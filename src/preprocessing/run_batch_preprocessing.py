"""Run and evaluate every selected EEG-fMRI preprocessing strategy.

Only raw EEG records whose file name contains ``fmri`` and which have a
matching ``Gradient_artifact_corrected`` reference are included. By default,
T0 is detected once with GradientSync and every strategy receives the same
600-second interval beginning at that T0. Evaluation uses the identical
interval from the clean reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mne
import numpy as np

try:
    from functions.bcggan_training_data import find_gradient_corrected_pairs
    from functions.preprocessing_paths import preprocessing_output_path
    from functions.appear_utils import load_subject_fmri_acquisition
    from functions.preprocessing_window import PreprocessingWindow, resolve_preprocessing_window
except ModuleNotFoundError:  # pragma: no cover - direct execution convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from functions.bcggan_training_data import find_gradient_corrected_pairs
    from functions.preprocessing_paths import preprocessing_output_path
    from functions.appear_utils import load_subject_fmri_acquisition
    from functions.preprocessing_window import PreprocessingWindow, resolve_preprocessing_window


DEFAULT_EEG_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_EEG")
DEFAULT_FMRI_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_MRI")
# Dataset1 derivatives live below the EEGLAB BIDS root.  Keep this default in
# sync with both training scripts, while allowing another dataset layout via
# --reference-root.
DEFAULT_REFERENCE_ROOT = DEFAULT_EEG_ROOT / "derivatives/Gradient_artifact_corrected"
DEFAULT_PREPROCESSING_ROOT = Path("data/preprocessing")
DEFAULT_EVALUATION_ROOT = Path("data/evaluation")
STRATEGIES = ("APPEAR", "BCGGAN", "AAS_AAS_PCA", "IVA_OBS_ICA", "DAR")
MANIFEST_FIELDS = (
    "strategy", "subject", "task", "raw", "reference", "processed", "status", "message",
    "preprocessing_seconds", "evaluation_seconds", "total_seconds", "reused_preprocessing",
    "window_enabled", "t0_seconds", "tf_seconds", "window_duration_seconds", "t0_source",
    "t0_detection_error",
    "completed_at_utc",
)
_AUXILIARY_TOKENS = ("ECG", "EKG", "VREF", "TRIG", "STI", "MISC", "RESP", "EOG", "EMG", "AUX")


def _task_from_path(path: Path) -> str:
    match = re.search(r"_task-(.+)_eeg\.set$", path.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not extract the BIDS task from {path.name}")
    return match.group(1)


def _subject_from_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name.lower().startswith("sub-"):
            return parent.name
    raise ValueError(f"Could not extract the BIDS subject from {path}")


def _script_path(strategy: str) -> Path:
    return Path(__file__).resolve().parent / f"{strategy}.py"


def _command_for_strategy(
    args: argparse.Namespace,
    strategy: str,
    subject: str,
    task: str,
    window: PreprocessingWindow | None,
) -> list[str]:
    command = [
        sys.executable, str(_script_path(strategy)), "--subject", subject, "--task", task,
        "--eeg-root", str(args.eeg_root), "--preprocessing-root", str(args.preprocessing_root),
        "--dataset-name", args.dataset_name,
    ]
    if window is not None:
        command.extend([
            "--crop-start-seconds", f"{window.start_sec:.12g}",
            "--crop-duration-seconds", f"{window.duration_sec:.12g}",
        ])
    if strategy in {"APPEAR", "AAS_AAS_PCA", "IVA_OBS_ICA"}:
        command.extend(["--fmri-root", str(args.fmri_root)])
        if args.use_fmri_checkpoint and window is None:
            command.append("--use-fmri-checkpoint")
    if strategy == "BCGGAN":
        command.extend(["--checkpoint", str(args.bcggan_checkpoint), "--batch-size", str(args.model_batch_size)])
    if strategy == "DAR":
        command.extend(["--checkpoint", str(args.dar_checkpoint), "--batch-size", str(args.model_batch_size)])
    return command


def _window_for_pair(
    args: argparse.Namespace,
    raw_path: Path,
    subject: str,
    task: str,
) -> PreprocessingWindow | None:
    if not args.use_gradient_sync_window:
        return None
    if args.gradient_sync_window_scope == "fmrirestingec" and task.lower() != "fmrirestingec":
        print(
            f"GradientSync window skipped for {subject}/{task}: scope is fmrirestingec.",
            flush=True,
        )
        return None
    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose="ERROR")
    names = [
        name for name in raw.ch_names
        if not any(token in name.upper() for token in _AUXILIARY_TOKENS)
    ]
    acquisition = load_subject_fmri_acquisition(args.fmri_root, subject)
    window = resolve_preprocessing_window(
        raw.get_data(picks=names),
        fs=float(raw.info["sfreq"]),
        tr_sec=float(acquisition["tr_sec"]),
        n_slices=int(acquisition["n_slices"]),
        channel_names=names,
        duration_sec=args.fmri_duration_seconds,
        calibration_seconds=args.calibration_seconds,
        threshold_sigma=args.threshold_sigma,
        fallback_t0_sec=args.fallback_t0_seconds,
    )
    if window.detection_error is not None:
        print(
            f"WARNING: GradientSync T0 failed for {subject}; using "
            f"{window.start_sec:.3f} s. {window.detection_error}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"GradientSync window for {subject}: {window.start_sec:.3f} -> "
            f"{window.stop_sec:.3f} s ({window.duration_sec:.3f} s)",
            flush=True,
        )
    return window


def _append_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Append batch results, upgrading manifests created by older versions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_rows: list[dict[str, str]] = []
    needs_upgrade = False
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                needs_upgrade = True
                previous_rows = list(reader)
    if needs_upgrade:
        # Preserve old rows and add empty values for the timing fields.
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in MANIFEST_FIELDS} for row in previous_rows)
        temporary.replace(path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _processed_matches_window(path: Path, window: PreprocessingWindow | None) -> bool:
    """Only reuse outputs produced from the same original-recording window."""

    if window is None:
        return True
    metadata_path = path.with_suffix(".json")
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            crop = metadata.get("input_crop", {})
            return (
                bool(crop.get("applied"))
                and int(crop.get("original_start_sample", -1)) == window.start_sample
                and int(crop.get("original_stop_sample", -1)) == window.stop_sample
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "crop_start_sec" not in archive or "crop_stop_sec" not in archive:
                    return False
                return (
                    np.isclose(float(np.asarray(archive["crop_start_sec"]).item()), window.start_sec)
                    and np.isclose(float(np.asarray(archive["crop_stop_sec"]).item()), window.stop_sec)
                )
        except (OSError, KeyError, ValueError):
            return False
    return False


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _store_execution_timing(
    summary_path: Path,
    preprocessing_seconds: float,
    evaluation_seconds: float,
    total_seconds: float,
    reused_preprocessing: bool,
) -> None:
    """Add timing fields to the same JSON file that holds quality metrics."""
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    summary.update({
        "preprocessing_seconds": preprocessing_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": total_seconds,
        "preprocessing_reused": reused_preprocessing,
        "timing_recorded_at_utc": datetime.now(UTC).isoformat(),
    })
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch preprocess and evaluate paired Dataset1 EEG-fMRI recordings.")
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--preprocessing-root", type=Path, default=DEFAULT_PREPROCESSING_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--dataset-name", default="Dataset1")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--bcggan-checkpoint", type=Path, default=Path("data/models/BCGGAN/training1/best.pt"))
    parser.add_argument("--dar-checkpoint", type=Path, default=Path("data/models/DAR/training1/best.pt"))
    parser.add_argument("--model-batch-size", type=int, default=4, help="Inference batch size for both neural models; increase only if GPU memory permits.")
    parser.add_argument(
        "--use-gradient-sync-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect T0 once per recording and process a fixed-duration window (enabled by default).",
    )
    parser.add_argument("--fmri-duration-seconds", type=float, default=600.0)
    parser.add_argument("--calibration-seconds", type=float, default=2.0)
    parser.add_argument("--threshold-sigma", type=float, default=4.0)
    parser.add_argument("--fallback-t0-seconds", type=float, default=10.0)
    parser.add_argument(
        "--gradient-sync-window-scope",
        choices=("fmrirestingec", "all"),
        default="fmrirestingec",
        help="Apply the T0-to-T0+duration window only to fmrirestingec (default) or to all tasks.",
    )
    parser.add_argument("--use-fmri-checkpoint", action="store_true", help="Enable morphology-based T0/Tf only for applicable methods.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse an existing .npz but still run its evaluation.")
    args = parser.parse_args()

    pairs = [pair for pair in find_gradient_corrected_pairs(args.eeg_root, args.reference_root) if "fmri" in pair.raw_set.name.lower()]
    if not pairs:
        raise FileNotFoundError("No complete paired EEG files with 'fmri' in their name were found.")
    missing = {"BCGGAN": args.bcggan_checkpoint, "DAR": args.dar_checkpoint}
    requested = [name for name in args.strategies if name not in missing or missing[name].is_file()]
    for name, checkpoint in missing.items():
        if name in args.strategies and not checkpoint.is_file():
            print(f"Skipping {name}: checkpoint not found: {checkpoint}")
    print(f"Matched fMRI recordings with references: {len(pairs)} | strategies: {', '.join(requested)}")

    manifest_rows: list[dict[str, str]] = []
    total_runs = len(pairs) * len(requested)
    completed_runs = 0
    batch_start = time.perf_counter()
    evaluator = Path(__file__).resolve().parents[1] / "functions" / "evaluate_eeg_preprocessing.py"
    for pair in pairs:
        subject, task = _subject_from_path(pair.raw_set), _task_from_path(pair.raw_set)
        window: PreprocessingWindow | None = None
        window_error: Exception | None = None
        try:
            window = _window_for_pair(args, pair.raw_set, subject, task)
        except (OSError, RuntimeError, ValueError) as error:
            window_error = error
        for strategy in requested:
            print(f"Running {strategy}: {subject}/{task}", flush=True)
            processed = preprocessing_output_path(args.preprocessing_root, args.dataset_name, strategy, subject, task)
            method_name = f"{strategy}_{processed.stem}"
            run_start = time.perf_counter()
            preprocessing_seconds = 0.0
            evaluation_seconds = 0.0
            reused = (
                args.skip_existing
                and processed.is_file()
                and _processed_matches_window(processed, window)
            )
            try:
                if window_error is not None:
                    raise window_error
                if not reused:
                    preprocessing_start = time.perf_counter()
                    subprocess.run(_command_for_strategy(args, strategy, subject, task, window), check=True)
                    preprocessing_seconds = time.perf_counter() - preprocessing_start
                else:
                    print(f"Reusing preprocessing: {processed}", flush=True)
                if not processed.is_file():
                    raise FileNotFoundError(f"The strategy did not create {processed}")
                evaluation_start = time.perf_counter()
                evaluation_command = [
                    sys.executable, str(evaluator), "--processed", str(processed), "--reference", str(pair.clean_set),
                    "--output-dir", str(args.evaluation_root / args.dataset_name / strategy), "--method-name", method_name,
                    "--resample-reference",
                ]
                if window is not None:
                    evaluation_command.extend([
                        "--reference-start-seconds", f"{window.start_sec:.12g}",
                        "--max-duration-seconds", f"{window.duration_sec:.12g}",
                    ])
                subprocess.run(evaluation_command, check=True)
                evaluation_seconds = time.perf_counter() - evaluation_start
                total_seconds = time.perf_counter() - run_start
                summary_path = args.evaluation_root / args.dataset_name / strategy / f"{method_name}_summary_metrics.json"
                _store_execution_timing(summary_path, preprocessing_seconds, evaluation_seconds, total_seconds, reused)
                status, message = "success", ""
            except (OSError, subprocess.CalledProcessError, ValueError) as error:
                status, message = "failed", str(error)
                print(f"WARNING: {strategy} failed for {subject} / {task}: {error}", file=sys.stderr, flush=True)
            total_seconds = time.perf_counter() - run_start
            completed_runs += 1
            manifest_rows.append({
                "strategy": strategy, "subject": subject, "task": task, "raw": str(pair.raw_set),
                "reference": str(pair.clean_set), "processed": str(processed), "status": status, "message": message,
                "preprocessing_seconds": f"{preprocessing_seconds:.3f}",
                "evaluation_seconds": f"{evaluation_seconds:.3f}", "total_seconds": f"{total_seconds:.3f}",
                "reused_preprocessing": str(reused), "completed_at_utc": datetime.now(UTC).isoformat(),
                "window_enabled": str(window is not None),
                "t0_seconds": "" if window is None else f"{window.start_sec:.6f}",
                "tf_seconds": "" if window is None else f"{window.stop_sec:.6f}",
                "window_duration_seconds": "" if window is None else f"{window.duration_sec:.6f}",
                "t0_source": "" if window is None else window.t0_source,
                "t0_detection_error": "" if window is None else (window.detection_error or ""),
            })
            _append_manifest(args.evaluation_root / args.dataset_name / "batch_manifest.csv", manifest_rows[-1:])
            elapsed = time.perf_counter() - batch_start
            mean_seconds = elapsed / completed_runs
            remaining = mean_seconds * (total_runs - completed_runs)
            print(
                f"Completed {completed_runs}/{total_runs}: {strategy} {subject}/{task} in {_format_duration(total_seconds)} "
                f"| elapsed {_format_duration(elapsed)} | estimated remaining {_format_duration(remaining)}",
                flush=True,
            )
    successes = sum(row["status"] == "success" for row in manifest_rows)
    print(f"\nFinished {successes}/{len(manifest_rows)} preprocessing/evaluation runs.")
    print(f"Manifest: {args.evaluation_root / args.dataset_name / 'batch_manifest.csv'}")


if __name__ == "__main__":
    main()

"""GA-AAS, BCG-AAS and PCA cleaning strategy for simultaneous EEG-fMRI."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import mne
import numpy as np

try:
    from functions.aas_ga import run_aas_pipeline
    from functions.appear_utils import apply_event_locked_aas, load_subject_fmri_acquisition
    from functions.gradient_sync import detect_gradient_artifact_start
    from functions.pca_artifact_rejection import apply_pca_artifact_rejection
    from functions.preprocessing_paths import preprocessing_output_path
    from functions.preprocessing_window import crop_mne_raw
except ModuleNotFoundError:  # pragma: no cover - direct execution convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from functions.aas_ga import run_aas_pipeline
    from functions.appear_utils import apply_event_locked_aas, load_subject_fmri_acquisition
    from functions.gradient_sync import detect_gradient_artifact_start
    from functions.pca_artifact_rejection import apply_pca_artifact_rejection
    from functions.preprocessing_paths import preprocessing_output_path
    from functions.preprocessing_window import crop_mne_raw


DEFAULT_EEG_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_EEG")
DEFAULT_FMRI_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_MRI")
_AUXILIARY_TOKENS = ("ECG", "EKG", "VREF", "TRIG", "STI", "MISC", "RESP", "EOG", "EMG", "AUX")


def _eeg_names(raw: mne.io.BaseRaw) -> list[str]:
    names = [name for name in raw.ch_names if not any(token in name.upper() for token in _AUXILIARY_TOKENS)]
    if len(names) < 2:
        raise ValueError("Fewer than two EEG channels remain after excluding auxiliary channels.")
    return names


def _ecg_name(raw: mne.io.BaseRaw, requested: str | None) -> str | None:
    if requested is not None:
        if requested not in raw.ch_names:
            raise ValueError(f"ECG channel not found: {requested}")
        return requested
    return next((name for name in raw.ch_names if "ECG" in name.upper() or "EKG" in name.upper()), None)


def _replace(raw: mne.io.BaseRaw, names: list[str], values: np.ndarray) -> mne.io.BaseRaw:
    output = raw.copy().load_data()
    output._data[[output.ch_names.index(name) for name in names]] = values
    return output


def _apply_ga_aas(
    data: np.ndarray, fs: float, slice_period_sec: float, start: int, stop: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply period-locked AAS only to the selected fMRI interval."""
    interval = data[:, start:stop]
    result = run_aas_pipeline(interval, TR=slice_period_sec, fs=fs, window_size=21)
    period = int(result["T_samples"])
    offset = int(result["offset"])
    n_epochs = int(np.asarray(result["cleaned_segments"]).shape[1])
    covered_start = start + offset
    covered_stop = covered_start + n_epochs * period
    cleaned = data.copy()
    cleaned[:, covered_start:covered_stop] = np.asarray(result["cleaned_signal"])[:, offset : offset + n_epochs * period]
    return cleaned, {"offset": offset, "period_samples": period, "covered_start_sample": covered_start, "covered_end_sample": covered_stop}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean EEG with GA-AAS, BCG-AAS and PCA artifact rejection.")
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--task", default="fmrirestingec")
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT)
    parser.add_argument("--preprocessing-root", type=Path, default=Path("data/preprocessing"))
    parser.add_argument("--dataset-name", default="Dataset1")
    parser.add_argument("--output", type=Path, help="Optional explicit output path.")
    parser.add_argument("--use-fmri-checkpoint", action="store_true", help="Detect T0/Tf from GA morphology; disabled by default.")
    parser.add_argument("--calibration-seconds", type=float, default=5.0)
    parser.add_argument("--crop-start-seconds", type=float)
    parser.add_argument("--crop-duration-seconds", type=float)
    parser.add_argument("--ecg-channel")
    parser.add_argument("--skip-bcg", action="store_true")
    parser.add_argument("--pca-exclude", type=int, nargs="*", default=())
    parser.add_argument("--no-auto-ocular-pca", action="store_false", dest="auto_ocular_pca", default=True,
                        help="Keep ocular PCA candidates instead of removing them automatically.")
    parser.add_argument("--no-auto-muscle-pca", action="store_false", dest="auto_muscle_pca", default=True,
                        help="Keep muscle PCA candidates instead of removing them automatically.")
    args = parser.parse_args()

    eeg_path = args.eeg_root / args.subject / "eeg" / f"{args.subject}_task-{args.task}_eeg.set"
    if not eeg_path.is_file():
        raise FileNotFoundError(f"EEGLAB header not found: {eeg_path}")
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose="ERROR")
    raw, crop_info = crop_mne_raw(raw, args.crop_start_seconds, args.crop_duration_seconds)
    acquisition = load_subject_fmri_acquisition(args.fmri_root, args.subject)
    tr_sec, n_slices = float(acquisition["tr_sec"]), int(acquisition["n_slices"])
    names, fs = _eeg_names(raw), float(raw.info["sfreq"])
    data = raw.get_data(picks=names)
    start, stop = 0, data.shape[1]
    detection = None
    if args.use_fmri_checkpoint:
        detection = detect_gradient_artifact_start(
            data, fs, tr_sec, n_slices,
            calibration_seconds=args.calibration_seconds,
            channel_names=names,
        )
        start, stop = detection.t_ga_sample, detection.t_ga_end_sample
    ga_clean, ga_info = _apply_ga_aas(data, fs, tr_sec / n_slices, start, stop)
    raw_bcg = _replace(raw, names, ga_clean)

    bcg_info: dict[str, object] = {"applied": False, "reason": "disabled" if args.skip_bcg else "no ECG channel"}
    ecg = _ecg_name(raw_bcg, args.ecg_channel)
    if not args.skip_bcg and ecg is not None:
        try:
            events, _, _ = mne.preprocessing.find_ecg_events(raw_bcg, ch_name=ecg, verbose="ERROR")
            corrected, details = apply_event_locked_aas(raw_bcg.get_data(picks=names), events[:, 0] - raw_bcg.first_samp, fs)
            raw_bcg = _replace(raw_bcg, names, corrected)
            bcg_info = {"applied": True, "ecg_channel": ecg, "n_events": int(len(details["event_samples"])), "template_window": 21}
        except (RuntimeError, ValueError) as error:
            warnings.warn(f"BCG AAS was skipped: {error}", RuntimeWarning, stacklevel=2)
            bcg_info = {"applied": False, "ecg_channel": ecg, "reason": str(error)}

    raw_clean, pca_info = apply_pca_artifact_rejection(
        raw_bcg, names, manual_exclude=args.pca_exclude,
        auto_ocular=args.auto_ocular_pca, auto_muscle=args.auto_muscle_pca,
    )
    output = args.output or preprocessing_output_path(args.preprocessing_root, args.dataset_name, "AAS_AAS_PCA", args.subject, args.task)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, cleaned_signal=raw_clean.get_data(), fs=float(raw_clean.info["sfreq"]), channel_names=np.asarray(raw_clean.ch_names))
    # Save the PCA decomposition separately, so it can be inspected without
    # reopening or recomputing this preprocessing run.
    pca_path = output.with_name(f"{output.stem}_pca_analysis.npz")
    np.savez_compressed(
        pca_path,
        channel_names=np.asarray(pca_info["picks"]),
        spatial_components=np.asarray(pca_info["spatial_components"]),
        component_scores=np.asarray(pca_info["component_scores"]),
        frontal_ratio=np.asarray(pca_info["frontal_ratio"]),
        low_frequency_power=np.asarray(pca_info["low_frequency_power"]),
        gamma_power=np.asarray(pca_info["gamma_power"]),
        excluded_components=np.asarray(pca_info["excluded_components"], dtype=int),
    )
    metadata = {
        "tr_sec": tr_sec, "n_slices": n_slices, "slice_period_sec": tr_sec / n_slices,
        "use_fmri_checkpoint": args.use_fmri_checkpoint, "fmri_start_sample": start, "fmri_end_sample": stop,
        "input_crop": crop_info,
        "ga_aas": ga_info, "bcg_aas": bcg_info,
        "pca_excluded_components": pca_info["excluded_components"],
        "pca_ocular_candidates": pca_info["ocular_candidates"], "pca_muscle_candidates": pca_info["muscle_candidates"],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved AAS_AAS_PCA-cleaned EEG to {output} ({raw_clean.get_data().shape}).")
    print(f"Saved PCA analysis to {pca_path}.")


if __name__ == "__main__":
    main()

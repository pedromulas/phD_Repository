"""Command-line preprocessing of Dataset1 EEGLAB EEG with band-pass + FASTR.

MNE reads an EEGLAB ``.set`` header and its adjacent ``.fdt`` payload, so no
manual binary parsing is needed.  The output is an ``.npz`` file with the
cleaned data, sampling rate, channel labels, and FASTR parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np

try:  # Supports both ``python -m functions...`` and direct script execution.
    from functions.fastr_ga import FASTRConfig, bandpass_filter, fastr_remove_gradient_artifact
except ModuleNotFoundError:  # pragma: no cover - direct execution convenience
    from fastr_ga import FASTRConfig, bandpass_filter, fastr_remove_gradient_artifact


DEFAULT_EEG_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_EEG")


def get_eeg_set_path(subject: str, task: str, eeg_root: Path = DEFAULT_EEG_ROOT) -> Path:
    """Return and validate the EEGLAB header whose data are stored in its .fdt."""
    path = eeg_root / subject / "eeg" / f"{subject}_task-{task}_eeg.set"
    if not path.is_file():
        raise FileNotFoundError(f"EEGLAB .set file not found: {path}")
    return path


def load_eeglab_fdt(set_path: Path) -> tuple[np.ndarray, float, list[str]]:
    """Load a .set/.fdt recording as a channels-by-samples NumPy array."""
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    return raw.get_data(), float(raw.info["sfreq"]), list(raw.ch_names)


def preprocess_recording(
    set_path: Path,
    artifact_period_s: float,
    low_hz: float = 0.1,
    high_hz: float = 70.0,
    template_window: int = 21,
    pca_components: int = 4,
    offset: int | None = None,
    use_fmri_bounds: bool = False,
    tr_sec: float | None = None,
    n_slices: int | None = None,
    calibration_seconds: float = 5.0,
    fmri_start_sample: int | None = None,
    fmri_end_sample: int | None = None,
) -> dict[str, object]:
    """Band-pass filter a Dataset1 recording and suppress the MRI gradient artefact."""
    data, fs, channel_names = load_eeglab_fdt(set_path)
    filtered = bandpass_filter(data, fs, low_hz=low_hz, high_hz=high_hz)
    result = fastr_remove_gradient_artifact(
        filtered,
        fs,
        FASTRConfig(
            artifact_period_s=artifact_period_s,
            template_window=template_window,
            pca_components=pca_components,
            offset=offset,
            use_fmri_bounds=use_fmri_bounds,
            tr_sec=tr_sec,
            n_slices=n_slices,
            calibration_seconds=calibration_seconds,
            fmri_start_sample=fmri_start_sample,
            fmri_end_sample=fmri_end_sample,
        ),
        detection_signal=data,
        channel_names=channel_names,
    )
    return {**result, "fs": fs, "channel_names": channel_names, "filtered_signal": filtered}


def main() -> None:
    parser = argparse.ArgumentParser(description="Band-pass and FASTR-preprocess a Dataset1 EEG recording.")
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--task", default="fmrirestingec")
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--artifact-period", type=float, default=2.0, help="Seconds per repeated GA artefact; set slice period when known.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=4)
    parser.add_argument("--use-fmri-bounds", action="store_true", help="Detect T0/Tf from GA morphology and clean only that interval.")
    parser.add_argument("--tr", type=float, help="fMRI TR in seconds; required with --use-fmri-bounds.")
    parser.add_argument("--n-slices", type=int, help="Number of fMRI slices; required with --use-fmri-bounds.")
    parser.add_argument("--fmri-start-sample", type=int, help="Optional manual T0; bypasses its automatic detection.")
    parser.add_argument("--fmri-end-sample", type=int, help="Optional manual Tf; bypasses its automatic detection.")
    parser.add_argument("--calibration-seconds", type=float, default=5.0, help="Initial non-fMRI EEG used to set the GA detection threshold.")
    args = parser.parse_args()

    set_path = get_eeg_set_path(args.subject, args.task, args.eeg_root)
    result = preprocess_recording(
        set_path, args.artifact_period, pca_components=args.pca_components,
        use_fmri_bounds=args.use_fmri_bounds, tr_sec=args.tr,
        n_slices=args.n_slices, calibration_seconds=args.calibration_seconds,
        fmri_start_sample=args.fmri_start_sample, fmri_end_sample=args.fmri_end_sample,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"Saved FASTR-cleaned EEG to {args.output} ({result['cleaned_signal'].shape}).")
    if args.use_fmri_bounds:
        print(f"Detected fMRI interval: {result['fmri_start_sec']:.3f} s to {result['fmri_end_sec']:.3f} s.")


if __name__ == "__main__":
    main()

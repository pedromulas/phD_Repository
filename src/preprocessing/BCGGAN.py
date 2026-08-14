"""Apply a trained BCGGAN model as a named preprocessing strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # Supports both ``python -m`` and direct script execution.
    from functions.apply_bcggan import apply_checkpoint
    from functions.preprocessing_paths import preprocessing_output_path
except ModuleNotFoundError:  # pragma: no cover - direct execution convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from functions.apply_bcggan import apply_checkpoint
    from functions.preprocessing_paths import preprocessing_output_path


DEFAULT_EEG_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_EEG")
DEFAULT_CHECKPOINT = Path("data/models/BCGGAN/training1/best.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean one EEG recording with a trained BCGGAN checkpoint.")
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--task", default="fmrirestingec")
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--preprocessing-root", type=Path, default=Path("data/preprocessing"))
    parser.add_argument("--dataset-name", default="Dataset1")
    parser.add_argument("--output", type=Path, help="Optional explicit output path; otherwise the standard BCGGAN path is used.")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Must match the 5-second training windows.")
    parser.add_argument("--batch-size", type=int, default=4, help="Reduce if CUDA runs out of memory.")
    parser.add_argument("--crop-start-seconds", type=float)
    parser.add_argument("--crop-duration-seconds", type=float)
    args = parser.parse_args()

    eeg_path = args.eeg_root / args.subject / "eeg" / f"{args.subject}_task-{args.task}_eeg.set"
    if not eeg_path.is_file():
        raise FileNotFoundError(f"EEGLAB header not found: {eeg_path}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"BCGGAN checkpoint not found: {args.checkpoint}")
    output_path = args.output or preprocessing_output_path(
        args.preprocessing_root, args.dataset_name, "BCGGAN", args.subject, args.task,
    )
    result = apply_checkpoint(
        args.checkpoint,
        eeg_path,
        output_path,
        window_s=args.window_seconds,
        stride_s=args.window_seconds,
        batch_size=args.batch_size,
        crop_start_seconds=args.crop_start_seconds,
        crop_duration_seconds=args.crop_duration_seconds,
    )
    print(f"Saved BCGGAN-cleaned EEG to {output_path} ({result['cleaned_signal'].shape}).")


if __name__ == "__main__":
    main()

"""DAR denoising-autoencoder preprocessing strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from functions.apply_dar import apply_dar_checkpoint
    from functions.preprocessing_paths import preprocessing_output_path
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from functions.apply_dar import apply_dar_checkpoint
    from functions.preprocessing_paths import preprocessing_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean one EEG recording with a trained DAR checkpoint.")
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--task", default="fmrirestingec")
    parser.add_argument("--eeg-root", type=Path, default=Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_EEG"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/models/DAR/training1/best.pt"))
    parser.add_argument("--preprocessing-root", type=Path, default=Path("data/preprocessing"))
    parser.add_argument("--dataset-name", default="Dataset1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    eeg_path = args.eeg_root / args.subject / "eeg" / f"{args.subject}_task-{args.task}_eeg.set"
    if not eeg_path.is_file(): raise FileNotFoundError(f"EEGLAB header not found: {eeg_path}")
    if not args.checkpoint.is_file(): raise FileNotFoundError(f"DAR checkpoint not found: {args.checkpoint}")
    output = args.output or preprocessing_output_path(args.preprocessing_root, args.dataset_name, "DAR", args.subject, args.task)
    result = apply_dar_checkpoint(args.checkpoint, eeg_path, output, args.batch_size)
    print(f"Saved DAR-cleaned EEG to {output} ({result['cleaned_signal'].shape}).")


if __name__ == "__main__": main()

"""Apply a trained BCGGAN checkpoint to contaminated EEG without clean data."""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np

try:  # Supports both ``python -m functions...`` and direct script execution.
    from functions.bcggan import BCGGANTrainer
except ModuleNotFoundError:  # pragma: no cover - direct execution convenience
    from bcggan import BCGGANTrainer


_AUXILIARY_TOKENS = ("ECG", "EKG", "VREF", "TRIG", "STI", "MISC", "RESP", "EOG", "EMG", "AUX")


def apply_checkpoint(
    checkpoint: Path,
    corrupted_set: Path,
    output: Path,
    window_s: float = 5.0,
    stride_s: float = 5.0,
    batch_size: int = 32,
) -> dict[str, object]:
    """Load only contaminated EEG, clean it, and save a portable NumPy result."""
    raw = mne.io.read_raw_eeglab(corrupted_set, preload=True, verbose="ERROR")
    corrupted = raw.get_data()
    trainer = BCGGANTrainer.load_checkpoint(checkpoint)
    if trainer.config.channels == 1:
        eeg_indices = [index for index, name in enumerate(raw.ch_names) if not any(token in name.upper() for token in _AUXILIARY_TOKENS)]
        cleaned = corrupted.copy()
        cleaned[eeg_indices] = trainer.clean_recording(corrupted[eeg_indices], float(raw.info["sfreq"]), window_s, stride_s, batch_size)
    else:
        cleaned = trainer.clean_recording(corrupted, float(raw.info["sfreq"]), window_s, stride_s, batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "cleaned_signal": cleaned,
        "corrupted_signal": corrupted,
        "fs": float(raw.info["sfreq"]),
        "channel_names": np.asarray(raw.ch_names),
    }
    np.savez_compressed(output, **result)
    # TODO: add test-set quality metrics once the project selects their definitions.
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean contaminated test EEG using a trained BCGGAN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corrupted-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Must match the 5-second windows used during training.")
    parser.add_argument("--stride-seconds", type=float, default=5.0, help="Use 5 seconds to concatenate non-overlapping model outputs.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = apply_checkpoint(args.checkpoint, args.corrupted_set, args.output, args.window_seconds, args.stride_seconds, args.batch_size)
    print(f"Saved cleaned EEG to {args.output} ({result['cleaned_signal'].shape}).")


if __name__ == "__main__":
    main()

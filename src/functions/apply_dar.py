"""Apply a trained DAR checkpoint to an EEGLAB EEG recording."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from functions.dar import DARTrainer
from functions.preprocessing_window import crop_mne_raw

_AUX = ("ECG", "EKG", "VREF", "TRIG", "STI", "MISC", "RESP", "EOG", "EMG", "AUX")


def apply_dar_checkpoint(
    checkpoint: Path,
    eeg_path: Path,
    output: Path,
    batch_size: int = 32,
    crop_start_seconds: float | None = None,
    crop_duration_seconds: float | None = None,
) -> dict[str, object]:
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose="ERROR")
    raw, crop_info = crop_mne_raw(raw, crop_start_seconds, crop_duration_seconds)
    original = raw.get_data()
    eeg_indices = [index for index, name in enumerate(raw.ch_names) if not any(token in name.upper() for token in _AUX)]
    trainer = DARTrainer.load_checkpoint(checkpoint)
    cleaned = original.copy()
    cleaned[eeg_indices] = trainer.clean_recording(original[eeg_indices], float(raw.info["sfreq"]), batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "cleaned_signal": cleaned,
        "corrupted_signal": original,
        "fs": float(raw.info["sfreq"]),
        "channel_names": np.asarray(raw.ch_names),
        "crop_start_sec": float(crop_info["start_sec"]),
        "crop_stop_sec": float(crop_info["stop_sec"]),
    }
    np.savez_compressed(output, **result)
    return result

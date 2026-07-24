"""Lazy single-channel, fixed-duration EEG windows for BCGGAN training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import mne
import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
except ImportError as error:  # pragma: no cover - depends on the training environment
    raise ImportError("BCGGAN training requires PyTorch. Install it with `pip install torch`.") from error


EXCLUDED_CHANNEL_TOKENS = ("ECG", "EKG", "VREF", "TRIG", "STI", "MISC", "RESP", "EOG", "EMG", "AUX")


@dataclass(frozen=True)
class EEGRecordPair:
    """One raw/gradient-corrected EEGLAB recording pair."""

    raw_set: Path
    clean_set: Path
    sfreq: float
    n_times: int
    channel_indices: tuple[int, ...]


@dataclass(frozen=True)
class WindowIndex:
    """Location of one single-channel segment in a recording pair."""

    record_index: int
    channel_index: int
    start_sample: int
    stop_sample: int
    group_index: int


def _eeg_channels(channel_names: list[str]) -> tuple[int, ...]:
    selected = [
        index for index, name in enumerate(channel_names)
        if not any(token in name.upper() for token in EXCLUDED_CHANNEL_TOKENS)
    ]
    if not selected:
        raise ValueError("No EEG channels remain after excluding auxiliary channels.")
    return tuple(selected)


def find_gradient_corrected_pairs(raw_root: Path, corrected_root: Path, task: str | None = None) -> list[EEGRecordPair]:
    """Match raw ``*_eeg.set`` files with ``*_gac_eeg.set`` derivatives."""
    raw_root, corrected_root = Path(raw_root), Path(corrected_root)
    raw_paths = sorted(raw_root.glob("sub-*/eeg/*_eeg.set"))
    pairs: list[EEGRecordPair] = []
    for raw_path in raw_paths:
        if task is not None and f"task-{task}" not in raw_path.name:
            continue
        clean_name = raw_path.name.replace("_eeg.set", "_gac_eeg.set")
        clean_path = corrected_root / clean_name
        if not clean_path.is_file():
            continue
        try:
            # EEGLAB headers may reference an absent .fdt file. Such a
            # recording cannot provide windows, so skip the whole pair.
            raw = mne.io.read_raw_eeglab(raw_path, preload=False, verbose="ERROR")
            clean = mne.io.read_raw_eeglab(clean_path, preload=False, verbose="ERROR")
        except FileNotFoundError as error:
            warnings.warn(f"Skipping incomplete EEGLAB pair {raw_path.name}: {error}", RuntimeWarning, stacklevel=2)
            continue
        if raw.ch_names != clean.ch_names:
            raise ValueError(f"Raw and corrected channel labels differ: {raw_path.name}")
        if raw.n_times != clean.n_times or raw.info["sfreq"] != clean.info["sfreq"]:
            raise ValueError(f"Raw and corrected sampling metadata differ: {raw_path.name}")
        pairs.append(
            EEGRecordPair(
                raw_set=raw_path,
                clean_set=clean_path,
                sfreq=float(raw.info["sfreq"]),
                n_times=raw.n_times,
                channel_indices=_eeg_channels(list(raw.ch_names)),
            )
        )
    if not pairs:
        raise FileNotFoundError("No matching raw/Gradient_artifact_corrected .set pairs were found.")
    return pairs


def build_window_index(records: list[EEGRecordPair], window_seconds: float = 5.0) -> list[WindowIndex]:
    """Create non-overlapping channel windows; incomplete final windows are omitted."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")
    index: list[WindowIndex] = []
    group = 0
    for record_index, record in enumerate(records):
        window_samples = int(round(window_seconds * record.sfreq))
        if window_samples < 2:
            raise ValueError("window_seconds is too short for this sampling frequency.")
        for start in range(0, record.n_times - window_samples + 1, window_samples):
            stop = start + window_samples
            # One group represents the same time interval in all channels;
            # it is never split across train/validation.
            for channel_index in record.channel_indices:
                index.append(WindowIndex(record_index, channel_index, start, stop, group))
            group += 1
    if not index:
        raise ValueError("No complete 5-second windows could be created.")
    return index


def split_window_index(index: list[WindowIndex], validation_fraction: float = 0.2, seed: int = 42) -> tuple[list[WindowIndex], list[WindowIndex]]:
    """Split temporal groups 80/20, keeping all channels of a group together."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    groups = np.unique([item.group_index for item in index])
    if groups.size < 2:
        raise ValueError("At least two complete time windows are required for train/validation splitting.")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_validation = max(1, int(round(groups.size * validation_fraction)))
    validation_groups = set(groups[:n_validation].tolist())
    train = [item for item in index if item.group_index not in validation_groups]
    validation = [item for item in index if item.group_index in validation_groups]
    return train, validation


class SingleChannelEEGWindowDataset(Dataset[Tensor]):
    """Read raw or GA-corrected one-channel windows lazily from EEGLAB/FDT."""

    def __init__(self, records: list[EEGRecordPair], index: list[WindowIndex], domain: str) -> None:
        if domain not in {"raw", "clean"}:
            raise ValueError("domain must be 'raw' or 'clean'.")
        self.records, self.index, self.domain = records, index, domain
        self._cache: dict[int, mne.io.BaseRaw] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _raw(self, record_index: int) -> mne.io.BaseRaw:
        if record_index not in self._cache:
            record = self.records[record_index]
            path = record.raw_set if self.domain == "raw" else record.clean_set
            self._cache[record_index] = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
        return self._cache[record_index]

    def __getitem__(self, item_index: int) -> Tensor:
        item = self.index[item_index]
        window = self._raw(item.record_index).get_data(
            picks=[item.channel_index], start=item.start_sample, stop=item.stop_sample
        ).astype(np.float32, copy=False)
        # Independent normalisation is appropriate for CycleGAN's unpaired
        # domains and prevents amplitude/session identity from dominating GAN.
        window = (window - window.mean(axis=-1, keepdims=True)) / np.maximum(window.std(axis=-1, keepdims=True), 1e-6)
        return torch.from_numpy(window)

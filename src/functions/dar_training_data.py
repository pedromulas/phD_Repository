"""Paired, channel-wise, overlapping EEG windows for supervised DAR training."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from functions.bcggan_training_data import EEGRecordPair
except ModuleNotFoundError:  # pragma: no cover - direct script execution convenience
    from bcggan_training_data import EEGRecordPair


@dataclass(frozen=True)
class DARWindowIndex:
    record_index: int
    channel_index: int
    start_sample: int
    stop_sample: int
    group_index: int


def build_dar_window_index(records: list[EEGRecordPair], window_seconds: float = 2.0, stride_seconds: float = 1.0) -> list[DARWindowIndex]:
    """Build overlapping 2-s windows while grouping every channel at one time."""
    result: list[DARWindowIndex] = []
    group = 0
    for record_index, record in enumerate(records):
        window, stride = int(round(window_seconds * record.sfreq)), int(round(stride_seconds * record.sfreq))
        starts = np.arange(0, record.n_times - window + 1, stride)
        for start in starts:
            for channel in record.channel_indices:
                result.append(DARWindowIndex(record_index, channel, int(start), int(start + window), group))
            group += 1
    if not result:
        raise ValueError("No complete DAR windows could be extracted.")
    return result


def split_dar_index(index: list[DARWindowIndex], validation_fraction: float = 0.2, seed: int = 42) -> tuple[list[DARWindowIndex], list[DARWindowIndex]]:
    groups = np.unique([item.group_index for item in index])
    if groups.size < 2 or not 0 < validation_fraction < 1:
        raise ValueError("At least two groups and a validation fraction in (0, 1) are required.")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    validation = set(groups[: max(1, int(round(groups.size * validation_fraction)))].tolist())
    return [item for item in index if item.group_index not in validation], [item for item in index if item.group_index in validation]


class DARPairedWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy paired raw/GAC reader; normalization uses noisy-window max amplitude."""

    def __init__(self, records: list[EEGRecordPair], index: list[DARWindowIndex]) -> None:
        self.records, self.index = records, index
        self._raw: dict[int, mne.io.BaseRaw] = {}
        self._clean: dict[int, mne.io.BaseRaw] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _recording(self, cache: dict[int, mne.io.BaseRaw], record_index: int, clean: bool) -> mne.io.BaseRaw:
        if record_index not in cache:
            path = self.records[record_index].clean_set if clean else self.records[record_index].raw_set
            cache[record_index] = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
        return cache[record_index]

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.index[position]
        noisy = self._recording(self._raw, item.record_index, False).get_data(picks=[item.channel_index], start=item.start_sample, stop=item.stop_sample).astype(np.float32, copy=False)
        clean = self._recording(self._clean, item.record_index, True).get_data(picks=[item.channel_index], start=item.start_sample, stop=item.stop_sample).astype(np.float32, copy=False)
        scale = max(float(np.max(np.abs(noisy))), 1e-8)
        return torch.from_numpy(noisy / scale), torch.from_numpy(clean / scale)

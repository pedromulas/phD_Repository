"""Reusable utilities for the APPEAR EEG-fMRI preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

try:
    from functions.load_fmri_metadata import find_fmri_files, load_fmri_metadata
    from functions.obs_ga import apply_sliding_window_obs
except ModuleNotFoundError:  # pragma: no cover - direct script execution convenience
    from load_fmri_metadata import find_fmri_files, load_fmri_metadata
    from obs_ga import apply_sliding_window_obs


def load_subject_fmri_acquisition(fmri_root: Path, subject: str) -> dict[str, object]:
    """Read TR and slice count for one subject from its BIDS MRI metadata."""
    paths = [path for path in find_fmri_files(Path(fmri_root)) if path.parent.parent.parent.name == subject]
    if not paths:
        raise FileNotFoundError(f"No BIDS fMRI file was found for {subject} under {fmri_root}.")
    metadata = load_fmri_metadata(paths[0])
    tr, n_slices = metadata.get("TR_s"), metadata.get("n_slices")
    if not isinstance(tr, (int, float)) or tr <= 0 or not isinstance(n_slices, int) or n_slices < 1:
        raise ValueError(f"Invalid TR or slice count in {metadata['json_path']}.")
    return {"tr_sec": float(tr), "n_slices": int(n_slices), "path": metadata["path"], "json_path": metadata["json_path"]}


def apply_event_locked_aas(
    signal: np.ndarray,
    event_samples: Sequence[int],
    fs: float,
    tmin_sec: float = -0.2,
    tmax_sec: float = 0.6,
    template_window: int = 21,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    """Subtract sliding AAS templates around cardiac-event samples.

    Contributions from overlapping cardiac windows are averaged. Samples not
    covered by a complete event window remain unchanged.
    """
    data = np.asarray(signal, dtype=np.float64)
    if data.ndim != 2 or fs <= 0 or tmin_sec >= tmax_sec or template_window < 1:
        raise ValueError("Invalid signal shape or event-AAS parameters.")
    start_offset, stop_offset = int(round(tmin_sec * fs)), int(round(tmax_sec * fs))
    valid = np.asarray([event for event in event_samples if -start_offset <= event and event + stop_offset <= data.shape[1]], dtype=int)
    if valid.size < 2:
        raise ValueError("At least two complete cardiac windows are required for BCG AAS.")
    epochs = np.stack([data[:, event + start_offset : event + stop_offset] for event in valid], axis=1)
    corrections = np.zeros_like(data)
    weights = np.zeros(data.shape[1], dtype=np.float64)
    half = template_window // 2
    for index, event in enumerate(valid):
        left = max(0, index - half)
        right = min(valid.size, left + template_window)
        left = max(0, right - template_window)
        template = epochs[:, left:right].mean(axis=1)
        start, stop = event + start_offset, event + stop_offset
        corrections[:, start:stop] += template
        weights[start:stop] += 1
    covered = weights > 0
    cleaned = data.copy()
    cleaned[:, covered] -= corrections[:, covered] / weights[covered]
    return cleaned, {
        "event_samples": valid,
        "tmin_sec": tmin_sec,
        "tmax_sec": tmax_sec,
        "template_window": template_window,
        "coverage_mask": covered,
    }


def apply_event_locked_obs(
    signal: np.ndarray,
    event_samples: Sequence[int],
    fs: float,
    tmin_sec: float = -0.2,
    tmax_sec: float = 0.6,
    window_size: int = 21,
    n_components: int = 4,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    """Suppress cardiac BCG using event-locked sliding-window OBS.

    OBS bases are learned independently per EEG channel from neighboring
    cardiac epochs. The reconstructed artifact contributions are averaged in
    overlapping epochs before subtraction from the continuous recording.
    """
    data = np.asarray(signal, dtype=np.float64)
    if data.ndim != 2 or fs <= 0 or tmin_sec >= tmax_sec:
        raise ValueError("Invalid signal shape or event-OBS parameters.")
    start_offset, stop_offset = int(round(tmin_sec * fs)), int(round(tmax_sec * fs))
    valid = np.asarray([event for event in event_samples if -start_offset <= event and event + stop_offset <= data.shape[1]], dtype=int)
    if valid.size < 2:
        raise ValueError("At least two complete cardiac windows are required for BCG OBS.")
    epochs = np.stack([data[:, event + start_offset : event + stop_offset] for event in valid], axis=1)
    corrections = np.zeros_like(data)
    weights = np.zeros(data.shape[1], dtype=np.float64)
    for channel in range(data.shape[0]):
        cleaned_epochs, _ = apply_sliding_window_obs(
            epochs[channel], window_size=window_size, n_components=n_components,
            remove_mean=False, normalize=False,
        )
        artifacts = epochs[channel] - cleaned_epochs
        for index, event in enumerate(valid):
            start, stop = event + start_offset, event + stop_offset
            corrections[channel, start:stop] += artifacts[index]
    for event in valid:
        weights[event + start_offset : event + stop_offset] += 1
    covered = weights > 0
    cleaned = data.copy()
    cleaned[:, covered] -= corrections[:, covered] / weights[covered]
    return cleaned, {
        "event_samples": valid,
        "tmin_sec": tmin_sec,
        "tmax_sec": tmax_sec,
        "window_size": window_size,
        "n_components": n_components,
        "coverage_mask": covered,
    }


def detect_bad_intervals_for_ica(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 7.0,
    threshold_db: float = 8.0,
    window_sec: float = 1.0,
    step_sec: float = 0.5,
) -> list[tuple[float, float]]:
    """Detect high-power low-frequency intervals for exclusion during ICA fit.

    This reproduces the intent of EEGLAB ``pop_rejcont`` using the APPEAR
    0.5--7 Hz and 8 dB settings. Returned intervals are in seconds.
    """
    data = np.asarray(signal, dtype=np.float64)
    if data.ndim != 2 or fs <= 0:
        raise ValueError("signal must be channels-by-samples and fs must be positive.")
    nyquist = fs / 2
    if not 0 < low_hz < high_hz < nyquist or threshold_db <= 0:
        raise ValueError("Invalid bad-interval filtering parameters.")
    window, step = int(round(window_sec * fs)), int(round(step_sec * fs))
    if window < 2 or step < 1 or data.shape[1] < window:
        return []
    sos = butter(3, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, data, axis=-1)
    starts = np.arange(0, data.shape[1] - window + 1, step)
    # Indexed data has shape (channels, windows, samples); aggregate channels
    # and samples to obtain one robust power estimate per temporal window.
    rms = np.sqrt(np.mean(filtered[:, starts[:, None] + np.arange(window)] ** 2, axis=(0, 2)))
    threshold = np.median(rms) * 10 ** (threshold_db / 20)
    flagged = starts[rms > threshold]
    if flagged.size == 0:
        return []
    intervals: list[tuple[int, int]] = []
    for start in flagged:
        stop = int(start + window)
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], stop))
        else:
            intervals.append((int(start), stop))
    return [(start / fs, stop / fs) for start, stop in intervals]


def add_bad_interval_annotations(raw: mne.io.BaseRaw, intervals: Sequence[tuple[float, float]]) -> mne.io.BaseRaw:
    """Attach BAD_APPEAR annotations so MNE excludes them while fitting ICA."""
    annotated = raw.copy()
    if intervals:
        onset = [start for start, _ in intervals]
        duration = [stop - start for start, stop in intervals]
        annotations = mne.Annotations(onset, duration, ["BAD_APPEAR"] * len(intervals), orig_time=annotated.info.get("meas_date"))
        annotated.set_annotations(annotated.annotations + annotations)
    return annotated

"""Shared fixed-duration window selection for batch EEG-fMRI preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

try:
    from functions.gradient_sync import GradientSyncDetection, detect_gradient_artifact_start
except ModuleNotFoundError:  # pragma: no cover - package-style test imports
    from .gradient_sync import GradientSyncDetection, detect_gradient_artifact_start


@dataclass(frozen=True)
class PreprocessingWindow:
    start_sample: int
    stop_sample: int
    start_sec: float
    stop_sec: float
    duration_sec: float
    t0_source: str
    detection_error: str | None


def fixed_duration_sample_bounds(
    n_samples: int,
    fs: float,
    start_sec: float,
    duration_sec: float,
) -> tuple[int, int]:
    """Convert a start and duration to an exact, validated sample interval."""

    if n_samples <= 0 or fs <= 0:
        raise ValueError("n_samples and fs must be strictly positive.")
    if start_sec < 0 or duration_sec <= 0:
        raise ValueError("start_sec must be non-negative and duration_sec strictly positive.")
    start = int(round(start_sec * fs))
    duration = int(round(duration_sec * fs))
    stop = start + duration
    if start >= n_samples:
        raise ValueError("The preprocessing T0 lies outside the EEG recording.")
    if stop > n_samples:
        available = (n_samples - start) / fs
        raise ValueError(
            f"The EEG contains only {available:.3f} s after T0; {duration_sec:.3f} s were requested."
        )
    return start, stop


def crop_mne_raw(
    raw: object,
    start_sec: float | None,
    duration_sec: float | None,
) -> tuple[object, dict[str, float | int | bool]]:
    """Crop an MNE Raw-like object to an exact fixed-duration sample window."""

    if start_sec is None and duration_sec is None:
        return raw, {
            "applied": False,
            "original_start_sample": 0,
            "original_stop_sample": int(raw.n_times),
            "start_sec": 0.0,
            "stop_sec": float(raw.n_times / raw.info["sfreq"]),
        }
    if start_sec is None or duration_sec is None:
        raise ValueError("crop start and duration must be provided together.")
    fs = float(raw.info["sfreq"])
    start, stop = fixed_duration_sample_bounds(int(raw.n_times), fs, start_sec, duration_sec)
    cropped = raw.copy().crop(start / fs, (stop - 1) / fs, include_tmax=True)
    return cropped, {
        "applied": True,
        "original_start_sample": start,
        "original_stop_sample": stop,
        "start_sec": start / fs,
        "stop_sec": stop / fs,
        "duration_sec": (stop - start) / fs,
    }


def resolve_preprocessing_window(
    signal: np.ndarray,
    *,
    fs: float,
    tr_sec: float,
    n_slices: int,
    channel_names: Sequence[str] | None,
    duration_sec: float = 600.0,
    calibration_seconds: float = 2.0,
    threshold_sigma: float = 4.0,
    fallback_t0_sec: float = 10.0,
    detector: Callable[..., GradientSyncDetection] = detect_gradient_artifact_start,
) -> PreprocessingWindow:
    """Detect T0 with GradientSync and fall back to a fixed 10-second onset."""

    data = np.asarray(signal)
    if data.ndim != 2:
        raise ValueError("signal must have shape channels x samples.")
    detection_error: str | None = None
    try:
        detection = detector(
            data,
            fs,
            tr_sec,
            n_slices,
            channel_names=channel_names,
            calibration_seconds=calibration_seconds,
            threshold_sigma=threshold_sigma,
        )
        start_sec = float(detection.t0_sec)
        source = "gradient_sync"
    except (RuntimeError, ValueError) as error:
        start_sec = float(fallback_t0_sec)
        source = "fallback"
        detection_error = f"{type(error).__name__}: {error}"

    start, stop = fixed_duration_sample_bounds(data.shape[1], fs, start_sec, duration_sec)
    return PreprocessingWindow(
        start_sample=start,
        stop_sample=stop,
        start_sec=start / fs,
        stop_sec=stop / fs,
        duration_sec=(stop - start) / fs,
        t0_source=source,
        detection_error=detection_error,
    )

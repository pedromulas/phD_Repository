from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


_DEFAULT_EXCLUDED_NAME_TOKENS = (
    "ECG",
    "EKG",
    "VREF",
    "TRIG",
    "STI",
    "MISC",
    "RESP",
    "EOG",
    "EMG",
    "AUX",
)


@dataclass(frozen=True)
class GradientSyncDetection:
    t0_sample: int
    t0_sec: float
    t_f_sample: int
    t_f_sec: float
    consensus_peaks: np.ndarray
    consensus_votes: np.ndarray
    consensus_amplitudes: np.ndarray
    threshold: float
    calibration_seconds: float
    calibration_mean: float
    calibration_std: float
    slice_period_sec: float
    tr_sec: float
    n_slices: int
    analysis_channel_indices: tuple[int, ...]
    analysis_channel_names: tuple[str, ...] | None


def _as_2d(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        return signal[np.newaxis, :]
    if signal.ndim != 2:
        raise ValueError("signal must be a 1D or 2D array.")
    return signal


def _default_analysis_channels(channel_names: Sequence[str] | None, n_channels: int) -> tuple[int, ...]:
    if channel_names is None:
        return tuple(range(n_channels))

    selected: list[int] = []
    for index, channel_name in enumerate(channel_names):
        upper_name = str(channel_name).upper()
        if any(token in upper_name for token in _DEFAULT_EXCLUDED_NAME_TOKENS):
            continue
        selected.append(index)

    if not selected:
        return tuple(range(n_channels))
    return tuple(selected)


def _bandpass_feature(signal_1d: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = fs / 2.0
    low = max(0.01, float(low_hz))
    high = min(float(high_hz), nyquist * 0.99)
    if low >= high:
        return np.abs(signal_1d - np.median(signal_1d))

    sos = butter(3, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, signal_1d)
    return np.abs(filtered)


def _cluster_channel_peaks(
    peaks_by_channel: list[np.ndarray],
    values_by_channel: list[np.ndarray],
    tolerance_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_events: list[tuple[int, int, float]] = []
    for channel_index, peak_samples in enumerate(peaks_by_channel):
        peak_values = values_by_channel[channel_index]
        for sample, value in zip(peak_samples.tolist(), peak_values.tolist(), strict=False):
            all_events.append((int(sample), int(channel_index), float(value)))

    if not all_events:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    all_events.sort(key=lambda item: item[0])
    clustered_samples: list[int] = []
    clustered_votes: list[int] = []
    clustered_amplitudes: list[float] = []

    current_samples = [all_events[0][0]]
    current_channels = {all_events[0][1]}
    current_values = [all_events[0][2]]

    for sample, channel_index, value in all_events[1:]:
        if sample - current_samples[-1] <= tolerance_samples:
            current_samples.append(sample)
            current_channels.add(channel_index)
            current_values.append(value)
            continue

        clustered_samples.append(int(round(float(np.median(current_samples)))))
        clustered_votes.append(len(current_channels))
        clustered_amplitudes.append(float(np.median(current_values)))

        current_samples = [sample]
        current_channels = {channel_index}
        current_values = [value]

    clustered_samples.append(int(round(float(np.median(current_samples)))))
    clustered_votes.append(len(current_channels))
    clustered_amplitudes.append(float(np.median(current_values)))

    return (
        np.asarray(clustered_samples, dtype=np.int64),
        np.asarray(clustered_votes, dtype=np.int64),
        np.asarray(clustered_amplitudes, dtype=np.float64),
    )


def _find_periodic_window_index(
    consensus_times: np.ndarray,
    *,
    required_peaks: int,
    expected_slice_period: float,
    period_tolerance: float,
    tr_sec: float,
    period_tolerance_fraction: float,
    reverse: bool = False,
) -> int | None:
    if required_peaks < 2 or consensus_times.size < required_peaks:
        return None

    candidate_range = range(0, consensus_times.size - required_peaks + 1)
    if reverse:
        candidate_range = range(consensus_times.size - required_peaks, -1, -1)

    for index in candidate_range:
        cycle = consensus_times[index : index + required_peaks]
        within_cycle = np.diff(cycle)
        if within_cycle.size == 0:
            continue
        if np.any(np.abs(within_cycle - expected_slice_period) > period_tolerance):
            continue
        if required_peaks > 2:
            total_interval = cycle[-1] - cycle[0]
            if abs(total_interval - tr_sec) > max(period_tolerance, tr_sec * period_tolerance_fraction):
                continue
        return int(index)

    return None


def detect_gradient_artifact_start(
    signal: np.ndarray,
    fs: float,
    tr_sec: float,
    n_slices: int,
    *,
    channel_names: Sequence[str] | None = None,
    calibration_seconds: float = 100.0,
    low_hz: float = 10.0,
    high_hz: float = 200.0,
    threshold_sigma: float = 6.0,
    refractory_ms: float = 20.0,
    cluster_tolerance_ms: float = 8.0,
    period_tolerance_fraction: float = 0.2,
    min_channels_fraction: float = 0.25,
    min_cycle_peaks: int = 4,
) -> GradientSyncDetection:
    """Estimate the first fMRI volume onset from gradient-artifact morphology."""
    if fs <= 0:
        raise ValueError("fs must be strictly positive.")
    if tr_sec <= 0:
        raise ValueError("tr_sec must be strictly positive.")
    if n_slices <= 0:
        raise ValueError("n_slices must be strictly positive.")
    if calibration_seconds <= 0:
        raise ValueError("calibration_seconds must be strictly positive.")
    if min_cycle_peaks < 2:
        raise ValueError("min_cycle_peaks must be at least 2.")

    signal_2d = _as_2d(signal)
    n_channels, n_samples = signal_2d.shape
    analysis_channel_indices = _default_analysis_channels(channel_names, n_channels)
    analysis_channel_names = None if channel_names is None else tuple(str(channel_names[index]) for index in analysis_channel_indices)

    calibration_samples = min(n_samples, max(1, int(round(calibration_seconds * fs))))
    refractory_samples = max(1, int(round(refractory_ms * fs / 1000.0)))
    cluster_tolerance_samples = max(refractory_samples, int(round(cluster_tolerance_ms * fs / 1000.0)))
    expected_slice_period = tr_sec / float(n_slices)
    period_tolerance = max(expected_slice_period * period_tolerance_fraction, 1.0 / fs)
    min_channels = max(1, int(ceil(len(analysis_channel_indices) * min_channels_fraction)))

    channel_features: list[np.ndarray] = []
    thresholds: list[float] = []
    peak_samples_by_channel: list[np.ndarray] = []
    peak_values_by_channel: list[np.ndarray] = []

    for channel_index in analysis_channel_indices:
        feature = _bandpass_feature(signal_2d[channel_index], fs, low_hz, high_hz)
        channel_features.append(feature)
        calibration_feature = feature[:calibration_samples]
        calibration_mean = float(np.mean(calibration_feature))
        calibration_std = float(np.std(calibration_feature))
        threshold = calibration_mean + threshold_sigma * calibration_std
        thresholds.append(threshold)

        peak_samples, peak_properties = find_peaks(feature, height=threshold, distance=refractory_samples)
        peak_samples_by_channel.append(peak_samples.astype(np.int64, copy=False))
        peak_values_by_channel.append(np.asarray(peak_properties.get("peak_heights", []), dtype=np.float64))

    consensus_peaks, consensus_votes, consensus_amplitudes = _cluster_channel_peaks(
        peak_samples_by_channel,
        peak_values_by_channel,
        tolerance_samples=cluster_tolerance_samples,
    )

    if consensus_peaks.size == 0:
        raise ValueError("No consensus gradient peaks were detected.")

    valid_mask = consensus_votes >= min_channels
    consensus_peaks = consensus_peaks[valid_mask]
    consensus_votes = consensus_votes[valid_mask]
    consensus_amplitudes = consensus_amplitudes[valid_mask]

    consensus_times = consensus_peaks / float(fs)
    differences = np.diff(consensus_times)

    if consensus_peaks.size < max(min_cycle_peaks, 2):
        t0_sample = int(consensus_peaks[0])
        t_f_sample = int(consensus_peaks[-1])
        t0_sec = t0_sample / float(fs)
        t_f_sec = t_f_sample / float(fs)
        concatenated_calibration = np.concatenate([feature[:calibration_samples] for feature in channel_features]) if channel_features else np.array([], dtype=np.float64)

        return GradientSyncDetection(
            t0_sample=t0_sample,
            t0_sec=t0_sec,
            t_f_sample=t_f_sample,
            t_f_sec=t_f_sec,
            consensus_peaks=consensus_peaks,
            consensus_votes=consensus_votes,
            consensus_amplitudes=consensus_amplitudes,
            threshold=float(np.mean(thresholds)),
            calibration_seconds=float(calibration_samples / fs),
            calibration_mean=float(np.mean(concatenated_calibration)) if concatenated_calibration.size else 0.0,
            calibration_std=float(np.std(concatenated_calibration)) if concatenated_calibration.size else 0.0,
            slice_period_sec=float(expected_slice_period),
            tr_sec=float(tr_sec),
            n_slices=int(n_slices),
            analysis_channel_indices=analysis_channel_indices,
            analysis_channel_names=analysis_channel_names,
        )

    candidate_index = None
    required_peaks = min(consensus_peaks.size, max(min_cycle_peaks, n_slices + 1))
    if required_peaks >= 2:
        candidate_index = _find_periodic_window_index(
            consensus_times,
            required_peaks=required_peaks,
            expected_slice_period=expected_slice_period,
            period_tolerance=period_tolerance,
            tr_sec=tr_sec,
            period_tolerance_fraction=period_tolerance_fraction,
            reverse=False,
        )
        candidate_index_end = _find_periodic_window_index(
            consensus_times,
            required_peaks=required_peaks,
            expected_slice_period=expected_slice_period,
            period_tolerance=period_tolerance,
            tr_sec=tr_sec,
            period_tolerance_fraction=period_tolerance_fraction,
            reverse=True,
        )
    else:
        candidate_index_end = None

    if candidate_index is None:
        median_interval = float(np.median(differences)) if differences.size else expected_slice_period
        tolerance = max(period_tolerance, median_interval * period_tolerance_fraction)
        for index in range(0, consensus_peaks.size - max(min_cycle_peaks, 2) + 1):
            cycle = consensus_times[index : index + max(min_cycle_peaks, 2)]
            within_cycle = np.diff(cycle)
            if within_cycle.size == 0:
                continue
            if np.all(np.abs(within_cycle - median_interval) <= tolerance):
                candidate_index = index
                break

    if candidate_index is None:
        t0_sample = int(consensus_peaks[0])
        t_f_sample = int(consensus_peaks[-1])
        t0_sec = t0_sample / float(fs)
        t_f_sec = t_f_sample / float(fs)
        concatenated_calibration = np.concatenate([feature[:calibration_samples] for feature in channel_features]) if channel_features else np.array([], dtype=np.float64)

        return GradientSyncDetection(
            t0_sample=t0_sample,
            t0_sec=t0_sec,
            t_f_sample=t_f_sample,
            t_f_sec=t_f_sec,
            consensus_peaks=consensus_peaks,
            consensus_votes=consensus_votes,
            consensus_amplitudes=consensus_amplitudes,
            threshold=float(np.mean(thresholds)),
            calibration_seconds=float(calibration_samples / fs),
            calibration_mean=float(np.mean(concatenated_calibration)) if concatenated_calibration.size else 0.0,
            calibration_std=float(np.std(concatenated_calibration)) if concatenated_calibration.size else 0.0,
            slice_period_sec=float(expected_slice_period),
            tr_sec=float(tr_sec),
            n_slices=int(n_slices),
            analysis_channel_indices=analysis_channel_indices,
            analysis_channel_names=analysis_channel_names,
        )

    t0_sample = int(consensus_peaks[candidate_index])
    if candidate_index_end is None:
        t_f_sample = int(consensus_peaks[-1])
    else:
        t_f_sample = int(consensus_peaks[min(candidate_index_end + required_peaks - 1, consensus_peaks.size - 1)])
    t0_sec = t0_sample / float(fs)
    t_f_sec = t_f_sample / float(fs)
    concatenated_calibration = np.concatenate([feature[:calibration_samples] for feature in channel_features]) if channel_features else np.array([], dtype=np.float64)

    return GradientSyncDetection(
        t0_sample=t0_sample,
        t0_sec=t0_sec,
        t_f_sample=t_f_sample,
        t_f_sec=t_f_sec,
        consensus_peaks=consensus_peaks,
        consensus_votes=consensus_votes,
        consensus_amplitudes=consensus_amplitudes,
        threshold=float(np.mean(thresholds)),
        calibration_seconds=float(calibration_samples / fs),
        calibration_mean=float(np.mean(concatenated_calibration)) if concatenated_calibration.size else 0.0,
        calibration_std=float(np.std(concatenated_calibration)) if concatenated_calibration.size else 0.0,
        slice_period_sec=float(expected_slice_period),
        tr_sec=float(tr_sec),
        n_slices=int(n_slices),
        analysis_channel_indices=analysis_channel_indices,
        analysis_channel_names=analysis_channel_names,
    )

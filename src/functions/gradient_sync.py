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
    t_f_end_sample: int
    t_f_end_sec: float
    t_f_valid: bool
    t_f_match_fraction: float
    t_f_discarded_trailing_peaks: int
    t_f_method: str
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
    t_ga_sample: int
    t_ga_sec: float
    t_ga_end_sample: int
    t_ga_end_sec: float
    t_bold_sample: int
    t_bold_sec: float
    t_bold_end_sample: int
    t_bold_end_sec: float
    energy_window_samples: int
    energy_samples: np.ndarray
    multichannel_energy: np.ndarray
    energy_threshold: float
    t_ga_coarse_sample: int
    t_ga_coarse_sec: float
    t_bold_valid: bool
    t_bold_match_fraction: float


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


def _select_quiet_baseline(
    feature: np.ndarray,
    fs: float,
    calibration_samples: int,
    window_seconds: float = 0.25,
) -> np.ndarray:
    """Use the initial calibration only when it resembles quiet EEG.

    If gradient activity is already present at the beginning, samples from the
    quietest short windows of the complete recording replace the initial
    segment. This avoids treating the first ``calibration_seconds`` as
    unconditionally artefact-free.
    """

    initial = feature[:calibration_samples]
    window_samples = max(1, int(round(window_seconds * fs)))
    starts = np.arange(0, feature.size, window_samples, dtype=np.int64)
    window_medians = np.asarray(
        [np.median(feature[start : min(start + window_samples, feature.size)]) for start in starts]
    )
    quiet_limit = float(np.quantile(window_medians, 0.05))
    quiet_window_indices = np.flatnonzero(window_medians <= quiet_limit)
    quiet_segments = [
        feature[starts[index] : min(starts[index] + window_samples, feature.size)]
        for index in quiet_window_indices.tolist()
    ]
    quiet = np.concatenate(quiet_segments) if quiet_segments else initial
    if np.median(initial) <= np.median(quiet) * 1.5:
        return initial
    return quiet


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
        # Compare with the cluster origin rather than the previous event.
        # Otherwise a dense chain of nearby detections can bridge unrelated
        # slice artefacts into one cluster spanning seconds or minutes.
        if sample - current_samples[0] <= tolerance_samples:
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


def _periodic_grid_match_fraction(
    consensus_peaks: np.ndarray,
    *,
    candidate_index: int,
    expected_period_samples: float,
    tolerance_samples: float,
    validation_samples: int,
    min_expected_peaks: int,
) -> float:
    """Return the fraction of expected slice slots matched near a candidate.

    Missing slice peaks are tolerated: observed peaks are projected onto the
    nearest position of the expected ``TR / n_slices`` grid and each grid slot
    can contribute at most one match.
    """

    candidate_sample = int(consensus_peaks[candidate_index])
    available_samples = min(validation_samples, int(consensus_peaks[-1]) - candidate_sample)
    expected_intervals = int(np.floor(available_samples / expected_period_samples))
    expected_count = expected_intervals + 1
    if expected_count < min_expected_peaks:
        return 0.0

    evaluation_end = candidate_sample + expected_intervals * expected_period_samples + tolerance_samples
    observed_end = int(np.searchsorted(consensus_peaks, evaluation_end, side="right"))
    observed = consensus_peaks[candidate_index:observed_end]
    relative = observed.astype(np.float64) - candidate_sample
    slots = np.rint(relative / expected_period_samples).astype(np.int64)
    valid_slots = (slots >= 0) & (slots <= expected_intervals)
    residuals = relative - slots * expected_period_samples
    matched_slots = np.unique(slots[valid_slots & (np.abs(residuals) <= tolerance_samples)])
    return float(matched_slots.size / expected_count)


def _find_local_periodic_grid_candidate(
    consensus_peaks: np.ndarray,
    *,
    t_ga_sample: int,
    fs: float,
    tr_sec: float,
    expected_slice_period: float,
    period_tolerance: float,
    min_cycle_peaks: int,
    min_match_fraction: float,
    search_trs: float,
    validation_trs: float,
) -> tuple[int | None, float, bool]:
    """Find the earliest periodic candidate close to the detected GA onset."""

    if consensus_peaks.size == 0:
        return None, 0.0, False

    expected_period_samples = expected_slice_period * fs
    tolerance_samples = period_tolerance * fs
    search_end = t_ga_sample + int(round(search_trs * tr_sec * fs))
    candidate_indices = np.flatnonzero(
        (consensus_peaks >= t_ga_sample) & (consensus_peaks <= search_end)
    )
    if candidate_indices.size == 0:
        return None, 0.0, False

    validation_samples = max(1, int(round(validation_trs * tr_sec * fs)))
    best_index = int(candidate_indices[0])
    best_fraction = -1.0
    for candidate_index in candidate_indices.tolist():
        match_fraction = _periodic_grid_match_fraction(
            consensus_peaks,
            candidate_index=int(candidate_index),
            expected_period_samples=expected_period_samples,
            tolerance_samples=tolerance_samples,
            validation_samples=validation_samples,
            min_expected_peaks=min_cycle_peaks,
        )
        if match_fraction >= min_match_fraction:
            return int(candidate_index), match_fraction, True
        if match_fraction > best_fraction:
            best_index = int(candidate_index)
            best_fraction = match_fraction

    return best_index, max(best_fraction, 0.0), False


def _track_connected_periodic_run_end(
    consensus_peaks: np.ndarray,
    *,
    t0_sample: int,
    expected_period_samples: float,
    tolerance_samples: float,
    validation_samples: int,
    min_expected_peaks: int,
    min_match_fraction: float,
) -> tuple[int | None, float, bool]:
    """Track the periodic peak run connected to T0 and return its last peak.

    Candidates are validated forward over ``validation_samples``. Once no
    valid candidate has appeared for a complete validation interval, tracking
    stops, so a later unrelated periodic block cannot replace the original GA
    run. The final partial block is recovered from the last valid local grid.
    """

    if consensus_peaks.size == 0:
        return None, 0.0, False
    start_index = int(np.searchsorted(consensus_peaks, t0_sample, side="left"))
    if start_index >= consensus_peaks.size:
        return None, 0.0, False

    last_valid_index: int | None = None
    last_valid_fraction = 0.0
    for candidate_index in range(start_index, consensus_peaks.size):
        candidate_sample = int(consensus_peaks[candidate_index])
        if (
            last_valid_index is not None
            and candidate_sample - int(consensus_peaks[last_valid_index]) > validation_samples
        ):
            break
        if last_valid_index is None and candidate_sample - t0_sample > validation_samples:
            break

        fraction = _periodic_grid_match_fraction(
            consensus_peaks,
            candidate_index=candidate_index,
            expected_period_samples=expected_period_samples,
            tolerance_samples=tolerance_samples,
            validation_samples=validation_samples,
            min_expected_peaks=min_expected_peaks,
        )
        if fraction >= min_match_fraction:
            last_valid_index = candidate_index
            last_valid_fraction = fraction

    if last_valid_index is None:
        return None, 0.0, False

    anchor_sample = int(consensus_peaks[last_valid_index])
    recovery_end = anchor_sample + validation_samples
    observed_end = int(np.searchsorted(consensus_peaks, recovery_end + tolerance_samples, side="right"))
    observed = consensus_peaks[last_valid_index:observed_end]
    relative = observed.astype(np.float64) - anchor_sample
    slots = np.rint(relative / expected_period_samples).astype(np.int64)
    residuals = relative - slots * expected_period_samples
    valid = (
        (slots >= 0)
        & (relative <= validation_samples + tolerance_samples)
        & (np.abs(residuals) <= tolerance_samples)
    )
    matched = observed[valid]
    if matched.size == 0:
        return anchor_sample, last_valid_fraction, False
    return int(matched[-1]), last_valid_fraction, True


def _persistent_activity_bounds(
    channel_features: list[np.ndarray],
    fs: float,
    tr_sec: float,
    calibration_seconds: float,
    threshold_sigma: float,
    window_seconds: float = 0.25,
) -> tuple[int, int, int, np.ndarray, np.ndarray, float]:
    """Detect the first/last sustained GA-energy interval across EEG channels.

    The baseline falls back to the quietest 5% of the whole recording when
    the initial calibration window already contains gradient artefacts.
    """
    if not channel_features:
        raise ValueError("At least one analysis channel is required.")
    n_samples = channel_features[0].size
    window_samples = max(1, int(round(window_seconds * fs)))
    scales = np.asarray([max(float(np.quantile(feature, 0.05)), np.finfo(float).eps) for feature in channel_features])
    normalized = np.median(np.vstack([feature / scale for feature, scale in zip(channel_features, scales, strict=True)]), axis=0)
    starts = np.arange(0, n_samples, window_samples, dtype=np.int64)
    energy = np.asarray([np.median(normalized[start : min(start + window_samples, n_samples)]) for start in starts])
    energy_samples = np.minimum(starts + window_samples // 2, n_samples - 1)

    calibration_windows = max(1, int(np.ceil(calibration_seconds * fs / window_samples)))
    initial = energy[:calibration_windows]
    global_quiet = energy[energy <= np.quantile(energy, 0.05)]
    # Prefer the intended initial baseline only when it is comparable to the
    # quietest part of the recording; otherwise fMRI had already begun there.
    baseline = initial if np.median(initial) <= np.median(global_quiet) * 1.5 else global_quiet
    center = float(np.median(baseline))
    robust_std = float(1.4826 * np.median(np.abs(baseline - center)))
    threshold = center + threshold_sigma * max(robust_std, np.finfo(float).eps)
    active = energy > threshold

    minimum_windows = max(2, int(np.ceil(max(2.0 * tr_sec, 1.0) * fs / window_samples)))
    required_active = int(np.ceil(0.75 * minimum_windows))
    activity_count = np.convolve(active.astype(int), np.ones(minimum_windows, dtype=int), mode="valid")
    valid_starts = np.flatnonzero(activity_count >= required_active)
    if valid_starts.size == 0:
        raise ValueError("No sustained high-frequency GA interval was detected.")
    confirmation_start = int(valid_starts[0])
    confirmation = active[confirmation_start : confirmation_start + minimum_windows]
    first_active = np.flatnonzero(confirmation)
    if first_active.size == 0:
        raise ValueError("The sustained GA interval contained no active energy window.")
    # The look-ahead window confirms persistence, but its beginning can precede
    # the true threshold crossing by up to the allowed inactive fraction.
    start_window = confirmation_start + int(first_active[0])
    last_confirmation_start = int(valid_starts[-1])
    last_confirmation = active[
        last_confirmation_start : last_confirmation_start + minimum_windows
    ]
    last_active = np.flatnonzero(last_confirmation)
    if last_active.size == 0:
        raise ValueError("The final sustained GA interval contained no active energy window.")
    # As at the onset, the persistence window only confirms the transition.
    # Its allowed inactive fraction must not extend the reported GA end.
    end_window = last_confirmation_start + int(last_active[-1])
    return (
        int(starts[start_window]),
        int(min(n_samples, starts[end_window] + window_samples)),
        window_samples,
        energy_samples,
        energy,
        float(threshold),
    )


def detect_gradient_artifact_start(
    signal: np.ndarray,
    fs: float,
    tr_sec: float,
    n_slices: int,
    *,
    channel_names: Sequence[str] | None = None,
    calibration_seconds: float = 2.0,
    low_hz: float = 10.0,
    high_hz: float = 200.0,
    threshold_sigma: float = 6.0,
    refractory_ms: float = 20.0,
    cluster_tolerance_ms: float = 8.0,
    period_tolerance_fraction: float = 0.2,
    min_channels_fraction: float = 0.25,
    min_cycle_peaks: int = 4,
    min_grid_match_fraction: float = 0.7,
    bold_search_trs: float = 1.0,
    grid_validation_trs: float = 2.0,
    tracking_min_channels_fraction: float = 0.25,
) -> GradientSyncDetection:
    """Estimate sustained GA bounds and the stable BOLD acquisition interval.

    ``t_ga_coarse_*`` marks the energy-window estimate and ``t_ga_*`` refines
    it to the first multichannel GS peak. ``t_bold_*`` is selected only within
    ``bold_search_trs`` of T_GA using a slice-period grid that tolerates missing
    peaks. T_F follows the connected multichannel periodic run from T0 across
    the complete recording; an energy-bounded reverse check is retained as a
    safe fallback when the stricter tracking consensus loses the weak tail.
    Legacy ``t0_*`` and ``t_f_*`` alias the BOLD bounds.
    """
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
    if not 0 < min_grid_match_fraction <= 1:
        raise ValueError("min_grid_match_fraction must be in (0, 1].")
    if bold_search_trs <= 0 or grid_validation_trs <= 0:
        raise ValueError("bold_search_trs and grid_validation_trs must be strictly positive.")
    if not 0 < tracking_min_channels_fraction <= 1:
        raise ValueError("tracking_min_channels_fraction must be in (0, 1].")

    signal_2d = _as_2d(signal)
    n_channels, n_samples = signal_2d.shape
    analysis_channel_indices = _default_analysis_channels(channel_names, n_channels)
    analysis_channel_names = None if channel_names is None else tuple(str(channel_names[index]) for index in analysis_channel_indices)

    calibration_samples = min(n_samples, max(1, int(round(calibration_seconds * fs))))
    refractory_samples = max(1, int(round(refractory_ms * fs / 1000.0)))
    cluster_tolerance_samples = max(refractory_samples, int(round(cluster_tolerance_ms * fs / 1000.0)))
    expected_slice_period = tr_sec / float(n_slices)
    period_tolerance = max(expected_slice_period * period_tolerance_fraction, 1.0 / fs)
    peak_detection_distance = max(
        refractory_samples,
        int(np.floor((expected_slice_period - period_tolerance) * fs)),
    )
    min_channels = max(1, int(ceil(len(analysis_channel_indices) * min_channels_fraction)))

    channel_features: list[np.ndarray] = []
    calibration_baselines: list[np.ndarray] = []
    thresholds: list[float] = []
    peak_samples_by_channel: list[np.ndarray] = []
    peak_values_by_channel: list[np.ndarray] = []

    for channel_index in analysis_channel_indices:
        feature = _bandpass_feature(signal_2d[channel_index], fs, low_hz, high_hz)
        channel_features.append(feature)
        calibration_feature = _select_quiet_baseline(feature, fs, calibration_samples)
        calibration_baselines.append(calibration_feature)
        calibration_mean = float(np.mean(calibration_feature))
        calibration_std = float(np.std(calibration_feature))
        threshold = calibration_mean + threshold_sigma * calibration_std
        thresholds.append(threshold)

        peak_samples, peak_properties = find_peaks(
            feature,
            height=threshold,
            distance=peak_detection_distance,
        )
        peak_samples_by_channel.append(peak_samples.astype(np.int64, copy=False))
        peak_values_by_channel.append(np.asarray(peak_properties.get("peak_heights", []), dtype=np.float64))

    (
        t_ga_coarse_sample,
        t_ga_end_sample,
        energy_window_samples,
        energy_samples,
        multichannel_energy,
        energy_threshold,
    ) = _persistent_activity_bounds(
        channel_features,
        fs,
        tr_sec,
        calibration_seconds,
        threshold_sigma,
    )

    t_ga_sample = t_ga_coarse_sample
    concatenated_calibration = np.concatenate(calibration_baselines) if calibration_baselines else np.array([], dtype=np.float64)

    def make_detection(
        t_bold_sample: int,
        t_bold_end_sample: int,
        *,
        t_bold_valid: bool = False,
        t_bold_match_fraction: float = 0.0,
        t_f_valid: bool = False,
        t_f_match_fraction: float = 0.0,
        t_f_discarded_trailing_peaks: int = 0,
        t_f_method: str = "energy_interval",
    ) -> GradientSyncDetection:
        t_f_end_sample = min(
            n_samples,
            int(round(t_bold_end_sample + expected_slice_period * fs)),
        )
        return GradientSyncDetection(
            t0_sample=t_bold_sample,
            t0_sec=t_bold_sample / float(fs),
            t_f_sample=t_bold_end_sample,
            t_f_sec=t_bold_end_sample / float(fs),
            t_f_end_sample=t_f_end_sample,
            t_f_end_sec=t_f_end_sample / float(fs),
            t_f_valid=t_f_valid,
            t_f_match_fraction=t_f_match_fraction,
            t_f_discarded_trailing_peaks=t_f_discarded_trailing_peaks,
            t_f_method=t_f_method,
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
            t_ga_sample=t_ga_sample,
            t_ga_sec=t_ga_sample / float(fs),
            t_ga_end_sample=t_ga_end_sample,
            t_ga_end_sec=t_ga_end_sample / float(fs),
            t_bold_sample=t_bold_sample,
            t_bold_sec=t_bold_sample / float(fs),
            t_bold_end_sample=t_bold_end_sample,
            t_bold_end_sec=t_bold_end_sample / float(fs),
            energy_window_samples=energy_window_samples,
            energy_samples=energy_samples,
            multichannel_energy=multichannel_energy,
            energy_threshold=energy_threshold,
            t_ga_coarse_sample=t_ga_coarse_sample,
            t_ga_coarse_sec=t_ga_coarse_sample / float(fs),
            t_bold_valid=t_bold_valid,
            t_bold_match_fraction=t_bold_match_fraction,
        )

    consensus_peaks, consensus_votes, consensus_amplitudes = _cluster_channel_peaks(
        peak_samples_by_channel,
        peak_values_by_channel,
        tolerance_samples=cluster_tolerance_samples,
    )

    if consensus_peaks.size == 0:
        empty = np.array([], dtype=np.float64)
        consensus_peaks = np.array([], dtype=np.int64)
        consensus_votes = np.array([], dtype=np.int64)
        consensus_amplitudes = empty
        return make_detection(t_ga_sample, t_ga_end_sample)

    valid_mask = consensus_votes >= min_channels
    consensus_peaks = consensus_peaks[valid_mask]
    consensus_votes = consensus_votes[valid_mask]
    consensus_amplitudes = consensus_amplitudes[valid_mask]

    if consensus_peaks.size == 0:
        return make_detection(t_ga_sample, t_ga_end_sample)

    # Keep a stricter multichannel series for T_F tracking. T0 may use a low
    # vote threshold for sensitivity, whereas a long connected tail needs
    # stronger spatial consensus to avoid chaining background noise.
    tracking_min_channels = max(
        min_channels,
        int(ceil(len(analysis_channel_indices) * tracking_min_channels_fraction)),
    )
    tracking_peaks = consensus_peaks[consensus_votes >= tracking_min_channels].copy()

    # Refine the energy-window estimate to the first multichannel peak inside
    # the sustained interval. This removes the look-ahead bias introduced by
    # persistence confirmation and restores sample-level precision.
    active_peaks = (consensus_peaks >= t_ga_coarse_sample) & (consensus_peaks <= t_ga_end_sample)
    consensus_peaks = consensus_peaks[active_peaks]
    consensus_votes = consensus_votes[active_peaks]
    consensus_amplitudes = consensus_amplitudes[active_peaks]
    if consensus_peaks.size == 0:
        return make_detection(t_ga_coarse_sample, t_ga_end_sample)

    t_ga_sample = int(consensus_peaks[0])

    if consensus_peaks.size < max(min_cycle_peaks, 2):
        return make_detection(int(consensus_peaks[0]), int(consensus_peaks[-1]))

    candidate_index, match_fraction, is_valid = _find_local_periodic_grid_candidate(
        consensus_peaks,
        t_ga_sample=t_ga_sample,
        fs=fs,
        tr_sec=tr_sec,
        expected_slice_period=expected_slice_period,
        period_tolerance=period_tolerance,
        min_cycle_peaks=min_cycle_peaks,
        min_match_fraction=min_grid_match_fraction,
        search_trs=bold_search_trs,
        validation_trs=grid_validation_trs,
    )
    if candidate_index is None:
        return make_detection(t_ga_sample, t_ga_end_sample)

    t_bold_sample = int(consensus_peaks[candidate_index])
    validation_samples = max(1, int(round(grid_validation_trs * tr_sec * fs)))
    tracked_end_sample, tracking_fraction, tracking_is_valid = _track_connected_periodic_run_end(
        tracking_peaks,
        t0_sample=t_bold_sample,
        expected_period_samples=expected_slice_period * fs,
        tolerance_samples=period_tolerance * fs,
        validation_samples=validation_samples,
        min_expected_peaks=min_cycle_peaks,
        min_match_fraction=min_grid_match_fraction,
    )
    if tracked_end_sample is None:
        t_bold_end_sample = int(consensus_peaks[-1])
        tail_match_fraction = 0.0
        tail_is_valid = False
    else:
        t_bold_end_sample = int(tracked_end_sample)
        peaks_through_end = tracking_peaks[tracking_peaks <= t_bold_end_sample]
        reverse_peaks = (
            t_bold_end_sample - peaks_through_end[::-1]
        ).astype(np.int64, copy=False)
        tail_match_fraction = _periodic_grid_match_fraction(
            reverse_peaks,
            candidate_index=0,
            expected_period_samples=expected_slice_period * fs,
            tolerance_samples=period_tolerance * fs,
            validation_samples=validation_samples,
            min_expected_peaks=min_cycle_peaks,
        )
        tail_is_valid = (
            tracking_is_valid
            and tracking_fraction >= min_grid_match_fraction
            and tail_match_fraction >= min_grid_match_fraction
        )
    t_f_method = "connected_periodic_tracking"

    if not tail_is_valid:
        # A strict tracking vote threshold can lose a weak but genuine final
        # slice. Fall back to the previous energy-bounded reverse validation
        # rather than returning a newly invalid or prematurely early T_F.
        energy_reverse_peaks = (
            consensus_peaks[-1] - consensus_peaks[::-1]
        ).astype(np.int64, copy=False)
        fallback_index, fallback_fraction, fallback_is_valid = _find_local_periodic_grid_candidate(
            energy_reverse_peaks,
            t_ga_sample=0,
            fs=fs,
            tr_sec=tr_sec,
            expected_slice_period=expected_slice_period,
            period_tolerance=period_tolerance,
            min_cycle_peaks=min_cycle_peaks,
            min_match_fraction=min_grid_match_fraction,
            search_trs=bold_search_trs,
            validation_trs=grid_validation_trs,
        )
        if fallback_index is not None and fallback_is_valid:
            original_index = consensus_peaks.size - 1 - int(fallback_index)
            t_bold_end_sample = int(consensus_peaks[original_index])
            tail_match_fraction = fallback_fraction
            tail_is_valid = True
            t_f_method = "energy_tail_fallback"
    return make_detection(
        t_bold_sample,
        t_bold_end_sample,
        t_bold_valid=is_valid,
        t_bold_match_fraction=match_fraction,
        t_f_valid=tail_is_valid,
        t_f_match_fraction=tail_match_fraction,
        t_f_discarded_trailing_peaks=int(np.sum(tracking_peaks > t_bold_end_sample)),
        t_f_method=t_f_method,
    )

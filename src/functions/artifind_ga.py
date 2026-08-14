"""Artifind-style gradient artefact onset detection for simultaneous EEG-fMRI.

This module ports the gradient-switch (GS) detector described by Nuttall et al.
(2023, MethodsX, doi:10.1016/j.mex.2023.102376) and its public MATLAB
implementation.  It deliberately remains separate from ``gradient_sync`` so
that the two approaches can be compared without sharing detection state.

Artifind labels a reproducible peak of each GS artefact.  Consequently, the
reported ``t0`` is the first retained artefact peak, not a hardware guarantee
of the physical MRI volume-onset time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.signal import find_peaks


TriggerType = Literal["volume", "slice"]
TriggerMode = Literal["auto", "volume", "slice"]


@dataclass(frozen=True)
class ArtifindGradientDetection:
    """Result returned by :func:`detect_artifind_gradient_onsets`."""

    t0_sample: int
    t0_sec: float
    t_f_sample: int
    t_f_sec: float
    t_f_end_sample: int
    t_f_end_sec: float
    t_f_valid: bool
    t_f_error_samples: float
    ga_onsets: np.ndarray
    ga_onsets_sec: np.ndarray
    detected_onsets_with_dummies: np.ndarray
    trigger_type: TriggerType
    selected_channel_index: int
    selected_channel_name: str | None
    threshold: float
    initial_threshold: float
    threshold_iteration: int
    expected_period_samples: float
    interval_mode_samples: int
    expected_onset_count: int
    detected_onset_count: int
    count_error: int
    period_error_samples: float
    meets_artifind_criteria: bool
    n_dummy_scans: int
    local_onset_valid: bool
    local_match_fraction: float
    discarded_leading_onsets: int
    discarded_trailing_onsets: int


@dataclass(frozen=True)
class _Candidate:
    onsets: np.ndarray
    trigger_type: TriggerType
    channel_index: int
    threshold: float
    threshold_iteration: int
    expected_period_samples: float
    interval_mode_samples: int
    expected_count_with_dummies: int
    count_error: int
    period_error_samples: float
    valid: bool


def _as_2d(signal: np.ndarray) -> np.ndarray:
    data = np.asarray(signal, dtype=np.float64)
    if data.ndim == 1:
        return data[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError("signal must be a 1D or 2D array shaped channels x samples.")
    return data


def _mode_interval(onsets: np.ndarray) -> int | None:
    differences = np.diff(onsets)
    if differences.size == 0:
        return None
    values, counts = np.unique(differences, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _threshold_schedule(
    initial_threshold: float,
    *,
    max_threshold_reductions: int,
    threshold_step: float | None,
) -> np.ndarray:
    if threshold_step is None:
        # The MATLAB release lowers the threshold by 100 amplifier units on
        # each iteration. EEG loaded by MNE is normally expressed in volts,
        # so an amplitude-relative step preserves the same descending search
        # without assuming a particular storage unit.
        threshold_step = abs(initial_threshold) / max(max_threshold_reductions, 1)
    if threshold_step <= 0:
        raise ValueError("threshold_step must be strictly positive.")
    return initial_threshold - threshold_step * np.arange(max_threshold_reductions + 1)


def _candidate_score(candidate: _Candidate) -> tuple[int, float, int, int]:
    """Rank fallback candidates deterministically when strict detection fails."""

    normalized_period_error = candidate.period_error_samples / max(candidate.expected_period_samples, 1.0)
    return (
        candidate.count_error,
        normalized_period_error,
        candidate.threshold_iteration,
        candidate.channel_index,
    )


def _periodic_run_match_fraction(
    onsets: np.ndarray,
    *,
    candidate_position: int,
    expected_period_samples: float,
    tolerance_samples: float,
    validation_samples: float,
) -> tuple[float, np.ndarray]:
    """Measure how well peaks after one candidate follow the expected grid."""

    expected_count = max(2, int(np.ceil(validation_samples / expected_period_samples)))
    targets = (
        float(onsets[candidate_position])
        + np.arange(expected_count, dtype=np.float64) * expected_period_samples
    )
    remaining = onsets[candidate_position:].astype(np.float64, copy=False)
    insertion_points = np.searchsorted(remaining, targets)
    matched = np.zeros(expected_count, dtype=bool)

    for target_index, (target, insertion_point) in enumerate(
        zip(targets.tolist(), insertion_points.tolist(), strict=True)
    ):
        neighbour_positions = []
        if insertion_point < remaining.size:
            neighbour_positions.append(insertion_point)
        if insertion_point > 0:
            neighbour_positions.append(insertion_point - 1)
        if neighbour_positions:
            closest_error = min(abs(float(remaining[position]) - target) for position in neighbour_positions)
            matched[target_index] = closest_error <= tolerance_samples

    return float(np.mean(matched)), matched


def _find_periodic_run_start(
    onsets: np.ndarray,
    *,
    expected_period_samples: float,
    tolerance_samples: float,
    validation_samples: float,
    min_match_fraction: float,
    min_consecutive_peaks: int,
) -> tuple[int | None, float]:
    """Return the first peak that starts a locally periodic acquisition run."""

    if onsets.size < 2:
        return None, 0.0

    best_fraction = 0.0
    for candidate_position in range(onsets.size - 1):
        match_fraction, matched = _periodic_run_match_fraction(
            onsets,
            candidate_position=candidate_position,
            expected_period_samples=expected_period_samples,
            tolerance_samples=tolerance_samples,
            validation_samples=validation_samples,
        )
        best_fraction = max(best_fraction, match_fraction)
        required_prefix = min(min_consecutive_peaks, matched.size)
        prefix_is_periodic = bool(np.all(matched[:required_prefix]))
        if prefix_is_periodic and match_fraction >= min_match_fraction:
            return candidate_position, match_fraction

    return None, best_fraction


def detect_artifind_gradient_onsets(
    signal: np.ndarray,
    fs: float,
    tr_sec: float,
    n_volumes: int,
    n_slices: int,
    *,
    n_dummy_scans: int = 0,
    channel_names: Sequence[str] | None = None,
    trigger_mode: TriggerMode = "auto",
    count_tolerance: int = 7,
    period_tolerance_samples: float = 2.0,
    max_threshold_reductions: int = 30,
    threshold_step: float | None = None,
    allow_approximate: bool = False,
    local_validation_trs: float = 2.0,
    local_min_match_fraction: float = 0.75,
    local_min_consecutive_peaks: int = 4,
) -> ArtifindGradientDetection:
    """Detect volume- or slice-wise GS peaks using the Artifind procedure.

    The search starts at the median of the per-channel maxima, lowers the
    threshold iteratively, and tests each EEG channel. Volume peaks are tried
    before slice peaks in ``auto`` mode, matching the public MATLAB code.

    Parameters
    ----------
    signal
        EEG array shaped ``channels x samples`` (or one one-dimensional
        channel). No filtering or absolute-value transform is applied because
        Artifind operates on the original positive artefact peaks.
    fs, tr_sec, n_volumes, n_slices
        EEG sampling frequency and MRI acquisition parameters.
    n_dummy_scans
        Number of complete volumes preceding the retained fMRI volumes.
    trigger_mode
        Try volume peaks, slice peaks, or both (volume first).
    allow_approximate
        If true, return the closest periodic candidate when no channel meets
        both the published period and count constraints. The result exposes
        this through ``meets_artifind_criteria=False``.
    local_validation_trs, local_min_match_fraction
        Require the selected onset to match the expected volume/slice grid
        over this many TRs. This rejects isolated peaks before the true GA run.
    local_min_consecutive_peaks
        Number of grid positions that must be present at the beginning of a
        slice-wise run. Volume-wise detection requires two consecutive peaks.

    Notes
    -----
    With slice-wise detection, one dummy volume contributes ``n_slices``
    artefact peaks. This corrects an inconsistency in the released MATLAB code
    and has no effect when ``n_dummy_scans=0``.
    """

    if fs <= 0 or tr_sec <= 0:
        raise ValueError("fs and tr_sec must be strictly positive.")
    if n_volumes <= 0 or n_slices <= 0:
        raise ValueError("n_volumes and n_slices must be strictly positive.")
    if n_dummy_scans < 0:
        raise ValueError("n_dummy_scans cannot be negative.")
    if trigger_mode not in {"auto", "volume", "slice"}:
        raise ValueError("trigger_mode must be 'auto', 'volume', or 'slice'.")
    if count_tolerance < 0 or period_tolerance_samples < 0:
        raise ValueError("Detection tolerances cannot be negative.")
    if max_threshold_reductions < 0:
        raise ValueError("max_threshold_reductions cannot be negative.")
    if local_validation_trs <= 0:
        raise ValueError("local_validation_trs must be strictly positive.")
    if not 0 < local_min_match_fraction <= 1:
        raise ValueError("local_min_match_fraction must be in (0, 1].")
    if local_min_consecutive_peaks < 2:
        raise ValueError("local_min_consecutive_peaks must be at least 2.")

    data = _as_2d(signal)
    n_channels, n_samples = data.shape
    if n_samples < 3:
        raise ValueError("signal is too short for peak detection.")
    if channel_names is not None and len(channel_names) != n_channels:
        raise ValueError("channel_names must match the number of signal channels.")

    channel_maxima = np.max(data, axis=1)
    initial_threshold = float(np.median(channel_maxima))
    if not np.isfinite(initial_threshold):
        raise ValueError("signal contains no finite channel maxima.")
    if initial_threshold <= 0:
        initial_threshold = float(np.max(channel_maxima))
    if initial_threshold <= 0:
        raise ValueError("Artifind requires positive gradient-artefact peaks.")

    thresholds = _threshold_schedule(
        initial_threshold,
        max_threshold_reductions=max_threshold_reductions,
        threshold_step=threshold_step,
    )
    trigger_types: tuple[TriggerType, ...]
    trigger_types = ("volume", "slice") if trigger_mode == "auto" else (trigger_mode,)
    best_candidate: _Candidate | None = None

    for trigger_type in trigger_types:
        if trigger_type == "volume":
            expected_period = tr_sec * fs
            expected_count_with_dummies = n_volumes + n_dummy_scans
        else:
            expected_period = tr_sec * fs / float(n_slices)
            expected_count_with_dummies = (n_volumes + n_dummy_scans) * n_slices

        minimum_distance = max(1, int(np.floor(expected_period - period_tolerance_samples)))

        for threshold_iteration, threshold in enumerate(thresholds):
            for channel_index in range(n_channels):
                onsets, _ = find_peaks(
                    data[channel_index],
                    height=float(threshold),
                    distance=minimum_distance,
                )
                onsets = onsets.astype(np.int64, copy=False)
                interval_mode = _mode_interval(onsets)
                if interval_mode is None:
                    continue

                count_error = abs(int(onsets.size) - expected_count_with_dummies)
                period_error = abs(float(interval_mode) - expected_period)
                valid = count_error <= count_tolerance and period_error <= period_tolerance_samples
                candidate = _Candidate(
                    onsets=onsets,
                    trigger_type=trigger_type,
                    channel_index=channel_index,
                    threshold=float(threshold),
                    threshold_iteration=threshold_iteration,
                    expected_period_samples=float(expected_period),
                    interval_mode_samples=interval_mode,
                    expected_count_with_dummies=expected_count_with_dummies,
                    count_error=count_error,
                    period_error_samples=period_error,
                    valid=valid,
                )

                if best_candidate is None or _candidate_score(candidate) < _candidate_score(best_candidate):
                    best_candidate = candidate
                if valid:
                    best_candidate = candidate
                    break
            if best_candidate is not None and best_candidate.valid:
                break
        if best_candidate is not None and best_candidate.valid:
            break

    if best_candidate is None or (not best_candidate.valid and not allow_approximate):
        details = "No peaks were detected." if best_candidate is None else (
            f"Best candidate had count error {best_candidate.count_error} and "
            f"period error {best_candidate.period_error_samples:.3f} samples."
        )
        raise ValueError(f"Artifind criteria were not satisfied. {details}")

    minimum_consecutive = 2 if best_candidate.trigger_type == "volume" else local_min_consecutive_peaks
    periodic_start, local_match_fraction = _find_periodic_run_start(
        best_candidate.onsets,
        expected_period_samples=best_candidate.expected_period_samples,
        tolerance_samples=period_tolerance_samples,
        validation_samples=local_validation_trs * tr_sec * fs,
        min_match_fraction=local_min_match_fraction,
        min_consecutive_peaks=minimum_consecutive,
    )
    if periodic_start is None:
        raise ValueError(
            "No detected peak started a locally periodic GA sequence "
            f"({local_match_fraction:.1%} was the best {local_validation_trs:g}-TR grid match)."
        )

    aligned_onsets = best_candidate.onsets[periodic_start:]
    dummy_onsets = n_dummy_scans if best_candidate.trigger_type == "volume" else n_dummy_scans * n_slices
    retained_onsets = aligned_onsets[dummy_onsets:]
    if retained_onsets.size == 0:
        raise ValueError("Removing dummy scans left no gradient-artefact onsets.")

    selected_channel_name = None if channel_names is None else str(channel_names[best_candidate.channel_index])
    t0_sample = int(retained_onsets[0])
    expected_onset_count = n_volumes if best_candidate.trigger_type == "volume" else n_volumes * n_slices
    theoretical_t_f = t0_sample + (expected_onset_count - 1) * best_candidate.expected_period_samples
    nearest_t_f_position = int(np.argmin(np.abs(retained_onsets.astype(np.float64) - theoretical_t_f)))
    nearest_t_f_sample = int(retained_onsets[nearest_t_f_position])
    t_f_error_samples = abs(float(nearest_t_f_sample) - theoretical_t_f)
    t_f_valid = t_f_error_samples <= period_tolerance_samples
    # When no detected peak is close enough, retain the metadata-derived end
    # rather than snapping T_F to an unrelated noise peak.
    t_f_sample = nearest_t_f_sample if t_f_valid else int(
        np.clip(round(theoretical_t_f), 0, n_samples - 1)
    )
    t_f_end_sample = min(n_samples, int(round(t_f_sample + best_candidate.expected_period_samples)))
    discarded_trailing_onsets = int(np.sum(retained_onsets > t_f_sample + period_tolerance_samples))
    return ArtifindGradientDetection(
        t0_sample=t0_sample,
        t0_sec=t0_sample / float(fs),
        t_f_sample=t_f_sample,
        t_f_sec=t_f_sample / float(fs),
        t_f_end_sample=t_f_end_sample,
        t_f_end_sec=t_f_end_sample / float(fs),
        t_f_valid=t_f_valid,
        t_f_error_samples=t_f_error_samples,
        ga_onsets=retained_onsets,
        ga_onsets_sec=retained_onsets / float(fs),
        detected_onsets_with_dummies=aligned_onsets,
        trigger_type=best_candidate.trigger_type,
        selected_channel_index=best_candidate.channel_index,
        selected_channel_name=selected_channel_name,
        threshold=best_candidate.threshold,
        initial_threshold=initial_threshold,
        threshold_iteration=best_candidate.threshold_iteration,
        expected_period_samples=best_candidate.expected_period_samples,
        interval_mode_samples=best_candidate.interval_mode_samples,
        expected_onset_count=expected_onset_count,
        detected_onset_count=int(retained_onsets.size),
        count_error=abs(
            int(retained_onsets.size)
            - (n_volumes if best_candidate.trigger_type == "volume" else n_volumes * n_slices)
        ),
        period_error_samples=best_candidate.period_error_samples,
        meets_artifind_criteria=best_candidate.valid,
        n_dummy_scans=n_dummy_scans,
        local_onset_valid=True,
        local_match_fraction=local_match_fraction,
        discarded_leading_onsets=int(periodic_start),
        discarded_trailing_onsets=discarded_trailing_onsets,
    )

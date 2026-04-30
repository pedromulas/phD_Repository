from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import correlate, correlation_lags


ArrayLike = np.ndarray


def tr_to_samples(TR: float, fs: float) -> int:
    """Convert the MRI repetition time from seconds to samples.

    Parameters
    ----------
    TR
        Repetition time in seconds.
    fs
        EEG sampling frequency in Hz.

    Returns
    -------
    int
        Number of EEG samples contained in one TR.

    Raises
    ------
    ValueError
        If ``TR`` or ``fs`` are not strictly positive, or if the rounded number
        of samples is smaller than one.
    """
    if TR <= 0:
        raise ValueError("TR must be strictly positive.")
    if fs <= 0:
        raise ValueError("fs must be strictly positive.")

    samples = int(round(TR * fs))
    if samples < 1:
        raise ValueError("TR * fs must correspond to at least one sample.")
    return samples


def _as_2d(signal: ArrayLike) -> tuple[np.ndarray, bool]:
    """Return the signal as ``(n_channels, n_samples)`` and whether it was 1D."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        return signal[np.newaxis, :], True
    if signal.ndim == 2:
        return signal, False
    raise ValueError("signal must be a 1D or 2D NumPy array.")


def _validate_offset(offset: int, T_samples: int) -> None:
    if T_samples <= 0:
        raise ValueError("T_samples must be strictly positive.")
    if offset < 0 or offset >= T_samples:
        raise ValueError("offset must satisfy 0 <= offset < T_samples.")


def _reference_channel_data(signal: ArrayLike, reference_channel: int) -> np.ndarray:
    """Extract one reference channel from a 1D or 2D signal."""
    signal_2d, was_1d = _as_2d(signal)
    if reference_channel < 0 or reference_channel >= signal_2d.shape[0]:
        raise ValueError("reference_channel is out of bounds.")
    if was_1d and reference_channel != 0:
        raise ValueError("reference_channel must be 0 for 1D signals.")
    return signal_2d[reference_channel]


def segment_signal(signal: ArrayLike, T_samples: int, offset: int) -> np.ndarray:
    """Segment a continuous signal into TR-sized epochs.

    Parameters
    ----------
    signal
        EEG array with shape ``(n_samples,)`` or ``(n_channels, n_samples)``.
    T_samples
        Number of samples per TR.
    offset
        Starting offset, in samples, of the first TR boundary.

    Returns
    -------
    np.ndarray
        Segments with shape ``(n_segments, T_samples)`` for 1D input or
        ``(n_channels, n_segments, T_samples)`` for 2D input.

    Raises
    ------
    ValueError
        If the signal is too short to contain at least one complete segment.
    """
    _validate_offset(offset, T_samples)
    signal_2d, was_1d = _as_2d(signal)
    n_channels, n_samples = signal_2d.shape

    usable_samples = n_samples - offset
    n_segments = usable_samples // T_samples
    if n_segments < 1:
        raise ValueError("The signal is too short to extract one full segment.")

    trimmed = signal_2d[:, offset : offset + n_segments * T_samples]
    segmented = trimmed.reshape(n_channels, n_segments, T_samples)
    if was_1d:
        return segmented[0]
    return segmented


def compute_alignment_score(segments: ArrayLike) -> float:
    """Compute a segment consistency score based on time-wise variance.

    Lower scores indicate that the segments are better aligned.

    Parameters
    ----------
    segments
        2D array with shape ``(n_segments, T_samples)``.

    Returns
    -------
    float
        Mean variance across time points.
    """
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 2:
        raise ValueError("segments must have shape (n_segments, T_samples).")
    if segments.shape[0] < 2:
        return float(np.inf)
    return float(np.mean(np.var(segments, axis=0, ddof=0)))


def find_best_offset(
    signal: ArrayLike,
    T_samples: int,
    reference_channel: int = 0,
) -> int:
    """Estimate the TR offset that best aligns gradient artifact repetitions.

    The optimal offset is defined as the one minimizing the average variance
    across TR-sized segments in the chosen reference channel.

    Parameters
    ----------
    signal
        EEG array with shape ``(n_samples,)`` or ``(n_channels, n_samples)``.
    T_samples
        Number of samples per TR.
    reference_channel
        Channel used to estimate the optimal offset for multichannel data.

    Returns
    -------
    int
        Optimal offset in samples, constrained to ``[0, T_samples)``.
    """
    reference = _reference_channel_data(signal, reference_channel)
    best_offset = 0
    best_score = np.inf

    for offset in range(T_samples):
        try:
            segments = segment_signal(reference, T_samples, offset)
        except ValueError:
            continue
        score = compute_alignment_score(segments)
        if score < best_score:
            best_score = score
            best_offset = offset

    if not np.isfinite(best_score):
        raise ValueError("Unable to estimate offset from the provided signal.")
    return best_offset


def compute_template(segments: ArrayLike) -> np.ndarray:
    """Compute the average artifact template across segments.

    Parameters
    ----------
    segments
        2D array with shape ``(n_segments, T_samples)``.

    Returns
    -------
    np.ndarray
        Template with shape ``(T_samples,)``.
    """
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 2:
        raise ValueError("segments must have shape (n_segments, T_samples).")
    if segments.shape[0] < 1:
        raise ValueError("At least one segment is required.")
    return np.mean(segments, axis=0)


def shift_signal(signal: ArrayLike, lag: int) -> np.ndarray:
    """Shift a 1D signal using zero padding instead of circular wrapping.

    Parameters
    ----------
    signal
        Input 1D array.
    lag
        Integer lag in samples. Positive values delay the signal to the right.
        Negative values advance the signal to the left.

    Returns
    -------
    np.ndarray
        Shifted signal with the same shape as the input.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be 1D.")

    shifted = np.zeros_like(signal)
    if lag == 0:
        shifted[:] = signal
        return shifted

    n_samples = signal.shape[0]
    if abs(lag) >= n_samples:
        return shifted

    if lag > 0:
        shifted[lag:] = signal[:-lag]
    else:
        shifted[:lag] = signal[-lag:]
    return shifted


def _best_lag_for_segment(
    segment: np.ndarray,
    template: np.ndarray,
    max_lag: int | None = None,
) -> int:
    """Estimate the lag that best matches one segment to the template."""
    segment = np.asarray(segment, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)

    segment_centered = segment - np.mean(segment)
    template_centered = template - np.mean(template)

    corr = correlate(segment_centered, template_centered, mode="full", method="auto")
    lags = correlation_lags(segment_centered.size, template_centered.size, mode="full")

    if max_lag is not None:
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative.")
        mask = np.abs(lags) <= max_lag
        corr = corr[mask]
        lags = lags[mask]

    if corr.size == 0:
        return 0
    return int(lags[np.argmax(corr)])


def realign_segments(
    segments: ArrayLike,
    template: ArrayLike,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Realign segments to a template using cross-correlation.

    Parameters
    ----------
    segments
        2D array with shape ``(n_segments, T_samples)``.
    template
        Reference template with shape ``(T_samples,)``.
    max_lag
        Optional maximum allowed lag in samples. Restricting the search is
        often useful in practice to avoid implausibly large shifts.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Realigned segments and the estimated lag for each segment. The shift
        applied to each segment is ``-lag``.
    """
    segments = np.asarray(segments, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    if segments.ndim != 2:
        raise ValueError("segments must have shape (n_segments, T_samples).")
    if template.ndim != 1:
        raise ValueError("template must be 1D.")
    if segments.shape[1] != template.shape[0]:
        raise ValueError("Segment length and template length must match.")

    n_segments = segments.shape[0]
    aligned = np.empty_like(segments)
    lags = np.zeros(n_segments, dtype=int)

    for idx in range(n_segments):
        lag = _best_lag_for_segment(segments[idx], template, max_lag=max_lag)
        aligned[idx] = shift_signal(segments[idx], -lag)
        lags[idx] = lag

    return aligned, lags


def iterative_realignment(
    segments: ArrayLike,
    n_iter: int = 5,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively refine segment alignment and the artifact template.

    Parameters
    ----------
    segments
        2D array with shape ``(n_segments, T_samples)``.
    n_iter
        Number of template-refinement iterations.
    max_lag
        Optional maximum allowed lag in samples for each realignment step.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Final aligned segments, the final template, and cumulative lags.
    """
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 2:
        raise ValueError("segments must have shape (n_segments, T_samples).")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")

    aligned = segments.copy()
    cumulative_lags = np.zeros(aligned.shape[0], dtype=int)

    for _ in range(n_iter):
        template = compute_template(aligned)
        aligned, lags = realign_segments(aligned, template, max_lag=max_lag)
        cumulative_lags += lags

    final_template = compute_template(aligned)
    return aligned, final_template, cumulative_lags


def apply_aas(segments: ArrayLike, window_size: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Apply sliding-window Average Artifact Subtraction to aligned segments.

    Parameters
    ----------
    segments
        2D array with shape ``(n_segments, T_samples)``.
    window_size
        Number of neighboring segments used to build the local artifact
        template. The window is centered on each segment when possible.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cleaned segments and the local templates that were subtracted.
    """
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 2:
        raise ValueError("segments must have shape (n_segments, T_samples).")
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    n_segments, T_samples = segments.shape
    half_window = window_size // 2

    cleaned = np.empty_like(segments)
    templates = np.empty((n_segments, T_samples), dtype=np.float64)

    for idx in range(n_segments):
        start = max(0, idx - half_window)
        stop = min(n_segments, idx + half_window + 1)
        local_template = np.mean(segments[start:stop], axis=0)
        templates[idx] = local_template
        cleaned[idx] = segments[idx] - local_template

    return cleaned, templates


def reconstruct_signal(
    segments: ArrayLike,
    original_length: int,
    offset: int,
) -> np.ndarray:
    """Reconstruct a continuous signal from TR-sized segments.

    Samples before ``offset`` and after the last complete segment are filled
    with zeros because they are not covered by the segmented representation.

    Parameters
    ----------
    segments
        Segmented data with shape ``(n_segments, T_samples)`` or
        ``(n_channels, n_segments, T_samples)``.
    original_length
        Desired number of samples in the reconstructed continuous signal.
    offset
        Sample index where the first segment starts.

    Returns
    -------
    np.ndarray
        Reconstructed signal with shape ``(original_length,)`` for 2D input or
        ``(n_channels, original_length)`` for 3D input.
    """
    segments = np.asarray(segments, dtype=np.float64)
    if original_length < 1:
        raise ValueError("original_length must be positive.")

    if segments.ndim == 2:
        n_segments, T_samples = segments.shape
        _validate_offset(offset, T_samples)
        reconstructed = np.zeros(original_length, dtype=np.float64)
        usable = min(n_segments * T_samples, max(0, original_length - offset))
        if usable > 0:
            reconstructed[offset : offset + usable] = segments.reshape(-1)[:usable]
        return reconstructed

    if segments.ndim == 3:
        n_channels, n_segments, T_samples = segments.shape
        _validate_offset(offset, T_samples)
        reconstructed = np.zeros((n_channels, original_length), dtype=np.float64)
        usable = min(n_segments * T_samples, max(0, original_length - offset))
        if usable > 0:
            reconstructed[:, offset : offset + usable] = segments.reshape(n_channels, -1)[:, :usable]
        return reconstructed

    raise ValueError("segments must have shape (n_segments, T) or (n_channels, n_segments, T).")


def run_aas_pipeline(
    signal: ArrayLike,
    TR: float,
    fs: float,
    reference_channel: int = 0,
    n_iter: int = 5,
    window_size: int = 21,
    max_lag: int | None = None,
) -> dict[str, Any]:
    """Run the full trigger-free GA removal pipeline.

    Parameters
    ----------
    signal
        EEG array with shape ``(n_samples,)`` or ``(n_channels, n_samples)``.
    TR
        MRI repetition time in seconds.
    fs
        EEG sampling frequency in Hz.
    reference_channel
        Channel used to estimate offset and lags for multichannel data.
    n_iter
        Number of iterative realignment steps.
    window_size
        Number of neighboring segments used in sliding-window AAS.
    max_lag
        Optional maximum allowed lag in samples.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the cleaned signal, alignment parameters, and
        intermediate arrays needed for inspection or debugging.
    """
    signal_2d, was_1d = _as_2d(signal)
    T_samples = tr_to_samples(TR, fs)
    offset = find_best_offset(signal_2d, T_samples, reference_channel=reference_channel)

    segmented = segment_signal(signal_2d, T_samples, offset)
    reference_segments = segmented[reference_channel]
    aligned_reference, final_template, lags = iterative_realignment(
        reference_segments,
        n_iter=n_iter,
        max_lag=max_lag,
    )

    aligned_all = np.empty_like(segmented)
    for channel_idx in range(segmented.shape[0]):
        for seg_idx, lag in enumerate(lags):
            aligned_all[channel_idx, seg_idx] = shift_signal(segmented[channel_idx, seg_idx], -lag)

    cleaned_all = np.empty_like(aligned_all)
    local_templates = np.empty_like(aligned_all)
    for channel_idx in range(aligned_all.shape[0]):
        cleaned_channel, template_channel = apply_aas(aligned_all[channel_idx], window_size=window_size)
        cleaned_all[channel_idx] = cleaned_channel
        local_templates[channel_idx] = template_channel

    cleaned_signal = reconstruct_signal(cleaned_all, signal_2d.shape[1], offset)
    if was_1d:
        segmented_out = segmented[0]
        aligned_out = aligned_all[0]
        cleaned_segments_out = cleaned_all[0]
        cleaned_signal_out = cleaned_signal[0]
        local_templates_out = local_templates[0]
    else:
        segmented_out = segmented
        aligned_out = aligned_all
        cleaned_segments_out = cleaned_all
        cleaned_signal_out = cleaned_signal
        local_templates_out = local_templates

    return {
        "cleaned_signal": cleaned_signal_out,
        "offset": offset,
        "T_samples": T_samples,
        "lags": lags,
        "segmented_signal": segmented_out,
        "aligned_segments": aligned_out,
        "cleaned_segments": cleaned_segments_out,
        "final_reference_template": final_template,
        "local_templates": local_templates_out,
        "aligned_reference_segments": aligned_reference,
    }

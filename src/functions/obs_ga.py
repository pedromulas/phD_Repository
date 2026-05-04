from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import correlate, correlation_lags


ArrayLike = np.ndarray


def tr_to_samples(TR: float, fs: float) -> int:
    """Convert the MRI repetition time from seconds to samples."""
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


def _validate_n_components(n_components: int) -> None:
    if n_components < 1:
        raise ValueError("n_components must be at least 1.")


def _fixed_window_bounds(n_segments: int, center_idx: int, window_size: int) -> tuple[int, int]:
    """Return fixed-size window bounds, shifting the window at the edges."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if window_size > n_segments:
        return 0, n_segments

    half_window = window_size // 2
    start = center_idx - half_window
    stop = start + window_size

    if start < 0:
        start = 0
        stop = window_size
    elif stop > n_segments:
        stop = n_segments
        start = n_segments - window_size

    return start, stop


def segment_signal(signal: ArrayLike, T_samples: int, offset: int = 0) -> np.ndarray:
    """Segment a continuous signal into TR-sized epochs.

    Parameters
    ----------
    signal
        EEG array with shape ``(n_samples,)`` or ``(n_channels, n_samples)``.
    T_samples
        Number of samples per TR.
    offset
        Starting offset, in samples, of the first segment boundary.

    Returns
    -------
    np.ndarray
        Segments with shape ``(n_segments, T_samples)`` for 1D input or
        ``(n_channels, n_segments, T_samples)`` for 2D input.
    """
    _validate_offset(offset, T_samples)
    signal_2d, was_1d = _as_2d(signal)
    _, n_samples = signal_2d.shape

    usable_samples = n_samples - offset
    n_segments = usable_samples // T_samples
    if n_segments < 1:
        raise ValueError("The signal is too short to extract one full segment.")

    trimmed = signal_2d[:, offset : offset + n_segments * T_samples]
    segmented = trimmed.reshape(signal_2d.shape[0], n_segments, T_samples)
    if was_1d:
        return segmented[0]
    return segmented


def shift_signal(signal: ArrayLike, lag: int) -> np.ndarray:
    """Shift a 1D signal using zero padding instead of circular wrapping."""
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


def compute_template(segments: ArrayLike) -> np.ndarray:
    """Compute the mean template across segments."""
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 2:
        raise ValueError("segments must have shape (n_segments, T_samples).")
    if segments.shape[0] < 1:
        raise ValueError("At least one segment is required.")
    return np.mean(segments, axis=0)


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
    template: ArrayLike | None = None,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Realign segments to a template using cross-correlation."""
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 2:
        raise ValueError("segments must have shape (n_segments, T_samples).")

    if template is None:
        template = compute_template(segments)
    else:
        template = np.asarray(template, dtype=np.float64)

    if template.ndim != 1:
        raise ValueError("template must be 1D.")
    if template.shape[0] != segments.shape[1]:
        raise ValueError("Segment length and template length must match.")

    aligned = np.empty_like(segments)
    lags = np.zeros(segments.shape[0], dtype=int)

    for idx in range(segments.shape[0]):
        lag = _best_lag_for_segment(segments[idx], template, max_lag=max_lag)
        aligned[idx] = shift_signal(segments[idx], -lag)
        lags[idx] = lag

    return aligned, lags


def prepare_obs_matrix(
    segments: ArrayLike,
    remove_mean: bool = True,
    normalize: bool = False,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build the OBS data matrix and return preprocessing parameters.

    Parameters
    ----------
    segments
        2D array with shape ``(n_segments, T_samples)``.
    remove_mean
        If True, subtract the mean of each segment before PCA.
    normalize
        If True, divide each segment by its standard deviation after mean
        removal. This is optional and disabled by default.
    eps
        Small positive constant used to avoid division by zero.
    """
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 2:
        raise ValueError("segments must have shape (n_segments, T_samples).")

    X = segments.copy()
    means = np.zeros(X.shape[0], dtype=np.float64)
    scales = np.ones(X.shape[0], dtype=np.float64)

    if remove_mean:
        means = np.mean(X, axis=1)
        X -= means[:, np.newaxis]

    if normalize:
        scales = np.std(X, axis=1, ddof=0)
        scales = np.where(scales < eps, 1.0, scales)
        X /= scales[:, np.newaxis]

    params = {
        "means": means,
        "scales": scales,
        "remove_mean": np.array(remove_mean, dtype=bool),
        "normalize": np.array(normalize, dtype=bool),
    }
    return X, params


def compute_obs_basis(X: ArrayLike, n_components: int = 4) -> np.ndarray:
    """Compute temporal OBS basis functions using PCA via SVD.

    Parameters
    ----------
    X
        Matrix with shape ``(n_segments, T_samples)``.
    n_components
        Number of temporal basis functions to retain.

    Returns
    -------
    np.ndarray
        Basis with shape ``(n_components, T_samples)``.
    """
    _validate_n_components(n_components)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (n_segments, T_samples).")
    if X.shape[0] < 1:
        raise ValueError("At least one segment is required.")

    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n_keep = min(n_components, vt.shape[0])
    return vt[:n_keep]


def project_onto_basis(
    segment: ArrayLike,
    basis: ArrayLike,
    mean: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Project one segment onto the OBS basis."""
    segment = np.asarray(segment, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if segment.ndim != 1:
        raise ValueError("segment must be 1D.")
    if basis.ndim != 2:
        raise ValueError("basis must have shape (n_components, T_samples).")
    if segment.shape[0] != basis.shape[1]:
        raise ValueError("segment length and basis length must match.")

    standardized = (segment - mean) / scale
    return standardized @ basis.T


def reconstruct_artifact(
    segment: ArrayLike,
    basis: ArrayLike,
    mean: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Reconstruct the artifact component of one segment from the OBS basis."""
    coeffs = project_onto_basis(segment, basis, mean=mean, scale=scale)
    basis = np.asarray(basis, dtype=np.float64)
    artifact_standardized = coeffs @ basis
    artifact = artifact_standardized * scale
    if mean != 0.0:
        artifact += mean
    return artifact


def apply_obs(
    segments: ArrayLike,
    n_components: int = 4,
    remove_mean: bool = True,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply global OBS artifact removal to segmented data."""
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 2:
        raise ValueError("segments must have shape (n_segments, T_samples).")

    X, params = prepare_obs_matrix(segments, remove_mean=remove_mean, normalize=normalize)
    basis = compute_obs_basis(X, n_components=n_components)

    cleaned = np.empty_like(segments)
    for idx in range(segments.shape[0]):
        artifact = reconstruct_artifact(
            segments[idx],
            basis,
            mean=float(params["means"][idx]),
            scale=float(params["scales"][idx]),
        )
        cleaned[idx] = segments[idx] - artifact

    return cleaned, basis


def apply_sliding_window_obs(
    segments: ArrayLike,
    window_size: int = 21,
    n_components: int = 4,
    remove_mean: bool = True,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply OBS using a fixed-size sliding window over segments."""
    _validate_n_components(n_components)
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 2:
        raise ValueError("segments must have shape (n_segments, T_samples).")
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    n_segments, T_samples = segments.shape
    cleaned = np.empty_like(segments)
    basis_bank = np.zeros((n_segments, min(n_components, T_samples), T_samples), dtype=np.float64)

    for idx in range(n_segments):
        start, stop = _fixed_window_bounds(n_segments, idx, window_size)
        window_segments = segments[start:stop]
        X, params = prepare_obs_matrix(window_segments, remove_mean=remove_mean, normalize=normalize)
        basis = compute_obs_basis(X, n_components=n_components)
        basis_bank[idx, : basis.shape[0]] = basis

        local_idx = idx - start
        artifact = reconstruct_artifact(
            window_segments[local_idx],
            basis,
            mean=float(params["means"][local_idx]),
            scale=float(params["scales"][local_idx]),
        )
        cleaned[idx] = segments[idx] - artifact

    return cleaned, basis_bank


def reconstruct_signal(
    segments: ArrayLike,
    original_length: int,
    offset: int,
) -> np.ndarray:
    """Reconstruct a continuous signal from TR-sized segments."""
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


def run_obs_pipeline(
    signal: ArrayLike,
    TR: float,
    fs: float,
    offset: int = 0,
    reference_channel: int = 0,
    n_components: int = 4,
    max_lag: int | None = None,
    window_size: int | None = 21,
    remove_mean: bool = True,
    normalize: bool = False,
) -> dict[str, Any]:
    """Run the full OBS artifact removal pipeline.

    OBS bases are estimated from the reference channel and then applied to all
    channels segment-by-segment.
    """
    signal_2d, was_1d = _as_2d(signal)
    if reference_channel < 0 or reference_channel >= signal_2d.shape[0]:
        raise ValueError("reference_channel is out of bounds.")

    T_samples = tr_to_samples(TR, fs)
    segmented = segment_signal(signal_2d, T_samples, offset)

    reference_segments = segmented[reference_channel]
    aligned_reference, lags = realign_segments(reference_segments, max_lag=max_lag)

    aligned_all = np.empty_like(segmented)
    for channel_idx in range(segmented.shape[0]):
        for seg_idx, lag in enumerate(lags):
            aligned_all[channel_idx, seg_idx] = shift_signal(segmented[channel_idx, seg_idx], -lag)

    cleaned_all = np.empty_like(aligned_all)

    if window_size is None:
        cleaned_reference, basis = apply_obs(
            aligned_reference,
            n_components=n_components,
            remove_mean=remove_mean,
            normalize=normalize,
        )
        cleaned_all[reference_channel] = cleaned_reference

        for channel_idx in range(aligned_all.shape[0]):
            if channel_idx == reference_channel:
                continue
            X_ref, params_ref = prepare_obs_matrix(
                aligned_all[channel_idx],
                remove_mean=remove_mean,
                normalize=normalize,
            )
            del X_ref
            for seg_idx in range(aligned_all.shape[1]):
                artifact = reconstruct_artifact(
                    aligned_all[channel_idx, seg_idx],
                    basis,
                    mean=float(params_ref["means"][seg_idx]),
                    scale=float(params_ref["scales"][seg_idx]),
                )
                cleaned_all[channel_idx, seg_idx] = aligned_all[channel_idx, seg_idx] - artifact
        basis_out: ArrayLike = basis
    else:
        cleaned_reference, basis_bank = apply_sliding_window_obs(
            aligned_reference,
            window_size=window_size,
            n_components=n_components,
            remove_mean=remove_mean,
            normalize=normalize,
        )
        cleaned_all[reference_channel] = cleaned_reference

        for channel_idx in range(aligned_all.shape[0]):
            if channel_idx == reference_channel:
                continue
            _, params = prepare_obs_matrix(
                aligned_all[channel_idx],
                remove_mean=remove_mean,
                normalize=normalize,
            )
            for seg_idx in range(aligned_all.shape[1]):
                basis = basis_bank[seg_idx]
                active = np.any(np.abs(basis) > 0, axis=1)
                basis_active = basis[active]
                artifact = reconstruct_artifact(
                    aligned_all[channel_idx, seg_idx],
                    basis_active,
                    mean=float(params["means"][seg_idx]),
                    scale=float(params["scales"][seg_idx]),
                )
                cleaned_all[channel_idx, seg_idx] = aligned_all[channel_idx, seg_idx] - artifact
        basis_out = basis_bank

    cleaned_signal = reconstruct_signal(cleaned_all, signal_2d.shape[1], offset)

    if was_1d:
        segmented_out = segmented[0]
        aligned_out = aligned_all[0]
        cleaned_segments_out = cleaned_all[0]
        cleaned_signal_out = cleaned_signal[0]
    else:
        segmented_out = segmented
        aligned_out = aligned_all
        cleaned_segments_out = cleaned_all
        cleaned_signal_out = cleaned_signal

    return {
        "cleaned_signal": cleaned_signal_out,
        "T_samples": T_samples,
        "offset": offset,
        "lags": lags,
        "segmented_signal": segmented_out,
        "aligned_segments": aligned_out,
        "cleaned_segments": cleaned_segments_out,
        "basis": basis_out,
    }
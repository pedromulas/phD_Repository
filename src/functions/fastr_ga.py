"""FASTR-inspired gradient-artifact suppression for simultaneous EEG-fMRI.

The original FASTR procedure (Niazy et al., 2005) combines an adaptive
slice/volume artefact template with a principal-component model of residual
artefact.  This implementation deliberately exposes the repetition period:
use the slice period when slice triggers are available, or the TR otherwise.
It is trigger-free by default, using the same offset estimation convention as
the repository AAS implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

try:  # Supports both ``python -m functions...`` and direct script execution.
    from functions.aas_ga import find_best_offset, segment_signal, tr_to_samples
    from functions.gradient_sync import GradientSyncDetection, detect_gradient_artifact_start
except ModuleNotFoundError:  # pragma: no cover - direct execution convenience
    from aas_ga import find_best_offset, segment_signal, tr_to_samples
    from gradient_sync import GradientSyncDetection, detect_gradient_artifact_start


@dataclass(frozen=True)
class FASTRConfig:
    """Parameters for the adaptive template and residual-PCA stages."""

    artifact_period_s: float
    template_window: int = 21
    pca_components: int = 4
    offset: int | None = None
    use_fmri_bounds: bool = False
    fmri_start_sample: int | None = None
    fmri_end_sample: int | None = None
    tr_sec: float | None = None
    n_slices: int | None = None
    calibration_seconds: float = 5.0


def _resolve_fmri_bounds(
    signal: np.ndarray,
    fs: float,
    config: FASTRConfig,
    channel_names: list[str] | None,
) -> tuple[int, int, GradientSyncDetection | None]:
    """Return user-provided or morphology-detected fMRI bounds."""
    n_samples = signal.shape[1]
    if not config.use_fmri_bounds:
        return 0, n_samples, None

    detection = None
    start, end = config.fmri_start_sample, config.fmri_end_sample
    if start is None or end is None:
        if config.tr_sec is None or config.n_slices is None:
            raise ValueError("tr_sec and n_slices are required when automatic fMRI-bound detection is enabled.")
        detection = detect_gradient_artifact_start(
            signal,
            fs,
            config.tr_sec,
            config.n_slices,
            channel_names=channel_names,
            calibration_seconds=config.calibration_seconds,
        )
        # FASTR must cover dummy scans too: clean from the first sustained GA,
        # not only from the later stable BOLD block.
        start = detection.t_ga_sample if start is None else start
        # Keep late periodic BOLD peaks too, even if the energy envelope has
        # already fallen below its sustained-activity threshold.
        end = max(detection.t_ga_end_sample, detection.t_bold_end_sample) if end is None else end

    start, end = int(start), int(end)
    if start < 0 or start >= n_samples:
        raise ValueError("fmri_start_sample is out of bounds.")
    if end <= start or end > n_samples:
        raise ValueError("fmri_end_sample must satisfy start < end <= n_samples.")
    return start, end, detection


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.1,
    high_hz: float = 70.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter for ``channels x samples`` EEG."""
    data = np.asarray(signal, dtype=np.float64)
    was_1d = data.ndim == 1
    if was_1d:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError("signal must have shape (samples,) or (channels, samples).")
    if fs <= 0 or not 0 < low_hz < high_hz < fs / 2:
        raise ValueError("Require 0 < low_hz < high_hz < fs / 2.")
    if order < 1:
        raise ValueError("order must be positive.")

    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, data, axis=-1)
    return filtered[0] if was_1d else filtered


def _window_bounds(n_epochs: int, index: int, window: int) -> tuple[int, int]:
    if window < 1:
        raise ValueError("template_window must be positive.")
    window = min(window, n_epochs)
    start = max(0, index - window // 2)
    stop = min(n_epochs, start + window)
    start = max(0, stop - window)
    return start, stop


def _adaptive_templates(epochs: np.ndarray, window: int) -> np.ndarray:
    """Create a robust, local artefact template for every epoch and channel."""
    n_channels, n_epochs, n_samples = epochs.shape
    templates = np.empty((n_channels, n_epochs, n_samples), dtype=np.float64)
    for epoch in range(n_epochs):
        start, stop = _window_bounds(n_epochs, epoch, window)
        # The median is less affected by transient EEG events than a mean.
        templates[:, epoch] = np.median(epochs[:, start:stop], axis=1)
    return templates


def _remove_residual_pca(residual: np.ndarray, templates: np.ndarray, n_components: int) -> np.ndarray:
    """Remove residual variation in the adaptive-template subspace.

    PCA is fitted independently per channel to the *template* trajectories,
    rather than the EEG residual, so that physiological EEG does not define the
    artefact subspace.
    """
    if n_components < 0:
        raise ValueError("pca_components must be non-negative.")
    if n_components == 0:
        return residual

    corrected = residual.copy()
    for channel in range(residual.shape[0]):
        template_matrix = templates[channel]
        centered = template_matrix - template_matrix.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        components = vh[: min(n_components, vh.shape[0])]
        if components.size:
            coefficients = residual[channel] @ components.T
            corrected[channel] -= coefficients @ components
    return corrected


def fastr_remove_gradient_artifact(
    signal: np.ndarray,
    fs: float,
    config: FASTRConfig,
    reference_channel: int = 0,
    detection_signal: np.ndarray | None = None,
    channel_names: list[str] | None = None,
) -> dict[str, object]:
    """Apply adaptive template subtraction and residual PCA artefact removal.

    Samples outside complete artefact periods are retained unchanged.  The
    returned ``cleaned_signal`` always has the same shape as ``signal``.
    """
    data = np.asarray(signal, dtype=np.float64)
    was_1d = data.ndim == 1
    if was_1d:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError("signal must have shape (samples,) or (channels, samples).")
    if not 0 <= reference_channel < data.shape[0]:
        raise ValueError("reference_channel is out of bounds.")

    detection_data = data if detection_signal is None else np.asarray(detection_signal, dtype=np.float64)
    if detection_data.shape != data.shape:
        raise ValueError("detection_signal must have the same shape as signal.")
    fmri_start, fmri_end, detection = _resolve_fmri_bounds(detection_data, fs, config, channel_names)
    fmri_data = data[:, fmri_start:fmri_end]

    period_samples = tr_to_samples(config.artifact_period_s, fs)
    offset = config.offset
    if offset is None:
        offset = find_best_offset(fmri_data, period_samples, reference_channel)
    if not 0 <= offset < period_samples:
        raise ValueError("offset must satisfy 0 <= offset < period_samples.")

    epochs = segment_signal(fmri_data, period_samples, offset)
    templates = _adaptive_templates(epochs, config.template_window)
    residual = epochs - templates
    cleaned_epochs = _remove_residual_pca(residual, templates, config.pca_components)

    cleaned = data.copy()
    usable = cleaned_epochs.shape[1] * period_samples
    cleaned_start = fmri_start + offset
    cleaned[:, cleaned_start : cleaned_start + usable] = cleaned_epochs.reshape(data.shape[0], -1)
    return {
        "cleaned_signal": cleaned[0] if was_1d else cleaned,
        "offset": offset,
        "period_samples": period_samples,
        "templates": templates[0] if was_1d else templates,
        "cleaned_epochs": cleaned_epochs[0] if was_1d else cleaned_epochs,
        "use_fmri_bounds": config.use_fmri_bounds,
        "fmri_start_sample": fmri_start,
        "fmri_end_sample": fmri_end,
        "fmri_start_sec": fmri_start / fs,
        "fmri_end_sec": fmri_end / fs,
        "gradient_detection_used": detection is not None,
        "t_bold_sample": detection.t_bold_sample if detection is not None else fmri_start,
        "t_bold_end_sample": detection.t_bold_end_sample if detection is not None else fmri_end,
        "t_bold_sec": detection.t_bold_sec if detection is not None else fmri_start / fs,
        "t_bold_end_sec": detection.t_bold_end_sec if detection is not None else fmri_end / fs,
    }

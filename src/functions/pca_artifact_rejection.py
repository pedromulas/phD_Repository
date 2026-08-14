"""Spatial PCA and conservative ocular/muscle component suggestions for EEG."""

from __future__ import annotations

from typing import Any, Sequence

import mne
import numpy as np
from scipy.signal import welch


def _band_power(frequencies: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (frequencies >= low) & (frequencies < high)
    if np.count_nonzero(mask) < 2:
        return np.zeros(psd.shape[0])
    return np.trapezoid(psd[:, mask], frequencies[mask], axis=-1)


def apply_pca_artifact_rejection(
    raw: mne.io.BaseRaw,
    picks: Sequence[str],
    n_components: int | None = None,
    manual_exclude: Sequence[int] = (),
    auto_ocular: bool = False,
    auto_muscle: bool = False,
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """Remove selected spatial PCA components and return the reconstructed Raw.

    PCA components are ranked only as *candidates*: frontal, low-frequency
    components can be marked ocular and high-frequency components muscular.
    Their thresholds are deliberately conservative and should be visually
    inspected for a final study analysis.
    """
    names = list(picks)
    if len(names) < 2 or any(name not in raw.ch_names for name in names):
        raise ValueError("picks must contain at least two valid EEG channel names.")
    data = raw.get_data(picks=names).astype(np.float64, copy=False)
    centered = data - data.mean(axis=1, keepdims=True)
    spatial, _, _ = np.linalg.svd(centered, full_matrices=False)
    maximum = spatial.shape[1]
    n_keep = maximum if n_components is None else int(n_components)
    if not 1 <= n_keep <= maximum:
        raise ValueError(f"n_components must be between 1 and {maximum}.")
    spatial = spatial[:, :n_keep]
    scores = spatial.T @ centered
    frequencies, psd = welch(scores, fs=float(raw.info["sfreq"]), axis=-1, nperseg=min(4096, scores.shape[-1]))
    delta_theta = _band_power(frequencies, psd, 1.0, 8.0)
    alpha_beta = _band_power(frequencies, psd, 8.0, 30.0)
    gamma = _band_power(frequencies, psd, 30.0, min(60.0, float(raw.info["sfreq"]) / 2))
    frontal_indices = [index for index, name in enumerate(names) if name.upper() in {"FP1", "FP2", "AF7", "AF8"}]
    frontal_ratio = np.zeros(n_keep)
    if frontal_indices:
        frontal_ratio = np.max(np.abs(spatial[frontal_indices]), axis=0) / np.maximum(np.median(np.abs(spatial), axis=0), 1e-12)
    ocular = np.flatnonzero((frontal_ratio >= 2.5) & (delta_theta > alpha_beta)).astype(int).tolist() if auto_ocular else []
    muscle = np.flatnonzero(gamma > (delta_theta + alpha_beta)).astype(int).tolist() if auto_muscle else []
    manual = [int(component) for component in manual_exclude]
    excluded = sorted(set(manual + ocular + muscle))
    invalid = [component for component in excluded if not 0 <= component < n_keep]
    if invalid:
        raise ValueError(f"PCA component indices out of range: {invalid}")
    retained = np.ones(n_keep, dtype=bool)
    retained[excluded] = False
    reconstructed = spatial[:, retained] @ scores[retained] + data.mean(axis=1, keepdims=True)
    cleaned = raw.copy().load_data()
    indices = [cleaned.ch_names.index(name) for name in names]
    cleaned._data[indices] = reconstructed
    return cleaned, {
        "picks": names,
        "n_components": n_keep,
        "excluded_components": excluded,
        "manual_components": manual,
        "ocular_candidates": ocular,
        "muscle_candidates": muscle,
        "frontal_ratio": frontal_ratio,
        "low_frequency_power": delta_theta,
        "gamma_power": gamma,
        # Keep the decomposition compact but available to callers that need to
        # save it for later visual inspection (channels x components).
        "spatial_components": spatial,
        "component_scores": scores,
    }

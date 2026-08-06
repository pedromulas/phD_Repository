"""Conventional FastICA-based EEG artefact removal utilities.

ICA is fitted only to EEG channels. Components may be excluded manually or,
for gradient artefacts, ranked by their spectral energy at the supplied slice
frequency and its harmonics. Automatic ranking is a convenience aid: inspect
the returned scores and component topographies before using it in research.
"""

from __future__ import annotations

from typing import Any, Sequence

import mne
import numpy as np
from scipy.signal import welch


def _eeg_picks(raw: mne.io.BaseRaw, picks: Sequence[str] | None) -> list[str]:
    """Resolve EEG channels, excluding ECG and other auxiliary channels."""
    if picks is not None:
        resolved = list(picks)
        missing = [name for name in resolved if name not in raw.ch_names]
        if missing:
            raise ValueError(f"Channels not found in Raw: {missing}")
        return resolved
    indices = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, misc=False)
    resolved = [raw.ch_names[index] for index in indices]
    if len(resolved) < 2:
        raise ValueError("ICA requires at least two EEG channels.")
    return resolved


def fit_fastica(
    raw: mne.io.BaseRaw,
    picks: Sequence[str] | None = None,
    n_components: int | float | None = 0.99,
    fit_l_freq: float | None = 1.0,
    fit_h_freq: float | None = None,
    method: str = "fastica",
    random_state: int | None = 97,
    max_iter: int | str = "auto",
) -> tuple[mne.preprocessing.ICA, mne.io.BaseRaw, list[str]]:
    """Fit conventional ICA to selected EEG channels.

    A copy can be high-pass filtered only for estimating the ICA decomposition;
    the resulting unmixing matrix is later applied to the unfiltered original
    recording, preserving low-frequency EEG in the output.
    """
    selected = _eeg_picks(raw, picks)
    fit_raw = raw.copy().load_data().pick(selected)
    if fit_l_freq is not None or fit_h_freq is not None:
        fit_raw.filter(l_freq=fit_l_freq, h_freq=fit_h_freq, verbose="ERROR")
    if method not in {"fastica", "infomax"}:
        raise ValueError("method must be 'fastica' or 'infomax'.")
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter=max_iter,
    )
    ica.fit(fit_raw, verbose="ERROR")
    return ica, fit_raw, selected


def rank_components_by_ga_frequency(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    ga_frequency_hz: float,
    n_harmonics: int = 4,
    bandwidth_hz: float = 0.5,
) -> np.ndarray:
    """Return one GA spectral-energy score per ICA component.

    The score is the fraction of each component's 1--70 Hz power concentrated
    at the supplied GA frequency and its harmonics. Higher scores are more
    consistent with a periodic scanner-gradient component.
    """
    if ga_frequency_hz <= 0 or n_harmonics < 1 or bandwidth_hz <= 0:
        raise ValueError("ga_frequency_hz, n_harmonics and bandwidth_hz must be positive.")
    sources = ica.get_sources(raw).get_data()
    fs = float(raw.info["sfreq"])
    frequencies, psd = welch(sources, fs=fs, axis=-1, nperseg=min(4096, sources.shape[-1]))
    usable = (frequencies >= 1.0) & (frequencies <= min(70.0, fs / 2))
    total_power = np.trapezoid(psd[:, usable], frequencies[usable], axis=-1)
    ga_power = np.zeros(psd.shape[0], dtype=np.float64)
    for harmonic in range(1, n_harmonics + 1):
        centre = harmonic * ga_frequency_hz
        if centre >= fs / 2:
            break
        band = (frequencies >= centre - bandwidth_hz) & (frequencies <= centre + bandwidth_hz)
        if np.count_nonzero(band) >= 2:
            ga_power += np.trapezoid(psd[:, band], frequencies[band], axis=-1)
    return ga_power / np.maximum(total_power, np.finfo(float).eps)


def suggest_ocular_muscle_components(ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw) -> tuple[list[int], list[int]]:
    """Conservatively suggest frontal ocular and high-frequency muscle ICs."""
    sources = ica.get_sources(raw).get_data()
    frequencies, psd = welch(sources, fs=float(raw.info["sfreq"]), axis=-1, nperseg=min(4096, sources.shape[-1]))
    def power(low: float, high: float) -> np.ndarray:
        mask = (frequencies >= low) & (frequencies < high)
        return np.trapezoid(psd[:, mask], frequencies[mask], axis=-1) if np.count_nonzero(mask) >= 2 else np.zeros(psd.shape[0])
    low_frequency, alpha_beta = power(1.0, 8.0), power(8.0, 30.0)
    gamma = power(30.0, min(60.0, float(raw.info["sfreq"]) / 2))
    topography = ica.get_components()
    frontal = [index for index, name in enumerate(ica.ch_names) if name.upper() in {"FP1", "FP2", "AF7", "AF8"}]
    frontal_ratio = np.zeros(topography.shape[1])
    if frontal:
        frontal_ratio = np.max(np.abs(topography[frontal]), axis=0) / np.maximum(np.median(np.abs(topography), axis=0), 1e-12)
    ocular = np.flatnonzero((frontal_ratio >= 2.5) & (low_frequency > alpha_beta)).astype(int).tolist()
    muscle = np.flatnonzero(gamma > (low_frequency + alpha_beta)).astype(int).tolist()
    return ocular, muscle


def apply_conventional_ica(
    raw: mne.io.BaseRaw,
    picks: Sequence[str] | None = None,
    n_components: int | float | None = 0.99,
    manual_exclude: Sequence[int] | None = None,
    ga_frequency_hz: float | None = None,
    n_auto_components: int = 0,
    n_harmonics: int = 4,
    fit_l_freq: float | None = 1.0,
    method: str = "fastica",
    auto_ocular: bool = False,
    auto_muscle: bool = False,
    random_state: int | None = 97,
    max_iter: int | str = "auto",
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """Fit FastICA, exclude selected components, and return a cleaned Raw.

    Set ``manual_exclude`` after visually inspecting components, or set both
    ``ga_frequency_hz`` and ``n_auto_components`` to remove the strongest
    scanner-periodic components automatically. With neither option, the fitted
    decomposition is returned but no component is removed.
    """
    if n_auto_components < 0:
        raise ValueError("n_auto_components must be non-negative.")
    ica, fit_raw, selected = fit_fastica(
        raw, picks, n_components, fit_l_freq, method=method, random_state=random_state, max_iter=max_iter,
    )
    scores: np.ndarray | None = None
    automatic: list[int] = []
    scoring_raw = raw.copy().pick(selected)
    if ga_frequency_hz is not None:
        scores = rank_components_by_ga_frequency(ica, scoring_raw, ga_frequency_hz, n_harmonics)
        if n_auto_components:
            automatic = np.argsort(scores)[::-1][: min(n_auto_components, scores.size)].astype(int).tolist()
    elif n_auto_components:
        raise ValueError("ga_frequency_hz is required when n_auto_components is greater than zero.")
    ocular, muscle = suggest_ocular_muscle_components(ica, scoring_raw)
    if auto_ocular:
        automatic.extend(ocular)
    if auto_muscle:
        automatic.extend(muscle)

    manual = [] if manual_exclude is None else [int(component) for component in manual_exclude]
    invalid = [component for component in manual + automatic if not 0 <= component < ica.n_components_]
    if invalid:
        raise ValueError(f"ICA component indices out of range: {invalid}")
    excluded = sorted(set(manual + automatic))
    cleaned = raw.copy().load_data()
    ica.apply(cleaned, exclude=excluded, verbose="ERROR")
    result: dict[str, Any] = {
        "ica": ica,
        "fit_raw": fit_raw,
        "picks": selected,
        "excluded_components": excluded,
        "manual_components": manual,
        "automatic_components": automatic,
        "ocular_candidates": ocular,
        "muscle_candidates": muscle,
        "ga_scores": scores,
        "ga_frequency_hz": ga_frequency_hz,
        "method": method,
    }
    return cleaned, result

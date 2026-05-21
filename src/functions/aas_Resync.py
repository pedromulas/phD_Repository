from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.signal import resample

from functions.aas_ga import _as_2d, run_aas_pipeline


ArrayLike = np.ndarray


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive.")


def _validate_reference_channel(signal_2d: np.ndarray, reference_channel: int) -> None:
    if reference_channel < 0 or reference_channel >= signal_2d.shape[0]:
        raise ValueError("reference_channel is out of bounds.")


def nominal_tr_samples(TR: float, fs: float) -> float:
    """Return the nominal number of EEG samples per fMRI TR."""
    _validate_positive("TR", TR)
    _validate_positive("fs", fs)
    return float(TR * fs)


def _segment_reference_for_resync(
    reference: np.ndarray,
    epoch_len: int,
    start: int = 0,
    n_epochs_limit: int | None = None,
) -> np.ndarray:
    """Extract complete nominal-TR epochs from one reference channel."""
    if epoch_len < 2:
        raise ValueError("epoch_len must be at least 2 samples.")
    if start < 0 or start >= reference.shape[0]:
        raise ValueError("start must satisfy 0 <= start < n_samples.")

    usable = reference.shape[0] - start
    n_epochs = usable // epoch_len
    if n_epochs_limit is not None:
        if n_epochs_limit < 1:
            raise ValueError("n_epochs_limit must be at least 1.")
        n_epochs = min(n_epochs, n_epochs_limit)
    if n_epochs < 3:
        raise ValueError("At least three TR epochs are required to estimate D.")

    trimmed = reference[start : start + n_epochs * epoch_len]
    return trimmed.reshape(n_epochs, epoch_len)


def _normalize_epochs(epochs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Center and scale each epoch for ACS/coherence scoring."""
    centered = epochs - np.mean(epochs, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return centered / norms


def _fft_phase_corrected_epochs(epoch_fft: np.ndarray, D: float) -> np.ndarray:
    """Apply the linear phase ramp implied by a candidate desynchronization D.

    If the true artifact period is ``T * D`` instead of the nominal ``T``,
    the Relative Timing Error (RTE) accumulates by ``m * T * (D - 1)``
    samples at epoch ``m``. A continuous phase ramp compensates that drift
    without direct fixed-rate upsampling.
    """
    if D <= 0:
        raise ValueError("D must be strictly positive.")

    n_epochs = epoch_fft.shape[0]
    epoch_len = (epoch_fft.shape[1] - 1) * 2
    omega = 2.0 * np.pi * np.fft.rfftfreq(epoch_len)
    epoch_index = np.arange(n_epochs, dtype=np.float64)
    drift = epoch_index * epoch_len * (D - 1.0)

    phase_ramp = np.exp(1j * drift[:, np.newaxis] * omega[np.newaxis, :])
    corrected_fft = epoch_fft * phase_ramp
    return np.fft.irfft(corrected_fft, n=epoch_len, axis=1)


def compute_resync_acs(epochs: ArrayLike, D: float) -> float:
    """Compute an ACS-like score after phase correction for a candidate D.

    The score approaches 1 when nominal-TR epochs become maximally consistent,
    which is the synchronization condition described by Resync.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    if epochs.ndim != 2:
        raise ValueError("epochs must have shape (n_epochs, T_samples).")

    corrected = _fft_phase_corrected_epochs(np.fft.rfft(epochs, axis=1), D)
    corrected = _normalize_epochs(corrected)
    template = np.mean(corrected, axis=0)
    template_norm = np.linalg.norm(template)
    if template_norm == 0:
        return 0.0
    template /= template_norm
    acs_peaks = corrected @ template
    return float(np.mean(acs_peaks))


def _first_stable_high_score(
    candidates: np.ndarray,
    scores: np.ndarray,
    plateau_fraction: float = 0.9,
) -> int:
    """Return the earliest candidate that enters the high-coherence plateau."""
    if candidates.ndim != 1 or scores.ndim != 1 or candidates.size != scores.size:
        raise ValueError("candidates and scores must be 1D arrays of equal length.")

    score_max = float(np.max(scores))
    score_min = float(np.min(scores))
    threshold = score_min + plateau_fraction * (score_max - score_min)
    stable_idx = np.flatnonzero(scores >= threshold)
    if stable_idx.size == 0:
        return int(candidates[int(np.argmax(scores))])
    return int(candidates[int(stable_idx[0])])


def _last_stable_high_score(
    candidates: np.ndarray,
    scores: np.ndarray,
    plateau_fraction: float = 0.9,
) -> int:
    """Return the latest candidate that still belongs to the high-coherence plateau."""
    if candidates.ndim != 1 or scores.ndim != 1 or candidates.size != scores.size:
        raise ValueError("candidates and scores must be 1D arrays of equal length.")

    score_max = float(np.max(scores))
    score_min = float(np.min(scores))
    threshold = score_min + plateau_fraction * (score_max - score_min)
    stable_idx = np.flatnonzero(scores >= threshold)
    if stable_idx.size == 0:
        return int(candidates[int(np.argmax(scores))])
    return int(candidates[int(stable_idx[-1])])


def _first_persistent_high_score(
    candidates: np.ndarray,
    after_scores: np.ndarray,
    after_fraction: float = 0.9,
    persistence: int = 3,
) -> int:
    """Return the earliest candidate that enters a persistent high-coherence regime."""
    if candidates.ndim != 1 or after_scores.ndim != 1 or candidates.size != after_scores.size:
        raise ValueError("candidates and after_scores must be 1D arrays of equal length.")
    if persistence < 1:
        raise ValueError("persistence must be at least 1.")

    after_thr = float(np.min(after_scores) + after_fraction * (np.max(after_scores) - np.min(after_scores)))
    high_mask = after_scores >= after_thr

    run_length = 0
    for idx, is_high in enumerate(high_mask):
        if is_high:
            run_length += 1
            if run_length >= persistence:
                start_idx = idx - persistence + 1
                return int(candidates[start_idx])
        else:
            run_length = 0

    return int(candidates[int(np.argmax(after_scores))])


def estimate_fmri_start(
    signal: ArrayLike,
    TR: float,
    fs: float,
    reference_channel: int = 0,
    search_step: int | None = None,
    n_epochs_eval: int = 6,
) -> dict[str, Any]:
    """Estimate the first EEG sample belonging to the fMRI sequence.

    The start is defined as the earliest candidate whose subsequent TR-sized
    epochs enter and remain in a high-coherence regime for several consecutive
    evaluations. This usually localizes the onset better than a pure maximum-
    jump criterion because the first stable plateau matters more than the
    largest score increase.
    """
    if n_epochs_eval < 3:
        raise ValueError("n_epochs_eval must be at least 3.")

    signal_2d, _ = _as_2d(signal)
    _validate_reference_channel(signal_2d, reference_channel)

    nominal_epoch_len = max(2, int(round(nominal_tr_samples(TR, fs))))
    reference = signal_2d[reference_channel]

    max_start = reference.shape[0] - n_epochs_eval * nominal_epoch_len
    if max_start < 0:
        raise ValueError("Signal is too short to estimate the fMRI start.")

    if search_step is None:
        search_step = max(1, nominal_epoch_len // 8)
    if search_step < 1:
        raise ValueError("search_step must be at least 1.")

    coarse_candidates = np.arange(0, max_start + 1, search_step, dtype=int)
    if coarse_candidates.size == 0 or coarse_candidates[-1] != max_start:
        coarse_candidates = np.append(coarse_candidates, max_start)

    coarse_after_scores = np.empty(coarse_candidates.shape[0], dtype=np.float64)
    for idx, start in enumerate(coarse_candidates):
        after_epochs = _segment_reference_for_resync(
            reference,
            nominal_epoch_len,
            start=int(start),
            n_epochs_limit=n_epochs_eval,
        )
        coarse_after_scores[idx] = compute_resync_acs(after_epochs, 1.0)

    coarse_start = _first_persistent_high_score(
        coarse_candidates,
        coarse_after_scores,
        persistence=3,
    )

    refine_left = max(0, coarse_start - search_step)
    refine_right = min(max_start, coarse_start + search_step)
    fine_candidates = np.arange(refine_left, refine_right + 1, dtype=int)
    fine_after_scores = np.empty(fine_candidates.shape[0], dtype=np.float64)
    for idx, start in enumerate(fine_candidates):
        after_epochs = _segment_reference_for_resync(
            reference,
            nominal_epoch_len,
            start=int(start),
            n_epochs_limit=n_epochs_eval,
        )
        fine_after_scores[idx] = compute_resync_acs(after_epochs, 1.0)

    fmri_start_sample = _first_persistent_high_score(
        fine_candidates,
        fine_after_scores,
        persistence=max(3, search_step // max(1, nominal_epoch_len // 16)),
    )
    fine_best_idx = int(np.where(fine_candidates == fmri_start_sample)[0][0])

    return {
        "fmri_start_sample": fmri_start_sample,
        "fmri_start_time_sec": fmri_start_sample / fs,
        "nominal_tr_samples": nominal_epoch_len,
        "search_step": search_step,
        "n_epochs_eval": n_epochs_eval,
        "coarse_candidates": coarse_candidates,
        "coarse_after_scores": coarse_after_scores,
        "fine_candidates": fine_candidates,
        "fine_after_scores": fine_after_scores,
        "start_score": float(fine_after_scores[fine_best_idx]),
    }


def estimate_fmri_end(
    signal: ArrayLike,
    TR: float,
    fs: float,
    reference_channel: int = 0,
    search_step: int | None = None,
    n_epochs_eval: int = 6,
) -> dict[str, Any]:
    """Estimate the first EEG sample after the fMRI sequence ends.

    The end is inferred by scanning backward and finding the latest interval
    whose preceding nominal-TR epochs still show strong TR-locked coherence.
    """
    if n_epochs_eval < 3:
        raise ValueError("n_epochs_eval must be at least 3.")

    signal_2d, _ = _as_2d(signal)
    _validate_reference_channel(signal_2d, reference_channel)

    nominal_epoch_len = max(2, int(round(nominal_tr_samples(TR, fs))))
    reference = signal_2d[reference_channel]
    block_len = n_epochs_eval * nominal_epoch_len

    max_end = reference.shape[0]
    min_end = block_len
    if max_end < min_end:
        raise ValueError("Signal is too short to estimate the fMRI end.")

    if search_step is None:
        search_step = max(1, nominal_epoch_len // 8)
    if search_step < 1:
        raise ValueError("search_step must be at least 1.")

    coarse_candidates = np.arange(min_end, max_end + 1, search_step, dtype=int)
    if coarse_candidates.size == 0 or coarse_candidates[-1] != max_end:
        coarse_candidates = np.append(coarse_candidates, max_end)

    coarse_scores = np.empty(coarse_candidates.shape[0], dtype=np.float64)
    for idx, end in enumerate(coarse_candidates):
        start = int(end - block_len)
        epochs = _segment_reference_for_resync(
            reference,
            nominal_epoch_len,
            start=start,
            n_epochs_limit=n_epochs_eval,
        )
        coarse_scores[idx] = compute_resync_acs(epochs, 1.0)

    coarse_end = _last_stable_high_score(coarse_candidates, coarse_scores)

    refine_left = max(min_end, coarse_end - search_step)
    refine_right = min(max_end, coarse_end + search_step)
    fine_candidates = np.arange(refine_left, refine_right + 1, dtype=int)
    fine_scores = np.empty(fine_candidates.shape[0], dtype=np.float64)
    for idx, end in enumerate(fine_candidates):
        start = int(end - block_len)
        epochs = _segment_reference_for_resync(
            reference,
            nominal_epoch_len,
            start=start,
            n_epochs_limit=n_epochs_eval,
        )
        fine_scores[idx] = compute_resync_acs(epochs, 1.0)

    fmri_end_sample = _last_stable_high_score(fine_candidates, fine_scores)
    fine_best_idx = int(np.where(fine_candidates == fmri_end_sample)[0][0])

    return {
        "fmri_end_sample": fmri_end_sample,
        "fmri_end_time_sec": fmri_end_sample / fs,
        "nominal_tr_samples": nominal_epoch_len,
        "search_step": search_step,
        "n_epochs_eval": n_epochs_eval,
        "coarse_candidates": coarse_candidates,
        "coarse_scores": coarse_scores,
        "fine_candidates": fine_candidates,
        "fine_scores": fine_scores,
        "end_score": float(fine_scores[fine_best_idx]),
    }


def estimate_resync_factor(
    signal: ArrayLike,
    TR: float,
    fs: float,
    reference_channel: int = 0,
    fmri_start_sample: int = 0,
    search_radius: float = 5e-3,
    grid_points: int = 41,
) -> dict[str, Any]:
    """Estimate the Resync desynchronization factor D from one EEG channel."""
    if grid_points < 5:
        raise ValueError("grid_points must be at least 5.")
    if search_radius <= 0:
        raise ValueError("search_radius must be positive.")

    signal_2d, _ = _as_2d(signal)
    _validate_reference_channel(signal_2d, reference_channel)

    nominal_samples = nominal_tr_samples(TR, fs)
    nominal_epoch_len = max(2, int(round(nominal_samples)))

    reference = signal_2d[reference_channel]
    epochs = _segment_reference_for_resync(reference, nominal_epoch_len, start=fmri_start_sample)
    epoch_fft = np.fft.rfft(epochs, axis=1)

    coarse_grid = np.linspace(1.0 - search_radius, 1.0 + search_radius, grid_points)
    coarse_scores = np.empty_like(coarse_grid)

    for idx, D in enumerate(coarse_grid):
        corrected = _fft_phase_corrected_epochs(epoch_fft, float(D))
        coarse_scores[idx] = compute_resync_acs(corrected, 1.0)

    best_idx = int(np.argmax(coarse_scores))
    D0 = float(coarse_grid[best_idx])

    lower = 1.0 - search_radius
    upper = 1.0 + search_radius

    def objective(x: np.ndarray) -> float:
        D = float(x[0])
        if D < lower or D > upper:
            return 1e6 + abs(D - 1.0)
        corrected = _fft_phase_corrected_epochs(epoch_fft, D)
        return -compute_resync_acs(corrected, 1.0)

    result = minimize(
        objective,
        x0=np.array([D0], dtype=np.float64),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 200},
    )

    D_opt = float(result.x[0])
    D_opt = min(max(D_opt, lower), upper)
    nominal_tr = nominal_tr_samples(TR, fs)
    synced_tr_samples = int(round(nominal_tr * D_opt))
    synced_tr_samples = max(1, synced_tr_samples)
    D_applied = synced_tr_samples / nominal_tr
    R_opt = D_opt - 1.0
    acs_nominal = compute_resync_acs(epochs, 1.0)
    acs_optimized = compute_resync_acs(epochs, D_opt)
    acs_applied = compute_resync_acs(epochs, D_applied)

    return {
        "D_opt": D_opt,
        "D_applied": D_applied,
        "R_opt": R_opt,
        "fmri_start_sample": fmri_start_sample,
        "nominal_tr_samples": nominal_tr,
        "synced_tr_samples": synced_tr_samples,
        "acs_nominal": acs_nominal,
        "acs_optimized": acs_optimized,
        "acs_applied": acs_applied,
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "coarse_grid": coarse_grid,
        "coarse_scores": coarse_scores,
    }


def fourier_resync_signal(signal: ArrayLike, D: float) -> tuple[np.ndarray, int]:
    """Apply global equidistant Fourier interpolation to resync the EEG.

    The full recording is mapped to a new time grid with ``Round(N * D)``
    samples. This keeps the interpolation global and phase-consistent across
    the entire channel instead of relying on piecewise fixed-rate upsampling.
    """
    if D <= 0:
        raise ValueError("D must be strictly positive.")

    signal_2d, was_1d = _as_2d(signal)
    n_out = int(round(signal_2d.shape[1] * D))
    n_out = max(1, n_out)
    resynced = resample(signal_2d, n_out, axis=1)
    if was_1d:
        return resynced[0], n_out
    return resynced, n_out


def run_resync_aas_pipeline(
    signal: ArrayLike,
    TR: float,
    fs: float,
    reference_channel: int = 0,
    fmri_start_sample: int | None = None,
    fmri_end_sample: int | None = None,
    start_search_step: int | None = None,
    end_search_step: int | None = None,
    start_n_epochs_eval: int = 6,
    end_n_epochs_eval: int = 6,
    search_radius: float = 5e-3,
    grid_points: int = 41,
    n_iter: int = 5,
    window_size: int = 21,
    max_lag: int | None = None,
    variance_threshold: float | None = None,
) -> dict[str, Any]:
    """Estimate fMRI start/end, run Resync, and apply AAS only on the fMRI part.

    After Resync, the new effective sampling rate becomes ``fs * D_applied`` so
    that one fMRI TR corresponds to an integer number of EEG samples. AAS is
    then applied only from the detected fMRI onset onward, leaving the earlier
    pre-scan EEG untouched.
    """
    signal_2d, was_1d = _as_2d(signal)
    if fmri_start_sample is None:
        start_info = estimate_fmri_start(
            signal=signal_2d,
            TR=TR,
            fs=fs,
            reference_channel=reference_channel,
            search_step=start_search_step,
            n_epochs_eval=start_n_epochs_eval,
        )
        fmri_start_sample = start_info["fmri_start_sample"]
    else:
        start_info = {
            "fmri_start_sample": int(fmri_start_sample),
            "fmri_start_time_sec": float(fmri_start_sample) / fs,
            "search_step": start_search_step,
            "n_epochs_eval": start_n_epochs_eval,
        }

    if fmri_end_sample is None:
        end_info = estimate_fmri_end(
            signal=signal_2d,
            TR=TR,
            fs=fs,
            reference_channel=reference_channel,
            search_step=end_search_step,
            n_epochs_eval=end_n_epochs_eval,
        )
        fmri_end_sample = end_info["fmri_end_sample"]
    else:
        end_info = {
            "fmri_end_sample": int(fmri_end_sample),
            "fmri_end_time_sec": float(fmri_end_sample) / fs,
            "search_step": end_search_step,
            "n_epochs_eval": end_n_epochs_eval,
        }

    fmri_start_sample = int(fmri_start_sample)
    fmri_end_sample = int(fmri_end_sample)
    if fmri_start_sample < 0 or fmri_start_sample >= signal_2d.shape[1]:
        raise ValueError("fmri_start_sample is out of bounds.")
    if fmri_end_sample <= fmri_start_sample or fmri_end_sample > signal_2d.shape[1]:
        raise ValueError("fmri_end_sample must satisfy start < end <= n_samples.")

    fmri_signal = signal_2d[:, fmri_start_sample:fmri_end_sample]

    if fmri_signal.shape[1] < max(3, int(round(nominal_tr_samples(TR, fs)))):
        raise ValueError("The detected fMRI interval is too short for Resync/AAS.")

    resync_info = estimate_resync_factor(
        signal=fmri_signal,
        TR=TR,
        fs=fs,
        reference_channel=reference_channel,
        fmri_start_sample=0,
        search_radius=search_radius,
        grid_points=grid_points,
    )

    resynced_fmri_signal, n_resynced = fourier_resync_signal(fmri_signal, resync_info["D_applied"])
    fs_resynced = fs * resync_info["D_applied"]

    aas_result = run_aas_pipeline(
        signal=resynced_fmri_signal,
        TR=TR,
        fs=fs_resynced,
        reference_channel=reference_channel,
        n_iter=n_iter,
        window_size=window_size,
        max_lag=max_lag,
        variance_threshold=variance_threshold,
        offset=0,
    )

    cleaned_fmri_2d, _ = _as_2d(aas_result["cleaned_signal"])
    fmri_length_resynced = resynced_fmri_signal.shape[1]
    total_length_resynced = (
        fmri_start_sample
        + fmri_length_resynced
        + (signal_2d.shape[1] - fmri_end_sample)
    )

    resynced_full = np.zeros((signal_2d.shape[0], total_length_resynced), dtype=np.float64)
    cleaned_full = np.zeros_like(resynced_full)

    pre_fmri = signal_2d[:, :fmri_start_sample]
    post_fmri = signal_2d[:, fmri_end_sample:]

    resynced_full[:, :fmri_start_sample] = pre_fmri
    cleaned_full[:, :fmri_start_sample] = pre_fmri

    fmri_stop_resynced = fmri_start_sample + fmri_length_resynced
    resynced_full[:, fmri_start_sample:fmri_stop_resynced] = resynced_fmri_signal
    cleaned_full[:, fmri_start_sample:fmri_stop_resynced] = cleaned_fmri_2d

    resynced_full[:, fmri_stop_resynced:] = post_fmri
    cleaned_full[:, fmri_stop_resynced:] = post_fmri

    if was_1d:
        resynced_signal_out = resynced_full[0]
        cleaned_signal_out = cleaned_full[0]
    else:
        resynced_signal_out = resynced_full
        cleaned_signal_out = cleaned_full

    output: dict[str, Any] = {
        "resynced_signal": resynced_signal_out,
        "n_resynced_samples": total_length_resynced,
        "fs_resynced": fs_resynced,
        "fmri_start_sample": fmri_start_sample,
        "fmri_start_time_sec": start_info["fmri_start_time_sec"],
        "fmri_end_sample": fmri_end_sample,
        "fmri_end_time_sec": end_info["fmri_end_time_sec"],
        "fmri_start_sample_resynced": fmri_start_sample,
        "fmri_start_time_sec_resynced": fmri_start_sample / fs_resynced,
        "fmri_end_sample_resynced": fmri_stop_resynced,
        "fmri_end_time_sec_resynced": fmri_stop_resynced / fs_resynced,
        "start_detection": start_info,
        "end_detection": end_info,
        "D_opt": resync_info["D_opt"],
        "D_applied": resync_info["D_applied"],
        "R_opt": resync_info["R_opt"],
        "nominal_tr_samples": resync_info["nominal_tr_samples"],
        "synced_tr_samples": resync_info["synced_tr_samples"],
        "acs_nominal": resync_info["acs_nominal"],
        "acs_optimized": resync_info["acs_optimized"],
        "acs_applied": resync_info["acs_applied"],
        "optimizer_success": resync_info["optimizer_success"],
        "optimizer_message": resync_info["optimizer_message"],
        "coarse_grid": resync_info["coarse_grid"],
        "coarse_scores": resync_info["coarse_scores"],
        "aas": aas_result,
        "cleaned_signal": cleaned_signal_out,
    }
    return output

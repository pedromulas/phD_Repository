from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import CubicSpline

from functions.aas_ga import _as_2d, find_best_offset, run_aas_pipeline, tr_to_samples
from functions.aas_Resync import estimate_fmri_end, estimate_fmri_start


ArrayLike = np.ndarray

if TYPE_CHECKING:
    import mne


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive.")


def _linear_resample_fixed_tr(
    signal: np.ndarray,
    h: float,
    tr_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample using linear interpolation while keeping a fixed TR length."""
    if h <= 0:
        raise ValueError("h must be strictly positive.")
    if tr_samples < 2:
        raise ValueError("tr_samples must be at least 2.")

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be 1D.")

    n_output = int(np.floor(signal.size * h))
    n_output -= n_output % tr_samples
    if n_output < 2 * tr_samples:
        raise ValueError("The resampled signal is too short for LASV optimization.")

    sample_positions = np.arange(signal.size, dtype=np.float64)
    query_positions = np.arange(n_output, dtype=np.float64) / h
    resampled = np.interp(
        query_positions,
        sample_positions,
        signal,
        left=signal[0],
        right=signal[-1],
    )
    return resampled, query_positions


def _segment_into_epochs(signal: np.ndarray, tr_samples: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be 1D.")
    n_epochs = signal.size // tr_samples
    if n_epochs < 2:
        raise ValueError("At least two complete epochs are required.")
    trimmed = signal[: n_epochs * tr_samples]
    return trimmed.reshape(n_epochs, tr_samples)


def lasv_objective(signal: np.ndarray, h: float, tr_samples: int) -> float:
    """Mean across-time variance of TR-sized epochs after linear interpolation."""
    resampled, _ = _linear_resample_fixed_tr(signal, h, tr_samples)
    epochs = _segment_into_epochs(resampled, tr_samples)
    return float(np.mean(np.var(epochs, axis=0, ddof=0)))


def _finite_difference_gradient(
    signal: np.ndarray,
    h: float,
    tr_samples: int,
    epsilon: float = 1e-6,
) -> float:
    h_plus = h + epsilon
    h_minus = max(h - epsilon, 1e-8)
    j_plus = lasv_objective(signal, h_plus, tr_samples)
    j_minus = lasv_objective(signal, h_minus, tr_samples)
    return float((j_plus - j_minus) / (h_plus - h_minus))


def select_lowest_snr_channel(
    signal: ArrayLike,
    fs: float,
    tr: float,
    n_iter: int = 3,
    window_size: int = 21,
    max_lag: int | None = 10,
) -> dict[str, Any]:
    """Select the channel with the strongest gradient artifact (lowest SNR)."""
    signal_2d, _ = _as_2d(signal)
    preliminary = run_aas_pipeline(
        signal=signal_2d,
        TR=tr,
        fs=fs,
        reference_channel=0,
        n_iter=n_iter,
        window_size=window_size,
        max_lag=max_lag,
    )

    cleaned_2d, _ = _as_2d(preliminary["cleaned_signal"])
    ga_estimate = signal_2d - cleaned_2d

    residual_var = np.var(cleaned_2d, axis=1, ddof=0)
    ga_var = np.var(ga_estimate, axis=1, ddof=0)
    snr = residual_var / np.maximum(ga_var, np.finfo(np.float64).eps)

    channel_idx = int(np.argmin(snr))
    return {
        "channel_index": channel_idx,
        "snr_per_channel": snr,
        "residual_variance": residual_var,
        "ga_variance": ga_var,
        "preliminary_aas": preliminary,
    }


def optimize_lasv_factor(
    reference_signal: np.ndarray,
    fs: float,
    tr: float,
    h0: float = 1.0,
    learning_rate: float = 1e-4,
    max_iter: int = 200,
    tol: float = 1e-10,
    gradient_epsilon: float = 1e-6,
    h_bounds: tuple[float, float] = (0.995, 1.005),
) -> dict[str, Any]:
    """Optimize the LASV scale factor h with batch gradient descent."""
    _validate_positive("fs", fs)
    _validate_positive("tr", tr)
    tr_samples = tr_to_samples(tr, fs)
    lower, upper = h_bounds
    if lower <= 0 or upper <= lower:
        raise ValueError("h_bounds must satisfy 0 < lower < upper.")

    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    if reference_signal.ndim != 1:
        raise ValueError("reference_signal must be 1D.")

    h = float(np.clip(h0, lower, upper))
    objective_history: list[float] = []
    gradient_history: list[float] = []
    h_history: list[float] = [h]

    for _ in range(max_iter):
        current_objective = lasv_objective(reference_signal, h, tr_samples)
        gradient = _finite_difference_gradient(
            reference_signal,
            h,
            tr_samples,
            epsilon=gradient_epsilon,
        )

        objective_history.append(current_objective)
        gradient_history.append(gradient)

        if abs(gradient) < tol:
            break

        step = learning_rate
        improved = False
        while step >= 1e-12:
            candidate_h = float(np.clip(h - step * gradient, lower, upper))
            candidate_objective = lasv_objective(reference_signal, candidate_h, tr_samples)
            if candidate_objective <= current_objective:
                h = candidate_h
                h_history.append(h)
                improved = True
                break
            step *= 0.5

        if not improved:
            break

        if len(h_history) >= 2 and abs(h_history[-1] - h_history[-2]) < tol:
            break

    final_objective = lasv_objective(reference_signal, h, tr_samples)
    return {
        "h_opt": h,
        "fs_opt": fs * h,
        "tr_samples": tr_samples,
        "objective": final_objective,
        "objective_history": np.asarray(objective_history, dtype=np.float64),
        "gradient_history": np.asarray(gradient_history, dtype=np.float64),
        "h_history": np.asarray(h_history, dtype=np.float64),
        "n_iterations": len(objective_history),
    }


def cubic_spline_resample(signal: ArrayLike, h: float) -> np.ndarray:
    """Resample the full multichannel signal using cubic spline interpolation."""
    signal_2d, was_1d = _as_2d(signal)
    if h <= 0:
        raise ValueError("h must be strictly positive.")

    n_samples = signal_2d.shape[1]
    n_output = int(np.floor(n_samples * h))
    n_output = max(2, n_output)

    original_positions = np.arange(n_samples, dtype=np.float64)
    target_positions = np.arange(n_output, dtype=np.float64) / h
    target_positions = np.clip(target_positions, 0.0, n_samples - 1.0)

    resampled = np.empty((signal_2d.shape[0], n_output), dtype=np.float64)
    for ch_idx in range(signal_2d.shape[0]):
        spline = CubicSpline(original_positions, signal_2d[ch_idx], bc_type="natural")
        resampled[ch_idx] = spline(target_positions)

    if was_1d:
        return resampled[0]
    return resampled


def run_lasv_aas_pipeline(
    signal: ArrayLike,
    fs: float = 1000.0,
    tr: float = 2.0,
    use_fmri_bounds: bool = False,
    fmri_start_sample: int | None = None,
    fmri_end_sample: int | None = None,
    start_search_step: int | None = None,
    end_search_step: int | None = None,
    start_n_epochs_eval: int = 6,
    end_n_epochs_eval: int = 6,
    learning_rate: float = 1e-4,
    max_iter: int = 200,
    tol: float = 1e-10,
    h_bounds: tuple[float, float] = (0.995, 1.005),
    initial_h: float = 1.0,
    n_iter_aas: int = 5,
    window_size_aas: int = 21,
    max_lag_aas: int | None = 10,
    variance_threshold_aas: float | None = None,
) -> dict[str, Any]:
    """Run LASV synchronization followed by AAS on the full signal or on the detected fMRI interval."""
    signal_2d, was_1d = _as_2d(signal)
    _validate_positive("fs", fs)
    _validate_positive("tr", tr)

    if not use_fmri_bounds:
        channel_info = select_lowest_snr_channel(
            signal=signal_2d,
            fs=fs,
            tr=tr,
            n_iter=min(n_iter_aas, 3),
            window_size=window_size_aas,
            max_lag=max_lag_aas,
        )
        reference_channel = int(channel_info["channel_index"])

        reference_signal = signal_2d[reference_channel]
        optimization = optimize_lasv_factor(
            reference_signal=reference_signal,
            fs=fs,
            tr=tr,
            h0=initial_h,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            h_bounds=h_bounds,
        )

        h_opt = float(optimization["h_opt"])
        resampled_signal = cubic_spline_resample(signal_2d, h_opt)
        fs_opt = float(optimization["fs_opt"])
        tr_samples_opt = tr_to_samples(tr, fs_opt)
        offset_opt = find_best_offset(resampled_signal, tr_samples_opt, reference_channel=reference_channel)

        aas_result = run_aas_pipeline(
            signal=resampled_signal,
            TR=tr,
            fs=fs_opt,
            reference_channel=reference_channel,
            n_iter=n_iter_aas,
            window_size=window_size_aas,
            max_lag=max_lag_aas,
            variance_threshold=variance_threshold_aas,
            offset=offset_opt,
        )

        cleaned_signal = aas_result["cleaned_signal"]
        if was_1d:
            resampled_out = resampled_signal[0]
            cleaned_out = cleaned_signal
        else:
            resampled_out = resampled_signal
            cleaned_out = cleaned_signal

        return {
            "reference_channel": reference_channel,
            "channel_selection": channel_info,
            "use_fmri_bounds": False,
            "fmri_start_sample": 0,
            "fmri_start_time_sec": 0.0,
            "fmri_end_sample": signal_2d.shape[1],
            "fmri_end_time_sec": signal_2d.shape[1] / fs,
            "fmri_start_sample_resampled": 0,
            "fmri_start_time_sec_resampled": 0.0,
            "fmri_end_sample_resampled": resampled_signal.shape[1],
            "fmri_end_time_sec_resampled": resampled_signal.shape[1] / fs_opt,
            "start_detection": {
                "fmri_start_sample": 0,
                "fmri_start_time_sec": 0.0,
                "mode": "full_signal",
            },
            "end_detection": {
                "fmri_end_sample": signal_2d.shape[1],
                "fmri_end_time_sec": signal_2d.shape[1] / fs,
                "mode": "full_signal",
            },
            "h_opt": h_opt,
            "fs_original": fs,
            "fs_opt": fs_opt,
            "tr": tr,
            "tr_samples_original": tr_to_samples(tr, fs),
            "tr_samples_opt": tr_samples_opt,
            "optimization": optimization,
            "resampled_signal": resampled_out,
            "cleaned_signal": cleaned_out,
            "offset_opt": offset_opt,
            "aas": aas_result,
        }

    if use_fmri_bounds:
        if fmri_start_sample is None:
            start_info = estimate_fmri_start(
                signal=signal_2d,
                TR=tr,
                fs=fs,
                reference_channel=0,
                search_step=start_search_step,
                n_epochs_eval=start_n_epochs_eval,
            )
            fmri_start_sample = int(start_info["fmri_start_sample"])
        else:
            fmri_start_sample = int(fmri_start_sample)
            start_info = {
                "fmri_start_sample": fmri_start_sample,
                "fmri_start_time_sec": fmri_start_sample / fs,
                "search_step": start_search_step,
                "n_epochs_eval": start_n_epochs_eval,
                "mode": "manual",
            }

        if fmri_end_sample is None:
            end_info = estimate_fmri_end(
                signal=signal_2d,
                TR=tr,
                fs=fs,
                reference_channel=0,
                search_step=end_search_step,
                n_epochs_eval=end_n_epochs_eval,
            )
            fmri_end_sample = int(end_info["fmri_end_sample"])
        else:
            fmri_end_sample = int(fmri_end_sample)
            end_info = {
                "fmri_end_sample": fmri_end_sample,
                "fmri_end_time_sec": fmri_end_sample / fs,
                "search_step": end_search_step,
                "n_epochs_eval": end_n_epochs_eval,
                "mode": "manual",
            }
    if fmri_start_sample < 0 or fmri_start_sample >= signal_2d.shape[1]:
        raise ValueError("fmri_start_sample is out of bounds.")
    if fmri_end_sample <= fmri_start_sample or fmri_end_sample > signal_2d.shape[1]:
        raise ValueError("fmri_end_sample must satisfy start < end <= n_samples.")

    fmri_signal = signal_2d[:, fmri_start_sample:fmri_end_sample]
    if fmri_signal.shape[1] < max(3, tr_to_samples(tr, fs)):
        raise ValueError("The detected fMRI interval is too short for LASV/AAS.")

    channel_info = select_lowest_snr_channel(
        signal=fmri_signal,
        fs=fs,
        tr=tr,
        n_iter=min(n_iter_aas, 3),
        window_size=window_size_aas,
        max_lag=max_lag_aas,
    )
    reference_channel = int(channel_info["channel_index"])

    reference_signal = fmri_signal[reference_channel]
    optimization = optimize_lasv_factor(
        reference_signal=reference_signal,
        fs=fs,
        tr=tr,
        h0=initial_h,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        h_bounds=h_bounds,
    )

    h_opt = float(optimization["h_opt"])
    resampled_signal = cubic_spline_resample(signal_2d, h_opt)
    fs_opt = float(optimization["fs_opt"])
    fmri_start_sample_resampled = int(round(fmri_start_sample * h_opt))
    fmri_end_sample_resampled = int(round(fmri_end_sample * h_opt))
    fmri_start_sample_resampled = max(0, min(fmri_start_sample_resampled, resampled_signal.shape[1] - 1))
    fmri_end_sample_resampled = max(
        fmri_start_sample_resampled + 1,
        min(fmri_end_sample_resampled, resampled_signal.shape[1]),
    )
    resampled_fmri_signal = resampled_signal[:, fmri_start_sample_resampled:fmri_end_sample_resampled]

    tr_samples_opt = tr_to_samples(tr, fs_opt)
    offset_opt = find_best_offset(
        resampled_fmri_signal,
        tr_samples_opt,
        reference_channel=reference_channel,
    )

    aas_result = run_aas_pipeline(
        signal=resampled_fmri_signal,
        TR=tr,
        fs=fs_opt,
        reference_channel=reference_channel,
        n_iter=n_iter_aas,
        window_size=window_size_aas,
        max_lag=max_lag_aas,
        variance_threshold=variance_threshold_aas,
        offset=offset_opt,
    )

    cleaned_fmri_signal, _ = _as_2d(aas_result["cleaned_signal"])
    aligned_segments = np.asarray(aas_result["aligned_segments"])
    if aligned_segments.ndim == 2:
        n_complete_segments = int(aligned_segments.shape[0])
    else:
        n_complete_segments = int(aligned_segments.shape[1])
    aas_covered_start = int(offset_opt)
    aas_covered_stop = min(
        resampled_fmri_signal.shape[1],
        aas_covered_start + n_complete_segments * tr_samples_opt,
    )

    cleaned_signal_2d = np.asarray(resampled_signal, dtype=np.float64).copy()
    cleaned_signal_2d[
        :,
        fmri_start_sample_resampled + aas_covered_start : fmri_start_sample_resampled + aas_covered_stop,
    ] = cleaned_fmri_signal[:, aas_covered_start:aas_covered_stop]

    if was_1d:
        resampled_out = resampled_signal[0]
        cleaned_out = cleaned_signal_2d[0]
    else:
        resampled_out = resampled_signal
        cleaned_out = cleaned_signal_2d

    return {
        "reference_channel": reference_channel,
        "channel_selection": channel_info,
        "use_fmri_bounds": use_fmri_bounds,
        "fmri_start_sample": fmri_start_sample,
        "fmri_start_time_sec": start_info["fmri_start_time_sec"],
        "fmri_end_sample": fmri_end_sample,
        "fmri_end_time_sec": end_info["fmri_end_time_sec"],
        "fmri_start_sample_resampled": fmri_start_sample_resampled,
        "fmri_start_time_sec_resampled": fmri_start_sample_resampled / fs_opt,
        "fmri_end_sample_resampled": fmri_end_sample_resampled,
        "fmri_end_time_sec_resampled": fmri_end_sample_resampled / fs_opt,
        "start_detection": start_info,
        "end_detection": end_info,
        "h_opt": h_opt,
        "fs_original": fs,
        "fs_opt": fs_opt,
        "tr": tr,
        "tr_samples_original": tr_to_samples(tr, fs),
        "tr_samples_opt": tr_samples_opt,
        "optimization": optimization,
        "resampled_signal": resampled_out,
        "cleaned_signal": cleaned_out,
        "offset_opt": offset_opt,
        "aas": aas_result,
    }


def run_lasv_aas_on_raw(
    raw: "mne.io.BaseRaw",
    tr: float = 2.0,
    use_fmri_bounds: bool = False,
    fmri_start_sample: int | None = None,
    fmri_end_sample: int | None = None,
    start_search_step: int | None = None,
    end_search_step: int | None = None,
    start_n_epochs_eval: int = 6,
    end_n_epochs_eval: int = 6,
    learning_rate: float = 1e-4,
    max_iter: int = 200,
    tol: float = 1e-10,
    h_bounds: tuple[float, float] = (0.995, 1.005),
    initial_h: float = 1.0,
    n_iter_aas: int = 5,
    window_size_aas: int = 21,
    max_lag_aas: int | None = 10,
    variance_threshold_aas: float | None = None,
) -> dict[str, Any]:
    """Apply LASV + AAS directly to an MNE Raw object."""
    import mne

    data = raw.get_data()
    fs = float(raw.info["sfreq"])

    result = run_lasv_aas_pipeline(
        signal=data,
        fs=fs,
        tr=tr,
        use_fmri_bounds=use_fmri_bounds,
        fmri_start_sample=fmri_start_sample,
        fmri_end_sample=fmri_end_sample,
        start_search_step=start_search_step,
        end_search_step=end_search_step,
        start_n_epochs_eval=start_n_epochs_eval,
        end_n_epochs_eval=end_n_epochs_eval,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        h_bounds=h_bounds,
        initial_h=initial_h,
        n_iter_aas=n_iter_aas,
        window_size_aas=window_size_aas,
        max_lag_aas=max_lag_aas,
        variance_threshold_aas=variance_threshold_aas,
    )

    cleaned_2d, _ = _as_2d(result["cleaned_signal"])
    info = mne.create_info(
        ch_names=raw.ch_names,
        sfreq=result["fs_opt"],
        ch_types=raw.get_channel_types(),
    )
    for key in ("bads", "description", "experimenter", "line_freq", "subject_info"):
        if key in raw.info:
            info[key] = raw.info[key]
    cleaned_raw = mne.io.RawArray(cleaned_2d, info, verbose="ERROR")

    result["cleaned_raw"] = cleaned_raw
    return result

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import warnings

if os.name == "nt" and "_MNE_FAKE_HOME_DIR" not in os.environ:
    _local_mne_home = Path(__file__).resolve().parents[2] / ".mne_local"
    _local_mne_home.mkdir(exist_ok=True)
    os.environ["_MNE_FAKE_HOME_DIR"] = str(_local_mne_home)

import mne
import numpy as np
from scipy.signal import welch

try:
    from independent_vector_analysis.iva_g import iva_g
    from independent_vector_analysis.iva_l_sos import iva_l_sos

    IVA_LIBRARY_AVAILABLE = True
    IVA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only when IVA is unavailable
    IVA_LIBRARY_AVAILABLE = False
    IVA_IMPORT_ERROR = exc


ArrayLike = np.ndarray


def tr_to_samples(TR: float, fs: float) -> int:
    """Convert the fMRI repetition time from seconds to samples."""
    if TR <= 0:
        raise ValueError("TR must be strictly positive.")
    if fs <= 0:
        raise ValueError("fs must be strictly positive.")

    samples = int(round(TR * fs))
    if samples < 1:
        raise ValueError("TR * fs must correspond to at least one sample.")
    return samples


def _as_2d(signal: ArrayLike) -> tuple[np.ndarray, bool]:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        return signal[np.newaxis, :], True
    if signal.ndim == 2:
        return signal, False
    raise ValueError("signal must be a 1D or 2D NumPy array.")


def segment_signal(signal: ArrayLike, T_samples: int, offset: int = 0) -> np.ndarray:
    """Segment a continuous signal into TR-sized epochs.

    Returns
    -------
    np.ndarray
        Segments with shape ``(n_epochs, T_samples)`` for 1D input or
        ``(n_channels, n_epochs, T_samples)`` for 2D input.
    """
    if T_samples <= 0:
        raise ValueError("T_samples must be strictly positive.")
    if offset < 0 or offset >= T_samples:
        raise ValueError("offset must satisfy 0 <= offset < T_samples.")

    signal_2d, was_1d = _as_2d(signal)
    n_channels, n_samples = signal_2d.shape
    usable_samples = n_samples - offset
    n_epochs = usable_samples // T_samples
    if n_epochs < 2:
        raise ValueError("At least two complete TR epochs are required for IVA.")

    trimmed = signal_2d[:, offset : offset + n_epochs * T_samples]
    segmented = trimmed.reshape(n_channels, n_epochs, T_samples)
    if was_1d:
        return segmented[0]
    return segmented


def epochs_to_iva_datasets(epochs: np.ndarray) -> np.ndarray:
    """Convert MNE-style epochs to the IVA convention ``N x T x K``.

    The paper describes one dataset per channel. Under that convention:
    ``N = n_epochs``, ``T = n_samples_per_epoch``, and ``K = n_channels``.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    if epochs.ndim != 3:
        raise ValueError("epochs must have shape (n_channels, n_epochs, n_samples).")
    return np.transpose(epochs, (1, 2, 0))


def iva_datasets_to_epochs(X: np.ndarray) -> np.ndarray:
    """Convert IVA data back to ``(n_channels, n_epochs, n_samples)``."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError("X must have shape (n_sources, n_samples, n_channels).")
    return np.transpose(X, (2, 0, 1))


def reconstruct_continuous_signal(
    segmented: np.ndarray,
    original_n_samples: int,
    offset: int,
) -> np.ndarray:
    """Restore segmented data into a continuous channels x samples array."""
    segmented = np.asarray(segmented, dtype=np.float64)
    if segmented.ndim != 3:
        raise ValueError("segmented must have shape (n_channels, n_epochs, n_samples).")

    n_channels, n_epochs, T_samples = segmented.shape
    usable = n_epochs * T_samples
    reconstructed = np.zeros((n_channels, original_n_samples), dtype=np.float64)
    reconstructed[:, offset : offset + usable] = segmented.reshape(n_channels, usable)
    return reconstructed


def _histogram_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 32,
    eps: float = 1e-12,
) -> float:
    """Estimate MI with a 2D histogram for robust, dependency-free scoring."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same number of samples.")
    if x.size < 4:
        return 0.0

    hist, _, _ = np.histogram2d(x, y, bins=n_bins)
    prob_xy = hist / np.maximum(hist.sum(), eps)
    prob_x = prob_xy.sum(axis=1, keepdims=True)
    prob_y = prob_xy.sum(axis=0, keepdims=True)
    mask = prob_xy > 0
    mi = np.sum(prob_xy[mask] * np.log(prob_xy[mask] / np.maximum((prob_x @ prob_y)[mask], eps)))
    return float(mi)


def _harmonic_power_ratio(
    signal: np.ndarray,
    fs: float,
    base_frequency: float = 14.0,
    n_harmonics: int = 6,
    bandwidth_hz: float = 1.0,
) -> float:
    """Measure how strongly a source concentrates power at 14 Hz harmonics."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be 1D.")
    if signal.size < 8:
        return 0.0

    freqs, psd = welch(signal, fs=fs, nperseg=min(signal.size, max(64, min(512, signal.size))))
    total_power = float(np.sum(psd))
    if total_power <= 0:
        return 0.0

    harmonic_power = 0.0
    nyquist = fs / 2.0
    for harmonic in range(1, n_harmonics + 1):
        center = harmonic * base_frequency
        if center >= nyquist:
            break
        mask = np.abs(freqs - center) <= bandwidth_hz
        harmonic_power += float(np.sum(psd[mask]))
    return harmonic_power / total_power


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    vmin = np.min(values)
    vmax = np.max(values)
    if np.isclose(vmax, vmin):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def run_iva_gl(
    X: np.ndarray,
    iva_g_max_iter: int = 256,
    iva_l_max_iter: int = 512,
    whiten: bool = True,
    verbose: bool = False,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Run IVA-G followed by IVA-L-SOS as an IVA-GL approximation."""
    if not IVA_LIBRARY_AVAILABLE:
        raise RuntimeError(
            "The package independent_vector_analysis could not be imported. "
            "A direct IVA fallback is not implemented in this repository yet."
        ) from IVA_IMPORT_ERROR

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError("X must have shape (n_sources, n_samples, n_channels).")
    if X.shape[2] < 2:
        raise ValueError("IVA requires at least two datasets/channels.")

    if random_state is not None:
        np.random.seed(random_state)

    W_g, cost_g, sigma_g, _ = iva_g(
        X,
        whiten=whiten,
        verbose=verbose,
        max_iter=iva_g_max_iter,
    )
    W_l, cost_l, sigma_l, _ = iva_l_sos(
        X,
        whiten=whiten,
        verbose=verbose,
        W_init=W_g,
        iva_g_initW=False,
        max_iter=iva_l_max_iter,
    )

    Y = np.zeros_like(X)
    A = np.zeros_like(W_l)
    for k in range(X.shape[2]):
        Y[:, :, k] = W_l[:, :, k] @ X[:, :, k]
        A[:, :, k] = np.linalg.pinv(W_l[:, :, k])

    return {
        "W_g": W_g,
        "W_l": W_l,
        "A_l": A,
        "Y": Y,
        "cost_g": np.asarray(cost_g, dtype=np.float64),
        "cost_l": np.asarray(cost_l, dtype=np.float64),
        "sigma_g": np.asarray(sigma_g, dtype=np.float64),
        "sigma_l": np.asarray(sigma_l, dtype=np.float64),
        "library_used": "independent_vector_analysis 0.3.6",
    }


def compute_scv_scores(
    raw_epochs: np.ndarray,
    Y: np.ndarray,
    sigma_scv: np.ndarray,
    fs: float,
    periodic_frequency_hz: float = 14.0,
    mi_bins: int = 32,
) -> dict[str, np.ndarray]:
    """Score SCVs by dependence, raw similarity, and 14 Hz harmonic structure."""
    raw_epochs = np.asarray(raw_epochs, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    sigma_scv = np.asarray(sigma_scv, dtype=np.float64)

    if raw_epochs.ndim != 3:
        raise ValueError("raw_epochs must have shape (n_channels, n_epochs, n_samples).")
    if Y.ndim != 3:
        raise ValueError("Y must have shape (n_components, n_samples, n_channels).")

    n_channels, n_epochs, n_samples = raw_epochs.shape
    n_components = Y.shape[0]
    if Y.shape[1] != n_samples or Y.shape[2] != n_channels:
        raise ValueError("Y shape is inconsistent with raw_epochs.")
    if sigma_scv.shape != (n_channels, n_channels, n_components):
        raise ValueError("sigma_scv must have shape (n_channels, n_channels, n_components).")

    channel_templates = np.mean(raw_epochs, axis=1)

    dependence_scores = np.zeros(n_components, dtype=np.float64)
    mi_scores = np.zeros(n_components, dtype=np.float64)
    harmonic_scores = np.zeros(n_components, dtype=np.float64)

    for component_idx in range(n_components):
        sigma = sigma_scv[:, :, component_idx]
        diagonal = np.sqrt(np.clip(np.diag(sigma), 1e-12, None))
        corr = sigma / np.outer(diagonal, diagonal)
        dependence_scores[component_idx] = np.mean(np.abs(corr - np.eye(n_channels)))

        mi_per_channel = []
        harmonic_per_channel = []
        for channel_idx in range(n_channels):
            source_signal = Y[component_idx, :, channel_idx]
            template = channel_templates[channel_idx]
            mi_per_channel.append(
                _histogram_mutual_information(source_signal, template, n_bins=mi_bins)
            )
            harmonic_per_channel.append(
                _harmonic_power_ratio(
                    source_signal,
                    fs=fs,
                    base_frequency=periodic_frequency_hz,
                )
            )

        mi_scores[component_idx] = float(np.mean(mi_per_channel))
        harmonic_scores[component_idx] = float(np.mean(harmonic_per_channel))

    combined_score = (
        _normalize_scores(dependence_scores)
        + _normalize_scores(mi_scores)
        + _normalize_scores(harmonic_scores)
    ) / 3.0

    return {
        "dependence_scores": dependence_scores,
        "mi_scores": mi_scores,
        "harmonic_scores": harmonic_scores,
        "combined_scores": combined_score,
    }


def identify_ga_component(
    raw_epochs: np.ndarray,
    Y: np.ndarray,
    sigma_scv: np.ndarray,
    fs: float,
    periodic_frequency_hz: float = 14.0,
    mi_bins: int = 32,
) -> dict[str, Any]:
    """Identify the SCV that best matches the gradient artifact signature."""
    scores = compute_scv_scores(
        raw_epochs=raw_epochs,
        Y=Y,
        sigma_scv=sigma_scv,
        fs=fs,
        periodic_frequency_hz=periodic_frequency_hz,
        mi_bins=mi_bins,
    )
    ga_component = int(np.argmax(scores["combined_scores"]))
    return {
        "ga_component": ga_component,
        **scores,
    }


def reconstruct_without_component(
    X: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    excluded_component: int,
) -> dict[str, np.ndarray]:
    """Back-project the data after suppressing one IVA component."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)

    if Y.ndim != 3 or A.ndim != 3:
        raise ValueError("Y and A must have shape (n_components, n_samples, n_channels) and (n_components, n_components, n_channels).")
    if excluded_component < 0 or excluded_component >= Y.shape[0]:
        raise ValueError("excluded_component is out of bounds.")

    component_contributions = np.zeros(
        (Y.shape[0], X.shape[0], X.shape[1], X.shape[2]),
        dtype=np.float64,
    )
    X_clean = np.zeros_like(X)
    Y_masked = Y.copy()
    Y_masked[excluded_component, :, :] = 0.0

    for k in range(X.shape[2]):
        X_clean[:, :, k] = A[:, :, k] @ Y_masked[:, :, k]
        for component_idx in range(Y.shape[0]):
            component_contributions[component_idx, :, :, k] = (
                A[:, [component_idx], k] @ Y[[component_idx], :, k]
            )

    return {
        "X_clean": X_clean,
        "component_contributions": component_contributions,
        "Y_masked": Y_masked,
    }


def _candidate_montages() -> list[str]:
    return [
        "standard_1020",
        "standard_1005",
        "biosemi32",
        "GSN-HydroCel-32",
        "GSN-HydroCel-64_1.0",
    ]


def _count_valid_channel_locs(info: mne.Info) -> int:
    valid = 0
    for ch in info["chs"]:
        loc = np.asarray(ch["loc"][:3], dtype=np.float64)
        if np.all(np.isfinite(loc)) and np.linalg.norm(loc) > 0:
            valid += 1
    return valid


def prepare_info_for_topomaps(raw: mne.io.BaseRaw, picks: list[str] | None = None) -> mne.Info:
    """Build an Info object with usable EEG positions for topomap plotting."""
    raw_topo = raw.copy()
    if picks is not None:
        raw_topo.pick(picks)
    else:
        raw_topo.pick("eeg")

    info = raw_topo.info.copy()
    if _count_valid_channel_locs(info) >= 3:
        return info

    best_info = info
    best_count = _count_valid_channel_locs(info)
    for montage_name in _candidate_montages():
        try:
            candidate = raw_topo.copy()
            candidate.set_montage(mne.channels.make_standard_montage(montage_name), on_missing="ignore")
            candidate_info = candidate.info.copy()
            candidate_count = _count_valid_channel_locs(candidate_info)
            if candidate_count > best_count:
                best_info = candidate_info
                best_count = candidate_count
        except Exception:
            continue

    if best_count < 3:
        raise RuntimeError(
            "Could not infer at least three valid electrode positions for topographic visualization."
        )
    return best_info


def compute_component_topographies(
    component_contributions: np.ndarray,
) -> np.ndarray:
    """Compute one scalp weight per channel from each reconstructed IVA component."""
    component_contributions = np.asarray(component_contributions, dtype=np.float64)
    if component_contributions.ndim != 4:
        raise ValueError(
            "component_contributions must have shape (n_components, n_epochs, n_samples, n_channels)."
        )
    energy = np.sqrt(np.mean(component_contributions**2, axis=(1, 2)))
    return np.asarray(energy, dtype=np.float64)


def plot_iva_component_topomaps(
    raw: mne.io.BaseRaw,
    result: dict[str, Any],
    picks: list[str] | None = None,
    n_cols: int = 5,
    cmap: str = "RdBu_r",
    show: bool = True,
):
    """Plot all extracted SCVs as ICA-style topomaps across the scalp."""
    if "component_contributions_epochs" not in result:
        raise ValueError("result does not contain component_contributions_epochs.")

    info = prepare_info_for_topomaps(raw, picks=picks)
    topographies = compute_component_topographies(result["component_contributions_epochs"])
    component_names = [f"SCV {idx}" for idx in range(topographies.shape[0])]
    titles = []
    combined = result.get("combined_scores")
    harmonic = result.get("harmonic_scores")
    for idx, name in enumerate(component_names):
        pieces = [name]
        if combined is not None:
            pieces.append(f"score={combined[idx]:.2f}")
        if harmonic is not None:
            pieces.append(f"h14={harmonic[idx]:.2f}")
        titles.append("\n".join(pieces))

    fig = mne.viz.plot_topomap(
        topographies[0],
        info,
        show=False,
    )[0].figure
    fig.clf()
    n_components = topographies.shape[0]
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_components / n_cols))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)

    for component_idx, ax in enumerate(axes.ravel()):
        if component_idx >= n_components:
            ax.axis("off")
            continue
        mne.viz.plot_topomap(
            topographies[component_idx],
            info,
            axes=ax,
            show=False,
            cmap=cmap,
            contours=0,
        )
        ax.set_title(titles[component_idx], fontsize=9)

    fig.suptitle("IVA SCV Topographies", fontsize=14)
    fig.tight_layout()
    if show:
        fig.show()
    return fig


def plot_iva_component_interactive(
    raw: mne.io.BaseRaw,
    result: dict[str, Any],
    picks: list[str] | None = None,
):
    """Create an interactive component browser for IVA SCVs in notebooks."""
    info = prepare_info_for_topomaps(raw, picks=picks)
    topographies = compute_component_topographies(result["component_contributions_epochs"])
    Y = result["Y"]
    fs = float(result["fs"])

    def _plot_one(component_idx: int) -> None:
        import matplotlib.pyplot as plt

        component_idx = int(component_idx)
        mean_source = np.mean(Y[component_idx], axis=1)
        time = np.arange(mean_source.size) / fs

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        mne.viz.plot_topomap(
            topographies[component_idx],
            info,
            axes=axes[0],
            show=False,
            contours=0,
        )
        axes[0].set_title(f"SCV {component_idx}")
        axes[1].plot(time, mean_source, linewidth=1.0)
        axes[1].set_title("Mean source waveform across channels")
        axes[1].set_xlabel("Time within TR (s)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    try:
        import ipywidgets as widgets
        from IPython.display import display

        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=topographies.shape[0] - 1,
            step=1,
            description="SCV",
            continuous_update=False,
        )
        out = widgets.interactive_output(_plot_one, {"component_idx": slider})
        display(widgets.VBox([slider, out]))
    except Exception:
        warnings.warn(
            "ipywidgets is not available; falling back to a static plot for SCV 0.",
            RuntimeWarning,
        )
        _plot_one(0)


def run_iva_ga_pipeline(
    signal: ArrayLike,
    TR: float,
    fs: float,
    offset: int = 0,
    periodic_frequency_hz: float = 14.0,
    iva_g_max_iter: int = 256,
    iva_l_max_iter: int = 512,
    whiten: bool = True,
    verbose: bool = False,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Run the full IVA-based gradient artifact removal pipeline."""
    signal_2d, was_1d = _as_2d(signal)
    T_samples = tr_to_samples(TR, fs)
    segmented_signal = segment_signal(signal_2d, T_samples=T_samples, offset=offset)
    X = epochs_to_iva_datasets(segmented_signal)

    iva_result = run_iva_gl(
        X,
        iva_g_max_iter=iva_g_max_iter,
        iva_l_max_iter=iva_l_max_iter,
        whiten=whiten,
        verbose=verbose,
        random_state=random_state,
    )
    component_result = identify_ga_component(
        raw_epochs=segmented_signal,
        Y=iva_result["Y"],
        sigma_scv=iva_result["sigma_l"],
        fs=fs,
        periodic_frequency_hz=periodic_frequency_hz,
    )
    reconstruction = reconstruct_without_component(
        X=X,
        Y=iva_result["Y"],
        A=iva_result["A_l"],
        excluded_component=component_result["ga_component"],
    )

    cleaned_epochs = iva_datasets_to_epochs(reconstruction["X_clean"])
    cleaned_signal = signal_2d.copy()
    reconstructed = reconstruct_continuous_signal(
        cleaned_epochs,
        original_n_samples=signal_2d.shape[1],
        offset=offset,
    )
    usable_samples = cleaned_epochs.shape[1] * cleaned_epochs.shape[2]
    cleaned_signal[:, offset : offset + usable_samples] = reconstructed[:, offset : offset + usable_samples]

    result: dict[str, Any] = {
        "cleaned_signal": cleaned_signal[0] if was_1d else cleaned_signal,
        "segmented_signal": segmented_signal,
        "cleaned_epochs": cleaned_epochs,
        "X": X,
        "Y": iva_result["Y"],
        "W_g": iva_result["W_g"],
        "W_l": iva_result["W_l"],
        "A_l": iva_result["A_l"],
        "cost_g": iva_result["cost_g"],
        "cost_l": iva_result["cost_l"],
        "sigma_g": iva_result["sigma_g"],
        "sigma_l": iva_result["sigma_l"],
        "T_samples": T_samples,
        "offset": offset,
        "TR": float(TR),
        "fs": float(fs),
        "library_used": iva_result["library_used"],
        "ga_component": component_result["ga_component"],
        "dependence_scores": component_result["dependence_scores"],
        "mi_scores": component_result["mi_scores"],
        "harmonic_scores": component_result["harmonic_scores"],
        "combined_scores": component_result["combined_scores"],
        "component_contributions": reconstruction["component_contributions"],
        "component_contributions_epochs": reconstruction["component_contributions"],
        "Y_masked": reconstruction["Y_masked"],
        "cleaned_X": reconstruction["X_clean"],
    }

    if X.shape[0] > 64:
        result["warning"] = (
            "The direct channel-wise IVA formulation used here sets N=n_epochs. "
            "For long recordings this may become computationally expensive because "
            f"N={X.shape[0]} epochs were passed to IVA."
        )

    return result


def apply_iva_ga_to_raw(
    raw: mne.io.BaseRaw,
    TR: float,
    picks: list[str] | None = None,
    exclude_last_channel: bool = False,
    offset: int = 0,
    periodic_frequency_hz: float = 14.0,
    iva_g_max_iter: int = 256,
    iva_l_max_iter: int = 512,
    whiten: bool = True,
    verbose: bool = False,
    random_state: int | None = 0,
) -> tuple[mne.io.RawArray, dict[str, Any]]:
    """Apply IVA-GA to an MNE Raw and return a cleaned RawArray."""
    raw_in = raw.copy().load_data()
    if picks is not None:
        picks = list(picks)
        raw_proc = raw_in.copy().pick(picks)
    elif exclude_last_channel:
        raw_proc = raw_in.copy().pick(raw_in.ch_names[:-1])
        picks = list(raw_proc.ch_names)
    else:
        raw_proc = raw_in.copy().pick("eeg")
        picks = list(raw_proc.ch_names)

    data = raw_proc.get_data()
    fs = float(raw_proc.info["sfreq"])
    result = run_iva_ga_pipeline(
        signal=data,
        TR=TR,
        fs=fs,
        offset=offset,
        periodic_frequency_hz=periodic_frequency_hz,
        iva_g_max_iter=iva_g_max_iter,
        iva_l_max_iter=iva_l_max_iter,
        whiten=whiten,
        verbose=verbose,
        random_state=random_state,
    )

    raw_clean_full = raw_in.copy()
    pick_indices = [raw_in.ch_names.index(ch_name) for ch_name in picks]
    raw_clean_full._data[pick_indices, :] = result["cleaned_signal"]
    raw_clean = mne.io.RawArray(raw_clean_full.get_data(), raw_clean_full.info.copy(), verbose="ERROR")
    return raw_clean, result

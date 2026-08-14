"""Evaluate any EEG preprocessing result against a clean-reference recording.

Supported processed inputs are the ``.npz`` files written by FASTR/BCGGAN and
EEGLAB ``.set`` or MNE ``.fif`` recordings.  The reference is normally the
matching ``*_gac_eeg.set`` file in ``Gradient_artifact_corrected``.
"""

from __future__ import annotations

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import resample_poly, welch


_AUXILIARY_TOKENS = ("ECG", "EKG", "VREF", "TRIG", "STI", "MISC", "RESP", "EOG", "EMG", "AUX")
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _load_signal(path: Path, signal_key: str) -> tuple[np.ndarray, float, list[str]]:
    """Load channels-by-samples data and metadata from a supported file."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if signal_key not in archive:
                available = ", ".join(archive.files)
                raise KeyError(f"{path} has no '{signal_key}' array (available: {available}).")
            if "fs" not in archive or "channel_names" not in archive:
                raise KeyError(f"{path} must contain 'fs' and 'channel_names'.")
            data = np.asarray(archive[signal_key], dtype=np.float64)
            fs = float(np.asarray(archive["fs"]).item())
            names = [str(name) for name in archive["channel_names"].tolist()]
    elif suffix == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        data, fs, names = raw.get_data(), float(raw.info["sfreq"]), list(raw.ch_names)
    elif suffix == ".fif":
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        data, fs, names = raw.get_data(), float(raw.info["sfreq"]), list(raw.ch_names)
    else:
        raise ValueError("Supported inputs are .npz, .set and .fif files.")
    if data.ndim != 2 or data.shape[0] != len(names):
        raise ValueError(f"Invalid channels-by-samples data in {path}.")
    return data, fs, names


def _select_common_eeg_channels(
    processed_names: list[str], reference_names: list[str], include_auxiliary: bool,
) -> list[tuple[str, int, int]]:
    reference_index = {name: index for index, name in enumerate(reference_names)}
    selected = []
    for processed_index, name in enumerate(processed_names):
        if name not in reference_index:
            continue
        if not include_auxiliary and any(token in name.upper() for token in _AUXILIARY_TOKENS):
            continue
        selected.append((name, processed_index, reference_index[name]))
    if not selected:
        raise ValueError("No common EEG channels remain. Check channel names or use --include-auxiliary.")
    return selected


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _band_power(frequencies: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    return float(np.trapezoid(psd[mask], frequencies[mask])) if np.count_nonzero(mask) >= 2 else float("nan")


def evaluate_signals(
    processed: np.ndarray, reference: np.ndarray, fs: float, channel_names: list[str],
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    """Compute temporal and spectral reconstruction metrics channel by channel."""
    rows: list[dict[str, float | str]] = []
    psd_data: dict[str, np.ndarray] = {}
    nperseg = min(4096, processed.shape[1])
    for name, candidate, target in zip(channel_names, processed, reference, strict=True):
        residual = candidate - target
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual**2)))
        target_variance = float(np.sum((target - np.mean(target)) ** 2))
        r2 = float(1.0 - np.sum(residual**2) / target_variance) if target_variance > 0 else float("nan")
        noise_power = float(np.mean(residual**2))
        signal_power = float(np.mean(target**2))
        reconstruction_snr_db = float(10 * np.log10(signal_power / noise_power)) if noise_power > 0 else float("inf")
        frequencies, candidate_psd = welch(candidate, fs=fs, nperseg=nperseg)
        _, reference_psd = welch(target, fs=fs, nperseg=nperseg)
        spectral_relative_error = float(np.mean(np.abs(candidate_psd - reference_psd)) / (np.mean(reference_psd) + np.finfo(float).eps))
        row: dict[str, float | str] = {
            "channel": name,
            "mae": mae,
            "rmse": rmse,
            "pearson_r": _safe_correlation(candidate, target),
            "r2": r2,
            "reconstruction_snr_db": reconstruction_snr_db,
            "spectral_relative_error": spectral_relative_error,
            "psd_correlation": _safe_correlation(candidate_psd, reference_psd),
        }
        for band_name, (low, high) in _BANDS.items():
            reference_power = _band_power(frequencies, reference_psd, low, high)
            candidate_power = _band_power(frequencies, candidate_psd, low, high)
            row[f"{band_name}_relative_power_error"] = float(
                abs(candidate_power - reference_power) / (abs(reference_power) + np.finfo(float).eps)
            )
        rows.append(row)
        psd_data[name] = np.vstack((frequencies, candidate_psd, reference_psd))
    return rows, psd_data


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, float | str]]) -> dict[str, float | int]:
    numeric_keys = [key for key in rows[0] if key != "channel"]
    return {
        "n_channels": len(rows),
        **{f"mean_{key}": float(np.nanmean([float(row[key]) for row in rows])) for key in numeric_keys},
        **{f"median_{key}": float(np.nanmedian([float(row[key]) for row in rows])) for key in numeric_keys},
    }


def _plot_summary(rows: list[dict[str, float | str]], path: Path, method_name: str) -> None:
    channels = [str(row["channel"]) for row in rows]
    correlation = [float(row["pearson_r"]) for row in rows]
    rmse = [float(row["rmse"]) for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(max(10, len(channels) * 0.35), 7), layout="constrained")
    axes[0].bar(channels, correlation, color="tab:blue")
    axes[0].set(title=f"{method_name}: temporal correlation", ylabel="Pearson r", ylim=(-1, 1))
    axes[1].bar(channels, rmse, color="tab:orange")
    axes[1].set(title=f"{method_name}: reconstruction error", ylabel="RMSE (V)", xlabel="Channel")
    for axis in axes:
        axis.tick_params(axis="x", rotation=90)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_psd(psd: np.ndarray, path: Path, channel: str, method_name: str) -> None:
    frequencies, candidate_psd, reference_psd = psd
    mask = (frequencies >= 0.5) & (frequencies <= min(70.0, frequencies[-1]))
    figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    axis.semilogy(frequencies[mask], reference_psd[mask], label="Reference clean", color="black")
    axis.semilogy(frequencies[mask], candidate_psd[mask], label=method_name, color="tab:blue")
    axis.set(xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)", title=f"PSD comparison: {channel}")
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare preprocessed EEG with a clean-reference EEG recording.")
    parser.add_argument("--processed", type=Path, required=True, help=".npz, .set or .fif produced by a preprocessing method.")
    parser.add_argument("--reference", type=Path, required=True, help="Clean-reference .set, .fif or .npz recording.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--method-name", help="Label used in generated reports; defaults to processed filename.")
    parser.add_argument("--signal-key", default="cleaned_signal", help="Array key used only for .npz processed files.")
    parser.add_argument("--reference-signal-key", default="cleaned_signal", help="Array key used only for .npz reference files.")
    parser.add_argument("--include-auxiliary", action="store_true", help="Also evaluate ECG and auxiliary channels.")
    parser.add_argument("--resample-reference", action="store_true", help="Resample the reference to the processed sampling rate when they differ.")
    parser.add_argument("--reference-start-seconds", type=float, default=0.0, help="Crop the reference from this original-recording time before evaluation.")
    parser.add_argument("--max-duration-seconds", type=float, help="Evaluate at most this duration from the processed and cropped-reference signals.")
    parser.add_argument("--psd-channel", help="Channel for the PSD comparison figure; defaults to the first evaluated channel.")
    args = parser.parse_args()

    processed, processed_fs, processed_names = _load_signal(args.processed, args.signal_key)
    reference, reference_fs, reference_names = _load_signal(args.reference, args.reference_signal_key)
    reference_resampled = False
    if not np.isclose(processed_fs, reference_fs):
        if not args.resample_reference:
            raise ValueError(f"Sampling rates differ ({processed_fs} vs {reference_fs} Hz); use --resample-reference or resample explicitly.")
        ratio = Fraction(processed_fs / reference_fs).limit_denominator(10_000)
        reference = resample_poly(reference, ratio.numerator, ratio.denominator, axis=-1)
        reference_fs = processed_fs
        reference_resampled = True
    if args.reference_start_seconds < 0:
        raise ValueError("--reference-start-seconds must be non-negative.")
    reference_start = int(round(args.reference_start_seconds * reference_fs))
    if reference_start >= reference.shape[1]:
        raise ValueError("The requested reference start lies outside the reference recording.")
    reference = reference[:, reference_start:]
    if args.max_duration_seconds is not None:
        if args.max_duration_seconds <= 0:
            raise ValueError("--max-duration-seconds must be strictly positive.")
        processed_limit = int(round(args.max_duration_seconds * processed_fs))
        reference_limit = int(round(args.max_duration_seconds * reference_fs))
        if processed.shape[1] < processed_limit:
            raise ValueError("The processed signal is shorter than --max-duration-seconds.")
        if reference.shape[1] < reference_limit:
            raise ValueError("The cropped reference is shorter than --max-duration-seconds.")
        processed = processed[:, :processed_limit]
        reference = reference[:, :reference_limit]
    common = _select_common_eeg_channels(processed_names, reference_names, args.include_auxiliary)
    n_samples = min(processed.shape[1], reference.shape[1])
    candidate = np.vstack([processed[processed_index, :n_samples] for _, processed_index, _ in common])
    target = np.vstack([reference[reference_index, :n_samples] for _, _, reference_index in common])
    channel_names = [name for name, _, _ in common]
    rows, psd_data = evaluate_signals(candidate, target, processed_fs, channel_names)

    method_name = args.method_name or args.processed.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / method_name
    csv_path = prefix.with_name(f"{prefix.name}_per_channel_metrics.csv")
    json_path = prefix.with_name(f"{prefix.name}_summary_metrics.json")
    summary_plot = prefix.with_name(f"{prefix.name}_summary.png")
    _write_csv(rows, csv_path)
    summary = _summary(rows) | {
        "method": method_name,
        "sampling_frequency_hz": processed_fs,
        "reference_resampled": reference_resampled,
        "reference_start_seconds": args.reference_start_seconds,
        "samples_evaluated": n_samples,
        "seconds_evaluated": n_samples / processed_fs,
    }
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    _plot_summary(rows, summary_plot, method_name)
    psd_channel = args.psd_channel or channel_names[0]
    if psd_channel not in psd_data:
        raise ValueError(f"PSD channel '{psd_channel}' was not evaluated.")
    psd_plot = prefix.with_name(f"{prefix.name}_psd_{psd_channel}.png")
    _plot_psd(psd_data[psd_channel], psd_plot, psd_channel, method_name)
    print(f"Evaluated {len(rows)} channels and {n_samples} samples ({n_samples / processed_fs:.2f} s).")
    print(f"Mean Pearson r: {summary['mean_pearson_r']:.4f} | mean RMSE: {summary['mean_rmse']:.6g} V")
    print(f"Per-channel metrics: {csv_path}\nSummary: {json_path}\nFigures: {summary_plot}, {psd_plot}")


if __name__ == "__main__":
    main()

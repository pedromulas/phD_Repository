"""IVA for GA, cardiac OBS for BCG, and Infomax ICA for ocular/muscle artefacts."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
from scipy.signal import resample_poly

try:
    from functions.appear_utils import apply_event_locked_obs, load_subject_fmri_acquisition
    from functions.gradient_sync import detect_gradient_artifact_start
    from functions.ica_conventional import apply_conventional_ica
    from functions.iva_ga import apply_iva_ga_to_raw
    from functions.iva_torch_gpu import cuda_available, run_blockwise_torch_iva_ga
    from functions.preprocessing_paths import preprocessing_output_path
    from functions.preprocessing_window import crop_mne_raw
except ModuleNotFoundError:  # pragma: no cover - direct execution convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from functions.appear_utils import apply_event_locked_obs, load_subject_fmri_acquisition
    from functions.gradient_sync import detect_gradient_artifact_start
    from functions.ica_conventional import apply_conventional_ica
    from functions.iva_ga import apply_iva_ga_to_raw
    from functions.iva_torch_gpu import cuda_available, run_blockwise_torch_iva_ga
    from functions.preprocessing_paths import preprocessing_output_path
    from functions.preprocessing_window import crop_mne_raw


DEFAULT_EEG_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_EEG")
DEFAULT_FMRI_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_MRI")
_AUXILIARY_TOKENS = ("ECG", "EKG", "VREF", "TRIG", "STI", "MISC", "RESP", "EOG", "EMG", "AUX")


def _eeg_names(raw: mne.io.BaseRaw) -> list[str]:
    names = [name for name in raw.ch_names if not any(token in name.upper() for token in _AUXILIARY_TOKENS)]
    if len(names) < 2:
        raise ValueError("Fewer than two EEG channels remain after excluding auxiliary channels.")
    return names


def _replace(raw: mne.io.BaseRaw, names: list[str], values: np.ndarray) -> mne.io.BaseRaw:
    output = raw.copy().load_data()
    output._data[[output.ch_names.index(name) for name in names]] = values
    return output


def _resample_to(data: np.ndarray, source_fs: float, target_fs: float, target_samples: int) -> np.ndarray:
    """Resample channels x samples and preserve the requested output length."""
    ratio = Fraction(target_fs / source_fs).limit_denominator(10_000)
    resampled = resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)
    if resampled.shape[1] >= target_samples:
        return resampled[:, :target_samples]
    padded = np.zeros((resampled.shape[0], target_samples), dtype=resampled.dtype)
    padded[:, : resampled.shape[1]] = resampled
    return padded


def _json_default(value: object) -> object:
    """Convert NumPy scalar/array metadata into standard JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean EEG with IVA-GA, OBS-BCG and ICA ocular/muscle rejection.")
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--task", default="fmrirestingec")
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT)
    parser.add_argument("--preprocessing-root", type=Path, default=Path("data/preprocessing"))
    parser.add_argument("--dataset-name", default="Dataset1")
    parser.add_argument("--output", type=Path, help="Optional explicit output path.")
    parser.add_argument("--use-fmri-checkpoint", action="store_true", help="Apply IVA only inside automatically detected T0/Tf; disabled by default.")
    parser.add_argument("--calibration-seconds", type=float, default=5.0)
    parser.add_argument("--crop-start-seconds", type=float)
    parser.add_argument("--crop-duration-seconds", type=float)
    parser.add_argument("--ecg-channel")
    parser.add_argument("--skip-bcg", action="store_true")
    parser.add_argument("--obs-components", type=int, default=4)
    parser.add_argument("--iva-sfreq", type=float, default=250.0, help="Sampling frequency used only while fitting IVA; use 0 to preserve the original rate.")
    parser.add_argument("--iva-epochs-per-block", type=int, default=40, help="Complete TR epochs per IVA block; use 0 for one global IVA fit.")
    parser.add_argument("--iva-overlap-epochs", type=int, default=4, help="TR-epoch overlap between IVA blocks.")
    parser.add_argument("--iva-g-max-iter", type=int, default=100)
    parser.add_argument("--iva-l-max-iter", type=int, default=200)
    parser.add_argument("--iva-backend", choices=("auto", "cpu", "gpu"), default="auto", help="Use CUDA IVA when available; 'cpu' uses the reference library.")
    parser.add_argument("--iva-gpu-learning-rate", type=float, default=1e-2)
    parser.add_argument("--ica-exclude", type=int, nargs="*", default=())
    parser.add_argument("--no-auto-ocular-ica", action="store_false", dest="auto_ocular_ica", default=True,
                        help="Keep ocular ICA candidates instead of removing them automatically.")
    parser.add_argument("--no-auto-muscle-ica", action="store_false", dest="auto_muscle_ica", default=True,
                        help="Keep muscle ICA candidates instead of removing them automatically.")
    args = parser.parse_args()

    eeg_path = args.eeg_root / args.subject / "eeg" / f"{args.subject}_task-{args.task}_eeg.set"
    if not eeg_path.is_file():
        raise FileNotFoundError(f"EEGLAB header not found: {eeg_path}")
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose="ERROR")
    raw, crop_info = crop_mne_raw(raw, args.crop_start_seconds, args.crop_duration_seconds)
    names, fs = _eeg_names(raw), float(raw.info["sfreq"])
    acquisition = load_subject_fmri_acquisition(args.fmri_root, args.subject)
    tr_sec, n_slices = float(acquisition["tr_sec"]), int(acquisition["n_slices"])
    start, stop, detection = 0, raw.n_times, None
    if args.use_fmri_checkpoint:
        detection = detect_gradient_artifact_start(
            raw.get_data(picks=names), fs, tr_sec, n_slices,
            calibration_seconds=args.calibration_seconds,
            channel_names=names,
        )
        start, stop = detection.t_ga_sample, detection.t_ga_end_sample

    # IVA is fitted to complete TR epochs. With a checkpoint, only the fMRI
    # interval is processed and then reinserted into the original recording.
    iva_input = raw.copy().crop(start / fs, (stop - 1) / fs, include_tmax=True) if args.use_fmri_checkpoint else raw
    iva_fs = fs if args.iva_sfreq <= 0 or args.iva_sfreq >= fs else float(args.iva_sfreq)
    iva_work = iva_input.copy()
    if not np.isclose(iva_fs, fs):
        iva_work.resample(iva_fs, npad="auto", verbose="ERROR")
    block_size = None if args.iva_epochs_per_block <= 0 else args.iva_epochs_per_block
    use_gpu = args.iva_backend == "gpu" or (args.iva_backend == "auto" and cuda_available())
    if args.iva_backend == "gpu" and not cuda_available():
        raise RuntimeError("--iva-backend gpu was requested, but CUDA is not available.")
    print(
        f"IVA configuration: {iva_work.n_times / iva_fs:.1f} s at {iva_fs:g} Hz | "
        f"backend={'gpu' if use_gpu else 'cpu'} | blocks={block_size or 'global'} TR | "
        f"iterations G/L={args.iva_g_max_iter}/{args.iva_l_max_iter}",
        flush=True,
    )
    if use_gpu:
        if block_size is None:
            raise ValueError("GPU IVA requires --iva-epochs-per-block greater than zero.")
        iva_result = run_blockwise_torch_iva_ga(
            iva_work.get_data(picks=names), TR=tr_sec, fs=iva_fs,
            epochs_per_block=block_size, overlap_epochs=args.iva_overlap_epochs,
            periodic_frequency_hz=n_slices / tr_sec, max_iter=args.iva_l_max_iter,
            learning_rate=args.iva_gpu_learning_rate, random_state=0,
        )
        iva_values = np.asarray(iva_result["cleaned_signal"])
    else:
        iva_clean, iva_result = apply_iva_ga_to_raw(
            iva_work, TR=tr_sec, picks=names, periodic_frequency_hz=n_slices / tr_sec,
            iva_g_max_iter=args.iva_g_max_iter, iva_l_max_iter=args.iva_l_max_iter,
            epochs_per_block=block_size, overlap_epochs=args.iva_overlap_epochs, random_state=0,
        )
        iva_values = iva_clean.get_data(picks=names)
    if not np.isclose(iva_fs, fs):
        iva_values = _resample_to(iva_values, iva_fs, fs, iva_input.n_times)
    raw_iva = raw.copy().load_data()
    if args.use_fmri_checkpoint:
        raw_iva._data[[raw_iva.ch_names.index(name) for name in names], start:stop] = iva_values[:, : stop - start]
    else:
        raw_iva._data[[raw_iva.ch_names.index(name) for name in names]] = iva_values[:, : raw_iva.n_times]

    ecg_name = args.ecg_channel or next((name for name in raw_iva.ch_names if "ECG" in name.upper() or "EKG" in name.upper()), None)
    obs_info: dict[str, object] = {"applied": False, "reason": "disabled" if args.skip_bcg else "no ECG channel"}
    raw_obs = raw_iva.copy().load_data()
    if not args.skip_bcg and ecg_name is not None:
        if ecg_name not in raw_obs.ch_names:
            raise ValueError(f"ECG channel not found: {ecg_name}")
        try:
            events, _, _ = mne.preprocessing.find_ecg_events(raw_obs, ch_name=ecg_name, verbose="ERROR")
            corrected, details = apply_event_locked_obs(raw_obs.get_data(picks=names), events[:, 0] - raw_obs.first_samp, fs, n_components=args.obs_components)
            raw_obs = _replace(raw_obs, names, corrected)
            obs_info = {"applied": True, "ecg_channel": ecg_name, "n_events": int(len(details["event_samples"])), "n_components": args.obs_components}
        except (RuntimeError, ValueError) as error:
            warnings.warn(f"BCG OBS was skipped: {error}", RuntimeWarning, stacklevel=2)
            obs_info = {"applied": False, "ecg_channel": ecg_name, "reason": str(error)}

    raw_clean, ica_result = apply_conventional_ica(
        raw_obs, picks=names, n_components=None, manual_exclude=args.ica_exclude,
        method="infomax", auto_ocular=args.auto_ocular_ica, auto_muscle=args.auto_muscle_ica,
        fit_l_freq=1.0, random_state=97,
    )
    output = args.output or preprocessing_output_path(args.preprocessing_root, args.dataset_name, "IVA_OBS_ICA", args.subject, args.task)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, cleaned_signal=raw_clean.get_data(), fs=float(raw_clean.info["sfreq"]), channel_names=np.asarray(raw_clean.ch_names))
    ica_path = output.with_name(f"{output.stem}_ica.fif")
    ica_result["ica"].save(ica_path, overwrite=True)
    np.savez_compressed(
        output.with_name(f"{output.stem}_ica_analysis.npz"),
        channel_names=np.asarray(ica_result["picks"]),
        topographies=np.asarray(ica_result["ica"].get_components()),
        excluded_components=np.asarray(ica_result["excluded_components"], dtype=int),
        ocular_candidates=np.asarray(ica_result["ocular_candidates"], dtype=int),
        muscle_candidates=np.asarray(ica_result["muscle_candidates"], dtype=int),
    )
    metadata = {
        "tr_sec": tr_sec, "n_slices": n_slices, "use_fmri_checkpoint": args.use_fmri_checkpoint,
        "input_crop": crop_info,
        "fmri_start_sample": start, "fmri_end_sample": stop,
        "iva_ga_component": int(iva_result["ga_component"]) if "ga_component" in iva_result else None,
        "iva_library": iva_result["library_used"], "iva_sampling_frequency_hz": iva_fs,
        "iva_backend": iva_result.get("backend", "cpu"), "iva_device": iva_result.get("device", "cpu"),
        "iva_epochs_per_block": block_size, "iva_overlap_epochs": args.iva_overlap_epochs,
        "iva_g_max_iter": args.iva_g_max_iter, "iva_l_max_iter": args.iva_l_max_iter,
        "iva_gpu_learning_rate": args.iva_gpu_learning_rate,
        "iva_blocks": iva_result.get("blocks"),
        "obs_bcg": obs_info, "ica_excluded_components": ica_result["excluded_components"],
        "ica_ocular_candidates": ica_result["ocular_candidates"], "ica_muscle_candidates": ica_result["muscle_candidates"],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    print(f"Saved IVA_OBS_ICA-cleaned EEG to {output} ({raw_clean.get_data().shape}).")
    print(f"Saved ICA decomposition to {ica_path}.")


if __name__ == "__main__":
    main()

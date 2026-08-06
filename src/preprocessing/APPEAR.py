"""APPEAR-inspired automatic EEG-fMRI cleaning pipeline.

The pipeline follows Mayeli et al. (2021): FASTR/OBS-like gradient correction,
filtering, cardiac-event AAS, bad-interval marking, and Infomax ICA.  Dataset1
does not provide scanner slice triggers, so the optional fMRI checkpoint uses
the morphology-based T0/Tf detector already implemented in ``functions``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import mne
import numpy as np

try:  # Supports both ``python -m`` and direct script execution.
    from functions.appear_utils import (
        add_bad_interval_annotations,
        apply_event_locked_aas,
        detect_bad_intervals_for_ica,
        load_subject_fmri_acquisition,
    )
    from functions.fastr_ga import FASTRConfig, fastr_remove_gradient_artifact
    from functions.ica_conventional import apply_conventional_ica
    from functions.preprocessing_paths import preprocessing_output_path
except ModuleNotFoundError:  # pragma: no cover - direct execution convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from functions.appear_utils import add_bad_interval_annotations, apply_event_locked_aas, detect_bad_intervals_for_ica, load_subject_fmri_acquisition
    from functions.fastr_ga import FASTRConfig, fastr_remove_gradient_artifact
    from functions.ica_conventional import apply_conventional_ica
    from functions.preprocessing_paths import preprocessing_output_path


DEFAULT_EEG_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_EEG")
DEFAULT_FMRI_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_MRI")
_AUXILIARY_TOKENS = ("ECG", "EKG", "VREF", "TRIG", "STI", "MISC", "RESP", "EOG", "EMG", "AUX")


def _eeg_channel_names(raw: mne.io.BaseRaw) -> list[str]:
    names = [name for name in raw.ch_names if not any(token in name.upper() for token in _AUXILIARY_TOKENS)]
    if len(names) < 2:
        raise ValueError("Fewer than two EEG channels remain after excluding auxiliary channels.")
    return names


def _ecg_channel_name(raw: mne.io.BaseRaw, requested: str | None) -> str | None:
    if requested is not None:
        if requested not in raw.ch_names:
            raise ValueError(f"ECG channel not found: {requested}")
        return requested
    return next((name for name in raw.ch_names if "ECG" in name.upper() or "EKG" in name.upper()), None)


def _set_selected_data(raw: mne.io.BaseRaw, names: list[str], data: np.ndarray) -> mne.io.BaseRaw:
    updated = raw.copy().load_data()
    indices = [updated.ch_names.index(name) for name in names]
    updated._data[indices] = data
    return updated


def _notch_frequencies(slice_frequency_hz: float, high_hz: float) -> list[float]:
    frequencies = [26.0, 60.0]
    harmonic = slice_frequency_hz
    while harmonic < high_hz:
        frequencies.append(harmonic)
        harmonic += slice_frequency_hz
    return sorted({round(frequency, 8) for frequency in frequencies if frequency < high_hz})


def run_appear(
    eeg_path: Path,
    fmri_root: Path,
    subject: str,
    use_fmri_checkpoint: bool = False,
    calibration_seconds: float = 5.0,
    low_hz: float = 1.0,
    high_hz: float = 70.0,
    downsample_hz: float = 250.0,
    ecg_channel: str | None = None,
    skip_bcg: bool = False,
    ica_exclude: tuple[int, ...] = (),
) -> tuple[mne.io.BaseRaw, dict[str, object]]:
    """Run the core APPEAR stages and return a cleaned MNE recording."""
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose="ERROR")
    acquisition = load_subject_fmri_acquisition(fmri_root, subject)
    tr_sec, n_slices = float(acquisition["tr_sec"]), int(acquisition["n_slices"])
    slice_period, slice_frequency = tr_sec / n_slices, n_slices / tr_sec
    eeg_names = _eeg_channel_names(raw)
    eeg_data = raw.get_data(picks=eeg_names)
    fs = float(raw.info["sfreq"])

    # APPEAR's fmrib_fastr stage is represented by the existing adaptive
    # FASTR/OBS-like function. With no scanner triggers, T0/Tf is optional.
    ga_result = fastr_remove_gradient_artifact(
        eeg_data,
        fs,
        FASTRConfig(
            artifact_period_s=slice_period,
            use_fmri_bounds=use_fmri_checkpoint,
            tr_sec=tr_sec,
            n_slices=n_slices,
            calibration_seconds=calibration_seconds,
        ),
        detection_signal=eeg_data,
        channel_names=eeg_names,
    )
    raw_ga = _set_selected_data(raw, eeg_names, np.asarray(ga_result["cleaned_signal"]))

    raw_filtered = raw_ga.copy().load_data()
    raw_filtered.filter(low_hz, high_hz, picks=eeg_names, method="fir", verbose="ERROR")
    notches = [frequency for frequency in _notch_frequencies(slice_frequency, high_hz) if frequency < raw_filtered.info["sfreq"] / 2]
    if notches:
        raw_filtered.notch_filter(notches, picks=eeg_names, method="fir", verbose="ERROR")
    if downsample_hz > 0 and downsample_hz < raw_filtered.info["sfreq"]:
        raw_filtered.resample(downsample_hz, npad="auto", verbose="ERROR")

    eeg_names = _eeg_channel_names(raw_filtered)
    raw_bcg = raw_filtered.copy().load_data()
    ecg_name = _ecg_channel_name(raw_bcg, ecg_channel)
    bcg_result: dict[str, object] = {"applied": False, "reason": "disabled" if skip_bcg else "no ECG channel"}
    if not skip_bcg and ecg_name is not None:
        try:
            events, _, _ = mne.preprocessing.find_ecg_events(raw_bcg, ch_name=ecg_name, verbose="ERROR")
            event_samples = events[:, 0] - raw_bcg.first_samp
            corrected, bcg_details = apply_event_locked_aas(raw_bcg.get_data(picks=eeg_names), event_samples, float(raw_bcg.info["sfreq"]))
            raw_bcg = _set_selected_data(raw_bcg, eeg_names, corrected)
            bcg_result = {"applied": True, "ecg_channel": ecg_name, "n_events": int(len(bcg_details["event_samples"])), "template_window": 21}
        except (RuntimeError, ValueError) as error:
            warnings.warn(f"BCG AAS was skipped: {error}", RuntimeWarning, stacklevel=2)
            bcg_result = {"applied": False, "reason": str(error), "ecg_channel": ecg_name}

    bad_intervals = detect_bad_intervals_for_ica(raw_bcg.get_data(picks=eeg_names), float(raw_bcg.info["sfreq"]))
    raw_for_ica = add_bad_interval_annotations(raw_bcg, bad_intervals)
    raw_clean, ica_result = apply_conventional_ica(
        raw_for_ica,
        picks=eeg_names,
        n_components=None,  # APPEAR estimates one IC per EEG channel.
        manual_exclude=ica_exclude,
        method="infomax",
        fit_l_freq=1.0,
        random_state=97,
    )
    ga_summary = {
        key: ga_result[key]
        for key in (
            "offset", "period_samples", "use_fmri_bounds", "fmri_start_sample", "fmri_end_sample",
            "fmri_start_sec", "fmri_end_sec", "gradient_detection_used", "t_bold_sample",
            "t_bold_end_sample", "t_bold_sec", "t_bold_end_sec",
        )
    }
    details: dict[str, object] = {
        "tr_sec": tr_sec,
        "n_slices": n_slices,
        "slice_period_sec": slice_period,
        "slice_frequency_hz": slice_frequency,
        "fmri_metadata": acquisition,
        "use_fmri_checkpoint": use_fmri_checkpoint,
        "ga": ga_summary,
        "notch_frequencies_hz": notches,
        "downsample_hz": float(raw_clean.info["sfreq"]),
        "bcg": bcg_result,
        "bad_intervals_sec": bad_intervals,
        "ica_method": ica_result["method"],
        "ica_excluded_components": ica_result["excluded_components"],
        "ica_components": int(ica_result["ica"].n_components_),
    }
    return raw_clean, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Run APPEAR-style EEG-fMRI preprocessing on one Dataset1 recording.")
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--task", default="fmrirestingec")
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT)
    parser.add_argument("--preprocessing-root", type=Path, default=Path("data/preprocessing"))
    parser.add_argument("--dataset-name", default="Dataset1")
    parser.add_argument("--output", type=Path, help="Optional explicit output path; otherwise the standard APPEAR path is used.")
    parser.add_argument("--use-fmri-checkpoint", action="store_true", help="Detect T0/Tf from GA morphology before FASTR; disabled by default.")
    parser.add_argument("--calibration-seconds", type=float, default=5.0)
    parser.add_argument("--low-hz", type=float, default=1.0, help="Use 0.1 Hz for task/ERP data, as in APPEAR.")
    parser.add_argument("--high-hz", type=float, default=70.0)
    parser.add_argument("--downsample-hz", type=float, default=250.0)
    parser.add_argument("--ecg-channel", help="ECG channel used for cardiac-event BCG AAS; autodetected by default.")
    parser.add_argument("--skip-bcg", action="store_true")
    parser.add_argument("--ica-exclude", type=int, nargs="*", default=(), help="Infomax component indices to exclude after inspection.")
    args = parser.parse_args()
    eeg_path = args.eeg_root / args.subject / "eeg" / f"{args.subject}_task-{args.task}_eeg.set"
    if not eeg_path.is_file():
        raise FileNotFoundError(f"EEGLAB header not found: {eeg_path}")
    clean, details = run_appear(
        eeg_path, args.fmri_root, args.subject, args.use_fmri_checkpoint,
        args.calibration_seconds, args.low_hz, args.high_hz, args.downsample_hz,
        args.ecg_channel, args.skip_bcg, tuple(args.ica_exclude),
    )
    output_path = args.output or preprocessing_output_path(
        args.preprocessing_root, args.dataset_name, "APPEAR", args.subject, args.task,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, cleaned_signal=clean.get_data(), fs=float(clean.info["sfreq"]), channel_names=np.asarray(clean.ch_names))
    metadata_path = output_path.with_suffix(".json")
    # Large intermediate NumPy arrays from FASTR and ICA are intentionally omitted.
    metadata_path.write_text(json.dumps(details, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)), encoding="utf-8")
    print(f"Saved APPEAR-cleaned EEG to {output_path} ({clean.get_data().shape}).")
    print(f"TR={details['tr_sec']} s | slices={details['n_slices']} | ICA components removed={details['ica_excluded_components']}")


if __name__ == "__main__":
    main()

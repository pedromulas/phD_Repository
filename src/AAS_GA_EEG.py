from pathlib import Path

import mne
import numpy as np

from functions.aas_ga import run_aas_pipeline


DEFAULT_EEG_ROOT = Path(
    r"C:\Users\pedro\Documents\DOCTORADO_Pedro\Code\phD_Repository\data\raw\Dataset1\Simultaneous_EEG_fMRI\BIDS_dataset_EEG"
)
DEFAULT_SUBJECT = "sub-001"
DEFAULT_TASK = "fmrirestingec"
DEFAULT_TR = 2.0


def get_eeg_set_path(
    subject: str,
    task: str = DEFAULT_TASK,
    eeg_root: Path = DEFAULT_EEG_ROOT,
) -> Path:
    """Return the EEGLAB .set path for one subject and task."""
    eeg_path = eeg_root / subject / "eeg" / f"{subject}_task-{task}_eeg.set"
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")
    return eeg_path


def load_eeg_array(eeg_path: Path) -> tuple[np.ndarray, float, list[str]]:
    """Load EEG data from EEGLAB and return channels x samples."""
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose="ERROR")
    data = raw.get_data()
    fs = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names)
    return data, fs, channel_names


def main() -> None:
    eeg_path = get_eeg_set_path(DEFAULT_SUBJECT)
    eeg_data, fs, channel_names = load_eeg_array(eeg_path)

    result = run_aas_pipeline(
        signal=eeg_data,
        TR=DEFAULT_TR,
        fs=fs,
        reference_channel=0,
        n_iter=4,
        window_size=21,
        max_lag=10,
    )

    cleaned_eeg = result["cleaned_signal"]

    print(f"Subject: {DEFAULT_SUBJECT}")
    print(f"Task: {DEFAULT_TASK}")
    print(f"EEG path: {eeg_path}")
    print(f"Input shape: {eeg_data.shape}")
    print(f"Cleaned shape: {cleaned_eeg.shape}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"TR: {DEFAULT_TR} s")
    print(f"Samples per TR: {result['T_samples']}")
    print(f"Estimated offset: {result['offset']} samples")
    print(f"Reference channel: 0 ({channel_names[0]})")
    print(f"First 10 lags: {result['lags'][:10]}")


if __name__ == "__main__":
    main()

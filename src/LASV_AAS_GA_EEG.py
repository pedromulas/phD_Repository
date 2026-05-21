from pathlib import Path

import mne

from functions.lasv_ga import run_lasv_aas_on_raw


DEFAULT_EEG_ROOT = Path(
    r"C:\Users\pedro\Documents\DOCTORADO_Pedro\Code\phD_Repository\data\raw\Dataset1\Simultaneous_EEG_fMRI\BIDS_dataset_EEG"
)
DEFAULT_SUBJECT = "sub-001"
DEFAULT_TASK = "fmrirestingec"
DEFAULT_TR = 2.0
DEFAULT_FS = 1000.0


def get_eeg_set_path(
    subject: str,
    task: str = DEFAULT_TASK,
    eeg_root: Path = DEFAULT_EEG_ROOT,
) -> Path:
    eeg_path = eeg_root / subject / "eeg" / f"{subject}_task-{task}_eeg.set"
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")
    return eeg_path


def main() -> None:
    eeg_path = get_eeg_set_path(DEFAULT_SUBJECT)
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose="ERROR")

    if float(raw.info["sfreq"]) != DEFAULT_FS:
        raise ValueError(
            f"The current script expects fs={DEFAULT_FS} Hz, "
            f"but the file contains {raw.info['sfreq']} Hz."
        )

    result = run_lasv_aas_on_raw(
        raw=raw,
        tr=DEFAULT_TR,
        learning_rate=1e-4,
        max_iter=200,
        tol=1e-10,
        h_bounds=(0.995, 1.005),
        initial_h=1.0,
        n_iter_aas=5,
        window_size_aas=21,
        max_lag_aas=10,
    )

    cleaned_raw = result["cleaned_raw"]

    print(f"Subject: {DEFAULT_SUBJECT}")
    print(f"Task: {DEFAULT_TASK}")
    print(f"EEG path: {eeg_path}")
    print(f"Original shape: {raw.get_data().shape}")
    print(f"Resampled shape: {result['resampled_signal'].shape}")
    print(f"Cleaned shape: {cleaned_raw.get_data().shape}")
    print(f"Original fs: {result['fs_original']} Hz")
    print(f"Optimized fs: {result['fs_opt']:.10f} Hz")
    print(f"TR: {result['tr']} s")
    print(f"Reference channel selected automatically: {result['reference_channel']}")
    print(f"Optimized h: {result['h_opt']:.12f}")
    print(f"Optimized offset: {result['offset_opt']} samples")
    print(f"LASV iterations: {result['optimization']['n_iterations']}")
    print(f"Final objective J(h): {result['optimization']['objective']:.6f}")


if __name__ == "__main__":
    main()

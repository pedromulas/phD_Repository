from pathlib import Path
import mne

EEG_FILENAME_SET_SUFFIX = "_task-fmrirestingec_eeg.set"


def find_eeg_set_files(eeg_root: Path) -> list[Path]:
    """Return all subject EEG .set files stored in the BIDS EEG dataset."""
    eeg_root = Path(eeg_root)
    return sorted(eeg_root.glob(f"sub-*/eeg/*{EEG_FILENAME_SET_SUFFIX}"))


def load_eeg_metadata(set_path: Path) -> dict:
    """Open an EEGLAB .set file and collect basic metadata for inspection."""
    
    set_path = Path(set_path)
    raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")
    info = raw.info

    return info

def summarize_eeg_dataset(eeg_root: Path) -> list[dict]:
    """Locate every EEG file in the dataset and return one summary per subject."""
    set_files = find_eeg_set_files(eeg_root)
    return [load_eeg_metadata(set_path) for set_path in set_files]
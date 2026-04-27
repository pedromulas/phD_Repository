import json
import pandas as pd
import mne
import nibabel as nib
from pathlib import Path

from functions.fMRI_Info import fMRI_Info

fmri_path = r"C:\Users\pedro\Documents\DOCTORADO_Pedro\Code\phD_Repository\data\raw\Dataset1\Simultaneous_EEG_fMRI\BIDS_dataset_MRI\sub-001\ses-001\func\sub-001_ses-001_task-rest_bold.nii.gz"
set_path = r"C:\Users\pedro\Documents\DOCTORADO_Pedro\Code\phD_Repository\data\raw\Dataset1\Simultaneous_EEG_fMRI\BIDS_dataset_EEG\sub-001\eeg\sub-001_task-fmrirestingec_eeg.set"

fmri_path = Path(fmri_path)
fmri_info = fMRI_Info(fmri_path)
print(fmri_info)

raw = mne.io.read_raw_eeglab(set_path, preload=True)
info = raw.info
eeg_info = {
        "n_channels": info["nchan"],
        "sfreq": float(info["sfreq"]),
        "duration_s": float(raw.n_times / info["sfreq"]),
        "n_samples": raw.n_times,
        "channel_names": info["ch_names"],
        "channel_types": raw.get_channel_types(),
        "highpass": float(info["highpass"]),
        "lowpass": float(info["lowpass"]),
        "line_freq": info.get("line_freq", None),
    }
print(eeg_info)


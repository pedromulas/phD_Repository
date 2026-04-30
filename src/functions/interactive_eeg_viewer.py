from __future__ import annotations

from pathlib import Path

import mne


DEFAULT_EEG_ROOT = Path(
    r"C:\Users\pedro\Documents\DOCTORADO_Pedro\Code\phD_Repository\data\raw\Dataset1\Simultaneous_EEG_fMRI\BIDS_dataset_EEG"
)


def find_subject_eeg_files(eeg_root: str | Path) -> dict[str, list[Path]]:
    eeg_root = Path(eeg_root)
    eeg_files = sorted(eeg_root.glob("sub-*/eeg/*_eeg.set"))

    subject_map: dict[str, list[Path]] = {}
    for eeg_path in eeg_files:
        subject_map.setdefault(eeg_path.parts[-3], []).append(eeg_path)

    return subject_map


def get_subject_eeg_files(eeg_root: str | Path, subject: str) -> list[Path]:
    subject_map = find_subject_eeg_files(eeg_root)
    if subject not in subject_map:
        raise ValueError(f"No se encontraron registros EEG para {subject}.")
    return subject_map[subject]


def get_subject_eeg_path(
    subject: str,
    eeg_root: str | Path = DEFAULT_EEG_ROOT,
    task: str | None = None,
) -> Path:
    eeg_options = get_subject_eeg_files(eeg_root, subject)

    if task is None:
        if len(eeg_options) != 1:
            names = ", ".join(path.name for path in eeg_options)
            raise ValueError(
                f"{subject} tiene varios registros EEG. Especifica una tarea de: {names}"
            )
        return eeg_options[0]

    for eeg_path in eeg_options:
        if task in eeg_path.name:
            return eeg_path

    available = ", ".join(path.name for path in eeg_options)
    raise ValueError(
        f"No se encontro un EEG que contenga '{task}' para {subject}. "
        f"Disponibles: {available}"
    )


def launch_interactive_eeg_viewer(eeg_path: str | Path) -> None:
    eeg_path = Path(eeg_path)
    raw = mne.io.read_raw_eeglab(eeg_path, preload=False)

    mne.viz.set_browser_backend("qt")
    raw.plot(
        n_channels=20,
        duration=10,
        show_scrollbars=True,
        block=True,
    )


def view_subject_eeg(
    subject: str,
    eeg_root: str | Path = DEFAULT_EEG_ROOT,
    task: str | None = None,
) -> None:
    eeg_path = get_subject_eeg_path(subject, eeg_root=eeg_root, task=task)
    launch_interactive_eeg_viewer(eeg_path)

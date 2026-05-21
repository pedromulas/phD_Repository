from __future__ import annotations

from pathlib import Path
from typing import Iterable

import mne
import numpy as np


def tr_to_samples(TR: float, fs: float) -> int:
    """Convert repetition time from seconds to samples."""
    if TR <= 0:
        raise ValueError("TR must be strictly positive.")
    if fs <= 0:
        raise ValueError("fs must be strictly positive.")

    samples = int(round(TR * fs))
    if samples < 1:
        raise ValueError("TR * fs must correspond to at least one sample.")
    return samples


def tr_to_times(
    n_samples: int,
    fs: float,
    TR: float,
    offset_samples: int = 0,
    anchor: str = "start",
) -> np.ndarray:
    """Build periodic artifact times from TR and a sample offset.

    Parameters
    ----------
    n_samples
        Total number of samples in the recording.
    fs
        Sampling frequency in Hz.
    TR
        Repetition time in seconds.
    offset_samples
        Sample index of the first artifact repetition.
    anchor
        Whether the returned times represent the ``"start"`` or ``"center"``
        of each TR segment. MNE's ``apply_pca_obs`` expects event times, so
        using the center is often a reasonable choice for symmetric windows.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if offset_samples < 0:
        raise ValueError("offset_samples must be non-negative.")
    if anchor not in {"start", "center"}:
        raise ValueError("anchor must be 'start' or 'center'.")

    T_samples = tr_to_samples(TR, fs)
    starts = np.arange(offset_samples, n_samples, T_samples, dtype=int)

    if anchor == "center":
        starts = starts + T_samples // 2

    starts = starts[starts < n_samples]
    return starts / fs


def segment_signal(signal: np.ndarray, T_samples: int, offset: int = 0) -> np.ndarray:
    """Segment a 1D or 2D signal into TR-sized epochs."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal_2d = signal[np.newaxis, :]
        was_1d = True
    elif signal.ndim == 2:
        signal_2d = signal
        was_1d = False
    else:
        raise ValueError("signal must be 1D or 2D.")

    if T_samples <= 0:
        raise ValueError("T_samples must be positive.")
    if offset < 0 or offset >= T_samples:
        raise ValueError("offset must satisfy 0 <= offset < T_samples.")

    usable_samples = signal_2d.shape[1] - offset
    n_segments = usable_samples // T_samples
    if n_segments < 1:
        raise ValueError("The signal is too short to extract one full segment.")

    trimmed = signal_2d[:, offset : offset + n_segments * T_samples]
    segmented = trimmed.reshape(signal_2d.shape[0], n_segments, T_samples)
    if was_1d:
        return segmented[0]
    return segmented


def load_raw_eeglab(eeg_path: str | Path, preload: bool = True) -> mne.io.BaseRaw:
    """Load an EEGLAB file as MNE Raw."""
    return mne.io.read_raw_eeglab(Path(eeg_path), preload=preload, verbose="ERROR")


def prepare_raw_for_obs(
    raw: mne.io.BaseRaw,
    picks: Iterable[str] | None = None,
    exclude_last_channel: bool = False,
) -> tuple[mne.io.BaseRaw, list[str]]:
    """Prepare a Raw object for OBS processing.

    Parameters
    ----------
    raw
        Input raw object.
    picks
        Optional channel names to process. If omitted, all channels are kept.
    exclude_last_channel
        If True, the last channel is removed. This is useful when the last
        channel corresponds to ECG and should not enter GA correction.
    """
    raw_obs = raw.copy()

    if picks is not None:
        picks = list(picks)
        raw_obs.pick(picks)
    elif exclude_last_channel:
        raw_obs.pick(raw_obs.ch_names[:-1])

    return raw_obs, list(raw_obs.ch_names)


def apply_obs_mne(
    raw: mne.io.BaseRaw,
    picks: list[str] | None,
    artifact_times: np.ndarray,
    n_components: int = 4,
    copy: bool = True,
    n_jobs: int | None = 1,
) -> mne.io.BaseRaw:
    """Apply MNE's PCA-OBS implementation to selected channels.

    Parameters
    ----------
    raw
        Input raw object.
    picks
        Channel names to process. If None, all data channels in ``raw`` are
        passed through to ``mne.preprocessing.apply_pca_obs``.
    artifact_times
        Event times in seconds corresponding to the repeated MRI artifact.
        For GA removal these can be TR-based times or true trigger times.
    n_components
        Number of PCA components to use.
    copy
        If True, operate on a copy. If False, modify ``raw`` in place.
    n_jobs
        Number of workers passed to MNE.
    """
    artifact_times = np.asarray(artifact_times, dtype=np.float64)
    if artifact_times.ndim != 1:
        raise ValueError("artifact_times must be a 1D array of times in seconds.")
    if artifact_times.size < 2:
        raise ValueError("At least two artifact times are required for OBS.")
    if n_components < 1:
        raise ValueError("n_components must be at least 1.")

    return mne.preprocessing.apply_pca_obs(
        raw,
        picks=picks,
        qrs_times=artifact_times,
        n_components=n_components,
        n_jobs=n_jobs,
        copy=copy,
    )


def run_obs_pipeline_mne(
    raw: mne.io.BaseRaw,
    TR: float,
    offset_samples: int = 0,
    picks: list[str] | None = None,
    n_components: int = 4,
    copy: bool = True,
    n_jobs: int | None = 1,
    anchor: str = "start",
) -> tuple[mne.io.BaseRaw, np.ndarray]:
    """Run OBS using MNE's implementation and TR-based artifact times.

    Notes
    -----
    This wrapper adapts MNE's cardiac PCA-OBS entry point
    ``mne.preprocessing.apply_pca_obs`` to the GA use case by providing a
    periodic list of artifact times derived from the known TR and a sample
    offset. If true trigger times are available, they should be passed
    directly to :func:`apply_obs_mne` instead of using this helper.
    """
    fs = float(raw.info["sfreq"])
    artifact_times = tr_to_times(
        n_samples=raw.n_times,
        fs=fs,
        TR=TR,
        offset_samples=offset_samples,
        anchor=anchor,
    )
    raw_clean = apply_obs_mne(
        raw,
        picks=picks,
        artifact_times=artifact_times,
        n_components=n_components,
        copy=copy,
        n_jobs=n_jobs,
    )
    return raw_clean, artifact_times
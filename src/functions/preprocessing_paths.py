"""Consistent output locations for named EEG preprocessing strategies."""

from __future__ import annotations

from pathlib import Path


def subject_output_name(subject: str) -> str:
    """Convert BIDS ``sub-001`` to the requested ``subject001`` label."""
    value = str(subject).strip()
    if value.lower().startswith("sub-"):
        value = value[4:]
    elif value.lower().startswith("subject"):
        value = value[7:]
    digits = "".join(character for character in value if character.isdigit())
    return f"subject{digits}" if digits else f"subject{value.lower()}"


def recording_output_name(task: str) -> str:
    """Create a compact recording label, e.g. ``fmrirestingec`` -> ``restingec``."""
    value = str(task).strip().lower().replace("-", "").replace("_", "")
    if value.startswith("fmri"):
        value = value[4:]
    if value.startswith("task"):
        value = value[4:]
    return value or "recording"


def preprocessing_output_path(
    preprocessing_root: Path,
    dataset_name: str,
    algorithm_name: str,
    subject: str,
    task: str,
) -> Path:
    """Return the standard ``.npz`` result path for one preprocessing run."""
    filename = f"{subject_output_name(subject)}_{recording_output_name(task)}.npz"
    return Path(preprocessing_root) / dataset_name / algorithm_name / filename

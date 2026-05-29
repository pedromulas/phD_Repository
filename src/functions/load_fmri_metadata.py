from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib


FMRI_FILENAME_SUFFIX = "_ses-001_task-rest_bold.nii.gz"


def _fmri_json_path(nifti_path: Path) -> Path:
    nifti_path = Path(nifti_path)
    if str(nifti_path).endswith(".nii.gz"):
        return Path(str(nifti_path)[:-7] + ".json")
    return nifti_path.with_suffix(".json")


def find_fmri_files(fmri_root: Path) -> list[Path]:
    fmri_root = Path(fmri_root)
    return sorted(fmri_root.glob(f"sub-*/ses-001/func/*{FMRI_FILENAME_SUFFIX}"))


def load_fmri_metadata(path: Path) -> dict:
    path = Path(path)
    img = nib.load(str(path))
    header = img.header
    zooms = header.get_zooms()
    json_path = _fmri_json_path(path)

    json_metadata: dict[str, object] = {}
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as fid:
            json_metadata = json.load(fid)

    slice_timing = json_metadata.get("SliceTiming")
    if isinstance(slice_timing, list):
        n_slices = len(slice_timing)
    else:
        n_slices = img.shape[2] if len(img.shape) >= 3 else None

    acquisition_matrix = json_metadata.get("AcquisitionMatrix")
    if not isinstance(acquisition_matrix, list):
        acquisition_matrix = None

    recon_matrix_pe = json_metadata.get("ReconMatrixPE")
    acquisition_matrix_pe = json_metadata.get("AcquisitionMatrixPE")
    repetition_time = json_metadata.get("RepetitionTime")
    echo_time = json_metadata.get("EchoTime")

    info = {
        "subject": path.parent.parent.parent.name,
        "path": str(path),
        "shape": img.shape,
        "ndim": len(img.shape),
        "affine": img.affine,
        "voxel_size_mm": zooms[:3],
        "datatype": str(header.get_data_dtype()),
        "json_path": str(json_path),
        "TR_header_s": zooms[3] if len(zooms) > 3 else None,
        "TR_json_s": repetition_time,
        "TR_s": repetition_time if repetition_time is not None else (zooms[3] if len(zooms) > 3 else None),
        "TE_s": echo_time,
        "n_slices": n_slices,
        "slice_timing": slice_timing,
        "recon_matrix_pe": recon_matrix_pe,
        "acquisition_matrix_pe": acquisition_matrix_pe,
        "acquisition_matrix": acquisition_matrix,
        "matrix_xy": img.shape[:2],
    }

    if len(img.shape) == 4:
        info.update({
            "n_volumes": img.shape[3],
            "TR": info["TR_s"],
        })
    else:
        info.update({
            "n_volumes": 1,
            "TR": info["TR_s"],
        })

    return info


def summarize_fmri_dataset(fmri_root: Path) -> list[dict]:
    fmri_files = find_fmri_files(fmri_root)
    return [load_fmri_metadata(fmri_path) for fmri_path in fmri_files]

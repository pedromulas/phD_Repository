from pathlib import Path
import nibabel as nib


FMRI_FILENAME_SUFFIX = "_ses-001_task-rest_bold.nii.gz"


def find_fmri_files(fmri_root: Path) -> list[Path]:
    fmri_root = Path(fmri_root)
    return sorted(fmri_root.glob(f"sub-*/ses-001/func/*{FMRI_FILENAME_SUFFIX}"))


def load_fmri_metadata(path: Path) -> dict:
    path = Path(path)
    img = nib.load(str(path))
    header = img.header
    zooms = header.get_zooms()

    info = {
        "subject": path.parent.parent.parent.name,
        "path": str(path),
        "shape": img.shape,
        "ndim": len(img.shape),
        "affine": img.affine,
        "voxel_size_mm": zooms[:3],
        "datatype": str(header.get_data_dtype()),
    }

    if len(img.shape) == 4:
        info.update({
            "n_volumes": img.shape[3],
            "TR": zooms[3] if len(zooms) > 3 else None,
        })
    else:
        info.update({
            "n_volumes": 1,
            "TR": None,
        })

    return info


def summarize_fmri_dataset(fmri_root: Path) -> list[dict]:
    fmri_files = find_fmri_files(fmri_root)
    return [load_fmri_metadata(fmri_path) for fmri_path in fmri_files]
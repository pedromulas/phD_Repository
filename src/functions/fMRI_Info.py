from pathlib import Path
import nibabel as nib
import numpy as np

def fMRI_Info(path: Path) -> dict:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    header = img.header

    info = {
        "path": str(path),
        "shape": img.shape,
        "ndim": data.ndim,
        "affine": img.affine,
        "voxel_size_mm": header.get_zooms(),
        "datatype": str(header.get_data_dtype()),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "std": float(np.nanstd(data)),
    }

    # Información específica para 4D (fMRI)
    if data.ndim == 4:
        info.update({
            "n_volumes": data.shape[3],
            "TR": header.get_zooms()[3] if len(header.get_zooms()) > 3 else None,
            "mean_volume": float(np.nanmean(data.mean(axis=3))),
            "std_volume": float(np.nanstd(data.mean(axis=3))),
        })
    else:
        info.update({
            "n_volumes": 1,
            "TR": None
        })

    return info
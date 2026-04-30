from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.widgets import Slider
from nilearn import image


DEFAULT_FMRI_ROOT = Path(
    r"C:\Users\pedro\Documents\DOCTORADO_Pedro\Code\phD_Repository\data\raw\Dataset1\Simultaneous_EEG_fMRI\BIDS_dataset_MRI"
)


def find_subject_functionals(fmri_root: str | Path) -> dict[str, list[Path]]:
    fmri_root = Path(fmri_root)
    functional_files = sorted(fmri_root.glob("sub-*/ses-001/func/*_bold.nii.gz"))

    subject_map: dict[str, list[Path]] = {}
    for fmri_path in functional_files:
        subject_map.setdefault(fmri_path.parts[-4], []).append(fmri_path)

    return subject_map


def find_anatomical_for_subject(fmri_root: str | Path, subject: str) -> Path | None:
    fmri_root = Path(fmri_root)
    anat_files = sorted((fmri_root / subject / "ses-001" / "anat").glob("*_T1w.nii.gz"))
    return anat_files[0] if anat_files else None


def get_subject_functionals(fmri_root: str | Path, subject: str) -> list[Path]:
    subject_map = find_subject_functionals(fmri_root)
    if subject not in subject_map:
        raise ValueError(f"No se encontraron adquisiciones fMRI para {subject}.")
    return subject_map[subject]


def get_subject_fmri_path(
    subject: str,
    fmri_root: str | Path = DEFAULT_FMRI_ROOT,
    acquisition: str | None = None,
) -> Path:
    fmri_options = get_subject_functionals(fmri_root, subject)

    if acquisition is None:
        if len(fmri_options) != 1:
            names = ", ".join(path.name for path in fmri_options)
            raise ValueError(
                f"{subject} tiene varias adquisiciones fMRI. Especifica una de: {names}"
            )
        return fmri_options[0]

    for fmri_path in fmri_options:
        if acquisition in fmri_path.name:
            return fmri_path

    available = ", ".join(path.name for path in fmri_options)
    raise ValueError(
        f"No se encontro una adquisicion que contenga '{acquisition}' para {subject}. "
        f"Disponibles: {available}"
    )


def launch_interactive_viewer(
    fmri_path: str | Path, anat_path: str | Path | None = None
) -> None:
    fmri_path = Path(fmri_path)
    img = nib.load(str(fmri_path))
    data = img.get_fdata(dtype=np.float32)

    if data.ndim != 4:
        raise ValueError(f"Se esperaba una imagen 4D funcional y se recibio shape={data.shape}")

    tr = img.header.get_zooms()[3] if len(img.header.get_zooms()) > 3 else 1.0
    times = np.arange(data.shape[3]) * tr

    anat_data = None
    if anat_path is not None:
        anat_img = nib.load(str(Path(anat_path)))
        anat_img = image.resample_to_img(
            anat_img,
            image.index_img(img, 0),
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )
        anat_data = anat_img.get_fdata(dtype=np.float32)
        if anat_data.ndim != 3:
            raise ValueError(
                f"Se esperaba una imagen anatomica 3D y se recibio shape={anat_data.shape}"
            )

    state = {
        "z": data.shape[2] // 2,
        "t": 0,
        "x": data.shape[0] // 2,
        "y": data.shape[1] // 2,
    }

    slice_fig, slice_ax = plt.subplots(figsize=(7, 7))
    ts_fig, ts_ax = plt.subplots(figsize=(10, 4))
    plt.subplots_adjust(bottom=0.18)
    display_slice = anat_data if anat_data is not None else data[:, :, :, state["t"]]

    slice_img = slice_ax.imshow(
        display_slice[:, :, state["z"]].T,
        cmap="gray",
        origin="lower",
    )
    marker, = slice_ax.plot([state["x"]], [state["y"]], "ro", markersize=8)
    vline = slice_ax.axvline(state["x"], color="deepskyblue", linestyle="--", linewidth=1)
    hline = slice_ax.axhline(state["y"], color="deepskyblue", linestyle="--", linewidth=1)
    slice_ax.set_xlabel("X")
    slice_ax.set_ylabel("Y")

    z_slider_ax = slice_fig.add_axes([0.15, 0.08, 0.7, 0.03])
    t_slider_ax = slice_fig.add_axes([0.15, 0.03, 0.7, 0.03])

    z_slider = Slider(
        z_slider_ax,
        "Z",
        0,
        data.shape[2] - 1,
        valinit=state["z"],
        valstep=1,
    )
    t_slider = Slider(
        t_slider_ax,
        "T",
        0,
        data.shape[3] - 1,
        valinit=state["t"],
        valstep=1,
    )

    def update_timeseries() -> None:
        voxel_ts = data[state["x"], state["y"], state["z"], :]
        ts_ax.clear()
        ts_ax.plot(times, voxel_ts, linewidth=1.5)
        ts_ax.axvline(times[state["t"]], color="crimson", linestyle="--", linewidth=1)
        ts_ax.set_title(
            f"Serie temporal voxel ({state['x']}, {state['y']}, {state['z']})"
        )
        ts_ax.set_xlabel("Tiempo (s)")
        ts_ax.set_ylabel("Intensidad BOLD")
        ts_ax.grid(True, alpha=0.3)
        ts_fig.canvas.draw_idle()

    def update_slice() -> None:
        current_slice = anat_data if anat_data is not None else data[:, :, :, state["t"]]
        slice_img.set_data(current_slice[:, :, state["z"]].T)
        marker.set_data([state["x"]], [state["y"]])
        vline.set_xdata([state["x"], state["x"]])
        hline.set_ydata([state["y"], state["y"]])
        base_name = Path(anat_path).name if anat_path is not None else fmri_path.name
        slice_ax.set_title(f"{base_name} | corte z={state['z']} | volumen t={state['t']}")
        slice_fig.canvas.draw_idle()

    def on_z_change(val: float) -> None:
        state["z"] = int(val)
        update_slice()
        update_timeseries()

    def on_t_change(val: float) -> None:
        state["t"] = int(val)
        update_slice()
        update_timeseries()

    def onclick(event) -> None:
        if event.inaxes != slice_ax or event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if not (0 <= x < data.shape[0] and 0 <= y < data.shape[1]):
            return

        state["x"] = x
        state["y"] = y
        update_slice()
        update_timeseries()

    z_slider.on_changed(on_z_change)
    t_slider.on_changed(on_t_change)
    slice_fig.canvas.mpl_connect("button_press_event", onclick)

    update_slice()
    update_timeseries()
    plt.show()


def view_subject_fmri(
    subject: str,
    fmri_root: str | Path = DEFAULT_FMRI_ROOT,
    acquisition: str | None = None,
    use_anatomical: bool = True,
) -> None:
    fmri_path = get_subject_fmri_path(subject, fmri_root=fmri_root, acquisition=acquisition)
    anat_path = find_anatomical_for_subject(fmri_root, subject) if use_anatomical else None
    launch_interactive_viewer(fmri_path, anat_path)

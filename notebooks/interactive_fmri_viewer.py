from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.widgets import Slider
from nilearn import image


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
        # Rejillas distintas entre T1 y BOLD son normales; remuestreamos la T1 al
        # espacio del funcional para poder usarla como fondo interactivo.
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


if __name__ == "__main__":
    launch_interactive_viewer(
        r"C:\Users\pedro\Documents\DOCTORADO_Pedro\Code\phD_Repository\data\raw\Dataset1\Simultaneous_EEG_fMRI\BIDS_dataset_MRI\sub-001\ses-001\func\sub-001_ses-001_task-eoec_bold.nii.gz",
        r"C:\Users\pedro\Documents\DOCTORADO_Pedro\Code\phD_Repository\data\raw\Dataset1\Simultaneous_EEG_fMRI\BIDS_dataset_MRI\sub-001\ses-001\anat\sub-001_ses-001_acq-highres_T1w.nii.gz",
    )

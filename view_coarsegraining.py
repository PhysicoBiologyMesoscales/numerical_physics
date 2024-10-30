import numpy as np
import holoviews as hv
import panel as pn
import argparse
import h5py

from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coarse-graining of active particle simulation data"
    )
    parser.add_argument(
        "sim_folder_path", help="Path to folder containing simulation data", type=str
    )
    return parser.parse_args()


def main():
    # Load arguments
    parms = parse_args()
    sim_path = parms.sim_folder_path
    hdf_file = h5py.File(join(sim_path, "data.h5py"))
    asp = hdf_file.attrs["L"] / hdf_file.attrs["l"]
    cg_data = hdf_file["coarse_grained"]
    x = cg_data["x"][()].flatten()
    y = cg_data["y"][()].flatten()

    # Set up widgets
    F_checkbox = pn.widgets.Checkbox(name="F")
    F_color = pn.widgets.ColorPicker(name="F color", value="red")
    list_t = list(cg_data["t"][()].flatten())
    t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)
    list_th = list(cg_data["theta"][()].flatten())
    th_slider = pn.widgets.DiscreteSlider(name="theta", options=list_th)
    select_cmap = pn.widgets.Select(
        name="Color Map", value="blues", options=["blues", "jet", "Reds"]
    )

    def plot_data(t, th, cmap, plot_F, F_color):
        t_idx = list_t.index(t)
        th_idx = list_th.index(th)
        psi = cg_data["psi"][t_idx, :, :, th_idx]
        plot = hv.HeatMap((x, y, psi)).opts(
            cmap=cmap,
            clim=(float(cg_data["psi"][()].min()), float(cg_data["psi"][()].max())),
            width=400,
            height=int(asp * 400),
        )
        F = cg_data["F"][t_idx, :, :, th_idx]
        plot = plot * hv.VectorField(
            (
                x,
                y,
                np.angle(F),
                np.abs(F),
            )
        ).opts(
            alpha=1.0 if plot_F else 0,
            color=F_color,
        ).opts(magnitude=hv.dim("Magnitude").norm())
        return plot

    dmap = hv.DynamicMap(
        pn.bind(
            plot_data,
            t=t_slider,
            th=th_slider,
            cmap=select_cmap,
            plot_F=F_checkbox,
            F_color=F_color,
        )
    )

    row = pn.Row(
        pn.WidgetBox(pn.Column(t_slider, th_slider, select_cmap, F_checkbox, F_color)),
        dmap,
    )

    return hdf_file, row


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    pn.extension()
    sim_path = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\Test_rebuild"
    args = ["prog", sim_path]
    with patch.object(sys, "argv", args):
        hdf_file, row = main()

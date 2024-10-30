import numpy as np
import holoviews as hv
import panel as pn
import argparse
import h5py
import param

from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualization of pair-correlation function"
    )
    parser.add_argument(
        "sim_folder_path", help="Path to folder containing simulation data", type=str
    )
    return parser.parse_args()


def main():
    parms = parse_args()
    sim_path = parms.sim_folder_path
    hdf_file = h5py.File(join(sim_path, "data.h5py"))
    N = hdf_file.attrs["N"]
    asp = hdf_file.attrs["asp"]
    v0 = hdf_file.attrs["v0"]

    sim_data = hdf_file["simulation_data"]

    list_t = list(sim_data["t"][()].flatten())
    t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)
    select_visualizer = pn.widgets.Select(
        name="Visualizer",
        value="theta",
        options=["random", "vx", "vy", "theta", "Fx", "Fy"],
    )
    select_cmap = pn.widgets.Select(
        name="Color Map", value="bwr", options=["viridis", "jet", "blues", "bwr"]
    )

    rand_color = np.arange(N)

    def plot_data(t, vis_field, cmap):
        t_idx = list_t.index(t)

        coords = np.array(
            [
                sim_data["r"][t_idx].real,
                sim_data["r"][t_idx].imag,
                rand_color,
                sim_data["theta"][t_idx],
                # sim_data["v"][t_idx].real,
                # sim_data["v"][t_idx].imag,
                sim_data["F"][t_idx].real,
                sim_data["F"][t_idx].imag,
            ]
        ).T

        kwargs = {}
        match vis_field:
            case "random":
                kwargs["clim"] = (0, N)
            case "theta":
                kwargs["clim"] = (0, 2 * np.pi)
            # case "vx" | "vy":
            #     kwargs["clim"] = (-v0, v0)
            case "Fx":
                Fmax = float(abs(sim_data["Fx"]).max())
                kwargs["clim"] = (-Fmax, Fmax)
            case "Fy":
                Fmax = float(abs(sim_data["Fy"]).max())
                kwargs["clim"] = (-Fmax, Fmax)

        plot = hv.Points(coords, vdims=["random", "theta", "Fx", "Fy"]).opts(
            width=400,
            height=int(asp * 400),
            cmap=cmap,
            size=0.1,
            color=vis_field,
            **kwargs,
        )
        return plot

    dmap = hv.DynamicMap(
        pn.bind(
            plot_data,
            t=t_slider,
            vis_field=select_visualizer,
            cmap=select_cmap,
        )
    )

    row = pn.Row(
        pn.WidgetBox(
            pn.Column(
                t_slider,
                select_visualizer,
                select_cmap,
            )
        ),
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

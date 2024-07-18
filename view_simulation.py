import numpy as np
import holoviews as hv
import panel as pn
import argparse
import xarray as xr

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
    sim_data = xr.open_dataset(join(sim_path, "data.nc"))

    N = sim_data.N
    asp = sim_data.asp
    v0 = sim_data.v0

    sim_data = sim_data.assign(
        vx=v0 * np.cos(sim_data.theta) + sim_data.Fx,
        vy=v0 * np.sin(sim_data.theta) + sim_data.Fy,
    )

    list_t = list(sim_data.t.data)
    t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)
    list_th = list(sim_data.theta.data)
    select_visualizer = pn.widgets.Select(
        name="Visualizer",
        value="random",
        options=["random", "vx", "vy", "theta", "Fx", "Fy"],
    )
    select_cmap = pn.widgets.Select(
        name="Color Map", value="viridis", options=["viridis", "jet", "blues", "Reds"]
    )

    rand_color = np.arange(N)

    def plot_data(t, vis_field, cmap):
        data = sim_data.sel(t=t)
        coords = np.array(
            [
                data.x.data,
                data.y.data,
                rand_color,
                data.theta.data,
                data.vx.data,
                data.vy.data,
                data.Fx.data,
                data.Fy.data,
            ]
        ).T

        kwargs = {}
        match vis_field:
            case "random":
                kwargs["clim"] = (0, N)
            case "theta":
                kwargs["clim"] = (0, 2 * np.pi)
            case "vx" | "vy":
                kwargs["clim"] = (-v0, v0)
            case "Fx":
                Fmax = float(abs(data.Fx).max())
                kwargs["clim"] = (-Fmax, Fmax)
            case "Fy":
                Fmax = float(abs(data.Fy).max())
                kwargs["clim"] = (-Fmax, Fmax)

        plot = hv.Points(
            coords, vdims=["random", "theta", "vx", "vy", "Fx", "Fy"]
        ).opts(
            width=400,
            height=int(asp * 400),
            cmap=cmap,
            size=0.3,
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
    return sim_data, row


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    pn.extension()
    sim_path = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\ar=1.5_N=100000_phi=0.4_v0=3_kc=3.0_k=10_h=0.0"
    args = ["prog", sim_path]
    with patch.object(sys, "argv", args):
        sim_data, row = main()

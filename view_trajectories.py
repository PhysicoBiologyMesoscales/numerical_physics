import numpy as np
import holoviews as hv
import panel as pn
import argparse
import h5py

from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualization of pair-correlation function"
    )
    parser.add_argument(
        "sim_folder_path", help="Path to folder containing simulation data", type=str
    )
    return parser.parse_args()


parms = parse_args()

## Load data
sim_path = parms.sim_folder_path
hdf_file = h5py.File(join(sim_path, "data.h5py"))
N = hdf_file.attrs["N"]
asp = hdf_file.attrs["asp"]
sim_data = hdf_file["simulation_data"]

## Create widgets
list_t = list(sim_data["t"][()].flatten())
t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)
select_visualizer = pn.widgets.Select(
    name="Visualizer",
    value="theta",
    options=["random", "theta", "px", "py", "Fx", "Fy"],
)
select_cmap = pn.widgets.Select(
    name="Color Map", value="bwr", options=["viridis", "jet", "blues", "bwr"]
)

# Random color for visualization
rand_color = np.arange(N)


def plot_data(t, vis_field, cmap):
    t_idx = list_t.index(t)

    coords = np.array(
        [
            sim_data["r"][t_idx].real,
            sim_data["r"][t_idx].imag,
            rand_color,
            sim_data["theta"][t_idx],
            np.cos(sim_data["theta"][t_idx]),
            np.sin(sim_data["theta"][t_idx]),
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
        case "px" | "py":
            kwargs["clim"] = (-1.0, 1.0)
        case "Fx":
            Fmax = float(abs(sim_data["F"][t_idx].real).max())
            kwargs["clim"] = (-Fmax, Fmax)
        case "Fy":
            Fmax = float(abs(sim_data["F"][t_idx].imag).max())
            kwargs["clim"] = (-Fmax, Fmax)

    plot = hv.Points(coords, vdims=["random", "theta", "px", "py", "Fx", "Fy"]).opts(
        width=800,
        height=int(asp * 800),
        cmap=cmap,
        size=np.sqrt(1e5 / N),
        color=vis_field,
        **kwargs,
    )
    return plot


hv.extension("bokeh")
pn.extension()
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
row.servable()


def main():
    anim = hv.HoloMap({t: plot_data(t, "theta", "hsv") for t in list_t})
    hv.save(anim, join(sim_path, "sim.gif"), backend="bokeh", fps=12)


if __name__ == "__main__":
    main()

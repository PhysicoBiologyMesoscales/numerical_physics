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
asp = hdf_file.attrs["asp"]
data = hdf_file["exp_data"]
Nmax = data.attrs.get("Nmax")

## Create widgets
list_t = list(data["t"][()].flatten())
t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)
select_visualizer = pn.widgets.Select(
    name="Visualizer",
    value="theta",
    options=["random", "theta", "px", "py"],
)
select_cmap = pn.widgets.Select(
    name="Color Map", value="hsv", options=["hsv", "viridis", "jet", "bwr"]
)

# Random color for visualization
rand_color = np.arange(Nmax)


def plot_data(t, vis_field, cmap):
    t_idx = list_t.index(t)

    r = data["r"][t_idx]
    mask = ~np.isnan(r)

    r = r[mask]
    color = rand_color[mask]
    theta = data["theta"][t_idx][mask]

    coords = np.array(
        [
            r.real,
            r.imag,
            color,
            theta,
            np.cos(theta),
            np.sin(theta),
        ]
    ).T

    kwargs = {}
    match vis_field:
        case "random":
            kwargs["clim"] = (0, Nmax)
        case "theta":
            kwargs["clim"] = (0, 2 * np.pi)
        case "px" | "py":
            kwargs["clim"] = (-1.0, 1.0)

    plot = hv.Points(coords, vdims=["random", "theta", "px", "py"]).opts(
        width=800,
        height=int(asp * 800),
        cmap=cmap,
        size=np.sqrt(1e5 / Nmax),
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

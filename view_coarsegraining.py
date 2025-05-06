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


# Load arguments
parms = parse_args()
sim_path = parms.sim_folder_path
hdf_file = h5py.File(join(sim_path, "data.h5py"))
asp = hdf_file.attrs["L"] / hdf_file.attrs["l"]
cg_data = hdf_file["coarse_grained"]
x = cg_data["x"][()].flatten()
y = cg_data["y"][()].flatten()

# Set up widgets
list_t = list(cg_data["t"][()].flatten())
t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)
list_th = list(cg_data["theta"][()].flatten())
th_slider = pn.widgets.DiscreteSlider(name="theta", options=list_th)
select_field = pn.widgets.Select(
    name="Field", value="px", options=["psi", "rho", "px", "py"]
)
select_cmap = pn.widgets.Select(
    name="Color Map", value="blues", options=["blues", "jet", "Reds"]
)


def plot_data(t, th, field, cmap):
    t_idx = list_t.index(t)
    th_idx = list_th.index(th)

    match field:
        case "psi":
            data = cg_data[field][t_idx, ..., th_idx]
            clim = (cg_data[field][()].min(), cg_data[field][()].max())
        case "px":
            data = np.real(cg_data["p"][t_idx])
            clim = (-1.0, 1.0)
        case "py":
            data = np.imag(cg_data["p"][t_idx])
            clim = (-1.0, 1.0)
        case "rho":
            data = cg_data[field][t_idx]
            clim = (cg_data[field][()].min(), cg_data[field][()].max())

    plot = hv.HeatMap((x, y, data.T)).opts(
        cmap=cmap,
        clim=clim,
        width=400,
        height=int(asp * 400),
    )
    return plot


def disable_th(x, y):
    th_slider.disabled = x != "psi"


pn.bind(disable_th, select_field, th_slider, watch=True)

dmap = hv.DynamicMap(
    pn.bind(plot_data, t=t_slider, th=th_slider, field=select_field, cmap=select_cmap)
)

app = pn.Row(
    pn.WidgetBox(pn.Column(t_slider, th_slider, select_field, select_cmap)),
    dmap,
)

app.servable()


def main():
    pass


if __name__ == "__main__":
    main()

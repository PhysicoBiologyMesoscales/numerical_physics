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
hdf_file = h5py.File(join(sim_path, "data.h5py"), "r")
pcf_grp = hdf_file["pair_correlation"]
pcf = pcf_grp["pcf"][()]
if len(pcf.shape) == 4:
    pcf = pcf[np.newaxis, ...]
r = pcf_grp["r"][()].flatten()
phi = pcf_grp["phi"][()].flatten()

## Create widgets
t_avg_checkbox = pn.widgets.Checkbox(name="Time Average", value=True)
th_avg_checkbox = pn.widgets.Checkbox(name="Theta Average", value=True)
dth_avg_checkbox = pn.widgets.Checkbox(name="Delta Theta Average")

list_t = [0]
if "t" in pcf_grp:
    list_t = list(pcf_grp["t"][()].flatten())
t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)
list_th = list(pcf_grp["theta"][()].flatten())
th_slider = pn.widgets.DiscreteSlider(name="Theta", options=list_th)
dth_slider = pn.widgets.DiscreteSlider(name="Delta Theta", options=list_th)
list_r = list(r)
rmax_slider = pn.widgets.DiscreteSlider(
    name="Max Radius", value=max(list_r), options=list_r
)
select_cmap = pn.widgets.Select(
    name="Color Map", value="viridis", options=["viridis", "jet", "bwr"]
)
vis_max = pn.widgets.EditableFloatSlider(
    name="Max value", start=1, end=10, step=0.005, value=2
)


def plot_data(t_avg, th_avg, dth_avg, t, th, dth, rmax, cmap, vis_max):
    t_idx = list_t.index(t)
    th_idx = list_th.index(th)
    dth_idx = list_th.index(dth)
    rmax_idx = list_r.index(rmax)
    _sel = {0: t_idx, 3: th_idx, 4: dth_idx}

    avg_dims = ()
    if t_avg:
        avg_dims += (0,)
        _sel.pop(0)
    if th_avg:
        avg_dims += (3,)
        _sel.pop(3)
    if dth_avg:
        avg_dims += (4,)
        _sel.pop(4)
    in_range = pcf[:, : rmax_idx + 1, :, :, :]
    mean_data = in_range.mean(axis=avg_dims, keepdims=True)
    sel_idx = [
        _sel[i] if i in _sel.keys() else (slice(None) if i in [1, 2] else 0)
        for i in range(5)
    ]
    data = mean_data[*sel_idx]

    PHI, R = np.meshgrid(phi, r[: rmax_idx + 1])
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)

    plot = hv.QuadMesh((X, Y, data)).opts(
        cmap=cmap,
        width=400,
        height=400,
        yticks=list(np.round(np.linspace(0, r[rmax_idx], 5), decimals=1)),
        xlim=(-rmax, rmax),
        ylim=(-rmax, rmax),
        clim=(0, vis_max),
        tools=["hover"],
    )
    return plot


dmap = hv.DynamicMap(
    pn.bind(
        plot_data,
        t_avg=t_avg_checkbox,
        th_avg=th_avg_checkbox,
        dth_avg=dth_avg_checkbox,
        t=t_slider,
        th=th_slider,
        dth=dth_slider,
        rmax=rmax_slider,
        cmap=select_cmap,
        vis_max=vis_max,
    )
)


def disable_t(x, y):
    t_slider.disabled = x


def disable_th(x, y):
    th_slider.disabled = x


def disable_dth(x, y):
    dth_slider.disabled = x


pn.bind(disable_t, t_avg_checkbox, t_slider, watch=True)
pn.bind(disable_th, th_avg_checkbox, th_slider, watch=True)
pn.bind(disable_dth, dth_avg_checkbox, dth_slider, watch=True)


hv.extension("bokeh")
pn.extension()
app = pn.Row(
    pn.WidgetBox(
        pn.Column(
            t_avg_checkbox,
            th_avg_checkbox,
            dth_avg_checkbox,
            t_slider,
            th_slider,
            dth_slider,
            rmax_slider,
            vis_max,
            select_cmap,
        )
    ),
    dmap,
)

app.servable()


def main():
    pass


if __name__ == "__main__":
    main()

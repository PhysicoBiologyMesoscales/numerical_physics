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
    pcf_ds = xr.open_dataset(join(sim_path, "pcf.nc"))

    t_avg_checkbox = pn.widgets.Checkbox(name="Time Average")
    th_avg_checkbox = pn.widgets.Checkbox(name="Theta Average")

    list_t = list(pcf_ds.t.data)
    t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)
    list_th = list(pcf_ds.theta.data)
    th_slider = pn.widgets.DiscreteSlider(name="theta", options=list_th)
    list_r = list(pcf_ds.r.data)
    rmax_slider = pn.widgets.DiscreteSlider(name="rmax", options=list_r)
    select_cmap = pn.widgets.Select(
        name="Color Map", value="viridis", options=["viridis", "jet", "bwr"]
    )

    # TODO Fix xticks !
    def plot_data(t_avg, th_avg, t, th, rmax, cmap):
        _sel = {"t": t, "theta": th}
        avg_dims = []
        if t_avg:
            avg_dims.append("t")
            _sel.pop("t")
        if th_avg:
            avg_dims.append("theta")
            _sel.pop("theta")
        mean_data = pcf_ds.mean(dim=avg_dims)
        data = mean_data.sel(**_sel).sel(r=slice(0, rmax))
        plot = hv.HeatMap((data["phi"], data["r"], data["g"])).opts(
            cmap=cmap,
            width=400,
            height=400,
            yticks=list(np.round(np.linspace(0, float(pcf_ds.r.max()), 5), decimals=1)),
            clim=(float(mean_data.g.min()), float(mean_data.g.max())),
            radial=True,
            radius_inner=0,
            tools=["hover"],
        )
        return plot

    dmap = hv.DynamicMap(
        pn.bind(
            plot_data,
            t_avg=t_avg_checkbox,
            th_avg=th_avg_checkbox,
            t=t_slider,
            th=th_slider,
            rmax=rmax_slider,
            cmap=select_cmap,
        )
    )

    def disable_t(x, y):
        t_slider.disabled = x

    def disable_th(x, y):
        th_slider.disabled = x

    pn.bind(disable_t, t_avg_checkbox, t_slider, watch=True)
    pn.bind(disable_th, th_avg_checkbox, th_slider, watch=True)

    row = pn.Row(
        pn.WidgetBox(
            pn.Column(
                t_avg_checkbox,
                th_avg_checkbox,
                t_slider,
                th_slider,
                rmax_slider,
                select_cmap,
            )
        ),
        dmap,
    )
    return pcf_ds, row


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    pn.extension()
    sim_path = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\ar=1.5_N=40000_phi=1.0_v0=3.0_kc=3.0_k=10.0_h=0.0_tmax=1.0"
    args = ["prog", sim_path]
    with patch.object(sys, "argv", args):
        pcf_ds, row = main()

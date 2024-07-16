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

    asp = pcf_ds.asp

    list_t = list(pcf_ds.t.data)
    t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)
    list_th = list(pcf_ds.theta.data)
    th_slider = pn.widgets.DiscreteSlider(name="theta", options=list_th)
    select_cmap = pn.widgets.Select(
        name="Color Map", value="viridis", options=["viridis", "jet", "blues", "Reds"]
    )

    # TODO Fix xticks !
    def plot_data(t, th, cmap):
        data = pcf_ds.sel(t=t, theta=th)
        plot = hv.HeatMap((data["phi"], data["r"], data["g"])).opts(
            cmap=cmap,
            width=400,
            height=400,
            yticks=list(np.arange(6)),
            clim=(float(pcf_ds.g.min()), float(pcf_ds.g.max())),
            radial=True,
            radius_inner=0,
            tools=["hover"],
        )
        return plot

    dmap = hv.DynamicMap(
        pn.bind(
            plot_data,
            t=t_slider,
            th=th_slider,
            cmap=select_cmap,
        )
    )

    row = pn.Row(
        pn.WidgetBox(
            pn.Column(
                t_slider,
                th_slider,
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
    sim_path = (
        r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\test"
    )
    args = ["prog", sim_path]
    with patch.object(sys, "argv", args):
        pcf_ds, row = main()

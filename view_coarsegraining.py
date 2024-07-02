import numpy as np
import pandas as pd
import holoviews as hv
import panel as pn
import json
import argparse

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
    parms = parse_args()
    sim_path = parms.sim_folder_path
    with open(join(sim_path, "cg_parms.json")) as jsonFile:
        parms = json.load(jsonFile)

    asp = parms["L"] / parms["l"]
    cg_data = pd.read_csv(join(sim_path, "cg_data.csv"))
    avg_data = pd.read_csv(join(sim_path, "avg_data.csv"))

    avg_data["F"] = np.sqrt(avg_data["Fx"] ** 2 + avg_data["Fy"] ** 2)
    avg_data["F_angle"] = np.arctan2(avg_data["Fy"], avg_data["Fx"])
    avg_data["p"] = np.sqrt(avg_data["px"] ** 2 + avg_data["py"] ** 2)
    avg_data["p_angle"] = np.arctan2(avg_data["py"], avg_data["px"])

    plot_F = pn.widgets.Checkbox(name="F")
    plot_p = pn.widgets.Checkbox(name="p")
    select_cmap = pn.widgets.Select(
        name="Color Map", value="blues", options=["blues", "jet"]
    )
    list_t = list(pd.unique(avg_data["t"]))
    t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)

    def plot_data(t, cmap, plot_F, plot_p):
        t_data = avg_data[avg_data["t"] == t]
        alpha_F = 0
        alpha_p = 0
        if plot_F:
            alpha_F = 1
        if plot_p:
            alpha_p = 1
        return (
            hv.HeatMap((t_data["x"], t_data["y"], t_data["rho"])).opts(cmap=cmap)
            * hv.VectorField(
                (t_data["x"], t_data["y"], t_data["F_angle"], t_data["F"])
            ).opts(alpha=alpha_F, color=(1, 0, 0))
            * hv.VectorField(
                (t_data["x"], t_data["y"], t_data["p_angle"], t_data["p"])
            ).opts(alpha=alpha_p)
        )

    dmap = hv.DynamicMap(
        pn.bind(plot_data, t=t_slider, cmap=select_cmap, plot_F=plot_F, plot_p=plot_p)
    )

    row = pn.Row(pn.Column(t_slider, select_cmap, plot_F, plot_p), dmap)

    return cg_data, avg_data, plot, row


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    sim_path = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\Gradient_oblique"
    args = ["prog", sim_path]
    with patch.object(sys, "argv", args):
        cg_data, avg_data, plot, row = main()
        row

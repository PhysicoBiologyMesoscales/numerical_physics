import numpy as np
import pandas as pd
import holoviews as hv
import panel as pn
import json
import argparse
import xarray as xr
from sklearn.linear_model import LinearRegression

from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coarse-graining of active particle simulation data"
    )
    parser.add_argument(
        "sim_folder_path", help="Path to folder containing simulation data", type=str
    )
    return parser.parse_args()


def set_data(ds):
    X = np.stack(
        [
            np.concatenate(
                [
                    ds.grad_rho_x.data.flatten(),
                    ds.grad_rho_y.data.flatten(),
                ]
            ),
            np.concatenate(
                [(ds.rho * ds.px).data.flatten(), (ds.rho * ds.py).data.flatten()]
            ),
        ],
        axis=-1,
    )
    y = np.concatenate([ds.Fx.data.flatten(), ds.Fy.data.flatten()])
    return X, y


def main():
    parms = parse_args()
    sim_path = parms.sim_folder_path
    # TODO Changer l'enregistrement des données pour charger directement des xarray
    cg_ds = xr.open_dataset(join(sim_path, "cg_data.nc"))
    avg_ds = xr.open_dataset(join(sim_path, "avg_data.nc"))
    asp = cg_ds.attrs["L"] / cg_ds.attrs["l"]

    avg_ds = avg_ds.assign(
        grad_rho_x=lambda arr: arr.rho.roll(x=-1) - arr.rho.roll(x=1)
    )
    avg_ds = avg_ds.assign(
        grad_rho_y=lambda arr: arr.rho.roll(y=-1) - arr.rho.roll(y=1)
    )

    # Linear regression fit of the forces with fields (grad_rho, p)
    X, y = set_data(avg_ds)
    lr = LinearRegression()
    lr.fit(X, y)
    Nx, Ny = avg_ds.Nx, avg_ds.Ny
    Fpred = lr.predict(X).reshape((2, avg_ds.sizes["t"], Ny, Nx))
    avg_ds["Fpredx"] = (["t", "y", "x"], Fpred[0])
    avg_ds["Fpredy"] = (["t", "y", "x"], Fpred[1])

    # avg_ds.assign(F_pred = lambda x: lr.predict(x[]))

    avg_ds["F"] = np.sqrt(avg_ds["Fx"] ** 2 + avg_ds["Fy"] ** 2)
    avg_ds["F_angle"] = np.arctan2(avg_ds["Fy"], avg_ds["Fx"])
    avg_ds["p"] = avg_ds["rho"] * np.sqrt(avg_ds["px"] ** 2 + avg_ds["py"] ** 2)
    avg_ds["p_angle"] = np.arctan2(avg_ds["py"], avg_ds["px"])
    avg_ds["Fpred"] = np.sqrt(avg_ds["Fpredx"] ** 2 + avg_ds["Fpredy"] ** 2)
    avg_ds["Fpred_angle"] = np.arctan2(avg_ds["Fpredy"], avg_ds["Fpredx"])

    plot_F = pn.widgets.Checkbox(name="F")
    color_F = pn.widgets.ColorPicker(name="F color", value="red")
    plot_Fpred = pn.widgets.Checkbox(name="Predicted")
    color_Fpred = pn.widgets.ColorPicker(name="Predicted F color", value="green")
    plot_p = pn.widgets.Checkbox(name="p")
    color_p = pn.widgets.ColorPicker(name="p color", value="yellow")
    select_cmap = pn.widgets.Select(
        name="Color Map", value="blues", options=["blues", "jet", "Reds"]
    )
    list_t = list(avg_ds.t.data)
    t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)

    def plot_data(t, cmap, plot_F, plot_p, plot_Fpred, f_col, p_col, fpred_col):
        t_data = avg_ds.sel(t=t)
        alpha_F = 0
        alpha_p = 0
        alpha_Fpred = 0
        if plot_F:
            alpha_F = 1
        if plot_p:
            alpha_p = 1
        if plot_Fpred:
            alpha_Fpred = 1
        return (
            hv.HeatMap((t_data["x"], t_data["y"], t_data["rho"])).opts(
                cmap=cmap,
                clim=(float(avg_ds.rho.min()), float(avg_ds.rho.max())),
                width=400,
                height=int(asp * 400),
            )
            * hv.VectorField(
                (
                    t_data["x"],
                    t_data["y"],
                    t_data["F_angle"],
                    t_data["F"] / t_data["F"].mean(),
                )
            )
            .opts(alpha=alpha_F, color=f_col)
            .opts(magnitude=hv.dim("Magnitude").norm())
            * hv.VectorField((t_data["x"], t_data["y"], t_data["p_angle"], t_data["p"]))
            .opts(alpha=alpha_p, color=p_col)
            .opts(magnitude=hv.dim("Magnitude").norm())
            * hv.VectorField(
                (
                    t_data["x"],
                    t_data["y"],
                    t_data["Fpred_angle"],
                    t_data["Fpred"] / t_data["F"].mean(),
                )
            )
            .opts(alpha=alpha_Fpred, color=fpred_col)
            .opts(magnitude=hv.dim("Magnitude").norm())
        )

    dmap = hv.DynamicMap(
        pn.bind(
            plot_data,
            t=t_slider,
            cmap=select_cmap,
            plot_F=plot_F,
            f_col=color_F,
            plot_p=plot_p,
            p_col=color_p,
            plot_Fpred=plot_Fpred,
            fpred_col=color_Fpred,
        )
    )

    row = pn.Row(
        pn.Column(
            t_slider,
            select_cmap,
            pn.Row(plot_F, color_F),
            pn.Row(plot_p, color_p),
            pn.Row(plot_Fpred, color_Fpred),
        ),
        dmap,
    )

    return cg_ds, avg_ds, row, lr


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    pn.extension()
    sim_path = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\Gradient_oblique"
    args = ["prog", sim_path]
    with patch.object(sys, "argv", args):
        cg_ds, avg_ds, row, lr = main()
        row

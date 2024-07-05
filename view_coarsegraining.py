import numpy as np
import holoviews as hv
import panel as pn
import argparse
import xarray as xr
from linear_regression import LinearRegression_xr

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
    cg_ds = xr.open_dataset(join(sim_path, "cg_data.nc"))
    asp = cg_ds.attrs["L"] / cg_ds.attrs["l"]

    cg_ds = cg_ds.assign(
        grad_rhox=lambda arr: (
            ["t", "y", "x"],
            (arr.rho.roll(x=-1) - arr.rho.roll(x=1)).data,
            {"type": "vector", "dir": "x"},
        )
    )
    cg_ds = cg_ds.assign(
        grad_rhoy=lambda arr: (
            ["t", "y", "x"],
            (arr.rho.roll(y=-1) - arr.rho.roll(y=1)).data,
            {"type": "vector", "dir": "y"},
        )
    )

    # Linear regression fit of the forces with fields (grad_rho, p)
    # lr = LinearRegression_xr()
    # lr.fit(cg_ds)
    # cg_ds = lr.predict_on_dataset(cg_ds)

    cg_ds["F"] = np.sqrt(cg_ds["Fx_avg"] ** 2 + cg_ds["Fy_avg"] ** 2)
    cg_ds["F_angle"] = np.arctan2(cg_ds["Fy_avg"], cg_ds["Fx_avg"])
    cg_ds["p"] = cg_ds["rho"] * np.sqrt(cg_ds["px"] ** 2 + cg_ds["py"] ** 2)
    cg_ds["p_angle"] = np.arctan2(cg_ds["py"], cg_ds["px"])
    # cg_ds["F_pred"] = np.sqrt(cg_ds["F_predx"] ** 2 + cg_ds["F_predy"] ** 2)
    # cg_ds["F_pred_angle"] = np.arctan2(cg_ds["F_predy"], cg_ds["F_predx"])

    plot_F = pn.widgets.Checkbox(name="F")
    color_F = pn.widgets.ColorPicker(name="F color", value="red")
    plot_Fpred = pn.widgets.Checkbox(name="Predicted")
    color_Fpred = pn.widgets.ColorPicker(name="Predicted F color", value="green")
    plot_p = pn.widgets.Checkbox(name="p")
    color_p = pn.widgets.ColorPicker(name="p color", value="yellow")
    select_cmap = pn.widgets.Select(
        name="Color Map", value="blues", options=["blues", "jet", "Reds"]
    )
    list_t = list(cg_ds.t.data)
    t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)

    def plot_data(t, cmap, plot_F, plot_p, plot_Fpred, f_col, p_col, fpred_col):
        t_data = cg_ds.drop_dims("theta").sel(t=t)
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
                clim=(float(cg_ds.rho.min()), float(cg_ds.rho.max())),
                width=400,
                height=int(asp * 400),
            )
            * hv.VectorField(
                (
                    t_data["x"],
                    t_data["y"],
                    t_data["F_angle"],
                    t_data["F"],
                )
            )
            .opts(alpha=alpha_F, color=f_col)
            .opts(magnitude=hv.dim("Magnitude").norm())
            * hv.VectorField((t_data["x"], t_data["y"], t_data["p_angle"], t_data["p"]))
            .opts(alpha=alpha_p, color=p_col)
            .opts(magnitude=hv.dim("Magnitude").norm())
            # * hv.VectorField(
            #     (
            #         t_data["x"],
            #         t_data["y"],
            #         t_data["F_pred_angle"],
            #         t_data["F_pred"],
            #     )
            # )
            # .opts(alpha=alpha_Fpred, color=fpred_col)
            # .opts(magnitude=hv.dim("Magnitude").norm())
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
    lr = None
    return cg_ds, row, lr


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    pn.extension()
    sim_path = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\ar=1.5_N=40000_phi=1.0_v0=2.0_kc=3.0_k=4.5_h=0.0"
    args = ["prog", sim_path]
    with patch.object(sys, "argv", args):
        cg_ds, row, lr = main()
        row

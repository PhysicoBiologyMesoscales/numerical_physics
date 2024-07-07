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
            {"name": "density_gradient", "average": 1, "type": "vector", "dir": "x"},
        )
    )
    cg_ds = cg_ds.assign(
        grad_rhoy=lambda arr: (
            ["t", "y", "x"],
            (arr.rho.roll(y=-1) - arr.rho.roll(y=1)).data,
            {"name": "density_gradient", "average": 1, "type": "vector", "dir": "y"},
        )
    )

    # Linear regression fit of the forces with fields (grad_rho, p)
    lr = LinearRegression_xr()
    lr.fit(cg_ds)
    cg_ds = lr.predict_on_dataset(cg_ds)

    def split_by_attr(ds, attr_name):
        fields = {}
        for var_name, variable in ds.items():
            attr_value = variable.attrs.get(attr_name)
            if not attr_value:
                raise (
                    ValueError(
                        f"Variable {var_name} doesn't have attribute {attr_name}"
                    )
                )
            if not attr_value in fields.keys():
                fields[attr_value] = [variable]
            else:
                fields[attr_value].append(variable)
        return fields

    dic_vector_widgets = {}
    dims = dict(cg_ds.sizes)
    dims.pop("theta")
    for vector_field in ["polarity", "force", "force_pred"]:
        x_data = (
            cg_ds.filter_by_attrs(type="vector", dir="x", name=vector_field, average=1)
            .to_dataarray()
            .data[0]
        )
        y_data = (
            cg_ds.filter_by_attrs(type="vector", dir="y", name=vector_field, average=1)
            .to_dataarray()
            .data[0]
        )
        cg_ds[f"{vector_field}_mag"] = (
            list(dims.keys()),
            np.sqrt(x_data**2 + y_data**2),
        )
        cg_ds[f"{vector_field}_angle"] = (
            list(dims.keys()),
            np.arctan2(y_data, x_data),
        )
        dic_vector_widgets[vector_field] = {
            "checkbox": pn.widgets.Checkbox(name=vector_field)
        }
        dic_vector_widgets[vector_field]["color"] = pn.widgets.ColorPicker(
            name=f"{vector_field} color", value="red"
        )

    select_cmap = pn.widgets.Select(
        name="Color Map", value="blues", options=["blues", "jet", "Reds"]
    )
    list_t = list(cg_ds.t.data)
    t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)

    def plot_data(t, cmap, dic_widgets):
        t_data = cg_ds.sel(t=t)
        plot = hv.HeatMap((t_data["x"], t_data["y"], t_data["rho"])).opts(
            cmap=cmap,
            clim=(float(cg_ds.rho.min()), float(cg_ds.rho.max())),
            width=400,
            height=int(asp * 400),
        )
        for vector_field in ["polarity", "force", "force_pred"]:
            plot = plot * hv.VectorField(
                (
                    t_data["x"],
                    t_data["y"],
                    t_data[f"{vector_field}_angle"],
                    t_data[f"{vector_field}_mag"],
                )
            ).opts(
                alpha=1.0 if dic_widgets[vector_field]["checkbox"].value else 0,
                color=dic_widgets[vector_field]["color"].value,
            ).opts(
                magnitude=hv.dim("Magnitude").norm()
            )
        return plot

    dmap = hv.DynamicMap(
        pn.bind(plot_data, t=t_slider, cmap=select_cmap, dic_widgets=dic_vector_widgets)
    )

    row = pn.Row(
        pn.WidgetBox(
            pn.Column(
                t_slider,
                select_cmap,
                *[
                    pn.Row(
                        dic_vector_widgets[field]["checkbox"],
                        dic_vector_widgets[field]["color"],
                    )
                    for field in dic_vector_widgets.keys()
                ],
            )
        ),
        dmap,
    )
    lr = None
    return cg_ds, row, lr, dic_vector_widgets


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    pn.extension()
    sim_path = (
        r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\_temp"
    )
    args = ["prog", sim_path]
    with patch.object(sys, "argv", args):
        cg_ds, row, lr, _dcw = main()
        row

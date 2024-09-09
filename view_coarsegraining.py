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
    # Load arguments
    parms = parse_args()
    sim_path = parms.sim_folder_path
    cg_ds = xr.open_dataset(join(sim_path, "cg_data.nc"))
    asp = cg_ds.attrs["L"] / cg_ds.attrs["l"]

    dx = cg_ds.dx
    dy = cg_ds.dy

    theta = cg_ds.theta

    def grad_x(da):
        return (da.roll(x=-1) - da.roll(x=1)) / 2 / dx

    def grad_y(da):
        return (da.roll(y=-1) - da.roll(y=1)) / 2 / dy

    def e_dot_grad(da):
        return (
            np.cos(theta) * (da.roll(x=-1) - da.roll(x=1)) / 2 / dx
            + np.sin(theta) * (da.roll(y=-1) - da.roll(y=1)) / 2 / dy
        )

    def div(ds: xr.Dataset, vec_field_name):
        return grad_x(ds[f"{vec_field_name}_x"]) + grad_y(ds[f"{vec_field_name}_y"])

    def dot(ds: xr.Dataset, fieldA_name, fieldB_name):
        return (
            ds[f"{fieldA_name}_x"] * ds[f"{fieldB_name}_x"]
            + ds[f"{fieldA_name}_y"] * ds[f"{fieldB_name}_y"]
        )

    def laplacian(da):
        return (da.roll(x=-1) + da.roll(x=1) - 2 * da) / dx**2 + (
            da.roll(y=-1) + da.roll(y=1) - 2 * da
        ) / dy**2

    ## Density gradient
    cg_ds = cg_ds.assign(
        grad_rho_x=(
            ["t", "y", "x"],
            grad_x(cg_ds.rho).data,
            {"name": "grad_rho", "type": "vector", "dir": "x"},
        ),
        grad_rho_y=(
            ["t", "y", "x"],
            grad_y(cg_ds.rho).data,
            {"name": "grad_rho", "type": "vector", "dir": "y"},
        ),
    )

    cg_ds = cg_ds.assign(
        e_x=(
            ["theta"],
            np.cos(theta).data,
            {"name": "e", "type": "vector", "dir": "x"},
        ),
        e_y=(
            ["theta"],
            np.sin(theta).data,
            {"name": "e", "type": "vector", "dir": "y"},
        ),
    )

    ## Training fields for density
    cg_ds = cg_ds.assign(
        f_001_x=(
            ["t", "theta", "y", "x"],
            (cg_ds.psi * cg_ds.rho * cg_ds.e_x).data,
            {"name": "f_001", "type": "vector", "dir": "x", "training": 1},
        ),
        f_001_y=(
            ["t", "theta", "y", "x"],
            (cg_ds.psi * cg_ds.rho * cg_ds.e_y).data,
            {"name": "f_001", "type": "vector", "dir": "y", "training": 1},
        ),
        f_011_x=(
            ["t", "theta", "y", "x"],
            (cg_ds.psi * cg_ds.grad_rho_x).data,
            {"name": "f_011", "type": "vector", "dir": "x", "training": 1},
        ),
        f_011_y=(
            ["t", "theta", "y", "x"],
            (cg_ds.psi * cg_ds.grad_rho_y).data,
            {"name": "f_011", "type": "vector", "dir": "y", "training": 1},
        ),
        f_012_x=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi * (e_dot_grad(cg_ds.rho) * cg_ds.e_x - 0.5 * cg_ds.grad_rho_x)
            ).data,
            {"name": "f_012", "type": "vector", "dir": "x", "training": 1},
        ),
        f_012_y=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi * (e_dot_grad(cg_ds.rho) * cg_ds.e_y - 0.5 * cg_ds.grad_rho_y)
            ).data,
            {"name": "f_012", "type": "vector", "dir": "y", "training": 1},
        ),
        f_021_x=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (2 * grad_x(e_dot_grad(cg_ds.rho)) + laplacian(cg_ds.rho) * cg_ds.e_x)
            ).data,
            {"name": "f_021", "type": "vector", "dir": "x", "training": 1},
        ),
        f_021_y=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (2 * grad_y(e_dot_grad(cg_ds.rho)) + laplacian(cg_ds.rho) * cg_ds.e_y)
            ).data,
            {"name": "f_021", "type": "vector", "dir": "y", "training": 1},
        ),
        f_022_x=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    e_dot_grad(e_dot_grad(cg_ds.rho)) * cg_ds.e_x
                    - 1
                    / 4
                    * (
                        2 * grad_x(e_dot_grad(cg_ds.rho))
                        + div(cg_ds, "grad_rho") * cg_ds.e_x
                    )
                )
            ).data,
            {"name": "f_022", "type": "vector", "dir": "x", "training": 1},
        ),
        f_022_y=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    e_dot_grad(e_dot_grad(cg_ds.rho)) * cg_ds.e_y
                    - 1
                    / 4
                    * (
                        2 * grad_y(e_dot_grad(cg_ds.rho))
                        + div(cg_ds, "grad_rho") * cg_ds.e_y
                    )
                )
            ).data,
            {"name": "f_022", "type": "vector", "dir": "y", "training": 1},
        ),
    )

    cg_ds = cg_ds.assign(
        rho_p_x=cg_ds.rho * cg_ds.p_x,
        rho_p_y=cg_ds.rho * cg_ds.p_y,
    )

    ## Training fields for polarity
    cg_ds = cg_ds.assign(
        f_101_x=(
            ["t", "theta", "y", "x"],
            (cg_ds.psi * dot(cg_ds, "e", "rho_p") * cg_ds.e_x).data,
            {"name": "f_101", "type": "vector", "dir": "x", "training": 1},
        ),
        f_101_y=(
            ["t", "theta", "y", "x"],
            (cg_ds.psi * dot(cg_ds, "e", "rho_p") * cg_ds.e_y).data,
            {"name": "f_101", "type": "vector", "dir": "y", "training": 1},
        ),
        f_102_x=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi * cg_ds.rho * (cg_ds.p_x - dot(cg_ds, "e", "p") * cg_ds.e_x)
            ).data,
            {"name": "f_102", "type": "vector", "dir": "x", "training": 1},
        ),
        f_102_y=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi * cg_ds.rho * (cg_ds.p_y - dot(cg_ds, "e", "p") * cg_ds.e_y)
            ).data,
            {"name": "f_102", "type": "vector", "dir": "y", "training": 1},
        ),
        f_111_x=(
            ["t", "theta", "y", "x"],
            (cg_ds.psi * grad_x(dot(cg_ds, "e", "rho_p"))).data,
            {"name": "f_111", "type": "vector", "dir": "x", "training": 1},
        ),
        f_111_y=(
            ["t", "theta", "y", "x"],
            (cg_ds.psi * grad_y(dot(cg_ds, "e", "rho_p"))).data,
            {"name": "f_111", "type": "vector", "dir": "y", "training": 1},
        ),
        f_112_x=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    div(cg_ds, "rho_p") * cg_ds.e_x
                    + e_dot_grad(cg_ds.rho_p_x)
                    - grad_x(cg_ds.rho * dot(cg_ds, "p", "e"))
                )
            ).data,
            {"name": "f_112", "type": "vector", "dir": "x", "training": 1},
        ),
        f_112_y=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    div(cg_ds, "rho_p") * cg_ds.e_y
                    + e_dot_grad(cg_ds.rho_p_y)
                    - grad_y(cg_ds.rho * dot(cg_ds, "p", "e"))
                )
            ).data,
            {"name": "f_112", "type": "vector", "dir": "y", "training": 1},
        ),
        f_113_x=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    e_dot_grad(dot(cg_ds, "e", "rho_p")) * cg_ds.e_x
                    - 1
                    / 4
                    * (
                        grad_x(dot(cg_ds, "e", "rho_p"))
                        + e_dot_grad(cg_ds.rho_p_x)
                        + div(cg_ds, "rho_p") * cg_ds.e_x
                    )
                )
            ).data,
            {"name": "f_113", "type": "vector", "dir": "x", "training": 1},
        ),
        f_113_y=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    e_dot_grad(dot(cg_ds, "e", "rho_p")) * cg_ds.e_y
                    - 1
                    / 4
                    * (
                        grad_y(dot(cg_ds, "e", "rho_p"))
                        + e_dot_grad(cg_ds.rho_p_y)
                        + div(cg_ds, "rho_p") * cg_ds.e_y
                    )
                )
            ).data,
            {"name": "f_113", "type": "vector", "dir": "y", "training": 1},
        ),
        f_121_x=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    2 * grad_x(e_dot_grad(dot(cg_ds, "e", "rho_p")))
                    + laplacian(dot(cg_ds, "e", "rho_p") * cg_ds.e_x)
                )
            ).data,
            {"name": "f_121", "type": "vector", "dir": "x", "training": 1},
        ),
        f_121_y=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    2 * grad_y(e_dot_grad(dot(cg_ds, "e", "rho_p")))
                    + laplacian(dot(cg_ds, "e", "rho_p") * cg_ds.e_y)
                )
            ).data,
            {"name": "f_121", "type": "vector", "dir": "y", "training": 1},
        ),
        f_122_x=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    2
                    * (
                        grad_x(
                            div(cg_ds, "rho_p") - e_dot_grad(dot(cg_ds, "e", "rho_p"))
                        )
                    )
                    + laplacian(cg_ds.rho_p_x - dot(cg_ds, "e", "rho_p") * cg_ds.e_x)
                )
            ).data,
            {"name": "f_122", "type": "vector", "dir": "x", "training": 1},
        ),
        f_122_y=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    2
                    * (
                        grad_y(
                            div(cg_ds, "rho_p") - e_dot_grad(dot(cg_ds, "e", "rho_p"))
                        )
                    )
                    + laplacian(cg_ds.rho_p_y - dot(cg_ds, "e", "rho_p") * cg_ds.e_y)
                )
            ).data,
            {"name": "f_122", "type": "vector", "dir": "y", "training": 1},
        ),
        f_123_x=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    2 * grad_x(div(cg_ds, "rho_p") - e_dot_grad(dot(cg_ds, "e", "rho_p")))
                    + laplacian(cg_ds.rho_p_x - dot(cg_ds, "e", "rho_p") * cg_ds.e_x)
                )
            ).data,
            {"name": "f_123", "type": "vector", "dir": "x", "training": 1},
        ),
        f_123_y=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    2
                    * grad_y(div(cg_ds, "rho_p") - e_dot_grad(dot(cg_ds, "e", "rho_p")))
                    + laplacian(cg_ds.rho_p_y - dot(cg_ds, "e", "rho_p") * cg_ds.e_y)
                )
            ).data,
            {"name": "f_123", "type": "vector", "dir": "y", "training": 1},
        ),
        f_124_x=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    e_dot_grad(
                        -e_dot_grad(dot(cg_ds, "e", "rho_p")) * cg_ds.e_x
                        + 1 / 2 * div(cg_ds, "rho_p") * cg_ds.e_x
                        + 1 / 2 * e_dot_grad(cg_ds.rho_p_x)
                    )
                    + 1
                    / 4
                    * laplacian(-cg_ds.rho_p_x + dot(cg_ds, "e", "rho_p") * cg_ds.e_x)
                )
            ).data,
            {"name": "f_124", "type": "vector", "dir": "x", "training": 1},
        ),
        f_124_y=(
            ["t", "theta", "y", "x"],
            (
                cg_ds.psi
                * (
                    e_dot_grad(
                        -e_dot_grad(dot(cg_ds, "e", "rho_p")) * cg_ds.e_y
                        + 1 / 2 * div(cg_ds, "rho_p") * cg_ds.e_y
                        + 1 / 2 * e_dot_grad(cg_ds.rho_p_y)
                    )
                    + 1
                    / 4
                    * laplacian(-cg_ds.rho_p_y + dot(cg_ds, "e", "rho_p") * cg_ds.e_y)
                )
            ).data,
            {"name": "f_124", "type": "vector", "dir": "y", "training": 1},
        ),
    )

    ## Training fields for nematic tensor
    cg_ds = cg_ds.assign()

    # Linear regression fit of the forces
    lr = LinearRegression_xr(target_field="F")
    lr.fit(cg_ds)
    # Predict forces on data
    cg_ds = lr.predict_on_dataset(cg_ds)

    def list_fields(ds: xr.Dataset, **kwargs):
        list_fields = []
        for field in ds.filter_by_attrs(**kwargs):
            field_name = ds[field].attrs["name"]
            if field_name in list_fields:
                continue
            list_fields.append(field_name)
        return list_fields

    dic_vector_widgets = {}

    plot_ds = cg_ds.broadcast_like(cg_ds)

    for vector_field in list_fields(plot_ds, type="vector"):
        x_data = plot_ds[f"{vector_field}_x"]
        y_data = plot_ds[f"{vector_field}_y"]
        plot_ds[f"{vector_field}_mag"] = np.sqrt(x_data**2 + y_data**2)
        plot_ds[f"{vector_field}_angle"] = np.arctan2(y_data, x_data)
        dic_vector_widgets[f"{vector_field}_checkbox"] = pn.widgets.Checkbox(
            name=vector_field
        )
        dic_vector_widgets[f"{vector_field}_color"] = pn.widgets.ColorPicker(
            name=f"{vector_field} color", value="red"
        )

    list_t = list(plot_ds.t.data)
    t_slider = pn.widgets.DiscreteSlider(name="t", options=list_t)
    list_th = list(plot_ds.theta.data)
    th_slider = pn.widgets.DiscreteSlider(name="theta", options=list_th)
    select_color_field = pn.widgets.Select(
        name="Color by field", value="psi", options=["psi", "rho"]
    )
    select_cmap = pn.widgets.Select(
        name="Color Map", value="blues", options=["blues", "jet", "Reds"]
    )

    def plot_data(t, th, col_field, cmap, **widgets):
        data = plot_ds.sel(t=t, theta=th)
        plot = hv.HeatMap((data["x"], data["y"], data[col_field])).opts(
            cmap=cmap,
            clim=(float(plot_ds[col_field].min()), float(plot_ds[col_field].max())),
            width=400,
            height=int(asp * 400),
        )
        for vector_field in list_fields(plot_ds, type="vector"):
            plot = plot * hv.VectorField(
                (
                    data["x"],
                    data["y"],
                    data[f"{vector_field}_angle"],
                    data[f"{vector_field}_mag"],
                )
            ).opts(
                alpha=1.0 if widgets[f"{vector_field}_checkbox"] else 0,
                color=widgets[f"{vector_field}_color"],
            ).opts(
                magnitude=hv.dim("Magnitude").norm()
            )
        return plot

    dmap = hv.DynamicMap(
        pn.bind(
            plot_data,
            t=t_slider,
            th=th_slider,
            col_field=select_color_field,
            cmap=select_cmap,
            **dic_vector_widgets,
        )
    )

    row = pn.Row(
        pn.WidgetBox(
            pn.Column(
                t_slider,
                th_slider,
                select_color_field,
                select_cmap,
                *[
                    pn.Row(
                        dic_vector_widgets[f"{field}_checkbox"],
                        dic_vector_widgets[f"{field}_color"],
                    )
                    for field in list_fields(plot_ds, type="vector")
                ],
            )
        ),
        dmap,
    )

    return cg_ds, row, lr


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    pn.extension()
    sim_path = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\ar=1.5_N=40000_phi=1.0_v0=3.0_kc=3.0_k=10.0_h=0.0_tmax=1.0"

    args = ["prog", sim_path]
    with patch.object(sys, "argv", args):
        cg_ds, row, lr = main()
        print(
            {
                field: coef
                for (field, coef) in zip(lr.training_fields, lr.coef_)
            }
        )
        print(
            {
                field: float(
                    coef
                    * np.sqrt(
                        cg_ds[f"{field}_x"] ** 2 + cg_ds[f"{field}_y"] ** 2
                    ).mean()
                )
                for (field, coef) in zip(lr.training_fields, lr.coef_)
            }
        )

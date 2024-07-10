import numpy as np
import json
import argparse
import pandas as pd
import xarray as xr
import scipy.sparse as sp


from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coarse-graining of active particle simulation data"
    )
    parser.add_argument(
        "sim_folder_path", help="Path to folder containing simulation data", type=str
    )
    parser.add_argument(
        "nx", help="Number of discretization points on x axis", type=int
    )
    parser.add_argument(
        "nth", help="Number of discretization points on theta", type=int
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sim_path = args.sim_folder_path
    # Load full sim data and parameters
    full_data = pd.read_csv(
        join(sim_path, "Data.csv"), index_col=["p_id", "t"]
    ).to_xarray()
    with open(join(sim_path, "parms.json")) as json_file:
        parm_dic = json.load(json_file)
    asp = parm_dic["aspect_ratio"]
    Nt = full_data.sizes["t"]
    N = full_data.sizes["p_id"]
    l = np.sqrt(N * np.pi / asp / parm_dic["phi"])
    L = asp * l
    # Load discretization parameters; y discretization is fixed by Nx and aspect ratio to get square tiles
    Nx = args.nx
    Ny = int(Nx * asp)
    Nth = args.nth
    dx = l / Nx
    dy = L / Ny
    dth = 2 * np.pi / Nth

    def coarsegrain_ds(ds):
        # Compute cell number in (theta, y, x) space
        cell_data = (
            (ds.theta // dth * Nx * Ny + ds.y // dy * Nx + ds.x // dx)
            .astype(int)
            .data.flatten()
        )
        # Compute coarse-graining matrix; Cij = 1 if particle j is in cell i, 0 otherwise
        C = sp.eye(Nx * Ny * Nth, format="csr")[cell_data].T / N * Nx * Ny * Nth

        new_ds = xr.Dataset(
            data_vars=dict(
                psi=(
                    ["theta", "y", "x"],
                    np.array(C.sum(axis=1)).reshape((Nth, Ny, Nx)),
                    {"name": "psi", "average": 0},
                )
            ),
            coords=dict(
                theta=np.arange(Nth) * dth, x=np.arange(Nx) * dx, y=np.arange(Ny) * dy
            ),
        )
        new_ds = new_ds.assign(
            Fx=(
                ["theta", "y", "x"],
                (C @ ds.Fx.data).reshape((Nth, Ny, Nx)),
                {"name": "F", "average": 0, "type": "vector", "dir": "x"},
            )
        )
        new_ds = new_ds.assign(
            Fy=(
                ["theta", "y", "x"],
                (C @ ds.Fy.data).reshape((Nth, Ny, Nx)),
                {"name": "F", "average": 0, "type": "vector", "dir": "y"},
            )
        )
        return new_ds

    cg_data = (
        full_data.groupby("t", squeeze=False)
        .apply(coarsegrain_ds)
        .assign_attrs(
            {
                "l": l,
                "L": L,
                "Nx": Nx,
                "Ny": Ny,
                "Nth": Nth,
                "dx": dx,
                "dy": dy,
                "dth": dth,
            }
        )
    )

    cg_data = cg_data.assign(
        rho=(
            ["t", "y", "x"],
            cg_data.psi.sum(dim="theta").data / Nth,
            {"name": "rho", "average": 1, "type": "scalar"},
        )
    )
    rho_wo_zero = np.where(cg_data.rho == 0, 1.0, cg_data.rho)
    cg_data = cg_data.assign(
        px=(
            ["t", "y", "x"],
            (cg_data.psi * np.cos(cg_data.theta)).sum(dim="theta").data
            * dth
            / rho_wo_zero,
            {"name": "p", "average": 1, "type": "vector", "dir": "x"},
        ),
        py=(
            ["t", "y", "x"],
            (cg_data.psi * np.sin(cg_data.theta)).sum(dim="theta").data
            * dth
            / rho_wo_zero,
            {"name": "p", "average": 1, "type": "vector", "dir": "y"},
        ),
        Fx_avg=(
            ["t", "y", "x"],
            (cg_data.psi * cg_data.Fx).sum(dim="theta").data * dth / rho_wo_zero,
            {"name": "F_avg", "average": 1, "type": "vector", "dir": "x"},
        ),
        Fy_avg=(
            ["t", "y", "x"],
            (cg_data.psi * cg_data.Fy).sum(dim="theta").data * dth / rho_wo_zero,
            {"name": "F_avg", "average": 1, "type": "vector", "dir": "y"},
        ),
    )

    cg_data.to_netcdf(join(sim_path, "cg_data.nc"))


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    sim_path = (
        r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\_temp"
    )
    args = ["prog", sim_path, "20", "20"]
    with patch.object(sys, "argv", args):
        main()

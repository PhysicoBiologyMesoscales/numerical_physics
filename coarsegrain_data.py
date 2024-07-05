import numpy as np
import pandas as pd
import json
import argparse
import scipy.sparse as sp
import xarray as xr


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
    full_data = pd.read_csv(join(sim_path, "Data.csv"), index_col=0)
    with open(join(sim_path, "parms.json")) as json_file:
        parm_dic = json.load(json_file)
    N = parm_dic["N"]
    asp = parm_dic["aspect_ratio"]
    l = np.sqrt(N * np.pi / asp / parm_dic["phi"])
    L = asp * l
    # Load discretization parameters; y discretization is fixed by Nx and aspect ratio to get square tiles
    Nx = args.nx
    Ny = int(Nx * asp)
    Nth = args.nth
    dx = l / Nx
    dy = L / Ny
    dth = 2 * np.pi / Nth

    def compute_coarsegrain_matrix(r, theta):
        # We compute forces in (x, y, theta) space
        r_th = np.concatenate([r, np.array([theta % (2 * np.pi)]).T], axis=-1)
        # Build matrix and 1D encoding in (x, y, theta) space
        Cijk = (r_th // np.array([dx, dy, dth])).astype(int)
        C1d_cg = np.ravel_multi_index(Cijk.T, (Nx, Ny, Nth), order="C")
        C = sp.eye(Nx * Ny * Nth, format="csr")[C1d_cg] / N / dx / dy / dth
        return C

    def compute_distribution(C):
        # Compute one-body distribution
        return np.array(C.T.sum(axis=1)).reshape((Nx, Ny, Nth)).swapaxes(0, 1)

    def coarsegrain_field(field, C):
        if field.ndim == 1:
            _field = np.array([field]).T
        else:
            _field = field
        data_ndims = field.shape[-1]
        field_cg = C.T @ _field
        return field_cg.reshape((Nx, Ny, Nth, data_ndims)).swapaxes(0, 1)

    def coarsegrain_df(df: pd.DataFrame) -> pd.DataFrame:
        r = np.stack([df["x"], df["y"]], axis=-1)
        F = np.stack([df["Fx"], df["Fy"]], axis=-1)

        C = compute_coarsegrain_matrix(r, df["theta"])

        psi = compute_distribution(C)

        F_cg = coarsegrain_field(F, C)

        data = np.stack(
            [
                psi.reshape(Nx * Ny * Nth),
                F_cg[:, :, :, 0].reshape(Nx * Ny * Nth),
                F_cg[:, :, :, 1].reshape(Nx * Ny * Nth),
            ],
            axis=-1,
        )

        index = pd.MultiIndex.from_product(
            [np.arange(Ny) * dy, np.arange(Nx) * dx, np.arange(Nth) * dth],
            names=["y", "x", "theta"],
        )

        cols = ["psi", "Fx", "Fy"]

        return pd.DataFrame(data, index=index, columns=cols)

    cg_data = (
        full_data.groupby("t")
        .apply(coarsegrain_df)
        .to_xarray()
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
            cg_data.psi.sum(dim="theta").data * dth,
            {"type": "scalar"},
        )
    )
    rho_wo_zero = np.where(cg_data.rho == 0, 1.0, cg_data.rho)
    cg_data = cg_data.assign(
        px=(
            ["t", "y", "x"],
            (cg_data.psi * np.cos(cg_data.theta)).sum(dim="theta").data
            * dth
            / rho_wo_zero,
            {"type": "vector", "dir": "x"},
        ),
        py=(
            ["t", "y", "x"],
            (cg_data.psi * np.sin(cg_data.theta)).sum(dim="theta").data
            * dth
            / rho_wo_zero,
            {"type": "vector", "dir": "y"},
        ),
        Fx_avg=(
            ["t", "y", "x"],
            (cg_data.psi * cg_data.Fx).sum(dim="theta").data * dth / rho_wo_zero,
            {"type": "vector", "dir": "x"},
        ),
        Fy_avg=(
            ["t", "y", "x"],
            (cg_data.psi * cg_data.Fy).sum(dim="theta").data * dth / rho_wo_zero,
            {"type": "vector", "dir": "y"},
        ),
    )

    cg_data.to_netcdf(join(sim_path, "cg_data.nc"))


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    sim_path = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\ar=1.5_N=40000_phi=1.0_v0=2.0_kc=3.0_k=4.5_h=0.0"
    args = ["prog", sim_path, "20", "20"]
    with patch.object(sys, "argv", args):
        main()

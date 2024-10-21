import numpy as np
import argparse
import scipy.sparse as sp
import h5py

from os.path import join

from scipy.stats import binned_statistic_dd


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
    with h5py.File(join(sim_path, "data.h5py"), "a") as hdf_file:
        # Load full sim data and parameters
        N = hdf_file.attrs["N"]
        asp = hdf_file.attrs["asp"]
        l = hdf_file.attrs["l"]
        L = hdf_file.attrs["L"]
        Nt = hdf_file.attrs["Nt"]
        # Load discretization parameters; y discretization is fixed by Nx and aspect ratio to get square tiles
        Nx = args.nx
        Ny = int(Nx * asp)
        Nth = args.nth
        dx = l / Nx
        dy = L / Ny
        dth = 2 * np.pi / Nth

        # Compute bins edges
        x_bins = np.linspace(0, l, Nx + 1)  # Binning x
        y_bins = np.linspace(0, L, Ny + 1)  # Binning y
        th_bins = np.linspace(0, 2 * np.pi, Nth + 1)

        # Load simulation datasets
        sim_data = hdf_file["simulation_data"]
        t = sim_data["t"]
        r = sim_data["r"]
        F = sim_data["F"]
        theta = sim_data["theta"]

        n_sim_points = r.size

        data = np.stack(
            [
                np.broadcast_to(t[()], r.shape).flatten(),
                r[()].flatten().real,
                r[()].flatten().imag,
                theta[()].flatten(),
            ],
            axis=-1,
        )

        psi = (
            binned_statistic_dd(
                data,
                np.arange(n_sim_points),
                bins=[Nt, x_bins, y_bins, th_bins],
                statistic="count",
            ).statistic
            / N
            / dx
            / dy
            / dth
        )

        F_cg = np.nan_to_num(
            binned_statistic_dd(
                data,
                [F[()].flatten().real, F[()].flatten().imag],
                bins=[Nt, x_bins, y_bins, th_bins],
                statistic="mean",
            ).statistic,
            nan=0,
        )

        if "coarse_grained" in hdf_file:
            del hdf_file["coarse_grained"]

        cg_grp = hdf_file.create_group("coarse_grained")
        cg_grp.attrs["Nx"] = Nx
        cg_grp.attrs["Ny"] = Ny
        cg_grp.attrs["Nth"] = Nth
        cg_grp.attrs["dx"] = dx
        cg_grp.attrs["dy"] = dy
        cg_grp.attrs["dth"] = dth
        cg_grp.create_dataset("x", data=((x_bins[:-1] + x_bins[1:]) / 2)[:, np.newaxis])
        cg_grp.create_dataset("y", data=((y_bins[:-1] + y_bins[1:]) / 2)[:, np.newaxis])
        cg_grp.create_dataset(
            "theta", data=((th_bins[:-1] + th_bins[1:]) / 2)[:, np.newaxis]
        )
        cg_grp.create_dataset("psi", data=psi)
        cg_grp.create_dataset("F", data=F_cg)


if __name__ == "__main__":
    main()

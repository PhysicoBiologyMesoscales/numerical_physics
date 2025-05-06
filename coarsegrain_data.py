import numpy as np
import argparse
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
        "Nth", help="Number of discretization points on theta", type=int
    )
    parser.add_argument(
        "Nx", help="Number of discretization points on x axis", type=int
    )
    return parser.parse_args()


class CoarseGraining:
    def __init__(self, hdf_file: h5py.File, Nx, Nth):
        self.hdf_file = hdf_file
        self.asp = hdf_file.attrs.get("asp")
        self.l = hdf_file.attrs.get("l")
        self.L = hdf_file.attrs.get("L")
        self.Nt = hdf_file.attrs.get("Nt")
        self.t = hdf_file["exp_data"]["t"][()]
        self.N = hdf_file["exp_data"]["N"][()]

        self.Nx = Nx
        self.Ny = int(self.asp * Nx)
        self.Nth = Nth

        self.dx = self.l / self.Nx
        self.dy = self.L / self.Ny
        self.dth = 2 * np.pi / self.Nth

        # Compute bins edges
        self.x_bins = np.linspace(0, self.l, self.Nx + 1)  # Binning x
        self.y_bins = np.linspace(0, self.L, self.Ny + 1)  # Binning y
        self.th_bins = np.linspace(0, 2 * np.pi, self.Nth + 1)

    def set_hdf_file(self):
        if "coarse_grained" in self.hdf_file:
            del self.hdf_file["coarse_grained"]

        cg_grp = self.hdf_file.create_group("coarse_grained")
        cg_grp.attrs.create("Nx", self.Nx)
        cg_grp.attrs.create("Ny", self.Ny)
        cg_grp.attrs.create("Nth", self.Nth)
        cg_grp.attrs.create("dx", self.dx)
        cg_grp.attrs.create("dy", self.dy)
        cg_grp.attrs.create("dth", self.dth)
        cg_grp.create_dataset("t", data=self.t)
        cg_grp.create_dataset("x", data=((self.x_bins[:-1] + self.x_bins[1:]) / 2))
        cg_grp.create_dataset("y", data=((self.y_bins[:-1] + self.y_bins[1:]) / 2))
        cg_grp.create_dataset(
            "theta", data=((self.th_bins[:-1] + self.th_bins[1:]) / 2)
        )
        cg_grp.create_dataset("psi", shape=(self.Nt, self.Nx, self.Ny, self.Nth))
        cg_grp.create_dataset("rho", shape=(self.Nt, self.Nx, self.Ny))
        cg_grp.create_dataset(
            "p", shape=(self.Nt, self.Nx, self.Ny), dtype=np.complex128
        )
        cg_grp.create_dataset("theta_p", shape=(self.Nt, self.Nx, self.Ny))

    def coarsegrain_sim(self):
        # Load experimental data
        data = self.hdf_file["exp_data"]
        r = data["r"]
        theta = data["theta"]
        # Coarse-grained data will be written in cg
        cg = self.hdf_file["coarse_grained"]

        for t in range(self.Nt):
            # Remove non-existing particles
            mask = ~np.isnan(r[t])
            data = np.stack(
                [
                    r[t, mask].real,
                    r[t, mask].imag,
                    theta[t, mask],
                ],
                axis=-1,
            )
            # Count particles in each bin
            counts = binned_statistic_dd(
                data,
                0,
                bins=[self.x_bins, self.y_bins, self.th_bins],
                statistic="count",
            ).statistic

            psi = counts / self.N[t] / self.dx / self.dy / self.dth
            rho = psi.sum(axis=-1) * self.dth
            p = (
                (psi * np.exp(1j * cg["theta"][()])[None, None, :]).sum(axis=-1)
                * self.dth
                / np.where(rho == 0, 1.0, rho)
            )
            cg["psi"][t] = psi
            cg["rho"][t] = rho
            cg["p"][t] = p
            cg["theta_p"][t] = np.angle(p)
            self.hdf_file.flush()


def main():
    args = parse_args()
    sim_path = args.sim_folder_path
    with h5py.File(join(sim_path, "data.h5py"), "a") as hdf_file:
        cg = CoarseGraining(hdf_file, args.Nx, args.Nth)
        cg.set_hdf_file()
        cg.coarsegrain_sim()


if __name__ == "__main__":
    main()

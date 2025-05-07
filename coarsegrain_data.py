import numpy as np
import argparse
import h5py

from os.path import join
from scipy.stats import binned_statistic_dd
from scipy.spatial import KDTree
from tqdm import tqdm


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
        # Grid reference points; values on theta are normalized so we can use only one distance measurement
        x_ref, y_ref, th_ref = np.meshgrid(
            cg["x"], cg["y"], cg["theta"] * self.dx / self.dth, indexing="ij"
        )
        ref_points = np.stack([x_ref.ravel(), y_ref.ravel(), th_ref.ravel()], axis=-1)
        ref_tree = KDTree(
            ref_points, boxsize=[1e20, 1e20, 2 * np.pi * self.dx / self.dth]
        )
        # Gaussian kernel dimensions
        sigma_r, sigma_th = 2 * self.dx, 2 * self.dth

        for t in range(self.Nt):
            # Remove non-existing particles
            mask = ~np.isnan(r[t])
            data = np.stack(
                [
                    r[t, mask].real,
                    r[t, mask].imag,
                    theta[t, mask] * self.dx / self.dth,
                ],
                axis=-1,
            )
            tree = KDTree(data, boxsize=[1e20, 1e20, 2 * np.pi * self.dx / self.dth])
            # Compute distance to grid points
            dist_matrix = tree.sparse_distance_matrix(
                ref_tree, max_distance=3 * sigma_r
            )
            # Create gaussian weight for each particle
            weights = -0.5 * (dist_matrix.power(2) / sigma_r**2)
            weights.data = 1 / (2 * np.pi * sigma_r * sigma_th) * np.exp(weights.data)
            # Sum over all particles to get psi (pdf over position and orientation) for each grid point
            psi = np.array(1 / self.N[t] * weights.sum(axis=0)).reshape(
                (self.Nx, self.Ny, self.Nth)
            )
            # Compute density and polarity from psi
            rho = psi.sum(axis=-1) * self.dth
            p = (
                (psi * np.exp(1j * cg["theta"][()])[None, None, :]).sum(axis=-1)
                * self.dth
                / np.where(rho == 0, 1.0, rho)
            )
            # Save data
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

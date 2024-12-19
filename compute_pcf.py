import numpy as np
import argparse
import h5py

from scipy.spatial import KDTree
from scipy.stats import binned_statistic, binned_statistic_dd
from tqdm import tqdm
from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(
        description="Computation of the pair-correlation function"
    )
    parser.add_argument(
        "sim_folder_path", help="Path to folder containing simulation data", type=str
    )
    parser.add_argument(
        "r_max", help="Maximum radius to compute pair-correlation function", type=float
    )
    parser.add_argument(
        "Nr", help="Number of discretization points on the radial dimension", type=int
    )
    parser.add_argument(
        "Nphi",
        help="Number of discretization points on the azimuthal angle",
        type=int,
    )
    parser.add_argument(
        "Nth",
        help="Number of discretization points on the particle orientation",
        type=int,
    )
    parser.add_argument("t_min", help="Start time to compute pcf", type=float)
    return parser.parse_args()


class PCFComputation:
    def __init__(
        self,
        rmax: float,
        rint: float,
        Nr: int,
        Nphi: int,
        Nth: int,
        hdf_file: h5py.File,
    ):
        self.rmax = rmax
        self.rint = rint
        self.Nr = Nr
        self.Nphi = Nphi
        self.Nth = Nth
        self.hdf_file = hdf_file
        self.Nt = hdf_file.attrs["Nt"]
        self.l = hdf_file.attrs["l"]
        self.L = hdf_file.attrs["L"]
        self.N = hdf_file.attrs["N"]
        self.t = hdf_file["simulation_data"]["t"][()]
        self.compute_bins()

    def compute_bins(self):
        # Compute bins edges
        self.r_bins = np.concatenate(
            [
                np.linspace(0, self.rint, 2),
                np.linspace(
                    self.rint + (self.rmax - self.rint) / (self.Nr - 1),
                    self.rmax,
                    self.Nr - 1,
                ),
            ]
        )
        # r_bins = np.linspace(0, rmax, Nr + 1)  # Binning r
        self.phi_bins = np.linspace(0, 2 * np.pi, self.Nphi + 1)  # Binning phi
        self.th_bins = np.linspace(0, 2 * np.pi, self.Nth + 1)  # Binning theta
        self.dr = np.diff(self.r_bins)
        self.dphi, self.dth = 2 * np.pi / self.Nphi, 2 * np.pi / self.Nth
        self.r = (self.r_bins[:-1] + self.r_bins[1:]) / 2
        self.phi = (self.phi_bins[:-1] + self.phi_bins[1:]) / 2
        self.th = (self.th_bins[:-1] + self.th_bins[1:]) / 2

    def set_hdf_group(self):
        if "pair_correlation" in self.hdf_file:
            del self.hdf_file["pair_correlation"]

        pcf_grp = self.hdf_file.create_group("pair_correlation")

        pcf_grp.attrs["rmax"] = self.rmax
        pcf_grp.attrs["Nr"] = self.Nr
        pcf_grp.attrs["Nphi"] = self.Nphi
        pcf_grp.attrs["Nth"] = self.Nth
        pcf_grp.attrs["dphi"] = self.dphi
        pcf_grp.attrs["dth"] = self.dth

        pcf_grp.create_dataset("t", data=self.t)
        pcf_grp.create_dataset("r", data=self.r)
        pcf_grp.create_dataset("rdr", data=(self.r * self.dr))
        pcf_grp.create_dataset("phi", data=self.phi)
        pcf_grp.create_dataset("theta", data=self.th)

        pcf_grp.create_dataset(
            "N_pairs", shape=(self.Nt, self.Nr, self.Nphi, self.Nth, self.Nth)
        )
        pcf_grp.create_dataset("p_th", shape=(self.Nt, self.Nth))
        pcf_grp.create_dataset("pcf", shape=(self.Nr, self.Nphi, self.Nth, self.Nth))

    def find_pairs(self, r, th, tree):
        if tree is None:
            r_array = np.stack([r.real, r.imag], axis=-1)
            tree = KDTree(r_array, boxsize=[self.l, self.L])
        # Query for in-range pairs
        pairs = tree.query_pairs(self.rmax, output_type="ndarray")
        ## Compute coordinates of each pair
        rij = r[pairs[:, 1]] - r[pairs[:, 0]]
        rij = np.where(rij.real > self.l / 2, rij - self.l, rij)
        rij = np.where(rij.real < -self.l / 2, self.l + rij, rij)
        rij = np.where(rij.imag > self.L / 2, rij - 1j * self.L, rij)
        rij = np.where(rij.imag < -self.L / 2, 1j * self.L + rij, rij)
        # Particle-particle distance
        dij = np.abs(rij)
        # Azimuthal angle
        e_t = np.exp(1j * th)[pairs[:, 0]]
        phi = np.angle(rij / e_t) % (2 * np.pi)
        # Angle of reference particle
        thi = th[pairs[:, 0]]
        # Angle of target particle
        thj = th[pairs[:, 1]]
        # data to bin
        data = np.stack([dij, phi, thi, thj], axis=-1)
        # Number of pairs in each 'cell' with coordinates (r, phi, theta_i, theta_j)
        N_pairs = binned_statistic_dd(
            data,
            0,
            bins=[self.r_bins, self.phi_bins, self.th_bins, self.th_bins],
            statistic="count",
        ).statistic
        return N_pairs

    def compute_p_th(self, th):
        p_th = binned_statistic(th, 0, bins=self.th_bins, statistic="count").statistic
        p_th /= self.N * self.dth
        return p_th

    def set_data(self, N_pairs, p_th, t_idx):
        corr = self.hdf_file["pair_correlation"]
        corr["N_pairs"][t_idx] = N_pairs
        corr["p_th"][t_idx] = p_th

    def compute_and_save_data(self, t_idx):
        sim = self.hdf_file["simulation_data"]
        r = sim["r"][t_idx]
        th = sim["theta"][t_idx]
        corr = self.hdf_file["pair_correlation"]
        corr["N_pairs"][t_idx] = self.find_pairs(r, th, tree=None)
        corr["p_th"][t_idx] = self.compute_p_th(th)

    def compute_pcf(self, t_min, t_max):
        t_min_idx = np.argmin(np.abs(self.t - t_min))
        t_max_idx = np.argmin(np.abs(self.t - t_max))
        corr = self.hdf_file["pair_correlation"]
        N_pairs = corr["N_pairs"][t_min_idx:t_max_idx].mean(axis=0)
        p_th = corr["p_th"][t_min_idx:t_max_idx].mean(axis=0)
        corr["pcf"][()] = (
            self.L
            * self.l
            * N_pairs
            / (self.N * (self.N - 1) / 2)
            / (
                (self.r * self.dr)[:, np.newaxis, np.newaxis, np.newaxis]
                * p_th[np.newaxis, np.newaxis, :, np.newaxis]
                * p_th[np.newaxis, np.newaxis, np.newaxis, :]
                * self.dphi
                * self.dth**2
            )
        )


def main():
    args = parse_args()
    sim_path = args.sim_folder_path
    Nr, Nphi, Nth = args.Nr, args.Nphi, args.Nth
    rmax = args.r_max
    t_min = args.t_min
    with h5py.File(join(sim_path, "data.h5py"), "a") as hdf_file:
        pcf = PCFComputation(rmax, 1.0, Nr, Nphi, Nth, hdf_file)
        pcf.set_hdf_group()
        for t_idx, t in enumerate(pcf.t):
            pcf.compute_and_save_data(t_idx)
        pcf.compute_pcf(t_min, pcf.t[-1])


if __name__ == "__main__":
    main()

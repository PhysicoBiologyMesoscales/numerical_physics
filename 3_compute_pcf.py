import numpy as np
import argparse
import h5py

from scipy.spatial import KDTree
from scipy.stats import binned_statistic_dd
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
    return parser.parse_args()


def main():
    args = parse_args()
    sim_path = args.sim_folder_path
    with h5py.File(join(sim_path, "data.h5py"), "a") as hdf_file:
        N = int(hdf_file.attrs["N"])
        Nt = int(hdf_file.attrs["Nt"])
        l = float(hdf_file.attrs["l"])
        L = float(hdf_file.attrs["L"])

        sim_data = hdf_file["simulation_data"]
        r = sim_data["r"]
        theta = sim_data["theta"]

        # Binning dimensions
        Nr, Nphi, Nth = args.Nr, args.Nphi, args.Nth
        rmax = args.r_max
        # Load only cells which are in the given range of interaction
        dr, dphi, dth = (
            rmax / Nr,
            2 * np.pi / Nphi,
            2 * np.pi / Nth,
        )

        # Compute bins edges
        r_bins = np.linspace(0, rmax, Nr + 1)  # Binning r
        rdr = ((r_bins[:-1] + r_bins[1:]) / 2) * dr
        phi_bins = np.linspace(0, 2 * np.pi, Nphi + 1)  # Binning phi
        th_bins = np.linspace(0, 2 * np.pi, Nth + 1)  # Binning theta

        if "pair_correlation" in hdf_file:
            del hdf_file["pair_correlation"]

        pcf_grp = hdf_file.create_group("pair_correlation")
        pcf_grp.attrs["Nr"] = Nr
        pcf_grp.attrs["Nphi"] = Nphi
        pcf_grp.attrs["Nth"] = Nth
        pcf_grp.attrs["dr"] = dr
        pcf_grp.attrs["dphi"] = dphi
        pcf_grp.attrs["dth"] = dth
        pcf_grp.create_dataset("t", data=sim_data["t"][()])
        pcf_grp.create_dataset(
            "r", data=((r_bins[:-1] + r_bins[1:]) / 2)[:, np.newaxis]
        )
        pcf_grp.create_dataset(
            "phi", data=((phi_bins[:-1] + phi_bins[1:]) / 2)[:, np.newaxis]
        )
        pcf_grp.create_dataset(
            "theta", data=((th_bins[:-1] + th_bins[1:]) / 2)[:, np.newaxis]
        )
        pcf = pcf_grp.create_dataset("pcf", shape=(Nt, Nr, Nphi, Nth, Nth))

        for i in tqdm(list(range(Nt))):
            r_t = r[i]
            th_t = theta[i]
            ## Build KDTree for efficient nearest-neighbour search
            pos = np.stack([r_t.real, r_t.imag], axis=-1)
            pos %= [l, L]
            tree = KDTree(pos, boxsize=[l, L])
            pairs = tree.query_pairs(rmax, output_type="ndarray")
            ## Compute coordinates of each pair
            rij = r_t[pairs[:, 1]] - r_t[pairs[:, 0]]
            rij = np.where(rij.real > l / 2, rij - l, rij)
            rij = np.where(rij.real < -l / 2, l + rij, rij)
            rij = np.where(rij.imag > L / 2, rij - 1j * L, rij)
            rij = np.where(rij.imag < -L / 2, 1j * L + rij, rij)
            # Particle-particle distance
            dij = np.abs(rij)
            # Azimuthal angle
            e_t = np.exp(1j * th_t)[pairs[:, 0]]
            phi = np.angle(rij / e_t) % (2 * np.pi)
            # Angle of reference particle
            thi = th_t[pairs[:, 0]]
            # Angle of target particle
            thj = th_t[pairs[:, 1]]
            # Polarity angle difference
            thij = (thj - thi) % (2 * np.pi)
            # data to bin
            data = np.stack([dij, phi, thi, thj], axis=-1)

            n_pairs = pairs.shape[0]

            p_th = binned_statistic_dd(
                th_t, np.arange(Nth), bins=[th_bins], statistic="count"
            ).statistic
            p_th /= N * 2 * np.pi

            N_pairs_t = binned_statistic_dd(
                data,
                np.arange(n_pairs),
                bins=[r_bins, phi_bins, th_bins, th_bins],
                statistic="count",
            ).statistic

            N_pairs_t /= N * (N - 1) / 2

            pcf_th = (
                L
                * l
                / N
                * N_pairs_t
                / (
                    rdr[:, np.newaxis, np.newaxis, np.newaxis]
                    * p_th[np.newaxis, np.newaxis, :, np.newaxis]
                    * p_th[np.newaxis, np.newaxis, np.newaxis, :]
                    * dth**2
                )
            )

            # Build pcf as g(r, phi, theta, delta_theta) with delta_theta = theta_j - theta_i
            i_indices = np.arange(Nth).reshape(-1, 1)  # Column vector for row indices
            j_indices = np.arange(Nth)  # Row vector for column offsets

            pcf[i] = pcf_th[:, :, i_indices, (i_indices + j_indices) % Nth]

            hdf_file.flush()


if __name__ == "__main__":
    main()

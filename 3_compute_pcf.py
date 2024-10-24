import numpy as np
import argparse
import h5py
from scipy.spatial import KDTree
from scipy.stats import binned_statistic_dd


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
        help="Number of discretization points on the particle position",
        type=int,
    )
    parser.add_argument(
        "Ndth",
        help="Number of discretization points for polarity difference between particles",
        type=int,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sim_path = args.sim_folder_path
    with h5py.File(join(sim_path, "data.h5py"), "a") as hdf_file:
        Nt = hdf_file.attrs["Nt"]

        sim_data = hdf_file["simulation_data"]
        r = sim_data["r"]
        theta = sim_data["theta"]

        # Binning dimensions
        Nr, Nphi, Nth, Ndth = args.Nr, args.Nphi, args.Nth, args.Ndth
        rmax = args.r_max
        # Load only cells which are in the given range of interaction
        dr2, dphi, dth, ddth = (
            rmax**2 / (Nr + 1),
            2 * np.pi / (Nphi + 1),
            2 * np.pi / (Nth + 1),
            2 * np.pi / (Ndth + 1),
        )

        # Compute bins edges
        r_bins = np.sqrt(np.linspace(0, rmax**2, Nr + 1))  # Binning r
        phi_bins = np.linspace(0, 2 * np.pi, Nphi + 1)  # Binning phi
        th_bins = np.linspace(0, 2 * np.pi, Nth + 1)  # Binning theta
        dth_bins = np.linspace(0, 2 * np.pi, Ndth + 1)  # Binning Delta_theta

        if "pair_correlation" in hdf_file:
            del hdf_file["pair_correlation"]

        pcf_grp = hdf_file.create_group("pair_correlation")
        pcf_grp.attrs["Nr"] = Nr
        pcf_grp.attrs["Nphi"] = Nphi
        pcf_grp.attrs["Nth"] = Nth
        pcf_grp.attrs["Ndth"] = Ndth
        pcf_grp.attrs["dr2"] = dr2
        pcf_grp.attrs["dphi"] = dphi
        pcf_grp.attrs["dth"] = dth
        pcf_grp.attrs["ddth"] = ddth
        pcf_grp.create_dataset(
            "r", data=((r_bins[:-1] + r_bins[1:]) / 2)[:, np.newaxis]
        )
        pcf_grp.create_dataset(
            "phi", data=((phi_bins[:-1] + phi_bins[1:]) / 2)[:, np.newaxis]
        )
        pcf_grp.create_dataset(
            "d_theta", data=((dth_bins[:-1] + dth_bins[1:]) / 2)[:, np.newaxis]
        )
        pcf = pcf_grp.create_dataset("pcf", shape=(Nt, Nr, Nphi, Nth, Ndth))

        for i in range(Nt):
            r_t = r[i]
            th_t = theta[i]
            ## Build KDTree for efficient nearest-neighbour search
            pos = np.stack([r_t.real, r_t.imag], axis=-1)
            tree = KDTree(pos)
            pairs = tree.query_pairs(rmax, output_type="ndarray")
            ## Compute coordinates of each pair
            rij = r_t[pairs[:, 0]] - r_t[pairs[:, 1]]
            # Particle-particle distance
            dij = np.abs(rij)
            # Azimuthal angle
            e_t = np.exp(1j * th_t)[pairs[:, 0]]
            phi = np.angle(rij / dij / e_t) % (2 * np.pi)
            # Angle of reference particle
            thi = th_t[pairs[:, 0]]
            # Polarity angle difference
            thij = (th_t[pairs[:, 0]] - th_t[pairs[:, 1]]) % (2 * np.pi)
            # data to bin
            data = np.stack([dij, phi, thi, thij], axis=-1)

            n_pairs = pairs.shape[0]

            pcf_t = binned_statistic_dd(
                data,
                np.arange(n_pairs),
                bins=[r_bins, phi_bins, th_bins, dth_bins],
                statistic="count",
            ).statistic

            pcf_t /= np.where(pcf_t.mean(axis=(2, 3)) > 0, pcf_t.mean(axis=(2, 3)), 1.0)

            pcf[i] = pcf_t

            hdf_file.flush()


if __name__ == "__main__":
    main()

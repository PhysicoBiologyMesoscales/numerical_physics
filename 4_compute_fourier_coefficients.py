import numpy as np
import argparse
import h5py

from os.path import join
from itertools import product


def parse_args():
    parser = argparse.ArgumentParser(
        description="Computation of Fourier coefficients and associated hydrodynamic parameters"
    )
    parser.add_argument(
        "sim_folder_path", help="Path to folder containing simulation data", type=str
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sim_path = args.sim_folder_path

    # Load data; select only t>0
    with h5py.File(join(sim_path, "data.h5py"), "a") as hdf_file:
        pcf_grp = hdf_file["pair_correlation"]
        # Recover pcf coordinates
        r = pcf_grp["r"][()].flatten()
        phi = pcf_grp["phi"][()].flatten()
        theta = pcf_grp["theta"][()].flatten()
        delta_theta = pcf_grp["d_theta"][()].flatten()

        pcf_full = pcf_grp["pcf"][1:]
        # Average over time
        pcf = pcf_full.mean(axis=0)

        dr2 = pcf_grp.attrs["dr2"] / 2
        dphi = pcf_grp.attrs["dphi"]
        dth = pcf_grp.attrs["dth"]
        ddth = pcf_grp.attrs["ddth"]

        def compute_fourier_coeff(pcf, l: int, m: int, n: int):
            """
            Compute the 8 Fourier coefficients associated with integers (l,m,n).
            Coefficients can be 0 if any of l,m equals 0.
            Arguments:
                pcf - array containing pair-correlation function; dimensions are (r, phi, theta, delta_theta)
                l, m, n - integers used to characterize Fourier mode
            """
            mul = 8 / (2 * np.pi) ** 3
            if l == 0:
                mul /= 2
            if m == 0:
                mul /= 2
            if n == 0:
                mul /= 2
            a_lmn = (
                mul
                * pcf
                * np.cos(l * phi)[np.newaxis, :, np.newaxis, np.newaxis]
                * np.cos(m * delta_theta)[np.newaxis, np.newaxis, np.newaxis, :]
                * np.cos(n * theta)[np.newaxis, np.newaxis, :, np.newaxis]
                * dth
                * dphi
                * ddth
            ).sum(axis=(1, 2, 3))
            b_lmn = (
                mul
                * pcf
                * np.sin(l * phi)[np.newaxis, :, np.newaxis, np.newaxis]
                * np.cos(m * delta_theta)[np.newaxis, np.newaxis, np.newaxis, :]
                * np.cos(n * theta)[np.newaxis, np.newaxis, :, np.newaxis]
                * dth
                * dphi
                * ddth
            ).sum(axis=(1, 2, 3))
            c_lmn = (
                mul
                * pcf
                * np.cos(l * phi)[np.newaxis, :, np.newaxis, np.newaxis]
                * np.sin(m * delta_theta)[np.newaxis, np.newaxis, np.newaxis, :]
                * np.cos(n * theta)[np.newaxis, np.newaxis, :, np.newaxis]
                * dth
                * dphi
                * ddth
            ).sum(axis=(1, 2, 3))
            d_lmn = (
                mul
                * pcf
                * np.sin(l * phi)[np.newaxis, :, np.newaxis, np.newaxis]
                * np.sin(m * delta_theta)[np.newaxis, np.newaxis, np.newaxis, :]
                * np.cos(n * theta)[np.newaxis, np.newaxis, :, np.newaxis]
                * dth
                * dphi
                * ddth
            ).sum(axis=(1, 2, 3))
            e_lmn = (
                mul
                * pcf
                * np.cos(l * phi)[np.newaxis, :, np.newaxis, np.newaxis]
                * np.cos(m * delta_theta)[np.newaxis, np.newaxis, np.newaxis, :]
                * np.sin(n * theta)[np.newaxis, np.newaxis, :, np.newaxis]
                * dth
                * dphi
                * ddth
            ).sum(axis=(1, 2, 3))
            f_lmn = (
                mul
                * pcf
                * np.sin(l * phi)[np.newaxis, :, np.newaxis, np.newaxis]
                * np.cos(m * delta_theta)[np.newaxis, np.newaxis, np.newaxis, :]
                * np.sin(n * theta)[np.newaxis, np.newaxis, :, np.newaxis]
                * dth
                * dphi
                * ddth
            ).sum(axis=(1, 2, 3))
            g_lmn = (
                mul
                * pcf
                * np.cos(l * phi)[np.newaxis, :, np.newaxis, np.newaxis]
                * np.sin(m * delta_theta)[np.newaxis, np.newaxis, np.newaxis, :]
                * np.sin(n * theta)[np.newaxis, np.newaxis, :, np.newaxis]
                * dth
                * dphi
                * ddth
            ).sum(axis=(1, 2, 3))
            h_lmn = (
                mul
                * pcf
                * np.sin(l * phi)[np.newaxis, :, np.newaxis, np.newaxis]
                * np.sin(m * delta_theta)[np.newaxis, np.newaxis, np.newaxis, :]
                * np.sin(n * theta)[np.newaxis, np.newaxis, :, np.newaxis]
                * dth
                * dphi
                * ddth
            ).sum(axis=(1, 2, 3))

            return a_lmn, b_lmn, c_lmn, d_lmn, e_lmn, f_lmn, g_lmn, h_lmn

        if "fourier_coefficients" in hdf_file:
            del hdf_file["fourier_coefficients"]

        coeff_grp = hdf_file.create_group("fourier_coefficients")
        coeff_grp.attrs["dr2"] = dr2
        for l, m, n in product(range(3), range(3), range(3)):
            grp = coeff_grp.create_group(f"{l}{m}{n}")
            for coeff_type, coeff in zip(
                list("abcdefgh"), compute_fourier_coeff(pcf, l, m, n)
            ):
                grp.create_dataset(f"{coeff_type}_{l}{m}{n}", data=coeff)


if __name__ == "__main__":
    main()

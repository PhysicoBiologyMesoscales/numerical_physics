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
    parser.add_argument("-l", "--l_range", help="Range for l index", type=str)
    parser.add_argument("-m", "--m_range", help="Range for m index", type=str)
    parser.add_argument("-n", "--n_range", help="Range for n index", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    sim_path = args.sim_folder_path
    l_range_str = args.l_range
    l_range = list(map(int, l_range_str.split("-")))
    m_range_str = args.m_range
    m_range = list(map(int, m_range_str.split("-")))
    n_range_str = args.n_range
    n_range = list(map(int, n_range_str.split("-")))

    # Load data; select only t>0
    with h5py.File(join(sim_path, "data.h5py"), "a") as hdf_file:
        pcf_grp = hdf_file["pair_correlation"]
        # Recover pcf coordinates
        r = pcf_grp["r"][()].flatten()
        phi = pcf_grp["phi"][()].flatten()
        theta = pcf_grp["theta"][()].flatten()

        pcf_full = pcf_grp["pcf"][1:]
        # Average over time
        pcf = pcf_full.mean(axis=0)

        dphi = pcf_grp.attrs["dphi"]
        dth = pcf_grp.attrs["dth"]

        def compute_fourier_coeff(pcf, l: int, m: int, n: int):
            """
            Compute the 8 Fourier coefficients associated with integers (l,m,n).
            Coefficients can be 0 if any of l,m equals 0.
            Arguments:
                pcf - array containing pair-correlation function; dimensions are (r, phi, theta, theta)
                l, m, n - integers used to characterize Fourier mode
            """
            mul = 8 / (2 * np.pi) ** 3
            if l == 0:
                mul /= 2
            if m == 0:
                mul /= 2
            if n == 0:
                mul /= 2
            fourier = np.zeros((pcf.shape[0], 8))
            f_arr = [np.cos, np.sin]
            for i, (f1, f2, f3) in enumerate(product(f_arr, repeat=3)):
                fourier[:, i] = (
                    mul
                    * dphi
                    * dth**2
                    * np.einsum(
                        "rptd,p,t,d",
                        pcf,
                        f2(l * phi),
                        f1(2 * n * theta),
                        f3(m * theta),
                    )
                )
            return fourier

        if "fourier_coefficients" in hdf_file:
            del hdf_file["fourier_coefficients"]

        coeff_grp = hdf_file.create_group("fourier_coefficients")
        coeff_grp.create_dataset("r", data=r)

        for l, m, n in product(
            range(l_range[0], l_range[1] + 1),
            range(m_range[0], m_range[1] + 1),
            range(n_range[0], n_range[1] + 1),
        ):
            coeff_grp.create_dataset(
                f"{l}{m}{n}", data=compute_fourier_coeff(pcf, l, m, n), dtype=np.float64
            )


if __name__ == "__main__":
    main()

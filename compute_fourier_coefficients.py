import numpy as np
import xarray as xr
import argparse
import json

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

    # Load data; select only for r<2 (particle_particle interaction range)
    pcf_ds = xr.open_dataset(join(sim_path, "pcf.nc")).sel(r=slice(0, 2))
    ds = pcf_ds.sel(t=slice(pcf_ds.dt_save, None)).mean(dim="t")

    kc = pcf_ds.kc
    v0 = pcf_ds.v0
    dr2 = pcf_ds.dr2 / 2
    dphi = pcf_ds.dphi
    dth = pcf_ds.dth

    def compute_fourier_coeff(ds, l: int, m: int):
        """
        Compute the 4 Fourier coefficients associated with integers (l,m).
        Coefficients can be 0 if any of l,m equals 0.
        """
        mul = 4 / (2 * np.pi) ** 2
        if l == 0:
            mul /= 2
        if m == 0:
            mul /= 2
        a_lm = (
            mul * ds.g * np.cos(l * ds.phi) * np.cos(m * ds.theta) * dth * dphi
        ).sum(dim=["theta", "phi"])
        b_lm = (
            mul * ds.g * np.sin(l * ds.phi) * np.cos(m * ds.theta) * dth * dphi
        ).sum(dim=["theta", "phi"])
        c_lm = (
            mul * ds.g * np.cos(l * ds.phi) * np.sin(m * ds.theta) * dth * dphi
        ).sum(dim=["theta", "phi"])
        d_lm = (
            mul * ds.g * np.sin(l * ds.phi) * np.sin(m * ds.theta) * dth * dphi
        ).sum(dim=["theta", "phi"])
        return (a_lm, b_lm, c_lm, d_lm)

    # Compute force array
    def F(r):
        return kc * v0 * (2 - np.where(r < 2, r, 0))

    F_arr = F(ds.r)

    # Dictionnary which will hold all hydrodynamic parameters
    parms = {}

    ## m=0 : density
    parms["chi"] = float(
        (np.pi * ds.r * compute_fourier_coeff(ds, 0, 0)[0] * F_arr * dr2).sum(dim="r")
    )
    parms["zeta"] = float(
        (np.pi * compute_fourier_coeff(ds, 1, 0)[0] * F_arr * dr2).sum(dim="r")
    )
    parms["kappa"] = -float(
        (np.pi * ds.r * compute_fourier_coeff(ds, 2, 0)[0] * F_arr * dr2).sum(dim="r")
    )

    ## m=1 : polarity
    parms["mu"] = -float(
        (np.pi * compute_fourier_coeff(ds, 1, 1)[0] * F_arr * dr2).sum(dim="r")
    )

    parms["alpha"] = -float(
        (np.pi * compute_fourier_coeff(ds, 1, 1)[3] * F_arr * dr2).sum(dim="r")
    )

    parms["chi_p"] = float(
        (np.pi * ds.r * compute_fourier_coeff(ds, 0, 1)[0] * F_arr * dr2).sum(dim="r")
    )

    parms["nu1"] = -float(
        (
            np.pi
            * ds.r
            * (compute_fourier_coeff(ds, 2, 1)[0] + compute_fourier_coeff(ds, 2, 1)[3])
            / 4
            * F_arr
            * dr2
        ).sum(dim="r")
    )

    parms["nu2"] = -float(
        (
            np.pi
            * ds.r
            * (compute_fourier_coeff(ds, 2, 1)[0] - compute_fourier_coeff(ds, 2, 1)[3])
            * F_arr
            * dr2
        ).sum(dim="r")
    )

    ## m=2 : nematic tensor
    parms["upsilon1"] = -float(
        (
            np.pi
            * (compute_fourier_coeff(ds, 1, 2)[0] + compute_fourier_coeff(ds, 1, 2)[3])
            / 2
            * F_arr
            * dr2
        ).sum(dim="r")
    )

    parms["upsilon2"] = -float(
        (
            np.pi
            * (compute_fourier_coeff(ds, 1, 2)[0] - compute_fourier_coeff(ds, 1, 2)[3])
            * F_arr
            * dr2
        ).sum(dim="r")
    )

    parms["xi1"] = -float(
        (
            np.pi
            * ds.r
            * (compute_fourier_coeff(ds, 2, 2)[0] + compute_fourier_coeff(ds, 2, 2)[3])
            / 4
            * F_arr
            * dr2
        ).sum(dim="r")
    )

    parms["xi2"] = -float(
        (
            np.pi
            * ds.r
            * (compute_fourier_coeff(ds, 2, 2)[0] - compute_fourier_coeff(ds, 2, 2)[3])
            * F_arr
            * dr2
        ).sum(dim="r")
    )

    parms["chi_Q"] = float(
        (np.pi * ds.r * compute_fourier_coeff(ds, 0, 2)[0] * F_arr * dr2).sum(dim="r")
    )

    with open(join(sim_path, "hydro_parms.json"), "w") as jsonFile:
        json.dump(parms, jsonFile)


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    with open("current_sim.json") as jsonFile:
        sim_path = json.load(jsonFile)["sim_path"]

    args = ["prog", sim_path]

    with patch.object(sys, "argv", args):
        main()

import numpy as np
import xarray as xr
import argparse
import json

from os.path import join


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

    for m in range(3):
        parms[f"a_1{m}"] = -float(
            (np.pi * ds.r * compute_fourier_coeff(ds, 1, m)[0] * F_arr * dr2).sum(
                dim="r"
            )
        )
        parms[f"a_2{m}"] = -float(
            (np.pi * ds.r * compute_fourier_coeff(ds, 1, m)[3] * F_arr * dr2).sum(
                dim="r"
            )
        )
        parms[f"a_3{m}"] = -float(
            (np.pi * ds.r**2 * compute_fourier_coeff(ds, 0, m)[0] * F_arr * dr2).sum(
                dim="r"
            )
        )
        parms[f"a_4{m}"] = -float(
            (np.pi * ds.r**2 * compute_fourier_coeff(ds, 2, m)[0] * F_arr * dr2).sum(
                dim="r"
            )
        )
        parms[f"a_5{m}"] = -float(
            (np.pi * ds.r**2 * compute_fourier_coeff(ds, 2, m)[3] * F_arr * dr2).sum(
                dim="r"
            )
        )
        parms[f"a_6{m}"] = -float(
            (
                np.pi / 8 * ds.r**3 * compute_fourier_coeff(ds, 1, m)[0] * F_arr * dr2
            ).sum(dim="r")
        )
        parms[f"a_7{m}"] = -float(
            (
                np.pi / 2 * ds.r**3 * compute_fourier_coeff(ds, 3, m)[0] * F_arr * dr2
            ).sum(dim="r")
        )
        parms[f"a_8{m}"] = -float(
            (
                np.pi / 8 * ds.r**3 * compute_fourier_coeff(ds, 1, m)[3] * F_arr * dr2
            ).sum(dim="r")
        )
        parms[f"a_9{m}"] = -float(
            (
                np.pi / 2 * ds.r**3 * compute_fourier_coeff(ds, 3, m)[3] * F_arr * dr2
            ).sum(dim="r")
        )
        parms[f"c_1{m}"] = parms[f"a_2{m}"]
        parms[f"c_2{m}"] = -(parms[f"a_3{m}"] + 0.5 * parms[f"a_4{m}"])
        parms[f"c_3{m}"] = parms[f"a_5{m}"] / 2
        parms[f"c_4{m}"] = -2 * (parms[f"a_6{m}"] - 0.25 * parms[f"a_7{m}"])
        parms[f"c_5{m}"] = -2 * (parms[f"a_8{m}"] - 0.25 * parms[f"a_9{m}"])
        parms[f"c_6{m}"] = 2 * parms[f"a_8{m}"]

    with open(join(sim_path, "hydro_parms.json"), "w") as jsonFile:
        json.dump(parms, jsonFile)


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    with open("save_parms.json") as jsonFile:
        save_parms = json.load(jsonFile)
        sim_path = join(save_parms["base_folder"], save_parms["sim"])

    args = ["prog", sim_path]

    with patch.object(sys, "argv", args):
        main()

import numpy as np
import xarray as xr
import pandas as pd
import scipy.sparse as sp
import json
import argparse

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
    return parser.parse_args()


def main():
    args = parse_args()
    sim_path = args.sim_folder_path
    sim_data = xr.open_dataset(join(sim_path, "data.nc"))

    asp = sim_data.asp
    N = sim_data.N
    l = np.sqrt(N * np.pi / asp / sim_data.phi)
    L = asp * l

    Nx = int(l / 2)
    Ny = int(L / 2)
    dx = l / Nx
    dy = L / Ny

    def build_neigbouring_matrix(neigh_range: int = 1):
        """
        Build neighbouring matrix. neighbours[i,j]==1 if i,j cells are neighbours, 0 otherwise.
        """
        datax = np.ones((1, Nx)).repeat(1 + 4 * neigh_range, axis=0)
        datay = np.ones((1, Ny)).repeat(1 + 4 * neigh_range, axis=0)
        offsetsx = np.array(
            [0]
            + [
                f(i)
                for i in range(1, neigh_range + 1)
                for f in (
                    lambda x: -Nx + x,
                    lambda x: Nx - x,
                    lambda x: -x,
                    lambda x: x,
                )
            ]
        )
        offsetsy = np.array(
            [0]
            + [
                f(i)
                for i in range(1, neigh_range + 1)
                for f in (
                    lambda x: -Ny + x,
                    lambda x: Ny - x,
                    lambda x: -x,
                    lambda x: x,
                )
            ]
        )
        neigh_x = sp.dia_matrix((datax, offsetsx), shape=(Nx, Nx))
        neigh_y = sp.dia_matrix((datay, offsetsy), shape=(Ny, Ny))
        return sp.kron(neigh_y, neigh_x)

    neigh_range = int(np.ceil(args.r_max))
    neigh = build_neigbouring_matrix(neigh_range=neigh_range)

    def compute_pcf(ds):
        cell_data = (ds.y // dy * Nx + ds.x // dx).astype(int).data.flatten()
        # Compute coarse-graining matrix; Cij = 1 if particle j is in cell i, 0 otherwise
        C = sp.eye(Nx * Ny, format="csr")[cell_data]
        # inRange[i,j]=1 if cells i,j are in neighbouring cells, 0 otherwise
        inRange = C.dot(neigh).dot(C.T)
        # Get positions and orientations for interacting cells
        x_ = inRange.multiply(ds.x)
        y_ = inRange.multiply(ds.y)
        th_ = inRange.multiply(ds.theta)

        def remove_diagonal(mat):
            return mat - sp.diags(mat.diagonal())

        x_ = remove_diagonal(x_)
        y_ = remove_diagonal(y_)
        th_ = remove_diagonal(th_)

        # Compute direction vectors and apply periodic boundary conditions
        xij = x_ - x_.T
        x_bound_plus = (xij.data > l / 2).astype(int)
        xij.data -= l * x_bound_plus
        x_bound_minus = (xij.data < -l / 2).astype(int)
        xij.data += l * x_bound_minus

        yij = y_ - y_.T
        y_bound_plus = (yij.data > L / 2).astype(int)
        yij.data -= L * y_bound_plus
        y_bound_minus = (yij.data < -L / 2).astype(int)
        yij.data += L * y_bound_minus

        # Compute orientation angle in [0, 2*pi] range
        thij = th_ - th_.T
        thij.data = np.where(thij.data < 0, 2 * np.pi + thij.data, thij.data)

        # Particle-particle distance for interacting particles
        dij = (xij.power(2) + yij.power(2)).power(0.5)

        r_vec = np.stack([xij.data / dij.data, yij.data / dij.data], axis=-1)
        e_vec = np.stack([np.cos(th_.data), np.sin(th_.data)], axis=-1)

        # Compute azimuthal angle between cell polarity and particle-particle vector
        phi_ij = xij.copy()
        phi_ij.data = np.arccos(np.einsum("ij,ij->i", r_vec, e_vec)) * np.sign(
            np.cross(e_vec, r_vec)
        )
        phi_ij.data = np.where(phi_ij.data < 0, 2 * np.pi + phi_ij.data, phi_ij.data)

        ## Compute pair-correlation function

        # Binning dimensions
        Nr, Nphi, Nth = 50, 20, 15
        rmax = neigh_range
        dr2, dth, dphi = rmax**2 / Nr, 2 * np.pi / Nth, 2 * np.pi / Nphi

        # Sparse matrix containing bin index for particle pair (i,j)
        bin_ij = dij.copy()
        bin_ij.data = (
            (dij.data**2 // dr2).astype(int) * Nphi * Nth
            + (phi_ij.data // dphi).astype(int) * Nth
            + (thij.data // dth).astype(int)
        ) + 1
        bin_ij.data = np.where(bin_ij.data > Nr * Nphi * Nth, 0, bin_ij.data)
        bin_ij.eliminate_zeros()
        bin_ij.data -= 1

        pcf = np.zeros((Nr, Nphi, Nth))
        idx_1d, counts = np.unique(bin_ij.data, return_counts=True)
        idx = np.unravel_index(idx_1d, shape=(Nr, Nphi, Nth))
        pcf[idx] = counts

        pcf_ds = xr.Dataset(
            data_vars={"g": (["r", "phi", "theta"], pcf)},
            coords=dict(
                r=np.sqrt(np.linspace(0, rmax**2, Nr)),
                phi=np.linspace(0, 2 * np.pi, Nphi),
                theta=np.linspace(0, 2 * np.pi, Nth),
            ),
        )
        return pcf_ds

    pcf_ds = sim_data.groupby("t", squeeze=False).apply(compute_pcf)
    pcf_ds = pcf_ds.assign_attrs(sim_data.attrs)
    pcf_ds.to_netcdf(join(sim_path, "pcf.nc"))


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    sim_path = (
        r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\test"
    )
    args = ["prog", sim_path, "5"]

    with patch.object(sys, "argv", args):
        main()

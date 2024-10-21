import numpy as np
import scipy.sparse as sp
import tkinter as tk
import pandas as pd
import argparse
import h5py

from os import makedirs, remove
from os.path import join
from tkinter import messagebox
from shutil import rmtree
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch simulation of active particles migrating on anisotropic substrate"
    )
    parser.add_argument("ar", help="Aspect ratio of simulation area", type=float)
    parser.add_argument(
        "N_max", help="Number of particles for a packing fraction phi=1", type=int
    )
    parser.add_argument("phi", help="Packing Fraction", type=float)
    parser.add_argument("v0", help="Particle velocity", type=float)
    parser.add_argument("kc", help="Interaction force intensity", type=float)
    parser.add_argument("k", help="Polarity-Velocity alignment strength", type=float)
    parser.add_argument("h", help="Nematic field intensity", type=float)
    parser.add_argument("t_max", help="Max simulation time", type=float)
    parser.add_argument("--dt", help="Base Time Step", type=float, default=5e-2)
    parser.add_argument(
        "--dt_save",
        help="Time interval between data saves",
        type=float,
        default=1.0,
    )
    parser.add_argument("--save_path", help="Path to save images", type=str)
    args = parser.parse_args()
    return args


def main():

    parms = parse_args()

    # Packing fraction and particle number
    phi = parms.phi
    N = int(parms.N_max * phi)
    # Frame aspect ratio
    aspectRatio = parms.ar
    # Frame width
    l = np.sqrt(parms.N_max * np.pi / aspectRatio)
    L = aspectRatio * l
    # Physical parameters
    v0 = parms.v0  # Propulsion force
    kc = parms.kc  # Collision force
    k = parms.k  # Polarity-velocity coupling
    h = parms.h  # Nematic field intensity
    dt_save = parms.dt_save

    # dt_max depends on v0 to avoid overshooting in particle collision; value is empirical
    dt_max = 5e-2 / v0
    dt = min(parms.dt, dt_max)
    # Compute frame interval between save points and make sure dt divides dt_save
    interval_btw_saves = int(np.ceil(dt_save / dt))
    dt = dt_save / interval_btw_saves

    t_max = parms.t_max
    t_arr = np.arange(0, t_max, dt)
    t_save_arr = np.arange(0, t_max, dt_save)
    Nt_save = len(t_save_arr)

    # Cells lists number
    Nx = int(l / 2)
    Ny = int(L / 2)

    # Cells lists dimensions
    wx = l / Nx
    wy = L / Ny

    def build_neigbouring_matrix():
        """
        Build neighbouring matrix. neighbours[i,j]==1 if i,j cells are neighbours, 0 otherwise.
        """
        datax = np.ones((1, Nx)).repeat(5, axis=0)
        datay = np.ones((1, Ny)).repeat(5, axis=0)
        offsetsx = np.array([-Nx + 1, -1, 0, 1, Nx - 1])
        offsetsy = np.array([-Ny + 1, -1, 0, 1, Ny - 1])
        neigh_x = sp.dia_matrix((datax, offsetsx), shape=(Nx, Nx))
        neigh_y = sp.dia_matrix((datay, offsetsy), shape=(Ny, Ny))
        return sp.kron(neigh_y, neigh_x)

    neighbours = build_neigbouring_matrix()

    def compute_forces(r):
        Cij = (r // np.array([wx, wy])).astype(int)
        # 1D array encoding the index of the cell containing the particle
        C1d = Cij[:, 0] + Nx * Cij[:, 1]
        # One-hot encoding of the 1D cell array as a sparse matrix
        C = sp.eye(Nx * Ny, format="csr")[C1d]
        # N x N array; inRange[i,j]=1 if particles i, j are in neighbouring cells, 0 otherwise
        inRange = C.dot(neighbours).dot(C.T)

        y_ = inRange.multiply(r[:, 1])
        x_ = inRange.multiply(r[:, 0])

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

        # particle-particle distance for interacting particles
        dij = (xij.power(2) + yij.power(2)).power(0.5)

        xij.data /= dij.data
        yij.data /= dij.data
        dij.data -= 2
        dij.data = np.where(dij.data < 0, dij.data, 0)
        dij.eliminate_zeros()
        Fij = -dij  # harmonic
        # Fij = 12 * (-dij).power(-13) - 6 * (-dij).power(-7)  # wca
        Fx = np.array(Fij.multiply(xij).sum(axis=0)).flatten()
        Fy = np.array(Fij.multiply(yij).sum(axis=0)).flatten()
        return Fx, Fy

    # Set initial values for fields
    r = np.random.uniform([0, 0], [l, L], size=(N, 2))
    theta = np.random.uniform(0, 2 * np.pi, size=N)

    save_path = parms.save_path

    try:
        makedirs(save_path)
    except FileExistsError:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        # Display a yes/no messagebox
        result = messagebox.askyesno(
            "Folder already exists", "Folder already exists; overwrite ?"
        )
        if not result:
            raise (FileExistsError("Folder already exists; not overwriting"))
        rmtree(save_path)
        makedirs(save_path)

    with h5py.File(join(save_path, "data.h5py"), "w") as h5py_file:
        ## Save simulation parameters
        h5py_file.attrs["N"] = N
        h5py_file.attrs["phi"] = phi
        h5py_file.attrs["l"] = l
        h5py_file.attrs["L"] = L
        h5py_file.attrs["asp"] = aspectRatio
        h5py_file.attrs["v0"] = v0
        h5py_file.attrs["k"] = k
        h5py_file.attrs["kc"] = kc
        h5py_file.attrs["h"] = h
        h5py_file.attrs["dt"] = dt_save
        h5py_file.attrs["Nt"] = Nt_save
        h5py_file.attrs["t_max"] = t_max
        ## Create group to store simulation results
        sim = h5py_file.create_group("simulation_data")
        sim.attrs["dt_sim"] = dt
        # Create datasets for coordinates
        sim.create_dataset("t", data=t_save_arr)
        sim.create_dataset("p_id", data=np.arange(N))
        # Create datasets for values
        r_ds = sim.create_dataset("r", shape=(Nt_save, N))
        F_ds = sim.create_dataset("F", shape=(Nt_save, N))
        th_ds = sim.create_dataset("theta", shape=(Nt_save, N))

        for i, t in enumerate(tqdm(t_arr)):
            ## Compute forces
            Fx, Fy = compute_forces(r)
            F = kc * np.stack([Fx, Fy], axis=-1)
            # Velocity = v0*(e + F)
            v = v0 * (
                np.stack(
                    [
                        np.cos(theta),
                        np.sin(theta),
                    ],
                    axis=-1,
                )
                + F
            )
            # Gaussian white noise
            xi = np.sqrt(2 * dt) * np.random.randn(N)
            ## Compute angular dynamics
            e_perp = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)

            ## Save data before position/orientation update
            if i % interval_btw_saves == 0:
                r_ds[i // interval_btw_saves] = r[:, 0] + 1j * r[:, 1]
                F_ds[i // interval_btw_saves] = Fx + 1j * Fy
                th_ds[i // interval_btw_saves] = theta
                h5py_file.flush()

            ## Update position
            r += dt * v
            # Periodic BC
            r %= np.array([l, L])
            ## Update orientation
            theta += (
                dt * (-h * np.sin(2 * theta) + k * np.einsum("ij, ij->i", F, e_perp))
                + xi
            )
            theta %= 2 * np.pi


if __name__ == "__main__":
    main()

import numpy as np
import scipy.sparse as sp
from os import makedirs, remove
from os.path import join
import tkinter as tk
from tkinter import messagebox
from shutil import rmtree
import pandas as pd
import argparse
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

    dt_max = (
        5e-2 / v0
    )  # Max dt depends on v0 to avoid overshooting in particle collision
    dt = min(parms.dt, dt_max)
    # Compute frame interval between save points and make sure dt divides dt_save
    interval_btw_saves = int(np.ceil(parms.dt_save / dt))
    dt = parms.dt_save / interval_btw_saves

    t_max = parms.t_max
    t_arr = np.arange(0, t_max, dt)

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

    # Initiate fields
    r = np.random.uniform([0, 0], [l, L], size=(N, 2))
    theta = np.random.uniform(-np.pi, np.pi, size=N)

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

    for i, t in enumerate(tqdm(t_arr)):
        ## Compute forces
        Fx, Fy = compute_forces(r)
        F = v0 * np.stack([kc * Fx, kc * Fy], axis=-1)
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
            data = {
                "t": t,
                "theta": theta,
                "x": r[:, 0],
                "y": r[:, 1],
                "Fx": F[:, 0],
                "Fy": F[:, 1],
            }
            header = False
            if i // interval_btw_saves == 0:
                header = True
            df = pd.DataFrame(data, index=np.arange(N))
            df.index.set_names("p_id", inplace=True)
            df.to_csv(join(save_path, "_temp_data.csv"), mode="a", header=header)

        ## Update position
        r += dt * v
        # Periodic BC
        r %= np.array([l, L])
        ## Update orientation
        theta += (
            dt * (-h * np.sin(2 * theta) + k * np.einsum("ij, ij->i", F, e_perp)) + xi
        )
        theta %= 2 * np.pi

    ds = (
        pd.read_csv(join(save_path, "_temp_data.csv"), index_col=["p_id", "t"])
        .to_xarray()
        .assign_attrs(
            {
                "asp": aspectRatio,
                "N": N,
                "phi": phi,
                "v0": v0,
                "kc": kc,
                "k": k,
                "h": h,
                "l": l,
                "L": L,
                "t_max": t_max,
                "dt": dt,
                "dt_save": parms.dt_save,
            }
        )
    )
    ds.to_netcdf(join(save_path, "data.nc"))
    remove(join(save_path, "_temp_data.csv"))


if __name__ == "__main__":
    import json
    import sys
    from unittest.mock import patch

    # Define sim parameters
    aspect_ratio = 1.5
    N = 10000
    phi = 1.0
    kc = 3.0
    h = 0.0
    dt = 5e-3
    dt_save = 5e-2
    t_max = 1.0
    v0 = 3.0
    k = 10.0
    sim_name = (
        f"ar={aspect_ratio}_N={N}_phi={phi}_v0={v0}_kc={kc}_k={k}_h={h}_tmax={t_max}"
    )

    with open("save_parms.json") as jsonFile:
        save_parms = json.load(jsonFile)
        base_folder = save_parms["base_folder"]

    # Write new folder as current sim
    with open("save_parms.json", "w") as jsonFile:
        save_parms["sim"] = sim_name
        json.dump(save_parms, jsonFile)

    save_path = join(base_folder, sim_name)

    args = [
        "prog",
        str(aspect_ratio),
        str(N),
        str(phi),
        str(v0),
        str(kc),
        str(k),
        str(h),
        str(t_max),
        "--dt",
        str(dt),
        "--dt_save",
        str(dt_save),
        "--save_path",
        save_path,
    ]
    with patch.object(sys, "argv", args):
        main()

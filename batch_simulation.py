import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sp
from os import makedirs
from os.path import join
import tkinter as tk
from tkinter import messagebox
from shutil import rmtree
import pandas as pd
import argparse
import matplotlib
import json


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
    parser.add_argument(
        "--plot_images",
        help="Plot simulation results in plt window",
        action="store_true",
    )
    parser.add_argument(
        "--save_images", help="Save simulation images to disk", action="store_true"
    )
    parser.add_argument("--save_path", help="Path to save images", type=str)
    args = parser.parse_args()
    return args


def main():
    matplotlib.use("Agg")
    plt.ioff()

    parms = parse_args()

    # Packing fraction and particle number
    phi = parms.phi
    N = int(parms.N_max * phi)
    # Frame aspect ratio
    aspectRatio = parms.ar
    # Frame width
    l = np.sqrt(N * np.pi / aspectRatio / phi)
    L = aspectRatio * l
    # Physical parameters
    v0 = parms.v0  # Propulsion force
    kc = parms.kc  # Collision force
    k = parms.k  # Polarity-velocity coupling
    h = parms.h  # Nematic field intensity

    dt = 5e-2 / v0
    t_max = parms.t_max
    Nt = int(t_max / dt)

    # Display parameters
    displayHeight = 7.0
    fig = plt.figure(figsize=(displayHeight / aspectRatio * 2, displayHeight))
    ax_ = fig.add_axes((0, 0, 1 / 2, 1))
    ax_theta = fig.add_axes((1 / 2, 0, 1 / 2, 1))
    for ax in [ax_, ax_theta]:
        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])

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
        x_bound = (xij.data > l / 2).astype(int)
        xij.data += l * (x_bound.T - x_bound)
        yij = y_ - y_.T
        y_bound = (yij.data > L / 2).astype(int)
        yij.data += L * (y_bound.T - y_bound)

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

    if parms.save_images:
        try:
            makedirs(join(save_path, "Images"))
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
            makedirs(join(save_path, "Images"))
        with open(join(save_path, "parms.json"), "w") as jsonFile:
            json.dump(
                {
                    "aspect_ratio": aspectRatio,
                    "N": N,
                    "phi": phi,
                    "v0": v0,
                    "kc": kc,
                    "k": k,
                    "h": h,
                },
                jsonFile,
            )

    for i in range(Nt):
        # Compute forces
        Fx, Fy = compute_forces(r)
        v = v0 * np.stack(
            [
                np.cos(theta),
                np.sin(theta),
            ],
            axis=-1,
        )
        F = v0 * np.stack([kc * Fx, kc * Fy], axis=-1)
        v += F
        xi = np.sqrt(2 * dt) * np.random.randn(N)
        e_perp = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
        theta += (
            dt * (-h * np.sin(2 * theta) + k * np.einsum("ij, ij->i", F, e_perp)) + xi
        )
        theta %= 2 * np.pi
        r += dt * v
        r %= np.array([l, L])

        if i % int(20 * v0) == 1:
            ax_.cla()
            ax_.set_xlim(0, l)
            ax_.set_ylim(0, L)
            ax_.scatter(
                r[:, 0],
                r[:, 1],
                s=np.pi * 1.25 * (72.0 / L * displayHeight) ** 2,
                c=np.arange(N),
                vmin=0,
                vmax=N,
            )
            ax_theta.cla()
            ax_theta.set_xlim(0, l)
            ax_theta.set_ylim(0, L)
            ax_theta.scatter(
                r[:, 0],
                r[:, 1],
                s=np.pi * 1.25 * (72.0 / L * displayHeight) ** 2,
                c=theta,
                vmin=0,
                vmax=2 * np.pi,
                cmap=cm.hsv,
            )
            if parms.plot_images:
                print("plotting image !")
                fig.show()
                plt.pause(0.1)
            if parms.save_images:
                fig.savefig(join(save_path, "Images", f"{i//int(20 * v0)}.png"))
                data = {
                    "t": dt * i * np.ones(N),
                    "theta": theta,
                    "x": r[:, 0],
                    "y": r[:, 1],
                }
                header = False
                if i // int(20 * v0) == 0:
                    header = True
                pd.DataFrame(data).to_csv(
                    join(save_path, "Data.csv"), mode="a", header=header
                )


if __name__ == "__main__":
    main()

import numpy as np
import argparse
import h5py

from os import makedirs
from os.path import join
from tkinter import messagebox
from shutil import rmtree
from tqdm import tqdm
from scipy.spatial import KDTree


def parse_args():
    parser = argparse.ArgumentParser(description="Batch simulation of active particles")
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
        default=0.1,
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

    save_path = parms.save_path

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

    try:
        makedirs(save_path)
    except FileExistsError:
        rmtree(save_path)
        makedirs(save_path)

    # Set initial values for fields
    r = np.random.uniform([0, 0], [l, L], size=(N, 2))
    theta = np.random.uniform(0, 2 * np.pi, size=N)

    # Initialize tree
    tree = KDTree(r, boxsize=[l, L])
    tree_ref = r.copy()

    def compute_forces(r, tree: KDTree):
        F = np.zeros((N, 2))
        pairs = tree.query_pairs(2.0, output_type="ndarray")
        rij = r[pairs[:, 1]] - r[pairs[:, 0]]
        rij[:, 0] = np.where(rij[:, 0] > l / 2, rij[:, 0] - l, rij[:, 0])
        rij[:, 0] = np.where(rij[:, 0] < -l / 2, l + rij[:, 0], rij[:, 0])
        rij[:, 1] = np.where(rij[:, 1] > L / 2, rij[:, 1] - L, rij[:, 1])
        rij[:, 1] = np.where(rij[:, 1] < -L / 2, L + rij[:, 1], rij[:, 1])
        dij = np.linalg.norm(rij, axis=1, keepdims=True)
        dij = np.where(dij == 0, 1.0, dij)
        uij = rij / dij
        np.add.at(F, pairs[:, 0], -kc * (2 * uij - rij))
        np.add.at(F, pairs[:, 1], kc * (2 * uij - rij))
        return F

    count_rebuild = 0

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
        sim.create_dataset("t", data=t_save_arr[:, np.newaxis])
        sim.create_dataset("p_id", data=np.arange(N)[:, np.newaxis])
        # Create datasets for values
        r_ds = sim.create_dataset("r", shape=(Nt_save, N), dtype=np.complex64)
        F_ds = sim.create_dataset("F", shape=(Nt_save, N), dtype=np.complex64)
        th_ds = sim.create_dataset("theta", shape=(Nt_save, N))

        for i, t in enumerate(tqdm(t_arr)):
            ## Compute forces
            F = compute_forces(r, tree)
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
                F_ds[i // interval_btw_saves] = F[:, 0] + 1j * F[:, 1]
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

            # Check if tree needs rebuilding
            disp = abs(r - tree_ref)
            disp[:, 0] = np.where(disp[:, 0] > l / 2, l - disp[:, 0], disp[:, 0])
            disp[:, 1] = np.where(disp[:, 1] > L / 2, L - disp[:, 1], disp[:, 1])
            if np.max(np.linalg.norm(disp, axis=1)) > 1.0:
                tree = KDTree(r, boxsize=[l, L])
                tree_ref = r.copy()
                count_rebuild += 1

        print(f"Tree was rebuilt {count_rebuild} times")


if __name__ == "__main__":
    main()

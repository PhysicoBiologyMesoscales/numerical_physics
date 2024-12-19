import numpy as np
import argparse
import h5py

from os import makedirs
from os.path import join
from shutil import rmtree
from tqdm import tqdm
from scipy.spatial import KDTree

from compute_pcf import PCFComputation


def parse_args():
    parser = argparse.ArgumentParser(description="Batch simulation of active particles")
    parser.add_argument("save_path", help="Path to save images", type=str)
    parser.add_argument("asp", help="Aspect ratio of simulation area", type=float)
    parser.add_argument(
        "N_max", help="Number of particles for a packing fraction phi=1", type=int
    )
    parser.add_argument("phi", help="Packing Fraction", type=float)
    parser.add_argument("v0", help="Particle velocity", type=float)
    parser.add_argument("kc", help="Interaction force intensity", type=float)
    parser.add_argument("k", help="Polarity-Velocity alignment strength", type=float)
    parser.add_argument("h", help="Nematic field intensity", type=float)
    parser.add_argument("D", help="Translational noise intensity", type=float)
    parser.add_argument("t_max", help="Max simulation time", type=float)
    parser.add_argument("--dt", help="Base Time Step", type=float, default=5e-2)
    parser.add_argument(
        "--dt_save",
        help="Time interval between data saves",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-v", "--verbose", help="Display information", type=bool, default=False
    )
    args = parser.parse_args()
    return args


def hexagonal_tiling(phi, l, L):
    phi_star = np.pi / 2 / np.sqrt(3)  # Optimal packing fraction
    D = np.sqrt(phi_star / phi) * 2
    Nx = int(l / D)
    Ny = int(L / (np.sqrt(3) * D / 2))
    x = np.repeat(np.arange(Nx) * D, Ny)
    x[::2] = (x[::2] + D / 2) % l
    y = np.tile(np.arange(Ny) * np.sqrt(3) * D / 2, Nx)
    return np.stack([x, y], axis=-1)


class Simulation:
    def __init__(self, save_path, N_max, phi, asp, v0, kc, k, h, D, dt_save, dt, t_max):
        self.save_path = save_path
        ## Set all parameters
        self.phi = phi
        self.N = int(N_max * phi)
        self.asp = asp
        # Frame dimensions are computed as a function of max number of particles. Dimension unit length is one particle radius.
        self.l = np.sqrt(N_max * np.pi / asp)
        self.L = asp * self.l
        self.v0 = v0  # Propulsion velocity
        self.kc = kc  # Collision force
        self.k = k  # Polarity-velocity coupling
        self.h = h  # Nematic field intensity
        self.D = D  # Translational noise
        ## Set time intervals
        self.dt_save = dt_save
        # dt_max depends on v0 to avoid overshooting in particle collision; value is empirical
        dt_max = 5e-2 / v0
        dt = min(dt, dt_max)
        # Compute frame interval between save points and make sure dt divides dt_save
        self.interval_btw_saves = int(np.ceil(dt_save / dt))
        self.dt = dt_save / self.interval_btw_saves
        self.t_max = t_max
        self.t_arr = np.arange(0, t_max, dt)
        self.t_save_arr = np.arange(0, t_max, dt_save)
        self.Nt_save = len(self.t_save_arr)
        self.count_rebuild = None

    def set_hdf_file(self):
        # Create output directory; remove directory if it already exists
        try:
            makedirs(self.save_path)
        except FileExistsError:
            rmtree(self.save_path)
            makedirs(self.save_path)
        with h5py.File(join(self.save_path, "data.h5py"), "w") as h5py_file:
            ## Save simulation parameters
            h5py_file.attrs["N"] = self.N
            h5py_file.attrs["phi"] = self.phi
            h5py_file.attrs["l"] = self.l
            h5py_file.attrs["L"] = self.L
            h5py_file.attrs["asp"] = self.asp
            h5py_file.attrs["v0"] = self.v0
            h5py_file.attrs["k"] = self.k
            h5py_file.attrs["kc"] = self.kc
            h5py_file.attrs["h"] = self.h
            h5py_file.attrs["dt_save"] = self.dt_save
            h5py_file.attrs["Nt"] = self.Nt_save
            h5py_file.attrs["t_max"] = self.t_max
            ## Create group to store simulation results
            sim = h5py_file.create_group("simulation_data")
            sim.attrs["dt_sim"] = self.dt
            # Create datasets for coordinates
            sim.create_dataset("t", data=self.t_save_arr)
            sim.create_dataset("p_id", data=np.arange(self.N))
            # Create datasets for values
            self.r_ds = sim.create_dataset(
                "r", shape=(self.Nt_save, self.N), dtype=np.complex128
            )
            self.F_ds = sim.create_dataset(
                "F", shape=(self.Nt_save, self.N), dtype=np.complex64
            )
            self.v_ds = sim.create_dataset(
                "v", shape=(self.Nt_save, self.N), dtype=np.complex64
            )
            self.th_ds = sim.create_dataset("theta", shape=(self.Nt_save, self.N))

    def initial_fields(self):
        r = np.random.uniform(0, self.l, size=self.N) + 1j * np.random.uniform(
            0, self.L, size=self.N
        )
        # theta = np.random.uniform(0, 2 * np.pi, size=N)
        theta = np.random.normal(0, 0.5, self.N)
        # Initialize tree
        tree = KDTree(np.stack([r.real, r.imag], axis=-1), boxsize=[self.l, self.L])
        tree_ref = r.copy()
        self.count_rebuild = 0
        return r, theta, tree, tree_ref

    def get_interacting_pairs(self, r, tree: KDTree):
        pairs = tree.query_pairs(2.0, output_type="ndarray")
        rij = r[pairs[:, 1]] - r[pairs[:, 0]]
        rij = np.where(rij.real > self.l / 2, rij - self.l, rij)
        rij = np.where(rij.real < -self.l / 2, self.l + rij, rij)
        rij = np.where(rij.imag > self.L / 2, rij - 1j * self.L, rij)
        rij = np.where(rij.imag < -self.L / 2, 1j * self.L + rij, rij)
        dij = np.abs(rij)
        return pairs, rij, dij

    def compute_forces(self, pairs, rij, dij):
        F = np.zeros(self.N, dtype=np.complex128)
        dij = np.where(dij == 0, 1.0, dij)
        uij = rij / dij
        np.add.at(F, pairs[:, 0], -self.kc * (2 * uij - rij))
        np.add.at(F, pairs[:, 1], self.kc * (2 * uij - rij))
        return F

    def sim_step(self, r, theta, F):
        v = self.v0 * (np.exp(1j * theta) + F)
        # Gaussian white noise
        xi = np.sqrt(2 * self.dt) * np.random.normal(size=self.N)
        # Translational noise
        eta = np.sqrt(2 * self.D * self.dt) * (
            np.random.normal(scale=np.sqrt(2) / 2, size=self.N)
            + 1j * np.random.normal(scale=np.sqrt(2) / 2, size=self.N)
        )
        ## Update position
        r += self.dt * v + eta
        # Periodic BC
        r.real %= self.l
        r.imag %= self.L

        ## Update orientation
        theta += (
            self.dt
            * (-self.h * np.sin(2 * theta) + self.k * (F * np.exp(-1j * theta)).imag)
            + xi
        )
        theta %= 2 * np.pi
        return r, theta

    def update_tree(self, r, tree, tree_ref):
        # Check if tree needs rebuilding
        disp = r - tree_ref
        disp.real = abs(disp.real)
        disp.imag = abs(disp.imag)
        disp = np.where(disp.real > self.l / 2, self.l - disp, disp)
        disp = np.where(disp.imag > self.L / 2, 1j * self.L - disp, disp)
        # Update tree if at least one particle moved one radius away from its ref position
        if np.max(np.abs(disp)) > 1.0:
            tree = KDTree(np.stack([r.real, r.imag], axis=-1), boxsize=[self.l, self.L])
            tree_ref = r.copy()
            self.count_rebuild += 1
        return tree, tree_ref

    def save_data(self, r, theta, F, save_idx, h5py_file):
        sim = h5py_file["simulation_data"]
        sim["r"][save_idx] = r
        sim["theta"][save_idx] = theta
        sim["F"][save_idx] = F

    def run_sim(self):
        r, theta, tree, tree_ref = self.initial_fields()
        with h5py.File(join(self.save_path, "data.h5py"), "a") as hdf_file:
            pcf = PCFComputation(2.0, 1.0, 20, 30, 30, hdf_file=hdf_file)
            pcf.compute_bins()
            pcf.set_hdf_group()
            # Temp arrays to store pcf statistics at each time step
            _N_pairs = np.zeros(
                (self.interval_btw_saves, pcf.Nr, pcf.Nphi, pcf.Nth, pcf.Nth)
            )
            _p_th = np.zeros((self.interval_btw_saves, pcf.Nth))
            for t_idx, t in enumerate(tqdm(self.t_arr)):
                # Find pairs and compute forces
                pairs, rij, dij = self.get_interacting_pairs(r, tree)
                F = self.compute_forces(pairs, rij, dij)
                # Store pcf data in temp arrays
                _N_pairs[t_idx % self.interval_btw_saves] = pcf.find_pairs(
                    r, theta, tree
                )
                _p_th[t_idx % self.interval_btw_saves] = pcf.compute_p_th(theta)
                # Check if data needs saving
                if t_idx % self.interval_btw_saves == 0:
                    save_idx = t_idx // self.interval_btw_saves
                    # Save simulation data
                    self.save_data(r, theta, F, save_idx, hdf_file)
                    # Save pcf data averaged over all timesteps
                    pcf.set_data(_N_pairs.mean(axis=0), _p_th.mean(axis=0), save_idx)
                    hdf_file.flush()
                # Perform simulation step
                r, theta = self.sim_step(r, theta, F)
                # Update tree if needed
                tree, tree_ref = self.update_tree(r, tree, tree_ref)
            # Compute average pcf between t=1.0 and t=t_max
            pcf.compute_pcf(1.0, self.t_max)
            hdf_file.flush()


def main():
    parms = parse_args()
    sim = Simulation(
        parms.save_path,
        parms.N_max,
        parms.phi,
        parms.asp,
        parms.v0,
        parms.kc,
        parms.k,
        parms.h,
        parms.D,
        parms.dt_save,
        parms.dt,
        parms.t_max,
    )
    sim.set_hdf_file()
    sim.run_sim()
    if parms.verbose:
        print(f"Tree was rebuilt {sim.count_rebuild} times")


if __name__ == "__main__":
    main()

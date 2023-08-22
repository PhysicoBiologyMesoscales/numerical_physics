import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from typing import Tuple
import os
from os.path import join, exists
import matplotlib.pyplot as plt


class Simulation:
    def __init__(
        self,
        N: float = 4000,
        f_0: float = 2.0,
        k: float = 5.0,
        h: float = 3.0,
        l: float = 1.0,
        L: float = 1.0,
        R: float = 1e-2,
        k_c: float = 4.0,
        D: float = 5e-4,
        Dr: float = 1.0,
        dt: float = 5e-4,
        N_t: int = 5000,
        plot_sim: bool = True,
        plot_modulo: int = 1,
        write_sim: bool = False,
        write_modulo: int = 1,
    ) -> None:
        """
        N : number of particles
        f_0 : propulsion force
        h : friction anisotropy
        k : velocity-polarity alignment strength
        l : box width
        L : box height
        R : particle radius
        k_c : contact stiffness
        D : diffusion coefficient (translation)
        Dr : rotationnal diffusion coefficient
        dt : time step
        N_t : max number of iterations
        plot_sim : do or do not plot sim result
        plot_modulo : number of sim iterations between each plot
        write_sim : do or do not write sim results to disk
        write_modulo : number of sim iterations between each write
        """
        self.N = N
        self.f_0 = f_0
        self.k = k
        self.h = h
        self.l = l
        self.L = L
        self.R = R
        self.k_c = k_c
        self.D = D
        self.Dr = Dr
        self.dt = dt
        self.N_t = N_t
        self.plot_sim = plot_sim
        self.fig = None
        self.ax = None
        if self.plot_sim:
            aspect_ratio = L / l
            self.fig = plt.figure(
                figsize=(8 / np.sqrt(aspect_ratio), 8 * np.sqrt(aspect_ratio))
            )
            self.ax = self.fig.add_subplot(111)
        self.plot_modulo = plot_modulo
        self.write_sim = write_sim
        self.write_modulo = write_modulo
        # Friction tensor
        self.gamma = np.array([[1, 0], [0, 1 / h]])
        self.out_fold_path = f"Data/N={N}_f0={f_0}_k={k}_h={h}"
        # Cells lists number
        self.Nx = int(self.l / (2 * self.R))
        self.Ny = int(self.L / (2 * self.R))
        # Cells lists width and height
        self.wx = self.l / self.Nx
        self.wy = self.L / self.Ny
        # Simulation iteration
        self.it = 0
        # Empty attributes, will be filled in later
        self.neighbours = None
        self.r = None
        self.theta = None

    def build_neigbouring_matrix(self):
        """
        Build neighbouring matrix. neighbours[i,j]==1 if i,j cells are neighbours, 0 otherwise.
        """
        datax = np.ones((1, self.Nx)).repeat(5, axis=0)
        datay = np.ones((1, self.Ny)).repeat(5, axis=0)
        offsetsx = np.array([-self.Nx + 1, -1, 0, 1, self.Nx - 1])
        offsetsy = np.array([-self.Ny + 1, -1, 0, 1, self.Ny - 1])
        neigh_x = sp.dia_matrix((datax, offsetsx), shape=(self.Nx, self.Nx))
        neigh_y = sp.dia_matrix((datay, offsetsy), shape=(self.Ny, self.Ny))
        self.neighbours = sp.kron(neigh_y, neigh_x)

    def initiate_fields(self):
        """
        Initiate fields r and theta to random values
        """
        self.r = np.random.uniform([0, 0], [self.l, self.L], size=(self.N, 2))
        self.theta = 2 * np.pi * np.random.random(size=self.N)

    def compute_forces(self):
        Cij = (self.r // np.array([self.wx, self.wy])).astype(int)
        # 1D array encoding the index of the cell containing the particle
        C1d = Cij[:, 0] + self.Nx * Cij[:, 1]
        # One-hot encoding of the 1D cell array as a sparse matrix
        C = sp.eye(self.Nx * self.Ny, format="csr")[C1d]
        # N x N array; inRange[i,j]=1 if particles i, j are in neighbouring cells, 0 otherwise
        inRange = C.dot(self.neighbours).dot(C.T)

        y_ = inRange.multiply(self.r[:, 1])
        x_ = inRange.multiply(self.r[:, 0])

        # Compute direction vectors and apply periodic boundary conditions
        xij = x_ - x_.T
        x_bound = (xij.data > self.l / 2).astype(int)
        xij.data += self.l * (x_bound.T - x_bound)
        yij = y_ - y_.T
        y_bound = (yij.data > self.L / 2).astype(int)
        yij.data += self.L * (y_bound.T - y_bound)

        # particle-particle distance for interacting particles
        dij = (xij.power(2) + yij.power(2)).power(0.5)

        xij.data /= dij.data
        yij.data /= dij.data
        dij.data -= 2 * self.R
        dij.data = np.where(dij.data < 0, dij.data, 0)
        dij.eliminate_zeros()
        Fij = -self.k_c / self.R * dij
        Fx = np.array(Fij.multiply(xij).sum(axis=0)).flatten()
        Fy = np.array(Fij.multiply(yij).sum(axis=0)).flatten()
        return Fx, Fy

    def compute_velocity(self):
        Fx, Fy = self.compute_forces()
        f = np.stack(
            [self.f_0 * np.cos(self.theta) + Fx, self.f_0 * np.sin(self.theta) + Fy],
            axis=-1,
        )
        v = f @ self.gamma
        return v

    def update_theta(self, v: npt.NDArray[np.float_]):
        e_perp = np.stack([-np.sin(self.theta), np.cos(self.theta)], axis=-1)
        # Rotational noise
        xi = np.sqrt(2 * self.Dr * self.dt) * np.random.randn(self.N)
        self.theta += self.k * np.einsum("ij, ij->i", v, e_perp) * self.dt + xi

    def sim_step(self):
        """
        Perform one step of the simulation and update iteration count
        """
        v = self.compute_velocity()
        self.update_theta(v)
        # Translational noise
        nu = np.sqrt(2 * self.D * self.dt) * np.random.randn(self.N, 2)
        self.r += self.dt * v + nu
        self.r %= np.array([self.l, self.L])
        # Update iteration
        self.it += 1

    def plot(self):
        self.ax.cla()
        self.ax.set_xlim(0, self.l)
        self.ax.set_ylim(0, self.L)
        self.ax.scatter(
            self.r[:, 0], self.r[:, 1], s=10, c=np.cos(self.theta), vmin=-1, vmax=1
        )
        self.fig.show()
        plt.pause(0.01)

    def write_fields(self):
        """
        Write current fields r and theta to disk.
        """
        r_fold_path = join(self.out_fold_path, "r")
        theta_fold_path = join(self.out_fold_path, "theta")
        if not exists(r_fold_path):
            print("Position folder not found, creating it.")
            os.makedirs(r_fold_path)
        if not exists(theta_fold_path):
            print("Angle folder not found, creating it.")
            os.makedirs(theta_fold_path)
        np.save(join(r_fold_path, f"{self.it}.npy"), self.r)
        np.save(join(theta_fold_path, f"{self.it}.npy"), self.theta)

    def check_events(self):
        # Write event
        if self.write_sim and self.it % self.write_modulo == 0:
            self.write_fields()
        # Plot event
        if self.plot_sim and self.it % self.plot_modulo == 0:
            self.plot()
        # TODO Pair correlation event

    def run(self):
        self.build_neigbouring_matrix()
        self.initiate_fields()
        while self.it < self.N_t:
            self.sim_step()
            self.check_events()


class SingleFile(Simulation):
    def initiate_fields(self):
        super().initiate_fields()
        self.r = np.random.uniform(
            [0, self.L / 2 - 1e-3], [self.l, self.L / 2 + 1e-3], size=(self.N, 2)
        )


if __name__ == "__main__":
    sim = SingleFile(
        N=50,
        f_0=2,
        h=np.infty,
        k=5,
        plot_sim=True,
        write_sim=False,
        l=1.0,
        L=0.2,
        plot_modulo=20,
        N_t=20000,
        D=0,
    )
    sim.run()

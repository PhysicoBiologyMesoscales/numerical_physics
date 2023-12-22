import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sp

plt.ion()


# Packing fraction and particle number
phi = 0.6
N = 5000
# Frame aspect ratio
aspectRatio = 1.0
# Frame width
l = np.sqrt(N * np.pi / aspectRatio / phi)
L = aspectRatio * l
# Physical parameters
F0 = 10  # Propulsion force
Kc = 5  # Collision force
K = 10  # Polarity-velocity coupling
h = np.infty
gamma = np.array([[1, 0], [0, 1 / h]])

Nt = 100000
dt = 5e-2 / F0

# Display parameters
displayWidth = 7.0
fig = plt.figure(figsize=(displayWidth, displayWidth * aspectRatio))
ax = fig.add_axes((0, 0, 1, 1))
fig2 = plt.figure()
ax2 = fig2.add_axes((0.1, 0.1, 0.9, 0.9))

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
theta = np.random.uniform(-np.pi + 1e-2, np.pi - 1e-2, size=N)

avg_cos = []
avg_cos3 = []

for i in range(Nt):
    # Compute forces
    # Fx, Fy = compute_forces(r)
    Fx, Fy = 0, 0
    v = (
        F0
        * np.stack(
            [np.cos(theta) + Kc * Fx, np.sin(theta) + Kc * Fy],
            axis=-1,
        )
        @ gamma
    )
    xi = np.sqrt(2 * dt) * np.random.randn(N)
    e_perp = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
    theta += dt * K * np.einsum("ij, ij->i", v, e_perp) + xi
    r += dt * v
    r %= np.array([l, L])

    if i % int(10 * F0) == 1:
        avg_cos.append((K * F0 - 1) * np.cos(theta).mean())
        avg_cos3.append((K * F0) * np.cos(3 * theta).mean())
        # ax.cla()
        # ax.set_xlim(0, l)
        # ax.set_ylim(0, L)
        # ax.scatter(
        #     r[:, 0],
        #     r[:, 1],
        #     s=np.pi * 1.25 * (72.0 / l * displayWidth) ** 2,
        #     c=np.cos(theta),
        #     vmin=-1,
        #     vmax=1,
        #     cmap=cm.bwr,
        # )
        fig.show()
        N_p = np.count_nonzero(np.cos(theta) > 0)
        ax2.cla()
        ax2.plot()
        ax2.plot(avg_cos)
        ax2.plot(avg_cos3)
        ax2.axhline(np.array(avg_cos).mean())
        ax2.axhline(np.array(avg_cos3).mean())
        # ax2.semilogy()
        fig2.show()
        plt.pause(0.1)

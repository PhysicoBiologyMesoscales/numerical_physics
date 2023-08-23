import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.ion()

# Dimensions
l = 1.0  # Box width
L = 1.0  # Box height
R = 1e-2  # Particle Radius
N = 2000  # Number of particles

# Cells lists number
Nx = int(l / (2 * R))
Ny = int(L / (2 * R))
# Cells lists width and height
wx = l / Nx
wy = L / Ny
# Build neighboring matrix
datax = np.ones((1, Nx)).repeat(5, axis=0)
datay = np.ones((1, Ny)).repeat(5, axis=0)
offsetsx = np.array([-Nx + 1, -1, 0, 1, Nx - 1])
offsetsy = np.array([-Ny + 1, -1, 0, 1, Ny - 1])
neigh_x = sp.dia_matrix((datax, offsetsx), shape=(Nx, Nx))
neigh_y = sp.dia_matrix((datay, offsetsy), shape=(Ny, Ny))
neighbours = sp.kron(neigh_y, neigh_x)

# Compute particle density
rho = np.pi * R**2 * N / (l * L)
print(f"rho={rho}")

# Initialize the fields
r = np.random.uniform([0, 0], [l, L], size=(N, 2))
theta = 2 * np.pi * np.random.random(size=N)

# Physical parameters
f_0 = 1
k = 5
kc = 4 / R
D = 0.0005
Dr = 1
h = 2

gamma = np.array([[np.sqrt(h), 0], [0, 1 / np.sqrt(h)]])

# Plotting parameters
draw_modulo = 5
## Coarse-graining
cgl = R * 2.6  # coarse-graining length
# Cells lists number
Nx_cg = int(l / cgl)
Ny_cg = int(L / cgl)
# Cells lists width and height
wx_cg = l / Nx_cg
wy_cg = L / Ny_cg

plt.figure(figsize=(7, 7))
cd = r[:, 1] / L


def compute_radial_correlation(r):
    r_max = 0.3
    N_r = 150  # number of steps between 0 and r_max
    dr = r_max / N_r
    bins = np.linspace(0, r_max, N_r + 1)
    r_arr = bins[1:] - bins[0]
    g = np.zeros(N_r)
    N_choice = 500
    for ref_idx in np.random.randint(0, r.shape[0], size=N_choice):
        r_diff = np.delete(r - r[ref_idx, :], ref_idx, 0)
        # Apply periodic BC
        r_diff = np.where(
            abs(r_diff) > [l / 2, np.infty],
            -np.sign(r_diff) * (l - abs(r_diff)),
            r_diff,
        )
        r_diff = np.where(
            abs(r_diff) > [np.infty, L / 2],
            -np.sign(r_diff) * (L - abs(r_diff)),
            r_diff,
        )
        norm = np.linalg.norm(r_diff, axis=1)
        count, bins = np.histogram(norm, bins=bins)
        g += count / (2 * np.pi * r_arr * dr * N * rho)
    g = g / N_choice
    return g, r_arr


def compute_pair_correlation(r, theta):
    r_max = 0.3
    N_x, N_y = 150, 150  # number of steps between 0 and r_max
    bins_x = np.linspace(-r_max, r_max, N_x + 1)
    bins_y = np.linspace(-r_max, r_max, N_y + 1)
    g = np.zeros((N_x, N_y))
    N_choice = 1000
    for ref_idx in np.random.randint(0, r.shape[0], size=N_choice):
        r_diff = np.delete(r - r[ref_idx, :], ref_idx, 0)
        # Apply periodic BC
        r_diff = np.where(
            abs(r_diff) > [l / 2, np.infty],
            -np.sign(r_diff) * (l - abs(r_diff)),
            r_diff,
        )
        r_diff = np.where(
            abs(r_diff) > [np.infty, L / 2],
            -np.sign(r_diff) * (L - abs(r_diff)),
            r_diff,
        )
        # Rotate frame to have particle orientation set to 0
        theta_0 = theta[ref_idx]
        c, s = np.cos(theta_0), np.sin(theta_0)
        R = np.array(((c, s), (-s, c)))
        r_diff = r_diff @ R.T
        # Count occurences in each bin
        count, binsx, binsy = np.histogram2d(
            r_diff[:, 0], r_diff[:, 1], bins=[bins_x, bins_y]
        )
        g += count
    g = g / N_choice
    return g


def compute_pair_correlation_anisotropic(r, theta):
    r_max = 0.3
    N_x, N_y, N_th = 150, 150, 20  # number of steps between 0 and r_max
    bins_x = np.linspace(-r_max, r_max, N_x + 1)
    bins_y = np.linspace(-r_max, r_max, N_y + 1)
    bins_th = np.linspace(0, 2 * np.pi, N_th)
    g = np.zeros((N_x, N_y, N_th))
    N_choice = 1000
    for ref_idx in np.random.randint(0, r.shape[0], size=N_choice):
        r_diff = np.delete(r - r[ref_idx, :], ref_idx, 0)
        # Apply periodic BC
        r_diff = np.where(
            abs(r_diff) > [l / 2, np.infty],
            -np.sign(r_diff) * (l - abs(r_diff)),
            r_diff,
        )
        r_diff = np.where(
            abs(r_diff) > [np.infty, L / 2],
            -np.sign(r_diff) * (L - abs(r_diff)),
            r_diff,
        )
        # Rotate frame to have particle orientation set to 0
        theta_0 = (theta[ref_idx]) % (2 * np.pi)
        idx = np.abs(bins_th - theta_0).argmin()
        c, s = np.cos(theta_0), np.sin(theta_0)
        R = np.array(((c, s), (-s, c)))
        r_diff = r_diff @ R.T
        # Count occurences in each bin
        count, binsx, binsy = np.histogram2d(
            r_diff[:, 0], r_diff[:, 1], bins=[bins_x, bins_y]
        )
        g[:, :, idx] += count
    g = g / N_choice
    return g


# Simulation parameters
dt = 0.0005
Nt = 5000

g = None

correlation_start = 1000
correlation_modulo = 200

for t in range(Nt):
    Cij = (r // np.array([wx, wy])).astype(int)
    C1d = Cij[:, 0] + Nx * Cij[:, 1]
    # 1D array encoding the index of the cell containing the particle
    C = sp.eye(Nx * Ny, format="csr")[C1d]
    # One-hot encoding of the 1D cell array as a sparse matrix

    inRange = C.dot(neighbours).dot(C.T)
    # N x N array; inRange[i,j]=1 if particles i, j are in neighbouring cells, 0 otherwise

    x_ = inRange.multiply(r[:, 0])
    y_ = inRange.multiply(r[:, 1])
    # Compute direction vectors and apply periodic boundary conditions
    xij = x_ - x_.T
    x_bound = (xij.data > l / 2).astype(int)
    xij.data += l * (x_bound.T - x_bound)
    yij = y_ - y_.T
    y_bound = (yij.data > L / 2).astype(int)
    yij.data += L * (y_bound.T - y_bound)

    dij = (xij.power(2) + yij.power(2)).power(0.5)
    # dij is particle-particle distance for interacting particles

    xij.data /= dij.data
    yij.data /= dij.data
    dij.data -= 2 * R
    dij.data = np.where(dij.data < 0, dij.data, 0)
    dij.eliminate_zeros()
    Fij = -kc * dij
    Fx = np.array(Fij.multiply(xij).sum(axis=0)).flatten()
    Fy = np.array(Fij.multiply(yij).sum(axis=0)).flatten()

    # Constant velocity
    f = np.stack([f_0 * np.cos(theta) + Fx, f_0 * np.sin(theta) + Fy], axis=-1)
    v = f @ gamma
    e_perp = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
    phi = np.arctan2(v[:, 1], v[:, 0])
    theta += k * np.einsum("ij, ij->i", f, e_perp) * dt + np.sqrt(
        2 * Dr * dt
    ) * np.random.randn(N)
    # theta += -k * f_0 * (np.sqrt(h) - 1 / np.sqrt(h)) / 2 * np.sin(
    #     2 * theta
    # ) * dt + np.sqrt(2 * Dr * dt) * np.random.randn(N)
    # theta += k * (phi - theta) * dt + np.sqrt(2 * Dr * dt) * np.random.randn(N)
    r += dt * v + np.sqrt(2 * D * dt) * np.random.randn(N, 2)
    r %= np.array([l, L])

    if t % draw_modulo == 0:
        plt.clf()
        plt.scatter(r[:, 0], r[:, 1], s=10, c=v[:, 0], vmin=-f_0, vmax=f_0)
        plt.show()
        plt.pause(0.01)

    if t > correlation_start and t % correlation_modulo == 0:
        if g is None:
            g = compute_pair_correlation_anisotropic(r, theta)
            continue
        g += compute_pair_correlation_anisotropic(r, theta)


# g, r_arr = compute_radial_correlation(r)
# plt.figure()
# plt.plot(r_arr, g)
# plt.show()
# plt.pause(2)

# plt.figure(figsize=(8, 8))
# plt.pcolormesh(g[:, :, 0])
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.pcolormesh(g[:, :, 10])
# plt.show()
# plt.pause(10)

np.save(f"pair_correlation_v={f_0}_h={h}_N={N}.npy", g)

import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import join
from scipy.special import iv
from scipy.integrate import trapezoid
from scipy.spatial import KDTree
from scipy.stats import binned_statistic_dd
from itertools import product


sim_path = r"C:\Users\nolan\Documents\PhD\Simulations\Test_gradient_descent"

Nth = 100
theta_bins = np.linspace(0, 2 * np.pi, Nth + 1)
theta = (theta_bins[:-1] + theta_bins[1:]) / 2

n_modes = (10, 5)

with h5py.File(join(sim_path, "data.h5py")) as hdf_file:
    r = hdf_file["simulation_data"]["r"][-1]
    theta_sim = hdf_file["simulation_data"]["theta"][-1]
    pcf = hdf_file["pair_correlation"]["pcf"][()]
    l = hdf_file.attrs["l"]
    L = hdf_file.attrs["L"]
    N = hdf_file.attrs["N"]
    kc = hdf_file.attrs["kc"]
    # compute fourier coeffs integrals
    fourier = hdf_file["fourier_coefficients"]
    radial_coord = hdf_file["pair_correlation"]["r"][()].flatten()
    d_array = np.zeros(n_modes + (len(radial_coord),))
    f_array = np.zeros(n_modes + (len(radial_coord),))
    for (i, m), (j, n) in product(
        enumerate(range(n_modes[0])), enumerate(range(n_modes[1]))
    ):
        d_array[i, j, :] = fourier[f"1{m}{n}"][:, 3]
        f_array[i, j, :] = fourier[f"1{m}{n}"][:, 5]

max_r_idx = np.nonzero(radial_coord < 2)[0][-1]
rad = radial_coord[: max_r_idx + 1]
alpha = np.pi * trapezoid(
    rad * kc * (2 - rad) * d_array[:, :, : max_r_idx + 1], rad, axis=-1
)
beta = np.pi * trapezoid(
    rad * kc * (2 - rad) * f_array[:, :, : max_r_idx + 1], rad, axis=-1
)
coeffs = np.zeros(n_modes + (2,))
coeffs[:, :, 0] = (alpha + beta) / (
    2 * np.arange(0, n_modes[1])[None, :] + np.arange(0, n_modes[0])[:, None]
)
coeffs[:, :, 1] = (alpha - beta) / (
    2 * np.arange(0, n_modes[1])[None, :] - np.arange(0, n_modes[0])[:, None]
)
coeffs = np.nan_to_num(coeffs, posinf=0, neginf=0)

cos = np.array(
    [
        [
            [np.cos((2 * n + m) * theta), np.cos((2 * n - m) * theta)]
            for n in range(n_modes[1])
        ]
        for m in range(n_modes[0])
    ]
)

r = np.stack([r.real, r.imag], axis=-1)

Nx = int(l / 10)
Ny = int(L / 10)
x_bins = np.linspace(0, l, Nx + 1)
y_bins = np.linspace(0, L, Ny + 1)
dx, dy, dth = l / Nx, L / Ny, 2 * np.pi / Nth

data = np.stack([r[:, 0], r[:, 1], theta_sim], axis=-1)
psi_0 = binned_statistic_dd(
    data, 0, bins=(x_bins, y_bins, theta_bins), statistic="count"
).statistic
psi_0 /= N * dx * dy * dth

px_avg = (np.cos(theta[None, :]) * psi_0).sum(axis=(0, 2)) * dx * dth
psi_0 = np.where((px_avg < 0)[None, :, None], np.roll(psi_0, Nth // 2, axis=2), psi_0)

psi_star = psi_0.sum(axis=(0, 1)) * dx * dy
psi_star /= trapezoid(psi_star, theta)


# number of (m,n) modes
p_star = np.array(
    [(np.cos(k * theta) * psi_star).sum() * dth for k in range(n_modes[0])]
)


# Plot distributions
def plot_distributions(H, K):
    plt.polar(theta, psi_star)
    plt.polar(theta, compute_psi(theta, H, K))


def compute_psi(theta, H, K):
    psi = np.exp(
        H / 2 * np.cos(2 * theta) + K * np.einsum("m,mnk,mnkj", p_star, coeffs, cos)
    )
    psi /= trapezoid(psi, theta)
    return psi


# def compute_gradient(H, K):
#     psi = compute_psi(theta, H, K)
#     cos_avg = trapezoid(cos * psi, theta)
#     cos2_avg = trapezoid(np.cos(2 * theta) * psi, theta)
#     grad_H = 0.5 * trapezoid((np.cos(2 * theta) - cos2_avg) * psi * (psi - psi_star))
#     grad_K = trapezoid(
#         np.einsum("m,mnk,mnkj", p_star, coeffs, cos - cos_avg[..., None])
#         * psi
#         * (psi - psi_star)
#     )
#     return grad_H, grad_K


def compute_gradient(H, K):
    psi = compute_psi(theta, H, K)
    e = np.cos(np.outer(np.arange(n_modes[0]), theta))
    p = trapezoid(e * psi, theta)
    cos2_avg = trapezoid(np.cos(2 * theta) * psi, theta)
    cos_avg = trapezoid(cos * psi, theta)

    grad_H = 0.5 * np.dot(
        trapezoid(e * (np.cos(2 * theta) - cos2_avg) * psi, theta), p - p_star
    )

    grad_K = np.dot(
        trapezoid(
            np.einsum(
                "m,mnk,ij,mnkj->ij", p_star, coeffs, e, (cos - cos_avg[..., None])
            )
            * psi,
            theta,
        ),
        p - p_star,
    )
    return grad_H, grad_K


H = 3.0  # initial guess
K = 5

lr = 0.1
tol = 1e-6
max_iter = 10000

for i in range(max_iter):
    grad_H, grad_K = compute_gradient(H, K)
    new_H = H - lr * grad_H
    new_K = K - lr * grad_K
    if abs(new_H - H) < tol and abs(new_K - K) < tol:
        H = new_H
        K = new_K
        print(f"Converged in {i} iterations")
        break
    H = new_H
    K = new_K

plot_distributions(H, K)

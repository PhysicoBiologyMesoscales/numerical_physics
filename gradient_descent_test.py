import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
from os.path import join
from scipy.special import iv
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.spatial import KDTree
from scipy.stats import binned_statistic_dd, norm
from itertools import product


sim_path = r"E:\Local_Sims\test_new_code"

matplotlib.use("TkAgg")

with h5py.File(join(sim_path, "data.h5py")) as hdf_file:
    kc = hdf_file.attrs["kc"]
    rho = hdf_file.attrs["N"] / hdf_file.attrs["l"] / hdf_file.attrs["L"]
    corr = hdf_file["pair_correlation"]
    Nth = corr.attrs["Nth"]
    r = corr["r"][()]
    phi = corr["phi"][()]
    theta = corr["theta"][()]
    dth = corr.attrs["dth"]
    dphi = corr.attrs["dphi"]
    pcf = corr["pcf"][()]
    p_th = corr["p_th"][1:].mean(axis=0)

i_indices = np.arange(Nth).reshape(-1, 1)  # Column vector for row indices
j_indices = np.arange(Nth)
pcf_th = pcf[:, :, i_indices, (j_indices - i_indices) % Nth]

F = -kc * (2 - r)

integrand = (
    np.einsum(
        "ijkl,i,j,k,l->ijkl", pcf_th, -r * F, np.exp(1j * phi), np.exp(1j * theta), p_th
    )
    * rho
)
F_stat = trapezoid(
    trapezoid(trapezoid(integrand, theta, axis=-1), phi, axis=1), r, axis=0
)

C = -(F_stat * np.exp(-1j * theta)).imag

# f = (
#     cumulative_trapezoid(C, theta, initial=0)
#     + cumulative_trapezoid(C[::-1], theta[::-1], initial=0)[::-1]
# ) / 2
f = (np.cumsum(C) - np.cumsum(C[::-1])[::-1]) / 2 * dth


# Plot distributions
def plot_distributions(H, K):
    plt.polar(theta, p_th)
    plt.polar(theta, compute_psi(theta, H, K))


def grad_psi_f():
    integrand = np.einsum("ijkl,i,j->ijkl", pcf_th, r * F, np.sin(phi)) * rho
    integral = trapezoid(integrand.sum(axis=1) * dphi, r, axis=0)
    grad = np.cumsum(integral, axis=0) * dth
    return grad


# TODO fix this !!
def gradient_descent_psi(H, K):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    psi = np.roll(norm.pdf(np.linspace(-np.pi, np.pi, Nth), scale=np.pi / 4), Nth // 2)
    lr = 0.1
    # mu = 10.0
    tol = 1e-6
    max_iter = 1000
    grad_f = grad_psi_f()
    for i in range(max_iter):
        ax.cla()
        psi_star = np.exp(H / 2 * np.cos(2 * theta) + K * compute_f(theta, psi))
        psi_star /= psi_star.sum() * dth
        grad_f_avg = (psi_star[:, np.newaxis] * grad_f).sum(axis=0, keepdims=True) * dth
        grad_L = (
            (psi - psi_star) / psi
            - K
            * ((grad_f - grad_f_avg) * (psi - psi_star)[:, np.newaxis]).sum(axis=0)
            * dth
            # + mu * np.sign(psi.sum() * dth - 1)
        )
        new_psi = psi - lr * grad_L
        new_psi /= new_psi.sum() * dth
        ax.plot(theta, new_psi)
        fig.show()
        if i == 0:
            plt.pause(1)
        plt.pause(0.01)
        if np.linalg.norm(psi - new_psi) < tol:
            print(f"Gradient descent converged in {i} iterations")
            return psi
        psi = new_psi
        # break
    return psi


def compute_psi(theta, H, K):

    psi = np.exp(H / 2 * np.cos(2 * theta) + K * f)
    psi /= psi.sum() * dth
    return psi


def compute_f(theta, psi):
    integrand = -(
        np.einsum(
            "ijkl,i,j,l->ijkl",
            pcf_th,
            r * F,
            np.sin(phi),
            psi,
        )
        * rho
    )
    C = trapezoid(integrand.sum(axis=(1, 3)) * dth * dphi, r, axis=0)

    f = (
        -(
            cumulative_trapezoid(C, theta, initial=0)
            + cumulative_trapezoid(C[::-1], theta[::-1], initial=0)[::-1]
        )
        / 2
    )
    return f


def grad_f(theta, psi):
    return


def compute_gradient(H, K):
    psi = compute_psi(theta, H, K)
    cos2_avg = (np.cos(2 * theta) * psi).sum() * dth
    f_avg = (f * psi).sum() * dth

    grad_H = (
        0.5
        * ((np.cos(2 * theta) - cos2_avg) * (np.log(psi) - np.log(p_th))).sum()
        * dth
    )
    grad_K = ((f - f_avg) * (np.log(psi) - np.log(p_th))).sum() * dth

    return grad_H, grad_K


# H = 1.0  # initial guess
# K = 5

# lr = 0.1
# tol = 1e-6
# max_iter = 10000

# for i in range(max_iter):
#     grad_H, grad_K = compute_gradient(H, K)
#     new_H = H - lr * grad_H
#     new_K = K - lr * grad_K
#     if abs(new_H - H) < tol and abs(new_K - K) < tol:
#         H = new_H
#         K = new_K
#         print(f"Converged in {i} iterations")
#         break
#     H = new_H
#     K = new_K

# print(f"H={H}, K={K}")
# plot_distributions(H, K)

psi = gradient_descent_psi(3.0, 10.0)
plt.polar(theta, psi)

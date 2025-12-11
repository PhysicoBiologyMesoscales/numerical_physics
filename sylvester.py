import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester
from scipy.special import jv, struve
from itertools import product
from tqdm import tqdm

s = 20
N = 2 * s + 1


def Vexp(k):
    return 2 * np.pi * np.exp(-0.5 * k**2)


def Vexp_r(r):
    return np.exp(-0.5 * r**2)


def dVexp_r(r):
    return -r * np.exp(-0.5 * r**2)


def Vharm(k):
    _v = (
        2
        * np.pi
        * (
            1
            / k**2
            * (
                -2 * jv(2, k)
                + np.pi * (jv(1, k) * struve(0, k) - jv(0, k) * struve(1, k))
            )
        )
    ) / 2
    return np.where(k == 0, 2 * np.pi / 24, _v)


def Vharm_r(r):
    return np.where(r <= 1, 0.5 * (1 - r) ** 2, 0)


lp = 1.0
phi = 0.5
eps = 1e-2


def L(k, lp, phi, eps, V):
    L = np.diag((np.arange(N) - s) ** 2) / lp - 1j * k / 2 * (
        np.eye(N, k=1) + np.eye(N, k=-1)
    )
    L[s, s] += 2 * phi / eps * V(k) * k**2
    return L


def RHS(k, eps, V):
    mat = np.zeros((N, N))
    mat[s, s] = -2 * k**2 * V(k) / eps
    return mat


def compute_g(lp, phi, eps, V, k_arr):
    Npoints = len(k_arr)
    g0 = np.zeros(Npoints, dtype=np.complex128)
    g1 = np.zeros(Npoints, dtype=np.complex128)

    X = np.zeros((N, N, Npoints), dtype=np.complex128)

    for i, k in enumerate(k_arr):
        X[..., i] = solve_sylvester(
            L(k, lp, phi, eps, V),
            L(k, lp, phi, eps, V).conj().T,
            RHS(k, eps, V),
        )
        g0[i] = X[s, s, i]
        g1[i] = X[s, s + 1, i]

    return X, g0, g1


def convert_to_r(g0, g1, k_arr, r_arr):
    kr = k_arr[:, None] * r_arr[None, :]
    g0_r = np.real(
        np.trapz(k_arr[:, None] * jv(0, kr) * g0[:, None], k_arr, axis=0) / 2 / np.pi
    )
    g1_r = -np.real(
        1j
        * np.trapz(k_arr[:, None] * jv(1, kr) * g1[:, None], k_arr, axis=0)
        / 2
        / np.pi
    )
    return g0_r, g1_r


def compute_P(lp, phi, eps, r_arr, dV, g0, g1):
    P = (
        phi
        / np.pi
        * (
            lp / 2
            - phi / 2 / eps * np.trapz(r_arr**2 * dV(r_arr) * (1 + g0), r_arr)
            + phi / eps * lp * np.trapz(r_arr * dV(r_arr) * g1, r_arr)
        )
    )
    return P


if __name__ == "__main__":
    eps = 1e-2
    Npoints_k = 100
    Npoints_r = 100
    k_arr = np.linspace(0, 10, Npoints_k)
    r_arr = np.linspace(0, 10, Npoints_r)
    N_lp = 10
    phi = 0.5
    lp_arr = np.linspace(10, 200, N_lp)
    list_P = np.zeros((3, N_lp))
    for i, lp in enumerate(tqdm(lp_arr)):
        _X, g0, g1 = compute_g(lp, phi, eps, Vexp, k_arr)
        g0_r, g1_r = convert_to_r(g0, g1, k_arr, r_arr)
        list_P[0, i] = phi * lp / 2
        list_P[1, i] = (
            -(phi**2)
            / 2
            / eps
            * np.trapz(r_arr**2 * dVexp_r(r_arr) * (1 + g0_r), r_arr)
        )
        list_P[2, i] = (
            phi**2 / eps * lp * np.trapz(r_arr * dVexp_r(r_arr) * g1_r, r_arr)
        )

plt.plot(lp_arr, list_P.T)

# ll, pp = np.meshgrid(lp_arr, phi_arr)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(ll, pp, list_P, linewidth=0, antialiased=False)

# X_r = np.trapz(
#     k_arr[None, None, :, None]
#     * jv(0, kr)[None, None, ...]
#     * (X[..., None] - np.eye(N)[..., None, None]),
#     k_arr,
#     axis=2,
# )


# g0_th = 1 / (
#     (1 + 4 * phi / lp / eps * V(k_arr)) * (1 + 2 * phi / eps * V(k_arr) * k_arr**2)
# )
# g1_th = (
#     1j
#     * lp
#     * k_arr
#     / 2
#     * 4
#     / eps
#     / lp
#     * V(k_arr)
#     / ((1 + 4 * phi / lp / eps * V(k_arr)) * (1 + 2 * phi / eps * V(k_arr) * k_arr**2))
# )
# g0_r_th = (
#     np.pi
#     / phi
#     * np.trapz(
#         k_arr[:, None] * jv(0, kr) * np.pi / phi * (g0_th[:, None] - 1), k_arr, axis=0
#     )
# )
# g1_r_th = (
#     np.pi / phi * np.trapz(k_arr[:, None] * jv(1, kr) * g1_th[:, None], k_arr, axis=0)
# )

# plt.plot(r_arr, 1j * g1_r)
# plt.plot(r_arr, 1j * g1_r_th)

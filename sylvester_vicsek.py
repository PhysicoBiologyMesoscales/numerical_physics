import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester
from scipy.special import jv, struve
from itertools import product
from tqdm import tqdm
from matplotlib.colors import CenteredNorm
from scipy.ndimage import shift

s = 10
N = 2 * s + 1


def G(k, r0=2):
    return 2 * np.where(k == 0, 0.5, jv(1, k * r0) / (k * r0))


def G_r(r, r0=2):
    return np.where(r <= r0, 1.0, 0.0) / (np.pi * r0**2)


def L(k, lp, phi, gamma, r0=2):
    L = np.diag((np.arange(N) - s) ** 2) + 1j * lp * k / 2 * (
        np.eye(N, k=1) + np.eye(N, k=-1)
    )
    L[s + 1, s + 1] += -phi * gamma / 2 * G(k, r0)
    L[s - 1, s - 1] += -phi * gamma / 2 * G(k, r0)
    return L


def RHS(k, lp, gamma, r0=2):
    mat = np.zeros((N, N), dtype=np.complex128)
    mat[s + 1, s + 1] = gamma * G(k, r0)
    mat[s - 1, s - 1] = gamma * G(k, r0)
    return mat


def RHS_deriv(k, lp, gamma, phi, h, r0=2):
    dL_h = (
        gamma
        * phi
        / 2
        * (
            (np.arange(N) - s)[:, None] * shift(h, [1, 0], cval=0)
            - (np.arange(N) - s)[None, :] * shift(h, [0, -1], cval=0)
        )
    )
    dL_h[s + 2, :] -= gamma * phi * G(k, r0) * h[s + 1, :]
    dL_h[:, s - 2] -= gamma * phi * G(k, r0) * h[:, s - 1]
    mat = np.zeros((N, N), dtype=np.complex128)
    mat[s + 2, s + 1] = gamma * 3.0 / 2.0 * G(k, r0)
    mat[s - 1, s - 2] = gamma * 3.0 / 2.0 * G(k, r0)
    mat[s + 1, s] = gamma / 2.0 * G(k, r0)
    mat[s, s - 1] = gamma / 2.0 * G(k, r0)
    mat += dL_h
    return mat


def compute_h(lp, phi, gamma, k, r0=2):
    h = solve_sylvester(
        L(k, lp, phi, gamma, r0),
        L(k, lp, phi, gamma, r0).conj().T,
        RHS(k, lp, gamma, r0),
    )
    return h


def compute_h_array(lp, phi, gamma, k_arr, r0=2):
    Npoints = len(k_arr)

    h = np.zeros((N, N, Npoints), dtype=np.complex128)
    X = np.zeros((N, N, Npoints), dtype=np.complex128)

    for i, k in enumerate(k_arr):
        _h = compute_h(lp, phi, gamma, k, r0)
        h[..., i] = _h
        X[..., i] = solve_sylvester(
            L(k, lp, phi, gamma, r0),
            L(k, lp, phi, gamma, r0).conj().T,
            RHS_deriv(k, lp, gamma, phi, _h, r0),
        )

    return h, X


def convert_to_r(g0, g2, k_arr, r_arr):
    kr = k_arr[:, None] * r_arr[None, :]
    g0_r = np.real(
        np.trapz(k_arr[:, None] * jv(0, kr) * g0[:, None], k_arr, axis=0) / 2 / np.pi
    )
    g2_r = np.real(
        np.trapz(k_arr[:, None] * jv(0, kr) * g2[:, None], k_arr, axis=0) / 2 / np.pi
    )
    return g0_r, g2_r


def convert_X_to_r(X, k_arr, r_arr):
    kr = k_arr[:, None] * r_arr[None, :]
    X_r = (
        np.trapz(
            k_arr[None, None, :, None] * jv(0, kr)[None, None, ...] * X[..., None],
            k_arr,
            axis=2,
        )
        / 2
        / np.pi
    )
    return X_r


def compute_sigma(gamma, phi, X, k_arr, r0=2):
    integrand = k_arr * G(k_arr, r0) / (2 * np.pi) * (X[s + 1, s] - X[s + 2, s + 1])
    sigma = gamma * phi / 2 * (1 + np.real(np.trapz(integrand, k_arr))) - 1.0
    return sigma


def k_grid(k0, k_max, N_k0, N_k):
    k_arr = np.concatenate(
        [
            np.linspace(0, k0, N_k0, endpoint=False),
            np.linspace(k0, k_max, N_k - N_k0),
        ]
    )
    return k_arr


def draw_B(B, r_arr, alpha):
    from mpl_toolkits.mplot3d import Axes3D

    ax = Axes3D(plt.figure())
    r, th = np.meshgrid(r_arr, alpha)
    plt.subplot(projection="polar")
    plt.pcolormesh(th, r, B.T)
    plt.show()


if __name__ == "__main__":
    r0 = 2.0
    N_phi = 20
    N_gamma = 20
    phi_arr = np.linspace(0.1, 1, N_phi)
    gamma = 3.8
    phi = 0.5
    lp = 10.0
    k_arr = k_grid(0.1, 20, 50, 200)
    r_arr = np.linspace(0, max(2 * r0, 4 * lp), 100)
    gamma_lim = np.zeros(N_phi)
    sigma_arr = np.zeros((N_phi, N_gamma))
    sigma_MF_arr = np.zeros((N_phi, N_gamma))
    full_gamma_arr = np.zeros((N_phi, N_gamma))
    tol = 1e-3
    Nmax = 100
    gamma_arr = np.linspace(0, 10, N_gamma)
    for i, phi in enumerate(phi_arr):
        # gamma_MF = 2 / phi
        # gamma_arr = np.linspace(0, gamma_MF, N_gamma)
        # full_gamma_arr[i, :] = gamma_arr
        # old_gamma = 0.0
        for j, gamma in enumerate(gamma_arr):
            h, X = compute_h_array(lp, phi, gamma, k_arr, r0)
            sigma = compute_sigma(gamma, phi, X, k_arr, r0)
            sigma_arr[i, j] = sigma
            sigma_MF_arr[i, j] = 1 - gamma * phi / 2
            # if sigma > 0:
            #     gamma_lim[i] = old_gamma
            #     break
            # old_gamma = gamma

    # plt.plot(phi_arr, gamma_lim)
# # Xr = convert_X_to_r(X, k_arr, r_arr)
# plt.plot(r_arr, Xr[s, s])
# plt.plot(r_arr, Xr[s+2, s+2])
# print(f"Computed sigma: {sigma}")

#     plt.imshow(
#         list_P.T,
#         extent=(phi_min, phi_max, lp_min, lp_max),
#         aspect="auto",
#         origin="lower",
#         cmap="seismic",
#         norm=CenteredNorm(0),
#     )
#     plt.colorbar()
#     plt.xlabel(r"$\Phi$")
#     plt.ylabel(r"$l_p$")
# ll, pp = np.meshgrid(lp_arr, phi_arr)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(ll, pp, list_P, linewidth=0, antialiased=False)

# plt.plot(lp_arr, list_P.T)

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

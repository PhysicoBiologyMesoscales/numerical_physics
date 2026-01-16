import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester
from scipy.special import jv, struve
from joblib import Parallel, delayed
from itertools import product
from tqdm import tqdm
from matplotlib.colors import CenteredNorm

from struct_aux import dot, L, Vexp, Vexp_r, dVexp_r
from struct_3b import compute_3bod

s = 10
N = 2 * s + 1

# RHS for sylvester solve


def RHS_S(k, lp, phi, eps, V):
    from scipy.linalg import eig, inv
    
    k2abs_arr = np.linspace(0, 10, 200)
    alpha = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    k2_arr = k2abs_arr[:, None] * np.exp(1j * alpha[None, :])
    S3arr = np.zeros((len(k2abs_arr), len(alpha), N, N), dtype=np.complex128)
    
    # Pre-compute L1-related values that are constant across the loop
    from struct_aux import L
    from struct_3b import compute_S
    L1 = L(k, lp, phi, eps, V)
    S1 = compute_S(L1, lp, s=s)
    l1, P1 = eig(L1)
    Q1 = inv(P1)
    
    for i, k2 in enumerate(k2_arr.flatten()):
        S3 = compute_3bod(k, k2, lp, phi, eps, V, s=s, L1=L1, S1=S1, l1=l1, P1=P1, Q1=Q1)
        S3arr[i // len(alpha), i % len(alpha), :, :] = S3[:, s, :]
    return np.diag(2 / lp * (np.arange(N) - s) ** 2) + np.trapz(
        np.trapz(S3arr, k2abs_arr, axis=0), alpha, axis=0
    )


def RHS_h(k, eps, V):
    mat = np.zeros((N, N))
    mat[s, s] = -2 * k**2 * V(k) / eps
    return mat


def compute_correlations(lp, phi, eps, V, k_arr, which="h", parallel="False"):
    """Computes correlation matrices for all values in k_arr"""
    match which:
        case "h":
            RHS = lambda k: RHS_h(k, eps, V)
        case "S":
            RHS = lambda _x: RHS_S(lp)

    if parallel:
        # Works best for large k_arr
        def solve_one(k):
            A = L(k, lp, phi, eps, V)
            return solve_sylvester(A, A.conj().T, RHS(k))

        Xs = Parallel(n_jobs=-1, prefer="processes")(
            delayed(solve_one)(k) for k in k_arr
        )
        X = np.stack(Xs, axis=-1)

        return X

    Npoints = len(k_arr)
    X = np.zeros((N, N, Npoints), dtype=np.complex128)

    for i, k in enumerate(k_arr):
        A = L(k, lp, phi, eps, V)
        X[..., i] = solve_sylvester(
            A,
            A.conj().T,
            RHS(k),
        )

    return X


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


def compute_B(phi, eps, lp, alpha, r_arr, V=Vexp, Npoints_k=100, kmax=10):
    """Computes correlation function from the reference frame of the 1st particle"""
    # Compute full correlation matrix
    k_arr = np.linspace(0, kmax, Npoints_k)
    G = compute_correlations(lp, phi, eps, V, k_arr, which="g")

    # Bessel functions for radial Fourier transform
    kr = k_arr[:, None] * r_arr[None, :]
    jint = np.stack([1j**n * jv(n, kr) for n in range(s + 1)], axis=0)

    # Compute correlation functions in real space
    C = G[s:, s]
    gn = np.real(np.trapz(C[..., None] * jint, k_arr, axis=1) / 2 / np.pi)

    # Resum the Fourier series

    n = np.arange(1, s + 1)
    B = gn[0, :, None] + 2 * np.sum(
        gn[1:, :, None] * np.cos(n[:, None, None] * alpha[None, None, :]), axis=0
    )

    return B


def draw_B(B, r_arr, alpha):
    from mpl_toolkits.mplot3d import Axes3D

    ax = Axes3D(plt.figure())
    r, th = np.meshgrid(r_arr, alpha)
    plt.subplot(projection="polar")
    plt.pcolormesh(th, r, B.T)
    plt.show()


# if __name__ == "__main__":
#     eps = 1e-2
#     Npoints_k = 200
#     Npoints_r = 100
#     k_arr = np.linspace(0, 20, Npoints_k)
#     r_arr = np.linspace(0, 4, Npoints_r)
#     N_lp = 10
#     N_phi = 10
#     phi_min = 0.0
#     phi_max = 1.0
#     lp_min = 0.01
#     lp_max = 200.0
#     lp_arr = np.linspace(lp_min, lp_max, N_lp)
#     phi_arr = np.linspace(phi_min, phi_max, N_phi)
#     list_P = np.zeros((N_phi, N_lp))
#     for i, (phi, lp) in enumerate(tqdm(product(phi_arr, lp_arr))):
#         _X, g0, g1 = compute_g(lp, phi, eps, Vexp, k_arr)
#         g0_r, g1_r = convert_to_r(g0, g1, k_arr, r_arr)
#         row, col = np.unravel_index(i, (N_phi, N_lp))
#         list_P[row, col] = compute_P(lp, phi, eps, r_arr, dVexp_r, g0_r, g1_r)

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

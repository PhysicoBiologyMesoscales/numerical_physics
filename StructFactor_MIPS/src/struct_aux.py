import numpy as np
from scipy.special import jv, struve
from scipy.optimize import fsolve
from scipy.special import iv, i0, i1
from tqdm import tqdm

## General utilities


def dot(u, v):
    """Scalar product of 2 complex numbers"""
    return np.real(0.5 * (u * np.conjugate(v) + np.conjugate(u) * v))


def L(k, lp, phi, eps, V, s, D=0.0):
    """Lengths in particle diameters (a = 1): rho0 = 4 phi / pi with phi
    the packing fraction  phi = pi rho0 (a/2)^2."""
    N = 2 * s + 1
    L = (
        np.diag((np.arange(N) - s) ** 2)
        + 1j / 2 * lp * (np.conjugate(k) * np.eye(N, k=1) + k * np.eye(N, k=-1))
        + D * np.abs(k) ** 2 * np.eye(N)
    )
    L[s, s] += 4 * phi / np.pi * eps * V(k) * np.abs(k) ** 2
    return L


def L_Vicsek(k, lp, phi, gamma, Kern, qn, s, D=0.0):
    """Lengths in particle diameters (a = 1): rho0 = 4 phi / pi with phi
    the packing fraction  phi = pi rho0 (a/2)^2.  Kern is the interaction Kernel of the particles,
    normalized such that \int d^2r Kern(r) = 1."""
    N = 2 * s + 1
    rho_0 = 4 * phi / np.pi
    L = (
        np.diag((np.arange(N) - s) ** 2)
        + 1j / 2 * lp * (np.conjugate(k) * np.eye(N, k=1) + k * np.eye(N, k=-1))
        + D * np.abs(k) ** 2 * np.eye(N)
    )
    n = np.arange(-s, s + 1)
    q1 = qn[s + 2]
    L += gamma * rho_0 / 2 * q1 * (np.diag(n[:-1], k=1) - np.diag(n[1:], k=-1))
    L[:, s - 1] += gamma * rho_0 / 2 * Kern(k) * n * qn[2:]
    L[:, s + 1] += -gamma * rho_0 / 2 * Kern(k) * n * qn[:-2]
    return L


def L_turning_away(k, lp, phi, eps, gamma, V, Kern, qn, s, D=0.0):
    N = 2 * s + 1
    rho_0 = 4 * phi / np.pi
    L = (
        np.diag((np.arange(N) - s) ** 2)
        + 1j / 2 * lp * (np.conjugate(k) * np.eye(N, k=1) + k * np.eye(N, k=-1))
        + D * np.abs(k) ** 2 * np.eye(N)
    )
    n = np.arange(-s, s + 1)
    # Repulsive interactions
    L[:, s] += eps * rho_0 * V(k) * np.abs(k) ** 2 * qn[1:-1]
    # Turning away torque
    L[:, s] += (
        -1j * gamma * rho_0 / 2 * Kern(k) * n * (qn[:-2] * k - qn[2:] * np.conjugate(k))
    )
    return L


# def L_Vicsek_anisotropic(k, lp, phi, gamma, Kern, s, qn=None, D=0.0):
#     N = 2 * s + 1
#     rho_0 = 4 * phi / np.pi
#     if gamma * rho_0 / 2 < 1:
#         raise ValueError(
#             "The anisotropic Vicsek model is only valid for 2*gamma*phi/pi>1"
#         )
#     if qn is None:
#         # Compute mean-field solution for the angular distribution
#         q1 = fsolve(
#             lambda q: q - iv(1, gamma * rho_0 * q) / iv(0, gamma * rho_0 * q), 1.0
#         )[0]
#         qn = iv(np.arange(-s - 1, s + 2), gamma * rho_0 * q1) / iv(
#             0, gamma * rho_0 * q1
#         )  # Compute one more harmonic, needed in the construction of L
#         if qn[-1] > 1e-3:
#             print(
#                 "Warning: the truncation s={} is too small for the anisotropic Vicsek model, qn[-1]={}".format(
#                     s, qn[-1]
#                 )
#             )
#     # Construct the relaxation operator
#     L = (
#         np.diag((np.arange(N) - s) ** 2)
#         - 1j / 2 * lp * (np.conjugate(k) * np.eye(N, k=1) + k * np.eye(N, k=-1))
#         + D * np.abs(k) ** 2 * np.eye(N)
#     )
#     n = np.arange(-s, s + 1)
#     q1 = qn[s + 2]
#     L += gamma * rho_0 / 2 * q1 * (np.diag(n[:-1], k=1) - np.diag(n[1:], k=-1))
#     L[:, s - 1] += gamma * rho_0 / 2 * Kern(k) * n * qn[2:]
#     L[:, s + 1] += -gamma * rho_0 / 2 * Kern(k) * n * qn[:-2]
#     return L


## Interaction potentials


def Vexp(k, r0=0.5):
    return 2 * np.pi * r0**2 * np.exp(-0.5 * (r0 * np.abs(k)) ** 2)


def Vexp_r(r, r0=0.5):
    return np.exp(-0.5 * (r / r0) ** 2)


def dVexp_r(r, r0=0.5):
    return -r / r0**2 * np.exp(-0.5 * (r / r0) ** 2)


def Vharm(k):
    """2D FT of Vharm_r(r) = (1 - r)^2 / 2 for r < 1 (range = 1 diameter)."""
    _v = (
        np.pi
        / np.abs(k) ** 2
        * (
            -2 * jv(2, np.abs(k))
            + np.pi
            * (
                jv(1, np.abs(k)) * struve(0, np.abs(k))
                - jv(0, np.abs(k)) * struve(1, np.abs(k))
            )
        )
    )

    return np.where(k == 0, np.pi / 12, _v)


def Vharm_r(r):
    return np.where(r <= 1, 0.5 * (1 - r) ** 2, 0)


def dVharm_r(r):
    return np.where(r <= 1, -(1 - r), 0)


## Interaction kernels (Vicsek model)


def Kern_constant(k, r0=1.0):
    """2D FT of Kern(r) = 1/(pi r0^2) for r < r0 (range = r0 diameter)."""
    _v = 2 * jv(1, np.abs(k) * r0) / (np.abs(k) * r0)
    return np.where(k == 0, 1, _v)


def Kern_gauss(k, r0=0.5):
    """2D FT of Kern(r) = 1/(2 pi r0^2) exp(-r^2/(2 r0^2)) (range = r0 diameter)."""
    _v = np.exp(-0.5 * (r0 * np.abs(k)) ** 2)
    return _v


def Kern_erfc(k, r0=0.5):
    """2D FT of Kern(r) = 1/(pi r0^2) * erfc(r/(sqrt(2) r0)) (range = r0 diameter)."""
    _v = (
        r0**2
        / 8
        * np.exp(-1.0 / 8.0 * (r0 * k) ** 2)
        * (i0((r0 * k) ** 2 / 8) - i1((r0 * k) ** 2 / 8))
    )
    return _v


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from itertools import product

    # lp = 10.0
    # gamma = 1.0
    phi = 0.4

    r_max = 10.0  # real-space extent of the correlation maps
    kmax = 5.0  # transform truncation: ringing has period 2pi/kmax, ~|X(kmax)|
    s = 30  # operator truncation: harmonics n = -s..s

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

    gamma_arr = np.linspace(np.pi / 2 / phi + 0.01, 10 * np.pi / 2 / phi, 100)
    gamma = 5.0
    k_arr = np.linspace(0, kmax, 100)
    phi_arr = np.linspace(0.5, 1.0, 10)
    lp_arr = np.linspace(1.0, 20.0, 10)
    min_eig = np.zeros((len(phi_arr), len(lp_arr)))
    min_k = np.zeros((len(phi_arr), len(lp_arr)))
    for (i, phi), (j, lp) in tqdm(
        product(enumerate(phi_arr), enumerate(lp_arr)), total=len(phi_arr) * len(lp_arr)
    ):
        l_k = []
        l_eig = []
        l_vec = []
        has_min = False
        for k in k_arr:
            L = L_Vicsek(k, lp, phi, gamma, Kern_gauss, s)
            eig, vec = np.linalg.eig(L)
            if not np.any(np.real(eig) < -1e-3):
                continue
            if np.any(np.real(eig) < -1e-3):
                has_min = True
                _where = np.argwhere(np.real(eig) < -1e-3)
            if _where.size > 1:
                print(
                    f"Warning: multiple unstable eigenvalues at k={k:.3f}: {np.real(eig[_where])}"
                )
            l_k.append(k)
            l_eig.append(np.real(eig[_where[0, 0]]))
            l_vec.append(vec[:, _where[0, 0]])
        if not has_min:
            print(
                f"Warning: no unstable eigenvalues found for phi={phi:.3f}, lp={lp:.3f}"
            )
            continue

        l_k, l_eig, l_vec = np.array(l_k), np.array(l_eig), np.array(l_vec)
        if j == 3:
            ax[0].plot(
                l_k, l_eig, color=(1 - i / len(phi_arr), 0, 0), label=f"phi={phi:.2f}"
            )
            ax[1].scatter(
                np.arange(15),
                np.real(l_vec[np.argmin(l_eig), s : s + 15]),
                c=(1 - i / len(phi_arr), 0, 0),
            )
            ax[1].plot(
                np.arange(15),
                np.real(l_vec[np.argmin(l_eig), s : s + 15]),
                c=(1 - i / len(phi_arr), 0, 0),
                alpha=0.2,
            )
        k_min = l_k[np.argmin(l_eig)]
        min_k[i, j] = k_min
        min_eig[i, j] = np.min(l_eig)

    ax[0].legend()
    ax[1].set_xlabel("Harmonic n")
    ax[0].set_ylabel("Re(eigenvalue)")
    ax[0].set_xlabel("Wavevector k")
    msh1 = ax2[0].pcolormesh(
        phi_arr, lp_arr, -min_eig.T, shading="auto", cmap="inferno"
    )
    ax2[0].set_ylabel("lp")
    ax2[0].set_xlabel("phi")
    msh2 = ax2[1].pcolormesh(phi_arr, lp_arr, min_k.T, shading="auto", cmap="inferno")
    ax2[1].set_ylabel("lp")
    ax2[1].set_xlabel("phi")
    fig2.colorbar(msh1, ax=ax2[0], label=r"$\sigma_{max}$")
    fig2.colorbar(msh2, ax=ax2[1], label="k_min")


# if __name__ == "__main__":
#     s = 30
#     kmax = 5.0  # transform truncation: ringing has period 2pi/kmax, ~|X(kmax)|
#     gamma = 18
#     lp = 10.0
#     phi = 0.1
#     k_arr = np.linspace(0, kmax, 100)
#     phi_arr = np.linspace(0.0, 1.0, 10)
#     for k in k_arr:
#         L = L_Vicsek_isotropic(k, lp, phi, gamma, Kern_gauss, s)
#         eig, vec = np.linalg.eig(L)
#         if np.any(np.real(eig) < -1e-3):
#             print(
#                 f"Warning: unstable eigenvalues at k={k:.3f}: {np.real(eig[np.argwhere(np.real(eig) < -1e-3)])}"
#             )

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import product
from scipy.fft import fft, ifft, fftfreq, fht, ifht, fhtoffset
from scipy.special import jv
from scipy.integrate import quad, quad_vec
from scipy.signal.windows import hann

# Matrix size
n = 2
N = 2 * n + 1

# Parameters
kc = 5
Pe = 100
l = 100
phi = 0.04


## Define interaction potential
# In real space
def compute_V(x):
    return (
        phi
        / np.pi
        * kc
        / 2
        * (2 - np.abs(x)) ** 2
        * np.heaviside(2 - np.abs(x), 0)
        * np.heaviside(np.abs(x), 1)
    )


# In Fourier space
def compute_U(k):
    # Works for a radial potential only
    dx = 1 / np.max(k)
    r = np.arange(0, 2, dx)
    U = (
        2
        * np.pi
        * np.trapz((r * compute_V(r))[None, ...] * jv(0, np.outer(k, r)), x=r, axis=-1)
    )
    if len(U) == 1:
        return U[0]
    return U


## Compute evolution matrices


def compute_M(w, k):
    M = np.zeros((len(k), N, N)).astype(np.complex128)
    # Rotational diffusion
    M += np.diag(np.arange(-n, n + 1) ** 2)[None, ...]
    # Time derivative and diffusion
    M += np.eye(N)[None, ...] * (1j * w + (l / Pe) ** 2 * np.abs(k) ** 2)[:, None, None]
    # # # Advection
    M += (
        1j
        * l
        / 2
        * np.diag(np.ones(N - 1), 1)[None, ...]
        * np.conjugate(k)[:, None, None]
    )
    M += 1j * l / 2 * np.diag(np.ones(N - 1), -1)[None, ...] * k[:, None, None]
    # Pairwise interaction
    M[:, n, n] += np.abs(k) ** 2 * compute_U(np.abs(k))
    return M


def compute_A(k):
    A = np.zeros((len(k), N, N)).astype(np.complex128)
    A += np.diag(np.arange(-n, n + 1) ** 2)[None, ...]
    A += (
        np.eye(N)[None, ...]
        * ((l / Pe) ** 2 * np.abs(k) ** 2)[
            :,
            None,
            None,
        ]
    )
    A *= phi / np.pi**2
    return A


def compute_S(w, k):
    M = compute_M(w, k)
    A = compute_A(k)
    M_inv = np.linalg.inv(M)
    # S = (M^-1).A.(M^-1)^H
    S = np.einsum(
        "kij, kjl, klm->kim", M_inv, A, np.conjugate(M_inv.transpose(0, 2, 1))
    )
    return S


def integrate_S(k):
    result = np.zeros((N, N)).astype(np.complex128)
    # Loop over each element in the matrix
    for i, j in product(range(N), repeat=2):
        # Define an integrand that extracts the (i, j) element of S(w, k)
        integrand = lambda w: compute_S(w, k)[i, j]
        # Integrate over w from -infty to infty
        result[i, j], _ = quad(integrand, -np.inf, np.inf, complex_func=True)
    return result


## Main

Nk = 64
Nphi = 32
k_length = np.logspace(-1, 1, Nk, base=10)
k_angle = np.linspace(0, 2 * np.pi, Nphi)
k = np.outer(k_length, np.exp(1j * k_angle)).flatten()
S = quad_vec(compute_S, -np.inf, np.inf, args=(k,))[0]


theta = np.exp(1j * np.outer(np.arange(-n, n + 1), k_angle))
corr = np.einsum("knm,ni,mj->kij", S, theta, 1 / theta)
corr_r_th = corr.reshape((Nk, Nphi, Nphi, Nphi)).sum(axis=-1) * 2 * np.pi / Nphi

i_indices = np.arange(Nphi).reshape(-1, 1)  # Column vector for row indices
j_indices = np.arange(Nphi)  # Row vector for column offsets

corr_r_phi = np.real(
    corr_r_th[:, (i_indices + j_indices) % Nphi, i_indices].sum(axis=-1)
    * 2
    * np.pi
    / Nphi
)

kk, pp = np.meshgrid(k_length, k_angle, indexing="ij")


fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
mesh = ax.pcolormesh(pp, kk, np.real(corr_r_phi))
ax.set_ylim((0, 10))

fig.colorbar(mesh)

# TODO Facteur 2 ??
# TODO L'intégration avec quad est très lente... Il vautdrait mieux donner des valeurs explicites pour theta, peut-être en logspace?

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import product
from scipy.fft import fft, ifft, fftfreq, fht, ifht, fhtoffset
from scipy.special import jv
from scipy.integrate import quad, quad_vec

from gauss_legendre_quadrature import legendre_integrate

from matplotlib.widgets import Slider

# Matrix size
n = 2
N = 2 * n + 1

# Parameters
kc = 2
D = 1
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
        * (2 - x) ** 2
        * np.heaviside(2 - x, 0)
        * np.heaviside(x, 1)
    )


# In Fourier space
def compute_U(k, n_points=100):
    # Works for a radial potential only
    integrand = (
        lambda r, k: 2
        * np.pi
        * np.einsum(
            "r, r... -> r...",
            r * compute_V(r),
            jv(0, np.einsum("r,...->r...", r, np.abs(k))),
        )
    )
    U = legendre_integrate(integrand, n_points, args=(k,))
    return U


def compute_passive_pair_correlation(r, n_points=100):
    integrand = (
        lambda k, r: 1
        / 2
        / np.pi
        * (k * (1 / (1 + compute_U(k, n_points=1000)) - 1))[..., None]
        * jv(0, np.outer(k, r))
        * np.heaviside(k, 1)[..., None]
    )
    g = legendre_integrate(integrand, n_points=n_points, args=(r,)) / phi * np.pi + 1
    return g


## Compute evolution matrices


def normalize_w(w, k):
    _w, _k = [
        ar.copy()
        for ar in np.broadcast_arrays(w.flatten()[:, None], k.flatten()[None, :])
    ]
    _w *= l * np.abs(_k)
    return _w, _k


def compute_M(w, k):
    if not isinstance(w, np.ndarray):
        w = np.array([w])
    if not isinstance(k, np.ndarray):
        k = np.array([k])

    match w.shape == k.shape:
        case False:
            _w, _k = np.broadcast_arrays(w.flatten()[:, None], k.flatten()[None, :])
        case _:
            _w, _k = w, k

    M = np.zeros((_w.shape[0], _k.shape[1], N, N)).astype(np.complex128)
    # Rotational diffusion
    M += np.diag(np.arange(-n, n + 1) ** 2)[None, None, ...]
    # Time derivative and diffusion
    M += np.eye(N)[None, None, ...] * (1j * _w + D * np.abs(_k) ** 2)[..., None, None]
    # Advection
    M += (
        1j
        * l
        / 2
        * np.diag(np.ones(N - 1), 1)[None, None, ...]
        * np.conjugate(_k)[..., None, None]
    )
    M += 1j * l / 2 * np.diag(np.ones(N - 1), -1)[None, None, ...] * _k[..., None, None]
    # Pairwise interaction
    M[..., n, n] += np.abs(_k) ** 2 * compute_U(_k)
    return M


def compute_A(k):
    if not isinstance(k, np.ndarray):
        k = np.array([k])

    A = np.zeros((len(k), N, N)).astype(np.complex128)
    A += np.diag(np.arange(-n, n + 1) ** 2)[None, ...]
    A += (
        np.eye(N)[None, ...]
        * (D * np.abs(k) ** 2)[
            :,
            None,
            None,
        ]
    )
    A *= phi / np.pi**2
    return A


def compute_S(w, k):
    _w, _k = normalize_w(w, k)
    M = compute_M(_w, _k)
    A = compute_A(k)
    M_inv = np.linalg.inv(M)
    # S = (M^-1).A.(M^-1)^H
    S = np.einsum(
        "wkij, kjl, wklm->wkim", M_inv, A, np.conjugate(M_inv.transpose(0, 1, 3, 2))
    )
    return S


r = np.linspace(0, 5, 100)


# integ = lambda k,r : (np.heaviside(k, 1)*k**2*compute_U(k)/(2*D*k**2+1)/(2+compute_U(k)))[:, None]*jv(1,np.outer(k,r))

# S1 = legendre_integrate(integ, 20, args=(r,), axis=0)


# Main

Nk = 32
Nphi = 16
k_length = np.logspace(-2, 1, Nk, base=10)
k_angle = np.linspace(0, 2 * np.pi * (1 - 1 / Nphi), Nphi)
k = np.outer(k_length, np.exp(1j * k_angle)).flatten()

Sint_unnorm = 1 / 2 / np.pi * legendre_integrate(compute_S, 50, args=(k,), axis=0)
Sint = np.einsum(
    "k, knm->knm",
    l * np.abs(k),
    Sint_unnorm,
)

# S = 1 / 2 / np.pi * legendre_integrate(compute_S, n_points=100, args=(k,), axis=0)

theta = np.exp(-1j * np.outer(np.arange(-n, n + 1), k_angle))
corr = (2 * np.pi**2 / phi) * np.einsum("knm,ni,mj->kij", Sint, theta, 1 / theta)
corr_r_th = corr.reshape((Nk, Nphi, Nphi, Nphi)).sum(axis=-1) / Nphi

i_indices = np.arange(Nphi).reshape(-1, 1)  # Column vector for row indices
j_indices = np.arange(Nphi)  # Row vector for column offsets

corr_r_phi = corr_r_th[:, (i_indices + j_indices) % Nphi, i_indices].sum(axis=-1) / Nphi

# # rho_corr = S[..., n, n].reshape((Nw, Nk, Nphi))
import matplotlib

matplotlib.use("TkAgg")
kk, pp = np.meshgrid(k_length, k_angle, indexing="ij")

X, Y = kk * np.cos(pp), kk * np.sin(pp)

fig = plt.figure()
ax_re = fig.add_axes([0, 0, 0.8, 1], projection="3d")  # <-- 3D plot axis

ax_re.plot_surface(X, Y, np.imag(corr_r_phi), cmap=plt.cm.YlGnBu_r)

plt.show()

# ax_sl = fig.add_axes([0.1, 0.85, 0.8, 0.1])

# # slide = Slider(ax_sl, r"$\omega$", -1, 1, valstep=x)


# # def on_change(val):
# #     i = list(x).index(val)
# #     ax_re.cla()
# #     # ax_im.cla()
# #     ax_re.plot_surface(X, Y, S[i], cmap=plt.cm.YlGnBu_r)
# #     # ax_im.plot_surface(X, Y, np.imag(rho_corr[i]))
# #     # plt.show()


# # slide.on_changed(on_change)

# plt.show()


# # corr_diff = corr_r_phi - 1 / (1 + compute_U(k_length))[:, None]

# # # np.save(r"D:\2_Numerical\Data\Correlations\S.npy", S)
# # # np.save(r"D:\2_Numerical\Data\Correlations\corr_r_phi.npy", corr_r_phi)

# # # S = np.load(r"D:\2_Numerical\Data\Correlations\S.npy")
# # # corr_r_phi = np.load(r"D:\2_Numerical\Data\Correlations\corr_r_phi.npy")

# # import matplotlib

# # matplotlib.use("TkAgg")
# # kk, pp = np.meshgrid(k_length, k_angle, indexing="ij")

# # X, Y = kk * np.cos(pp), kk * np.sin(pp)

# # fig = plt.figure()
# # ax = fig.add_subplot(projection="3d")
# # ax.plot_surface(X, Y, corr_r_phi, cmap=plt.cm.YlGnBu_r)

# # plt.show()

# # # fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
# # # mesh = ax.pcolormesh(pp, kk, np.real(corr_r_phi))
# # # ax.set_ylim((0, 10))

# # # fig.colorbar(mesh)

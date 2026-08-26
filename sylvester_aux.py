import numpy as np
from scipy.special import jv, struve

## General utilities


def dot(u, v):
    """Scalar product of 2 complex numbers"""
    return np.real(0.5 * (u * np.conjugate(v) + np.conjugate(u) * v))


def L(N, s, k, lp, phi, eps, V):
    """Lengths in particle diameters (a = 1): rho0 = 4 phi / pi with phi
    the packing fraction  phi = pi rho0 (a/2)^2."""
    L = np.diag((np.arange(N) - s) ** 2) / lp - 1j / 2 * (
        np.conjugate(k) * np.eye(N, k=1) + k * np.eye(N, k=-1)
    )
    L[s, s] += 4 * phi / np.pi / eps * V(k) * k**2
    return L


## Interaction potentials


def Vexp(k, r0=0.5):
    return 2 * np.pi * r0**2 * np.exp(-0.5 * (r0 * k) ** 2)


def Vexp_r(r, r0=0.5):
    return np.exp(-0.5 * (r / r0) ** 2)


def dVexp_r(r, r0=0.5):
    return -r / r0**2 * np.exp(-0.5 * (r / r0) ** 2)


def Vharm(k):
    """2D FT of Vharm_r(r) = (1 - r)^2 / 2 for r < 1 (range = 1 diameter)."""
    _v = (
        np.pi
        / k**2
        * (
            -2 * jv(2, k)
            + np.pi * (jv(1, k) * struve(0, k) - jv(0, k) * struve(1, k))
        )
    )
    return np.where(k == 0, np.pi / 12, _v)


def Vharm_r(r):
    return np.where(r <= 1, 0.5 * (1 - r) ** 2, 0)


def dVharm_r(r):
    return np.where(r <= 1, -(1 - r), 0)

import numpy as np
from scipy.special import jv, struve

## General utilities


def dot(u, v):
    """Scalar product of 2 complex numbers"""
    return np.real(0.5 * (u * np.conjugate(v) + np.conjugate(u) * v))


def L(k, lp, phi, eps, V, s=10):
    N = 2 * s + 1
    L = np.diag((np.arange(N) - s) ** 2) / lp - 1j / 2 * (
        np.conjugate(k) * np.eye(N, k=1) + k * np.eye(N, k=-1)
    )
    L[s, s] += phi / np.pi / eps * V(k) * np.abs(k) ** 2
    return L


## Interaction potentials


def Vexp(k, r0=0.5):
    return 2 * np.pi * r0**2 * np.exp(-0.5 * (r0 * np.abs(k)) ** 2)


def Vexp_r(r, r0=0.5):
    return np.exp(-0.5 * (r / r0) ** 2)


def dVexp_r(r, r0=0.5):
    return -r / r0 * np.exp(-0.5 * (r / r0) ** 2)


def Vharm(k):
    _v = (
        4
        * np.pi
        * (
            1
            / np.abs(k) ** 2
            * (
                -2 * jv(2, 2 * np.abs(k))
                + np.pi
                * (
                    jv(1, 2 * np.abs(k)) * struve(0, 2 * np.abs(k))
                    - jv(0, 2 * np.abs(k)) * struve(1, 2 * np.abs(k))
                )
            )
        )
    )
    return np.where(k == 0, 16 * np.pi / 12, _v)


def Vharm_r(r):
    return np.where(r <= 2, 0.5 * (2 - r) ** 2, 0)


def dVharm_r(r):
    return np.where(r <= 2, -(2 - r), 0)

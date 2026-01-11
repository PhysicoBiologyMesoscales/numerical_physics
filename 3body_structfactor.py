import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester, eig, inv
from scipy.special import jv, struve
from itertools import product
from tqdm import tqdm
from matplotlib.colors import CenteredNorm

s = 10
N = 2 * s + 1
lp = 10
phi = 1.0
eps = 1e-2


def Vexp(k, r0=0.5):
    return 2 * np.pi * r0**2 * np.exp(-0.5 * (r0 * np.abs(k)) ** 2)


def L(k, lp, phi, eps, V):
    L = np.diag((np.arange(N) - s) ** 2) / lp - 1j * 1 / 2 * (
        np.conjugate(k) * np.eye(N, k=1) + k * np.eye(N, k=-1)
    )
    L[s, s] += phi / np.pi / eps * V(k) * k**2
    return L


k1 = 1.0
k2 = 1.0 * np.exp(2 * np.pi / 3 * 1j)

L1 = L(k1, lp, phi, eps, Vexp)
L2 = L(k2, lp, phi, eps, Vexp)
L12 = L(-k1 - k2, lp, phi, eps, Vexp)

l1, P1 = eig(L1)
l2, P2 = eig(L2)
l12, P12 = eig(L12)

Q1, Q2, Q12 = inv(P1), inv(P2), inv(P12)


def compute_S(L, lp):
    RHS = 2 * np.diag((np.arange(N) - s) ** 2) / lp
    S = solve_sylvester(L, L.conj().T, RHS)
    return S


S1, S2, S12 = compute_S(L1, lp), compute_S(L2, lp), compute_S(L12, lp)


def extend_S(S):
    """Extend S by assuming S_{nm} = \delta_{nm} for abs(n)>s or abs(m)>s"""
    newS = np.eye(N + 2 * s, dtype=np.complex128)
    newS[s : s + N, s : s + N] = S
    return newS


def noise_vertex(S1, S2, S12, lp):
    def partial_vtx(S, axis):
        match axis:
            case 0:
                j, k, ijk = "m", "l", "nml"
            case 1:
                j, k, ijk = "l", "n", "mln"
            case 2:
                j, k, ijk = "n", "m", "lnm"

        _S = extend_S(S)
        idx = np.arange(N) - s
        ml = 2 * s - idx[:, None] - idx[None, :]
        gathered = np.take(_S, ml, axis=1)[s : s + N]
        tot = np.einsum(f"{j}, {k}, {ijk} -> nml", idx, idx, gathered)
        return tot

    return -2 / lp * (partial_vtx(S1, 0) + partial_vtx(S2, 1) + partial_vtx(S12, 2))


def dot(u, v):
    """Scalar product of 2 complex numbers"""
    return np.real(0.5 * (u * np.conjugate(v) + np.conjugate(u) * v))


def collision_vertex(S1, S2, S12, k1, k2, V):
    idx = 2 * s - np.arange(N)
    return (
        phi
        / np.pi
        / eps
        * (
            dot(k1, k2) * V(k2) * S2[None, :, (s,)] * S12.T[idx, None, :]
            - dot(k1, k1 + k2) * V(k1 + k2) * S12[None, None, :, s] * S2.T[idx, :, None]
            + dot(k2, k1) * V(k1) * S1[:, (s,), None] * S12.T[None, idx, :]
            - dot(k2, k1 + k2) * V(k1 + k2) * S12[None, None, :, s] * S1[:, idx, None]
            - dot(k1 + k2, k1) * V(k1) * S1[:, (s,), None] * S2[None, :, idx]
            - dot(k1 + k2, k2) * V(k2) * S2[None, :, (s,)] * S1[:, None, idx]
        )
    )


T = noise_vertex(S1, S2, S12, lp) + collision_vertex(S1, S2, S12, k1, k2, Vexp)

U = np.einsum("ni, mj, lk , ijk -> nml", Q1, Q2, Q12, T)
U2 = U / (l1[:, None, None] + l2[None, :, None] + l12[None, None, :])

S_3body = np.einsum("ni, mj, lk , ijk -> nml", P1, P2, P12, U2)

idx = np.arange(N, dtype=np.complex128) - s
n = idx[:, None, None]
ml = idx[:, None] + idx[None, :]
delta = n == (-ml)[None, ...]

# Test the possible solution S3_nml = S_ni S_mj S_lk \delta_{i+j+k, 0}
S_th = np.einsum("ni, mj, lk, ijk -> nml", S1, S2, S12, delta)

# G = (S - delta) * np.pi / phi

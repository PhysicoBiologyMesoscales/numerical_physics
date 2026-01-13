import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester, eig, inv
from sylvester_aux import dot, Vexp

s = 10
N = 2 * s + 1
lp = 0.01
phi = 0.1
eps = 1e-2


def L(k, lp, phi, eps, V):
    L = np.diag((np.arange(N) - s) ** 2) / lp - 1j * 1 / 2 * (
        np.conjugate(k) * np.eye(N, k=1) + k * np.eye(N, k=-1)
    )
    L[s, s] += phi / np.pi / eps * V(k) * k**2
    return L


k1 = 1.0
k2 = 2.0 * np.exp(2 * np.pi / 3 * 1j)

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
# S1, S2, S12 = np.eye(N), np.eye(N), np.eye(N)


def extend_S(S):
    """Extend S by assuming S_{nm} = \delta_{nm} for abs(n)>s or abs(m)>s"""
    newS = np.eye(N + 2 * s, dtype=np.complex128)
    newS[s : s + N, s : s + N] = S
    return newS


def noise_vertex(S1, S2, S12, lp):
    _S1, _S2, _S12 = extend_S(S1), extend_S(S2), extend_S(S12)
    idx = np.arange(N) - s
    lm = 2 * s - idx[:, None] + idx[None, :]
    # gathered = np.take(_S, ml, axis=1)[s : s + N]
    # tot = sign * np.einsum(f"{j}, {k}, {ijk} -> nml", idx, idx, gathered)
    # return tot

    # return 2 / lp * (partial_vtx(S1, 0) + partial_vtx(S2, 1) + partial_vtx(S12, 2))
    v1 = np.einsum(
        "m, l, nml -> nml",
        idx,
        idx,
        np.take(_S1, 2 * s - idx[:, None] + idx[None, :], axis=1)[s : s + N],
    )
    v2 = np.einsum(
        "n, l, mnl -> nml",
        idx,
        idx,
        np.take(_S2, 2 * s - idx[:, None] + idx[None, :], axis=1)[s : s + N],
    )
    v12 = -np.einsum(
        "n, m, lnm -> nml",
        idx,
        idx,
        np.take(_S12, 2 * s - idx[:, None] - idx[None, :], axis=1)[s : s + N][s - idx],
    )
    return 2 / lp * (v1 + v2 + v12)


def collision_vertex(S1, S2, S12, k1, k2, V):
    idx = 2 * s - np.arange(N)
    return (
        phi
        / np.pi
        / eps
        * (
            dot(k1, k2) * V(k2) * np.einsum("m, ln -> nml", S2[:, s], S12[idx][:, idx])
            - dot(k1, k1 + k2)
            * V(k1 + k2)
            * np.einsum("l, mn -> nml", S12[idx, s], S2[:, idx])
            + dot(k2, k1)
            * V(k1)
            * np.einsum("n, lm -> nml", S1[:, s], S12[idx][:, idx])
            - dot(k2, k1 + k2)
            * V(k1 + k2)
            * np.einsum("l, nm -> nml", S12[idx, s], S1[:, idx])
            - dot(k1 + k2, k1) * V(k1) * np.einsum("n, ml -> nml", S1[:, s], S2)
            - dot(k1 + k2, k2) * V(k2) * np.einsum("m, nl -> nml", S2[:, s], S1)
        )
    )


T = noise_vertex(S1, S2, S12, lp) + collision_vertex(S1, S2, S12, k1, k2, Vexp)

U = np.einsum("ni, mj, kl , ijk -> nml", Q1, Q2, P12, T)
U2 = U / (l1[:, None, None] + l2[None, :, None] + l12[None, None, :])

S_3b = np.einsum("ni, mj, kl , ijk -> nml", P1, P2, Q12, U2)

idx = np.arange(N, dtype=np.complex128) - s
n = idx[:, None, None]
ml = idx[:, None] + idx[None, :]
delta = n == (-ml)[None, ...]
rev_idx = 2 * s - np.arange(N)

# # Test the possible solution S3_nml = S_ni S_mj S_lk \delta_{i+j+k, 0}
S_th = np.einsum("ni, mj, lk, ijk -> nml", S1, S2, S12[rev_idx], delta)

# # G = (S - delta) * np.pi / phi

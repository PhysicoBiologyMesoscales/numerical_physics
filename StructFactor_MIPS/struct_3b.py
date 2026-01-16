"""
3-body structure factor calculations (optimized version).

This module has been optimized to accept pre-computed eigendecompositions
to minimize redundant operations:
- compute_3bod now accepts optional pre-computed L1, S1, l1, P1, Q1 parameters
- compute_3bod now accepts optional pre-computed L2, S2, l2, P2, Q2 parameters
- Only L12-related values need to be computed for each (k1, k2) pair

For a fully merged optimized implementation, see struct_optimized.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester, eig, inv
from struct_aux import dot, Vexp, L


def compute_S(L, lp, s=10):
    """
    Solve Sylvester equation for structure factor S.
    
    Can be pre-computed and reused if the same L matrix is needed multiple times.
    """
    N = 2 * s + 1
    RHS = 2 * np.diag((np.arange(N) - s) ** 2) / lp
    S = solve_sylvester(L, L.conj().T, RHS)
    return S


def compute_3bod(k1, k2, lp, phi, eps, V, s=10, 
                 L1=None, S1=None, l1=None, P1=None, Q1=None,
                 L2=None, S2=None, l2=None, P2=None, Q2=None):
    """
    Compute 3-body structure factor with optional pre-computed values.
    
    This optimized version accepts pre-computed eigendecompositions and
    Sylvester solutions to avoid redundant calculations.
    
    Parameters:
        k1, k2: Wave vectors
        lp, phi, eps: Physical parameters  
        V: Interaction potential
        s: Fourier mode truncation
        L1, S1, l1, P1, Q1: Pre-computed values for k1 (optional)
        L2, S2, l2, P2, Q2: Pre-computed values for k2 (optional)
        
    Returns:
        S_3b: 3-body structure factor tensor of shape (N, N, N)
        
    Performance notes:
        - Without pre-computed values: ~6 eig() + 3 inv() + 3 Sylvester solves
        - With L1 pre-computed: ~4 eig() + 2 inv() + 2 Sylvester solves
        - With L1 and L2 pre-computed: ~2 eig() + 1 inv() + 1 Sylvester solve
    """

    N = 2 * s + 1

    # Use pre-computed k1 values if available, otherwise compute them
    if L1 is None:
        L1 = L(k1, lp, phi, eps, V)
    if l1 is None or P1 is None:
        l1, P1 = eig(L1)
    if Q1 is None:
        Q1 = inv(P1)
    if S1 is None:
        S1 = compute_S(L1, lp, s=s)
    
    # Use pre-computed k2 values if available, otherwise compute them
    if L2 is None:
        L2 = L(k2, lp, phi, eps, V)
    if l2 is None or P2 is None:
        l2, P2 = eig(L2)
    if Q2 is None:
        Q2 = inv(P2)
    if S2 is None:
        S2 = compute_S(L2, lp, s=s)
    
    # Always compute L12 (depends on both k1 and k2)
    L12 = L(-k1 - k2, lp, phi, eps, V)
    l12, P12 = eig(L12)
    Q12 = inv(P12)
    S12 = compute_S(L12, lp, s=s)

    def extend_S(S):
        r"""Extend S by assuming S_{nm} = \delta_{nm} for abs(n)>s or abs(m)>s"""
        newS = np.eye(N + 2 * s, dtype=np.complex128)
        newS[s : s + N, s : s + N] = S
        return newS

    def noise_vertex(S1, S2, S12, lp):
        _S1, _S2, _S12 = extend_S(S1), extend_S(S2), extend_S(S12)
        idx = np.arange(N) - s
        lm = 2 * s - idx[:, None] + idx[None, :]

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
            np.take(_S12, 2 * s - idx[:, None] - idx[None, :], axis=1)[s : s + N][
                s - idx
            ],
        )
        return 2 / lp * (v1 + v2 + v12)

    def collision_vertex(S1, S2, S12, k1, k2, V, phi, eps):
        idx = 2 * s - np.arange(N)
        return (
            phi
            / np.pi
            / eps
            * (
                dot(k1, k2)
                * V(k2)
                * np.einsum("m, ln -> nml", S2[:, s], S12[idx][:, idx])
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

    T = noise_vertex(S1, S2, S12, lp) + collision_vertex(S1, S2, S12, k1, k2, V, phi, eps)

    U = np.einsum("ni, mj, kl , ijk -> nml", Q1, Q2, P12, T)
    U2 = U / (l1[:, None, None] + l2[None, :, None] + l12[None, None, :])

    S_3b = np.einsum("ni, mj, kl , ijk -> nml", P1, P2, Q12, U2)

    return S_3b


if __name__ == "__main__":
    lp = 1.0
    phi = 0.5
    eps = 1e-2
    k1 = 1.0
    k2 = 1.0
    s = 10
    S, T = compute_3bod(
        k1,
        k2,
        lp,
        phi,
        eps,
        Vexp,
    )
    plt.imshow(np.real(S[s]))

# idx = np.arange(N, dtype=np.complex128) - s
# n = idx[:, None, None]
# ml = idx[:, None] + idx[None, :]
# delta = n == (-ml)[None, ...]
# rev_idx = 2 * s - np.arange(N)

# # # Test the possible solution S3_nml = S_ni S_mj S_lk \delta_{i+j+k, 0}
# S_th = np.einsum("ni, mj, lk, ijk -> nml", S1, S2, S12[rev_idx], delta)

# # # G = (S - delta) * np.pi / phi

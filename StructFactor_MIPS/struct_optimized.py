"""
Optimized structure factor computation for 2-body and 3-body correlations.

This module merges and optimizes struct_2b.py and struct_3b.py to minimize
costly operations (eigendecompositions, matrix inversions, Sylvester solves).

Key optimizations:
1. Pre-compute L1, S1, eigendecomposition for fixed k1 (eliminates ~1999/2000 redundant operations)
2. Pre-compute L2, S2, eigendecomposition for all k2 values (reusable across k1 iterations)
3. Only compute L12, S12, eigendecomposition per (k1, k2) pair
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester, eig, inv
from scipy.special import jv, struve
from joblib import Parallel, delayed
from itertools import product
from tqdm import tqdm
from matplotlib.colors import CenteredNorm

from struct_aux import dot, L, Vexp, Vexp_r, dVexp_r

s = 10
N = 2 * s + 1


# ============================================================================
# Core computational functions
# ============================================================================

def compute_S(L, lp, s=10):
    """Solve Sylvester equation for structure factor S"""
    N = 2 * s + 1
    RHS = 2 * np.diag((np.arange(N) - s) ** 2) / lp
    S = solve_sylvester(L, L.conj().T, RHS)
    return S


def precompute_k2_values(k2abs_arr, alpha, lp, phi, eps, V, s=10):
    """
    Pre-compute all k2-related matrices that can be reused across different k1 values.
    
    Returns:
        dict: Dictionary with (i, j) index tuples as keys mapping to precomputed matrices.
              The indices correspond to k2abs_arr[i] * exp(1j * alpha[j])
    """
    k2_arr = k2abs_arr[:, None] * np.exp(1j * alpha[None, :])
    k2_cache = {}
    
    for i in range(len(k2abs_arr)):
        for j in range(len(alpha)):
            k2 = k2_arr[i, j]
            L2 = L(k2, lp, phi, eps, V)
            S2 = compute_S(L2, lp, s=s)
            l2, P2 = eig(L2)
            Q2 = inv(P2)
            
            k2_cache[(i, j)] = {
                'k2': k2,
                'L2': L2,
                'S2': S2,
                'l2': l2,
                'P2': P2,
                'Q2': Q2
            }
    
    return k2_cache


def compute_3bod_optimized(k1, k2, lp, phi, eps, V, s=10,
                          L1=None, S1=None, l1=None, P1=None, Q1=None,
                          L2=None, S2=None, l2=None, P2=None, Q2=None):
    """
    Compute 3-body structure factor with pre-computed values.
    
    Parameters:
        k1, k2: Wave vectors
        lp, phi, eps: Physical parameters
        V: Interaction potential
        s: Fourier mode truncation
        L1, S1, l1, P1, Q1: Pre-computed values for k1 (optional)
        L2, S2, l2, P2, Q2: Pre-computed values for k2 (optional)
    
    Returns:
        S_3b: 3-body structure factor tensor
    """
    N = 2 * s + 1

    # Use pre-computed k1 values if available
    if L1 is None:
        L1 = L(k1, lp, phi, eps, V)
    if l1 is None or P1 is None:
        l1, P1 = eig(L1)
    if Q1 is None:
        Q1 = inv(P1)
    if S1 is None:
        S1 = compute_S(L1, lp, s=s)
    
    # Use pre-computed k2 values if available
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

    T = noise_vertex(S1, S2, S12, lp) + collision_vertex(S1, S2, S12, k1, k2, Vexp, phi, eps)

    U = np.einsum("ni, mj, kl , ijk -> nml", Q1, Q2, P12, T)
    U2 = U / (l1[:, None, None] + l2[None, :, None] + l12[None, None, :])

    S_3b = np.einsum("ni, mj, kl , ijk -> nml", P1, P2, Q12, U2)

    return S_3b


# ============================================================================
# RHS functions for Sylvester solve
# ============================================================================

def RHS_S(k, lp, phi, eps, V):
    """
    Compute RHS for structure factor S with optimized k2 pre-computation.
    
    This is the main optimization point:
    - Pre-computes L1, S1, eigendecomposition once
    - Pre-computes all k2 values once (can be cached across k1 iterations)
    - Only computes L12 for each (k1, k2) pair
    """
    k2abs_arr = np.linspace(0, 10, 200)
    alpha = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    k2_arr = k2abs_arr[:, None] * np.exp(1j * alpha[None, :])
    S3arr = np.zeros((len(k2abs_arr), len(alpha), N, N), dtype=np.complex128)
    
    # Pre-compute L1-related values (constant for this k)
    L1 = L(k, lp, phi, eps, V)
    S1 = compute_S(L1, lp, s=s)
    l1, P1 = eig(L1)
    Q1 = inv(P1)
    
    # Pre-compute all k2-related values (reusable across different k1 values)
    k2_cache = precompute_k2_values(k2abs_arr, alpha, lp, phi, eps, V, s=s)
    
    # Main loop - only computes L12 for each iteration
    for i in range(len(k2abs_arr)):
        for j in range(len(alpha)):
            k2_data = k2_cache[(i, j)]
            k2 = k2_data['k2']
            S3 = compute_3bod_optimized(
                k, k2, lp, phi, eps, V, s=s,
                L1=L1, S1=S1, l1=l1, P1=P1, Q1=Q1,
                L2=k2_data['L2'], S2=k2_data['S2'], 
                l2=k2_data['l2'], P2=k2_data['P2'], Q2=k2_data['Q2']
            )
            S3arr[i, j, :, :] = S3[:, s, :]
    
    return np.diag(2 / lp * (np.arange(N) - s) ** 2) + np.trapz(
        np.trapz(S3arr, k2abs_arr, axis=0), alpha, axis=0
    )


def RHS_S_with_k2_cache(k, lp, phi, eps, V, k2_cache, k2abs_arr, alpha):
    """
    Compute RHS_S with externally provided k2 cache for maximum efficiency.
    
    Use this when calling RHS_S multiple times with different k values
    but the same k2 grid, lp, phi, eps, and V.
    
    Parameters:
        k2_cache: Dictionary with (i, j) tuples as keys, from precompute_k2_values()
    """
    S3arr = np.zeros((len(k2abs_arr), len(alpha), N, N), dtype=np.complex128)
    
    # Pre-compute L1-related values (constant for this k)
    L1 = L(k, lp, phi, eps, V)
    S1 = compute_S(L1, lp, s=s)
    l1, P1 = eig(L1)
    Q1 = inv(P1)
    
    # Main loop - only computes L12 for each iteration
    for i in range(len(k2abs_arr)):
        for j in range(len(alpha)):
            k2_data = k2_cache[(i, j)]
            k2 = k2_data['k2']
            S3 = compute_3bod_optimized(
                k, k2, lp, phi, eps, V, s=s,
                L1=L1, S1=S1, l1=l1, P1=P1, Q1=Q1,
                L2=k2_data['L2'], S2=k2_data['S2'], 
                l2=k2_data['l2'], P2=k2_data['P2'], Q2=k2_data['Q2']
            )
            S3arr[i, j, :, :] = S3[:, s, :]
    
    return np.diag(2 / lp * (np.arange(N) - s) ** 2) + np.trapz(
        np.trapz(S3arr, k2abs_arr, axis=0), alpha, axis=0
    )


def RHS_h(k, eps, V):
    """Compute RHS for density-density correlation h"""
    mat = np.zeros((N, N))
    mat[s, s] = -2 * k**2 * V(k) / eps
    return mat


# ============================================================================
# Correlation computation functions (from struct_2b.py)
# ============================================================================

def compute_correlations(lp, phi, eps, V, k_arr, which="h", parallel="False"):
    """Computes correlation matrices for all values in k_arr"""
    match which:
        case "h":
            RHS = lambda k: RHS_h(k, eps, V)
        case "S":
            RHS = lambda k: RHS_S(k, lp, phi, eps, V)
        case _:
            raise ValueError(f"Unknown correlation type: {which}. Must be 'h' or 'S'.")

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
    """Convert correlation functions from k-space to r-space"""
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
    """Compute pressure"""
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
    """
    Computes correlation function from the reference frame of the 1st particle.
    
    WARNING: This function is not fully implemented. It requires the 'g' case
    in compute_correlations() which is not defined. Calling this function will
    raise a ValueError. This function is preserved from the original code for
    reference but should not be used.
    """
    raise NotImplementedError(
        "compute_B requires the 'g' case in compute_correlations which is not "
        "implemented. This function is preserved for backward compatibility but "
        "cannot be used until the 'g' correlation type is implemented."
    )

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
    """Visualize correlation function B"""
    from mpl_toolkits.mplot3d import Axes3D

    ax = Axes3D(plt.figure())
    r, th = np.meshgrid(r_arr, alpha)
    plt.subplot(projection="polar")
    plt.pcolormesh(th, r, B.T)
    plt.show()


# ============================================================================
# Backward compatibility - original function signatures
# ============================================================================

# For backward compatibility, keep the original compute_3bod function
compute_3bod = compute_3bod_optimized

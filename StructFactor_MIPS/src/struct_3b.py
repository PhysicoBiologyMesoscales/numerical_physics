"""3-body correlation function h3(k1, k2).

The (N, N, N) tensor h3 solves the three-index Sylvester equation

    L(k1).h3 + L(k2).h3 + h3.L(k12)^H = T(k1, k2),    k12 = k1 + k2,

with each L acting on its own index (the right-multiplication by
L(-k1-k2) = L(k12)^H realizes the L(k12)^dagger of the theory).  The
RHS T is a collision-type vertex built from the 2-body correlations
h(k1), h(k2), h(k12); see _rhs_vertex for the exact expression.

The operator L(k), the 2-body source and the vertex all come from a
``Model`` (see ``models.py``): every solver function takes a single
``model`` argument instead of the loose ``(lp, phi, eps, V, ..., D)``
tuple, so a different microscopic model plugs in without editing this
file.  Dimensionless prefactors of the default (MIPS) vertex: mu -> 1/eps
and rho0 = 4 phi / pi (phi the packing fraction of diameter-1 disks).

The equation is solved in the eigenbases of the three L operators; the
index rotations are evaluated as three successive matrix products
(BLAS).  `spectral_data` bundles everything compute_3bod needs for one
wave vector; precompute it for k1 and/or k2 when they are reused across
many (k1, k2) pairs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester, eig, inv

from struct_aux import dot, Vexp


def compute_h(L_mat, k, model, s=None):
    """2-body correlation h(k): solves L h + h L^H = model.rhs_2b(k).

    For the ABP model the source is -2 |k|^2 V(k)/eps at the (0, 0)
    harmonic, written with |k|^2 = dot(k, k) so that it also holds for
    complex (vector-valued) k.  ``L_mat`` is passed explicitly (it need
    not equal ``model.L(k)`` -- e.g. the third leg uses L(k12)^H).
    ``s`` is unused (the truncation is baked into the model); kept for
    call-site compatibility.
    """
    return solve_sylvester(L_mat, L_mat.conj().T, model.rhs_2b(k))


def spectral_data(k, model, s=None):
    """L(k), its eigendecomposition L = P diag(l) Q with Q = P^-1, and h(k).

    ``s`` is unused (the truncation is baked into the model); kept for
    call-site compatibility.
    """
    Lk = model.L(k)
    l, P = eig(Lk)
    Q = inv(P)
    h = compute_h(Lk, k, model)
    return Lk, l, P, Q, h


def _vertex_terms(h1, h2, h12, k1, k2, V, phi, eps, s):
    r"""The six rank-structured terms of the MIPS vertex T as (slot, vec, mat).

    Each term of T is a vector living on one index times a matrix on the
    other two: slot 0 -> T_nml = vec_n mat_ml, slot 1 -> vec_m mat_nl,
    slot 2 -> vec_l mat_nm.  The overall factor eps is absorbed in vec.

    This is the pairwise-force (MIPS) vertex, carried by a ``Model`` as
    its default ``vertex_terms``; a different model supplies its own
    callable with the same signature.
    """
    N = 2 * s + 1
    k12 = k1 + k2
    rev = 2 * s - np.arange(N)
    delta0 = np.zeros(N)
    delta0[s] = 1.0
    rho0 = 4 * phi / np.pi

    h2_mn = h2[:, rev].T  # [n, m] = h_{m,-n}(k2)
    h1_nm = h1[:, rev]  # [n, m] = h_{n,-m}(k1)

    return [
        (0, eps * delta0, V(k1) * (dot(k1, k2 - k1) * h12 + dot(k1, -k1 - k12) * h2)),
        (1, eps * delta0, V(k2) * (dot(k2, k1 - k2) * h12 + dot(k2, -k12 - k2) * h1)),
        (
            2,
            eps * delta0,
            -V(k12) * (dot(k12, k1 + k12) * h2_mn + dot(k12, k2 + k12) * h1_nm),
        ),
        (
            0,
            eps * rho0 * h1[:, s],
            V(k1) * (dot(k1, k2) * h12 - dot(k1, k12) * h2),
        ),
        (
            1,
            eps * rho0 * h2[:, s],
            V(k2) * (dot(k2, k1) * h12 - dot(k2, k12) * h1),
        ),
        (
            2,
            eps * rho0 * h12[s],
            -V(k12) * (dot(k12, k1) * h2_mn + dot(k12, k2) * h1_nm),
        ),
    ]


def _assemble_vertex(vertex_terms, h1, h2, h12, k1, k2, V, phi, eps, s):
    r"""RHS tensor T of the 3-body equation from a ``vertex_terms`` builder.

    The default MIPS vertex T_nml, with k12 = k1 + k2, mu -> 1/eps and
    rho0 = 4 phi/pi:

    T_nml = mu [ d_{n0} V(k1) k1.((k2 - k1) h_{ml}(k12) + (-k1 - k2) h_{ml}(k2))
               + d_{m0} V(k2) k2.((k1 - k2) h_{nl}(k12) + (-k1 - k2) h_{nl}(k1))
               - d_{l0} V(k12) k12.((k1 + k12) h_{m,-n}(k2) + (k2 + k12) h_{n,-m}(k1))
               + rho0 ( h_{n0}(k1) V(k1) k1.(k2 h_{ml}(k12) - k12 h_{ml}(k2))
                      + h_{m0}(k2) V(k2) k2.(k1 h_{nl}(k12) - k12 h_{nl}(k1))
                      - h_{0l}(k12) V(k12) k12.(k1 h_{m,-n}(k2) + k2 h_{n,-m}(k1)) ) ]
    """
    N = 2 * s + 1
    subs = {0: "n, ml -> nml", 1: "m, nl -> nml", 2: "l, nm -> nml"}
    T = np.zeros((N, N, N), dtype=np.complex128)
    for slot, vec, mat in vertex_terms(h1, h2, h12, k1, k2, V, phi, eps, s):
        T += np.einsum(subs[slot], vec, mat)
    return T


def _rhs_vertex(h1, h2, h12, k1, k2, V, phi, eps, s):
    """RHS of the 3-body equation from the default MIPS vertex terms."""
    return _assemble_vertex(_vertex_terms, h1, h2, h12, k1, k2, V, phi, eps, s)


def _rotate(T, A, B, C):
    """einsum('ni,mj,kl,ijk->nml', A, B, C, T) via three BLAS matrix products."""
    X = np.tensordot(A, T, axes=(1, 0))  # (n, j, k)
    X = np.tensordot(B, X, axes=(1, 1))  # (m, n, k)
    X = np.tensordot(X, C, axes=(2, 0))  # (m, n, l)
    return X.transpose(1, 0, 2)


def compute_3bod(
    k1,
    k2,
    model,
    s=None,
    L1=None,
    h1=None,
    l1=None,
    P1=None,
    Q1=None,
    L2=None,
    h2=None,
    l2=None,
    P2=None,
    Q2=None,
    L12=None,
    h12=None,
    l12=None,
    P12=None,
    Q12=None,
):
    """Compute the 3-body correlation tensor h3[n, m, l] for (k1, k2).

    The optional arguments take precomputed spectral data for k1, k2 and
    the third leg (see spectral_data); anything missing is computed
    here.  Third-leg conventions: L12, l12, P12, Q12 refer to
    L(-k1-k2) = L(k12)^H (the eigenbasis the solve runs in), while h12
    is the 2-body correlation at +k12 = k1 + k2 (what the vertex needs).
    """
    model.require_3b()
    if s is None:
        s = model.s
    elif s != model.s:
        raise ValueError(
            f"s={s} does not match model.s={model.s}; rebuild the model with this s"
        )
    if L1 is None:
        L1 = model.L(k1)
    if l1 is None or P1 is None:
        l1, P1 = eig(L1)
    if Q1 is None:
        Q1 = inv(P1)
    if h1 is None:
        h1 = compute_h(L1, k1, model)

    if L2 is None:
        L2 = model.L(k2)
    if l2 is None or P2 is None:
        l2, P2 = eig(L2)
    if Q2 is None:
        Q2 = inv(P2)
    if h2 is None:
        h2 = compute_h(L2, k2, model)

    # Third leg: the solve runs in the eigenbasis of L(-k12) = L(k12)^H
    # (right-multiplication), while the vertex needs h at +k12.
    k12 = k1 + k2
    if L12 is None and (l12 is None or P12 is None or Q12 is None or h12 is None):
        L12 = model.L(-k12)
    if l12 is None or P12 is None:
        l12, P12 = eig(L12)
    if Q12 is None:
        Q12 = inv(P12)
    if h12 is None:
        h12 = compute_h(L12.conj().T, k12, model)  # L(k12) = L(-k12)^H exactly

    T = _assemble_vertex(
        model.vertex_terms, h1, h2, h12, k1, k2, model.V, model.phi, model.eps, s
    )

    U = _rotate(T, Q1, Q2, P12)
    U /= l1[:, None, None] + l2[None, :, None] + l12[None, None, :]
    return _rotate(U, P1, P2, Q12)


def compute_3bod_slice0(
    k1, k2, model, s, h1, h2, h12, l1, P1, Q1, l2, P2, Q2, l12, P12, Q12
):
    """h3[s] (the n=0 slice of compute_3bod) in O(N^3) instead of O(N^4).

    Exploits the rank structure of the vertex: each term is a vector on
    one index times a matrix on the other two, so the rotation into the
    eigenbases costs one matrix product per term instead of a full
    tensor rotation, and only the row P1[s] of the rotation back is
    needed.  All spectral data is required (same conventions as
    compute_3bod: l12/P12/Q12 belong to L(-k1-k2), h12 to +k1+k2).
    """
    N = 2 * s + 1
    U = np.zeros((N, N, N), dtype=np.complex128)
    for slot, vec, mat in model.vertex_terms(
        h1, h2, h12, k1, k2, model.V, model.phi, model.eps, s
    ):
        if slot == 0:
            U += np.einsum("i, jk -> ijk", Q1 @ vec, Q2 @ mat @ P12)
        elif slot == 1:
            U += np.einsum("j, ik -> ijk", Q2 @ vec, Q1 @ mat @ P12)
        else:
            U += np.einsum("k, ij -> ijk", P12.T @ vec, Q1 @ mat @ Q2.T)
    U /= l1[:, None, None] + l2[None, :, None] + l12[None, None, :]
    A = np.tensordot(P1[s], U, axes=(0, 0))  # contract the n = 0 row
    return P2 @ A @ Q12


def compute_3bod_slice0_batch(
    k1,
    k2,
    model,
    s,
    h1,
    h2,
    h12,
    l1,
    P1,
    Q1,
    l2,
    P2,
    Q2,
    l12,
    P12,
    Q12,
    max_bytes=2**26,
):
    """compute_3bod_slice0 stacked over a leading node axis.

    ``k1``, ``k2`` are (T,) wave vectors and the leg-1/leg-2 spectral
    data carry a matching leading axis (h1, P1, Q1: (T, N, N); l1:
    (T, N); same for leg 2); the third leg (h12, l12, P12, Q12) is
    shared by every node, as in the bipolar collision integral where
    k12 = k for all nodes.  Returns the stacked n = 0 slices, shape
    (T, N, N).

    For the default MIPS vertex the six rank-structured terms are
    evaluated as stacked matrix products (one BLAS call each instead of
    one per node), and the (T, N, N, N) resolvent denominator is built
    in chunks capped at ``max_bytes``.  Any other ``vertex_terms``
    falls back to the per-node compute_3bod_slice0.
    """
    T = len(k1)
    N = 2 * s + 1
    if model.vertex_terms is not _vertex_terms:
        return np.stack(
            [
                compute_3bod_slice0(
                    k1[t], k2[t], model, s, h1[t], h2[t], h12,
                    l1[t], P1[t], Q1[t], l2[t], P2[t], Q2[t], l12, P12, Q12,
                )
                for t in range(T)
            ]
        )

    eps, phi, V = model.eps, model.phi, model.V
    rho0 = 4 * phi / np.pi
    rev = 2 * s - np.arange(N)
    delta0 = np.zeros(N)
    delta0[s] = 1.0
    k12 = k1 + k2
    V1, V2, V12 = V(k1), V(k2), V(k12)

    def sc(x):  # (T,) scalar -> broadcast against (T, N, N) matrices
        return x[:, None, None]

    # the six (slot, vec, mat) terms of _vertex_terms, batched over t
    h12b = h12[None]
    h2_mn = np.transpose(h2[:, :, rev], (0, 2, 1))  # [t, n, m] = h2[t, m, -n]
    h1_nm = h1[:, :, rev]  # [t, n, m] = h1[t, n, -m]
    terms = [
        (0, eps * delta0,
         sc(V1) * (sc(dot(k1, k2 - k1)) * h12b + sc(dot(k1, -k1 - k12)) * h2)),
        (1, eps * delta0,
         sc(V2) * (sc(dot(k2, k1 - k2)) * h12b + sc(dot(k2, -k12 - k2)) * h1)),
        (2, eps * delta0,
         -sc(V12) * (sc(dot(k12, k1 + k12)) * h2_mn + sc(dot(k12, k2 + k12)) * h1_nm)),
        (0, eps * rho0 * h1[:, :, s],
         sc(V1) * (sc(dot(k1, k2)) * h12b - sc(dot(k1, k12)) * h2)),
        (1, eps * rho0 * h2[:, :, s],
         sc(V2) * (sc(dot(k2, k1)) * h12b - sc(dot(k2, k12)) * h1)),
        (2, eps * rho0 * h12[s],
         -sc(V12) * (sc(dot(k12, k1)) * h2_mn + sc(dot(k12, k2)) * h1_nm)),
    ]

    def matvec(A, v):  # (T, N, N) @ vec, vec (N,) shared or (T, N) per node
        if v.ndim == 1:
            return np.matmul(A, v)
        return np.matmul(A, v[..., None])[..., 0]

    # rotate each term into the eigenbases (same products as slice0)
    Q2T = np.transpose(Q2, (0, 2, 1))
    slot0, slot1, slot2 = [], [], []
    for slot, vec, mat in terms:
        if slot == 0:
            slot0.append((matvec(Q1, vec), np.matmul(np.matmul(Q2, mat), P12)))
        elif slot == 1:
            slot1.append((matvec(Q2, vec), np.matmul(np.matmul(Q1, mat), P12)))
        else:  # slot-2 vecs are node-independent, transformed once
            slot2.append((P12.T @ vec, np.matmul(np.matmul(Q1, mat), Q2T)))

    out = np.empty((T, N, N), dtype=np.complex128)
    P1s = P1[:, s, :]
    chunk = max(1, int(max_bytes // (2 * 16 * N**3)))
    for a in range(0, T, chunk):
        c = slice(a, min(a + chunk, T))
        # Resolvent weights W[t, i, j, k] = P1[s, i] / (l1_i + l2_j + l12_k).
        # The n = 0 slice is A = sum_i W U with U the rank-structured
        # vertex; contracting each term against W directly keeps every
        # pass over the (T, N, N, N) array to a single read instead of
        # materializing U (memory-bound otherwise).
        W = P1s[c][:, :, None, None] / (
            l1[c][:, :, None, None]
            + l2[c][:, None, :, None]
            + l12[None, None, None, :]
        )
        A = np.zeros((W.shape[0], N, N), dtype=np.complex128)
        for w, M in slot0:  # U[t, i, j, k] = w_i M_jk
            A += np.einsum("tijk, ti -> tjk", W, w[c]) * M[c]
        for w, M in slot1:  # U[t, i, j, k] = w_j M_ik
            A += w[c][:, :, None] * np.einsum("tijk, tik -> tjk", W, M[c])
        for v, M in slot2:  # U[t, i, j, k] = v_k M_ij
            A += v[None, None, :] * np.einsum("tijk, tij -> tjk", W, M[c])
        out[c] = np.matmul(np.matmul(P2[c], A), Q12)
    return out


if __name__ == "__main__":
    from models import abp_model

    s = 10
    model = abp_model(lp=1.0, phi=0.5, eps=1e-2, V=Vexp, s=s)
    k1 = 1.0
    k2 = 1.0
    h3 = compute_3bod(k1, k2, model)
    plt.imshow(np.real(h3[s]))
    plt.show()

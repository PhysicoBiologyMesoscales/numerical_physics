"""2-body structure factor / correlation calculations.

For each k in k_arr the correlation matrix X(k) solves the Sylvester
equation  A X + X A^H = RHS(k)  with A = L(k).  For which="3b" the RHS
contains the collision integral over q wave vectors (2-body feedback
term and 3-body correlation h3).

The operator L(k), the 2-body source RHS and the 3-body vertex all come
from a ``Model`` (see ``models.py``): every function takes a single
``model`` argument instead of the loose ``(lp, phi, eps, V, ..., D)``
tuple, so a different microscopic model plugs in without copying this
file.  ``models.abp_model`` reproduces the original pairwise-force model.

Two discretizations of the q-integral:
- method="bipolar" (default, RHS_3b_bipolar): nodes (q, p = |k - q|)
  with both radii on a fixed 1D grid; exact angular cell measures
  absorb the Jacobian; all angular spectral data follows from the
  rotation identity L(q e^{ia}) = R L(q) R^H (R = diag(e^{i n a})), so
  the only eigendecompositions are the 1D table of wavenumber moduli
  (precompute_q_table, computed once) and one per k.
- method="polar" (RHS_3b, legacy): fixed polar grid centered at the
  origin.  Beware: it under-resolves the q ~ k feature of h(k - q)
  unless the alpha grid grows with k, and truncates the V(q) tail at
  k2abs[-1]; kept for cross-checks.
"""

from collections import namedtuple

from matplotlib import colors
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester, eig, inv
from scipy.special import jv
from joblib import Parallel, delayed
from tqdm import tqdm

# from StructFactor_MIPS.src import models
from struct_aux import Kern_erfc, Kern_gauss, dot, Vexp
from struct_3b import (
    compute_3bod,
    compute_3bod_slice0,
    compute_3bod_slice0_batch,
    compute_h,
    spectral_data,
)

# Fixed polar grid over which the 3-body term is integrated in RHS_S
K2ABS = np.linspace(0, 10, 200)
ALPHA = np.linspace(0, 2 * np.pi, 10, endpoint=False)

K2Grid = namedtuple("K2Grid", ["k2abs", "alpha", "k2", "L", "l", "P", "Q", "h"])


def precompute_k2_grid(model, k2abs=K2ABS, alpha=ALPHA, s=30):
    """Spectral data of L(k2) and h(k2) for every k2 on the polar grid.

    Independent of k, hence computed once per (model, s) and reused for
    all k values in compute_correlations.
    """
    N = 2 * s + 1
    k2 = (k2abs[:, None] * np.exp(1j * alpha[None, :])).ravel()
    M = k2.size
    l2 = np.empty((M, N), dtype=np.complex128)
    L2 = np.empty((M, N, N), dtype=np.complex128)
    P2 = np.empty_like(L2)
    Q2 = np.empty_like(L2)
    h2 = np.empty_like(L2)
    for i, k2i in enumerate(k2):
        L2[i], l2[i], P2[i], Q2[i], h2[i] = spectral_data(k2i, model, s=s)
    return K2Grid(k2abs, alpha, k2, L2, l2, P2, Q2, h2)


def RHS_3b(k, model, s=30, k2_grid=None, only_h=False):
    r"""RHS of the 2-body Sylvester equation of the h-hierarchy.

    (L h + h L^+)_{nm} = -2 mu V(k) k^2 d_{n0} d_{m0}
        - 2   \int d2q/(2pi)^2 mu (k.q) V(q) h_{nm}(k - q)
        - rho0 \int d2q/(2pi)^2 mu (k.q) V(q)
                   ( h3_{0nm}(q, k-q) + h3_{m0n}(q-k, -q) )

    with mu -> 1/eps and rho0 = 4 phi/pi (phi the packing fraction of
    diameter-1 disks).  Without the integral terms this reduces to
    RHS_2b.  The integration variable q runs over the fixed polar grid
    (k2_grid takes the output of precompute_k2_grid, which must have been
    built with the same model and s; it is built on the fly if not
    provided).

    The h3_{m0n}(q-k, -q) term integrates to the conjugate transpose of
    the h3_{0nm}(q, k-q) term (exactly, ring by ring on the grid, via
    the reflection identity h3(conj k1, conj k2) = h3(k1, k2) with all
    indices reversed), so only the first one is computed.  This
    requires real k and the uniform full-circle alpha grid.
    """
    if np.imag(k) != 0:
        raise ValueError("RHS_3b assumes real k (Hermitian shortcut)")
    model.require_3b()

    if k2_grid is None:
        k2_grid = precompute_k2_grid(model, s=s)

    N = 2 * s + 1
    if k2_grid.l.shape[-1] != N:
        raise ValueError("k2_grid was built with a different s")

    n_alpha = len(k2_grid.alpha)
    shape = (len(k2_grid.k2abs), n_alpha, N, N)
    integrand_h = np.zeros(shape, dtype=np.complex128)
    integrand_3 = np.zeros(shape, dtype=np.complex128)

    # data at k, fixed across the q loop
    _, lk, Pk, Qk, hk = spectral_data(k, model, s=s)
    # eigendata of L(-k) = L(k)^H: eigenvalues conj(lk), eigenvectors Qk^H.
    # This is the third leg of h3(q, k-q), whose k12 = q + (k-q) = k.
    lk_c, Pk_c, Qk_c = np.conj(lk), Qk.conj().T, Pk.conj().T

    for i, q in enumerate(k2_grid.k2):
        F = dot(k, q) * model.V(q)
        if F == 0.0:  # |q| = 0: no contribution
            continue

        # spectral data at k - q, shared by both integrands
        kq = k - q
        Lkq = model.L(kq)
        lkq, Pkq = eig(Lkq)
        Qkq = inv(Pkq)
        hkq = compute_h(Lkq, kq, model, s=s)

        iq, ia = divmod(i, n_alpha)

        if only_h:
            # integrand of the 2-body term only (no h3)
            integrand_h[iq, ia] = F * hkq
            continue
        # h3_{0nm}(q, k-q); third leg k12 = k
        h3A = compute_3bod(
            q,
            kq,
            model,
            s=s,
            L1=k2_grid.L[i],
            h1=k2_grid.h[i],
            l1=k2_grid.l[i],
            P1=k2_grid.P[i],
            Q1=k2_grid.Q[i],
            L2=Lkq,
            h2=hkq,
            l2=lkq,
            P2=Pkq,
            Q2=Qkq,
            l12=lk_c,
            P12=Pk_c,
            Q12=Qk_c,
            h12=hk,
        )

        integrand_h[iq, ia] = F * hkq
        integrand_3[iq, ia] = F * h3A[s]

    # d2q = |q| d|q| dalpha: periodic rectangle rule in alpha (uniform
    # grid over the full circle), trapezoid in |q|
    def _integrate(arr):
        I_alpha = arr.sum(axis=1) * (2 * np.pi / n_alpha)
        return np.trapz(k2_grid.k2abs[:, None, None] * I_alpha, k2_grid.k2abs, axis=0)

    I_h = _integrate(integrand_h)
    I_3 = _integrate(integrand_3)
    I = 2 * I_h + 4 * model.phi / np.pi * (I_3 + I_3.conj().T)

    return RHS_2b(k, model, s=s) - I * model.eps / (2 * np.pi) ** 2


def RHS_2b(k, model, s=None):
    """2-body source: L h + h L^H = RHS_2b, from ``model.rhs_2b``.

    ``s`` is unused (the truncation is baked into the model); kept for
    call-site compatibility.
    """
    return model.rhs_2b(k)


QTable = namedtuple("QTable", ["q", "l", "P", "Q", "h"])


def make_q_grid(
    q_grid_max, dq=0.05, q_break=15.0, dq_max=0.3, growth=1.05, q_min=1e-3, n_small=8
):
    """1D grid of wavenumber moduli for precompute_q_table.

    Log-spaced nodes from q_min to dq (the small-k structure of h,
    which lives at k ~ 1/lp and sharpens near the MIPS spinodal --
    lower q_min / raise n_small for large lp or when approaching the
    spinodal), uniform spacing dq up to q_break (interaction-scale
    structure: the peak of h and the period-pi oscillations of Vharm),
    then geometric coarsening (ratio `growth`, capped at dq_max) out to
    q_grid_max.
    """
    small = np.geomspace(q_min, dq, n_small, endpoint=False)
    fine = np.arange(dq, min(q_break, q_grid_max) + dq / 2, dq)
    q, step = fine[-1], dq
    tail = []
    while q < q_grid_max:
        step = min(step * growth, dq_max)
        q += step
        tail.append(q)
    return np.concatenate([small, fine, np.asarray(tail)])


def precompute_q_table(model, q_nodes, s=30):
    """Spectral data of L and h on a 1D grid of real positive wavenumbers.

    Data at any k = q e^{ia} follows exactly from the rotation identity

        L(q e^{ia}) = R L(q) R^H,   R = diag(e^{i n a}), n = -s..s,

    so l is unchanged, P -> R P, Q -> Q R^H, h -> R h R^H.  One 1D
    table therefore serves every angle (it replaces the 2D polar
    precompute_k2_grid).  Independent of k; computed once per
    (model, s).
    """
    N = 2 * s + 1
    M = len(q_nodes)
    l = np.empty((M, N), dtype=np.complex128)
    P = np.empty((M, N, N), dtype=np.complex128)
    Q = np.empty_like(P)
    h = np.empty_like(P)
    for i, qi in enumerate(q_nodes):
        _, l[i], P[i], Q[i], h[i] = spectral_data(qi, model, s=s)
    return QTable(np.asarray(q_nodes, dtype=float), l, P, Q, h)


def _trapz_weights(x):
    """Trapezoidal quadrature weights on the (possibly non-uniform) grid x."""
    w = np.zeros_like(x)
    if len(x) > 1:
        w[0] = (x[1] - x[0]) / 2
        w[-1] = (x[-1] - x[-2]) / 2
        w[1:-1] = (x[2:] - x[:-2]) / 2
    return w


def RHS_3b_bipolar(k, model, s=30, table=None, qmax=30.0, only_h=False, stride_3b=3):
    r"""Same RHS as RHS_3b, with the q-integral in bipolar coordinates.

    Integration nodes are (q, p = |k - q|) with both moduli on the
    fixed 1D grid of `table` (precompute_q_table): the sharp structures
    of the integrand -- V(q) on one side, h(k - q) and the h3 legs on
    the other -- are resolved by 1D grids independently of k, instead
    of chasing the q ~ k feature with an origin-centered polar grid.
    Spectral data at complex wave vectors comes from the table through
    the rotation identity, so the loop contains no eigendecompositions.

    For each q the p-integral uses the exact angular measure of each
    p-cell, w = Delta arccos((k^2 + q^2 - p^2) / 2kq), which integrates
    the 1/sqrt Jacobian edge singularities analytically; the integrand
    is evaluated at the p-nodes.  The integral is folded onto the upper
    half plane alpha in [0, pi] via the reflection identity
    h(conj u) = J h(u) J (J = index reversal), the lower half entering
    as I[::-1, ::-1]; the folded node set is conjugation-symmetric, so
    the Hermitian shortcut for the second h3 term (see RHS_3b) still
    applies.  The h3 part is integrated on every stride_3b-th node in
    both q and p (with cells merged accordingly); the h part always
    uses the full table density.

    Convergence knobs: qmax (V(q)-tail truncation; Vharm decays only as
    q^-5/2), the table's dq / q_break (must cover k + qmax and resolve
    h), and stride_3b.
    """
    if np.imag(k) != 0:
        raise ValueError("RHS_3b_bipolar assumes real k (Hermitian shortcut)")
    model.require_3b()
    k = float(np.real(k))
    if k < 0:
        raise ValueError("RHS_3b_bipolar assumes k >= 0")
    if k == 0.0:
        return RHS_2b(k, model, s=s)  # the integrand carries a (k.q) factor
    if table is None:
        table = precompute_q_table(model, make_q_grid(k + qmax), s=s)
    qs = table.q
    if k + qmax > qs[-1] + 1e-9:
        raise ValueError("q table too short: need q[-1] >= k + qmax")

    N = 2 * s + 1
    if table.l.shape[-1] != N:
        raise ValueError("q table was built with a different s")
    n_idx = np.arange(-s, s + 1)

    # data at k: third leg of h3(q, k - q), whose k12 = q + (k - q) = k.
    # Eigendata of L(-k) = L(k)^H: eigenvalues conj(lk), eigenvectors Qk^H.
    _, lk, Pk, Qk, hk = spectral_data(k, model, s=s)
    lk_c, Pk_c, Qk_c = np.conj(lk), Qk.conj().T, Pk.conj().T

    # p-cells: one cell per table node, edges at the midpoints
    edges = np.empty(len(qs) + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (qs[1:] + qs[:-1])
    edges[-1] = qs[-1] + 0.5 * (qs[-1] - qs[-2])

    iq_all = np.nonzero(qs <= qmax)[0]
    qq = qs[iq_all]
    wq = _trapz_weights(qq)
    is3 = np.zeros(len(iq_all), dtype=bool)
    is3[::stride_3b] = True
    wq3 = np.zeros(len(iq_all))
    wq3[is3] = _trapz_weights(qq[is3])

    Ih = np.zeros((N, N), dtype=np.complex128)
    I3 = np.zeros((N, N), dtype=np.complex128)

    def alpha_of(p, q):
        c = (k**2 + q**2 - p**2) / (2 * k * q)
        return np.arccos(np.clip(c, -1.0, 1.0))

    for a, iq in enumerate(iq_all):
        q = qq[a]
        Vq = model.V(q)
        if q == 0.0 or Vq == 0.0:
            continue
        pmin, pmax = abs(k - q), k + q

        # table nodes whose p-cell intersects the kinematic range
        jj = np.nonzero((edges[1:] > pmin) & (edges[:-1] < pmax))[0]
        if len(jj) == 0:
            continue

        # exact angular measure of each (clipped) p-cell
        w = alpha_of(np.clip(edges[jj + 1], pmin, pmax), q) - alpha_of(
            np.clip(edges[jj], pmin, pmax), q
        )
        alph = alpha_of(qs[jj], q)  # node angles (clamped in the edge cells)
        # k - q e^{i alpha} = p e^{i beta} with p = qs[jj]
        beta = np.arctan2(-q * np.sin(alph), k - q * np.cos(alph))
        F = k * q * np.cos(alph) * Vq  # dot(k, q e^{i alpha}) V(q)

        # h term, vectorized over the p-block: h(p e^{ib}) = R_b h(p) R_b^H
        ph = np.exp(1j * beta[:, None] * n_idx[None, :])
        Ih += (wq[a] * q) * np.einsum(
            "j, jn, jnm, jm -> nm", w * F, ph, table.h[jj], ph.conj()
        )

        if only_h or not is3[a]:
            continue

        # h3 term on the strided (q, p) sub-grid, merged cells.  The
        # kinematic p-window [|k - q|, k + q] is only 2 min(k, q) wide:
        # when it holds few table nodes (small k), striding it leaves
        # too coarse a quadrature, so the stride backs off to keep at
        # least ~32 p-nodes (full density on narrow windows).
        st = max(1, min(stride_3b, len(jj) // 32))
        jsel = jj[::st]
        e3 = np.empty(len(jsel) + 1)
        e3[0], e3[-1] = pmin, pmax
        e3[1:-1] = 0.5 * (qs[jsel[1:]] + qs[jsel[:-1]])
        w3 = alpha_of(e3[1:], q) - alpha_of(e3[:-1], q)
        al3 = alpha_of(qs[jsel], q)
        be3 = np.arctan2(-q * np.sin(al3), k - q * np.cos(al3))
        F3 = k * q * np.cos(al3) * Vq

        # stacked spectral data over the p-block: rotation identity as
        # diagonal phases on the table entries, one batched call
        ph1 = np.exp(1j * al3[:, None] * n_idx[None, :])
        ph2 = np.exp(1j * be3[:, None] * n_idx[None, :])
        h3s = compute_3bod_slice0_batch(
            q * np.exp(1j * al3),
            qs[jsel] * np.exp(1j * be3),
            model,
            s,
            h1=ph1[:, :, None] * ph1.conj()[:, None, :] * table.h[iq][None],
            h2=ph2[:, :, None] * ph2.conj()[:, None, :] * table.h[jsel],
            h12=hk,
            l1=np.broadcast_to(table.l[iq], ph1.shape),
            P1=ph1[:, :, None] * table.P[iq][None],
            Q1=table.Q[iq][None] * ph1.conj()[:, None, :],
            l2=table.l[jsel],
            P2=ph2[:, :, None] * table.P[jsel],
            Q2=table.Q[jsel] * ph2.conj()[:, None, :],
            l12=lk_c,
            P12=Pk_c,
            Q12=Qk_c,
        )
        I3 += np.einsum("t, tnm -> nm", wq3[a] * q * w3 * F3, h3s)

    # add the lower half plane (alpha -> -alpha): the integrand maps to
    # J (.) J with J the index reversal, since h(conj u) = J h(u) J and
    # h3(conj k1, conj k2) = h3(k1, k2) with all indices reversed.
    Ih = Ih + Ih[::-1, ::-1]
    I = 2 * Ih
    if not only_h:
        I3 = I3 + I3[::-1, ::-1]
        I = I + 4 * model.phi / np.pi * (I3 + I3.conj().T)

    return RHS_2b(k, model, s=s) + I * model.eps / (2 * np.pi) ** 2


def make_k_grid(kmax, rmax, dk_factor=8.0, kmin=1e-3, n_log=30):
    """k-grid for compute_correlations and the Hankel transforms of B_from_X.

    Uniform part with dk = pi / (dk_factor * rmax), so that the
    trapezoidal transform resolves the oscillations of J_n(k r) up to
    r = rmax (dk_factor points per half period; >= 6 recommended),
    preceded by a log-spaced part from kmin resolving the small-k
    structure of S(k) (which sharpens near the MIPS spinodal -- lower
    kmin / raise n_log when approaching it).  kmax sets the truncation
    ringing of the transform: artificial oscillations of period
    2 pi / kmax with amplitude ~ |X(kmax)|.  The grid never exceeds
    kmax (it ends at the largest multiple of dk below it), so a q table
    sized to kmax + qmax always covers it.
    """
    dk = np.pi / (dk_factor * rmax)
    lin = dk * np.arange(1, int(kmax / dk) + 1)
    return np.concatenate([np.geomspace(kmin, dk, n_log, endpoint=False), lin])


def _solve_one(
    k,
    model,
    which,
    k2_grid,
    s=30,
    only_h=False,
    method="bipolar",
    table=None,
    qmax=30.0,
    stride_3b=3,
):
    A = model.L(k)
    if which == "2b":
        rhs = RHS_2b(k, model)
    elif method == "bipolar":
        rhs = RHS_3b_bipolar(
            k,
            model,
            s=s,
            table=table,
            qmax=qmax,
            only_h=only_h,
            stride_3b=stride_3b,
        )
    else:
        rhs = RHS_3b(k, model, s=s, k2_grid=k2_grid, only_h=only_h)
    return solve_sylvester(A, A.conj().T, rhs)


def compute_correlations(
    model,
    k_arr,
    s=None,
    which="2b",
    parallel=False,
    only_h=False,
    method="bipolar",
    table=None,
    qmax=30.0,
    stride_3b=3,
):
    """Computes correlation matrices for all values in k_arr.

    ``model`` is a ``models.Model`` (e.g. ``abp_model(...)``) carrying
    the operator L, the 2-body source and, for which="3b", the collision
    vertex.  The operator is truncated to harmonics n = -s..s, so the
    returned array has shape (2s+1, 2s+1, len(k_arr)).  For which="3b",
    `method` selects the q-integral discretization ("bipolar" or the
    legacy "polar"; see the module docstring); a precomputed QTable can
    be passed via `table` (it must have been built with the SAME model
    and s -- it stores spectral data, not just a grid), otherwise one is
    built covering max(k_arr) + qmax.
    """
    if which not in ("2b", "3b"):
        raise ValueError(f"Unknown correlation type {which!r}; expected '2b' or '3b'")

    # the truncation lives on the model; an explicit s must agree with it
    if s is None:
        s = model.s
    elif s != model.s:
        raise ValueError(
            f"s={s} does not match model.s={model.s}; rebuild the model with this s"
        )

    k2_grid = None
    if which == "3b":
        model.require_3b()
        if method == "bipolar":
            if table is None:
                q_grid_max = float(np.max(np.real(k_arr))) + qmax
                table = precompute_q_table(model, make_q_grid(q_grid_max), s=s)
        elif method == "polar":
            k2_grid = precompute_k2_grid(model, s=s)
        else:
            raise ValueError(
                f"Unknown method {method!r}; expected 'bipolar' or 'polar'"
            )

    if parallel:
        # Works best for large k_arr; the precomputed grid arrays are
        # memmapped once by joblib instead of being pickled for every
        # task.  return_as="generator" yields results (in k_arr order)
        # as they complete, so tqdm can track progress.
        results = Parallel(n_jobs=-1, prefer="processes", return_as="generator")(
            delayed(_solve_one)(
                k,
                model,
                which,
                k2_grid,
                s=s,
                only_h=only_h,
                method=method,
                table=table,
                qmax=qmax,
                stride_3b=stride_3b,
            )
            for k in k_arr
        )
        Xs = list(results)
    else:
        Xs = [
            _solve_one(
                k,
                model,
                which,
                k2_grid,
                s=s,
                only_h=only_h,
                method=method,
                table=table,
                qmax=qmax,
                stride_3b=stride_3b,
            )
            for k in k_arr
        ]

    return np.stack(Xs, axis=-1)


def convert_to_real_space(X, k_arr, r_arr, n, m):
    N = X.shape[0]
    s = (N - 1) // 2
    # Create Bessel function arguments for the Hankel transforms
    kr = k_arr[:, None] * r_arr[None, :]
    J = jv(n - m, kr)
    integral = (
        np.trapz(k_arr[:, None] * J * X[s + n, s + m, :, None], k_arr, axis=0)
        / 2
        / np.pi
    )
    return 1j ** (n - m) * integral


def compute_A_from_X(X, k_arr, r_arr, alpha_arr):
    N = X.shape[0]
    s = (N - 1) // 2
    n_range = np.arange(-s, s + 1)
    harmonic_n = np.stack(
        [
            np.exp(-1j * n * alpha_arr[:, None])
            * convert_to_real_space(X, k_arr, r_arr, n, 0)[None, :]
            for n in n_range
        ]
    )
    A = np.real(np.sum(harmonic_n, axis=0, keepdims=False))
    return A


def compute_A_from_X_anisotropic(X_n, k_arr, alpha_arr, r_arr):
    """Real-space map A(n, r, alpha) from correlations on a polar k-grid.

    ``X_n[n, b, k]`` are correlations evaluated at the wave vectors
    ``k_arr[k] * exp(1j * alpha_arr[b])``; ``alpha_arr`` must be uniform
    on [0, 2 pi).  Direct quadrature of the 2D Fourier integral needs
    the dense (n, k, r, alpha, alpha') phase tensor, which does not fit
    in memory.  Instead, the Jacobi--Anger expansion
        e^{i k r cos(a - b)} = sum_p i^p J_p(k r) e^{i p (a - b)}
    diagonalizes the kernel in angular harmonics: an FFT over the
    k-angle, then one order-p Hankel transform per harmonic
    (|p| <= n_angle // 2, the harmonics the angular grid resolves).
    """
    n_angle = len(alpha_arr)
    # Harmonics over the k-angle: the uniform-grid quadrature
    # (1/2pi) int X e^{-i p b} db is exactly the DFT / n_angle.
    Xp = np.fft.fft(X_n, axis=1) / n_angle  # (n, p, k), p in fft order
    p_arr = np.fft.fftfreq(n_angle, d=1.0 / n_angle).astype(int)

    kr = k_arr[:, None] * r_arr[None, :]
    H = np.empty((n_angle, X_n.shape[0], len(r_arr)), dtype=np.complex128)
    for i, p in enumerate(p_arr):
        H[i] = 1j**p * np.trapz(
            k_arr[None, :, None] * Xp[:, i, :, None] * jv(p, kr)[None],
            k_arr,
            axis=1,
        )
    phases = np.exp(1j * p_arr[:, None] * alpha_arr[None, :])
    return np.einsum("pnr, pa -> nra", H, phases)


def pressure_k_grid(kcut=9.0, n=24):
    """Gauss-Legendre nodes and weights on [0, kcut] for the pressure.

    The pressure integrands are smooth in k and weighted by k^2 V(k),
    so a few GL nodes replace the dense Hankel grid of make_k_grid
    (which is sized for real-space transforms).  kcut must sit where
    k^2 V(k) has decayed: k ~ 9 for Vexp with r0 = 0.5; take it larger
    for slowly decaying potentials (Vharm).  Integrate with the
    returned weights (compute_P_Fourier(..., weights=w)), not trapz.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * kcut * (x + 1.0), 0.5 * kcut * w


def s_for_pressure(lp, kcut=9.0, s_min=6, margin=0):
    """Harmonic truncation for a pressure run at persistence length lp.

    Advection excites harmonics until the free relaxation n^2 beats the
    coupling lp k / 2, i.e. up to n* ~ sqrt(lp kcut / 2) (~ sqrt(4 Pe)
    for the Vexp cutoff kcut ~ 9); ``margin`` extra harmonics absorb
    the tail.  Validate at the largest Pe of a scan by bumping s.
    """
    return max(s_min, int(np.ceil(np.sqrt(lp * kcut / 2))) + margin)


def compute_P_Fourier(lp, phi, eps, k_arr, h0, h1, V, dV, s, weights=None):
    """Pressure from the Fourier-space correlations h0 = X_00, h1 = X_10.

    P = rho0 lp^2/2 + (rho0^2/2) eps \\int d2r V
        - (rho0^2/4) eps \\int d2r (r.grad V) h00(r)
        - (lp/2) rho0^2 eps \\int d2r grad V . h10(r)

    in code units (D_r = mu = 1, v0 = lp, potential eps V).  ``dV`` is
    the k-derivative of the Fourier potential (for Vexp with range r0:
    -r0^2 k V(k)).  With ``weights`` given, integrals over k_arr use
    sum(weights * f) (e.g. Gauss-Legendre from pressure_k_grid);
    otherwise trapezoid on k_arr.
    """
    rho0 = 4 * phi / np.pi
    if weights is None:
        integrate = lambda f: np.trapz(f, k_arr)
    else:
        integrate = lambda f: np.sum(weights * f)
    P_int_unif = eps * rho0**2 * V(0) / 2
    P_passive = (
        eps
        * rho0**2
        / 2
        * integrate(k_arr * (V(k_arr) + k_arr / 2 * dV(k_arr)) * h0)
        / 2
        / np.pi
    )
    P_active = 1j * (
        eps * rho0**2 * lp / 2 * integrate(k_arr**2 * V(k_arr) * h1) / 2 / np.pi
    )
    P = rho0 * lp**2 / 2 + P_int_unif + P_passive + P_active
    return P, P_int_unif, P_passive, P_active


def compute_P(lp, phi, eps, r_arr, dV, g0, g1):
    """Real-space counterpart of compute_P_Fourier.

    ``dV`` is dV/dr (e.g. dVexp_r); g0, g1 are the real parts of
    convert_to_real_space(X, k, r, 0, 0) and (1, 0) (the vector-valued
    harmonic is the radial field h10(r) = g1(r) e_r).
    """
    rho0 = 4 * phi / np.pi  # phi = pi rho0 (a/2)^2, a = 1 the diameter
    P = rho0 * (
        lp**2 / 2
        - np.pi * rho0 / 2 * eps * np.trapz(r_arr**2 * dV(r_arr) * (1 + g0), r_arr)
        - np.pi * rho0 * eps * lp * np.trapz(r_arr * dV(r_arr) * g1, r_arr)
    )
    return P


def compute_B(
    model,
    alpha,
    r_arr,
    Npoints_k=100,
    kmax=10,
    which="2b",
    s=None,
    only_h=False,
):
    """Computes correlation function from the reference frame of the 1st particle"""
    if s is None:
        s = model.s
    # Compute full correlation matrix
    k_arr = np.linspace(0, kmax, Npoints_k)
    G = compute_correlations(model, k_arr, s=s, which=which, only_h=only_h)

    # Bessel functions for radial Fourier transform
    kr = k_arr[:, None] * r_arr[None, :]
    jint = np.stack([1j**n * jv(n, kr) for n in range(s + 1)], axis=0)

    # Compute correlation functions in real space
    C = G[s:, s]
    gn = np.real(
        np.trapz(np.abs(k_arr)[None, :, None] * C[..., None] * jint, k_arr, axis=1)
        / 2
        / np.pi
    )

    # Resum the Fourier series
    n = np.arange(1, s + 1)
    B = gn[0, :, None] + 2 * np.sum(
        gn[1:, :, None] * np.cos(n[:, None, None] * alpha[None, None, :]), axis=0
    )

    return B


def B_from_X(X, k_arr, alpha, r_arr):
    """Computes correlation function from the reference frame of the 1st particle.

    The truncation order is read off the matrix: X has shape
    (2s+1, 2s+1, len(k_arr)).
    """
    s = (X.shape[0] - 1) // 2
    # Bessel functions for radial Fourier transform
    kr = k_arr[:, None] * r_arr[None, :]
    jint = np.stack([1j**n * jv(n, kr) for n in range(s + 1)], axis=0)

    # Compute correlation functions in real space
    C = X[s:, s]
    gn = np.real(
        np.trapz(np.abs(k_arr)[None, :, None] * C[..., None] * jint, k_arr, axis=1)
        / 2
        / np.pi
    )

    # Resum the Fourier series
    n = np.arange(1, s + 1)
    B = gn[0, :, None] + 2 * np.sum(
        gn[1:, :, None] * np.cos(n[:, None, None] * alpha[None, None, :]), axis=0
    )

    return B


def draw_B(B, r_arr, alpha):
    r, th = np.meshgrid(r_arr, alpha)
    plt.subplot(projection="polar")
    plt.pcolormesh(th, r, B.T)
    plt.show()


def plot_struct_2b_and_corrections():
    import time

    from models import abp_model

    list_s = [5, 10, 20, 30]
    list_Dr = [10.0, 1.0, 0.1, 0.01]
    fig, ax = plt.subplots(
        1,
        len(list_s),
        subplot_kw={"projection": "polar"},
        layout="constrained",
        figsize=(len(list_s) * 5, 5),
    )

    r_max = 10.0  # real-space extent of the correlation maps
    kmax = 12.0  # transform truncation: ringing has period 2pi/kmax, ~|X(kmax)|
    qmax = 4.0  # V(q)-tail cutoff of the collision integral
    s = 30  # operator truncation: harmonics n = -s..s

    for i, (s, D_r) in enumerate(zip(list_s, list_Dr)):
        # Physical parameters, adimensionalized with D_r = 0.01
        eps = 1.0 / D_r
        lp = 10.0 / D_r  # h has small-k structure down to k ~ 1/lp
        D = 0.0 / D_r
        phi = 0.04
        V = Vexp
        model = abp_model(lp=lp, phi=phi, eps=eps, V=V, D=D, s=s)

        # Outer k grid: dk = pi / (8 r_max) resolves J_n(k r) out to r_max;
        # the log section covers the small-k structure at k ~ 1/lp.
        k_arr = make_k_grid(kmax, r_max, dk_factor=8.0, kmin=0.1 / lp, n_log=40)
        # q table refined at small q for lp = 1000 (structure at k ~ 1/lp);
        # sized from the actual grid so it always covers max(k) + qmax
        t0 = time.perf_counter()
        table = precompute_q_table(
            model,
            make_q_grid(k_arr.max() + qmax, q_min=1e-4, n_small=30),
            s=s,
        )
        print(f"q table: {len(table.q)} nodes, {time.perf_counter() - t0:.1f} s")

        t0 = time.perf_counter()
        X2b = compute_correlations(model, k_arr, s=s, which="2b", parallel=True)
        print(f"2b solve: {len(k_arr)} k values, {time.perf_counter() - t0:.1f} s")

        t0 = time.perf_counter()
        X3b = compute_correlations(
            model,
            k_arr,
            s=s,
            which="3b",
            parallel=True,
            only_h=False,
            table=table,
            qmax=qmax,
        )
        print(f"3b (only_h) solve: {time.perf_counter() - t0:.1f} s")

        np.savez_compressed(
            f"struct_2b_vs_3b_Dr{D_r}.npz",
            k_arr=k_arr,
            X2b=X2b,
            X3b=X3b,
            lp=lp,
            phi=phi,
            eps=eps,
            D=D,
            qmax=qmax,
        )

        # Correlation maps in the frame of particle 1
        r_arr = np.linspace(0, r_max, 201)
        alpha = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        r, th = np.meshgrid(r_arr, alpha)
        B2b = B_from_X(X2b, k_arr, alpha, r_arr) / (2 * np.pi)
        B3b = B_from_X(X3b, k_arr, alpha, r_arr)

        vmax = np.abs(B2b).max()
        norm = colors.SymLogNorm(
            linthresh=1e-3, linscale=0.2, vmin=-0.1, vmax=0.1, base=10
        )
        msh = ax[0, i].pcolormesh(
            th, r, B2b.T, norm=norm, cmap="RdBu_r", rasterized=True
        )
        ax[0, i].set_title(f"Pe={lp:.1f}")
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([0, 10])
        msh = ax[1, i].pcolormesh(
            th, r, B3b.T, norm=norm, cmap="RdBu_r", rasterized=True
        )
        ax[1, i].set_title("3-body feedback")
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([0, 10])
    plt.colorbar(msh, ax=ax)
    fig.savefig(f"struct_2b_Pe_eps{eps:.1f}_phi{phi:.1f}_Pe{lp:.1f}.svg", dpi=300)
    plt.show()


def plot_corr_vicsek_isotropic():
    import matplotlib.pyplot as plt
    from models import vicsek_model
    from struct_aux import Kern_gauss, Kern_constant

    matplotlib.rcParams.update({"font.size": 22})

    D_r = 0.01
    lp = 0.1 / D_r
    phi = 0.1
    gamma_thresh = D_r * np.pi / 2 / phi
    list_gamma = gamma_thresh * np.array([0.1, 0.3, 0.7, 0.95])
    list_lp = [0.02 / D_r, 0.05 / D_r, 1.0 / D_r]
    fig, ax = plt.subplots(
        len(list_lp),
        len(list_gamma),
        figsize=(4 * len(list_gamma), 4 * len(list_lp)),
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )
    r_max = 10.0  # real-space extent of the correlation maps
    kmax = 10.0  # transform truncation: ringing has period 2pi/kmax, ~|X(kmax)|
    # qmax = 4.0  # V(q)-tail cutoff of the collision integral
    s = 30  # operator truncation: harmonics n = -s..s
    k_arr = make_k_grid(kmax, r_max, dk_factor=8.0, kmin=0.1 / lp, n_log=40)
    r_arr = np.linspace(0, r_max, 201)
    alpha = np.linspace(0, 2 * np.pi, 201, endpoint=False)
    rr, aa = np.meshgrid(r_arr, alpha)

    for i, _gamma in enumerate(list_gamma):
        for j, _lp in enumerate(list_lp):
            gamma = _gamma / D_r
            lp = _lp
            print(f"Growth rate {gamma/2*phi*4/np.pi-1.0:.3f}")
            model = vicsek_model(lp=lp, phi=phi, gamma=gamma, Kern=Kern_gauss, s=s)
            X = compute_correlations(model, k_arr, s=s, which="2b", parallel=True)
            A = compute_A_from_X(X, k_arr, r_arr, alpha) / 2 / np.pi
            norm = colors.SymLogNorm(
                linthresh=1e-3, linscale=0.2, vmin=-1.0, vmax=1.0, base=10
            )
            # norm = colors.CenteredNorm(vmin=-0.1, vmax=0.5)
            msh = ax[j, i].pcolormesh(
                aa, rr, A, cmap="RdBu_r", norm=norm, rasterized=True
            )
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])

    plt.colorbar(msh, ax=ax)


def plot_struct_anisotropic():
    import matplotlib.pyplot as plt
    from models import vicsek_anisotropic_model
    from struct_aux import Kern_gauss

    D_r = 0.01
    lp = 0.1 / D_r
    phi = 0.04
    gamma = 10.0 / D_r

    model = vicsek_anisotropic_model(lp=lp, phi=phi, gamma=gamma, Kern=Kern_gauss)
    r_max = 10.0
    kmax = 10.0
    s = 30
    k_arr = make_k_grid(kmax, r_max, dk_factor=8.0, kmin=0.1 / lp, n_log=40)
    r_arr = np.linspace(0, r_max, 201)
    n_angle = 51
    alpha = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    r_vec_arr = r_arr[None, :] * np.exp(1j * alpha[:, None])
    k_vec_arr = k_arr[None, :] * np.exp(1j * alpha[:, None])
    _orig_shape = k_vec_arr.shape
    k_vec_arr = k_vec_arr.reshape(-1)

    X = compute_correlations(model, k_vec_arr, s=s, which="2b", parallel=True)

    # Reshape to (n, m, alpha, k)
    X = X.reshape((X.shape[0], X.shape[1]) + _orig_shape)

    integral = compute_A_from_X_anisotropic(X[:, s - 1], k_arr, alpha, r_arr)

    theta = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    eith = np.exp(-1j * np.arange(-s, s + 1)[:, None] * theta[None, :])
    A = 1 / (2 * np.pi) * np.einsum("nrp, nt -> rpt", integral, eith)
    A = np.real(A)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6, 6),
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )
    aa, rr = np.meshgrid(alpha, r_arr)
    msh = ax.pcolormesh(aa, rr, A[..., 0], cmap="RdBu_r", rasterized=True)
    plt.colorbar(msh)


def _vicsek_growth_rate_correction(gamma_factor, lp, phi, s, k_arr):
    """One (gamma, lp) point of the __main__ scan: loop-correction integral.

    Module-level so joblib worker processes can pickle it on Windows.
    """
    from models import vicsek_model

    gamma_thresh = np.pi / 2 / phi
    gamma = gamma_factor * gamma_thresh

    model = vicsek_model(lp=lp, phi=phi, gamma=gamma, Kern=Kern_gauss, s=s)
    sigma_arr = np.zeros(len(k_arr), dtype=np.complex128)
    for l, k in enumerate(k_arr):
        L = model.L(k)
        RHS = model.rhs_2b(k)
        h = solve_sylvester(L, L.conj().T, RHS)

        RHS_deriv = model.rhs_deriv(k, h)
        X = solve_sylvester(L, L.conj().T, RHS_deriv)
        sigma_arr[l] = -X[s + 1, s] + X[s + 2, s + 1]

    return np.trapz(k_arr * model.Kern(k_arr) * sigma_arr, k_arr) / 2 / np.pi


def plot_vicsek_growth_rate_correction():
    from itertools import product

    s = 30

    D_r = 0.1
    lp = 0.25 / D_r
    gamma = 1.0 / D_r
    phi = 0.5

    gamma_arr = np.linspace(0.8, 1.2, 10)
    lp_arr = np.linspace(0.1, 2.0, 10) / D_r

    k_arr = np.linspace(0, 8, 200)

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_vicsek_growth_rate_correction)(gamma, lp, phi, s, k_arr)
        for gamma, lp in product(gamma_arr, lp_arr)
    )
    delta_sigma_arr = np.array(results).reshape(len(gamma_arr), len(lp_arr))

    # gamma_thresh = np.pi / 2 / phi_arr

    fig, ax = plt.subplots(layout="constrained")
    msh = plt.pcolormesh(
        gamma_arr,
        lp_arr,
        np.real(delta_sigma_arr - 1 / gamma_arr[:, None] + 1).T,
        cmap="RdBu_r",
        rasterized=True,
        vmax=1.0,
        vmin=-1.0,
    )
    # ax.set_xticks([0.8, 0.9, 1.0])
    fig.colorbar(msh, ax=ax)
    # fig.savefig("../figures/vicsek/growth_rate_renormalization_phi.svg", dpi=300)


def plot_turning_away_corr_map():
    from models import turning_away_model
    from itertools import product

    s = 30
    V = Vexp
    Kern = Kern_erfc
    D_r = 0.1
    lp = 0.5 / D_r
    phi = 0.1
    eps = 20 * lp
    gamma = 3.0 / D_r

    model = turning_away_model(
        lp=lp, phi=phi, eps=eps, gamma=gamma, V=V, Kern=Kern, s=s
    )

    kmax = 25.0
    rmax = 4.0
    r_arr = np.linspace(0, rmax, 101)
    phi_arr = np.linspace(0, 2 * np.pi, 101, endpoint=False)
    k_arr = make_k_grid(kmax, rmax, dk_factor=12.0, kmin=0.1 / lp, n_log=40)
    X = compute_correlations(model, k_arr, s=s, which="2b", parallel=True)

    n = np.arange(-s, s + 1)

    jv_arr = np.zeros((2 * s + 1, len(k_arr), len(r_arr)), dtype=np.complex128)
    for i, n1 in enumerate(tqdm(range(2 * s + 1))):
        kr = k_arr[:, None] * r_arr[None, :]
        jv_arr[i] = jv(n1, kr)

    choose_jv = jv_arr[abs(n[None, :] - n[:, None])]
    integral = (
        np.trapz(
            k_arr[None, None, :, None]
            * (-1j) ** abs(n[:, None, None, None] - n[None, :, None, None])
            * X[:, :, :, None]
            * choose_jv,
            k_arr,
            axis=2,
        )
        / 2
        / np.pi
    )
    fig, ax = plt.subplots(
        2, 2, subplot_kw={"projection": "polar"}, layout="constrained", figsize=(10, 10)
    )
    for i, theta2 in enumerate([np.pi / 6, 2 * np.pi / 3, -2 * np.pi / 3, -np.pi / 6]):
        exp_factor = np.exp(
            1j * n[:, None, None] * phi_arr[None, None, :]
            + 1j * n[None, :, None] * (theta2 - phi_arr)
        )

        int_realspace = np.einsum("nmp, nmr -> rp", exp_factor, integral)

        l, j = np.unravel_index(i, (2, 2))

        msh = ax[l, j].pcolormesh(
            phi_arr,
            r_arr,
            np.real(int_realspace) + 1,
            vmin=0.0,
            vmax=4.0,
            cmap="jet",
            rasterized=True,
        )
        ax[l, j].set_xticks([])
        ax[l, j].set_yticks([])
    fig.colorbar(msh, ax=ax)
    fig.savefig("../figures/turning_away_corr_map_no_coll.svg", dpi=300)


def _turning_away_growth_rate(gamma, lp, phi, V, Kern, s, k_arr):
    """One (gamma, lp) point of the __main__ scan: loop-correction integral.

    Module-level so joblib worker processes can pickle it on Windows.
    """
    from models import turning_away_model

    eps = 20 * lp

    model = turning_away_model(
        lp=lp, phi=phi, eps=eps, gamma=gamma, V=V, Kern=Kern, s=s
    )
    sigma_arr = np.zeros(len(k_arr), dtype=np.complex128)
    for l, k in enumerate(k_arr):
        L = model.L(k)
        RHS = model.rhs_2b(k)
        h = solve_sylvester(L, L.conj().T, RHS)

        RHS_deriv = model.rhs_deriv(k, h)
        X = solve_sylvester(L, L.conj().T, RHS_deriv)
        sigma_arr[l] = 1j * k * (X[s, s] - X[s + 2, s])

    return np.trapz(k_arr * model.Kern(k_arr) * sigma_arr, k_arr) / 2 / np.pi


def plot_turning_away_growth_rate():
    from models import turning_away_model
    from itertools import product

    s = 8
    V = Vexp
    Kern = Kern_erfc
    D_r = 0.1
    # eps = 4.0 / D_r
    gamma = 6.5 / D_r

    lp_arr = np.linspace(0.1, 1.0, 10) / D_r
    phi_arr = np.linspace(0.01, 0.25, 9)

    kmax = 14.0
    rmax = 10.0
    k_grid = make_k_grid(kmax, rmax, dk_factor=8.0, kmin=0.1 / 100.0, n_log=40)

    result = Parallel(n_jobs=-1, verbose=10)(
        delayed(_turning_away_growth_rate)(gamma, lp, phi, V, Kern, s, k_grid)
        for lp, phi in product(lp_arr, phi_arr)
    )
    result = np.array(result).reshape(len(lp_arr), len(phi_arr))

    norm = colors.SymLogNorm(linthresh=1e-2, linscale=0.2, vmin=-0.5, vmax=0.5, base=10)
    fig, ax = plt.subplots(layout="constrained")
    msh = ax.pcolormesh(
        lp_arr,
        phi_arr,
        np.real(gamma * phi_arr[None, :] * 2 / np.pi * result - 1).T,
        cmap="RdBu_r",
        rasterized=True,
        norm=norm,
    )
    fig.colorbar(msh, ax=ax)


def plot_pressure_scan(eps_factor):
    from matplotlib import colors
    from models import abp_model
    from itertools import product

    which = "2b"  # "2b" for the closure without the collision integral
    Dr = 0.1
    lp_arr = np.logspace(np.log10(0.5), np.log10(10.0), 30) / Dr
    phi_max = 0.7
    phi_arr = np.linspace(0.01, phi_max, 30)
    # lp_arr = np.array([5.0])
    # phi_arr = np.array([0.2])
    V = Vexp
    dV = lambda k: -0.25 * k * Vexp(k)  # dV/dk of Vexp with r0 = 0.5
    kcut = 6.0  # pressure-integral truncation: k^2 V(k) decayed for Vexp
    qmax = 6.0  # collision-integral V(q) tail cutoff (Gaussian for Vexp)
    stride_3b = 5
    k_arr, wk = pressure_k_grid(kcut=kcut, n=24)
    # eps_factor = 4.0  # eps = eps_factor * lp, for the Vexp potential with r0 = 0.5

    def pressure_point(lp, phi, eps_factor):
        eps = eps_factor * lp
        s = s_for_pressure(lp, kcut=kcut, margin=-1)
        model = abp_model(lp=lp, phi=phi, eps=eps, V=V, D=0.0, s=s)
        table = (
            precompute_q_table(model, make_q_grid(kcut + qmax), s=s)
            if which == "3b"
            else None
        )
        X = compute_correlations(
            model,
            k_arr,
            s=s,
            which=which,
            table=table,
            qmax=qmax,
            stride_3b=stride_3b,
        )
        return compute_P_Fourier(
            lp, phi, eps, k_arr, X[s, s], X[s + 1, s], V, dV, s, weights=wk
        )

    result = Parallel(n_jobs=-1, verbose=10)(
        delayed(pressure_point)(lp, phi, eps_factor)
        for lp, phi in product(lp_arr, phi_arr)
    )
    P_arr, P_int_arr, P_passive_arr, P_active_arr = np.real(
        np.array(result).T.reshape(4, len(lp_arr), len(phi_arr))
    )
    np.savez_compressed(
        f"pressure_scan_{which}_eps{eps_factor}_phimax{phi_max}.npz",
        lp_arr=lp_arr,
        phi_arr=phi_arr,
        P=P_arr,
        P_int=P_int_arr,
        P_passive=P_passive_arr,
        P_active=P_active_arr,
    )
    P0 = phi_arr[None, :] * (lp_arr[:, None] ** 2) * 2 / np.pi

    norm_log_press = colors.SymLogNorm(
        vmin=0, vmax=1.0, linthresh=1e-1, linscale=0.2, base=10, clip=False
    )
    norm_log_press_neg = colors.SymLogNorm(
        vmin=min(-1.0, np.min(P_active_arr / P0)),
        vmax=0.0,
        linthresh=1e-1,
        linscale=0.2,
        base=10,
    )

    norm_log_sym = colors.SymLogNorm(
        linthresh=1e-1, linscale=0.2, base=10, vmin=-100, vmax=100
    )
    cmap = plt.get_cmap("inferno").with_extremes(
        under="white",  # < vmin
        bad="grey",  # NaN / masked
    )

    fig, ax = plt.subplots(1, 4, figsize=(20, 4), layout="constrained")
    msh1 = ax[0].pcolormesh(
        phi_arr,
        lp_arr,
        P_arr / (lp_arr[:, None] ** 2) * 2,
        cmap=cmap,
        rasterized=True,
        norm=norm_log_press,
    )
    msh2 = ax[1].pcolormesh(
        phi_arr,
        lp_arr,
        (P_int_arr + P_passive_arr) / (lp_arr[:, None] ** 2) * 2,
        cmap="inferno",
        rasterized=True,
        norm=norm_log_press,
    )
    msh3 = ax[2].pcolormesh(
        phi_arr,
        lp_arr,
        P_active_arr / (lp_arr[:, None] ** 2) * 2,
        cmap="inferno_r",
        rasterized=True,
        norm=norm_log_press_neg,
    )
    msh4 = ax[3].pcolormesh(
        phi_arr[1:],
        lp_arr,
        np.diff(P_arr, axis=1) / np.diff(phi_arr)[None, :],
        cmap="RdBu_r",
        rasterized=True,
        norm=norm_log_sym,
    )
    for a, title in zip(ax, ["P", "PD", "PI", "dP/dphi"]):
        a.set_title(title)
        a.semilogy()
        a.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    fig.colorbar(msh1, ax=ax[0])
    fig.colorbar(msh2, ax=ax[1])
    fig.colorbar(msh3, ax=ax[2])
    fig.colorbar(msh4, ax=ax[3])
    # fig.savefig(
    #     f"../figures/pressure/pressure_scan_{which}_eps{eps_factor}_phimax{phi_max}.svg",
    #     dpi=300,
    # )


if __name__ == "__main__":
    for eps_factor in [3.5]:
        plot_pressure_scan(eps_factor)

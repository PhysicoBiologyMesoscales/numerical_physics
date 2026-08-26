r"""Real-space 3-body correlation B(r, s, theta1, theta2) from h3.

    B(r, s, th1, th2) = int dth3/2pi < psi(0, th1) psi(r, th2) psi(s, th3) >

with psi the microscopic phase-space density (uniform value rho/2pi).
Particle 1 sits at the origin, particle 2 at r (heading th2), particle 3
at s (heading integrated out -> harmonic 0 on that leg).  In the c_n
conventions of ``abp_structure_factor`` (c_n(k) = sum_j e^{i n theta_j}
e^{+i k.r_j}, S = delta + rho X validated against struct_2b), the
harmonics of the 3-point function pair as

    h3_{nml}(k1, k2)  <->  < c_n(k1) c_m(k2) conj(c_l(k1+k2)) >:

n rides the k1 leg, m the k2 leg, l the conjugated k1+k2 leg -- the
conjugated leg is the reference particle, so k1 (resp. k2) is Fourier-
conjugate to r (resp. s) measured FROM particle 1.  Everything returned
here is dimensionless (g-normalized):

    B = (rho/2pi)^3 * [ B3_disconnected + B3_connected ],

B3_disconnected carrying 1 + the three 2-body terms (from the pair
correlation X of struct_2b) and B3_connected the irreducible 3-body
part (from h3 of struct_3b).  The bracket -> 1 when all three particles
are far apart.

Geometry: r = r u_x (all angles are measured from r_hat), s = s
e^{i phi_s}, headings th1, th2 from the same axis.  With G3 the inverse
transform of the middle-index-0 slice (theta3 integrated),

    B3_connected = sum_{n,l} e^{i(l th1 - n th2)} G3_{nl}(r, s vec),

    G3_{nl} = (-i)^{n-l}/(2pi)^2 sum_p (-1)^p e^{-i p phi_s}
              int k1 dk1 k2 dk2 J_{n-l+p}(k1 r) J_p(k2 s) h3p_{nl}(k1,k2),

    h3p_{nl}(k1,k2) = int dbeta/2pi e^{i p beta} h3_{n,0,l}(k1, k2 e^{i beta}).

The k1-angle integral is NOT a single Bessel term: derotating k1 onto
the real axis (joint rotation identity, phase e^{i(n-l)a}) drags the
angles of both k2 and s along, which couples the two angular integrals;
expanding e^{-i k2.s} in the relative angle beta = arg k2 - arg k1
produces the p-sum above (checked against brute-force 2D angular
quadrature to 3e-13).

Workflow (all the expensive solves live in step 1, reused for every
(r, s, theta) afterwards):

1. table = precompute_B3_table(lp, phi, eps, V, k_arr, ...)   [slow]
2. B3_connected(table, r, s, phi_s, th1, th2)     -> product grids
   B3_connected_mesh(table, r, svec, th1, th2)    -> dense maps at one r
   B3_connected_points(table, rvec, svec, th1, th2) -> scattered points
3. X2 = compute_correlations(..., which="2b");
   B3_disconnected(...) / B3_disconnected_points(...)

Cost/quality knobs: k_arr (Hankel resolution, use struct_2b.make_k_grid
with dk ~ pi/(6 rmax)), n_beta (resolution of the relative angle; h3 is
smooth in beta away from sharp small-|k12| structure), pmax (kept
relative-angle harmonics; J_p(k2 s) needs decay of h3p by |p| ~ pmax,
raise together with n_beta for large |s|), s_out (harmonics kept in the
output; the solver truncation s can be larger).  Table memory is
nk^2 (2 pmax + 1) (2 s_out + 1)^2 * 16 bytes.
"""

import functools
import time
from collections import namedtuple

from matplotlib import colors
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import CubicSpline
from scipy.linalg import eig, inv
from scipy.special import jv
from tqdm import tqdm

from struct_3b import compute_3bod_slice0, compute_h, spectral_data
from models import abp_model

B3Table = namedtuple(
    "B3Table", ["k_arr", "p_arr", "H", "s", "s_out", "lp", "phi", "eps", "D"]
)

# k12 = k1 + k2 e^{i beta} vanishes at beta = pi on the k1 = k2 diagonal,
# where the h(k12) solve is 0/0 (L(0) has a zero eigenvalue while its
# source ~ |k|^2).  h has a finite k -> 0 limit, evaluated at |k| = K12_TOL.
K12_TOL = 1e-3


def _timed(func):
    """Print the wall-clock time of each call."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter() - t0:.2f} s")
        return out

    return wrapper


def _trapz_kweights(k_arr):
    """Trapezoid weights times k: int k dk f(k) ~ sum wk f."""
    w = np.zeros_like(k_arr)
    w[0] = (k_arr[1] - k_arr[0]) / 2
    w[-1] = (k_arr[-1] - k_arr[-2]) / 2
    w[1:-1] = (k_arr[2:] - k_arr[:-2]) / 2
    return w * k_arr


def _h_from_eig(l12, P12, Q12, k12, eps, V, s):
    """h(+k12) from the eigendata of L(-k12) = L(k12)^H, O(N^2).

    L(k12) = Q12^H diag(conj l12) P12^H, and the source of compute_h is
    rank-1 (only the (s, s) entry), so the Sylvester solve reduces to
    one outer product in the eigenbasis.
    """
    c = -2 * np.abs(k12) ** 2 * V(k12) * eps
    u = np.conj(P12[s, :])
    Y = c * np.outer(u, u.conj()) / (np.conj(l12)[:, None] + l12[None, :])
    return Q12.conj().T @ Y @ Q12


def _b3_row(i1, k_arr, spec, model, s, n_beta, p_sel, crop):
    """One k1-row of the table: h3p_{nl}(k1, k2) for all k2, p kept.

    The middle-index-0 slice comes from the leg-swap symmetry
    h3_{m,0,l}(k1, k2) = h3_{0,m,l}(k2, k1) so compute_3bod_slice0 runs
    with swapped arguments; the k2 e^{i beta} leg data follows from the
    real-k2 table through the rotation identity, the third leg
    (k12 = k1 + k2 e^{i beta}) needs one eig per beta node.  beta is
    only computed on [0, pi]: h3 at -beta is the (n, l)-reversed value
    (mirror identity).
    """
    N = 2 * s + 1
    n_idx = np.arange(-s, s + 1)
    beta = 2 * np.pi * np.arange(n_beta) / n_beta
    nb_half = n_beta // 2
    k1 = k_arr[i1]
    l1, P1, Q1, h1 = spec[i1]

    M = np.empty((len(k_arr), n_beta, N, N), dtype=np.complex128)
    for i2, k2 in enumerate(k_arr):
        l2, P2, Q2, h2 = spec[i2]
        for jb in range(nb_half + 1):
            ph = np.exp(1j * n_idx * beta[jb])
            k2b = k2 * np.exp(1j * beta[jb])
            k12 = k1 + k2b
            L12 = model.L(-k12)
            l12, P12 = eig(L12)
            Q12 = inv(P12)
            if np.abs(k12) >= K12_TOL:
                h12 = _h_from_eig(l12, P12, Q12, k12, model.eps, model.V, s)
            else:
                k12r = K12_TOL if k12 == 0 else K12_TOL * k12 / np.abs(k12)
                h12 = compute_h(model.L(k12r), k12r, model)
            M[i2, jb] = compute_3bod_slice0(
                k2b,
                k1,
                model,
                s,
                (ph[:, None] * ph.conj()[None, :]) * h2,
                h1,
                h12,
                l2,
                ph[:, None] * P2,
                Q2 * ph.conj()[None, :],
                l1,
                P1,
                Q1,
                l12,
                P12,
                Q12,
            )
        for jb in range(nb_half + 1, n_beta):
            M[i2, jb] = M[i2, n_beta - jb][::-1, ::-1]

    # h3p = (1/2pi) int dbeta e^{i p beta} h3(beta): ifft gives
    # (1/n) sum_j e^{+2pi i j p / n} M_j in fftfreq order.
    Hp = np.fft.ifft(M, axis=1)[:, p_sel]
    return Hp[:, :, crop, :][:, :, :, crop]


def precompute_B3_table(
    lp,
    phi,
    eps,
    V,
    k_arr,
    s=10,
    D=0.0,
    n_beta=48,
    pmax=16,
    s_out=6,
    parallel=False,
):
    """All k-space data B3_connected needs: h3p[a, b, p, n, l].

    a, b index k_arr (the k1 = r leg and k2 = s leg share the grid,
    which must not contain 0), p the kept relative-angle harmonics
    (|p| <= pmax <= n_beta // 2), n, l = -s_out..s_out.  The solver
    truncation s only affects accuracy (and cost: one eig of size
    2s+1 per (k1, k2, beta) node).
    """
    k_arr = np.asarray(k_arr, dtype=float)
    if np.any(k_arr <= 0):
        raise ValueError("k_arr must be strictly positive")
    s_out = min(s_out, s)
    pmax = min(pmax, n_beta // 2)
    p_all = np.rint(np.fft.fftfreq(n_beta) * n_beta).astype(int)
    p_sel = np.nonzero(np.abs(p_all) <= pmax)[0]
    crop = slice(s - s_out, s + s_out + 1)

    # the real-space 3-body transform is inherently the MIPS pairwise model
    model = abp_model(lp, phi, eps, V, D, s=s)
    spec = [spectral_data(k, model, s=s)[1:] for k in k_arr]

    args = (k_arr, spec, model, s, n_beta, p_sel, crop)
    if parallel:
        rows = Parallel(n_jobs=-1, prefer="processes", return_as="generator")(
            delayed(_b3_row)(i1, *args) for i1 in range(len(k_arr))
        )
        rows = list(tqdm(rows, total=len(k_arr)))
    else:
        rows = [_b3_row(i1, *args) for i1 in tqdm(range(len(k_arr)))]

    return B3Table(
        k_arr, p_all[p_sel], np.stack(rows, axis=0), s, s_out, lp, phi, eps, D
    )


def _resum_theta(G, theta1, theta2, s_out):
    """Real part of sum_{n,l} G[..., n, l] e^{i l th1} e^{-i n th2}.

    Output shape: G.shape[:-2] + (len(th1), len(th2)).
    """
    n_out = np.arange(-s_out, s_out + 1)
    e1 = np.exp(1j * np.outer(n_out, np.atleast_1d(theta1)))  # (l, t1)
    e2 = np.exp(-1j * np.outer(n_out, np.atleast_1d(theta2)))  # (n, t2)
    return np.real(np.einsum("...nl,lt,nu->...tu", G, e1, e2))


@_timed
def B3_connected(table, r, s, phi_s=0.0, theta1=0.0, theta2=0.0):
    """Connected part of B on the product grid r x s x phi_s x th1 x th2.

    r, s are moduli (r along u_x, s vec = s e^{i phi_s}); all angles
    are measured from r_hat.  Returns the dimensionless real array of
    shape (len(r), len(s), len(phi_s), len(theta1), len(theta2)); the
    physical B carries an extra (rho/2pi)^3.  This is the fast path:
    the k contractions run once per (p, r, s) modulus pair, the phi_s
    and theta dependences are analytic phase resummations.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    s_mod = np.atleast_1d(np.asarray(s, dtype=float))
    phi_s = np.atleast_1d(np.asarray(phi_s, dtype=float))
    N_out = 2 * table.s_out + 1
    n_out = np.arange(-table.s_out, table.s_out + 1)
    nl_diff = n_out[:, None] - n_out[None, :]
    wk = _trapz_kweights(table.k_arr)

    G = np.zeros((len(r), len(s_mod), len(phi_s), N_out, N_out), dtype=np.complex128)
    for jp, p in enumerate(table.p_arr):
        J2 = jv(p, np.outer(table.k_arr, s_mod))  # (b, j)
        C = np.einsum("b,bj,abnl->janl", wk, J2, table.H[:, :, jp])
        J1 = jv(
            nl_diff[None, None] + p, np.outer(r, table.k_arr)[:, :, None, None]
        )  # (i, a, n, l)
        T = np.einsum("a,ianl,janl->ijnl", wk, J1, C)
        phase = (-1.0) ** p * np.exp(-1j * p * phi_s)
        G += phase[None, None, :, None, None] * T[:, :, None]
    G *= (-1j) ** nl_diff / (2 * np.pi) ** 2

    return _resum_theta(G, theta1, theta2, table.s_out)


@_timed
def B3_connected_points(table, r_vec, s_vec, theta1=0.0, theta2=0.0, chunk=256):
    """Connected part of B at arbitrary points (slow path).

    r_vec, s_vec are complex positions (x + i y) of particles 2 and 3
    relative to particle 1, theta1/theta2 lab-frame headings; the four
    arguments broadcast together and the result has the broadcast
    shape.  Each point is derotated onto r along u_x and evaluated
    individually -- use B3_connected when the arguments live on a
    (r, s, phi_s, theta) product grid.
    """
    r_vec, s_vec, th1, th2 = np.broadcast_arrays(
        np.asarray(r_vec, dtype=complex),
        np.asarray(s_vec, dtype=complex),
        theta1,
        theta2,
    )
    shape = r_vec.shape
    r_vec, s_vec = r_vec.ravel(), s_vec.ravel()
    phi_r = np.angle(r_vec)
    r = np.abs(r_vec)
    s_mod = np.abs(s_vec)
    phi_s = np.angle(s_vec) - phi_r
    th1 = np.asarray(th1, dtype=float).ravel() - phi_r
    th2 = np.asarray(th2, dtype=float).ravel() - phi_r

    N_out = 2 * table.s_out + 1
    n_out = np.arange(-table.s_out, table.s_out + 1)
    nl_diff = n_out[:, None] - n_out[None, :]
    wk = _trapz_kweights(table.k_arr)

    out = np.empty(r.size)
    for i0 in range(0, r.size, chunk):
        sl = slice(i0, min(i0 + chunk, r.size))
        G = np.zeros((r[sl].size, N_out, N_out), dtype=np.complex128)
        for jp, p in enumerate(table.p_arr):
            J2 = jv(p, np.outer(table.k_arr, s_mod[sl]))  # (b, m)
            C = np.einsum("b,bm,abnl->manl", wk, J2, table.H[:, :, jp])
            J1 = jv(
                nl_diff[None, None] + p, np.outer(r[sl], table.k_arr)[:, :, None, None]
            )
            T = np.einsum("a,manl,manl->mnl", wk, J1, C)
            G += ((-1.0) ** p * np.exp(-1j * p * phi_s[sl]))[:, None, None] * T
        G *= (-1j) ** nl_diff / (2 * np.pi) ** 2
        ph1 = np.exp(1j * np.outer(th1[sl], n_out))  # (m, l)
        ph2 = np.exp(-1j * np.outer(th2[sl], n_out))  # (m, n)
        out[sl] = np.real(np.einsum("mnl,ml,mn->m", G, ph1, ph2))
    return out.reshape(shape)


@_timed
def B3_connected_mesh(table, r, s_vec, theta1=0.0, theta2=0.0, s_grid=None):
    """Connected part at one scalar r for arbitrary complex s_vec.

    Same frame as B3_connected (r along u_x, angles measured from
    r_hat), but s_vec = x + i y need not lie on any (s, phi_s) product
    grid.  The k contractions -- the expensive part -- run once per p
    on a 1D |s| grid (default spacing pi / (8 kmax), resolving the
    fastest Bessel oscillation of the table); each mesh node then only
    costs a cubic interpolation in |s| and the exact phi_s / theta
    phase resummation.  theta1, theta2 broadcast against s_vec.  Use
    this for dense non-product maps (e.g. bipolar meshes) at a single
    r; B3_connected_points re-contracts the k integrals per point and
    scales much worse.
    """
    r = float(r)
    s_vec, th1, th2 = np.broadcast_arrays(
        np.asarray(s_vec, dtype=complex), theta1, theta2
    )
    shape = s_vec.shape
    s_vec = s_vec.ravel()
    s_mod = np.abs(s_vec)
    phi_s = np.angle(s_vec)
    th1 = np.asarray(th1, dtype=float).ravel()
    th2 = np.asarray(th2, dtype=float).ravel()

    if s_grid is None:
        ds = np.pi / (8 * table.k_arr[-1])
        lo, hi = s_mod.min(), s_mod.max()
        hi = max(hi, lo + 10 * ds)  # degenerate range guard
        s_grid = np.linspace(lo, hi, int(np.ceil((hi - lo) / ds)) + 1)

    N_out = 2 * table.s_out + 1
    n_out = np.arange(-table.s_out, table.s_out + 1)
    nl_diff = n_out[:, None] - n_out[None, :]
    wk = _trapz_kweights(table.k_arr)

    G = np.zeros((s_vec.size, N_out, N_out), dtype=np.complex128)
    for jp, p in enumerate(table.p_arr):
        J2 = jv(p, np.outer(table.k_arr, s_grid))  # (b, j)
        C = np.einsum("b,bj,abnl->janl", wk, J2, table.H[:, :, jp], optimize=True)
        J1 = jv(nl_diff + p, r * table.k_arr[:, None, None])  # (a, n, l)
        T = np.einsum("a,anl,janl->jnl", wk, J1, C)  # radial kernel on s_grid
        Tm = CubicSpline(s_grid, T, axis=0)(s_mod)  # (m, n, l)
        G += ((-1.0) ** p * np.exp(-1j * p * phi_s))[:, None, None] * Tm
    G *= (-1j) ** nl_diff / (2 * np.pi) ** 2

    ph1 = np.exp(1j * np.outer(th1, n_out))  # (m, l)
    ph2 = np.exp(-1j * np.outer(th2, n_out))  # (m, n)
    return np.real(np.einsum("mnl,ml,mn->m", G, ph1, ph2)).reshape(shape)


def _pair_harmonics(X2, k_arr, x, s_out):
    """gh[n, m](x) = i^{m-n}/(2pi) int k dk J_{n-m}(k x) X_{nm}(k).

    The inverse of the (simulation-validated) bridge X_nm =
    2pi i^{n-m} int r dr J_{n-m}(k r) [g_nm - delta]: the derotated
    pair-correlation harmonics, n on the particle at the head of x.
    x may have any shape; output (2 s_out + 1, 2 s_out + 1) + x.shape.
    """
    s2 = (X2.shape[0] - 1) // 2
    if s_out > s2:
        raise ValueError("s_out exceeds the truncation of X2")
    wk = _trapz_kweights(k_arr)
    x = np.asarray(x, dtype=float)
    N_out = 2 * s_out + 1
    n_out = np.arange(-s_out, s_out + 1)
    crop = slice(s2 - s_out, s2 + s_out + 1)
    Xc = X2[crop, crop]
    gh = np.zeros((N_out, N_out) + x.shape, dtype=np.complex128)
    kx = np.multiply.outer(x, k_arr)
    nl = n_out[:, None] - n_out[None, :]
    for nu in range(2 * s_out + 1):
        Jnu = jv(nu, kx)  # x.shape + (nk,); J_{-nu} = (-1)^nu J_nu
        signs = [(nu, 1.0)] if nu == 0 else [(nu, 1.0), (-nu, (-1.0) ** nu)]
        for v, sgn in signs:
            pairs = np.nonzero(nl == v)
            block = np.einsum("...a,a,pa->p...", Jnu, wk, Xc[pairs])
            gh[pairs] = sgn * 1j ** (-v) / (2 * np.pi) * block
    return gh


def _gh_spline(X2, k_arr, lo, hi, s_out):
    """Cubic spline in |x| of the pair harmonics, for bulk evaluation.

    The direct Bessel sums of _pair_harmonics cost ~(4 s_out + 1) nk
    jv evaluations PER POINT; on dense grids almost all of that is
    redundant, since gh only depends on the modulus and oscillates no
    faster than kmax.  Knots spaced pi / (8 kmax) resolve the fastest
    oscillation, so the spline is interchangeable with the direct sum
    to ~1e-6 relative while being orders of magnitude cheaper.
    """
    ds = np.pi / (8 * k_arr[-1])
    hi = max(hi, lo + 10 * ds)  # degenerate range guard
    grid = np.linspace(lo, hi, int(np.ceil((hi - lo) / ds)) + 1)
    gh = _pair_harmonics(X2, k_arr, grid, s_out)  # (n, m, j)
    return CubicSpline(grid, np.moveaxis(gh, -1, 0), axis=0)


def _gh_eval(sp, x):
    """Spline of _gh_spline evaluated with _pair_harmonics' layout."""
    return np.moveaxis(sp(np.asarray(x, dtype=float)), (-2, -1), (0, 1))


@_timed
def B3_disconnected(
    X2, k_arr, r, s, phi_s=0.0, theta1=0.0, theta2=0.0, s_out=6, interp=True
):
    """1 + the three 2-body terms of B on the same product grid.

    X2 is the pair Sylvester solution on k_arr (struct_2b
    compute_correlations, which="2b" -- or "3b" for the corrected
    pair correlation).  Same geometry and normalization as
    B3_connected; adding the two gives the full dimensionless bracket,
    -> 1 at large separations.  The bracket structure is

        d_n0 d_l0 + gh_{nl}(r) + d_n0 e^{-i l phi_s} gh_{0l}(s)
                  + d_l0 e^{i n arg(r-s)} gh_{n0}(|r - s|),

    the three pairs (1,2), (1,3), (2,3) each dressed by the uniform
    third particle.  With interp=True (default) the pair harmonics
    come from one shared modulus spline (_gh_spline, ~1e-6 relative)
    instead of per-node Bessel sums, which dominate the cost of the
    (2,3) term on dense grids; interp=False is the exact path.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    s_mod = np.atleast_1d(np.asarray(s, dtype=float))
    phi_s = np.atleast_1d(np.asarray(phi_s, dtype=float))
    N_out = 2 * s_out + 1
    sc = s_out  # center index of the cropped harmonic axes

    d = r[:, None, None] - s_mod[None, :, None] * np.exp(1j * phi_s[None, None, :])
    if interp:
        lo = min(r.min(), s_mod.min(), np.abs(d).min())
        hi = max(r.max(), s_mod.max(), np.abs(d).max())
        sp = _gh_spline(X2, k_arr, lo, hi, s_out)
        gh = lambda x: _gh_eval(sp, x)
    else:
        gh = lambda x: _pair_harmonics(X2, k_arr, x, s_out)

    G = np.zeros((len(r), len(s_mod), len(phi_s), N_out, N_out), dtype=np.complex128)
    G[..., sc, sc] += 1.0

    # pair (1,2): head at r on u_x
    G += gh(r).transpose(2, 0, 1)[:, None, None]

    # pair (1,3): head at s e^{i phi_s}, n = 0 forced
    gh_b = gh(s_mod)[sc]  # (l, j)
    ph = np.exp(-1j * np.outer(phi_s, np.arange(-s_out, s_out + 1)))  # (f, l)
    G[:, :, :, sc, :] += gh_b.T[None, :, None, :] * ph[None, None]

    # pair (2,3): head at r - s vec, l = 0 forced
    gh_c = gh(np.abs(d))[:, sc]  # (n, i, j, f)
    ph = np.exp(
        1j * np.multiply.outer(np.arange(-s_out, s_out + 1), np.angle(d))
    )  # (n, i, j, f)
    G[..., sc] += np.moveaxis(gh_c * ph, 0, -1)

    return _resum_theta(G, theta1, theta2, s_out)


@_timed
def B3_disconnected_points(
    X2, k_arr, r_vec, s_vec, theta1=0.0, theta2=0.0, s_out=6, interp=True
):
    """1 + the three 2-body terms at arbitrary points.

    Point analogue of B3_disconnected: r_vec, s_vec are complex
    positions (x + i y) of particles 2 and 3 relative to particle 1,
    theta1/theta2 lab-frame headings; the four arguments broadcast
    together and the result has the broadcast shape.  interp as in
    B3_disconnected (shared modulus spline vs exact Bessel sums).
    """
    r_vec, s_vec, th1, th2 = np.broadcast_arrays(
        np.asarray(r_vec, dtype=complex),
        np.asarray(s_vec, dtype=complex),
        theta1,
        theta2,
    )
    shape = r_vec.shape
    r_vec, s_vec = r_vec.ravel(), s_vec.ravel()
    phi_r = np.angle(r_vec)
    th1 = np.asarray(th1, dtype=float).ravel() - phi_r
    th2 = np.asarray(th2, dtype=float).ravel() - phi_r
    s_rel = s_vec * np.exp(-1j * phi_r)  # derotated: r along u_x
    d = np.abs(r_vec) - s_rel  # particle 3 as seen from particle 2

    if interp:
        mods = (np.abs(r_vec), np.abs(s_rel), np.abs(d))
        sp = _gh_spline(
            X2, k_arr, min(m.min() for m in mods), max(m.max() for m in mods), s_out
        )
        gh = lambda x: _gh_eval(sp, x)
    else:
        gh = lambda x: _pair_harmonics(X2, k_arr, x, s_out)

    N_out = 2 * s_out + 1
    n_out = np.arange(-s_out, s_out + 1)
    sc = s_out
    G = np.zeros((r_vec.size, N_out, N_out), dtype=np.complex128)
    G[:, sc, sc] = 1.0

    # pair (1,2): head at r on u_x
    G += np.moveaxis(gh(np.abs(r_vec)), -1, 0)

    # pair (1,3): head at s_rel, n = 0 forced
    gh_b = gh(np.abs(s_rel))[sc]  # (l, m)
    G[:, sc, :] += gh_b.T * np.exp(-1j * np.outer(np.angle(s_rel), n_out))

    # pair (2,3): head at d, l = 0 forced
    gh_c = gh(np.abs(d))[:, sc]  # (n, m)
    G[:, :, sc] += gh_c.T * np.exp(1j * np.outer(np.angle(d), n_out))

    ph1 = np.exp(1j * np.outer(th1, n_out))  # (m, l)
    ph2 = np.exp(-1j * np.outer(th2, n_out))  # (m, n)
    return np.real(np.einsum("mnl,ml,mn->m", G, ph1, ph2)).reshape(shape)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import os
    from struct_2b import compute_correlations, make_k_grid
    from struct_aux import Vexp as V

    # Physical parameters
    Dr = 1.0

    lp, phi, eps, D = 10.0 / Dr, 0.04, 1.0 / Dr, 0.0

    # Choose truncation so that diffusion compensates advection
    s = 16
    # s, s_out = 6, 6
    s_out = 6
    rmax = 10.0
    k_arr = make_k_grid(kmax=14.0, rmax=rmax, dk_factor=8.0)
    n_beta, pmax = 36, 16
    # every knob that shapes the table goes into the cache name -- a knob
    # change with a stale name silently reloads the old table otherwise
    fname = (
        f"B3_table_lp{lp:g}_phi{phi:g}_eps{eps:g}_rm{rmax:g}"
        f"_s{s}_so{s_out}_nb{n_beta}_pm{pmax}_km{k_arr[-1]:.4g}.npz"
    )
    if not os.path.exists(fname):
        table = precompute_B3_table(
            lp,
            phi,
            eps,
            V,
            k_arr,
            s=s,
            s_out=s_out,
            n_beta=n_beta,
            pmax=pmax,
            parallel=True,
        )
        np.savez(fname, **table._asdict())
    else:
        data = np.load(fname)
        table = B3Table(
            data["k_arr"],
            data["p_arr"],
            data["H"],
            int(data["s"]),
            int(data["s_out"]),
            float(data["lp"]),
            float(data["phi"]),
            float(data["eps"]),
            float(data["D"]),
        )

    X2 = compute_correlations(
        abp_model(lp, phi, eps, V, D, s=s), k_arr, s=s, which="2b"
    )

    # Third particle mapped in bipolar coordinates (sigma, tau) with the
    # foci on particles 1 and 2, which play symmetric roles; the frame is
    # centered on the midpoint x = r/2.  With a = r/2,
    #     x + i y = a (sinh tau + i sin sigma) / (cosh tau - cos sigma),
    # tau -> -inf converges on particle 1, tau -> +inf on particle 2,
    # sigma = pi is the segment between them, sigma -> 0 (mod 2pi) with
    # tau -> 0 escapes to infinity.
    l_th12 = [
        (0.0, 0.0),
        (0.0, np.pi),
        (np.pi, 0.0),
        (np.pi / 2, np.pi / 2),
        (np.pi / 2, -np.pi / 2),
    ]
    l_r0 = [1.0, 2.0, 3.0, 4.0]
    fig, ax = plt.subplots(
        len(l_th12),
        len(l_r0),
        figsize=(len(l_th12) * 3, len(l_r0) * 3),
        layout="constrained",
    )

    l_Bc = []
    l_Bd = []
    for i, (th1, th2) in tqdm(enumerate(l_th12), total=len(l_th12)):
        for j, r0 in enumerate(l_r0):
            a = r0 / 2
            tau = np.linspace(-3.0, 3.0, 121)
            sigma = np.linspace(0.0, 2 * np.pi, 161)
            tg, sg = np.meshgrid(tau, sigma, indexing="ij")

            with np.errstate(divide="ignore", invalid="ignore"):
                w = a * (np.sinh(tg) + 1j * np.sin(sg)) / (np.cosh(tg) - np.cos(sg))
            s_vec = w + a  # module frame: particle 1 at the origin, r along u_x
            ok = np.isfinite(w) & (np.abs(s_vec) <= rmax) & (np.abs(s_vec - r0) <= rmax)

            Bc = np.full(w.shape, np.nan)
            Bd = np.full(w.shape, np.nan)
            Bc[ok] = B3_connected_mesh(table, r0, s_vec[ok], th1, th2)
            Bd[ok] = B3_disconnected_points(
                X2, k_arr, r0 + 0j, s_vec[ok], th1, th2, s_out=s_out
            )
            l_Bc.append(Bc)
            l_Bd.append(Bd)

            # the sigma = 0 (mod 2pi), tau = 0 nodes map to infinity: give them a
            # finite dummy position (their cells are dropped through the NaNs in Z)
            w_plot = np.where(np.isfinite(w), w, 0)

            norm = colors.SymLogNorm(
                linthresh=1e-3, linscale=0.2, vmin=-0.1, vmax=0.1, base=10
            )
            panels = [(Bc, "connected"), (Bc + Bd - 1, "full bracket - 1")]
            _ax = ax[i, j] if len(l_th12) > 1 and len(l_r0) > 1 else ax[max(i, j)]
            _ax.pcolormesh(
                w_plot.real,
                w_plot.imag,
                Bc,
                norm=norm,
                cmap="RdBu_r",
                shading="gouraud",
                rasterized=True,
            )
            r = 6.0
            clip = Circle((0, 0), r, transform=_ax.transData)

            _ax.set_aspect("equal")  # do this BEFORE, or the circle is an ellipse
            for art in _ax.get_children():
                if art is not _ax.patch:
                    art.set_clip_path(clip)

            _ax.set_xlim(-r, r)
            _ax.set_ylim(-r, r)
            _ax.set_frame_on(False)
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.add_patch(
                Circle((0, 0), r, fc="none", ec="k", lw=1.5, zorder=10, clip_on=False)
            )

        # fig.colorbar(pb, ax=ax[row, :])
    plt.show()

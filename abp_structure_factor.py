"""
Soft repulsive active Brownian particles in 2D and their equal-time
correlation matrix S_nm(k), for direct comparison with the linear
h-hierarchy of ``StructFactor_MIPS/struct_2b.py``.

Model
-----
N particles obey the overdamped Langevin dynamics (mobility absorbed
into ``eps``)

    dr_i/dt     = v0 u(theta_i) - eps sum_j grad V(|r_i - r_j|) + sqrt(2 D)  xi_i
    dtheta_i/dt =                                                 sqrt(2 Dr) eta_i

with u(theta) = (cos theta, sin theta) and independent unit white
noises.  The pair potential (selected by ``potential``) is

    "harm" : V(r) = (1 - r/lV)^2 / 2   for r < lV   (else 0)
    "exp"  : V(r) = exp(-r^2 / (2 lV^2))

Units and mapping to the structure-factor code
----------------------------------------------
All parameters are dimensional; the recommended choice is a = 1 (the
particle diameter / interaction length) and v0 = 1, so time is measured
in a/v0 and Pe = lp = v0/(a Dr) is set by Dr alone.  The h-hierarchy
code uses time unit 1/Dr instead; given a simulation (v0, Dr, D, eps),
call ``struct_2b.compute_correlations`` with

    lp  = v0 / (a Dr),   eps -> eps / (a^2 Dr),   D -> D / (a^2 Dr)

and the SAME phi: the standard packing fraction of diameter-a disks,

    phi = pi rho (a/2)^2,   i.e.   rho = N / L^2 = 4 phi / (pi a^2)

(rho0 = 4 phi/pi in the struct code, same convention as
``HardSphereABP``).  The k-space kernels matching the two potentials
are ``Vk_harm`` / ``Vk_exp`` below (for lV = a, ``Vk_harm`` coincides
with ``struct_aux.Vharm``; for lV = 0.5 a, ``Vk_exp`` coincides with
``struct_aux.Vexp``).

Observable
----------
For every box wavevector k = (2 pi / L)(nx, ny) kept (see below), the
angular-harmonic Fourier modes

    c_n(k) = sum_j exp(i n theta_j) exp(+i k.r_j),   n = -s..s

are accumulated into  S_nm(k) = < c_n(k) conj(c_m(k)) > / N.  The
RELATIVE sign of the two exponents matters and is fixed by the theory:
the advection block of ``struct_aux.L`` (super-diagonal -i/2 lp k*,
sub-diagonal -i/2 lp k) is the Fokker-Planck operator for mode
amplitudes f_n(k) of psi = sum f_n e^{-i k.r} e^{-i n theta}, whose
measured amplitude is c_n above; flipping only one of the two signs
conjugates S (it flips Im S_10 -- checked against the 2b solution).
Each wavevector is *derotated* onto the real axis using the rotation
identity  S(q e^{i alpha}) = R S(q) R^H,  R = diag(e^{i n alpha})
(same identity as L(q e^{i a}) = R L(q) R^H in the h-hierarchy), then
ring-averaged over |k| bins and symmetrized under the two exact
symmetries of the isotropic ensemble: index reversal J S J (mirror,
theta -> -theta) and Hermitian conjugation.  Since both the simulation
average and the theory matrix satisfy X = J X J at real k, flipping
BOTH signs at once changes nothing:

    S_nm(k)  <->  delta_nm + rho * X_nm(k),      rho = 4 phi / (pi a^2),

with X the Sylvester solution of ``compute_correlations`` (its (0,0)
element gives the ordinary structure factor S(k) = 1 + rho h(k)).

k selection: all half-plane box vectors with |k| <= kmax are binned in
|k| with width dk; bins holding more than ``n_kvec`` vectors keep only
``n_kvec`` of them, evenly spread in angle (low-|k| rings keep every
vector, large-|k| rings are angularly subsampled).

Real space: the same samples are also binned directly into the
derotated angular-harmonic pair correlation

    g_nm(r) = < sum_{j != l} e^{i n (theta_j - alpha)}
                             e^{-i m (theta_l - alpha)}
               delta^2(r - r_jl) > / (2 pi r rho N),

where alpha is the angle of the separation r_jl = r_j - r_l, i.e. n
attaches to the particle at the HEAD of the separation vector (the
real-space counterpart of the k-derotation R^H S R above).  g_00 is
the ordinary g(r) -> 1, all other harmonics -> 0 at large r.  Counting
both orders of every pair (the reverse order sees alpha + pi) makes

    g_nm(r) = (-1)^(n-m) conj(g_mn(r))

exact per sample -- the real-space analog of Hermiticity -- while the
mirror symmetry g_nm = g_{-n,-m} is enforced in ``_finalize``.  The
bridge to the k-space observable is the Hankel transform

    S_nm(k) = delta_nm + rho X_nm(k),
    X_nm(k) = 2 pi i^(n-m) int_0^inf dr r J_{n-m}(k r)
              [g_nm(r) - delta_n0 delta_m0],

so the binned g_nm gives an independent, truncation-free-in-k check of
the same X compared against ``struct_2b.compute_correlations``.
``measure_sk=False`` skips the k-space accumulation entirely (usually
the dominant sampling cost) and measures only g_nm(r).

Statistics: the error of a ring bin is ~ 1/sqrt(n_vec * n_indep) with
n_indep the number of *independent* samples, limited by the relaxation
time of the slowest mode in the bin (~ 1/(D_eff k^2) at small k, with
D_eff = v0^2/(2 Dr) + D).  The lowest few bins hold O(few) wavevectors
and decorrelate over hundreds of a/v0, so they are far noisier than
the rest -- run long, and drop them (or average runs) when comparing.

Performance
-----------
Cell lists + Numba for the dynamics; the S_nm accumulation is Numba
parallel over wavevectors.  Cost per sample ~ n_kvec_total * N * s.
"""

from __future__ import annotations

import os

import numpy as np
from numba import njit, prange
from tqdm import tqdm

from hard_sphere_abp import _wrap, _min_image, _build_cells

TWOPI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# k-space kernels of the two potentials (for the theory comparison)            #
# --------------------------------------------------------------------------- #
def Vk_exp(k, lV=0.5):
    """2D FT of exp(-r^2/(2 lV^2)):  2 pi lV^2 exp(-(lV k)^2 / 2).

    Equals ``struct_aux.Vexp`` for lV = 0.5."""
    return TWOPI * lV**2 * np.exp(-0.5 * (lV * np.abs(k)) ** 2)


def Vk_harm(k, lV=1.0):
    """2D FT of the harmonic repulsion (1 - r/lV)^2 / 2 for r < lV.

    Equals ``struct_aux.Vharm`` for lV = 1 (diameter units).  The
    k -> 0 limit is pi lV^2 / 12."""
    from scipy.special import jv, struve

    q = np.abs(k) * lV / 2.0  # half-range variable: FT_1(q) has range 2
    _v = (
        np.pi
        / np.where(q == 0, 1.0, q) ** 2
        * (
            -2 * jv(2, 2 * q)
            + np.pi
            * (jv(1, 2 * q) * struve(0, 2 * q) - jv(0, 2 * q) * struve(1, 2 * q))
        )
    )
    return np.where(q == 0, np.pi * lV**2 / 12.0, lV**2 / 4.0 * _v)


# --------------------------------------------------------------------------- #
# Numba kernels                                                                #
# --------------------------------------------------------------------------- #
@njit
def _seed_numba(seed):
    # np.random.seed called *inside* an njit function seeds Numba's own RNG
    # (the interpreter-side np.random.seed does not).
    np.random.seed(seed)


@njit
def _forces(pos, L, eps, pot_id, lV, rcut, head, linked, ncx, ncy, cell_size, fx, fy):
    """Pairwise repulsive forces  f_i = -eps sum_j V'(r_ij) rhat_ij  via the
    cell list (pot_id: 0 = harm, 1 = exp)."""
    N = pos.shape[0]
    rcut2 = rcut * rcut
    for i in range(N):
        fx[i] = 0.0
        fy[i] = 0.0
    for i in range(N):
        cx = int(pos[i, 0] / cell_size) % ncx
        cy = int(pos[i, 1] / cell_size) % ncy
        for oy in (-1, 0, 1):
            ny = (cy + oy) % ncy
            for ox in (-1, 0, 1):
                nx = (cx + ox) % ncx
                c = nx + ncx * ny
                j = head[c]
                while j != -1:
                    if j > i:
                        dx = _min_image(pos[j, 0] - pos[i, 0], L)
                        dy = _min_image(pos[j, 1] - pos[i, 1], L)
                        r2 = dx * dx + dy * dy
                        if 1e-18 < r2 < rcut2:
                            r = np.sqrt(r2)
                            if pot_id == 0:
                                fmag = eps * (1.0 - r / lV) / lV if r < lV else 0.0
                            else:
                                fmag = (
                                    eps * r / (lV * lV) * np.exp(-0.5 * r2 / (lV * lV))
                                )
                            if fmag != 0.0:
                                fr = fmag / r  # repulsive: j pushed along +d
                                fx[j] += fr * dx
                                fy[j] += fr * dy
                                fx[i] -= fr * dx
                                fy[i] -= fr * dy
                    j = linked[j]


@njit(parallel=True)
def _accumulate_sk(pos, theta, kxs, kys, derot, s, sk_acc):
    """Add  c_a(k) conj(c_b(k))  (derotated onto the real k-axis) into
    ``sk_acc[ik]`` for every wavevector, parallel over wavevectors.

    ``derot[ik, s+n] = exp(-i n alpha_k)`` implements S(q) = R^H S(k) R."""
    nk = kxs.size
    N = pos.shape[0]
    nh = 2 * s + 1
    for ik in prange(nk):
        kx = kxs[ik]
        ky = kys[ik]
        c = np.zeros(nh, dtype=np.complex128)
        for j in range(N):
            arg = kx * pos[j, 0] + ky * pos[j, 1]
            ph = complex(np.cos(arg), np.sin(arg))  # e^{+i k.r_j}
            u = complex(np.cos(theta[j]), np.sin(theta[j]))
            uc = u.conjugate()
            c[s] += ph
            wp = ph
            wm = ph
            for n in range(1, s + 1):
                wp = wp * u
                wm = wm * uc
                c[s + n] += wp
                c[s - n] += wm
        for n in range(nh):
            c[n] = c[n] * derot[ik, n]
        for a in range(nh):
            ca = c[a]
            for b in range(nh):
                sk_acc[ik, a, b] += ca * np.conj(c[b])


@njit
def _accumulate_gr(
    pos, theta, L, s, r_max, dr_bin, head, linked, ncg, cell_size, gr_acc
):
    """Bin every ordered pair into the derotated angular-harmonic pair
    correlation:  gr_acc[ir, s+n, s+m] += e^{i n (theta_head - alpha)}
    e^{-i m (theta_tail - alpha)}  with alpha the angle of the separation
    head - tail.  Each unordered pair is visited once (j > i in a 3x3
    cell neighbourhood, valid because cell_size >= r_max) and both
    orders are added; the reverse order sees alpha + pi, which makes
    g_nm = (-1)^(n-m) conj(g_mn) exact per sample."""
    N = pos.shape[0]
    rmax2 = r_max * r_max
    nh = 2 * s + 1
    nbin = gr_acc.shape[0]
    p1 = np.empty(nh, dtype=np.complex128)
    p2 = np.empty(nh, dtype=np.complex128)
    for i in range(N):
        cx = int(pos[i, 0] / cell_size) % ncg
        cy = int(pos[i, 1] / cell_size) % ncg
        for oy in (-1, 0, 1):
            ny = (cy + oy) % ncg
            for ox in (-1, 0, 1):
                nx = (cx + ox) % ncg
                c = nx + ncg * ny
                j = head[c]
                while j != -1:
                    if j > i:
                        dx = _min_image(pos[i, 0] - pos[j, 0], L)
                        dy = _min_image(pos[i, 1] - pos[j, 1], L)
                        r2 = dx * dx + dy * dy
                        if 1e-18 < r2 < rmax2:
                            ib = int(np.sqrt(r2) / dr_bin)
                            if ib < nbin:
                                alpha = np.arctan2(dy, dx)  # angle of r_i - r_j
                                a1 = theta[i] - alpha
                                a2 = theta[j] - alpha
                                u1 = complex(np.cos(a1), np.sin(a1))
                                u2 = complex(np.cos(a2), np.sin(a2))
                                # p[s+n] = e^{i n (theta - alpha)}
                                p1[s] = 1.0 + 0.0j
                                p2[s] = 1.0 + 0.0j
                                w1 = complex(1.0, 0.0)
                                w2 = complex(1.0, 0.0)
                                for n in range(1, s + 1):
                                    w1 = w1 * u1
                                    w2 = w2 * u2
                                    p1[s + n] = w1
                                    p1[s - n] = np.conj(w1)
                                    p2[s + n] = w2
                                    p2[s - n] = np.conj(w2)
                                for a in range(nh):
                                    for b in range(nh):
                                        w = p1[a] * np.conj(p2[b])  # ordered (i, j)
                                        wr = p2[a] * np.conj(
                                            p1[b]
                                        )  # reverse, alpha + pi
                                        if (a - b) & 1:
                                            wr = -wr
                                        gr_acc[ib, a, b] += w + wr
                    j = linked[j]


@njit
def _run_chunk(
    pos,
    theta,
    L,
    v0,
    Dr,
    D,
    eps,
    pot_id,
    lV,
    rcut,
    dt,
    step_start,
    n_chunk,
    burn_in,
    sample_every,
    kxs,
    kys,
    derot,
    s,
    sk_acc,
    r_max,
    dr_bin,
    gr_acc,
    fx,
    fy,
):
    """Advance ``n_chunk`` steps from absolute step ``step_start``,
    accumulating S_nm(k) and g_nm(r) on sample steps.  Returns the
    number of samples."""
    N = pos.shape[0]

    ncx = max(3, int(L / rcut))
    cell_size = L / ncx  # exact tiling, cell_size >= rcut
    head = np.empty(ncx * ncx, dtype=np.int64)
    linked = np.empty(N, dtype=np.int64)

    # Coarser cell list for the real-space binning (cell_size_g >= r_max,
    # guaranteed by r_max <= L/3 enforced in the constructor).
    ncg = int(L / r_max)
    cell_size_g = L / ncg
    head_g = np.empty(ncg * ncg, dtype=np.int64)
    linked_g = np.empty(N, dtype=np.int64)

    sig_t = np.sqrt(2.0 * D * dt) if D > 0.0 else 0.0
    sig_r = np.sqrt(2.0 * Dr * dt) if Dr > 0.0 else 0.0
    n_samples = 0

    for st in range(n_chunk):
        step = step_start + st

        # --- forces ---
        if eps > 0.0:
            _build_cells(pos, L, cell_size, head, linked, ncx, ncx)
            _forces(
                pos, L, eps, pot_id, lV, rcut, head, linked, ncx, ncx, cell_size, fx, fy
            )
        else:
            for i in range(N):
                fx[i] = 0.0
                fy[i] = 0.0

        # --- Euler-Maruyama update ---
        for i in range(N):
            th = theta[i]
            pos[i, 0] += dt * (v0 * np.cos(th) + fx[i])
            pos[i, 1] += dt * (v0 * np.sin(th) + fy[i])
            if sig_t > 0.0:
                pos[i, 0] += sig_t * np.random.normal()
                pos[i, 1] += sig_t * np.random.normal()
            pos[i, 0] = _wrap(pos[i, 0], L)
            pos[i, 1] = _wrap(pos[i, 1], L)
        if sig_r > 0.0:
            for i in range(N):
                theta[i] += sig_r * np.random.normal()

        # --- sample ---
        if step >= burn_in and (step % sample_every == 0):
            if kxs.size > 0:
                _accumulate_sk(pos, theta, kxs, kys, derot, s, sk_acc)
            _build_cells(pos, L, cell_size_g, head_g, linked_g, ncg, ncg)
            _accumulate_gr(
                pos,
                theta,
                L,
                s,
                r_max,
                dr_bin,
                head_g,
                linked_g,
                ncg,
                cell_size_g,
                gr_acc,
            )
            n_samples += 1

    return n_samples


# --------------------------------------------------------------------------- #
# Public interface                                                             #
# --------------------------------------------------------------------------- #
class ABPStructureFactor:
    """Soft repulsive ABP simulation + equal-time correlation matrix S_nm(k).

    Parameters
    ----------
    N : int
        Number of particles.
    phi : float
        Packing fraction  phi = pi rho (a/2)^2  (sets the box size; same
        convention as ``HardSphereABP`` and, since the diameter-unit
        migration, the struct code: rho0 = 4 phi/pi).
    v0, Dr, eps : float
        Self-propulsion speed, rotational diffusion rate, interaction
        strength (force = -eps grad V; mobility absorbed).
    D : float, optional
        Translational diffusion (default 0).
    a : float, optional
        Particle diameter = length unit of the theory code (default 1).
    potential : {"harm", "exp"}
        Pair potential (see module docstring).
    lV : float, optional
        Interaction length: harm -> range of the ramp (default a);
        exp -> Gaussian width (default 0.5 a, matching ``struct_aux``).
    rcut : float, optional
        Force cutoff (harm: lV; exp: 4 lV by default).
    s : int
        Angular harmonics kept, n = -s..s (matrix size 2s+1).
    kmax : float, optional
        Largest |k| sampled (default 12/a).
    dk : float, optional
        |k| bin width for the ring average (default 2 pi / L).
    n_kvec : int
        Max wavevectors kept per |k| bin (angular subsampling).
    r_max : float, optional
        Range of the direct real-space pair binning (default
        min(10 a, L/3); always capped at L/3 so the 3x3 cell search
        sees every pair).
    n_r : int
        Number of radial bins in (0, r_max) for g_nm(r).
    measure_sk : bool
        Set False to skip the k-space S_nm(k) accumulation entirely (no
        wavevectors are built; the k-space attributes and saved arrays
        are empty) and measure only the real-space g_nm(r).
    dt : float, optional
        Time step.  Default min(0.01 a/v0, 0.1 lV^2/eps, 0.005 lV^2/D).
    seed : int
        RNG seed (seeds both the initial condition and the Numba RNG).
    verbose : bool
        Print a banner and show a progress bar.

    After ``run``:  ``self.k`` (bin centers), ``self.S`` (n_bins, 2s+1,
    2s+1) complex, ``self.S00`` = S(k) the ordinary structure factor,
    ``self.S00_kvec`` per-wavevector S(k) for anisotropy checks;
    ``self.r`` (bin centers), ``self.g`` (n_r, 2s+1, 2s+1) complex =
    the derotated pair-correlation harmonics g_nm(r) binned directly in
    real space (module docstring), ``self.g00`` = the ordinary g(r).

    Example
    -------
    >>> sim = ABPStructureFactor(N=1000, phi=0.2, v0=1.0, Dr=0.5, eps=0.5)
    >>> sim.run(n_steps=200_000, burn_in=10_000, sample_every=100,
    ...         save_path="abp_sk")
    >>> # theory: S = I + rho X with X from struct_2b.compute_correlations
    """

    def __init__(
        self,
        N: int,
        phi: float,
        v0: float,
        Dr: float,
        eps: float,
        D: float = 0.0,
        a: float = 1.0,
        potential: str = "harm",
        lV: float | None = None,
        rcut: float | None = None,
        s: int = 8,
        kmax: float | None = None,
        dk: float | None = None,
        n_kvec: int = 48,
        r_max: float | None = None,
        n_r: int = 400,
        measure_sk: bool = True,
        dt: float | None = None,
        seed: int = 0,
        verbose: bool = True,
    ):
        if potential not in ("harm", "exp"):
            raise ValueError(f"Unknown potential {potential!r}")
        self.N = int(N)
        self.phi = phi
        self.a = a
        self.v0 = v0
        self.Dr = Dr
        self.D = D
        self.eps = eps
        self.potential = potential
        self.pot_id = 0 if potential == "harm" else 1
        self.lV = (a if potential == "harm" else 0.5 * a) if lV is None else lV
        self.rcut = (
            (self.lV if potential == "harm" else 4.0 * self.lV)
            if rcut is None
            else rcut
        )
        self.s = int(s)
        self.n_harm = 2 * self.s + 1
        self.seed = seed
        self.verbose = verbose

        # Box size from the packing fraction: rho = 4 phi / (pi a^2).
        self.rho = 4.0 * phi / (np.pi * a * a)
        self.L = np.sqrt(self.N / self.rho)
        if self.L < 3.0 * self.rcut:
            raise ValueError("Box too small for the cell list: L < 3 rcut")

        # Mapping to the h-hierarchy units (time 1/Dr).
        self.lp = v0 / (a * Dr) if Dr > 0 else np.inf
        self.eps_struct = eps / (a * a * Dr) if Dr > 0 else np.inf
        self.D_struct = D / (a * a * Dr) if Dr > 0 else np.inf

        # Time step: resolve propulsion over lV, potential relaxation, noise.
        if dt is None:
            cands = [0.01 * a / v0 if v0 > 0 else np.inf]
            if eps > 0:
                cands.append(0.1 * self.lV**2 / eps)
            if D > 0:
                cands.append(0.005 * self.lV**2 / D)
            dt = min(cands)
            if not np.isfinite(dt):
                raise ValueError("Cannot pick dt: v0, eps and D are all zero")
        self.dt = dt

        # ---- wavevector set ----
        self.kmax = (12.0 / a) if kmax is None else kmax
        self.dk = (TWOPI / self.L) if dk is None else dk
        self.n_kvec = int(n_kvec)
        self.measure_sk = bool(measure_sk)
        if self.measure_sk:
            self._build_kvecs()
        else:  # empty k set: the sk accumulation becomes a no-op
            self._kxs = np.empty(0)
            self._kys = np.empty(0)
            self._bin_of_vec = np.empty(0, dtype=np.int64)
            self._derot = np.empty((0, self.n_harm), dtype=np.complex128)

        # ---- real-space binning ----
        self.r_max = min(10.0 * a if r_max is None else r_max, self.L / 3.0)
        self.n_r = int(n_r)
        self.dr_bin = self.r_max / self.n_r

        self.S = None
        self.S00 = None
        self.k = None
        self.g = None
        self.g00 = None
        self.r = None
        self.n_samples = 0

    # ------------------------------------------------------------------ #
    def _build_kvecs(self):
        """Half-plane box wavevectors binned in |k|; angularly subsampled
        rings (at most ``n_kvec`` vectors per bin, evenly spread)."""
        kunit = TWOPI / self.L
        n_bins = int(np.ceil(self.kmax / self.dk))
        M = int(self.kmax / kunit) + 1

        nx, ny = np.meshgrid(np.arange(-M, M + 1), np.arange(0, M + 1), indexing="ij")
        keep = (ny > 0) | ((ny == 0) & (nx > 0))  # half plane, k != 0
        kx = kunit * nx[keep]
        ky = kunit * ny[keep]
        kk = np.hypot(kx, ky)
        inside = kk <= self.kmax
        kx, ky, kk = kx[inside], ky[inside], kk[inside]
        bins = np.minimum((kk / self.dk).astype(np.int64), n_bins - 1)

        sel_kx, sel_ky, sel_bin = [], [], []
        for b in range(n_bins):
            idx = np.nonzero(bins == b)[0]
            if idx.size == 0:
                continue
            if idx.size > self.n_kvec:
                order = idx[np.argsort(np.arctan2(ky[idx], kx[idx]))]
                take = np.round(
                    np.linspace(0, idx.size, self.n_kvec, endpoint=False)
                ).astype(np.int64)
                idx = order[take]
            sel_kx.append(kx[idx])
            sel_ky.append(ky[idx])
            sel_bin.append(np.full(idx.size, b, dtype=np.int64))

        self._kxs = np.concatenate(sel_kx)
        self._kys = np.concatenate(sel_ky)
        self._bin_of_vec = np.concatenate(sel_bin)
        alphas = np.arctan2(self._kys, self._kxs)
        n_idx = np.arange(-self.s, self.s + 1)
        self._derot = np.exp(-1j * alphas[:, None] * n_idx[None, :])

    # ------------------------------------------------------------------ #
    def _initial_state(self):
        rng = np.random.default_rng(self.seed)
        pos = rng.uniform(0.0, self.L, size=(self.N, 2))
        theta = rng.uniform(0.0, TWOPI, size=self.N)
        return pos, theta

    # ------------------------------------------------------------------ #
    def run(
        self,
        n_steps: int,
        burn_in: int = 0,
        sample_every: int = 100,
        chunk: int | None = None,
        save_path: str | None = None,
        checkpoint_every: int | None = None,
    ):
        """Run and accumulate S_nm(k) / g_nm(r); results land in ``self.S``,
        ``self.g`` etc.

        The accumulators are fixed-size running sums (no growth with run
        length), so there is nothing to flush for speed; but if
        ``checkpoint_every`` is given, the correctly normalized partial
        averages are additionally saved to ``save_path`` every that many
        samples (atomic overwrite of the same file), so long runs can be
        monitored mid-flight and survive a crash."""
        n_steps = int(n_steps)
        burn_in = int(burn_in)
        sample_every = int(sample_every)
        if chunk is None:
            chunk = max(sample_every, n_steps // 100)
        chunk = max(1, int(chunk))
        if checkpoint_every is not None and save_path is None:
            raise ValueError("checkpoint_every requires save_path")

        if self.verbose:
            if self.measure_sk:
                mem = self._kxs.size * self.n_harm**2 * 16 / 1e6
                kinfo = f"{self._kxs.size} wavevectors, s={self.s} ({mem:.0f} MB)"
            else:
                kinfo = f"S(k) disabled, s={self.s}"
            print(
                f"ABPStructureFactor: N={self.N}, phi={self.phi} "
                f"(rho={self.rho:.4f}), L={self.L:.2f}, v0={self.v0}, "
                f"Dr={self.Dr}, D={self.D}, eps={self.eps}, "
                f"{self.potential}(lV={self.lV}), dt={self.dt:.4g}"
            )
            print(
                f"  theory units: lp={self.lp:.4g}, eps={self.eps_struct:.4g}, "
                f"D={self.D_struct:.4g}, phi={self.phi} | {kinfo} | "
                f"g_nm(r): r_max={self.r_max:.3g}, {self.n_r} bins"
            )

        _seed_numba(self.seed)
        pos, theta = self._initial_state()
        sk_acc = np.zeros(
            (self._kxs.size, self.n_harm, self.n_harm), dtype=np.complex128
        )
        gr_acc = np.zeros((self.n_r, self.n_harm, self.n_harm), dtype=np.complex128)
        fx = np.empty(self.N)
        fy = np.empty(self.N)
        n_samples = 0

        bar = tqdm(
            total=n_steps,
            disable=not self.verbose,
            unit="step",
            unit_scale=True,
            desc="simulating",
        )
        step = 0
        last_ckpt = 0
        while step < n_steps:
            n_chunk = min(chunk, n_steps - step)
            n_samples += _run_chunk(
                pos,
                theta,
                self.L,
                self.v0,
                self.Dr,
                self.D,
                self.eps,
                self.pot_id,
                self.lV,
                self.rcut,
                self.dt,
                step,
                n_chunk,
                burn_in,
                sample_every,
                self._kxs,
                self._kys,
                self._derot,
                self.s,
                sk_acc,
                self.r_max,
                self.dr_bin,
                gr_acc,
                fx,
                fy,
            )
            step += n_chunk
            bar.update(n_chunk)
            bar.set_postfix(samples=n_samples)
            if (
                checkpoint_every is not None
                and n_samples - last_ckpt >= checkpoint_every
                and step < n_steps  # the final save happens below anyway
            ):
                self.n_samples = n_samples
                self._finalize(sk_acc, gr_acc)
                try:
                    self.save(save_path)
                except OSError as e:
                    # e.g. the npz is open in a viewer (Windows file lock):
                    # a failed checkpoint must not kill the run.
                    tqdm.write(f"checkpoint save failed ({e}); continuing")
                last_ckpt = n_samples
        bar.close()

        self.n_samples = n_samples
        self.pos, self.theta = pos, theta
        self._finalize(sk_acc, gr_acc)

        if self.verbose:
            print(f"done: {n_samples} samples")
        if save_path is not None:
            self.save(save_path)
        return self.k, self.S

    # ------------------------------------------------------------------ #
    def _finalize(self, sk_acc, gr_acc):
        """Normalize, ring-average and symmetrize the accumulated data."""
        c = self.s
        if self._kxs.size > 0:
            Sk = sk_acc / (self.n_samples * self.N)
            self.S00_kvec = np.real(Sk[:, c, c])

            kk = np.hypot(self._kxs, self._kys)
            bins_present = np.unique(self._bin_of_vec)
            S_list, k_list, n_list = [], [], []
            for b in bins_present:
                sel = self._bin_of_vec == b
                Sb = Sk[sel].mean(axis=0)
                Sb = 0.5 * (Sb + Sb[::-1, ::-1])  # mirror (J) symmetry
                Sb = 0.5 * (Sb + Sb.conj().T)  # Hermiticity
                S_list.append(Sb)
                k_list.append(kk[sel].mean())
                n_list.append(int(sel.sum()))
            self.k = np.array(k_list)
            self.S = np.stack(S_list, axis=0)
            self.S00 = np.real(self.S[:, c, c])
            self.n_kvec_bin = np.array(n_list)
        else:  # measure_sk=False: keep the npz schema with empty arrays
            self.S00_kvec = np.empty(0)
            self.k = np.empty(0)
            self.S = np.empty((0, self.n_harm, self.n_harm), dtype=np.complex128)
            self.S00 = np.empty(0)
            self.n_kvec_bin = np.empty(0, dtype=np.int64)

        # Real-space g_nm(r): the accumulator counts ordered pairs, so
        # dividing by (n_samples N rho shell_area) sends g_00 -> 1.
        edges = self.dr_bin * np.arange(self.n_r + 1)
        shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        g = gr_acc / (self.n_samples * self.N * self.rho * shell[:, None, None])
        g = 0.5 * (g + g[:, ::-1, ::-1])  # mirror (J) symmetry
        self.r = 0.5 * (edges[:-1] + edges[1:])
        self.g = g
        self.g00 = np.real(g[:, c, c])

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> str:
        """Save results + parameters to ``.npz`` (load with ``np.load``).

        The write is atomic (tmp file + replace) so a checkpoint that is
        interrupted mid-write never corrupts the previous save."""
        if not path.endswith(".npz"):
            path = path + ".npz"
        tmp = path + ".tmp.npz"
        np.savez_compressed(
            tmp,
            k=self.k,
            S=self.S,
            S00=self.S00,
            n_kvec_bin=self.n_kvec_bin,
            # real-space pair correlation (direct binning)
            r=self.r,
            g=self.g,
            g00=self.g00,
            r_max=self.r_max,
            n_r=self.n_r,
            # per-wavevector diagnostics (anisotropy check)
            kvec_kx=self._kxs,
            kvec_ky=self._kys,
            S00_kvec=self.S00_kvec,
            # parameters
            N=self.N,
            phi=self.phi,
            rho=self.rho,
            L=self.L,
            a=self.a,
            v0=self.v0,
            Dr=self.Dr,
            D=self.D,
            eps=self.eps,
            potential=self.potential,
            lV=self.lV,
            rcut=self.rcut,
            s=self.s,
            kmax=self.kmax,
            dk=self.dk,
            n_kvec=self.n_kvec,
            dt=self.dt,
            seed=self.seed,
            n_samples=self.n_samples,
            # h-hierarchy mapping (time unit 1/Dr)
            lp=self.lp,
            eps_struct=self.eps_struct,
            D_struct=self.D_struct,
        )
        os.replace(tmp, path)
        if self.verbose:
            tqdm.write(f"saved results to {path}")
        return path


# --------------------------------------------------------------------------- #
# Example                                                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Moderate-coupling active run set up for a direct overlay with the
    # 2-body h-hierarchy (see compare_struct_sim.py):
    #   lp = 2, eps_struct = 1, D_struct = 0.5, phi = 0.05 (rho0 = 0.064),
    #   V = Vexp.

    sim = ABPStructureFactor(
        N=1000,
        phi=0.04,
        v0=10.0,
        Dr=0.01,
        eps=1.0,
        D=0.25,
        potential="exp",
        lV=0.5,
        s=30,
        kmax=10.0,
        seed=1,
        measure_sk=False,
    )
    sim.run(
        n_steps=400_000,
        burn_in=20_000,
        sample_every=100,
        save_path="abp_structure_factor.npz",
    )

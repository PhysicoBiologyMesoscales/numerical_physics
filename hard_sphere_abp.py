"""
Hard-sphere active Brownian particles in 2D and their pair-correlation function.

Model
-----
N disks of diameter ``a`` move overdamped, each self-propelled along its own
orientation ``theta_i`` at speed ``v0`` and subject to rotational diffusion at
rate ``Dr`` (an optional translational diffusion ``Dt`` is available but is 0 by
default).  The hard-core constraint ``|r_i - r_j| >= a`` is enforced exactly by
projecting overlaps out at every step (overdamped contact = normal push, no
tangential friction), so particles can *slide* along each other while remaining
in contact.

Because there is no translational noise, two particles propelled into each other
stay glued at exactly ``r = a`` until rotational diffusion turns them apart.
This produces a genuine singular accumulation of pairs on the contact circle, so
the pair distribution splits into

    rho_2(r, alpha, beta) = (rho/2pi)^2 [ g_bulk(r, alpha, beta)
                                          + g_surf(alpha, beta) delta(r-a) ]

with ``rho`` the number density.  ``g_bulk`` is the usual (dimensionless) pair
correlation for r > a; ``g_surf`` is the *surface density* of contacts (units of
length, carried by the delta) -- both divided by the bulk factor (rho/2pi)^2.

Angles
------
Everything is expressed with the two collision-frame angles.  For a pair with
separation vector d = r_j - r_i and  phi = atan2(d_y, d_x):

    alpha = phi - (theta_i + theta_j)/2 + pi/2     (separation vs. mean heading)
    beta  = (theta_j - theta_i)/2                  (half the heading mismatch)

The headings are wrapped into [0, 2pi) before forming these (the stored thetas
are an unbounded random walk): this makes the half-sum single-valued, so e.g. a
head-on contact always lands at alpha ~ 0 rather than being scattered to
alpha ~ pi by an arbitrary 2pi winding.  See ``_collision_angles``.

With theta_i, theta_j in [0, 2pi), alpha covers [0, 2pi) and beta covers
[-pi, pi) -- so beta separates head-on (beta = +pi/2) from back-to-back
(beta = -pi/2).  alpha is uniform under the ideal gas, but beta is *not* (its
weight is triangular, peaked at parallel headings beta = 0 and vanishing at
beta = +-pi), so the normalisation divides by the Monte-Carlo reference
``_angular_reference`` rather than by a flat weight.

The *bulk* g is histogrammed in (alpha, beta).  The *contact* density g_surf is
instead histogrammed in the (sin beta, cos alpha) plane, since the pair approach
rate is dr/dt = -2 v0 sin(beta) cos(alpha): contacts live where
sin(beta) cos(alpha) > 0, and folding beta -> sin(beta) also maps the sparsely
sampled beta ~ +-pi onto the well sampled sin(beta) ~ 0.  Its ideal-gas weight
is again captured by ``_angular_reference``.

In addition, both histograms are also binned on the *single-frame* angle

    psi = phi - theta_i          (partner's bearing in particle i's heading frame)

which ignores the partner's orientation entirely: g_bulk_psi(r, psi) and
g_surf_psi(psi) are the pair correlation / contact density around a tagged
particle averaged over the orientation of the second particle (psi = 0 is dead
ahead of the tagged particle; psi is binned in [-pi, pi)).  This is the map in
which the low-density Pe -> inf theory predicts an untouched forward cone
|psi| < arccos(a/r), tangent-caustic rims, and depletion wings at the rear
sides (see StructFactor_MIPS/depletion_HS.py).  Its ideal-gas measure is
exactly flat, so no Monte-Carlo reference is needed.  Note psi is related to
the collision-frame angles by psi = alpha + beta - pi/2.

Performance
-----------
Cell lists + Numba.  Dynamics use a fine cell list (size a); the g-histogram is
accumulated only every ``sample_every`` steps with a coarser cell list (size
r_max), so sampling is cheap.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from tqdm import tqdm

TWOPI = 2.0 * np.pi
HALFPI = 0.5 * np.pi


# --------------------------------------------------------------------------- #
# Low-level helpers                                                           #
# --------------------------------------------------------------------------- #
@njit(inline="always")
def _wrap(x: float, L: float) -> float:
    if x >= L:
        x -= L * np.floor(x / L)
    elif x < 0.0:
        x += L * (1.0 + np.floor((-x) / L))
    return x


@njit(inline="always")
def _min_image(dx: float, L: float) -> float:
    return dx - L * np.floor((dx + 0.5 * L) / L)


@njit(inline="always")
def _alpha_bin(alpha: float, n: int) -> int:
    """Bin alpha into [0, 2pi)."""
    b = int((alpha - TWOPI * np.floor(alpha / TWOPI)) * (n / TWOPI))
    if b >= n:
        b -= n
    return b


@njit(inline="always")
def _beta_bin(beta: float, n: int) -> int:
    """Bin beta into [-pi, pi) (period 2pi)."""
    f = beta + np.pi
    f -= TWOPI * np.floor(f / TWOPI)  # -> [0, 2pi)
    b = int(f * (n / TWOPI))
    if b >= n:
        b -= n
    return b


@njit(inline="always")
def _unit_bin(x: float, n: int) -> int:
    """Bin x in [-1, 1] into n uniform bins (clamped at the closed endpoints)."""
    b = int((x + 1.0) * (0.5 * n))
    if b < 0:
        b = 0
    elif b >= n:
        b = n - 1
    return b


@njit(inline="always")
def _collision_angles(phi: float, thi: float, thj: float):
    """Collision-frame angles (alpha, beta) for a pair, ordering (i, j).

        alpha = phi - (theta_i + theta_j)/2 + pi/2     (separation vs. mean heading)
        beta  = (theta_j - theta_i)/2                  (half the heading mismatch)

    with ``phi`` the separation azimuth atan2(d_y, d_x), d = r_j - r_i.

    The headings are wrapped into [0, 2pi) first.  The stored thetas are an
    unbounded random walk, so without this the half-sum (theta_i + theta_j)/2
    would jump by pi for an arbitrary 2pi winding -- scattering, e.g., a head-on
    contact from alpha ~ 0 to alpha ~ pi.  Wrapping gives each heading a unique
    representative, so the angles are single-valued.  With theta_i, theta_j in
    [0, 2pi) the half-difference beta lands in (-pi, pi)."""
    thi -= TWOPI * np.floor(thi / TWOPI)  # -> [0, 2pi)
    thj -= TWOPI * np.floor(thj / TWOPI)  # -> [0, 2pi)
    return phi - 0.5 * (thi + thj) + HALFPI, 0.5 * (thj - thi)


@njit
def _build_cells(pos, L, cell_size, head, linked, ncx, ncy):
    for c in range(head.size):
        head[c] = -1
    for i in range(pos.shape[0]):
        cx = int(pos[i, 0] / cell_size) % ncx
        cy = int(pos[i, 1] / cell_size) % ncy
        c = cx + ncx * cy
        linked[i] = head[c]
        head[c] = i


# --------------------------------------------------------------------------- #
# Overlap resolution (overdamped hard-core projection)                        #
# --------------------------------------------------------------------------- #
@njit
def _resolve(pos, a, L, head, linked, ncx, ncy, cell_size, n_sweeps):
    """Gauss-Seidel projection: push overlapping pairs apart along their line
    of centres by half the overlap each, repeated ``n_sweeps`` times."""
    N = pos.shape[0]
    a2 = a * a
    for _ in range(n_sweeps):
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
                            if 1e-18 < r2 < a2:
                                r = np.sqrt(r2)
                                corr = 0.5 * (a - r) / r
                                ddx = corr * dx
                                ddy = corr * dy
                                pos[i, 0] -= ddx
                                pos[i, 1] -= ddy
                                pos[j, 0] += ddx
                                pos[j, 1] += ddy
                        j = linked[j]


# --------------------------------------------------------------------------- #
# Contact detection (active hard-core constraints, before the projection moves)#
# --------------------------------------------------------------------------- #
@njit
def _detect_contacts(
    pos, a, L, head, linked, ncx, ncy, cell_size, contact_i, contact_j
):
    """Record the pairs that the propulsion step drove into overlap, i.e. the
    pairs ``_resolve`` is about to push apart on its first sweep.

    These are the *physical* contacts: in overdamped dynamics a hard-core
    constraint is active iff the two particles' self-propulsion is compressive
    ``(u_j - u_i)·n < 0``, which to lowest order in ``dt`` is exactly ``r < a``
    right after the propulsion sub-step.  Detecting them *here* -- before the
    projection moves anyone -- means a pair that the projection later shoves
    into a third particle is never mistaken for a contact (the spurious overlap
    only exists after a projection move).  Only the few overlapping pairs are
    stored; the count is returned."""
    N = pos.shape[0]
    a2 = a * a
    n_contact = 0
    max_contact = contact_i.shape[0]
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
                        if 1e-18 < r2 < a2 and n_contact < max_contact:
                            contact_i[n_contact] = i
                            contact_j[n_contact] = j
                            n_contact += 1
                    j = linked[j]
    return n_contact


# --------------------------------------------------------------------------- #
# Histogram accumulation (bulk + surface), binned directly in (alpha, beta)    #
# --------------------------------------------------------------------------- #
@njit
def _accumulate(
    pos,
    theta,
    a,
    L,
    r_max,
    contact_band,
    inv_dr,
    n_r,
    n_alpha,
    n_beta,
    n_psi,
    hist_bulk,
    hist_psi,
    head,
    linked,
    ncx,
    ncy,
    cell_size,
):
    """Bin every neighbour pair (both orderings) into the bulk histogram using
    the collision-frame angles (alpha, beta), and into the single-frame
    histogram using psi = phi - theta (the partner's bearing in each particle's
    own heading frame, partner orientation ignored).  Pairs within the contact
    shell r <= a*(1+contact_band) are skipped -- those are the contacts, which
    are accumulated separately from the projection (see
    ``_accumulate_contacts``) -- so they do not pollute the first radial bin."""
    N = pos.shape[0]
    r_contact = a * (1.0 + contact_band)
    rmax2 = r_max * r_max
    for i in range(N):
        thi = theta[i]
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
                        if r_contact * r_contact < r2 < rmax2:
                            r = np.sqrt(r2)
                            thj = theta[j]
                            phi = np.arctan2(dy, dx)
                            al, be = _collision_angles(phi, thi, thj)
                            # ordering (i, j)
                            ia = _alpha_bin(al, n_alpha)
                            ib = _beta_bin(be, n_beta)
                            # ordering (j, i): separation reversed -> alpha+pi, beta->-beta
                            ja = _alpha_bin(al + np.pi, n_alpha)
                            jb = _beta_bin(-be, n_beta)
                            kr = int((r - a) * inv_dr)
                            if 0 <= kr < n_r:
                                hist_bulk[kr, ia, ib] += 1.0
                                hist_bulk[kr, ja, jb] += 1.0
                                # single-frame angle psi = phi - theta, one
                                # entry per ordering ((j, i) sees -d, i.e.
                                # phi + pi, and its own heading thj)
                                hist_psi[kr, _beta_bin(phi - thi, n_psi)] += 1.0
                                hist_psi[kr, _beta_bin(phi + np.pi - thj, n_psi)] += 1.0
                    j = linked[j]


@njit
def _accumulate_contacts(
    pos,
    theta,
    L,
    n_cosa,
    n_sinb,
    n_psi,
    hist_surf,
    hist_surf_psi,
    contact_i,
    contact_j,
    n_contact,
):
    """Bin the contact pairs found by ``_detect_contacts`` into the surface
    histogram in the (sin beta, cos alpha) plane, both orderings, and into the
    single-frame surface histogram over psi = phi - theta.

    The pair approach rate is ``dr/dt = -2 v0 sin(beta) cos(alpha)``, so
    (sin beta, cos alpha) are the natural contact variables -- a compressive
    contact has ``sin(beta) cos(alpha) > 0``.  Axis 0 of ``hist_surf`` is
    sin(beta) (``n_sinb`` bins over [-1, 1]); axis 1 is cos(alpha) (``n_cosa``
    bins over [-1, 1]).  The (j, i) ordering sends (alpha, beta) ->
    (alpha+pi, -beta), i.e. (sin beta, cos alpha) -> (-sin beta, -cos alpha).
    Angles are evaluated on the sampled configuration (post-projection,
    post-rotation), consistent with the bulk pairs."""
    for k in range(n_contact):
        i = contact_i[k]
        j = contact_j[k]
        dx = _min_image(pos[j, 0] - pos[i, 0], L)
        dy = _min_image(pos[j, 1] - pos[i, 1], L)
        thi = theta[i]
        thj = theta[j]
        phi = np.arctan2(dy, dx)
        al, be = _collision_angles(phi, thi, thj)
        s = np.sin(be)  # sin(beta)
        c = np.cos(al)  # cos(alpha)
        # ordering (i, j)
        hist_surf[_unit_bin(s, n_sinb), _unit_bin(c, n_cosa)] += 1.0
        # ordering (j, i)
        hist_surf[_unit_bin(-s, n_sinb), _unit_bin(-c, n_cosa)] += 1.0
        # single-frame angle psi = phi - theta, one entry per ordering
        hist_surf_psi[_beta_bin(phi - thi, n_psi)] += 1.0
        hist_surf_psi[_beta_bin(phi + np.pi - thj, n_psi)] += 1.0


# --------------------------------------------------------------------------- #
# Main driver                                                                  #
# --------------------------------------------------------------------------- #
@njit
def _run_chunk(
    pos,
    theta,
    a,
    L,
    v0,
    Dr,
    Dt,
    dt,
    step_start,
    n_chunk,
    burn_in,
    sample_every,
    n_sweeps,
    r_max,
    contact_band,
    n_r,
    n_alpha,
    n_beta,
    n_psi,
    hist_bulk,
    hist_surf,
    hist_psi,
    hist_surf_psi,
):
    """Advance the system by ``n_chunk`` steps starting at absolute step
    ``step_start``, accumulating into the (in-place) histograms.  Returns the
    number of samples taken in this chunk."""
    N = pos.shape[0]
    inv_dr = n_r / (r_max - a)

    # Fine cell list for the dynamics (cell size a).
    cs_d = a
    ncx_d = max(3, int(L / cs_d))
    ncy_d = ncx_d
    head_d = np.empty(ncx_d * ncy_d, dtype=np.int64)
    linked_d = np.empty(N, dtype=np.int64)

    # Coarse cell list for sampling (cell size r_max).
    cs_a = r_max
    ncx_a = max(3, int(L / cs_a))
    ncy_a = ncx_a
    head_a = np.empty(ncx_a * ncy_a, dtype=np.int64)
    linked_a = np.empty(N, dtype=np.int64)

    # Buffers for the contact pairs flagged by _detect_contacts on a sample step
    # (each disk has at most ~6 neighbours in 2D, so <= 3N pairs; allow margin).
    max_contact = 8 * N
    contact_i = np.empty(max_contact, dtype=np.int64)
    contact_j = np.empty(max_contact, dtype=np.int64)

    sig_t = np.sqrt(2.0 * Dt * dt) if Dt > 0.0 else 0.0
    sig_r = np.sqrt(2.0 * Dr * dt) if Dr > 0.0 else 0.0
    n_samples = 0

    for s in range(n_chunk):
        step = step_start + s
        # --- propulsion (+ optional translational noise) ---
        for i in range(N):
            th = theta[i]
            pos[i, 0] += dt * v0 * np.cos(th)
            pos[i, 1] += dt * v0 * np.sin(th)
            if sig_t > 0.0:
                pos[i, 0] += sig_t * np.random.normal()
                pos[i, 1] += sig_t * np.random.normal()

        # --- hard-core projection ---
        is_sample = step >= burn_in and (step % sample_every == 0)
        _build_cells(pos, L, cs_d, head_d, linked_d, ncx_d, ncy_d)
        # On sample steps, flag the contacts (post-propulsion overlaps) *before*
        # the projection moves anyone, so projection-induced overlaps are never
        # mistaken for physical contacts.
        n_contact = 0
        if is_sample:
            n_contact = _detect_contacts(
                pos, a, L, head_d, linked_d, ncx_d, ncy_d, cs_d, contact_i, contact_j
            )
        _resolve(pos, a, L, head_d, linked_d, ncx_d, ncy_d, cs_d, n_sweeps)

        # --- wrap into the box ---
        for i in range(N):
            pos[i, 0] = _wrap(pos[i, 0], L)
            pos[i, 1] = _wrap(pos[i, 1], L)

        # --- rotational diffusion ---
        if sig_r > 0.0:
            for i in range(N):
                theta[i] += sig_r * np.random.normal()

        # --- sample ---
        if is_sample:
            _build_cells(pos, L, cs_a, head_a, linked_a, ncx_a, ncy_a)
            _accumulate(
                pos,
                theta,
                a,
                L,
                r_max,
                contact_band,
                inv_dr,
                n_r,
                n_alpha,
                n_beta,
                n_psi,
                hist_bulk,
                hist_psi,
                head_a,
                linked_a,
                ncx_a,
                ncy_a,
                cs_a,
            )
            _accumulate_contacts(
                pos,
                theta,
                L,
                n_alpha,
                n_beta,
                n_psi,
                hist_surf,
                hist_surf_psi,
                contact_i,
                contact_j,
                n_contact,
            )
            n_samples += 1

    return n_samples


# --------------------------------------------------------------------------- #
# Plain dynamics stepping (for visualisation; no histogramming)                #
# --------------------------------------------------------------------------- #
@njit
def _advance(pos, theta, a, L, v0, Dr, Dt, dt, n_steps, n_sweeps):
    """Advance the system ``n_steps`` steps in place (propulsion, hard-core
    projection, wrap, rotational diffusion) without sampling.  Used to render a
    movie frame by frame."""
    N = pos.shape[0]
    cs_d = a
    ncx_d = max(3, int(L / cs_d))
    ncy_d = ncx_d
    head_d = np.empty(ncx_d * ncy_d, dtype=np.int64)
    linked_d = np.empty(N, dtype=np.int64)

    sig_t = np.sqrt(2.0 * Dt * dt) if Dt > 0.0 else 0.0
    sig_r = np.sqrt(2.0 * Dr * dt) if Dr > 0.0 else 0.0

    for _ in range(n_steps):
        for i in range(N):
            th = theta[i]
            pos[i, 0] += dt * v0 * np.cos(th)
            pos[i, 1] += dt * v0 * np.sin(th)
            if sig_t > 0.0:
                pos[i, 0] += sig_t * np.random.normal()
                pos[i, 1] += sig_t * np.random.normal()

        _build_cells(pos, L, cs_d, head_d, linked_d, ncx_d, ncy_d)
        _resolve(pos, a, L, head_d, linked_d, ncx_d, ncy_d, cs_d, n_sweeps)

        for i in range(N):
            pos[i, 0] = _wrap(pos[i, 0], L)
            pos[i, 1] = _wrap(pos[i, 1], L)

        if sig_r > 0.0:
            for i in range(N):
                theta[i] += sig_r * np.random.normal()


# --------------------------------------------------------------------------- #
# Public interface                                                             #
# --------------------------------------------------------------------------- #
class HardSphereABP:
    """Hard-sphere active Brownian particle simulation + pair correlation.

    Parameters
    ----------
    N : int
        Number of particles.
    phi : float
        Packing fraction  phi = N * pi (a/2)^2 / L^2  (sets the box size L).
    v0 : float
        Self-propulsion speed.
    Dr : float
        Rotational diffusion rate.
    a : float
        Particle diameter.
    Dt : float, optional
        Translational diffusion (default 0 -> genuine contact delta).
    r_max : float, optional
        Maximum distance for the bulk pair correlation (default 4a).
    n_r : int
        Radial bins for r in (a, r_max).
    n_alpha, n_beta : int
        Bin counts for the two angular axes.  ``g_bulk`` is binned in
        (alpha in [0, 2pi), beta in [-pi, pi)) with (n_alpha, n_beta) bins.
        ``g_surf`` is binned in the (sin beta, cos alpha) plane (both in
        [-1, 1]) with sin beta on axis 0 (n_beta bins) and cos alpha on axis 1
        (n_alpha bins).  The single-frame histograms ``g_bulk_psi`` /
        ``g_surf_psi`` use n_alpha bins over psi = phi - theta_i in [-pi, pi)
        (partner's bearing in the tagged particle's heading frame, partner
        orientation averaged out; psi = 0 is dead ahead).
    dt : float, optional
        Time step (default 0.01 a / v0, so propulsion moves << a per step).
    n_sweeps : int
        Gauss-Seidel projection sweeps per step.
    contact_band : float
        Width of the contact shell excluded from the bulk histogram, r <=
        a (1 + contact_band).  Contacts themselves are not defined by this
        threshold: a contact is a pair whose hard-core constraint is active,
        detected (in ``_detect_contacts``) as a post-propulsion overlap before
        the projection moves anyone.  The band only keeps those pairs (which sit
        at r ~ a after projection) out of the first bulk bin.
    seed : int
        RNG seed.
    verbose : bool
        Print a banner/summary and show a progress bar.

    Example
    -------
    >>> sim = HardSphereABP(N=2000, phi=0.3, v0=1.0, Dr=0.1)
    >>> sim.run(n_steps=200_000, burn_in=20_000, sample_every=50, save_path="run")
    >>> g_bulk = sim.g_bulk          # (n_r, n_alpha, n_beta) over (r, alpha, beta)
    >>> g_surf = sim.g_surf          # (n_beta, n_alpha) over (sin beta, cos alpha)
    >>> g_psi = sim.g_bulk_psi       # (n_r, n_alpha) over (r, psi), theta_j-averaged
    >>> gs_psi = sim.g_surf_psi      # (n_alpha,) contact density over psi
    """

    def __init__(
        self,
        N: int,
        phi: float,
        v0: float,
        Dr: float,
        a: float = 1.0,
        Dt: float = 0.0,
        r_max: float | None = None,
        n_r: int = 60,
        n_alpha: int = 72,
        n_beta: int = 36,
        dt: float | None = None,
        n_sweeps: int = 20,
        contact_band: float = 1e-3,
        seed: int = 0,
        verbose: bool = True,
    ):
        self.N = int(N)
        self.phi = phi
        self.a = a
        self.v0 = v0
        self.Dr = Dr
        self.Dt = Dt
        self.r_max = 4.0 * a if r_max is None else r_max
        self.n_r = int(n_r)
        self.n_alpha = int(n_alpha)
        self.n_beta = int(n_beta)
        self.dt = (0.01 * a / v0) if dt is None else dt
        self.n_sweeps = int(n_sweeps)
        self.contact_band = contact_band
        self.seed = seed
        self.verbose = verbose

        # Box size from packing fraction.
        self.L = np.sqrt(self.N * np.pi * a * a / (4.0 * phi))
        self.rho = self.N / (self.L * self.L)

        # Coordinate grids.
        self.r_edges = np.linspace(a, self.r_max, self.n_r + 1)
        self.r = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        self.alpha_edges = np.linspace(0.0, TWOPI, self.n_alpha + 1)
        self.alpha = 0.5 * (self.alpha_edges[:-1] + self.alpha_edges[1:])
        self.beta_edges = np.linspace(-np.pi, np.pi, self.n_beta + 1)
        self.beta = 0.5 * (self.beta_edges[:-1] + self.beta_edges[1:])
        # g_surf is binned in the (sin beta, cos alpha) plane (both in [-1, 1]):
        # sin beta uses n_beta bins (axis 0), cos alpha uses n_alpha bins (axis 1).
        self.sin_beta_edges = np.linspace(-1.0, 1.0, self.n_beta + 1)
        self.sin_beta = 0.5 * (self.sin_beta_edges[:-1] + self.sin_beta_edges[1:])
        self.cos_alpha_edges = np.linspace(-1.0, 1.0, self.n_alpha + 1)
        self.cos_alpha = 0.5 * (self.cos_alpha_edges[:-1] + self.cos_alpha_edges[1:])
        # Single-frame angle psi = phi - theta_i in [-pi, pi) (partner bearing
        # in the tagged particle's heading frame, partner orientation ignored).
        self.psi_edges = np.linspace(-np.pi, np.pi, self.n_alpha + 1)
        self.psi = 0.5 * (self.psi_edges[:-1] + self.psi_edges[1:])

        self.g_bulk = None
        self.g_surf = None
        self.g_bulk_psi = None
        self.g_surf_psi = None
        self.g_of_r = None
        self.n_samples = 0
        self._n_contacts = 0.0

    # ------------------------------------------------------------------ #
    def _initial_state(self):
        """Random non-overlapping-ish start; a few projection sweeps clean it."""
        rng = np.random.default_rng(self.seed)
        pos = rng.uniform(0.0, self.L, size=(self.N, 2))
        theta = rng.uniform(0.0, TWOPI, size=self.N)
        # Relax initial overlaps.
        cs = self.a
        ncx = max(3, int(self.L / cs))
        head = np.empty(ncx * ncx, dtype=np.int64)
        linked = np.empty(self.N, dtype=np.int64)
        for _ in range(50):
            _build_cells(pos, self.L, cs, head, linked, ncx, ncx)
            _resolve(pos, self.a, self.L, head, linked, ncx, ncx, cs, 1)
            pos[:, 0] = np.mod(pos[:, 0], self.L)
            pos[:, 1] = np.mod(pos[:, 1], self.L)
        return pos, theta

    def run(
        self,
        n_steps: int,
        burn_in: int = 0,
        sample_every: int = 50,
        chunk: int | None = None,
        save_path: str | None = None,
    ):
        """Run the simulation and compute g_bulk(r, alpha, beta) and
        g_surf(alpha, beta).

        The integration is split into chunks so a progress bar can update
        between them (the Numba inner loop cannot be interrupted otherwise).
        Set ``self.verbose = False`` to silence the bar and the summary prints.
        If ``save_path`` is given, the results are written there (see ``save``).
        """
        n_steps = int(n_steps)
        burn_in = int(burn_in)
        sample_every = int(sample_every)
        if chunk is None:
            chunk = max(sample_every, n_steps // 100)
        chunk = max(1, int(chunk))

        if self.verbose:
            if self.Dr == 0:
                print(
                    f"HardSphereABP: N={self.N}, phi={self.phi}, L={self.L:.3f}, "
                    f"rho={self.rho:.4f}, a={self.a}, v0={self.v0}, Dr={self.Dr}, "
                    f"Dt={self.Dt}, dt={self.dt:.4g}, Pe=inf"
                )
            else:
                print(
                    f"HardSphereABP: N={self.N}, phi={self.phi}, L={self.L:.3f}, "
                    f"rho={self.rho:.4f}, a={self.a}, v0={self.v0}, Dr={self.Dr}, "
                    f"Dt={self.Dt}, dt={self.dt:.4g}, Pe={self.v0 / (self.a * self.Dr):.3g}"
                )

        np.random.seed(self.seed)
        pos, theta = self._initial_state()

        hist_bulk = np.zeros((self.n_r, self.n_alpha, self.n_beta))
        # g_surf is binned in the (sin beta, cos alpha) plane: axis 0 sin beta
        # (n_beta bins), axis 1 cos alpha (n_alpha bins).
        hist_surf = np.zeros((self.n_beta, self.n_alpha))
        # single-frame histograms over psi = phi - theta_i (n_alpha bins).
        hist_psi = np.zeros((self.n_r, self.n_alpha))
        hist_surf_psi = np.zeros(self.n_alpha)
        n_samples = 0

        bar = tqdm(
            total=n_steps,
            disable=not self.verbose,
            unit="step",
            unit_scale=True,
            desc="simulating",
        )
        step = 0
        while step < n_steps:
            n_chunk = min(chunk, n_steps - step)
            n_samples += _run_chunk(
                pos,
                theta,
                self.a,
                self.L,
                self.v0,
                self.Dr,
                self.Dt,
                self.dt,
                step,
                n_chunk,
                burn_in,
                sample_every,
                self.n_sweeps,
                self.r_max,
                self.contact_band,
                self.n_r,
                self.n_alpha,
                self.n_beta,
                self.n_alpha,
                hist_bulk,
                hist_surf,
                hist_psi,
                hist_surf_psi,
            )
            step += n_chunk
            bar.update(n_chunk)
            bar.set_postfix(samples=n_samples)
        bar.close()

        self.n_samples = n_samples
        self.pos, self.theta = pos, theta
        self._normalize(hist_bulk, hist_surf, hist_psi, hist_surf_psi)

        if self.verbose:
            print(
                f"done: {n_samples} samples, "
                f"contacts/particle = {self.contact_fraction():.3f}"
            )
        if save_path is not None:
            self.save(save_path)
        return self.g_bulk, self.g_surf

    # ------------------------------------------------------------------ #
    def animate(
        self,
        n_steps: int = 4000,
        frame_every: int = 20,
        interval: int = 40,
        save_path: str | None = None,
        fps: int = 25,
        dpi: int = 120,
        color_by_orientation: bool = True,
        seed: int | None = None,
    ):
        """Render a movie of the disks moving under the dynamics.

        Runs an independent trajectory (it does **not** touch the histograms or
        any state set by :meth:`run`): the system is initialised, then advanced
        ``frame_every`` steps per frame for a total of ``n_steps`` steps.  Disks
        are drawn at their true diameter ``a`` and, by default, coloured by
        orientation with a cyclic colormap.

        Parameters
        ----------
        n_steps : int
            Total dynamics steps to play through.
        frame_every : int
            Steps advanced between captured frames (sets the time resolution).
        interval : int
            Delay between frames in ms for on-screen playback.
        save_path : str, optional
            If given, write the movie here.  ``.gif`` uses the Pillow writer,
            anything else (e.g. ``.mp4``) uses ffmpeg.  If omitted the animation
            is shown interactively with ``plt.show()``.
        fps : int
            Frames per second when saving.
        dpi : int
            Resolution when saving.
        color_by_orientation : bool
            Colour each disk by its heading ``theta`` (cyclic ``hsv``); if
            False all disks share one colour.
        seed : int, optional
            Override the simulation seed for this movie only.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation object (keep a reference alive while it plays).
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import EllipseCollection

        n_steps = int(n_steps)
        frame_every = max(1, int(frame_every))
        n_frames = n_steps // frame_every

        np.random.seed(self.seed if seed is None else seed)
        pos, theta = self._initial_state()

        # Collect frames (independent of run()'s state).
        frames_pos = [pos.copy()]
        frames_theta = [theta.copy()]
        bar = tqdm(
            total=n_frames, disable=not self.verbose, desc="rendering", unit="frame"
        )
        for _ in range(n_frames):
            _advance(
                pos,
                theta,
                self.a,
                self.L,
                self.v0,
                self.Dr,
                self.Dt,
                self.dt,
                frame_every,
                self.n_sweeps,
            )
            frames_pos.append(pos.copy())
            frames_theta.append(theta.copy())
            bar.update(1)
        bar.close()

        # Set up the figure: a box of side L with true-size disks.
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0.0, self.L)
        ax.set_ylim(0.0, self.L)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"N={self.N}, $\\phi$={self.phi}, " f"Pe={self.v0 / (self.a * self.Dr):.0f}"
        )

        coll = EllipseCollection(
            widths=self.a,
            heights=self.a,
            angles=0.0,
            units="xy",
            offsets=frames_pos[0],
            offset_transform=ax.transData,
            edgecolor="k",
            linewidth=0.3,
            cmap="hsv",
        )
        if color_by_orientation:
            coll.set_array(np.mod(frames_theta[0], TWOPI))
            coll.set_clim(0.0, TWOPI)
        else:
            coll.set_facecolor("tab:blue")
        ax.add_collection(coll)

        def update(f):
            coll.set_offsets(frames_pos[f])
            if color_by_orientation:
                coll.set_array(np.mod(frames_theta[f], TWOPI))
            return (coll,)

        anim = FuncAnimation(
            fig,
            update,
            frames=len(frames_pos),
            interval=interval,
            blit=True,
        )

        if save_path is not None:
            writer = "pillow" if save_path.lower().endswith(".gif") else "ffmpeg"
            anim.save(save_path, writer=writer, fps=fps, dpi=dpi)
            if self.verbose:
                print(f"saved movie to {save_path}")
        else:
            plt.show()
        return anim

    # ------------------------------------------------------------------ #
    def _angular_reference(self, n_mc: int = 10_000_000):
        """Ideal-gas reference weights (each normalised to sum to 1):

            w_bulk(alpha, beta)         for the bulk histogram, and
            w_surf(sin beta, cos alpha) for the contact histogram.

        These are the fractions of uniformly-oriented pairs falling in each
        cell.  Doing it by Monte-Carlo captures the exact (non-flat) measures --
        triangular in beta, and the arcsine-like measures of sin beta / cos
        alpha -- so they divide out consistently with how the data are binned."""
        rng = np.random.default_rng(self.seed + 1)
        ti = rng.uniform(0.0, TWOPI, n_mc)
        tj = rng.uniform(0.0, TWOPI, n_mc)
        phi = rng.uniform(0.0, TWOPI, n_mc)
        # Same angles as _collision_angles (ti, tj are already in [0, 2pi)).
        al = np.mod(phi - 0.5 * (ti + tj) + HALFPI, TWOPI)
        be = 0.5 * (tj - ti)  # in (-pi, pi)
        w_bulk, _, _ = np.histogram2d(al, be, bins=[self.alpha_edges, self.beta_edges])
        w_surf, _, _ = np.histogram2d(
            np.sin(be), np.cos(al), bins=[self.sin_beta_edges, self.cos_alpha_edges]
        )
        return w_bulk / w_bulk.sum(), w_surf / w_surf.sum()

    def _normalize(self, hist_bulk, hist_surf, hist_psi, hist_surf_psi):
        """Divide the histograms by the ideal-gas (g=1) expectation.

        The angular dependence factorises through the reference weights (bulk in
        (alpha, beta), surface in (sin beta, cos alpha)); the radial/contact
        measures are
            bulk : pi (r_{k+1}^2 - r_k^2)
            surf : 2 pi a            (contact circle, carries 1/length)
        The single-frame angle psi = phi - theta_i is exactly uniform under the
        ideal gas (phi and theta_i are independent and uniform), so its
        reference weight is flat: 1/n_psi per bin.
        """
        w_bulk, w_surf = self._angular_reference()  # each sums to 1
        pref = self.n_samples * self.N * self.rho
        self._n_contacts = float(hist_surf.sum())

        # Bulk: g_bulk(r, alpha, beta).
        ring = np.pi * (self.r_edges[1:] ** 2 - self.r_edges[:-1] ** 2)
        denom_bulk = pref * ring[:, None, None] * w_bulk[None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            self.g_bulk = np.where(denom_bulk > 0, hist_bulk / denom_bulk, np.nan)

        # Angle-averaged bulk g(r) (radial marginal, robust validation -> 1).
        self.g_of_r = hist_bulk.sum(axis=(1, 2)) / (pref * ring)

        # Surface (contact) density g_surf(sin beta, cos alpha), units of length.
        denom_surf = pref * (TWOPI * self.a) * w_surf
        with np.errstate(invalid="ignore", divide="ignore"):
            self.g_surf = np.where(denom_surf > 0, hist_surf / denom_surf, np.nan)

        # Single-frame histograms: flat psi measure (1/n_psi per bin).
        n_psi = hist_psi.shape[1]
        self.g_bulk_psi = hist_psi / (pref * ring[:, None] / n_psi)
        self.g_surf_psi = hist_surf_psi / (pref * (TWOPI * self.a) / n_psi)

    # ------------------------------------------------------------------ #
    def contact_fraction(self) -> float:
        """Mean number of contacts per particle (sanity diagnostic)."""
        if self.n_samples == 0:
            return 0.0
        return self._n_contacts / (self.N * self.n_samples)

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> str:
        """Save all results + parameters to a compressed ``.npz`` file.

        Reload with ``np.load(path)``; scalars come back as 0-d arrays."""
        if not path.endswith(".npz"):
            path = path + ".npz"
        np.savez_compressed(
            path,
            # coordinate grids
            r=self.r,
            alpha=self.alpha,
            beta=self.beta,
            r_edges=self.r_edges,
            alpha_edges=self.alpha_edges,
            beta_edges=self.beta_edges,
            # g_surf is binned in (sin beta, cos alpha)
            sin_beta=self.sin_beta,
            cos_alpha=self.cos_alpha,
            sin_beta_edges=self.sin_beta_edges,
            cos_alpha_edges=self.cos_alpha_edges,
            # single-frame angle psi = phi - theta_i
            psi=self.psi,
            psi_edges=self.psi_edges,
            # correlation functions
            g_bulk=self.g_bulk,
            g_surf=self.g_surf,
            g_bulk_psi=self.g_bulk_psi,
            g_surf_psi=self.g_surf_psi,
            g_of_r=self.g_of_r,
            # parameters
            N=self.N,
            phi=self.phi,
            a=self.a,
            v0=self.v0,
            Dr=self.Dr,
            Dt=self.Dt,
            L=self.L,
            rho=self.rho,
            dt=self.dt,
            r_max=self.r_max,
            n_r=self.n_r,
            n_alpha=self.n_alpha,
            n_beta=self.n_beta,
            n_sweeps=self.n_sweeps,
            contact_band=self.contact_band,
            seed=self.seed,
            n_samples=self.n_samples,
            contacts_per_particle=self.contact_fraction(),
        )
        if self.verbose:
            print(f"saved results to {path}")
        return path


# --------------------------------------------------------------------------- #
# Example                                                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for Dr in [0.005, 0.001]:
        if Dr == 0.0:
            save_path = f"hard_sphere_abp_lowphi_Peinf.npz"
        else:
            save_path = f"hard_sphere_abp_lowphi_Pe{10.0/Dr:g}.npz"
        sim = HardSphereABP(
            N=10000,
            phi=0.01,
            a=1.0,
            v0=10.0,
            Dr=Dr,
            n_r=60,
            n_alpha=72,
            n_beta=72,
        )
        sim.run(
            n_steps=200_000,
            burn_in=2_000,
            sample_every=20,
            save_path=save_path,
        )

    # g = np.load("hard_sphere_abp_Pe20.npz")
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="polar")
    # r = g["r"]
    # alpha = g["alpha"]
    # beta = g["beta"]
    # g_bulk = g["g_bulk"]
    # rr, aa = np.meshgrid(r, alpha, indexing="ij")
    # # Plot g_bulk in the (r, \alpha) plane
    # c = ax.pcolormesh(aa, rr, g_bulk[:, :, 0], shading="auto", cmap="viridis")
    # ax.set_title("g_bulk(r, alpha) at beta=0")
    # fig.colorbar(c, ax=ax, label="g_bulk")

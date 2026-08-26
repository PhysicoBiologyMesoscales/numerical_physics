"""
Contact pair-correlation g_surf of an *isolated* pair of hard-sphere ABPs,
sampled from a large ensemble of independent two-particle systems.

Why this exists
---------------
In the full many-body simulation (``hard_sphere_abp.py``) the Pe -> inf
(Dr = 0) contact correlation ``g_surf(sin beta, cos alpha)`` should be exactly
1 on its support ``sin(beta) cos(alpha) > 0`` -- this is the flux-balance result
for an isolated glued pair.  The many-body data instead show a boundary layer:
a dip of ``g_surf`` for ``sin(beta) -> 0`` that grows linearly with packing
fraction ``phi``.  The mechanism is *three-body interruption*: a nearly aligned
pair (small ``sin beta``) stays glued for a time ``~ a / (2 v0 sin beta)``, which
diverges as ``sin beta -> 0``; once that lifetime exceeds the mean free time to
be hit by a third particle, the contact is broken before the pair finishes
sliding, depleting the small-``sin beta`` contact statistics.

This module removes the third particle *by construction*: it evolves a huge
ensemble of independent 2-particle systems, so a glued pair can only leave
contact by sliding off naturally (reaching the tangent point cos(alpha) -> 0).
If the boundary layer is really a finite-density effect, ``g_surf`` here must be
flat and equal to 1 all the way down to ``sin beta = 0``.

Method
------
At Dr = 0 each particle's heading is frozen, so for a pair (i, j) the relative
coordinate ``d = r_j - r_i`` obeys purely ballistic motion at the constant
relative velocity

    u_rel = v0 (e_j - e_i),   |u_rel| = 2 v0 |sin beta|,

with a single hard-core constraint ``|d| >= a``.  We therefore integrate only
``d`` on a periodic box (a point drifting past one excluded disk of radius ``a``
at the origin).  This is exactly the two-body reduction of the many-body update
in ``hard_sphere_abp.py``:

    * propulsion            d += dt u_rel
    * nearest image         d -= L round(d / L)
    * contact detection     flag  1e-18 < |d|^2 < a^2   (post-propulsion overlap)
    * hard-core projection  d *= a / |d|   (the 2-body limit of the half-overlap
                                            Gauss-Seidel push: d_new = d a/|d|)

Contacts are accumulated in the (sin beta, cos alpha) plane exactly as in
``_accumulate_contacts`` (both orderings), and normalised against the same
Monte-Carlo ideal-gas reference.  The ideal ordered-pair contact measure per box
is ``N (N-1) / L^2 * 2 pi a = 2 / L^2 * 2 pi a`` (exactly, using N = 2), summed
over the ``M`` boxes.

Molecular chaos (``rerandomize=True``, default)
-----------------------------------------------
At Dr = 0 a single pair in a periodic box is *deterministic* (an
infinite-horizon Lorentz gas): every contact ends with a grazing exit tangent
to the disk, and these tangent trajectories re-impinge on the disk instead of
decorrelating.  The stationary free density is then non-uniform -- a
``(r - a)^(-1/2)`` caustic at the disk and a depleted far field -- which
inflates the ``g_surf`` plateau by a factor growing like ``~ 0.74 ln(L/a) +
0.30`` (measured) while leaving its *shape* flat.

To restore molecular chaos, ``rerandomize=True`` re-emits the pair from the
disk surface each time it detaches: the relative coordinate is placed at the
downstream intersection of the circle ``|d| = a`` with a chord of uniform
random impact parameter ``b in (-a, a)``, moving away along ``u_rel``.  The
free flights are then exactly the Kac chord ensemble of the periodic Lorentz
gas -- uniform stationary free-space density, mean free path
``(L^2 - pi a^2) / (2 a)``, and a uniform incoming impact parameter at the next
collision -- which is the molecular-chaos flux for which ``g_surf = 1`` is
exact.  (Teleporting to a uniform *position* instead does NOT work: by the
renewal inspection paradox the mean flight after such a teleport is the
length-biased ``<l^2>/(2<l>)`` rather than the Kac ``<l>``, so the collision
rate no longer matches the box-mean density reference and the plateau stays
biased, ~1.18 at L = 4.)

Rotational diffusion (``Dr > 0``)
---------------------------------
With ``Dr > 0`` both headings diffuse (``theta += sqrt(2 Dr dt) xi``), the
relative velocity is rebuilt every step, and the collision angles are computed
from the *current* headings at each contact sample -- still the exact two-body
reduction of the many-body update.  Two finite-``Dr`` subtleties in the
re-emission rule:

* a contact at finite Dr detaches and re-attaches many times before the pair
  truly separates; re-emitting at the first detachment would cut these
  sequences short and distort g_surf.  Re-emission therefore only triggers
  when the separation first exceeds ``r_reemit`` (default ``2 a`` at
  ``Dr > 0``) after a contact, so the near field ``r < r_reemit`` keeps the
  true two-body dynamics.  At ``Dr = 0`` the default ``r_reemit = a``
  reproduces the validated Pe = infinity scheme exactly (detachment there is
  final, so the two rules coincide).
* at re-emission both headings are redrawn -- the next collision must be with
  a fresh, uncorrelated partner (molecular chaos); keeping them would let
  successive collisions share heading memory whenever the free flight is
  shorter than the persistence time 1/Dr.  The redraw is *flux-weighted*, not
  uniform: incoming pairs arrive at a rate proportional to
  ``|u_rel| = 2 v0 |sin(beta)|``, so ``theta_i`` is uniform and
  ``delta = theta_j - theta_i`` is drawn from the ``|sin(delta/2)|`` measure
  (``delta = 2 arccos(1 - 2 u)``).  This regenerates the uniform free-space
  heading measure and the correct per-beta collision rates for any Dr.  (A
  uniform redraw would make all beta collide equally often; at ``Dr = 0`` the
  headings are simply kept, since a fixed-beta pair produces the ``v_rel``
  collision-rate weighting dynamically -- the validated case.)
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

TWOPI = 2.0 * np.pi
HALFPI = 0.5 * np.pi


@njit(inline="always")
def _unit_bin(x: float, n: int) -> int:
    """Bin x in [-1, 1] into n uniform bins (clamped at the closed endpoints).
    Identical to hard_sphere_abp._unit_bin so the two g_surf are comparable."""
    b = int((x + 1.0) * (0.5 * n))
    if b < 0:
        b = 0
    elif b >= n:
        b = n - 1
    return b


@njit(inline="always")
def _xorshift_uniform(st):
    """Advance one xorshift64 state; return (new_state, uniform in [0, 1))."""
    st ^= st << np.uint64(13)
    st ^= st >> np.uint64(7)
    st ^= st << np.uint64(17)
    return st, (st >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(parallel=True, cache=True, fastmath=True)
def _run_pairs(
    d0,
    thi,
    thj,
    a,
    L,
    v0,
    Dr,
    dt,
    n_steps,
    burn_in,
    sample_every,
    n_sinb,
    n_cosa,
    n_blocks,
    n_r,
    r_max,
    rerandomize,
    r_reemit,
    rng_seed,
):
    """Evolve ``M`` independent pairs (relative coordinate only) and bin their
    contacts into ``hist_surf`` (n_sinb, n_cosa) and their non-contact
    separations into a radial ``hist_bulk`` (n_r) over r in (a, r_max).

    At ``Dr > 0`` both headings perform rotational diffusion and the relative
    velocity ``u_rel = v0 (e_j - e_i)`` is rebuilt every step; collision angles
    are evaluated from the current headings at each contact sample, exactly as
    ``hard_sphere_abp._collision_angles`` does.

    If ``rerandomize`` is true, a pair is re-emitted from the disk surface at a
    uniform random impact parameter, moving away along ``u_rel``, the first
    time its separation exceeds ``r_reemit`` after a contact; at ``Dr > 0`` the
    headings are also redrawn from the flux-weighted measure at that moment
    (molecular chaos; see module docstring).  Each pair carries its own
    xorshift64* stream seeded from ``rng_seed`` and the pair index, so the
    parallel blocks stay independent and reproducible.

    Parallelism: the ``M`` pairs are split into ``n_blocks`` contiguous blocks;
    block ``b`` (one prange iteration) writes only to ``hist[b]`` / ``histr[b]``,
    so there is no write race and no dependence on thread-id APIs.  The per-block
    histograms are summed at the end.  Returns (hist_surf, hist_bulk_r,
    n_contacts)."""
    M = d0.shape[0]
    a2 = a * a
    rmax2 = r_max * r_max
    remit2 = r_reemit * r_reemit
    inv_dr = n_r / (r_max - a)
    sig = np.sqrt(2.0 * Dr * dt)
    hist = np.zeros((n_blocks, n_sinb, n_cosa))
    histr = np.zeros((n_blocks, n_r))
    block = (M + n_blocks - 1) // n_blocks
    for b in prange(n_blocks):
        p0 = b * block
        p1 = p0 + block
        if p1 > M:
            p1 = M
        for p in range(p0, p1):
            dx = d0[p, 0]
            dy = d0[p, 1]
            # per-pair xorshift64* state (nonzero by construction)
            st = (np.uint64(p + 1) * np.uint64(2862933555777941757)) ^ np.uint64(
                rng_seed
            )
            for _ in range(3):  # decorrelate from the linear seeding
                st, _u = _xorshift_uniform(st)
            armed = False
            ti = thi[p]
            tj = thj[p]
            ux = v0 * (np.cos(tj) - np.cos(ti))
            uy = v0 * (np.sin(tj) - np.sin(ti))
            # Wrap the two headings into [0, 2pi), exactly as _collision_angles
            # does.  At Dr = 0 the headings are frozen, so the collision-frame
            # quantities are fixed for the whole run and precomputed here; at
            # Dr > 0 they are recomputed at every contact sample.
            tiw = ti - TWOPI * np.floor(ti / TWOPI)
            tjw = tj - TWOPI * np.floor(tj / TWOPI)
            halfsum = 0.5 * (tiw + tjw)
            sinb = np.sin(0.5 * (tjw - tiw))
            ib_pos = _unit_bin(sinb, n_sinb)  # (i, j) ordering, axis-0 bin
            ib_neg = _unit_bin(-sinb, n_sinb)  # (j, i) ordering, axis-0 bin
            for s in range(n_steps):
                if sig > 0.0:
                    # --- rotational diffusion (Box-Muller from xorshift) ---
                    st, u1 = _xorshift_uniform(st)
                    st, u2 = _xorshift_uniform(st)
                    amp = sig * np.sqrt(-2.0 * np.log(1.0 - u1))
                    ang = TWOPI * u2
                    ti += amp * np.cos(ang)
                    tj += amp * np.sin(ang)
                    ux = v0 * (np.cos(tj) - np.cos(ti))
                    uy = v0 * (np.sin(tj) - np.sin(ti))
                # --- propulsion (relative) ---
                dx += dt * ux
                dy += dt * uy
                # --- nearest image into [-L/2, L/2) ---
                dx -= L * np.floor(dx / L + 0.5)
                dy -= L * np.floor(dy / L + 0.5)
                # --- contact detection (post-propulsion overlap) + projection ---
                r2 = dx * dx + dy * dy
                is_sample = s >= burn_in and (s % sample_every) == 0
                if 1e-18 < r2 < a2:
                    r = np.sqrt(r2)
                    sc = a / r
                    dx *= sc
                    dy *= sc
                    armed = True
                    if is_sample:
                        if sig > 0.0:
                            # collision frame from the *current* headings
                            tiw = ti - TWOPI * np.floor(ti / TWOPI)
                            tjw = tj - TWOPI * np.floor(tj / TWOPI)
                            halfsum = 0.5 * (tiw + tjw)
                            sinb = np.sin(0.5 * (tjw - tiw))
                            ib_pos = _unit_bin(sinb, n_sinb)
                            ib_neg = _unit_bin(-sinb, n_sinb)
                        phi = np.arctan2(dy, dx)
                        cosa = np.cos(phi - halfsum + HALFPI)
                        # ordering (i, j) and (j, i): (sinb, cosa) -> (-sinb, -cosa)
                        hist[b, ib_pos, _unit_bin(cosa, n_cosa)] += 1.0
                        hist[b, ib_neg, _unit_bin(-cosa, n_cosa)] += 1.0
                else:
                    if armed and rerandomize and r2 >= remit2:
                        # collision event over: re-emit from the disk surface
                        # at a uniform impact parameter, moving away along
                        # u_rel (molecular-chaos exit -> Kac chord ensemble)
                        if sig > 0.0:
                            # fresh partner: redraw headings from the
                            # flux-weighted measure |sin(delta/2)| d(delta)
                            sp2 = 0.0
                            while sp2 <= 1e-24:
                                st, u1 = _xorshift_uniform(st)
                                st, u2 = _xorshift_uniform(st)
                                ti = TWOPI * u1
                                tj = ti + 2.0 * np.arccos(1.0 - 2.0 * u2)
                                ux = v0 * (np.cos(tj) - np.cos(ti))
                                uy = v0 * (np.sin(tj) - np.sin(ti))
                                sp2 = ux * ux + uy * uy
                        else:
                            sp2 = ux * ux + uy * uy
                        if sp2 > 1e-24:
                            st, u1 = _xorshift_uniform(st)
                            bimp = (2.0 * u1 - 1.0) * a
                            sp = np.sqrt(sp2)
                            ex = ux / sp
                            ey = uy / sp
                            cchord = np.sqrt(a2 - bimp * bimp)
                            # d = cchord * u_hat + bimp * u_hat_perp
                            dx = cchord * ex - bimp * ey
                            dy = cchord * ey + bimp * ex
                            r2 = a2
                        armed = False
                    if is_sample and r2 < rmax2:
                        # bulk radial g(r): both orderings share the same r
                        kr = int((np.sqrt(r2) - a) * inv_dr)
                        if 0 <= kr < n_r:
                            histr[b, kr] += 2.0
    out = hist.sum(axis=0)
    outr = histr.sum(axis=0)
    return out, outr, out.sum()


def _angular_reference_surf(sin_beta_edges, cos_alpha_edges, seed, n_mc=20_000_000):
    """Ideal-gas reference weight w_surf(sin beta, cos alpha), normalised to sum
    to 1 -- identical construction to HardSphereABP._angular_reference (surf
    part): uniform ordered pairs (theta_i, theta_j, phi)."""
    rng = np.random.default_rng(seed + 1)
    ti = rng.uniform(0.0, TWOPI, n_mc)
    tj = rng.uniform(0.0, TWOPI, n_mc)
    phi = rng.uniform(0.0, TWOPI, n_mc)
    al = np.mod(phi - 0.5 * (ti + tj) + HALFPI, TWOPI)
    be = 0.5 * (tj - ti)
    w_surf, _, _ = np.histogram2d(
        np.sin(be), np.cos(al), bins=[sin_beta_edges, cos_alpha_edges]
    )
    return w_surf / w_surf.sum()


def run_two_particle(
    M=300_000,
    L=4.0,
    v0=10.0,
    a=1.0,
    Dr=0.0,
    dt=1e-3,
    n_steps=100_000,
    burn_in=20_000,
    sample_every=20,
    n_sinb=72,
    n_cosa=72,
    n_blocks=256,
    r_max=None,
    n_r=60,
    seed=0,
    rerandomize=True,
    r_reemit=None,
    save_path="two_particle_gsurf.npz",
    verbose=True,
):
    """Run the ensemble of ``M`` isolated pairs and return
    ``(g_surf, sin_beta, cos_alpha)``.

    ``g_surf`` has shape (n_sinb, n_cosa) in the (sin beta, cos alpha) plane,
    with the same normalisation as ``HardSphereABP.g_surf`` -- so a flat value of
    1 on the support ``sin(beta) cos(alpha) > 0`` is the theoretical target.  The
    radial bulk correlation ``g_of_r`` (r in (a, r_max)) is also measured and
    stored, normalised by the same box-mean density, so ``g_surf`` and
    ``g_of_r`` share one reference.

    ``Dr`` is the rotational diffusion constant of each heading
    (Pe = v0 / (a Dr); ``Dr = 0`` is the frozen-heading Pe = infinity limit).

    ``rerandomize=True`` (default) re-emits a pair from the disk surface at a
    uniform random impact parameter once its separation exceeds ``r_reemit``
    after a contact (redrawing the headings from the flux-weighted measure when
    ``Dr > 0``), enforcing molecular chaos so the Pe = infinity plateau is 1
    independently of ``L``; ``rerandomize=False`` keeps the closed single-pair
    dynamics, whose grazing-exit recollisions inflate the plateau ~ ln L at
    ``Dr = 0`` (the shape stays flat -- see module docstring).  ``r_reemit``
    defaults to ``a`` at ``Dr = 0`` (re-emit right at detachment) and ``2 a``
    at ``Dr > 0`` (so detach/re-attach sequences of a wiggling contact are not
    cut short)."""
    if L <= 2.0 * a:
        raise ValueError("need L > 2a so the excluded disk fits inside [-L/2, L/2)")
    r_max = min(0.5 * L, 4.0 * a) if r_max is None else r_max
    if r_reemit is None:
        r_reemit = a if Dr == 0.0 else min(2.0 * a, 0.45 * L)
    if not (a <= r_reemit < 0.5 * L):
        raise ValueError("need a <= r_reemit < L/2")

    rng = np.random.default_rng(seed)
    thi = rng.uniform(0.0, TWOPI, M)
    thj = rng.uniform(0.0, TWOPI, M)
    # random initial relative position in the box
    d0 = rng.uniform(-0.5 * L, 0.5 * L, size=(M, 2))

    sin_beta_edges = np.linspace(-1.0, 1.0, n_sinb + 1)
    cos_alpha_edges = np.linspace(-1.0, 1.0, n_cosa + 1)
    sin_beta = 0.5 * (sin_beta_edges[:-1] + sin_beta_edges[1:])
    cos_alpha = 0.5 * (cos_alpha_edges[:-1] + cos_alpha_edges[1:])
    r_edges = np.linspace(a, r_max, n_r + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])

    steps = np.arange(n_steps)
    n_samples = int(np.count_nonzero((steps >= burn_in) & (steps % sample_every == 0)))

    if verbose:
        pe_str = "inf" if Dr == 0.0 else f"{v0 / (a * Dr):g}"
        mode = (
            f"rerandomize at r={r_reemit} (molecular chaos)"
            if rerandomize
            else "closed dynamics"
        )
        print(
            f"two-particle ensemble: M={M} pairs, L={L}a, v0={v0}, a={a}, dt={dt}, "
            f"Dr={Dr} (Pe={pe_str}), {mode}\n  n_steps={n_steps}, burn_in={burn_in}, "
            f"sample_every={sample_every} -> n_samples={n_samples}, r_max={r_max}"
        )

    hist_surf, hist_r, n_contacts = _run_pairs(
        d0,
        thi,
        thj,
        a,
        L,
        v0,
        Dr,
        dt,
        n_steps,
        burn_in,
        sample_every,
        n_sinb,
        n_cosa,
        n_blocks,
        n_r,
        r_max,
        rerandomize,
        r_reemit,
        np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(0xDA442D24),
    )

    w_surf = _angular_reference_surf(sin_beta_edges, cos_alpha_edges, seed)
    # exact ordered-pair contact reference: sum over M boxes of N(N-1)/L^2 * 2 pi a
    pref = n_samples * M * (2.0 / (L * L))
    denom = pref * (TWOPI * a) * w_surf
    with np.errstate(invalid="ignore", divide="ignore"):
        g_surf = np.where(denom > 0, hist_surf / denom, np.nan)

    # radial bulk g(r), same box-mean reference density: shell area pi(r_{k+1}^2-r_k^2)
    ring = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    g_of_r = hist_r / (pref * ring)

    if verbose:
        print(
            f"  contacts binned = {n_contacts:.3g} "
            f"({n_contacts / (M * n_samples):.4f} per pair-sample); "
            f"g(r) near contact = {g_of_r[0]:.3f}, far (r~r_max) = {g_of_r[-1]:.3f}"
        )

    if save_path is not None:
        np.savez_compressed(
            save_path,
            g_surf=g_surf,
            g_of_r=g_of_r,
            r=r,
            r_edges=r_edges,
            sin_beta=sin_beta,
            cos_alpha=cos_alpha,
            sin_beta_edges=sin_beta_edges,
            cos_alpha_edges=cos_alpha_edges,
            hist_surf=hist_surf,
            M=M,
            L=L,
            v0=v0,
            a=a,
            dt=dt,
            Dr=Dr,
            r_max=r_max,
            n_r=n_r,
            n_steps=n_steps,
            burn_in=burn_in,
            sample_every=sample_every,
            n_samples=n_samples,
            n_sinb=n_sinb,
            n_cosa=n_cosa,
            seed=seed,
            n_contacts=n_contacts,
            rerandomize=rerandomize,
            r_reemit=r_reemit,
        )
        if verbose:
            print(f"  saved to {save_path}")

    return g_surf, sin_beta, cos_alpha


def _folded_profile(g_surf, sin_beta, cos_alpha, ca_lo=0.2, ca_hi=0.95):
    """g_surf averaged over the supported cos(alpha) band at each sin(beta) > 0,
    folding in the symmetric (sin beta < 0, cos alpha < 0) quadrant.  Matches the
    profile extracted from the many-body runs in compare_layers.py."""
    jmask = (cos_alpha > ca_lo) & (cos_alpha < ca_hi)
    jmask_neg = (cos_alpha < -ca_lo) & (cos_alpha > -ca_hi)
    pos = sin_beta > 0
    neg = sin_beta < 0
    prof_pos = np.nanmean(g_surf[np.ix_(pos, jmask)], axis=1)
    prof_neg = np.nanmean(g_surf[np.ix_(neg, jmask_neg)], axis=1)[::-1]
    return sin_beta[pos], 0.5 * (prof_pos + prof_neg)


if __name__ == "__main__":
    import os

    for Dr in [0.0, 5, 1.0, 0.5, 0.1, 0.05, 0.01]:
        if Dr == 0.0:
            save_path = f"two_particle_gsurf_Peinf.npz"
        else:
            save_path = f"two_particle_gsurf_Pe{10.0/Dr:g}.npz"
        print(f"Saving to {save_path}")
        g_surf, sin_beta, cos_alpha = run_two_particle(
            M=300_000,
            L=4.0,
            v0=10.0,
            a=1.0,
            Dr=Dr,
            dt=1e-3,
            n_steps=100_000,
            burn_in=20_000,
            sample_every=20,
            n_sinb=72,
            n_cosa=72,
            n_blocks=256,
            r_max=None,
            n_r=60,
            seed=0,
            rerandomize=True,
            r_reemit=None,
            save_path=save_path,
        )

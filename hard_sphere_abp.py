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

Both are invariant under a global rotation of the whole system.  Computed
directly from the raw (phi, theta_i, theta_j) of each pair they cover
alpha in [0, 2pi), beta in [-pi/2, pi/2) uniformly, so the ideal-gas reference
is flat and there are no inaccessible cells.

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
    """Bin beta into [-pi/2, pi/2) (period pi)."""
    f = beta + HALFPI
    f -= np.pi * np.floor(f / np.pi)  # -> [0, pi)
    b = int(f * (n / np.pi))
    if b >= n:
        b -= n
    return b


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
    hist_bulk,
    hist_surf,
    head,
    linked,
    ncx,
    ncy,
    cell_size,
):
    """Bin every neighbour pair (both orderings) into the bulk or surface
    histogram using the collision-frame angles (alpha, beta).  Contacts are
    pairs with r <= a*(1+contact_band)."""
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
                        if 1e-18 < r2 < rmax2:
                            r = np.sqrt(r2)
                            thj = theta[j]
                            phi = np.arctan2(dy, dx)
                            half_sum = 0.5 * (thi + thj)
                            # ordering (i, j)
                            al = phi - half_sum + HALFPI
                            be = 0.5 * (thj - thi)
                            ia = _alpha_bin(al, n_alpha)
                            ib = _beta_bin(be, n_beta)
                            # ordering (j, i): separation reversed -> alpha+pi, beta->-beta
                            ja = _alpha_bin(al + np.pi, n_alpha)
                            jb = _beta_bin(-be, n_beta)
                            if r <= r_contact:
                                hist_surf[ia, ib] += 1.0
                                hist_surf[ja, jb] += 1.0
                            else:
                                kr = int((r - a) * inv_dr)
                                if 0 <= kr < n_r:
                                    hist_bulk[kr, ia, ib] += 1.0
                                    hist_bulk[kr, ja, jb] += 1.0
                    j = linked[j]


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
    hist_bulk,
    hist_surf,
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
        _build_cells(pos, L, cs_d, head_d, linked_d, ncx_d, ncy_d)
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
        if step >= burn_in and (step % sample_every == 0):
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
                hist_bulk,
                hist_surf,
                head_a,
                linked_a,
                ncx_a,
                ncy_a,
                cs_a,
            )
            n_samples += 1

    return n_samples


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
        Bins for alpha in [0, 2pi) and beta in [-pi/2, pi/2).
    dt : float, optional
        Time step (default 0.01 a / v0, so propulsion moves << a per step).
    n_sweeps : int
        Gauss-Seidel projection sweeps per step.
    contact_band : float
        A pair counts as a contact when r <= a (1 + contact_band).
    seed : int
        RNG seed.
    verbose : bool
        Print a banner/summary and show a progress bar.

    Example
    -------
    >>> sim = HardSphereABP(N=2000, phi=0.3, v0=1.0, Dr=0.1)
    >>> sim.run(n_steps=200_000, burn_in=20_000, sample_every=50, save_path="run")
    >>> g_bulk = sim.g_bulk          # (n_r, n_alpha, n_beta)
    >>> g_surf = sim.g_surf          # (n_alpha, n_beta), units of length
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
        self.beta_edges = np.linspace(-HALFPI, HALFPI, self.n_beta + 1)
        self.beta = 0.5 * (self.beta_edges[:-1] + self.beta_edges[1:])

        self.g_bulk = None
        self.g_surf = None
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
            print(
                f"HardSphereABP: N={self.N}, phi={self.phi}, L={self.L:.3f}, "
                f"rho={self.rho:.4f}, a={self.a}, v0={self.v0}, Dr={self.Dr}, "
                f"Dt={self.Dt}, dt={self.dt:.4g}, Pe={self.v0 / (self.a * self.Dr):.3g}"
            )

        np.random.seed(self.seed)
        pos, theta = self._initial_state()

        hist_bulk = np.zeros((self.n_r, self.n_alpha, self.n_beta))
        hist_surf = np.zeros((self.n_alpha, self.n_beta))
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
                hist_bulk,
                hist_surf,
            )
            step += n_chunk
            bar.update(n_chunk)
            bar.set_postfix(samples=n_samples)
        bar.close()

        self.n_samples = n_samples
        self.pos, self.theta = pos, theta
        self._normalize(hist_bulk, hist_surf)

        if self.verbose:
            print(
                f"done: {n_samples} samples, "
                f"contacts/particle = {self.contact_fraction():.3f}"
            )
        if save_path is not None:
            self.save(save_path)
        return self.g_bulk, self.g_surf

    # ------------------------------------------------------------------ #
    def _angular_reference(self, n_mc: int = 10_000_000) -> np.ndarray:
        """Ideal-gas angular weight w(alpha, beta): the (normalised) fraction of
        uniformly-oriented pairs falling in each (alpha, beta) cell.  Computing
        alpha, beta from raw uniform (theta_i, theta_j, phi) it is essentially
        flat, but doing it by Monte-Carlo automatically captures the exact
        measure (and any edge effects)."""
        rng = np.random.default_rng(self.seed + 1)
        ti = rng.uniform(0.0, TWOPI, n_mc)
        tj = rng.uniform(0.0, TWOPI, n_mc)
        phi = rng.uniform(0.0, TWOPI, n_mc)
        al = np.mod(phi - 0.5 * (ti + tj) + HALFPI, TWOPI)
        be = np.mod(0.5 * (tj - ti) + HALFPI, np.pi) - HALFPI
        ref, _, _ = np.histogram2d(al, be, bins=[self.alpha_edges, self.beta_edges])
        return ref / ref.sum()

    def _normalize(self, hist_bulk, hist_surf):
        """Divide the histograms by the ideal-gas (g=1) expectation.

        The angular dependence factorises through the flat reference weight
        w(alpha, beta); the radial/contact measures are
            bulk : pi (r_{k+1}^2 - r_k^2)
            surf : 2 pi a            (contact circle, carries 1/length)
        """
        w = self._angular_reference()  # (n_alpha, n_beta), sums to 1
        pref = self.n_samples * self.N * self.rho
        self._n_contacts = float(hist_surf.sum())

        # Bulk: g_bulk(r, alpha, beta).
        ring = np.pi * (self.r_edges[1:] ** 2 - self.r_edges[:-1] ** 2)
        denom_bulk = pref * ring[:, None, None] * w[None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            self.g_bulk = np.where(denom_bulk > 0, hist_bulk / denom_bulk, np.nan)

        # Angle-averaged bulk g(r) (radial marginal, robust validation -> 1).
        self.g_of_r = hist_bulk.sum(axis=(1, 2)) / (pref * ring)

        # Surface (contact) density g_surf(alpha, beta), units of length.
        denom_surf = pref * (TWOPI * self.a) * w
        with np.errstate(invalid="ignore", divide="ignore"):
            self.g_surf = np.where(denom_surf > 0, hist_surf / denom_surf, np.nan)

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
            # correlation functions
            g_bulk=self.g_bulk,
            g_surf=self.g_surf,
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

    sim = HardSphereABP(
        N=2000,
        phi=0.05,
        a=1.0,
        v0=10.0,
        Dr=0.1,  # Peclet  v0/(a Dr) = 100
        n_r=60,
        n_alpha=72,
        n_beta=36,
    )
    sim.run(n_steps=20_000, burn_in=0, sample_every=50, save_path="hard_sphere_abp")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(sim.r / sim.a, sim.g_of_r, "-o", ms=3)
    axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel(r"$r/a$")
    axes[0].set_ylabel(r"$\langle g_{\rm bulk}\rangle_{\alpha,\beta}(r)$")
    axes[0].set_title("Angle-averaged bulk pair correlation")

    im = axes[1].imshow(
        sim.g_surf.T,
        origin="lower",
        extent=(0, 360, -90, 90),
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set_xlabel(r"$\alpha = \phi-(\theta_1+\theta_2)/2+\pi/2$ (deg)")
    axes[1].set_ylabel(r"$\beta = (\theta_2-\theta_1)/2$ (deg)")
    axes[1].set_title(r"Contact density $g_{\rm surf}(\alpha,\beta)$")
    fig.colorbar(im, ax=axes[1])

    fig.tight_layout()
    fig.savefig("hard_sphere_abp_g.png", dpi=200)
    print("Saved hard_sphere_abp_g.png")

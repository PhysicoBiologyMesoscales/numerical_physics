"""
Three overdamped active hard spheres in free space (no rotational diffusion).

Model
-----
Three disks of diameter ``a`` move overdamped in the plane.  Each is
self-propelled along a *fixed* orientation ``theta_i`` at speed ``v0``:

    dr_i/dt = v0 * (cos theta_i, sin theta_i) + (hard-core constraint force)

There is no rotational diffusion (the ``theta_i`` are constant) and no
translational noise, so the trajectory is fully deterministic given the initial
positions and orientations.  The hard-core constraint ``|r_i - r_j| >= a`` is
enforced exactly by projecting overlaps out at every step: an overlapping pair
is pushed apart along its line of centres by half the overlap each (equal
mobility => symmetric split).  A few Gauss-Seidel sweeps resolve the case where
all three are simultaneously in contact (a force chain / jam).

Because there is no noise, two particles propelled into each other stay glued at
exactly ``r = a`` and *slide* along one another: the projection removes only the
normal (compressive) part of the relative motion, so the tangential part
survives.  Three particles whose propulsions all point inward jam into a stuck
triangle; otherwise they eventually slide apart.  This is the same contact
physics as ``hard_sphere_abp.py``, specialised to N=3 and free (non-periodic)
space so you can watch a single scattering event.

Usage
-----
>>> import numpy as np
>>> positions   = np.array([[-1.2, 0.0], [1.2, 0.0], [0.0, 2.0]])
>>> orientations = np.array([0.0, np.pi, -np.pi / 2])          # radians
>>> sim = ThreeBodyActiveHS(positions, orientations, v0=1.0, a=1.0)
>>> sim.animate(n_steps=2000, frame_every=8, save_path="three_body.mp4")

To start from an exact contact chain instead of explicit positions, use
:meth:`ThreeBodyActiveHS.from_chain`, giving the two bond angles from the
central disk to its neighbours (both placed at r = a):

>>> sim = ThreeBodyActiveHS.from_chain(
...     phi_12=0.0, phi_13=2 * np.pi / 3, orientations=orientations, center=0)
"""

from __future__ import annotations

import numpy as np

# The three unordered pairs among particles {0, 1, 2}.
_PAIRS = ((0, 1), (0, 2), (1, 2))


def _resolve(pos: np.ndarray, a: float, n_sweeps: int) -> None:
    """Push overlapping pairs apart along their line of centres by half the
    overlap each, repeated ``n_sweeps`` times (Gauss-Seidel).  In place."""
    for _ in range(n_sweeps):
        for i, j in _PAIRS:
            d = pos[j] - pos[i]
            r2 = d @ d
            if r2 < a * a and r2 > 1e-18:
                r = np.sqrt(r2)
                corr = 0.5 * (a - r) / r
                pos[i] -= corr * d
                pos[j] += corr * d


def _step(
    pos: np.ndarray, drift: np.ndarray, a: float, dt: float, n_sweeps: int
) -> None:
    """Advance one time step in place: free propulsion, then contact projection.

    ``drift`` is the pre-computed per-particle propulsion velocity
    ``v0 * (cos theta, sin theta)`` (constant, since orientations are fixed)."""
    pos += dt * drift
    _resolve(pos, a, n_sweeps)


class ThreeBodyActiveHS:
    """Overdamped active hard-sphere dynamics for exactly three disks.

    Parameters
    ----------
    positions : (3, 2) array_like
        Initial centres.  Overlapping inputs are relaxed before the run.
    orientations : (3,) array_like
        Fixed propulsion headings ``theta_i`` in radians.
    v0 : float
        Self-propulsion speed.
    a : float
        Disk diameter (contact distance).
    dt : float, optional
        Time step.  Default ``0.01 a / v0`` so propulsion moves << a per step.
    n_sweeps : int
        Gauss-Seidel projection sweeps per step (>= a few for a stable jam).
    """

    def __init__(
        self,
        positions,
        orientations,
        v0: float = 1.0,
        a: float = 1.0,
        dt: float | None = None,
        n_sweeps: int = 20,
    ):
        pos = np.asarray(positions, dtype=float)
        theta = np.asarray(orientations, dtype=float).ravel()
        if pos.shape != (3, 2):
            raise ValueError(f"positions must have shape (3, 2), got {pos.shape}")
        if theta.shape != (3,):
            raise ValueError(f"orientations must have shape (3,), got {theta.shape}")

        self.pos0 = pos.copy()
        self.theta = theta.copy()
        self.v0 = float(v0)
        self.a = float(a)
        self.dt = (0.01 * a / v0) if dt is None else float(dt)
        self.n_sweeps = int(n_sweeps)

        # Fixed propulsion velocity of each particle (orientations never change).
        self.drift = self.v0 * np.stack(
            [np.cos(self.theta), np.sin(self.theta)], axis=-1
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_chain(
        cls,
        phi_12: float,
        phi_13: float,
        orientations,
        center: int = 0,
        a: float = 1.0,
        center_position=(0.0, 0.0),
        **kwargs,
    ):
        """Build a *contact chain*: one central disk touching the other two at
        exactly ``r = a``.

        The ``center`` particle is placed at ``center_position`` and its two
        neighbours are placed one contact length ``a`` away along the bond
        directions ``phi_12`` and ``phi_13`` (radians).  With the default
        ``center=0`` this is exactly the "particle 1 in the middle" chain:

            particle 0 (centre) -> center_position
            particle 1          -> center_position + a (cos phi_12, sin phi_12)
            particle 2          -> center_position + a (cos phi_13, sin phi_13)

        For another ``center`` the two neighbours are the remaining indices in
        ascending order (e.g. ``center=1`` puts particle 1 in the middle with
        the bonds pointing at particles 0 then 2).

        This is a *chain*, not a triangle: only the two centre bonds are set to
        contact.  The outer pair is itself in contact only if the opening angle
        ``|phi_12 - phi_13|`` equals 60 degrees; wider than that and they start
        apart.  The initial overlap relaxation in :meth:`simulate` leaves an
        exact ``r = a`` contact untouched (the projection only acts on ``r < a``).

        Parameters
        ----------
        phi_12, phi_13 : float
            Bond directions (radians) from the centre to its first and second
            neighbour.
        orientations : (3,) array_like
            Fixed propulsion headings for particles 0, 1, 2 -- same convention
            as the main constructor, independent of the chain geometry.
        center : int
            Which particle (0, 1 or 2, zero-indexed) sits in the middle.
        a : float
            Contact distance / disk diameter.
        center_position : (2,) array_like
            Where to place the central disk (default the origin).
        **kwargs
            Forwarded to the constructor (``v0``, ``dt``, ``n_sweeps``).
        """
        if center not in (0, 1, 2):
            raise ValueError(f"center must be 0, 1 or 2, got {center}")
        first, second = (k for k in range(3) if k != center)  # ascending indices

        c = np.asarray(center_position, dtype=float).reshape(2)
        positions = np.empty((3, 2), dtype=float)
        positions[center] = c
        positions[first] = c + a * np.array([np.cos(phi_12), np.sin(phi_12)])
        positions[second] = c + a * np.array([np.cos(phi_13), np.sin(phi_13)])

        return cls(positions, orientations, a=a, **kwargs)

    # ------------------------------------------------------------------ #
    def simulate(self, n_steps: int, frame_every: int = 8):
        """Integrate ``n_steps`` steps, capturing a frame every ``frame_every``.

        Returns
        -------
        frames_pos : (n_frames, 3, 2) ndarray
            Particle centres at each captured frame (frame 0 is the initial,
            overlap-relaxed, configuration).
        times : (n_frames,) ndarray
            Simulation time of each frame.
        """
        n_steps = int(n_steps)
        frame_every = max(1, int(frame_every))

        pos = self.pos0.copy()
        _resolve(pos, self.a, self.n_sweeps)  # clean any initial overlap

        frames = [pos.copy()]
        times = [0.0]
        for s in range(n_steps):
            _step(pos, self.drift, self.a, self.dt, self.n_sweeps)
            if (s + 1) % frame_every == 0:
                frames.append(pos.copy())
                times.append((s + 1) * self.dt)

        return np.asarray(frames), np.asarray(times)

    # ------------------------------------------------------------------ #
    def animate(
        self,
        n_steps: int = 2000,
        frame_every: int = 8,
        save_path: str | None = None,
        fps: int = 25,
        dpi: int = 120,
        interval: int = 40,
        follow_com: bool = False,
        window: float | None = None,
        show_arrows: bool = True,
        show_trails: bool = True,
    ):
        """Render a movie of the three disks moving under the dynamics.

        Parameters
        ----------
        n_steps, frame_every : int
            Total integration steps and steps between captured frames.
        save_path : str, optional
            Where to write the movie.  ``.gif`` uses the Pillow writer, anything
            else (e.g. ``.mp4``) uses ffmpeg.  If omitted the animation is shown
            interactively with ``plt.show()``.
        fps, dpi : int
            Frame rate and resolution when saving.
        interval : int
            Delay between frames in ms for on-screen playback.
        follow_com : bool
            If True the camera tracks the centre of mass with a fixed-size
            window (good when the group drifts far).  If False the view is a
            fixed box bounding the whole trajectory, so everything stays visible.
        window : float, optional
            Half-width of the view.  Defaults to a few diameters (follow_com) or
            is derived from the trajectory bounding box (fixed view).
        show_arrows : bool
            Draw a propulsion-direction arrow on each disk.
        show_trails : bool
            Draw the path each centre has traced so far.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Keep a reference alive while it plays.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import EllipseCollection

        frames, _ = self.simulate(n_steps, frame_every)
        n_frames = frames.shape[0]
        colors = plt.cm.hsv(np.mod(self.theta, 2 * np.pi) / (2 * np.pi))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("3 active hard spheres (overdamped, no rotational diffusion)")

        if follow_com:
            half = 4.0 * self.a if window is None else float(window)
        else:
            lo = frames.reshape(-1, 2).min(axis=0) - 1.0 * self.a
            hi = frames.reshape(-1, 2).max(axis=0) + 1.0 * self.a
            centre = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo).max() if window is None else float(window)
            ax.set_xlim(centre[0] - half, centre[0] + half)
            ax.set_ylim(centre[1] - half, centre[1] + half)

        # Disks at their true diameter, coloured by (fixed) orientation.
        coll = EllipseCollection(
            widths=self.a,
            heights=self.a,
            angles=0.0,
            units="xy",
            offsets=frames[0],
            offset_transform=ax.transData,
            facecolors=colors,
            edgecolor="k",
            linewidth=0.6,
        )
        ax.add_collection(coll)

        trails = []
        if show_trails:
            for k in range(3):
                (line,) = ax.plot([], [], "-", color=colors[k], lw=1.0, alpha=0.5)
                trails.append(line)

        arrow_len = 0.6 * self.a
        quiver = None
        if show_arrows:
            quiver = ax.quiver(
                frames[0, :, 0],
                frames[0, :, 1],
                arrow_len * np.cos(self.theta),
                arrow_len * np.sin(self.theta),
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="k",
                width=0.006,
                zorder=5,
            )

        def update(f):
            coll.set_offsets(frames[f])
            artists = [coll]
            if follow_com:
                com = frames[f].mean(axis=0)
                ax.set_xlim(com[0] - half, com[0] + half)
                ax.set_ylim(com[1] - half, com[1] + half)
            if show_trails:
                for k, line in enumerate(trails):
                    line.set_data(frames[: f + 1, k, 0], frames[: f + 1, k, 1])
                artists.extend(trails)
            if show_arrows:
                quiver.set_offsets(frames[f])
                artists.append(quiver)
            return artists

        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            blit=not follow_com,  # blitting can't handle changing axis limits
        )

        if save_path is not None:
            writer = "pillow" if save_path.lower().endswith(".gif") else "ffmpeg"
            anim.save(save_path, writer=writer, fps=fps, dpi=dpi)
            print(f"saved movie to {save_path}")
        else:
            plt.show()
        return anim


# --------------------------------------------------------------------------- #
# Example                                                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Two particles driven head-on into each other, a third dropping in from
    # above -- they make contact, slide, and scatter.
    orientations = np.array([0.0, np.pi, -np.pi / 6])  # ->, <-, down

    sim = ThreeBodyActiveHS.from_chain(
        phi_12=1.9144,
        phi_13=1.4682,
        orientations=orientations,
        center=1,
        v0=0.1,
        a=1.0,
    )
    sim.animate(
        n_steps=2500,
        frame_every=8,
        save_path="three_body_active_hs.mp4",
        follow_com=True,
    )

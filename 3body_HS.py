import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ThreeBodyG:
    """Compute the pair correlation function g along characteristics
    for a 3-body hard-sphere problem, given velocity vectors u and v.

    Usage
    -----
    >>> solver = ThreeBodyG(u, v)
    >>> t, g, l0 = solver.compute_g(l0, n_times=100)
    >>> phi, lx, ly, g_grid = solver.compute_g_on_grid()
    """

    def __init__(self, u: np.ndarray, v: np.ndarray, plane: int = 1):
        self.u = np.asarray(u, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.plane = plane

        # Derived geometry
        self.u_norm = np.linalg.norm(self.u)
        self.v_norm = np.linalg.norm(self.v)
        self.u_hat = self.u / self.u_norm
        self.v_hat = self.v / self.v_norm
        self.vu = self.v - self.u
        self.vu_hat = self.vu / np.linalg.norm(self.vu)

        # Characteristic drift: l(t) = l0 + t * drift
        self.drift = 0.5 * self.u - self.v

        # Cone direction (clockwise / counter-clockwise)
        angle_v = np.arctan2(self.v[1], self.v[0])
        angle_vu = np.arctan2(self.vu[1], self.vu[0])
        cone_angle = (angle_vu - angle_v + np.pi) % (2 * np.pi) - np.pi
        self.angle_dir = np.sign(cone_angle)

    # -- Characteristic trajectory ------------------------------------------

    def phi(self, t: np.ndarray) -> np.ndarray:
        """Angle of the 1-2 rotation at backward-time *t*."""
        _t = np.atleast_1d(t)
        result = (
            np.pi
            + np.arctan2(self.u[1], self.u[0])
            + self.plane * np.arccos(np.tanh(self.u_norm * _t))
        )
        return np.squeeze((result + np.pi) % (2 * np.pi) - np.pi)

    def absolute_position(self, t: np.ndarray, l0: np.ndarray) -> np.ndarray:
        """Position of particle 3 relative to the 1-2 centre along the
        characteristic.  Returns shape (n_times, n_particles, 2)."""
        _t = np.atleast_1d(t)[:, None, None]
        _l0 = np.atleast_2d(l0)[None, :, :]
        return np.squeeze(_l0 + _t * self.drift)

    def relative_positions(self, l: np.ndarray, phi: np.ndarray):
        """Positions of particle 3 relative to particles 1 and 2.
        Returns (s, sr) each with same shape as *l*."""
        # _phi = phi[:, None] if l.ndim == 3 else phi
        half_rot = 0.5 * np.stack([np.cos(phi), np.sin(phi)], axis=-1)
        return l - half_rot, l + half_rot

    def t_from_phi(self, phi_target: np.ndarray) -> np.ndarray:
        """Invert phi(t) to recover t from target phi values."""
        alpha = np.arctan2(self.u[1], self.u[0])
        phi_target = np.asarray(phi_target, dtype=float)
        # phi_unwrapped = pi + alpha + plane * theta,  theta = arccos(tanh(|u|*t)) in [0, pi/2]
        # Find the 2*k*pi offset that puts theta in [0, pi/2]
        for k in [-1, 0, 1]:
            theta = (phi_target + 2 * k * np.pi - np.pi - alpha) / self.plane
            if np.all(theta >= -1e-10) and np.all(theta <= np.pi / 2 + 1e-10):
                break
        theta = np.clip(theta, 0, np.pi / 2)
        cos_theta = np.clip(np.cos(theta), 0, 1 - 1e-15)
        return np.arctanh(cos_theta) / self.u_norm

    # -- Masking (trivially g = 1) ------------------------------------------

    def trivial_mask(self, l0: np.ndarray) -> np.ndarray:
        """Boolean mask: True where g = 1 can be determined without
        tracing the characteristic (no collision, no shadow crossing)."""
        proj_v = np.dot(l0, self.v)
        proj_vu = np.dot(l0, self.vu)
        d = self.angle_dir
        return np.logical_or.reduce(
            (
                (np.linalg.norm(l0, axis=-1) > 1.5) & (proj_v < 0) & (proj_vu < 0),
                (proj_v > 0) & (d * np.cross(l0, self.v_hat) > 1.5),
                (proj_vu > 0) & (-d * np.cross(l0, self.vu_hat) > 1.5),
            )
        )

    # -- Shadow detection ---------------------------------------------------

    def shadow_masks(self, s, sr):
        """Boolean shadow masks (n_times, n_particles) for the tubes along
        v (particle 1) and v-u (particle 2)."""
        s_cross = self.angle_dir * np.cross(s, self.v_hat)
        sr_cross = -self.angle_dir * np.cross(sr, self.vu_hat)
        in_v = (s_cross > -1) & (s_cross < 1) & (np.dot(s, self.v) > 0)
        in_vu = (sr_cross > -1) & (sr_cross < 1) & (np.dot(sr, self.vu) > 0)
        return in_v, in_vu, s_cross, sr_cross

    # -- Collision times ----------------------------------------------------

    @staticmethod
    def _first_crossing_idx(dist: np.ndarray) -> np.ndarray:
        """For each particle (axis 0), return the time index where *dist*
        first becomes negative, or NaN if it never does."""
        # TODO add a more precise method to compute crossing time
        collides = np.any(dist < 0, axis=0)
        idx = np.full(dist.shape[1], np.nan)
        idx[collides] = np.argmax(
            np.diff(dist[:, collides] < 0, axis=0, prepend=False), axis=0
        )
        return idx

    def collision_times(self, s, sr):
        """Earliest collision-time index with particle 1 or 2."""
        s_mag = np.linalg.norm(s, axis=-1) - 1
        sr_mag = np.linalg.norm(sr, axis=-1) - 1
        return np.minimum(
            self._first_crossing_idx(s_mag),
            self._first_crossing_idx(sr_mag),
        )

    # -- Local solutions ----------------------------------------------------

    def g_light(self, t, t_bound, g_boundary):
        """Analytic solution in a light (unshadowed) region."""
        return 1 + (g_boundary - 1) * np.cosh(self.u_norm * t) / np.cosh(
            self.u_norm * t_bound
        )

    def g_shadow(self, t, t_bound, g_boundary):
        """Analytic solution in a shadow region."""
        return g_boundary * np.cosh(self.u_norm * t) / np.cosh(self.u_norm * t_bound)

    # -- Core propagation ---------------------------------------------------

    def _propagate_grid(
        self, t0: np.ndarray, l0: np.ndarray, n_prop: int = 200
    ) -> np.ndarray:
        """Given a grid of initial positions l0 and offset initial times t0,
        propagate g along characteristics and return the grid of g values."""

        trivial = self.trivial_mask(l0)
        need_trace = ~trivial

        # Remove points for which g=1 is trivial
        l0_initial = l0[need_trace]
        _t, _x = np.meshgrid(t0, l0_initial[:, 0], indexing="xy")
        _, _y = np.meshgrid(t0, l0_initial[:, 1], indexing="xy")
        grid = np.stack([_t, _x, _y], axis=-1).transpose(1, 0, 2)  # (n_times, n_pts, 3)
        grid = grid.reshape(-1, 3)  # (n_times * n_pts, 3)

        # Propagation time: enough for each characteristic to be fully
        # resolved, then take the worst case.
        drift_speed = max(np.linalg.norm(self.drift), 1e-10)
        per_point_t = 1.5 * np.linalg.norm(l0_initial, axis=-1) / drift_speed
        t_prop_max = np.max(per_point_t)
        t_prop = np.linspace(0, t_prop_max, n_prop)

        characteristics = np.zeros((len(t_prop), *grid.shape))

        t = t_prop[:, None] + grid[:, 0][None, :]

        characteristics[..., 0] = self.phi(t)
        characteristics[..., 1:3] = (
            grid[None, :, 1:] + t_prop[:, None, None] * self.drift[None, None, :]
        )

        s, sr = self.relative_positions(
            characteristics[..., 1:3], characteristics[..., 0]
        )

        coll_time = self.collision_times(s, sr)

        # Initial conditions
        n_cand = grid.shape[0]
        has_coll = ~np.isnan(coll_time)
        ic_idx = np.where(has_coll, coll_time, n_prop - 1).astype(int)
        ic_val = np.where(has_coll, 0.0, 1.0)

        # Shadow masks & signed distances
        in_shadow_v, in_shadow_vu, s_cross, sr_cross = self.shadow_masks(s, sr)
        in_shadow = in_shadow_v | in_shadow_vu

        # Propagate from IC toward t = 0
        g_cand = np.ones((n_prop, n_cand))
        g_cand[ic_idx, np.arange(n_cand)] = ic_val

        g_boundary = ic_val.copy()
        t_bnd_idx = ic_idx.copy()
        particle_idx = np.arange(n_cand)

        for i in range(n_prop - 2, -1, -1):
            active = i < ic_idx
            if not np.any(active):
                continue

            # TODO replace with real boundary corrections
            delta_g_v_plus = 1.0
            delta_g_v_minus = 1.0
            delta_g_vu_plus = 1.0
            delta_g_vu_minus = 1.0

            # Detect shadow ↔ light transitions per tube & side
            changed_v = (in_shadow_v[i] != in_shadow_v[i + 1]) & active
            changed_vu = (in_shadow_vu[i] != in_shadow_vu[i + 1]) & active
            tr_v_plus = changed_v & (s_cross[i] > 0)
            tr_v_minus = changed_v & (s_cross[i] < 0)
            tr_vu_plus = changed_vu & (sr_cross[i] > 0)
            tr_vu_minus = changed_vu & (sr_cross[i] < 0)
            transition = tr_v_plus | tr_v_minus | tr_vu_plus | tr_vu_minus

            # Update boundary value with jump correction
            g_prev = g_cand[i + 1]
            g_boundary = np.where(tr_v_plus, g_prev + delta_g_v_plus, g_boundary)
            g_boundary = np.where(tr_v_minus, g_prev + delta_g_v_minus, g_boundary)
            g_boundary = np.where(tr_vu_plus, g_prev + delta_g_vu_plus, g_boundary)
            g_boundary = np.where(tr_vu_minus, g_prev + delta_g_vu_minus, g_boundary)
            t_bnd_idx = np.where(transition, i + 1, t_bnd_idx)

            # Evaluate local solutions
            g_s = self.g_shadow(t[i], t[t_bnd_idx, particle_idx], g_boundary)
            g_l = self.g_light(t[i], t[t_bnd_idx, particle_idx], g_boundary)
            g_cand[i] = np.where(active, np.where(in_shadow[i], g_s, g_l), g_cand[i])

        inside = (np.linalg.norm(s, axis=-1) < 1) | (np.linalg.norm(sr, axis=-1) < 1)
        g_cand[inside] = np.nan

        g = np.full((len(t0), l0.shape[0]), np.nan)
        g[:, trivial] = 1.0
        g[:, need_trace] = g_cand[0, :].reshape((len(t0), l0_initial.shape[0]))

        return g

    def visualize_3d(
        self,
        l0: np.ndarray,
        n_times: int = 200,
        n_chars: int = 100,
        l_bounds: tuple[float, float] = (-5.0, 5.0),
        vmin: float = 0.0,
        vmax: float = 2.0,
    ):
        """Interactive 3D plot of characteristics in (phi, lx, ly) space,
        colored by local g, with |s|=1 and |sr|=1 exclusion surfaces.

        Uses plotly for interactive rotation/zoom.

        Parameters
        ----------
        l0 : (n_pts, 2)   Starting positions (a random subset is drawn).
        n_times : int      Number of time steps for the characteristics.
        n_chars : int      Max number of characteristics to draw.
        l_bounds : (lo, hi) Spatial domain bounds for display.
        vmin, vmax : float Color scale range for g.
        """
        import plotly.graph_objects as go

        l0 = np.atleast_2d(l0)
        if l0.shape[0] > n_chars:
            idx = np.random.choice(l0.shape[0], n_chars, replace=False)
            l0 = l0[idx]

        # Compute g along characteristics
        t, g, l0 = self.compute_g(l0, n_times=n_times)
        _l = self.characteristic(t, l0)  # (n_times, n_pts, 2)
        _phi = self.phi(t)  # (n_times,)

        fig = go.Figure()

        # -- Draw characteristics as colored lines --
        phi_broadcast = np.broadcast_to(_phi[:, None], g.shape)

        for j in range(l0.shape[0]):
            gj = g[:, j]
            valid = ~np.isnan(gj)
            if not np.any(valid):
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=_l[valid, j, 0],
                    y=_l[valid, j, 1],
                    z=phi_broadcast[valid, j],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=gj[valid],
                        colorscale="Viridis",
                        cmin=vmin,
                        cmax=vmax,
                        showscale=(j == 0),
                        colorbar=dict(title="g") if j == 0 else None,
                    ),
                    showlegend=False,
                    hovertemplate=(
                        "lx=%{x:.2f}<br>ly=%{y:.2f}<br>"
                        "φ=%{z:.3f}<br>g=%{marker.color:.3f}<extra></extra>"
                    ),
                )
            )

        # -- Exclusion surfaces |s|=1 and |sr|=1 --
        # These are cylinders of radius 1 centred on the rotating particle
        # positions, swept over phi.
        n_phi_surf = 60
        n_theta = 40
        phi_lo, phi_hi = float(_phi[0]), float(_phi[-1])
        phi_arr = np.linspace(phi_lo, phi_hi, n_phi_surf)
        theta = np.linspace(0, 2 * np.pi, n_theta)
        PHI_S, TH = np.meshgrid(phi_arr, theta, indexing="ij")

        for sign, name, color in [
            (-1, "|s|=1 (particle 1)", "rgba(255,80,80,0.25)"),
            (+1, "|sr|=1 (particle 2)", "rgba(80,80,255,0.25)"),
        ]:
            # Centre of the exclusion sphere at each phi
            cx = sign * 0.5 * np.cos(PHI_S)
            cy = sign * 0.5 * np.sin(PHI_S)
            X = cx + np.cos(TH)
            Y = cy + np.sin(TH)
            Z = PHI_S

            lo, hi = l_bounds
            mask = (X >= lo) & (X <= hi) & (Y >= lo) & (Y <= hi)
            X = np.where(mask, X, np.nan)
            Y = np.where(mask, Y, np.nan)

            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    surfacecolor=np.zeros_like(Z),
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=0.3,
                    name=name,
                    hoverinfo="skip",
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis_title="lx",
                yaxis_title="ly",
                zaxis_title="φ",
                aspectmode="manual",
                aspectratio=dict(
                    x=1,
                    y=1,
                    z=0.5,
                ),
            ),
            title="Characteristics in (lx, ly, φ) space — colored by g",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.show()


# ---- Example usage --------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    u = np.array([np.cos(-np.pi), np.sin(-np.pi)])
    v = np.array([np.cos(np.pi / 2), np.sin(np.pi / 2)])
    solver = ThreeBodyG(u, v, plane=1)
    _coord = np.linspace(-1.5, 6, 100)
    l0 = np.stack(
        np.meshgrid(_coord, _coord, indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    phi_grid = np.linspace(0, np.pi / 2, 10)
    t0 = solver.t_from_phi(phi_grid)
    g = solver._propagate_grid(t0, l0)

    fig = plt.figure(figsize=(10, 6))
    for i in range(6):
        ax, grid = plt.subplot(2, 3, i + 1), g[i].reshape((100, 100))
        ax.imshow(grid.T, origin="lower", extent=(-3, 3, -3, 3), vmin=0, vmax=2)
        ax.plot([0, v[0]], [0, v[1]], "r-", label="v")
        ax.plot([0, v[0] - u[0]], [0, v[1] - u[1]], "g-", label="v-u")
        ax.plot([0, v[0] - u[0] / 2], [0, v[1] - u[1] / 2], "y-", label="v-u/2")

    # phi_grid, lx, ly, g_grid = solver.compute_g_on_grid(
    #     l_bounds=(-5, 5), n_l=80, n_times=20
    # )

    # # 3D interactive visualisation
    # l0_vis = np.random.uniform([-5, -5], [5, 5], size=(500, 2))
    # solver.visualize_3d(l0_vis, n_times=200, n_chars=150, l_bounds=(-5, 5))

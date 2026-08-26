r"""Physical models for the h-hierarchy structure-factor solver.

Everything the 2-body/3-body machinery in ``struct_2b`` and ``struct_3b``
needs to know about a particular microscopic model is bundled into one
frozen :class:`Model`:

    L(k)         -> (2s+1, 2s+1) linear operator L(k)
    rhs_2b(k)    -> (2s+1, 2s+1) source of  L h + h L^H = rhs_2b
    vertex_terms : builder of the 3-body collision RHS (None if undefined)
    phi, eps, V  : couplings/potential the 3-body collision integral reuses
    s            : harmonic truncation, fixed at construction

`L` and `rhs_2b` carry the model's own parameters baked in -- including
the truncation ``s``, so they take the wave vector alone and the solver
functions take a single ``model`` argument instead of the loose
``(lp, phi, eps, V, ..., D)`` tuple.  To try a new model you build a
``Model`` (or write a small ``*_model`` factory like the two below) and
pass it in -- no need to copy a whole solver file and swap an import.

The 3-body vertex is kept model-specific: ``vertex_terms`` defaults to the
MIPS pairwise-force vertex (``struct_3b._vertex_terms``).  A model whose
``vertex_terms is None`` supports the 2-body path only; requesting a
3-body quantity on it raises a clear error.

Built-ins
---------
``abp_model``    : the pairwise-force ABP/MIPS model (full 2b + 3b).
``vicsek_model`` : the isotropic alignment (Vicsek) model.  Operator and
                   2-body source only -- its 3-body vertex is left blank
                   (``vertex_terms=None``) to be filled in later.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from struct_aux import (
    L_turning_away,
    dot,
    L as _L_abp,
    L_Vicsek,
    Vexp,
)
from struct_3b import _vertex_terms as _mips_vertex_terms
from scipy.special import iv
from scipy.optimize import fsolve


@dataclass(frozen=True)
class Model:
    r"""A microscopic model, as seen by the structure-factor solver.

    Attributes
    ----------
    L : callable ``(k) -> (2s+1, 2s+1) ndarray``
        The linear operator L(k); k may be a complex (vector-valued)
        wavenumber.  All physical parameters, including the truncation
        ``s``, are baked in.
    rhs_2b : callable ``(k) -> (2s+1, 2s+1) ndarray``
        Source term of the 2-body equation  L h + h L^H = rhs_2b(k).
    s : int
        Harmonic truncation (matrices span n = -s..s).  Fixed at
        construction; build a new model to change it.
    phi : float
        Packing fraction; sets ``rho0 = 4 phi / pi`` in the collision
        integral.
    eps : float
        Overall coupling reused by the 3-body collision integral and its
        vertex.
    V : callable or None
        Interaction potential (Fourier space) reused by the 3-body
        collision integrand ``dot(k, q) V(q)`` and by ``vertex_terms``.
    vertex_terms : callable or None
        Builder of the rank-structured terms of the 3-body RHS, same
        signature as ``struct_3b._vertex_terms``
        ``(h1, h2, h12, k1, k2, V, phi, eps, s) -> list``.  ``None`` marks
        a model without a 3-body closure (2-body path only).
    name : str
        Human-readable label.
    """

    L: Callable[[complex], np.ndarray]
    rhs_2b: Callable[[complex], np.ndarray]
    phi: float
    s: int = 30
    eps: float = 1.0
    qn: Optional[np.ndarray] = None
    rhs_deriv: Optional[Callable[[complex, int], np.ndarray]] = None
    V: Optional[Callable] = None
    Kern: Optional[Callable] = None
    vertex_terms: Optional[Callable] = _mips_vertex_terms
    name: str = ""

    @property
    def rho0(self) -> float:
        """Reference density rho0 = 4 phi / pi (phi = pi rho0 (a/2)^2, a = 1)."""
        return 4 * self.phi / np.pi

    @property
    def supports_3b(self) -> bool:
        """Whether the 3-body path is defined (needs a vertex and a potential)."""
        return self.vertex_terms is not None and self.V is not None

    def require_3b(self) -> None:
        """Raise a helpful error if this model has no 3-body closure."""
        if not self.supports_3b:
            raise NotImplementedError(
                f"model {self.name!r} has no 3-body vertex "
                "(vertex_terms/V is None); only which='2b' is available. "
                "Define model.vertex_terms to enable the 3-body path."
            )


def _abp_source(V, eps, s):
    """MIPS/ABP 2-body source: -2 |k|^2 V(k) eps at the (0, 0) harmonic.

    ``dot(k, k) = |k|^2`` so this also holds for complex vector-valued k
    (matches ``struct_3b.compute_h``); for real k it equals the old
    ``RHS_2b`` (-2 k^2 V(k) eps).
    """

    def rhs_2b(k):
        N = 2 * s + 1
        mat = np.zeros((N, N), dtype=np.complex128)
        mat[s, s] = -2 * dot(k, k) * V(k) * eps
        return mat

    return rhs_2b


def abp_model(lp, phi, eps=1.0, V=Vexp, D=0.0, s=30, name=None):
    """Pairwise-force ABP / MIPS model (the original struct_2b/3b model).

    Full 2-body and 3-body support.  ``L`` is ``struct_aux.L`` and the
    2-body source is ``-2 |k|^2 V(k) eps`` at n = 0; the 3-body vertex is
    the MIPS pairwise-force vertex.
    """
    return Model(
        L=lambda k: _L_abp(k, lp, phi, eps, V, s=s, D=D),
        rhs_2b=_abp_source(V, eps, s),
        s=s,
        phi=phi,
        eps=eps,
        V=V,
        vertex_terms=_mips_vertex_terms,
        name=name or f"ABP(lp={lp:g}, phi={phi:g}, eps={eps:g}, D={D:g})",
    )


# def _vicsek_source(Kern, gamma, phi):
#     r"""Vicsek 2-body source, structurally analogous to the ABP one.

#     The MIPS source equals ``-2/rho0`` times the interaction (non-free)
#     part of L, placed at the harmonic the interaction couples.  The
#     isotropic Vicsek interaction sits at n = +/-1 with strength
#     ``rho0 gamma/2 Kern(k)``, so the same rule gives ``-gamma Kern(k)``
#     at n = +/-1.  This is an assumption carried over from the pairwise
#     case; override ``model.rhs_2b`` if your derivation differs.
#     """

#     def rhs_2b(k, s):
#         N = 2 * s + 1
#         mat = np.zeros((N, N), dtype=np.complex128)
#         mat[s + 1, s + 1] = gamma * Kern(k)
#         mat[s - 1, s - 1] = gamma * Kern(k)
#         return mat

#     return rhs_2b


def _vicsek_source(Kern, gamma, qn, s):
    r"""Vicsek 2-body source, structurally analogous to the ABP one.

    The MIPS source equals ``-2/rho0`` times the interaction (non-free)
    part of L, placed at the harmonic the interaction couples.  The
    isotropic Vicsek interaction sits at n = +/-1 with strength
    ``rho0 gamma/2 Kern(k)``, so the same rule gives ``-gamma Kern(k)``
    at n = +/-1.  This is an assumption carried over from the pairwise
    case; override ``model.rhs_2b`` if your derivation differs.
    """

    def rhs_2b(k):
        N = 2 * s + 1
        n = np.arange(-s, s + 1)
        mat = np.zeros((N, N), dtype=np.complex128)
        mat += (
            gamma
            * Kern(k)
            / 2
            * (n[:, None] + n[None, :])
            * (
                qn[:-2, None] * qn[None, :-2].conjugate()
                - qn[2:, None] * qn[None, 2:].conjugate()
            )
        )
        return mat

    return rhs_2b


def vicsek_source_deriv(Kern, gamma, phi, qn, s):
    """RHS for the Sylvester-Lyapunov equation of the derivative of H
    with respect to the polar order parameter in the Vicsek model."""

    def rhs_deriv(k, h):
        N = 2 * s + 1
        n = np.arange(-s, s + 1)
        if h.shape[0] == N:
            # Pad h to include the n = +/- (s+1) harmonics, which are needed for the derivative
            _h = np.pad(h, ((1, 1), (1, 1)), mode="constant")
        else:
            _h = h

        rho_0 = 4 * phi / np.pi
        mat = np.zeros((N, N), dtype=np.complex128)
        mat[s + 2, :] += gamma * Kern(k) / 2 * (2 + n) * qn[:-2].conjugate()
        mat[:, s] += gamma * Kern(k) / 2 * n * qn[:-2]
        mat[s, :] += -gamma * Kern(k) / 2 * n * qn[2:].conjugate()
        mat[:, s - 2] += -gamma * Kern(k) / 2 * (n - 2) * qn[2:]
        mat[s + 2, :] += gamma * rho_0 * Kern(k) * _h[s + 2, 1:-1]
        mat[:, s - 2] += gamma * rho_0 * Kern(k) * _h[1:-1, s]
        mat += (
            gamma * rho_0 / 2 * (n[:, None] * _h[:-2, 1:-1] - _h[1:-1, 2:] * n[None, :])
        )
        return mat

    return rhs_deriv


def vicsek_model(lp, phi, gamma, Kern, s=30, D=0.0, name=None):
    """Isotropic alignment (Vicsek) model.

    Operator ``struct_aux.L_Vicsek_isotropic`` (alignment coupling at the
    n = +/-1 harmonics) with the analogous 2-body source.  The 3-body
    vertex is intentionally left blank (``vertex_terms=None``): the
    2-body path works, ``which='3b'`` raises until a vertex is supplied.
    """
    rho_0 = 4 * phi / np.pi
    q1 = fsolve(lambda q: q - iv(1, gamma * rho_0 * q) / iv(0, gamma * rho_0 * q), 1.0)[
        0
    ]
    qn = iv(np.arange(-s - 1, s + 2), gamma * rho_0 * q1) / iv(
        0, gamma * rho_0 * q1
    )  # Compute one more harmonic, needed in the construction of L
    if qn[-1] > 1e-3:
        print(
            "Warning: the truncation s={} is too small for the Vicsek model, qn[-1]={}".format(
                s, qn[-1]
            )
        )

    return Model(
        L=lambda k: L_Vicsek(k, lp, phi, gamma, Kern, qn, s=s, D=D),
        rhs_2b=_vicsek_source(Kern, gamma, qn, s),
        rhs_deriv=vicsek_source_deriv(Kern, gamma, phi, qn, s),
        qn=qn,
        s=s,
        phi=phi,
        eps=gamma,
        Kern=Kern,
        vertex_terms=None,  # TODO: fill in the Vicsek 3-body vertex
        name=name or f"Vicsek(lp={lp:g}, phi={phi:g}, gamma={gamma:g}, D={D:g})",
    )


def turning_away_source(gamma, eps, V, Kern, qn, s):

    def rhs_turning_away(k):
        N = 2 * s + 1
        n = np.arange(-s, s + 1)
        mat = np.zeros((N, N), dtype=np.complex128)
        # Repulsive interactions
        mat += (
            -2
            * eps
            * V(k)
            * np.abs(k) ** 2
            * qn[1:-1, None]
            * np.conjugate(qn[None, 1:-1])
        )
        # Turning away torque
        mat += (
            1j
            * gamma
            / 2
            * Kern(k)
            * (
                n[:, None]
                * (qn[:-2, None] * k - qn[2:, None] * np.conjugate(k))
                * np.conjugate(qn[None, 1:-1])
                + n[None, :]
                * qn[1:-1, None]
                * (
                    np.conjugate(qn[None, 2:]) * k
                    - np.conjugate(qn[None, :-2]) * np.conjugate(k)
                )
            )
        )
        return mat

    return rhs_turning_away


def turning_away_source_deriv(gamma, eps, V, Kern, phi, s):

    def rhs_deriv(k, h):
        rho_0 = 4 * phi / np.pi
        N = 2 * s + 1
        n = np.arange(-s, s + 1)
        if h.shape[0] == N:
            # Pad h to include the n = +/- (s+1) harmonics, which are needed for the derivative
            _h = np.pad(h, ((1, 1), (1, 1)), mode="constant")
        else:
            _h = h

        mat = np.zeros((N, N), dtype=np.complex128)
        # Repulsive interactions
        mat[s + 1, :] += -eps * rho_0 * V(k) * _h[s + 1, 1:-1] * np.abs(k) ** 2
        mat[:, s - 1] += -eps * rho_0 * V(k) * _h[1:-1, s + 1] * np.abs(k) ** 2
        mat[s + 2, :] += 1j * gamma * rho_0 * k * Kern(k) * _h[s + 1, 1:-1]
        mat[:, s - 2] += -1j * gamma * rho_0 * k * Kern(k) * _h[1:-1, s + 1]
        mat[s + 1, s] += -2 * eps * V(k) * np.abs(k) ** 2
        mat[s, s - 1] += -2 * eps * V(k) * np.abs(k) ** 2
        mat[s + 2, s] += 1j * gamma * k * Kern(k)
        mat[s, s - 2] += -1j * gamma * k * Kern(k)

        mat[s - 1, s - 1] += 1j * gamma * Kern(k) * np.conjugate(k) / 2
        mat[s + 1, s + 1] += -1j * gamma * Kern(k) * np.conjugate(k) / 2
        return mat

    return rhs_deriv


def turning_away_model(lp, phi, eps, gamma, V, Kern, s=30, D=0.0, name=None):
    """Turning-away model (repulsive + torque).

    Operator ``struct_aux.L_turning_away`` with the analogous 2-body source.
    The 3-body vertex is intentionally left blank (``vertex_terms=None``):
    the 2-body path works, ``which='3b'`` raises until a vertex is supplied.
    """
    qn = np.zeros(2 * s + 3, dtype=np.complex128)
    qn[s + 1] = 1.0  # n = 0
    return Model(
        L=lambda k: L_turning_away(k, lp, phi, eps, gamma, V, Kern, qn=qn, s=s, D=D),
        rhs_2b=turning_away_source(gamma, eps, V, Kern, qn=qn, s=s),
        rhs_deriv=turning_away_source_deriv(gamma, eps, V, Kern, phi, s),
        s=s,
        phi=phi,
        eps=eps,
        V=V,
        Kern=Kern,
        qn=qn,
        vertex_terms=None,
        name=name
        or f"TurningAway(lp={lp:g}, phi={phi:g}, eps={eps:g}, gamma={gamma:g}, D={D:g})",
    )

"""Vicsek (isotropic alignment) model on the generic structure-factor core.

This used to be a full copy of ``struct_2b.py`` with the operator import
swapped -- exactly the file-duplication the ``Model`` abstraction removes.
Now the whole solver is shared: you only describe the model once (its
operator and 2-body source live in ``models.vicsek_model``) and hand it to
the same ``compute_correlations`` the ABP model uses.

The Vicsek 3-body vertex has not been written yet, so ``vicsek_model``
ships with ``vertex_terms=None``: the 2-body path works and ``which="3b"``
raises a clear error until the vertex is filled in (see ``models.py``).
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

# the solver library lives in ../src (flat, mutually-importing modules)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from models import vicsek_model, abp_model  # noqa: E402
from struct_2b import compute_correlations, make_k_grid  # noqa: E402
from struct_aux import Kern_gauss, Vexp  # noqa: E402


if __name__ == "__main__":
    # Alignment model: gamma sets the coupling strength, Kern its range.
    lp, phi, gamma, D = 10.0, 0.04, 1.0, 0.0
    s = 20
    model = vicsek_model(lp=lp, phi=phi, gamma=gamma, Kern=Kern_gauss, D=D, s=s)
    print("model:", model.name, "| supports 3b:", model.supports_3b)

    # 2-body structure factor -- identical call to the ABP workflow,
    # only the `model` argument changes.
    k_arr = make_k_grid(kmax=12.0, rmax=10.0, dk_factor=8.0, kmin=0.1 / lp, n_log=40)
    X = compute_correlations(model, k_arr, s=s, which="2b", parallel=True)

    # S_00(k) = 1 + rho0 X_00(k), with rho0 = 4 phi / pi
    S00 = 1 + model.rho0 * np.real(X[s, s])

    # which="3b" raises until the Vicsek vertex is defined:
    try:
        compute_correlations(model, k_arr[:1], s=s, which="3b")
    except NotImplementedError as err:
        print("3b not available yet:", err)

    # For comparison, the ABP model runs through the *same* functions.
    abp = abp_model(lp=lp, phi=phi, eps=1.0, V=Vexp, D=D, s=s)
    X_abp = compute_correlations(abp, k_arr, s=s, which="2b", parallel=True)
    S00_abp = 1 + abp.rho0 * np.real(X_abp[s, s])

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(k_arr, S00, label="Vicsek $S_{00}$")
    ax.plot(k_arr, S00_abp, label="ABP $S_{00}$")
    ax.axhline(1.0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$S_{00}(k)$")
    ax.legend()
    plt.show()

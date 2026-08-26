#!/usr/bin/env python
"""Overlay a simulated S_nm(k) (``abp_structure_factor.py``) with the
h-hierarchy prediction of ``StructFactor_MIPS/struct_2b.py``.

The simulation ``.npz`` stores the theory-unit parameters (lp,
eps_struct, D_struct, phi) and the potential; the script solves the
2-body (optionally 3-body) Sylvester equation on the simulated k-grid
and plots

    S_nm(k)  vs  delta_nm + rho0 X_nm(k),      rho0 = 4 phi / pi,

for a few low harmonics (the (0,0) element is the ordinary structure
factor).  k is converted to theory units 1/a.

Usage
-----
    python compare_struct_sim.py abp_structure_factor.npz [--which 2b|3b]
        [--out compare.png] [--kmin 0.05]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "StructFactor_MIPS", "src"))

from struct_2b import compute_correlations  # noqa: E402
from models import abp_model  # noqa: E402
from abp_structure_factor import Vk_exp, Vk_harm  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", help="Simulation output from ABPStructureFactor.save")
    p.add_argument("--which", default="2b", choices=["2b", "3b"])
    p.add_argument("--out", default=None, help="Figure path (default: <npz>_vs_<which>.png)")
    p.add_argument("--kmin", type=float, default=0.0, help="Drop sim bins below this k (units 1/a)")
    p.add_argument("--s", type=int, default=30, help="Theory truncation: harmonics n = -s..s")
    p.add_argument("--no-show", action="store_true", help="Only save the figure (headless)")
    args = p.parse_args(argv)

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(args.npz)
    a = float(d["a"])
    lp = float(d["lp"])
    eps = float(d["eps_struct"])
    D = float(d["D_struct"])
    phi = float(d["phi"])
    lV = float(d["lV"]) / a  # theory units
    potential = str(d["potential"])
    s_sim = int(d["s"])
    if not np.isfinite(lp):
        raise SystemExit("Dr = 0 in the simulation: no theory-unit mapping (lp = inf)")

    if potential == "exp":
        V = lambda k: Vk_exp(k, lV)  # noqa: E731
    else:
        V = lambda k: Vk_harm(k, lV)  # noqa: E731

    k_sim = np.asarray(d["k"]) * a  # -> units 1/a
    keep = k_sim >= args.kmin
    k_sim = k_sim[keep]
    S_sim = np.asarray(d["S"])[keep]

    print(
        f"theory: which={args.which}, lp={lp:g}, phi={phi:g}, eps={eps:g}, "
        f"D={D:g}, V={potential}(lV={lV:g}), {k_sim.size} k-points"
    )
    model = abp_model(lp, phi, eps, V, D, s=args.s)
    X = compute_correlations(
        model, k_sim, s=args.s, which=args.which, parallel=True
    )

    rho0 = 4 * phi / np.pi
    c_th, c_sim = args.s, s_sim

    def S_theory(n, m):
        pred = rho0 * X[c_th + n, c_th + m, :]
        if n == m:
            pred = pred + 1.0
        return pred

    def S_simulated(n, m):
        return S_sim[:, c_sim + n, c_sim + m]

    elements = [
        (0, 0, np.real, "Re"),
        (1, 0, np.real, "Re"),
        (1, 0, np.imag, "Im"),
        (1, 1, np.real, "Re"),
        (1, -1, np.real, "Re"),
        (2, 0, np.real, "Re"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained", sharex=True)
    for ax, (n, m, part, pname) in zip(axes.ravel(), elements):
        ax.plot(k_sim, part(S_theory(n, m)), "-", color="C1", label=f"theory {args.which}")
        ax.plot(k_sim, part(S_simulated(n, m)), ".", ms=4, color="C0", label="simulation")
        ax.axhline(1.0 if (n == m and pname == "Re") else 0.0, color="0.8", lw=0.8, zorder=0)
        ax.set_title(f"{pname} $S_{{{n},{m}}}(k)$")
        ax.set_xlabel("$k a$")
    axes[0, 0].legend()
    fig.suptitle(
        f"$l_p$={lp:g}, $\\phi$={phi:g}, $\\epsilon$={eps:g}, D={D:g}, "
        f"V={potential}($l_V$={lV:g}) | N={int(d['N'])}, {int(d['n_samples'])} samples"
    )

    out = args.out or args.npz.replace(".npz", f"_vs_{args.which}.png")
    fig.savefig(out, dpi=180)
    print(f"saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

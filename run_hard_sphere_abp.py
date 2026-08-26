#!/usr/bin/env python
"""Command-line runner for a single :class:`HardSphereABP` simulation.

Designed for cluster (PBS) jobs: every physical and numerical parameter is a CLI
option, so a PBS array job can sweep them through environment variables.  The run
computes ``g_bulk`` / ``g_surf`` / ``g_of_r`` and writes ``<out_dir>/<name>.npz``
(via :meth:`HardSphereABP.save`, which also stores all parameters so repeats can
be grouped afterwards by ``Cluster/collect_results.py``).

Example
-------
    python run_hard_sphere_abp.py results/run0 --N 2000 --phi 0.3 --v0 10 --Dr 0.5 \
        --n_steps 200000 --burn_in 20000 --sample_every 50 --seed 7
"""

from __future__ import annotations

import argparse
import os

from hard_sphere_abp import HardSphereABP


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run one HardSphereABP simulation and save its pair correlations."
    )
    p.add_argument("out_dir", help="Directory to write the .npz result into (created if needed).")

    # physical parameters
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--phi", type=float, default=0.3)
    p.add_argument("--v0", type=float, default=10.0)
    p.add_argument("--Dr", type=float, default=0.5)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--Dt", type=float, default=0.0)

    # histogram / numerical parameters
    p.add_argument("--r_max", type=float, default=None, help="Default: 4*a.")
    p.add_argument("--n_r", type=int, default=60)
    p.add_argument("--n_alpha", type=int, default=72)
    p.add_argument("--n_beta", type=int, default=36)
    p.add_argument("--dt", type=float, default=None, help="Default: 0.01*a/v0.")
    p.add_argument("--n_sweeps", type=int, default=20)
    p.add_argument("--contact_band", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)

    # run control
    p.add_argument("--n_steps", type=int, default=200_000)
    p.add_argument("--burn_in", type=int, default=20_000)
    p.add_argument("--sample_every", type=int, default=50)
    p.add_argument("--name", default="hard_sphere_abp", help="Base filename for the .npz output.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    sim = HardSphereABP(
        N=args.N,
        phi=args.phi,
        v0=args.v0,
        Dr=args.Dr,
        a=args.a,
        Dt=args.Dt,
        r_max=args.r_max,
        n_r=args.n_r,
        n_alpha=args.n_alpha,
        n_beta=args.n_beta,
        dt=args.dt,
        n_sweeps=args.n_sweeps,
        contact_band=args.contact_band,
        seed=args.seed,
        verbose=True,
    )
    sim.run(
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        sample_every=args.sample_every,
        save_path=os.path.join(args.out_dir, args.name),
    )


if __name__ == "__main__":
    main()

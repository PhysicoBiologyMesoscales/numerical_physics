"""Check the depletion wings of the orientation-averaged pair correlation.

Compares three things at low density (phi = 0.01) and high Pe:

1.  Theory (StructFactor_MIPS/depletion_HS.py): the low-density Pe -> inf
    prediction for h(r, psi) = g(r, psi) - 1, where psi is the bearing of the
    second particle in the tagged particle's heading frame (psi = 0 dead
    ahead), averaged over the orientation of the second particle:

        B = 0                                   |psi| < arccos(a/r)   (upstream)
        B = (jac + ac - |psi|)/pi               ac < |psi| < pi - ac  (wings)
        B = 2 (jac - arcsin(a/r))/pi            |psi| > pi - ac       (behind)

    with jac = 1/sqrt(r^2 - 1) (tangent-caustic fold) -- so an untouched
    forward cone, a positive caustic rim, *negative depletion wings* at the
    rear sides, and a small excess dead behind.

2.  Reconstruction from a saved (alpha, beta)-binned run: psi = alpha + beta
    - pi/2, so the old 3-D histogram can be marginalised onto psi after
    undoing the Monte-Carlo angular normalisation (regenerated with the same
    seed, so it cancels exactly).

3.  A fresh run with the new direct psi-binning (g_bulk_psi in
    hard_sphere_abp.py), if its .npz is present.
"""

import os

import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_RUN = os.path.join(HERE, "hard_sphere_abp_lowphi_Pe10000.npz")
NEW_RUN = os.path.join(HERE, "hard_sphere_abp_lowphi_Pe10000_psi.npz")
TWOPI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
def theory_B(r, psi):
    """Depletion-wing prediction of depletion_HS.py (units a = 1; r > 1)."""
    ac = np.arccos(1.0 / r)
    asi = np.arcsin(1.0 / r)
    jac = 1.0 / np.sqrt(r**2 - 1.0)
    return np.where(
        np.abs(psi) > np.pi - ac,
        2.0 / np.pi * (jac - asi),
        np.where(np.abs(psi) > ac, (jac + ac - np.abs(psi)) / np.pi, 0.0),
    )


# --------------------------------------------------------------------------- #
def reconstruct_psi(path):
    """Marginalise a saved g_bulk(r, alpha, beta) onto psi = alpha + beta - pi/2.

    Undoes the ideal-gas normalisation with the *identical* Monte-Carlo weights
    (same seed and sample count as HardSphereABP._angular_reference), recovers
    the raw counts, and re-bins them on psi, whose ideal-gas measure is flat.
    Each (alpha, beta) cell spans exactly two psi bins (triangular weight
    centred on a bin edge), so its counts are split half-half.
    """
    d = np.load(path)
    g_bulk = d["g_bulk"]
    alpha, beta = d["alpha"], d["beta"]
    alpha_edges, beta_edges = d["alpha_edges"], d["beta_edges"]
    n_psi = alpha.size
    seed = int(d["seed"])

    # Same MC draws as _angular_reference -> the weights cancel exactly.
    rng = np.random.default_rng(seed + 1)
    n_mc = 10_000_000
    ti = rng.uniform(0.0, TWOPI, n_mc)
    tj = rng.uniform(0.0, TWOPI, n_mc)
    phi = rng.uniform(0.0, TWOPI, n_mc)
    al = np.mod(phi - 0.5 * (ti + tj) + 0.5 * np.pi, TWOPI)
    be = 0.5 * (tj - ti)
    w, _, _ = np.histogram2d(al, be, bins=[alpha_edges, beta_edges])
    w /= w.sum()

    counts = np.where(np.isnan(g_bulk), 0.0, g_bulk) * w[None, :, :]

    psi_edges = np.linspace(-np.pi, np.pi, n_psi + 1)
    psi = 0.5 * (psi_edges[:-1] + psi_edges[1:])
    dpsi = TWOPI / n_psi

    g_psi = np.zeros((g_bulk.shape[0], n_psi))
    aa, bb = np.meshgrid(alpha, beta, indexing="ij")
    psi_c = aa + bb - 0.5 * np.pi  # lands exactly on psi-bin edges
    for shift in (-0.25 * dpsi, +0.25 * dpsi):
        p = np.mod(psi_c + shift + np.pi, TWOPI) - np.pi
        idx = np.clip(((p + np.pi) / dpsi).astype(int), 0, n_psi - 1)
        np.add.at(
            g_psi, (slice(None), idx.ravel()), 0.5 * counts.reshape(counts.shape[0], -1)
        )
    g_psi *= n_psi

    # Sanity: the psi-average must reproduce the stored g_of_r.
    err = np.nanmax(np.abs(g_psi.mean(axis=1) - d["g_of_r"]))
    print(f"[reconstruct] max |mean_psi g - g_of_r| = {err:.2e}")
    return d["r"], psi, g_psi


# --------------------------------------------------------------------------- #
def main_old():
    r_rec, psi_rec, g_rec = reconstruct_psi(OLD_RUN)

    have_new = os.path.exists(NEW_RUN)
    if have_new:
        dn = np.load(NEW_RUN)
        r_new, psi_new, g_new = dn["r"], dn["psi"], dn["g_bulk_psi"]
        print(
            f"[direct] loaded {os.path.basename(NEW_RUN)}: "
            f"{int(dn['n_samples'])} samples"
        )

    # ---- figure: polar maps + cuts ---------------------------------------- #
    fig = plt.figure(figsize=(13, 8.5))
    n_maps = 3 if have_new else 2
    titles = [
        "theory  $B(r,\\psi)$  (depletion_HS)",
        "reconstructed from $(\\alpha,\\beta)$ run,  $g-1$",
    ]
    fields = [None, g_rec - 1.0]
    coords = [(r_rec, psi_rec), (r_rec, psi_rec)]
    if have_new:
        titles.append("direct $\\psi$-binning (new code),  $g-1$")
        fields.append(g_new - 1.0)
        coords.append((r_new, psi_new))

    pp, rr = np.meshgrid(psi_rec, r_rec)
    fields[0] = theory_B(rr, pp)

    vlim = 0.35
    for k in range(n_maps):
        ax = fig.add_subplot(2, n_maps, k + 1, projection="polar")
        r_k, psi_k = coords[k]
        ppk, rrk = np.meshgrid(psi_k, r_k)
        msh = ax.pcolormesh(
            ppk,
            rrk,
            fields[k],
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            shading="nearest",
        )
        ax.set_title(titles[k], fontsize=10)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    fig.colorbar(msh, ax=fig.axes[:n_maps], shrink=0.8, label="$g-1$")

    # ---- 1D cuts ----------------------------------------------------------- #
    cut_radii = [1.4, 2.0, 3.0]
    psi_fine = np.linspace(-np.pi, np.pi, 721)
    for k, r0 in enumerate(cut_radii):
        ax = fig.add_subplot(2, 3, 4 + k)
        kr = int(np.argmin(np.abs(r_rec - r0)))
        ax.plot(
            np.degrees(psi_fine),
            theory_B(r_rec[kr], psi_fine),
            "k-",
            lw=1.5,
            label="theory",
        )
        ax.plot(
            np.degrees(psi_rec), g_rec[kr] - 1.0, "C0.", ms=4, label="reconstructed"
        )
        if have_new:
            krn = int(np.argmin(np.abs(r_new - r0)))
            ax.plot(
                np.degrees(psi_new),
                g_new[krn] - 1.0,
                "C3.",
                ms=4,
                label="direct $\\psi$ bins",
            )
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.set_title(f"$r = {r_rec[kr]:.2f}\\,a$", fontsize=10)
        ax.set_xlabel("$\\psi$ (deg)")
        if k == 0:
            ax.set_ylabel("$g(r,\\psi) - 1$")
            ax.legend(fontsize=8)

    fig.suptitle(
        "Orientation-averaged pair correlation around a tagged ABP "
        "($\\phi=0.01$, Pe $=10^4$): depletion wings",
        fontsize=12,
    )
    out = os.path.join(HERE, "depletion_wings_check.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")

    # ---- quantitative wing check ------------------------------------------ #
    print("\nwing depth (minimum of g-1 over psi) at fixed r:")
    print(
        f"{'r':>6} {'theory':>9} {'reconstr':>9}"
        + ("{:>9}".format("direct") if have_new else "")
    )
    for r0 in cut_radii:
        kr = int(np.argmin(np.abs(r_rec - r0)))
        th = theory_B(r_rec[kr], psi_fine).min()
        rc = np.nanmin(g_rec[kr] - 1.0)
        row = f"{r_rec[kr]:6.2f} {th:9.3f} {rc:9.3f}"
        if have_new:
            krn = int(np.argmin(np.abs(r_new - r0)))
            row += f" {np.nanmin(g_new[krn] - 1.0):9.3f}"
        print(row)


def main():
    r_rec, psi_rec, g_rec = reconstruct_psi(OLD_RUN)
    pp, rr = np.meshgrid(psi_rec, r_rec)
    fields = [None, g_rec - 1.0]
    fields[0] = theory_B(rr, pp)
    psi_inrange = (psi_rec + np.pi) % (2 * np.pi) - np.pi
    fields[0][0] += 1 - np.abs(psi_inrange) / np.pi

    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "polar"})
    ax[0].pcolormesh(
        pp,
        rr,
        fields[0],
        cmap="RdBu_r",
        vmin=-0.35,
        vmax=0.35,
        shading="nearest",
        rasterized=True,
    )
    msh = ax[1].pcolormesh(
        pp,
        rr,
        fields[1],
        cmap="RdBu_r",
        vmin=-0.35,
        vmax=0.35,
        shading="nearest",
        rasterized=True,
    )
    ax[0].set_xticks([])
    ax[0].set_yticks([1.0])
    ax[0].set_yticklabels([])
    ax[0].set_thetamin(0)
    ax[0].set_thetamax(180)
    ax[1].set_xticks([])
    ax[1].set_yticks([1.0])
    ax[1].set_yticklabels([])
    ax[1].set_thetamin(180)
    ax[1].set_thetamax(360)
    fig.colorbar(msh, ax=ax, shrink=0.8)
    fig.savefig(
        os.path.join(HERE, "depletion_wings_comparison.svg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":

    main()

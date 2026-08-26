r"""Check of the hierarchy 3-body correlation against the Kirkwood closure.

Compares the 3-body correlation tensor h3_{nml}(k1, k2) of the h-hierarchy
(``struct_3b.compute_3bod``, ABP model without alignment) with the Kirkwood
superposition approximation (KSA) written directly in Fourier space:

    h3^KSA_{nml}(k1, k2) =
          sum_a [ h_{na}(k1) h_{m,l-a}(k2)          (pairs 13.23)
                + h_{na}(k1) h_{m+a,l}(k12)         (pairs 12.23)
                + h_{ma}(k2) h_{n+a,l}(k12) ]       (pairs 12.13)
        + rho0 sum_{ab} h_{na}(k1) h_{mb}(k2) h_{a+b,l}(k12),

with k12 = k1 + k2, rho0 = 4 phi / pi and h = X the pair harmonics of
``struct_2b`` (conventions of ``struct_3b_realspace``: h3_{nml}(k1,k2) pairs
with < c_n(k1) c_m(k2) conj(c_l(k1+k2)) >, c_n(k) = sum_j e^{i n theta_j}
e^{+i k.r_j}).  The triple term uses the convolution approximation (the
exact KSA triple product is a 2D convolution); with it the formula is
exactly the harmonic generalization of the S3 factorization

    S3_{nml}(k1, k2) = sum_{ab} S_{na}(k1) S_{mb}(k2) S_{a+b,l}(k12).

Term (12.13) is implemented both in the raw form
sum_nu h_{nu,-m}(-k2) h_{n-nu,l}(k12) and in the Hermiticity-reduced form
above; they agree to machine precision (checked in stage "test").

Usage:  python check_kirkwood_3b.py [test|scan|map|all]
"""

import pathlib
import sys
import time

import numpy as np

# the solver library lives in ../src (flat, mutually-importing modules)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from models import abp_model  # noqa: E402
from struct_3b import compute_3bod, compute_h  # noqa: E402
from struct_aux import Vexp  # noqa: E402

# ----------------------------------------------------------------------
# KSA builder
# ----------------------------------------------------------------------


def _row_shift(M, d):
    """R[i, :] = M[i + d, :], zero-padded outside the truncation."""
    N = M.shape[0]
    R = np.zeros_like(M)
    if d >= 0:
        R[: N - d] = M[d:]
    else:
        R[-d:] = M[: N + d]
    return R


def _col_shift(M, d):
    """R[:, j] = M[:, j - d], zero-padded outside the truncation."""
    N = M.shape[1]
    R = np.zeros_like(M)
    if d >= 0:
        R[:, d:] = M[:, : N - d]
    else:
        R[:, : N + d] = M[:, -d:]
    return R


def kirkwood_h3(h1, h2, h12, rho0, s):
    """KSA tensor and its pieces: returns (doubles, triple).

    h1 = h(k1), h2 = h(k2), h12 = h(k1 + k2) are pair Sylvester solutions
    (matrix index i <-> harmonic i - s).  Internal harmonic sums are
    truncated to |a|, |b| <= s and shifted indices falling outside the
    truncation are dropped, consistently with the hierarchy truncation.
    The full KSA tensor is doubles + rho0 * triple.
    """
    N = 2 * s + 1
    doubles = np.zeros((N, N, N), dtype=np.complex128)
    triple = np.zeros_like(doubles)
    for a in range(N):
        al = a - s
        h2c = _col_shift(h2, al)  # [m, l] = h2[m, l - al]
        h12r = _row_shift(h12, al)  # [x, l] = h12[x + al, l]
        doubles += h1[:, a][:, None, None] * h2c[None, :, :]
        doubles += h1[:, a][:, None, None] * h12r[None, :, :]
        doubles += h2[:, a][None, :, None] * h12r[:, None, :]
        triple += h1[:, a][:, None, None] * (h2 @ h12r)[None, :, :]
    return doubles, triple


def kirkwood_h3_brute(h1, h2, h12, rho0, s):
    """Nested-loop reference implementation (small s only)."""
    N = 2 * s + 1
    rng = range(-s, s + 1)

    def g(M, i, j):
        if abs(i) > s or abs(j) > s:
            return 0.0
        return M[i + s, j + s]

    H = np.zeros((N, N, N), dtype=np.complex128)
    for n in rng:
        for m in rng:
            for l in rng:
                v = 0.0
                for a in rng:
                    v += g(h1, n, a) * g(h2, m, l - a)
                    v += g(h1, n, a) * g(h12, m + a, l)
                    v += g(h2, m, a) * g(h12, n + a, l)
                    for b in rng:
                        v += rho0 * g(h1, n, a) * g(h2, m, b) * g(h12, a + b, l)
                H[n + s, m + s, l + s] = v
    return H


def term_1213_raw(h2m, h12, s):
    """Pairs (12.13) without Hermiticity: sum_nu h_{nu,-m}(-k2) h_{n-nu,l}(k12).

    h2m is the pair solution at the reflected wavevector -k2.  Equal to the
    reduced form sum_a h_{ma}(k2) h_{n+a,l}(k12) by reality + Hermiticity
    of the pair harmonics; used as a convention cross-check.
    """
    N = 2 * s + 1
    C = np.zeros((N, N, N), dtype=np.complex128)
    h2m_rev = h2m[:, ::-1]  # [v, m] = h_{v-s, -(m-s)}(-k2)
    for v in range(N):
        nu = v - s
        h12r = _row_shift(h12, -nu)  # [n, l] = h12[n - nu, l]
        C += h2m_rev[v, :][None, :, None] * h12r[:, None, :]
    return C


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------


def h_of(k, model, s, cache=None):
    """Pair harmonics h(k) from the 2-body Sylvester solve, with caching."""
    if cache is not None:
        key = (round(complex(k).real, 12), round(complex(k).imag, 12))
        if key in cache:
            return cache[key]
    h = compute_h(model.L(k), k, model)
    if cache is not None:
        cache[key] = h
    return h


def compare_point(k1, k2, model, s, cache=None):
    """(h3_hier, h3_ksa, doubles, triple) at one (k1, k2)."""
    h1 = h_of(k1, model, s, cache)
    h2 = h_of(k2, model, s, cache)
    h12 = h_of(k1 + k2, model, s, cache)
    doubles, triple = kirkwood_h3(h1, h2, h12, model.rho0, s)
    hier = compute_3bod(k1, k2, model, s=s)
    return hier, doubles + model.rho0 * triple, doubles, triple


def crop(T, s, c):
    sl = slice(s - c, s + c + 1)
    return T[sl, sl, sl]


def relerr(a, b):
    """||a - b||_F / ||b||_F."""
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ----------------------------------------------------------------------
# Stage 1: algebraic self-tests
# ----------------------------------------------------------------------


def stage_test():
    print("=== stage 1: self-tests ===")
    rng = np.random.default_rng(0)

    # 1. vectorized builder vs brute force, random matrices
    s = 3
    N = 2 * s + 1
    h1, h2, h12 = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)) for _ in range(3))
    rho0 = 0.7
    d, t = kirkwood_h3(h1, h2, h12, rho0, s)
    ref = kirkwood_h3_brute(h1, h2, h12, rho0, s)
    err = np.abs(d + rho0 * t - ref).max()
    print(f"builder vs brute force:                 max|diff| = {err:.2e}")
    assert err < 1e-12

    # 2. raw vs Hermiticity-reduced (12.13) term, with real solver h's
    s = 12
    model = abp_model(lp=10.0, phi=0.04, eps=1.0, V=Vexp, D=0.0, s=s)
    k1, k2 = 1.7, 2.3 * np.exp(1j * 2.1)
    cache = {}
    h2m = h_of(-k2, model, s, cache)
    h12 = h_of(k1 + k2, model, s, cache)
    h2 = h_of(k2, model, s, cache)
    C_raw = term_1213_raw(h2m, h12, s)
    N = 2 * s + 1
    C_red = np.zeros((N, N, N), dtype=np.complex128)
    for a in range(N):
        h12r = _row_shift(h12, a - s)
        C_red += h2[:, a][None, :, None] * h12r[:, None, :]
    err = np.abs(C_raw - C_red).max() / np.abs(C_red).max()
    print(f"(12.13) raw vs Hermiticity-reduced:     rel max diff = {err:.2e}")

    # 3. exchange symmetry 1 <-> 2 of both tensors
    hier12, ksa12, _, _ = compare_point(k1, k2, model, s, cache)
    hier21, ksa21, _, _ = compare_point(k2, k1, model, s, cache)
    e_h = np.abs(hier12 - hier21.transpose(1, 0, 2)).max() / np.abs(hier12).max()
    e_k = np.abs(ksa12 - ksa21.transpose(1, 0, 2)).max() / np.abs(ksa12).max()
    print(f"exchange symmetry hier / KSA:           {e_h:.2e} / {e_k:.2e}")

    # 4. mirror identity h3(conj k1, conj k2) = h3(k1, k2) reversed
    hier_c, ksa_c, _, _ = compare_point(np.conj(k1), np.conj(k2), model, s, cache)
    e_h = np.abs(hier_c - hier12[::-1, ::-1, ::-1]).max() / np.abs(hier12).max()
    e_k = np.abs(ksa_c - ksa12[::-1, ::-1, ::-1]).max() / np.abs(ksa12).max()
    print(f"mirror identity hier / KSA:             {e_h:.2e} / {e_k:.2e}")

    # 5. near-passive weak-coupling canary: the hierarchy h3 should reduce
    # to the KSA doubles (equilibrium low-density limit, triple ~ rho0)
    s_eq = 8
    model_eq = abp_model(lp=0.05, phi=0.005, eps=0.2, V=Vexp, D=1.0, s=s_eq)
    for kk1, kk2 in [(1.2, 0.8 * np.exp(2j * np.pi / 3)), (0.6, 1.5 * np.exp(0.9j))]:
        hier, ksa, _, _ = compare_point(kk1, kk2, model_eq, s_eq)
        c = 4
        e = relerr(crop(ksa, s_eq, c), crop(hier, s_eq, c))
        print(f"near-passive canary k1={kk1:.2f} |k2|={abs(kk2):.2f}: rel err = {e:.3e}")

    # 6. timing at production truncation
    s = 16
    model = abp_model(lp=10.0, phi=0.04, eps=1.0, V=Vexp, D=0.0, s=s)
    t0 = time.perf_counter()
    compare_point(2.0, 3.0 * np.exp(1j), model, s)
    print(f"one compare_point at s={s}: {time.perf_counter() - t0:.3f} s")


# ----------------------------------------------------------------------
# Stage 2: line scans
# ----------------------------------------------------------------------

GEOMETRIES = [
    ("collinear", 0.0),
    ("right angle", np.pi / 2),
    ("equilateral", 2 * np.pi / 3),
]

COMPONENTS = [(0, 0, 0), (1, 0, 1), (1, -1, 0), (1, 1, 2)]

S_SOLVE = 16
C_CROP = 6
K_SCAN = np.linspace(0.25, 12.0, 40)


def scan_config(model, s=S_SOLVE, c=C_CROP, k_scan=K_SCAN):
    """Scans |k1| = |k2| = k for the three geometries; returns cropped data."""
    nc = 2 * c + 1
    out = {}
    for name, beta in GEOMETRIES:
        cache = {}
        hier_c = np.zeros((len(k_scan), nc, nc, nc), dtype=np.complex128)
        ksa_c = np.zeros_like(hier_c)
        tri_frac = np.zeros(len(k_scan))
        for i, k in enumerate(k_scan):
            k1, k2 = k, k * np.exp(1j * beta)
            hier, ksa, doubles, triple = compare_point(k1, k2, model, s, cache)
            hier_c[i] = crop(hier, s, c)
            ksa_c[i] = crop(ksa, s, c)
            tri_frac[i] = (
                model.rho0
                * np.linalg.norm(crop(triple, s, c))
                / np.linalg.norm(ksa_c[i])
            )
        out[name] = (hier_c, ksa_c, tri_frac)
    return out


def stage_scan():
    print("=== stage 2: line scans ===")
    configs = [
        ("phi=0.04, eps=1", dict(lp=10.0, phi=0.04, eps=1.0), True),
        ("phi=0.2, eps=1", dict(lp=10.0, phi=0.2, eps=1.0), True),
        ("phi=0.04, eps=0.3", dict(lp=10.0, phi=0.04, eps=0.3), False),
        ("phi=0.04, eps=0.1", dict(lp=10.0, phi=0.04, eps=0.1), False),
    ]
    results = {}
    for label, pars, with_components in configs:
        t0 = time.perf_counter()
        model = abp_model(V=Vexp, D=0.0, s=S_SOLVE, **pars)
        results[label] = (scan_config(model), pars, with_components)
        print(f"scanned {label}: {time.perf_counter() - t0:.1f} s")

    # summary table
    print("\nrelative Frobenius error (|n|,|m|,|l| <= %d):" % C_CROP)
    print(f"{'config':<22}{'geometry':<14}{'median':>10}{'max':>10}{'max tri/KSA':>13}")
    summary = {}
    for label, (scans, pars, _) in results.items():
        for name, (hier_c, ksa_c, tri) in scans.items():
            re_k = np.array(
                [relerr(ksa_c[i], hier_c[i]) for i in range(len(K_SCAN))]
            )
            summary[(label, name)] = re_k
            print(
                f"{label:<22}{name:<14}{np.median(re_k):>10.3f}"
                f"{re_k.max():>10.3f}{tri.max():>13.3f}"
            )

    np.savez_compressed(
        "kirkwood_check_scans.npz",
        k_scan=K_SCAN,
        c_crop=C_CROP,
        s_solve=S_SOLVE,
        **{
            f"{label}|{name}|{arr}": data[idx]
            for label, (scans, pars, _) in results.items()
            for name, data in scans.items()
            for idx, arr in enumerate(("hier", "ksa", "trifrac"))
        },
    )
    print("saved kirkwood_check_scans.npz")
    plot_scans(results, summary)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREEN = "#009E73"
OI_VERM = "#D55E00"
OI_PURPLE = "#CC79A7"
GRAY = "#666666"


def _style(ax):
    ax.grid(True, lw=0.4, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_scans(results, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = C_CROP

    # component figures, one per config that carries them
    for label, (scans, pars, with_components) in results.items():
        if not with_components:
            continue
        fig, ax = plt.subplots(
            len(COMPONENTS),
            len(GEOMETRIES),
            figsize=(11, 11),
            sharex=True,
            layout="constrained",
        )
        for j, (name, _) in enumerate(GEOMETRIES):
            hier_c, ksa_c, _ = scans[name]
            for i, (n, m, l) in enumerate(COMPONENTS):
                hy = hier_c[:, n + c, m + c, l + c]
                ky = ksa_c[:, n + c, m + c, l + c]
                a = ax[i, j]
                a.plot(K_SCAN, hy.real, color=OI_BLUE, lw=1.8, label="hierarchy Re")
                a.plot(
                    K_SCAN, ky.real, "o", ms=3.5, mfc="none", color=OI_BLUE,
                    label="Kirkwood Re",
                )
                a.plot(
                    K_SCAN, hy.imag, color=OI_ORANGE, lw=1.8, label="hierarchy Im"
                )
                a.plot(
                    K_SCAN, ky.imag, "o", ms=3.5, mfc="none", color=OI_ORANGE,
                    label="Kirkwood Im",
                )
                a.axhline(0, color=GRAY, lw=0.6)
                _style(a)
                if i == 0:
                    a.set_title(name, fontsize=11)
                if j == 0:
                    a.set_ylabel(rf"$h^{{(3)}}_{{{n},{m},{l}}}$")
                if i == len(COMPONENTS) - 1:
                    a.set_xlabel(r"$k = |k_1| = |k_2|$")
        ax[0, 0].legend(fontsize=8, frameon=False)
        fig.suptitle(
            f"3-body hierarchy vs Kirkwood superposition -- ABP {label}"
            f" (lp={pars['lp']:g}, s={S_SOLVE})",
            fontsize=12,
        )
        tag = label.replace(", ", "_").replace("=", "")
        fig.savefig(f"kirkwood_vs_h3_components_{tag}.png", dpi=200)
        fig.savefig(f"kirkwood_vs_h3_components_{tag}.svg")
        plt.close(fig)
        print(f"saved kirkwood_vs_h3_components_{tag}.png/svg")

    # relative-error figure
    colors = [OI_BLUE, OI_VERM, OI_GREEN, OI_PURPLE]
    fig, ax = plt.subplots(
        1, len(GEOMETRIES), figsize=(11, 3.6), sharey=True, layout="constrained"
    )
    for j, (name, _) in enumerate(GEOMETRIES):
        for col, (label, _) in zip(colors, results.items()):
            ax[j].plot(K_SCAN, summary[(label, name)], color=col, lw=1.8, label=label)
        ax[j].set_yscale("log")
        ax[j].set_title(name, fontsize=11)
        ax[j].set_xlabel(r"$k = |k_1| = |k_2|$")
        _style(ax[j])
    ax[0].set_ylabel("rel. Frobenius error")
    ax[-1].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Kirkwood superposition vs hierarchy: relative error"
        rf" ($|n|,|m|,|l|\leq{C_CROP}$)",
        fontsize=12,
    )
    fig.savefig("kirkwood_vs_h3_relerr.png", dpi=200)
    fig.savefig("kirkwood_vs_h3_relerr.svg")
    plt.close(fig)
    print("saved kirkwood_vs_h3_relerr.png/svg")


# ----------------------------------------------------------------------
# Stage 3: 2D map over k2 at fixed k1
# ----------------------------------------------------------------------


def _map_point(k1, k2, model, s, c):
    hier, ksa, _, _ = compare_point(k1, k2, model, s)
    hc, kc = crop(hier, s, c), crop(ksa, s, c)
    return hier[s, s, s], ksa[s, s, s], relerr(kc, hc)


def stage_map():
    print("=== stage 3: k2 map at fixed k1 ===")
    from joblib import Parallel, delayed

    model = abp_model(lp=10.0, phi=0.04, eps=1.0, V=Vexp, D=0.0, s=S_SOLVE)
    s, c = S_SOLVE, C_CROP
    k1 = 2.0
    kx = np.linspace(-6.0, 6.0, 41)
    ky = np.linspace(-6.0, 6.0, 41)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX + 1j * KY
    mask = (np.abs(K2) > 0.2) & (np.abs(k1 + K2) > 0.2)
    pts = K2[mask]

    t0 = time.perf_counter()
    res = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_map_point)(k1, k2, model, s, c) for k2 in pts
    )
    print(f"map: {mask.sum()} points, {time.perf_counter() - t0:.1f} s")

    h000 = np.full(K2.shape, np.nan, dtype=np.complex128)
    k000 = np.full(K2.shape, np.nan, dtype=np.complex128)
    err = np.full(K2.shape, np.nan)
    h000[mask] = [r[0] for r in res]
    k000[mask] = [r[1] for r in res]
    err[mask] = [r[2] for r in res]

    np.savez_compressed(
        "kirkwood_check_map.npz", k1=k1, kx=kx, ky=ky, h000=h000, k000=k000, err=err
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    vmax = np.nanmax(np.abs(h000.real))
    norm = colors.SymLogNorm(
        linthresh=1e-3 * vmax, linscale=0.4, vmin=-vmax, vmax=vmax, base=10
    )
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), layout="constrained")
    for a, Z, title in zip(
        ax[:3],
        [h000.real, k000.real, h000.real - k000.real],
        [
            r"hierarchy $\mathrm{Re}\,h^{(3)}_{000}$",
            r"Kirkwood $\mathrm{Re}\,h^{(3)}_{000}$",
            "difference",
        ],
    ):
        pc = a.pcolormesh(KX, KY, Z, norm=norm, cmap="RdBu_r", rasterized=True)
        a.set_aspect("equal")
        a.set_title(title, fontsize=11)
        a.set_xlabel(r"$k_{2x}$")
        a.annotate(
            "", xy=(k1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5)
        )
    fig.colorbar(pc, ax=ax[:3], shrink=0.9)
    pe = ax[3].pcolormesh(
        KX, KY, err, norm=colors.LogNorm(vmin=1e-3, vmax=1.0), cmap="viridis",
        rasterized=True,
    )
    ax[3].set_aspect("equal")
    ax[3].set_title(rf"rel. error ($|n|,|m|,|l|\leq{c}$)", fontsize=11)
    ax[3].set_xlabel(r"$k_{2x}$")
    ax[0].set_ylabel(r"$k_{2y}$")
    fig.colorbar(pe, ax=ax[3], shrink=0.9)
    fig.suptitle(
        rf"ABP $\varphi=0.04$, $\ell_p=10$, $\varepsilon=1$:"
        rf" $h^{{(3)}}_{{000}}(k_1={k1:g}\,\hat x,\ k_2)$",
        fontsize=12,
    )
    fig.savefig("kirkwood_vs_h3_map.png", dpi=200)
    fig.savefig("kirkwood_vs_h3_map.svg")
    plt.close(fig)
    print("saved kirkwood_vs_h3_map.png/svg")


# ----------------------------------------------------------------------
# Stage 4: activity dependence and truncation check
# ----------------------------------------------------------------------


def stage_lp():
    """Median KSA error vs persistence length, at fixed D = 1 (passive limit
    lp -> 0 well defined), phi = 0.04, equilateral geometry."""
    print("=== stage 4: activity dependence ===")
    lps = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    k_scan = np.linspace(0.25, 12.0, 24)
    s, c = S_SOLVE, C_CROP
    beta = 2 * np.pi / 3
    curves = {}
    for eps in (1.0, 0.1):
        med = np.zeros(len(lps))
        for il, lp in enumerate(lps):
            model = abp_model(lp=lp, phi=0.04, eps=eps, V=Vexp, D=1.0, s=s)
            cache = {}
            errs = []
            for k in k_scan:
                hier, ksa, _, _ = compare_point(k, k * np.exp(1j * beta), model, s, cache)
                errs.append(relerr(crop(ksa, s, c), crop(hier, s, c)))
            med[il] = np.median(errs)
            print(f"eps={eps:g} lp={lp:g}: median rel err = {med[il]:.4f}")
        curves[eps] = med

    np.savez_compressed("kirkwood_check_lp.npz", lps=lps, **{
        f"eps{eps:g}": med for eps, med in curves.items()
    })

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
    for col, (eps, med) in zip((OI_BLUE, OI_VERM), curves.items()):
        ax.loglog(lps, med, "o-", color=col, lw=1.8, ms=5, label=rf"$\varepsilon={eps:g}$")
    ax.set_xlabel(r"persistence length $\ell_p$  ($D=1$, $\varphi=0.04$)")
    ax.set_ylabel("median rel. error")
    ax.set_title("Kirkwood error vs activity (equilateral)", fontsize=11)
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig("kirkwood_vs_h3_lp.png", dpi=200)
    fig.savefig("kirkwood_vs_h3_lp.svg")
    plt.close(fig)
    print("saved kirkwood_vs_h3_lp.png/svg")


def stage_strunc():
    """Truncation sanity: does the error at s = 16 persist at s = 24?"""
    print("=== stage 5: truncation check ===")
    for k in (1.0, 3.0, 6.0):
        k1, k2 = k, k * np.exp(2j * np.pi / 3)
        for s in (12, 16, 24):
            model = abp_model(lp=10.0, phi=0.04, eps=1.0, V=Vexp, D=0.0, s=s)
            hier, ksa, _, _ = compare_point(k1, k2, model, s)
            e = relerr(crop(ksa, s, C_CROP), crop(hier, s, C_CROP))
            print(f"k={k:g}, s={s}: rel err = {e:.4f}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("test", "all"):
        stage_test()
    if what in ("scan", "all"):
        stage_scan()
    if what in ("map", "all"):
        stage_map()
    if what in ("lp", "all"):
        stage_lp()
    if what in ("strunc", "all"):
        stage_strunc()

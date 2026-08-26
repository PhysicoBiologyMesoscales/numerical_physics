import numpy as np
import matplotlib.pyplot as plt

list_Pe = [2, 10, 100, 1000, "inf"]


fig, ax = plt.subplots(1, 1)

# for(i, Pe) in enumerate(list_Pe):
#     print(f"Pe={Pe}")
#     g = np.load(f"hard_sphere_abp_lowphi_Pe{Pe}.npz")
#     a = g["alpha"]
#     b = g["beta"]
#     r = g["r"]
#     gb = g["g_bulk"]

#     if Pe == "inf":
#         label = r"$Pe=\infty$"
#     else:
#         label = f"Pe={Pe}"
#     ax.plot(a, gb[1:5, :, 1:36].mean(axis=-1).mean(axis=0), label=label)
#     ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
#     ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
# ax.legend()
# fig.savefig("gb_alpha.svg", dpi=300, bbox_inches="tight")
# fig, ax = plt.subplots(
#     1, len(list_Pe), figsize=(15, 3), sharey=True, layout="constrained"
# )

# for i, Pe in enumerate(list_Pe):
#     print(f"Pe={Pe}")
#     g = np.load(f"hard_sphere_abp_lowphi_Pe{Pe}.npz")
#     ca = g["cos_alpha"]
#     sb = g["sin_beta"]
#     gs = g["g_surf"]

#     cc, ss = np.meshgrid(ca, sb)
#     msh = ax[i].pcolormesh(
#         cc, ss, gs, shading="auto", vmin=0.0, vmax=1.1, cmap="inferno", rasterized=True
#     )
#     ax[i].set_xticks([-1, 0, 1])
#     ax[i].set_yticks([-1, 0, 1])
#     ax[i].set_title(f"Pe={Pe}")

# fig.colorbar(msh, ax=ax, orientation="vertical", label=r"$g_{surf}$", ticks=[0, 0.5, 1])
# fig.savefig("gs_Pe.svg", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(
    1,
    len(list_Pe),
    figsize=(15, 3),
    sharey=True,
    layout="constrained",
    subplot_kw={"projection": "polar"},
)

for i, Pe in enumerate(list_Pe):
    print(f"Pe={Pe}")
    g = np.load(f"hard_sphere_abp_lowphi_Pe{Pe}.npz")
    r = g["r"]
    a = g["alpha"]
    gb = g["g_bulk"][..., 50:].mean(axis=-1)

    rr, aa = np.meshgrid(r, a)
    msh = ax[i].pcolormesh(
        aa, rr, gb.T, shading="auto", vmin=0.0, vmax=2, cmap="plasma", rasterized=True
    )
    ax[i].set_title(f"Pe={Pe}")
    ax[i].set_xticks([])
    ax[i].set_yticks([0])
    ax[i].set_yticklabels([])

fig.colorbar(msh, ax=ax, orientation="vertical", label=r"$g_{bulk}$", ticks=[0, 1, 2])
# fig.savefig("gb_Pe.svg", dpi=300, bbox_inches="tight")

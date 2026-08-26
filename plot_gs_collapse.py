import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

list_Pe = [2, 10, 20, 100, 200, 1000, 2000, 10000]
n_Pe = len(list_Pe)

## Plot gs raw data
fig, ax = plt.subplots(1, 1)
for i, Pe in enumerate(list_Pe):
    print(f"Pe={Pe}")
    g = np.load(f"hard_sphere_abp_lowphi_Pe{Pe}.npz")
    ca = g["cos_alpha"]
    sb = g["sin_beta"]
    gs = g["g_surf"]
    for j in range(60, 72):
        x = sb
        y = gs[:, j]
        # y = -(1 - y)*(1 + ca[j])/2
        ax.scatter(
            x,
            y,
            color=(i / float(n_Pe), i / float(n_Pe), i / float(n_Pe)),
            label=f"Pe={Pe}",
        )

ax.set_xlim(-0.25)
fig.savefig("gs_raw.svg")

## Plot the gs collapse
fig, ax = plt.subplots(1, 1)
arr = np.zeros((0, 2))
handles = []
labels = [f"Pe={Pe}" for Pe in list_Pe]

for i, Pe in enumerate(list_Pe):
    print(f"Pe={Pe}")
    g = np.load(f"hard_sphere_abp_lowphi_Pe{Pe}.npz")
    ca = g["cos_alpha"]
    sb = g["sin_beta"]
    gs = g["g_surf"]
    scat_plots = []
    for j in range(60, 72):
        x = sb * (4 * Pe) ** (1.0 / 3.0)
        y = np.where(sb > 0, 1 - (1 - gs[:, j]), 0)
        # y = -(1 - y)*(1 + ca[j])/2
        arr = np.concatenate((arr, np.stack((x, y), axis=-1)))
        scat = ax.scatter(
            x,
            y,
            color=(i / float(n_Pe), i / float(n_Pe), i / float(n_Pe)),
            label=f"Pe={Pe}",
        )
        scat_plots.append(scat)
    scat_plots = tuple(scat_plots)
    handles.append(scat_plots)

from scipy.special import airy

x = np.linspace(0, 15)
y = 1 - airy(x)[0] / airy(0)[0]
(line_plot,) = plt.plot(x, y, c="red")
ax.set_xlim(-2, 10)
handles.append((line_plot,))
labels.append(rf"y = 1-Ai(x)/Ai(0)")
ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)})

x_sim = arr[:, 0]
_idx = np.argwhere(x_sim > 0).flatten()
x_sim = x_sim[_idx]
y_sim = arr[_idx, 1]
y_pred = 1 - airy(x_sim)[0] / airy(0)[0]

fig.savefig("gs_collapse.svg")

R2 = 1 - ((y_sim - y_pred) ** 2).sum() / ((y_sim - y_sim.mean()) ** 2).sum()
R2

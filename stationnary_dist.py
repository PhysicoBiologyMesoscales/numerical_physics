import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib

matplotlib.use("TkAgg")

p_possible = np.linspace(0, 1, 1000)
Nk, Nh = 80, 50
K = np.concatenate(
    [np.linspace(0, 0.4, 10), np.linspace(0.4, 1.5, Nk - 20), np.linspace(1.5, 5, 10)]
)
H = np.linspace(0, 5, Nh)


th = np.linspace(0, 2 * np.pi, 100)[:, *(np.newaxis,) * 3]
_p = p_possible[np.newaxis, :, *(np.newaxis,) * 2]
_K = K[*(np.newaxis,) * 2, :, np.newaxis]
_H = H[*(np.newaxis,) * 3, :]

num_vals = np.trapz(
    np.cos(th) * np.exp(_H * np.cos(2 * th) + 2 * _K * _p * np.cos(th)),
    th[:, 0, 0, 0],
    axis=0,
)
den_vals = np.trapz(
    np.exp(_H * np.cos(2 * th) + 2 * _K * _p * np.cos(th)), th[:, 0, 0, 0], axis=0
)

volume = num_vals / den_vals - p_possible[:, *(np.newaxis,) * 2]

matches = np.argwhere(np.diff(np.sign(volume), axis=0) == -2)

p = np.zeros((Nk, Nh))
p[matches[:, 1], matches[:, 2]] = p_possible[matches[:, 0]]

X, Y = np.meshgrid(K, H)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X.T, Y.T, p, cmap=cm.coolwarm, linewidth=0, antialiased=True)
ax.set_xlabel("K")
ax.set_ylabel("H")
ax.set_zlabel("p")

plt.show()

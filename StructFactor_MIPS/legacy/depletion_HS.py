import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"})

r = np.linspace(1.01, 1.5, 101)
phi = np.linspace(-np.pi, np.pi, 261, endpoint=False)
rr, phiphi = np.meshgrid(r, phi)

ac = np.arccos(1 / rr)
asi = np.arcsin(1 / rr)
jac = 1 / np.sqrt(rr**2 - 1.0)

# jac = 0
B = np.where(
    np.abs(phiphi) > np.pi - ac,
    2 / np.pi * (jac - np.abs(asi)),
    np.where(np.abs(phiphi) > ac, jac / np.pi + ac / np.pi - np.abs(phiphi) / np.pi, 0),
)
msh = ax.pcolormesh(phiphi, rr, B)
fig.colorbar(msh, ax=ax)

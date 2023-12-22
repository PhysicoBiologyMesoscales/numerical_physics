import numpy as np
import matplotlib.pyplot as plt

Lx = 1
Ly = 4
Nx = 10
Ny = 40

alpha = 2.0
h = 3.0

theta = np.random.random(size=(Ny, Nx)) * 2 * np.pi
px = np.cos(theta)
py = np.sin(theta)

dx = Lx / Nx
dy = Ly / Ny

dt = 0.01
N_t = 10000

it = 0

plt.figure()
while it < N_t:
    p = np.sqrt(px**2 + py**2)
    dx_p = 1 / dx / 2 * (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))
    dy_p = 1 / dy / 2 * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))
    px += dt * (-dx_p + (alpha * (1 - 1 / h) * p - 1) * px)
    py += dt * (-1 / h * dx_p - (alpha * (1 - 1 / h) * p + 1) * px)
    it += 1
    plt.imshow(px)
    plt.pause(0.05)

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

N = 1000  # Number of sites
eps = 1e-16  # Noise initial amplitude
p = eps * (1 - 2 * np.random.random((N, 1)))  # Initial spins

N_t = 10000
dt = 0.01

alpha = 0.01
beta = 10

plt.figure()

for i in range(N_t):
    p_copy = p.copy()
    idx_sat = np.argwhere(abs(p) >= 1)
    p += dt * (alpha * p + beta * (np.roll(p, -1) + np.roll(p, 1))) * (1 - abs(p))
    # p[idx_sat] = p_copy[idx_sat]
    if i % 10 == 0:
        plt.clf()
        plt.imshow(p, aspect="auto")
        plt.show()
        plt.pause(0.02)

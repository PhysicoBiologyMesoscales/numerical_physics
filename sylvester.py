import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester
from scipy.special import jv

s = 50
N = 2 * s + 1


def V(k):
    return np.exp(-0.5 * k**2)


Pe = 0.1
phi = 1.0
eps = 1e-2


def L(k):
    L = np.diag((np.arange(N) - s) ** 2) * 2 / Pe - 1j * k / 2 * (
        np.eye(N, k=1) + np.eye(N, k=-1)
    )
    L[s, s] += 2 * phi / eps * V(k)
    return L


D = 4 * np.diag((np.arange(N) - s) ** 2) / Pe

Npoints = 100

k_arr = np.linspace(0, 10, Npoints)
g0 = np.zeros(Npoints, dtype=np.complex128)
g1 = np.zeros(Npoints, dtype=np.complex128)

X = np.zeros((N, N, Npoints), dtype=np.complex128)

for i, k in enumerate(k_arr):
    X[..., i] = solve_sylvester(L(k), L(k).conj().T, D)
    g0[i] = X[s, s, i]
    g1[i] = X[s, s + 1, i]

plt.plot(k_arr, g0)
plt.plot(k_arr, np.abs(g1))

r_arr = np.linspace(0, 5.0, 120)
kr = k_arr[:, None] * r_arr[None, :]


# X_r = np.trapz(
#     k_arr[None, None, :, None]
#     * jv(0, kr)[None, None, ...]
#     * (X[..., None] - np.eye(N)[..., None, None]),
#     k_arr,
#     axis=2,
# )

g0_r = np.trapz(k_arr[:, None] * jv(0, kr) * (g0[:, None] - 1), k_arr, axis=0)
g1_r = np.trapz(k_arr[:, None] * jv(1, kr) * g1[:, None], k_arr, axis=0)

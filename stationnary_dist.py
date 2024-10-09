import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(0, 100, 10000)


def trunc_series(z, n, n_max=30):
    if n == n_max:
        return 1 + z / n / (n + 1)
    else:
        return 1 + z / n / (n + 1) / trunc_series(z, n + 1, n_max)


y = trunc_series(z, 1, n_max=100)

Nk = 100

k_arr = np.linspace(1, 10, Nk)

kz = np.stack([k_arr, np.zeros(Nk)], axis=-1)

for i, k in enumerate(k_arr):
    kz[i, 1] = z[np.argwhere(np.diff(np.sign(y - k))).flatten()[0]]

p2 = kz[:, 1] / kz[:, 0] ** 2
plt.plot(k_arr, np.sqrt(p2))
plt.axhline(1, color="orange", linestyle="--")
plt.xlabel("k")
plt.ylabel("p")
plt.xlim((0, 10))
plt.xticks(np.linspace(0, 10, 11))
plt.ylim((0, 1.05))

# for i in range(5,30):
#     plt.plot(z, trunc_series(z, 1, n_max=i))

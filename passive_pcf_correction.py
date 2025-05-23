import numpy as np
from scipy import integrate


def v(k, v0, l):
    return v0 * np.exp(-0.5 * (l * k) ** 2)


def integrand(phi, k, q, v0, l):
    qq, kk, phi_ = np.meshgrid(q, k, phi, indexing="ij")
    q_dot_k = qq * kk * np.cos(phi_)
    qmk = np.sqrt(qq**2 + kk**2 - 2 * q_dot_k)
    vk = v(kk, v0, l)
    vq = v(qq, v0, l)
    vqmk = v(qmk, v0, l)

    num1 = q_dot_k * vk + (qq**2 - q_dot_k) * vqmk
    num2 = (
        2 * (q_dot_k - kk * 2) / (1 + vq)
        - q_dot_k / (1 + vqmk)
        - (qq * 2 - q_dot_k) / (1 + vk)
    )
    den = (
        qq**2 * (1 + vq) * (qq * 2 * (1 + vq) + kk**2 * (1 + vk) + qmk**2 * (1 + vqmk))
    )

    return np.where(den == 0, 0, num1 * num2 / den)


def compute_I_of_q(q, v0, l):
    # 1) define the φ–integral as a function of k
    def inner_phi(k):
        # do a 100-point Gauss-Legendre over [0,2π]
        val, _ = integrate.fixed_quad(
            integrand, 0, 2 * np.pi, args=(k, q, v0, l), n=100
        )
        return val

    # 2) outer integral ∫₀^∞ k * [inner_phi(k)] dk
    result, error = integrate.fixed_quad(lambda kk: kk * inner_phi(kk), 0, 10, n=100)
    return result, error


q = np.linspace(0, 12, 100)
v0 = 20
l = 5.0

S0 = 1 / (1 + v(q, v0, l))
I = compute_I_of_q(q, v0, l)[0]

import matplotlib.pyplot as plt

plt.plot(q, S0)
plt.plot(q, S0 - I)

# k, phi = np.linspace(0, 5, 100), np.linspace(0, 2 * np.pi, 50)
# integ = integrand(phi, k, 10, v0, l)[0]

# import matplotlib

# matplotlib.use("TkAgg")
# kk, pp = np.meshgrid(k, phi, indexing="ij")

# X, Y = kk * np.cos(pp), kk * np.sin(pp)

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.plot_surface(X, Y, integ, cmap=plt.cm.YlGnBu_r)

# plt.show()

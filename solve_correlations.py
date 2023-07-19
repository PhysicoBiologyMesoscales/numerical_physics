import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.special import iv

import numpy.typing as npt
from typing import Union

from tqdm import tqdm

plt.ion()

N_points = 50
d_theta = 2 * np.pi / N_points
theta = np.linspace(d_theta / 2, 2 * np.pi - d_theta / 2, N_points)
sintheta_minus = np.roll(np.sin(2 * theta), 1)
sintheta_plus = np.roll(np.sin(2 * theta), -1)


def RHS(t: float, p: npt.NDArray[np.float_], alpha: float) -> npt.NDArray[np.float_]:
    """
    Given current time t and values of probabilities on the [0,2*pi] ring, returns the RHS of the system with spatial derivatives
    """
    d_sinp = (sintheta_plus * np.roll(p, -1) - sintheta_minus * np.roll(p, 1)) / (
        2 * d_theta
    )
    lap = (np.roll(p, 1) + np.roll(p, -1) - 2 * p) / d_theta**2
    return alpha * d_sinp + lap


def compute_first_fourier_coefficients(p: np.ndarray):
    a1 = 1 / np.pi * d_theta * (p * np.cos(theta)[None, :, None]).sum(axis=1)
    b1 = 1 / np.pi * d_theta * (p * np.sin(theta)[None, :, None]).sum(axis=1)
    return a1, b1


def p_stat(theta, alpha):
    return 1 / 2 / np.pi / iv(0, alpha / 2) * np.exp(np.cos(2 * theta) * alpha / 2)


def fourier_th(theta0, alpha, t):
    return (
        np.exp(-t)
        * (
            np.cos(theta0)
            * (1 + alpha * t)
            # - alpha * np.cos(3 * theta0) * (1 - np.exp(-8 * np.pi * t)) / 8
        )
        / np.pi
    )


# def diracs(x):
#     p = np.diag(np.ones(len(x)))/d_theta
#     return p

t0 = 0.0
tf = 5

alpha = 0.5

N_t = 200
t_eval = np.linspace(t0, tf, N_t)

initial_conditions = np.diag(np.ones(len(theta))) / d_theta

sols = np.zeros((N_points, N_points, N_t))

# sol = solve_ivp(RHS, t_span=(t0, tf), t_eval=t_eval, y0=initial_conditions[10], args=(alpha,))

for i, p0 in enumerate(tqdm(initial_conditions)):
    sols[i, :, :] = solve_ivp(
        RHS, t_span=(t0, tf), t_eval=t_eval, y0=p0, args=(alpha,)
    )["y"]

a1, b1 = compute_first_fourier_coefficients(sols)
corr = (
    np.pi
    * d_theta
    * np.sum(
        p_stat(theta, alpha)[:, None]
        * (np.cos(theta)[:, None] * a1 + np.sin(theta)[:, None] * b1),
        axis=0,
    )
)

# plt.figure()
# for i, t in enumerate(t_eval):
#     plt.clf()
#     plt.polar(theta, sols[0, :, i])
#     plt.polar(theta, sols[25, :, i])
#     plt.pause(0.05)

plt.plot(t_eval, corr)
plt.plot(t_eval, np.exp(-t_eval))
plt.plot(
    t_eval,
    np.exp(-t_eval)
    * (
        1
        + alpha
        * (t_eval - (1 - np.exp(-8 * t_eval)) / 8)
        * iv(1, alpha / 2)
        / iv(0, alpha / 2)
    ),
)
# plt.plot(t_eval, np.exp(-t_eval)*(np.cosh(alpha*t_eval)+np.sinh(alpha*t_eval)*iv(1, alpha/2)/iv(0, alpha/2)))
plt.plot(
    t_eval,
    np.exp(-t_eval)
    * (
        1
        + alpha
        * (t_eval - (1 - np.exp(-8 * t_eval)) / 8)
        * iv(1, alpha / 2)
        / iv(0, alpha / 2)
        + alpha**2 * ((1 - np.exp(-8 * t_eval)) / 64 - t_eval / 8 + t_eval**2 / 2)
    ),
)
# plt.figure()
# plt.plot(t_eval, a1[25,:])
# plt.plot(t_eval, fourier_th(theta[25], 0, t_eval))
# plt.plot(t_eval, fourier_th(theta[25], alpha, t_eval))

# plt.figure()
# plt.plot(t_eval, a1[12,:])
# plt.plot(t_eval, fourier_th(theta[12], 0, t_eval))
# plt.plot(t_eval, fourier_th(theta[12], alpha, t_eval))

# plt.plot(t_eval, 1 - 2 * np.exp(-alpha) * t_eval)
# plt.semilogy()

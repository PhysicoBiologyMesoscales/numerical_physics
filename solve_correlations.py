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
theta = np.linspace(-np.pi, np.pi, N_points)
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
    p = 1 / 2 / np.pi / iv(0, alpha) * np.exp(np.cos(2 * theta) * alpha)
    return p / p.sum() / d_theta


# def diracs(x):
#     p = np.diag(np.ones(len(x)))/d_theta
#     return p

t0 = 0.0
tf = 10.0

alpha = 10

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

plt.plot(t_eval, corr)
# plt.plot(t_eval, np.exp(-t_eval))
plt.plot(t_eval, 1 - 2 * np.exp(-alpha) * t_eval)
# plt.semilogy()

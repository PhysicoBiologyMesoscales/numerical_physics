import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.special import iv

import numpy.typing as npt
from typing import Union

plt.ion()

R = 0.1  # Ratio between rotationnal diffusion and alignment strength
N_points = 500
d_theta = 2 * np.pi / N_points
theta = np.linspace(-np.pi, np.pi, N_points)
sintheta_minus = np.roll(np.sin(2 * theta), 1)
sintheta_plus = np.roll(np.sin(2 * theta), -1)

p_stat = np.exp(np.cos(2 * theta) / 2 / R) / (2 * np.pi * iv(0, 1 / 2 / R))


def RHS(t: float, p: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Given current time t and values of probabilities on the [0,2*pi] ring, returns the RHS of the system with spatial derivatives
    """
    d_sinp = (sintheta_plus * np.roll(p, -1) - sintheta_minus * np.roll(p, 1)) / (
        2 * d_theta
    )
    lap = (np.roll(p, 1) + np.roll(p, -1) - 2 * p) / d_theta**2
    return d_sinp + R * lap


def find_bounds_idx(x: float, arr: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
    index_sorted = np.argsort(
        abs(arr - x)
    )  # Arrays with the indexes of arr values sorted by proximity
    return index_sorted[:2]


def sol_interpolated(sol: dict, t: float):
    if t > sol["t"].max():
        raise (ValueError("The solution has not be calculated for the time t"))
    # Interpolate between values of the solutions to approximate solution at time t
    t_bounds_idx = find_bounds_idx(t, sol["t"])
    t_bounds = sol["t"][t_bounds_idx]
    p_bounds = sol["y"][:, t_bounds_idx]
    x = (t - t_bounds[0]) / (
        t_bounds[1] - t_bounds[0]
    )  # interpolation coordinate (in the range[0,1])
    return (1 - x) * p_bounds[:, 0] + x * p_bounds[:, 1]


def compute_fourier_coefficient(p: np.ndarray, n: Union[int, npt.NDArray[np.int_]]):
    if isinstance(n, int):
        order_0_correction = 1 - (n == 0) * 0.5
    else:
        order_0_correction = np.ones(n.shape) - (n == 0) * 0.5
    return (
        2
        / np.pi
        * np.einsum(
            "j...,ij,i->...i",
            p,
            np.cos(np.outer(n, theta)) * d_theta,
            order_0_correction,
        )
    )


def gaussian(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(((x % (np.pi) - mu) / sigma) ** 2))


def mean_squared_diff(t, y):
    tol = 2e-5
    return np.mean((y - p_stat) ** 2) > tol


mean_squared_diff.terminal = True

theta_0 = 0
p_0 = gaussian(theta, theta_0, 0.1) + gaussian(np.pi - theta, theta_0, 0.1)
p_0 /= p_0.sum() * d_theta

sol = solve_ivp(RHS, t_span=(0, 20), y0=p_0, events=mean_squared_diff)

plt.figure(0)
plt.plot(theta, sol["y"].T[-1], label=f"Computed solution (t = {sol['t'][-1]:.3})")
plt.plot(theta, p_stat, label="Stationnary solution (analytical)")
plt.legend()

p = sol["y"].T[-1]

n_arr = np.arange(-20, 21, 2)
plt.figure(1)
plt.scatter(n_arr, compute_fourier_coefficient(p_0, n_arr))
plt.scatter(n_arr, compute_fourier_coefficient(p, n_arr))

# visualize_step = 1
# plt.figure(2)
# for t, y in zip(sol["t"][::visualize_step], sol["y"].T[::visualize_step]):
#     plt.clf()
#     plt.polar(theta, y)
#     plt.title(f"t={t}")
#     plt.pause(0.03)

n_arr = np.arange(0, 21, 2)
fourier_coeff = compute_fourier_coefficient(sol["y"], n_arr)

plt.figure(3)
plt.plot(sol["t"], fourier_coeff, label=[rf"$p_{{{i//2}}}$" for i in n_arr])
plt.legend(loc="lower right", ncol=3)
plt.tight_layout()


# t_arr = np.linspace(0, 3, 3)
# for t in t_arr:
#     p = sol_interpolated(sol, t)

#     n_arr = np.arange(-20, 21, 2)
#     l_fourier = []
#     for n in n_arr:
#         l_fourier.append(compute_fourier_coefficient(p, n))
#     plt.scatter(n_arr, l_fourier)

# ax = plt.gca()
# _ = ax.set_ylim([-2 / np.pi - 0.1, 2 / np.pi + 0.1])
# yticks = np.arange(-2 / np.pi, 2 / np.pi + 0.1, 1 / (np.pi))
# n_ticks = len(yticks)
# _ = ax.set_yticks(yticks)
# labels = [
#     rf"${n}\cdot$" + r"$\frac{1}{\pi}$"
#     for n in range(-(n_ticks // 2), n_ticks // 2 + 1)
# ]
# _ = ax.set_yticklabels(labels)
# ax.grid(linestyle="-.")

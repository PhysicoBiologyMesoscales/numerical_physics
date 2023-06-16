import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


R = 0.1  # Ratio between rotationnal diffusion and alignment strength
N_points = 500
d_theta = 2 * np.pi / N_points
theta = np.linspace(-np.pi, np.pi, N_points)
sintheta_minus = np.roll(np.sin(2 * theta), 1)
sintheta_plus = np.roll(np.sin(2 * theta), -1)


def RHS(t: float, p: np.ndarray):
    """
    Given current time t and values of probabilities on the [0,2*pi] ring, returns the RHS of the system with spatial derivatives
    """
    d_sinp = (sintheta_plus * np.roll(p, -1) - sintheta_minus * np.roll(p, 1)) / (
        2 * d_theta
    )
    lap = (np.roll(p, 1) + np.roll(p, -1) - 2 * p) / d_theta**2
    return d_sinp + R * lap


def find_bounds_idx(x: float, arr: np.ndarray):
    index_sorted = np.argsort(
        abs(arr - x)
    )  # Arrays with the indexes of arr values sorted by proximity
    return index_sorted[:2]


def sol_interpolated(sol: dict, t: float):
    if t > sol["t"].max():
        raise (ValueError("The solution has not be calculated for the time t"))
    # Interpolate between values of the solutions to approximate solution at time t
    t_bounds = find_bounds_idx(t, sol["t"])
    p_bounds = sol["y"][:, t_bounds]
    x = (t - t_bounds[0]) / (
        t_bounds[1] - t_bounds[0]
    )  # interpolation coordinate (in the range[0,1])
    return (1 - x) * p_bounds[:, 0] + x * p_bounds[:, 1]


def compute_fourier_coefficient(p: np.ndarray, n: int):
    if p.shape != theta.shape:
        raise (ValueError("p and theta must have the same dimensions"))
    if n == 0:
        return 1 / np.pi * np.sum(p * d_theta)
    return 2 / np.pi * np.sum(p * np.cos(n * theta) * d_theta)


def gaussian(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(((x % (np.pi) - mu) / sigma) ** 2))


theta_0 = np.pi / 3
p_0 = gaussian(theta, theta_0, 0.1) + gaussian(np.pi - theta, theta_0, 0.1)
p_0 /= p_0.sum() * d_theta

sol = solve_ivp(RHS, t_span=(0, 3), y0=p_0)

t_arr = np.linspace(0, 3, 3)
for t in t_arr:
    p = sol_interpolated(sol, t)

    n_arr = np.arange(-20, 21, 2)
    l_fourier = []
    for n in n_arr:
        l_fourier.append(compute_fourier_coefficient(p, n))
    plt.scatter(n_arr, l_fourier)

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

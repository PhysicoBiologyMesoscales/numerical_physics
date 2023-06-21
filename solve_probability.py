import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.special import iv

import numpy.typing as npt
from typing import Union

plt.ion()

N_points = 500
d_theta = 2 * np.pi / N_points
theta = np.linspace(-np.pi, np.pi, N_points)
sintheta_minus = np.roll(np.sin(2 * theta), 1)
sintheta_plus = np.roll(np.sin(2 * theta), -1)


def RHS(
    t: float, p: npt.NDArray[np.float_], R: float, y_t: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
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


def mean_squared_diff(t, y, R, y_t):
    tol = 1e-6
    return np.mean((y - y_t) ** 2) > tol


mean_squared_diff.terminal = True

theta_0 = 0
p_0 = gaussian(theta, theta_0, 0.1) + gaussian(np.pi - theta, theta_0, 0.1)
p_0 /= p_0.sum() * d_theta

R_list = [2.0, 0.1, 0.0008]
alpha_list = [1 / 2 / R for R in R_list]

fig, axes = plt.subplots(nrows=len(R_list), ncols=3, figsize=(19, 15))

for i, R in enumerate(R_list):
    print(f"R={R}")
    p_stat = np.exp(np.cos(2 * theta) / 2 / R) / (2 * np.pi * iv(0, 1 / 2 / R))
    sol = solve_ivp(
        RHS, t_span=(0, 10), y0=p_0, events=mean_squared_diff, args=(R, p_stat)
    )

    ax1 = axes[i, 0]
    ax1.plot(theta, sol["y"].T[-1], label=f"Computed solution (t = {sol['t'][-1]:.3})")
    ax1.plot(theta, p_stat, label="Stationnary solution (analytical)")
    ax1.legend(loc="upper center")

    ax2 = axes[i, 1]

    n_arr = np.arange(-20, 21, 2)
    ax2.scatter(n_arr, compute_fourier_coefficient(p_0, n_arr), label="t=0")
    ax2.scatter(
        n_arr,
        compute_fourier_coefficient(sol["y"][:, -1], n_arr),
        label=f"t={sol['t'][-1]:.3}",
    )
    ax2.set_xticks(ticks=n_arr, labels=n_arr // 2)
    ax2.legend()

    ax3 = axes[i, 2]
    n_arr = np.arange(0, 11, 2)
    fourier_coeff = compute_fourier_coefficient(sol["y"], n_arr)
    plot_until_time = 2
    last_index = find_bounds_idx(plot_until_time, sol["t"])[1]
    lines = ax3.plot(
        sol["t"][1:last_index],
        fourier_coeff[1:last_index, 1:],
        label=[rf"$p_{{{i//2}}}$" for i in n_arr[1:]],
    )
    [
        ax3.axhline(
            2 / np.pi * iv(n // 2, 1 / 2 / R) / iv(0, 1 / 2 / R),
            linestyle="-.",
            color=lines[i].get_color(),
        )
        for (i, n) in enumerate(n_arr[1:])
    ]
    ax3.legend(ncol=3, prop={"size": 10})

cols = [r"$P(\theta)$", r"Fourier coefficients", r"$p_n(t)$"]
rows = [rf"$\alpha={alpha:.1E}$" for alpha in alpha_list]

for ax, col in zip(axes[0], cols):
    ax.set_title(col)

for ax, row in zip(axes[:, 0], rows):
    ax.set_ylabel(row, size="large")

fig.tight_layout()

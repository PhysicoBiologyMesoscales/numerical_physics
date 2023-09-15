import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from itertools import product


# Define your system of ODEs
def system(t, y, alpha):
    return -alpha * np.cos(y).mean() * np.sin(y)


# Set up the time span and initial conditions
t_span = (0, 200)
initial_conditions_list = product(np.linspace(0, np.pi, 20), np.linspace(0, np.pi, 20))

# Solve the system of ODEs
for initial_conditions in initial_conditions_list:
    sol = solve_ivp(
        system, t_span, initial_conditions, args=(1,), t_eval=np.linspace(0, 20, 1000)
    )
    x_values = sol.y[0]
    y_values = sol.y[1]
    plt.quiver(
        x_values[:-1],
        y_values[:-1],
        x_values[1:] - x_values[:-1],
        y_values[1:] - y_values[:-1],
        scale_units="xy",
        angles="xy",
        scale=1,
        label=f"IC={initial_conditions}",
    )

plt.plot(np.linspace(0, np.pi / 2, 100), np.pi - np.linspace(0, np.pi / 2, 100))

# plt.plot(x_values, y_values, label='Phase Portrait', color='b')
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.title("Phase Portrait of Coupled ODEs")
plt.grid(True)
# plt.legend()
plt.show()

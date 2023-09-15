import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from itertools import product
from mpl_toolkits.mplot3d import Axes3D

plt.ion()


# Define your system of ODEs
def system(t, y, alpha):
    return -alpha * np.cos(y).mean() * np.sin(y)


# Set up the time span and initial conditions
t_span = (0, 200)
initial_conditions_list = []
for th1, th2, th3 in product(np.linspace(0.1, np.pi - 0.1, 10), repeat=3):
    if np.cos(th1) <= 0.5 * (np.cos(th2) + np.cos(th3)) or np.cos(th3) >= 0.5 * (
        np.cos(th1) + np.cos(th2)
    ):
        continue
    initial_conditions_list.append([th1, th2, th3])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Solve the system of ODEs
for initial_conditions in initial_conditions_list:
    sol = solve_ivp(
        system, t_span, initial_conditions, args=(1,), t_eval=np.linspace(0, 20, 1000)
    )
    th1_values = sol.y[0]
    th2_values = sol.y[1]
    th3_values = sol.y[2]
    lines = ax.plot(th1_values, th2_values, th3_values)

ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.set_zlabel(r"$\theta_3$")

plt.show()
plt.pause(0)

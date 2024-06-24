import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, 2 * np.pi, 100)
phi, theta = np.meshgrid(phi, theta)
R = 1
r = 0.3


def toroidal_harmonic(phi, theta, l, m):
    return (
        r * np.cos(l * phi) * np.cos(m * theta),
        r * np.sin(l * phi) * np.cos(m * theta),
        r * np.cos(l * phi) * np.sin(m * theta),
        r * np.sin(l * phi) * np.sin(m * theta),
    )


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

harm = toroidal_harmonic(phi, theta, 2, 1)[1]
X = (R + harm * np.cos(theta)) * np.cos(phi)
Y = (R + harm * np.cos(theta)) * np.sin(phi)
Z = harm * np.sin(theta)
my_col = cm.jet(np.sign(harm), min=-1, max=1)

ax.axes.set_zlim3d(bottom=-2 * r, top=2 * r)
ax.plot_surface(X, Y, Z, facecolors=my_col, antialiased=False)

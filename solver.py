# %matplotlib

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lg
import matplotlib.pyplot as plt
from os.path import join

plt.ion()
np.seterr(all="raise")


def BuildLAPP():
    """
    Laplacian with Periodic BC on x and y

    """

    NXi = nx
    NYi = ny

    ###### 1D Laplace operator

    ###### X-axis
    ### Diagonal terms
    dataNXi = [
        np.ones(NXi),
        np.ones(NXi),
        -2 * np.ones(NXi),
        np.ones(NXi),
        np.ones(NXi),
    ]

    ###### Their positions
    offsetsX = np.array([-NXi + 1, -1, 0, 1, NXi - 1])
    DXX = sp.dia_matrix((dataNXi, offsetsX), shape=(NXi, NXi)) / dx**2

    ###### Y-axis
    ### Diagonal terms
    dataNYi = [
        np.ones(NYi),
        np.ones(NYi),
        -2 * np.ones(NYi),
        np.ones(NYi),
        np.ones(NYi),
    ]

    ###### Their positions
    offsetsY = np.array([-NYi + 1, -1, 0, 1, NYi - 1])
    DYY = sp.dia_matrix((dataNYi, offsetsY), shape=(NYi, NYi)) / dy**2

    # Build 2D Laplacian
    LAPP = sp.kron(DXX, sp.eye(NYi, NYi)) + sp.kron(sp.eye(NXi, NXi), DYY)

    return LAPP


def BuildID():
    """
    Laplacian with Periodic BC on x and y

    """

    NXi = nx
    NYi = ny

    return sp.kron(sp.eye(NXi, NXi), sp.eye(NYi, NYi))


def LUdecomposition(LAP):
    """
    return the Incomplete LU decomposition
    of a sparse matrix LAP
    """
    return lg.splu(
        LAP.tocsc(),
    )


def Resolve(splu, RHS):
    """
    solve the system

    SPLU * x = RHS

    Args:
    --RHS: 2D array((NY,NX))
    --splu: (Incomplete) LU decomposed matrix
            shape (NY*NX, NY*NX)

    Return: x = array[NY,NX]

    Rem1: taille matrice fonction des CL

    """
    # array 2D -> array 1D
    f2 = RHS.ravel()

    # Solving the linear system
    x = splu.solve(f2)

    return x.reshape(RHS.shape)


def Semilag(u, v, q):
    """
    1st order semi-Lagrangian advection
    Adapted for periodic BC
    """
    ADVq = np.zeros((NX, NY))

    # Matrices where 1 is right, 0 is left or center
    Mx2 = np.sign(np.sign(u) + 1.0)
    Mx1 = 1.0 - Mx2

    # Matrices where 1 is up, 0 is down or center
    My2 = np.sign(np.sign(v) + 1.0)
    My1 = 1.0 - My2

    # Matrices of absolute values for u and v
    au = abs(u)
    av = abs(v)

    # Matrices of coefficients respectively central, external, same x, same y
    Cc = (dx - au * dt) * (dy - av * dt) / dx / dy
    Ce = dt * dt * au * av / dx / dy
    Cmx = (dx - au * dt) * av * dt / dx / dy
    Cmy = dt * au * (dy - dt * av) / dx / dy

    # Computes the advected quantity
    ADVq = (
        Cc * q
        + Ce
        * (
            Mx1 * My1 * np.roll(q, (-1, -1), axis=(0, 1))
            + Mx2 * My1 * np.roll(q, (1, -1), axis=(0, 1))
            + Mx1 * My2 * np.roll(q, (-1, 1), axis=(0, 1))
            + Mx2 * My2 * np.roll(q, (1, 1), axis=(0, 1))
        )
        + Cmx * (My1 * np.roll(q, -1, axis=1) + My2 * np.roll(q, 1, axis=1))
        + Cmy * (Mx1 * np.roll(q, -1, axis=0) + Mx2 * np.roll(q, 1, axis=0))
    )

    return ADVq


def Semilag2(u, v, q):
    """
    Second order semi-Lagrangian advection
    """

    qstar = Semilag(u, v, q)
    qtilde = Semilag(-u, -v, qstar)
    qstar = q + (q - qtilde) / 2
    ADVq = Semilag(u, v, qstar)

    return ADVq


def divergence(u, v):
    """
    divergence of vectorial field (u,v) with ghost points
    never use the boundary values

    """
    div = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / dx / 2 + (
        np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)
    ) / dy / 2

    return div


def vorticity(u, v):
    """
    Antisymmetric part of the velocity gradient tensor
    """

    w = 0.5 * (
        (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / dx / 2
        - (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / dy / 2
    )

    return w


def laplacien(x):
    """
    Laplacien of scalar field x(i,j)

    """

    lap = (np.roll(x, -1, axis=0) + np.roll(x, 1, axis=0) - 2 * x) / dx**2 + (
        np.roll(x, -1, axis=1) + np.roll(x, 1, axis=1) - 2 * x
    ) / dy**2

    return lap


# Dimensions
asp_r = 2
LX = 63.0
LY = asp_r * LX

# Number of points (including ghost points)
NX = int(100)
NY = int(asp_r * NX)
# Number of points in real domain (change if BC are modified)
nx = NX
ny = NY
# Grid spacing
dx = LX / NX
dy = LY / NY
x = np.linspace(0, LX, NX)
y = np.linspace(0, LY, NY)
xx, yy = np.meshgrid(x, y)

### Initialize fields
# Velocity
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))

# Polarity
px = 1e-3 * (2 * np.random.random((nx, ny)) - 1)
py = 1e-3 * (2 * np.random.random((nx, ny)) - 1)

# Density
phi = np.ones((nx, ny), dtype=float)

# px[: nx // 2] = 0.1
# py = np.zeros((nx, ny))

# theta_0 = np.pi / 3
# px = np.ones((NX, NY)) * np.cos(theta_0)
# py = np.ones((NX, NY)) * np.sin(theta_0)


### Simulation parameters
# Physical parameters
v0 = 1.6  # Particle velocity
K = 2.0  # Torque intensity
H = 5.0  # Nematic field intensity
epsilon = 0.001  # Dimensionless particle size
Re = 0.0001
k0 = 5

dt = 0.05

LAPP = BuildLAPP()
ID = BuildID()
LUPP = LUdecomposition(LAPP)

t = 0.0

fig = plt.figure(figsize=(5, 5 * asp_r))


i = 0


try:
    while t < 10000:
        t += dt
        ## Compute velocity
        ustar = v0 * px
        vstar = v0 * py
        # Compute divergence
        divstar = divergence(ustar, vstar)
        # Leray projector to get pressure
        phi = Resolve(LUPP, RHS=divstar)
        # Compute pressure gradient
        gradphix = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / dx / 2
        gradphiy = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / dy / 2
        # Compute velocity
        u = ustar - gradphix
        v = vstar - gradphiy
        ## Update polarity
        pxa = Semilag2(u, v, px)
        pya = Semilag2(u, v, py)
        # pxa = px
        # pya = py

        px = pxa + dt * (
            (H / 2 * (1 - pxa**2 + 3 * pya**2) + K * v0 / 2 * (1 - px**2 - py**2) - 1)
            * pxa
            + (
                (1 - (pxa**2 - pya**2))
                / 2
                * (K * v0 / 8 * laplacien(pxa) - k0 * gradphix)
                - pxa * pya * (K * v0 / 8 * laplacien(pya) - k0 * gradphiy)
            )
        )
        py = pya + dt * (
            (-H / 2 * (1 - pya**2 + 3 * pxa**2) + K * v0 / 2 * (1 - px**2 - py**2) - 1)
            * pya
            + (
                -pxa * pya * (K * v0 / 8 * laplacien(pxa) - k0 * gradphix)
                + (1 - (pxa**2 - pya**2))
                / 2
                * (K * v0 / 8 * laplacien(pya) - k0 * gradphiy)
            )
        )

        # Check CFL condition
        while np.max(np.sqrt(u**2 + v**2)) > 0.9 * dx / dt:
            dt *= 0.9

        x_pl = xx[::10, ::10]
        y_pl = yy[::10, ::10]
        u_pl = u[::10, ::10]
        v_pl = v[::10, ::10]

        if t // 0.2 > i:
            plt.clf()
            plt.pcolormesh(xx, yy, px.T, vmin=-1.0, vmax=1.0)
            # plt.quiver(x_pl, y_pl, u_pl, v_pl)
            plt.colorbar()
            plt.title(f"t={t:.2f}; dt={dt:.3f}")
            plt.pause(0.1)
            # plt.savefig(
            #     join(
            #         r"C:\Users\nolan\Documents\PhD\Simulations\Data\Group_Meeting_0603\Num_solve_k0=5_with_adv",
            #         f"t={i}.png",
            #     )
            # )
            i += 1

except KeyboardInterrupt:
    pass

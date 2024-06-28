import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from os.path import join
from sim_analysis import (
    compute_coarsegrain_matrix,
    coarsegrain_field,
    compute_distribution,
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from scipy.ndimage import sobel

data_folder = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\Gradient_oblique"
df_full = pd.read_csv(join(data_folder, "Data.csv"), index_col=0)

with open(join(data_folder, "parms.json")) as json_file:
    parm_dic = json.load(json_file)

N = parm_dic["N"]
asp = parm_dic["aspect_ratio"]
l = np.sqrt(N * np.pi / asp / parm_dic["phi"])
L = asp * l

Nx, Nth = 20, 10
Ny = int(Nx * asp)
dx, dy = l / Nx, L / Ny


def coarsegrain_df(df):
    r = np.stack([df["x"], df["y"]], axis=-1)
    F = np.stack([df["Fx"], df["Fy"]], axis=-1)
    e = np.stack([np.cos(df["theta"]), np.sin(df["theta"])], axis=-1)

    C = compute_coarsegrain_matrix(r, df["theta"], l, L, Nx, Ny, Nth, N)

    psi = compute_distribution(C, Nx, Ny, Nth)

    F_cg = coarsegrain_field(F, C, Nx, Ny, Nth)
    e_cg = coarsegrain_field(e, C, Nx, Ny, Nth)

    rho = psi.sum(axis=2) * 2 * np.pi / Nth

    grad_rho_x = sobel(rho, axis=1, mode="wrap") / 8 / dx
    grad_rho_y = sobel(rho, axis=0, mode="wrap") / 8 / dy

    # # grad_rho_x = (np.roll(rho, -1, axis=0) - np.roll(rho, 1, axis=0)) / 2 / dx
    # # grad_rho_y = (np.roll(rho, 1, axis=1) - np.roll(rho, -1, axis=1)) / 2 / dy

    F_avg = (
        F_cg.sum(axis=2) * 2 * np.pi / Nth / np.where(rho == 0, 1, rho)[..., np.newaxis]
    )
    Fx = F_avg[:, :, 0]
    Fy = F_avg[:, :, 1]
    p = e_cg.sum(axis=2) * 2 * np.pi / Nth / np.where(rho == 0, 1, rho)[..., np.newaxis]
    px, py = p[:, :, 0], p[:, :, 1]
    data = np.stack(
        [
            rho.reshape(Nx * Ny),
            grad_rho_x.reshape(Nx * Ny),
            grad_rho_y.reshape(Nx * Ny),
            px.reshape(Nx * Ny),
            py.reshape(Nx * Ny),
            Fx.reshape(Nx * Ny),
            Fy.reshape(Nx * Ny),
        ],
        axis=-1,
    )
    cols = ["rho", "grad_rho_x", "grad_rho_y", "px", "py", "Fx", "Fy"]
    index = np.arange(Nx * Ny)
    return pd.DataFrame(data, columns=cols, index=index)


df_cg = df_full.groupby("t").apply(coarsegrain_df)

timepoints = df_cg.index.unique(level="t")

df_fit = df_cg
df_test = df_cg.loc[timepoints[10]]


def set_data(df):
    X = np.stack(
        [
            np.concatenate([df["grad_rho_x"], df["grad_rho_y"]]),
            np.concatenate([df["rho"] * df["px"], df["rho"] * df["py"]]),
        ],
        axis=-1,
    )
    y = np.concatenate([df["Fx"], df["Fy"]])
    return X, y


X_fit, y_fit = set_data(df_fit)
X_test, y_test = set_data(df_test)

reg = LinearRegression()
reg.fit(X_fit, y_fit)


def get_field(df, field):
    return np.array(df[field]).reshape((Ny, Nx))


rho = get_field(df_test, "rho")
Fx_true = get_field(df_test, "Fx")
Fy_true = get_field(df_test, "Fy")

y_pred = reg.predict(X_test)
F_pred = y_pred.reshape((Ny * Nx, 2), order="F").reshape((Ny, Nx, 2))

plt.imshow(rho, origin="lower")
plt.quiver(Fx_true, Fy_true)
plt.quiver(
    F_pred.reshape((Ny, Nx, 2))[:, :, 0],
    F_pred.reshape((Ny, Nx, 2))[:, :, 1],
    color=(1, 0, 0),
)

reg.score(X_test, y_test)


def test_regression(sim_folder_path, timepoint, reg):
    df = pd.read_csv(join(sim_folder_path, "Data.csv"), index_col=0)
    df_t = df[df["t"] == pd.unique(df["t"])[timepoint]]
    df_cg = coarsegrain_df(df_t)
    Xtest, ytest = set_data(df_cg)
    rho = get_field(df_cg, "rho")
    Fx_true = get_field(df_cg, "Fx")
    Fy_true = get_field(df_cg, "Fy")
    Fpred = reg.predict(Xtest)
    return rho, Fx_true, Fy_true, Fpred, Xtest, ytest


# grad_rho = np.stack([grad_rho_x, grad_rho_y], axis=-1)


# # X = grad_rho.reshape((Nx*Ny*2,1))
# y = F_avg.reshape(Nx * Ny * 2)
# reg = LinearRegression().fit(X, y)
# F_pred = reg.predict(X).reshape(Ny, Nx, 2)
# plt.figure(figsize=(10, asp * 10))
# plt.imshow(rho, origin="lower")
# plt.quiver(F_avg[:, :, 0], F_avg[:, :, 1])
# plt.quiver(F_pred[:, :, 0], F_pred[:, :, 1], color=(1, 0, 0))

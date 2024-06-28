import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse
import scipy.sparse as sp


from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coarse-graining of active particle simulation data"
    )
    parser.add_argument(
        "sim_folder_path", help="Path to folder containing simulation data", type=str
    )
    parser.add_argument(
        "nx", help="Number of discretization points on x axis", type=int
    )
    parser.add_argument(
        "nth", help="Number of discretization points on theta", type=int
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sim_path = args.sim_folder_path
    # Load full sim data and parameters
    full_data = pd.read_csv(join(sim_path, "Data.csv"), index_col=0)
    with open(join(sim_path, "parms.json")) as json_file:
        parm_dic = json.load(json_file)
    N = parm_dic["N"]
    asp = parm_dic["aspect_ratio"]
    l = np.sqrt(N * np.pi / asp / parm_dic["phi"])
    L = asp * l
    # Load discretization parameters; y discretization is fixed by Nx and aspect ratio to get square tiles
    Nx = args.nx
    Ny = int(Nx * asp)
    Nth = args.nth

    def compute_coarsegrain_matrix(r, theta):
        dx = l / Nx
        dy = L / Ny
        dth = 2 * np.pi / Nth
        # We compute forces in (x, y, theta) space
        r_th = np.concatenate([r, np.array([theta % (2 * np.pi)]).T], axis=-1)
        # Build matrix and 1D encoding in (x, y, theta) space
        Cijk = (r_th // np.array([dx, dy, dth])).astype(int)
        C1d_cg = np.ravel_multi_index(Cijk.T, (Nx, Ny, Nth), order="C")
        C = sp.eye(Nx * Ny * Nth, format="csr")[C1d_cg] / N
        return C

    def compute_distribution(C):
        # Compute one-body distribution
        return np.array(C.T.sum(axis=1)).reshape((Nx, Ny, Nth)).swapaxes(0, 1)

    def coarsegrain_field(field, C):
        if field.ndim == 1:
            _field = np.array([field]).T
        else:
            _field = field
        data_ndims = field.shape[-1]
        field_cg = C.T @ _field
        return field_cg.reshape((Nx, Ny, Nth, data_ndims)).swapaxes(0, 1)

    def coarsegrain_df(df: pd.DataFrame) -> pd.DataFrame:
        r = np.stack([df["x"], df["y"]], axis=-1)
        F = np.stack([df["Fx"], df["Fy"]], axis=-1)

        C = compute_coarsegrain_matrix(r, df["theta"])

        psi = compute_distribution(C)

        F_cg = coarsegrain_field(F, C)

        data = np.stack(
            [
                psi.reshape(Nx * Ny * Nth),
                F_cg[:, :, :, 0].reshape(Nx * Ny * Nth),
                F_cg[:, :, :, 1].reshape(Nx * Ny * Nth),
            ],
            axis=-1,
        )

        index = pd.MultiIndex.from_product(
            [np.arange(Nx), np.arange(Ny), np.arange(Nth)], names=["x", "y", "theta"]
        )

        cols = ["psi", "Fx", "Fy"]

        return pd.DataFrame(data, index=index, columns=cols)

    cg_data = full_data.groupby("t").apply(coarsegrain_df)
    cg_data.to_csv(join(sim_path, "cg_data.csv"))
    with open(join(sim_path, "cg_parms.json"), "w") as jsonFile:
        json.dump({"l": l, "L": L, "Nx": Nx, "Ny": Ny, "Nth": Nth}, jsonFile)


if __name__ == "__main__":
    import sys
    from unittest.mock import patch

    sim_path = (
        r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\temp"
    )
    args = ["prog", sim_path, "20", "20"]
    with patch.object(sys, "argv", args):
        main()


#         rho = psi.sum(axis=2) * 2 * np.pi / Nth

#         grad_rho_x = sobel(rho, axis=1, mode="wrap") / 8 / dx
#         grad_rho_y = sobel(rho, axis=0, mode="wrap") / 8 / dy

#         # # grad_rho_x = (np.roll(rho, -1, axis=0) - np.roll(rho, 1, axis=0)) / 2 / dx
#         # # grad_rho_y = (np.roll(rho, 1, axis=1) - np.roll(rho, -1, axis=1)) / 2 / dy

#         F_avg = (
#             F_cg.sum(axis=2)
#             * 2
#             * np.pi
#             / Nth
#             / np.where(rho == 0, 1, rho)[..., np.newaxis]
#         )
#         Fx = F_avg[:, :, 0]
#         Fy = F_avg[:, :, 1]
#         p = (
#             e_cg.sum(axis=2)
#             * 2
#             * np.pi
#             / Nth
#             / np.where(rho == 0, 1, rho)[..., np.newaxis]
#         )
#         px, py = p[:, :, 0], p[:, :, 1]
#         data = np.stack(
#             [
#                 rho.reshape(Nx * Ny),
#                 grad_rho_x.reshape(Nx * Ny),
#                 grad_rho_y.reshape(Nx * Ny),
#                 px.reshape(Nx * Ny),
#                 py.reshape(Nx * Ny),
#                 Fx.reshape(Nx * Ny),
#                 Fy.reshape(Nx * Ny),
#             ],
#             axis=-1,
#         )
#         cols = ["rho", "grad_rho_x", "grad_rho_y", "px", "py", "Fx", "Fy"]
#         index = np.arange(Nx * Ny)
#         return pd.DataFrame(data, columns=cols, index=index)


# data_folder = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\Gradient_oblique"
# df_full = pd.read_csv(join(data_folder, "Data.csv"), index_col=0)

# with open(join(data_folder, "parms.json")) as json_file:
#     parm_dic = json.load(json_file)

# N = parm_dic["N"]
# asp = parm_dic["aspect_ratio"]
# l = np.sqrt(N * np.pi / asp / parm_dic["phi"])
# L = asp * l

# Nx, Nth = 20, 10
# Ny = int(Nx * asp)
# dx, dy = l / Nx, L / Ny


# df_cg = df_full.groupby("t").apply(coarsegrain_df)

# timepoints = df_cg.index.unique(level="t")

# df_fit = df_cg
# df_test = df_cg.loc[timepoints[10]]


# def set_data(df):
#     X = np.stack(
#         [
#             np.concatenate([df["grad_rho_x"], df["grad_rho_y"]]),
#             np.concatenate([df["rho"] * df["px"], df["rho"] * df["py"]]),
#         ],
#         axis=-1,
#     )
#     y = np.concatenate([df["Fx"], df["Fy"]])
#     return X, y


# X_fit, y_fit = set_data(df_fit)
# X_test, y_test = set_data(df_test)

# reg = LinearRegression()
# reg.fit(X_fit, y_fit)


# def get_field(df, field):
#     return np.array(df[field]).reshape((Ny, Nx))


# rho = get_field(df_test, "rho")
# Fx_true = get_field(df_test, "Fx")
# Fy_true = get_field(df_test, "Fy")

# y_pred = reg.predict(X_test)
# F_pred = y_pred.reshape((Ny * Nx, 2), order="F").reshape((Ny, Nx, 2))

# plt.imshow(rho, origin="lower")
# plt.quiver(Fx_true, Fy_true)
# plt.quiver(
#     F_pred.reshape((Ny, Nx, 2))[:, :, 0],
#     F_pred.reshape((Ny, Nx, 2))[:, :, 1],
#     color=(1, 0, 0),
# )

# reg.score(X_test, y_test)


# def test_regression(sim_folder_path, timepoint, reg):
#     df = pd.read_csv(join(sim_folder_path, "Data.csv"), index_col=0)
#     df_t = df[df["t"] == pd.unique(df["t"])[timepoint]]
#     df_cg = coarsegrain_df(df_t)
#     Xtest, ytest = set_data(df_cg)
#     rho = get_field(df_cg, "rho")
#     Fx_true = get_field(df_cg, "Fx")
#     Fy_true = get_field(df_cg, "Fy")
#     Fpred = reg.predict(Xtest)
#     return rho, Fx_true, Fy_true, Fpred, Xtest, ytest


# # grad_rho = np.stack([grad_rho_x, grad_rho_y], axis=-1)


# # # X = grad_rho.reshape((Nx*Ny*2,1))
# # y = F_avg.reshape(Nx * Ny * 2)
# # reg = LinearRegression().fit(X, y)
# # F_pred = reg.predict(X).reshape(Ny, Nx, 2)
# # plt.figure(figsize=(10, asp * 10))
# # plt.imshow(rho, origin="lower")
# # plt.quiver(F_avg[:, :, 0], F_avg[:, :, 1])
# # plt.quiver(F_pred[:, :, 0], F_pred[:, :, 1], color=(1, 0, 0))

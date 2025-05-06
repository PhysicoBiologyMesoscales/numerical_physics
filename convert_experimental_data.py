import pickle
import h5py
import pandas as pd
import numpy as np

from os.path import join

sim_path = r"E:\Experiments\2025\Test_Paule_data"

with open(join(sim_path, "trajectories.pkl"), "rb") as f:
    pkl_data = pickle.load(f)

# pkl_data.to_hdf(
#     join(sim_path, "data.h5py"),
#     key="data",
#     mode="w",
#     format="fixed",
#     data_columns=["frame", "particle", "x", "y"],
# )

grouped_data = pkl_data.groupby(["frame"])

t = pd.unique(pkl_data["frame"])
Nt = len(t)

Nmax = int(pkl_data["particle"].max()) + 1
r = np.ones((Nt, Nmax), dtype=np.complex128) * np.nan
theta = np.ones((Nt, Nmax)) * np.nan
N = np.zeros(Nt)
# TODO temp, should be stored in the experimental data
l = np.max(pkl_data["x"])
L = np.max(pkl_data["y"])


for frame, ds in grouped_data:
    N[frame] = len(ds)
    r[frame, ds["particle"]] = ds["x"] + 1j * ds["y"]
    theta[frame, ds["particle"]] = np.where(
        ds["angle_cyto_nuclei"] < 0,
        2 * np.pi + ds["angle_cyto_nuclei"],
        ds["angle_cyto_nuclei"],
    )


hdf_file = h5py.File(join(sim_path, "data.h5py"), "w")
hdf_file.attrs.create("l", l)
hdf_file.attrs.create("L", L)
hdf_file.attrs.create("Nt", Nt)


hdf_data = hdf_file.create_group("exp_data")

hdf_data.create_dataset("N", data=N)
hdf_data.create_dataset("t", data=t)
hdf_data.create_dataset("r", data=r)
hdf_data.create_dataset("theta", data=theta)

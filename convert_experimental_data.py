import pickle
import h5py
import pandas as pd
import numpy as np
import argparse

from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualization of pair-correlation function"
    )
    parser.add_argument(
        "exp_folder_path", help="Path to folder containing experimental data", type=str
    )
    return parser.parse_args()


def convert(ds, out_path):
    """
    Convert experimental data stored in DataSet ds into hdf datasets using h5py
    """
    ## Create attributes matching experimental data
    t = pd.unique(ds["frame"])
    Nt = len(t)
    Nmax = int(ds["particle"].max()) + 1
    # FOV dimensions
    # TODO temp, should be stored in experimental data
    l = np.max(ds["x"])
    L = np.max(ds["y"])

    # Position array
    r = np.ones((Nt, Nmax), dtype=np.complex128) * np.nan
    # Orientation array
    theta = np.ones((Nt, Nmax)) * np.nan
    # Number of segmented celles for each frame
    N = np.zeros(Nt)

    ## Loop over all frames and store data
    grouped_data = ds.groupby(["frame"])
    for frame, ds in grouped_data:
        N[frame] = len(ds)
        r[frame, ds["particle"]] = ds["x"] + 1j * ds["y"]
        # Ensure angles are in [0, 2*Pi]
        theta[frame, ds["particle"]] = ds["angle_cyto_nuclei"] % (2 * np.pi)

    ## Create h5py file and write into it
    hdf_file = h5py.File(out_path, "w")
    hdf_file.attrs.create("l", l)
    hdf_file.attrs.create("L", L)
    hdf_file.attrs.create("asp", L / l)
    hdf_file.attrs.create("Nt", Nt)
    # Create datasets storing experimental values
    hdf_data = hdf_file.create_group("exp_data")
    hdf_data.attrs.create("Nmax", Nmax)
    hdf_data.create_dataset("N", data=N)
    hdf_data.create_dataset("t", data=t)
    hdf_data.create_dataset("r", data=r)
    hdf_data.create_dataset("theta", data=theta)

    hdf_file.flush()
    hdf_file.close()


def main():
    parms = parse_args()
    sim_path = parms.sim_folder_path
    # ! This assumes the experimental data is named trajectories.pkl
    with open(join(sim_path, "trajectories.pkl"), "rb") as f:
        # TODO could be improved by delaying load if dataset is very large
        pkl_data = pickle.load(f)
    convert(pkl_data, join(sim_path, "data.h5py"))

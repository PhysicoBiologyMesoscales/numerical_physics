import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import matplotlib.cm as cm
from tqdm import tqdm
from numpy.random import default_rng
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute radial correlation function for particle simulation results"
    )
    parser.add_argument("data_path", help="Path to the data folder", type=str)
    parser.add_argument(
        "--N_r",
        help="Number of distance points where correlations are computed",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--N_sample",
        help="Number of sample particules to use to compute correlations",
        type=int,
        default=1000,
    )
    return parser.parse_args()


def main():
    parms = parse_args()
    with open(join(parms.data_path, "parms.json")) as jsonFile:
        sim_parms = json.load(jsonFile)

    print("Loading data...")

    # Load data
    df = pd.read_csv(
        join(parms.data_path, "Data.csv"), sep=",", header=0, index_col=0, dtype=float
    )
    df.index = df.index.astype(int)

    # Compute polarity
    df["px"] = np.cos(df["theta"])
    df["py"] = np.sin(df["theta"])

    # Select only one time point to compute correlations
    df_corr = df[df["t"] == df["t"].max()]

    aspectRatio = sim_parms["aspect_ratio"]
    N = sim_parms["N"]
    phi = sim_parms["phi"]
    l = np.sqrt(N * np.pi / aspectRatio / phi)
    L = aspectRatio * l

    def get_distances(row, **kwargs):
        """Compute distances from the particle represented by row with all other particles"""
        x_arr = kwargs["x_arr"]
        y_arr = kwargs["y_arr"]
        size_x = kwargs["size_x"]
        size_y = kwargs["size_y"]
        x_0, y_0 = row["x"], row["y"]
        x_diff = abs(x_arr - x_0)
        x_diff = np.where(x_diff > 0.5 * size_x, size_x - x_diff, x_diff)
        y_diff = abs(y_arr - y_0)
        y_diff = np.where(y_diff > 0.5 * size_y, size_y - y_diff, y_diff)
        return np.sqrt(x_diff**2 + y_diff**2)

    print("Computing distances...")

    dist_matrix_df = df_corr.apply(
        get_distances,
        x_arr=df_corr["x"],
        y_arr=df_corr["y"],
        size_x=L,
        size_y=L,
        axis=1,
        result_type="expand",
    )

    # Set all distances on the diagonal to nan to avoid overcounting
    def set_self_to_nan(row):
        row[row.name] = np.nan
        return row

    dist_matrix_df.apply(set_self_to_nan)

    # Divide the domain in 'onion slices' of size dr
    dr = L / np.sqrt(2) / parms.N_r
    df_modulo = dist_matrix_df // dr

    def compute_polarity_correlations(index, **kwargs):
        df = kwargs["data"]
        # Reference position
        px = df["px"].iloc[index.name]
        py = df["py"].iloc[index.name]
        pol_df = df.iloc[index[0]][["px", "py"]]
        pol_df["corr"] = pol_df["px"] * px + pol_df["py"] * py
        return [len(pol_df["corr"]), pol_df["corr"].sum()]

    possible_distances = np.arange(0, np.sqrt(l**2 + L**2) / 2 // dr, 1)
    corr_arr = np.vstack([possible_distances * dr, np.zeros(len(possible_distances))])

    # Select a random sample of particles to compute the correlations
    rng = default_rng()
    particle_selection = rng.choice(N, size=parms.N_sample, replace=False)

    print("Computing correlations...")
    for i, dist in enumerate(tqdm(possible_distances)):
        df_pair = pd.DataFrame(df_modulo.apply(lambda x: x.index[x == dist], axis=1))
        count_and_sum = (
            df_pair.iloc[particle_selection]
            .apply(
                compute_polarity_correlations,
                axis=1,
                result_type="expand",
                data=df_corr,
            )
            .sum(axis=0)
        )
        corr_arr[1, i] = count_and_sum[1] / count_and_sum[0]

    # Save correlations to disk
    np.save(
        join(
            parms.data_path,
            "correlations.npy",
        ),
        corr_arr,
    )
    print("Done !")


if __name__ == "__main__":
    main()

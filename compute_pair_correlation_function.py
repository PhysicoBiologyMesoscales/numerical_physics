import numpy as np
import xarray as xr
import pandas as pd

from os.path import join

sim_path = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Compute_forces\Batch\_temp"
sim_data = pd.read_csv(join(sim_path, "Data.csv"), index_col=["p_id", "t"]).to_xarray()
cg_data = xr.open_dataset(join(sim_path, "cg_data.nc"))


interp_data = sim_data.assign(
    psi=cg_data.psi.interp(
        x=sim_data.x, y=sim_data.y, theta=sim_data.theta, t=sim_data.t
    )
    .reset_coords(["x", "y", "theta"])
    .psi
)

Nx, Ny, Nth = 20, 30, 20

# TODO : interpolate values

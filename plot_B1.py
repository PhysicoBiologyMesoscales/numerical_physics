import matplotlib.pyplot as plt
import numpy as np
import h5py

from os.path import join

sim_path = r"E:\Local_Sims\ABP_no_alignment_low_density_velocity"

sim_file = h5py.File(join(sim_path, "data.h5py"))

pcf = sim_file["pair_correlation"]["pcf"][()]
r = sim_file["pair_correlation"]["r"][()]
theta = sim_file["pair_correlation"]["theta"][()]

plt.plot(r, (pcf * np.cos(theta)[None, :, None, None]).mean(axis=(1, 2, 3)))

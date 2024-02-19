import numpy as np
from scipy.optimize import curve_fit
import os
import json
import matplotlib.pyplot as plt


def f(x, a, b, l):
    return a * np.exp(-x / l) + b


data_path = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Circle_Meeting\Batch"
os.chdir(data_path)

v_arr = np.linspace(1.6, 2.1, 10)
k_arr = np.linspace(5, 8, 10)
xx, yy = np.meshgrid(v_arr, k_arr)
l_arr = np.zeros(xx.shape)

for folder in os.listdir():
    with open(os.path.join(folder, "parms.json")) as jsonFile:
        parms = json.load(jsonFile)
    v0 = parms["v0"]
    k = parms["k"]
    phi = parms["phi"]
    N = parms["N"]
    # Frame aspect ratio
    aspectRatio = parms["aspect_ratio"]
    # Frame width
    l_syst = np.sqrt(N * np.pi / aspectRatio / phi)
    L_syst = aspectRatio * l_syst

    if not "correlations.npy" in os.listdir(folder):
        continue
    corr_arr = np.load(os.path.join(folder, "correlations.npy"))
    _1, _2, l = curve_fit(f, corr_arr[0, :], corr_arr[1, :])[0]

    if l > np.sqrt(l_syst**2 + L_syst**2) / 2:
        l = np.sqrt(l_syst**2 + L_syst**2) / 2

    l_arr[*np.argwhere(np.logical_and(xx == v0, yy == k))[0]] = l

plt.pcolormesh(k_arr, v_arr, l_arr.T)
plt.plot(k_arr, 10 / k_arr, color="red", linestyle="--", linewidth=2)

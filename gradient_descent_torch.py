import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py

from os.path import join
from torch.optim.lr_scheduler import ExponentialLR

sim_path = r"E:\Local_Sims\test_new_code"

with h5py.File(join(sim_path, "data.h5py")) as hdf_file:
    H = hdf_file.attrs["h"]
    K = hdf_file.attrs["k"] * hdf_file.attrs["phi"] / np.pi
    kc = hdf_file.attrs["kc"]
    rho = hdf_file.attrs["N"] / hdf_file.attrs["l"] / hdf_file.attrs["L"]
    corr = hdf_file["pair_correlation"]
    Nth = corr.attrs["Nth"]
    r = corr["r"][()]
    phi = corr["phi"][()]
    theta = corr["theta"][()]
    dth = corr.attrs["dth"]
    dphi = corr.attrs["dphi"]
    pcf = corr["pcf"][()]
    p_th = corr["p_th"][-1]

i_indices = np.arange(Nth).reshape(-1, 1)  # Column vector for row indices
j_indices = np.arange(Nth)
pcf = pcf[:, :, i_indices, (j_indices - i_indices) % Nth]

# Convert numpy values to tensors
r = torch.from_numpy(r)
phi = torch.from_numpy(phi)
theta = torch.from_numpy(theta)
pcf = torch.from_numpy(pcf)
p_th = torch.from_numpy(p_th)

Nth = theta.shape[0]

# Compute force magnitude
F = kc * (2 - r)

# Define the parameter vector that represents ψ(θ) at each grid point.
# We initialize it (arbitrarily) as small random values or zeros.
mu_var = torch.nn.Parameter(torch.zeros(Nth))
psi_init_values = torch.ones(Nth) * 1e-10
psi_init_values[0] = 0.5 / dth
psi_init_values[-1] = 0.5 / dth
psi_init_values /= torch.sum(psi_init_values) * dth
psi_var = torch.nn.Parameter(psi_init_values)


# --- 3) A function that maps mu -> psi, ensuring sum(psi)*dth = 1 and psi >= 0 ---
def softmax_distribution(mu):
    _mu = mu - torch.max(mu)
    exp_mu = torch.exp(_mu)
    return exp_mu / torch.sum(exp_mu) / dth


def C(psi):
    # Torque
    integrand = -torch.einsum("i,j,l,ijkl->ijkl", r * F, torch.sin(phi), psi, pcf)
    C = torch.sum(torch.trapezoid(integrand, r, dim=0), dim=(0, 2)) * dphi * dth
    return C


def f(psi):
    torque = C(psi)
    f = torch.cumsum(torque, dim=0) * dth
    return f


def estimate_psi(psi, theta):
    pdf = torch.exp(H / 2 * torch.cos(2 * theta) + K * f(psi))
    pdf = pdf / torch.sum(pdf)
    return pdf


def loss_function(mu):
    psi = softmax_distribution(mu)
    psi_star = estimate_psi(psi, theta)
    diff = psi - psi_star
    L = 0.5 * torch.sum(diff**2) * dth
    return L


def loss_function_psi(psi):
    psi_star = estimate_psi(psi, theta)
    diff = psi - psi_star
    L = 0.5 * torch.sum(diff**2) * dth
    return L


matplotlib.use("TkAgg")
fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw=dict(polar=True))

optimizer = torch.optim.Adam([psi_var], lr=1e-2)
# scheduler1 = ExponentialLR(optimizer, gamma=0.9)
# torch.autograd.set_detect_anomaly(True)

max_iter = 1000

for i in range(max_iter):
    optimizer.zero_grad()
    L_val = loss_function_psi(psi_var)
    prev_L = L_val.item()
    L_val.backward()
    optimizer.step()
    new_L = loss_function_psi(psi_var).item()
    if abs(new_L - prev_L) < 1e-6:
        break

    if (i + 1) % 50 == 0:
        print(f"Iter {i+1}, L = {L_val.item():.6f}")
        psi_plot = psi_var.detach()
        axs[0].cla()
        axs[0].plot(theta, psi_plot)
        axs[1].cla()
        axs[1].plot(theta, abs(C(psi_plot)))
        axs[2].cla()
        axs[2].plot(theta, f(psi_plot))
        plt.pause(0.05)

plt.close(fig)


with torch.no_grad():
    psi_final = psi_var.detach()
    psi_final = torch.where(psi_final < 0, 0, psi_final)
    plt.polar(theta, psi_final / torch.sum(psi_final) / dth)
    plt.polar(theta, p_th)
    plt.show()

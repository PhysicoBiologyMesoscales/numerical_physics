import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py

from os.path import join

sim_path = r"E:\Local_Sims\test_new_code"

with h5py.File(join(sim_path, "data.h5py")) as hdf_file:
    H = hdf_file.attrs["h"]
    K = hdf_file.attrs["k"] * hdf_file.attrs["phi"] / np.pi
    L = hdf_file.attrs["L"]
    l = hdf_file.attrs["l"]
    N = int(hdf_file.attrs["N"])
    kc = hdf_file.attrs["kc"]
    rho = hdf_file.attrs["N"] / hdf_file.attrs["l"] / hdf_file.attrs["L"]
    corr = hdf_file["pair_correlation"]
    Nth = corr.attrs["Nth"]
    r = corr["r"][()]
    dr = corr["rdr"][()] / corr["r"][()]
    phi = corr["phi"][()]
    theta = corr["theta"][()]
    dth = corr.attrs["dth"]
    dphi = corr.attrs["dphi"]
    N_pairs = corr["N_pairs"][50:].mean(axis=0)
    p_th = corr["p_th"][50:].mean(axis=0)


pcf = (
    L
    * l
    * N_pairs
    / (N * (N - 1) / 2)
    / (
        (r * dr)[:, np.newaxis, np.newaxis, np.newaxis]
        * p_th[np.newaxis, np.newaxis, :, np.newaxis]
        * p_th[np.newaxis, np.newaxis, np.newaxis, :]
        * dphi
        * dth**2
    )
)

H_init, K_init = 5.0, 6.0

# i_indices = np.arange(Nth).reshape(-1, 1)  # Column vector for row indices
# j_indices = np.arange(Nth)
# pcf = pcf[:, :, i_indices, (j_indices - i_indices) % Nth]

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

H_var = torch.nn.Parameter(torch.tensor(H_init))
K_var = torch.nn.Parameter(torch.tensor(K_init))


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


def estimate_psi(psi, H, K):
    pdf = torch.exp(H / 2 * torch.cos(2 * theta) + K * f(psi))
    pdf = pdf / torch.sum(pdf) / dth
    pdf = torch.where(pdf < 0, 0, pdf)
    mu_f = torch.log(pdf)
    pdf = softmax_distribution(mu_f)
    return pdf


def loss_function(mu, H, K):
    psi = softmax_distribution(mu)
    psi_star = estimate_psi(psi, H, K)
    diff = psi - psi_star
    L = 0.5 * torch.sum(diff**2) * dth
    return L


def loss_function_psi(psi, H, K):
    psi_star = estimate_psi(psi, H, K)
    diff = psi - psi_star
    L = 0.5 * torch.sum(diff**2) * dth
    return L


def loss_function_HK(psi, H, K):
    diff = estimate_psi(psi, H, K) - p_th
    L = 0.5 * torch.sum(diff**2) * dth
    return L


matplotlib.use("TkAgg")
fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw=dict(polar=True))

optimizer_psi = torch.optim.Adam([psi_var], lr=1e-2)
optimizer_HZ = torch.optim.Adam([K_var], lr=0.1)
# scheduler1 = ExponentialLR(optimizer, gamma=0.9)
# torch.autograd.set_detect_anomaly(True)

max_iter_psi = 1000
max_iter_HK = 1000

dL_HK = np.inf

for k in range(max_iter_HK):
    for i in range(max_iter_psi):
        optimizer_psi.zero_grad()
        L_val = loss_function_psi(psi_var, 3.0, K_var)
        prev_L = L_val.item()
        L_val.backward(inputs=[psi_var])
        optimizer_psi.step()
        new_L = loss_function_psi(psi_var, 3.0, K_var).item()
        dL_psi = new_L - prev_L
        if abs(dL_psi) < 1e-8:
            break
    optimizer_HZ.zero_grad()
    L_HK = loss_function_HK(psi_var, 3.0, K_var)
    L_HK.backward(inputs=[K_var])
    optimizer_HZ.step()
    new_L_HK = loss_function_HK(psi_var, 3.0, K_var)
    if abs((new_L_HK - L_HK).item()) < 1e-8:
        break
    if (k + 1) % 20 == 0:
        print(f"H={3.0}; K={K_var.item()}; L={L_HK.item():.6f}")
        psi_plot = psi_var.detach()
        axs[0].cla()
        axs[0].plot(theta, psi_plot)
        axs[1].cla()
        axs[1].plot(theta, abs(C(psi_plot)))
        axs[2].cla()
        axs[2].plot(theta, f(psi_plot))
        plt.pause(0.05)
    if (i + 1) % 50 == 0:
        print(f"Iter {i+1}, L = {L_val.item():.6f}")

plt.close(fig)


with torch.no_grad():
    psi_final = psi_var.detach()
    psi_final = torch.where(psi_final < 0, 0, psi_final)
    mu_f = torch.log(psi_final)
    psi_final = softmax_distribution(mu_f)
    plt.polar(theta, psi_final)
    plt.polar(theta, p_th)
    plt.show()

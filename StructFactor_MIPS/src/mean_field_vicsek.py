import numpy as np
import matplotlib.pyplot as plt
from struct_aux import Kern_gauss

if __name__ == "__main__":
    from models import vicsek_model

    Dr = 0.1
    lp = 0.5 / Dr
    phi = 0.5
    s = 30

    gamma_thresh = Dr * np.pi / 2 / phi

    gamma = 5.0 * gamma_thresh / Dr

    v_model = vicsek_model(lp=lp, phi=phi, gamma=gamma, Kern=Kern_gauss, s=s)

    k_arr = np.linspace(0, 10.0, 300)
    l_eigs = np.zeros((len(k_arr), 2 * s + 1), dtype=np.complex128)
    for i, k in enumerate(k_arr):
        L = v_model.L(k)
        eigs, vecs = np.linalg.eig(L)
        l_eigs[i] = eigs
        if np.any(np.real(eigs) < 0):
            print(f"Unstable mode at k={k:.2f}, eigenvalues: {eigs}")

    fig, ax = plt.subplots()
    for i in range(2 * s + 1):
        ax.scatter(k_arr, np.real(l_eigs[:, i]), s=5, color="grey")
    ax.set_ylim(0, 20)
    fig.savefig(f"../figures/vicsek/Re_eigs_k.svg")

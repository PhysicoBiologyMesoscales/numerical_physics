import numpy as np
import matplotlib.pyplot as plt

from scipy.special import iv

N = 1000
theta = np.linspace(0, 2 * np.pi, N)
alpha = np.logspace(-1, 0.2, 5)
p_stat = 1 / np.pi / iv(0, alpha)[None, :] * np.exp(np.outer(np.cos(2 * theta), alpha))

plt.polar(
    theta, p_stat, label=[rf"$\tilde{{\alpha}}={alpha_i:.3}$" for alpha_i in alpha]
)
plt.legend()
plt.title(
    r"Angular distribution for different values of $\tilde{{\alpha}}=\frac{{\alpha}}{{2R}}$"
)

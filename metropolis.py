import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table, label


class Metropolis:
    def __init__(self, Lx, Ly) -> None:
        self.lattice = np.random.choice([-1, 1], size=(Ly, Lx))
        self.Lx = Lx
        self.Ly = Ly
        self.beta = 100
        self.line_increment = np.repeat([np.arange(self.Ly)*self.Lx], self.Lx, axis=0).T
        

    def update(self):
        N_moves = self.Lx**2
        for k in range(N_moves):
            i, j = np.random.randint([self.Ly, self.Lx])
            alpha = 0.01  # Weight of the coupling between lines
            h = (
                self.lattice[i, (j - 1) % self.Lx] + self.lattice[i, (j + 1) % self.Lx]
            ) + alpha * (
                self.lattice[(i + 1) % self.Ly, (j - 1) % self.Lx]
                + self.lattice[(i + 1) % self.Ly, (j + 1) % self.Lx]
                + self.lattice[(i - 1) % self.Ly, (j - 1) % self.Lx]
                + self.lattice[(i - 1) % self.Ly, (j + 1) % self.Lx]
            )
            delta = h * self.lattice[i, j]
            gamma = np.random.uniform(0, 1)
            if np.log(gamma) < -self.beta * delta:
                self.lattice[i, j] *= -1
    
    def segment_domains(self):
        pass


if __name__ == "__main__":
    plt.ion()
    met = Metropolis(200, 500)
    N_t = 1000
    plt.figure()
    for t in range(N_t):
        met.update()
        plt.clf()
        plt.imshow(met.lattice)
        plt.pause(0.1)

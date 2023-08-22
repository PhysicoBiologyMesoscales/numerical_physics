import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.typing import NDArray


class Lattice1D:
    def __init__(self, size: int, N_particles: int, N_t: int) -> None:
        assert size >= N_particles, "The lattice is too small"
        self.size = size
        self.N = N_particles
        self.pos = sorted(
            np.random.choice(np.arange(self.size), size=self.N, replace=False)
        )
        self.dir = np.random.choice([-1, 1], size=self.N)
        self.iter = 0
        self.N_t = N_t

    @classmethod
    def from_arrays(cls, size: int, pos: NDArray, dir: NDArray, N_t: int):
        assert len(pos) == len(dir), "pos and dir must have the same dimension"
        assert len(pos) < size, "The lattice is too small"
        lat = cls(size, len(pos), N_t)
        lat.pos = pos.copy()
        lat.dir = dir.copy()
        return lat

    def get_consecutive(self):
        return np.diff(self.pos, append=self.pos[0]) % self.size == 1

    def get_colliding(self, consec):
        convergent = np.diff(self.dir, append=self.dir[0]) == -2
        return consec * convergent

    def get_trailing(self, consec):
        aligned = np.diff(self.dir, append=self.dir[0]) == 0
        return consec * aligned

    def stop_colliding(self, collide, trailing):
        consec_groups = np.split(
            np.arange(self.N), np.where(1 - np.logical_or(trailing, collide))[0] + 1
        )
        if np.logical_or(trailing, collide)[-1]:
            if not len(consec_groups) <= 1:
                consec_groups[0] = np.concatenate(
                    [consec_groups.pop(), consec_groups[0]]
                )
        for consec_indices in consec_groups:
            if np.any(collide[consec_indices]):
                self.attempt_move[consec_indices] = 0

    def overlap(self, trailing):
        # Get targeted destinations
        targets = (self.pos + self.attempt_move) % self.size
        u, inv = np.unique(targets, return_inverse=True)
        # Index of particles which are heading to the same site as their opposite neighbour
        overlap_idx = np.where(np.diff(inv, append=inv[0]) == 0)[0]
        # Choose randomly which particle gets to the site
        stop = np.random.choice([0, 1], size=len(overlap_idx))
        # Stop the other particles and their trails
        stopped = np.zeros(self.N)
        self.attempt_move[(overlap_idx + stop) % self.N] = 0
        stopped[(overlap_idx + stop) % self.N] = True
        consec_groups = np.split(np.arange(self.N), np.where(1 - trailing)[0] + 1)
        # Deal with periodic BC
        if trailing[-1]:
            if not len(consec_groups) <= 1:
                consec_groups[0] = np.concatenate(
                    [consec_groups.pop(), consec_groups[0]]
                )
        for consec_indices in consec_groups:
            if np.any(stopped[consec_indices]):
                self.attempt_move[consec_indices] = 0

    def switch_orient(self, collide, trailing):
        collisions_idx = np.where(collide)[0]
        weights = np.zeros(len(collisions_idx))
        # Loop through collisions; set weights according to train size
        for i, coll_idx in enumerate(collisions_idx):
            left_idx = coll_idx - 1
            left_count = 1
            while trailing[left_idx] == 1:
                left_idx -= 1
                left_count += 1
            right_idx = (coll_idx + 1) % self.N
            right_count = 1
            while trailing[right_idx % self.N] == 1:
                right_idx += 1
                right_count += 1
            weights[i] = left_count / (left_count + right_count)
        # Random integers between 0 and 1 (left or right colliding particle chosen for switch)
        switch = (np.random.rand(len(collisions_idx)) < weights).astype(int)
        self.dir[(collisions_idx + switch) % self.N] *= -1

    def update(self):
        self.attempt_move = self.dir.copy()
        consec = self.get_consecutive()
        # Check for collisions and trailing particles
        collide = self.get_colliding(consec)
        trailing = self.get_trailing(consec)
        # Set all colliding particles moves to 0
        self.stop_colliding(collide, trailing)
        self.overlap(trailing)
        # Randomly change orientation for collisions
        self.switch_orient(collide, trailing)
        # Update positions
        self.pos += self.attempt_move
        self.pos %= self.size
        self.iter += 1


class Lattice2D:
    def __init__(
        self, size_x: int, size_y: int, N_particles: int, N_t: int, alpha: float
    ) -> None:
        self.size_x = size_x
        self.size_y = size_y
        self.N = N_particles
        self.alpha = alpha
        pos1d = sorted(
            np.random.choice(
                np.arange(self.size_x * size_y), size=self.N, replace=False
            )
        )
        self.pos = np.array(np.unravel_index(pos1d, (self.size_y, self.size_x)))
        self.dir = np.random.choice([-1, 1], size=self.N)
        # nan means there's no particle on the site, +/-1 means there is a particle going to the right/left
        lattice = np.zeros((self.size_y, self.size_x), dtype=int)
        lattice[*self.pos] = self.dir
        self.lattice = np.ma.array(lattice, mask=1 - abs(lattice))

        self.iter = 0
        self.N_t = N_t

    def get_colliding_idx(self):
        return np.argwhere((self.lattice - np.roll(self.lattice, -1, axis=1)) == 2)

    def get_overlap_idx(self):
        # Overlap when 2 particles go towards the same empty site
        return np.argwhere(
            (self.lattice - np.roll(self.lattice, -2, axis=1) == 2).data
            & np.roll(self.lattice.mask, -1, axis=1),
        )

    def get_neighbours_index(self, direction):
        neighbours_idx = self.pos.copy()
        neighbours_idx[0] += direction
        neighbours_idx[0] %= self.size_y
        neighbours_idx[1] += self.dir
        neighbours_idx[1] %= self.size_x
        return neighbours_idx

    def coupling_between_lanes(self):
        up_neighbours_idx = self.get_neighbours_index(1)
        down_neighbours_idx = self.get_neighbours_index(-1)
        opp_to_upper = self.dir != self.lattice[*up_neighbours_idx]
        opp_to_lower = self.dir != self.lattice[*down_neighbours_idx]
        opp_to_one = opp_to_upper ^ opp_to_lower
        opp_to_both = opp_to_upper & opp_to_lower
        switch_proba = np.zeros(self.N)
        switch_proba[opp_to_one] = self.alpha
        switch_proba[opp_to_both] = 2 * self.alpha
        switch = np.random.random(self.N) < switch_proba
        self.dir[switch] *= -1
        self.lattice[*self.pos] = self.dir

    def stop_and_switch_colliding(self):
        # Get array containing indices (row, column) of +1 colliding particles
        collisions_idx = self.get_colliding_idx()
        # Weight array containing probability to switch direction
        weights = np.zeros(len(collisions_idx))
        for i, coll_idx in enumerate(collisions_idx):
            row = coll_idx[0]
            # Count all particles to the left of the collision
            left_idx = coll_idx[1]
            left_count = 0
            while self.lattice[row, left_idx % self.size_x]:
                if self.lattice[row, left_idx % self.size_x] in [0, -1]:
                    break
                self.lattice[row, left_idx % self.size_x] = 0
                left_idx -= 1
                left_count += 1
            # Count all particles to the right of the collision
            right_idx = (coll_idx[1] + 1) % self.size_x
            right_count = 0
            while self.lattice[row, right_idx % self.size_x]:
                if self.lattice[row, right_idx % self.size_x] in [0, 1]:
                    break
                self.lattice[row, right_idx % self.size_x] = 0
                right_idx += 1
                right_count += 1
            # Probability to flip is proportional to the weight of the colliding train
            weights[i] = left_count / (right_count + left_count)
        # Switch direction randomly
        switch = (np.random.rand(len(collisions_idx)) < weights).astype(int)
        new_idx = collisions_idx.copy().T
        new_idx[1] += switch
        new_idx[1] %= self.size_x
        self.next_lattice[*new_idx] *= -1

    def overlap(self):
        # Get array containing indices (row, col) of +1 overlapping particles
        overlaps_idx = self.get_overlap_idx()
        for overlap_idx in overlaps_idx:
            row = overlap_idx[0]
            col = overlap_idx[1]
            # Choose randomly which particle gets to the empty site
            left_or_right = np.random.choice([-1, 1])
            # Stop the other train of particles
            match left_or_right:
                case -1:
                    while self.lattice[row, col]:
                        if self.lattice[row, col] == -1 or self.lattice[row, col] == 0:
                            break
                        self.lattice[row, col] = 0
                        col -= 1
                        col %= self.size_x
                case 1:
                    col += 2
                    col %= self.size_x
                    while self.lattice[row, col]:
                        if self.lattice[row, col] == 1 or self.lattice[row, col] == 0:
                            break
                        self.lattice[row, col] = 0
                        col += 1
                        col %= self.size_x

    def update_pos_and_lattice(self):
        self.pos[1] += self.lattice[*self.pos]
        self.pos[1] %= self.size_x
        lattice = np.zeros((self.size_y, self.size_x), dtype=int)
        lattice[*self.pos] = self.dir
        self.lattice = np.ma.array(lattice, mask=(lattice == 0))

    def update(self):
        self.coupling_between_lanes()
        self.next_lattice = self.lattice.copy()
        self.stop_and_switch_colliding()
        self.dir = self.next_lattice[*self.pos]
        self.overlap()
        self.update_pos_and_lattice()
        self.iter += 1


if __name__ == "__main__":
    lat = Lattice2D(100, 200, 20000, 1000, alpha=0.2)
    cmap = cm.jet
    cmap.set_bad("black", 1.0)
    plt.figure()
    while lat.iter < lat.N_t:
        plt.clf()
        plt.imshow(lat.lattice, interpolation="nearest", cmap=cmap)
        plt.pause(0.1)
        lat.update()
        # u, c = np.unique(lat.pos, axis=1, return_counts=True)
        # if 2 in c:
        #     print("Overlapping particles")
        #     break

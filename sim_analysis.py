import numpy as np
import scipy.sparse as sp


def compute_coarsegrain_matrix(r, theta, l, L, Nx, Ny, Nth, N):
    dx = l / Nx
    dy = L / Ny
    dth = 2 * np.pi / Nth
    # We compute forces in (x, y, theta) space
    r_th = np.concatenate([r, np.array([theta % (2 * np.pi)]).T], axis=-1)
    # Build matrix and 1D encoding in (x, y, theta) space
    Cijk = (r_th // np.array([dx, dy, dth])).astype(int)
    C1d_cg = np.ravel_multi_index(Cijk.T, (Nx, Ny, Nth), order="C")
    C = sp.eye(Nx * Ny * Nth, format="csr")[C1d_cg] / N
    return C


def compute_distribution(C, Nx, Ny, Nth):
    # Compute one-body distribution
    return np.array(C.T.sum(axis=1)).reshape((Nx, Ny, Nth)).swapaxes(0, 1)


def coarsegrain_field(field, C, Nx, Ny, Nth):
    if field.ndim == 1:
        _field = np.array([field]).T
    else:
        _field = field
    data_ndims = field.shape[-1]
    field_cg = C.T @ _field
    return field_cg.reshape((Nx, Ny, Nth, data_ndims)).swapaxes(0, 1)

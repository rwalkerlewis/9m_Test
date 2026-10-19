"""2-D Helmholtz (frequency-domain) solver."""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from acoustic_sim.model import VelocityModel


def solve_helmholtz(
    model: VelocityModel,
    source_xy: np.ndarray,
    frequency_hz: float,
    damping_width: int | None = None,
) -> np.ndarray:
    """Solve the 2-D Helmholtz equation on the velocity-model grid.

    Parameters
    ----------
    model : VelocityModel
        The velocity field to use.
    source_xy : array_like, shape (2,)
        Source location (x, y).
    frequency_hz : float
        Driving frequency in Hz.
    damping_width : int or None
        Absorbing-layer width in cells (auto-chosen if *None*).

    Returns
    -------
    field : np.ndarray, shape (ny, nx)
        Pressure magnitude |p(x, y)|.
    """
    nx, ny = model.nx, model.ny
    dx = model.dx
    omega = 2.0 * np.pi * frequency_hz
    k_map = omega / np.maximum(model.values, 1.0)

    if damping_width is None:
        damping_width = max(3, int(0.08 * min(nx, ny)))

    # Absorbing boundary layer
    damping = np.zeros((ny, nx), dtype=np.float64)
    for iy in range(ny):
        for ix in range(nx):
            d_edge = min(ix, nx - 1 - ix, iy, ny - 1 - iy)
            if d_edge < damping_width:
                damping[iy, ix] = (
                    ((damping_width - d_edge) / damping_width) ** 2 * 0.7
                )

    total = nx * ny
    a_mat = lil_matrix((total, total), dtype=np.complex128)
    b_vec = np.zeros(total, dtype=np.complex128)
    inv_dx2 = 1.0 / (dx * dx)

    def flat(ix: int, iy: int) -> int:
        return iy * nx + ix

    for iy in range(ny):
        for ix in range(nx):
            row = flat(ix, iy)
            k_local = k_map[iy, ix]
            sigma = damping[iy, ix]
            diag = -4.0 * inv_dx2 + (k_local**2) * (1.0 + 1j * sigma)
            a_mat[row, row] = diag
            if ix > 0:
                a_mat[row, flat(ix - 1, iy)] = inv_dx2
            if ix < nx - 1:
                a_mat[row, flat(ix + 1, iy)] = inv_dx2
            if iy > 0:
                a_mat[row, flat(ix, iy - 1)] = inv_dx2
            if iy < ny - 1:
                a_mat[row, flat(ix, iy + 1)] = inv_dx2

    sx = int(np.argmin(np.abs(model.x - source_xy[0])))
    sy = int(np.argmin(np.abs(model.y - source_xy[1])))
    b_vec[flat(sx, sy)] = -1.0 / (dx * dx)

    p = spsolve(a_mat.tocsr(), b_vec)
    field = np.abs(p.reshape((ny, nx)))
    if not np.all(np.isfinite(field)):
        raise RuntimeError("Helmholtz solution produced non-finite values.")
    return field

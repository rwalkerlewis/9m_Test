"""3D velocity model dataclass and creation helpers.

Extends the 2D ``VelocityModel`` with a z-dimension.  The velocity field
is a 3-D array **[nz, ny, nx]** of wave speeds in m/s on a uniform
Cartesian grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityModel3D:
    """3D velocity model on a regular grid.

    Attributes
    ----------
    x : np.ndarray
        1-D array of x cell-centre coordinates.
    y : np.ndarray
        1-D array of y cell-centre coordinates.
    z : np.ndarray
        1-D array of z cell-centre coordinates.
    values : np.ndarray
        3-D array **[nz, ny, nx]** of wave speeds in m/s.
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    dz : float
        Grid spacing in the z direction.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    values: np.ndarray
    dx: float
    dy: float
    dz: float

    @property
    def nx(self) -> int:
        return len(self.x)

    @property
    def ny(self) -> int:
        return len(self.y)

    @property
    def nz(self) -> int:
        return len(self.z)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values.shape

    @property
    def extent_xy(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) for imshow *extent* of an x-y slice."""
        return (
            float(self.x[0]),
            float(self.x[-1]),
            float(self.y[0]),
            float(self.y[-1]),
        )

    @property
    def extent_xz(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, z_min, z_max) for an x-z slice."""
        return (
            float(self.x[0]),
            float(self.x[-1]),
            float(self.z[0]),
            float(self.z[-1]),
        )

    @property
    def c_min(self) -> float:
        return float(np.min(self.values))

    @property
    def c_max(self) -> float:
        return float(np.max(self.values))

    def velocity_at(self, px: float, py: float, pz: float) -> float:
        """Nearest-neighbour velocity look-up at an arbitrary (x, y, z) point."""
        ix = int(np.clip(np.round((px - self.x[0]) / self.dx), 0, self.nx - 1))
        iy = int(np.clip(np.round((py - self.y[0]) / self.dy), 0, self.ny - 1))
        iz = int(np.clip(np.round((pz - self.z[0]) / self.dz), 0, self.nz - 1))
        return float(self.values[iz, iy, ix])


# ---------------------------------------------------------------------------
# Creation helpers
# ---------------------------------------------------------------------------


def create_uniform_model_3d(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    dx: float,
    velocity: float = 343.0,
) -> VelocityModel3D:
    """Constant-velocity 3D model."""
    x = np.arange(x_min, x_max + 0.5 * dx, dx)
    y = np.arange(y_min, y_max + 0.5 * dx, dx)
    z = np.arange(z_min, z_max + 0.5 * dx, dx)
    values = np.full((len(z), len(y), len(x)), velocity, dtype=np.float64)
    return VelocityModel3D(x=x, y=y, z=z, values=values, dx=dx, dy=dx, dz=dx)


def create_layered_z_model_3d(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    dx: float,
    layers: list[tuple[float, float]],
    background: float = 343.0,
) -> VelocityModel3D:
    """3D model with horizontal layers defined by z boundaries.

    Parameters
    ----------
    layers : list of (z_boundary, velocity)
        Each entry sets the velocity for the region *below* ``z_boundary``.
        Processed bottom-to-top; ``background`` fills regions above the
        highest boundary.
    """
    model = create_uniform_model_3d(x_min, x_max, y_min, y_max,
                                     z_min, z_max, dx, background)
    sorted_layers = sorted(layers, key=lambda t: t[0])
    for iz, zval in enumerate(model.z):
        vel = background
        for z_boundary, layer_vel in sorted_layers:
            if zval <= z_boundary:
                vel = layer_vel
                break
        model.values[iz, :, :] = vel
    return model


def model_3d_from_array(
    values: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> VelocityModel3D:
    """Wrap an existing 3-D numpy array as a :class:`VelocityModel3D`.

    Parameters
    ----------
    values : np.ndarray
        Shape **[nz, ny, nx]** of wave speeds in m/s.
    """
    if values.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {values.shape}")
    nz, ny, nx = values.shape
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("Velocity array must be at least 2×2×2.")
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    return VelocityModel3D(x=x, y=y, z=z,
                            values=values.astype(np.float64),
                            dx=dx, dy=dy, dz=dz)

"""Velocity model dataclass and creation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityModel:
    """2D velocity model on a regular grid.

    Attributes
    ----------
    x : np.ndarray
        1-D array of x cell-centre coordinates.
    y : np.ndarray
        1-D array of y cell-centre coordinates.
    values : np.ndarray
        2-D array **[ny, nx]** of wave speeds in m/s.
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    """

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    dx: float
    dy: float

    @property
    def nx(self) -> int:
        return len(self.x)

    @property
    def ny(self) -> int:
        return len(self.y)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) for imshow *extent*."""
        return (
            float(self.x[0]),
            float(self.x[-1]),
            float(self.y[0]),
            float(self.y[-1]),
        )

    @property
    def c_min(self) -> float:
        return float(np.min(self.values))

    @property
    def c_max(self) -> float:
        return float(np.max(self.values))

    def velocity_at(self, px: float, py: float) -> float:
        """Nearest-neighbour velocity look-up at an arbitrary (x, y) point."""
        ix = int(np.clip(np.round((px - self.x[0]) / self.dx), 0, self.nx - 1))
        iy = int(np.clip(np.round((py - self.y[0]) / self.dy), 0, self.ny - 1))
        return float(self.values[iy, ix])


# ---------------------------------------------------------------------------
# Creation helpers
# ---------------------------------------------------------------------------


def create_uniform_model(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    dx: float,
    velocity: float = 343.0,
) -> VelocityModel:
    """Constant-velocity model."""
    x = np.arange(x_min, x_max + 0.5 * dx, dx)
    y = np.arange(y_min, y_max + 0.5 * dx, dx)
    values = np.full((len(y), len(x)), velocity, dtype=np.float64)
    return VelocityModel(x=x, y=y, values=values, dx=dx, dy=dx)


def create_layered_model(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    dx: float,
    layers: list[tuple[float, float]],
    background: float = 343.0,
) -> VelocityModel:
    """Horizontally layered model.

    Parameters
    ----------
    layers : list of (y_top, velocity)
        Each entry sets the velocity for the region *below* ``y_top``.
        Entries are processed bottom-to-top; the ``background`` velocity
        fills any region above the highest boundary.
    """
    model = create_uniform_model(x_min, x_max, y_min, y_max, dx, background)
    sorted_layers = sorted(layers, key=lambda t: t[0])
    for iy, yval in enumerate(model.y):
        vel = background
        for y_top, layer_vel in sorted_layers:
            if yval <= y_top:
                vel = layer_vel
                break
        model.values[iy, :] = vel
    return model


def create_gradient_model(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    dx: float,
    v_bottom: float = 360.0,
    v_top: float = 330.0,
) -> VelocityModel:
    """Linear velocity gradient from ``v_bottom`` (y_min) to ``v_top`` (y_max)."""
    model = create_uniform_model(x_min, x_max, y_min, y_max, dx)
    for iy, yval in enumerate(model.y):
        frac = (yval - y_min) / max(y_max - y_min, 1e-12)
        model.values[iy, :] = v_bottom + frac * (v_top - v_bottom)
    return model


def create_checkerboard_model(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    dx: float,
    cell_size: float = 4.0,
    v_base: float = 343.0,
    perturbation: float = 20.0,
) -> VelocityModel:
    """Checkerboard velocity perturbation (useful for resolution tests)."""
    model = create_uniform_model(x_min, x_max, y_min, y_max, dx, v_base)
    xx, yy = np.meshgrid(model.x, model.y)
    checker = np.sign(
        np.sin(np.pi * xx / cell_size) * np.sin(np.pi * yy / cell_size)
    )
    model.values[:] = v_base + perturbation * checker
    return model


def model_from_array(
    values: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> VelocityModel:
    """Wrap an existing 2-D numpy array as a :class:`VelocityModel`.

    Parameters
    ----------
    values : np.ndarray
        Shape **[ny, nx]** of wave speeds in m/s.
    x_min, x_max, y_min, y_max : float
        Physical extent of the grid.
    """
    if values.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {values.shape}")
    ny, nx = values.shape
    if nx < 2 or ny < 2:
        raise ValueError("Velocity array must be at least 2×2.")
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return VelocityModel(x=x, y=y, values=values.astype(np.float64), dx=dx, dy=dy)


# ---------------------------------------------------------------------------
# Anomaly injection
# ---------------------------------------------------------------------------


def add_circle_anomaly(
    model: VelocityModel,
    cx: float,
    cy: float,
    radius: float,
    velocity: float,
) -> VelocityModel:
    """Return a copy of *model* with a circular region overwritten."""
    xx, yy = np.meshgrid(model.x, model.y)
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    new_values = model.values.copy()
    new_values[mask] = velocity
    return VelocityModel(
        x=model.x, y=model.y, values=new_values, dx=model.dx, dy=model.dy
    )


def add_rectangle_anomaly(
    model: VelocityModel,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    velocity: float,
) -> VelocityModel:
    """Return a copy of *model* with a rectangular region overwritten."""
    xx, yy = np.meshgrid(model.x, model.y)
    mask = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    new_values = model.values.copy()
    new_values[mask] = velocity
    return VelocityModel(
        x=model.x, y=model.y, values=new_values, dx=model.dx, dy=model.dy
    )

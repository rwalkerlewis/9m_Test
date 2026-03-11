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


# ---------------------------------------------------------------------------
# Valley model with randomised hill profiles
# ---------------------------------------------------------------------------

def _random_hill_profile(
    x: np.ndarray,
    base_y: float,
    peak_height: float,
    base_width: float,
    rng: np.random.Generator,
    n_bumps: int = 6,
) -> np.ndarray:
    """Return an organic-looking hill surface height for each *x* value.

    A Gaussian envelope is modulated by a sum of random sinusoidal bumps so
    the ridge-line is never perfectly smooth.
    """
    mid_x = 0.5 * (x[0] + x[-1])
    sigma = base_width / 4.0
    envelope = peak_height * np.exp(-0.5 * ((x - mid_x) / sigma) ** 2)
    bumps = np.zeros_like(x)
    for _ in range(n_bumps):
        freq = rng.uniform(0.05, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.05, 0.25) * peak_height
        bumps += amp * np.sin(2 * np.pi * freq * (x - mid_x) + phase)
    return base_y + envelope + bumps


def create_valley_model(
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    dx: float = 0.5,
    air_velocity: float = 343.0,
    dirt_velocity: float = 1500.0,
    seed: int = 42,
    hill_south_y: float = -20.0,
    hill_north_y: float = 20.0,
    hill_peak_height: float = 18.0,
    hill_base_width: float = 60.0,
    saddle_x: float = 0.0,
    saddle_width: float = 12.0,
    saddle_depth_frac: float = 0.55,
) -> VelocityModel:
    """Create a valley between two ridges with a saddle notch.

    The southern and northern ridges have organic, randomised profiles.
    The northern ridge contains a saddle (low pass) centred at *saddle_x*.
    Everything below a ridge surface is dirt (``dirt_velocity``); the open
    valley and air above the ridges use ``air_velocity``.  The strong
    impedance contrast produces visible reflections.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(x_min, x_max + dx / 2, dx)
    y = np.arange(y_min, y_max + dx / 2, dx)
    nx, ny = len(x), len(y)

    # Start with air everywhere
    values = np.full((ny, nx), air_velocity, dtype=np.float64)

    # Southern ridge (grows upward from hill_south_y)
    south_profile = _random_hill_profile(
        x, hill_south_y, hill_peak_height, hill_base_width, rng,
    )
    # Northern ridge (grows downward from hill_north_y, i.e. toward positive y)
    north_profile = _random_hill_profile(
        x, hill_north_y, hill_peak_height, hill_base_width, rng,
    )

    # Carve the saddle notch into the northern ridge
    saddle_env = np.exp(-0.5 * ((x - saddle_x) / (saddle_width / 2)) ** 2)
    saddle_cut = saddle_depth_frac * hill_peak_height * saddle_env
    north_profile -= saddle_cut

    # Fill dirt: south ridge fills below its profile, north ridge fills above
    yy = y[:, np.newaxis]  # (ny, 1) for broadcasting
    south_mask = yy <= south_profile[np.newaxis, :]  # below southern crest
    north_mask = yy >= north_profile[np.newaxis, :]  # above northern crest
    values[south_mask] = dirt_velocity
    values[north_mask] = dirt_velocity

    return VelocityModel(x=x, y=y, values=values, dx=dx, dy=dx)

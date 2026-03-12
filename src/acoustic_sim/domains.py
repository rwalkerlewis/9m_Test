"""Domain builders for FDTD simulations.

Each builder returns ``(VelocityModel, DomainMeta)`` — the velocity field
plus metadata describing wind, attenuation zones, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from acoustic_sim.model import VelocityModel, create_uniform_model


@dataclass
class DomainMeta:
    """Extra per-cell physics that supplement the velocity model."""

    wind_vx: float = 0.0
    wind_vy: float = 0.0
    attenuation: np.ndarray | None = None  # (ny, nx) damping coefficients
    description: str = ""


# ---------------------------------------------------------------------------
# Isotropic (uniform, no wind, no attenuation)
# ---------------------------------------------------------------------------

def create_isotropic_domain(
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    dx: float = 0.5,
    velocity: float = 343.0,
) -> tuple[VelocityModel, DomainMeta]:
    """Uniform velocity, no wind, no attenuation."""
    model = create_uniform_model(x_min, x_max, y_min, y_max, dx, velocity)
    meta = DomainMeta(description="Isotropic (uniform velocity, no wind)")
    return model, meta


# ---------------------------------------------------------------------------
# Isotropic + wind
# ---------------------------------------------------------------------------

def create_wind_domain(
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    dx: float = 0.5,
    velocity: float = 343.0,
    wind_speed: float = 15.0,
    wind_direction_deg: float = 45.0,
) -> tuple[VelocityModel, DomainMeta]:
    """Uniform velocity with a constant wind field.

    Parameters
    ----------
    wind_speed : float
        Wind speed in m/s (must be subsonic).
    wind_direction_deg : float
        Meteorological direction the wind is *coming from* in degrees,
        measured clockwise from the +y axis.  Internally converted to
        Cartesian vx, vy components.
    """
    model = create_uniform_model(x_min, x_max, y_min, y_max, dx, velocity)
    rad = np.deg2rad(wind_direction_deg)
    vx = wind_speed * np.sin(rad)
    vy = wind_speed * np.cos(rad)
    meta = DomainMeta(
        wind_vx=float(vx),
        wind_vy=float(vy),
        description=(
            f"Wind domain: {wind_speed:.1f} m/s from {wind_direction_deg:.0f}°"
        ),
    )
    return model, meta


# ---------------------------------------------------------------------------
# Hills + vegetation
# ---------------------------------------------------------------------------

def _random_hill_profile(
    x: np.ndarray,
    base_y: float,
    peak_height: float,
    base_width: float,
    rng: np.random.Generator,
    n_bumps: int = 6,
) -> np.ndarray:
    """Organic hill surface height at each *x* — Gaussian + random bumps."""
    mid_x = 0.5 * (x[0] + x[-1])
    sigma = base_width / 4.0
    envelope = peak_height * np.exp(-0.5 * ((x - mid_x) / sigma) ** 2)
    bumps = np.zeros_like(x)
    for _ in range(n_bumps):
        freq = rng.uniform(0.05, 0.3)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.05, 0.25) * peak_height
        bumps += amp * np.sin(2.0 * np.pi * freq * (x - mid_x) + phase)
    return base_y + envelope + bumps


def create_hills_vegetation_domain(
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    dx: float = 0.5,
    air_velocity: float = 343.0,
    dirt_velocity: float = 1500.0,
    veg_thickness: float = 4.0,
    veg_attenuation: float = 0.15,
    seed: int = 42,
    hill_south_y: float = -20.0,
    hill_north_y: float = 20.0,
    hill_peak_height: float = 18.0,
    hill_base_width: float = 60.0,
) -> tuple[VelocityModel, DomainMeta]:
    """2-D slice with two ridges, a valley, and vegetation zones.

    Hills are solid (``dirt_velocity``); a thin vegetation layer above
    each ridge surface adds extra damping.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(x_min, x_max + dx / 2, dx)
    y = np.arange(y_min, y_max + dx / 2, dx)
    nx, ny = len(x), len(y)

    values = np.full((ny, nx), air_velocity, dtype=np.float64)
    attenuation = np.zeros((ny, nx), dtype=np.float64)

    # Southern ridge (grows upward).
    south_profile = _random_hill_profile(
        x, hill_south_y, hill_peak_height, hill_base_width, rng,
    )
    # Northern ridge (grows downward toward +y).
    north_profile = _random_hill_profile(
        x, hill_north_y, hill_peak_height, hill_base_width, rng,
    )

    yy = y[:, np.newaxis]  # (ny, 1) for broadcasting
    south_mask = yy <= south_profile[np.newaxis, :]
    north_mask = yy >= north_profile[np.newaxis, :]
    values[south_mask] = dirt_velocity
    values[north_mask] = dirt_velocity

    # Vegetation: thin layer just *above* each ridge surface.
    south_veg = (yy > south_profile[np.newaxis, :]) & (
        yy <= south_profile[np.newaxis, :] + veg_thickness
    )
    north_veg = (yy < north_profile[np.newaxis, :]) & (
        yy >= north_profile[np.newaxis, :] - veg_thickness
    )
    attenuation[south_veg] = veg_attenuation
    attenuation[north_veg] = veg_attenuation

    model = VelocityModel(x=x, y=y, values=values, dx=dx, dy=dx)
    meta = DomainMeta(
        attenuation=attenuation,
        description="Hills + vegetation (2-D valley slice)",
    )
    return model, meta


# ---------------------------------------------------------------------------
# Echo-prone domains
# ---------------------------------------------------------------------------

def create_echo_canyon_domain(
    x_min: float = -100.0,
    x_max: float = 100.0,
    y_min: float = -100.0,
    y_max: float = 100.0,
    dx: float = 0.2,
    air_velocity: float = 343.0,
    wall_velocity: float = 2000.0,
    canyon_y_south: float = -60.0,
    canyon_y_north: float = 60.0,
    canyon_wall_thickness: float = 5.0,
) -> tuple["VelocityModel", DomainMeta]:
    """Domain with two parallel walls forming a canyon.

    The strong impedance contrast between air (343 m/s) and wall
    material (2000 m/s) produces clear reflections — ideal for testing
    echo discrimination.

    Parameters
    ----------
    canyon_y_south, canyon_y_north : float
        y-coordinates of the inner edge of each wall.
    canyon_wall_thickness : float
        Thickness of each wall [m].
    """
    x = np.arange(x_min, x_max + dx / 2, dx)
    y = np.arange(y_min, y_max + dx / 2, dx)
    nx, ny = len(x), len(y)
    values = np.full((ny, nx), air_velocity, dtype=np.float64)

    # South wall: from canyon_y_south - thickness to canyon_y_south.
    for iy, yv in enumerate(y):
        if canyon_y_south - canyon_wall_thickness <= yv <= canyon_y_south:
            values[iy, :] = wall_velocity
        if canyon_y_north <= yv <= canyon_y_north + canyon_wall_thickness:
            values[iy, :] = wall_velocity

    model = VelocityModel(x=x, y=y, values=values, dx=dx, dy=dx)
    meta = DomainMeta(description="Echo canyon (parallel walls)")
    return model, meta


def create_urban_echo_domain(
    x_min: float = -100.0,
    x_max: float = 100.0,
    y_min: float = -100.0,
    y_max: float = 100.0,
    dx: float = 0.2,
    air_velocity: float = 343.0,
    building_velocity: float = 2500.0,
    n_buildings: int = 4,
    building_size: float = 15.0,
    seed: int = 42,
) -> tuple["VelocityModel", DomainMeta]:
    """Domain with rectangular buildings that produce complex multipath.

    Buildings are high-impedance blocks placed semi-randomly in the
    domain, away from the centre (where the mic array typically sits).

    Parameters
    ----------
    n_buildings : int
        Number of buildings.
    building_size : float
        Side length of each square building [m].
    """
    rng = np.random.default_rng(seed)
    x = np.arange(x_min, x_max + dx / 2, dx)
    y = np.arange(y_min, y_max + dx / 2, dx)
    nx, ny = len(x), len(y)
    values = np.full((ny, nx), air_velocity, dtype=np.float64)

    xx, yy = np.meshgrid(x, y)

    for _ in range(n_buildings):
        # Place building away from centre (at least 30 m from origin).
        while True:
            bx = rng.uniform(x_min + building_size, x_max - building_size)
            by = rng.uniform(y_min + building_size, y_max - building_size)
            if np.hypot(bx, by) > 30.0:
                break
        mask = (
            (xx >= bx - building_size / 2) & (xx <= bx + building_size / 2) &
            (yy >= by - building_size / 2) & (yy <= by + building_size / 2)
        )
        values[mask] = building_velocity

    model = VelocityModel(x=x, y=y, values=values, dx=dx, dy=dx)
    meta = DomainMeta(description=f"Urban echo ({n_buildings} buildings)")
    return model, meta

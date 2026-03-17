"""Domain builders for FDTD simulations.

Each builder returns ``(VelocityModel, DomainMeta)`` — the velocity field
plus metadata describing wind, attenuation zones, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from acoustic_sim.model import (
    VelocityModel,
    VelocityModel3D,
    create_uniform_model,
    create_uniform_model_3d,
)


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

    # Place buildings away from centre — at least 1/3 of domain half-width.
    min_dist = max((x_max - x_min) / 6.0, 2.0 * building_size)
    for _ in range(n_buildings):
        for _attempt in range(200):
            bx = rng.uniform(x_min + building_size, x_max - building_size)
            by = rng.uniform(y_min + building_size, y_max - building_size)
            if np.hypot(bx, by) > min_dist:
                break
        mask = (
            (xx >= bx - building_size / 2) & (xx <= bx + building_size / 2) &
            (yy >= by - building_size / 2) & (yy <= by + building_size / 2)
        )
        values[mask] = building_velocity

    model = VelocityModel(x=x, y=y, values=values, dx=dx, dy=dx)
    meta = DomainMeta(description=f"Urban echo ({n_buildings} buildings)")
    return model, meta


# ═════════════════════════════════════════════════════════════════════
# Merged from domains_3d
# ═════════════════════════════════════════════════════════════════════

@dataclass
class DomainMeta3D:
    """Extra per-cell physics for a 3D domain."""

    wind_vx: float = 0.0
    wind_vy: float = 0.0
    wind_vz: float = 0.0
    attenuation: np.ndarray | None = None  # (nz, ny, nx) damping coefficients
    description: str = ""


# ---------------------------------------------------------------------------
# Isotropic (uniform, no wind)
# ---------------------------------------------------------------------------

def create_isotropic_domain_3d(
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    z_min: float = 0.0,
    z_max: float = 100.0,
    dx: float = 1.0,
    velocity: float = 343.0,
) -> tuple[VelocityModel3D, DomainMeta3D]:
    """Uniform velocity, no wind, no attenuation."""
    model = create_uniform_model_3d(x_min, x_max, y_min, y_max,
                                     z_min, z_max, dx, velocity)
    meta = DomainMeta3D(description="3D Isotropic (uniform velocity, no wind)")
    return model, meta


# ---------------------------------------------------------------------------
# Isotropic + wind
# ---------------------------------------------------------------------------

def create_wind_domain_3d(
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    z_min: float = 0.0,
    z_max: float = 100.0,
    dx: float = 1.0,
    velocity: float = 343.0,
    wind_speed: float = 15.0,
    wind_direction_deg: float = 45.0,
    wind_vz: float = 0.0,
) -> tuple[VelocityModel3D, DomainMeta3D]:
    """Uniform velocity with a constant 3D wind field."""
    model = create_uniform_model_3d(x_min, x_max, y_min, y_max,
                                     z_min, z_max, dx, velocity)
    rad = np.deg2rad(wind_direction_deg)
    vx = wind_speed * np.sin(rad)
    vy = wind_speed * np.cos(rad)
    meta = DomainMeta3D(
        wind_vx=float(vx),
        wind_vy=float(vy),
        wind_vz=float(wind_vz),
        description=(
            f"3D Wind domain: {wind_speed:.1f} m/s from {wind_direction_deg:.0f}°"
        ),
    )
    return model, meta


# ---------------------------------------------------------------------------
# Ground layer (air + dirt)
# ---------------------------------------------------------------------------

def create_ground_layer_domain_3d(
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    z_min: float = -10.0,
    z_max: float = 100.0,
    dx: float = 1.0,
    air_velocity: float = 343.0,
    ground_velocity: float = 1500.0,
    ground_z: float = 0.0,
) -> tuple[VelocityModel3D, DomainMeta3D]:
    """3D domain with air above and ground (high velocity) below.

    The impedance contrast at ``ground_z`` produces a ground reflection
    — useful for testing multipath in the FDTD.
    """
    model = create_uniform_model_3d(x_min, x_max, y_min, y_max,
                                     z_min, z_max, dx, air_velocity)
    for iz, zval in enumerate(model.z):
        if zval <= ground_z:
            model.values[iz, :, :] = ground_velocity
    meta = DomainMeta3D(
        description=f"3D Ground layer (air/ground at z={ground_z}m)",
    )
    return model, meta


# ---------------------------------------------------------------------------
# Hills + vegetation (3D extrusion of the 2D valley)
# ---------------------------------------------------------------------------

def create_hills_vegetation_domain_3d(
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    z_min: float = -5.0,
    z_max: float = 50.0,
    dx: float = 1.0,
    air_velocity: float = 343.0,
    dirt_velocity: float = 1500.0,
    veg_thickness: float = 4.0,
    veg_attenuation: float = 0.15,
    seed: int = 42,
    hill_south_y: float = -20.0,
    hill_north_y: float = 20.0,
    hill_peak_height: float = 18.0,
    hill_base_width: float = 60.0,
) -> tuple[VelocityModel3D, DomainMeta3D]:
    """3-D valley between two ridges, extruded from the 2-D model.

    The two ridges run roughly east-west (along the x-axis).  In the
    2-D model, the y-axis encoded *both* horizontal distance from the
    ridge centre *and* the "height" of the ridge.  Here we separate
    them: x and y are horizontal, z is altitude.

    **Terrain height map** ``H(x, y)``
    -----------------------------------
    Each ridge has a 1-D peak-height profile ``h(x)`` generated by
    ``_random_hill_profile``.  The ridge extends as a Gaussian-shaped
    band in y centred at ``hill_south_y`` / ``hill_north_y``::

        H_south(x, y) = h_south(x) * exp(-0.5 * ((y - hill_south_y) / sigma_y)^2)
        H_north(x, y) = h_north(x) * exp(-0.5 * ((y - hill_north_y) / sigma_y)^2)

    where ``sigma_y`` controls the ridge width perpendicular to the
    ridge line.  The overall terrain height is the envelope::

        H(x, y) = max(H_south, H_north, 0)

    Grid cells with ``z < H(x, y)`` → dirt velocity.
    Grid cells with ``H(x, y) ≤ z < H(x, y) + veg_thickness`` →
    vegetation attenuation.
    Everything else → air.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        Horizontal extent [m].
    z_min, z_max : float
        Altitude extent [m].  ``z_min`` can be negative (sub-surface).
    dx : float
        Uniform grid spacing in all three dimensions.
    air_velocity, dirt_velocity : float
        Wave speeds for air and solid ground [m/s].
    veg_thickness : float
        Thickness of the vegetation damping layer above each ridge [m].
    veg_attenuation : float
        Peak damping coefficient inside the vegetation layer.
    seed : int
        RNG seed for the random hill bumps (same seed → same terrain as
        the 2-D model).
    hill_south_y, hill_north_y : float
        y-coordinates of the two ridge centres.
    hill_peak_height : float
        Maximum ridge elevation [m] above z = 0.
    hill_base_width : float
        Controls the width of the ridge along x (fed to the Gaussian
        envelope in ``_random_hill_profile``).
    """
    rng = np.random.default_rng(seed)

    x = np.arange(x_min, x_max + dx / 2, dx)
    y = np.arange(y_min, y_max + dx / 2, dx)
    z = np.arange(z_min, z_max + dx / 2, dx)
    nx, ny, nz = len(x), len(y), len(z)

    # -- 1-D peak-height profiles along x (same RNG sequence as 2-D) ------
    # _random_hill_profile returns base_y + envelope + bumps, so the
    # *relative* peak height above the base is  profile(x) - base_y.
    south_profile_abs = _random_hill_profile(
        x, hill_south_y, hill_peak_height, hill_base_width, rng,
    )
    north_profile_abs = _random_hill_profile(
        x, hill_north_y, hill_peak_height, hill_base_width, rng,
    )
    # Relative peak heights (metres above z=0).
    south_peak = south_profile_abs - hill_south_y  # shape (nx,)
    north_peak = north_profile_abs - hill_north_y  # shape (nx,)

    # -- Gaussian ridge cross-section in y ---------------------------------
    # sigma_y: half the distance between the ridge centre and the valley
    # floor ensures the ridge tapers to ~0 near the valley.
    half_valley = abs(hill_north_y - hill_south_y) / 2.0
    sigma_y = half_valley / 2.5  # Gaussian ~0 at ~2.5σ from centre

    # -- Build terrain height map H(x, y) → (ny, nx) ----------------------
    # south_gauss[iy] = exp(...)   for each y value
    south_gauss = np.exp(
        -0.5 * ((y - hill_south_y) / sigma_y) ** 2
    )  # (ny,)
    north_gauss = np.exp(
        -0.5 * ((y - hill_north_y) / sigma_y) ** 2
    )  # (ny,)

    # H_south(iy, ix) = south_peak[ix] * south_gauss[iy]
    H_south = south_gauss[:, np.newaxis] * south_peak[np.newaxis, :]  # (ny, nx)
    H_north = north_gauss[:, np.newaxis] * north_peak[np.newaxis, :]  # (ny, nx)
    terrain_height = np.maximum(np.maximum(H_south, H_north), 0.0)   # (ny, nx)

    # -- Fill velocity array (nz, ny, nx) ----------------------------------
    values = np.full((nz, ny, nx), air_velocity, dtype=np.float64)
    attenuation = np.zeros((nz, ny, nx), dtype=np.float64)

    for iz, zval in enumerate(z):
        dirt_mask = zval < terrain_height                     # (ny, nx)
        values[iz][dirt_mask] = dirt_velocity

        veg_mask = (
            (zval >= terrain_height)
            & (zval < terrain_height + veg_thickness)
            & (terrain_height > 0.5)   # skip vegetation on flat ground
        )
        attenuation[iz][veg_mask] = veg_attenuation

    model = VelocityModel3D(x=x, y=y, z=z, values=values,
                             dx=dx, dy=dx, dz=dx)
    meta = DomainMeta3D(
        attenuation=attenuation,
        description="3D Hills + vegetation (valley between two ridges)",
    )
    return model, meta

"""3D domain builders for FDTD simulations.

Each builder returns ``(VelocityModel3D, DomainMeta3D)`` — the 3D velocity
field plus metadata describing wind, attenuation zones, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_sim.model_3d import VelocityModel3D, create_uniform_model_3d


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

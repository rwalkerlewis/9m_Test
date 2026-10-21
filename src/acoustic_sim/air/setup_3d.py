"""3-D counterparts of ``air/setup.py``: domain, receiver array, time step and source builders for
:class:`~acoustic_sim.solver.fdtd.FDTD3DSolver` (used by ``examples/air/01_simulation/run_fdtd_3d.py`` and
``ml/fno_data_gen.py``)."""

from __future__ import annotations

import math

import numpy as np

from acoustic_sim.air.domains import (
    DomainMeta3D,
    create_ground_layer_domain_3d,
    create_hills_vegetation_domain_3d,
    create_isotropic_domain_3d,
    create_wind_domain_3d,
)
from acoustic_sim.solver.fdtd import fd2_coefficients, fd2_cfl_factor, FDTD3DSolver
from acoustic_sim.model import VelocityModel3D
from acoustic_sim.receivers import (
    create_receiver_circle_3d,
    create_receiver_line_3d,
    create_receiver_concentric,
    create_receiver_l_shaped,
    create_receiver_random,
)
from acoustic_sim.sources import (
    make_drone_harmonics,
    make_source_from_file,
    make_source_noise,
    make_source_propeller,
    make_source_tone,
    make_wavelet_ricker,
    MovingSource3D,
    StaticSource3D,
)


# ---------------------------------------------------------------------------
# Domain builder
# ---------------------------------------------------------------------------

def build_domain_3d(
    domain: str = "isotropic",
    *,
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    z_min: float = -5.0,
    z_max: float = 50.0,
    dx: float = 1.0,
    velocity: float = 343.0,
    wind_speed: float = 0.0,
    wind_direction_deg: float = 0.0,
    dirt_velocity: float = 1500.0,
    seed: int = 42,
) -> tuple[VelocityModel3D, DomainMeta3D]:
    """Dispatch to the appropriate 3-D domain builder."""
    grid_kw = dict(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_min, z_max=z_max,
        dx=dx,
    )
    if domain == "isotropic":
        return create_isotropic_domain_3d(velocity=velocity, **grid_kw)
    if domain == "wind":
        return create_wind_domain_3d(
            velocity=velocity,
            wind_speed=wind_speed,
            wind_direction_deg=wind_direction_deg,
            **grid_kw,
        )
    if domain == "hills_vegetation":
        return create_hills_vegetation_domain_3d(
            air_velocity=velocity,
            dirt_velocity=dirt_velocity,
            seed=seed,
            **grid_kw,
        )
    if domain == "ground_layer":
        return create_ground_layer_domain_3d(
            air_velocity=velocity,
            ground_velocity=dirt_velocity,
            ground_z=0.0,
            **grid_kw,
        )
    raise ValueError(f"Unknown 3D domain: {domain!r}")


# ---------------------------------------------------------------------------
# Receiver builder
# ---------------------------------------------------------------------------

def build_receivers_3d(
    array: str = "circular",
    *,
    count: int = 16,
    radius: float = 2.0,
    radii: list[float] | None = None,
    x0: float = -40.0,
    y0: float = 0.0,
    x1: float = 40.0,
    y1: float = 0.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
    spacing: float = 3.0,
    n1: int = 8,
    n2: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Build a 3-D receiver array.  Returns shape ``(n_recv, 3)``."""
    if radii is None:
        radii = [10.0, 20.0, 30.0, 40.0]

    if array == "circular":
        return create_receiver_circle_3d(
            center_x, center_y, radius, count, z=center_z,
        )
    if array == "linear":
        return create_receiver_line_3d(
            x0, y0, x1, y1, count, z=center_z,
        )
    if array == "concentric":
        pos_2d = create_receiver_concentric(
            center_x, center_y, radii, count,
        )
        z_arr = np.full(pos_2d.shape[0], center_z)
        return np.column_stack([pos_2d, z_arr])
    if array == "l_shaped":
        pos_2d = create_receiver_l_shaped(n1, n2, spacing, center_x, center_y)
        z_arr = np.full(pos_2d.shape[0], center_z)
        return np.column_stack([pos_2d, z_arr])
    if array == "random":
        pos_2d = create_receiver_random(count, x0, x1, y0, y1, seed=seed)
        z_arr = np.full(pos_2d.shape[0], center_z)
        return np.column_stack([pos_2d, z_arr])
    raise ValueError(f"Unknown 3D array type: {array!r}")


# ---------------------------------------------------------------------------
# CFL / dt for 3-D
# ---------------------------------------------------------------------------

def compute_dt_3d(
    model: VelocityModel3D,
    meta: DomainMeta3D,
    cfl_safety: float = 0.9,
    fd_order: int = 2,
) -> tuple[float, float]:
    """Return ``(dt, f_max)`` from 3-D CFL and spatial sampling.

    The 3-D CFL limit uses a √3 factor (vs √2 in 2-D) to account
    for the extra spatial dimension.
    """
    coeffs = fd2_coefficients(fd_order)
    spec_radius = fd2_cfl_factor(coeffs)

    c_max = float(np.max(model.values))
    v_wind = math.sqrt(
        meta.wind_vx ** 2 + meta.wind_vy ** 2 + meta.wind_vz ** 2
    )
    dt = (
        cfl_safety
        * 2.0
        * model.dx
        / ((c_max + v_wind) * math.sqrt(3.0 * spec_radius))
    )
    f_max = float(np.min(model.values)) / (10.0 * model.dx)
    return dt, f_max


# ---------------------------------------------------------------------------
# Source builder
# ---------------------------------------------------------------------------

def build_source_3d(
    source_type: str = "static",
    signal_type: str = "ricker",
    *,
    n_steps: int,
    dt: float,
    f_max: float,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    x1: float = 30.0,
    y1: float = 0.0,
    z1: float = 0.0,
    speed: float = 50.0,
    arc_height: float = 0.0,
    freq: float = 25.0,
    blade_count: int = 3,
    rpm: float = 3600.0,
    harmonics: int = 14,
    seed: int = 42,
    wav_path: str = "audio/input.wav",
    max_seconds: float | None = None,
    source_level_dB: float = 90.0,
    n_harmonics: int = 4,
    fundamental_freq: float = 150.0,
) -> StaticSource3D | MovingSource3D:
    """Build a 3-D source object with its signal."""
    sig = _build_signal(
        signal_type,
        n_steps=n_steps, dt=dt, f_max=f_max,
        freq=freq, blade_count=blade_count, rpm=rpm,
        harmonics=harmonics, seed=seed,
        wav_path=wav_path, max_seconds=max_seconds,
        source_level_dB=source_level_dB,
        n_harmonics=n_harmonics,
        fundamental_freq=fundamental_freq,
    )

    if source_type == "static":
        return StaticSource3D(x=x, y=y, z=z, signal=sig)
    if source_type == "moving":
        return MovingSource3D(
            x0=x, y0=y, z0=z,
            x1=x1, y1=y1, z1=z1,
            speed=speed, signal=sig,
            arc_height=arc_height,
        )
    raise ValueError(f"Unknown 3D source type: {source_type!r}")


def _build_signal(
    kind: str,
    *,
    n_steps: int,
    dt: float,
    f_max: float,
    freq: float,
    blade_count: int,
    rpm: float,
    harmonics: int,
    seed: int,
    wav_path: str,
    max_seconds: float | None,
    source_level_dB: float = 90.0,
    n_harmonics: int = 4,
    fundamental_freq: float = 150.0,
) -> np.ndarray:
    if kind == "ricker":
        return make_wavelet_ricker(n_steps, dt, freq)
    if kind == "tone":
        return make_source_tone(n_steps, dt, freq)
    if kind == "noise":
        return make_source_noise(n_steps, dt, f_low=5.0, f_high=f_max, seed=seed)
    if kind == "propeller":
        return make_source_propeller(
            n_steps, dt, f_max=f_max,
            blade_count=blade_count, rpm=rpm,
            harmonics=harmonics, seed=seed,
        )
    if kind == "file":
        return make_source_from_file(
            wav_path, n_steps, dt, f_max, max_seconds=max_seconds,
        )
    if kind == "drone_harmonics":
        return make_drone_harmonics(
            n_steps, dt,
            fundamental=fundamental_freq,
            n_harmonics=n_harmonics,
            source_level_dB=source_level_dB,
            f_max=f_max,
        )
    raise ValueError(f"Unknown signal type: {kind!r}")

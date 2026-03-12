"""Simulation setup helpers.

Construct domain, source, and receiver objects from plain parameters.
These functions centralise the CLI-argument-to-object mapping so that
``run_fdtd.py`` stays a thin script and the same builders can be
reused from notebooks or other entry points.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from acoustic_sim.domains import (
    DomainMeta,
    create_echo_canyon_domain,
    create_hills_vegetation_domain,
    create_isotropic_domain,
    create_urban_echo_domain,
    create_wind_domain,
)
from acoustic_sim.model import VelocityModel
from acoustic_sim.receivers import (
    create_receiver_circle,
    create_receiver_concentric,
    create_receiver_custom,
    create_receiver_l_shaped,
    create_receiver_line,
    create_receiver_random,
)
from acoustic_sim.sources import (
    CircularOrbitSource,
    CustomTrajectorySource,
    EvasiveSource,
    FigureEightSource,
    LoiterApproachSource,
    MovingSource,
    StaticSource,
    make_drone_harmonics,
    make_source_from_file,
    make_source_noise,
    make_source_propeller,
    make_source_tone,
    make_stationary_tonal,
    make_wavelet_ricker,
)


# -----------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------

def build_domain(
    domain: str = "isotropic",
    *,
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    dx: float = 0.5,
    velocity: float = 343.0,
    wind_speed: float = 15.0,
    wind_direction_deg: float = 45.0,
    dirt_velocity: float = 1500.0,
    seed: int = 42,
) -> tuple[VelocityModel, DomainMeta]:
    """Build a velocity model and domain metadata.

    Parameters
    ----------
    domain : str
        One of ``"isotropic"``, ``"wind"``, ``"hills_vegetation"``.
    """
    grid_kw: dict[str, Any] = dict(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, dx=dx,
    )
    if domain == "isotropic":
        return create_isotropic_domain(velocity=velocity, **grid_kw)
    if domain == "wind":
        return create_wind_domain(
            velocity=velocity,
            wind_speed=wind_speed,
            wind_direction_deg=wind_direction_deg,
            **grid_kw,
        )
    if domain == "hills_vegetation":
        return create_hills_vegetation_domain(
            air_velocity=velocity,
            dirt_velocity=dirt_velocity,
            seed=seed,
            **grid_kw,
        )
    if domain == "echo_canyon":
        return create_echo_canyon_domain(air_velocity=velocity, **grid_kw)
    if domain == "urban_echo":
        return create_urban_echo_domain(
            air_velocity=velocity, seed=seed, **grid_kw,
        )
    raise ValueError(f"Unknown domain: {domain!r}")


# -----------------------------------------------------------------------
# Receivers
# -----------------------------------------------------------------------

def build_receivers(
    array: str = "circular",
    *,
    count: int = 16,
    radius: float = 15.0,
    radii: Sequence[float] = (10.0, 20.0, 30.0, 40.0),
    x0: float = -40.0,
    y0: float = 0.0,
    x1: float = 40.0,
    y1: float = 0.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
    spacing: float = 3.0,
    n1: int = 8,
    n2: int = 8,
    positions: list[tuple[float, float]] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Build a receiver array.  Returns shape ``(n_recv, 2)``.

    Parameters
    ----------
    array : str
        One of ``"circular"``, ``"concentric"``, ``"linear"``,
        ``"l_shaped"``, ``"random"``, ``"custom"``.
    """
    if array == "circular":
        return create_receiver_circle(center_x, center_y, radius, count)
    if array == "concentric":
        return create_receiver_concentric(center_x, center_y, list(radii), count)
    if array == "linear":
        return create_receiver_line(x0, y0, x1, y1, count)
    if array == "l_shaped":
        return create_receiver_l_shaped(n1, n2, spacing, center_x, center_y)
    if array == "random":
        return create_receiver_random(count, x0, x1, y0, y1, seed=seed)
    if array == "custom":
        if positions is None:
            raise ValueError("array='custom' requires 'positions' argument")
        return create_receiver_custom(positions)
    raise ValueError(f"Unknown array type: {array!r}")


# -----------------------------------------------------------------------
# CFL / dt helper
# -----------------------------------------------------------------------

def compute_dt(
    model: VelocityModel,
    meta: DomainMeta,
    cfl_safety: float = 0.9,
    fd_order: int = 2,
) -> tuple[float, float]:
    """Return ``(dt, f_max)`` from CFL and spatial sampling.

    ``f_max`` is the maximum frequency the grid can resolve at 10
    points per wavelength.
    """
    from acoustic_sim.fdtd import fd2_coefficients, fd2_cfl_factor

    coeffs = fd2_coefficients(fd_order)
    spec_radius = fd2_cfl_factor(coeffs)

    c_max = float(np.max(model.values))
    v_wind = math.sqrt(meta.wind_vx ** 2 + meta.wind_vy ** 2)
    dt = cfl_safety * 2.0 * model.dx / ((c_max + v_wind) * math.sqrt(2.0 * spec_radius))
    f_max = float(np.min(model.values)) / (10.0 * model.dx)
    return dt, f_max


# -----------------------------------------------------------------------
# Source
# -----------------------------------------------------------------------

def build_source(
    source_type: str = "static",
    signal_type: str = "ricker",
    *,
    n_steps: int,
    dt: float,
    f_max: float,
    x: float = 0.0,
    y: float = 0.0,
    x1: float = 30.0,
    y1: float = 0.0,
    speed: float = 50.0,
    arc_height: float = 0.0,
    freq: float = 25.0,
    blade_count: int = 3,
    rpm: float = 3600.0,
    harmonics: int = 14,
    seed: int = 42,
    wav_path: str = "audio/input.wav",
    max_seconds: float | None = None,
    # ── Drone-detection specific ──
    source_level_dB: float = 90.0,
    harmonic_amplitudes: list[float] | None = None,
    n_harmonics: int = 4,
    fundamental_freq: float = 150.0,
    # Circular orbit
    orbit_cx: float = 0.0,
    orbit_cy: float = 50.0,
    orbit_radius: float = 40.0,
    orbit_start_angle: float = 0.0,
    # Figure-eight
    fig8_cx: float = 0.0,
    fig8_cy: float = 50.0,
    fig8_x_amp: float = 40.0,
    fig8_y_amp: float = 20.0,
    fig8_x_freq: float = 0.1,
    fig8_y_freq: float = 0.2,
    fig8_phase_offset: float = 1.5708,
    # Loiter-approach
    loiter_orbit_cx: float = 0.0,
    loiter_orbit_cy: float = 80.0,
    loiter_orbit_radius: float = 30.0,
    loiter_orbit_duration: float = 3.0,
    loiter_approach_x: float = 0.0,
    loiter_approach_y: float = 0.0,
    # Evasive
    evasive_heading: float = 0.0,
    evasive_speed_var: float = 2.0,
    evasive_heading_var: float = 0.3,
    # Custom trajectory
    trajectory_times: np.ndarray | None = None,
    trajectory_positions: np.ndarray | None = None,
    # Stationary tonal
    stationary_n_harmonics: int = 4,
    stationary_broadband: float = 0.1,
) -> StaticSource | MovingSource | CircularOrbitSource | FigureEightSource | LoiterApproachSource | EvasiveSource | CustomTrajectorySource:
    """Build a source object with its signal.

    Parameters
    ----------
    source_type : str
        ``"static"``, ``"moving"``, ``"circular_orbit"``,
        ``"figure_eight"``, ``"loiter_approach"``, ``"evasive"``, or
        ``"custom_trajectory"``.
    signal_type : str
        ``"ricker"``, ``"tone"``, ``"noise"``, ``"propeller"``,
        ``"file"``, ``"drone_harmonics"``, or ``"stationary_tonal"``.
    n_steps, dt, f_max
        Simulation timing — needed to generate the signal at the correct
        sample rate and length.
    """
    sig = _build_signal(
        signal_type, n_steps=n_steps, dt=dt, f_max=f_max,
        freq=freq, blade_count=blade_count, rpm=rpm,
        harmonics=harmonics, seed=seed,
        wav_path=wav_path, max_seconds=max_seconds,
        source_level_dB=source_level_dB,
        harmonic_amplitudes=harmonic_amplitudes,
        n_harmonics=n_harmonics,
        fundamental_freq=fundamental_freq,
        stationary_n_harmonics=stationary_n_harmonics,
        stationary_broadband=stationary_broadband,
    )

    if source_type == "static":
        return StaticSource(x=x, y=y, signal=sig)
    if source_type == "moving":
        return MovingSource(x0=x, y0=y, x1=x1, y1=y1, speed=speed,
                            signal=sig, arc_height=arc_height)
    if source_type == "circular_orbit":
        omega = speed / max(orbit_radius, 1e-6)
        return CircularOrbitSource(
            cx=orbit_cx, cy=orbit_cy, radius=orbit_radius,
            angular_velocity=omega, start_angle=orbit_start_angle, signal=sig,
        )
    if source_type == "figure_eight":
        return FigureEightSource(
            cx=fig8_cx, cy=fig8_cy,
            x_amp=fig8_x_amp, y_amp=fig8_y_amp,
            x_freq=fig8_x_freq, y_freq=fig8_y_freq,
            phase_offset=fig8_phase_offset, signal=sig,
        )
    if source_type == "loiter_approach":
        return LoiterApproachSource(
            orbit_cx=loiter_orbit_cx, orbit_cy=loiter_orbit_cy,
            orbit_radius=loiter_orbit_radius,
            orbit_duration=loiter_orbit_duration,
            approach_target_x=loiter_approach_x,
            approach_target_y=loiter_approach_y,
            approach_speed=speed, signal=sig,
        )
    if source_type == "evasive":
        return EvasiveSource(
            x0=x, y0=y, heading=evasive_heading,
            mean_speed=speed, speed_var=evasive_speed_var,
            heading_var=evasive_heading_var, signal=sig, seed=seed,
        )
    if source_type == "custom_trajectory":
        if trajectory_times is None or trajectory_positions is None:
            raise ValueError(
                "source_type='custom_trajectory' requires "
                "trajectory_times and trajectory_positions"
            )
        return CustomTrajectorySource(
            times=trajectory_times, positions=trajectory_positions, signal=sig,
        )
    raise ValueError(f"Unknown source type: {source_type!r}")


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
    harmonic_amplitudes: list[float] | None = None,
    n_harmonics: int = 4,
    fundamental_freq: float = 150.0,
    stationary_n_harmonics: int = 4,
    stationary_broadband: float = 0.1,
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
        return make_source_from_file(wav_path, n_steps, dt, f_max,
                                     max_seconds=max_seconds)
    if kind == "drone_harmonics":
        return make_drone_harmonics(
            n_steps, dt,
            fundamental=fundamental_freq,
            n_harmonics=n_harmonics,
            harmonic_amplitudes=harmonic_amplitudes,
            source_level_dB=source_level_dB,
            f_max=f_max,
        )
    if kind == "stationary_tonal":
        return make_stationary_tonal(
            n_steps, dt,
            base_freq=freq,
            n_harmonics=stationary_n_harmonics,
            source_level_dB=source_level_dB,
            broadband_level=stationary_broadband,
            f_max=f_max,
            seed=seed,
        )
    raise ValueError(f"Unknown signal type: {kind!r}")

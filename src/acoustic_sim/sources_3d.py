"""3D source signal generators and trajectory helpers.

Extends the 2D source classes from ``sources.py`` with a z-coordinate.
Every source class exposes ``position_at(step, dt) -> (x, y, z)``.

When all z-coordinates are zero, outputs are identical to the 2D classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from acoustic_sim.sources import (
    _P_REF,
    _normalize,
    load_wav_mono,
    make_drone_harmonics,
    make_source_from_file,
    make_source_noise,
    make_source_propeller,
    make_source_tone,
    make_stationary_tonal,
    make_wavelet_ricker,
    prepare_source_signal,
)


# ---------------------------------------------------------------------------
# 3D Source position helpers
# ---------------------------------------------------------------------------

@dataclass
class StaticSource3D:
    """Fixed-position point source in 3D."""
    x: float
    y: float
    z: float
    signal: np.ndarray

    def position_at(self, step: int, dt: float = 0.0) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class MovingSource3D:
    """Linear trajectory from (x0,y0,z0) to (x1,y1,z1) at constant speed.

    z interpolates linearly from z0 to z1.
    """
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    speed: float
    signal: np.ndarray
    arc_height: float = 0.0

    def position_at(self, step: int, dt: float) -> tuple[float, float, float]:
        dx = self.x1 - self.x0
        dy = self.y1 - self.y0
        dz = self.z1 - self.z0
        dist_total = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist_total < 1e-12:
            return (self.x0, self.y0, self.z0)
        t = step * dt
        frac = min((self.speed * t) / dist_total, 1.0)
        x = self.x0 + frac * dx
        y = self.y0 + frac * dy
        z = self.z0 + frac * dz
        # Parabolic arc on y (preserving 2D behavior).
        y += self.arc_height * 4.0 * frac * (1.0 - frac)
        return (x, y, z)


@dataclass
class CircularOrbitSource3D:
    """Drone orbiting at constant altitude.

    z is constant (configurable altitude, default 50m).
    """
    cx: float
    cy: float
    radius: float
    angular_velocity: float
    start_angle: float
    signal: np.ndarray
    altitude: float = 50.0

    def position_at(self, step: int, dt: float) -> tuple[float, float, float]:
        t = step * dt
        theta = self.start_angle + self.angular_velocity * t
        x = self.cx + self.radius * math.cos(theta)
        y = self.cy + self.radius * math.sin(theta)
        return (x, y, self.altitude)


@dataclass
class FigureEightSource3D:
    """Lissajous figure-eight trajectory with optional z oscillation.

    z is constant or has a configurable sinusoidal variation.
    """
    cx: float
    cy: float
    x_amp: float
    y_amp: float
    x_freq: float
    y_freq: float
    phase_offset: float
    signal: np.ndarray
    altitude: float = 50.0
    z_amplitude: float = 0.0
    z_freq: float = 0.0

    def position_at(self, step: int, dt: float) -> tuple[float, float, float]:
        t = step * dt
        x = self.cx + self.x_amp * math.sin(2.0 * math.pi * self.x_freq * t)
        y = self.cy + self.y_amp * math.sin(
            2.0 * math.pi * self.y_freq * t + self.phase_offset
        )
        z = self.altitude
        if self.z_amplitude > 0 and self.z_freq > 0:
            z += self.z_amplitude * math.sin(2.0 * math.pi * self.z_freq * t)
        return (x, y, z)


@dataclass
class LoiterApproachSource3D:
    """Drone orbits at standoff altitude, then descends on approach.

    During orbit phase: z = orbit_altitude (constant).
    During approach phase: z descends linearly at descent_rate m/s.
    """
    orbit_cx: float
    orbit_cy: float
    orbit_radius: float
    orbit_duration: float
    approach_target_x: float
    approach_target_y: float
    approach_speed: float
    signal: np.ndarray
    orbit_altitude: float = 50.0
    approach_target_z: float = 0.0
    descent_rate: float = 2.0

    def position_at(self, step: int, dt: float) -> tuple[float, float, float]:
        t = step * dt
        omega = self.approach_speed / max(self.orbit_radius, 1e-6)

        if t <= self.orbit_duration:
            theta = omega * t
            x = self.orbit_cx + self.orbit_radius * math.cos(theta)
            y = self.orbit_cy + self.orbit_radius * math.sin(theta)
            return (x, y, self.orbit_altitude)

        # Transition point.
        theta_end = omega * self.orbit_duration
        x_start = self.orbit_cx + self.orbit_radius * math.cos(theta_end)
        y_start = self.orbit_cy + self.orbit_radius * math.sin(theta_end)

        # Approach phase: linear toward target.
        dx = self.approach_target_x - x_start
        dy = self.approach_target_y - y_start
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return (x_start, y_start, self.orbit_altitude)

        dt_approach = t - self.orbit_duration
        frac = min((self.approach_speed * dt_approach) / dist, 1.0)
        x = x_start + frac * dx
        y = y_start + frac * dy

        # Altitude: descend linearly.
        z = self.orbit_altitude - self.descent_rate * dt_approach
        z = max(z, self.approach_target_z)
        return (x, y, z)


@dataclass
class EvasiveSource3D:
    """Random-walk heading with altitude variation.

    z has random walk with configurable variance, superimposed on mean
    altitude.
    """
    x0: float
    y0: float
    z0: float
    heading: float
    mean_speed: float
    speed_var: float
    heading_var: float
    signal: np.ndarray
    mean_altitude: float = 50.0
    z_variance: float = 5.0
    seed: int = 42
    _xs: np.ndarray = dc_field(default_factory=lambda: np.empty(0), repr=False)
    _ys: np.ndarray = dc_field(default_factory=lambda: np.empty(0), repr=False)
    _zs: np.ndarray = dc_field(default_factory=lambda: np.empty(0), repr=False)
    _dt_cache: float = 0.0

    def _build_path(self, n_steps: int, dt: float) -> None:
        rng = np.random.default_rng(self.seed)

        heading_increments = rng.normal(0.0, self.heading_var * dt, n_steps)
        speed_perturbations = rng.normal(0.0, self.speed_var, n_steps)
        z_increments = rng.normal(0.0, self.z_variance * dt, n_steps)

        headings = self.heading + np.cumsum(heading_increments)
        speeds = np.maximum(self.mean_speed + speed_perturbations, 0.0)

        vx = speeds * np.cos(headings) * dt
        vy = speeds * np.sin(headings) * dt

        self._xs = self.x0 + np.cumsum(vx)
        self._ys = self.y0 + np.cumsum(vy)
        self._zs = self.mean_altitude + np.cumsum(z_increments)
        # Clamp z to positive.
        self._zs = np.maximum(self._zs, 0.0)

        self._xs = np.concatenate([[self.x0], self._xs[:-1]])
        self._ys = np.concatenate([[self.y0], self._ys[:-1]])
        self._zs = np.concatenate([[self.z0], self._zs[:-1]])
        self._dt_cache = dt

    def position_at(self, step: int, dt: float) -> tuple[float, float, float]:
        if len(self._xs) == 0 or abs(self._dt_cache - dt) > 1e-15:
            self._build_path(len(self.signal), dt)
        idx = min(step, len(self._xs) - 1)
        return (float(self._xs[idx]), float(self._ys[idx]), float(self._zs[idx]))


@dataclass
class CustomTrajectorySource3D:
    """User-supplied trajectory as (t, x, y, z) arrays.

    Positions are linearly interpolated between the supplied waypoints.
    """
    times: np.ndarray
    positions: np.ndarray  # (N, 3) — x, y, z
    signal: np.ndarray

    def position_at(self, step: int, dt: float) -> tuple[float, float, float]:
        t = step * dt
        x = float(np.interp(t, self.times, self.positions[:, 0]))
        y = float(np.interp(t, self.times, self.positions[:, 1]))
        z = float(np.interp(t, self.times, self.positions[:, 2]))
        return (x, y, z)


# ---------------------------------------------------------------------------
# 3D velocity helper
# ---------------------------------------------------------------------------

def source_velocity_at_3d(
    source: object,
    step: int,
    dt: float,
    eps_steps: int = 1,
) -> tuple[float, float, float]:
    """Compute instantaneous 3D velocity via central finite differences.

    Returns ``(vx, vy, vz)`` in m/s.
    """
    s0 = max(step - eps_steps, 0)
    s1 = step + eps_steps
    x0, y0, z0 = source.position_at(s0, dt)  # type: ignore[union-attr]
    x1, y1, z1 = source.position_at(s1, dt)  # type: ignore[union-attr]
    delta_t = (s1 - s0) * dt
    if delta_t < 1e-30:
        return (0.0, 0.0, 0.0)
    return ((x1 - x0) / delta_t, (y1 - y0) / delta_t, (z1 - z0) / delta_t)

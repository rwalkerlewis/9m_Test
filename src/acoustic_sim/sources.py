"""Source signal generators and source-position helpers for FDTD.

Supports audio-file sources (WAV), synthetic propeller / tone / noise
generators, drone harmonic signals, and the classic Ricker wavelet.
Each source produces a 1-D time-series that the FDTD solver injects at
the source grid location.

Trajectory helpers
==================
Every source class exposes ``position_at(step, dt) -> (x, y)``.
The standalone helper ``source_velocity_at`` computes instantaneous
velocity via finite differences for *any* source that implements
``position_at``.

Available trajectory types:

* ``StaticSource`` — fixed position
* ``MovingSource`` — linear A→B with optional parabolic arc
* ``CircularOrbitSource`` — circular orbit
* ``FigureEightSource`` — Lissajous figure-eight
* ``LoiterApproachSource`` — orbit then linear approach
* ``EvasiveSource`` — random-walk heading overlaid on mean course
* ``CustomTrajectorySource`` — user-supplied (t, x, y) arrays
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy import signal as sp_signal
from scipy.io import wavfile

# Reference pressure in air (20 µPa, threshold of hearing).
_P_REF = 20e-6  # Pa


# ---------------------------------------------------------------------------
# WAV loading
# ---------------------------------------------------------------------------

def _pcm_to_float32(data: np.ndarray) -> np.ndarray:
    """Convert PCM integer audio to float32 in [-1, 1]."""
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32)
    if data.dtype == np.int16:
        return (data.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if data.dtype == np.int32:
        return (data.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    if data.dtype == np.uint8:
        return ((data.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    return data.astype(np.float32)


def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file and return ``(mono_float32, sample_rate)``."""
    fs, data = wavfile.read(path)
    audio = _pcm_to_float32(data)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, int(fs)


# ---------------------------------------------------------------------------
# Signal conditioning for the FD grid
# ---------------------------------------------------------------------------

def _normalize(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    mx = np.max(np.abs(x))
    if mx > 1e-12:
        x = x * (peak / mx)
    return x


def prepare_source_signal(
    raw: np.ndarray,
    fs_audio: int,
    dt_sim: float,
    f_max: float,
) -> np.ndarray:
    """Low-pass filter and resample an audio signal for the FD simulation.

    Parameters
    ----------
    raw : 1-D float array
        Source audio at *fs_audio* Hz.
    fs_audio : int
        Original sample rate of *raw*.
    dt_sim : float
        FDTD timestep in seconds.
    f_max : float
        Maximum frequency the grid can resolve (``c_min / (ppw * dx)``).

    Returns
    -------
    np.ndarray
        Resampled & filtered signal at rate ``1 / dt_sim``, peak-normalised.
    """
    fs_sim = 1.0 / dt_sim

    # Low-pass at f_max (must be below Nyquist of *both* sample rates).
    nyq_audio = fs_audio / 2.0
    nyq_sim = fs_sim / 2.0
    cutoff = min(f_max, nyq_audio * 0.95, nyq_sim * 0.95)
    if cutoff > 1.0:
        sos = sp_signal.butter(6, cutoff, btype="low", fs=fs_audio, output="sos")
        raw = sp_signal.sosfilt(sos, raw).astype(np.float32)

    # Resample: rational approximation of fs_sim / fs_audio.
    from math import gcd

    fs_sim_int = int(round(fs_sim))
    g = gcd(fs_sim_int, fs_audio)
    up, down = fs_sim_int // g, fs_audio // g
    # Clamp the poly-phase factors so memory stays bounded.
    MAX_FACTOR = 512
    if up > MAX_FACTOR or down > MAX_FACTOR:
        # Fall back to scipy.signal.resample (FFT-based).
        n_out = int(len(raw) * fs_sim / fs_audio)
        resampled = sp_signal.resample(raw, max(n_out, 1)).astype(np.float64)
    else:
        resampled = sp_signal.resample_poly(raw, up, down).astype(np.float64)

    return _normalize(resampled)


# ---------------------------------------------------------------------------
# Synthetic source generators
# ---------------------------------------------------------------------------

def make_wavelet_ricker(n_steps: int, dt: float, f0: float) -> np.ndarray:
    """Ricker (Mexican-hat) wavelet centred at *t0 = 1.5 / f0*.

    Parameters
    ----------
    n_steps : int
        Number of simulation time-steps.
    dt : float
        Time-step size [s].
    f0 : float
        Peak frequency [Hz].
    """
    t = np.arange(n_steps) * dt
    t0 = 1.5 / f0
    u = (np.pi * f0 * (t - t0)) ** 2
    return ((1.0 - 2.0 * u) * np.exp(-u)).astype(np.float64)


def make_source_propeller(
    n_steps: int,
    dt: float,
    f_max: float | None = None,
    blade_count: int = 3,
    rpm: float = 3600.0,
    harmonics: int = 14,
    mod_depth: float = 0.25,
    broadband_level: float = 0.12,
    seed: int = 42,
) -> np.ndarray:
    """Synthetic propeller / rotor noise.

    Blade-pass-frequency harmonics with amplitude modulation and broadband
    turbulent noise — the same model used in the legacy ``simulate_array``
    code.
    """
    fs_sim = 1.0 / dt
    t = np.arange(n_steps) * dt
    bpf = blade_count * rpm / 60.0
    rotor_freq = rpm / 60.0

    sig = np.zeros(n_steps, dtype=np.float64)
    for h in range(1, harmonics + 1):
        freq_h = bpf * h
        if f_max is not None and freq_h > f_max:
            break
        amp = 1.0 / math.sqrt(h)
        sig += amp * np.sin(2.0 * np.pi * freq_h * t + 0.17 * h)

    # Amplitude modulation at rotor frequency.
    mod = 1.0 + mod_depth * np.sin(2.0 * np.pi * rotor_freq * t)
    sig *= mod

    # Broadband turbulent noise component.
    rng = np.random.default_rng(seed)
    bb = rng.standard_normal(n_steps)
    nyq = fs_sim / 2.0
    hi = min(8000.0, nyq * 0.95)
    lo = min(100.0, hi * 0.5)
    if hi > lo > 0:
        sos = sp_signal.butter(2, [lo, hi], btype="bandpass", fs=fs_sim, output="sos")
        bb = sp_signal.sosfilt(sos, bb)
    sig += broadband_level * bb

    return _normalize(sig)


def make_source_tone(
    n_steps: int,
    dt: float,
    frequency_hz: float = 40.0,
) -> np.ndarray:
    """Pure sine-wave source at *frequency_hz*."""
    t = np.arange(n_steps) * dt
    return np.sin(2.0 * np.pi * frequency_hz * t).astype(np.float64)


def make_source_noise(
    n_steps: int,
    dt: float,
    f_low: float = 5.0,
    f_high: float = 60.0,
    seed: int = 42,
) -> np.ndarray:
    """Band-limited coloured noise source."""
    fs_sim = 1.0 / dt
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n_steps)
    nyq = fs_sim / 2.0
    hi = min(f_high, nyq * 0.95)
    lo = min(f_low, hi * 0.5)
    if hi > lo > 0:
        sos = sp_signal.butter(4, [lo, hi], btype="bandpass", fs=fs_sim, output="sos")
        noise = sp_signal.sosfilt(sos, noise)
    return _normalize(noise.astype(np.float64))


def make_source_from_file(
    path: str,
    n_steps: int,
    dt: float,
    f_max: float,
    max_seconds: float | None = None,
) -> np.ndarray:
    """Load a WAV file and prepare it for FD injection.

    The audio is low-pass filtered at *f_max*, resampled to ``1/dt``, and
    padded or truncated to *n_steps*.
    """
    audio, fs = load_wav_mono(path)
    if max_seconds is not None:
        audio = audio[: int(max_seconds * fs)]
    sig = prepare_source_signal(audio, fs, dt, f_max)
    # Pad or truncate to n_steps.
    if len(sig) >= n_steps:
        sig = sig[:n_steps]
    else:
        sig = np.pad(sig, (0, n_steps - len(sig)))
    return sig


def make_drone_harmonics(
    n_steps: int,
    dt: float,
    fundamental: float = 150.0,
    n_harmonics: int = 4,
    harmonic_amplitudes: list[float] | None = None,
    source_level_dB: float = 90.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Synthetic drone rotor signal as a sum of harmonics.

    Parameters
    ----------
    n_steps : int
        Number of simulation time-steps.
    dt : float
        Time-step size [s].
    fundamental : float
        Fundamental rotor frequency [Hz], default 150 Hz.
    n_harmonics : int
        Number of harmonics (1 = fundamental only).
    harmonic_amplitudes : list[float] or None
        Relative amplitude weights for each harmonic.
        Default ``[1.0, 0.6, 0.3, 0.15]``.
    source_level_dB : float
        Source level in dB re 20 µPa at 1 m, default 90 dB.
    f_max : float or None
        Maximum resolvable frequency — harmonics above this are skipped.

    Returns
    -------
    np.ndarray
        Signal of length *n_steps* with peak amplitude equal to the
        physical source pressure in Pascals.  Use with
        ``FDTDConfig(source_amplitude=1.0)`` so the FDTD injects the
        signal directly in Pascals.

    Notes
    -----
    At 90 dB re 20 µPa the source pressure is::

        p_source = 20e-6 * 10^(90/20) ≈ 0.632 Pa

    At 100 m (1/r decay) the received level is ≈ 50 dB.
    """
    if harmonic_amplitudes is None:
        harmonic_amplitudes = [1.0, 0.6, 0.3, 0.15]
    # Extend or truncate amplitude list to n_harmonics.
    amps = list(harmonic_amplitudes)
    while len(amps) < n_harmonics:
        amps.append(amps[-1] * 0.5)
    amps = amps[:n_harmonics]

    t = np.arange(n_steps) * dt
    sig = np.zeros(n_steps, dtype=np.float64)
    for k in range(n_harmonics):
        freq_k = fundamental * (k + 1)
        if f_max is not None and freq_k > f_max:
            break
        sig += amps[k] * np.sin(2.0 * np.pi * freq_k * t)

    # Scale so peak equals physical source pressure.
    p_source = _P_REF * 10.0 ** (source_level_dB / 20.0)
    mx = np.max(np.abs(sig))
    if mx > 1e-30:
        sig *= p_source / mx

    return sig


def make_stationary_tonal(
    n_steps: int,
    dt: float,
    base_freq: float = 60.0,
    n_harmonics: int = 4,
    source_level_dB: float = 70.0,
    broadband_level: float = 0.1,
    f_max: float | None = None,
    seed: int = 99,
) -> np.ndarray:
    """Tonal signal for a stationary coherent noise source.

    Generates harmonics at *base_freq*, 2·base_freq, … plus broadband
    noise, scaled to physical Pascals.
    """
    t = np.arange(n_steps) * dt
    sig = np.zeros(n_steps, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        freq_k = base_freq * k
        if f_max is not None and freq_k > f_max:
            break
        amp = 1.0 / k
        sig += amp * np.sin(2.0 * np.pi * freq_k * t)

    # Broadband component.
    rng = np.random.default_rng(seed)
    fs_sim = 1.0 / dt
    bb = rng.standard_normal(n_steps)
    nyq = fs_sim / 2.0
    hi = min(base_freq * (n_harmonics + 2), nyq * 0.95)
    lo = max(base_freq * 0.5, 1.0)
    if hi > lo:
        sos = sp_signal.butter(2, [lo, hi], btype="bandpass", fs=fs_sim, output="sos")
        bb = sp_signal.sosfilt(sos, bb)
    sig += broadband_level * bb / max(np.max(np.abs(bb)), 1e-30)

    p_source = _P_REF * 10.0 ** (source_level_dB / 20.0)
    mx = np.max(np.abs(sig))
    if mx > 1e-30:
        sig *= p_source / mx
    return sig


# ---------------------------------------------------------------------------
# Source position helpers
# ---------------------------------------------------------------------------

@dataclass
class StaticSource:
    """Fixed-position point source."""

    x: float
    y: float
    signal: np.ndarray  # length = n_steps

    def position_at(self, step: int, dt: float = 0.0) -> tuple[float, float]:  # noqa: ARG002
        return (self.x, self.y)


@dataclass
class MovingSource:
    """Moving point source with optional parabolic arc.

    Travels from ``(x0, y0)`` to ``(x1, y1)`` at *speed* m/s.  When
    ``arc_height != 0`` a parabolic y-offset is added, peaking at the
    path midpoint.  After reaching the end point the source stays there.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    speed: float  # m/s
    signal: np.ndarray
    arc_height: float = 0.0

    def position_at(self, step: int, dt: float) -> tuple[float, float]:
        dist_total = math.hypot(self.x1 - self.x0, self.y1 - self.y0)
        if dist_total < 1e-12:
            return (self.x0, self.y0)
        t = step * dt
        frac = min((self.speed * t) / dist_total, 1.0)
        x = self.x0 + frac * (self.x1 - self.x0)
        y = self.y0 + frac * (self.y1 - self.y0)
        # Parabolic arc: peaks at frac=0.5
        y += self.arc_height * 4.0 * frac * (1.0 - frac)
        return (x, y)


@dataclass
class CircularOrbitSource:
    """Drone orbiting a centre point at constant angular velocity.

    Parameters
    ----------
    cx, cy : float
        Orbit centre.
    radius : float
        Orbit radius [m].
    angular_velocity : float
        Angular speed [rad/s].  Alternatively set *speed* and compute
        ``angular_velocity = speed / radius``.
    start_angle : float
        Starting angle in radians (0 = +x axis).
    signal : np.ndarray
        Source waveform (length = n_steps).
    """

    cx: float
    cy: float
    radius: float
    angular_velocity: float
    start_angle: float
    signal: np.ndarray

    def position_at(self, step: int, dt: float) -> tuple[float, float]:
        t = step * dt
        theta = self.start_angle + self.angular_velocity * t
        x = self.cx + self.radius * math.cos(theta)
        y = self.cy + self.radius * math.sin(theta)
        return (x, y)


@dataclass
class FigureEightSource:
    """Lissajous figure-eight trajectory.

    Parameters
    ----------
    cx, cy : float
        Trajectory centre.
    x_amp, y_amp : float
        Amplitude along each axis [m].
    x_freq, y_freq : float
        Angular frequencies (in cycles/s) for each axis.
    phase_offset : float
        Phase offset for the y-axis [rad].
    signal : np.ndarray
        Source waveform.
    """

    cx: float
    cy: float
    x_amp: float
    y_amp: float
    x_freq: float
    y_freq: float
    phase_offset: float
    signal: np.ndarray

    def position_at(self, step: int, dt: float) -> tuple[float, float]:
        t = step * dt
        x = self.cx + self.x_amp * math.sin(2.0 * math.pi * self.x_freq * t)
        y = self.cy + self.y_amp * math.sin(
            2.0 * math.pi * self.y_freq * t + self.phase_offset
        )
        return (x, y)


@dataclass
class LoiterApproachSource:
    """Drone orbits at standoff, then transitions to linear approach.

    During ``t < orbit_duration`` the drone circles at
    ``(orbit_cx, orbit_cy)`` with the given radius and speed.
    After the orbit phase it flies a straight line toward the approach
    target at ``approach_speed``.

    Parameters
    ----------
    orbit_cx, orbit_cy : float
        Orbit centre.
    orbit_radius : float
        Orbit radius [m].
    orbit_duration : float
        Time spent orbiting [s].
    approach_target_x, approach_target_y : float
        Point the drone flies toward after the orbit phase.
    approach_speed : float
        Linear approach speed [m/s].
    signal : np.ndarray
        Source waveform.
    """

    orbit_cx: float
    orbit_cy: float
    orbit_radius: float
    orbit_duration: float
    approach_target_x: float
    approach_target_y: float
    approach_speed: float
    signal: np.ndarray

    def position_at(self, step: int, dt: float) -> tuple[float, float]:
        t = step * dt
        omega = self.approach_speed / max(self.orbit_radius, 1e-6)

        if t <= self.orbit_duration:
            # Orbiting phase.
            theta = omega * t
            x = self.orbit_cx + self.orbit_radius * math.cos(theta)
            y = self.orbit_cy + self.orbit_radius * math.sin(theta)
            return (x, y)

        # Transition point (position at end of orbit).
        theta_end = omega * self.orbit_duration
        x_start = self.orbit_cx + self.orbit_radius * math.cos(theta_end)
        y_start = self.orbit_cy + self.orbit_radius * math.sin(theta_end)

        # Approach phase: linear toward target.
        dx = self.approach_target_x - x_start
        dy = self.approach_target_y - y_start
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return (x_start, y_start)

        dt_approach = t - self.orbit_duration
        frac = min((self.approach_speed * dt_approach) / dist, 1.0)
        x = x_start + frac * dx
        y = y_start + frac * dy
        return (x, y)


@dataclass
class EvasiveSource:
    """Random-walk heading overlaid on a general course.

    The trajectory is pre-computed on construction via ``__post_init__``
    to ensure reproducibility when many FDTD steps query the same path.

    Parameters
    ----------
    x0, y0 : float
        Start position.
    heading : float
        General heading [rad].
    mean_speed : float
        Mean forward speed [m/s].
    speed_var : float
        Standard deviation of speed perturbation [m/s].
    heading_var : float
        Standard deviation of heading perturbation per second [rad/s].
    signal : np.ndarray
        Source waveform.
    seed : int
        RNG seed.
    _dt_cache : float
        Internal — timestep used to pre-compute the path (set automatically).
    """

    x0: float
    y0: float
    heading: float
    mean_speed: float
    speed_var: float
    heading_var: float
    signal: np.ndarray
    seed: int = 42
    # Cached trajectory (filled by __post_init__).
    _xs: np.ndarray = dc_field(default_factory=lambda: np.empty(0), repr=False)
    _ys: np.ndarray = dc_field(default_factory=lambda: np.empty(0), repr=False)
    _dt_cache: float = 0.0

    def _build_path(self, n_steps: int, dt: float) -> None:
        """Pre-compute the random-walk trajectory (vectorised)."""
        rng = np.random.default_rng(self.seed)

        # Generate all random increments at once.
        heading_increments = rng.normal(0.0, self.heading_var * dt, n_steps)
        speed_perturbations = rng.normal(0.0, self.speed_var, n_steps)

        # Cumulative heading.
        headings = self.heading + np.cumsum(heading_increments)
        speeds = np.maximum(self.mean_speed + speed_perturbations, 0.0)

        # Velocity components.
        vx = speeds * np.cos(headings) * dt
        vy = speeds * np.sin(headings) * dt

        # Cumulative position from start.
        self._xs = self.x0 + np.cumsum(vx)
        self._ys = self.y0 + np.cumsum(vy)
        # Shift so index 0 is the start position.
        self._xs = np.concatenate([[self.x0], self._xs[:-1]])
        self._ys = np.concatenate([[self.y0], self._ys[:-1]])
        self._dt_cache = dt

    def position_at(self, step: int, dt: float) -> tuple[float, float]:
        # Lazy build on first call (or if dt changed).
        if len(self._xs) == 0 or abs(self._dt_cache - dt) > 1e-15:
            self._build_path(len(self.signal), dt)
        idx = min(step, len(self._xs) - 1)
        return (float(self._xs[idx]), float(self._ys[idx]))


@dataclass
class CustomTrajectorySource:
    """User-supplied trajectory as (t, x, y) arrays.

    Positions are linearly interpolated between the supplied waypoints.

    Parameters
    ----------
    times : np.ndarray
        1-D array of time values [s].
    positions : np.ndarray
        2-D array shape ``(N, 2)`` of (x, y) positions.
    signal : np.ndarray
        Source waveform.
    """

    times: np.ndarray
    positions: np.ndarray
    signal: np.ndarray

    def position_at(self, step: int, dt: float) -> tuple[float, float]:
        t = step * dt
        x = float(np.interp(t, self.times, self.positions[:, 0]))
        y = float(np.interp(t, self.times, self.positions[:, 1]))
        return (x, y)


# ---------------------------------------------------------------------------
# Velocity helper (works with any source that has position_at)
# ---------------------------------------------------------------------------

def source_velocity_at(
    source: object,
    step: int,
    dt: float,
    eps_steps: int = 1,
) -> tuple[float, float]:
    """Compute instantaneous velocity via central finite differences.

    Works with *any* source object that exposes
    ``position_at(step, dt) -> (x, y)``.

    Returns ``(vx, vy)`` in m/s.
    """
    s0 = max(step - eps_steps, 0)
    s1 = step + eps_steps
    x0, y0 = source.position_at(s0, dt)  # type: ignore[union-attr]
    x1, y1 = source.position_at(s1, dt)  # type: ignore[union-attr]
    delta_t = (s1 - s0) * dt
    if delta_t < 1e-30:
        return (0.0, 0.0)
    return ((x1 - x0) / delta_t, (y1 - y0) / delta_t)


# ---------------------------------------------------------------------------
# Source injection
# ---------------------------------------------------------------------------

def inject_source(
    field: np.ndarray,
    sx: float,
    sy: float,
    amplitude: float,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    dx: float,
    dy: float,
    ix_offset: int = 0,
    iy_offset: int = 0,
) -> None:
    """Inject *amplitude* into *field* at ``(sx, sy)`` using bilinear weights.

    ``ix_offset`` / ``iy_offset`` are the global indices of the first
    interior cell of *field* (used for MPI sub-domains).
    """
    # Continuous grid indices (global).
    gx = (sx - float(x_arr[0])) / dx
    gy = (sy - float(y_arr[0])) / dy
    ix0 = int(math.floor(gx))
    iy0 = int(math.floor(gy))
    fx = gx - ix0
    fy = gy - iy0

    ny_field, nx_field = field.shape
    for diy, wy in ((0, 1.0 - fy), (1, fy)):
        for dix, wx in ((0, 1.0 - fx), (1, fx)):
            li = ix0 + dix - ix_offset
            lj = iy0 + diy - iy_offset
            if 0 <= li < nx_field and 0 <= lj < ny_field:
                field[lj, li] += amplitude * wx * wy

"""Source signal generators and source-position helpers for FDTD.

Supports audio-file sources (WAV), synthetic propeller / tone / noise
generators, and the classic Ricker wavelet.  Each source produces a 1-D
time-series that the FDTD solver injects at the source grid location.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal
from scipy.io import wavfile


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

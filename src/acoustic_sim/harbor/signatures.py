"""Source signatures and level calibration.

Levels are expressed as the free-field rms pressure [Pa] a source produces at
a reference range (100 m) in the 2-D model.  :func:`to_injection` converts a
"signal as received at the reference range" into the waveform that must be
injected into the FDTD so that this is true.

The FDTD point source gives ``p(r, f) = K |H0(k r)|`` per unit injected
amplitude with ``K = (dx / (c dt))^2 / 4`` (validated against the solver in
``tests/harbor/test_fdtd_runner.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import hankel1

R_REF = 100.0          # reference range for source levels [m]
C_NOMINAL = 1500.0     # sound speed the band limit is defined with [m/s]
CELLS_PER_WAVELENGTH = 3.75   # 80 Hz on a 5 m grid at 1500 m/s: the limit the chain was validated at
F_MIN = 5.0


def f_max_for_grid(dx: float, c: float = C_NOMINAL) -> float:
    """Highest frequency [Hz] a grid of spacing *dx* propagates accurately enough for this package.

    ``CELLS_PER_WAVELENGTH`` cells per wavelength at *c*; 80 Hz for the 5 m default grid.
    Any scenario on a finer grid should take its band from this instead of a constant.
    """
    return c / (CELLS_PER_WAVELENGTH * dx)


F_MAX = f_max_for_grid(5.0)   # 80 Hz: the default 5 m grid


@dataclass(frozen=True)
class InjectionCalibration:
    """Constants needed to convert reference-range pressure to FDTD injection."""

    dx: float
    dt: float
    c: float
    r_ref: float = R_REF

    @property
    def k_const(self) -> float:
        return (self.dx / (self.c * self.dt)) ** 2 / 4.0


def to_injection(ref_signal: np.ndarray, fs: float, cal: InjectionCalibration, f_max: float | None = None) -> np.ndarray:
    """Waveform to inject so the free-field pressure at ``cal.r_ref`` equals *ref_signal*.

    The spectrum is divided by the 2-D Green's function magnitude and
    restricted to [F_MIN, f_max] (the band the grid propagates accurately).
    *f_max* defaults to :func:`f_max_for_grid` of ``cal.dx``.
    """
    f_max = f_max_for_grid(cal.dx) if f_max is None else f_max
    n = len(ref_signal)
    spec = np.fft.rfft(ref_signal)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = (f >= F_MIN) & (f <= f_max)
    gain = np.zeros_like(f)
    k = 2.0 * np.pi * f[band] / cal.c
    gain[band] = 1.0 / (cal.k_const * np.abs(hankel1(0, k * cal.r_ref)))
    return np.fft.irfft(spec * gain, n)


def shaped_noise(
    n: int,
    fs: float,
    rng: np.random.Generator,
    f_lo: float = 10.0,
    f_hi: float = F_MAX,
    slope: float = 0.0,
) -> np.ndarray:
    """Band-limited Gaussian noise, amplitude spectrum ``~ f**-slope``, rms normalised to 1."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    taper = np.clip((f - f_lo) / max(0.1 * f_lo, 1e-9), 0, 1) * np.clip((f_hi - f) / (0.1 * f_hi), 0, 1)
    shape = np.where(f > 0, (np.maximum(f, 1e-9) / f_lo) ** (-slope), 0.0)
    out = np.fft.irfft(spec * taper * shape, n)
    return out / np.sqrt(np.mean(out ** 2))


def tonal_stack(
    f_inst: np.ndarray,
    fs: float,
    harmonic_db: list[float],
    rng: np.random.Generator,
    am_depth: float = 0.0,
    am_freq: float = 1.0,
    f_limit: float | None = None,
) -> np.ndarray:
    """Harmonic series following the instantaneous fundamental *f_inst*, rms normalised to 1.

    Harmonics that would exceed *f_limit* (default ``F_MAX - 5``) at any time are dropped.
    """
    f_limit = F_MAX - 5.0 if f_limit is None else f_limit
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    out = np.zeros_like(f_inst, dtype=float)
    for h, db in enumerate(harmonic_db, start=1):
        if h * np.max(f_inst) > f_limit:
            break
        out += 10 ** (db / 20.0) * np.cos(h * phase + rng.uniform(0, 2 * np.pi))
    if am_depth > 0:
        t = np.arange(len(f_inst)) / fs
        out *= 1.0 + am_depth * np.sin(2 * np.pi * am_freq * t + rng.uniform(0, 2 * np.pi))
    return out / np.sqrt(np.mean(out ** 2))


def ship_reference(speed: np.ndarray, fs: float, rng: np.random.Generator, f_max: float = F_MAX) -> np.ndarray:
    """Surface ship signature at the reference range (rms ~ 1 at nominal speed 5 m/s).

    Engine/propeller line series (fundamental rises with speed) with shaft-rate
    amplitude modulation, plus broadband noise whose level grows with speed.
    *f_max* is the top of the radiated band (the grid limit by default).
    """
    n = len(speed)
    f0 = rng.uniform(14.0, 22.0)
    f_inst = f0 + 1.2 * speed
    tonal = tonal_stack(f_inst, fs, [0.0, -3.0, -6.0, -9.0], rng, am_depth=0.3, am_freq=1.5 + 0.1 * rng.uniform(),
                        f_limit=f_max - 5.0)
    broad = shaped_noise(n, fs, rng, 10.0, f_max, slope=0.7)
    base = np.sqrt(0.5) * tonal + np.sqrt(0.5) * broad
    gain = 0.3 + 0.7 * np.minimum(speed / 5.0, 1.5)
    return gain * base / np.sqrt(np.mean(base ** 2))


def uuv_reference(speed: np.ndarray, fs: float, rng: np.random.Generator, f_max: float = F_MAX) -> np.ndarray:
    """Small UUV signature: a sparse low tonal comb (motor/pump lines), little broadband.

    Fundamental 30-40 Hz drifting slightly with speed, one 2nd harmonic,
    very weak broadband (about -10 dB relative to the tonal power).
    """
    n = len(speed)
    f0 = rng.uniform(30.0, 40.0)
    f_inst = f0 + 0.3 * speed
    tonal = tonal_stack(f_inst, fs, [0.0, -6.0], rng, am_depth=0.05, am_freq=0.3, f_limit=f_max - 5.0)
    broad = shaped_noise(n, fs, rng, 20.0, f_max, slope=0.0)
    base = np.sqrt(0.9) * tonal + np.sqrt(0.1) * broad
    gain = 0.5 + 0.5 * np.minimum(speed / 2.0, 1.5)
    return gain * base / np.sqrt(np.mean(base ** 2))


def activity_gate(active: np.ndarray, fs: float, ramp_s: float = 1.0) -> np.ndarray:
    """Smooth 0/1 gate from a boolean activity mask (moving-average edge of *ramp_s* seconds)."""
    w = max(int(ramp_s * fs), 1)
    return np.clip(np.convolve(active.astype(float), np.ones(w) / w, mode="same"), 0.0, 1.0)


def make_reference_signal(
    kind: str, speed: np.ndarray, active: np.ndarray, fs: float, seed: int, f_max: float = F_MAX
) -> np.ndarray:
    """Unit-level reference signature (rms ~ 1 Pa at 100 m) for one vessel, radiating up to *f_max*."""
    rng = np.random.default_rng(seed)
    if kind == "ship":
        sig = ship_reference(speed, fs, rng, f_max)
    elif kind == "uuv":
        sig = uuv_reference(speed, fs, rng, f_max)
    else:
        raise ValueError(f"unknown vessel kind {kind!r}")
    return sig * activity_gate(active, fs)

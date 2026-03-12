"""Post-hoc noise generators for microphone traces.

Wind noise and sensor self-noise are *not* acoustic phenomena that can
be modelled by the FDTD wave solver:

* **Wind noise** — turbulent pressure fluctuations at sub-metre scale.
  Modelled as spatially correlated, spectrally shaped (1/f below a
  corner frequency) random pressure added to each microphone trace.
* **Sensor self-noise** — electronic noise floor of the microphone,
  modelled as uncorrelated white Gaussian noise.

Both are added to the FDTD output traces post-hoc.  Because the wave
equation is linear this is equivalent to having the noise present
during propagation (superposition principle).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

# Reference pressure in air.
_P_REF = 20e-6  # 20 µPa


def generate_wind_noise(
    mic_positions: np.ndarray,
    n_samples: int,
    dt: float,
    level_dB: float = 60.0,
    corner_freq: float = 15.0,
    correlation_length: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Spatially correlated, spectrally shaped wind noise.

    Parameters
    ----------
    mic_positions : (n_mics, 2)
        Microphone (x, y) positions.
    n_samples : int
        Number of time samples per trace.
    dt : float
        Time step [s].
    level_dB : float
        Broadband RMS level in dB re 20 µPa.
    corner_freq : float
        Wind noise corner frequency [Hz].  Spectrum rolls off above this.
    correlation_length : float
        Spatial coherence decay length [m]:
        ``coherence(r) = exp(-r / correlation_length)``.
    seed : int
        Random-number-generator seed.

    Returns
    -------
    np.ndarray, shape ``(n_mics, n_samples)``
    """
    rng = np.random.default_rng(seed)
    n_mics = mic_positions.shape[0]

    # ── Spatial correlation (Cholesky) ──────────────────────────────────
    dx = mic_positions[:, np.newaxis, :] - mic_positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(dx ** 2, axis=-1))  # (n_mics, n_mics)
    C = np.exp(-dist / max(correlation_length, 1e-6))
    # Add small diagonal for numerical stability.
    C += np.eye(n_mics) * 1e-8
    L = np.linalg.cholesky(C)

    # ── White noise → spatially correlated ──────────────────────────────
    white = rng.standard_normal((n_mics, n_samples))
    correlated = L @ white  # (n_mics, n_samples)

    # ── Spectral shaping per trace ──────────────────────────────────────
    fs = 1.0 / dt
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    # 1/sqrt(f) below corner, steep rolloff above.
    f_safe = np.maximum(freqs, freqs[1] if len(freqs) > 1 else 1e-6)
    shape = np.ones_like(freqs)
    below = f_safe <= corner_freq
    shape[below] = 1.0 / np.sqrt(f_safe[below] / corner_freq)
    # 4th-order rolloff above corner.
    above = f_safe > corner_freq
    shape[above] = (corner_freq / f_safe[above]) ** 2
    shape[0] = 0.0  # remove DC

    for i in range(n_mics):
        spec = np.fft.rfft(correlated[i])
        spec *= shape
        correlated[i] = np.fft.irfft(spec, n=n_samples)

    # ── Scale to target RMS level ───────────────────────────────────────
    p_target = _P_REF * 10.0 ** (level_dB / 20.0)
    rms = np.sqrt(np.mean(correlated ** 2))
    if rms > 1e-30:
        correlated *= p_target / rms

    return correlated


def generate_sensor_noise(
    n_mics: int,
    n_samples: int,
    dt: float,
    level_dB: float = 30.0,
    seed: int = 43,
) -> np.ndarray:
    """Uncorrelated white Gaussian sensor self-noise.

    Parameters
    ----------
    n_mics : int
        Number of microphones.
    n_samples : int
        Number of time samples per trace.
    dt : float
        Time step [s] (unused, but kept for API symmetry).
    level_dB : float
        RMS noise level in dB re 20 µPa.
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray, shape ``(n_mics, n_samples)``
    """
    rng = np.random.default_rng(seed)
    p_target = _P_REF * 10.0 ** (level_dB / 20.0)
    noise = rng.standard_normal((n_mics, n_samples)) * p_target
    return noise


def add_all_noise(
    traces: np.ndarray,
    stationary_traces: np.ndarray | None,
    mic_positions: np.ndarray,
    dt: float,
    *,
    wind_enabled: bool = True,
    wind_level_dB: float = 60.0,
    wind_corner_freq: float = 15.0,
    wind_correlation_length: float = 3.0,
    sensor_enabled: bool = True,
    sensor_level_dB: float = 30.0,
    seed: int = 42,
) -> np.ndarray:
    """Sum drone traces, optional stationary traces, and post-hoc noise.

    Parameters
    ----------
    traces : (n_mics, n_samples)
        Clean drone traces from FDTD.
    stationary_traces : (n_mics, n_samples) or None
        Clean stationary-source traces from a second FDTD run.
    mic_positions : (n_mics, 2)
    dt : float
    wind_enabled, wind_level_dB, wind_corner_freq, wind_correlation_length
        Wind-noise parameters.
    sensor_enabled, sensor_level_dB
        Sensor self-noise parameters.
    seed : int

    Returns
    -------
    np.ndarray, shape ``(n_mics, n_samples)``
    """
    n_mics, n_samples = traces.shape
    combined = traces.copy()

    if stationary_traces is not None:
        combined += stationary_traces

    if wind_enabled:
        combined += generate_wind_noise(
            mic_positions, n_samples, dt,
            level_dB=wind_level_dB,
            corner_freq=wind_corner_freq,
            correlation_length=wind_correlation_length,
            seed=seed,
        )

    if sensor_enabled:
        combined += generate_sensor_noise(
            n_mics, n_samples, dt,
            level_dB=sensor_level_dB,
            seed=seed + 1,
        )

    return combined

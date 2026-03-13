"""Feature extraction for ML classification.

Mel spectrogram for acoustic classification.
Kinematic feature vector for fusion classification.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal as sp_signal


def compute_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: float,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Compute log-mel spectrogram of a 1D signal.

    Parameters
    ----------
    signal : 1D array
    sample_rate : float (Hz)
    n_fft, hop_length, n_mels : int
    f_min, f_max : float (Hz)

    Returns
    -------
    (n_mels, n_time_frames) log-power mel spectrogram.
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    # STFT with Hann window.
    window = np.hanning(n_fft)
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    if n_frames < 1:
        # Pad signal if too short.
        padded = np.zeros(n_fft)
        padded[:len(signal)] = signal
        signal = padded
        n_frames = 1

    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + n_fft] * window
        stft[:, i] = np.fft.rfft(frame)

    # Power spectrum.
    power = np.abs(stft) ** 2

    # Mel filterbank.
    mel_fb = _mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)

    # Apply mel filterbank.
    mel_spec = mel_fb @ power  # (n_mels, n_frames)

    # Log compression.
    mel_spec = np.log(np.maximum(mel_spec, 1e-10))

    return mel_spec


def _mel_filterbank(
    sample_rate: float,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Create a mel-scale filterbank matrix.

    Returns (n_mels, n_fft // 2 + 1).
    """
    def _hz_to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def _mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freq = n_fft // 2 + 1
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_freq - 1)

    fb = np.zeros((n_mels, n_freq))
    for m in range(n_mels):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]
        for k in range(f_left, f_center):
            if f_center > f_left:
                fb[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                fb[m, k] = (f_right - k) / (f_right - f_center)

    return fb


def compute_kinematic_features(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float = 0.1,
) -> np.ndarray:
    """Compute 14-dimensional kinematic feature vector.

    Parameters
    ----------
    positions : (N, 3) — x, y, z positions from tracker.
    velocities : (N, 3) — vx, vy, vz from tracker.
    dt : float — time between consecutive measurements.

    Returns
    -------
    (14,) feature vector.
    """
    N = len(positions)
    if N < 3:
        return np.zeros(14)

    vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]
    z = positions[:, 2]

    # Speed.
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    speed_mean = float(np.mean(speed))
    speed_std = float(np.std(speed))
    speed_min = float(np.min(speed))

    # Heading rate.
    heading = np.arctan2(vy, vx)
    # Unwrap heading to avoid +/-pi discontinuity.
    heading_unwrapped = np.unwrap(heading)
    heading_rate = np.diff(heading_unwrapped) / dt
    heading_rate_mean = float(np.mean(np.abs(heading_rate)))
    heading_rate_std = float(np.std(heading_rate))

    # Path curvature: |heading_rate| / speed.
    speed_safe = np.maximum(speed[:-1], 0.5)
    curvature = np.abs(heading_rate) / speed_safe
    # Set curvature to 0 where speed < 0.5 m/s.
    curvature[speed[:-1] < 0.5] = 0.0
    curvature_mean = float(np.mean(curvature))
    curvature_std = float(np.std(curvature))

    # Altitude statistics.
    z_mean = float(np.mean(z))
    z_std = float(np.std(z))
    z_rate = np.diff(z) / dt
    z_rate_std = float(np.std(z_rate))
    z_is_zero = float(np.mean(np.abs(z) < 1.0))

    # Hover fraction.
    hover_fraction = float(np.mean(speed < 1.0))

    # Autocorrelation of heading rate at lag 1.
    if len(heading_rate) > 1:
        hr_centered = heading_rate - np.mean(heading_rate)
        var = np.var(heading_rate)
        if var > 1e-12:
            autocorr = float(np.mean(hr_centered[:-1] * hr_centered[1:]) / var)
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0

    return np.array([
        speed_mean, speed_std, speed_min,
        heading_rate_mean, heading_rate_std,
        curvature_mean, curvature_std,
        z_mean, z_std, z_rate_std, z_is_zero,
        hover_fraction,
        autocorr,
        0.0,  # reserved (14th feature to match spec)
    ], dtype=np.float64)

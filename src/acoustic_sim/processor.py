"""Matched field processor (MFP) for passive acoustic source detection.

Implements the seismological back-projection technique applied to
acoustics: for each candidate source position on a search grid, predict
the travel times to all microphones, time-shift and stack the recorded
traces, and measure the coherent beam power.  The grid point with the
highest coherence is the estimated source location.

The processor operates in sliding time windows with configurable length
and overlap.  A bandpass filter bank isolates the expected drone
harmonic frequencies before beamforming (critical — wind noise dominates
if you beamform unfiltered broadband data).

Stationary source rejection uses the coefficient of variation of the
beam power at each grid point over a rolling history; points with
persistently high, stable power are masked.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


# -----------------------------------------------------------------------
# Travel-time table
# -----------------------------------------------------------------------

def compute_travel_times(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mic_positions: np.ndarray,
    sound_speed: float = 343.0,
) -> np.ndarray:
    """Pre-compute travel times from every grid point to every microphone.

    Parameters
    ----------
    grid_x : (n_gx,)
        Grid x-coordinates.
    grid_y : (n_gy,)
        Grid y-coordinates.
    mic_positions : (n_mics, 2)
        Microphone positions.
    sound_speed : float
        Uniform speed of sound [m/s].

    Returns
    -------
    np.ndarray, shape ``(n_gx, n_gy, n_mics)``
        Travel time in seconds.
    """
    gx = grid_x[:, np.newaxis, np.newaxis]   # (n_gx, 1, 1)
    gy = grid_y[np.newaxis, :, np.newaxis]   # (1, n_gy, 1)
    mx = mic_positions[:, 0][np.newaxis, np.newaxis, :]  # (1, 1, n_mics)
    my = mic_positions[:, 1][np.newaxis, np.newaxis, :]
    dist = np.sqrt((gx - mx) ** 2 + (gy - my) ** 2)
    return dist / sound_speed


# -----------------------------------------------------------------------
# Bandpass filter bank
# -----------------------------------------------------------------------

def create_filter_bank(
    fundamental: float,
    n_harmonics: int,
    bandwidth: float,
    sample_rate: float,
) -> list[np.ndarray]:
    """Design narrow bandpass filters centred on each harmonic.

    Parameters
    ----------
    fundamental : float
        Fundamental frequency [Hz].
    n_harmonics : int
        Number of harmonics (1 = fundamental only).
    bandwidth : float
        Half-width of each bandpass [Hz].
    sample_rate : float
        Sampling rate [Hz].

    Returns
    -------
    list of SOS arrays
        One per harmonic that falls below Nyquist.
    """
    nyq = sample_rate / 2.0
    filters: list[np.ndarray] = []
    for k in range(1, n_harmonics + 1):
        fc = fundamental * k
        lo = max(fc - bandwidth, 1.0)
        hi = min(fc + bandwidth, nyq * 0.95)
        if lo >= hi:
            continue
        sos = sp_signal.butter(4, [lo, hi], btype="bandpass",
                               fs=sample_rate, output="sos")
        filters.append(sos)
    return filters


def apply_filter_bank(
    traces: np.ndarray,
    filters: list[np.ndarray],
) -> np.ndarray:
    """Apply each bandpass filter and sum the outputs.

    Parameters
    ----------
    traces : (n_mics, n_samples)
    filters : list of SOS arrays from :func:`create_filter_bank`.

    Returns
    -------
    np.ndarray, same shape as *traces*.
    """
    out = np.zeros_like(traces)
    for sos in filters:
        for i in range(traces.shape[0]):
            out[i] += sp_signal.sosfilt(sos, traces[i])
    return out


# -----------------------------------------------------------------------
# Beam power / coherence
# -----------------------------------------------------------------------

def compute_beam_power(
    filtered_traces: np.ndarray,
    travel_time_samples: np.ndarray,
    window_start: int,
    window_length: int,
) -> np.ndarray:
    """Compute normalised coherence map for one time window.

    Parameters
    ----------
    filtered_traces : (n_mics, n_samples)
        Bandpass-filtered microphone traces.
    travel_time_samples : (n_gx, n_gy, n_mics)
        Travel times expressed as *integer* sample offsets.
    window_start : int
        First sample of the analysis window.
    window_length : int
        Window length in samples.

    Returns
    -------
    np.ndarray, shape ``(n_gx, n_gy)``
        Coherence values in [0, 1].
    """
    n_mics, n_samples = filtered_traces.shape
    n_gx, n_gy, _ = travel_time_samples.shape
    wend = window_start + window_length

    # Individual trace power in window (for normalisation).
    indiv_power = np.zeros(n_mics)
    for m in range(n_mics):
        seg = filtered_traces[m, window_start:wend]
        indiv_power[m] = np.sum(seg ** 2)
    total_indiv = np.sum(indiv_power)
    if total_indiv < 1e-30:
        return np.zeros((n_gx, n_gy))

    coherence = np.zeros((n_gx, n_gy))

    # Vectorised over grid — for each grid point, shift and stack.
    for ix in range(n_gx):
        for iy in range(n_gy):
            stack = np.zeros(window_length)
            for m in range(n_mics):
                shift = int(travel_time_samples[ix, iy, m])
                s = window_start - shift
                e = s + window_length
                if s < 0 or e > n_samples:
                    continue
                stack += filtered_traces[m, s:e]
            beam = np.sum(stack ** 2)
            coherence[ix, iy] = beam / total_indiv

    return coherence


# -----------------------------------------------------------------------
# Stationary source rejection
# -----------------------------------------------------------------------

def detect_stationary(
    beam_power_history: list[np.ndarray],
    cv_threshold: float = 0.2,
) -> np.ndarray:
    """Identify grid points with persistently high, stable beam power.

    Parameters
    ----------
    beam_power_history : list of (n_gx, n_gy)
        Recent coherence maps.
    cv_threshold : float
        Grid points with coefficient of variation below this are
        considered stationary.

    Returns
    -------
    np.ndarray, shape ``(n_gx, n_gy)``, dtype bool
        ``True`` where a stationary source is detected.
    """
    if len(beam_power_history) < 2:
        return np.zeros(beam_power_history[0].shape, dtype=bool)

    stack = np.stack(beam_power_history, axis=0)  # (n_hist, n_gx, n_gy)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    cv = std / np.maximum(mean, 1e-12)

    # Stationary = low CV *and* mean power above a modest threshold.
    mean_threshold = 0.5 * np.mean(mean)
    return (cv < cv_threshold) & (mean > mean_threshold)


# -----------------------------------------------------------------------
# Sliding-window processor
# -----------------------------------------------------------------------

def matched_field_process(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    *,
    sound_speed: float = 343.0,
    grid_spacing: float = 5.0,
    grid_x_range: tuple[float, float] | None = None,
    grid_y_range: tuple[float, float] | None = None,
    window_length: float = 0.1,
    window_overlap: float = 0.5,
    detection_threshold: float = 0.3,
    fundamental: float = 150.0,
    n_harmonics: int = 4,
    harmonic_bandwidth: float = 20.0,
    stationary_history: int = 10,
    stationary_cv_threshold: float = 0.2,
) -> dict:
    """Run matched field processing on microphone traces.

    Parameters
    ----------
    traces : (n_mics, n_samples)
        Microphone data (may include noise).
    mic_positions : (n_mics, 2)
    dt : float
        Time step between samples.
    sound_speed : float
    grid_spacing : float
        MFP search-grid spacing [m].
    grid_x_range, grid_y_range : (min, max) or None
        Search region.  *None* → infer from mic array ± 100 m.
    window_length : float
        Analysis window in seconds.
    window_overlap : float
        Fractional overlap (0–1).
    detection_threshold : float
        Minimum coherence for a detection (0–1).
    fundamental, n_harmonics, harmonic_bandwidth : float
        Bandpass filter bank parameters.
    stationary_history, stationary_cv_threshold : float
        Stationary source rejection parameters.

    Returns
    -------
    dict with keys:
        ``detections``  — list of dicts per window
        ``grid_x``, ``grid_y`` — 1-D grid coordinate arrays
        ``filtered_traces`` — the bandpass-filtered traces
    """
    n_mics, n_samples = traces.shape
    fs = 1.0 / dt

    # ── Build search grid ───────────────────────────────────────────────
    if grid_x_range is None:
        cx = np.mean(mic_positions[:, 0])
        grid_x_range = (cx - 100.0, cx + 100.0)
    if grid_y_range is None:
        cy = np.mean(mic_positions[:, 1])
        grid_y_range = (cy - 100.0, cy + 100.0)

    grid_x = np.arange(grid_x_range[0], grid_x_range[1] + 0.5 * grid_spacing,
                        grid_spacing)
    grid_y = np.arange(grid_y_range[0], grid_y_range[1] + 0.5 * grid_spacing,
                        grid_spacing)

    # ── Travel-time table (seconds → integer sample offsets) ────────────
    tt_sec = compute_travel_times(grid_x, grid_y, mic_positions, sound_speed)
    tt_samples = np.round(tt_sec / dt).astype(np.int64)

    # ── Bandpass filter bank ────────────────────────────────────────────
    filters = create_filter_bank(fundamental, n_harmonics,
                                 harmonic_bandwidth, fs)
    if len(filters) == 0:
        # No harmonics below Nyquist — fall back to broadband
        filtered = traces.copy()
    else:
        filtered = apply_filter_bank(traces, filters)

    # ── Sliding windows ─────────────────────────────────────────────────
    win_len = max(int(round(window_length * fs)), 1)
    hop = max(int(round(win_len * (1.0 - window_overlap))), 1)

    detections: list[dict] = []
    history: list[np.ndarray] = []

    pos = 0
    while pos + win_len <= n_samples:
        coh = compute_beam_power(filtered, tt_samples, pos, win_len)

        # Stationary rejection.
        history.append(coh.copy())
        if len(history) > stationary_history:
            history.pop(0)
        mask = detect_stationary(history, stationary_cv_threshold)
        coh_masked = coh.copy()
        coh_masked[mask] = 0.0

        # Detection.
        peak_val = float(np.max(coh_masked))
        ix, iy = np.unravel_index(np.argmax(coh_masked), coh_masked.shape)
        detected = peak_val >= detection_threshold

        t_center = (pos + win_len / 2.0) * dt
        detections.append({
            "time": t_center,
            "x": float(grid_x[ix]) if detected else float("nan"),
            "y": float(grid_y[iy]) if detected else float("nan"),
            "coherence": peak_val,
            "detected": detected,
            "beam_power_map": coh,
        })

        pos += hop

    return {
        "detections": detections,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "filtered_traces": filtered,
    }

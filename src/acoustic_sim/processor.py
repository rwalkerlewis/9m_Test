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

Robustness features
===================
* **Multi-peak detection** — iterative peak-finding with exclusion
  zones to detect multiple simultaneous sources.
* **Sensor fault detection** — compare per-sensor power to the median;
  outlier sensors are down-weighted in the beamformer.
* **Transient blanking** — short-time energy thresholding zeros out
  impulsive events (explosions) before they corrupt the MFP.
* **Stationary source rejection** — coefficient-of-variation filter
  on rolling beam-power history.
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

    Returns shape ``(n_gx, n_gy, n_mics)`` in seconds.
    """
    gx = grid_x[:, np.newaxis, np.newaxis]
    gy = grid_y[np.newaxis, :, np.newaxis]
    mx = mic_positions[:, 0][np.newaxis, np.newaxis, :]
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
    """Narrow bandpass filters centred on each harmonic below Nyquist."""
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
    """Apply each bandpass filter and sum the outputs."""
    out = np.zeros_like(traces)
    for sos in filters:
        for i in range(traces.shape[0]):
            out[i] += sp_signal.sosfilt(sos, traces[i])
    return out


# -----------------------------------------------------------------------
# Sensor fault detection / robust weighting
# -----------------------------------------------------------------------

def compute_sensor_weights(
    traces: np.ndarray,
    method: str = "median_power",
    fault_threshold: float = 10.0,
) -> np.ndarray:
    """Compute per-sensor reliability weights.

    Parameters
    ----------
    traces : (n_mics, n_samples)
    method : str
        ``"median_power"`` — sensors whose total power deviates from
        the median by more than *fault_threshold*× are zeroed.
    fault_threshold : float
        Ratio above/below median that flags a faulty sensor.

    Returns
    -------
    np.ndarray, shape ``(n_mics,)``
        Weights in {0, 1}.
    """
    n_mics = traces.shape[0]
    power = np.array([np.sum(traces[m] ** 2) for m in range(n_mics)])
    median_p = np.median(power)
    if median_p < 1e-30:
        return np.ones(n_mics)
    ratio = power / median_p
    weights = np.ones(n_mics)
    weights[ratio > fault_threshold] = 0.0
    weights[ratio < 1.0 / fault_threshold] = 0.0
    return weights


# -----------------------------------------------------------------------
# Transient blanking
# -----------------------------------------------------------------------

def blank_transients(
    traces: np.ndarray,
    dt: float,
    subwindow_ms: float = 5.0,
    threshold_factor: float = 10.0,
) -> np.ndarray:
    """Zero out sub-windows where short-time energy exceeds threshold.

    A running median of sub-window energies is used as the reference.
    Sub-windows with energy > *threshold_factor* × median are zeroed.

    Returns a copy with transients blanked.
    """
    out = traces.copy()
    n_mics, n_samples = traces.shape
    sub_len = max(int(subwindow_ms * 1e-3 / dt), 1)
    n_subs = n_samples // sub_len

    if n_subs < 3:
        return out

    # Compute per-sub-window energy (summed across all mics).
    energies = np.zeros(n_subs)
    for s in range(n_subs):
        start = s * sub_len
        end = start + sub_len
        energies[s] = np.sum(traces[:, start:end] ** 2)

    # Running median (use full-array median as a simple robust estimator).
    med = np.median(energies)
    if med < 1e-30:
        return out

    for s in range(n_subs):
        if energies[s] > threshold_factor * med:
            start = s * sub_len
            end = start + sub_len
            out[:, start:end] = 0.0

    return out


# -----------------------------------------------------------------------
# Beam power / coherence
# -----------------------------------------------------------------------

def compute_beam_power(
    filtered_traces: np.ndarray,
    travel_time_samples: np.ndarray,
    window_start: int,
    window_length: int,
    sensor_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute normalised coherence map for one time window.

    Parameters
    ----------
    filtered_traces : (n_mics, n_samples)
    travel_time_samples : (n_gx, n_gy, n_mics)  integer sample offsets
    window_start, window_length : int
    sensor_weights : (n_mics,) or None
        Per-sensor weights (0 = exclude, 1 = include).

    Returns
    -------
    np.ndarray, shape ``(n_gx, n_gy)``
    """
    n_mics, n_samples = filtered_traces.shape
    n_gx, n_gy, _ = travel_time_samples.shape
    wend = window_start + window_length

    if sensor_weights is None:
        sensor_weights = np.ones(n_mics)

    # Count active sensors and compute per-sensor power for normalisation.
    n_active = 0
    total_indiv = 0.0
    for m in range(n_mics):
        if sensor_weights[m] < 0.5:
            continue
        seg = filtered_traces[m, window_start:wend]
        total_indiv += np.sum(seg ** 2)
        n_active += 1
    if total_indiv < 1e-30 or n_active == 0:
        return np.zeros((n_gx, n_gy))

    # Normalisation factor: for perfect coherence among N sensors,
    # beam_power = N² × per-sensor-power, so dividing by N × total_indiv
    # gives a coherence in [0, 1].
    norm = n_active * total_indiv

    coherence = np.zeros((n_gx, n_gy))

    for ix in range(n_gx):
        for iy in range(n_gy):
            stack = np.zeros(window_length)
            n_valid = 0
            for m in range(n_mics):
                if sensor_weights[m] < 0.5:
                    continue
                shift = int(travel_time_samples[ix, iy, m])
                s = window_start + shift
                e = s + window_length
                if s < 0 or e > n_samples:
                    continue
                stack += sensor_weights[m] * filtered_traces[m, s:e]
                n_valid += 1
            if n_valid == 0:
                continue
            beam = np.sum(stack ** 2)
            coherence[ix, iy] = beam / norm

    return coherence


# -----------------------------------------------------------------------
# Multi-peak detection
# -----------------------------------------------------------------------

def find_multiple_peaks(
    coherence_map: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    threshold: float = 0.3,
    min_separation_m: float = 20.0,
    max_sources: int = 5,
) -> list[dict]:
    """Find up to *max_sources* peaks in the coherence map.

    Uses iterative peak-finding with exclusion zones: find the strongest
    peak, mask a circle of ``min_separation_m`` around it, repeat.

    Returns list of ``{x, y, coherence}`` dicts sorted by coherence
    (descending).
    """
    coh = coherence_map.copy()
    dx = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
    dy = grid_y[1] - grid_y[0] if len(grid_y) > 1 else 1.0
    excl_ix = max(int(np.ceil(min_separation_m / dx)), 1)
    excl_iy = max(int(np.ceil(min_separation_m / dy)), 1)

    peaks: list[dict] = []
    for _ in range(max_sources):
        peak_val = float(np.max(coh))
        if peak_val < threshold:
            break
        ix, iy = np.unravel_index(np.argmax(coh), coh.shape)
        peaks.append({
            "x": float(grid_x[ix]),
            "y": float(grid_y[iy]),
            "coherence": peak_val,
        })
        # Mask exclusion zone.
        ix_lo = max(ix - excl_ix, 0)
        ix_hi = min(ix + excl_ix + 1, coh.shape[0])
        iy_lo = max(iy - excl_iy, 0)
        iy_hi = min(iy + excl_iy + 1, coh.shape[1])
        coh[ix_lo:ix_hi, iy_lo:iy_hi] = 0.0

    return peaks


# -----------------------------------------------------------------------
# Stationary source rejection
# -----------------------------------------------------------------------

def detect_stationary(
    beam_power_history: list[np.ndarray],
    cv_threshold: float = 0.2,
) -> np.ndarray:
    """Identify grid points with persistently high, stable beam power."""
    if len(beam_power_history) < 2:
        return np.zeros(beam_power_history[0].shape, dtype=bool)

    stack = np.stack(beam_power_history, axis=0)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    cv = std / np.maximum(mean, 1e-12)
    mean_threshold = 0.5 * np.mean(mean)
    return (cv < cv_threshold) & (mean > mean_threshold)


# -----------------------------------------------------------------------
# Position self-calibration
# -----------------------------------------------------------------------

def calibrate_positions(
    traces: np.ndarray,
    reported_positions: np.ndarray,
    dt: float,
    sound_speed: float = 343.0,
    max_lag_m: float = 10.0,
) -> np.ndarray:
    """Estimate corrected microphone positions via cross-correlation.

    For each microphone pair, cross-correlate to find the observed
    inter-sensor travel-time difference (TDOA).  Then solve a
    least-squares problem to find position corrections that minimise
    the residual between observed and predicted TDOAs.

    Parameters
    ----------
    traces : (n_mics, n_samples)
    reported_positions : (n_mics, 2)
    dt : float
    sound_speed : float
    max_lag_m : float
        Maximum expected position error [m] — limits the cross-correlation
        search window.

    Returns
    -------
    np.ndarray, shape ``(n_mics, 2)``
        Corrected positions.
    """
    n_mics = traces.shape[0]
    max_lag_samples = int(np.ceil(max_lag_m / sound_speed / dt))

    # ── Measure TDOAs via cross-correlation ─────────────────────────────
    pairs = []
    observed_tdoa = []
    predicted_tdoa = []
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            cc = np.correlate(
                traces[i, max_lag_samples:-max_lag_samples] if max_lag_samples > 0
                else traces[i],
                traces[j],
                mode="full" if max_lag_samples == 0 else "valid",
            )
            if len(cc) == 0:
                continue
            # For "valid" mode the lag axis is [-max_lag_samples, +max_lag_samples]
            peak_idx = np.argmax(np.abs(cc))
            if max_lag_samples > 0:
                lag_samples = peak_idx - max_lag_samples
            else:
                lag_samples = peak_idx - (len(cc) // 2)
            obs_tdoa = lag_samples * dt

            # Predicted TDOA from reported positions (relative to array centre).
            d_i = np.linalg.norm(reported_positions[i] - np.mean(reported_positions, axis=0))
            d_j = np.linalg.norm(reported_positions[j] - np.mean(reported_positions, axis=0))
            pred_tdoa = (d_i - d_j) / sound_speed

            pairs.append((i, j))
            observed_tdoa.append(obs_tdoa)
            predicted_tdoa.append(pred_tdoa)

    if len(pairs) < n_mics:
        # Not enough pairs — return reported positions unchanged.
        return reported_positions.copy()

    # ── Least-squares position correction ───────────────────────────────
    # Small correction model: δ_pos for each mic.
    # For each pair (i,j), the TDOA residual ≈ (n̂_ij · δ_i - n̂_ij · δ_j) / c
    # where n̂_ij is the unit vector from mic i to mic j.
    # We solve for δ = [δx_0, δy_0, δx_1, δy_1, ...] (2*n_mics unknowns)
    # with a zero-mean constraint to remove the translational ambiguity.

    n_pairs = len(pairs)
    A = np.zeros((n_pairs + 2, 2 * n_mics))
    b = np.zeros(n_pairs + 2)

    for p, (i, j) in enumerate(pairs):
        diff = reported_positions[j] - reported_positions[i]
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            continue
        n_hat = diff / dist
        # ∂(TDOA)/∂(pos_i) ≈ -n_hat / c,  ∂(TDOA)/∂(pos_j) ≈ +n_hat / c
        A[p, 2 * i] = -n_hat[0] / sound_speed
        A[p, 2 * i + 1] = -n_hat[1] / sound_speed
        A[p, 2 * j] = n_hat[0] / sound_speed
        A[p, 2 * j + 1] = n_hat[1] / sound_speed
        b[p] = observed_tdoa[p] - predicted_tdoa[p]

    # Zero-mean constraint (remove bulk translation).
    for m in range(n_mics):
        A[n_pairs, 2 * m] = 1.0 / n_mics
        A[n_pairs + 1, 2 * m + 1] = 1.0 / n_mics

    # Solve via least-squares.
    result = np.linalg.lstsq(A, b, rcond=None)
    delta = result[0]

    corrected = reported_positions.copy()
    for m in range(n_mics):
        corrected[m, 0] += delta[2 * m]
        corrected[m, 1] += delta[2 * m + 1]

    return corrected


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
    # ── Robustness options ──
    enable_sensor_weights: bool = False,
    sensor_fault_threshold: float = 10.0,
    enable_transient_blanking: bool = False,
    transient_subwindow_ms: float = 5.0,
    transient_threshold_factor: float = 10.0,
    max_sources: int = 1,
    min_source_separation_m: float = 20.0,
    enable_position_calibration: bool = False,
    position_calibration_max_lag_m: float = 10.0,
) -> dict:
    """Run matched field processing on microphone traces.

    Returns dict with keys:
        ``detections``       — list of single-best-peak dicts per window
        ``multi_detections`` — list of lists (multi-peak) per window
        ``grid_x``, ``grid_y`` — 1-D grid coordinate arrays
        ``filtered_traces``  — the bandpass-filtered traces
        ``sensor_weights``   — per-sensor weights used
        ``calibrated_positions`` — mic positions used (may differ from input)
    """
    n_mics, n_samples = traces.shape
    fs = 1.0 / dt

    # ── Position calibration ────────────────────────────────────────────
    used_positions = mic_positions
    if enable_position_calibration:
        used_positions = calibrate_positions(
            traces, mic_positions, dt, sound_speed,
            max_lag_m=position_calibration_max_lag_m,
        )

    # ── Transient blanking ──────────────────────────────────────────────
    working_traces = traces
    if enable_transient_blanking:
        working_traces = blank_transients(
            traces, dt,
            subwindow_ms=transient_subwindow_ms,
            threshold_factor=transient_threshold_factor,
        )

    # ── Sensor weights ──────────────────────────────────────────────────
    weights = np.ones(n_mics)
    if enable_sensor_weights:
        weights = compute_sensor_weights(
            working_traces, fault_threshold=sensor_fault_threshold,
        )

    # ── Build search grid ───────────────────────────────────────────────
    if grid_x_range is None:
        cx = np.mean(used_positions[:, 0])
        grid_x_range = (cx - 100.0, cx + 100.0)
    if grid_y_range is None:
        cy = np.mean(used_positions[:, 1])
        grid_y_range = (cy - 100.0, cy + 100.0)

    grid_x = np.arange(grid_x_range[0], grid_x_range[1] + 0.5 * grid_spacing,
                        grid_spacing)
    grid_y = np.arange(grid_y_range[0], grid_y_range[1] + 0.5 * grid_spacing,
                        grid_spacing)

    # ── Travel-time table ───────────────────────────────────────────────
    tt_sec = compute_travel_times(grid_x, grid_y, used_positions, sound_speed)
    tt_samples = np.round(tt_sec / dt).astype(np.int64)

    # ── Bandpass filter bank ────────────────────────────────────────────
    filters = create_filter_bank(fundamental, n_harmonics,
                                 harmonic_bandwidth, fs)
    if len(filters) == 0:
        filtered = working_traces.copy()
    else:
        filtered = apply_filter_bank(working_traces, filters)

    # ── Sliding windows ─────────────────────────────────────────────────
    win_len = max(int(round(window_length * fs)), 1)
    hop = max(int(round(win_len * (1.0 - window_overlap))), 1)

    detections: list[dict] = []
    multi_detections: list[list[dict]] = []
    history: list[np.ndarray] = []

    pos = 0
    while pos + win_len <= n_samples:
        coh = compute_beam_power(filtered, tt_samples, pos, win_len,
                                 sensor_weights=weights)

        # Stationary rejection.
        history.append(coh.copy())
        if len(history) > stationary_history:
            history.pop(0)
        mask = detect_stationary(history, stationary_cv_threshold)
        coh_masked = coh.copy()
        coh_masked[mask] = 0.0

        t_center = (pos + win_len / 2.0) * dt

        # ── Multi-peak detection ────────────────────────────────────────
        peaks = find_multiple_peaks(
            coh_masked, grid_x, grid_y,
            threshold=detection_threshold,
            min_separation_m=min_source_separation_m,
            max_sources=max_sources,
        )
        for pk in peaks:
            pk["time"] = t_center
            pk["detected"] = True
        multi_detections.append(peaks)

        # ── Single-best detection (backward compatible) ─────────────────
        if peaks:
            best = peaks[0]
            detections.append({
                "time": t_center,
                "x": best["x"],
                "y": best["y"],
                "coherence": best["coherence"],
                "detected": True,
                "beam_power_map": coh,
            })
        else:
            detections.append({
                "time": t_center,
                "x": float("nan"),
                "y": float("nan"),
                "coherence": float(np.max(coh_masked)) if coh_masked.size else 0.0,
                "detected": False,
                "beam_power_map": coh,
            })

        pos += hop

    return {
        "detections": detections,
        "multi_detections": multi_detections,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "filtered_traces": filtered,
        "sensor_weights": weights,
        "calibrated_positions": used_positions,
    }

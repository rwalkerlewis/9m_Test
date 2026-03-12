"""Matched field processor (MFP) for passive acoustic source detection.

Implements the seismological back-projection technique applied to
acoustics.  Fully vectorised with NumPy; optional CUDA acceleration
via CuPy for the compute-intensive beam-power grid search.

Robustness features
===================
* **Multi-peak detection** — iterative peak-finding with exclusion zones.
* **Sensor fault detection** — median-power outlier down-weighting.
* **Transient blanking** — short-time energy thresholding.
* **Stationary source rejection** — CV filter on rolling beam-power history.
* **Position self-calibration** — cross-correlation TDOA least-squares.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
from scipy import signal as sp_signal


def _get_xp(use_cuda: bool) -> tuple[ModuleType, bool]:
    """Return ``(array_module, is_cuda)`` — NumPy or CuPy."""
    if use_cuda:
        try:
            import cupy  # type: ignore[import-untyped]
            cupy.cuda.Device(0).compute_capability
            return cupy, True
        except Exception:
            pass
    return np, False


# -----------------------------------------------------------------------
# Travel-time table
# -----------------------------------------------------------------------

def compute_travel_times(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mic_positions: np.ndarray,
    sound_speed: float = 343.0,
    use_cuda: bool = False,
) -> np.ndarray:
    """Pre-compute travel times.  Returns ``(n_gx, n_gy, n_mics)``."""
    xp, _ = _get_xp(use_cuda)
    gx = xp.asarray(grid_x)[:, None, None]
    gy = xp.asarray(grid_y)[None, :, None]
    mx = xp.asarray(mic_positions[:, 0])[None, None, :]
    my = xp.asarray(mic_positions[:, 1])[None, None, :]
    dist = xp.sqrt((gx - mx) ** 2 + (gy - my) ** 2)
    result = dist / sound_speed
    return result.get() if use_cuda and hasattr(result, "get") else np.asarray(result)


# -----------------------------------------------------------------------
# Bandpass filter bank
# -----------------------------------------------------------------------

def create_filter_bank(
    fundamental: float,
    n_harmonics: int,
    bandwidth: float,
    sample_rate: float,
) -> list[np.ndarray]:
    """Narrow bandpass SOS filters for each harmonic below Nyquist."""
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
    """Apply each bandpass filter and sum the outputs (vectorised)."""
    out = np.zeros_like(traces)
    for sos in filters:
        # sosfilt along axis=1 processes all mics in one call.
        out += sp_signal.sosfilt(sos, traces, axis=1)
    return out


# -----------------------------------------------------------------------
# Sensor fault detection / robust weighting
# -----------------------------------------------------------------------

def compute_sensor_weights(
    traces: np.ndarray,
    method: str = "median_power",
    fault_threshold: float = 10.0,
) -> np.ndarray:
    """Per-sensor reliability weights (vectorised)."""
    power = np.sum(traces ** 2, axis=1)          # (n_mics,)
    median_p = float(np.median(power))
    if median_p < 1e-30:
        return np.ones(traces.shape[0])
    ratio = power / median_p
    weights = np.ones(traces.shape[0])
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
    """Zero out impulsive sub-windows (vectorised)."""
    n_mics, n_samples = traces.shape
    sub_len = max(int(subwindow_ms * 1e-3 / dt), 1)
    n_subs = n_samples // sub_len
    if n_subs < 3:
        return traces.copy()

    usable = n_subs * sub_len
    # Reshape to (n_mics, n_subs, sub_len), sum energy per sub-window.
    reshaped = traces[:, :usable].reshape(n_mics, n_subs, sub_len)
    energies = np.sum(reshaped ** 2, axis=(0, 2))   # (n_subs,)

    med = float(np.median(energies))
    if med < 1e-30:
        return traces.copy()

    bad = energies > threshold_factor * med          # (n_subs,) bool
    out = traces.copy()
    # Blank bad sub-windows.
    bad_mask = np.repeat(bad, sub_len)               # (usable,) bool
    out[:, :usable][:, bad_mask] = 0.0
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
    use_cuda: bool = False,
) -> np.ndarray:
    """Compute normalised coherence map for one time window.

    Vectorised over mics; optional CUDA acceleration for the large
    stack accumulation and beam-power reduction.
    """
    xp, is_cuda = _get_xp(use_cuda)

    n_mics, n_samples = filtered_traces.shape
    n_gx, n_gy, _ = travel_time_samples.shape

    if sensor_weights is None:
        sensor_weights = np.ones(n_mics)

    active = sensor_weights >= 0.5
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active == 0:
        return np.zeros((n_gx, n_gy))

    # Normalisation: sum of individual trace powers in the window.
    window_data = filtered_traces[active, window_start:window_start + window_length]
    total_indiv = float(np.sum(window_data ** 2))
    if total_indiv < 1e-30:
        return np.zeros((n_gx, n_gy))
    norm = n_active * total_indiv

    # Move to device if CUDA.
    ft = xp.asarray(filtered_traces) if is_cuda else filtered_traces
    tt = xp.asarray(travel_time_samples) if is_cuda else travel_time_samples
    sw = xp.asarray(sensor_weights) if is_cuda else sensor_weights

    stack = xp.zeros((n_gx, n_gy, window_length), dtype=np.float64)

    for m_idx in active_idx:
        w = float(sensor_weights[m_idx])
        shifts = tt[:, :, int(m_idx)]  # (n_gx, n_gy)

        unique_shifts = np.unique(travel_time_samples[:, :, int(m_idx)])
        for sh in unique_shifts:
            s = window_start + int(sh)
            e = s + window_length
            if s < 0 or e > n_samples:
                continue
            mask = shifts == int(sh)
            seg = ft[int(m_idx), s:e]
            stack[mask] += w * seg

    beam = xp.sum(stack ** 2, axis=2)

    result = beam / norm
    if is_cuda and hasattr(result, "get"):
        return result.get()
    return np.asarray(result)


# -----------------------------------------------------------------------
# Multi-peak detection
# -----------------------------------------------------------------------

def _parabolic_interp_1d(y_minus: float, y_center: float, y_plus: float) -> float:
    """Sub-sample offset from parabolic fit through 3 points.

    Returns offset in [-0.5, 0.5] from the centre sample.
    """
    denom = 2.0 * (2.0 * y_center - y_minus - y_plus)
    if abs(denom) < 1e-30:
        return 0.0
    return (y_minus - y_plus) / denom


def find_multiple_peaks(
    coherence_map: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    threshold: float = 0.3,
    min_separation_m: float = 20.0,
    max_sources: int = 5,
) -> list[dict]:
    """Find up to *max_sources* peaks with exclusion zones.

    Uses parabolic sub-grid interpolation around each peak for
    sub-grid-cell position accuracy.
    """
    coh = coherence_map.copy()
    n_gx, n_gy = coh.shape
    dx = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
    dy = grid_y[1] - grid_y[0] if len(grid_y) > 1 else 1.0
    excl_ix = max(int(np.ceil(min_separation_m / dx)), 1)
    excl_iy = max(int(np.ceil(min_separation_m / dy)), 1)

    peaks: list[dict] = []
    for _ in range(max_sources):
        peak_val = float(np.max(coh))
        if peak_val < threshold:
            break
        ix, iy = np.unravel_index(int(np.argmax(coh)), coh.shape)

        # Sub-grid parabolic interpolation.
        off_x = 0.0
        if 0 < ix < n_gx - 1:
            off_x = _parabolic_interp_1d(
                coh[ix - 1, iy], coh[ix, iy], coh[ix + 1, iy])
        off_y = 0.0
        if 0 < iy < n_gy - 1:
            off_y = _parabolic_interp_1d(
                coh[ix, iy - 1], coh[ix, iy], coh[ix, iy + 1])

        x_interp = float(grid_x[ix]) + off_x * dx
        y_interp = float(grid_y[iy]) + off_y * dy

        peaks.append({
            "x": x_interp,
            "y": y_interp,
            "coherence": peak_val,
        })
        coh[max(ix - excl_ix, 0):min(ix + excl_ix + 1, n_gx),
            max(iy - excl_iy, 0):min(iy + excl_iy + 1, n_gy)] = 0.0

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
    """Estimate corrected microphone positions via cross-correlation TDOA."""
    n_mics = traces.shape[0]
    max_lag_samples = int(np.ceil(max_lag_m / sound_speed / dt))

    pairs = []
    observed_tdoa = []
    predicted_tdoa = []
    centre = np.mean(reported_positions, axis=0)

    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            if max_lag_samples > 0 and max_lag_samples < traces.shape[1] // 2:
                cc = np.correlate(
                    traces[i, max_lag_samples:-max_lag_samples],
                    traces[j], mode="valid",
                )
            else:
                cc = np.correlate(traces[i], traces[j], mode="full")
                max_lag_samples = 0
            if len(cc) == 0:
                continue
            peak_idx = int(np.argmax(np.abs(cc)))
            lag = peak_idx - max_lag_samples if max_lag_samples > 0 else peak_idx - len(cc) // 2
            obs = lag * dt

            d_i = float(np.linalg.norm(reported_positions[i] - centre))
            d_j = float(np.linalg.norm(reported_positions[j] - centre))
            pred = (d_i - d_j) / sound_speed

            pairs.append((i, j))
            observed_tdoa.append(obs)
            predicted_tdoa.append(pred)

    if len(pairs) < n_mics:
        return reported_positions.copy()

    n_pairs = len(pairs)
    A = np.zeros((n_pairs + 2, 2 * n_mics))
    b = np.zeros(n_pairs + 2)

    pairs_arr = np.array(pairs)                # (n_pairs, 2)
    diffs = reported_positions[pairs_arr[:, 1]] - reported_positions[pairs_arr[:, 0]]  # (n_pairs, 2)
    dists = np.linalg.norm(diffs, axis=1, keepdims=True)
    dists = np.maximum(dists, 1e-6)
    n_hats = diffs / dists                     # (n_pairs, 2)

    for p in range(n_pairs):
        i, j = pairs[p]
        A[p, 2 * i]     = -n_hats[p, 0] / sound_speed
        A[p, 2 * i + 1] = -n_hats[p, 1] / sound_speed
        A[p, 2 * j]     =  n_hats[p, 0] / sound_speed
        A[p, 2 * j + 1] =  n_hats[p, 1] / sound_speed
        b[p] = observed_tdoa[p] - predicted_tdoa[p]

    # Zero-mean constraint.
    A[n_pairs, 0::2] = 1.0 / n_mics
    A[n_pairs + 1, 1::2] = 1.0 / n_mics

    delta, *_ = np.linalg.lstsq(A, b, rcond=None)
    corrected = reported_positions + delta.reshape(n_mics, 2)
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
    enable_sensor_weights: bool = False,
    sensor_fault_threshold: float = 10.0,
    enable_transient_blanking: bool = False,
    transient_subwindow_ms: float = 5.0,
    transient_threshold_factor: float = 10.0,
    max_sources: int = 1,
    min_source_separation_m: float = 20.0,
    enable_position_calibration: bool = False,
    position_calibration_max_lag_m: float = 10.0,
    use_cuda: bool = False,
) -> dict:
    """Run matched field processing on microphone traces.

    Fully vectorised; optional ``use_cuda=True`` for GPU acceleration
    of the beam-power grid search.
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
        cx = float(np.mean(used_positions[:, 0]))
        grid_x_range = (cx - 100.0, cx + 100.0)
    if grid_y_range is None:
        cy = float(np.mean(used_positions[:, 1]))
        grid_y_range = (cy - 100.0, cy + 100.0)

    grid_x = np.arange(grid_x_range[0], grid_x_range[1] + 0.5 * grid_spacing,
                        grid_spacing)
    grid_y = np.arange(grid_y_range[0], grid_y_range[1] + 0.5 * grid_spacing,
                        grid_spacing)

    # ── Travel-time table ───────────────────────────────────────────────
    tt_sec = compute_travel_times(grid_x, grid_y, used_positions, sound_speed,
                                  use_cuda=use_cuda)
    tt_samples = np.round(tt_sec / dt).astype(np.int64)

    # ── Bandpass filter bank ────────────────────────────────────────────
    filters = create_filter_bank(fundamental, n_harmonics,
                                 harmonic_bandwidth, fs)
    filtered = apply_filter_bank(working_traces, filters) if filters else working_traces.copy()

    # ── Sliding windows ─────────────────────────────────────────────────
    win_len = max(int(round(window_length * fs)), 1)
    hop = max(int(round(win_len * (1.0 - window_overlap))), 1)

    detections: list[dict] = []
    multi_detections: list[list[dict]] = []
    history: list[np.ndarray] = []

    pos = 0
    while pos + win_len <= n_samples:
        coh = compute_beam_power(filtered, tt_samples, pos, win_len,
                                 sensor_weights=weights, use_cuda=use_cuda)

        history.append(coh.copy())
        if len(history) > stationary_history:
            history.pop(0)
        mask = detect_stationary(history, stationary_cv_threshold)
        coh_masked = coh.copy()
        coh_masked[mask] = 0.0

        t_center = (pos + win_len / 2.0) * dt

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

        if peaks:
            best = peaks[0]
            detections.append({
                "time": t_center,
                "x": best["x"], "y": best["y"],
                "coherence": best["coherence"],
                "detected": True, "beam_power_map": coh,
            })
        else:
            detections.append({
                "time": t_center,
                "x": float("nan"), "y": float("nan"),
                "coherence": float(np.max(coh_masked)) if coh_masked.size else 0.0,
                "detected": False, "beam_power_map": coh,
            })

        pos += hop

    return {
        "detections": detections,
        "multi_detections": multi_detections,
        "grid_x": grid_x, "grid_y": grid_y,
        "filtered_traces": filtered,
        "sensor_weights": weights,
        "calibrated_positions": used_positions,
    }

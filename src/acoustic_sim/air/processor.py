"""Frequency-domain matched field processor (MFP) with MVDR beamforming.

Implements broadband MFP on a polar grid optimised for a compact
microphone array where bearing is the primary observable and range
is poorly resolved.

Algorithm (per time window)
===========================
1. Hann-taper and FFT each microphone trace.
2. Select frequency bins at the drone harmonic frequencies.
3. Compute the cross-spectral density matrix (CSDM) at each selected
   frequency, averaged over sub-windows.
4. Apply diagonal loading for robustness.
5. Compute steering vectors on a polar (azimuth × range) grid.
6. Compute MVDR (Capon) beam power at each grid point and frequency.
   Fall back to conventional if the CSDM is ill-conditioned.
7. Weight each frequency by (f / f_max)² — upper harmonics contribute
   more because the compact array resolves them better.
8. Sum across frequencies → broadband beam power map.
9. Normalise, detect peaks, sub-grid interpolation in polar coords.

Robustness features (carried over)
===================================
* Sensor fault detection / weighting
* Transient blanking
* Stationary source rejection
* Position self-calibration
"""

from __future__ import annotations

import math
from types import ModuleType

import numpy as np
from scipy import signal as sp_signal


def _get_xp(use_cuda: bool) -> tuple[ModuleType, bool]:
    if use_cuda:
        try:
            import cupy  # type: ignore[import-untyped]
            cupy.cuda.Device(0).compute_capability
            return cupy, True
        except Exception:
            pass
    return np, False


# =====================================================================
#  Steering vectors & travel times
# =====================================================================

def build_polar_grid(
    azimuth_spacing_deg: float = 1.0,
    range_min: float = 20.0,
    range_max: float = 500.0,
    range_spacing: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build polar search grid.

    Returns ``(azimuths_rad, ranges_m)`` — 1-D arrays.
    """
    azimuths = np.deg2rad(np.arange(0, 360, azimuth_spacing_deg))
    ranges = np.arange(range_min, range_max + 0.5 * range_spacing, range_spacing)
    return azimuths, ranges


def polar_to_cartesian(
    azimuths: np.ndarray,
    ranges: np.ndarray,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert polar grid to Cartesian.

    Returns ``(gx, gy)`` each of shape ``(n_az, n_range)``.
    """
    az = azimuths[:, np.newaxis]
    rng = ranges[np.newaxis, :]
    gx = center_x + rng * np.cos(az)
    gy = center_y + rng * np.sin(az)
    return gx, gy


def compute_travel_times_polar(
    gx: np.ndarray,
    gy: np.ndarray,
    mic_positions: np.ndarray,
    sound_speed: float = 343.0,
) -> np.ndarray:
    """Travel times from each grid point to each mic.

    Parameters
    ----------
    gx, gy : (n_az, n_range)
    mic_positions : (n_mics, 2)

    Returns
    -------
    (n_az, n_range, n_mics)
    """
    mx = mic_positions[:, 0]  # (n_mics,)
    my = mic_positions[:, 1]
    dx = gx[:, :, np.newaxis] - mx[np.newaxis, np.newaxis, :]
    dy = gy[:, :, np.newaxis] - my[np.newaxis, np.newaxis, :]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return dist / sound_speed


def compute_steering_vectors(
    freqs: np.ndarray,
    travel_times: np.ndarray,
) -> np.ndarray:
    """Complex steering vectors.

    Parameters
    ----------
    freqs : (n_freq,)
    travel_times : (n_az, n_range, n_mics)

    Returns
    -------
    (n_freq, n_az, n_range, n_mics)  complex128
    """
    # a_i(f) = exp(-j 2π f τ_i)
    phase = -2.0 * np.pi * freqs[:, None, None, None] * travel_times[None, :, :, :]
    return np.exp(1j * phase)


# =====================================================================
#  CSDM computation
# =====================================================================

def select_harmonic_bins(
    n_fft: int,
    dt: float,
    fundamental: float,
    n_harmonics: int,
    bandwidth: float,
) -> np.ndarray:
    """Return FFT bin indices for each harmonic ± bandwidth."""
    fs = 1.0 / dt
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    selected = []
    for k in range(1, n_harmonics + 1):
        fc = fundamental * k
        lo = fc - bandwidth
        hi = fc + bandwidth
        mask = (freqs >= lo) & (freqs <= hi) & (freqs < fs / 2)
        selected.extend(np.where(mask)[0].tolist())
    return np.unique(np.array(selected, dtype=int))


def compute_csdm(
    traces: np.ndarray,
    dt: float,
    window_start: int,
    window_length: int,
    freq_bins: np.ndarray,
    n_subwindows: int = 4,
) -> np.ndarray:
    """Cross-spectral density matrix averaged over sub-windows.

    Parameters
    ----------
    traces : (n_mics, n_samples)
    dt : float
    window_start, window_length : int
    freq_bins : (n_freq,)  FFT bin indices
    n_subwindows : int

    Returns
    -------
    (n_freq, n_mics, n_mics) complex128
    """
    n_mics = traces.shape[0]
    sub_len = window_length // max(n_subwindows, 1)
    if sub_len < 4:
        sub_len = window_length
        n_subwindows = 1
    n_freq = len(freq_bins)

    hann = np.hanning(sub_len)
    csdm = np.zeros((n_freq, n_mics, n_mics), dtype=np.complex128)

    for s in range(n_subwindows):
        s0 = window_start + s * sub_len
        s1 = s0 + sub_len
        if s1 > traces.shape[1]:
            break
        # Windowed FFT for each mic.
        spectra = np.zeros((n_mics, sub_len // 2 + 1), dtype=np.complex128)
        for m in range(n_mics):
            spectra[m] = np.fft.rfft(traces[m, s0:s1] * hann)

        # Select bins and outer product.
        for fi, fb in enumerate(freq_bins):
            if fb >= spectra.shape[1]:
                continue
            d = spectra[:, fb]  # (n_mics,) complex
            csdm[fi] += np.outer(d, d.conj())

    csdm /= max(n_subwindows, 1)
    return csdm


# =====================================================================
#  Beamforming
# =====================================================================

def mvdr_beam_power(
    csdm: np.ndarray,
    steering: np.ndarray,
    epsilon: float = 0.01,
) -> np.ndarray:
    """MVDR (Capon) beam power.

    Parameters
    ----------
    csdm : (n_freq, n_mics, n_mics) complex
    steering : (n_freq, n_az, n_range, n_mics) complex
    epsilon : float — diagonal loading fraction

    Returns
    -------
    (n_freq, n_az, n_range)
    """
    n_freq, n_az, n_range, n_mics = steering.shape
    power = np.zeros((n_freq, n_az, n_range))

    for fi in range(n_freq):
        C = csdm[fi].copy()
        # Diagonal loading.
        load = epsilon * np.real(np.trace(C)) / n_mics
        C += load * np.eye(n_mics)

        # Check condition number; fall back to conventional if bad.
        try:
            cond = np.linalg.cond(C)
            if cond > 1e6:
                raise np.linalg.LinAlgError("ill-conditioned")
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            # Conventional fallback: P = a^H C a.
            for ia in range(n_az):
                for ir in range(n_range):
                    a = steering[fi, ia, ir]  # (n_mics,)
                    power[fi, ia, ir] = np.real(a.conj() @ C @ a)
            continue

        # MVDR: P = 1 / (a^H C^{-1} a).
        for ia in range(n_az):
            for ir in range(n_range):
                a = steering[fi, ia, ir]
                denom = np.real(a.conj() @ C_inv @ a)
                power[fi, ia, ir] = 1.0 / max(denom, 1e-30)

    return power


def conventional_beam_power(
    csdm: np.ndarray,
    steering: np.ndarray,
) -> np.ndarray:
    """Conventional (delay-and-sum) beam power: P = a^H C a."""
    n_freq, n_az, n_range, n_mics = steering.shape
    power = np.zeros((n_freq, n_az, n_range))
    for fi in range(n_freq):
        C = csdm[fi]
        for ia in range(n_az):
            for ir in range(n_range):
                a = steering[fi, ia, ir]
                power[fi, ia, ir] = np.real(a.conj() @ C @ a)
    return power


def broadband_weighted_sum(
    beam_powers: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Frequency-weighted broadband sum: w(f) = (f / f_max)².

    Parameters
    ----------
    beam_powers : (n_freq, n_az, n_range)
    freqs : (n_freq,) Hz

    Returns
    -------
    (n_az, n_range)
    """
    f_max = np.max(freqs) if len(freqs) > 0 else 1.0
    weights = (freqs / f_max) ** 2
    return np.tensordot(weights, beam_powers, axes=([0], [0]))


# =====================================================================
#  Peak finding + sub-grid interpolation (polar)
# =====================================================================

def _parabolic_interp(y_m: float, y_c: float, y_p: float) -> float:
    denom = 2.0 * (2.0 * y_c - y_m - y_p)
    if abs(denom) < 1e-30:
        return 0.0
    return (y_m - y_p) / denom


def find_peaks_polar(
    bpm: np.ndarray,
    azimuths: np.ndarray,
    ranges: np.ndarray,
    threshold: float = 0.25,
    max_sources: int = 5,
    min_sep_deg: float = 10.0,
) -> list[dict]:
    """Find peaks in the polar beam-power map with sub-grid interpolation."""
    n_az, n_range = bpm.shape
    d_az = azimuths[1] - azimuths[0] if n_az > 1 else 1.0
    d_rng = ranges[1] - ranges[0] if n_range > 1 else 1.0
    excl_az = max(int(np.ceil(np.deg2rad(min_sep_deg) / d_az)), 1)
    excl_rng = max(int(np.ceil(20.0 / d_rng)), 1)

    coh = bpm.copy()
    peaks: list[dict] = []
    for _ in range(max_sources):
        pv = float(np.max(coh))
        if pv < threshold:
            break
        ia, ir = np.unravel_index(int(np.argmax(coh)), coh.shape)

        # Sub-grid interpolation in azimuth (circular wrap).
        off_az = 0.0
        if n_az > 2:
            ia_m = (ia - 1) % n_az
            ia_p = (ia + 1) % n_az
            off_az = _parabolic_interp(coh[ia_m, ir], coh[ia, ir], coh[ia_p, ir])
        off_rng = 0.0
        if 0 < ir < n_range - 1:
            off_rng = _parabolic_interp(coh[ia, ir - 1], coh[ia, ir], coh[ia, ir + 1])

        az_refined = float(azimuths[ia]) + off_az * d_az
        rng_refined = float(ranges[ir]) + off_rng * d_rng

        peaks.append({
            "bearing": az_refined,
            "bearing_deg": float(np.degrees(az_refined)) % 360,
            "range": max(rng_refined, 1.0),
            "coherence": pv,
        })

        # Mask exclusion zone.
        for di in range(-excl_az, excl_az + 1):
            idx = (ia + di) % n_az
            lo_r = max(ir - excl_rng, 0)
            hi_r = min(ir + excl_rng + 1, n_range)
            coh[idx, lo_r:hi_r] = 0.0

    return peaks


# =====================================================================
#  Sensor fault detection + transient blanking (reused)
# =====================================================================

def compute_sensor_weights(
    traces: np.ndarray,
    fault_threshold: float = 10.0,
) -> np.ndarray:
    """Per-sensor reliability weights (vectorised)."""
    power = np.sum(traces ** 2, axis=1)
    median_p = float(np.median(power))
    if median_p < 1e-30:
        return np.ones(traces.shape[0])
    ratio = power / median_p
    weights = np.ones(traces.shape[0])
    weights[ratio > fault_threshold] = 0.0
    weights[ratio < 1.0 / fault_threshold] = 0.0
    return weights


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
    reshaped = traces[:, :usable].reshape(n_mics, n_subs, sub_len)
    energies = np.sum(reshaped ** 2, axis=(0, 2))
    med = float(np.median(energies))
    if med < 1e-30:
        return traces.copy()
    bad = energies > threshold_factor * med
    out = traces.copy()
    bad_mask = np.repeat(bad, sub_len)
    out[:, :usable][:, bad_mask] = 0.0
    return out


def detect_stationary(
    history: list[np.ndarray],
    cv_threshold: float = 0.15,
) -> np.ndarray:
    """Identify stationary grid points from beam-power history."""
    if len(history) < 2:
        return np.zeros(history[0].shape, dtype=bool)
    stack = np.stack(history, axis=0)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    cv = std / np.maximum(mean, 1e-12)
    return (cv < cv_threshold) & (mean > 0.5 * np.mean(mean))


def calibrate_positions(
    traces: np.ndarray,
    reported_positions: np.ndarray,
    dt: float,
    sound_speed: float = 343.0,
    max_lag_m: float = 2.0,
) -> np.ndarray:
    """Cross-correlation TDOA self-calibration (unchanged)."""
    n_mics = traces.shape[0]
    max_lag_samples = int(np.ceil(max_lag_m / sound_speed / dt))
    centre = np.mean(reported_positions, axis=0)

    pairs, obs, pred = [], [], []
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            if max_lag_samples > 0 and max_lag_samples < traces.shape[1] // 2:
                cc = np.correlate(traces[i, max_lag_samples:-max_lag_samples],
                                  traces[j], mode="valid")
            else:
                cc = np.correlate(traces[i], traces[j], mode="full")
                max_lag_samples = 0
            if len(cc) == 0:
                continue
            pk = int(np.argmax(np.abs(cc)))
            lag = pk - max_lag_samples if max_lag_samples > 0 else pk - len(cc) // 2
            obs.append(lag * dt)
            d_i = float(np.linalg.norm(reported_positions[i] - centre))
            d_j = float(np.linalg.norm(reported_positions[j] - centre))
            pred.append((d_i - d_j) / sound_speed)
            pairs.append((i, j))

    if len(pairs) < n_mics:
        return reported_positions.copy()

    n_p = len(pairs)
    A = np.zeros((n_p + 2, 2 * n_mics))
    b = np.zeros(n_p + 2)
    for p, (i, j) in enumerate(pairs):
        diff = reported_positions[j] - reported_positions[i]
        d = max(np.linalg.norm(diff), 1e-6)
        n_hat = diff / d
        A[p, 2*i]   = -n_hat[0] / sound_speed
        A[p, 2*i+1] = -n_hat[1] / sound_speed
        A[p, 2*j]   =  n_hat[0] / sound_speed
        A[p, 2*j+1] =  n_hat[1] / sound_speed
        b[p] = obs[p] - pred[p]
    A[n_p, 0::2] = 1.0 / n_mics
    A[n_p+1, 1::2] = 1.0 / n_mics
    delta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return reported_positions + delta.reshape(n_mics, 2)


# =====================================================================
#  Main processor
# =====================================================================

def matched_field_process(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    *,
    sound_speed: float = 343.0,
    # Polar grid
    azimuth_spacing_deg: float = 1.0,
    range_min: float = 20.0,
    range_max: float = 500.0,
    range_spacing: float = 5.0,
    # Window
    window_length: float = 0.2,
    window_overlap: float = 0.5,
    n_subwindows: int = 4,
    # Detection
    detection_threshold: float = 0.25,
    min_signal_rms: float = 0.0,  # Minimum RMS to consider window valid
    # Harmonics
    fundamental: float = 150.0,
    n_harmonics: int = 6,
    harmonic_bandwidth: float = 10.0,
    # Stationary rejection
    stationary_history: int = 10,
    stationary_cv_threshold: float = 0.15,
    # MVDR
    diagonal_loading: float = 0.01,
    # Robustness
    enable_sensor_weights: bool = False,
    sensor_fault_threshold: float = 10.0,
    enable_transient_blanking: bool = False,
    transient_subwindow_ms: float = 5.0,
    transient_threshold_factor: float = 10.0,
    enable_position_calibration: bool = False,
    position_calibration_max_lag_m: float = 2.0,
    # Multi-source
    max_sources: int = 1,
    min_source_separation_deg: float = 10.0,
    # Legacy Cartesian params (ignored — kept for API compat)
    grid_spacing: float = 5.0,
    grid_x_range: tuple[float, float] | None = None,
    grid_y_range: tuple[float, float] | None = None,
    min_source_separation_m: float = 20.0,
    # CUDA
    use_cuda: bool = False,
) -> dict:
    """Broadband frequency-domain MVDR matched field processing.

    Takes ONLY sensor observables.  Returns detections per window on
    a polar grid plus the broadband beam-power maps.
    """
    n_mics, n_samples = traces.shape
    fs = 1.0 / dt

    # ── Robustness pre-processing ───────────────────────────────────────
    # Use mic positions directly - they define where time series were sampled.
    # No coordinate transformation needed.
    used_positions = mic_positions.copy()
    
    if enable_position_calibration:
        used_positions = calibrate_positions(
            traces, used_positions, dt, sound_speed,
            max_lag_m=position_calibration_max_lag_m,
        )
    working = traces
    if enable_transient_blanking:
        working = blank_transients(traces, dt,
                                   subwindow_ms=transient_subwindow_ms,
                                   threshold_factor=transient_threshold_factor)
    weights = np.ones(n_mics)
    if enable_sensor_weights:
        weights = compute_sensor_weights(working, sensor_fault_threshold)

    # Zero-weight faulty sensors.
    active = weights >= 0.5
    if not np.all(active):
        working = working.copy()
        working[~active] = 0.0

    # ── Polar grid + travel times + steering vectors ────────────────────
    azimuths, ranges = build_polar_grid(azimuth_spacing_deg,
                                         range_min, range_max, range_spacing)
    # Mic positions are relative to array centre; grid is absolute.
    # Convert grid to absolute Cartesian using the array centroid.
    cx = float(np.mean(used_positions[:, 0]))
    cy = float(np.mean(used_positions[:, 1]))
    gx, gy = polar_to_cartesian(azimuths, ranges, cx, cy)
    tt = compute_travel_times_polar(gx, gy, used_positions, sound_speed)

    # ── Identify harmonic frequency bins ────────────────────────────────
    win_len = max(int(round(window_length * fs)), 1)
    sub_len = win_len // max(n_subwindows, 1)
    if sub_len < 4:
        sub_len = win_len
    freq_bins = select_harmonic_bins(sub_len, dt, fundamental,
                                     n_harmonics, harmonic_bandwidth)
    if len(freq_bins) == 0:
        # No harmonics below Nyquist.
        freq_bins = np.array([1])

    fft_freqs = np.fft.rfftfreq(sub_len, d=dt)
    selected_freqs = fft_freqs[freq_bins[freq_bins < len(fft_freqs)]]
    if len(selected_freqs) == 0:
        selected_freqs = np.array([fundamental])

    # Pre-compute steering vectors for selected frequencies.
    sv = compute_steering_vectors(selected_freqs, tt)  # (n_freq, n_az, n_rng, n_mics)

    # ── Sliding windows ─────────────────────────────────────────────────
    hop = max(int(round(win_len * (1.0 - window_overlap))), 1)
    detections: list[dict] = []
    multi_detections: list[list[dict]] = []
    history: list[np.ndarray] = []

    pos = 0
    while pos + win_len <= n_samples:
        # Check signal level before processing.
        window_data = working[:, pos:pos+win_len]
        window_rms = float(np.sqrt(np.mean(window_data**2)))
        
        t_center = (pos + win_len / 2.0) * dt
        
        # Skip if signal too weak (no source present).
        if window_rms < min_signal_rms:
            detections.append({
                "time": t_center,
                "bearing": float("nan"),
                "bearing_deg": float("nan"),
                "range": float("nan"),
                "x": float("nan"), "y": float("nan"),
                "coherence": 0.0,
                "detected": False,
                "beam_power_map": np.zeros((len(azimuths), len(ranges))),
                "window_rms": window_rms,
            })
            multi_detections.append([])
            pos += hop
            continue
        
        # CSDM.
        csdm = compute_csdm(working, dt, pos, win_len,
                            freq_bins, n_subwindows)
        # MVDR beam power.
        bp = mvdr_beam_power(csdm, sv, diagonal_loading)
        # Broadband weighted sum.
        bpm = broadband_weighted_sum(bp, selected_freqs)

        # Normalise.
        bpm_max = float(np.max(bpm))
        if bpm_max > 1e-30:
            bpm /= bpm_max

        # Stationary rejection.
        history.append(bpm.copy())
        if len(history) > stationary_history:
            history.pop(0)
        stat_mask = detect_stationary(history, stationary_cv_threshold)
        bpm_masked = bpm.copy()
        bpm_masked[stat_mask] = 0.0

        # Peak finding.
        peaks = find_peaks_polar(
            bpm_masked, azimuths, ranges,
            threshold=detection_threshold,
            max_sources=max_sources,
            min_sep_deg=min_source_separation_deg,
        )
        # Convert polar peaks to Cartesian for compatibility.
        for pk in peaks:
            pk["time"] = t_center
            pk["detected"] = True
            pk["x"] = cx + pk["range"] * math.cos(pk["bearing"])
            pk["y"] = cy + pk["range"] * math.sin(pk["bearing"])
        multi_detections.append(peaks)

        if peaks:
            best = peaks[0]
            detections.append({
                "time": t_center,
                "bearing": best["bearing"],
                "bearing_deg": best["bearing_deg"],
                "range": best["range"],
                "x": best["x"], "y": best["y"],
                "coherence": best["coherence"],
                "detected": True,
                "beam_power_map": bpm,
                "window_rms": window_rms,
            })
        else:
            detections.append({
                "time": t_center,
                "bearing": float("nan"),
                "bearing_deg": float("nan"),
                "range": float("nan"),
                "x": float("nan"), "y": float("nan"),
                "coherence": 0.0,
                "detected": False,
                "beam_power_map": bpm,
                "window_rms": window_rms,
            })

        pos += hop

    return {
        "detections": detections,
        "multi_detections": multi_detections,
        "azimuths": azimuths,
        "ranges": ranges,
        "grid_x": gx,
        "grid_y": gy,
        "sensor_weights": weights,
        "calibrated_positions": used_positions,
        "selected_freqs": selected_freqs,
        "filtered_traces": working,  # for gather plot compat
    }


# ═════════════════════════════════════════════════════════════════════
# Merged from processor_3d
# ═════════════════════════════════════════════════════════════════════



# =====================================================================
#  3D grid + travel times
# =====================================================================

def build_3d_grid(
    azimuth_spacing_deg: float = 1.0,
    range_min: float = 20.0,
    range_max: float = 500.0,
    range_spacing: float = 5.0,
    z_min: float = 0.0,
    z_max: float = 200.0,
    z_spacing: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build polar+z search grid.

    Returns ``(azimuths_rad, ranges_m, z_values_m)`` — 1-D arrays.
    """
    azimuths = np.deg2rad(np.arange(0, 360, azimuth_spacing_deg))
    ranges = np.arange(range_min, range_max + 0.5 * range_spacing, range_spacing)
    z_values = np.arange(z_min, z_max + 0.5 * z_spacing, z_spacing)
    if len(z_values) == 0:
        z_values = np.array([0.0])
    return azimuths, ranges, z_values


def compute_travel_times_3d(
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    mic_positions: np.ndarray,
    sound_speed: float = 343.0,
) -> np.ndarray:
    """Travel times from each 3D grid point to each mic.

    Parameters
    ----------
    gx, gy : (n_az, n_range) — Cartesian x,y of horizontal grid.
    gz : (n_gz,) — z-values for the vertical grid.
    mic_positions : (n_mics, 2) or (n_mics, 3)

    Returns
    -------
    (n_az, n_range, n_gz, n_mics)
    """
    mic_pos = np.asarray(mic_positions, dtype=np.float64)
    if mic_pos.ndim == 1:
        mic_pos = mic_pos.reshape(1, -1)
    if mic_pos.shape[1] == 2:
        mic_pos = np.column_stack([mic_pos, np.zeros(mic_pos.shape[0])])

    n_az, n_range = gx.shape
    n_gz = len(gz)
    n_mics = mic_pos.shape[0]

    mx = mic_pos[:, 0]  # (n_mics,)
    my = mic_pos[:, 1]
    mz = mic_pos[:, 2]

    # dx, dy: (n_az, n_range, n_mics)
    dx = gx[:, :, np.newaxis] - mx[np.newaxis, np.newaxis, :]
    dy = gy[:, :, np.newaxis] - my[np.newaxis, np.newaxis, :]
    horiz_sq = dx ** 2 + dy ** 2  # (n_az, n_range, n_mics)

    # For each z value, compute full 3D distance.
    travel_times = np.zeros((n_az, n_range, n_gz, n_mics))
    for iz, zv in enumerate(gz):
        dz_sq = (zv - mz) ** 2  # (n_mics,)
        dist = np.sqrt(horiz_sq + dz_sq[np.newaxis, np.newaxis, :])
        travel_times[:, :, iz, :] = dist / sound_speed

    return travel_times


def compute_steering_vectors_3d(
    freqs: np.ndarray,
    travel_times: np.ndarray,
) -> np.ndarray:
    """Complex steering vectors for 3D grid.

    Parameters
    ----------
    freqs : (n_freq,)
    travel_times : (n_az, n_range, n_gz, n_mics)

    Returns
    -------
    (n_freq, n_az, n_range, n_gz, n_mics) complex128
    """
    phase = -2.0 * np.pi * freqs[:, None, None, None, None] * travel_times[None, :, :, :, :]
    return np.exp(1j * phase)


def mvdr_beam_power_3d(
    csdm: np.ndarray,
    steering: np.ndarray,
    epsilon: float = 0.01,
) -> np.ndarray:
    """MVDR beam power for 3D steering vectors.

    Parameters
    ----------
    csdm : (n_freq, n_mics, n_mics) complex
    steering : (n_freq, n_az, n_range, n_gz, n_mics) complex

    Returns
    -------
    (n_freq, n_az, n_range, n_gz)
    """
    n_freq, n_az, n_range, n_gz, n_mics = steering.shape
    power = np.zeros((n_freq, n_az, n_range, n_gz))

    for fi in range(n_freq):
        C = csdm[fi].copy()
        load = epsilon * np.real(np.trace(C)) / n_mics
        C += load * np.eye(n_mics)

        try:
            cond = np.linalg.cond(C)
            if cond > 1e6:
                raise np.linalg.LinAlgError("ill-conditioned")
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            for ia in range(n_az):
                for ir in range(n_range):
                    for iz in range(n_gz):
                        a = steering[fi, ia, ir, iz]
                        power[fi, ia, ir, iz] = np.real(a.conj() @ C @ a)
            continue

        for ia in range(n_az):
            for ir in range(n_range):
                for iz in range(n_gz):
                    a = steering[fi, ia, ir, iz]
                    denom = np.real(a.conj() @ C_inv @ a)
                    power[fi, ia, ir, iz] = 1.0 / max(denom, 1e-30)

    return power


def broadband_weighted_sum_3d(
    beam_powers: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Frequency-weighted broadband sum: w(f) = (f / f_max)².

    Parameters
    ----------
    beam_powers : (n_freq, n_az, n_range, n_gz)
    freqs : (n_freq,)

    Returns
    -------
    (n_az, n_range, n_gz)
    """
    f_max = np.max(freqs) if len(freqs) > 0 else 1.0
    weights = (freqs / f_max) ** 2
    return np.tensordot(weights, beam_powers, axes=([0], [0]))


def find_peaks_3d(
    bpm: np.ndarray,
    azimuths: np.ndarray,
    ranges: np.ndarray,
    z_values: np.ndarray,
    threshold: float = 0.25,
    max_sources: int = 5,
    min_sep_deg: float = 10.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> list[dict]:
    """Find peaks in the 3D beam-power map.

    Parameters
    ----------
    bpm : (n_az, n_range, n_gz)
    azimuths : (n_az,) in radians
    ranges : (n_range,) in metres
    z_values : (n_gz,) in metres

    Returns
    -------
    list of dicts with bearing, range, z, x, y, coherence.
    """
    n_az, n_range, n_gz = bpm.shape

    # If only one z-slice, delegate to 2D peak finder on that slice.
    if n_gz == 1:
        peaks_2d = find_peaks_polar(
            bpm[:, :, 0], azimuths, ranges,
            threshold=threshold, max_sources=max_sources,
            min_sep_deg=min_sep_deg,
        )
        for pk in peaks_2d:
            pk["z"] = float(z_values[0])
            pk["x"] = center_x + pk["range"] * math.cos(pk["bearing"])
            pk["y"] = center_y + pk["range"] * math.sin(pk["bearing"])
        return peaks_2d

    # Full 3D peak finding.
    d_az = azimuths[1] - azimuths[0] if n_az > 1 else 1.0
    d_rng = ranges[1] - ranges[0] if n_range > 1 else 1.0
    excl_az = max(int(np.ceil(np.deg2rad(min_sep_deg) / d_az)), 1)
    excl_rng = max(int(np.ceil(20.0 / d_rng)), 1)
    excl_z = max(n_gz // 5, 1)

    coh = bpm.copy()
    peaks: list[dict] = []
    for _ in range(max_sources):
        pv = float(np.max(coh))
        if pv < threshold:
            break
        flat_idx = int(np.argmax(coh))
        ia, ir, iz = np.unravel_index(flat_idx, coh.shape)

        az_refined = float(azimuths[ia])
        rng_refined = float(ranges[ir])
        z_refined = float(z_values[iz])

        peaks.append({
            "bearing": az_refined,
            "bearing_deg": float(np.degrees(az_refined)) % 360,
            "range": max(rng_refined, 1.0),
            "z": z_refined,
            "coherence": pv,
            "x": center_x + rng_refined * math.cos(az_refined),
            "y": center_y + rng_refined * math.sin(az_refined),
        })

        # Mask exclusion zone.
        for di in range(-excl_az, excl_az + 1):
            idx_a = (ia + di) % n_az
            lo_r = max(ir - excl_rng, 0)
            hi_r = min(ir + excl_rng + 1, n_range)
            lo_z = max(iz - excl_z, 0)
            hi_z = min(iz + excl_z + 1, n_gz)
            coh[idx_a, lo_r:hi_r, lo_z:hi_z] = 0.0

    return peaks


# =====================================================================
#  Main 3D processor
# =====================================================================

def matched_field_process_3d(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    *,
    sound_speed: float = 343.0,
    # Polar grid.
    azimuth_spacing_deg: float = 1.0,
    range_min: float = 20.0,
    range_max: float = 500.0,
    range_spacing: float = 5.0,
    # Z grid.
    z_min: float = 0.0,
    z_max: float = 200.0,
    z_spacing: float = 10.0,
    # Window.
    window_length: float = 0.2,
    window_overlap: float = 0.5,
    n_subwindows: int = 4,
    # Detection.
    detection_threshold: float = 0.25,
    min_signal_rms: float = 0.0,
    # Harmonics.
    fundamental: float = 150.0,
    n_harmonics: int = 6,
    harmonic_bandwidth: float = 10.0,
    # Stationary rejection.
    stationary_history: int = 10,
    stationary_cv_threshold: float = 0.15,
    # MVDR.
    diagonal_loading: float = 0.01,
    # Robustness.
    enable_sensor_weights: bool = False,
    sensor_fault_threshold: float = 10.0,
    enable_transient_blanking: bool = False,
    transient_subwindow_ms: float = 5.0,
    transient_threshold_factor: float = 10.0,
    enable_position_calibration: bool = False,
    position_calibration_max_lag_m: float = 2.0,
    # Multi-source.
    max_sources: int = 1,
    min_source_separation_deg: float = 10.0,
    # CUDA (placeholder).
    use_cuda: bool = False,
) -> dict:
    """3D broadband frequency-domain MVDR matched field processing.

    Extends the 2D processor with a z-dimension in the search grid.
    When ``z_min == z_max == 0``, output is identical to the 2D processor.
    """
    n_mics, n_samples = traces.shape
    fs = 1.0 / dt

    # ── Mic positions: ensure 3D ────────────────────────────────────────
    mic_pos = np.asarray(mic_positions, dtype=np.float64)
    if mic_pos.shape[1] == 2:
        mic_pos = np.column_stack([mic_pos, np.zeros(n_mics)])

    # ── Robustness pre-processing ───────────────────────────────────────
    used_positions = mic_pos.copy()
    working = traces
    if enable_transient_blanking:
        working = blank_transients(traces, dt,
                                   subwindow_ms=transient_subwindow_ms,
                                   threshold_factor=transient_threshold_factor)
    weights = np.ones(n_mics)
    if enable_sensor_weights:
        weights = compute_sensor_weights(working, sensor_fault_threshold)
    active = weights >= 0.5
    if not np.all(active):
        working = working.copy()
        working[~active] = 0.0

    # ── 3D Grid + travel times ──────────────────────────────────────────
    azimuths, ranges, z_values = build_3d_grid(
        azimuth_spacing_deg, range_min, range_max, range_spacing,
        z_min, z_max, z_spacing,
    )
    cx = float(np.mean(used_positions[:, 0]))
    cy = float(np.mean(used_positions[:, 1]))
    gx, gy = polar_to_cartesian(azimuths, ranges, cx, cy)
    tt = compute_travel_times_3d(gx, gy, z_values, used_positions, sound_speed)

    # ── Identify harmonic frequency bins ────────────────────────────────
    win_len = max(int(round(window_length * fs)), 1)
    sub_len = win_len // max(n_subwindows, 1)
    if sub_len < 4:
        sub_len = win_len
    freq_bins = select_harmonic_bins(sub_len, dt, fundamental,
                                     n_harmonics, harmonic_bandwidth)
    if len(freq_bins) == 0:
        freq_bins = np.array([1])

    fft_freqs = np.fft.rfftfreq(sub_len, d=dt)
    selected_freqs = fft_freqs[freq_bins[freq_bins < len(fft_freqs)]]
    if len(selected_freqs) == 0:
        selected_freqs = np.array([fundamental])

    # Pre-compute 3D steering vectors.
    sv = compute_steering_vectors_3d(selected_freqs, tt)

    # ── Sliding windows ─────────────────────────────────────────────────
    hop = max(int(round(win_len * (1.0 - window_overlap))), 1)
    detections: list[dict] = []
    multi_detections: list[list[dict]] = []
    # For stationary rejection, collapse z dimension to 2D.
    history: list[np.ndarray] = []

    pos = 0
    while pos + win_len <= n_samples:
        window_data = working[:, pos:pos + win_len]
        window_rms = float(np.sqrt(np.mean(window_data ** 2)))
        t_center = (pos + win_len / 2.0) * dt

        if window_rms < min_signal_rms:
            detections.append({
                "time": t_center,
                "bearing": float("nan"),
                "bearing_deg": float("nan"),
                "range": float("nan"),
                "z": float("nan"),
                "x": float("nan"), "y": float("nan"),
                "coherence": 0.0,
                "detected": False,
                "beam_power_map": np.zeros((len(azimuths), len(ranges))),
                "beam_power_map_3d": np.zeros((len(azimuths), len(ranges), len(z_values))),
                "window_rms": window_rms,
            })
            multi_detections.append([])
            pos += hop
            continue

        # CSDM.
        csdm = compute_csdm(working, dt, pos, win_len,
                            freq_bins, n_subwindows)
        # 3D MVDR beam power.
        bp = mvdr_beam_power_3d(csdm, sv, diagonal_loading)
        # Broadband weighted sum.
        bpm_3d = broadband_weighted_sum_3d(bp, selected_freqs)

        # Normalise.
        bpm_max = float(np.max(bpm_3d))
        if bpm_max > 1e-30:
            bpm_3d /= bpm_max

        # 2D projection for stationary rejection (max over z).
        bpm_2d = np.max(bpm_3d, axis=2)

        history.append(bpm_2d.copy())
        if len(history) > stationary_history:
            history.pop(0)
        stat_mask = detect_stationary(history, stationary_cv_threshold)

        # Apply stationary mask to all z-slices.
        bpm_masked = bpm_3d.copy()
        for iz in range(len(z_values)):
            bpm_masked[:, :, iz][stat_mask] = 0.0

        # Peak finding in 3D.
        peaks = find_peaks_3d(
            bpm_masked, azimuths, ranges, z_values,
            threshold=detection_threshold,
            max_sources=max_sources,
            min_sep_deg=min_source_separation_deg,
            center_x=cx, center_y=cy,
        )
        for pk in peaks:
            pk["time"] = t_center
            pk["detected"] = True
        multi_detections.append(peaks)

        if peaks:
            best = peaks[0]
            detections.append({
                "time": t_center,
                "bearing": best["bearing"],
                "bearing_deg": best["bearing_deg"],
                "range": best["range"],
                "z": best["z"],
                "x": best["x"], "y": best["y"],
                "coherence": best["coherence"],
                "detected": True,
                "beam_power_map": bpm_2d,
                "beam_power_map_3d": bpm_3d,
                "window_rms": window_rms,
            })
        else:
            detections.append({
                "time": t_center,
                "bearing": float("nan"),
                "bearing_deg": float("nan"),
                "range": float("nan"),
                "z": float("nan"),
                "x": float("nan"), "y": float("nan"),
                "coherence": 0.0,
                "detected": False,
                "beam_power_map": bpm_2d,
                "beam_power_map_3d": bpm_3d,
                "window_rms": window_rms,
            })

        pos += hop

    return {
        "detections": detections,
        "multi_detections": multi_detections,
        "azimuths": azimuths,
        "ranges": ranges,
        "z_values": z_values,
        "grid_x": gx,
        "grid_y": gy,
        "sensor_weights": weights,
        "calibrated_positions": used_positions,
        "selected_freqs": selected_freqs,
        "filtered_traces": working,
    }

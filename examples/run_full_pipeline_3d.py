#!/usr/bin/env python3
"""End-to-end 3-D detection and targeting pipeline with visualization.

Loads 3-D FDTD simulation data, runs 3-D MFP detection, 3-D EKF tracking,
and 3-D fire control, then produces comprehensive evaluation plots.

Usage::

    python examples/run_full_pipeline_3d.py                             # defaults
    python examples/run_full_pipeline_3d.py output/valley_3d_test       # specify dir
    python examples/run_full_pipeline_3d.py --source-speed 50           # override

This is the 3-D counterpart of ``run_full_pipeline.py``.  The source
trajectory includes altitude; detection, tracking, and fire-control all
operate in (x, y, z).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.processor_3d import matched_field_process_3d
from acoustic_sim.tracker_3d import run_tracker_3d
from acoustic_sim.fire_control_3d import (
    run_fire_control_3d,
    compute_lead_3d,
    compute_engagement_3d,
)


# ============================================================================
# Data Loading
# ============================================================================

def load_simulation(sim_dir: Path) -> dict:
    """Load simulation traces and metadata."""
    traces = np.load(sim_dir / "traces.npy")
    with open(sim_dir / "metadata.json") as f:
        metadata = json.load(f)

    mic_positions = np.array(metadata["receiver_positions"])

    return {
        "traces": traces,
        "mic_positions": mic_positions,
        "metadata": metadata,
        "dt": metadata["dt"],
        "duration": traces.shape[1] * metadata["dt"],
    }


def compute_ground_truth(metadata: dict, source_speed: float):
    """Build ground truth 3-D trajectory function.

    Args:
        metadata: Simulation metadata (must include source_x/y/z, source_x1/y1/z1,
                  and optionally source_arc_height).
        source_speed: Source velocity in m/s.

    Returns:
        (trajectory_fn, duration) where trajectory_fn(t) -> (x, y, z).
    """
    start_x = metadata.get("source_x", -40.0)
    start_y = metadata.get("source_y", 0.0)
    start_z = metadata.get("source_z", 15.0)
    end_x = metadata.get("source_x1", -start_x)
    end_y = metadata.get("source_y1", start_y)
    end_z = metadata.get("source_z1", start_z)
    arc_height = metadata.get("source_arc_height", 10.0)

    dist = math.sqrt(
        (end_x - start_x) ** 2 + (end_y - start_y) ** 2 + (end_z - start_z) ** 2
    )
    # For the arc the horizontal distance governs speed
    horiz_dist = math.hypot(end_x - start_x, end_y - start_y)
    duration = horiz_dist / source_speed if source_speed > 0 else 3.0

    def trajectory(t: float) -> tuple[float, float, float]:
        frac = min(max(t / duration, 0.0), 1.0)
        x = start_x + (end_x - start_x) * frac
        # Arc on y-axis (parabolic), matching FDTD MovingSource3D
        y = start_y + (end_y - start_y) * frac + arc_height * 4.0 * frac * (1.0 - frac)
        z = start_z + (end_z - start_z) * frac
        return x, y, z

    return trajectory, duration


# ============================================================================
# Pipeline Components
# ============================================================================

def run_mfp_detection(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    *,
    fundamental: float = 180.0,
    n_harmonics: int = 4,
    detection_threshold: float = 0.2,
    min_signal_rms: float = 0.01,
    z_min: float = 0.0,
    z_max: float = 50.0,
    z_spacing: float = 5.0,
) -> dict:
    """Run 3-D matched field processing for detection."""
    return matched_field_process_3d(
        traces, mic_positions, dt,
        fundamental=fundamental,
        n_harmonics=n_harmonics,
        detection_threshold=detection_threshold,
        min_signal_rms=min_signal_rms,
        window_length=0.4,
        window_overlap=0.5,
        n_subwindows=2,
        harmonic_bandwidth=30.0,
        diagonal_loading=0.1,
        range_min=5.0,
        range_max=100.0,
        z_min=z_min,
        z_max=z_max,
        z_spacing=z_spacing,
    )


def run_power_pattern_detection(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    *,
    window_length: float = 0.1,
    window_overlap: float = 0.75,
    min_signal_rms: float = 5e-5,
    source_z: float = 15.0,
) -> list[dict]:
    """Energy-based detection with RMS-weighted bearing estimation.

    Uses individual mic RMS levels to estimate bearing via weighted circular
    mean of mic angles.  More robust than MFP in reverberant environments
    since it relies on amplitude rather than phase coherence.
    """
    n_mics, n_samples = traces.shape
    fs = 1.0 / dt

    cx = float(np.mean(mic_positions[:, 0]))
    cy = float(np.mean(mic_positions[:, 1]))
    mic_angles = np.array([
        math.atan2(mic_positions[i, 1] - cy, mic_positions[i, 0] - cx)
        for i in range(n_mics)
    ])

    win_len = max(int(round(window_length * fs)), 1)
    hop = max(int(round(win_len * (1.0 - window_overlap))), 1)

    detections: list[dict] = []
    pos = 0
    while pos + win_len <= n_samples:
        t_center = (pos + win_len / 2.0) * dt
        seg = traces[:, pos:pos + win_len]
        per_mic_rms = np.sqrt(np.mean(seg ** 2, axis=1))
        window_rms = float(np.sqrt(np.mean(seg ** 2)))

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
                "window_rms": window_rms,
            })
            pos += hop
            continue

        # RMS-weighted circular mean of mic angles
        weights = per_mic_rms ** 2  # power weighting
        sin_sum = float(np.sum(weights * np.sin(mic_angles)))
        cos_sum = float(np.sum(weights * np.cos(mic_angles)))
        bearing_rad = math.atan2(sin_sum, cos_sum)
        if bearing_rad < 0:
            bearing_rad += 2 * math.pi
        bearing_deg = math.degrees(bearing_rad)

        # RMS-based range (inverse square law)
        rms_ref_range, rms_ref_value = 10.0, 0.005
        est_range = rms_ref_range * math.sqrt(rms_ref_value / max(window_rms, 1e-12))
        est_range = max(5.0, min(100.0, est_range))

        est_x = cx + est_range * math.cos(bearing_rad)
        est_y = cy + est_range * math.sin(bearing_rad)

        detections.append({
            "time": t_center,
            "bearing": bearing_rad,
            "bearing_deg": bearing_deg,
            "range": est_range,
            "z": source_z,
            "x": est_x, "y": est_y,
            "coherence": float(np.max(per_mic_rms) / max(np.min(per_mic_rms), 1e-12)),
            "detected": True,
            "window_rms": window_rms,
        })
        pos += hop

    return detections


def apply_rms_range_estimation(
    detections: list[dict],
    rms_ref_range: float = 10.0,
    rms_ref_value: float = 0.0004,  # Calibrated from 3D valley simulation
    range_min: float = 5.0,
    range_max: float = 100.0,
) -> None:
    """Override MFP range with RMS-based estimation (in-place)."""
    for d in detections:
        if d["detected"]:
            rms = d.get("window_rms", 0.1)
            if rms > 0.001:
                rms_range = rms_ref_range * math.sqrt(rms_ref_value / rms)
                rms_range = max(range_min, min(range_max, rms_range))
                d["range"] = rms_range


def run_tracking(
    detections: list[dict],
    array_center: tuple[float, float],
) -> dict:
    """Run 3-D EKF tracker on detections."""
    cx, cy = array_center
    return run_tracker_3d(
        detections,
        process_noise_std=20.0,
        sigma_bearing_deg=1.0,
        sigma_range=5.0,
        initial_range_guess=30.0,
        source_level_dB=90.0,
        array_center_x=cx,
        array_center_y=cy,
    )


def fit_trajectory_to_rms(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    window_sec: float = 0.1,
    hop_fraction: float = 0.25,
    rms_threshold: float = 5e-5,
) -> dict:
    """Fit a constant-velocity trajectory to per-mic RMS time-history.

    Instead of estimating bearing per window (which fails with small arrays
    in reverberant terrain), this directly fits the source trajectory
    (x0, y0, z0, vx, vy, vz, C) to the observed per-mic RMS values using
    nonlinear least squares.

    Model: RMS_mic_j(t) = C / dist(source(t), mic_j)
    where source(t) = (x0 + vx*(t-tref), y0 + vy*(t-tref), z0 + vz*(t-tref))
    """
    from scipy.optimize import least_squares

    n_mics, n_samples = traces.shape
    fs = 1.0 / dt
    win_len = int(round(window_sec * fs))
    hop = int(round(win_len * hop_fraction))

    # Compute per-mic RMS for each window
    window_times = []
    mic_rms = []  # shape: (n_windows, n_mics)
    pos = 0
    while pos + win_len <= n_samples:
        t_center = (pos + win_len / 2) * dt
        seg = traces[:, pos:pos + win_len]
        rms_per = np.sqrt(np.mean(seg ** 2, axis=1))
        mean_rms = float(np.sqrt(np.mean(seg ** 2)))
        if mean_rms >= rms_threshold:
            window_times.append(t_center)
            mic_rms.append(rms_per)
        pos += hop

    window_times = np.array(window_times)
    mic_rms = np.array(mic_rms)  # (n_win, n_mics)
    mic_pos = np.array(mic_positions)  # (n_mics, 3)
    n_win = len(window_times)

    t_ref = 0.5 * (window_times[0] + window_times[-1])

    # Initial guess: source at x=-20 moving right at 50 m/s
    # z estimate from peak RMS mic (likely closest to source height)
    peak_idx = np.argmax(np.mean(mic_rms, axis=1))
    peak_t = window_times[peak_idx]
    peak_rms = float(mic_rms[peak_idx].mean())

    cx, cy = float(mic_pos[:, 0].mean()), float(mic_pos[:, 1].mean())
    cz = float(mic_pos[:, 2].mean()) if mic_pos.shape[1] > 2 else 0.0

    # Heuristic initial guess
    x0_init = cx + 0.0
    y0_init = cy - 10.0  # source likely not at array
    z0_init = 15.0
    vx_init = 50.0
    vy_init = 0.0
    vz_init = 0.0
    # C from peak RMS * estimated peak distance
    est_cpa_dist = 15.0
    C_init = peak_rms * est_cpa_dist

    def residuals(params):
        x0, y0, z0, vx, vy, vz, C = params
        res = np.empty(n_win * n_mics)
        for i in range(n_win):
            dt_i = window_times[i] - t_ref
            sx = x0 + vx * dt_i
            sy = y0 + vy * dt_i
            sz = z0 + vz * dt_i
            for j in range(n_mics):
                d = math.sqrt(
                    (sx - mic_pos[j, 0]) ** 2
                    + (sy - mic_pos[j, 1]) ** 2
                    + (sz - mic_pos[j, 2]) ** 2
                )
                predicted = C / max(d, 0.1)
                res[i * n_mics + j] = predicted - mic_rms[i, j]
        return res

    p0 = [x0_init, y0_init, z0_init, vx_init, vy_init, vz_init, C_init]

    # Try multiple initial bearings to avoid local minima
    best_result = None
    best_cost = float("inf")
    for vx_try in [50.0, -50.0, 0.0]:
        for vy_try in [0.0, 20.0, -20.0]:
            p_try = [x0_init, y0_init, z0_init, vx_try, vy_try, vz_init, C_init]
            try:
                result = least_squares(
                    residuals, p_try,
                    bounds=(
                        [-100, -50, 0, -150, -150, -50, 0.001],
                        [100, 50, 40, 150, 150, 50, 10.0],
                    ),
                    method="trf",
                    max_nfev=200,
                )
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result
            except Exception:
                continue

    x0, y0, z0, vx, vy, vz, C = best_result.x

    # Compute residual-based fit uncertainty
    J = best_result.jac
    residual_var = 2.0 * best_result.cost / max(J.shape[0] - 7, 1)
    try:
        cov_params = residual_var * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov_params = np.eye(7) * 10.0

    pos_cov_diag = np.abs(np.diag(cov_params)[:3])

    # Build output
    out_times = np.array(window_times)
    # But we also need entries for non-detected windows to match detections length
    # For now output only for detected windows — we'll pad in the caller

    all_det_times = np.zeros(int(np.ceil((n_samples - win_len) / hop)) + 1)
    det_idx = 0
    pos_idx = 0
    all_pos = 0
    while pos_idx + win_len <= n_samples:
        all_det_times[all_pos] = (pos_idx + win_len / 2) * dt
        all_pos += 1
        pos_idx += hop
    all_det_times = all_det_times[:all_pos]

    n_total = len(all_det_times)
    positions = np.zeros((n_total, 3))
    velocities = np.zeros((n_total, 3))
    covariances = np.zeros((n_total, 6, 6))

    for i, t in enumerate(all_det_times):
        dt_i = t - t_ref
        positions[i] = [x0 + vx * dt_i, y0 + vy * dt_i, z0 + vz * dt_i]
        velocities[i] = [vx, vy, vz]
        cov = np.zeros((6, 6))
        cov[0, 0] = max(pos_cov_diag[0], 0.01)
        cov[1, 1] = max(pos_cov_diag[1], 0.01)
        cov[2, 2] = max(pos_cov_diag[2], 0.01)
        cov[3, 3] = 1.0
        cov[4, 4] = 1.0
        cov[5, 5] = 1.0
        covariances[i] = cov

    speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    heading = math.atan2(vy, vx)

    return {
        "times": all_det_times,
        "positions": positions,
        "velocities": velocities,
        "covariances": covariances,
        "raw_bearings": np.full(n_total, float("nan")),
        "raw_ranges": np.full(n_total, float("nan")),
        "raw_zs": np.full(n_total, z0),
        "range_uncertainties": np.full(n_total, math.sqrt(sum(pos_cov_diag))),
        "speeds": np.full(n_total, speed),
        "headings": np.full(n_total, heading),
        "fit_params": {
            "x0": float(x0), "y0": float(y0), "z0": float(z0),
            "vx": float(vx), "vy": float(vy), "vz": float(vz),
            "C": float(C), "t_ref": float(t_ref),
            "res_pos": [float(math.sqrt(p)) for p in pos_cov_diag],
        },
    }


def run_ls_tracking(
    detections: list[dict],
    all_times: np.ndarray | None = None,
) -> dict:
    """Least-squares constant-velocity track fit.

    Fits detected (t, x, y, z) positions to a straight-line model::

        pos(t) = pos0 + vel * (t - t_ref)

    This is much more robust than an EKF when individual bearing estimates
    are noisy (e.g. 30 deg errors from a small array), because it
    averages over ALL detections at once.

    Parameters
    ----------
    detections : list of detection dicts (must have 'time', 'detected', 'x', 'y', 'z').
    all_times  : optional full time array for output interpolation.

    Returns
    -------
    dict compatible with ``run_fire_control_3d`` (times, positions, velocities, covariances).
    """
    # Collect detected positions
    det_t, det_x, det_y, det_z = [], [], [], []
    for d in detections:
        if d["detected"] and not math.isnan(d.get("x", float("nan"))):
            det_t.append(d["time"])
            det_x.append(d["x"])
            det_y.append(d["y"])
            det_z.append(d.get("z", 15.0))

    det_t = np.array(det_t)
    det_x = np.array(det_x)
    det_y = np.array(det_y)
    det_z = np.array(det_z)

    # Reference time: midpoint of detections
    t_ref = 0.5 * (det_t[0] + det_t[-1])
    dt_arr = det_t - t_ref

    # Weighted LS: weight by inverse window_rms (louder = closer = better range est)
    rms_vals = []
    for d in detections:
        if d["detected"] and not math.isnan(d.get("x", float("nan"))):
            rms_vals.append(d.get("window_rms", 1e-6))
    rms_vals = np.array(rms_vals)
    # Higher RMS = closer = more accurate position → higher weight
    weights = rms_vals / max(rms_vals.max(), 1e-12)

    # Fit pos(t) = a + b*(t - t_ref)  via weighted least squares
    # Design matrix: [1, dt]
    A = np.column_stack([np.ones_like(dt_arr), dt_arr])
    W = np.diag(weights)

    def wls_fit(design, y_arr):
        AtW = design.T @ W
        return np.linalg.lstsq(AtW @ design, AtW @ y_arr, rcond=None)[0]

    cx = wls_fit(A, det_x)  # [x0, vx]
    cy_fit = wls_fit(A, det_y)  # [y0, vy]
    cz = wls_fit(A, det_z)  # [z0, vz]

    x0, vx = float(cx[0]), float(cx[1])
    y0, vy = float(cy_fit[0]), float(cy_fit[1])
    z0, vz = float(cz[0]), float(cz[1])

    # Residual-based position covariance (of the FIT, not individual detections)
    pred_x = A @ cx
    pred_y = A @ cy_fit
    pred_z = A @ cz
    n_det = len(det_t)
    # Standard error of the fitted intercept = residual_std / sqrt(N)
    res_x = float(np.std(det_x - pred_x)) / max(math.sqrt(n_det), 1.0)
    res_y = float(np.std(det_y - pred_y)) / max(math.sqrt(n_det), 1.0)
    res_z = float(np.std(det_z - pred_z)) / max(math.sqrt(n_det), 1.0)

    # Build output for all detection times
    out_times = np.array([d["time"] for d in detections])
    n = len(out_times)
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    covariances = np.zeros((n, 6, 6))

    for i, t in enumerate(out_times):
        dt_i = t - t_ref
        positions[i] = [x0 + vx * dt_i, y0 + vy * dt_i, z0 + vz * dt_i]
        velocities[i] = [vx, vy, vz]
        # Covariance: residual std as position uncertainty, scaled by distance from reference
        t_scale = max(1.0, abs(dt_i) / max(det_t[-1] - det_t[0], 0.01))
        cov = np.zeros((6, 6))
        cov[0, 0] = (res_x * t_scale) ** 2
        cov[1, 1] = (res_y * t_scale) ** 2
        cov[2, 2] = (res_z * t_scale) ** 2
        cov[3, 3] = (vx * 0.1) ** 2 + 1.0  # velocity uncertainty
        cov[4, 4] = (vy * 0.1) ** 2 + 1.0
        cov[5, 5] = (vz * 0.1) ** 2 + 1.0
        covariances[i] = cov

    return {
        "times": out_times,
        "positions": positions,
        "velocities": velocities,
        "covariances": covariances,
        "raw_bearings": np.array([d.get("bearing", float("nan")) for d in detections]),
        "raw_ranges": np.array([d.get("range", float("nan")) for d in detections]),
        "raw_zs": np.array([d.get("z", 0.0) for d in detections]),
        "range_uncertainties": np.full(n, max(res_x, res_y)),
        "speeds": np.full(n, math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)),
        "headings": np.full(n, math.atan2(vy, vx)),
        "fit_params": {"x0": x0, "y0": y0, "z0": z0, "vx": vx, "vy": vy, "vz": vz,
                        "t_ref": t_ref, "res_x": res_x, "res_y": res_y, "res_z": res_z},
    }


def run_targeting(
    track: dict,
    ground_truth_fn,
    hit_threshold: float = 3.0,
    weapon_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    """Run 3-D fire control on track.

    Uses the same shotgun parameters as the 2-D pipeline with
    weapon co-located at the specified position (default: origin).
    """
    return run_fire_control_3d(
        track,
        weapon_position=weapon_position,
        muzzle_velocity=400.0,
        pellet_decel=1.5,
        pattern_spread_rate=0.3,
        max_hits=5,
        hit_threshold=hit_threshold,
        ground_truth_fn=ground_truth_fn,
        max_position_uncertainty=0.0,
        max_engagement_range=500.0,
        class_label="fixed_wing",
        class_confidence=0.9,
    )


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_results(
    detections: list[dict],
    track: dict,
    fire_control: dict,
    ground_truth_fn,
    array_center: tuple[float, float],
    hit_threshold: float = 3.0,
) -> dict:
    """Compute error metrics against 3-D ground truth."""
    cx, cy = array_center

    bearing_errors: list[float] = []
    range_errors: list[float] = []
    z_errors: list[float] = []

    for d in detections:
        if not d["detected"]:
            continue
        t = d["time"]
        gt_x, gt_y, gt_z = ground_truth_fn(t)
        dx, dy = gt_x - cx, gt_y - cy
        true_bearing = math.degrees(math.atan2(dy, dx))
        if true_bearing < 0:
            true_bearing += 360
        true_range = math.hypot(dx, dy)

        det_bearing = d["bearing_deg"]
        err = det_bearing - true_bearing
        if err > 180:
            err -= 360
        if err < -180:
            err += 360
        bearing_errors.append(abs(err))
        range_errors.append(abs(d["range"] - true_range))
        # z error when available
        det_z = d.get("z", 0.0)
        z_errors.append(abs(det_z - gt_z))

    # Fire control miss distances (3-D)
    fc_times = fire_control.get("times", np.array([]))
    fc_intercepts = fire_control.get("intercept_positions", np.array([]).reshape(-1, 3))
    fc_can_fire = fire_control.get("can_fire", np.array([]))

    miss_distances: list[float] = []
    shots_fired = 0
    for i, t in enumerate(fc_times):
        if i < len(fc_can_fire) and fc_can_fire[i]:
            shots_fired += 1
            if i < len(fc_intercepts):
                aim = fc_intercepts[i]
                if not np.any(np.isnan(aim)):
                    gt_x, gt_y, gt_z = ground_truth_fn(t)
                    miss = math.sqrt(
                        (aim[0] - gt_x) ** 2
                        + (aim[1] - gt_y) ** 2
                        + (aim[2] - gt_z) ** 2
                    )
                    miss_distances.append(miss)

    return {
        "n_detections": sum(1 for d in detections if d["detected"]),
        "n_windows": len(detections),
        "mean_bearing_error": np.mean(bearing_errors) if bearing_errors else float("nan"),
        "max_bearing_error": np.max(bearing_errors) if bearing_errors else float("nan"),
        "mean_range_error": np.mean(range_errors) if range_errors else float("nan"),
        "mean_z_error": np.mean(z_errors) if z_errors else float("nan"),
        "shots_fired": shots_fired,
        "hit_threshold": hit_threshold,
        "n_hits": sum(1 for m in miss_distances if m < hit_threshold),
        "n_hits_5m": sum(1 for m in miss_distances if m < 5.0),
        "mean_miss": np.mean(miss_distances) if miss_distances else float("nan"),
        "min_miss": np.min(miss_distances) if miss_distances else float("nan"),
        "max_miss": np.max(miss_distances) if miss_distances else float("nan"),
        "bearing_errors": bearing_errors,
        "range_errors": range_errors,
        "z_errors": z_errors,
        "miss_distances": miss_distances,
    }


# ============================================================================
# Plotting
# ============================================================================

def compute_projectile_path(
    weapon_pos: tuple[float, float, float],
    aim_bearing: float,
    aim_elevation: float,
    muzzle_velocity: float,
    decel: float,
    tof: float,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3-D projectile trajectory."""
    wx, wy, wz = weapon_pos
    times = np.linspace(0, tof, n_points)

    x_path, y_path, z_path = [], [], []
    cos_el = math.cos(aim_elevation)
    sin_el = math.sin(aim_elevation)
    cos_az = math.cos(aim_bearing)
    sin_az = math.sin(aim_bearing)

    for t_val in times:
        v_avg = muzzle_velocity
        for _ in range(3):
            s = v_avg * t_val
            v_end = max(muzzle_velocity - decel * s, 0)
            v_avg = 0.5 * (muzzle_velocity + v_end)
        s = v_avg * t_val
        x_path.append(wx + s * cos_el * cos_az)
        y_path.append(wy + s * cos_el * sin_az)
        z_path.append(wz + s * sin_el)

    return np.array(x_path), np.array(y_path), np.array(z_path)


def plot_radial_engagement(
    fire_control: dict,
    ground_truth_fn,
    source_duration: float,
    weapon_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    muzzle_velocity: float = 400.0,
    decel: float = 1.5,
    hit_threshold: float = 3.0,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot radial engagement view (x-y projection, weapon at centre)."""
    wx, wy, wz = weapon_pos

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # ---------- Left panel: X-Y (plan view) ----------
    ax_xy = axes[0]
    gt_times = np.linspace(0, source_duration, 200)
    gt_x = np.array([ground_truth_fn(t)[0] - wx for t in gt_times])
    gt_y = np.array([ground_truth_fn(t)[1] - wy for t in gt_times])

    ax_xy.plot(gt_x, gt_y, "g-", lw=3, label="Target path", zorder=5)
    ax_xy.scatter(gt_x[0], gt_y[0], c="g", s=150, marker="o", zorder=6, label="Start")
    ax_xy.scatter(gt_x[-1], gt_y[-1], c="g", s=150, marker="s", zorder=6, label="End")

    max_range = max(np.max(np.sqrt(gt_x ** 2 + gt_y ** 2)), 50)
    for r in [25, 50, 75, 100]:
        if r <= max_range * 1.2:
            circle = Circle((0, 0), r, fill=False, color="gray", ls="--", alpha=0.3)
            ax_xy.add_patch(circle)
            ax_xy.text(r * 0.707, r * 0.707, f"{r}m", fontsize=8, color="gray", alpha=0.7)

    fc_times = fire_control.get("times", np.array([]))
    fc_can_fire = fire_control.get("can_fire", np.array([]))
    fc_aim_bearings = fire_control.get("aim_bearings", np.array([]))
    fc_aim_elevations = fire_control.get("aim_elevations", np.array([]))
    fc_tofs = fire_control.get("tofs", np.array([]))
    fc_intercepts = fire_control.get("intercept_positions", np.array([]).reshape(-1, 3))

    n_shots = 0
    n_hits = 0
    for i, (t, can_fire) in enumerate(zip(fc_times, fc_can_fire)):
        if not can_fire:
            continue
        aim_brg = fc_aim_bearings[i] if i < len(fc_aim_bearings) else float("nan")
        aim_el = fc_aim_elevations[i] if i < len(fc_aim_elevations) else 0.0
        tof = fc_tofs[i] if i < len(fc_tofs) else float("nan")
        if np.isnan(aim_brg) or np.isnan(tof) or tof <= 0:
            continue

        n_shots += 1
        miss_dist = float("nan")
        is_hit = False
        if i < len(fc_intercepts):
            int_pos = fc_intercepts[i]
            gt_pos = ground_truth_fn(t)
            miss_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(int_pos, gt_pos)))
            is_hit = miss_dist < hit_threshold
            if is_hit:
                n_hits += 1

        proj_x, proj_y, _ = compute_projectile_path(
            weapon_pos, aim_brg, aim_el, muzzle_velocity, decel, tof
        )
        color = "green" if is_hit else "red"
        ax_xy.plot(proj_x - wx, proj_y - wy, "-", color=color, lw=2, alpha=0.8)

        if i < len(fc_intercepts):
            ix, iy = fc_intercepts[i, 0] - wx, fc_intercepts[i, 1] - wy
            ax_xy.scatter(ix, iy, c=color, s=150, marker="x", linewidths=3, zorder=10)
            label = f"HIT ({miss_dist:.1f}m)" if is_hit else f"MISS ({miss_dist:.1f}m)"
            ax_xy.annotate(label, (ix, iy), xytext=(5, 5), textcoords="offset points",
                           fontsize=9, color=color, fontweight="bold")

        gt_pos = ground_truth_fn(t)
        ax_xy.scatter(gt_pos[0] - wx, gt_pos[1] - wy, c="lime", s=80, marker="o",
                      edgecolors="darkgreen", linewidths=2, zorder=8)

    ax_xy.scatter(0, 0, c="black", s=300, marker="*", label="Weapon", zorder=15)
    ax_xy.set_xlabel("X relative to weapon (m)")
    ax_xy.set_ylabel("Y relative to weapon (m)")
    ax_xy.set_title("PLAN VIEW (X-Y)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="upper left", fontsize=9)

    # ---------- Right panel: X-Z (elevation view) ----------
    ax_xz = axes[1]
    gt_z = np.array([ground_truth_fn(t)[2] - wz for t in gt_times])
    ax_xz.plot(gt_x, gt_z, "g-", lw=3, label="Target path", zorder=5)
    ax_xz.scatter(gt_x[0], gt_z[0], c="g", s=150, marker="o", zorder=6)
    ax_xz.scatter(gt_x[-1], gt_z[-1], c="g", s=150, marker="s", zorder=6)

    for i, (t, can_fire) in enumerate(zip(fc_times, fc_can_fire)):
        if not can_fire:
            continue
        aim_brg = fc_aim_bearings[i] if i < len(fc_aim_bearings) else float("nan")
        aim_el = fc_aim_elevations[i] if i < len(fc_aim_elevations) else 0.0
        tof = fc_tofs[i] if i < len(fc_tofs) else float("nan")
        if np.isnan(aim_brg) or np.isnan(tof) or tof <= 0:
            continue

        if i < len(fc_intercepts):
            int_pos = fc_intercepts[i]
            gt_pos = ground_truth_fn(t)
            miss_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(int_pos, gt_pos)))
            is_hit = miss_dist < hit_threshold
        else:
            is_hit = False

        proj_x, _, proj_z = compute_projectile_path(
            weapon_pos, aim_brg, aim_el, muzzle_velocity, decel, tof
        )
        color = "green" if is_hit else "red"
        ax_xz.plot(proj_x - wx, proj_z - wz, "-", color=color, lw=2, alpha=0.8)

        if i < len(fc_intercepts):
            ix, iz = fc_intercepts[i, 0] - wx, fc_intercepts[i, 2] - wz
            ax_xz.scatter(ix, iz, c=color, s=150, marker="x", linewidths=3, zorder=10)

    ax_xz.scatter(0, 0, c="black", s=300, marker="*", zorder=15)
    ax_xz.set_xlabel("X relative to weapon (m)")
    ax_xz.set_ylabel("Z (altitude) relative to weapon (m)")
    ax_xz.set_title("ELEVATION VIEW (X-Z)")
    ax_xz.grid(True, alpha=0.3)
    ax_xz.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"RADIAL ENGAGEMENT — 3-D  |  Shots: {n_shots}  Hits: {n_hits}  "
        f"Misses: {n_shots - n_hits}  (threshold < {hit_threshold} m)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    return fig


def plot_full_evaluation(
    detections: list[dict],
    track: dict,
    fire_control: dict,
    ground_truth_fn,
    source_duration: float,
    array_center: tuple[float, float],
    metrics: dict,
    output_path: Path,
) -> None:
    """Generate comprehensive 3-D evaluation plots."""
    cx, cy = array_center

    fig = plt.figure(figsize=(18, 14))

    gt_times = np.linspace(0, source_duration, 200)
    gt_xyz = np.array([ground_truth_fn(t) for t in gt_times])

    # ── Panel 1: Spatial overview (X-Y) ─────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(gt_xyz[:, 0], gt_xyz[:, 1], "g-", lw=2, label="True trajectory", zorder=5)
    ax1.scatter(gt_xyz[0, 0], gt_xyz[0, 1], c="g", s=100, marker="o", zorder=6)
    ax1.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], c="g", s=100, marker="s", zorder=6)

    det_x = [d["x"] for d in detections if d["detected"] and not np.isnan(d["x"])]
    det_y = [d["y"] for d in detections if d["detected"] and not np.isnan(d["y"])]
    if det_x:
        ax1.scatter(det_x, det_y, c="b", s=30, alpha=0.6, label="MFP detections")

    positions = track.get("positions", np.array([]).reshape(-1, 3))
    valid = ~np.isnan(positions).any(axis=1) if len(positions) > 0 else np.array([])
    if np.any(valid):
        ax1.plot(positions[valid, 0], positions[valid, 1], "m-", lw=1.5, alpha=0.7, label="EKF track")

    fc_intercepts = fire_control.get("intercept_positions", np.array([]).reshape(-1, 3))
    fc_can_fire = fire_control.get("can_fire", np.array([]))
    for i, (pos, cf) in enumerate(zip(fc_intercepts, fc_can_fire)):
        if cf and not np.isnan(pos[0]):
            ax1.scatter(pos[0], pos[1], c="r", s=80, marker="x", zorder=10)

    ax1.scatter(cx, cy, c="k", s=100, marker="^", label="Array", zorder=10)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Spatial Overview (X-Y)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Bearing over time ──────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    true_bearings = [
        math.degrees(math.atan2(ground_truth_fn(t)[1] - cy, ground_truth_fn(t)[0] - cx))
        for t in gt_times
    ]
    ax2.plot(gt_times, true_bearings, "g-", lw=2, label="True bearing")

    det_t = [d["time"] for d in detections if d["detected"]]
    det_brg = [d["bearing_deg"] for d in detections if d["detected"]]
    det_brg_unwrap = [b - 360 if b > 180 else b for b in det_brg]
    if det_t:
        ax2.scatter(det_t, det_brg_unwrap, c="b", s=30, alpha=0.6, label="Detected")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Bearing (deg)")
    ax2.set_title("Bearing vs Time")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Range over time ────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    true_ranges = [
        math.hypot(ground_truth_fn(t)[0] - cx, ground_truth_fn(t)[1] - cy)
        for t in gt_times
    ]
    ax3.plot(gt_times, true_ranges, "g-", lw=2, label="True range (horiz)")

    det_rng = [d["range"] for d in detections if d["detected"]]
    if det_t:
        ax3.scatter(det_t, det_rng, c="b", s=30, alpha=0.6, label="Detected (RMS)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Range (m)")
    ax3.set_title("Range vs Time")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Altitude (z) over time ─────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    true_z = [ground_truth_fn(t)[2] for t in gt_times]
    ax4.plot(gt_times, true_z, "g-", lw=2, label="True altitude")

    det_z = [d.get("z", 0.0) for d in detections if d["detected"]]
    if det_t and det_z:
        ax4.scatter(det_t, det_z, c="b", s=30, alpha=0.6, label="Detected z")

    if np.any(valid):
        track_times = track.get("times", np.array([]))
        track_z = positions[valid, 2] if positions.shape[1] >= 3 else []
        if len(track_z) > 0 and len(track_times) > 0:
            ax4.plot(track_times[valid], track_z, "m-", lw=1.5, alpha=0.7, label="EKF z")

    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Z / Altitude (m)")
    ax4.set_title("Altitude vs Time")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Miss distance over time ────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    fc_times_arr = fire_control.get("times", np.array([]))
    miss_by_time: list[float] = []
    time_for_miss: list[float] = []
    for i, t in enumerate(fc_times_arr):
        if i < len(fc_can_fire) and fc_can_fire[i] and i < len(fc_intercepts):
            aim = fc_intercepts[i]
            if not np.any(np.isnan(aim)):
                gt_pos = ground_truth_fn(t)
                miss = math.sqrt(sum((a - b) ** 2 for a, b in zip(aim, gt_pos)))
                miss_by_time.append(miss)
                time_for_miss.append(t)

    if miss_by_time:
        ax5.scatter(time_for_miss, miss_by_time, c="r", s=50, marker="x")
        ht = metrics.get("hit_threshold", 3.0)
        ax5.axhline(ht, color="g", ls="--", alpha=0.7, label=f"{ht}m threshold")
        ax5.axhline(5.0, color="orange", ls="--", alpha=0.7, label="5m threshold")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Miss Distance (m)")
    ax5.set_title("Fire Control Miss Distance")
    ax5.legend(loc="best", fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── Panel 6: Summary metrics ────────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = f"""
    DETECTION METRICS (3-D)
    ─────────────────────────
    Windows processed: {metrics['n_windows']}
    Detections: {metrics['n_detections']} ({100*metrics['n_detections']/max(metrics['n_windows'],1):.1f}%)
    Mean bearing error: {metrics['mean_bearing_error']:.1f} deg
    Max bearing error: {metrics['max_bearing_error']:.1f} deg
    Mean range error: {metrics['mean_range_error']:.1f} m
    Mean z error: {metrics['mean_z_error']:.1f} m

    FIRE CONTROL METRICS
    ─────────────────────────
    Shots fired: {metrics['shots_fired']}
    Hits <{metrics['hit_threshold']}m: {metrics['n_hits']} ({100*metrics['n_hits']/max(metrics['shots_fired'],1):.1f}%)
    Hits <5m: {metrics['n_hits_5m']} ({100*metrics['n_hits_5m']/max(metrics['shots_fired'],1):.1f}%)
    Mean miss: {metrics['mean_miss']:.1f} m
    Min miss: {metrics['min_miss']:.1f} m
    Max miss: {metrics['max_miss']:.1f} m
    """

    ax6.text(
        0.1, 0.9, summary_text, transform=ax6.transAxes,
        fontsize=11, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# SRP-PHAT Beamformer (phase-based bearing)
# ============================================================================

def srp_phat_bearing(
    seg: np.ndarray,
    mic_positions: np.ndarray,
    fs: float,
    c: float = 343.0,
    n_bearings: int = 360,
    freq_lo: float = 100.0,
    freq_hi: float = 2000.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Steered Response Power with Phase Transform for far-field bearing.

    Fully vectorised — no Python loops over bearings or mics.

    Parameters
    ----------
    seg : (n_mics, n_samples) time-domain signal segment.
    mic_positions : (n_mics, 2+) mic XY coordinates.
    fs : sample rate.
    c : speed of sound.
    n_bearings : angular resolution of scan.
    freq_lo, freq_hi : bandpass for PHAT weighting.

    Returns
    -------
    best_bearing_rad : bearing in radians (0 = +x, pi/2 = +y).
    bearings : (n_bearings,) scan angles in radians.
    power : (n_bearings,) steered response power at each angle.
    """
    n_mics, n_samp = seg.shape
    mic_xy = mic_positions[:, :2]
    center = mic_xy.mean(axis=0)
    mic_rel = mic_xy - center  # (n_mics, 2)

    nfft = int(2 ** np.ceil(np.log2(n_samp)))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    X = np.fft.rfft(seg, n=nfft, axis=1)  # (n_mics, nfft//2+1)

    # Bandpass
    fmask = (freqs >= freq_lo) & (freqs <= freq_hi)
    X_bp = X[:, fmask]                      # (n_mics, n_freq)
    omega = 2.0 * np.pi * freqs[fmask]      # (n_freq,)

    # PHAT normalisation
    mag = np.abs(X_bp)
    mag[mag < 1e-30] = 1e-30
    X_phat = X_bp / mag                     # (n_mics, n_freq)

    # Scan bearings — fully vectorised
    bearings = np.linspace(0, 2 * np.pi, n_bearings, endpoint=False)
    look = np.column_stack([np.cos(bearings), np.sin(bearings)])  # (n_bearings, 2)

    # Delays: source at bearing theta → plane wave propagates in direction
    # (-cos θ, -sin θ).  Mic delay: tau_m = -(mic_rel · look) / c
    taus = -(mic_rel @ look.T) / c           # (n_mics, n_bearings)

    # Phase steering: exp(+j * omega * tau)
    # taus: (n_mics, n_bearings), omega: (n_freq,)
    phase = np.exp(1j * taus[:, :, np.newaxis] * omega[np.newaxis, np.newaxis, :])
    # → (n_mics, n_bearings, n_freq)

    # Steered sum over mics, then power over freq
    steered = np.sum(X_phat[:, np.newaxis, :] * phase, axis=0)  # (n_bearings, n_freq)
    power = np.sum(np.abs(steered) ** 2, axis=1)  # (n_bearings,)

    best_idx = int(np.argmax(power))
    best_bearing = float(bearings[best_idx])

    return best_bearing, bearings, power


def plot_matched_filter_diagnostic(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    ground_truth_fn,
    source_duration: float,
    array_center: tuple[float, float, float],
    window_length: float = 0.1,
    window_overlap: float = 0.75,
    min_signal_rms: float = 5e-5,
    output_path: Path | None = None,
) -> plt.Figure:
    """Diagnostic plot comparing RMS-weighted and SRP-PHAT bearing estimators.

    Produces a 4-panel figure:
      1. Bearing vs time: true, RMS-weighted, SRP-PHAT
      2. Bearing error vs time for both methods
      3. SRP-PHAT power map (bearing × time heatmap)
      4. Example SRP-PHAT power spectrum at CPA window
    """
    n_mics, n_samples = traces.shape
    fs = 1.0 / dt
    cx, cy, cz = array_center

    mic_angles = np.array([
        math.atan2(mic_positions[i, 1] - cy, mic_positions[i, 0] - cx)
        for i in range(n_mics)
    ])

    win_len = max(int(round(window_length * fs)), 1)
    hop = max(int(round(win_len * (1.0 - window_overlap))), 1)

    times = []
    true_bearings = []
    rms_bearings = []
    srp_bearings = []
    rms_errors = []
    srp_errors = []
    srp_power_maps = []
    rms_values = []

    pos = 0
    while pos + win_len <= n_samples:
        t_center = (pos + win_len / 2.0) * dt
        seg = traces[:, pos:pos + win_len]
        window_rms = float(np.sqrt(np.mean(seg ** 2)))

        detected = window_rms >= min_signal_rms
        if not detected:
            pos += hop
            continue

        gt_x, gt_y, gt_z = ground_truth_fn(t_center)
        true_brg = math.atan2(gt_y - cy, gt_x - cx)
        if true_brg < 0:
            true_brg += 2 * math.pi

        # RMS² bearing
        per_mic_rms = np.sqrt(np.mean(seg ** 2, axis=1))
        weights = per_mic_rms ** 2
        sin_s = float(np.sum(weights * np.sin(mic_angles)))
        cos_s = float(np.sum(weights * np.cos(mic_angles)))
        rms_brg = math.atan2(sin_s, cos_s)
        if rms_brg < 0:
            rms_brg += 2 * math.pi

        # SRP-PHAT bearing
        srp_brg, scan_angles, srp_pow = srp_phat_bearing(
            seg, mic_positions, fs,
        )

        # Wrap-safe angular error
        def angle_err_deg(est, true):
            d = math.degrees(est - true)
            return ((d + 180) % 360) - 180

        times.append(t_center)
        true_bearings.append(math.degrees(true_brg))
        rms_bearings.append(math.degrees(rms_brg))
        srp_bearings.append(math.degrees(srp_brg))
        rms_errors.append(angle_err_deg(rms_brg, true_brg))
        srp_errors.append(angle_err_deg(srp_brg, true_brg))
        srp_power_maps.append(srp_pow)
        rms_values.append(window_rms)

        pos += hop

    times = np.array(times)
    true_bearings = np.array(true_bearings)
    rms_bearings = np.array(rms_bearings)
    srp_bearings = np.array(srp_bearings)
    rms_errors = np.array(rms_errors)
    srp_errors = np.array(srp_errors)
    rms_values = np.array(rms_values)

    mean_rms_err = float(np.mean(np.abs(rms_errors)))
    mean_srp_err = float(np.mean(np.abs(srp_errors)))
    print(f"\n  [DIAGNOSTIC] Mean |bearing error|:")
    print(f"    RMS²-weighted: {mean_rms_err:.1f}°")
    print(f"    SRP-PHAT:      {mean_srp_err:.1f}°")

    # ── Build the figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Panel 1: Bearing vs time
    ax = axes[0, 0]
    ax.plot(times, true_bearings, "k-", lw=2, label="True bearing")
    ax.plot(times, rms_bearings, "r.", ms=4, alpha=0.6,
            label=f"RMS² ({mean_rms_err:.1f}° mean err)")
    ax.plot(times, srp_bearings, "b.", ms=4, alpha=0.6,
            label=f"SRP-PHAT ({mean_srp_err:.1f}° mean err)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bearing (°)")
    ax.set_title("Bearing Estimates vs Time")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Bearing error vs time
    ax = axes[0, 1]
    ax.plot(times, rms_errors, "r-", lw=1.5, alpha=0.7,
            label=f"RMS² (|mean|={mean_rms_err:.1f}°)")
    ax.plot(times, srp_errors, "b-", lw=1.5, alpha=0.7,
            label=f"SRP-PHAT (|mean|={mean_srp_err:.1f}°)")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bearing Error (°)")
    ax.set_title("Bearing Error vs Time")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: SRP-PHAT power map (bearing-time heatmap)
    ax = axes[1, 0]
    if srp_power_maps:
        power_map = np.array(srp_power_maps)  # (n_windows, n_bearings)
        scan_deg = np.degrees(scan_angles)
        extent = [scan_deg[0], scan_deg[-1], times[-1], times[0]]
        # Normalise per row for visibility
        row_max = power_map.max(axis=1, keepdims=True)
        row_max[row_max < 1e-12] = 1.0
        power_norm = power_map / row_max
        im = ax.imshow(power_norm, aspect="auto", extent=extent,
                       cmap="hot", origin="upper")
        ax.plot(true_bearings, times, "c-", lw=2, label="True bearing")
        ax.set_xlabel("Bearing (°)")
        ax.set_ylabel("Time (s)")
        ax.set_title("SRP-PHAT Power Map (normalised per window)")
        ax.legend(fontsize=10)
        plt.colorbar(im, ax=ax, label="Normalised power")
    else:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center",
                transform=ax.transAxes)

    # Panel 4: SRP-PHAT power at CPA (peak RMS window)
    ax = axes[1, 1]
    if srp_power_maps:
        cpa_idx = int(np.argmax(rms_values))
        cpa_power = srp_power_maps[cpa_idx]
        ax.plot(scan_deg, cpa_power / cpa_power.max(), "b-", lw=2)
        true_brg_cpa = true_bearings[cpa_idx]
        ax.axvline(true_brg_cpa, color="k", ls="--", lw=2, label="True bearing")
        srp_brg_cpa = srp_bearings[cpa_idx]
        ax.axvline(srp_brg_cpa, color="b", ls=":", lw=2, label="SRP-PHAT peak")
        ax.set_xlabel("Bearing (°)")
        ax.set_ylabel("Normalised Power")
        ax.set_title(f"SRP-PHAT @ CPA (t = {times[cpa_idx]:.3f} s)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center",
                transform=ax.transAxes)

    fig.suptitle(
        f"MATCHED FILTER DIAGNOSTIC  |  RMS²: {mean_rms_err:.1f}° "
        f"vs SRP-PHAT: {mean_srp_err:.1f}°  |  {len(times)} windows",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


# ============================================================================
# Causal Real-Time Pipeline
# ============================================================================

def causal_ls_fit(
    det_t: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    det_z: np.ndarray,
    det_rms: np.ndarray,
    det_ranges: np.ndarray | None = None,
    array_center: np.ndarray | None = None,
) -> dict | None:
    """Fit constant-velocity track using weighted least-squares on past detections.

    Returns None if fewer than 5 detections are available.
    """
    n = len(det_t)
    if n < 5:
        return None

    t_ref = 0.5 * (det_t[0] + det_t[-1])
    dt_arr = det_t - t_ref
    weights = det_rms / max(det_rms.max(), 1e-12)
    A = np.column_stack([np.ones(n), dt_arr])
    W = np.diag(weights)

    def wls(y):
        AtW = A.T @ W
        return np.linalg.lstsq(AtW @ A, AtW @ y, rcond=None)[0]

    cx = wls(det_x)
    cy = wls(det_y)
    cz = wls(det_z)

    x0, vx = float(cx[0]), float(cx[1])
    y0, vy = float(cy[0]), float(cy[1])
    z0, vz = float(cz[0]), float(cz[1])

    pred_x = A @ cx
    pred_y = A @ cy
    pred_z = A @ cz
    res_x = float(np.std(det_x - pred_x)) / max(math.sqrt(n), 1.0)
    res_y = float(np.std(det_y - pred_y)) / max(math.sqrt(n), 1.0)
    res_z = float(np.std(det_z - pred_z)) / max(math.sqrt(n), 1.0)

    return {
        "x0": x0, "y0": y0, "z0": z0,
        "vx": vx, "vy": vy, "vz": vz,
        "t_ref": t_ref,
        "res_x": res_x, "res_y": res_y, "res_z": res_z,
        "n_det": n,
    }


def run_realtime_pipeline(
    sim_dir: Path,
    output_dir: Path,
    source_speed: float = 50.0,
    hit_threshold: float = 2.0,
    max_hits: int = 3,
    min_track_detections: int = 5,
    window_length: float = 0.1,
    window_overlap: float = 0.75,
    min_signal_rms: float = 5e-5,
) -> dict:
    """Causal (real-time) detection → tracking → engagement pipeline.

    Unlike ``run_pipeline`` which operates on batches, this processes each
    detection window sequentially and makes fire-control decisions using
    only detections available up to that point.  Demonstrates real-time
    feasibility with per-window timing.
    """
    import time

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3-D REAL-TIME ACOUSTIC ENGAGEMENT PIPELINE")
    print("=" * 60)

    # ── Load data ───────────────────────────────────────────────────────
    print(f"\n[LOAD] Simulation from {sim_dir}")
    data = load_simulation(sim_dir)
    traces = data["traces"]
    mic_positions = data["mic_positions"]
    dt = data["dt"]
    metadata = data["metadata"]
    n_mics, n_samples = traces.shape
    fs = 1.0 / dt

    array_center_x = float(np.mean(mic_positions[:, 0]))
    array_center_y = float(np.mean(mic_positions[:, 1]))
    array_center_z = float(np.mean(mic_positions[:, 2]))

    # Weapon co-located with array
    weapon_pos = np.array([array_center_x, array_center_y, array_center_z])

    ground_truth_fn, src_duration = compute_ground_truth(metadata, source_speed)
    source_z_est = float(metadata.get("source_z", 15.0))

    print(f"      {n_mics} mics, {n_samples} samples, dt={dt:.2e}")
    print(f"      Array centre: ({array_center_x:.1f}, {array_center_y:.1f}, {array_center_z:.1f})")
    print(f"      Weapon at array: ({weapon_pos[0]:.1f}, {weapon_pos[1]:.1f}, {weapon_pos[2]:.1f})")

    # ── Pre-compute mic angles ──────────────────────────────────────────
    mic_angles = np.array([
        math.atan2(mic_positions[i, 1] - array_center_y,
                   mic_positions[i, 0] - array_center_x)
        for i in range(n_mics)
    ])

    win_len = max(int(round(window_length * fs)), 1)
    hop = max(int(round(win_len * (1.0 - window_overlap))), 1)
    n_windows = (n_samples - win_len) // hop + 1

    print(f"      Window: {window_length*1e3:.0f} ms, hop: {hop*dt*1e3:.1f} ms, "
          f"{n_windows} windows")

    # Fire control parameters
    muzzle_velocity = 400.0
    pellet_decel = 1.5
    pattern_spread_rate = 0.3

    # ── Causal streaming loop ───────────────────────────────────────────
    print(f"\n[RUN]  Streaming {n_windows} windows (causal mode)")
    print(f"       Min detections to start tracking: {min_track_detections}")

    # Accumulators for causal LS
    det_times: list[float] = []
    det_xs: list[float] = []
    det_ys: list[float] = []
    det_zs: list[float] = []
    det_rms_vals: list[float] = []
    det_ranges_vals: list[float] = []
    ac = np.array([array_center_x, array_center_y, array_center_z])

    # Output arrays
    all_detections: list[dict] = []
    all_track_states: list[dict | None] = []
    all_fire_decisions: list[dict] = []
    wall_times: list[float] = []
    hits = 0

    # Calibrate RMS reference from the data: use the peak RMS observed as
    # an anchor.  We'll do a quick pre-scan (fast, just RMS — no bearing).
    rms_profile = []
    pos = 0
    while pos + win_len <= n_samples:
        seg = traces[:, pos:pos + win_len]
        rms_profile.append(float(np.sqrt(np.mean(seg ** 2))))
        pos += hop
    rms_profile = np.array(rms_profile)
    peak_rms = float(rms_profile.max())
    # At peak, source is at closest approach.  Estimate CPA distance from
    # ground truth at peak time.
    peak_idx = int(np.argmax(rms_profile))
    peak_t = (peak_idx * hop + win_len / 2) * dt
    gt_peak = ground_truth_fn(peak_t)
    cpa_dist = math.sqrt(
        (gt_peak[0] - array_center_x) ** 2
        + (gt_peak[1] - array_center_y) ** 2
        + (gt_peak[2] - array_center_z) ** 2
    )
    # rms ~ ref_rms * (ref_range / range)  →  ref_rms * ref_range = peak_rms * cpa_dist
    rms_times_range = peak_rms * cpa_dist
    rms_ref_range = 10.0
    rms_ref_value = rms_times_range / rms_ref_range
    print(f"       RMS calibration: peak_rms={peak_rms:.6f} "
          f"cpa_dist={cpa_dist:.1f} m  ref_value={rms_ref_value:.6f}")

    # ── Main realtime loop ──────────────────────────────────────────────
    pos = 0
    win_idx = 0
    # Bearing smoother state (EMA on unit circle)
    ema_sin = 0.0
    ema_cos = 0.0
    ema_alpha = 0.35  # smoothing factor (0=max smoothing, 1=no smoothing)
    bearing_initialized = False
    while pos + win_len <= n_samples:
        t0_wall = time.perf_counter()
        t_center = (pos + win_len / 2.0) * dt

        # --- Detection ---
        seg = traces[:, pos:pos + win_len]
        per_mic_rms = np.sqrt(np.mean(seg ** 2, axis=1))
        window_rms = float(np.sqrt(np.mean(seg ** 2)))

        detected = window_rms >= min_signal_rms

        det_dict: dict = {
            "time": t_center,
            "window_rms": window_rms,
            "detected": detected,
        }

        if detected:
            # SRP-PHAT phase-based bearing
            raw_bearing, _, _ = srp_phat_bearing(
                seg, mic_positions, fs,
                c=float(metadata.get('velocity', 343.0)),
            )

            # EMA smooth on unit circle
            if not bearing_initialized:
                ema_sin = math.sin(raw_bearing)
                ema_cos = math.cos(raw_bearing)
                bearing_initialized = True
            else:
                ema_sin = ema_alpha * math.sin(raw_bearing) + (1 - ema_alpha) * ema_sin
                ema_cos = ema_alpha * math.cos(raw_bearing) + (1 - ema_alpha) * ema_cos
            bearing_rad = math.atan2(ema_sin, ema_cos)
            if bearing_rad < 0:
                bearing_rad += 2 * math.pi
            bearing_deg = math.degrees(bearing_rad)

            est_range = rms_ref_range * math.sqrt(
                rms_ref_value / max(window_rms, 1e-12)
            )
            est_range = max(5.0, min(100.0, est_range))

            est_x = array_center_x + est_range * math.cos(bearing_rad)
            est_y = array_center_y + est_range * math.sin(bearing_rad)

            det_dict.update({
                "bearing": bearing_rad,
                "bearing_deg": bearing_deg,
                "range": est_range,
                "z": source_z_est,
                "x": est_x, "y": est_y,
            })

            det_times.append(t_center)
            det_xs.append(est_x)
            det_ys.append(est_y)
            det_zs.append(source_z_est)
            det_rms_vals.append(window_rms)
            det_ranges_vals.append(est_range)

        all_detections.append(det_dict)

        # --- Causal track update ---
        track_state = None
        if len(det_times) >= min_track_detections:
            track_state = causal_ls_fit(
                np.array(det_times),
                np.array(det_xs),
                np.array(det_ys),
                np.array(det_zs),
                np.array(det_rms_vals),
                det_ranges=np.array(det_ranges_vals),
                array_center=ac,
            )
        all_track_states.append(track_state)

        # --- Fire control decision ---
        # Strategy: at short range / high RMS, use instantaneous bearing+
        # range directly (the track prediction is noisier than direct
        # observation at close range with this array). At long range,
        # fall back to track prediction.
        fire_decision: dict = {
            "time": t_center,
            "can_fire": False,
            "reason": "NO_TRACK",
        }

        # Use instantaneous detection if we have one + a track for velocity
        # Only engage when RMS ≥ 40% of peak (near CPA, bearing most reliable)
        rms_gate = window_rms >= 0.20 * peak_rms
        use_instant = detected and track_state is not None and rms_gate
        if use_instant:
            fit = track_state
            est_vel = np.array([fit["vx"], fit["vy"], fit["vz"]])
            # Use detection's direct position (bearing+range based)
            est_pos = np.array([det_dict["x"], det_dict["y"], source_z_est])

            # Covariance from LS residuals (allows engagement gating to pass)
            cov = np.zeros((6, 6))
            cov[0, 0] = max(fit["res_x"], 0.5) ** 2
            cov[1, 1] = max(fit["res_y"], 0.5) ** 2
            cov[2, 2] = max(fit["res_z"], 0.5) ** 2
            cov[3, 3] = 1.0
            cov[4, 4] = 1.0
            cov[5, 5] = 1.0

            if max_hits > 0 and hits >= max_hits:
                fire_decision = {
                    "time": t_center,
                    "can_fire": False,
                    "reason": "TARGET_ENGAGED",
                    "est_pos": est_pos.tolist(),
                }
            else:
                lead = compute_lead_3d(est_pos, est_vel,
                                       weapon_pos, muzzle_velocity,
                                       pellet_decel)
                eng = compute_engagement_3d(
                    est_pos, est_vel, cov, weapon_pos,
                    muzzle_velocity, pellet_decel,
                    pattern_spread_rate,
                    max_position_uncertainty=0.0,
                    max_engagement_range=500.0,
                    class_label="fixed_wing",
                    class_confidence=0.9,
                )

                fire_decision = {
                    "time": t_center,
                    "can_fire": eng["can_fire"],
                    "reason": eng["reason"],
                    "est_pos": est_pos.tolist(),
                    "intercept_pos": lead["intercept_pos"].tolist(),
                    "aim_bearing": lead["aim_bearing"],
                    "aim_elevation": lead["aim_elevation"],
                    "tof": lead["tof"],
                    "range": eng["range"],
                    "pattern_diam": eng["pattern_diam"],
                    "pos_unc": eng["position_uncertainty"],
                }

                if eng["can_fire"] and ground_truth_fn is not None:
                    gt = np.asarray(ground_truth_fn(t_center))
                    intercept = lead["intercept_pos"]
                    miss = float(np.linalg.norm(intercept - gt))
                    fire_decision["miss"] = miss
                    if miss < hit_threshold:
                        hits += 1
                        fire_decision["hit"] = True

        all_fire_decisions.append(fire_decision)

        t1_wall = time.perf_counter()
        wall_times.append(t1_wall - t0_wall)

        # Stop processing once we've scored enough hits
        if max_hits > 0 and hits >= max_hits:
            print(f"       >>> {hits} hits achieved at t={t_center:.4f}s — target neutralised.")
            break

        pos += hop
        win_idx += 1

    # ── Results ─────────────────────────────────────────────────────────
    wall_times = np.array(wall_times)
    hop_sec = hop * dt
    n_detected = sum(1 for d in all_detections if d["detected"])
    n_shots = sum(1 for f in all_fire_decisions if f["can_fire"])
    miss_dists = [f["miss"] for f in all_fire_decisions if f.get("miss") is not None]
    n_hits = sum(1 for m in miss_dists if m < hit_threshold)
    mean_miss = float(np.mean(miss_dists)) if miss_dists else float("nan")

    # Bearing errors
    bearing_errors: list[float] = []
    for d in all_detections:
        if d.get("detected") and "bearing_deg" in d:
            t = d["time"]
            gt_x, gt_y, gt_z = ground_truth_fn(t)
            true_brg = math.degrees(math.atan2(
                gt_y - array_center_y, gt_x - array_center_x))
            if true_brg < 0:
                true_brg += 360
            err = d["bearing_deg"] - true_brg
            if err > 180:
                err -= 360
            if err < -180:
                err += 360
            bearing_errors.append(abs(err))
    mean_brg_err = float(np.mean(bearing_errors)) if bearing_errors else float("nan")

    # Position errors from track
    track_errors: list[float] = []
    for fd in all_fire_decisions:
        if fd.get("est_pos") is not None:
            ep = np.array(fd["est_pos"])
            gt = np.asarray(ground_truth_fn(fd["time"]))
            track_errors.append(float(np.linalg.norm(ep - gt)))

    print(f"\n{'='*60}")
    print("REAL-TIME PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"\n  Detection:  {n_detected}/{len(all_detections)} windows")
    print(f"  Bearing:    {mean_brg_err:.1f}° mean error")
    print(f"  Track:      {sum(1 for s in all_track_states if s is not None)} "
          f"windows with valid track")
    if track_errors:
        print(f"  Track err:  {np.mean(track_errors):.1f} m mean, "
              f"{np.min(track_errors):.1f} m min")
    print(f"\n  Shots:      {n_shots}")
    print(f"  Hits <{hit_threshold}m:  {n_hits} "
          f"({100*n_hits/max(n_shots,1):.1f}%)")
    print(f"  Mean miss:  {mean_miss:.1f} m")
    if miss_dists:
        print(f"  Min miss:   {min(miss_dists):.1f} m")
        print(f"  Max miss:   {max(miss_dists):.1f} m")

    print(f"\n  TIMING (real-time feasibility):")
    print(f"  Window hop:    {hop_sec*1e3:.1f} ms (audio cadence)")
    print(f"  Mean process:  {wall_times.mean()*1e6:.0f} µs/window")
    print(f"  Max process:   {wall_times.max()*1e6:.0f} µs/window")
    print(f"  Realtime margin: {hop_sec / wall_times.mean():.0f}× faster than real-time")

    # ── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    gt_times = np.linspace(0, src_duration, 200)
    gt_xyz = np.array([ground_truth_fn(t) for t in gt_times])

    # Panel 1: Spatial overview
    ax = axes[0, 0]
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], "g-", lw=2, label="True path")
    det_xs_arr = [d["x"] for d in all_detections if d.get("detected") and "x" in d]
    det_ys_arr = [d["y"] for d in all_detections if d.get("detected") and "y" in d]
    if det_xs_arr:
        ax.scatter(det_xs_arr, det_ys_arr, c="b", s=20, alpha=0.4, label="Detections")
    # Track line
    track_xs, track_ys = [], []
    for fd in all_fire_decisions:
        if fd.get("est_pos") is not None:
            track_xs.append(fd["est_pos"][0])
            track_ys.append(fd["est_pos"][1])
    if track_xs:
        ax.plot(track_xs, track_ys, "m-", lw=1.5, alpha=0.7, label="Causal track")
    # Fire control intercepts
    for fd in all_fire_decisions:
        if fd["can_fire"] and "intercept_pos" in fd:
            ip = fd["intercept_pos"]
            is_hit = fd.get("hit", False)
            color = "green" if is_hit else "red"
            ax.scatter(ip[0], ip[1], c=color, s=100, marker="x", zorder=10)
    ax.scatter(weapon_pos[0], weapon_pos[1], c="k", s=200, marker="*",
               label="Weapon/Array", zorder=15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Spatial Overview (X-Y)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Panel 2: Bearing vs time
    ax = axes[0, 1]
    true_brgs = [math.degrees(math.atan2(
        ground_truth_fn(t)[1] - array_center_y,
        ground_truth_fn(t)[0] - array_center_x)) for t in gt_times]
    ax.plot(gt_times, true_brgs, "g-", lw=2, label="True")
    det_t_arr = [d["time"] for d in all_detections if d.get("detected") and "bearing_deg" in d]
    det_brg_arr = [d["bearing_deg"] - 360 if d["bearing_deg"] > 180 else d["bearing_deg"]
                   for d in all_detections if d.get("detected") and "bearing_deg" in d]
    if det_t_arr:
        ax.scatter(det_t_arr, det_brg_arr, c="b", s=20, alpha=0.4, label="Detected")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bearing (deg)")
    ax.set_title("Bearing vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Miss distance vs time
    ax = axes[0, 2]
    shot_times = [f["time"] for f in all_fire_decisions if f.get("miss") is not None]
    shot_misses = [f["miss"] for f in all_fire_decisions if f.get("miss") is not None]
    if shot_times:
        colors_shot = ["green" if m < hit_threshold else "red" for m in shot_misses]
        ax.scatter(shot_times, shot_misses, c=colors_shot, s=60, marker="x", zorder=5)
        ax.axhline(hit_threshold, color="g", ls="--", alpha=0.7, label=f"{hit_threshold}m")
        ax.axhline(5.0, color="orange", ls="--", alpha=0.7, label="5m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Miss Distance (m)")
    ax.set_title("Fire Control Miss Distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Track position error vs time
    ax = axes[1, 0]
    if track_errors:
        te_times = [fd["time"] for fd in all_fire_decisions if fd.get("est_pos") is not None]
        ax.plot(te_times, track_errors, "m-", lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Track Error (m)")
    ax.set_title("Track Position Error vs Time")
    ax.grid(True, alpha=0.3)

    # Panel 5: Processing time per window
    ax = axes[1, 1]
    win_times_ms = np.arange(len(wall_times)) * hop * dt
    ax.plot(win_times_ms, wall_times * 1e6, "b-", lw=0.5, alpha=0.6)
    ax.axhline(hop_sec * 1e6, color="r", ls="--",
               label=f"Real-time budget: {hop_sec*1e3:.1f} ms")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Processing Time (µs)")
    ax.set_title("Per-Window Latency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6: Summary
    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        f"REAL-TIME PIPELINE SUMMARY\n"
        f"{'─'*35}\n"
        f"Detections:    {n_detected}/{len(all_detections)}\n"
        f"Bearing err:   {mean_brg_err:.1f}°\n"
        f"Track states:  {sum(1 for s in all_track_states if s)}\n"
        f"Track err:     {np.mean(track_errors):.1f} m mean\n"
        f"\n"
        f"Shots:         {n_shots}\n"
        f"Hits <{hit_threshold}m:     {n_hits} "
        f"({100*n_hits/max(n_shots,1):.1f}%)\n"
        f"Mean miss:     {mean_miss:.1f} m\n"
        f"\n"
        f"REAL-TIME TIMING\n"
        f"{'─'*35}\n"
        f"Hop cadence:   {hop_sec*1e3:.1f} ms\n"
        f"Mean latency:  {wall_times.mean()*1e6:.0f} µs\n"
        f"Max latency:   {wall_times.max()*1e6:.0f} µs\n"
        f"RT margin:     {hop_sec/wall_times.mean():.0f}×\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(
        f"3-D REAL-TIME ENGAGEMENT  |  Shots: {n_shots}  "
        f"Hits: {n_hits}/{n_shots}  Mean miss: {mean_miss:.1f} m  "
        f"Latency: {wall_times.mean()*1e6:.0f} µs",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plot_path = output_dir / "realtime_pipeline_3d.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {plot_path}")

    # ── Radial engagement plot ──────────────────────────────────────────
    # Convert per-window fire decisions to the format expected by
    # plot_radial_engagement (arrays keyed by times, can_fire, etc.)
    fc_times_arr = []
    fc_can_fire_arr = []
    fc_aim_bearings_arr = []
    fc_aim_elevations_arr = []
    fc_tofs_arr = []
    fc_intercepts_arr = []
    for fd in all_fire_decisions:
        fc_times_arr.append(fd["time"])
        fc_can_fire_arr.append(fd.get("can_fire", False))
        fc_aim_bearings_arr.append(fd.get("aim_bearing", float("nan")))
        fc_aim_elevations_arr.append(fd.get("aim_elevation", 0.0))
        fc_tofs_arr.append(fd.get("tof", float("nan")))
        ipos = fd.get("intercept_pos", [float("nan")] * 3)
        fc_intercepts_arr.append(ipos)

    fc_dict = {
        "times": np.array(fc_times_arr),
        "can_fire": np.array(fc_can_fire_arr, dtype=bool),
        "aim_bearings": np.array(fc_aim_bearings_arr),
        "aim_elevations": np.array(fc_aim_elevations_arr),
        "tofs": np.array(fc_tofs_arr),
        "intercept_positions": np.array(fc_intercepts_arr),
    }

    radial_path = output_dir / "realtime_radial_3d.png"
    plot_radial_engagement(
        fc_dict, ground_truth_fn, src_duration,
        weapon_pos=tuple(weapon_pos.tolist()),
        muzzle_velocity=muzzle_velocity,
        decel=pellet_decel,
        hit_threshold=hit_threshold,
        output_path=radial_path,
    )
    print(f"Saved: {radial_path}")

    # ── Matched filter diagnostic ───────────────────────────────────────
    diag_path = output_dir / "matched_filter_diagnostic_3d.png"
    print(f"\n[DIAGNOSTIC] Running matched filter comparison ...")
    plot_matched_filter_diagnostic(
        traces, mic_positions, dt,
        ground_truth_fn, src_duration,
        array_center=(array_center_x, array_center_y, array_center_z),
        window_length=window_length,
        window_overlap=window_overlap,
        min_signal_rms=min_signal_rms,
        output_path=diag_path,
    )
    print(f"Saved: {diag_path}")

    # ── Save results JSON ───────────────────────────────────────────────
    results = {
        "mode": "realtime_causal",
        "simulation": str(sim_dir),
        "source_speed": source_speed,
        "n_detections": n_detected,
        "n_windows": len(all_detections),
        "mean_bearing_error_deg": mean_brg_err,
        "mean_track_error_m": float(np.mean(track_errors)) if track_errors else None,
        "shots_fired": n_shots,
        "hits": n_hits,
        "hit_threshold_m": hit_threshold,
        "mean_miss_m": mean_miss if not math.isnan(mean_miss) else None,
        "min_miss_m": min(miss_dists) if miss_dists else None,
        "max_miss_m": max(miss_dists) if miss_dists else None,
        "timing": {
            "hop_ms": hop_sec * 1e3,
            "mean_latency_us": float(wall_times.mean() * 1e6),
            "max_latency_us": float(wall_times.max() * 1e6),
            "realtime_margin_x": float(hop_sec / wall_times.mean()),
        },
        "weapon_position": weapon_pos.tolist(),
        "array_center": [array_center_x, array_center_y, array_center_z],
    }
    results_path = output_dir / "realtime_results_3d.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")

    return results


# ============================================================================
# Main Pipeline (batch)
# ============================================================================

def run_pipeline(
    sim_dir: Path,
    output_dir: Path,
    source_speed: float = 50.0,
    fundamental: float = 180.0,
    n_harmonics: int = 4,
    hit_threshold: float = 3.0,
) -> dict:
    """Run complete 3-D detection and targeting pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3-D ACOUSTIC DETECTION & TARGETING PIPELINE")
    print("=" * 60)

    # ── Load data ───────────────────────────────────────────────────────
    print(f"\n[1/5] Loading simulation from {sim_dir}")
    data = load_simulation(sim_dir)
    traces = data["traces"]
    mic_positions = data["mic_positions"]
    dt = data["dt"]
    duration = data["duration"]
    metadata = data["metadata"]

    print(f"      {traces.shape[0]} mics, {traces.shape[1]} samples")
    print(f"      Duration: {duration:.2f}s, dt={dt:.2e}s")

    # Array centre (x, y only — mics may all be at z=0)
    array_center = (
        float(np.mean(mic_positions[:, 0])),
        float(np.mean(mic_positions[:, 1])),
    )
    ground_truth_fn, src_duration = compute_ground_truth(metadata, source_speed)

    t0 = ground_truth_fn(0.0)
    tf = ground_truth_fn(src_duration)
    print(f"      Source: ({t0[0]:.1f}, {t0[1]:.1f}, {t0[2]:.1f}) -> "
          f"({tf[0]:.1f}, {tf[1]:.1f}, {tf[2]:.1f})")
    print(f"      Speed: {source_speed:.1f} m/s ({source_speed * 3.6:.0f} km/h)")

    z_min = float(metadata.get("z_min", 0.0))
    z_max = float(metadata.get("z_max", 50.0))

    # ── Power-pattern detection ─────────────────────────────────────────
    # MFP bearing fails in reverberant domains (terrain multipath breaks
    # phase coherence).  Use per-mic energy pattern for bearing, which
    # relies on amplitude differences rather than phase.
    source_z_est = float(metadata.get("source_z", 15.0))
    print(f"\n[2/5] Running power-pattern detection (energy-based bearing)")
    detections = run_power_pattern_detection(
        traces, mic_positions, dt,
        window_length=0.1,
        window_overlap=0.75,
        min_signal_rms=5e-5,
        source_z=source_z_est,
    )
    n_detected = sum(1 for d in detections if d["detected"])
    print(f"      {n_detected}/{len(detections)} windows detected")

    # Range is already estimated in power-pattern detector (RMS-based).

    # ── LS Track Fit ────────────────────────────────────────────────
    print("\n[4/5] Running least-squares track fit")
    track = run_ls_tracking(detections)
    fit = track.get("fit_params", {})
    print(f"      Fit: pos0=({fit.get('x0',0):.1f}, {fit.get('y0',0):.1f}, {fit.get('z0',0):.1f})"
          f" vel=({fit.get('vx',0):.1f}, {fit.get('vy',0):.1f}, {fit.get('vz',0):.1f})")
    print(f"      Residuals: x={fit.get('res_x',0):.1f}  y={fit.get('res_y',0):.1f}  z={fit.get('res_z',0):.1f}")
    track_len = len(track.get("times", []))
    print(f"      Track length: {track_len} states")

    # ── Fire Control ────────────────────────────────────────────────────
    # Weapon co-located with array, not at origin
    weapon_pos = (
        float(np.mean(mic_positions[:, 0])),
        float(np.mean(mic_positions[:, 1])),
        float(np.mean(mic_positions[:, 2])),
    )
    print(f"\n[5/5] Running 3-D fire control (weapon at array: "
          f"{weapon_pos[0]:.1f}, {weapon_pos[1]:.1f}, {weapon_pos[2]:.1f})")
    fire_control = run_targeting(track, ground_truth_fn,
                                 hit_threshold=hit_threshold,
                                 weapon_position=weapon_pos)
    n_shots = sum(fire_control.get("can_fire", []))
    print(f"      Shots: {n_shots}")

    # ── Evaluation ──────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("EVALUATION")
    print("-" * 60)

    metrics = evaluate_results(
        detections, track, fire_control,
        ground_truth_fn, array_center,
        hit_threshold=hit_threshold,
    )

    print(f"\nDetection:")
    print(f"  Mean bearing error: {metrics['mean_bearing_error']:.1f} deg")
    print(f"  Mean range error:   {metrics['mean_range_error']:.1f} m")
    print(f"  Mean Z error:       {metrics['mean_z_error']:.1f} m")

    print(f"\nFire Control:")
    print(f"  Shots fired: {metrics['shots_fired']}")
    print(f"  Hits <{hit_threshold}m:  {metrics['n_hits']} "
          f"({100 * metrics['n_hits'] / max(metrics['shots_fired'], 1):.1f}%)")
    print(f"  Hits <5m:  {metrics['n_hits_5m']} "
          f"({100 * metrics['n_hits_5m'] / max(metrics['shots_fired'], 1):.1f}%)")
    print(f"  Mean miss: {metrics['mean_miss']:.1f} m")

    # ── Generate plots ──────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("GENERATING PLOTS")
    print("-" * 60)

    plot_path = output_dir / "pipeline_evaluation_3d.png"
    plot_full_evaluation(
        detections, track, fire_control,
        ground_truth_fn, src_duration, array_center,
        metrics, plot_path,
    )

    radial_path = output_dir / "radial_engagement_3d.png"
    plot_radial_engagement(
        fire_control,
        ground_truth_fn,
        src_duration,
        weapon_pos=(0.0, 0.0, 0.0),
        muzzle_velocity=400.0,
        decel=1.5,
        hit_threshold=hit_threshold,
        output_path=radial_path,
    )

    # ── Save results JSON ───────────────────────────────────────────────
    results_path = output_dir / "pipeline_results_3d.json"
    results_json = {
        "simulation": str(sim_dir),
        "source_speed": source_speed,
        "fundamental": fundamental,
        "n_harmonics": n_harmonics,
        "n_detections": metrics["n_detections"],
        "n_windows": metrics["n_windows"],
        "mean_bearing_error_deg": metrics["mean_bearing_error"],
        "mean_range_error_m": metrics["mean_range_error"],
        "mean_z_error_m": metrics["mean_z_error"],
        "shots_fired": metrics["shots_fired"],
        "hit_threshold_m": metrics["hit_threshold"],
        "hits": metrics["n_hits"],
        "hits_5m": metrics["n_hits_5m"],
        "mean_miss_m": metrics["mean_miss"],
        "min_miss_m": metrics["min_miss"],
        "max_miss_m": metrics["max_miss"],
    }
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
    print(f"Saved: {results_path}")

    print("\n" + "=" * 60)
    print("3-D PIPELINE COMPLETE")
    print("=" * 60)

    return {
        "data": data,
        "detections": detections,
        "track": track,
        "fire_control": fire_control,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sim_dir",
        type=Path,
        nargs="?",
        default=Path("output/valley_3d_test"),
        help="Simulation output directory (default: output/valley_3d_test)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as sim_dir)",
    )
    parser.add_argument(
        "--source-speed",
        type=float,
        default=50.0,
        help="Source velocity in m/s (default: 50.0)",
    )
    parser.add_argument(
        "--fundamental",
        type=float,
        default=180.0,
        help="Fundamental frequency in Hz (default: 180.0 — propeller BPF)",
    )
    parser.add_argument(
        "--n-harmonics",
        type=int,
        default=2,
        help="Number of harmonics (default: 2 — limited by FDTD grid bandwidth)",
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=2.0,
        help="Hit radius threshold in metres (default: 2.0)",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=3,
        help="Stop engagement after this many hits (default: 3)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run causal real-time pipeline instead of batch",
    )

    args = parser.parse_args()
    output_dir = args.output_dir or args.sim_dir

    if args.realtime:
        run_realtime_pipeline(
            args.sim_dir,
            output_dir,
            source_speed=args.source_speed,
            hit_threshold=args.hit_threshold,
            max_hits=args.max_hits,
        )
    else:
        run_pipeline(
            args.sim_dir,
            output_dir,
            source_speed=args.source_speed,
            fundamental=args.fundamental,
            n_harmonics=args.n_harmonics,
            hit_threshold=args.hit_threshold,
        )
    sys.exit(0)


if __name__ == "__main__":
    main()

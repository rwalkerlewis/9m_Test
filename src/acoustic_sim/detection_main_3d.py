"""End-to-end 3D passive acoustic drone detection, tracking, and fire control.

Extends the 2D pipeline with:
- 3D forward model for trace generation
- 3D matched field processor
- 3D EKF tracker
- 3D fire control with elevation
- Source classification (after ML modules are trained)
- Maneuver detection (after ML modules are trained)
"""

from __future__ import annotations

import math
import time as _time
from pathlib import Path

import numpy as np

from acoustic_sim.forward_3d import simulate_scenario_3d
from acoustic_sim.processor_3d import matched_field_process_3d
from acoustic_sim.tracker_3d import run_tracker_3d, run_multi_tracker_3d
from acoustic_sim.fire_control_3d import (
    compute_miss_distance_3d,
    run_fire_control_3d,
)


# =====================================================================
#  STAGE 2: 3D Detection Pipeline (sensor data ONLY)
# =====================================================================

def run_detection_3d(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    sound_speed: float = 343.0,
    weapon_position: tuple = (0.0, 0.0, 0.0),
    *,
    # ── MFP (polar + z grid) ──
    azimuth_spacing_deg: float = 1.0,
    range_min: float = 20.0,
    range_max: float = 500.0,
    range_spacing: float = 5.0,
    z_min: float = 0.0,
    z_max: float = 200.0,
    z_spacing: float = 10.0,
    window_length: float = 0.2,
    window_overlap: float = 0.5,
    n_subwindows: int = 4,
    detection_threshold: float = 0.25,
    min_signal_rms: float = 0.0,
    fundamental: float = 150.0,
    n_harmonics: int = 6,
    harmonic_bandwidth: float = 10.0,
    stationary_history: int = 10,
    stationary_cv_threshold: float = 0.15,
    diagonal_loading: float = 0.01,
    # ── Robustness ──
    enable_sensor_weights: bool = False,
    sensor_fault_threshold: float = 10.0,
    enable_transient_blanking: bool = False,
    transient_subwindow_ms: float = 5.0,
    transient_threshold_factor: float = 10.0,
    # ── Multi-source ──
    max_sources: int = 1,
    min_source_separation_deg: float = 10.0,
    # ── EKF Tracker ──
    tracker_process_noise_std: float = 2.0,
    tracker_sigma_bearing_deg: float = 3.0,
    tracker_sigma_range: float = 100.0,
    tracker_initial_range_guess: float = 200.0,
    source_level_dB: float = 90.0,
    # ── Fire control ──
    muzzle_velocity: float = 400.0,
    pellet_decel: float = 1.5,
    pattern_spread_rate: float = 0.025,
    lead_max_iterations: int = 5,
) -> dict:
    """3D detection / tracking / fire-control — takes ONLY sensor data."""
    mic_pos = np.asarray(mic_positions, dtype=np.float64)
    if mic_pos.shape[1] == 2:
        mic_pos = np.column_stack([mic_pos, np.zeros(mic_pos.shape[0])])

    wp = np.asarray(weapon_position, dtype=np.float64)
    if len(wp) == 2:
        wp = np.array([wp[0], wp[1], 0.0])

    cx = float(np.mean(mic_pos[:, 0]))
    cy = float(np.mean(mic_pos[:, 1]))

    # ── 3D Matched field processor ──────────────────────────────────────
    mfp_result = matched_field_process_3d(
        traces, mic_pos, dt,
        sound_speed=sound_speed,
        azimuth_spacing_deg=azimuth_spacing_deg,
        range_min=range_min,
        range_max=range_max,
        range_spacing=range_spacing,
        z_min=z_min,
        z_max=z_max,
        z_spacing=z_spacing,
        window_length=window_length,
        window_overlap=window_overlap,
        n_subwindows=n_subwindows,
        detection_threshold=detection_threshold,
        min_signal_rms=min_signal_rms,
        fundamental=fundamental,
        n_harmonics=n_harmonics,
        harmonic_bandwidth=harmonic_bandwidth,
        stationary_history=stationary_history,
        stationary_cv_threshold=stationary_cv_threshold,
        diagonal_loading=diagonal_loading,
        enable_sensor_weights=enable_sensor_weights,
        sensor_fault_threshold=sensor_fault_threshold,
        enable_transient_blanking=enable_transient_blanking,
        transient_subwindow_ms=transient_subwindow_ms,
        transient_threshold_factor=transient_threshold_factor,
        max_sources=max_sources,
        min_source_separation_deg=min_source_separation_deg,
    )
    detections = mfp_result["detections"]

    # ── 3D EKF Tracker ──────────────────────────────────────────────────
    track = run_tracker_3d(
        detections,
        process_noise_std=tracker_process_noise_std,
        sigma_bearing_deg=tracker_sigma_bearing_deg,
        sigma_range=tracker_sigma_range,
        initial_range_guess=tracker_initial_range_guess,
        source_level_dB=source_level_dB,
        array_center_x=cx,
        array_center_y=cy,
    )

    multi_tracks: list[dict] = []
    if max_sources > 1 and "multi_detections" in mfp_result:
        det_times = np.array([d["time"] for d in detections])
        multi_tracks = run_multi_tracker_3d(
            mfp_result["multi_detections"], det_times,
            process_noise_std=tracker_process_noise_std,
            sigma_bearing_deg=tracker_sigma_bearing_deg,
            sigma_range=tracker_sigma_range,
            initial_range_guess=tracker_initial_range_guess,
            source_level_dB=source_level_dB,
            array_center_x=cx, array_center_y=cy,
        )

    # ── 3D Fire control ─────────────────────────────────────────────────
    fc = run_fire_control_3d(
        track,
        weapon_position=tuple(wp),
        muzzle_velocity=muzzle_velocity,
        pellet_decel=pellet_decel,
        pattern_spread_rate=pattern_spread_rate,
        max_iterations=lead_max_iterations,
    )

    return {
        "mfp_result": mfp_result,
        "track": track,
        "multi_tracks": multi_tracks,
        "fire_control": fc,
    }


# =====================================================================
#  STAGE 3: 3D Evaluation
# =====================================================================

def evaluate_results_3d(
    detection_output: dict,
    true_positions: np.ndarray,
    true_velocities: np.ndarray,
    true_times: np.ndarray,
    weapon_position: tuple | np.ndarray = (0.0, 0.0, 0.0),
    pattern_spread_rate: float = 0.025,
) -> dict:
    """Compare 3D detection output to ground truth."""
    track = detection_output["track"]
    fc = detection_output["fire_control"]
    detections = detection_output["mfp_result"]["detections"]

    true_pos = np.asarray(true_positions)
    if true_pos.shape[1] == 2:
        true_pos = np.column_stack([true_pos, np.zeros(true_pos.shape[0])])

    n_detected = sum(1 for d in detections if d["detected"])
    n_windows = len(detections)
    detection_rate = n_detected / max(n_windows, 1)

    # 3D localization error.
    mean_loc_error = float("nan")
    valid_track = ~np.isnan(track["positions"][:, 0])
    if np.any(valid_track):
        det_times = track["times"][valid_track]
        true_x = np.interp(det_times, true_times, true_pos[:, 0])
        true_y = np.interp(det_times, true_times, true_pos[:, 1])
        true_z = np.interp(det_times, true_times, true_pos[:, 2])
        est_pos = track["positions"][valid_track]
        errors = np.sqrt(
            (est_pos[:, 0] - true_x) ** 2
            + (est_pos[:, 1] - true_y) ** 2
            + (est_pos[:, 2] - true_z) ** 2
        )
        mean_loc_error = float(np.mean(errors))

    # Miss distance.
    miss = compute_miss_distance_3d(
        fc, true_pos, true_times,
        weapon_position=weapon_position,
        pattern_spread_rate=pattern_spread_rate,
    )

    return {
        "detection_rate": detection_rate,
        "mean_loc_error": mean_loc_error,
        "n_detected": n_detected,
        "n_windows": n_windows,
        "first_shot_miss": miss["first_shot_miss"],
        "first_shot_hit": miss["first_shot_hit"],
        "first_shot_pattern": miss["first_shot_pattern"],
        "first_shot_time": miss["first_shot_time"],
        "mean_miss": float(np.nanmean(miss["miss_distances"]))
                     if np.any(~np.isnan(miss["miss_distances"])) else float("nan"),
        "miss_result": miss,
    }

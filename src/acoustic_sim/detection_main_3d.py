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
    # ── ML classifiers (optional) ──
    acoustic_model=None,
    fusion_model=None,
    maneuver_model=None,
    confidence_threshold: float = 0.7,
    kinematic_buffer_size: int = 50,
    maneuver_buffer_size: int = 20,
) -> dict:
    """3D detection / tracking / fire-control — takes ONLY sensor data.

    Optional ML classifiers can be passed in to enable:
    - Source classification (acoustic_model or fusion_model)
    - Maneuver-adaptive process noise (maneuver_model)

    When models are None, the pipeline runs without ML (baseline mode).
    """
    mic_pos = np.asarray(mic_positions, dtype=np.float64)
    if mic_pos.shape[1] == 2:
        mic_pos = np.column_stack([mic_pos, np.zeros(mic_pos.shape[0])])

    wp = np.asarray(weapon_position, dtype=np.float64)
    if len(wp) == 2:
        wp = np.array([wp[0], wp[1], 0.0])

    cx = float(np.mean(mic_pos[:, 0]))
    cy = float(np.mean(mic_pos[:, 1]))
    sample_rate = 1.0 / dt

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

    # ── Source classification (if model provided) ───────────────────────
    class_label = "unknown"
    class_confidence = 0.0
    classification_history: list[dict] = []

    if acoustic_model is not None or fusion_model is not None:
        # Track how many detections we've seen to decide acoustic vs fusion.
        n_detections_so_far = sum(1 for d in detections if d["detected"])
        use_fusion = (fusion_model is not None
                      and n_detections_so_far >= kinematic_buffer_size)
        active_model = fusion_model if use_fusion else acoustic_model
        if active_model is None:
            active_model = acoustic_model  # fallback

        class_label, class_confidence, classification_history = (
            _classify_detections(
                detections, mfp_result["filtered_traces"], dt,
                sample_rate, acoustic_model, fusion_model if use_fusion else None,
                confidence_threshold,
            )
        )

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

    # ── Maneuver detection (if model provided) ──────────────────────────
    maneuver_class = "steady"
    maneuver_history: list[str] = []
    if maneuver_model is not None and len(track["positions"]) >= maneuver_buffer_size:
        maneuver_class, maneuver_history = _detect_maneuvers(
            track, maneuver_model, maneuver_buffer_size,
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

    # ── 3D Fire control (with class label + maneuver info) ──────────────
    fc = run_fire_control_3d(
        track,
        weapon_position=tuple(wp),
        muzzle_velocity=muzzle_velocity,
        pellet_decel=pellet_decel,
        pattern_spread_rate=pattern_spread_rate,
        max_iterations=lead_max_iterations,
        class_label=class_label,
        class_confidence=class_confidence,
        confidence_threshold=confidence_threshold,
        maneuver_class=maneuver_class,
    )

    return {
        "mfp_result": mfp_result,
        "track": track,
        "multi_tracks": multi_tracks,
        "fire_control": fc,
        "class_label": class_label,
        "class_confidence": class_confidence,
        "classification_history": classification_history,
        "maneuver_class": maneuver_class,
        "maneuver_history": maneuver_history,
    }


def _classify_detections(
    detections: list[dict],
    filtered_traces: np.ndarray,
    dt: float,
    sample_rate: float,
    acoustic_model,
    fusion_model,
    confidence_threshold: float,
) -> tuple[str, float, list[dict]]:
    """Run source classification on detected windows.

    Uses the beamformed trace (mean of all channels) for each detected
    window and runs the acoustic or fusion classifier.

    Returns (final_label, final_confidence, per_window_history).
    """
    import torch
    from acoustic_sim.ml.features import compute_mel_spectrogram
    from acoustic_sim.ml.data_generation import SOURCE_CLASSES

    history = []
    class_votes = {c: 0.0 for c in SOURCE_CLASSES}

    model = fusion_model if fusion_model is not None else acoustic_model
    if model is None:
        return "unknown", 0.0, history

    model.eval()
    device = next(model.parameters()).device

    for det in detections:
        if not det["detected"]:
            history.append({"label": "unknown", "confidence": 0.0})
            continue

        # Beamformed trace from filtered traces for this window.
        beamformed = np.mean(filtered_traces, axis=0)

        # Compute mel spectrogram.
        mel = compute_mel_spectrogram(beamformed, sample_rate)
        mel_tensor = torch.tensor(
            mel[np.newaxis, np.newaxis, :, :], dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            logits = model(mel_tensor) if acoustic_model is model else None
            if logits is None:
                # Fusion model needs kinematic input — use zeros as fallback.
                kin = torch.zeros(1, 14, device=device)
                logits = model(mel_tensor, kin)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])
        pred_label = SOURCE_CLASSES[pred_idx]

        if pred_conf < confidence_threshold:
            pred_label = "unknown"

        history.append({"label": pred_label, "confidence": pred_conf})
        class_votes[pred_label] += pred_conf

    # Final label: weighted majority vote over all windows.
    if class_votes:
        final_label = max(class_votes, key=class_votes.get)
        total = sum(class_votes.values())
        final_confidence = class_votes[final_label] / max(total, 1e-12)
    else:
        final_label = "unknown"
        final_confidence = 0.0

    return final_label, final_confidence, history


def _detect_maneuvers(
    track: dict,
    maneuver_model,
    buffer_size: int = 20,
) -> tuple[str, list[str]]:
    """Run maneuver detection on tracker history.

    Returns (current_maneuver_class, per_window_maneuver_history).
    """
    import torch
    from acoustic_sim.ml.data_generation import MANEUVER_CLASSES

    MULTIPLIERS = {
        "steady": 1.0,
        "turning": 5.0,
        "accelerating": 3.0,
        "diving": 5.0,
        "evasive": 10.0,
        "hovering": 0.5,
    }

    positions = track["positions"]
    velocities = track["velocities"]
    n = len(positions)
    history = []

    maneuver_model.eval()
    device = next(maneuver_model.parameters()).device

    current_class = "steady"

    for i in range(buffer_size, n + 1):
        window_pos = positions[i - buffer_size:i]
        window_vel = velocities[i - buffer_size:i]

        if np.any(np.isnan(window_pos)) or np.any(np.isnan(window_vel)):
            history.append("steady")
            continue

        # Normalize positions.
        mean_pos = np.mean(window_pos, axis=0)
        norm_pos = window_pos - mean_pos

        # Build feature: (buffer_size, 6) → (1, 6, buffer_size) for Conv1d.
        features = np.hstack([norm_pos, window_vel])  # (buffer_size, 6)
        x = torch.tensor(
            features.T[np.newaxis, :, :], dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            logits = maneuver_model(x)
            pred_idx = int(logits.argmax(dim=1).item())

        current_class = MANEUVER_CLASSES[pred_idx]
        history.append(current_class)

    return current_class, history


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

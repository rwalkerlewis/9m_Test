"""End-to-end passive acoustic drone detection, tracking, and fire control.

Architecture
============
The pipeline is split into three **completely independent** stages:

1. **simulate_scenario** — runs FDTD to generate synthetic sensor data.
   Produces ONLY what a real sensor system would observe: pressure
   traces, sensor positions, sample rate, local sound speed.

2. **run_detection** — the detection / tracking / fire-control pipeline.
   Takes ONLY sensor observables as input.  Knows NOTHING about the
   FDTD domain, velocity model, source trajectory, or simulation
   parameters.  This is the part that would run on real hardware.

3. **evaluate_results** — compares detection output to ground truth.
   Computes miss distances, detection rates, localization errors.
   Only used for testing / validation.

One FDTD run can feed many detection runs (e.g., sweeping thresholds,
injecting faults, testing different parameters) without re-running
the expensive forward model.
"""

from __future__ import annotations

import argparse
import math
import sys
import time as _time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from acoustic_sim.config import DetectionConfig


# =====================================================================
#  STAGE 1: FDTD Scenario Generation
# =====================================================================

def simulate_scenario(config: DetectionConfig | None = None) -> dict:
    """Run FDTD to produce synthetic sensor data.

    Returns a dict with two sections:

    **Sensor observables** (what a real system would have):
        ``traces`` (n_mics, n_samples), ``mic_positions`` (n_mics, 2),
        ``dt`` (float), ``sound_speed`` (float),
        ``weapon_position`` (2,)

    **Ground truth** (for evaluation only, NOT passed to detection):
        ``true_positions`` (n_steps, 2), ``true_velocities`` (n_steps, 2),
        ``true_times`` (n_steps,)
    """
    from acoustic_sim.domains import DomainMeta
    from acoustic_sim.fdtd import FDTDConfig, FDTDSolver
    from acoustic_sim.noise import add_all_noise
    from acoustic_sim.setup import build_domain, build_receivers, build_source, compute_dt
    from acoustic_sim.sources import source_velocity_at

    if config is None:
        config = DetectionConfig()

    # ── Domain ──────────────────────────────────────────────────────────
    model, meta = build_domain(
        config.domain_type,
        x_min=config.x_min, x_max=config.x_max,
        y_min=config.y_min, y_max=config.y_max,
        dx=config.dx, velocity=config.sound_speed,
        wind_speed=config.wind_speed,
        wind_direction_deg=config.wind_direction_deg,
        dirt_velocity=config.dirt_velocity,
        seed=config.seed,
    )

    # ── Receivers ───────────────────────────────────────────────────────
    receivers = build_receivers(
        config.array_type,
        count=config.n_mics, radius=config.array_radius,
        center_x=config.array_center_x, center_y=config.array_center_y,
        spacing=config.array_spacing, positions=config.mic_positions,
        seed=config.seed,
    )

    # ── Timing ──────────────────────────────────────────────────────────
    dt, f_max = compute_dt(model, meta, fd_order=config.fd_order)
    n_steps = int(math.ceil(config.total_time / dt))

    # ── Build drone source ──────────────────────────────────────────────
    src_type_map = {
        "linear": "moving", "circular": "circular_orbit",
        "figure_eight": "figure_eight",
        "loiter_approach": "loiter_approach", "evasive": "evasive",
    }
    source_type = src_type_map.get(config.trajectory_type, config.trajectory_type)

    drone_source = build_source(
        source_type, config.source_signal,
        n_steps=n_steps, dt=dt, f_max=f_max,
        x=config.source_start[0], y=config.source_start[1],
        x1=config.source_end[0], y1=config.source_end[1],
        speed=config.drone_speed,
        source_level_dB=config.source_level_dB,
        harmonic_amplitudes=config.harmonic_amplitudes,
        n_harmonics=config.n_harmonics,
        fundamental_freq=config.fundamental_freq,
        orbit_cx=config.orbit_center[0], orbit_cy=config.orbit_center[1],
        orbit_radius=config.orbit_radius,
        orbit_start_angle=config.orbit_start_angle,
        fig8_cx=config.fig8_center[0], fig8_cy=config.fig8_center[1],
        fig8_x_amp=config.fig8_x_amp, fig8_y_amp=config.fig8_y_amp,
        fig8_x_freq=config.fig8_x_freq, fig8_y_freq=config.fig8_y_freq,
        fig8_phase_offset=config.fig8_phase_offset,
        loiter_orbit_cx=config.loiter_orbit_center[0],
        loiter_orbit_cy=config.loiter_orbit_center[1],
        loiter_orbit_radius=config.loiter_orbit_radius,
        loiter_orbit_duration=config.loiter_orbit_duration,
        loiter_approach_x=config.loiter_approach_target[0],
        loiter_approach_y=config.loiter_approach_target[1],
        evasive_heading=config.evasive_heading,
        evasive_speed_var=config.evasive_speed_var,
        evasive_heading_var=config.evasive_heading_var,
        seed=config.seed,
    )

    # ── Record ground truth ─────────────────────────────────────────────
    true_positions = np.zeros((n_steps, 2))
    true_velocities = np.zeros((n_steps, 2))
    for i in range(n_steps):
        true_positions[i] = drone_source.position_at(i, dt)
        true_velocities[i] = source_velocity_at(drone_source, i, dt)
    true_times = np.arange(n_steps) * dt

    # ── Run FDTD ────────────────────────────────────────────────────────
    fdtd_cfg = FDTDConfig(
        total_time=config.total_time, dt=dt,
        damping_width=config.damping_width,
        damping_max=config.damping_max,
        air_absorption=config.air_absorption,
        source_amplitude=1.0,
        snapshot_interval=config.snapshot_interval,
        fd_order=config.fd_order,
    )
    solver = FDTDSolver(model, fdtd_cfg, drone_source, receivers, meta)
    drone_result = solver.run(verbose=True)
    drone_traces = drone_result["traces"]

    # ── Stationary source (optional, second FDTD) ──────────────────────
    stationary_traces = None
    if config.stationary_source_enabled:
        from acoustic_sim.sources import StaticSource, make_stationary_tonal
        stat_sig = make_stationary_tonal(
            n_steps, dt,
            base_freq=config.stationary_source_freq,
            n_harmonics=config.stationary_source_n_harmonics,
            source_level_dB=config.stationary_source_level_dB,
            f_max=f_max, seed=config.seed + 100,
        )
        stat_source = StaticSource(
            x=config.stationary_source_pos[0],
            y=config.stationary_source_pos[1],
            signal=stat_sig,
        )
        solver2 = FDTDSolver(model, fdtd_cfg, stat_source, receivers, meta)
        stat_result = solver2.run(verbose=True)
        stationary_traces = stat_result["traces"]

    # ── Add noise ───────────────────────────────────────────────────────
    traces = add_all_noise(
        drone_traces, stationary_traces, receivers, dt,
        wind_enabled=config.wind_noise_enabled,
        wind_level_dB=config.wind_noise_level_dB,
        wind_corner_freq=config.wind_corner_freq,
        wind_correlation_length=config.wind_correlation_length,
        sensor_enabled=config.sensor_noise_enabled,
        sensor_level_dB=config.sensor_noise_level_dB,
        seed=config.seed,
    )

    return {
        # ── Sensor observables (what detection sees) ────────────────────
        "traces": traces,
        "mic_positions": receivers.copy(),
        "dt": dt,
        "sound_speed": config.sound_speed,
        "weapon_position": np.array(config.weapon_position),
        # ── Ground truth (for evaluation only) ──────────────────────────
        "true_positions": true_positions,
        "true_velocities": true_velocities,
        "true_times": true_times,
        # ── Metadata (for plotting, not used by detection) ──────────────
        "model": model,
        "config": config,
    }


# =====================================================================
#  STAGE 2: Detection Pipeline (sensor data ONLY)
# =====================================================================

def run_detection(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    sound_speed: float = 343.0,
    weapon_position: tuple[float, float] | np.ndarray = (500.0, 500.0),
    *,
    # ── MFP (polar grid) ──
    azimuth_spacing_deg: float = 1.0,
    range_min: float = 20.0,
    range_max: float = 500.0,
    range_spacing: float = 5.0,
    window_length: float = 0.2,
    window_overlap: float = 0.5,
    n_subwindows: int = 4,
    detection_threshold: float = 0.25,
    min_signal_rms: float = 0.01,  # Minimum signal RMS to detect
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
    enable_position_calibration: bool = False,
    position_calibration_max_lag_m: float = 2.0,
    # ── Multi-source ──
    max_sources: int = 1,
    min_source_separation_deg: float = 10.0,
    # ── EKF Tracker ──
    tracker_process_noise_std: float = 2.0,
    tracker_sigma_bearing_deg: float = 3.0,
    tracker_sigma_range: float = 100.0,
    tracker_initial_range_guess: float = 200.0,
    tracker_gate_threshold: float = 30.0,
    tracker_max_missed: int = 5,
    source_level_dB: float = 90.0,
    # ── Fire control ──
    muzzle_velocity: float = 400.0,
    pellet_decel: float = 1.5,
    pattern_spread_rate: float = 0.025,
    lead_max_iterations: int = 5,
    priority_w_range: float = 1.0,
    priority_w_closing: float = 2.0,
    priority_w_quality: float = 0.5,
    # ── CUDA ──
    use_cuda: bool = False,
    # ── Legacy (ignored) ──
    grid_spacing: float = 5.0,
    grid_x_range: tuple[float, float] | None = None,
    grid_y_range: tuple[float, float] | None = None,
    min_source_separation_m: float = 20.0,
    tracker_measurement_noise_std: float = 5.0,
) -> dict:
    """Detection / tracking / fire-control — takes ONLY sensor data.

    This function knows NOTHING about the FDTD domain, velocity model,
    source trajectory, or simulation parameters.  It receives only what
    a real sensor system would provide.
    """
    from acoustic_sim.fire_control import run_fire_control, run_multi_fire_control
    from acoustic_sim.processor import matched_field_process
    from acoustic_sim.tracker import run_multi_tracker, run_tracker

    wp = np.asarray(weapon_position, dtype=np.float64)
    cx = float(np.mean(mic_positions[:, 0]))
    cy = float(np.mean(mic_positions[:, 1]))

    # ── Matched field processor ─────────────────────────────────────────
    mfp_result = matched_field_process(
        traces, mic_positions, dt,
        sound_speed=sound_speed,
        azimuth_spacing_deg=azimuth_spacing_deg,
        range_min=range_min,
        range_max=range_max,
        range_spacing=range_spacing,
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
        enable_position_calibration=enable_position_calibration,
        position_calibration_max_lag_m=position_calibration_max_lag_m,
        max_sources=max_sources,
        min_source_separation_deg=min_source_separation_deg,
        use_cuda=use_cuda,
    )
    detections = mfp_result["detections"]

    # ── EKF Tracker ─────────────────────────────────────────────────────
    track = run_tracker(
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
        multi_tracks = run_multi_tracker(
            mfp_result["multi_detections"], det_times,
            process_noise_std=tracker_process_noise_std,
            sigma_bearing_deg=tracker_sigma_bearing_deg,
            sigma_range=tracker_sigma_range,
            initial_range_guess=tracker_initial_range_guess,
            gate_threshold=tracker_gate_threshold,
            max_missed=tracker_max_missed,
            source_level_dB=source_level_dB,
            array_center_x=cx, array_center_y=cy,
        )

    # ── Fire control ────────────────────────────────────────────────────
    fc = run_fire_control(
        track, weapon_position=tuple(wp),
        muzzle_velocity=muzzle_velocity, pellet_decel=pellet_decel,
        pattern_spread_rate=pattern_spread_rate,
        max_iterations=lead_max_iterations,
    )

    fc_multi: list[dict] = []
    if multi_tracks:
        fc_multi = run_multi_fire_control(
            multi_tracks, weapon_position=tuple(wp),
            muzzle_velocity=muzzle_velocity, pellet_decel=pellet_decel,
            pattern_spread_rate=pattern_spread_rate,
            max_iterations=lead_max_iterations,
            w_range=priority_w_range, w_closing=priority_w_closing,
            w_quality=priority_w_quality,
        )

    return {
        "mfp_result": mfp_result,
        "track": track,
        "multi_tracks": multi_tracks,
        "fire_control": fc,
        "fire_control_multi": fc_multi,
    }


# =====================================================================
#  STAGE 3: Evaluation (detection output + ground truth)
# =====================================================================

def evaluate_results(
    detection_output: dict,
    true_positions: np.ndarray,
    true_velocities: np.ndarray,
    true_times: np.ndarray,
    weapon_position: tuple[float, float] | np.ndarray = (0.0, 0.0),
    pattern_spread_rate: float = 0.025,
) -> dict:
    """Compare detection output to ground truth.

    Returns detection rate, localization error, miss distances, etc.
    """
    from acoustic_sim.fire_control import compute_miss_distance

    track = detection_output["track"]
    fc = detection_output["fire_control"]
    detections = detection_output["mfp_result"]["detections"]

    n_detected = sum(1 for d in detections if d["detected"])
    n_windows = len(detections)
    detection_rate = n_detected / max(n_windows, 1)

    # Localization error.
    mean_loc_error = float("nan")
    valid_track = ~np.isnan(track["positions"][:, 0])
    if np.any(valid_track):
        det_times = track["times"][valid_track]
        true_x = np.interp(det_times, true_times, true_positions[:, 0])
        true_y = np.interp(det_times, true_times, true_positions[:, 1])
        est_pos = track["positions"][valid_track]
        errors = np.sqrt((est_pos[:, 0] - true_x) ** 2 +
                         (est_pos[:, 1] - true_y) ** 2)
        mean_loc_error = float(np.mean(errors))

    # Miss distance.
    miss = compute_miss_distance(
        fc, true_positions, true_times,
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


# =====================================================================
#  Convenience: run everything (FDTD → detect → evaluate)
# =====================================================================

def run_detection_pipeline(config: DetectionConfig | None = None) -> dict:
    """Run the full pipeline: simulate → detect → evaluate → plot.

    This is a convenience wrapper that calls all three stages.
    """
    from acoustic_sim.noise import (
        inject_sensor_faults,
        inject_transient,
        perturb_mic_positions,
    )
    from acoustic_sim.plotting import (
        plot_beam_power,
        plot_detection_domain,
        plot_detection_gather,
        plot_polar_beam_power,
        plot_tracking,
        plot_vespagram,
    )
    from acoustic_sim.validate import run_all_checks
    # The new MVDR processor operates in the frequency domain and does not
    # produce time-domain filtered traces.  For the SNR sanity check we
    # pass the raw traces — the check is a rough diagnostic, not exact.

    if config is None:
        config = DetectionConfig()

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_wall_start = _time.time()

    print("=" * 60)
    print("  PASSIVE ACOUSTIC DRONE DETECTION PIPELINE")
    print("=" * 60)

    # ── Stage 1: FDTD simulation ────────────────────────────────────────
    print("\n[Stage 1] Running FDTD scenario generation…")
    scenario = simulate_scenario(config)

    traces = scenario["traces"]
    mic_positions = scenario["mic_positions"]
    dt = scenario["dt"]

    # ── Post-hoc injections (faults, transients, position errors) ───────
    faulted_sensors: list[int] = []
    if config.inject_faults:
        traces, faulted_sensors = inject_sensor_faults(
            traces, fault_type=config.fault_type,
            fault_sensors=config.fault_sensors,
            fault_fraction=config.fault_fraction,
            fault_level_dB=config.fault_level_dB,
            seed=config.seed + 10,
        )
        print(f"   Faults injected: {config.fault_type} on {faulted_sensors}")

    if config.inject_transient:
        traces = inject_transient(
            traces, dt,
            event_time=config.transient_time,
            event_pos=config.transient_pos,
            mic_positions=mic_positions,
            level_dB=config.transient_level_dB,
            duration_ms=config.transient_duration_ms,
            sound_speed=config.sound_speed,
            seed=config.seed + 20,
        )
        print(f"   Transient injected: {config.transient_level_dB} dB")

    reported_positions = mic_positions
    if config.inject_position_error:
        reported_positions = perturb_mic_positions(
            mic_positions, error_std=config.position_error_std,
            seed=config.seed + 30,
        )
        print(f"   Position errors injected: std={config.position_error_std} m")

    # ── Sanity checks ───────────────────────────────────────────────────
    print("\n   Running sanity checks…")
    sample_rate = 1.0 / dt
    filtered_check = traces  # MVDR processor works in freq domain
    run_all_checks(
        traces, filtered_check, dt,
        reported_positions, scenario["true_positions"],
        source_level_dB=config.source_level_dB,
        sound_speed=config.sound_speed,
    )

    # ── Stage 2: Detection (sensor data only) ───────────────────────────
    print("[Stage 2] Running detection pipeline (sensor data only)…")
    detection_output = run_detection(
        traces, reported_positions, dt,
        sound_speed=config.sound_speed,
        weapon_position=config.weapon_position,
        # Polar grid
        azimuth_spacing_deg=config.mfp_azimuth_spacing_deg,
        range_min=config.mfp_range_min,
        range_max=config.mfp_range_max,
        range_spacing=config.mfp_range_spacing,
        window_length=config.mfp_window_length,
        window_overlap=config.mfp_window_overlap,
        n_subwindows=config.mfp_n_subwindows,
        detection_threshold=config.mfp_detection_threshold,
        min_signal_rms=config.mfp_min_signal_rms,
        fundamental=config.fundamental_freq,
        n_harmonics=config.n_harmonics,
        harmonic_bandwidth=config.mfp_harmonic_bandwidth,
        stationary_history=config.mfp_stationary_history,
        stationary_cv_threshold=config.mfp_stationary_cv_threshold,
        diagonal_loading=config.mfp_diagonal_loading,
        # Robustness
        enable_sensor_weights=config.enable_sensor_weights,
        sensor_fault_threshold=config.sensor_fault_threshold,
        enable_transient_blanking=config.enable_transient_blanking,
        transient_subwindow_ms=config.transient_subwindow_ms,
        transient_threshold_factor=config.transient_threshold_factor,
        enable_position_calibration=config.enable_position_calibration,
        position_calibration_max_lag_m=config.position_calibration_max_lag_m,
        max_sources=config.max_sources,
        # EKF tracker
        tracker_process_noise_std=config.tracker_process_noise_std,
        tracker_sigma_bearing_deg=config.tracker_sigma_bearing_deg,
        tracker_sigma_range=config.tracker_sigma_range,
        tracker_initial_range_guess=config.tracker_initial_range_guess,
        tracker_gate_threshold=config.tracker_gate_threshold,
        tracker_max_missed=config.tracker_max_missed,
        source_level_dB=config.source_level_dB,
        # Fire control
        muzzle_velocity=config.muzzle_velocity,
        pellet_decel=config.pellet_decel,
        pattern_spread_rate=config.pattern_spread_rate,
        lead_max_iterations=config.lead_max_iterations,
        priority_w_range=config.priority_w_range,
        priority_w_closing=config.priority_w_closing,
        priority_w_quality=config.priority_w_quality,
        use_cuda=config.use_cuda,
    )

    # ── Stage 3: Evaluation ─────────────────────────────────────────────
    print("\n[Stage 3] Evaluating results against ground truth…")
    metrics = evaluate_results(
        detection_output,
        scenario["true_positions"],
        scenario["true_velocities"],
        scenario["true_times"],
        weapon_position=config.weapon_position,
        pattern_spread_rate=config.pattern_spread_rate,
    )

    # Print results.
    print(f"   Detection rate: {metrics['detection_rate']*100:.0f}% "
          f"({metrics['n_detected']}/{metrics['n_windows']})")
    if np.isfinite(metrics["mean_loc_error"]):
        print(f"   Mean loc error: {metrics['mean_loc_error']:.1f} m")
    fm = metrics["first_shot_miss"]
    fp = metrics["first_shot_pattern"]
    fh = metrics["first_shot_hit"]
    if np.isfinite(fm):
        print(f"   1st shot miss:  {fm:.2f} m (pattern {fp:.2f} m → "
              f"{'HIT' if fh else 'MISS'})")

    # ── Plots ───────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    mfp = detection_output["mfp_result"]
    track = detection_output["track"]
    fc = detection_output["fire_control"]
    tp = scenario["true_positions"]
    tt = scenario["true_times"]

    plot_detection_domain(
        scenario["model"], reported_positions, tp,
        weapon_pos=config.weapon_position,
        stationary_pos=(config.stationary_source_pos
                        if config.stationary_source_enabled else None),
        output_path=str(out / "detection_domain.png"),
    )
    plot_detection_gather(
        traces, mfp["filtered_traces"], dt,
        output_path=str(out / "detection_gather.png"),
    )
    plot_beam_power(
        mfp["detections"], tp, mfp["grid_x"], mfp["grid_y"],
        output_path=str(out / "beam_power.png"),
    )
    plot_tracking(
        track, tp, tt, fc, config.weapon_position,
        output_path=str(out / "tracking.png"),
    )
    plot_vespagram(
        traces, reported_positions, dt,
        output_path=str(out / "vespagram.png"),
        sound_speed=config.sound_speed,
    )

    if "azimuths" in mfp and "ranges" in mfp:
        plot_polar_beam_power(
            mfp["detections"], mfp["azimuths"], mfp["ranges"],
            tp,
            array_center=(config.array_center_x, config.array_center_y),
            output_path=str(out / "polar_beam_power.png"),
        )

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = _time.time() - t_wall_start
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Detection rate: {metrics['detection_rate']*100:.0f}%")
    if np.isfinite(metrics["mean_loc_error"]):
        print(f"  Mean loc error: {metrics['mean_loc_error']:.1f} m")
    if np.isfinite(fm):
        print(f"  1st shot miss:  {fm:.2f} m  "
              f"(pattern {fp:.2f} m → {'HIT' if fh else 'MISS'})")
    print(f"  Wall time:      {elapsed:.1f} s")
    print(f"  Output:         {out}")
    print(f"{'=' * 60}")

    return {
        "scenario": scenario,
        "detection_output": detection_output,
        "metrics": metrics,
        "config": config,
        # Backward-compatible flat keys.
        "detection_rate": metrics["detection_rate"],
        "mean_loc_error": metrics["mean_loc_error"],
        "first_shot_miss": metrics["first_shot_miss"],
        "first_shot_hit": metrics["first_shot_hit"],
        "first_shot_pattern": metrics["first_shot_pattern"],
        "mean_miss": metrics["mean_miss"],
        "faulted_sensors": faulted_sensors,
        # Flat references for plotting code.
        "track": track,
        "fire_control": fc,
        "mfp_result": mfp,
        "noisy_traces": traces,
        "true_positions": tp,
        "true_times": tt,
        "dt": dt,
        "receivers": reported_positions,
    }


# =====================================================================
#  CLI
# =====================================================================

def parse_args(argv: list[str] | None = None) -> DetectionConfig:
    p = argparse.ArgumentParser(
        description="Passive acoustic drone detection pipeline",
    )
    p.add_argument("--trajectory", default="linear",
                   choices=["linear", "circular", "figure_eight",
                            "loiter_approach", "evasive"])
    p.add_argument("--domain", default="isotropic",
                   choices=["isotropic", "wind", "hills_vegetation",
                            "echo_canyon", "urban_echo"])
    p.add_argument("--total-time", type=float, default=0.5)
    p.add_argument("--dx", type=float, default=0.05)
    p.add_argument("--x-min", type=float, default=-15.0)
    p.add_argument("--x-max", type=float, default=15.0)
    p.add_argument("--y-min", type=float, default=-15.0)
    p.add_argument("--y-max", type=float, default=15.0)
    p.add_argument("--n-mics", type=int, default=16)
    p.add_argument("--array-type", default="circular")
    p.add_argument("--array-radius", type=float, default=0.5)
    p.add_argument("--drone-speed", type=float, default=15.0)
    p.add_argument("--source-level-dB", type=float, default=90.0)
    p.add_argument("--fundamental-freq", type=float, default=150.0)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--no-stationary", action="store_true")
    p.add_argument("--output-dir", default="output/detection")
    p.add_argument("--grid-spacing", type=float, default=1.0)
    p.add_argument("--detection-threshold", type=float, default=0.15)
    p.add_argument("--snapshot-interval", type=int, default=0)

    args = p.parse_args(argv)
    cfg = DetectionConfig(
        trajectory_type=args.trajectory, domain_type=args.domain,
        total_time=args.total_time, dx=args.dx,
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        n_mics=args.n_mics, array_type=args.array_type,
        array_radius=args.array_radius, drone_speed=args.drone_speed,
        source_level_dB=args.source_level_dB,
        fundamental_freq=args.fundamental_freq,
        output_dir=args.output_dir,
        mfp_grid_spacing=args.grid_spacing,
        mfp_detection_threshold=args.detection_threshold,
        snapshot_interval=args.snapshot_interval,
    )
    if args.no_noise:
        cfg.wind_noise_enabled = False
        cfg.sensor_noise_enabled = False
        cfg.stationary_source_enabled = False
    if args.no_stationary:
        cfg.stationary_source_enabled = False
    return cfg


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    run_detection_pipeline(config)


if __name__ == "__main__":
    main()

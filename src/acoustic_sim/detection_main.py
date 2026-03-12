"""End-to-end passive acoustic drone detection, tracking, and fire control.

Orchestrates the full pipeline:

1. Build domain (velocity model) and microphone array
2. Run FDTD for the drone source → drone traces
3. Optionally run FDTD for a stationary noise source → stationary traces
4. Add post-hoc noise (wind, sensor self-noise)
5. Run sanity checks
6. Run matched field processor → detections
7. Run Kalman tracker → smoothed track
8. Compute fire-control solution
9. Generate visualisations
10. Print summary statistics

Usage::

    python -m acoustic_sim.detection_main [options]
    python src/acoustic_sim/detection_main.py [options]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time as _time
from pathlib import Path

import numpy as np

# Ensure the package is importable when run as a script.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from acoustic_sim.config import DetectionConfig
from acoustic_sim.domains import DomainMeta
from acoustic_sim.fdtd import FDTDConfig, FDTDSolver
from acoustic_sim.fire_control import (
    compute_miss_distance,
    run_fire_control,
    run_multi_fire_control,
)
from acoustic_sim.noise import (
    add_all_noise,
    inject_sensor_faults,
    inject_transient,
    perturb_mic_positions,
)
from acoustic_sim.plotting import (
    plot_beam_power,
    plot_detection_domain,
    plot_detection_gather,
    plot_tracking,
    plot_vespagram,
)
from acoustic_sim.processor import matched_field_process
from acoustic_sim.setup import build_domain, build_receivers, build_source, compute_dt
from acoustic_sim.sources import source_velocity_at
from acoustic_sim.tracker import run_tracker, run_multi_tracker
from acoustic_sim.validate import run_all_checks


# -----------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------

def run_detection_pipeline(config: DetectionConfig | None = None) -> dict:
    """Run the full detection pipeline.

    Parameters
    ----------
    config : DetectionConfig or None
        Pipeline parameters.  Uses defaults if *None*.

    Returns
    -------
    dict with all intermediate and final results.
    """
    if config is None:
        config = DetectionConfig()

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t_wall_start = _time.time()

    # ── 1. Domain ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  PASSIVE ACOUSTIC DRONE DETECTION PIPELINE")
    print("=" * 60)
    print(f"\n[1/10] Building domain: {config.domain_type}")
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
    print(f"   Grid: {model.shape}, c=[{model.c_min:.1f}, {model.c_max:.1f}] m/s")

    # ── 2. Receivers ────────────────────────────────────────────────────
    print(f"\n[2/10] Building microphone array: {config.array_type}")
    receivers = build_receivers(
        config.array_type,
        count=config.n_mics,
        radius=config.array_radius,
        center_x=config.array_center_x,
        center_y=config.array_center_y,
        spacing=config.array_spacing,
        positions=config.mic_positions,
        seed=config.seed,
    )
    print(f"   {receivers.shape[0]} microphones")

    # ── 3. Timing ───────────────────────────────────────────────────────
    dt, f_max = compute_dt(model, meta, fd_order=config.fd_order)
    n_steps = int(math.ceil(config.total_time / dt))
    sample_rate = 1.0 / dt
    print(f"\n[3/10] FDTD timing: dt={dt:.2e}s, n_steps={n_steps}, "
          f"f_max={f_max:.1f} Hz, effective fs={sample_rate:.0f} Hz")

    # ── 4. Build drone source + run FDTD ────────────────────────────────
    print(f"\n[4/10] Building drone source: trajectory={config.trajectory_type}, "
          f"signal={config.source_signal}")

    # Map trajectory_type to build_source source_type and extra kwargs.
    src_type_map = {
        "linear": "moving",
        "circular": "circular_orbit",
        "figure_eight": "figure_eight",
        "loiter_approach": "loiter_approach",
        "evasive": "evasive",
    }
    source_type = src_type_map.get(config.trajectory_type, config.trajectory_type)

    drone_source = build_source(
        source_type, config.source_signal,
        n_steps=n_steps, dt=dt, f_max=f_max,
        # Linear / evasive start
        x=config.source_start[0], y=config.source_start[1],
        x1=config.source_end[0], y1=config.source_end[1],
        speed=config.drone_speed,
        # Drone harmonics
        source_level_dB=config.source_level_dB,
        harmonic_amplitudes=config.harmonic_amplitudes,
        n_harmonics=config.n_harmonics,
        fundamental_freq=config.fundamental_freq,
        # Circular orbit
        orbit_cx=config.orbit_center[0], orbit_cy=config.orbit_center[1],
        orbit_radius=config.orbit_radius,
        orbit_start_angle=config.orbit_start_angle,
        # Figure-eight
        fig8_cx=config.fig8_center[0], fig8_cy=config.fig8_center[1],
        fig8_x_amp=config.fig8_x_amp, fig8_y_amp=config.fig8_y_amp,
        fig8_x_freq=config.fig8_x_freq, fig8_y_freq=config.fig8_y_freq,
        fig8_phase_offset=config.fig8_phase_offset,
        # Loiter-approach
        loiter_orbit_cx=config.loiter_orbit_center[0],
        loiter_orbit_cy=config.loiter_orbit_center[1],
        loiter_orbit_radius=config.loiter_orbit_radius,
        loiter_orbit_duration=config.loiter_orbit_duration,
        loiter_approach_x=config.loiter_approach_target[0],
        loiter_approach_y=config.loiter_approach_target[1],
        # Evasive
        evasive_heading=config.evasive_heading,
        evasive_speed_var=config.evasive_speed_var,
        evasive_heading_var=config.evasive_heading_var,
        seed=config.seed,
    )

    # Record true positions and velocities.
    true_positions = np.zeros((n_steps, 2))
    true_velocities = np.zeros((n_steps, 2))
    for i in range(n_steps):
        true_positions[i] = drone_source.position_at(i, dt)
        true_velocities[i] = source_velocity_at(drone_source, i, dt)
    true_times = np.arange(n_steps) * dt

    # FDTD run for drone.
    print("   Running FDTD for drone source…")
    fdtd_cfg = FDTDConfig(
        total_time=config.total_time, dt=dt,
        damping_width=config.damping_width,
        damping_max=config.damping_max,
        air_absorption=config.air_absorption,
        source_amplitude=1.0,  # signal is already in Pa
        snapshot_interval=config.snapshot_interval,
        fd_order=config.fd_order,
    )
    solver = FDTDSolver(model, fdtd_cfg, drone_source, receivers, meta)
    drone_result = solver.run(verbose=True)
    drone_traces = drone_result["traces"]
    print(f"   Drone traces: {drone_traces.shape}")

    # ── 5. Stationary source (optional) ─────────────────────────────────
    stationary_traces = None
    if config.stationary_source_enabled:
        print(f"\n[5/10] Running FDTD for stationary source at "
              f"{config.stationary_source_pos}")
        from acoustic_sim.sources import StaticSource
        from acoustic_sim.sources import make_stationary_tonal

        stat_sig = make_stationary_tonal(
            n_steps, dt,
            base_freq=config.stationary_source_freq,
            n_harmonics=config.stationary_source_n_harmonics,
            source_level_dB=config.stationary_source_level_dB,
            f_max=f_max,
            seed=config.seed + 100,
        )
        stat_source = StaticSource(
            x=config.stationary_source_pos[0],
            y=config.stationary_source_pos[1],
            signal=stat_sig,
        )
        solver2 = FDTDSolver(model, fdtd_cfg, stat_source, receivers, meta)
        stat_result = solver2.run(verbose=True)
        stationary_traces = stat_result["traces"]
        print(f"   Stationary traces: {stationary_traces.shape}")
    else:
        print("\n[5/10] Stationary source disabled — skipping")

    # ── 6. Add noise ────────────────────────────────────────────────────
    print("\n[6/10] Adding post-hoc noise")
    noisy_traces = add_all_noise(
        drone_traces, stationary_traces, receivers, dt,
        wind_enabled=config.wind_noise_enabled,
        wind_level_dB=config.wind_noise_level_dB,
        wind_corner_freq=config.wind_corner_freq,
        wind_correlation_length=config.wind_correlation_length,
        sensor_enabled=config.sensor_noise_enabled,
        sensor_level_dB=config.sensor_noise_level_dB,
        seed=config.seed,
    )
    # Sensor fault injection.
    faulted_sensors: list[int] = []
    if config.inject_faults:
        noisy_traces, faulted_sensors = inject_sensor_faults(
            noisy_traces,
            fault_type=config.fault_type,
            fault_sensors=config.fault_sensors,
            fault_fraction=config.fault_fraction,
            fault_level_dB=config.fault_level_dB,
            seed=config.seed + 10,
        )
        print(f"   Sensor faults injected: {config.fault_type} on sensors {faulted_sensors}")

    # Transient (explosion) injection.
    if config.inject_transient:
        noisy_traces = inject_transient(
            noisy_traces, dt,
            event_time=config.transient_time,
            event_pos=config.transient_pos,
            mic_positions=receivers,
            level_dB=config.transient_level_dB,
            duration_ms=config.transient_duration_ms,
            sound_speed=config.sound_speed,
            seed=config.seed + 20,
        )
        print(f"   Transient injected: {config.transient_level_dB} dB at t={config.transient_time}s")

    # Position error injection.
    reported_positions = receivers
    if config.inject_position_error:
        reported_positions = perturb_mic_positions(
            receivers,
            error_std=config.position_error_std,
            seed=config.seed + 30,
        )
        pos_errs = np.sqrt(np.sum((reported_positions - receivers) ** 2, axis=1))
        print(f"   Position errors injected: mean={pos_errs.mean():.2f} m, max={pos_errs.max():.2f} m")

    print(f"   Noisy traces RMS: "
          f"{np.sqrt(np.mean(noisy_traces**2)):.2e} Pa")

    # ── 7. Sanity checks ───────────────────────────────────────────────
    print("\n[7/10] Running sanity checks")
    # We need filtered traces for the SNR check — compute a quick
    # broadband filter.
    from acoustic_sim.processor import apply_filter_bank, create_filter_bank
    fb = create_filter_bank(config.fundamental_freq, config.n_harmonics,
                            config.mfp_harmonic_bandwidth, sample_rate)
    filtered_for_check = apply_filter_bank(noisy_traces, fb) if fb else noisy_traces

    run_all_checks(
        noisy_traces, filtered_for_check, dt,
        receivers, true_positions,
        source_level_dB=config.source_level_dB,
        sound_speed=config.sound_speed,
    )

    # ── 8. Matched field processor ──────────────────────────────────────
    # Use reported (possibly perturbed) positions for the MFP.
    mic_pos_for_mfp = reported_positions if config.inject_position_error else receivers

    print("[8/10] Running matched field processor")
    mfp_result = matched_field_process(
        noisy_traces, mic_pos_for_mfp, dt,
        sound_speed=config.sound_speed,
        grid_spacing=config.mfp_grid_spacing,
        grid_x_range=config.mfp_grid_x_range,
        grid_y_range=config.mfp_grid_y_range,
        window_length=config.mfp_window_length,
        window_overlap=config.mfp_window_overlap,
        detection_threshold=config.mfp_detection_threshold,
        fundamental=config.fundamental_freq,
        n_harmonics=config.n_harmonics,
        harmonic_bandwidth=config.mfp_harmonic_bandwidth,
        stationary_history=config.mfp_stationary_history,
        stationary_cv_threshold=config.mfp_stationary_cv_threshold,
        # Robustness options
        enable_sensor_weights=config.enable_sensor_weights,
        sensor_fault_threshold=config.sensor_fault_threshold,
        enable_transient_blanking=config.enable_transient_blanking,
        transient_subwindow_ms=config.transient_subwindow_ms,
        transient_threshold_factor=config.transient_threshold_factor,
        max_sources=config.max_sources,
        min_source_separation_m=config.min_source_separation_m,
        enable_position_calibration=config.enable_position_calibration,
        position_calibration_max_lag_m=config.position_calibration_max_lag_m,
        use_cuda=config.use_cuda,
    )
    detections = mfp_result["detections"]
    n_detected = sum(1 for d in detections if d["detected"])
    n_windows = len(detections)
    print(f"   {n_detected}/{n_windows} windows detected "
          f"({100*n_detected/max(n_windows,1):.0f}%)")

    # ── 9. Tracker ──────────────────────────────────────────────────────
    print("\n[9/10] Running tracker")
    multi_tracks: list[dict] = []
    if config.max_sources > 1 and "multi_detections" in mfp_result:
        # Multi-target tracker.
        det_times_arr = np.array([d["time"] for d in detections])
        multi_tracks = run_multi_tracker(
            mfp_result["multi_detections"], det_times_arr,
            process_noise_std=config.tracker_process_noise_std,
            measurement_noise_std=config.tracker_measurement_noise_std,
            gate_threshold=config.tracker_gate_threshold,
            max_missed=config.tracker_max_missed,
        )
        print(f"   Multi-target: {len(multi_tracks)} tracks")

    # Always run single-target tracker for backward compatibility / metrics.
    track = run_tracker(
        detections,
        process_noise_std=config.tracker_process_noise_std,
        measurement_noise_std=config.tracker_measurement_noise_std,
    )

    # Compute localization error where we have both true and estimated.
    valid_track = ~np.isnan(track["positions"][:, 0])
    mean_err = float("nan")
    if np.any(valid_track):
        det_times = track["times"][valid_track]
        true_x_interp = np.interp(det_times, true_times, true_positions[:, 0])
        true_y_interp = np.interp(det_times, true_times, true_positions[:, 1])
        est_pos = track["positions"][valid_track]
        errors = np.sqrt((est_pos[:, 0] - true_x_interp) ** 2 +
                         (est_pos[:, 1] - true_y_interp) ** 2)
        mean_err = float(np.mean(errors))
        print(f"   Mean localisation error: {mean_err:.1f} m")

        true_vx_interp = np.interp(det_times, true_times, true_velocities[:, 0])
        true_vy_interp = np.interp(det_times, true_times, true_velocities[:, 1])
        est_vel = track["velocities"][valid_track]
        vel_err = np.sqrt((est_vel[:, 0] - true_vx_interp) ** 2 +
                          (est_vel[:, 1] - true_vy_interp) ** 2)
        print(f"   Mean velocity error: {np.mean(vel_err):.1f} m/s")
    else:
        print("   No valid track positions")

    # ── 10. Fire control ────────────────────────────────────────────────
    print("\n[10/10] Computing fire-control solution")
    if multi_tracks:
        fc_multi = run_multi_fire_control(
            multi_tracks,
            weapon_position=config.weapon_position,
            muzzle_velocity=config.muzzle_velocity,
            pellet_decel=config.pellet_decel,
            pattern_spread_rate=config.pattern_spread_rate,
            max_iterations=config.lead_max_iterations,
            w_range=config.priority_w_range,
            w_closing=config.priority_w_closing,
            w_quality=config.priority_w_quality,
        )
        for i, t in enumerate(fc_multi):
            n_f = int(np.sum(t["fire_control"]["can_fire"]))
            print(f"   Track {t['track_id']}: priority={t['priority_score']:.2f}, "
                  f"range={t['range']:.0f}m, fire_windows={n_f}")
    else:
        fc_multi = []

    fc = run_fire_control(
        track,
        weapon_position=config.weapon_position,
        muzzle_velocity=config.muzzle_velocity,
        pellet_decel=config.pellet_decel,
        pattern_spread_rate=config.pattern_spread_rate,
        max_iterations=config.lead_max_iterations,
    )
    n_fire = int(np.sum(fc["can_fire"]))
    print(f"   Primary track engagement windows: {n_fire}/{len(fc['can_fire'])}")

    # ── Miss distance (the real success metric) ─────────────────────────
    miss_result = compute_miss_distance(
        fc, true_positions, true_times,
        weapon_position=config.weapon_position,
        pattern_spread_rate=config.pattern_spread_rate,
    )
    first_miss = miss_result["first_shot_miss"]
    first_pat = miss_result["first_shot_pattern"]
    first_hit = miss_result["first_shot_hit"]
    first_time = miss_result["first_shot_time"]

    if not np.isnan(first_miss):
        print(f"\n   ** FIRST SHOT ANALYSIS **")
        print(f"   Time of first shot:   {first_time:.3f} s")
        print(f"   Miss distance:        {first_miss:.2f} m")
        print(f"   Pattern diameter:     {first_pat:.2f} m")
        print(f"   WOULD HIT:            {'YES' if first_hit else 'NO'}")
        valid_miss = ~np.isnan(miss_result["miss_distances"])
        if np.any(valid_miss):
            print(f"   Mean miss (all shots): {np.nanmean(miss_result['miss_distances']):.2f} m")
            print(f"   Min miss:             {np.nanmin(miss_result['miss_distances']):.2f} m")
            n_hits = int(np.sum(miss_result["would_hit"]))
            n_valid = int(np.sum(valid_miss))
            print(f"   Hit rate (all shots): {n_hits}/{n_valid} "
                  f"({100*n_hits/max(n_valid,1):.0f}%)")
    else:
        print("\n   ** No valid fire-control solution — cannot compute miss **")

    # ── Visualisations ──────────────────────────────────────────────────
    print("\nGenerating plots…")

    plot_detection_domain(
        model, receivers, true_positions,
        weapon_pos=config.weapon_position,
        stationary_pos=(config.stationary_source_pos
                        if config.stationary_source_enabled else None),
        output_path=str(out / "detection_domain.png"),
    )

    plot_detection_gather(
        noisy_traces, mfp_result["filtered_traces"], dt,
        output_path=str(out / "detection_gather.png"),
    )

    plot_beam_power(
        detections, true_positions,
        mfp_result["grid_x"], mfp_result["grid_y"],
        output_path=str(out / "beam_power.png"),
    )

    plot_tracking(
        track, true_positions, true_times, fc,
        config.weapon_position,
        output_path=str(out / "tracking.png"),
    )

    plot_vespagram(
        noisy_traces, receivers, dt,
        output_path=str(out / "vespagram.png"),
        sound_speed=config.sound_speed,
    )

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = _time.time() - t_wall_start
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Domain:         {config.x_max - config.x_min:.0f} × "
          f"{config.y_max - config.y_min:.0f} m, dx={config.dx}")
    print(f"  Duration:       {config.total_time:.1f} s "
          f"({n_steps} steps)")
    print(f"  Trajectory:     {config.trajectory_type}")
    print(f"  Detection rate: {100*n_detected/max(n_windows,1):.0f}% "
          f"({n_detected}/{n_windows})")
    if not np.isnan(mean_err):
        print(f"  Mean loc error: {mean_err:.1f} m")
    print(f"  Fire windows:   {n_fire}/{len(fc['can_fire'])}")
    if not np.isnan(first_miss):
        print(f"  1st shot miss:  {first_miss:.2f} m  "
              f"(pattern {first_pat:.2f} m → {'HIT' if first_hit else 'MISS'})")
    print(f"  Wall time:      {elapsed:.1f} s")
    print(f"  Output:         {out}")
    print("=" * 60)

    return {
        "config": config,
        "model": model,
        "receivers": receivers,
        "reported_positions": reported_positions if config.inject_position_error else receivers,
        "drone_traces": drone_traces,
        "stationary_traces": stationary_traces,
        "noisy_traces": noisy_traces,
        "true_positions": true_positions,
        "true_velocities": true_velocities,
        "true_times": true_times,
        "mfp_result": mfp_result,
        "track": track,
        "multi_tracks": multi_tracks,
        "fire_control": fc,
        "fire_control_multi": fc_multi,
        "dt": dt,
        "n_steps": n_steps,
        "detection_rate": n_detected / max(n_windows, 1),
        "mean_loc_error": mean_err,
        "faulted_sensors": faulted_sensors,
        "miss_result": miss_result,
        "first_shot_miss": first_miss,
        "first_shot_hit": first_hit,
        "first_shot_pattern": first_pat,
        "mean_miss": float(np.nanmean(miss_result["miss_distances"]))
                     if np.any(~np.isnan(miss_result["miss_distances"])) else float("nan"),
    }


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> DetectionConfig:
    """Parse command-line arguments into a :class:`DetectionConfig`."""
    p = argparse.ArgumentParser(
        description="Passive acoustic drone detection pipeline",
    )

    # Key overrides (most users only need these).
    p.add_argument("--trajectory", default="linear",
                   choices=["linear", "circular", "figure_eight",
                            "loiter_approach", "evasive"],
                   help="Drone trajectory type")
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
    p.add_argument("--no-noise", action="store_true",
                   help="Disable all noise sources")
    p.add_argument("--no-stationary", action="store_true",
                   help="Disable the stationary noise source")
    p.add_argument("--output-dir", default="output/detection")
    p.add_argument("--grid-spacing", type=float, default=1.0)
    p.add_argument("--detection-threshold", type=float, default=0.15)
    p.add_argument("--snapshot-interval", type=int, default=0)

    args = p.parse_args(argv)

    cfg = DetectionConfig(
        trajectory_type=args.trajectory,
        domain_type=args.domain,
        total_time=args.total_time,
        dx=args.dx,
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        n_mics=args.n_mics,
        array_type=args.array_type,
        array_radius=args.array_radius,
        drone_speed=args.drone_speed,
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

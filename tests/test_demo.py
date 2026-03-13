#!/usr/bin/env python3
"""Module 4.2 + 4.4: Adaptive tracker comparison and end-to-end demonstration.

4.2: Compare tracker with/without maneuver-adaptive process noise on
     maneuvering trajectories. Report 3D RMSE.
4.4: Full end-to-end demonstration scenario.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.sources_3d import (
    CircularOrbitSource3D,
    EvasiveSource3D,
    LoiterApproachSource3D,
    MovingSource3D,
    StaticSource3D,
    source_velocity_at_3d,
)
from acoustic_sim.sources import make_drone_harmonics, make_stationary_tonal
from acoustic_sim.forward_3d import simulate_3d_traces, simulate_scenario_3d
from acoustic_sim.receivers_3d import create_receiver_l_shaped_3d
from acoustic_sim.processor_3d import matched_field_process_3d
from acoustic_sim.tracker_3d import EKFTracker3D, run_tracker_3d
from acoustic_sim.fire_control_3d import run_fire_control_3d
from acoustic_sim.detection_main_3d import run_detection_3d, evaluate_results_3d
from acoustic_sim.plotting_3d import (
    plot_3d_trajectory,
    plot_altitude_vs_time,
    plot_tracking_3d,
)
from acoustic_sim.ml.data_generation import (
    MANEUVER_CLASSES,
    MANEUVER_TO_IDX,
    _make_bird_signal,
    generate_maneuver_dataset,
)
from acoustic_sim.ml.maneuver_classifier import ManeuverClassifier
from acoustic_sim.ml.training import train_classifier


# =====================================================================
#  4.2: Tracker Performance with/without Maneuver Detection
# =====================================================================

def _run_adaptive_tracker(
    detections: list[dict],
    maneuver_model,
    process_noise_std: float,
    array_center_x: float,
    array_center_y: float,
    buffer_size: int = 20,
) -> dict:
    """Run tracker with maneuver-adaptive process noise."""
    MULTIPLIERS = {
        "steady": 1.0,
        "turning": 5.0,
        "accelerating": 3.0,
        "diving": 5.0,
        "evasive": 10.0,
        "hovering": 0.5,
    }

    sigma_bearing_rad = math.radians(3.0)
    kf = EKFTracker3D(
        process_noise_std=process_noise_std,
        sigma_bearing=sigma_bearing_rad,
        sigma_range=100.0,
        initial_range_guess=200.0,
    )
    cx, cy = array_center_x, array_center_y

    times, positions, velocities = [], [], []
    maneuver_labels = []
    prev_t = None
    history_buffer: list[np.ndarray] = []

    device = next(maneuver_model.parameters()).device
    maneuver_model.eval()

    for det in detections:
        t = det["time"]
        is_det = det["detected"]
        bearing = det.get("bearing", float("nan"))
        range_est = det.get("range", float("nan"))
        z_est = det.get("z", 0.0) or 0.0
        amplitude = det.get("coherence", 0.0)

        if not kf._initialised:
            if is_det and not math.isnan(bearing):
                kf.initialise_from_detection(bearing, range_est, z_est, cx, cy)
            else:
                times.append(t)
                positions.append([float("nan")] * 3)
                velocities.append([0.0] * 3)
                maneuver_labels.append("steady")
                prev_t = t
                continue

        if prev_t is not None:
            dt_step = t - prev_t
            if dt_step > 0:
                kf.predict(dt_step)

        if is_det and not math.isnan(bearing):
            kf.update(bearing, range_est, amplitude, z_est, cx, cy)

        state = kf.get_state()
        times.append(t)
        positions.append(list(state[:3]))
        velocities.append(list(state[3:6]))

        # Maneuver detection on buffer.
        history_buffer.append(state.copy())
        if len(history_buffer) > buffer_size:
            history_buffer.pop(0)

        maneuver_class = "steady"
        if len(history_buffer) >= buffer_size:
            window = np.array(history_buffer[-buffer_size:])
            mean_pos = np.mean(window[:, :3], axis=0)
            norm_pos = window[:, :3] - mean_pos
            features = np.hstack([norm_pos, window[:, 3:6]])  # (buf, 6)
            x = torch.tensor(
                features.T[np.newaxis, :, :], dtype=torch.float32
            ).to(device)
            with torch.no_grad():
                logits = maneuver_model(x)
                pred_idx = int(logits.argmax(dim=1).item())
            maneuver_class = MANEUVER_CLASSES[pred_idx]
            # Adapt process noise for next step.
            kf.set_process_noise_multiplier(MULTIPLIERS.get(maneuver_class, 1.0))

        maneuver_labels.append(maneuver_class)
        prev_t = t

    return {
        "times": np.array(times),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "maneuver_labels": maneuver_labels,
    }


def _compute_3d_rmse(
    track: dict,
    true_positions: np.ndarray,
    true_times: np.ndarray,
) -> float:
    """Compute 3D position RMSE between track and truth."""
    valid = ~np.isnan(track["positions"][:, 0])
    if not np.any(valid):
        return float("nan")
    det_times = track["times"][valid]
    true_x = np.interp(det_times, true_times, true_positions[:, 0])
    true_y = np.interp(det_times, true_times, true_positions[:, 1])
    true_z = np.interp(det_times, true_times, true_positions[:, 2])
    est_pos = track["positions"][valid]
    errors = np.sqrt(
        (est_pos[:, 0] - true_x) ** 2
        + (est_pos[:, 1] - true_y) ** 2
        + (est_pos[:, 2] - true_z) ** 2
    )
    return float(np.sqrt(np.mean(errors ** 2)))


def test_tracker_comparison():
    """Compare baseline vs adaptive tracker on maneuvering trajectories."""
    print("\n" + "=" * 60)
    print("  4.2: Tracker Performance — Baseline vs Adaptive")
    print("=" * 60)

    out_dir = Path("output/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train maneuver classifier.
    print("\n  Training maneuver classifier...")
    rng = np.random.default_rng(42)
    dataset = generate_maneuver_dataset(n_samples_per_class=400, seed=42)
    X = torch.tensor(dataset["features"].transpose(0, 2, 1), dtype=torch.float32)
    y = torch.tensor(dataset["labels"], dtype=torch.long)
    n = len(y)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    model = ManeuverClassifier(len(MANEUVER_CLASSES))
    train_classifier(model, X[idx[:split]], y[idx[:split]],
                     X[idx[split:]], y[idx[split:]],
                     n_epochs=50, lr=1e-3, verbose=False)

    # Test scenarios.
    dt = 1.0 / 4000
    n_steps = 12000  # 3 seconds
    sound_speed = 343.0
    mics = create_receiver_l_shaped_3d(5, 5, spacing=0.3,
                                        origin_x=295.0, origin_y=295.0, z=0.0)
    cx = float(np.mean(mics[:, 0]))
    cy = float(np.mean(mics[:, 1]))

    scenarios = []

    # Scenario 1: Loiter-and-approach with descent.
    sig1 = make_drone_harmonics(n_steps, dt, fundamental=150.0, n_harmonics=6,
                                 source_level_dB=90.0)
    src1 = LoiterApproachSource3D(
        orbit_cx=300.0, orbit_cy=200.0, orbit_radius=60.0,
        orbit_duration=1.5,
        approach_target_x=300.0, approach_target_y=300.0,
        approach_speed=15.0, signal=sig1,
        orbit_altitude=60.0, descent_rate=3.0,
    )
    scenarios.append(("Loiter-approach+descent", src1))

    # Scenario 2: Evasive with altitude changes.
    sig2 = make_drone_harmonics(n_steps, dt, fundamental=120.0, n_harmonics=6,
                                 source_level_dB=90.0)
    src2 = EvasiveSource3D(
        x0=250.0, y0=250.0, z0=50.0,
        heading=0.5, mean_speed=12.0, speed_var=3.0,
        heading_var=0.5, signal=sig2,
        mean_altitude=50.0, z_variance=8.0, seed=42,
    )
    scenarios.append(("Evasive+altitude", src2))

    # Scenario 3: Vertical descent.
    sig3 = make_drone_harmonics(n_steps, dt, fundamental=180.0, n_harmonics=6,
                                 source_level_dB=90.0)
    src3 = MovingSource3D(
        x0=310.0, y0=310.0, z0=100.0,
        x1=310.0, y1=310.0, z1=20.0,
        speed=30.0, signal=sig3,
    )
    scenarios.append(("Vertical descent", src3))

    print(f"\n  {'Scenario':<30s} {'Baseline RMSE':>15s} {'Adaptive RMSE':>15s} {'Improvement':>12s}")
    print(f"  {'-' * 72}")

    for name, src in scenarios:
        # Generate traces.
        traces = simulate_3d_traces(src, mics, dt, n_steps, sound_speed)
        # Add noise.
        noise = np.random.default_rng(42).standard_normal(traces.shape) * 1e-4
        traces += noise

        # Ground truth.
        true_positions = np.array([src.position_at(i, dt) for i in range(n_steps)])
        true_times = np.arange(n_steps) * dt

        # Run MFP.
        mfp = matched_field_process_3d(
            traces, mics, dt,
            sound_speed=sound_speed,
            azimuth_spacing_deg=2.0,
            range_min=5.0, range_max=150.0, range_spacing=10.0,
            z_min=0.0, z_max=120.0, z_spacing=20.0,
            detection_threshold=0.15,
            fundamental=150.0, n_harmonics=4,
        )
        detections = mfp["detections"]

        # Baseline tracker (constant Q).
        track_baseline = run_tracker_3d(
            detections,
            process_noise_std=2.0,
            sigma_bearing_deg=3.0,
            sigma_range=100.0,
            array_center_x=cx, array_center_y=cy,
        )
        rmse_baseline = _compute_3d_rmse(track_baseline, true_positions, true_times)

        # Adaptive tracker.
        track_adaptive = _run_adaptive_tracker(
            detections, model,
            process_noise_std=2.0,
            array_center_x=cx, array_center_y=cy,
        )
        rmse_adaptive = _compute_3d_rmse(track_adaptive, true_positions, true_times)

        improvement = (rmse_baseline - rmse_adaptive) / max(rmse_baseline, 1e-6) * 100
        print(f"  {name:<30s} {rmse_baseline:>15.1f} {rmse_adaptive:>15.1f} {improvement:>11.1f}%")

    print()


# =====================================================================
#  4.4: End-to-End Demonstration
# =====================================================================

def test_end_to_end_demo():
    """Full end-to-end demonstration scenario per spec Section 4.4."""
    print("\n" + "=" * 60)
    print("  4.4: End-to-End Demonstration")
    print("=" * 60)

    out_dir = Path("output/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / 4000
    n_steps = 20000  # 5 seconds
    sound_speed = 343.0

    # ── Sources ─────────────────────────────────────────────────────
    # Quadcopter: loiter at z=60m, approach, descend to z=20m.
    sig_drone = make_drone_harmonics(
        n_steps, dt, fundamental=150.0, n_harmonics=6, source_level_dB=90.0,
    )
    drone = LoiterApproachSource3D(
        orbit_cx=300.0, orbit_cy=100.0, orbit_radius=80.0,
        orbit_duration=3.0,
        approach_target_x=300.0, approach_target_y=300.0,
        approach_speed=15.0, signal=sig_drone,
        orbit_altitude=60.0, descent_rate=2.0,
    )

    # Bird at z=40m with flap-glide altitude oscillation.
    sig_bird = _make_bird_signal(n_steps, dt, wing_beat_freq=6.0, seed=99)
    bird = MovingSource3D(
        x0=250.0, y0=200.0, z0=40.0,
        x1=350.0, y1=200.0, z1=45.0,
        speed=12.0, signal=sig_bird,
    )

    # Stationary generator noise at z=0.
    sig_gen = make_stationary_tonal(
        n_steps, dt, base_freq=60.0, source_level_dB=75.0,
    )
    generator = StaticSource3D(x=350.0, y=250.0, z=0.0, signal=sig_gen)

    # ── Array ───────────────────────────────────────────────────────
    mics = create_receiver_l_shaped_3d(
        5, 5, spacing=0.3, origin_x=295.0, origin_y=295.0, z=0.0,
    )

    # ── Forward model ───────────────────────────────────────────────
    print("\n  Generating scenario traces (3 sources)...")
    scenario = simulate_scenario_3d(
        [drone, bird, generator], mics, dt, n_steps,
        sound_speed=sound_speed,
        wind_noise_enabled=True, wind_noise_level_dB=55.0,
        sensor_noise_enabled=True, sensor_noise_level_dB=40.0,
        seed=42,
    )

    # ── Detection pipeline ──────────────────────────────────────────
    print("  Running 3D detection pipeline...")
    detection_output = run_detection_3d(
        scenario["traces"], mics, dt,
        sound_speed=sound_speed,
        weapon_position=(300.0, 300.0, 0.0),
        azimuth_spacing_deg=2.0,
        range_min=10.0, range_max=200.0, range_spacing=10.0,
        z_min=0.0, z_max=100.0, z_spacing=20.0,
        detection_threshold=0.15,
        fundamental=150.0, n_harmonics=6,
        max_sources=3,
    )

    # ── Results ─────────────────────────────────────────────────────
    n_det = sum(1 for d in detection_output["mfp_result"]["detections"]
                if d["detected"])
    n_win = len(detection_output["mfp_result"]["detections"])
    print(f"  Detections: {n_det}/{n_win}")

    # Evaluate against drone (source 0).
    drone_truth = scenario["true_positions"][0]
    drone_times = scenario["true_times"]
    eval_result = evaluate_results_3d(
        detection_output, drone_truth, scenario["true_velocities"][0],
        drone_times, weapon_position=(300.0, 300.0, 0.0),
    )
    print(f"  Detection rate: {eval_result['detection_rate']:.1%}")
    if np.isfinite(eval_result["mean_loc_error"]):
        print(f"  Mean 3D loc error: {eval_result['mean_loc_error']:.1f} m")

    # Fire control status.
    fc = detection_output["fire_control"]
    n_fire = int(np.sum(fc["can_fire"]))
    print(f"  Fire windows: {n_fire}/{len(fc['can_fire'])}")

    # ── Plots ───────────────────────────────────────────────────────
    print("\n  Generating demonstration plots...")

    # 3D trajectory.
    plot_3d_trajectory(
        drone_truth,
        estimated_positions=detection_output["track"]["positions"],
        mic_positions=mics,
        weapon_pos=np.array([300.0, 300.0, 0.0]),
        title="End-to-End Demo: Quadcopter + Bird + Generator",
        output_path=str(out_dir / "demo_trajectory_3d.png"),
    )

    # Altitude vs time.
    track = detection_output["track"]
    plot_altitude_vs_time(
        drone_times, drone_truth[:, 2],
        estimated_z=track["positions"][:, 2],
        estimated_times=track["times"],
        title="Drone Altitude Tracking",
        output_path=str(out_dir / "demo_altitude.png"),
    )

    # Full tracking display.
    plot_tracking_3d(
        track, drone_truth, drone_times, fc,
        weapon_pos=(300.0, 300.0, 0.0),
        output_path=str(out_dir / "demo_tracking_3d.png"),
    )

    # Bird trajectory.
    bird_truth = scenario["true_positions"][1]
    plot_3d_trajectory(
        bird_truth,
        title="Bird Trajectory (non-threat)",
        output_path=str(out_dir / "demo_bird_3d.png"),
    )

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n  {'=' * 50}")
    print(f"  DEMONSTRATION SUMMARY")
    print(f"  {'=' * 50}")
    print(f"  Sources: quadcopter (z=60m→20m), bird (z=40m), generator (z=0)")
    print(f"  Array: 9-element L-shaped at z=0")
    print(f"  Wind noise: 55 dB")
    print(f"  Detection rate: {eval_result['detection_rate']:.1%}")
    if np.isfinite(eval_result["mean_loc_error"]):
        print(f"  Mean 3D loc error: {eval_result['mean_loc_error']:.1f} m")
    print(f"  Fire control windows: {n_fire}")
    print(f"  Classification: {detection_output.get('class_label', 'N/A')}")
    print(f"  Maneuver: {detection_output.get('maneuver_class', 'N/A')}")
    print(f"  Output: {out_dir}")
    print(f"\n  *** DEMONSTRATION COMPLETE ***")


# =====================================================================
#  Run all
# =====================================================================

def run_all():
    test_tracker_comparison()
    test_end_to_end_demo()


if __name__ == "__main__":
    run_all()

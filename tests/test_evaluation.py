#!/usr/bin/env python3
"""Module 4: Full evaluation and demonstration.

Runs all classifiers, compares performance, and generates the
end-to-end demonstration scenario.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.ml.data_generation import (
    SOURCE_CLASSES,
    MANEUVER_CLASSES,
    generate_classification_dataset,
    generate_maneuver_dataset,
)
from acoustic_sim.ml.features import compute_kinematic_features, compute_mel_spectrogram
from acoustic_sim.ml.acoustic_classifier import AcousticClassifier
from acoustic_sim.ml.maneuver_classifier import ManeuverClassifier
from acoustic_sim.ml.fusion_classifier import (
    FusionClassifier,
    KinematicOnlyClassifier,
)
from acoustic_sim.ml.training import (
    evaluate_classifier,
    evaluate_fusion_classifier,
    prepare_acoustic_data,
    train_classifier,
    train_fusion_classifier,
)

from acoustic_sim.sources_3d import (
    CircularOrbitSource3D,
    EvasiveSource3D,
    LoiterApproachSource3D,
    MovingSource3D,
    StaticSource3D,
    source_velocity_at_3d,
)
from acoustic_sim.sources import make_drone_harmonics
from acoustic_sim.forward_3d import simulate_3d_traces, simulate_scenario_3d
from acoustic_sim.receivers_3d import create_receiver_l_shaped_3d
from acoustic_sim.detection_main_3d import run_detection_3d, evaluate_results_3d
from acoustic_sim.fire_control_3d import run_fire_control_3d
from acoustic_sim.tracker_3d import run_tracker_3d
from acoustic_sim.processor_3d import matched_field_process_3d
from acoustic_sim.plotting_3d import (
    plot_3d_trajectory,
    plot_altitude_vs_time,
    plot_tracking_3d,
    plot_kinematic_scatter,
)


def run_full_evaluation():
    """Run the complete Module 4 evaluation."""
    print("\n" + "=" * 60)
    print("  MODULE 4: FULL EVALUATION AND DEMONSTRATION")
    print("=" * 60)

    out_dir = Path("output/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── 4.1: Classification comparison ─────────────────────────────
    print("\n" + "-" * 60)
    print("  4.1: Classification Performance Comparison")
    print("-" * 60)

    # Generate shared dataset.
    print("\n  Generating classification dataset...")
    dataset = generate_classification_dataset(
        n_samples_per_class=200, dt=1.0 / 4000,
        window_duration=0.5, seed=42,
    )
    sample_rate = 1.0 / dataset["dt"]

    # Prepare acoustic data.
    X_acoustic, y = prepare_acoustic_data(
        dataset["signals"], dataset["labels"], sample_rate,
    )

    # Generate kinematic features for each sample.
    rng = np.random.default_rng(42)
    kinematic_features = []
    for params in dataset["params"]:
        # Simulate kinematic features based on class.
        cls = params["class"]
        speed = params["speed"]
        alt = params["altitude"]
        window_size = 50
        dt_kin = 0.1

        positions = np.zeros((window_size, 3))
        velocities = np.zeros((window_size, 3))

        if cls == "quadcopter":
            vx = speed * 0.7
            vy = speed * 0.5
            for i in range(window_size):
                positions[i] = [i * dt_kin * vx, i * dt_kin * vy, alt + rng.normal(0, 2)]
                velocities[i] = [vx + rng.normal(0, 1), vy + rng.normal(0, 1), rng.normal(0, 0.5)]
        elif cls == "hexacopter":
            vx = speed * 0.6
            vy = speed * 0.6
            for i in range(window_size):
                positions[i] = [i * dt_kin * vx, i * dt_kin * vy, alt + rng.normal(0, 1.5)]
                velocities[i] = [vx + rng.normal(0, 0.8), vy + rng.normal(0, 0.8), rng.normal(0, 0.3)]
        elif cls == "fixed_wing":
            vx = speed
            for i in range(window_size):
                positions[i] = [i * dt_kin * vx, rng.normal(0, 0.5), alt + rng.normal(0, 0.5)]
                velocities[i] = [vx + rng.normal(0, 0.5), rng.normal(0, 0.3), rng.normal(0, 0.2)]
        elif cls == "bird":
            for i in range(window_size):
                t = i * dt_kin
                vx = speed * math.cos(t * 0.5) + rng.normal(0, 2)
                vy = speed * math.sin(t * 0.3) + rng.normal(0, 2)
                z_osc = alt + 10 * math.sin(t * 0.3)
                positions[i] = [50 + vx * t, 50 + vy * t, z_osc]
                velocities[i] = [vx, vy, 10 * 0.3 * math.cos(t * 0.3)]
        elif cls == "ground_vehicle":
            vx = speed
            for i in range(window_size):
                positions[i] = [i * dt_kin * vx, rng.normal(0, 0.3), 0]
                velocities[i] = [vx + rng.normal(0, 0.3), rng.normal(0, 0.1), 0]
        else:  # unknown
            for i in range(window_size):
                positions[i] = [rng.normal(0, 5), rng.normal(0, 5), rng.uniform(0, 50)]
                velocities[i] = [rng.normal(0, 3), rng.normal(0, 3), rng.normal(0, 1)]

        # Add noise.
        positions += rng.normal(0, 2, positions.shape)
        velocities += rng.normal(0, 1, velocities.shape)
        kf = compute_kinematic_features(positions, velocities, dt_kin)
        kinematic_features.append(kf)

    X_kinematic = torch.tensor(np.array(kinematic_features), dtype=torch.float32)

    # Split.
    n = len(y)
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_a_train, y_train = X_acoustic[train_idx], y[train_idx]
    X_a_val, y_val = X_acoustic[val_idx], y[val_idx]
    X_k_train = X_kinematic[train_idx]
    X_k_val = X_kinematic[val_idx]
    snr_vals = np.array(dataset["snr_dbs"])[val_idx]

    # Train Classifier A: Acoustic-only.
    print("\n  Training Classifier A (acoustic-only)...")
    model_a = AcousticClassifier(len(SOURCE_CLASSES))
    train_classifier(model_a, X_a_train, y_train, X_a_val, y_val,
                     n_epochs=50, lr=1e-3, verbose=True)
    metrics_a = evaluate_classifier(model_a, X_a_val, y_val, SOURCE_CLASSES)

    # Train Classifier C: Kinematic-only.
    print("\n  Training Classifier C (kinematic-only)...")
    model_c = KinematicOnlyClassifier(len(SOURCE_CLASSES))
    train_classifier(model_c, X_k_train, y_train, X_k_val, y_val,
                     n_epochs=50, lr=1e-3, verbose=True)
    metrics_c = evaluate_classifier(model_c, X_k_val, y_val, SOURCE_CLASSES)

    # Train Classifier B: Fusion.
    print("\n  Training Classifier B (fusion)...")
    model_b = FusionClassifier(len(SOURCE_CLASSES))
    model_b.load_acoustic_weights(model_a)
    train_fusion_classifier(
        model_b, X_a_train, X_k_train, y_train,
        X_a_val, X_k_val, y_val,
        n_epochs=50, lr=5e-4, verbose=True,
    )
    metrics_b = evaluate_fusion_classifier(
        model_b, X_a_val, X_k_val, y_val, SOURCE_CLASSES,
    )

    # Comparison table.
    print("\n  " + "=" * 60)
    print("  Classification Comparison")
    print("  " + "=" * 60)
    print(f"  {'Classifier':<20s} {'Accuracy':>10s}")
    print(f"  {'-' * 30}")
    print(f"  {'A: Acoustic-only':<20s} {metrics_a['accuracy']:>10.3f}")
    print(f"  {'B: Fusion':<20s} {metrics_b['accuracy']:>10.3f}")
    print(f"  {'C: Kinematic-only':<20s} {metrics_c['accuracy']:>10.3f}")

    print(f"\n  Per-class F1 comparison:")
    print(f"  {'Class':>14s} {'Acoustic':>10s} {'Fusion':>10s} {'Kinematic':>10s}")
    for name in SOURCE_CLASSES:
        fa = metrics_a["per_class"][name]["f1"]
        fb = metrics_b["per_class"][name]["f1"]
        fc = metrics_c["per_class"][name]["f1"]
        print(f"  {name:>14s} {fa:>10.3f} {fb:>10.3f} {fc:>10.3f}")

    # SNR-stratified comparison.
    print(f"\n  SNR-stratified accuracy:")
    print(f"  {'SNR Range':>15s} {'Acoustic':>10s} {'Fusion':>10s} {'Kinematic':>10s}")
    snr_ranges = [("High (>15 dB)", snr_vals > 15),
                  ("Med (5-15 dB)", (snr_vals >= 5) & (snr_vals <= 15)),
                  ("Low (<5 dB)", snr_vals < 5)]
    for label, mask in snr_ranges:
        if not np.any(mask):
            continue
        y_sub = y_val[mask].numpy()
        acc_a = float(np.mean(metrics_a["predictions"][mask] == y_sub))
        acc_b = float(np.mean(metrics_b["predictions"][mask] == y_sub))
        acc_c = float(np.mean(metrics_c["predictions"][mask] == y_sub))
        print(f"  {label:>15s} {acc_a:>10.3f} {acc_b:>10.3f} {acc_c:>10.3f}")

    # ─── 4.2: Tracker performance ───────────────────────────────────
    print("\n" + "-" * 60)
    print("  4.2: Tracker Performance (baseline vs adaptive)")
    print("-" * 60)

    # Train maneuver classifier for adaptive tracking.
    print("\n  Training maneuver classifier...")
    man_dataset = generate_maneuver_dataset(
        n_samples_per_class=400, window_size=20, seed=42,
    )
    X_man = torch.tensor(man_dataset["features"].transpose(0, 2, 1), dtype=torch.float32)
    y_man = torch.tensor(man_dataset["labels"], dtype=torch.long)
    n_man = len(y_man)
    idx_man = rng.permutation(n_man)
    split_man = int(0.8 * n_man)
    model_man = ManeuverClassifier(len(MANEUVER_CLASSES))
    train_classifier(
        model_man, X_man[idx_man[:split_man]], y_man[idx_man[:split_man]],
        X_man[idx_man[split_man:]], y_man[idx_man[split_man:]],
        n_epochs=50, lr=1e-3, verbose=True,
    )
    metrics_man = evaluate_classifier(
        model_man, X_man[idx_man[split_man:]], y_man[idx_man[split_man:]],
        MANEUVER_CLASSES,
    )
    print(f"\n  Maneuver classifier accuracy: {metrics_man['accuracy']:.3f}")

    # ─── 4.3: Kinematic discriminant analysis ───────────────────────
    print("\n" + "-" * 60)
    print("  4.3: Kinematic Discriminant Analysis")
    print("-" * 60)

    kin_by_class = {}
    for i, name in enumerate(SOURCE_CLASSES):
        mask = y_val.numpy() == i
        if np.any(mask):
            kin_by_class[name] = X_kinematic[val_idx][mask].numpy()

    # Plot 1: speed_std vs heading_rate_std.
    scatter1 = {}
    for name, feats in kin_by_class.items():
        scatter1[name] = feats[:, [1, 4]]  # speed_std, heading_rate_std
    plot_kinematic_scatter(
        scatter1, ("speed_std", "heading_rate_std"),
        title="Speed Variability vs Heading Rate Variability",
        output_path=str(out_dir / "scatter_speed_heading.png"),
    )

    # Plot 2: z_std vs hover_fraction.
    scatter2 = {}
    for name, feats in kin_by_class.items():
        scatter2[name] = feats[:, [8, 11]]  # z_std, hover_fraction
    plot_kinematic_scatter(
        scatter2, ("z_std", "hover_fraction"),
        title="Altitude Variability vs Hover Fraction",
        output_path=str(out_dir / "scatter_z_hover.png"),
    )

    # Plot 3: curvature_std vs speed_min.
    scatter3 = {}
    for name, feats in kin_by_class.items():
        scatter3[name] = feats[:, [6, 2]]  # curvature_std, speed_min
    plot_kinematic_scatter(
        scatter3, ("curvature_std", "speed_min"),
        title="Curvature Variability vs Minimum Speed",
        output_path=str(out_dir / "scatter_curvature_speed.png"),
    )

    # ─── 4.4: End-to-end demonstration ─────────────────────────────
    print("\n" + "-" * 60)
    print("  4.4: End-to-End Demonstration")
    print("-" * 60)

    dt = 1.0 / 4000
    n_steps = 20000  # 5 seconds
    sound_speed = 343.0

    # Quadcopter: loiter then approach.
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

    # Bird.
    from acoustic_sim.ml.data_generation import _make_bird_signal
    sig_bird = _make_bird_signal(n_steps, dt, wing_beat_freq=6.0, seed=99)
    bird = MovingSource3D(
        x0=250.0, y0=200.0, z0=40.0,
        x1=350.0, y1=200.0, z1=45.0,
        speed=12.0, signal=sig_bird,
    )

    # Stationary generator.
    from acoustic_sim.sources import make_stationary_tonal
    sig_gen = make_stationary_tonal(n_steps, dt, base_freq=60.0,
                                    source_level_dB=75.0)
    generator = StaticSource3D(x=350.0, y=250.0, z=0.0, signal=sig_gen)

    # Microphones.
    mics = create_receiver_l_shaped_3d(5, 5, spacing=0.3,
                                        origin_x=295.0, origin_y=295.0, z=0.0)

    # Generate traces.
    print("\n  Generating scenario traces...")
    scenario = simulate_scenario_3d(
        [drone, bird, generator], mics, dt, n_steps,
        sound_speed=sound_speed,
        wind_noise_enabled=True, wind_noise_level_dB=55.0,
        sensor_noise_enabled=True, sensor_noise_level_dB=40.0,
        seed=42,
    )

    # Run detection pipeline.
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

    n_det = sum(1 for d in detection_output["mfp_result"]["detections"]
                if d["detected"])
    n_windows = len(detection_output["mfp_result"]["detections"])
    print(f"  Detections: {n_det}/{n_windows}")

    # Evaluate against drone truth.
    drone_truth = scenario["true_positions"][0]  # First source = drone
    drone_times = scenario["true_times"]
    eval_result = evaluate_results_3d(
        detection_output, drone_truth, scenario["true_velocities"][0],
        drone_times, weapon_position=(300.0, 300.0, 0.0),
    )
    print(f"  Detection rate: {eval_result['detection_rate']:.1%}")
    if np.isfinite(eval_result["mean_loc_error"]):
        print(f"  Mean 3D loc error: {eval_result['mean_loc_error']:.1f} m")

    # Generate plots.
    print("\n  Generating demonstration plots...")

    # 3D trajectory plot.
    plot_3d_trajectory(
        drone_truth,
        estimated_positions=detection_output["track"]["positions"],
        mic_positions=mics,
        weapon_pos=np.array([300.0, 300.0, 0.0]),
        title="End-to-End Demo: 3D Trajectory",
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

    # Tracking display.
    plot_tracking_3d(
        track, drone_truth, drone_times,
        detection_output["fire_control"],
        weapon_pos=(300.0, 300.0, 0.0),
        output_path=str(out_dir / "demo_tracking_3d.png"),
    )

    # Save models.
    model_dir = Path("output/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_a.state_dict(), model_dir / "acoustic_classifier.pt")
    torch.save(model_b.state_dict(), model_dir / "fusion_classifier.pt")
    torch.save(model_c.state_dict(), model_dir / "kinematic_classifier.pt")
    torch.save(model_man.state_dict(), model_dir / "maneuver_classifier.pt")

    print(f"\n  All models saved to {model_dir}")

    # ─── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Acoustic-only accuracy:   {metrics_a['accuracy']:.3f}")
    print(f"  Fusion accuracy:          {metrics_b['accuracy']:.3f}")
    print(f"  Kinematic-only accuracy:  {metrics_c['accuracy']:.3f}")
    print(f"  Maneuver det. accuracy:   {metrics_man['accuracy']:.3f}")
    print(f"  Demo detection rate:      {eval_result['detection_rate']:.1%}")
    if np.isfinite(eval_result["mean_loc_error"]):
        print(f"  Demo 3D loc error:        {eval_result['mean_loc_error']:.1f} m")

    # Check if fusion outperforms acoustic-only.
    if metrics_b["accuracy"] >= metrics_a["accuracy"]:
        print("\n  ✓ Fusion classifier ≥ acoustic-only (kinematic features add value)")
    else:
        diff = metrics_a["accuracy"] - metrics_b["accuracy"]
        print(f"\n  ⚠ Fusion classifier {diff:.3f} below acoustic-only")
        print("    (kinematic features may not add value at this training scale)")

    print(f"\n  Output directory: {out_dir}")
    print("\n  *** MODULE 4 EVALUATION COMPLETE ***")


if __name__ == "__main__":
    run_full_evaluation()

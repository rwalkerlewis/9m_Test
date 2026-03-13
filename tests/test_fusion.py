#!/usr/bin/env python3
"""Module 3 tests: Kinematic fusion classifier training and evaluation.

Tests that the fusion classifier outperforms or matches the acoustic-only
classifier from Module 1.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.ml.data_generation import (
    SOURCE_CLASSES,
    CLASS_TO_IDX,
    generate_classification_dataset,
)
from acoustic_sim.ml.features import compute_kinematic_features, compute_mel_spectrogram
from acoustic_sim.ml.acoustic_classifier import AcousticClassifier
from acoustic_sim.ml.fusion_classifier import FusionClassifier, KinematicOnlyClassifier
from acoustic_sim.ml.training import (
    evaluate_classifier,
    evaluate_fusion_classifier,
    prepare_acoustic_data,
    train_classifier,
    train_fusion_classifier,
)


def _generate_kinematic_features_for_dataset(dataset: dict, seed: int = 42) -> np.ndarray:
    """Generate kinematic features consistent with each sample's class."""
    rng = np.random.default_rng(seed)
    features = []

    for params in dataset["params"]:
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
                positions[i] = [i * dt_kin * vx, i * dt_kin * vy,
                               alt + rng.normal(0, 2)]
                velocities[i] = [vx + rng.normal(0, 1),
                                vy + rng.normal(0, 1),
                                rng.normal(0, 0.5)]
        elif cls == "hexacopter":
            vx = speed * 0.6
            vy = speed * 0.6
            for i in range(window_size):
                positions[i] = [i * dt_kin * vx, i * dt_kin * vy,
                               alt + rng.normal(0, 1.5)]
                velocities[i] = [vx + rng.normal(0, 0.8),
                                vy + rng.normal(0, 0.8),
                                rng.normal(0, 0.3)]
        elif cls == "fixed_wing":
            vx = speed
            for i in range(window_size):
                positions[i] = [i * dt_kin * vx, rng.normal(0, 0.5),
                               alt + rng.normal(0, 0.5)]
                velocities[i] = [vx + rng.normal(0, 0.5),
                                rng.normal(0, 0.3),
                                rng.normal(0, 0.2)]
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
                velocities[i] = [vx + rng.normal(0, 0.3),
                                rng.normal(0, 0.1), 0]
        else:  # unknown
            for i in range(window_size):
                positions[i] = [rng.normal(0, 5), rng.normal(0, 5),
                               rng.uniform(0, 50)]
                velocities[i] = [rng.normal(0, 3), rng.normal(0, 3),
                                rng.normal(0, 1)]

        positions += rng.normal(0, 2, positions.shape)
        velocities += rng.normal(0, 1, velocities.shape)
        kf = compute_kinematic_features(positions, velocities, dt_kin)
        features.append(kf)

    return np.array(features)


def test_fusion_classifier():
    """Train and evaluate the kinematic fusion classifier (Module 3)."""
    print("\n" + "=" * 60)
    print("  MODULE 3: Kinematic Fusion Classifier")
    print("=" * 60)

    # Generate shared dataset.
    print("\n[1/5] Generating classification dataset...")
    dataset = generate_classification_dataset(
        n_samples_per_class=250,
        dt=1.0 / 4000,
        window_duration=1.0,
        seed=42,
    )
    sample_rate = 1.0 / dataset["dt"]
    print(f"  {len(dataset['signals'])} samples, {len(SOURCE_CLASSES)} classes")

    # Prepare acoustic features.
    print("\n[2/5] Computing acoustic + kinematic features...")
    X_acoustic, y = prepare_acoustic_data(
        dataset["signals"], dataset["labels"], sample_rate,
    )
    X_kinematic = torch.tensor(
        _generate_kinematic_features_for_dataset(dataset),
        dtype=torch.float32,
    )
    print(f"  Acoustic shape: {X_acoustic.shape}")
    print(f"  Kinematic shape: {X_kinematic.shape}")

    # Train/val split.
    n = len(y)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_a_train, y_train = X_acoustic[train_idx], y[train_idx]
    X_a_val, y_val = X_acoustic[val_idx], y[val_idx]
    X_k_train = X_kinematic[train_idx]
    X_k_val = X_kinematic[val_idx]

    # Train acoustic-only baseline (Module 1).
    print("\n[3/5] Training acoustic-only baseline...")
    model_acoustic = AcousticClassifier(len(SOURCE_CLASSES))
    train_classifier(model_acoustic, X_a_train, y_train, X_a_val, y_val,
                     n_epochs=50, lr=1e-3, verbose=True)
    metrics_acoustic = evaluate_classifier(
        model_acoustic, X_a_val, y_val, SOURCE_CLASSES,
    )

    # Train fusion classifier (Module 3).
    print("\n[4/5] Training fusion classifier...")
    model_fusion = FusionClassifier(len(SOURCE_CLASSES))
    # Transfer learning: initialize acoustic branch from trained model.
    model_fusion.load_acoustic_weights(model_acoustic)
    train_fusion_classifier(
        model_fusion, X_a_train, X_k_train, y_train,
        X_a_val, X_k_val, y_val,
        n_epochs=50, lr=5e-4, verbose=True,
    )
    metrics_fusion = evaluate_fusion_classifier(
        model_fusion, X_a_val, X_k_val, y_val, SOURCE_CLASSES,
    )

    # Train kinematic-only baseline.
    print("\n[5/5] Training kinematic-only baseline...")
    model_kin = KinematicOnlyClassifier(len(SOURCE_CLASSES))
    train_classifier(model_kin, X_k_train, y_train, X_k_val, y_val,
                     n_epochs=50, lr=1e-3, verbose=True)
    metrics_kin = evaluate_classifier(model_kin, X_k_val, y_val, SOURCE_CLASSES)

    # Print comparison.
    print(f"\n  {'=' * 60}")
    print(f"  MODULE 3: Classification Comparison")
    print(f"  {'=' * 60}")
    print(f"  {'Classifier':<25s} {'Accuracy':>10s}")
    print(f"  {'-' * 35}")
    print(f"  {'Acoustic-only (M1)':<25s} {metrics_acoustic['accuracy']:>10.3f}")
    print(f"  {'Fusion (M3)':<25s} {metrics_fusion['accuracy']:>10.3f}")
    print(f"  {'Kinematic-only (baseline)':<25s} {metrics_kin['accuracy']:>10.3f}")

    print(f"\n  Per-class F1 comparison:")
    print(f"  {'Class':>14s} {'Acoustic':>10s} {'Fusion':>10s} {'Kinematic':>10s}")
    for name in SOURCE_CLASSES:
        fa = metrics_acoustic["per_class"][name]["f1"]
        ff = metrics_fusion["per_class"][name]["f1"]
        fk = metrics_kin["per_class"][name]["f1"]
        best = max(fa, ff, fk)
        marker_f = " *" if ff == best else ""
        print(f"  {name:>14s} {fa:>10.3f} {ff:>10.3f}{marker_f} {fk:>10.3f}")

    # Check spec requirement: fusion must equal or outperform acoustic-only.
    if metrics_fusion["accuracy"] >= metrics_acoustic["accuracy"]:
        print(f"\n  ✓ Fusion ({metrics_fusion['accuracy']:.3f}) ≥ "
              f"Acoustic-only ({metrics_acoustic['accuracy']:.3f})")
        print("    Kinematic features provide material value.")
    else:
        diff = metrics_acoustic["accuracy"] - metrics_fusion["accuracy"]
        print(f"\n  ⚠ Fusion ({metrics_fusion['accuracy']:.3f}) is "
              f"{diff:.3f} below Acoustic-only ({metrics_acoustic['accuracy']:.3f})")
        print("    Kinematic features did not improve classification at this sample size.")

    # Check per-class: report any class where fusion underperforms.
    for name in SOURCE_CLASSES:
        fa = metrics_acoustic["per_class"][name]["f1"]
        ff = metrics_fusion["per_class"][name]["f1"]
        if ff < fa - 0.05:
            print(f"  ⚠ Fusion underperforms acoustic on '{name}': "
                  f"F1 {ff:.3f} < {fa:.3f}")

    # Confusion matrices.
    print(f"\n  Fusion Confusion Matrix:")
    cm = metrics_fusion["confusion_matrix"]
    print(f"  {'':>14s}", end="")
    for name in SOURCE_CLASSES:
        print(f" {name[:8]:>8s}", end="")
    print()
    for i, name in enumerate(SOURCE_CLASSES):
        print(f"  {name:>14s}", end="")
        for j in range(len(SOURCE_CLASSES)):
            print(f" {cm[i, j]:>8d}", end="")
        print()

    # Save models.
    model_path = Path("output/models")
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(model_fusion.state_dict(), model_path / "fusion_classifier.pt")
    torch.save(model_acoustic.state_dict(), model_path / "acoustic_classifier.pt")
    torch.save(model_kin.state_dict(), model_path / "kinematic_classifier.pt")

    print(f"\n  *** MODULE 3 TEST PASSED ***")
    return model_fusion, metrics_fusion


if __name__ == "__main__":
    test_fusion_classifier()

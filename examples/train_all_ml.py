#!/usr/bin/env python3
"""Train all ML classifiers and save checkpoints.

Generates synthetic training data and trains three models end-to-end:

1. AcousticClassifier — CNN on mel spectrograms (6 source classes)
2. ManeuverClassifier — 1D CNN on kinematic time series (6 maneuver classes)
3. FusionClassifier — Two-branch acoustic+kinematic network (6 source classes)

Checkpoints are saved to ``output/models/``.  This script must be
runnable standalone::

    python examples/train_all_ml.py
    python examples/train_all_ml.py --quick   # smaller dataset for testing
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.ml.acoustic_classifier import AcousticClassifier
from acoustic_sim.ml.maneuver_classifier import ManeuverClassifier
from acoustic_sim.ml.fusion_classifier import FusionClassifier
from acoustic_sim.ml.data_generation import (
    SOURCE_CLASSES,
    MANEUVER_CLASSES,
    generate_classification_dataset,
    generate_maneuver_dataset,
)
from acoustic_sim.ml.features import compute_kinematic_features
from acoustic_sim.ml.training import (
    evaluate_classifier,
    evaluate_fusion_classifier,
    prepare_acoustic_data,
    train_classifier,
    train_fusion_classifier,
)


def _generate_kinematic_features_for_dataset(
    dataset: dict, seed: int = 42,
) -> np.ndarray:
    """Generate kinematic features consistent with each sample's class.

    Adapted from tests/test_fusion.py to produce realistic kinematic
    signatures for each source class.
    """
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


def train_acoustic(n_samples: int, n_epochs: int, model_dir: Path) -> tuple:
    """Train the acoustic source classifier."""
    print("\n" + "=" * 60)
    print("  STEP 1: Acoustic Source Classifier")
    print("=" * 60)

    t0 = time.perf_counter()
    print(f"\n  Generating {n_samples * 6} training samples "
          f"({n_samples}/class) ...")
    dataset = generate_classification_dataset(
        n_samples_per_class=n_samples,
        dt=1.0 / 4000,
        window_duration=0.5,
        seed=42,
    )
    sample_rate = 1.0 / dataset["dt"]
    elapsed = time.perf_counter() - t0
    print(f"  Generated in {elapsed:.1f}s")

    print("  Computing mel spectrograms ...")
    X, y = prepare_acoustic_data(dataset["signals"], dataset["labels"],
                                  sample_rate)
    print(f"  Tensor shape: X={tuple(X.shape)}, y={tuple(y.shape)}")

    # 80/20 split.
    n = len(y)
    indices = np.random.default_rng(42).permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X[indices[:split]], y[indices[:split]]
    X_val, y_val = X[indices[split:]], y[indices[split:]]
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}")

    print(f"\n  Training for {n_epochs} epochs ...")
    model = AcousticClassifier(n_classes=len(SOURCE_CLASSES))
    history = train_classifier(model, X_train, y_train, X_val, y_val,
                                n_epochs=n_epochs, lr=1e-3, batch_size=32)

    metrics = evaluate_classifier(model, X_val, y_val, SOURCE_CLASSES)
    print(f"\n  Final validation accuracy: {metrics['accuracy']:.3f}")

    for name in SOURCE_CLASSES:
        m = metrics["per_class"][name]
        print(f"    {name:>15s}  P={m['precision']:.2f}  "
              f"R={m['recall']:.2f}  F1={m['f1']:.2f}")

    path = model_dir / "acoustic_classifier.pt"
    torch.save(model.state_dict(), path)
    print(f"\n  Saved: {path}")
    return model, dataset, X, y, sample_rate


def train_maneuver(n_samples: int, n_epochs: int, model_dir: Path):
    """Train the maneuver detection classifier."""
    print("\n" + "=" * 60)
    print("  STEP 2: Maneuver Detection Classifier")
    print("=" * 60)

    t0 = time.perf_counter()
    print(f"\n  Generating {n_samples * 6} training samples "
          f"({n_samples}/class) ...")
    dataset = generate_maneuver_dataset(
        n_samples_per_class=n_samples,
        window_size=20,
        dt_tracker=0.1,
        seed=42,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Generated in {elapsed:.1f}s")

    features = dataset["features"]  # (N, 20, 6)
    labels = dataset["labels"]
    print(f"  Feature shape: {features.shape}")

    # Reshape for Conv1d: (N, 6, 20).
    X = torch.tensor(features.transpose(0, 2, 1), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # 80/20 split.
    n = len(y)
    indices = np.random.default_rng(42).permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X[indices[:split]], y[indices[:split]]
    X_val, y_val = X[indices[split:]], y[indices[split:]]
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}")

    print(f"\n  Training for {n_epochs} epochs ...")
    model = ManeuverClassifier(n_classes=len(MANEUVER_CLASSES))
    history = train_classifier(model, X_train, y_train, X_val, y_val,
                                n_epochs=n_epochs, lr=1e-3, batch_size=32)

    metrics = evaluate_classifier(model, X_val, y_val, MANEUVER_CLASSES)
    print(f"\n  Final validation accuracy: {metrics['accuracy']:.3f}")

    for name in MANEUVER_CLASSES:
        m = metrics["per_class"][name]
        print(f"    {name:>15s}  P={m['precision']:.2f}  "
              f"R={m['recall']:.2f}  F1={m['f1']:.2f}")

    path = model_dir / "maneuver_classifier.pt"
    torch.save(model.state_dict(), path)
    print(f"\n  Saved: {path}")
    return model


def train_fusion(acoustic_model, dataset, X_acoustic, y, sample_rate,
                 n_epochs: int, model_dir: Path):
    """Train the fusion classifier with transfer learning."""
    print("\n" + "=" * 60)
    print("  STEP 3: Fusion Classifier (Transfer Learning)")
    print("=" * 60)

    print("\n  Computing kinematic features ...")
    X_kinematic = torch.tensor(
        _generate_kinematic_features_for_dataset(dataset),
        dtype=torch.float32,
    )
    print(f"  Acoustic shape: {tuple(X_acoustic.shape)}")
    print(f"  Kinematic shape: {tuple(X_kinematic.shape)}")

    # Same split as acoustic.
    n = len(y)
    indices = np.random.default_rng(42).permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_a_train, y_train = X_acoustic[train_idx], y[train_idx]
    X_a_val, y_val = X_acoustic[val_idx], y[val_idx]
    X_k_train = X_kinematic[train_idx]
    X_k_val = X_kinematic[val_idx]

    print(f"  Train: {len(y_train)}, Val: {len(y_val)}")

    model = FusionClassifier(n_classes=len(SOURCE_CLASSES))
    # Transfer learning: initialize acoustic branch from trained model.
    model.load_acoustic_weights(acoustic_model)
    print("  Loaded acoustic branch weights from acoustic classifier")

    print(f"\n  Training for {n_epochs} epochs ...")
    history = train_fusion_classifier(
        model, X_a_train, X_k_train, y_train,
        X_a_val, X_k_val, y_val,
        n_epochs=n_epochs, lr=5e-4, batch_size=32,
    )

    metrics = evaluate_fusion_classifier(
        model, X_a_val, X_k_val, y_val, SOURCE_CLASSES,
    )
    print(f"\n  Final validation accuracy: {metrics['accuracy']:.3f}")

    for name in SOURCE_CLASSES:
        m = metrics["per_class"][name]
        print(f"    {name:>15s}  P={m['precision']:.2f}  "
              f"R={m['recall']:.2f}  F1={m['f1']:.2f}")

    path = model_dir / "fusion_classifier.pt"
    torch.save(model.state_dict(), path)
    print(f"\n  Saved: {path}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller datasets for quick testing")
    parser.add_argument("--n-class-samples", type=int, default=None,
                        help="Override samples per class for classification")
    parser.add_argument("--n-maneuver-samples", type=int, default=None,
                        help="Override samples per class for maneuver")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/models"),
                        help="Model checkpoint directory")
    args = parser.parse_args()

    if args.quick:
        n_class = args.n_class_samples or 100
        n_maneuver = args.n_maneuver_samples or 200
        n_epochs = args.n_epochs or 30
    else:
        n_class = args.n_class_samples or 500
        n_maneuver = args.n_maneuver_samples or 800
        n_epochs = args.n_epochs or 100

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ML CLASSIFIER TRAINING SUITE")
    print("=" * 60)
    print(f"  Classification: {n_class} samples/class, {n_epochs} epochs")
    print(f"  Maneuver: {n_maneuver} samples/class, {n_epochs} epochs")
    print(f"  Output: {args.output_dir}")

    t_total = time.perf_counter()

    # Step 1: Acoustic classifier.
    acoustic_model, dataset, X_acoustic, y, sr = train_acoustic(
        n_class, n_epochs, args.output_dir,
    )

    # Step 2: Maneuver classifier.
    train_maneuver(n_maneuver, n_epochs, args.output_dir)

    # Step 3: Fusion classifier (depends on step 1).
    train_fusion(acoustic_model, dataset, X_acoustic, y, sr,
                 n_epochs, args.output_dir)

    elapsed = time.perf_counter() - t_total
    print("\n" + "=" * 60)
    print(f"  ALL TRAINING COMPLETE — {elapsed:.1f}s total")
    print(f"  Checkpoints in: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

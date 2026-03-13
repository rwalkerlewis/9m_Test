#!/usr/bin/env python3
"""Module 1 tests: Source classification training and evaluation."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.ml.data_generation import (
    SOURCE_CLASSES,
    generate_classification_dataset,
)
from acoustic_sim.ml.acoustic_classifier import AcousticClassifier
from acoustic_sim.ml.training import (
    evaluate_classifier,
    prepare_acoustic_data,
    train_classifier,
)


def test_acoustic_classifier():
    """Train and evaluate the acoustic-only source classifier."""
    print("\n" + "=" * 60)
    print("  MODULE 1: Source Classification")
    print("=" * 60)

    # Generate training data.
    print("\n[1/3] Generating training data...")
    dataset = generate_classification_dataset(
        n_samples_per_class=300,
        dt=1.0 / 4000,
        window_duration=1.0,
        seed=42,
    )
    sample_rate = 1.0 / dataset["dt"]
    print(f"  {len(dataset['signals'])} samples, {len(SOURCE_CLASSES)} classes")
    print(f"  Sample rate: {sample_rate:.0f} Hz")

    # Prepare mel spectrograms.
    print("\n[2/3] Computing mel spectrograms...")
    X, y = prepare_acoustic_data(
        dataset["signals"], dataset["labels"], sample_rate,
    )
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    # Train/val split (80/20).
    n = len(y)
    indices = np.random.default_rng(42).permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}")

    # Train.
    print("\n[3/3] Training acoustic classifier...")
    model = AcousticClassifier(n_classes=len(SOURCE_CLASSES))
    history = train_classifier(
        model, X_train, y_train, X_val, y_val,
        n_epochs=50, lr=1e-3, batch_size=32,
    )

    # Evaluate.
    metrics = evaluate_classifier(model, X_val, y_val, SOURCE_CLASSES)

    print(f"\n  Validation accuracy: {metrics['accuracy']:.3f}")
    print(f"\n  Confusion matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  {'':>14s}", end="")
    for name in SOURCE_CLASSES:
        print(f" {name[:8]:>8s}", end="")
    print()
    for i, name in enumerate(SOURCE_CLASSES):
        print(f"  {name:>14s}", end="")
        for j in range(len(SOURCE_CLASSES)):
            print(f" {cm[i, j]:>8d}", end="")
        print()

    print(f"\n  Per-class metrics:")
    print(f"  {'Class':>14s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    for name in SOURCE_CLASSES:
        m = metrics["per_class"][name]
        print(f"  {name:>14s} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")

    # Save model.
    model_path = Path("output/models")
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path / "acoustic_classifier.pt")
    print(f"\n  Model saved to {model_path / 'acoustic_classifier.pt'}")

    print("\n  *** MODULE 1 TEST PASSED ***")
    return model, metrics, dataset


if __name__ == "__main__":
    test_acoustic_classifier()

#!/usr/bin/env python3
"""Module 2 tests: Maneuver detection training and evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # optional dependency: pip install -e ".[ml]"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.ml.data_generation import (
    MANEUVER_CLASSES,
    generate_maneuver_dataset,
)
from acoustic_sim.ml.maneuver_classifier import ManeuverClassifier
from acoustic_sim.ml.training import evaluate_classifier, train_classifier


def test_maneuver_classifier(tmp_path):
    """Train and evaluate the maneuver detection classifier."""
    print("\n" + "=" * 60)
    print("  MODULE 2: Maneuver Detection")
    print("=" * 60)

    # Generate training data.
    print("\n[1/3] Generating maneuver training data...")
    dataset = generate_maneuver_dataset(
        n_samples_per_class=400,
        window_size=20,
        dt_tracker=0.1,
        seed=42,
    )
    features = dataset["features"]  # (N, 20, 6)
    labels = dataset["labels"]
    print(f"  {len(labels)} samples, {len(MANEUVER_CLASSES)} classes")
    print(f"  Feature shape: {features.shape}")

    # Reshape for Conv1d: (N, 6, 20).
    X = torch.tensor(features.transpose(0, 2, 1), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Train/val split.
    n = len(y)
    indices = np.random.default_rng(42).permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X[indices[:split]], y[indices[:split]]
    X_val, y_val = X[indices[split:]], y[indices[split:]]
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}")

    # Train.
    print("\n[2/3] Training maneuver classifier...")
    model = ManeuverClassifier(n_classes=len(MANEUVER_CLASSES))
    history = train_classifier(
        model, X_train, y_train, X_val, y_val,
        n_epochs=50, lr=1e-3, batch_size=32,
    )

    # Evaluate.
    print("\n[3/3] Evaluating...")
    metrics = evaluate_classifier(model, X_val, y_val, MANEUVER_CLASSES)

    print(f"\n  Validation accuracy: {metrics['accuracy']:.3f}")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':>14s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    for name in MANEUVER_CLASSES:
        m = metrics["per_class"][name]
        print(f"  {name:>14s} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")

    # Save and reload the weights (to a temporary directory: output/models holds the
    # committed, trained weights and must not be overwritten by tests).
    checkpoint = tmp_path / "maneuver_classifier.pt"
    torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint))

    print("\n  *** MODULE 2 TEST PASSED ***")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_maneuver_classifier(Path(tmp))

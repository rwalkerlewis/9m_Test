"""Training loops for all ML classifiers.

Provides train/evaluate functions for:
- AcousticClassifier (Module 1)
- ManeuverClassifier (Module 2)
- FusionClassifier (Module 3)
- KinematicOnlyClassifier (Module 4 baseline)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from acoustic_sim.ml.features import compute_mel_spectrogram


def prepare_acoustic_data(
    signals: list[np.ndarray],
    labels: list[int],
    sample_rate: float,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw signals to mel spectrogram tensors.

    Returns (X, y) where X is (N, 1, n_mels, n_time) and y is (N,).
    """
    specs = []
    max_time = 0
    for sig in signals:
        mel = compute_mel_spectrogram(sig, sample_rate, n_fft, hop_length, n_mels)
        specs.append(mel)
        max_time = max(max_time, mel.shape[1])

    # Pad all to same length.
    X = np.zeros((len(specs), 1, n_mels, max_time), dtype=np.float32)
    for i, mel in enumerate(specs):
        X[i, 0, :, :mel.shape[1]] = mel

    return torch.tensor(X), torch.tensor(labels, dtype=torch.long)


def train_classifier(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    verbose: bool = True,
) -> dict:
    """Generic training loop for single-input classifiers.

    Works for AcousticClassifier, ManeuverClassifier, KinematicOnlyClassifier.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # Validation.
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_loss = criterion(val_logits, y_val.to(device)).item()
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = float(np.mean(val_preds == y_val.numpy()))

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

    return history


def train_fusion_classifier(
    model: nn.Module,
    X_acoustic_train: torch.Tensor,
    X_kinematic_train: torch.Tensor,
    y_train: torch.Tensor,
    X_acoustic_val: torch.Tensor,
    X_kinematic_val: torch.Tensor,
    y_val: torch.Tensor,
    n_epochs: int = 50,
    lr: float = 5e-4,
    batch_size: int = 32,
    verbose: bool = True,
) -> dict:
    """Training loop for the two-input FusionClassifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_acoustic_train, X_kinematic_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for X_a, X_k, y_batch in train_loader:
            X_a = X_a.to(device)
            X_k = X_k.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_a, X_k)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(
                X_acoustic_val.to(device),
                X_kinematic_val.to(device),
            )
            val_loss = criterion(val_logits, y_val.to(device)).item()
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = float(np.mean(val_preds == y_val.numpy()))

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

    return history


def evaluate_classifier(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    class_names: list[str],
) -> dict:
    """Evaluate a single-input classifier and return metrics.

    Returns confusion matrix, per-class metrics, overall accuracy.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    y_true = y_test.numpy()

    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, preds):
        confusion[t, p] += 1

    accuracy = float(np.mean(preds == y_true))

    per_class = {}
    for i, name in enumerate(class_names):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "confusion_matrix": confusion,
        "accuracy": accuracy,
        "per_class": per_class,
        "predictions": preds,
        "probabilities": probs,
    }


def evaluate_fusion_classifier(
    model: nn.Module,
    X_acoustic: torch.Tensor,
    X_kinematic: torch.Tensor,
    y_test: torch.Tensor,
    class_names: list[str],
) -> dict:
    """Evaluate the fusion classifier."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(X_acoustic.to(device), X_kinematic.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    y_true = y_test.numpy()

    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, preds):
        confusion[t, p] += 1

    accuracy = float(np.mean(preds == y_true))

    per_class = {}
    for i, name in enumerate(class_names):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "confusion_matrix": confusion,
        "accuracy": accuracy,
        "per_class": per_class,
        "predictions": preds,
        "probabilities": probs,
    }

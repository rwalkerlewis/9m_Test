#!/usr/bin/env python3
"""Train, evaluate, and visualise the acoustic source classifier.

Generates synthetic training data from the 3D analytical forward model,
trains the CNN acoustic classifier on mel spectrograms, evaluates on a
held-out test set, and produces seven demonstration plots covering
spectral signatures, training dynamics, confusion, per-class metrics,
SNR robustness, ROC analysis, and detection-gate integration.

Fully self-contained — no external data or pre-trained weights required.

Usage::

    # Default settings (300 samples/class, 80 epochs)
    python examples/run_ml_demo.py

    # Quick smoke test
    python examples/run_ml_demo.py --n-samples 50 --n-epochs 5

    # Custom output directory
    python examples/run_ml_demo.py --output-dir output/my_run
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print(
        "ERROR: PyTorch is required but not installed.\n"
        "Install it with:  pip install torch --index-url "
        "https://download.pytorch.org/whl/cpu"
    )
    sys.exit(1)

from acoustic_sim.forward import simulate_3d_traces
from acoustic_sim.ml.acoustic_classifier import AcousticClassifier
from acoustic_sim.ml.data_generation import (
    CLASS_TO_IDX,
    SOURCE_CLASSES,
    generate_classification_dataset,
    generate_source_signal,
)
from acoustic_sim.ml.features import compute_mel_spectrogram
from acoustic_sim.ml.training import (
    evaluate_classifier,
    prepare_acoustic_data,
    train_classifier,
)
from acoustic_sim.receivers import create_receiver_l_shaped_3d
from acoustic_sim.sources import MovingSource3D


# ====================================================================
# Step 1 — Generate synthetic training data
# ====================================================================

def generate_data(n_samples_per_class: int) -> dict:
    """Generate classification dataset and split into train/val/test."""
    print("[STEP 1] Generating synthetic training data "
          f"({n_samples_per_class} samples/class, "
          f"{n_samples_per_class * len(SOURCE_CLASSES)} total) ...")
    t0 = time.perf_counter()

    dataset = generate_classification_dataset(
        n_samples_per_class=n_samples_per_class,
        dt=1.0 / 4000,
        window_duration=0.5,
        seed=42,
    )

    signals = dataset["signals"]
    labels = dataset["labels"]
    snr_dbs = dataset["snr_dbs"]
    dt = dataset["dt"]
    sample_rate = 1.0 / dt
    n = len(labels)

    # 70 / 15 / 15 split.
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    split_train = int(0.70 * n)
    split_val = int(0.85 * n)

    train_idx = indices[:split_train]
    val_idx = indices[split_train:split_val]
    test_idx = indices[split_val:]

    elapsed = time.perf_counter() - t0
    print(f"         Generated {n} samples in {elapsed:.1f}s")
    print(f"         Split: train={len(train_idx)}, "
          f"val={len(val_idx)}, test={len(test_idx)}")

    return {
        "signals": signals,
        "labels": labels,
        "snr_dbs": snr_dbs,
        "sample_rate": sample_rate,
        "dt": dt,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


# ====================================================================
# Step 2 — Mel spectrogram examples
# ====================================================================

def plot_mel_examples(
    signals: list[np.ndarray],
    labels: list[int],
    sample_rate: float,
    output_dir: Path,
) -> None:
    """Plot one representative mel spectrogram per source class."""
    print("[STEP 2] Plotting mel spectrogram examples ...")

    specs = []
    for cls_idx, cls_name in enumerate(SOURCE_CLASSES):
        # Find the first sample of this class.
        for i, lab in enumerate(labels):
            if lab == cls_idx:
                mel = compute_mel_spectrogram(signals[i], sample_rate)
                specs.append((cls_name, mel))
                break

    vmin = min(s.min() for _, s in specs)
    vmax = max(s.max() for _, s in specs)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.ravel()
    im = None
    for ax, (cls_name, mel) in zip(axes, specs):
        im = ax.imshow(
            mel, aspect="auto", origin="lower", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(cls_name)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Mel band")

    fig.colorbar(im, ax=axes.tolist(), label="Log power")
    fig.suptitle("Mel Spectrogram Examples by Source Class", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "mel_spectrogram_examples.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"         Saved {path}")


# ====================================================================
# Step 3 — Train the acoustic classifier
# ====================================================================

def train_model(
    signals: list[np.ndarray],
    labels: list[int],
    sample_rate: float,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    n_epochs: int,
) -> tuple:
    """Prepare data tensors and train the CNN classifier.

    Returns (model, history, X_test, y_test).
    """
    print("[STEP 3] Preparing mel spectrogram tensors ...")
    X, y = prepare_acoustic_data(signals, labels, sample_rate)
    print(f"         Tensor shape: X={tuple(X.shape)}, y={tuple(y.shape)}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = AcousticClassifier(n_classes=len(SOURCE_CLASSES))
    print(f"         Training for {n_epochs} epochs "
          f"(lr=1e-3, batch_size=32) ...")
    history = train_classifier(
        model, X_train, y_train, X_val, y_val,
        n_epochs=n_epochs, lr=1e-3, batch_size=32,
    )
    final_acc = history["val_acc"][-1]
    print(f"         Final val accuracy: {final_acc:.3f}")
    return model, history, X_test, y_test


# ====================================================================
# Step 4 — Training curves plot
# ====================================================================

def plot_training_curves(history: dict, output_dir: Path) -> None:
    """Plot train/val loss and val accuracy vs epoch."""
    print("[STEP 4] Plotting training curves ...")

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_tl = "#1f77b4"
    color_vl = "#ff7f0e"
    color_va = "#2ca02c"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ln1 = ax1.plot(epochs, history["train_loss"], color=color_tl,
                   label="Train loss")
    ln2 = ax1.plot(epochs, history["val_loss"], color=color_vl,
                   label="Val loss")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ln3 = ax2.plot(epochs, history["val_acc"], color=color_va,
                   linestyle="--", label="Val accuracy")
    ax2.set_ylim(0, 1.05)

    lns = ln1 + ln2 + ln3
    labs = [ln.get_label() for ln in lns]
    ax1.legend(lns, labs, loc="center right")

    ax1.set_title("Acoustic Classifier Training")
    fig.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"         Saved {path}")


# ====================================================================
# Step 5 — Evaluate on test set
# ====================================================================

def evaluate_model(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Evaluate the trained model on the held-out test set."""
    print("[STEP 5] Evaluating on test set ...")
    metrics = evaluate_classifier(model, X_test, y_test, SOURCE_CLASSES)
    print(f"         Test accuracy: {metrics['accuracy']:.3f}")
    for name in SOURCE_CLASSES:
        m = metrics["per_class"][name]
        print(f"           {name:>15s}  "
              f"P={m['precision']:.2f}  R={m['recall']:.2f}  "
              f"F1={m['f1']:.2f}")
    return metrics


# ====================================================================
# Step 6 — Confusion matrix plot
# ====================================================================

def plot_confusion_matrix(metrics: dict, output_dir: Path) -> None:
    """Plot row-normalised confusion matrix."""
    print("[STEP 6] Plotting confusion matrix ...")

    cm = metrics["confusion_matrix"].astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    n_cls = len(SOURCE_CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(n_cls):
        for j in range(n_cls):
            val = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(SOURCE_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(SOURCE_CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(
        f"Confusion Matrix (accuracy={metrics['accuracy']:.1%})"
    )
    fig.colorbar(im, ax=ax, label="Fraction")
    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"         Saved {path}")


# ====================================================================
# Step 7 — Per-class metrics bar chart
# ====================================================================

def plot_per_class_metrics(metrics: dict, output_dir: Path) -> None:
    """Grouped bar chart of precision, recall, F1 per class."""
    print("[STEP 7] Plotting per-class metrics ...")

    classes = SOURCE_CLASSES
    precision = [metrics["per_class"][c]["precision"] for c in classes]
    recall = [metrics["per_class"][c]["recall"] for c in classes]
    f1 = [metrics["per_class"][c]["f1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#1f77b4")
    ax.bar(x, recall, width, label="Recall", color="#ff7f0e")
    ax.bar(x + width, f1, width, label="F1", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title("Per-Class Classification Metrics")
    fig.tight_layout()
    path = output_dir / "per_class_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"         Saved {path}")


# ====================================================================
# Step 8 — Accuracy vs SNR
# ====================================================================

def plot_accuracy_vs_snr(
    metrics: dict,
    test_snr_dbs: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot classification accuracy as a function of SNR."""
    print("[STEP 8] Plotting accuracy vs SNR ...")

    preds = metrics["predictions"]
    correct = (preds == y_test).astype(np.float64)

    bin_edges = [(-5, 0), (0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
    centers = []
    accs = []
    errs = []

    for lo, hi in bin_edges:
        if hi == 30:
            mask = (test_snr_dbs >= lo) & (test_snr_dbs <= hi)
        else:
            mask = (test_snr_dbs >= lo) & (test_snr_dbs < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        acc = float(correct[mask].mean())
        se = math.sqrt(acc * (1 - acc) / max(n_bin, 1))
        centers.append((lo + hi) / 2.0)
        accs.append(acc)
        errs.append(se)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(centers, accs, yerr=errs, fmt="o-", capsize=4,
                linewidth=2, markersize=6, color="#1f77b4")
    ax.axhline(1.0 / len(SOURCE_CLASSES), color="gray", linestyle="--",
               label="Chance (1/6)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Classification Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification Accuracy vs SNR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "accuracy_vs_snr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"         Saved {path}")


# ====================================================================
# Step 9 — ROC curves
# ====================================================================

def plot_roc_curves(
    metrics: dict,
    y_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot one-vs-rest ROC curves for each class."""
    print("[STEP 9] Plotting ROC curves ...")

    probs = metrics["probabilities"]  # (N, n_classes)
    n_cls = len(SOURCE_CLASSES)

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_cls))

    for cls_idx in range(n_cls):
        y_bin = (y_test == cls_idx).astype(int)
        scores = probs[:, cls_idx]

        # Sort by descending score.
        order = np.argsort(-scores)
        y_sorted = y_bin[order]

        tps = np.cumsum(y_sorted)
        fps = np.cumsum(1 - y_sorted)
        n_pos = int(y_bin.sum())
        n_neg = len(y_bin) - n_pos

        tpr = tps / max(n_pos, 1)
        fpr = fps / max(n_neg, 1)

        # Prepend origin.
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        auc = float(np.trapezoid(tpr, fpr))
        ax.plot(fpr, tpr, color=colors[cls_idx],
                label=f"{SOURCE_CLASSES[cls_idx]} (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"         Saved {path}")


# ====================================================================
# Step 10 — Detection gate integration demo
# ====================================================================

def plot_detection_gate(
    model: torch.nn.Module,
    sample_rate: float,
    dt: float,
    output_dir: Path,
) -> None:
    """Simulate a flyover and show RMS gate + classifier confidence."""
    print("[STEP 10] Running detection gate integration demo ...")

    n_windows = 50
    window_duration = 0.5
    n_steps = int(window_duration / dt)
    sound_speed = 343.0

    mics = create_receiver_l_shaped_3d(
        5, 5, spacing=0.3, origin_x=0.0, origin_y=0.0, z=0.0,
    )

    rng = np.random.default_rng(123)
    device = next(model.parameters()).device
    model.eval()

    rms_values = np.zeros(n_windows)
    max_probs = np.zeros(n_windows)
    pred_classes = np.full(n_windows, -1, dtype=int)
    ground_truth = np.zeros(n_windows)

    noise_std = 1e-4

    for w in range(n_windows):
        if 15 <= w < 35:
            # Active window: quadcopter at decreasing range.
            frac = (w - 15) / 19.0  # 0 → 1 over 20 windows
            horiz_dist = 400.0 - frac * 350.0  # 400m → 50m
            bearing = 0.3
            altitude = 50.0
            sx = horiz_dist * math.cos(bearing)
            sy = horiz_dist * math.sin(bearing)

            sig, params = generate_source_signal(
                "quadcopter", n_steps, dt, rng,
            )
            src = MovingSource3D(
                x0=sx, y0=sy, z0=altitude,
                x1=sx + 5.0 * window_duration * math.cos(bearing),
                y1=sy + 5.0 * window_duration * math.sin(bearing),
                z1=altitude,
                speed=5.0,
                signal=sig,
            )
            traces = simulate_3d_traces(
                src, mics, dt, n_steps, sound_speed,
                air_absorption=0.005,
            )
            noise = rng.standard_normal(traces.shape) * noise_std
            noisy_traces = traces + noise
            beamformed = np.mean(noisy_traces, axis=0)
            ground_truth[w] = 1.0
        else:
            # Noise-only window.
            beamformed = rng.standard_normal(n_steps) * noise_std

        rms_values[w] = float(np.sqrt(np.mean(beamformed ** 2)))

        # Classify.
        mel = compute_mel_spectrogram(beamformed, sample_rate)
        mel_tensor = torch.tensor(
            mel[np.newaxis, np.newaxis, :, :], dtype=torch.float32,
        ).to(device)
        with torch.no_grad():
            logits = model(mel_tensor)
            prob = F.softmax(logits, dim=1).cpu().numpy()[0]
        max_probs[w] = float(prob.max())
        pred_classes[w] = int(prob.argmax())

    # Detection threshold: 3× median RMS of noise-only windows.
    noise_rms = np.concatenate([rms_values[:15], rms_values[35:]])
    rms_threshold = 3.0 * float(np.median(noise_rms))

    # --- Plot ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True,
    )
    windows = np.arange(n_windows)

    # Top: RMS energy.
    ax1.plot(windows, rms_values, "k-", linewidth=1.2)
    ax1.axhline(rms_threshold, color="red", linestyle="--",
                label="Detection threshold")
    ax1.set_ylabel("RMS Energy")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title("Detection Gate + Classifier Integration")

    # Middle: classifier confidence, colored by predicted class.
    colors_map = plt.cm.tab10(np.linspace(0, 1, len(SOURCE_CLASSES)))
    for w in range(n_windows):
        c = colors_map[pred_classes[w]] if pred_classes[w] >= 0 else "gray"
        ax2.bar(w, max_probs[w], width=0.8, color=c, edgecolor="none")
    ax2.set_ylabel("Max class probability")
    ax2.set_ylim(0, 1.05)
    # Legend for class colours.
    handles = []
    for ci, cname in enumerate(SOURCE_CLASSES):
        handles.append(plt.Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor=colors_map[ci], markersize=7, label=cname,
        ))
    ax2.legend(handles=handles, loc="upper left", fontsize=6,
               ncol=3)

    # Bottom: ground truth.
    ax3.fill_between(windows, ground_truth, step="mid",
                     alpha=0.4, color="#2ca02c")
    ax3.step(windows, ground_truth, where="mid", color="#2ca02c",
             linewidth=1.5)
    ax3.set_ylabel("Source present")
    ax3.set_xlabel("Window index")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Absent", "Present"])

    fig.tight_layout()
    path = output_dir / "detection_gate_demo.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"         Saved {path}")


# ====================================================================
# Step 11 — Save model weights
# ====================================================================

def save_model(model: torch.nn.Module, output_dir: Path) -> None:
    """Save trained model weights and print parameter count."""
    print("[STEP 11] Saving model weights ...")
    path = output_dir / "acoustic_classifier.pt"
    torch.save(model.state_dict(), path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"          Saved {path}")
    print(f"          Model parameters: {n_params:,}")


# ====================================================================
# Main
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate acoustic source classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/ml_demo"),
        help="Directory for plots and model weights "
             "(default: output/ml_demo)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=300,
        help="Samples per class for training data "
             "(default: 300, total = 6 × N)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=80,
        help="Training epochs (default: 80)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print("=" * 60)
    print("  ACOUSTIC CLASSIFIER DEMONSTRATION")
    print("=" * 60)
    print()

    # Step 1: generate data.
    data = generate_data(args.n_samples)

    # Step 2: mel spectrogram examples.
    plot_mel_examples(
        data["signals"], data["labels"],
        data["sample_rate"], args.output_dir,
    )

    # Step 3: train classifier.
    model, history, X_test, y_test = train_model(
        data["signals"], data["labels"], data["sample_rate"],
        data["train_idx"], data["val_idx"], data["test_idx"],
        args.n_epochs,
    )

    # Step 4: training curves.
    plot_training_curves(history, args.output_dir)

    # Step 5: evaluate on test set.
    metrics = evaluate_model(model, X_test, y_test)

    # Step 6: confusion matrix.
    plot_confusion_matrix(metrics, args.output_dir)

    # Step 7: per-class metrics.
    plot_per_class_metrics(metrics, args.output_dir)

    # Step 8: accuracy vs SNR.
    test_snr_dbs = np.array(
        [data["snr_dbs"][i] for i in data["test_idx"]]
    )
    plot_accuracy_vs_snr(
        metrics, test_snr_dbs, y_test.numpy(), args.output_dir,
    )

    # Step 9: ROC curves.
    plot_roc_curves(metrics, y_test.numpy(), args.output_dir)

    # Step 10: detection gate demo.
    plot_detection_gate(
        model, data["sample_rate"], data["dt"], args.output_dir,
    )

    # Step 11: save model.
    save_model(model, args.output_dir)

    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 60)
    print(f"  DONE — total runtime {elapsed:.1f}s")
    print(f"  Output: {args.output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

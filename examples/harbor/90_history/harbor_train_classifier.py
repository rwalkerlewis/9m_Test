#!/usr/bin/env python3
"""Harbor Sub-Task 4: train the CNN acoustic classifier on the classifier dataset.

Loads ``classifier_dataset.npz``, performs a seed-stratified train/test split
(seeds 0–5 = train, seeds 6–7 = test), trains ``AcousticClassifier(n_classes=3)``
with cross-entropy loss, evaluates per-class accuracy, and saves a checkpoint.

Usage::

    python examples/harbor/90_history/harbor_train_classifier.py \
        [--dataset output/harbor/classifier_dataset.npz] \
        [--out output/harbor] \
        [--epochs 30] \
        [--batch 64] \
        [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from acoustic_sim.ml.acoustic_classifier import AcousticClassifier  # noqa: E402

LABEL_NAMES = ["ship", "uuv", "diver"]
N_CLASSES = 3

# Seeds 0-5 → train; seeds 6-7 → test.
TRAIN_SEEDS = set(range(6))
TEST_SEEDS = {6, 7}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="output/harbor/classifier_dataset.npz")
    ap.add_argument("--out", default="output/harbor")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset.
    # ------------------------------------------------------------------
    with np.load(args.dataset, allow_pickle=False) as d:
        X_all = d["X"].astype(np.float32)        # (N, 1, n_mels, T)
        y_all = d["y"].astype(np.int64)           # (N,)
        seeds_all = d["seed"].astype(np.int32)    # (N,)
        label_names = list(d["label_names"])

    print(f"Loaded {len(y_all)} examples, shape {X_all.shape}")
    print("Label names from file:", label_names)

    # ------------------------------------------------------------------
    # Seed-stratified split: whole seeds in each partition.
    # ------------------------------------------------------------------
    train_mask = np.isin(seeds_all, sorted(TRAIN_SEEDS))
    test_mask = np.isin(seeds_all, sorted(TEST_SEEDS))

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    print(f"Train: {len(y_train)} examples (seeds {sorted(TRAIN_SEEDS)})")
    print(f"Test:  {len(y_test)} examples  (seeds {sorted(TEST_SEEDS)})")

    # ------------------------------------------------------------------
    # Build DataLoader.
    # ------------------------------------------------------------------
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    # ------------------------------------------------------------------
    # Model, optimiser, loss.
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AcousticClassifier(n_classes=N_CLASSES).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Training loop.
    # ------------------------------------------------------------------
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={epoch_loss / max(n_batches, 1):.4f}")

    # ------------------------------------------------------------------
    # Evaluation on test set (batched to avoid OOM on GPU).
    # ------------------------------------------------------------------
    model.eval()
    preds_list = []
    eval_dl = DataLoader(TensorDataset(torch.from_numpy(X_test)), batch_size=args.batch, shuffle=False)
    with torch.no_grad():
        for (X_b,) in eval_dl:
            preds_list.append(model(X_b.to(device)).argmax(dim=1).cpu())
    preds = torch.cat(preds_list).numpy()

    overall_acc = float(np.mean(preds == y_test))
    print(f"\nOverall test accuracy: {overall_acc * 100:.1f}%")
    for cls_idx, cls_name in enumerate(LABEL_NAMES):
        mask = y_test == cls_idx
        if mask.sum() == 0:
            print(f"  {cls_name}: no test examples")
        else:
            acc = float(np.mean(preds[mask] == cls_idx))
            print(f"  {cls_name}: {acc * 100:.1f}%  ({mask.sum()} examples)")

    # ------------------------------------------------------------------
    # Save checkpoint.
    # ------------------------------------------------------------------
    ckpt_path = out / "cnn_classifier.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_names": LABEL_NAMES,
            "n_classes": N_CLASSES,
        },
        ckpt_path,
    )
    print(f"\nCheckpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Harbor Sub-Task 5: train the CVAE anomaly detector on ship-only mel-spectrograms.

Loads ``classifier_dataset.npz``, filters to ship-only rows (label 0), trains
``ConvVAE`` using the standard VAE training loop, and saves the checkpoint to
``output/harbor/cvae_anomaly.pt``.

After training, evaluates the anomaly score on ship vs. non-ship examples and
prints mean scores per class so the ship/non-ship separation can be assessed.

Usage::

    python examples/harbor/90_history/harbor_train_anomaly.py \\
        [--dataset output/harbor/classifier_dataset.npz] \\
        [--out output/harbor] \\
        [--epochs 50] \\
        [--batch 64] \\
        [--lr 1e-3] \\
        [--latent-dim 32]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from acoustic_sim.ml.anomaly_detector import ConvVAE  # noqa: E402
from acoustic_sim.ml.anomaly_training import train_anomaly_detector  # noqa: E402

# Label index used for the "ship" class in classifier_dataset.npz.
SHIP_LABEL = 0


def _evaluate_scores(
    model: ConvVAE,
    X: torch.Tensor,
    y: np.ndarray,
    label_names: list[str],
    device: torch.device,
) -> None:
    """Print mean anomaly score per class on the given dataset split."""
    model.eval()
    scores: list[float] = []
    for i in range(X.size(0)):
        x_i = X[i : i + 1].to(device)
        score, _ = model.compute_anomaly_score(x_i)
        scores.append(score)
    scores_arr = np.array(scores)
    print("\nAnomaly score evaluation (mean ± std per class):")
    for cls_idx, name in enumerate(label_names):
        mask = y == cls_idx
        if mask.sum() == 0:
            print(f"  {name}: no examples")
            continue
        cls_scores = scores_arr[mask]
        print(f"  {name:12s}: mean={cls_scores.mean():.4f}  std={cls_scores.std():.4f}"
              f"  n={mask.sum()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="output/harbor/classifier_dataset.npz",
                    help="Path to classifier_dataset.npz")
    ap.add_argument("--out", default="output/harbor",
                    help="Output directory for the CVAE checkpoint")
    ap.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs (default: 50)")
    ap.add_argument("--batch", type=int, default=64,
                    help="Batch size (default: 64)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate (default: 1e-3)")
    ap.add_argument("--latent-dim", type=int, default=32,
                    help="CVAE latent dimensionality (default: 32)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset.
    # ------------------------------------------------------------------
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}", file=sys.stderr)
        print("Run harbor_build_classifier_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    with np.load(str(dataset_path), allow_pickle=False) as d:
        X_all = d["X"].astype(np.float32)     # (N, 1, n_mels, T)
        y_all = d["y"].astype(np.int64)        # (N,)
        label_names = list(d["label_names"])

    print(f"Loaded {len(y_all)} examples, shape {X_all.shape}")
    print(f"Label names: {label_names}")

    # ------------------------------------------------------------------
    # Filter to ship-only rows.
    # ------------------------------------------------------------------
    ship_mask = y_all == SHIP_LABEL
    X_ship = X_all[ship_mask]
    print(f"Ship examples (label={SHIP_LABEL}): {len(X_ship)} "
          f"of {len(y_all)} total")

    if len(X_ship) < 2:
        print("ERROR: not enough ship examples to train.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Train/val split (80/20) on ship examples.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    n = len(X_ship)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    X_train = torch.from_numpy(X_ship[idx[:split]])
    X_val = torch.from_numpy(X_ship[idx[split:]])
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    n_mels = X_ship.shape[2]
    input_dim = int(np.prod(X_ship.shape[1:]))  # 1 * n_mels * T

    # ------------------------------------------------------------------
    # Build and train the CVAE.
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Latent dim: {args.latent_dim}  Epochs: {args.epochs}  "
          f"Batch: {args.batch}  LR: {args.lr}")

    model = ConvVAE(latent_dim=args.latent_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvVAE parameters: {n_params:,}")

    history = train_anomaly_detector(
        model, X_train, X_val,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        kl_weight=1.0,
        verbose=True,
    )
    print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}  "
          f"val loss: {history['val_loss'][-1]:.4f}")

    # ------------------------------------------------------------------
    # Evaluate anomaly score on a subsample of each class.
    # ------------------------------------------------------------------
    model.to(device)
    # Use at most 200 examples per class to keep evaluation fast.
    max_eval = 200
    rng2 = np.random.default_rng(0)
    eval_indices: list[int] = []
    for cls_idx in range(len(label_names)):
        cls_idx_arr = np.where(y_all == cls_idx)[0]
        chosen = rng2.choice(
            cls_idx_arr,
            size=min(max_eval, len(cls_idx_arr)),
            replace=False,
        )
        eval_indices.extend(chosen.tolist())
    eval_indices_arr = np.array(sorted(eval_indices))
    X_eval = torch.from_numpy(X_all[eval_indices_arr])
    y_eval = y_all[eval_indices_arr]
    _evaluate_scores(model, X_eval, y_eval, label_names, device)

    # ------------------------------------------------------------------
    # Save checkpoint.
    # ------------------------------------------------------------------
    ckpt_path = out / "cvae_anomaly.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "latent_dim": args.latent_dim,
            "label_trained_on": "ship",
            "n_mels": n_mels,
        },
        ckpt_path,
    )
    print(f"\nCVAE checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()

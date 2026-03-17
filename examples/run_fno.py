"""End-to-end FNO surrogate workflow: generate data, train, infer.

Demonstrates the full pipeline for training a Fourier Neural Operator
surrogate that replaces the FDTD solver for fast acoustic trace
prediction.

Steps
-----
1. Generate a small FDTD training dataset.
2. Train the FNO model.
3. Run inference and compare against an FDTD reference.

Usage
-----
    python examples/run_fno.py --n-samples 20 --epochs 50

    # Skip data generation if dataset exists:
    python examples/run_fno.py --data-dir data/fno_demo --epochs 100

    # Evaluate an existing checkpoint:
    python examples/run_fno.py --checkpoint checkpoints/fno/fno_final.pt --evaluate-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Ensure project root is importable when running from examples/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def generate_data(data_dir: Path, n_samples: int, seed: int) -> None:
    """Generate FDTD training samples."""
    from acoustic_sim.ml.fno_data_gen import generate_dataset

    print(f"\n=== Generating {n_samples} FDTD samples → {data_dir} ===\n")
    generate_dataset(
        n_samples=n_samples,
        output_dir=str(data_dir),
        dims="2d",
        seed=seed,
        dx=1.0,           # coarser grid for speed
        total_time=0.15,   # shorter sim for demo
    )


def train(data_dir: Path, epochs: int, checkpoint_dir: Path) -> Path:
    """Train the FNO and return path to final checkpoint."""
    import torch
    from torch.utils.data import DataLoader

    from acoustic_sim.ml.fno import AcousticFNO
    from acoustic_sim.ml.fno_training import FNODataset, train_fno

    print(f"\n=== Training FNO for {epochs} epochs ===\n")

    dataset = FNODataset(data_dir, max_receivers=24)
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = AcousticFNO(
        modes1=12, modes2=12, width=32, n_layers=3,
        n_time_steps=dataset.max_time_steps,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    print(f"Dataset: {len(dataset)} samples ({n_train} train, {n_val} val)")
    print(f"Max time steps: {dataset.max_time_steps}\n")

    history = train_fno(
        model, train_loader, val_loader,
        n_epochs=epochs, lr=1e-3,
        checkpoint_dir=str(checkpoint_dir),
    )

    # Save history.
    hist_path = checkpoint_dir / "history.json"
    with open(str(hist_path), "w") as f:
        json.dump(history, f, indent=2)

    final = checkpoint_dir / "fno_final.pt"
    print(f"\nFinal checkpoint: {final}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    if history["val_loss"]:
        print(f"Final val loss:   {history['val_loss'][-1]:.6f}")

    return final


def evaluate(
    checkpoint: Path,
    data_dir: Path,
    n_time_steps: int | None = None,
) -> None:
    """Compare FNO predictions against FDTD ground truth."""
    from acoustic_sim.ml.fno_inference import FNOForwardModel

    print(f"\n=== Evaluating FNO checkpoint: {checkpoint} ===\n")

    # Load a sample for comparison.
    samples = sorted(data_dir.glob("sample_*.npz"))
    if not samples:
        print("No samples found for evaluation.")
        return

    sample = np.load(str(samples[0]))
    vel = sample["velocity_field"]
    gx = sample["grid_x"]
    gy = sample["grid_y"]
    recv = sample["receiver_positions"]
    fdtd_traces = sample["traces"]
    src_x = float(sample["source_x"])
    src_y = float(sample["source_y"])
    src_freq = float(sample["source_freq"])

    if n_time_steps is None:
        n_time_steps = fdtd_traces.shape[1]

    fno = FNOForwardModel(
        checkpoint,
        modes=12, width=32, n_layers=3,
        n_time_steps=n_time_steps,
    )
    print(f"Model: {fno}")

    fno_traces = fno.predict(
        velocity_field=vel,
        grid_x=gx, grid_y=gy,
        receiver_positions=recv,
        source_x=src_x, source_y=src_y,
        source_freq=src_freq,
    )

    # Metrics.
    n_recv = min(fdtd_traces.shape[0], fno_traces.shape[0])
    n_steps = min(fdtd_traces.shape[1], fno_traces.shape[1])
    fdtd_sub = fdtd_traces[:n_recv, :n_steps]
    fno_sub = fno_traces[:n_recv, :n_steps]

    mse = np.mean((fdtd_sub - fno_sub) ** 2)
    fdtd_energy = np.mean(fdtd_sub ** 2)
    rel_l2 = np.sqrt(mse / max(fdtd_energy, 1e-20))

    print(f"\nSample 0 comparison ({n_recv} receivers, {n_steps} steps):")
    print(f"  MSE:           {mse:.6e}")
    print(f"  Relative L2:   {rel_l2:.4f}")
    print(f"  FDTD RMS:      {np.sqrt(fdtd_energy):.6e}")
    print(f"  FNO RMS:       {np.sqrt(np.mean(fno_sub**2)):.6e}")

    # Optional: save comparison plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        t = np.arange(n_steps)
        recv_idx = 0

        axes[0].plot(t, fdtd_sub[recv_idx], label="FDTD", alpha=0.8)
        axes[0].plot(t, fno_sub[recv_idx], label="FNO", alpha=0.8, ls="--")
        axes[0].set_ylabel("Pressure")
        axes[0].set_title(f"Receiver {recv_idx} trace comparison")
        axes[0].legend()

        axes[1].plot(t, fdtd_sub[recv_idx] - fno_sub[recv_idx], color="red")
        axes[1].set_ylabel("Residual")
        axes[1].set_xlabel("Time step")

        fig.tight_layout()
        plot_path = checkpoint.parent / "comparison.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"  Saved plot: {plot_path}")
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FNO surrogate model demo pipeline",
    )
    parser.add_argument("--data-dir", default="data/fno_demo",
                        help="Directory for training data")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of FDTD samples to generate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint-dir", default="checkpoints/fno_demo")
    parser.add_argument("--checkpoint", default=None,
                        help="Existing checkpoint (skips training)")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoint_dir)

    # Step 1: generate data (if needed).
    if not args.evaluate_only:
        existing = list(data_dir.glob("sample_*.npz"))
        if not existing:
            generate_data(data_dir, args.n_samples, args.seed)
        else:
            print(f"Found {len(existing)} existing samples in {data_dir}")

    # Step 2: train (unless checkpoint provided).
    if args.checkpoint:
        final_ckpt = Path(args.checkpoint)
    elif not args.evaluate_only:
        final_ckpt = train(data_dir, args.epochs, ckpt_dir)
    else:
        final_ckpt = ckpt_dir / "fno_final.pt"

    # Step 3: evaluate.
    if final_ckpt.exists():
        # Determine n_time_steps from dataset.
        samples = sorted(data_dir.glob("sample_*.npz"))
        n_ts = None
        if samples:
            with np.load(str(samples[0])) as d:
                n_ts = d["traces"].shape[1]
        evaluate(final_ckpt, data_dir, n_time_steps=n_ts)
    else:
        print(f"Checkpoint not found: {final_ckpt}")
        sys.exit(1)


if __name__ == "__main__":
    main()

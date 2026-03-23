"""Training script for the CVAE anomaly detector.

Generates synthetic mel spectrograms from known aerial threat classes
(quadcopter, hexacopter, fixed_wing) using the same source signal
generators and feature extraction pipeline as the CNN classifier.
Trains the ConvVAE on known-class spectrograms only, then calibrates
an anomaly threshold on a held-out validation set.

Outputs
-------
- Model weights:  ``output/models/anomaly_detector.pt``
- Threshold file: ``output/models/anomaly_threshold.json``

The threshold JSON contains::

    {
        "mean": <float>,        # mean recon error on validation set
        "std": <float>,         # std of recon error on validation set
        "threshold_3sigma": <float>  # recommended threshold (mean + 3*std)
    }

Usage::

    python src/acoustic_sim/ml/anomaly_training.py
    python src/acoustic_sim/ml/anomaly_training.py --quick
    python src/acoustic_sim/ml/anomaly_training.py --n-samples 500 --n-epochs 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from acoustic_sim.ml.anomaly_detector import ConvVAE, vae_loss
from acoustic_sim.ml.data_generation import generate_source_signal
from acoustic_sim.ml.features import compute_mel_spectrogram
from acoustic_sim.sources import MovingSource3D
from acoustic_sim.forward import simulate_3d_traces
from acoustic_sim.receivers import create_receiver_l_shaped_3d

# Known aerial threat classes used for CVAE training.
KNOWN_THREAT_CLASSES = ["quadcopter", "hexacopter", "fixed_wing"]


def generate_anomaly_training_data(
    n_samples_per_class: int = 300,
    dt: float = 1.0 / 4000,
    window_duration: float = 0.5,
    sound_speed: float = 343.0,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64,
    seed: int = 42,
) -> dict:
    """Generate mel spectrogram training data from known threat classes.

    Uses the same source signal generators and forward model as the
    classification dataset in
    :func:`~acoustic_sim.ml.data_generation.generate_classification_dataset`,
    but restricted to known aerial threat classes only.

    Parameters
    ----------
    n_samples_per_class : int
        Number of samples to generate per threat class.
    dt : float
        Simulation time step (1/sample_rate).
    window_duration : float
        Duration of each sample in seconds.
    sound_speed : float
        Speed of sound in m/s.
    n_fft, hop_length, n_mels : int
        Mel spectrogram parameters (must match the CNN classifier).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        spectrograms : list of 2D arrays (n_mels, n_time)
        class_names : list of str
        sample_rate : float
    """
    rng = np.random.default_rng(seed)
    n_steps = int(window_duration / dt)
    sample_rate = 1.0 / dt

    # Microphone array (same L-shaped geometry as classification dataset).
    mics = create_receiver_l_shaped_3d(
        5, 5, spacing=0.3, origin_x=0.0, origin_y=0.0, z=0.0,
    )

    spectrograms: list[np.ndarray] = []

    for cls_name in KNOWN_THREAT_CLASSES:
        for _ in range(n_samples_per_class):
            sig, params = generate_source_signal(cls_name, n_steps, dt, rng)

            # Random source position.
            horiz_dist = rng.uniform(50, 400)
            bearing = rng.uniform(0, 2 * math.pi)
            alt_lo, alt_hi = params.get("altitude_range", (20, 100))
            altitude = rng.uniform(max(alt_lo, 0), max(alt_hi, 0.1))
            sx = horiz_dist * math.cos(bearing)
            sy = horiz_dist * math.sin(bearing)

            # Speed and heading.
            min_speed = params.get("min_speed", 0)
            speed = rng.uniform(max(min_speed, 5), 25)
            if cls_name == "fixed_wing":
                speed = max(speed, 10)
            heading = rng.uniform(0, 2 * math.pi)

            src = MovingSource3D(
                x0=sx, y0=sy, z0=altitude,
                x1=sx + speed * window_duration * math.cos(heading),
                y1=sy + speed * window_duration * math.sin(heading),
                z1=altitude,
                speed=speed, signal=sig,
            )

            # Forward model.
            traces = simulate_3d_traces(
                src, mics, dt, n_steps, sound_speed,
                air_absorption=0.005,
            )

            # Add noise at random SNR.
            target_snr_dB = rng.uniform(-5, 30)
            sig_power = np.mean(traces ** 2)
            noise_power = sig_power / max(10 ** (target_snr_dB / 10), 1e-30)
            noise_std = math.sqrt(max(noise_power, 1e-30))
            noise = rng.standard_normal(traces.shape) * noise_std
            noisy_traces = traces + noise

            # Beamform (delay-and-sum at origin).
            beamformed = np.mean(noisy_traces, axis=0)

            # Compute mel spectrogram.
            mel = compute_mel_spectrogram(
                beamformed, sample_rate,
                n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            )
            spectrograms.append(mel)

    return {
        "spectrograms": spectrograms,
        "class_names": KNOWN_THREAT_CLASSES,
        "sample_rate": sample_rate,
    }


def prepare_anomaly_tensors(
    spectrograms: list[np.ndarray],
) -> torch.Tensor:
    """Convert spectrograms to a padded (N, 1, n_mels, max_time) tensor.

    Parameters
    ----------
    spectrograms : list of (n_mels, n_time) arrays.

    Returns
    -------
    (N, 1, n_mels, max_time) float32 tensor.
    """
    max_time = max(s.shape[1] for s in spectrograms)
    n_mels = spectrograms[0].shape[0]
    X = np.zeros((len(spectrograms), 1, n_mels, max_time), dtype=np.float32)
    for i, mel in enumerate(spectrograms):
        X[i, 0, :, :mel.shape[1]] = mel
    return torch.tensor(X)


def train_anomaly_detector(
    model: ConvVAE,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    kl_weight: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Train the CVAE anomaly detector.

    Parameters
    ----------
    model : ConvVAE
    X_train : (N_train, 1, n_mels, n_time)
    X_val : (N_val, 1, n_mels, n_time)
    n_epochs : int
    lr : float
    batch_size : int
    kl_weight : float
        β weighting for the KL divergence term.
    verbose : bool

    Returns
    -------
    dict with keys: train_loss, val_loss, train_recon, train_kl,
                    val_recon, val_kl (all lists of floats per epoch).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(X_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "train_kl": [],
        "val_recon": [], "val_kl": [],
    }

    for epoch in range(n_epochs):
        # -- Training --
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0
        for (X_batch,) in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = model(X_batch)
            loss, recon_loss, kl_loss = vae_loss(
                recon, X_batch, mu, log_var, kl_weight=kl_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        avg_train = total_loss / max(n_batches, 1)
        avg_recon = total_recon / max(n_batches, 1)
        avg_kl = total_kl / max(n_batches, 1)
        history["train_loss"].append(avg_train)
        history["train_recon"].append(avg_recon)
        history["train_kl"].append(avg_kl)

        # -- Validation --
        model.eval()
        with torch.no_grad():
            recon_v, mu_v, log_var_v = model(X_val.to(device))
            val_loss, val_recon, val_kl = vae_loss(
                recon_v, X_val.to(device), mu_v, log_var_v,
                kl_weight=kl_weight,
            )
        history["val_loss"].append(val_loss.item())
        history["val_recon"].append(val_recon.item())
        history["val_kl"].append(val_kl.item())

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}: "
                  f"train={avg_train:.4f} (recon={avg_recon:.4f}, "
                  f"kl={avg_kl:.4f}), val={val_loss.item():.4f}")

    return history


def calibrate_threshold(
    model: ConvVAE,
    X_val: torch.Tensor,
) -> dict:
    """Calibrate the anomaly threshold from the validation set.

    Computes per-sample anomaly scores on the known-class validation
    data and derives a threshold at mean + 3*std (3-sigma rule).

    Parameters
    ----------
    model : ConvVAE (should be in eval mode)
    X_val : (N_val, 1, n_mels, n_time)

    Returns
    -------
    dict with keys: mean, std, threshold_3sigma (all floats).
    """
    device = next(model.parameters()).device
    model.eval()

    scores: list[float] = []
    for i in range(X_val.size(0)):
        x_i = X_val[i:i + 1].to(device)
        score, _ = model.compute_anomaly_score(x_i)
        scores.append(score)

    scores_arr = np.array(scores)
    mean_score = float(np.mean(scores_arr))
    std_score = float(np.std(scores_arr))
    threshold = mean_score + 3.0 * std_score

    return {
        "mean": mean_score,
        "std": std_score,
        "threshold_3sigma": threshold,
    }


def main():
    """Train the CVAE anomaly detector and save model + threshold."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller dataset for quick testing")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Samples per class (default: 300 or 80 for --quick)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Training epochs (default: 100 or 30 for --quick)")
    parser.add_argument("--latent-dim", type=int, default=12,
                        help="Latent space dimensionality (default: 12)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/models"),
                        help="Output directory for model and threshold")
    args = parser.parse_args()

    if args.quick:
        n_samples = args.n_samples or 80
        n_epochs = args.n_epochs or 30
    else:
        n_samples = args.n_samples or 300
        n_epochs = args.n_epochs or 100

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ANOMALY DETECTOR TRAINING (CVAE)")
    print("=" * 60)
    print(f"  Known classes: {KNOWN_THREAT_CLASSES}")
    print(f"  Samples/class: {n_samples}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Output: {args.output_dir}")

    # ── Generate training data ──────────────────────────────────────
    t0 = time.perf_counter()
    print(f"\n[1/3] Generating {n_samples * len(KNOWN_THREAT_CLASSES)} "
          f"known-class spectrograms...")
    data = generate_anomaly_training_data(
        n_samples_per_class=n_samples,
        dt=1.0 / 4000,
        window_duration=0.5,
        seed=42,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Generated {len(data['spectrograms'])} spectrograms "
          f"in {elapsed:.1f}s")

    # Prepare tensors.
    X = prepare_anomaly_tensors(data["spectrograms"])
    print(f"  Tensor shape: {tuple(X.shape)}")

    # Train/val split (80/20).
    n = X.size(0)
    indices = np.random.default_rng(42).permutation(n)
    split = int(0.8 * n)
    X_train = X[indices[:split]]
    X_val = X[indices[split:]]
    print(f"  Train: {X_train.size(0)}, Val: {X_val.size(0)}")

    # ── Train ───────────────────────────────────────────────────────
    print(f"\n[2/3] Training ConvVAE for {n_epochs} epochs...")
    model = ConvVAE(latent_dim=args.latent_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    t0 = time.perf_counter()
    history = train_anomaly_detector(
        model, X_train, X_val,
        n_epochs=n_epochs, lr=1e-3, batch_size=32,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Training completed in {elapsed:.1f}s")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss:   {history['val_loss'][-1]:.4f}")

    # ── Calibrate threshold ─────────────────────────────────────────
    print("\n[3/3] Calibrating anomaly threshold...")
    threshold_info = calibrate_threshold(model, X_val)
    print(f"  Validation scores: mean={threshold_info['mean']:.4f}, "
          f"std={threshold_info['std']:.4f}")
    print(f"  Recommended threshold (3σ): {threshold_info['threshold_3sigma']:.4f}")

    # ── Save ────────────────────────────────────────────────────────
    model_path = args.output_dir / "anomaly_detector.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    threshold_path = args.output_dir / "anomaly_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold_info, f, indent=2)
    print(f"  Threshold saved: {threshold_path}")

    print(f"\n{'=' * 60}")
    print(f"  ANOMALY DETECTOR TRAINING COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

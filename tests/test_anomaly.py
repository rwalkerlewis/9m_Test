#!/usr/bin/env python3
"""Anomaly detection module tests.

Tests covering:
1. CVAE forward/backward pass with random input
2. Reconstruction error is low for in-distribution spectrograms after brief training
3. Reconstruction error is high for out-of-distribution input
4. Integration wrapper returns correct dataclass fields
5. Pipeline integration does not break existing detection flow
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.ml.anomaly_detector import ConvVAE, vae_loss
from acoustic_sim.ml.features import compute_mel_spectrogram
from acoustic_sim.ml.data_generation import generate_source_signal


# =====================================================================
#  Test 1: CVAE forward / backward pass
# =====================================================================

def test_cvae_forward_backward():
    """Verify ConvVAE forward pass, output shapes, and gradient flow."""
    print("\n" + "=" * 60)
    print("  TEST 1: CVAE Forward / Backward Pass")
    print("=" * 60)

    for latent_dim in [8, 12, 16]:
        model = ConvVAE(latent_dim=latent_dim)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  latent_dim={latent_dim}: {n_params:,} parameters")
        assert n_params < 2_000_000, (
            f"Model too large: {n_params:,} params (limit: 2,000,000)")

        # Test with various input time dimensions.
        for n_time in [5, 8, 16, 32]:
            x = torch.randn(4, 1, 64, n_time)
            model.train()
            recon, mu, log_var = model(x)

            assert recon.shape == x.shape, (
                f"Shape mismatch for T={n_time}: "
                f"recon={recon.shape} vs input={x.shape}")
            assert mu.shape == (4, latent_dim)
            assert log_var.shape == (4, latent_dim)

            # Compute loss and backward.
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, log_var)
            loss.backward()

            # Verify gradients flow to all parameters.
            for name, param in model.named_parameters():
                assert param.grad is not None, f"No gradient for {name}"

            model.zero_grad()

        # Test compute_anomaly_score.
        model.eval()
        x_single = torch.randn(1, 1, 64, 8)
        score, latent = model.compute_anomaly_score(x_single)
        assert isinstance(score, float)
        assert latent.shape == (latent_dim,)
        assert np.isfinite(score), f"Non-finite anomaly score: {score}"

        print(f"    All shape/gradient checks passed for latent_dim={latent_dim}")

    print("\n  *** TEST 1 PASSED ***")


# =====================================================================
#  Test 2: Low reconstruction error for in-distribution data
# =====================================================================

def test_reconstruction_low_for_in_distribution():
    """Train CVAE briefly and verify low recon error on known classes."""
    print("\n" + "=" * 60)
    print("  TEST 2: Low Reconstruction Error (In-Distribution)")
    print("=" * 60)

    # Generate known-class spectrograms.
    rng = np.random.default_rng(42)
    dt = 1.0 / 4000
    sample_rate = 4000.0
    n_steps = int(0.5 / dt)  # 0.5 seconds
    known_classes = ["quadcopter", "hexacopter", "fixed_wing"]

    print("\n  Generating known-class spectrograms...")
    specs = []
    for cls_name in known_classes:
        for _ in range(50):
            sig, _ = generate_source_signal(cls_name, n_steps, dt, rng)
            mel = compute_mel_spectrogram(sig, sample_rate)
            specs.append(mel)
    print(f"  Generated {len(specs)} spectrograms")

    # Pad to uniform size and create tensor.
    max_time = max(s.shape[1] for s in specs)
    n_mels = specs[0].shape[0]
    X = np.zeros((len(specs), 1, n_mels, max_time), dtype=np.float32)
    for i, s in enumerate(specs):
        X[i, 0, :, :s.shape[1]] = s
    X = torch.tensor(X)

    # Split.
    n = len(X)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    X_train = X[idx[:split]]
    X_val = X[idx[split:]]

    # Train.
    n_epochs = 80
    print(f"  Training ConvVAE ({n_epochs} epochs, {X_train.size(0)} train, "
          f"{X_val.size(0)} val)...")
    model = ConvVAE(latent_dim=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        # Mini-batch training for better convergence.
        perm = torch.randperm(X_train.size(0))
        batch_size = 32
        for start in range(0, X_train.size(0), batch_size):
            batch = X_train[perm[start:start + batch_size]]
            recon, mu, log_var = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Measure reconstruction error on validation set.
    model.eval()
    val_scores = []
    for i in range(X_val.size(0)):
        score, _ = model.compute_anomaly_score(X_val[i:i + 1])
        val_scores.append(score)
    val_scores = np.array(val_scores)
    mean_val_score = float(np.mean(val_scores))
    std_val_score = float(np.std(val_scores))

    # Also compute untrained baseline for comparison.
    untrained_model = ConvVAE(latent_dim=12)
    untrained_model.eval()
    untrained_scores = []
    for i in range(X_val.size(0)):
        score, _ = untrained_model.compute_anomaly_score(X_val[i:i + 1])
        untrained_scores.append(score)
    untrained_mean = float(np.mean(untrained_scores))

    print(f"  Untrained baseline score: {untrained_mean:.4f}")
    print(f"  Trained validation score: {mean_val_score:.4f}")
    print(f"  Improvement: {(1 - mean_val_score / untrained_mean) * 100:.1f}%")

    # After training, reconstruction error should decrease significantly
    # compared to the untrained model.
    assert mean_val_score < untrained_mean, (
        f"Trained model ({mean_val_score:.4f}) should have lower error than "
        f"untrained ({untrained_mean:.4f})")
    print(f"  In-distribution score ({mean_val_score:.4f}) is below threshold")

    print("\n  *** TEST 2 PASSED ***")
    return model, mean_val_score, std_val_score, X_val


# =====================================================================
#  Test 3: High reconstruction error for out-of-distribution input
# =====================================================================

def test_reconstruction_high_for_ood(model=None, in_dist_mean=None,
                                      in_dist_std=None, X_val_known=None):
    """Verify OOD inputs produce higher reconstruction error."""
    print("\n" + "=" * 60)
    print("  TEST 3: High Reconstruction Error (Out-of-Distribution)")
    print("=" * 60)

    # If no model provided, train one quickly.
    if model is None:
        print("  Training fresh model for OOD test...")
        _, model_ret, mean_ret, std_ret, xv = test_reconstruction_low_for_in_distribution()
        model = model_ret
        in_dist_mean = mean_ret
        in_dist_std = std_ret
        X_val_known = xv

    n_mels = 64
    max_time = X_val_known.size(3)
    dt = 1.0 / 4000
    sample_rate = 4000.0

    # Generate OOD spectrograms.
    print("\n  Generating OOD spectrograms...")
    ood_specs = []
    rng = np.random.default_rng(99)

    # Type 1: Pure random noise.
    for _ in range(30):
        noise = rng.standard_normal(int(0.5 / dt))
        mel = compute_mel_spectrogram(noise, sample_rate)
        ood_specs.append(mel)

    # Type 2: Pure tones (not in training set).
    for freq in [500, 1000, 2000, 3000, 5000]:
        t = np.arange(int(0.5 / dt)) * dt
        for _ in range(6):
            tone = np.sin(2 * math.pi * freq * t + rng.uniform(0, 2 * math.pi))
            mel = compute_mel_spectrogram(tone, sample_rate)
            ood_specs.append(mel)

    print(f"  Generated {len(ood_specs)} OOD spectrograms")

    # Pad and create tensor.
    X_ood = np.zeros((len(ood_specs), 1, n_mels, max_time), dtype=np.float32)
    for i, s in enumerate(ood_specs):
        t_len = min(s.shape[1], max_time)
        X_ood[i, 0, :, :t_len] = s[:, :t_len]
    X_ood = torch.tensor(X_ood)

    # Compute OOD scores.
    model.eval()
    ood_scores = []
    for i in range(X_ood.size(0)):
        score, _ = model.compute_anomaly_score(X_ood[i:i + 1])
        ood_scores.append(score)
    ood_scores = np.array(ood_scores)
    mean_ood = float(np.mean(ood_scores))

    # Compute in-distribution scores for comparison.
    in_scores = []
    for i in range(X_val_known.size(0)):
        score, _ = model.compute_anomaly_score(X_val_known[i:i + 1])
        in_scores.append(score)
    in_scores = np.array(in_scores)
    mean_in = float(np.mean(in_scores))

    print(f"  In-distribution mean score:  {mean_in:.4f}")
    print(f"  Out-of-distribution mean score: {mean_ood:.4f}")
    print(f"  Ratio (OOD / in-dist): {mean_ood / max(mean_in, 1e-10):.2f}x")

    # OOD should have higher reconstruction error than in-distribution.
    assert mean_ood > mean_in, (
        f"OOD mean ({mean_ood:.4f}) should exceed in-dist mean ({mean_in:.4f})")
    print(f"  OOD score ({mean_ood:.4f}) > in-dist score ({mean_in:.4f}) ✓")

    print("\n  *** TEST 3 PASSED ***")


# =====================================================================
#  Test 4: Integration wrapper returns correct dataclass fields
# =====================================================================

def test_integration_wrapper_dataclass():
    """Verify AnomalyDetector wrapper returns correct AnomalyResult."""
    print("\n" + "=" * 60)
    print("  TEST 4: Integration Wrapper Dataclass")
    print("=" * 60)

    from acoustic_sim.ml.anomaly_integration import AnomalyDetector, AnomalyResult

    # Create and save a model + threshold.
    model = ConvVAE(latent_dim=12)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "anomaly_detector.pt"
        threshold_path = Path(tmpdir) / "anomaly_threshold.json"

        torch.save(model.state_dict(), model_path)
        threshold_info = {
            "mean": 0.5,
            "std": 0.1,
            "threshold_3sigma": 0.8,
        }
        with open(threshold_path, "w") as f:
            json.dump(threshold_info, f)

        # Instantiate wrapper.
        detector = AnomalyDetector(
            model_path=model_path,
            threshold_path=threshold_path,
            latent_dim=12,
            sample_rate=4000.0,
        )
        print(f"  Loaded detector, threshold={detector.threshold}")

        # Test process_frame with random audio.
        audio = np.random.randn(2000).astype(np.float64)
        result = detector.process_frame(audio, sample_rate=4000.0)

        assert isinstance(result, AnomalyResult), (
            f"Expected AnomalyResult, got {type(result)}")
        assert isinstance(result.is_novel, bool), (
            f"is_novel should be bool, got {type(result.is_novel)}")
        assert isinstance(result.reconstruction_error, float), (
            f"reconstruction_error should be float, got "
            f"{type(result.reconstruction_error)}")
        assert isinstance(result.threshold, float), (
            f"threshold should be float, got {type(result.threshold)}")
        assert isinstance(result.latent_vector, np.ndarray), (
            f"latent_vector should be ndarray, got "
            f"{type(result.latent_vector)}")
        assert result.latent_vector.shape == (12,), (
            f"latent_vector shape should be (12,), got "
            f"{result.latent_vector.shape}")
        assert np.isfinite(result.reconstruction_error)
        assert result.threshold == 0.8

        print(f"  process_frame result:")
        print(f"    is_novel: {result.is_novel}")
        print(f"    reconstruction_error: {result.reconstruction_error:.4f}")
        print(f"    threshold: {result.threshold}")
        print(f"    latent_vector shape: {result.latent_vector.shape}")

        # Test process_mel_spectrogram.
        mel = compute_mel_spectrogram(audio, 4000.0)
        result2 = detector.process_mel_spectrogram(mel)
        assert isinstance(result2, AnomalyResult)
        assert isinstance(result2.is_novel, bool)
        assert result2.latent_vector.shape == (12,)
        print(f"  process_mel_spectrogram: OK")

    print("\n  *** TEST 4 PASSED ***")


# =====================================================================
#  Test 5: Pipeline integration does not break existing detection flow
# =====================================================================

def test_pipeline_integration_no_break():
    """Verify pipeline config and imports work with anomaly detection."""
    print("\n" + "=" * 60)
    print("  TEST 5: Pipeline Integration (No Breakage)")
    print("=" * 60)

    # 1. Verify all anomaly module imports work.
    from acoustic_sim.ml.anomaly_detector import ConvVAE, vae_loss
    from acoustic_sim.ml.anomaly_integration import AnomalyDetector, AnomalyResult
    from acoustic_sim.ml.anomaly_training import (
        generate_anomaly_training_data,
        prepare_anomaly_tensors,
        train_anomaly_detector,
        calibrate_threshold,
        KNOWN_THREAT_CLASSES,
    )
    print("  All anomaly module imports: OK")

    # 2. Verify pipeline config includes anomaly fields.
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_pipeline",
        str(Path(__file__).parent.parent / "examples" / "run_pipeline.py"),
    )
    pipeline_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_mod)

    cfg = pipeline_mod.load_config(None)
    ml_cfg = cfg["ml"]

    required_keys = [
        "enable_anomaly_detection",
        "anomaly_checkpoint",
        "anomaly_threshold_file",
        "anomaly_override_confidence_threshold",
    ]
    for key in required_keys:
        assert key in ml_cfg, f"Missing config key: {key}"
        print(f"    Config key '{key}': {ml_cfg[key]}")

    # 3. Verify load_ml_models returns anomaly key.
    models = pipeline_mod.load_ml_models(ml_cfg)
    assert "anomaly" in models, "load_ml_models missing 'anomaly' key"
    # With defaults, anomaly is disabled, so it should be None.
    assert models["anomaly"] is None, (
        "Anomaly model should be None when disabled")
    print("  load_ml_models returns anomaly=None when disabled: OK")

    # 4. Verify that existing detection imports still work.
    from acoustic_sim.detection import DetectionEngine, WindowDetection
    from acoustic_sim.ml.acoustic_classifier import AcousticClassifier
    from acoustic_sim.ml.maneuver_classifier import ManeuverClassifier
    from acoustic_sim.ml.fusion_classifier import FusionClassifier
    print("  Existing detection/ML imports: OK")

    # 5. Verify AnomalyResult fields are correct.
    import dataclasses
    fields = {f.name: f.type for f in dataclasses.fields(AnomalyResult)}
    assert "is_novel" in fields
    assert "reconstruction_error" in fields
    assert "threshold" in fields
    assert "latent_vector" in fields
    print(f"  AnomalyResult fields: {list(fields.keys())}")

    print("\n  *** TEST 5 PASSED ***")


# =====================================================================
#  Run all
# =====================================================================

def run_all():
    """Run all anomaly detection tests."""
    print("\n" + "=" * 60)
    print("  ANOMALY DETECTION MODULE TESTS")
    print("=" * 60)

    test_cvae_forward_backward()
    model, mean_score, std_score, X_val = test_reconstruction_low_for_in_distribution()
    test_reconstruction_high_for_ood(model, mean_score, std_score, X_val)
    test_integration_wrapper_dataclass()
    test_pipeline_integration_no_break()

    print("\n" + "=" * 60)
    print("  ALL ANOMALY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all()

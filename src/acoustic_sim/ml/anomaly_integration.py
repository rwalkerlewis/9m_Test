"""Integration wrapper for CVAE-based acoustic anomaly detection.

Provides a high-level :class:`AnomalyDetector` that takes a raw audio
frame (or pre-computed mel spectrogram), runs it through the trained
:class:`~acoustic_sim.ml.anomaly_detector.ConvVAE`, and returns an
:class:`AnomalyResult` with novelty flag, reconstruction error,
threshold, and latent vector.

Example
-------
::

    from acoustic_sim.ml.anomaly_integration import AnomalyDetector

    detector = AnomalyDetector(
        model_path="output/models/anomaly_detector.pt",
        threshold_path="output/models/anomaly_threshold.json",
    )
    result = detector.process_frame(audio_frame, sample_rate=4000)
    if result.is_novel:
        print(f"Novel threat detected! error={result.reconstruction_error:.4f}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from acoustic_sim.ml.anomaly_detector import ConvVAE
from acoustic_sim.ml.features import compute_mel_spectrogram


@dataclass
class AnomalyResult:
    """Result of anomaly detection on a single audio frame.

    Attributes
    ----------
    is_novel : bool
        ``True`` if the reconstruction error exceeds the calibrated
        threshold, indicating an acoustically novel target.
    reconstruction_error : float
        Combined anomaly score (MSE reconstruction error + KL
        divergence from the standard normal prior).
    threshold : float
        The calibrated threshold used for the novelty decision.
    latent_vector : np.ndarray
        The latent-space embedding (shape ``(latent_dim,)``), useful
        for downstream clustering of novel threats into groups.
    """

    is_novel: bool
    reconstruction_error: float
    threshold: float
    latent_vector: np.ndarray


class AnomalyDetector:
    """Integration wrapper for CVAE-based anomaly detection.

    Loads a trained :class:`ConvVAE` and threshold calibration, then
    provides :meth:`process_frame` and :meth:`process_mel_spectrogram`
    for easy integration into the detection pipeline.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved ConvVAE state dict (``.pt`` file).
    threshold_path : str or Path
        Path to the threshold calibration JSON file produced by
        :func:`~acoustic_sim.ml.anomaly_training.calibrate_threshold`.
    latent_dim : int
        Must match the ``latent_dim`` used during training.
    sample_rate : float
        Default sample rate for :meth:`process_frame`.
    n_fft : int
        FFT window size for mel spectrogram (must match training).
    hop_length : int
        Hop length for mel spectrogram (must match training).
    n_mels : int
        Number of mel bands (must match training).
    device : str
        PyTorch device (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold_path: str | Path,
        latent_dim: int = 12,
        sample_rate: float = 8000.0,
        n_fft: int = 512,
        hop_length: int = 128,
        n_mels: int = 64,
        device: str = "cpu",
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.device = torch.device(device)

        # Load model.
        self.model = ConvVAE(latent_dim=latent_dim)
        self.model.load_state_dict(
            torch.load(str(model_path), map_location=self.device,
                        weights_only=True),
        )
        self.model.to(self.device)
        self.model.eval()

        # Load threshold calibration.
        with open(str(threshold_path)) as f:
            cal = json.load(f)
        self.threshold = float(cal["threshold_3sigma"])
        self.cal_mean = float(cal["mean"])
        self.cal_std = float(cal["std"])

    def process_frame(
        self,
        audio_frame: np.ndarray,
        sample_rate: float | None = None,
    ) -> AnomalyResult:
        """Run anomaly detection on a raw audio frame.

        Computes the mel spectrogram using the same parameters as the
        CNN classifier, then feeds it through the CVAE.

        Parameters
        ----------
        audio_frame : 1D numpy array
            Raw audio samples (beamformed or single-channel).
        sample_rate : float or None
            Sample rate of the audio.  If ``None``, uses the default
            ``self.sample_rate``.

        Returns
        -------
        AnomalyResult
        """
        sr = sample_rate if sample_rate is not None else self.sample_rate
        mel = compute_mel_spectrogram(
            audio_frame, sr,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        return self.process_mel_spectrogram(mel)

    def process_mel_spectrogram(
        self,
        mel_spec: np.ndarray,
    ) -> AnomalyResult:
        """Run anomaly detection on a pre-computed mel spectrogram.

        Parameters
        ----------
        mel_spec : (n_mels, n_time) numpy array
            Log-mel spectrogram (same format as produced by
            :func:`~acoustic_sim.ml.features.compute_mel_spectrogram`).

        Returns
        -------
        AnomalyResult
        """
        # Convert to tensor: (1, 1, n_mels, n_time).
        x = torch.tensor(
            mel_spec[np.newaxis, np.newaxis, :, :],
            dtype=torch.float32,
        ).to(self.device)

        score, latent = self.model.compute_anomaly_score(x)

        return AnomalyResult(
            is_novel=score > self.threshold,
            reconstruction_error=score,
            threshold=self.threshold,
            latent_vector=latent,
        )

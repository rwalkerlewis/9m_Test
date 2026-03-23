"""Convolutional Variational Autoencoder (CVAE) for acoustic anomaly detection.

Provides open-set threat recognition by learning the latent distribution
of known aerial threat classes (quadcopter, hexacopter, fixed_wing).
At inference, acoustically novel targets produce high reconstruction
error and KL divergence from the learned prior, flagging them as
potential novel threats without requiring them in the training
distribution.

Architecture
------------
The CVAE operates on the same mel spectrogram representation used by
:class:`~acoustic_sim.ml.acoustic_classifier.AcousticClassifier`:

    Input: (batch, 1, n_mels, n_time)

**Encoder:**

    Conv2d(1, 16, 3, stride=2, padding=1) + BatchNorm2d + ReLU
    Conv2d(16, 32, 3, stride=2, padding=1) + BatchNorm2d + ReLU
    Conv2d(32, 64, 3, stride=2, padding=1) + BatchNorm2d + ReLU
    AdaptiveAvgPool2d(4, 4) → Flatten → (batch, 1024)
    Linear(1024, latent_dim)  [mu]
    Linear(1024, latent_dim)  [log_var]

**Decoder:**

    Linear(latent_dim, 1024) → Reshape → (batch, 64, 4, 4)
    ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1) + BN + ReLU
    ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1) + BN + ReLU
    ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
    Interpolate to original input spatial dimensions

The model is lightweight (~85K parameters with ``latent_dim=12``)
and suitable for Jetson-class edge deployment.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder for anomaly detection.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space (recommended 8–16).
    """

    def __init__(self, latent_dim: int = 12):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────────────────
        self.enc_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Latent heads.
        self.fc_mu = nn.Linear(64 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(64 * 4 * 4, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────
        self.fc_decode = nn.Linear(latent_dim, 64 * 4 * 4)
        self.dec_conv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1,
        )
        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_conv2 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1,
        )
        self.dec_bn2 = nn.BatchNorm2d(16)
        self.dec_conv3 = nn.ConvTranspose2d(
            16, 1, kernel_size=3, stride=2, padding=1, output_padding=1,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input mel spectrogram to latent distribution parameters.

        Parameters
        ----------
        x : (batch, 1, n_mels, n_time)

        Returns
        -------
        mu : (batch, latent_dim)
        log_var : (batch, latent_dim)
        """
        h = torch.relu(self.enc_bn1(self.enc_conv1(x)))
        h = torch.relu(self.enc_bn2(self.enc_conv2(h)))
        h = torch.relu(self.enc_bn3(self.enc_conv3(h)))
        h = self.enc_pool(h)  # (batch, 64, 4, 4)
        h = h.view(h.size(0), -1)  # (batch, 1024)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Sample latent vector using the reparameterization trick.

        Parameters
        ----------
        mu : (batch, latent_dim)
        log_var : (batch, latent_dim)

        Returns
        -------
        z : (batch, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(
        self,
        z: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode latent vector to reconstructed mel spectrogram.

        Parameters
        ----------
        z : (batch, latent_dim)
        output_size : (height, width) target spatial dimensions.
            If ``None``, returns the raw decoder output without
            interpolation.

        Returns
        -------
        (batch, 1, H, W) reconstructed mel spectrogram.
        """
        h = torch.relu(self.fc_decode(z))
        h = h.view(h.size(0), 64, 4, 4)
        h = torch.relu(self.dec_bn1(self.dec_conv1(h)))
        h = torch.relu(self.dec_bn2(self.dec_conv2(h)))
        h = self.dec_conv3(h)
        if output_size is not None:
            h = F.interpolate(h, size=output_size, mode="bilinear",
                              align_corners=False)
        return h

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → reparameterize → decode.

        Parameters
        ----------
        x : (batch, 1, n_mels, n_time)

        Returns
        -------
        recon : (batch, 1, n_mels, n_time) — reconstructed spectrogram.
        mu : (batch, latent_dim)
        log_var : (batch, latent_dim)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        output_size = (x.size(2), x.size(3))
        recon = self.decode(z, output_size=output_size)
        return recon, mu, log_var

    @torch.no_grad()
    def compute_anomaly_score(
        self, x: torch.Tensor,
    ) -> tuple[float, np.ndarray]:
        """Compute anomaly score for a single mel spectrogram.

        The score combines MSE reconstruction error and KL divergence
        from the standard normal prior.

        Parameters
        ----------
        x : (1, 1, n_mels, n_time) — single spectrogram batch.

        Returns
        -------
        score : float
            Combined anomaly score (reconstruction_mse + kl_divergence).
        latent_vector : (latent_dim,) numpy array
            The mean of the latent distribution (useful for downstream
            clustering of novel threats).
        """
        was_training = self.training
        self.eval()

        mu, log_var = self.encode(x)
        z = mu  # deterministic at inference
        output_size = (x.size(2), x.size(3))
        recon = self.decode(z, output_size=output_size)

        recon_error = float(F.mse_loss(recon, x, reduction="mean"))
        kl_div = float(
            -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        )
        score = recon_error + kl_div
        latent = mu.squeeze(0).cpu().numpy()

        if was_training:
            self.train()
        return score, latent


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss: MSE reconstruction + KL divergence.

    Parameters
    ----------
    recon_x : (batch, 1, H, W) — reconstructed spectrogram.
    x : (batch, 1, H, W) — original spectrogram.
    mu : (batch, latent_dim)
    log_var : (batch, latent_dim)
    kl_weight : float
        Weighting factor for the KL term (β-VAE parameter).

    Returns
    -------
    total_loss : scalar tensor
    recon_loss : scalar tensor (MSE component)
    kl_loss : scalar tensor (KL divergence component)
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss

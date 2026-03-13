"""Acoustic source classifier using a simple CNN on mel spectrograms.

Architecture (under 50 lines of model code):
    Input: (batch, 1, n_mel_bands, n_time_frames)
    Conv2d(1, 16, 3, padding=1) + BatchNorm2d + ReLU
    Conv2d(16, 32, 3, padding=1) + BatchNorm2d + ReLU
    MaxPool2d(2)
    Conv2d(32, 64, 3, padding=1) + BatchNorm2d + ReLU
    GlobalAvgPool → (batch, 64)
    Linear(64, n_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AcousticClassifier(nn.Module):
    """Small CNN for acoustic source classification."""

    def __init__(self, n_classes: int = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch, 1, n_mels, n_time)

        Returns
        -------
        (batch, n_classes) logits.
        """
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        # Global average pooling.
        x = x.mean(dim=(-2, -1))  # (batch, 64)
        x = self.fc(x)
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get the 64-dimensional acoustic embedding (before FC layer)."""
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=(-2, -1))
        return x

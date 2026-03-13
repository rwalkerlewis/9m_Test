"""Maneuver detection classifier using a 1D CNN.

Architecture:
    Input: (batch, 6, N) — 6 features over N time steps
    Conv1d(6, 32, kernel_size=5, padding=2) + ReLU
    Conv1d(32, 64, kernel_size=5, padding=2) + ReLU
    GlobalAvgPool → (batch, 64)
    Linear(64, n_maneuver_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ManeuverClassifier(nn.Module):
    """1D CNN for maneuver state classification."""

    def __init__(self, n_classes: int = 6):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch, 6, N) — 6 features × N time steps.

        Returns
        -------
        (batch, n_classes) logits.
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=-1)  # Global average pooling → (batch, 64)
        x = self.fc(x)
        return x

"""Kinematic fusion classifier — two-branch network.

Branch A: acoustic (identical to AcousticClassifier up to final layer)
Branch B: kinematic MLP (14 → 32 → 32)
Fusion: concatenate (96) → 64 → n_classes
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KinematicBranch(nn.Module):
    """MLP for kinematic features."""

    def __init__(self, n_features: int = 14, embed_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(n_features, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class FusionClassifier(nn.Module):
    """Two-branch acoustic + kinematic fusion classifier."""

    def __init__(self, n_classes: int = 6, n_kinematic_features: int = 14):
        super().__init__()
        # Acoustic branch (same architecture as AcousticClassifier).
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Kinematic branch.
        self.kinematic = KinematicBranch(n_kinematic_features, 32)
        # Fusion.
        self.fusion_fc1 = nn.Linear(96, 64)
        self.fusion_fc2 = nn.Linear(64, n_classes)

    def forward(
        self,
        mel_spec: torch.Tensor,
        kinematic: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        mel_spec : (batch, 1, n_mels, n_time)
        kinematic : (batch, 14)

        Returns
        -------
        (batch, n_classes) logits.
        """
        # Acoustic branch.
        a = torch.relu(self.bn1(self.conv1(mel_spec)))
        a = torch.relu(self.bn2(self.conv2(a)))
        a = self.pool(a)
        a = torch.relu(self.bn3(self.conv3(a)))
        a = a.mean(dim=(-2, -1))  # (batch, 64)
        # Kinematic branch.
        k = self.kinematic(kinematic)  # (batch, 32)
        # Fusion.
        fused = torch.cat([a, k], dim=1)  # (batch, 96)
        x = torch.relu(self.fusion_fc1(fused))
        x = self.fusion_fc2(x)
        return x

    def load_acoustic_weights(self, acoustic_model) -> None:
        """Initialize acoustic branch from a trained AcousticClassifier."""
        self.conv1.load_state_dict(acoustic_model.conv1.state_dict())
        self.bn1.load_state_dict(acoustic_model.bn1.state_dict())
        self.conv2.load_state_dict(acoustic_model.conv2.state_dict())
        self.bn2.load_state_dict(acoustic_model.bn2.state_dict())
        self.conv3.load_state_dict(acoustic_model.conv3.state_dict())
        self.bn3.load_state_dict(acoustic_model.bn3.state_dict())


class KinematicOnlyClassifier(nn.Module):
    """Kinematic-only classifier (baseline for comparison)."""

    def __init__(self, n_classes: int = 6, n_features: int = 14):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""Plotting utilities for velocity models and wavefields."""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from acoustic_sim.model import VelocityModel

matplotlib.use("Agg")


def plot_velocity_model(
    model: VelocityModel,
    output_path: str = "velocity_model.png",
    receivers: np.ndarray | None = None,
    source_xy: np.ndarray | None = None,
    title: str = "Velocity Model",
) -> None:
    """Save a figure of the velocity field with optional overlays."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ext = model.extent
    im = ax.imshow(
        model.values,
        origin="lower",
        extent=[ext[0], ext[1], ext[2], ext[3]],
        cmap="seismic",
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wave speed [m/s]")
    if receivers is not None:
        ax.scatter(
            receivers[:, 0], receivers[:, 1],
            s=30, c="cyan", edgecolors="black", zorder=5, label="Receivers",
        )
    if source_xy is not None:
        ax.scatter(
            source_xy[0], source_xy[1],
            s=80, c="yellow", marker="*", edgecolors="black", zorder=6,
            label="Source",
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    if receivers is not None or source_xy is not None:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote velocity model plot to {output_path}")


def plot_wavefield(
    model: VelocityModel,
    field: np.ndarray,
    output_path: str = "wavefield.png",
    receivers: np.ndarray | None = None,
    source_xy: np.ndarray | None = None,
    title: str = "Helmholtz wavefield",
) -> None:
    """Save a figure of the Helmholtz pressure magnitude with optional overlays."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ext = model.extent
    im = ax.imshow(
        field,
        origin="lower",
        extent=[ext[0], ext[1], ext[2], ext[3]],
        cmap="magma",
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|p(x, y)| (pressure magnitude)")

    if receivers is not None:
        ax.scatter(
            receivers[:, 0], receivers[:, 1],
            s=30, c="cyan", edgecolors="black", zorder=5, label="Receivers",
        )
    if source_xy is not None:
        ax.scatter(
            source_xy[0], source_xy[1],
            s=80, c="yellow", marker="*", edgecolors="black", zorder=6,
            label="Source",
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    if receivers is not None or source_xy is not None:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote wavefield plot to {output_path}")

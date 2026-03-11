"""Plotting utilities for velocity models and wavefields."""

from __future__ import annotations

import os
from pathlib import Path

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
        interpolation="bicubic",
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
        interpolation="bicubic",
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


# Reference pressure for dB SPL (threshold of hearing in air).
_P_REF = 20e-6  # 20 µPa


def _to_db_spl(p: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Convert pressure to dB SPL, floored at *floor_db* below peak."""
    mag = np.abs(p)
    mag = np.where(mag < _P_REF * 1e-6, _P_REF * 1e-6, mag)  # avoid log(0)
    db = 20.0 * np.log10(mag / _P_REF)
    db_max = float(np.max(db))
    return np.clip(db, db_max + floor_db, None)


# ---------------------------------------------------------------------------
# FDTD gather plot
# ---------------------------------------------------------------------------

def plot_gather(
    traces: np.ndarray,
    dt: float,
    output_path: str = "gather.png",
    title: str = "Receiver Gather",
    db_range: float = 60.0,
    cmap: str = "inferno",
) -> None:
    """Plot receiver traces as a dB SPL gather.

    Parameters
    ----------
    traces : np.ndarray, shape ``(n_receivers, n_samples)``
    dt : float
        Timestep [s] between samples.
    db_range : float
        Dynamic range in dB below peak to display.
    """
    n_recv, n_samp = traces.shape
    t_axis = np.arange(n_samp) * dt

    db = _to_db_spl(traces, floor_db=-db_range)
    db_max = float(np.max(db))
    db_min = db_max - db_range

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        db.T,
        aspect="auto",
        cmap=cmap,
        vmin=db_min,
        vmax=db_max,
        origin="upper",
        extent=[0, n_recv, t_axis[-1], t_axis[0]],
        interpolation="bicubic",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("SPL [dB re 20 µPa]")
    ax.set_xlabel("Receiver index")
    ax.set_ylabel("Time [s]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote gather plot to {output_path}")


# ---------------------------------------------------------------------------
# FDTD wavefield snapshot
# ---------------------------------------------------------------------------

def save_snapshot(
    model: VelocityModel,
    field: np.ndarray,
    step: int,
    output_dir: str,
    receivers: np.ndarray | None = None,
    source_xy: np.ndarray | None = None,
    db_range: float = 60.0,
    title: str | None = None,
) -> None:
    """Save a single wavefield snapshot as a numbered PNG in dB SPL.

    File is written to ``{output_dir}/snapshot_{step:06d}.png``.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, f"snapshot_{step:06d}.png")

    db = _to_db_spl(field, floor_db=-db_range)
    db_max = float(np.max(db))
    db_min = db_max - db_range

    ext = model.extent
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        db,
        origin="lower",
        extent=[ext[0], ext[1], ext[2], ext[3]],
        cmap="inferno",
        aspect="equal",
        vmin=db_min,
        vmax=db_max,
        interpolation="bicubic",
    )
    fig.colorbar(im, ax=ax, label="SPL [dB re 20 µPa]")
    if receivers is not None:
        ax.scatter(
            receivers[:, 0], receivers[:, 1],
            s=12, c="cyan", edgecolors="black", linewidths=0.3, zorder=5,
        )
    if source_xy is not None:
        ax.scatter(
            source_xy[0], source_xy[1],
            s=60, c="yellow", marker="*", edgecolors="black", zorder=6,
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title or f"Step {step}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Domain visualisation (velocity + attenuation + wind + receivers)
# ---------------------------------------------------------------------------

def plot_domain(
    model: VelocityModel,
    output_path: str = "domain.png",
    receivers: np.ndarray | None = None,
    source_xy: np.ndarray | None = None,
    source_path: np.ndarray | None = None,
    attenuation: np.ndarray | None = None,
    wind_vx: float = 0.0,
    wind_vy: float = 0.0,
    title: str = "Domain",
) -> None:
    """Plot the velocity model with optional overlays.

    * Semi-transparent green overlay where *attenuation > 0* (vegetation).
    * A quiver arrow showing wind direction / magnitude.
    * An arrowed dashed line for *source_path* (shape ``(N, 2)``).
    * Receiver and source markers.
    """
    ext = model.extent
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        model.values,
        origin="lower",
        extent=[ext[0], ext[1], ext[2], ext[3]],
        cmap="terrain",
        aspect="equal",
        interpolation="bicubic",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wave speed [m/s]")

    # Vegetation overlay.
    if attenuation is not None:
        veg_mask = np.ma.masked_where(attenuation < 1e-6, attenuation)
        ax.imshow(
            veg_mask,
            origin="lower",
            extent=[ext[0], ext[1], ext[2], ext[3]],
            cmap="Greens",
            alpha=0.45,
            aspect="equal",
        )

    # Wind arrow.
    if abs(wind_vx) > 1e-6 or abs(wind_vy) > 1e-6:
        # Place arrow in the upper-left corner of the domain.
        cx = ext[0] + 0.12 * (ext[1] - ext[0])
        cy = ext[3] - 0.10 * (ext[3] - ext[2])
        mag = (wind_vx**2 + wind_vy**2) ** 0.5
        scale = 0.10 * (ext[1] - ext[0]) / max(mag, 1e-8)
        ax.annotate(
            "",
            xy=(cx + wind_vx * scale, cy + wind_vy * scale),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color="white", lw=2),
        )
        ax.text(
            cx, cy - 0.04 * (ext[3] - ext[2]),
            f"Wind {mag:.1f} m/s",
            color="white", fontsize=7, ha="center",
        )

    if receivers is not None:
        ax.scatter(
            receivers[:, 0], receivers[:, 1],
            s=20, c="cyan", edgecolors="black", linewidths=0.4,
            zorder=5, label="Receivers",
        )
    # Source path arrows for moving sources.
    if source_path is not None and len(source_path) >= 2:
        ax.plot(
            source_path[:, 0], source_path[:, 1],
            ls="--", lw=1.4, color="yellow", alpha=0.8, zorder=5,
        )
        # Draw arrowheads along the path.
        n_arrows = min(5, len(source_path) - 1)
        idxs = np.linspace(0, len(source_path) - 2, n_arrows, dtype=int)
        for i in idxs:
            dx = source_path[i + 1, 0] - source_path[i, 0]
            dy = source_path[i + 1, 1] - source_path[i, 1]
            ax.annotate(
                "",
                xy=(source_path[i + 1, 0], source_path[i + 1, 1]),
                xytext=(source_path[i, 0], source_path[i, 1]),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.6),
            )
        # Mark start and end.
        ax.scatter(
            source_path[0, 0], source_path[0, 1],
            s=80, c="yellow", marker="*", edgecolors="black",
            zorder=6, label="Source start",
        )
        ax.scatter(
            source_path[-1, 0], source_path[-1, 1],
            s=60, c="orange", marker="s", edgecolors="black",
            zorder=6, label="Source end",
        )
    elif source_xy is not None:
        ax.scatter(
            source_xy[0], source_xy[1],
            s=80, c="yellow", marker="*", edgecolors="black",
            zorder=6, label="Source",
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    if receivers is not None or source_xy is not None:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote domain plot to {output_path}")

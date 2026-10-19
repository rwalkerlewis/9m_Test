"""Wavefield snapshot writers used by the FDTD solvers (``snapshot_interval``).

Re-exported from :mod:`acoustic_sim.air.plotting`, their former home.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from acoustic_sim.model import VelocityModel

matplotlib.use("Agg")


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
# 3-D wavefield snapshot (two-panel: X-Y slab + X-Z cross-section)
# ---------------------------------------------------------------------------

def save_snapshot_3d(
    field_3d: np.ndarray,
    step: int,
    output_dir: str,
    *,
    extent_xy: tuple[float, float, float, float],
    extent_xz: tuple[float, float, float, float],
    z_index: int | None = None,
    y_index: int | None = None,
    receivers: np.ndarray | None = None,
    source_xyz: np.ndarray | None = None,
    db_range: float = 60.0,
    title: str | None = None,
) -> None:
    """Save a two-panel wavefield snapshot as a numbered PNG.

    Left panel: X-Y slice at the given *z_index* (default: middle).
    Right panel: X-Z slice at the given *y_index* (default: middle).

    Parameters
    ----------
    field_3d : ndarray, shape ``(nz, ny, nx)``
        Full 3-D pressure field.
    step : int
        Current timestep number (used in filename and title).
    output_dir : str
        Directory for output PNGs (created if needed).
    extent_xy / extent_xz : tuple
        Imshow extents for each panel.
    z_index / y_index : int or None
        Slice indices. ``None`` → middle of that axis.
    receivers : ndarray (n_recv, 3) or None
    source_xyz : ndarray (3,) or None
    db_range : float
        Dynamic range in dB below peak.
    title : str or None
    """
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    nz, ny, nx = field_3d.shape
    if z_index is None:
        z_index = nz // 2
    if y_index is None:
        y_index = ny // 2
    z_index = min(z_index, nz - 1)
    y_index = min(y_index, ny - 1)

    xy_slice = field_3d[z_index, :, :]  # (ny, nx)
    xz_slice = field_3d[:, y_index, :]  # (nz, nx)

    db_xy = _to_db_spl(xy_slice, floor_db=-db_range)
    db_xz = _to_db_spl(xz_slice, floor_db=-db_range)
    vmax = max(float(np.max(db_xy)), float(np.max(db_xz)))
    vmin = vmax - db_range

    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(16, 6))

    # ── X-Y plan view ──────────────────────────────────────────────────
    im1 = ax_xy.imshow(
        db_xy, origin="lower",
        extent=list(extent_xy),
        cmap="inferno", aspect="equal",
        vmin=vmin, vmax=vmax, interpolation="bicubic",
    )
    fig.colorbar(im1, ax=ax_xy, label="SPL [dB re 20 µPa]")
    if receivers is not None:
        ax_xy.scatter(receivers[:, 0], receivers[:, 1],
                      s=12, c="cyan", edgecolors="black", linewidths=0.3, zorder=5)
    if source_xyz is not None:
        ax_xy.scatter(source_xyz[0], source_xyz[1],
                      s=60, c="yellow", marker="*", edgecolors="black", zorder=6)
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title(f"X-Y slice at z-index {z_index}")

    # ── X-Z elevation view ─────────────────────────────────────────────
    im2 = ax_xz.imshow(
        db_xz, origin="lower",
        extent=list(extent_xz),
        cmap="inferno", aspect="auto",
        vmin=vmin, vmax=vmax, interpolation="bicubic",
    )
    fig.colorbar(im2, ax=ax_xz, label="SPL [dB re 20 µPa]")
    if source_xyz is not None:
        ax_xz.scatter(source_xyz[0], source_xyz[2],
                       s=60, c="yellow", marker="*", edgecolors="black", zorder=6)
    ax_xz.set_xlabel("x [m]")
    ax_xz.set_ylabel("z [m]")
    ax_xz.set_title(f"X-Z slice at y-index {y_index}")

    fig.suptitle(title or f"Step {step}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = str(Path(output_dir) / f"snapshot3d_{step:06d}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

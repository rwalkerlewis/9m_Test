"""Plotting utilities for velocity models and wavefields."""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

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


# =====================================================================
#  Detection / tracking / fire-control visualisations
# =====================================================================

def plot_detection_domain(
    model: VelocityModel,
    receivers: np.ndarray,
    source_positions: np.ndarray,
    weapon_pos: tuple[float, float] | np.ndarray | None = None,
    stationary_pos: tuple[float, float] | np.ndarray | None = None,
    output_path: str = "detection_domain.png",
    title: str = "Detection Domain",
) -> None:
    """Domain overview with microphones, drone trajectory, and weapon.

    Parameters
    ----------
    model : VelocityModel
    receivers : (n_mics, 2)
    source_positions : (n_steps, 2)   true drone positions
    weapon_pos : (2,) or None
    stationary_pos : (2,) or None
    """
    ext = model.extent
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        model.values, origin="lower",
        extent=[ext[0], ext[1], ext[2], ext[3]],
        cmap="terrain", aspect="equal", interpolation="bicubic",
    )
    fig.colorbar(im, ax=ax, label="Wave speed [m/s]")

    # Microphones.
    ax.scatter(receivers[:, 0], receivers[:, 1],
               s=40, c="cyan", marker="^", edgecolors="black",
               linewidths=0.5, zorder=5, label="Microphones")

    # Drone trajectory.
    ax.plot(source_positions[:, 0], source_positions[:, 1],
            ls="--", lw=1.5, color="yellow", alpha=0.8, zorder=5,
            label="Drone trajectory")
    ax.scatter(source_positions[0, 0], source_positions[0, 1],
               s=70, c="yellow", marker="*", edgecolors="black", zorder=6)

    # Weapon.
    if weapon_pos is not None:
        wp = np.asarray(weapon_pos)
        ax.scatter(wp[0], wp[1], s=200, c="red", marker="*",
                   edgecolors="black", zorder=7, label="Weapon")

    # Stationary noise source.
    if stationary_pos is not None:
        sp = np.asarray(stationary_pos)
        ax.scatter(sp[0], sp[1], s=80, c="magenta", marker="s",
                   edgecolors="black", zorder=6, label="Stationary src")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote detection domain plot to {output_path}")


def plot_detection_gather(
    traces: np.ndarray,
    filtered_traces: np.ndarray,
    dt: float,
    output_path: str = "detection_gather.png",
    title: str = "Receiver Gather",
    db_range: float = 60.0,
) -> None:
    """Two-panel seismological gather: raw (left) and filtered (right).

    Parameters
    ----------
    traces : (n_mics, n_samples) raw traces.
    filtered_traces : (n_mics, n_samples) bandpass-filtered.
    dt : float
    """
    n_recv, n_samp = traces.shape
    t_axis = np.arange(n_samp) * dt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for ax, data, label in [(ax1, traces, "Raw"),
                             (ax2, filtered_traces, "Filtered")]:
        db = _to_db_spl(data, floor_db=-db_range)
        db_max = float(np.max(db))
        db_min = db_max - db_range
        im = ax.imshow(
            db.T, aspect="auto", cmap="inferno",
            vmin=db_min, vmax=db_max, origin="upper",
            extent=[0, n_recv, t_axis[-1], t_axis[0]],
            interpolation="bicubic",
        )
        fig.colorbar(im, ax=ax, label="SPL [dB re 20 µPa]")
        ax.set_xlabel("Receiver index")
        ax.set_title(label)

    ax1.set_ylabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote detection gather to {output_path}")


def plot_beam_power(
    results: list[dict],
    true_positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    output_path: str = "beam_power.png",
    n_panels: int = 9,
) -> None:
    """Multi-panel beam-power / coherence snapshots.

    Parameters
    ----------
    results : list of detection dicts (from processor).
    true_positions : (n_steps, 2) true drone positions at each time step.
    grid_x, grid_y : 1-D MFP grid coordinates.
    n_panels : int
        Number of panels to show.
    """
    n = len(results)
    if n == 0:
        return
    n_panels = min(n_panels, n)
    indices = np.linspace(0, n - 1, n_panels, dtype=int)

    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    # Track history for blue line.
    est_xs, est_ys = [], []

    for panel, idx in enumerate(indices):
        r = results[idx]
        row, col = divmod(panel, ncols)
        ax = axes[row, col]

        bpm = r["beam_power_map"]
        ax.imshow(
            bpm.T, origin="lower", aspect="equal",
            extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
            cmap="hot", interpolation="bicubic",
        )

        # True position (interpolate to detection time).
        t = r["time"]
        n_steps_true = true_positions.shape[0]
        frac = t  # we'll use time directly if available
        # Find closest true position by time index.
        total_det = len(results)
        true_idx = min(int(idx * n_steps_true / max(total_det, 1)),
                       n_steps_true - 1)
        ax.plot(true_positions[true_idx, 0], true_positions[true_idx, 1],
                "ro", ms=5, label="True" if panel == 0 else "")

        # Estimated position.
        if r["detected"]:
            ax.plot(r["x"], r["y"], "bx", ms=6, mew=2,
                    label="Est" if panel == 0 else "")
            est_xs.append(r["x"])
            est_ys.append(r["y"])

        # Track history.
        if len(est_xs) > 1:
            ax.plot(est_xs, est_ys, "b-", lw=0.8, alpha=0.6)

        ax.set_title(f"t={t:.2f}s", fontsize=8)
        ax.tick_params(labelsize=6)

    # Hide unused axes.
    for panel in range(n_panels, nrows * ncols):
        row, col = divmod(panel, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Beam Power / Coherence Maps")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote beam power plot to {output_path}")


def plot_tracking(
    track: dict,
    true_positions: np.ndarray,
    true_times: np.ndarray,
    fire_control: dict,
    weapon_pos: tuple[float, float] | np.ndarray,
    output_path: str = "tracking.png",
) -> None:
    """Four-panel tracking and fire-control display.

    Panels: bearing vs time, range vs time, lead angle vs time,
    engagement status vs time.
    """
    wp = np.asarray(weapon_pos, dtype=np.float64)
    t_track = track["times"]
    pos_track = track["positions"]

    # True bearing & range vs time.
    true_dx = true_positions[:, 0] - wp[0]
    true_dy = true_positions[:, 1] - wp[1]
    true_bearing = np.degrees(np.arctan2(true_dy, true_dx))
    true_range = np.sqrt(true_dx ** 2 + true_dy ** 2)

    # Estimated bearing & range.
    est_dx = pos_track[:, 0] - wp[0]
    est_dy = pos_track[:, 1] - wp[1]
    est_bearing = np.degrees(np.arctan2(est_dy, est_dx))
    est_range = np.sqrt(est_dx ** 2 + est_dy ** 2)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Panel 1: Bearing.
    ax = axes[0]
    ax.plot(true_times, true_bearing, "r-", lw=1, label="True")
    valid = ~np.isnan(pos_track[:, 0])
    ax.plot(t_track[valid], est_bearing[valid], "b.", ms=3, label="Estimated")
    ax.set_ylabel("Bearing [deg]")
    ax.legend(fontsize=7)
    ax.set_title("Bearing to Target")

    # Panel 2: Range.
    ax = axes[1]
    ax.plot(true_times, true_range, "r-", lw=1, label="True")
    ax.plot(t_track[valid], est_range[valid], "b.", ms=3, label="Estimated")
    ax.set_ylabel("Range [m]")
    ax.legend(fontsize=7)
    ax.set_title("Range to Target")

    # Panel 3: Lead angle.
    ax = axes[2]
    leads = np.degrees(fire_control["lead_angles"])
    valid_fc = ~np.isnan(leads)
    ax.plot(fire_control["times"][valid_fc], leads[valid_fc], "g.-", ms=3,
            lw=0.8, label="Lead angle")
    ax.set_ylabel("Lead angle [deg]")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.legend(fontsize=7)
    ax.set_title("Lead Angle")

    # Panel 4: Engagement envelope.
    ax = axes[3]
    cf = fire_control["can_fire"]
    t_fc = fire_control["times"]
    ax.fill_between(t_fc, 0, cf.astype(float),
                    step="mid", alpha=0.4, color="green", label="FIRE")
    ax.fill_between(t_fc, 0, (~cf).astype(float),
                    step="mid", alpha=0.3, color="red", label="NO FIRE")
    ax.set_ylim(-0.1, 1.2)
    ax.set_ylabel("Engagement")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=7)
    ax.set_title("Engagement Envelope")

    fig.suptitle("Tracking & Fire Control")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote tracking plot to {output_path}")


def plot_vespagram(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    output_path: str = "vespagram.png",
    sound_speed: float = 343.0,
    slowness_range: tuple[float, float] | None = None,
    n_slowness: int = 101,
    title: str = "Vespagram",
) -> None:
    """Vespagram: beam power as a function of slowness and time.

    Parameters
    ----------
    traces : (n_mics, n_samples)
    mic_positions : (n_mics, 2)
    dt : float
    sound_speed : float
        Reference speed for slowness axis centre.
    slowness_range : (s_min, s_max) in s/m, or None (auto).
    n_slowness : int
        Number of slowness values to sweep.
    """
    n_mics, n_samples = traces.shape

    # Reference slowness and range.
    s_ref = 1.0 / sound_speed
    if slowness_range is None:
        slowness_range = (0.0, 2.0 * s_ref)
    slownesses = np.linspace(slowness_range[0], slowness_range[1], n_slowness)

    # Array reference point and offsets.
    ref = np.mean(mic_positions, axis=0)
    dx = mic_positions - ref  # (n_mics, 2)

    # Use x-component of offset for plane-wave delay (1-D projection).
    # For a 2-D array, project along the principal axis.
    _, _, Vt = np.linalg.svd(dx, full_matrices=False)
    proj = dx @ Vt[0]  # (n_mics,) — signed distance along principal axis

    # Window: slide through time with 50% overlap.
    win_len = max(int(0.05 / dt), 32)
    hop = win_len // 2
    n_windows = max((n_samples - win_len) // hop, 1)

    vespa = np.zeros((n_slowness, n_windows))

    for wi in range(n_windows):
        w_start = wi * hop
        for si, s in enumerate(slownesses):
            stack = np.zeros(win_len)
            for m in range(n_mics):
                delay_samples = int(round(s * proj[m] / dt))
                idx_start = w_start - delay_samples
                idx_end = idx_start + win_len
                if idx_start < 0 or idx_end > n_samples:
                    continue
                stack += traces[m, idx_start:idx_end]
            vespa[si, wi] = np.sum(stack ** 2)

    # Normalise columns.
    col_max = np.max(vespa, axis=0, keepdims=True)
    col_max = np.where(col_max < 1e-30, 1.0, col_max)
    vespa /= col_max

    t_axis = np.arange(n_windows) * hop * dt

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        vespa, origin="lower", aspect="auto",
        extent=[t_axis[0], t_axis[-1], slownesses[0] * 1e3,
                slownesses[-1] * 1e3],
        cmap="hot", interpolation="bicubic",
    )
    fig.colorbar(im, ax=ax, label="Normalised beam power")
    ax.axhline(s_ref * 1e3, color="cyan", ls="--", lw=0.8,
               label=f"1/c = {s_ref*1e3:.2f} ms/m")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Slowness [ms/m]")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote vespagram to {output_path}")


# =====================================================================
#  Study comparison and multi-track plots
# =====================================================================

def plot_study_comparison(
    labels: list[str],
    metrics: dict[str, list[float]],
    output_path: str = "study_comparison.png",
    title: str = "Study Comparison",
) -> None:
    """Bar-chart comparison of metrics across study cases.

    Parameters
    ----------
    labels : list of case names (x-axis categories).
    metrics : dict mapping metric name → list of values (one per case).
    """
    n_metrics = len(metrics)
    n_cases = len(labels)
    if n_metrics == 0 or n_cases == 0:
        return

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_cases)
    for ax, (name, values) in zip(axes, metrics.items()):
        vals = [v if np.isfinite(v) else 0.0 for v in values]
        ax.bar(x, vals, color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(name)
        ax.set_title(name, fontsize=9)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote study comparison to {output_path}")


def plot_multi_track(
    tracks: list[dict],
    true_positions_list: list[np.ndarray] | None = None,
    output_path: str = "multi_track.png",
    title: str = "Multi-Target Tracking",
) -> None:
    """Spatial plot of multiple tracks + optional true trajectories.

    Parameters
    ----------
    tracks : list of track dicts from MultiTargetTracker.
    true_positions_list : list of (n_steps, 2) arrays, one per true source.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    colours = plt.cm.tab10.colors  # type: ignore[attr-defined]

    if true_positions_list:
        for i, tp in enumerate(true_positions_list):
            c = colours[i % len(colours)]
            ax.plot(tp[:, 0], tp[:, 1], "--", color=c, lw=1, alpha=0.5,
                    label=f"True {i}")

    for i, tr in enumerate(tracks):
        c = colours[i % len(colours)]
        pos = tr["positions"]
        ax.plot(pos[:, 0], pos[:, 1], ".-", color=c, lw=1.5, ms=4,
                label=f"Track {tr.get('track_id', i)}")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote multi-track plot to {output_path}")


# =====================================================================
#  Polar beam power plot
# =====================================================================

def plot_polar_beam_power(
    results: list[dict],
    azimuths: np.ndarray,
    ranges: np.ndarray,
    true_positions: np.ndarray,
    array_center: tuple[float, float] = (500.0, 500.0),
    output_path: str = "polar_beam_power.png",
    n_panels: int = 6,
) -> None:
    """Multi-panel polar beam-power maps.

    Each panel shows the beam power map for one time window in
    polar coordinates (azimuth vs range), with the true and
    estimated source positions.
    """
    n = len(results)
    if n == 0:
        return
    n_panels = min(n_panels, n)
    indices = np.linspace(0, n - 1, n_panels, dtype=int)

    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_panels == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    cx, cy = array_center
    az_deg = np.degrees(azimuths)

    for panel, idx in enumerate(indices):
        r = results[idx]
        row, col = divmod(panel, ncols)
        ax = axes[row, col]

        bpm = r.get("beam_power_map", None)
        if bpm is None:
            continue

        ax.imshow(
            bpm.T, origin="lower", aspect="auto",
            extent=[az_deg[0], az_deg[-1], ranges[0], ranges[-1]],
            cmap="hot", interpolation="bicubic",
        )

        # True position → polar.
        n_true = true_positions.shape[0]
        true_idx = min(int(idx * n_true / max(n, 1)), n_true - 1)
        tp = true_positions[true_idx]
        true_az = np.degrees(np.arctan2(tp[1] - cy, tp[0] - cx)) % 360
        true_rng = np.hypot(tp[0] - cx, tp[1] - cy)
        ax.plot(true_az, true_rng, "ro", ms=5)

        # Estimated.
        if r["detected"]:
            ax.plot(r.get("bearing_deg", 0), r.get("range", 0), "cx", ms=6, mew=2)

        ax.set_title(f"t={r['time']:.2f}s", fontsize=8)
        ax.set_xlabel("Azimuth [°]")
        if col == 0:
            ax.set_ylabel("Range [m]")
        ax.tick_params(labelsize=6)

    for panel in range(n_panels, nrows * ncols):
        row, col = divmod(panel, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Polar Beam Power (MVDR)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote polar beam power plot to {output_path}")


# ═════════════════════════════════════════════════════════════════════
# Merged from plotting_3d
# ═════════════════════════════════════════════════════════════════════

def plot_3d_trajectory(
    true_positions: np.ndarray,
    estimated_positions: np.ndarray | None = None,
    mic_positions: np.ndarray | None = None,
    weapon_pos: np.ndarray | None = None,
    title: str = "3D Trajectory",
    output_path: str = "trajectory_3d.png",
) -> None:
    """Plot 3D source trajectory with optional estimated track.

    Parameters
    ----------
    true_positions : (N, 3) — true (x, y, z) positions.
    estimated_positions : (M, 3) or None — estimated positions.
    mic_positions : (n_mics, 3) or None.
    weapon_pos : (3,) or None.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # True trajectory.
    ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2],
            "g-", lw=2, label="True trajectory")
    ax.scatter(*true_positions[0], c="g", s=100, marker="o", zorder=5)
    ax.scatter(*true_positions[-1], c="g", s=100, marker="s", zorder=5)

    # Estimated track.
    if estimated_positions is not None:
        valid = ~np.isnan(estimated_positions).any(axis=1)
        if np.any(valid):
            ep = estimated_positions[valid]
            ax.plot(ep[:, 0], ep[:, 1], ep[:, 2],
                    "b.-", lw=1, ms=3, label="Estimated")

    # Microphones.
    if mic_positions is not None:
        mp = np.asarray(mic_positions)
        if mp.shape[1] == 2:
            mp = np.column_stack([mp, np.zeros(mp.shape[0])])
        ax.scatter(mp[:, 0], mp[:, 1], mp[:, 2],
                   c="cyan", s=40, marker="^", label="Microphones")

    # Weapon.
    if weapon_pos is not None:
        wp = np.asarray(weapon_pos)
        if len(wp) == 2:
            wp = np.array([wp[0], wp[1], 0.0])
        ax.scatter(wp[0], wp[1], wp[2],
                   c="red", s=200, marker="*", label="Weapon")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote 3D trajectory plot to {output_path}")


def plot_altitude_vs_time(
    times: np.ndarray,
    true_z: np.ndarray,
    estimated_z: np.ndarray | None = None,
    estimated_times: np.ndarray | None = None,
    title: str = "Altitude vs Time",
    output_path: str = "altitude_time.png",
) -> None:
    """Plot altitude over time for true and estimated trajectories."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, true_z, "g-", lw=2, label="True altitude")
    if estimated_z is not None and estimated_times is not None:
        valid = ~np.isnan(estimated_z)
        if np.any(valid):
            ax.plot(estimated_times[valid], estimated_z[valid],
                    "b.", ms=4, label="Estimated altitude")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote altitude plot to {output_path}")


def plot_tracking_3d(
    track: dict,
    true_positions: np.ndarray,
    true_times: np.ndarray,
    fire_control: dict,
    weapon_pos: tuple | np.ndarray,
    output_path: str = "tracking_3d.png",
    maneuver_labels: list[str] | None = None,
    class_label: str | None = None,
    class_confidence: float | None = None,
) -> None:
    """Six-panel tracking and fire-control display with altitude and maneuver.

    Panels: bearing, range, altitude, lead angle, engagement, maneuver/class.
    """
    wp = np.asarray(weapon_pos, dtype=np.float64)
    if len(wp) == 2:
        wp = np.array([wp[0], wp[1], 0.0])
    t_track = track["times"]
    pos_track = track["positions"]

    # True values.
    true_dx = true_positions[:, 0] - wp[0]
    true_dy = true_positions[:, 1] - wp[1]
    true_dz = true_positions[:, 2] - wp[2] if true_positions.shape[1] > 2 else np.zeros(len(true_dx))
    true_bearing = np.degrees(np.arctan2(true_dy, true_dx))
    true_range = np.sqrt(true_dx ** 2 + true_dy ** 2 + true_dz ** 2)
    true_z = true_positions[:, 2] if true_positions.shape[1] > 2 else np.zeros(len(true_dx))

    # Estimated values.
    est_dx = pos_track[:, 0] - wp[0]
    est_dy = pos_track[:, 1] - wp[1]
    est_dz = pos_track[:, 2] - wp[2] if pos_track.shape[1] > 2 else np.zeros(len(est_dx))
    est_bearing = np.degrees(np.arctan2(est_dy, est_dx))
    est_range = np.sqrt(est_dx ** 2 + est_dy ** 2 + est_dz ** 2)
    est_z = pos_track[:, 2] if pos_track.shape[1] > 2 else np.zeros(len(est_dx))

    valid = ~np.isnan(pos_track[:, 0])

    n_panels = 6 if maneuver_labels or class_label else 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=True)

    # Panel 1: Bearing.
    ax = axes[0]
    ax.plot(true_times, true_bearing, "r-", lw=1, label="True")
    ax.plot(t_track[valid], est_bearing[valid], "b.", ms=3, label="Estimated")
    ax.set_ylabel("Bearing [deg]")
    ax.legend(fontsize=7)
    ax.set_title("Bearing to Target")

    # Panel 2: Range.
    ax = axes[1]
    ax.plot(true_times, true_range, "r-", lw=1, label="True")
    ax.plot(t_track[valid], est_range[valid], "b.", ms=3, label="Estimated")
    ax.set_ylabel("Range [m]")
    ax.legend(fontsize=7)
    ax.set_title("Range to Target")

    # Panel 3: Altitude.
    ax = axes[2]
    ax.plot(true_times, true_z, "r-", lw=1, label="True")
    ax.plot(t_track[valid], est_z[valid], "b.", ms=3, label="Estimated")
    ax.set_ylabel("Altitude [m]")
    ax.legend(fontsize=7)
    ax.set_title("Target Altitude")

    # Panel 4: Lead angle.
    ax = axes[3]
    leads = np.degrees(fire_control["lead_angles"])
    valid_fc = ~np.isnan(leads)
    ax.plot(fire_control["times"][valid_fc], leads[valid_fc], "g.-", ms=3,
            lw=0.8, label="Lead angle (az)")
    if "lead_angles_el" in fire_control:
        leads_el = np.degrees(fire_control["lead_angles_el"])
        valid_el = ~np.isnan(leads_el)
        ax.plot(fire_control["times"][valid_el], leads_el[valid_el], "m.-",
                ms=3, lw=0.8, label="Lead angle (el)")
    ax.set_ylabel("Lead angle [deg]")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.legend(fontsize=7)
    ax.set_title("Lead Angle")

    # Panel 5: Engagement.
    ax = axes[4]
    cf = fire_control["can_fire"]
    t_fc = fire_control["times"]
    ax.fill_between(t_fc, 0, cf.astype(float),
                    step="mid", alpha=0.4, color="green", label="FIRE")
    ax.fill_between(t_fc, 0, (~cf).astype(float),
                    step="mid", alpha=0.3, color="red", label="NO FIRE")
    ax.set_ylim(-0.1, 1.2)
    ax.set_ylabel("Engagement")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=7)
    ax.set_title("Engagement Envelope")

    # Panel 6: Maneuver classification + source class (optional).
    if n_panels == 6:
        ax = axes[5]
        if maneuver_labels:
            # Map maneuver labels to numeric codes for step plot.
            maneuver_names = sorted(set(maneuver_labels))
            maneuver_map = {n: i for i, n in enumerate(maneuver_names)}
            maneuver_codes = [maneuver_map[m] for m in maneuver_labels]
            # Use track times or generate equally spaced times.
            if len(maneuver_codes) == len(t_track):
                ax.step(t_track, maneuver_codes, where="mid", lw=1.5,
                        color="purple", label="Maneuver")
            else:
                t_man = np.linspace(t_track[0], t_track[-1], len(maneuver_codes))
                ax.step(t_man, maneuver_codes, where="mid", lw=1.5,
                        color="purple", label="Maneuver")
            ax.set_yticks(range(len(maneuver_names)))
            ax.set_yticklabels(maneuver_names, fontsize=7)
        if class_label:
            conf_str = f" ({class_confidence:.2f})" if class_confidence else ""
            ax.axhline(0, color="gray", ls="--", lw=0.5)
            ax.text(0.02, 0.95, f"Class: {class_label}{conf_str}",
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Maneuver / Class")
        ax.set_title("Maneuver Detection & Source Classification")
        ax.legend(fontsize=7)

    fig.suptitle("3D Tracking & Fire Control")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote 3D tracking plot to {output_path}")


def plot_kinematic_scatter(
    features_by_class: dict[str, np.ndarray],
    feature_names: tuple[str, str],
    title: str = "Kinematic Feature Scatter",
    output_path: str = "kinematic_scatter.png",
) -> None:
    """2D scatter plot of kinematic features colored by class.

    Parameters
    ----------
    features_by_class : dict mapping class_name → (N, 2) array of feature pairs.
    feature_names : (x_name, y_name).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors
    for i, (cls, feats) in enumerate(features_by_class.items()):
        c = colors[i % len(colors)]
        ax.scatter(feats[:, 0], feats[:, 1], c=[c], s=20, alpha=0.6, label=cls)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote kinematic scatter plot to {output_path}")


# ---------------------------------------------------------------------------
# 3-D wavefield snapshot (two-panel: X-Y slab + X-Z cross-section)
# ---------------------------------------------------------------------------

_P_REF = 20e-6  # 20 µPa


def _to_db_spl(p: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Convert pressure to dB SPL, floored at *floor_db* below peak."""
    mag = np.abs(p)
    mag = np.where(mag < _P_REF * 1e-6, _P_REF * 1e-6, mag)
    db = 20.0 * np.log10(mag / _P_REF)
    db_max = float(np.max(db))
    return np.clip(db, db_max + floor_db, None)


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


# ====================================================================
# Pipeline engagement / diagnostic plots
# ====================================================================

def plot_radial_engagement(
    fire_decisions: list[dict],
    ground_truth_fn,
    source_duration: float,
    weapon_pos: np.ndarray,
    is_3d: bool,
    cfg_fc: dict,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Radial engagement plot (plan-view + elevation for 3-D).

    Projectile trajectories are drawn to the CPA point for hits and
    extended to maximum pellet range for misses.
    """
    from acoustic_sim.fire_control import (
        projectile_path,
        time_of_flight,
    )

    muzzle_velocity = cfg_fc["muzzle_velocity_mps"]
    decel = cfg_fc["pellet_decel_mps2"]
    hit_threshold = cfg_fc["hit_threshold_m"]
    wx, wy, wz = float(weapon_pos[0]), float(weapon_pos[1]), float(weapon_pos[2])
    ncols = 2 if is_3d else 1
    fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 10))
    if ncols == 1:
        axes = [axes]

    gt_times = np.linspace(0, source_duration, 200)
    gt_xyz = np.array([ground_truth_fn(t) for t in gt_times])
    gt_x = gt_xyz[:, 0] - wx
    gt_y = gt_xyz[:, 1] - wy

    ax_xy = axes[0]
    ax_xy.plot(gt_x, gt_y, "g-", lw=3, label="Target path", zorder=5)
    ax_xy.scatter(gt_x[0], gt_y[0], c="g", s=150, marker="o", zorder=6,
                  label="Start")
    ax_xy.scatter(gt_x[-1], gt_y[-1], c="g", s=150, marker="s", zorder=6,
                  label="End")

    max_range = max(float(np.max(np.sqrt(gt_x**2 + gt_y**2))), 50)
    for r in [25, 50, 75, 100]:
        if r <= max_range * 1.2:
            circle = Circle((0, 0), r, fill=False, color="gray", ls="--",
                            alpha=0.3)
            ax_xy.add_patch(circle)
            ax_xy.text(r * 0.707, r * 0.707, f"{r}m", fontsize=8,
                       color="gray", alpha=0.7)

    n_shots = n_hits = 0
    max_pellet_range = muzzle_velocity / decel
    for fd in fire_decisions:
        if not fd.get("can_fire"):
            continue
        aim_brg = fd.get("aim_bearing", float("nan"))
        aim_el = fd.get("aim_elevation", 0.0)
        tof = fd.get("tof", float("nan"))
        if np.isnan(aim_brg) or np.isnan(tof) or tof <= 0:
            continue
        n_shots += 1
        miss_dist = fd.get("miss", float("nan"))
        is_hit = fd.get("hit", False)
        if is_hit:
            n_hits += 1
        if is_hit:
            draw_range = fd.get("cpa_range", None)
        else:
            draw_range = max_pellet_range
        if draw_range is not None and draw_range > 0:
            draw_tof = time_of_flight(draw_range, muzzle_velocity, decel)
            if draw_tof == float("inf"):
                draw_tof = tof
        else:
            draw_tof = tof
        proj_x, proj_y, _ = projectile_path(
            weapon_pos, aim_brg, aim_el, muzzle_velocity, decel, draw_tof)
        color = "green" if is_hit else "red"
        ax_xy.plot(proj_x - wx, proj_y - wy, "-", color=color, lw=2,
                   alpha=0.8)
        ipos = fd.get("intercept_pos")
        if ipos is not None:
            ix, iy = ipos[0] - wx, ipos[1] - wy
            ax_xy.scatter(ix, iy, c=color, s=150, marker="x",
                          linewidths=3, zorder=10)
            label = (f"HIT ({miss_dist:.1f}m)" if is_hit
                     else f"MISS ({miss_dist:.1f}m)")
            ax_xy.annotate(label, (ix, iy), xytext=(5, 5),
                           textcoords="offset points", fontsize=9,
                           color=color, fontweight="bold")
        gt_pos = ground_truth_fn(fd["time"])
        ax_xy.scatter(gt_pos[0] - wx, gt_pos[1] - wy, c="lime", s=80,
                      marker="o", edgecolors="darkgreen", linewidths=2,
                      zorder=8)

    ax_xy.scatter(0, 0, c="black", s=300, marker="*", label="Weapon",
                  zorder=15)
    ax_xy.set_xlabel("X relative to weapon (m)")
    ax_xy.set_ylabel("Y relative to weapon (m)")
    ax_xy.set_title("PLAN VIEW (X-Y)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="upper left", fontsize=9)

    if is_3d:
        ax_xz = axes[1]
        gt_z = gt_xyz[:, 2] - wz
        ax_xz.plot(gt_x, gt_z, "g-", lw=3, label="Target path", zorder=5)
        ax_xz.scatter(gt_x[0], gt_z[0], c="g", s=150, marker="o", zorder=6)
        ax_xz.scatter(gt_x[-1], gt_z[-1], c="g", s=150, marker="s",
                      zorder=6)
        for fd in fire_decisions:
            if not fd.get("can_fire"):
                continue
            aim_brg = fd.get("aim_bearing", float("nan"))
            aim_el = fd.get("aim_elevation", 0.0)
            tof = fd.get("tof", float("nan"))
            if np.isnan(aim_brg) or np.isnan(tof) or tof <= 0:
                continue
            is_hit = fd.get("hit", False)
            if is_hit:
                draw_range = fd.get("cpa_range", None)
            else:
                draw_range = max_pellet_range
            if draw_range is not None and draw_range > 0:
                draw_tof = time_of_flight(draw_range, muzzle_velocity,
                                          decel)
                if draw_tof == float("inf"):
                    draw_tof = tof
            else:
                draw_tof = tof
            proj_x, _, proj_z = projectile_path(
                weapon_pos, aim_brg, aim_el, muzzle_velocity, decel,
                draw_tof)
            color = "green" if is_hit else "red"
            ax_xz.plot(proj_x - wx, proj_z - wz, "-", color=color, lw=2,
                       alpha=0.8)
            ipos = fd.get("intercept_pos")
            if ipos is not None:
                ax_xz.scatter(ipos[0] - wx, ipos[2] - wz, c=color, s=150,
                              marker="x", linewidths=3, zorder=10)
        ax_xz.scatter(0, 0, c="black", s=300, marker="*", zorder=15)
        ax_xz.set_xlabel("X relative to weapon (m)")
        ax_xz.set_ylabel("Z (altitude) relative to weapon (m)")
        ax_xz.set_title("ELEVATION VIEW (X-Z)")
        ax_xz.grid(True, alpha=0.3)
        ax_xz.legend(loc="upper left", fontsize=9)

    dim = "3-D" if is_3d else "2-D"
    fig.suptitle(
        f"RADIAL ENGAGEMENT -- {dim}  |  Shots: {n_shots}  Hits: {n_hits}  "
        f"Misses: {n_shots - n_hits}  (threshold < {hit_threshold} m)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    return fig


def plot_pipeline_summary(
    all_detections: list[dict],
    all_fire_decisions: list[dict],
    all_track_states: list,
    wall_times: np.ndarray,
    ground_truth_fn,
    source_duration: float,
    array_center: np.ndarray,
    weapon_pos: np.ndarray,
    is_3d: bool,
    hop_sec: float,
    hit_threshold: float,
    metrics: dict,
    output_path: Path | str,
) -> None:
    """6-panel real-time pipeline summary figure."""
    cx, cy = float(array_center[0]), float(array_center[1])
    gt_times = np.linspace(0, source_duration, 200)
    gt_xyz = np.array([ground_truth_fn(t) for t in gt_times])

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1 -- spatial overview X-Y
    ax = axes[0, 0]
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], "g-", lw=2, label="True path")
    det_xs = [d["x"] for d in all_detections
              if d.get("detected") and "x" in d]
    det_ys = [d["y"] for d in all_detections
              if d.get("detected") and "y" in d]
    if det_xs:
        ax.scatter(det_xs, det_ys, c="b", s=20, alpha=0.4,
                   label="Detections")
    track_xs = [fd["est_pos"][0] for fd in all_fire_decisions
                if fd.get("est_pos")]
    track_ys = [fd["est_pos"][1] for fd in all_fire_decisions
                if fd.get("est_pos")]
    if track_xs:
        ax.plot(track_xs, track_ys, "m-", lw=1.5, alpha=0.7,
                label="Causal track")
    for fd in all_fire_decisions:
        if fd["can_fire"] and "intercept_pos" in fd:
            ip = fd["intercept_pos"]
            color = "green" if fd.get("hit") else "red"
            ax.scatter(ip[0], ip[1], c=color, s=100, marker="x", zorder=10)
    ax.scatter(weapon_pos[0], weapon_pos[1], c="k", s=200, marker="*",
               label="Weapon/Array", zorder=15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Spatial Overview (X-Y)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Panel 2 -- bearing vs time
    ax = axes[0, 1]
    true_brgs = [
        math.degrees(math.atan2(
            ground_truth_fn(t)[1] - cy, ground_truth_fn(t)[0] - cx))
        for t in gt_times
    ]
    ax.plot(gt_times, true_brgs, "g-", lw=2, label="True")
    det_t = [d["time"] for d in all_detections
             if d.get("detected") and "bearing_deg" in d]
    det_brg = [
        d["bearing_deg"] - 360 if d["bearing_deg"] > 180
        else d["bearing_deg"]
        for d in all_detections
        if d.get("detected") and "bearing_deg" in d
    ]
    if det_t:
        ax.scatter(det_t, det_brg, c="b", s=20, alpha=0.4,
                   label="Detected")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bearing (deg)")
    ax.set_title("Bearing vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 -- miss distance vs time
    ax = axes[0, 2]
    shot_times = [f["time"] for f in all_fire_decisions
                  if f.get("miss") is not None]
    shot_misses = [f["miss"] for f in all_fire_decisions
                   if f.get("miss") is not None]
    if shot_times:
        colors_shot = ["green" if m < hit_threshold else "red"
                       for m in shot_misses]
        ax.scatter(shot_times, shot_misses, c=colors_shot, s=60,
                   marker="x", zorder=5)
        ax.axhline(hit_threshold, color="g", ls="--", alpha=0.7,
                   label=f"{hit_threshold}m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Miss Distance (m)")
    ax.set_title("Fire Control Miss Distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4 -- track error vs time
    ax = axes[1, 0]
    track_errors, te_times = [], []
    for fd in all_fire_decisions:
        if fd.get("est_pos") is not None:
            ep = np.array(fd["est_pos"])
            gt = np.asarray(ground_truth_fn(fd["time"]))
            track_errors.append(float(np.linalg.norm(ep - gt)))
            te_times.append(fd["time"])
    if track_errors:
        ax.plot(te_times, track_errors, "m-", lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Track Error (m)")
    ax.set_title("Track Position Error vs Time")
    ax.grid(True, alpha=0.3)

    # Panel 5 -- latency
    ax = axes[1, 1]
    win_times_ms = np.arange(len(wall_times)) * hop_sec
    ax.plot(win_times_ms, wall_times * 1e6, "b-", lw=0.5, alpha=0.6)
    ax.axhline(hop_sec * 1e6, color="r", ls="--",
               label=f"Real-time budget: {hop_sec * 1e3:.1f} ms")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Processing Time (us)")
    ax.set_title("Per-Window Latency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6 -- text summary
    ax = axes[1, 2]
    ax.axis("off")
    n_shots = metrics["shots_fired"]
    n_hits = metrics["n_hits"]
    mean_miss = metrics["mean_miss"]
    mean_brg = metrics["mean_bearing_error"]
    dim = "3-D" if is_3d else "2-D"
    summary = (
        f"PIPELINE SUMMARY ({dim})\n"
        f"{'=' * 35}\n"
        f"Detections:    {metrics['n_detections']}/{metrics['n_windows']}\n"
        f"Bearing err:   {mean_brg:.1f} deg\n"
        f"Track states:  {sum(1 for s in all_track_states if s)}\n"
        f"\n"
        f"Shots:         {n_shots}\n"
        f"Hits <{hit_threshold}m:     {n_hits} "
        f"({100 * n_hits / max(n_shots, 1):.1f}%)\n"
        f"Mean miss:     {mean_miss:.1f} m\n"
        f"\n"
        f"TIMING\n"
        f"{'=' * 35}\n"
        f"Hop cadence:   {hop_sec * 1e3:.1f} ms\n"
        f"Mean latency:  {wall_times.mean() * 1e6:.0f} us\n"
        f"Max latency:   {wall_times.max() * 1e6:.0f} us\n"
        f"RT margin:     {hop_sec / wall_times.mean():.0f}x\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(
        f"{dim} ENGAGEMENT  |  Shots: {n_shots}  "
        f"Hits: {n_hits}/{n_shots}  Mean miss: {mean_miss:.1f} m  "
        f"Latency: {wall_times.mean() * 1e6:.0f} us",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_beamformer_diagnostic(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    ground_truth_fn,
    source_duration: float,
    array_center: np.ndarray,
    beamformer,
    cfg_det: dict,
    output_path: Path | str | None = None,
) -> None:
    """4-panel diagnostic comparing RMS-weighted vs SRP-PHAT bearing."""
    n_mics, n_samples = traces.shape
    fs = 1.0 / dt
    cx, cy = float(array_center[0]), float(array_center[1])
    min_signal_rms = cfg_det["min_signal_rms"]

    mic_angles = np.array([
        math.atan2(mic_positions[i, 1] - cy, mic_positions[i, 0] - cx)
        for i in range(n_mics)
    ])

    win_len = max(int(round(cfg_det["window_length_s"] * fs)), 1)
    hop = max(int(round(win_len * (1.0 - cfg_det["window_overlap"]))), 1)

    times, true_bearings = [], []
    rms_bearings, srp_bearings = [], []
    rms_errors, srp_errors = [], []
    srp_power_maps, rms_values = [], []
    scan_angles: np.ndarray | None = None

    pos = 0
    while pos + win_len <= n_samples:
        t_center = (pos + win_len / 2.0) * dt
        seg = traces[:, pos:pos + win_len]
        window_rms = float(np.sqrt(np.mean(seg**2)))
        if window_rms < min_signal_rms:
            pos += hop
            continue

        gt_x, gt_y, _ = ground_truth_fn(t_center)
        true_brg = math.atan2(gt_y - cy, gt_x - cx)
        if true_brg < 0:
            true_brg += 2 * math.pi

        per_mic_rms = np.sqrt(np.mean(seg**2, axis=1))
        weights = per_mic_rms**2
        rms_brg = math.atan2(
            float(np.sum(weights * np.sin(mic_angles))),
            float(np.sum(weights * np.cos(mic_angles))))
        if rms_brg < 0:
            rms_brg += 2 * math.pi

        result = beamformer.estimate(seg, max_sources=1)
        srp_brg = (result.detections[0].bearing_rad
                   if result.detections else 0.0)
        srp_pow = result.spectrum
        if scan_angles is None and result.bearings_rad is not None:
            scan_angles = result.bearings_rad

        def angle_err_deg(est, true):
            d = math.degrees(est - true)
            return ((d + 180) % 360) - 180

        times.append(t_center)
        true_bearings.append(math.degrees(true_brg))
        rms_bearings.append(math.degrees(rms_brg))
        srp_bearings.append(math.degrees(srp_brg))
        rms_errors.append(angle_err_deg(rms_brg, true_brg))
        srp_errors.append(angle_err_deg(srp_brg, true_brg))
        srp_power_maps.append(srp_pow)
        rms_values.append(window_rms)
        pos += hop

    times = np.array(times)
    rms_errors = np.array(rms_errors)
    srp_errors = np.array(srp_errors)
    rms_values = np.array(rms_values)
    mean_rms_err = float(np.mean(np.abs(rms_errors)))
    mean_srp_err = float(np.mean(np.abs(srp_errors)))
    print(f"\n  [DIAGNOSTIC] Mean |bearing error|:")
    print(f"    RMS-weighted: {mean_rms_err:.1f} deg")
    print(f"    SRP-PHAT:     {mean_srp_err:.1f} deg")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax = axes[0, 0]
    ax.plot(times, true_bearings, "k-", lw=2, label="True bearing")
    ax.plot(times, rms_bearings, "r.", ms=4, alpha=0.6,
            label=f"RMS ({mean_rms_err:.1f} deg)")
    ax.plot(times, srp_bearings, "b.", ms=4, alpha=0.6,
            label=f"SRP-PHAT ({mean_srp_err:.1f} deg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bearing (deg)")
    ax.set_title("Bearing Estimates vs Time")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, rms_errors, "r-", lw=1.5, alpha=0.7,
            label=f"RMS ({mean_rms_err:.1f} deg)")
    ax.plot(times, srp_errors, "b-", lw=1.5, alpha=0.7,
            label=f"SRP-PHAT ({mean_srp_err:.1f} deg)")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bearing Error (deg)")
    ax.set_title("Bearing Error vs Time")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if srp_power_maps:
        power_map = np.array(srp_power_maps)
        scan_deg = np.degrees(scan_angles)
        row_max = power_map.max(axis=1, keepdims=True)
        row_max[row_max < 1e-12] = 1.0
        power_norm = power_map / row_max
        extent = [scan_deg[0], scan_deg[-1], times[-1], times[0]]
        im = ax.imshow(power_norm, aspect="auto", extent=extent,
                       cmap="hot", origin="upper")
        ax.plot(true_bearings, times, "c-", lw=2, label="True bearing")
        ax.set_xlabel("Bearing (deg)")
        ax.set_ylabel("Time (s)")
        ax.set_title("SRP-PHAT Power Map")
        ax.legend(fontsize=10)
        plt.colorbar(im, ax=ax, label="Normalised power")

    ax = axes[1, 1]
    if srp_power_maps:
        cpa_idx = int(np.argmax(rms_values))
        cpa_power = srp_power_maps[cpa_idx]
        ax.plot(scan_deg, cpa_power / cpa_power.max(), "b-", lw=2)
        ax.axvline(true_bearings[cpa_idx], color="k", ls="--", lw=2,
                   label="True")
        ax.axvline(srp_bearings[cpa_idx], color="b", ls=":", lw=2,
                   label="SRP-PHAT")
        ax.set_xlabel("Bearing (deg)")
        ax.set_ylabel("Normalised Power")
        ax.set_title(f"SRP-PHAT @ CPA (t = {times[cpa_idx]:.3f} s)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"BEAMFORMER DIAGNOSTIC  |  RMS: {mean_rms_err:.1f} deg "
        f"vs SRP-PHAT: {mean_srp_err:.1f} deg  |  {len(times)} windows",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

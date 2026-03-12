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

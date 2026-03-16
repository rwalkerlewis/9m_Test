"""3D visualization utilities for the drone detection system.

Provides 3D trajectory plots, altitude-vs-time subplots, and x-y
projection views.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


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

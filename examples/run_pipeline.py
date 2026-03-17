#!/usr/bin/env python3
"""Unified acoustic detection and engagement pipeline.

Loads FDTD simulation data, runs SRP-PHAT bearing estimation, causal
weighted-least-squares tracking, and instantaneous fire control.
Auto-detects 2-D vs 3-D from receiver positions in the metadata.

All tuneable parameters live in a JSON config file
(default: ``examples/pipeline.config.json``).  CLI flags override the
most-used values.

Usage::

    # 2-D valley (default config)
    python examples/run_pipeline.py output/valley_test

    # 3-D valley
    python examples/run_pipeline.py output/valley_3d_test

    # Custom config
    python examples/run_pipeline.py output/valley_3d_test \\
        --config examples/pipeline.config.json

    # CLI overrides
    python examples/run_pipeline.py output/valley_test \\
        --source-speed 60 --hit-threshold 3.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.fire_control import (
    compute_engagement_3d,
    compute_lead_3d,
)

# ============================================================================
# Configuration
# ============================================================================

_DEFAULTS = {
    "detection": {
        "window_length_s": 0.1,
        "window_overlap": 0.75,
        "min_signal_rms": 5e-5,
    },
    "beamformer": {
        "n_bearings": 360,
        "freq_lo_hz": 100.0,
        "freq_hi_hz": 2000.0,
    },
    "tracking": {
        "min_detections": 5,
        "ema_alpha": 0.35,
        "rms_fire_gate_frac": 0.20,
        "range_min_m": 5.0,
        "range_max_m": 100.0,
        "rms_ref_range_m": 10.0,
        "covariance_floor": 0.5,
        "covariance_cap": 1.0,
    },
    "fire_control": {
        "muzzle_velocity_mps": 400.0,
        "pellet_decel_mps2": 1.5,
        "pattern_spread_rate": 0.3,
        "max_engagement_range_m": 500.0,
        "max_position_uncertainty_m": 0.0,
        "class_label": "fixed_wing",
        "class_confidence": 0.9,
        "hit_threshold_m": 2.0,
        "max_hits": 3,
    },
    "source": {
        "speed_mps": 50.0,
        "altitude_estimate_m": None,
    },
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into a copy of *base*."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Path | None) -> dict:
    """Load JSON config and merge with built-in defaults."""
    if config_path is not None and config_path.exists():
        with open(config_path) as f:
            user = json.load(f)
        return _deep_merge(_DEFAULTS, user)
    return dict(_DEFAULTS)


# ============================================================================
# Data Loading
# ============================================================================

def load_simulation(sim_dir: Path) -> dict:
    """Load traces and metadata; promote 2-D mic positions to 3-D."""
    traces = np.load(sim_dir / "traces.npy")
    with open(sim_dir / "metadata.json") as f:
        metadata = json.load(f)

    mic_positions = np.array(metadata["receiver_positions"], dtype=np.float64)
    is_3d = mic_positions.shape[1] >= 3
    if not is_3d:
        mic_positions = np.column_stack(
            [mic_positions, np.zeros(mic_positions.shape[0])]
        )

    return {
        "traces": traces,
        "mic_positions": mic_positions,
        "metadata": metadata,
        "dt": metadata["dt"],
        "duration": traces.shape[1] * metadata["dt"],
        "is_3d": is_3d,
    }


def compute_ground_truth(metadata: dict, source_speed: float):
    """Build a ground-truth trajectory returning ``(x, y, z)``.

    For 2-D simulations that lack ``source_z``, z defaults to 0 and a
    sine arc is applied; for 3-D a parabolic arc is used.
    """
    start_x = metadata.get("source_x", -40.0)
    start_y = metadata.get("source_y", 0.0)
    start_z = metadata.get("source_z", 0.0)
    end_x = metadata.get("source_x1", -start_x)
    end_y = metadata.get("source_y1", start_y)
    end_z = metadata.get("source_z1", start_z)

    has_z = "source_z" in metadata
    default_arc = 10.0 if has_z else 15.0
    arc_height = metadata.get("source_arc_height", default_arc)
    horiz_dist = math.hypot(end_x - start_x, end_y - start_y)
    duration = horiz_dist / source_speed if source_speed > 0 else 3.0

    def trajectory(t: float) -> tuple[float, float, float]:
        frac = min(max(t / duration, 0.0), 1.0)
        x = start_x + (end_x - start_x) * frac
        if has_z:
            y = (start_y + (end_y - start_y) * frac
                 + arc_height * 4.0 * frac * (1.0 - frac))
        else:
            y_base = start_y + (end_y - start_y) * frac
            y = y_base + arc_height * math.sin(math.pi * frac)
        z = start_z + (end_z - start_z) * frac
        return x, y, z

    return trajectory, duration


# ============================================================================
# SRP-PHAT Beamformer
# ============================================================================

class SRPBeamformer:
    """SRP-PHAT beamformer with pre-computed steering vectors."""

    def __init__(
        self,
        mic_positions: np.ndarray,
        fs: float,
        window_samples: int,
        c: float = 343.0,
        n_bearings: int = 180,
        freq_lo: float = 100.0,
        freq_hi: float = 1000.0,
    ):
        self.n_mics = mic_positions.shape[0]
        self.fs = fs
        self.c = c
        self.n_bearings = n_bearings

        mic_xy = mic_positions[:, :2]
        center = mic_xy.mean(axis=0)
        mic_rel = mic_xy - center

        self.nfft = int(2 ** np.ceil(np.log2(window_samples)))
        freqs = np.fft.rfftfreq(self.nfft, d=1.0 / fs)
        fmask = (freqs >= freq_lo) & (freqs <= freq_hi)
        self.fmask = fmask
        omega = 2.0 * np.pi * freqs[fmask]

        self.bearings = np.linspace(0, 2 * np.pi, n_bearings, endpoint=False)
        look = np.column_stack([np.cos(self.bearings), np.sin(self.bearings)])

        taus = -(mic_rel @ look.T) / c
        self.steering = np.exp(
            1j * taus[:, :, np.newaxis] * omega[np.newaxis, np.newaxis, :]
        ).astype(np.complex64)

    def __call__(self, seg: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Return ``(best_bearing_rad, bearings, power)``."""
        X = np.fft.rfft(seg.astype(np.float32), n=self.nfft, axis=1)
        X_bp = X[:, self.fmask]
        mag = np.abs(X_bp)
        mag[mag < 1e-30] = 1e-30
        X_phat = (X_bp / mag).astype(np.complex64)
        steered = np.einsum("mf,mbf->bf", X_phat, self.steering)
        power = np.sum(np.abs(steered) ** 2, axis=1)
        best_idx = int(np.argmax(power))
        return float(self.bearings[best_idx]), self.bearings, power


# ============================================================================
# Causal WLS track fitter
# ============================================================================

def causal_ls_fit(
    det_t: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    det_z: np.ndarray,
    det_rms: np.ndarray,
) -> dict | None:
    """Constant-velocity WLS fit on past detections. ``None`` if < 5 pts."""
    n = len(det_t)
    if n < 5:
        return None

    t_ref = 0.5 * (det_t[0] + det_t[-1])
    dt_arr = det_t - t_ref
    weights = det_rms / max(det_rms.max(), 1e-12)
    A = np.column_stack([np.ones(n), dt_arr])
    W = np.diag(weights)

    def wls(y):
        AtW = A.T @ W
        return np.linalg.lstsq(AtW @ A, AtW @ y, rcond=None)[0]

    cx = wls(det_x)
    cy = wls(det_y)
    cz = wls(det_z)
    sq_n = max(math.sqrt(n), 1.0)

    return {
        "x0": float(cx[0]), "y0": float(cy[0]), "z0": float(cz[0]),
        "vx": float(cx[1]), "vy": float(cy[1]), "vz": float(cz[1]),
        "t_ref": t_ref,
        "res_x": float(np.std(det_x - A @ cx)) / sq_n,
        "res_y": float(np.std(det_y - A @ cy)) / sq_n,
        "res_z": float(np.std(det_z - A @ cz)) / sq_n,
        "n_det": n,
    }


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_results(
    detections: list[dict],
    fire_decisions: list[dict],
    ground_truth_fn,
    array_center: np.ndarray,
    hit_threshold: float,
) -> dict:
    """Compute bearing / range / miss-distance metrics."""
    cx, cy = array_center[0], array_center[1]
    bearing_errors: list[float] = []
    range_errors: list[float] = []

    for d in detections:
        if not d.get("detected"):
            continue
        t = d["time"]
        gt_x, gt_y, gt_z = ground_truth_fn(t)
        true_brg = math.degrees(math.atan2(gt_y - cy, gt_x - cx))
        if true_brg < 0:
            true_brg += 360
        err = d["bearing_deg"] - true_brg
        if err > 180:
            err -= 360
        if err < -180:
            err += 360
        bearing_errors.append(abs(err))
        true_range = math.hypot(gt_x - cx, gt_y - cy)
        range_errors.append(abs(d["range"] - true_range))

    miss_dists = [f["miss"] for f in fire_decisions if f.get("miss") is not None]
    n_shots = sum(1 for f in fire_decisions if f["can_fire"])
    n_hits = sum(1 for m in miss_dists if m < hit_threshold)

    return {
        "n_detections": sum(1 for d in detections if d.get("detected")),
        "n_windows": len(detections),
        "mean_bearing_error": float(np.mean(bearing_errors)) if bearing_errors else float("nan"),
        "max_bearing_error": float(np.max(bearing_errors)) if bearing_errors else float("nan"),
        "mean_range_error": float(np.mean(range_errors)) if range_errors else float("nan"),
        "shots_fired": n_shots,
        "hit_threshold": hit_threshold,
        "n_hits": n_hits,
        "mean_miss": float(np.mean(miss_dists)) if miss_dists else float("nan"),
        "min_miss": float(np.min(miss_dists)) if miss_dists else float("nan"),
        "max_miss": float(np.max(miss_dists)) if miss_dists else float("nan"),
        "bearing_errors": bearing_errors,
        "range_errors": range_errors,
        "miss_distances": miss_dists,
    }


# ============================================================================
# Plotting helpers
# ============================================================================

def _projectile_path(
    weapon_pos: np.ndarray,
    aim_bearing: float,
    aim_elevation: float,
    muzzle_velocity: float,
    decel: float,
    tof: float,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wx, wy, wz = weapon_pos[:3]
    times = np.linspace(0, tof, n_points)
    cos_el = math.cos(aim_elevation)
    sin_el = math.sin(aim_elevation)
    cos_az = math.cos(aim_bearing)
    sin_az = math.sin(aim_bearing)

    x_path, y_path, z_path = [], [], []
    for tv in times:
        v_avg = muzzle_velocity
        for _ in range(3):
            s = v_avg * tv
            v_end = max(muzzle_velocity - decel * s, 0)
            v_avg = 0.5 * (muzzle_velocity + v_end)
        s = v_avg * tv
        x_path.append(wx + s * cos_el * cos_az)
        y_path.append(wy + s * cos_el * sin_az)
        z_path.append(wz + s * sin_el)
    return np.array(x_path), np.array(y_path), np.array(z_path)


def plot_radial_engagement(
    fire_decisions: list[dict],
    ground_truth_fn,
    source_duration: float,
    weapon_pos: np.ndarray,
    is_3d: bool,
    cfg_fc: dict,
    output_path: Path | None = None,
) -> plt.Figure:
    """Radial engagement plot (plan-view + elevation for 3-D)."""
    muzzle_velocity = cfg_fc["muzzle_velocity_mps"]
    decel = cfg_fc["pellet_decel_mps2"]
    hit_threshold = cfg_fc["hit_threshold_m"]
    wx, wy, wz = weapon_pos[:3]
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
    ax_xy.scatter(gt_x[0], gt_y[0], c="g", s=150, marker="o", zorder=6, label="Start")
    ax_xy.scatter(gt_x[-1], gt_y[-1], c="g", s=150, marker="s", zorder=6, label="End")

    max_range = max(float(np.max(np.sqrt(gt_x**2 + gt_y**2))), 50)
    for r in [25, 50, 75, 100]:
        if r <= max_range * 1.2:
            circle = Circle((0, 0), r, fill=False, color="gray", ls="--", alpha=0.3)
            ax_xy.add_patch(circle)
            ax_xy.text(r * 0.707, r * 0.707, f"{r}m", fontsize=8, color="gray", alpha=0.7)

    n_shots = n_hits = 0
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
        proj_x, proj_y, _ = _projectile_path(
            weapon_pos, aim_brg, aim_el, muzzle_velocity, decel, tof)
        color = "green" if is_hit else "red"
        ax_xy.plot(proj_x - wx, proj_y - wy, "-", color=color, lw=2, alpha=0.8)
        ipos = fd.get("intercept_pos")
        if ipos is not None:
            ix, iy = ipos[0] - wx, ipos[1] - wy
            ax_xy.scatter(ix, iy, c=color, s=150, marker="x", linewidths=3, zorder=10)
            label = f"HIT ({miss_dist:.1f}m)" if is_hit else f"MISS ({miss_dist:.1f}m)"
            ax_xy.annotate(label, (ix, iy), xytext=(5, 5), textcoords="offset points",
                           fontsize=9, color=color, fontweight="bold")
        gt_pos = ground_truth_fn(fd["time"])
        ax_xy.scatter(gt_pos[0] - wx, gt_pos[1] - wy, c="lime", s=80, marker="o",
                      edgecolors="darkgreen", linewidths=2, zorder=8)

    ax_xy.scatter(0, 0, c="black", s=300, marker="*", label="Weapon", zorder=15)
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
        ax_xz.scatter(gt_x[-1], gt_z[-1], c="g", s=150, marker="s", zorder=6)
        for fd in fire_decisions:
            if not fd.get("can_fire"):
                continue
            aim_brg = fd.get("aim_bearing", float("nan"))
            aim_el = fd.get("aim_elevation", 0.0)
            tof = fd.get("tof", float("nan"))
            if np.isnan(aim_brg) or np.isnan(tof) or tof <= 0:
                continue
            is_hit = fd.get("hit", False)
            proj_x, _, proj_z = _projectile_path(
                weapon_pos, aim_brg, aim_el, muzzle_velocity, decel, tof)
            color = "green" if is_hit else "red"
            ax_xz.plot(proj_x - wx, proj_z - wz, "-", color=color, lw=2, alpha=0.8)
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
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")
    return fig


def plot_summary(
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
    output_path: Path,
) -> None:
    """6-panel real-time pipeline summary figure."""
    cx, cy = array_center[0], array_center[1]
    gt_times = np.linspace(0, source_duration, 200)
    gt_xyz = np.array([ground_truth_fn(t) for t in gt_times])

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1 -- spatial overview X-Y
    ax = axes[0, 0]
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], "g-", lw=2, label="True path")
    det_xs = [d["x"] for d in all_detections if d.get("detected") and "x" in d]
    det_ys = [d["y"] for d in all_detections if d.get("detected") and "y" in d]
    if det_xs:
        ax.scatter(det_xs, det_ys, c="b", s=20, alpha=0.4, label="Detections")
    track_xs = [fd["est_pos"][0] for fd in all_fire_decisions if fd.get("est_pos")]
    track_ys = [fd["est_pos"][1] for fd in all_fire_decisions if fd.get("est_pos")]
    if track_xs:
        ax.plot(track_xs, track_ys, "m-", lw=1.5, alpha=0.7, label="Causal track")
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
    true_brgs = [math.degrees(math.atan2(
        ground_truth_fn(t)[1] - cy, ground_truth_fn(t)[0] - cx)) for t in gt_times]
    ax.plot(gt_times, true_brgs, "g-", lw=2, label="True")
    det_t = [d["time"] for d in all_detections if d.get("detected") and "bearing_deg" in d]
    det_brg = [d["bearing_deg"] - 360 if d["bearing_deg"] > 180 else d["bearing_deg"]
               for d in all_detections if d.get("detected") and "bearing_deg" in d]
    if det_t:
        ax.scatter(det_t, det_brg, c="b", s=20, alpha=0.4, label="Detected")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bearing (deg)")
    ax.set_title("Bearing vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 -- miss distance vs time
    ax = axes[0, 2]
    shot_times = [f["time"] for f in all_fire_decisions if f.get("miss") is not None]
    shot_misses = [f["miss"] for f in all_fire_decisions if f.get("miss") is not None]
    if shot_times:
        colors_shot = ["green" if m < hit_threshold else "red" for m in shot_misses]
        ax.scatter(shot_times, shot_misses, c=colors_shot, s=60, marker="x", zorder=5)
        ax.axhline(hit_threshold, color="g", ls="--", alpha=0.7, label=f"{hit_threshold}m")
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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_beamformer_diagnostic(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    ground_truth_fn,
    source_duration: float,
    array_center: np.ndarray,
    beamformer: SRPBeamformer,
    cfg_det: dict,
    output_path: Path | None = None,
) -> None:
    """4-panel diagnostic comparing RMS-weighted vs SRP-PHAT bearing."""
    n_mics, n_samples = traces.shape
    fs = 1.0 / dt
    cx, cy = array_center[0], array_center[1]
    min_signal_rms = cfg_det["min_signal_rms"]

    mic_angles = np.array([
        math.atan2(mic_positions[i, 1] - cy, mic_positions[i, 0] - cx)
        for i in range(n_mics)
    ])

    win_len = max(int(round(cfg_det["window_length_s"] * fs)), 1)
    hop = max(int(round(win_len * (1.0 - cfg_det["window_overlap"]))), 1)

    times, true_bearings, rms_bearings, srp_bearings = [], [], [], []
    rms_errors, srp_errors, srp_power_maps, rms_values = [], [], [], []

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

        srp_brg, scan_angles, srp_pow = beamformer(seg)

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
        im = ax.imshow(power_norm, aspect="auto", extent=extent, cmap="hot", origin="upper")
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
        ax.axvline(true_bearings[cpa_idx], color="k", ls="--", lw=2, label="True")
        ax.axvline(srp_bearings[cpa_idx], color="b", ls=":", lw=2, label="SRP-PHAT")
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
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


# ============================================================================
# Pipeline
# ============================================================================

def run_pipeline(
    sim_dir: Path,
    output_dir: Path,
    cfg: dict,
) -> dict:
    """Causal real-time detection / tracking / engagement pipeline.

    Works identically for 2-D and 3-D data.
    """
    cfg_det = cfg["detection"]
    cfg_bf = cfg["beamformer"]
    cfg_trk = cfg["tracking"]
    cfg_fc = cfg["fire_control"]
    cfg_src = cfg["source"]

    source_speed = cfg_src["speed_mps"]
    hit_threshold = cfg_fc["hit_threshold_m"]
    max_hits = cfg_fc["max_hits"]
    min_track_det = cfg_trk["min_detections"]
    ema_alpha = cfg_trk["ema_alpha"]
    rms_fire_gate = cfg_trk["rms_fire_gate_frac"]
    range_min = cfg_trk["range_min_m"]
    range_max = cfg_trk["range_max_m"]
    rms_ref_range = cfg_trk["rms_ref_range_m"]
    cov_floor = cfg_trk["covariance_floor"]
    cov_cap = cfg_trk["covariance_cap"]
    muzzle_velocity = cfg_fc["muzzle_velocity_mps"]
    pellet_decel = cfg_fc["pellet_decel_mps2"]
    pattern_spread_rate = cfg_fc["pattern_spread_rate"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # -- load ----------------------------------------------------------------
    data = load_simulation(sim_dir)
    traces = data["traces"]
    mic_positions = data["mic_positions"]
    dt_sim = data["dt"]
    metadata = data["metadata"]
    is_3d = data["is_3d"]
    n_mics, n_samples = traces.shape
    fs = 1.0 / dt_sim

    array_center = mic_positions.mean(axis=0)
    weapon_pos = array_center.copy()

    ground_truth_fn, src_duration = compute_ground_truth(metadata, source_speed)
    source_z_est = cfg_src["altitude_estimate_m"]
    if source_z_est is None:
        source_z_est = float(metadata.get("source_z", 0.0))

    dim = "3-D" if is_3d else "2-D"
    print("=" * 60)
    print(f"{dim} ACOUSTIC ENGAGEMENT PIPELINE")
    print("=" * 60)
    print(f"\n[LOAD] {sim_dir}")
    print(f"       {n_mics} mics, {n_samples} samples, dt={dt_sim:.2e}")
    print(f"       Array centre: ({array_center[0]:.1f}, "
          f"{array_center[1]:.1f}, {array_center[2]:.1f})")
    print(f"       Weapon at array")

    # -- windowing -----------------------------------------------------------
    win_len = max(int(round(cfg_det["window_length_s"] * fs)), 1)
    hop = max(int(round(win_len * (1.0 - cfg_det["window_overlap"]))), 1)
    hop_sec = hop * dt_sim
    n_windows = (n_samples - win_len) // hop + 1
    print(f"       Window: {cfg_det['window_length_s'] * 1e3:.0f} ms, "
          f"hop: {hop_sec * 1e3:.1f} ms, {n_windows} windows")

    # -- beamformer ----------------------------------------------------------
    c_sound = float(metadata.get("velocity", 343.0))
    beamformer = SRPBeamformer(
        mic_positions, fs, win_len, c=c_sound,
        n_bearings=cfg_bf["n_bearings"],
        freq_lo=cfg_bf["freq_lo_hz"],
        freq_hi=cfg_bf["freq_hi_hz"],
    )
    print(f"       SRP-PHAT: {beamformer.n_bearings} bearings, "
          f"nfft={beamformer.nfft}, steering {beamformer.steering.shape}")

    # -- RMS calibration -----------------------------------------------------
    rms_profile = np.array([
        float(np.sqrt(np.mean(traces[:, p:p + win_len] ** 2)))
        for p in range(0, n_samples - win_len + 1, hop)
    ])
    peak_rms = float(rms_profile.max())
    peak_idx = int(np.argmax(rms_profile))
    peak_t = (peak_idx * hop + win_len / 2) * dt_sim
    gt_peak = ground_truth_fn(peak_t)
    cpa_dist = max(float(np.linalg.norm(np.array(gt_peak) - array_center)), 1.0)
    rms_times_range = peak_rms * cpa_dist
    rms_ref_value = rms_times_range / rms_ref_range
    print(f"       RMS cal: peak={peak_rms:.6f}, CPA dist={cpa_dist:.1f} m")

    # -- main loop -----------------------------------------------------------
    print(f"\n[RUN]  Streaming {n_windows} windows (causal mode)")

    det_times: list[float] = []
    det_xs: list[float] = []
    det_ys: list[float] = []
    det_zs: list[float] = []
    det_rms_list: list[float] = []

    all_detections: list[dict] = []
    all_track_states: list = []
    all_fire_decisions: list[dict] = []
    wall_times_list: list[float] = []
    hits = 0

    ema_sin = 0.0
    ema_cos = 0.0
    bearing_init = False
    min_rms = cfg_det["min_signal_rms"]

    pos = 0
    while pos + win_len <= n_samples:
        t0_wall = time.perf_counter()
        t_center = (pos + win_len / 2.0) * dt_sim
        seg = traces[:, pos:pos + win_len]
        window_rms = float(np.sqrt(np.mean(seg ** 2)))
        detected = window_rms >= min_rms

        det_dict: dict = {"time": t_center, "window_rms": window_rms,
                          "detected": detected}

        if detected:
            raw_bearing, _, _ = beamformer(seg)

            if not bearing_init:
                ema_sin = math.sin(raw_bearing)
                ema_cos = math.cos(raw_bearing)
                bearing_init = True
            else:
                ema_sin = ema_alpha * math.sin(raw_bearing) + (1 - ema_alpha) * ema_sin
                ema_cos = ema_alpha * math.cos(raw_bearing) + (1 - ema_alpha) * ema_cos
            bearing_rad = math.atan2(ema_sin, ema_cos)
            if bearing_rad < 0:
                bearing_rad += 2 * math.pi
            bearing_deg = math.degrees(bearing_rad)

            est_range = rms_ref_range * math.sqrt(
                rms_ref_value / max(window_rms, 1e-12))
            est_range = max(range_min, min(range_max, est_range))

            est_x = array_center[0] + est_range * math.cos(bearing_rad)
            est_y = array_center[1] + est_range * math.sin(bearing_rad)

            det_dict.update({
                "bearing": bearing_rad, "bearing_deg": bearing_deg,
                "range": est_range, "z": source_z_est,
                "x": est_x, "y": est_y,
            })
            det_times.append(t_center)
            det_xs.append(est_x)
            det_ys.append(est_y)
            det_zs.append(source_z_est)
            det_rms_list.append(window_rms)

        all_detections.append(det_dict)

        # -- causal track update ---------------------------------------------
        track_state = None
        if len(det_times) >= min_track_det:
            track_state = causal_ls_fit(
                np.array(det_times), np.array(det_xs),
                np.array(det_ys), np.array(det_zs),
                np.array(det_rms_list),
            )
        all_track_states.append(track_state)

        # -- fire control ----------------------------------------------------
        fire_decision: dict = {"time": t_center, "can_fire": False,
                               "reason": "NO_TRACK"}

        rms_gate = window_rms >= rms_fire_gate * peak_rms
        use_instant = detected and track_state is not None and rms_gate

        if use_instant:
            fit = track_state
            est_vel = np.array([fit["vx"], fit["vy"], fit["vz"]])
            est_pos = np.array([det_dict["x"], det_dict["y"], source_z_est])

            cov = np.zeros((6, 6))
            cov[0, 0] = min(max(fit["res_x"], cov_floor), cov_cap) ** 2
            cov[1, 1] = min(max(fit["res_y"], cov_floor), cov_cap) ** 2
            cov[2, 2] = min(max(fit["res_z"], cov_floor), cov_cap) ** 2
            cov[3, 3] = cov[4, 4] = cov[5, 5] = 1.0

            if max_hits > 0 and hits >= max_hits:
                fire_decision = {"time": t_center, "can_fire": False,
                                 "reason": "TARGET_ENGAGED",
                                 "est_pos": est_pos.tolist()}
            else:
                lead = compute_lead_3d(est_pos, est_vel, weapon_pos,
                                       muzzle_velocity, pellet_decel)
                eng = compute_engagement_3d(
                    est_pos, est_vel, cov, weapon_pos,
                    muzzle_velocity, pellet_decel, pattern_spread_rate,
                    max_position_uncertainty=cfg_fc["max_position_uncertainty_m"],
                    max_engagement_range=cfg_fc["max_engagement_range_m"],
                    class_label=cfg_fc["class_label"],
                    class_confidence=cfg_fc["class_confidence"],
                )
                fire_decision = {
                    "time": t_center,
                    "can_fire": eng["can_fire"],
                    "reason": eng["reason"],
                    "est_pos": est_pos.tolist(),
                    "intercept_pos": lead["intercept_pos"].tolist(),
                    "aim_bearing": lead["aim_bearing"],
                    "aim_elevation": lead["aim_elevation"],
                    "tof": lead["tof"],
                    "range": eng["range"],
                    "pattern_diam": eng["pattern_diam"],
                    "pos_unc": eng["position_uncertainty"],
                }
                if eng["can_fire"]:
                    gt = np.asarray(ground_truth_fn(t_center))
                    intercept = lead["intercept_pos"]
                    miss = float(np.linalg.norm(intercept - gt))
                    fire_decision["miss"] = miss
                    if miss < hit_threshold:
                        hits += 1
                        fire_decision["hit"] = True

        all_fire_decisions.append(fire_decision)
        wall_times_list.append(time.perf_counter() - t0_wall)

        if max_hits > 0 and hits >= max_hits:
            print(f"       >>> {hits} hits at t={t_center:.4f}s "
                  f"-- target neutralised.")
            break
        pos += hop

    # -- results -------------------------------------------------------------
    wall_times = np.array(wall_times_list)
    metrics = evaluate_results(all_detections, all_fire_decisions,
                               ground_truth_fn, array_center, hit_threshold)

    n_detected = metrics["n_detections"]
    n_shots = metrics["shots_fired"]
    n_hits_val = metrics["n_hits"]
    mean_miss = metrics["mean_miss"]
    mean_brg_err = metrics["mean_bearing_error"]
    miss_dists = metrics["miss_distances"]

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({dim})")
    print(f"{'=' * 60}")
    print(f"\n  Detection:  {n_detected}/{len(all_detections)} windows")
    print(f"  Bearing:    {mean_brg_err:.1f} deg mean error")
    print(f"\n  Shots:      {n_shots}")
    print(f"  Hits <{hit_threshold}m:  {n_hits_val} "
          f"({100 * n_hits_val / max(n_shots, 1):.1f}%)")
    print(f"  Mean miss:  {mean_miss:.1f} m")
    if miss_dists:
        print(f"  Min miss:   {min(miss_dists):.1f} m")
        print(f"  Max miss:   {max(miss_dists):.1f} m")
    print(f"\n  TIMING:")
    print(f"  Mean process:  {wall_times.mean() * 1e6:.0f} us/window")
    print(f"  Max process:   {wall_times.max() * 1e6:.0f} us/window")
    print(f"  Realtime margin: {hop_sec / wall_times.mean():.0f}x "
          f"faster than real-time")

    # -- plots ---------------------------------------------------------------
    suffix = "_3d" if is_3d else "_2d"

    plot_summary(
        all_detections, all_fire_decisions, all_track_states,
        wall_times, ground_truth_fn, src_duration,
        array_center, weapon_pos, is_3d, hop_sec,
        hit_threshold, metrics,
        output_dir / f"pipeline_summary{suffix}.png",
    )
    plot_radial_engagement(
        all_fire_decisions, ground_truth_fn, src_duration,
        weapon_pos, is_3d, cfg_fc,
        output_path=output_dir / f"radial_engagement{suffix}.png",
    )
    print(f"\n[DIAGNOSTIC] Running beamformer comparison ...")
    plot_beamformer_diagnostic(
        traces, mic_positions, dt_sim,
        ground_truth_fn, src_duration,
        array_center, beamformer, cfg_det,
        output_path=output_dir / f"beamformer_diagnostic{suffix}.png",
    )

    # -- JSON ----------------------------------------------------------------
    results = {
        "dimensions": dim,
        "simulation": str(sim_dir),
        "config": cfg,
        "n_detections": n_detected,
        "n_windows": len(all_detections),
        "mean_bearing_error_deg": mean_brg_err,
        "shots_fired": n_shots,
        "hits": n_hits_val,
        "hit_threshold_m": hit_threshold,
        "mean_miss_m": mean_miss if not math.isnan(mean_miss) else None,
        "min_miss_m": min(miss_dists) if miss_dists else None,
        "max_miss_m": max(miss_dists) if miss_dists else None,
        "timing": {
            "hop_ms": hop_sec * 1e3,
            "mean_latency_us": float(wall_times.mean() * 1e6),
            "max_latency_us": float(wall_times.max() * 1e6),
            "realtime_margin_x": float(hop_sec / wall_times.mean()),
        },
    }
    results_path = output_dir / f"results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sim_dir", type=Path, nargs="?",
        default=Path("output/valley_3d_test"),
        help="Simulation output directory (default: output/valley_3d_test)",
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent / "pipeline.config.json",
        help="JSON config file (default: examples/pipeline.config.json)",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for plots (default: sim_dir)")
    parser.add_argument("--source-speed", type=float, default=None,
                        help="Override source.speed_mps from config")
    parser.add_argument("--hit-threshold", type=float, default=None,
                        help="Override fire_control.hit_threshold_m")
    parser.add_argument("--max-hits", type=int, default=None,
                        help="Override fire_control.max_hits")

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.source_speed is not None:
        cfg["source"]["speed_mps"] = args.source_speed
    if args.hit_threshold is not None:
        cfg["fire_control"]["hit_threshold_m"] = args.hit_threshold
    if args.max_hits is not None:
        cfg["fire_control"]["max_hits"] = args.max_hits

    output_dir = args.output_dir or args.sim_dir

    run_pipeline(args.sim_dir, output_dir, cfg)
    sys.exit(0)


if __name__ == "__main__":
    main()

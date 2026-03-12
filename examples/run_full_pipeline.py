#!/usr/bin/env python3
"""End-to-end detection and targeting pipeline with visualization.

Loads FDTD simulation data, runs MFP detection, EKF tracking, and fire control,
then produces comprehensive evaluation plots.

Usage:
    python run_full_pipeline.py                          # Use defaults
    python run_full_pipeline.py ../output/valley_test    # Specify sim dir
    python run_full_pipeline.py --source-speed 50        # Override source speed
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.processor import matched_field_process
from acoustic_sim.tracker import run_tracker
from acoustic_sim.fire_control import run_fire_control


# ============================================================================
# Data Loading
# ============================================================================

def load_simulation(sim_dir: Path) -> dict:
    """Load simulation traces and metadata."""
    traces = np.load(sim_dir / "traces.npy")
    with open(sim_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    mic_positions = np.array(metadata["receiver_positions"])
    
    return {
        "traces": traces,
        "mic_positions": mic_positions,
        "metadata": metadata,
        "dt": metadata["dt"],
        "duration": traces.shape[1] * metadata["dt"],
    }


def compute_ground_truth(metadata: dict, source_speed: float) -> callable:
    """Build ground truth trajectory function.
    
    Args:
        metadata: Simulation metadata
        source_speed: Source velocity in m/s
        
    Returns:
        Function mapping time to (x, y) position
    """
    start_x = metadata.get("source_x", -40.0)
    start_y = metadata.get("source_y", 0.0)
    end_x = metadata.get("source_x1", -start_x)
    end_y = metadata.get("source_y1", start_y)
    arc_height = metadata.get("source_arc_height", 15.0)
    
    dist = math.hypot(end_x - start_x, end_y - start_y)
    duration = dist / source_speed if source_speed > 0 else 3.0
    
    def trajectory(t: float) -> tuple[float, float]:
        frac = min(max(t / duration, 0.0), 1.0)
        x = start_x + (end_x - start_x) * frac
        y_base = start_y + (end_y - start_y) * frac
        y = y_base + arc_height * math.sin(math.pi * frac)
        return x, y
    
    return trajectory, duration


# ============================================================================
# Pipeline Components
# ============================================================================

def run_mfp_detection(
    traces: np.ndarray,
    mic_positions: np.ndarray,
    dt: float,
    *,
    fundamental: float = 180.0,
    n_harmonics: int = 4,
    detection_threshold: float = 0.2,
    min_signal_rms: float = 0.01,
) -> dict:
    """Run matched field processing for detection."""
    return matched_field_process(
        traces, mic_positions, dt,
        fundamental=fundamental,
        n_harmonics=n_harmonics,
        detection_threshold=detection_threshold,
        min_signal_rms=min_signal_rms,
        window_length=0.1,
        window_overlap=0.5,
        range_min=5.0,
        range_max=100.0,
    )


def apply_rms_range_estimation(
    detections: list[dict],
    rms_ref_range: float = 10.0,
    rms_ref_value: float = 0.24,  # Calibrated from valley_test simulation
    range_min: float = 5.0,
    range_max: float = 100.0,
) -> None:
    """Override MFP range with RMS-based estimation (in-place)."""
    for d in detections:
        if d["detected"]:
            rms = d.get("window_rms", 0.1)
            if rms > 0.001:
                rms_range = rms_ref_range * math.sqrt(rms_ref_value / rms)
                rms_range = max(range_min, min(range_max, rms_range))
                d["range"] = rms_range


def run_tracking(
    detections: list[dict],
    array_center: tuple[float, float],
) -> dict:
    """Run EKF tracker on detections."""
    cx, cy = array_center
    return run_tracker(
        detections,
        process_noise_std=50.0,
        sigma_bearing_deg=1.0,
        sigma_range=2.0,
        initial_range_guess=30.0,
        source_level_dB=90.0,
        array_center_x=cx,
        array_center_y=cy,
    )


def run_targeting(track: dict, ground_truth_fn: callable, hit_threshold: float = 3.0) -> dict:
    """Run fire control on track.
    
    Uses realistic shotgun parameters:
    - 400 m/s muzzle velocity (typical 12ga slug)
    - 1.5 m/s per meter deceleration
    - 0.2 pattern spread rate
    - Stop after 2 confirmed hits
    - Engage as soon as track is available (no uncertainty gate)
    - 50m max engagement range
    """
    return run_fire_control(
        track,
        weapon_position=(0.0, 0.0),
        muzzle_velocity=400.0,
        pellet_decel=1.5,
        pattern_spread_rate=0.2,
        max_hits=2,
        hit_threshold=hit_threshold,
        ground_truth_fn=ground_truth_fn,
        max_position_uncertainty=0.0,
        max_engagement_range=500.0,
    )


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_results(
    detections: list[dict],
    track: dict,
    fire_control: dict,
    ground_truth_fn: callable,
    array_center: tuple[float, float],
    hit_threshold: float = 3.0,
) -> dict:
    """Compute error metrics against ground truth."""
    cx, cy = array_center
    
    # Detection bearing errors
    bearing_errors = []
    range_errors = []
    for d in detections:
        if not d["detected"]:
            continue
        t = d["time"]
        gt_x, gt_y = ground_truth_fn(t)
        dx, dy = gt_x - cx, gt_y - cy
        true_bearing = math.degrees(math.atan2(dy, dx))
        if true_bearing < 0:
            true_bearing += 360
        true_range = math.hypot(dx, dy)
        
        det_bearing = d["bearing_deg"]
        err = det_bearing - true_bearing
        if err > 180:
            err -= 360
        if err < -180:
            err += 360
        bearing_errors.append(abs(err))
        range_errors.append(abs(d["range"] - true_range))
    
    # Fire control miss distances
    fc_times = fire_control.get("times", np.array([]))
    fc_intercepts = fire_control.get("intercept_positions", np.array([]).reshape(-1, 2))
    fc_can_fire = fire_control.get("can_fire", np.array([]))
    
    miss_distances = []
    shots_fired = 0
    for i, t in enumerate(fc_times):
        if i < len(fc_can_fire) and fc_can_fire[i]:
            shots_fired += 1
            if i < len(fc_intercepts):
                aim_x, aim_y = fc_intercepts[i]
                if not (np.isnan(aim_x) or np.isnan(aim_y)):
                    gt_x, gt_y = ground_truth_fn(t)
                    miss = math.hypot(aim_x - gt_x, aim_y - gt_y)
                    miss_distances.append(miss)
    
    return {
        "n_detections": sum(1 for d in detections if d["detected"]),
        "n_windows": len(detections),
        "mean_bearing_error": np.mean(bearing_errors) if bearing_errors else float("nan"),
        "max_bearing_error": np.max(bearing_errors) if bearing_errors else float("nan"),
        "mean_range_error": np.mean(range_errors) if range_errors else float("nan"),
        "shots_fired": shots_fired,
        "hit_threshold": hit_threshold,
        "n_hits": sum(1 for m in miss_distances if m < hit_threshold),
        "n_hits_5m": sum(1 for m in miss_distances if m < 5.0),
        "mean_miss": np.mean(miss_distances) if miss_distances else float("nan"),
        "min_miss": np.min(miss_distances) if miss_distances else float("nan"),
        "max_miss": np.max(miss_distances) if miss_distances else float("nan"),
        "bearing_errors": bearing_errors,
        "range_errors": range_errors,
        "miss_distances": miss_distances,
    }


# ============================================================================
# Plotting
# ============================================================================

def compute_projectile_path(
    weapon_pos: tuple[float, float],
    aim_bearing: float,
    muzzle_velocity: float,
    decel: float,
    tof: float,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute projectile trajectory from weapon to intercept.
    
    Args:
        weapon_pos: (x, y) weapon position
        aim_bearing: Aim direction in radians
        muzzle_velocity: Initial velocity m/s
        decel: Velocity loss per meter
        tof: Time of flight in seconds
        n_points: Number of points to sample
        
    Returns:
        (x_array, y_array) trajectory points
    """
    wx, wy = weapon_pos
    times = np.linspace(0, tof, n_points)
    
    x_path = []
    y_path = []
    
    for t in times:
        # Iterative approach for accuracy with drag
        v_avg = muzzle_velocity
        for _ in range(3):
            s = v_avg * t
            v_end = max(muzzle_velocity - decel * s, 0)
            v_avg = 0.5 * (muzzle_velocity + v_end)
        
        s = v_avg * t
        x = wx + s * math.cos(aim_bearing)
        y = wy + s * math.sin(aim_bearing)
        x_path.append(x)
        y_path.append(y)
    
    return np.array(x_path), np.array(y_path)


def plot_radial_engagement(
    fire_control: dict,
    ground_truth_fn: callable,
    source_duration: float,
    weapon_pos: tuple[float, float] = (0.0, 0.0),
    muzzle_velocity: float = 400.0,
    decel: float = 1.5,
    hit_threshold: float = 3.0,
    output_path: Path = None,
) -> plt.Figure:
    """Plot radial engagement view with weapon at center.
    
    Shows:
    - Weapon at origin
    - Target trajectory (ground truth)
    - Projectile flight paths for each shot
    - Range rings
    - Hit/Miss status for each shot (green=hit, red=miss)
    """
    wx, wy = weapon_pos
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Ground truth trajectory (relative to weapon)
    gt_times = np.linspace(0, source_duration, 200)
    gt_x = np.array([ground_truth_fn(t)[0] - wx for t in gt_times])
    gt_y = np.array([ground_truth_fn(t)[1] - wy for t in gt_times])
    
    ax.plot(gt_x, gt_y, 'g-', lw=3, label='Target path', zorder=5)
    ax.scatter(gt_x[0], gt_y[0], c='g', s=150, marker='o', zorder=6, label='Target start')
    ax.scatter(gt_x[-1], gt_y[-1], c='g', s=150, marker='s', zorder=6, label='Target end')
    
    # Draw range rings
    max_range = max(np.max(np.sqrt(gt_x**2 + gt_y**2)), 50)
    for r in [25, 50, 75, 100]:
        if r <= max_range * 1.2:
            circle = Circle((0, 0), r, fill=False, color='gray', linestyle='--', alpha=0.3)
            ax.add_patch(circle)
            ax.text(r * 0.707, r * 0.707, f'{r}m', fontsize=8, color='gray', alpha=0.7)
    
    # Extract fire control data
    fc_times = fire_control.get("times", np.array([]))
    fc_can_fire = fire_control.get("can_fire", np.array([]))
    fc_aim_bearings = fire_control.get("aim_bearings", np.array([]))
    fc_tofs = fire_control.get("tofs", np.array([]))
    fc_intercepts = fire_control.get("intercept_positions", np.array([]).reshape(-1, 2))
    
    # Plot projectile paths for each shot
    shot_num = 0
    n_shots = int(np.sum(fc_can_fire)) if len(fc_can_fire) > 0 else 0
    n_hits = 0
    
    for i, (t, can_fire, aim_brg, tof) in enumerate(zip(fc_times, fc_can_fire, fc_aim_bearings, fc_tofs)):
        if not can_fire or np.isnan(aim_brg) or np.isnan(tof) or tof <= 0:
            continue
        
        shot_num += 1
        
        # Compute miss distance
        if i < len(fc_intercepts):
            int_x, int_y = fc_intercepts[i]
            gt_x_t, gt_y_t = ground_truth_fn(t)
            miss_dist = math.hypot(int_x - gt_x_t, int_y - gt_y_t)
            is_hit = miss_dist < hit_threshold
            if is_hit:
                n_hits += 1
        else:
            miss_dist = float('nan')
            is_hit = False
        
        # Compute projectile trajectory
        proj_x, proj_y = compute_projectile_path(
            weapon_pos, aim_brg, muzzle_velocity, decel, tof
        )
        
        # Convert to relative coordinates
        proj_x_rel = proj_x - wx
        proj_y_rel = proj_y - wy
        
        # Color based on hit/miss
        path_color = 'green' if is_hit else 'red'
        
        ax.plot(proj_x_rel, proj_y_rel, '-', color=path_color, lw=2, alpha=0.8)
        
        # Mark intercept point with hit/miss color
        if i < len(fc_intercepts):
            int_x = fc_intercepts[i, 0] - wx
            int_y = fc_intercepts[i, 1] - wy
            marker_color = 'green' if is_hit else 'red'
            ax.scatter(int_x, int_y, c=marker_color, s=150, marker='x', linewidths=3, zorder=10)
            # Add miss distance label
            label = f"HIT ({miss_dist:.1f}m)" if is_hit else f"MISS ({miss_dist:.1f}m)"
            ax.annotate(label, (int_x, int_y), xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=marker_color, fontweight='bold')
        
        # Mark target position at shot time
        gt_x_t, gt_y_t = ground_truth_fn(t)
        gt_x_rel = gt_x_t - wx
        gt_y_rel = gt_y_t - wy
        ax.scatter(gt_x_rel, gt_y_rel, c='lime', s=80, marker='o', 
                   edgecolors='darkgreen', linewidths=2, zorder=8)
    
    # Weapon at center
    ax.scatter(0, 0, c='black', s=300, marker='*', label='Weapon', zorder=15)
    
    # Configure axes
    ax.set_xlabel('X relative to weapon (m)', fontsize=12)
    ax.set_ylabel('Y relative to weapon (m)', fontsize=12)
    ax.set_title(f'RADIAL ENGAGEMENT VIEW\n(Weapon at Center, Hit threshold: <{hit_threshold}m)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='g', lw=3, label='Target path'),
        Line2D([0], [0], marker='o', color='g', linestyle='None', markersize=10, label='Target start'),
        Line2D([0], [0], marker='s', color='g', linestyle='None', markersize=10, label='Target end'),
        Line2D([0], [0], color='green', lw=2, label='HIT (<3m)'),
        Line2D([0], [0], color='red', lw=2, label='MISS (>=3m)'),
        Line2D([0], [0], marker='o', color='lime', linestyle='None', markersize=10, 
               markeredgecolor='darkgreen', markeredgewidth=2, label='Target at shot time'),
        Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=15, label='Weapon'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add shot count annotation with hit/miss breakdown
    ax.text(0.98, 0.02, f'Shots: {n_shots}  |  Hits: {n_hits}  |  Misses: {n_shots - n_hits}', 
            transform=ax.transAxes, fontsize=12, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    return fig


def plot_full_evaluation(
    detections: list[dict],
    track: dict,
    fire_control: dict,
    ground_truth_fn: callable,
    source_duration: float,
    array_center: tuple[float, float],
    metrics: dict,
    output_path: Path,
) -> None:
    """Generate comprehensive evaluation plots."""
    cx, cy = array_center
    
    fig = plt.figure(figsize=(16, 12))
    
    # ── Panel 1: Spatial overview ───────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Ground truth trajectory
    gt_times = np.linspace(0, source_duration, 200)
    gt_x = [ground_truth_fn(t)[0] for t in gt_times]
    gt_y = [ground_truth_fn(t)[1] for t in gt_times]
    ax1.plot(gt_x, gt_y, "g-", lw=2, label="True trajectory", zorder=5)
    ax1.scatter(gt_x[0], gt_y[0], c="g", s=100, marker="o", zorder=6)
    ax1.scatter(gt_x[-1], gt_y[-1], c="g", s=100, marker="s", zorder=6)
    
    # Detections
    det_x = [d["x"] for d in detections if d["detected"] and not np.isnan(d["x"])]
    det_y = [d["y"] for d in detections if d["detected"] and not np.isnan(d["y"])]
    if det_x:
        ax1.scatter(det_x, det_y, c="b", s=30, alpha=0.6, label="MFP detections")
    
    # Track
    positions = track.get("positions", np.array([]).reshape(-1, 2))
    valid_pos = ~np.isnan(positions).any(axis=1) if len(positions) > 0 else np.array([])
    if np.any(valid_pos):
        track_x = positions[valid_pos, 0]
        track_y = positions[valid_pos, 1]
        ax1.plot(track_x, track_y, "m-", lw=1.5, alpha=0.7, label="EKF track")
    
    # Fire control intercepts
    fc_intercepts = fire_control.get("intercept_positions", np.array([]).reshape(-1, 2))
    fc_can_fire = fire_control.get("can_fire", np.array([]))
    for i, (pos, can_fire) in enumerate(zip(fc_intercepts, fc_can_fire)):
        if can_fire and not np.isnan(pos[0]):
            ax1.scatter(pos[0], pos[1], c="r", s=80, marker="x", zorder=10)
    
    # Array center
    ax1.scatter(cx, cy, c="k", s=100, marker="^", label="Array", zorder=10)
    
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Spatial Overview")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    
    # ── Panel 2: Bearing over time ──────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    
    # True bearing
    true_bearings = []
    for t in gt_times:
        gx, gy = ground_truth_fn(t)
        brg = math.degrees(math.atan2(gy - cy, gx - cx))
        true_bearings.append(brg)
    ax2.plot(gt_times, true_bearings, "g-", lw=2, label="True bearing")
    
    # Detected bearings
    det_t = [d["time"] for d in detections if d["detected"]]
    det_brg = [d["bearing_deg"] for d in detections if d["detected"]]
    # Unwrap for plotting
    det_brg_unwrap = []
    for b in det_brg:
        if b > 180:
            b -= 360
        det_brg_unwrap.append(b)
    if det_t:
        ax2.scatter(det_t, det_brg_unwrap, c="b", s=30, alpha=0.6, label="Detected")
    
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Bearing (°)")
    ax2.set_title("Bearing vs Time")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ── Panel 3: Range over time ────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    
    # True range
    true_ranges = [math.hypot(ground_truth_fn(t)[0] - cx, ground_truth_fn(t)[1] - cy) for t in gt_times]
    ax3.plot(gt_times, true_ranges, "g-", lw=2, label="True range")
    
    # Detected ranges
    det_rng = [d["range"] for d in detections if d["detected"]]
    if det_t:
        ax3.scatter(det_t, det_rng, c="b", s=30, alpha=0.6, label="Detected (RMS)")
    
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Range (m)")
    ax3.set_title("Range vs Time")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # ── Panel 4: Bearing error histogram ────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    if metrics["bearing_errors"]:
        ax4.hist(metrics["bearing_errors"], bins=20, color="steelblue", edgecolor="white")
        ax4.axvline(metrics["mean_bearing_error"], color="r", ls="--", label=f'Mean: {metrics["mean_bearing_error"]:.1f}°')
    ax4.set_xlabel("Bearing Error (°)")
    ax4.set_ylabel("Count")
    ax4.set_title("Bearing Error Distribution")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # ── Panel 5: Miss distance over time ────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    
    fc_times = fire_control.get("times", np.array([]))
    miss_by_time = []
    time_for_miss = []
    for i, t in enumerate(fc_times):
        if i < len(fc_can_fire) and fc_can_fire[i] and i < len(fc_intercepts):
            aim_x, aim_y = fc_intercepts[i]
            if not (np.isnan(aim_x) or np.isnan(aim_y)):
                gt_x, gt_y = ground_truth_fn(t)
                miss = math.hypot(aim_x - gt_x, aim_y - gt_y)
                miss_by_time.append(miss)
                time_for_miss.append(t)
    
    if miss_by_time:
        ax5.scatter(time_for_miss, miss_by_time, c="r", s=50, marker="x")
        ht = metrics.get("hit_threshold", 3.0)
        ax5.axhline(ht, color="g", ls="--", alpha=0.7, label=f"{ht}m threshold")
        ax5.axhline(5.0, color="orange", ls="--", alpha=0.7, label="5m threshold")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Miss Distance (m)")
    ax5.set_title("Fire Control Miss Distance")
    ax5.legend(loc="best", fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # ── Panel 6: Summary metrics ────────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    
    summary_text = f"""
    DETECTION METRICS
    ─────────────────────────
    Windows processed: {metrics['n_windows']}
    Detections: {metrics['n_detections']} ({100*metrics['n_detections']/max(metrics['n_windows'],1):.1f}%)
    Mean bearing error: {metrics['mean_bearing_error']:.1f}°
    Max bearing error: {metrics['max_bearing_error']:.1f}°
    Mean range error: {metrics['mean_range_error']:.1f} m
    
    FIRE CONTROL METRICS
    ─────────────────────────
    Shots fired: {metrics['shots_fired']}
    Hits <{metrics['hit_threshold']}m: {metrics['n_hits']} ({100*metrics['n_hits']/max(metrics['shots_fired'],1):.1f}%)
    Hits <5m: {metrics['n_hits_5m']} ({100*metrics['n_hits_5m']/max(metrics['shots_fired'],1):.1f}%)
    Mean miss: {metrics['mean_miss']:.1f} m
    Min miss: {metrics['min_miss']:.1f} m
    Max miss: {metrics['max_miss']:.1f} m
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(
    sim_dir: Path,
    output_dir: Path,
    source_speed: float = 50.0,
    fundamental: float = 180.0,
    n_harmonics: int = 4,
    hit_threshold: float = 3.0,
) -> dict:
    """Run complete detection and targeting pipeline.
    
    Args:
        sim_dir: Path to simulation output directory
        output_dir: Path for output plots and results
        source_speed: Source velocity in m/s
        fundamental: Fundamental frequency in Hz
        n_harmonics: Number of harmonics to use
        
    Returns:
        dict with all results and metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ACOUSTIC DETECTION & TARGETING PIPELINE")
    print("=" * 60)
    
    # ── Load data ───────────────────────────────────────────────────────
    print(f"\n[1/5] Loading simulation from {sim_dir}")
    data = load_simulation(sim_dir)
    traces = data["traces"]
    mic_positions = data["mic_positions"]
    dt = data["dt"]
    duration = data["duration"]
    
    print(f"      {traces.shape[0]} mics, {traces.shape[1]} samples")
    print(f"      Duration: {duration:.2f}s, dt={dt:.2e}s")
    
    array_center = tuple(np.mean(mic_positions, axis=0))
    ground_truth_fn, src_duration = compute_ground_truth(data["metadata"], source_speed)
    
    t0 = ground_truth_fn(0.0)
    tf = ground_truth_fn(src_duration)
    print(f"      Source: ({t0[0]:.1f}, {t0[1]:.1f}) → ({tf[0]:.1f}, {tf[1]:.1f})")
    print(f"      Speed: {source_speed:.1f} m/s ({source_speed*3.6:.0f} km/h)")
    
    # ── MFP Detection ───────────────────────────────────────────────────
    print(f"\n[2/5] Running MFP detection (f0={fundamental}Hz, {n_harmonics} harmonics)")
    mfp_result = run_mfp_detection(
        traces, mic_positions, dt,
        fundamental=fundamental,
        n_harmonics=n_harmonics,
    )
    detections = mfp_result["detections"]
    n_detected = sum(1 for d in detections if d["detected"])
    print(f"      {n_detected}/{len(detections)} windows detected")
    
    # ── RMS Range Estimation ────────────────────────────────────────────
    print("\n[3/5] Applying RMS-based range estimation")
    apply_rms_range_estimation(detections)
    
    # ── EKF Tracking ────────────────────────────────────────────────────
    print("\n[4/5] Running EKF tracker")
    track = run_tracking(detections, array_center)
    track_len = len(track.get("times", []))
    print(f"      Track length: {track_len} states")
    
    # ── Fire Control ────────────────────────────────────────────────────
    print("\n[5/5] Running fire control")
    fire_control = run_targeting(track, ground_truth_fn, hit_threshold=hit_threshold)
    n_shots = sum(fire_control.get("can_fire", []))
    print(f"      Shots: {n_shots}")
    
    # ── Evaluation ──────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("EVALUATION")
    print("-" * 60)
    
    metrics = evaluate_results(
        detections, track, fire_control,
        ground_truth_fn, array_center,
        hit_threshold=hit_threshold,
    )
    
    print(f"\nDetection:")
    print(f"  Mean bearing error: {metrics['mean_bearing_error']:.1f}°")
    print(f"  Mean range error:   {metrics['mean_range_error']:.1f}m")
    
    print(f"\nFire Control:")
    print(f"  Shots fired: {metrics['shots_fired']}")
    print(f"  Hits <{hit_threshold}m:  {metrics['n_hits']} ({100*metrics['n_hits']/max(metrics['shots_fired'],1):.1f}%)")
    print(f"  Hits <5m:  {metrics['n_hits_5m']} ({100*metrics['n_hits_5m']/max(metrics['shots_fired'],1):.1f}%)")
    print(f"  Mean miss: {metrics['mean_miss']:.1f}m")
    
    # ── Generate plots ──────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("GENERATING PLOTS")
    print("-" * 60)
    
    plot_path = output_dir / "pipeline_evaluation.png"
    plot_full_evaluation(
        detections, track, fire_control,
        ground_truth_fn, src_duration, array_center,
        metrics, plot_path,
    )
    
    # Radial engagement plot
    radial_path = output_dir / "radial_engagement.png"
    plot_radial_engagement(
        fire_control,
        ground_truth_fn,
        src_duration,
        weapon_pos=(0.0, 0.0),
        muzzle_velocity=400.0,
        decel=1.5,
        hit_threshold=hit_threshold,
        output_path=radial_path,
    )
    
    # ── Save results JSON ───────────────────────────────────────────────
    results_path = output_dir / "pipeline_results.json"
    results_json = {
        "simulation": str(sim_dir),
        "source_speed": source_speed,
        "fundamental": fundamental,
        "n_harmonics": n_harmonics,
        "n_detections": metrics["n_detections"],
        "n_windows": metrics["n_windows"],
        "mean_bearing_error_deg": metrics["mean_bearing_error"],
        "mean_range_error_m": metrics["mean_range_error"],
        "shots_fired": metrics["shots_fired"],
        "hit_threshold_m": metrics["hit_threshold"],
        "hits": metrics["n_hits"],
        "hits_5m": metrics["n_hits_5m"],
        "mean_miss_m": metrics["mean_miss"],
        "min_miss_m": metrics["min_miss"],
        "max_miss_m": metrics["max_miss"],
    }
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=lambda x: None if np.isnan(x) else x)
    print(f"Saved: {results_path}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return {
        "data": data,
        "detections": detections,
        "track": track,
        "fire_control": fire_control,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sim_dir",
        type=Path,
        nargs="?",
        default=Path("../output/valley_test"),
        help="Simulation output directory (default: ../output/valley_test)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as sim_dir)",
    )
    parser.add_argument(
        "--source-speed",
        type=float,
        default=50.0,
        help="Source velocity in m/s (default: 50.0)",
    )
    parser.add_argument(
        "--fundamental",
        type=float,
        default=180.0,
        help="Fundamental frequency in Hz (default: 180.0)",
    )
    parser.add_argument(
        "--n-harmonics",
        type=int,
        default=4,
        help="Number of harmonics (default: 4)",
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=3.0,
        help="Hit radius threshold in metres (default: 3.0)",
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.sim_dir
    
    run_pipeline(
        args.sim_dir,
        output_dir,
        source_speed=args.source_speed,
        fundamental=args.fundamental,
        n_harmonics=args.n_harmonics,
        hit_threshold=args.hit_threshold,
    )


if __name__ == "__main__":
    main()

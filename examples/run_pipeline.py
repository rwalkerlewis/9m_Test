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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.detection import (
    DetectionEngine,
    WindowDetection,
    available_bearing_methods,
)
from acoustic_sim.fire_control import (
    compute_bearing_rate,
    compute_engagement_3d,
    compute_lead_3d,
    find_cpa,
    pattern_diameter,
    time_of_flight,
)
from acoustic_sim.plotting import (
    plot_beamformer_diagnostic,
    plot_pipeline_summary,
    plot_radial_engagement,
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
        "method": "srp_phat",
        "n_bearings": 360,
        "freq_lo_hz": 100.0,
        "freq_hi_hz": 2000.0,
        "max_sources": 1,
    },
    "ranging": {
        "method": "auto",
        "n_range_bins": 80,
        "auto_cpa_threshold_m": 3.0,
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
        "pattern_spread_rate": 0.2,
        "max_engagement_range_m": 500.0,
        "max_position_uncertainty_m": 0.0,
        "class_label": "quadcopter",
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


def compute_ground_truth(metadata: dict, source_speed: float,
                         sim_dir: Path | None = None):
    """Build a ground-truth trajectory returning ``(x, y, z)``.

    For 2-D simulations that lack ``source_z``, z defaults to 0 and a
    sine arc is applied only when ``source_arc_height`` is explicitly
    set in *metadata*; for 3-D a parabolic arc is used similarly.
    If the metadata does not contain ``source_arc_height`` the path is
    assumed to be a straight line (matching :class:`MovingSource`
    default ``arc_height = 0``).

    For erratic trajectories (``source_type == "erratic_quadcopter"``),
    the ground-truth positions are loaded from a saved trajectory file.
    """
    # -- Erratic trajectory from file ----------------------------------
    traj_file = metadata.get("source_trajectory_file")
    if traj_file and sim_dir is not None:
        traj_path = sim_dir / traj_file
        if traj_path.exists():
            traj = np.load(str(traj_path))  # (n_steps, 3)
            dt_sim = metadata["dt"]
            duration = (traj.shape[0] - 1) * dt_sim

            def trajectory(t: float) -> tuple[float, float, float]:
                frac = min(max(t / duration, 0.0), 1.0) if duration > 0 else 0.0
                idx_f = frac * (traj.shape[0] - 1)
                i0 = int(idx_f)
                i1 = min(i0 + 1, traj.shape[0] - 1)
                w = idx_f - i0
                pos = traj[i0] * (1.0 - w) + traj[i1] * w
                return float(pos[0]), float(pos[1]), float(pos[2])

            return trajectory, duration

    start_x = metadata.get("source_x", -40.0)
    start_y = metadata.get("source_y", 0.0)
    start_z = metadata.get("source_z", 0.0)
    end_x = metadata.get("source_x1", -start_x)
    end_y = metadata.get("source_y1", start_y)
    end_z = metadata.get("source_z1", start_z)

    has_z = "source_z" in metadata
    arc_height = metadata.get("source_arc_height", 0.0)
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


# SRPBeamformer and causal_ls_fit have been moved to the
# acoustic_sim.detection module.  The pipeline now uses DetectionEngine.


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
    n_hits = sum(1 for f in fire_decisions if f.get("hit"))

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
    max_track_hist = cfg_trk.get("max_history", 20)
    min_rms = cfg_det["min_signal_rms"]
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

    ground_truth_fn, src_duration = compute_ground_truth(metadata, source_speed,
                                                          sim_dir=sim_dir)
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

    # -- detection engine -----------------------------------------------------
    c_sound = float(metadata.get("velocity", 343.0))
    bearing_method = cfg_bf.get("method", "srp_phat")
    max_sources = cfg_bf.get("max_sources", 1)
    cfg_rng = cfg.get("ranging", {})
    range_method = cfg_rng.get("method", "auto")

    # -- RMS profile (needed for fire gate regardless of range method) ------
    rms_profile = np.array([
        float(np.sqrt(np.mean(traces[:, p:p + win_len] ** 2)))
        for p in range(0, n_samples - win_len + 1, hop)
    ])
    peak_rms = float(rms_profile.max())
    peak_idx = int(np.argmax(rms_profile))
    peak_t = (peak_idx * hop + win_len / 2) * dt_sim
    gt_peak = ground_truth_fn(peak_t)
    cpa_dist = max(float(np.linalg.norm(np.array(gt_peak) - array_center)), 1.0)

    # -- auto-select range method based on CPA geometry ---------------------
    if range_method == "auto":
        cpa_threshold = cfg_rng.get("auto_cpa_threshold_m", 3.0)
        if cpa_dist <= cpa_threshold:
            range_method = "tdoa"
            print(f"       [AUTO] CPA={cpa_dist:.1f}m <= {cpa_threshold:.0f}m "
                  f"-> using TDOA ranging (RMS unreliable at near-zero CPA)")
        else:
            range_method = "rms"
            print(f"       [AUTO] CPA={cpa_dist:.1f}m > {cpa_threshold:.0f}m "
                  f"-> using RMS ranging")

    range_kwargs: dict = dict(
        range_min=range_min,
        range_max=range_max,
    )
    if range_method == "rms":
        range_kwargs["ref_range"] = rms_ref_range
    elif range_method == "bearing_rate":
        range_kwargs["source_speed"] = source_speed
        range_kwargs["hop_sec"] = hop_sec
        range_kwargs["ema_alpha"] = ema_alpha
    else:
        # tdoa / nearfield
        range_kwargs["freq_lo"] = cfg_bf["freq_lo_hz"]
        range_kwargs["freq_hi"] = cfg_bf["freq_hi_hz"]
        range_kwargs["n_range_bins"] = cfg_rng.get("n_range_bins", 80)

    engine = DetectionEngine(
        mic_positions=mic_positions,
        fs=fs,
        window_samples=win_len,
        bearing_method=bearing_method,
        range_method=range_method,
        max_sources=max_sources,
        min_signal_rms=min_rms,
        ema_alpha=ema_alpha,
        source_z_estimate=source_z_est,
        c=c_sound,
        bearing_kwargs=dict(
            n_bearings=cfg_bf["n_bearings"],
            freq_lo=cfg_bf["freq_lo_hz"],
            freq_hi=cfg_bf["freq_hi_hz"],
        ),
        range_kwargs=range_kwargs,
        tracker_min_detections=min_track_det,
        tracker_max_history=max_track_hist,
    )

    print(f"       Bearing: {bearing_method} "
          f"({cfg_bf['n_bearings']} bearings, "
          f"{cfg_bf['freq_lo_hz']}-{cfg_bf['freq_hi_hz']} Hz)"
          + (f", max_sources={max_sources}" if max_sources > 1 else "")
          + f"\n       Range: {range_method}"
          + f"\n       Available methods: {available_bearing_methods()}")

    if range_method == "rms":
        engine.calibrate_range(peak_rms, cpa_dist)
        print(f"       RMS cal: peak={peak_rms:.6f}, CPA dist={cpa_dist:.1f} m")

    # -- main loop -----------------------------------------------------------
    print(f"\n[RUN]  Streaming {n_windows} windows (causal mode)")

    all_detections: list[dict] = []
    all_track_states: list = []
    all_fire_decisions: list[dict] = []
    wall_times_list: list[float] = []
    hits = 0

    # Bearing-rate tracking for fire gate.
    bearing_history: list[tuple[float, float]] = []  # (time, bearing_rad)
    BEARING_RATE_WINDOW = 0.15  # seconds of history for rate estimate
    BEARING_RATE_THRESHOLD = 15.0  # deg/s — below this, bearing is "stable"

    pos = 0
    while pos + win_len <= n_samples:
        t0_wall = time.perf_counter()
        t_center = (pos + win_len / 2.0) * dt_sim
        seg = traces[:, pos:pos + win_len]

        det = engine.process_window(seg, t_center)

        det_dict: dict = {"time": t_center, "window_rms": det.window_rms,
                          "detected": det.detected}
        if det.detected and not math.isnan(det.bearing_rad):
            det_dict.update({
                "bearing": det.bearing_rad, "bearing_deg": det.bearing_deg,
                "range": det.range_m, "z": det.z,
                "x": det.x, "y": det.y,
            })
        all_detections.append(det_dict)

        track_state = det.track
        all_track_states.append(track_state)

        # -- fire control ----------------------------------------------------
        fire_decision: dict = {"time": t_center, "can_fire": False,
                               "reason": "NO_TRACK"}

        # Track bearing rate — if the angular position is stable the
        # target is heading roughly radially (toward or away).  Shooting
        # when bearing is constant means zero angular lead is needed;
        # just point where the target is.
        bearing_rate_dps = float("inf")
        if det.detected and not math.isnan(det.bearing_rad):
            bearing_rate_dps, bearing_history = compute_bearing_rate(
                bearing_history, t_center, det.bearing_rad,
                BEARING_RATE_WINDOW)

        stable_bearing = bearing_rate_dps < BEARING_RATE_THRESHOLD
        rms_gate = det.window_rms >= rms_fire_gate * peak_rms

        # Decide which fire mode to use:
        #  - stable bearing + rms gate → radial shot (no lead needed)
        #  - changing bearing + rms gate → lead shot (need pos/vel)
        #  - stable bearing but weak rms → wait (don't waste ammo at range)
        can_engage = det.detected and track_state is not None and rms_gate

        if can_engage and max_hits > 0 and hits >= max_hits:
            fire_decision = {"time": t_center, "can_fire": False,
                             "reason": "TARGET_ENGAGED"}
        elif can_engage and stable_bearing:
            # ----- RADIAL SHOT -----
            # Bearing isn't changing → target is on the same ray as
            # the weapon.  No lead needed.  Just shoot along the
            # current bearing; the projectile and target share the
            # same line.  We don't need position or velocity estimates.
            brg = det.bearing_rad
            elev = 0.0
            if dim == "3-D" and not math.isnan(det.z):
                raw_pos = np.array([det.x, det.y, det.z])
                dx = raw_pos - weapon_pos
                r_horiz = math.sqrt(dx[0]**2 + dx[1]**2)
                elev = math.atan2(dx[2], max(r_horiz, 1e-6))
            # Aim direction from bearing + elevation.
            # SRP-PHAT uses math convention: bearing θ → (cos θ, sin θ).
            aim_dir = np.array([
                math.cos(brg) * math.cos(elev),
                math.sin(brg) * math.cos(elev),
                math.sin(elev),
            ])
            # Use estimated range only for pattern-spread calculation.
            est_range = det.range_m if det.range_m > 0 else 20.0
            tof = time_of_flight(est_range, muzzle_velocity, pellet_decel)
            if tof == float("inf"):
                tof = est_range / muzzle_velocity
            intercept = weapon_pos + aim_dir * est_range
            pat_diam = pattern_diameter(est_range, pattern_spread_rate)

            # Evaluate hit: find closest point of approach along
            # the entire flight path (projectile vs moving target).
            cpa = find_cpa(weapon_pos, aim_dir, muzzle_velocity,
                          pellet_decel, ground_truth_fn, t_center)
            miss = cpa["miss"]
            cpa_range = cpa["cpa_range"]
            in_front = cpa["in_range"]

            pat_diam_cpa = pattern_diameter(max(cpa_range, 0.1),
                                            pattern_spread_rate)
            pattern_radius = pat_diam_cpa / 2.0
            effective_threshold = max(pattern_radius, hit_threshold)

            cpa_pos = cpa["cpa_pos"]
            fire_decision = {
                "time": t_center,
                "can_fire": in_front,
                "reason": "RADIAL_FIRE" if in_front else "BEHIND",
                "est_pos": intercept.tolist(),
                "intercept_pos": cpa_pos.tolist(),
                "aim_bearing": brg,
                "aim_elevation": elev,
                "tof": tof,
                "cpa_range": cpa_range,
                "range": est_range,
                "pattern_diam": pat_diam_cpa,
                "pos_unc": 0.0,
                "miss": miss,
                "pattern_radius": pattern_radius,
            }
            if in_front and miss < effective_threshold:
                hits += 1
                fire_decision["hit"] = True
        elif can_engage and not stable_bearing:
            # ----- LEAD SHOT -----
            # Bearing is changing → need position + velocity for lead.
            est_vel = track_state.velocity
            raw_pos = np.array([det.x, det.y, det.z])
            fit_pos = track_state.position_at(t_center)
            rng = float(np.linalg.norm(raw_pos - weapon_pos))
            blend = min(rng / 20.0, 1.0)
            est_pos = (1.0 - blend) * raw_pos + blend * fit_pos
            cov = track_state.covariance_6x6(floor=cov_floor, cap=cov_cap)

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
                # Find closest point of approach between the pellet
                # cone and the moving target along the entire flight.
                aim_dir = lead["intercept_pos"] - weapon_pos
                aim_len = float(np.linalg.norm(aim_dir))
                if aim_len > 1e-6:
                    aim_dir = aim_dir / aim_len
                cpa = find_cpa(weapon_pos, aim_dir, muzzle_velocity,
                              pellet_decel, ground_truth_fn, t_center)
                miss = cpa["miss"]
                cpa_range = cpa["cpa_range"]
                pat_r_at_gt = pattern_diameter(max(cpa_range, 0.1),
                                               pattern_spread_rate) / 2.0
                effective_threshold = max(pat_r_at_gt, hit_threshold)
                fire_decision["miss"] = miss
                fire_decision["pattern_radius"] = pat_r_at_gt
                fire_decision["cpa_range"] = cpa_range
                fire_decision["intercept_pos"] = cpa["cpa_pos"].tolist()
                if cpa["in_range"] and miss < effective_threshold:
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
    # Compute pattern info for display.
    pattern_rads = [f.get("pattern_radius", 0) for f in all_fire_decisions
                    if f.get("can_fire") and f.get("pattern_radius")]
    avg_pat_rad = np.mean(pattern_rads) if pattern_rads else 0.0
    print(f"  Hits:       {n_hits_val} "
          f"({100 * n_hits_val / max(n_shots, 1):.1f}%)"
          f"  [pattern radius={avg_pat_rad:.1f}m, "
          f"point threshold={hit_threshold}m]")
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

    plot_pipeline_summary(
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
        array_center, engine.bearing_estimator, cfg_det,
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

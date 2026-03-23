#!/usr/bin/env python3
"""Coupled acoustic + seismic detection and engagement pipeline.

Processes microphone (air) and geophone (ground) data from a coupled
elastic FDTD simulation.  The seismic wave-field propagates at the
ground P-wave velocity (~1500 m/s), so geophones detect the drone
well before microphones (c_air ~ 343 m/s).  The pipeline runs two
independent detection streams and fuses them for tracking and fire
control.

Usage::

    python examples/run_pipeline_coupled.py output/coupled_moving_3d

    python examples/run_pipeline_coupled.py output/coupled_moving_3d \\
        --config examples/pipeline_coupled.config.json
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
)
from acoustic_sim.fire_control import (
    compute_bearing_rate,
    compute_engagement_3d,
    compute_lead_3d,
    find_cpa,
    pattern_diameter,
    time_of_flight,
)

# ============================================================================
# Defaults
# ============================================================================

_DEFAULTS = {
    "acoustic": {
        "detection": {
            "window_length_s": 0.05,
            "window_overlap": 0.75,
            "min_signal_rms": 1e-5,
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
            "min_detections": 3,
            "max_history": 20,
            "ema_alpha": 0.35,
            "range_min_m": 5.0,
            "range_max_m": 150.0,
            "rms_ref_range_m": 10.0,
            "covariance_floor": 0.5,
            "covariance_cap": 1.0,
        },
    },
    "seismic": {
        "detection": {
            "window_length_s": 0.02,
            "window_overlap": 0.75,
            "min_signal_rms": 1e-10,
        },
        "beamformer": {
            "method": "srp_phat",
            "n_bearings": 360,
            "freq_lo_hz": 10.0,
            "freq_hi_hz": 500.0,
            "max_sources": 1,
        },
        "ranging": {
            "method": "rms",
            "n_range_bins": 80,
        },
        "tracking": {
            "min_detections": 3,
            "max_history": 30,
            "ema_alpha": 0.45,
            "range_min_m": 5.0,
            "range_max_m": 200.0,
            "rms_ref_range_m": 10.0,
            "covariance_floor": 1.0,
            "covariance_cap": 3.0,
        },
    },
    "fusion": {
        "bearing_weight_acoustic": 0.6,
        "bearing_weight_seismic": 0.4,
        "range_weight_acoustic": 0.5,
        "range_weight_seismic": 0.5,
        "seismic_early_warning_only": False,
    },
    "fire_control": {
        "muzzle_velocity_mps": 400.0,
        "pellet_decel_mps2": 1.5,
        "pattern_spread_rate": 0.3,
        "max_engagement_range_m": 500.0,
        "max_position_uncertainty_m": 0.0,
        "class_label": "quadcopter",
        "class_confidence": 0.9,
        "hit_threshold_m": 2.0,
        "max_hits": 3,
        "rms_fire_gate_frac": 0.20,
    },
    "source": {
        "speed_mps": 50.0,
        "altitude_estimate_m": None,
    },
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None) -> dict:
    if path is not None and path.exists():
        with open(path) as f:
            user = json.load(f)
        return _deep_merge(_DEFAULTS, user)
    return dict(_DEFAULTS)


# ============================================================================
# Data
# ============================================================================

def load_coupled_simulation(sim_dir: Path) -> dict:
    """Load mic and geophone traces + metadata from a coupled simulation."""
    with open(sim_dir / "metadata.json") as f:
        metadata = json.load(f)

    mic_traces = np.load(sim_dir / "mic_traces.npy")
    geo_traces = np.load(sim_dir / "geo_traces.npy")
    all_traces = np.load(sim_dir / "traces.npy")

    mic_positions = np.array(metadata["mic_positions"], dtype=np.float64)
    geo_positions = np.array(metadata["geo_positions"], dtype=np.float64)

    dt = metadata["dt"]
    c_air = float(metadata.get("velocity", 343.0))
    ground = metadata.get("ground", {})
    c_ground = float(ground.get("vp", 1500.0))

    return {
        "mic_traces": mic_traces,
        "geo_traces": geo_traces,
        "all_traces": all_traces,
        "mic_positions": mic_positions,
        "geo_positions": geo_positions,
        "metadata": metadata,
        "dt": dt,
        "c_air": c_air,
        "c_ground": c_ground,
        "fs": 1.0 / dt,
        "duration": mic_traces.shape[1] * dt,
    }


def compute_ground_truth(metadata: dict, source_speed: float):
    """Build ground-truth trajectory from metadata."""
    start_x = metadata.get("source_x", metadata.get("source_start", [-40])[0])
    start_y = metadata.get("source_y", metadata.get("source_start", [0, 0])[1])
    start_z = metadata.get("source_z", metadata.get("source_start", [0, 0, 0])[2])
    end_x = metadata.get("source_x1", metadata.get("source_end", [40])[0])
    end_y = metadata.get("source_y1", metadata.get("source_end", [0, 0])[1])
    end_z = metadata.get("source_z1", metadata.get("source_end", [0, 0, 0])[2])
    arc_height = metadata.get("source_arc_height", 0.0)

    horiz_dist = math.hypot(end_x - start_x, end_y - start_y)
    duration = horiz_dist / source_speed if source_speed > 0 else 3.0

    def trajectory(t: float) -> tuple[float, float, float]:
        frac = min(max(t / duration, 0.0), 1.0)
        x = start_x + (end_x - start_x) * frac
        y = (start_y + (end_y - start_y) * frac
             + arc_height * 4.0 * frac * (1.0 - frac))
        z = start_z + (end_z - start_z) * frac
        return x, y, z

    return trajectory, duration


# ============================================================================
# Single-modality stream processor
# ============================================================================

def _build_engine(
    positions: np.ndarray,
    fs: float,
    win_len: int,
    c: float,
    cfg_bf: dict,
    cfg_rng: dict,
    cfg_trk: dict,
    cfg_det: dict,
    source_z_est: float,
) -> DetectionEngine:
    """Build a DetectionEngine for one sensor modality."""
    bearing_method = cfg_bf.get("method", "srp_phat")
    max_sources = cfg_bf.get("max_sources", 1)
    range_method = cfg_rng.get("method", "rms")

    range_kwargs: dict = dict(
        range_min=cfg_trk.get("range_min_m", 5.0),
        range_max=cfg_trk.get("range_max_m", 150.0),
    )
    if range_method == "rms":
        range_kwargs["ref_range"] = cfg_trk.get("rms_ref_range_m", 10.0)
    elif range_method not in ("rms", "auto"):
        range_kwargs["freq_lo"] = cfg_bf["freq_lo_hz"]
        range_kwargs["freq_hi"] = cfg_bf["freq_hi_hz"]
        range_kwargs["n_range_bins"] = cfg_rng.get("n_range_bins", 80)

    return DetectionEngine(
        mic_positions=positions,
        fs=fs,
        window_samples=win_len,
        bearing_method=bearing_method,
        range_method=range_method,
        max_sources=max_sources,
        min_signal_rms=cfg_det.get("min_signal_rms", 1e-5),
        ema_alpha=cfg_trk.get("ema_alpha", 0.35),
        source_z_estimate=source_z_est,
        c=c,
        bearing_kwargs=dict(
            n_bearings=cfg_bf.get("n_bearings", 360),
            freq_lo=cfg_bf["freq_lo_hz"],
            freq_hi=cfg_bf["freq_hi_hz"],
        ),
        range_kwargs=range_kwargs,
        tracker_min_detections=cfg_trk.get("min_detections", 3),
        tracker_max_history=cfg_trk.get("max_history", 20),
    )


def _process_stream(
    engine: DetectionEngine,
    traces: np.ndarray,
    fs: float,
    win_len: int,
    hop: int,
    dt: float,
    peak_rms: float,
    cpa_dist: float,
    range_method: str,
) -> list[dict]:
    """Run a detection engine over all windows and return detections."""
    if range_method == "rms":
        engine.calibrate_range(peak_rms, cpa_dist)

    n_samples = traces.shape[1]
    detections: list[dict] = []
    pos = 0
    while pos + win_len <= n_samples:
        t_center = (pos + win_len / 2.0) * dt
        seg = traces[:, pos:pos + win_len]
        det = engine.process_window(seg, t_center)
        d: dict = {
            "time": t_center,
            "window_rms": det.window_rms,
            "detected": det.detected,
        }
        if det.detected and not math.isnan(det.bearing_rad):
            d.update({
                "bearing": det.bearing_rad,
                "bearing_deg": det.bearing_deg,
                "range": det.range_m,
                "x": det.x, "y": det.y, "z": det.z,
            })
        if det.track is not None:
            d["track"] = det.track
        detections.append(d)
        pos += hop
    return detections


# ============================================================================
# Fusion
# ============================================================================

def _fuse_detections(
    acoustic_dets: list[dict],
    seismic_dets: list[dict],
    cfg_fuse: dict,
    array_center: np.ndarray,
) -> list[dict]:
    """Merge acoustic and seismic detection streams.

    For each time step that both sensors report, fuse bearing and range
    using configurable weights.  When only one modality detects, use
    that result directly.
    """
    w_brg_a = cfg_fuse.get("bearing_weight_acoustic", 0.6)
    w_brg_s = cfg_fuse.get("bearing_weight_seismic", 0.4)
    w_rng_a = cfg_fuse.get("range_weight_acoustic", 0.5)
    w_rng_s = cfg_fuse.get("range_weight_seismic", 0.5)
    early_only = cfg_fuse.get("seismic_early_warning_only", False)

    # Index seismic detections by time for lookup.
    seis_by_time: dict[float, dict] = {}
    for d in seismic_dets:
        seis_by_time[round(d["time"], 8)] = d

    fused: list[dict] = []

    # Walk all unique times from both streams.
    all_times: dict[float, None] = {}
    for d in acoustic_dets:
        all_times[round(d["time"], 8)] = None
    for d in seismic_dets:
        all_times[round(d["time"], 8)] = None

    acou_by_time: dict[float, dict] = {}
    for d in acoustic_dets:
        acou_by_time[round(d["time"], 8)] = d

    for t_key in sorted(all_times):
        a = acou_by_time.get(t_key)
        s = seis_by_time.get(t_key)

        a_det = a is not None and a.get("detected") and "bearing" in a
        s_det = s is not None and s.get("detected") and "bearing" in s

        if not a_det and not s_det:
            # Neither detected.
            src = a or s
            fused.append({
                "time": src["time"],
                "window_rms": src.get("window_rms", 0.0),
                "detected": False,
                "source": "none",
            })
            continue

        if a_det and s_det and not early_only:
            # Both modalities — weighted fusion.
            brg_a = a["bearing"]
            brg_s = s["bearing"]
            # Circular mean of two bearings.
            cx = w_brg_a * math.cos(brg_a) + w_brg_s * math.cos(brg_s)
            cy = w_brg_a * math.sin(brg_a) + w_brg_s * math.sin(brg_s)
            fused_brg = math.atan2(cy, cx) % (2 * math.pi)
            fused_rng = w_rng_a * a["range"] + w_rng_s * s["range"]
            fused_x = array_center[0] + fused_rng * math.cos(fused_brg)
            fused_y = array_center[1] + fused_rng * math.sin(fused_brg)
            fused_z = a.get("z", s.get("z", 0.0))
            fused.append({
                "time": a["time"],
                "window_rms": max(a["window_rms"], s["window_rms"]),
                "detected": True,
                "source": "fused",
                "bearing": fused_brg,
                "bearing_deg": math.degrees(fused_brg) % 360,
                "range": fused_rng,
                "x": fused_x, "y": fused_y, "z": fused_z,
                "acoustic_bearing_deg": a["bearing_deg"],
                "seismic_bearing_deg": s["bearing_deg"],
                "acoustic_range": a["range"],
                "seismic_range": s["range"],
            })
        elif a_det:
            entry = dict(a)
            entry["source"] = "acoustic"
            fused.append(entry)
        else:
            entry = dict(s)
            entry["source"] = "seismic"
            fused.append(entry)

    return fused


# ============================================================================
# Evaluation
# ============================================================================

def _evaluate(
    detections: list[dict],
    ground_truth_fn,
    array_center: np.ndarray,
    label: str,
) -> dict:
    """Bearing and range error metrics for a detection stream."""
    cx, cy = array_center[0], array_center[1]
    bearing_errors: list[float] = []
    range_errors: list[float] = []
    first_detection_t: float | None = None

    for d in detections:
        if not d.get("detected"):
            continue
        t = d["time"]
        if first_detection_t is None:
            first_detection_t = t
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
        if "range" in d:
            range_errors.append(abs(d["range"] - true_range))

    return {
        "label": label,
        "n_detections": sum(1 for d in detections if d.get("detected")),
        "n_windows": len(detections),
        "first_detection_s": first_detection_t,
        "mean_bearing_error": float(np.mean(bearing_errors)) if bearing_errors else float("nan"),
        "max_bearing_error": float(np.max(bearing_errors)) if bearing_errors else float("nan"),
        "mean_range_error": float(np.mean(range_errors)) if range_errors else float("nan"),
    }


# ============================================================================
# Fire control (on fused detections)
# ============================================================================

def _run_fire_control(
    fused_dets: list[dict],
    ground_truth_fn,
    weapon_pos: np.ndarray,
    cfg_fc: dict,
    peak_rms: float,
) -> list[dict]:
    """Simplified fire-control loop on fused detections."""
    muzzle_velocity = cfg_fc["muzzle_velocity_mps"]
    pellet_decel = cfg_fc["pellet_decel_mps2"]
    pattern_spread = cfg_fc["pattern_spread_rate"]
    hit_thresh = cfg_fc["hit_threshold_m"]
    max_hits = cfg_fc["max_hits"]
    rms_fire_gate = cfg_fc.get("rms_fire_gate_frac", 0.20)
    hits = 0

    bearing_history: list[tuple[float, float]] = []
    BEARING_RATE_WINDOW = 0.15
    BEARING_RATE_THRESH = 15.0

    fire_decisions: list[dict] = []

    for d in fused_dets:
        t = d["time"]
        fd: dict = {"time": t, "can_fire": False, "reason": "NO_DETECTION"}

        if not d.get("detected"):
            fire_decisions.append(fd)
            continue

        rms_gate = d.get("window_rms", 0.0) >= rms_fire_gate * peak_rms

        bearing_rate_dps = float("inf")
        if "bearing" in d:
            bearing_rate_dps, bearing_history = compute_bearing_rate(
                bearing_history, t, d["bearing"], BEARING_RATE_WINDOW)
        stable = bearing_rate_dps < BEARING_RATE_THRESH

        if max_hits > 0 and hits >= max_hits:
            fd["reason"] = "TARGET_ENGAGED"
            fire_decisions.append(fd)
            continue

        if not rms_gate:
            fd["reason"] = "RMS_GATE"
            fire_decisions.append(fd)
            continue

        brg = d["bearing"]
        est_range = d.get("range", 20.0) if d.get("range", 0) > 0 else 20.0
        elev = 0.0
        if "z" in d and not math.isnan(d["z"]):
            dx = np.array([d["x"], d["y"], d["z"]]) - weapon_pos
            r_h = math.sqrt(dx[0]**2 + dx[1]**2)
            elev = math.atan2(dx[2], max(r_h, 1e-6))

        aim_dir = np.array([
            math.cos(brg) * math.cos(elev),
            math.sin(brg) * math.cos(elev),
            math.sin(elev),
        ])

        tof = time_of_flight(est_range, muzzle_velocity, pellet_decel)
        if tof == float("inf"):
            tof = est_range / muzzle_velocity
        intercept = weapon_pos + aim_dir * est_range

        cpa = find_cpa(weapon_pos, aim_dir, muzzle_velocity,
                       pellet_decel, ground_truth_fn, t)
        miss = cpa["miss"]
        cpa_range = cpa["cpa_range"]
        in_front = cpa["in_range"]

        pat_diam = pattern_diameter(max(cpa_range, 0.1), pattern_spread)
        effective_thresh = max(pat_diam / 2.0, hit_thresh)

        fd = {
            "time": t,
            "can_fire": in_front and stable,
            "reason": "FIRE" if (in_front and stable) else "WAIT",
            "est_pos": intercept.tolist(),
            "intercept_pos": cpa["cpa_pos"].tolist(),
            "aim_bearing": brg,
            "aim_elevation": elev,
            "tof": tof,
            "range": est_range,
            "cpa_range": cpa_range,
            "pattern_diam": pat_diam,
            "miss": miss,
            "source": d.get("source", "unknown"),
        }
        if in_front and stable and miss < effective_thresh:
            hits += 1
            fd["hit"] = True
        if max_hits > 0 and hits >= max_hits:
            fire_decisions.append(fd)
            break
        fire_decisions.append(fd)

    return fire_decisions


# ============================================================================
# Plotting
# ============================================================================

def _plot_coupled_summary(
    acoustic_dets: list[dict],
    seismic_dets: list[dict],
    fused_dets: list[dict],
    fire_decisions: list[dict],
    ground_truth_fn,
    source_duration: float,
    array_center: np.ndarray,
    weapon_pos: np.ndarray,
    metrics: dict,
    output_path: Path,
) -> None:
    """8-panel summary: acoustic vs seismic timelines, fusion, fire control."""
    gt_times = np.linspace(0, source_duration, 300)
    gt_xyz = np.array([ground_truth_fn(t) for t in gt_times])
    cx, cy = float(array_center[0]), float(array_center[1])
    gt_brg = np.degrees(np.arctan2(gt_xyz[:, 1] - cy, gt_xyz[:, 0] - cx)) % 360
    gt_rng = np.sqrt((gt_xyz[:, 0] - cx)**2 + (gt_xyz[:, 1] - cy)**2)

    fig, axes = plt.subplots(4, 2, figsize=(18, 22))

    # ── Row 0: Spatial overview ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], "g-", lw=2, label="True path")
    for dets, color, lbl in [
        (acoustic_dets, "blue", "Acoustic"),
        (seismic_dets, "red", "Seismic"),
    ]:
        xs = [d["x"] for d in dets if d.get("detected") and "x" in d]
        ys = [d["y"] for d in dets if d.get("detected") and "y" in d]
        if xs:
            ax.scatter(xs, ys, c=color, s=12, alpha=0.4, label=lbl)
    # Fused track.
    fx = [d["x"] for d in fused_dets if d.get("detected") and "x" in d]
    fy = [d["y"] for d in fused_dets if d.get("detected") and "y" in d]
    if fx:
        ax.plot(fx, fy, "m-", lw=1.5, alpha=0.7, label="Fused track")
    for fd in fire_decisions:
        if fd.get("can_fire") and "intercept_pos" in fd:
            ip = fd["intercept_pos"]
            color = "lime" if fd.get("hit") else "red"
            ax.scatter(ip[0], ip[1], c=color, s=100, marker="x", zorder=10)
    ax.scatter(weapon_pos[0], weapon_pos[1], c="k", s=200, marker="*",
               label="Weapon/Array", zorder=15)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Spatial overview (X-Y)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── Row 0 col 1: RMS comparison ─────────────────────────────────
    ax = axes[0, 1]
    a_times = [d["time"] for d in acoustic_dets]
    a_rms = [d["window_rms"] for d in acoustic_dets]
    s_times = [d["time"] for d in seismic_dets]
    s_rms = [d["window_rms"] for d in seismic_dets]
    ax.semilogy(a_times, a_rms, "b-", lw=1, alpha=0.8, label="Acoustic RMS")
    ax.semilogy(s_times, s_rms, "r-", lw=1, alpha=0.8, label="Seismic RMS")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Window RMS")
    ax.set_title("Signal level comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 1: Bearing timelines ─────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(gt_times, gt_brg, "g-", lw=2, label="True bearing")
    for dets, color, lbl in [
        (acoustic_dets, "blue", "Acoustic"),
        (seismic_dets, "red", "Seismic"),
    ]:
        ts = [d["time"] for d in dets if d.get("detected") and "bearing_deg" in d]
        bs = [d["bearing_deg"] for d in dets if d.get("detected") and "bearing_deg" in d]
        if ts:
            ax.scatter(ts, bs, c=color, s=8, alpha=0.5, label=lbl)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bearing [deg]")
    ax.set_title("Bearing: acoustic vs seismic")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 1 col 1: Fused bearing ───────────────────────────────────
    ax = axes[1, 1]
    ax.plot(gt_times, gt_brg, "g-", lw=2, label="True bearing")
    f_ts = [d["time"] for d in fused_dets
            if d.get("detected") and "bearing_deg" in d]
    f_bs = [d["bearing_deg"] for d in fused_dets
            if d.get("detected") and "bearing_deg" in d]
    if f_ts:
        colors = []
        for d in fused_dets:
            if d.get("detected") and "bearing_deg" in d:
                src = d.get("source", "none")
                colors.append(
                    "blue" if src == "acoustic"
                    else "red" if src == "seismic"
                    else "purple"
                )
        ax.scatter(f_ts, f_bs, c=colors, s=10, alpha=0.6, label="Fused")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bearing [deg]")
    ax.set_title("Fused bearing (blue=acoustic, red=seismic, purple=both)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 2: Range timelines ───────────────────────────────────────
    ax = axes[2, 0]
    ax.plot(gt_times, gt_rng, "g-", lw=2, label="True range")
    for dets, color, lbl in [
        (acoustic_dets, "blue", "Acoustic"),
        (seismic_dets, "red", "Seismic"),
    ]:
        ts = [d["time"] for d in dets if d.get("detected") and "range" in d
              and not math.isnan(d["range"])]
        rs = [d["range"] for d in dets if d.get("detected") and "range" in d
              and not math.isnan(d["range"])]
        if ts:
            ax.scatter(ts, rs, c=color, s=8, alpha=0.5, label=lbl)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Range [m]")
    ax.set_title("Range: acoustic vs seismic")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 2 col 1: Detection timeline ──────────────────────────────
    ax = axes[2, 1]
    a_det_t = [d["time"] for d in acoustic_dets if d.get("detected")]
    s_det_t = [d["time"] for d in seismic_dets if d.get("detected")]
    # Marker width: hop duration for each modality.
    a_hop_s = (acoustic_dets[1]["time"] - acoustic_dets[0]["time"]
               if len(acoustic_dets) > 1 else 0.001)
    s_hop_s = (seismic_dets[1]["time"] - seismic_dets[0]["time"]
               if len(seismic_dets) > 1 else 0.001)
    if s_det_t:
        for t in s_det_t:
            ax.barh("Seismic", width=s_hop_s, left=t,
                    color="red", alpha=0.3, height=0.5)
        ax.axvline(s_det_t[0], color="red", ls="--", lw=1.5,
                   label=f"Seismic 1st: {s_det_t[0]*1e3:.1f} ms")
    if a_det_t:
        for t in a_det_t:
            ax.barh("Acoustic", width=a_hop_s, left=t,
                    color="blue", alpha=0.3, height=0.5)
        ax.axvline(a_det_t[0], color="blue", ls="--", lw=1.5,
                   label=f"Acoustic 1st: {a_det_t[0]*1e3:.1f} ms")
    if s_det_t and a_det_t:
        lead = (a_det_t[0] - s_det_t[0]) * 1e3
        ax.set_title(f"Detection timeline  (seismic leads by {lead:.1f} ms)")
    else:
        ax.set_title("Detection timeline")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    # ── Row 3: Fire control ──────────────────────────────────────────
    ax = axes[3, 0]
    fire_t = [fd["time"] for fd in fire_decisions if fd.get("can_fire")]
    miss_vals = [fd["miss"] for fd in fire_decisions
                 if fd.get("can_fire") and "miss" in fd]
    if fire_t and miss_vals:
        colors = ["lime" if fd.get("hit") else "red"
                  for fd in fire_decisions
                  if fd.get("can_fire") and "miss" in fd]
        ax.scatter(fire_t[:len(miss_vals)], miss_vals, c=colors, s=40,
                   edgecolors="k", linewidths=0.5, zorder=5)
        ax.axhline(metrics["fire"]["hit_threshold"],
                   color="orange", ls="--", lw=1, label="Hit threshold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Miss distance [m]")
    ax.set_title("Fire control — miss distances")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 3 col 1: Summary text ────────────────────────────────────
    ax = axes[3, 1]
    ax.axis("off")
    m_a = metrics["acoustic"]
    m_s = metrics["seismic"]
    m_f = metrics["fused"]
    m_fc = metrics["fire"]
    lines = [
        "COUPLED PIPELINE RESULTS",
        "=" * 40,
        "",
        f"Acoustic:  {m_a['n_detections']}/{m_a['n_windows']} windows detected",
        f"  First detection:  {m_a['first_detection_s']:.4f} s" if m_a['first_detection_s'] else "  No detections",
        f"  Mean bearing err: {m_a['mean_bearing_error']:.1f} deg",
        "",
        f"Seismic:   {m_s['n_detections']}/{m_s['n_windows']} windows detected",
        f"  First detection:  {m_s['first_detection_s']:.4f} s" if m_s['first_detection_s'] else "  No detections",
        f"  Mean bearing err: {m_s['mean_bearing_error']:.1f} deg",
        "",
        f"Fused:     {m_f['n_detections']}/{m_f['n_windows']} windows detected",
        f"  Mean bearing err: {m_f['mean_bearing_error']:.1f} deg",
        "",
        f"Ground Vp = {metrics.get('c_ground', '?')} m/s  vs  c_air = {metrics.get('c_air', 343)} m/s",
    ]
    if m_s['first_detection_s'] and m_a['first_detection_s']:
        lead_ms = (m_a['first_detection_s'] - m_s['first_detection_s']) * 1e3
        lines.append(f"Seismic early warning: {lead_ms:.1f} ms")
    lines += [
        "",
        f"Shots: {m_fc['shots']}   Hits: {m_fc['hits']}",
        f"Hit threshold: {m_fc['hit_threshold']:.1f} m",
    ]
    if m_fc.get("mean_miss") is not None:
        lines.append(f"Mean miss: {m_fc['mean_miss']:.1f} m")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, va="top", ha="left", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Coupled Acoustic + Seismic Pipeline", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(sim_dir: Path, output_dir: Path, cfg: dict) -> dict:
    """Run the coupled acoustic + seismic pipeline."""
    cfg_acou = cfg["acoustic"]
    cfg_seis = cfg["seismic"]
    cfg_fuse = cfg["fusion"]
    cfg_fc = cfg["fire_control"]
    cfg_src = cfg["source"]

    source_speed = cfg_src["speed_mps"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load ----------------------------------------------------------------
    data = load_coupled_simulation(sim_dir)
    mic_traces = data["mic_traces"]
    geo_traces = data["geo_traces"]
    mic_pos = data["mic_positions"]
    geo_pos = data["geo_positions"]
    metadata = data["metadata"]
    dt = data["dt"]
    fs = data["fs"]
    c_air = data["c_air"]
    c_ground = data["c_ground"]

    mic_center = mic_pos.mean(axis=0)
    geo_center = geo_pos.mean(axis=0)
    array_center = mic_center.copy()
    weapon_pos = array_center.copy()

    ground_truth_fn, src_duration = compute_ground_truth(metadata, source_speed)

    source_z_est = cfg_src.get("altitude_estimate_m")
    if source_z_est is None:
        source_z_est = float(metadata.get("source_z", 10.0))

    n_mic, n_samples = mic_traces.shape
    n_geo = geo_traces.shape[0]

    print("=" * 65)
    print("COUPLED ACOUSTIC + SEISMIC ENGAGEMENT PIPELINE")
    print("=" * 65)
    print(f"\n[LOAD]  {sim_dir}")
    print(f"        {n_mic} microphones, {n_geo} geophones, {n_samples} samples")
    print(f"        dt = {dt:.2e} s,  fs = {fs:.0f} Hz")
    print(f"        c_air  = {c_air:.0f} m/s")
    print(f"        c_ground (Vp) = {c_ground:.0f} m/s  "
          f"({c_ground / c_air:.1f}x faster)")
    print(f"        Mic centre: ({mic_center[0]:.1f}, "
          f"{mic_center[1]:.1f}, {mic_center[2]:.1f})")
    print(f"        Geo centre: ({geo_center[0]:.1f}, "
          f"{geo_center[1]:.1f}, {geo_center[2]:.1f})")

    # -- Acoustic stream setup -----------------------------------------------
    cfg_a_det = cfg_acou["detection"]
    cfg_a_bf = cfg_acou["beamformer"]
    cfg_a_rng = cfg_acou.get("ranging", {})
    cfg_a_trk = cfg_acou["tracking"]

    a_win_len = max(int(round(cfg_a_det["window_length_s"] * fs)), 1)
    a_hop = max(int(round(a_win_len * (1.0 - cfg_a_det["window_overlap"]))), 1)

    a_range_method = cfg_a_rng.get("method", "rms")
    # Auto-select for acoustic.
    a_rms_profile = np.array([
        float(np.sqrt(np.mean(mic_traces[:, p:p+a_win_len]**2)))
        for p in range(0, n_samples - a_win_len + 1, a_hop)
    ])
    a_peak_rms = float(a_rms_profile.max()) if len(a_rms_profile) else 1e-10
    a_peak_idx = int(np.argmax(a_rms_profile)) if len(a_rms_profile) else 0
    a_peak_t = (a_peak_idx * a_hop + a_win_len / 2) * dt
    a_gt_peak = ground_truth_fn(a_peak_t)
    a_cpa_dist = max(float(np.linalg.norm(
        np.array(a_gt_peak) - mic_center)), 1.0)

    if a_range_method == "auto":
        cpa_thr = cfg_a_rng.get("auto_cpa_threshold_m", 3.0)
        a_range_method = "tdoa" if a_cpa_dist <= cpa_thr else "rms"
    # Override in the config copy for engine build.
    cfg_a_rng_eff = dict(cfg_a_rng, method=a_range_method)

    acoustic_engine = _build_engine(
        mic_pos, fs, a_win_len, c_air,
        cfg_a_bf, cfg_a_rng_eff, cfg_a_trk, cfg_a_det,
        source_z_est,
    )
    print(f"\n[ACOUSTIC] window={cfg_a_det['window_length_s']*1e3:.0f} ms, "
          f"hop={a_hop * dt * 1e3:.1f} ms, "
          f"range_method={a_range_method}")

    # -- Seismic stream setup ------------------------------------------------
    cfg_s_det = cfg_seis["detection"]
    cfg_s_bf = cfg_seis["beamformer"]
    cfg_s_rng = cfg_seis.get("ranging", {})
    cfg_s_trk = cfg_seis["tracking"]

    s_win_len = max(int(round(cfg_s_det["window_length_s"] * fs)), 1)
    s_hop = max(int(round(s_win_len * (1.0 - cfg_s_det["window_overlap"]))), 1)

    s_range_method = cfg_s_rng.get("method", "rms")
    s_rms_profile = np.array([
        float(np.sqrt(np.mean(geo_traces[:, p:p+s_win_len]**2)))
        for p in range(0, n_samples - s_win_len + 1, s_hop)
    ])
    s_peak_rms = float(s_rms_profile.max()) if len(s_rms_profile) else 1e-10
    s_peak_idx = int(np.argmax(s_rms_profile)) if len(s_rms_profile) else 0
    s_peak_t = (s_peak_idx * s_hop + s_win_len / 2) * dt
    s_gt_peak = ground_truth_fn(s_peak_t)
    s_cpa_dist = max(float(np.linalg.norm(
        np.array(s_gt_peak) - geo_center)), 1.0)

    seismic_engine = _build_engine(
        geo_pos, fs, s_win_len, c_ground,
        cfg_s_bf, cfg_s_rng, cfg_s_trk, cfg_s_det,
        source_z_est,
    )
    print(f"[SEISMIC]  window={cfg_s_det['window_length_s']*1e3:.0f} ms, "
          f"hop={s_hop * dt * 1e3:.1f} ms, "
          f"range_method={s_range_method}")

    # -- Process both streams ------------------------------------------------
    t0 = time.perf_counter()

    print(f"\n[RUN]  Processing acoustic stream ...")
    acoustic_dets = _process_stream(
        acoustic_engine, mic_traces, fs, a_win_len, a_hop, dt,
        a_peak_rms, a_cpa_dist, a_range_method,
    )

    print(f"[RUN]  Processing seismic stream ...")
    seismic_dets = _process_stream(
        seismic_engine, geo_traces, fs, s_win_len, s_hop, dt,
        s_peak_rms, s_cpa_dist, s_range_method,
    )

    wall_both = time.perf_counter() - t0

    n_a_det = sum(1 for d in acoustic_dets if d.get("detected"))
    n_s_det = sum(1 for d in seismic_dets if d.get("detected"))
    a_first = next((d["time"] for d in acoustic_dets if d.get("detected")), None)
    s_first = next((d["time"] for d in seismic_dets if d.get("detected")), None)
    print(f"       Acoustic: {n_a_det}/{len(acoustic_dets)} detected"
          + (f", first at {a_first*1e3:.1f} ms" if a_first else ""))
    print(f"       Seismic:  {n_s_det}/{len(seismic_dets)} detected"
          + (f", first at {s_first*1e3:.1f} ms" if s_first else ""))
    if s_first and a_first:
        print(f"       >> Seismic leads acoustic by "
              f"{(a_first - s_first)*1e3:.1f} ms")

    # -- Fuse ----------------------------------------------------------------
    print(f"\n[FUSE] Merging acoustic + seismic detections ...")
    fused_dets = _fuse_detections(
        acoustic_dets, seismic_dets, cfg_fuse, array_center)
    n_f_det = sum(1 for d in fused_dets if d.get("detected"))
    print(f"       Fused: {n_f_det} detections")

    # -- Fire control --------------------------------------------------------
    print(f"\n[FIRE] Running fire control on fused detections ...")
    fire_decisions = _run_fire_control(
        fused_dets, ground_truth_fn, weapon_pos, cfg_fc,
        max(a_peak_rms, s_peak_rms),
    )

    n_shots = sum(1 for f in fire_decisions if f.get("can_fire"))
    n_hits = sum(1 for f in fire_decisions if f.get("hit"))
    miss_vals = [f["miss"] for f in fire_decisions
                 if f.get("can_fire") and "miss" in f]

    # -- Evaluate ------------------------------------------------------------
    m_acoustic = _evaluate(acoustic_dets, ground_truth_fn, mic_center, "acoustic")
    m_seismic = _evaluate(seismic_dets, ground_truth_fn, geo_center, "seismic")
    m_fused = _evaluate(fused_dets, ground_truth_fn, array_center, "fused")

    metrics = {
        "acoustic": m_acoustic,
        "seismic": m_seismic,
        "fused": m_fused,
        "c_air": c_air,
        "c_ground": c_ground,
        "fire": {
            "shots": n_shots,
            "hits": n_hits,
            "hit_threshold": cfg_fc["hit_threshold_m"],
            "mean_miss": float(np.mean(miss_vals)) if miss_vals else None,
            "min_miss": float(np.min(miss_vals)) if miss_vals else None,
            "max_miss": float(np.max(miss_vals)) if miss_vals else None,
        },
    }

    # -- Print results -------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("RESULTS — COUPLED PIPELINE")
    print(f"{'=' * 65}")
    print(f"\n  Ground Vp = {c_ground:.0f} m/s   c_air = {c_air:.0f} m/s  "
          f"(ratio {c_ground/c_air:.1f}x)")
    if s_first is not None and a_first is not None:
        print(f"  Seismic early warning: {(a_first - s_first)*1e3:.1f} ms")
    print(f"\n  ACOUSTIC  {m_acoustic['n_detections']}/{m_acoustic['n_windows']} det"
          f"   brg err: {m_acoustic['mean_bearing_error']:.1f} deg"
          f"   rng err: {m_acoustic['mean_range_error']:.1f} m")
    print(f"  SEISMIC   {m_seismic['n_detections']}/{m_seismic['n_windows']} det"
          f"   brg err: {m_seismic['mean_bearing_error']:.1f} deg"
          f"   rng err: {m_seismic['mean_range_error']:.1f} m")
    print(f"  FUSED     {m_fused['n_detections']}/{m_fused['n_windows']} det"
          f"   brg err: {m_fused['mean_bearing_error']:.1f} deg"
          f"   rng err: {m_fused['mean_range_error']:.1f} m")
    print(f"\n  Shots: {n_shots}   Hits: {n_hits}")
    if miss_vals:
        print(f"  Mean miss: {np.mean(miss_vals):.1f} m"
              f"   Min: {np.min(miss_vals):.1f} m")
    print(f"\n  Processing time: {wall_both:.3f} s")

    # -- Plots ---------------------------------------------------------------
    _plot_coupled_summary(
        acoustic_dets, seismic_dets, fused_dets, fire_decisions,
        ground_truth_fn, src_duration, array_center, weapon_pos,
        metrics, output_dir / "pipeline_coupled_summary.png",
    )

    # -- Save ----------------------------------------------------------------
    results = {
        "simulation": str(sim_dir),
        "config": cfg,
        "c_air": c_air,
        "c_ground": c_ground,
        "seismic_lead_ms": ((a_first - s_first) * 1e3
                            if s_first and a_first else None),
        "acoustic": {
            "n_detections": m_acoustic["n_detections"],
            "n_windows": m_acoustic["n_windows"],
            "first_detection_s": m_acoustic["first_detection_s"],
            "mean_bearing_error_deg": m_acoustic["mean_bearing_error"],
        },
        "seismic": {
            "n_detections": m_seismic["n_detections"],
            "n_windows": m_seismic["n_windows"],
            "first_detection_s": m_seismic["first_detection_s"],
            "mean_bearing_error_deg": m_seismic["mean_bearing_error"],
        },
        "fused": {
            "n_detections": m_fused["n_detections"],
            "n_windows": m_fused["n_windows"],
            "mean_bearing_error_deg": m_fused["mean_bearing_error"],
        },
        "fire_control": {
            "shots": n_shots,
            "hits": n_hits,
            "mean_miss_m": metrics["fire"]["mean_miss"],
        },
        "processing_time_s": wall_both,
    }
    res_path = output_dir / "results_coupled.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {res_path}")
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
        default=Path("output/coupled_moving_3d"),
        help="Coupled simulation output directory "
             "(default: output/coupled_moving_3d)",
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent / "pipeline_coupled.config.json",
        help="JSON config (default: examples/pipeline_coupled.config.json)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-speed", type=float, default=None)
    parser.add_argument("--hit-threshold", type=float, default=None)
    parser.add_argument("--max-hits", type=int, default=None)

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


if __name__ == "__main__":
    main()

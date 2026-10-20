"""Pipeline engagement and diagnostic plots (``examples/run_pipeline.py``).

Split out of :mod:`acoustic_sim.air.plotting`, which re-exports these functions.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

matplotlib.use("Agg")


def _annotate(provenance) -> None:
    """Print the provenance flags (oracle inputs, simulator knowledge) on the current figure."""
    if provenance is not None:
        from acoustic_sim.contract import annotate_figure

        annotate_figure(plt.gcf(), provenance, y=-0.01, fontsize=10)


def plot_radial_engagement(
    fire_decisions: list[dict],
    ground_truth_fn,
    source_duration: float,
    weapon_pos: np.ndarray,
    is_3d: bool,
    cfg_fc: dict,
    output_path: Path | str | None = None,
    provenance=None,
) -> plt.Figure:
    """Radial engagement plot (plan-view + elevation for 3-D).

    Projectile trajectories are drawn to the CPA point for hits and
    extended to maximum pellet range for misses.
    """
    from acoustic_sim.air.fire_control import (
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
    _annotate(provenance)
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
    provenance=None,
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
    fired = [f for f in all_fire_decisions if f.get("miss") is not None]
    shot_times = [f["time"] for f in fired]
    shot_misses = [f["miss"] for f in fired]
    shot_eff_thresh = [
        max(f.get("pattern_radius", 0.0), hit_threshold) for f in fired
    ]
    if shot_times:
        colors_shot = ["green" if f.get("hit") else "red" for f in fired]
        ax.scatter(shot_times, shot_misses, c=colors_shot, s=60,
                   marker="x", zorder=5)
        ax.scatter(shot_times, shot_eff_thresh, c="gray", s=30,
                   marker="_", linewidths=2, alpha=0.6, zorder=4,
                   label="Eff. threshold")
        ax.axhline(hit_threshold, color="g", ls="--", alpha=0.7,
                   label=f"Point threshold {hit_threshold}m")
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
    _annotate(provenance)
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
    provenance=None,
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
    _annotate(provenance)
    if output_path:
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

"""Pipeline engagement and diagnostic plots (``examples/air/02_pipeline/run_pipeline.py``).

Split out of :mod:`acoustic_sim.air.plotting`, which re-exports these functions.  Every figure is one panel, written by
:func:`acoustic_sim.plotting.save_figure`; a display of several quantities is one figure per quantity, named
``<figure>_<panel>.png`` beside the path the caller gave.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from acoustic_sim.plotting import new_figure, recording_from_path, save_figure

matplotlib.use("Agg")

PANEL_SIZE = (12.0, 8.0)
MAP_SIZE = (12.0, 11.0)


def panel_path(output_path: Path | str, panel: str) -> Path:
    """``<stem>_<panel>.png`` beside *output_path*: one file per panel of a display."""
    p = Path(output_path)
    return p.with_name(f"{p.stem}_{panel}{p.suffix or '.png'}")


def finish(fig, ax, output_path: Path | str | None, title: str, provenance=None, *, allow_two_panels: bool = False,
           verbose: bool = True) -> None:
    """Title the panel (naming the scenario read from the output path, and the provenance flags when given), then
    write it through :func:`save_figure` (or close it when there is no path)."""
    scenario = recording_from_path(output_path) if output_path else None
    if scenario and scenario not in title:
        title = f"{scenario}: {title}"
    if provenance is not None:
        title = f"{title}\n{provenance.label()}"
    ax.set_title(title)
    if output_path is None:
        plt.close(fig)
        return
    save_figure(fig, output_path, allow_two_panels=allow_two_panels)
    if verbose:
        print(f"Saved: {output_path}")


def _shot_tof(fd, max_pellet_range, muzzle_velocity, decel, tof):
    from acoustic_sim.air.fire_control import time_of_flight

    draw_range = fd.get("cpa_range", None) if fd.get("hit", False) else max_pellet_range
    if draw_range is not None and draw_range > 0:
        draw_tof = time_of_flight(draw_range, muzzle_velocity, decel)
        if draw_tof == float("inf"):
            draw_tof = tof
    else:
        draw_tof = tof
    return draw_tof


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
    """Radial engagement: the plan view (``<stem>_plan.png``) and, for 3-D, the elevation view
    (``<stem>_elevation.png``), each its own figure.

    Projectile trajectories are drawn to the CPA point for hits and
    extended to maximum pellet range for misses.  Returns the plan view figure.
    """
    from acoustic_sim.air.fire_control import projectile_path

    muzzle_velocity = cfg_fc["muzzle_velocity_mps"]
    decel = cfg_fc["pellet_decel_mps2"]
    hit_threshold = cfg_fc["hit_threshold_m"]
    wx, wy, wz = float(weapon_pos[0]), float(weapon_pos[1]), float(weapon_pos[2])

    gt_times = np.linspace(0, source_duration, 200)
    gt_xyz = np.array([ground_truth_fn(t) for t in gt_times])
    gt_x = gt_xyz[:, 0] - wx
    gt_y = gt_xyz[:, 1] - wy

    fig, ax_xy = new_figure(MAP_SIZE)
    ax_xy.plot(gt_x, gt_y, "g-", lw=3, label="Target path", zorder=5)
    ax_xy.scatter(gt_x[0], gt_y[0], c="g", s=150, marker="o", zorder=6, label="Start")
    ax_xy.scatter(gt_x[-1], gt_y[-1], c="g", s=150, marker="s", zorder=6, label="End")

    max_range = max(float(np.max(np.sqrt(gt_x**2 + gt_y**2))), 50)
    for r in [25, 50, 75, 100]:
        if r <= max_range * 1.2:
            ax_xy.add_patch(Circle((0, 0), r, fill=False, color="gray", ls="--", alpha=0.3))
            ax_xy.text(r * 0.707, r * 0.707, f"{r} m", fontsize=11, color="gray", alpha=0.7)

    n_shots = n_hits = 0
    max_pellet_range = muzzle_velocity / decel
    shots = []
    for fd in fire_decisions:
        if not fd.get("can_fire"):
            continue
        aim_brg = fd.get("aim_bearing", float("nan"))
        aim_el = fd.get("aim_elevation", 0.0)
        tof = fd.get("tof", float("nan"))
        if np.isnan(aim_brg) or np.isnan(tof) or tof <= 0:
            continue
        n_shots += 1
        is_hit = fd.get("hit", False)
        if is_hit:
            n_hits += 1
        draw_tof = _shot_tof(fd, max_pellet_range, muzzle_velocity, decel, tof)
        proj_x, proj_y, proj_z = projectile_path(weapon_pos, aim_brg, aim_el, muzzle_velocity, decel, draw_tof)
        shots.append((fd, is_hit, proj_x, proj_z))
        color = "green" if is_hit else "red"
        ax_xy.plot(proj_x - wx, proj_y - wy, "-", color=color, lw=2, alpha=0.8)
        ipos = fd.get("intercept_pos")
        if ipos is not None:
            ix, iy = ipos[0] - wx, ipos[1] - wy
            # no per shot text (the labels of close shots overlap): the miss distances are in _miss_distance.png
            ax_xy.scatter(ix, iy, c=color, s=150, marker="x", linewidths=3, zorder=10)
        gt_pos = ground_truth_fn(fd["time"])
        ax_xy.scatter(gt_pos[0] - wx, gt_pos[1] - wy, c="lime", s=80, marker="o", edgecolors="darkgreen", linewidths=2,
                      zorder=8)
    ax_xy.plot([], [], "-", color="green", lw=2, label="Projectile (hit)")
    ax_xy.plot([], [], "-", color="red", lw=2, label="Projectile (miss)")
    ax_xy.scatter([], [], c="green", s=150, marker="x", linewidths=3, label="Intercept point (hit)")
    ax_xy.scatter([], [], c="red", s=150, marker="x", linewidths=3, label="Intercept point (miss)")
    ax_xy.scatter([], [], c="lime", s=80, marker="o", edgecolors="darkgreen", linewidths=2, label="Target at the shot")
    ax_xy.scatter(0, 0, c="black", s=300, marker="*", label="Weapon", zorder=15)
    ax_xy.set_xlabel("x relative to weapon [m]")
    ax_xy.set_ylabel("y relative to weapon [m]")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="upper left")

    dim = "3-D" if is_3d else "2-D"
    head = (f"radial engagement, plan view ({dim}): shots {n_shots}, hits {n_hits}, misses {n_shots - n_hits} "
            f"(threshold {hit_threshold} m)")
    if output_path:
        finish(fig, ax_xy, panel_path(output_path, "plan"), head, provenance)
    else:
        ax_xy.set_title(head)

    if is_3d:
        fig_z, ax_xz = new_figure(PANEL_SIZE)
        gt_z = gt_xyz[:, 2] - wz
        ax_xz.plot(gt_x, gt_z, "g-", lw=3, label="Target path", zorder=5)
        ax_xz.scatter(gt_x[0], gt_z[0], c="g", s=150, marker="o", zorder=6, label="Start")
        ax_xz.scatter(gt_x[-1], gt_z[-1], c="g", s=150, marker="s", zorder=6, label="End")
        for fd, is_hit, proj_x, proj_z in shots:
            color = "green" if is_hit else "red"
            ax_xz.plot(proj_x - wx, proj_z - wz, "-", color=color, lw=2, alpha=0.8)
            ipos = fd.get("intercept_pos")
            if ipos is not None:
                ax_xz.scatter(ipos[0] - wx, ipos[2] - wz, c=color, s=150, marker="x", linewidths=3, zorder=10)
        ax_xz.plot([], [], "-", color="green", lw=2, label="Projectile (hit)")
        ax_xz.plot([], [], "-", color="red", lw=2, label="Projectile (miss)")
        ax_xz.scatter(0, 0, c="black", s=300, marker="*", zorder=15, label="Weapon")
        ax_xz.set_xlabel("x relative to weapon [m]")
        ax_xz.set_ylabel("z (altitude) relative to weapon [m]")
        ax_xz.grid(True, alpha=0.3)
        ax_xz.legend(loc="upper left")
        finish(fig_z, ax_xz, panel_path(output_path, "elevation") if output_path else None,
               f"radial engagement, elevation view ({dim})", provenance)
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
    """The real time pipeline summary, one figure per quantity: ``<stem>_map.png`` (plan view), ``_bearing.png``,
    ``_miss_distance.png``, ``_track_error.png``, ``_latency.png``; the summary numbers go to ``<stem>_summary.txt``."""
    cx, cy = float(array_center[0]), float(array_center[1])
    gt_times = np.linspace(0, source_duration, 200)
    gt_xyz = np.array([ground_truth_fn(t) for t in gt_times])
    n_shots = metrics["shots_fired"]
    n_hits = metrics["n_hits"]
    mean_miss = metrics["mean_miss"]
    mean_brg = metrics["mean_bearing_error"]
    dim = "3-D" if is_3d else "2-D"

    # Plan view
    fig, ax = new_figure(MAP_SIZE)
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
            ax.scatter(ip[0], ip[1], c="green" if fd.get("hit") else "red", s=100, marker="x", zorder=10)
    ax.scatter([], [], c="green", s=100, marker="x", label="Intercept (hit)")
    ax.scatter([], [], c="red", s=100, marker="x", label="Intercept (miss)")
    ax.scatter(weapon_pos[0], weapon_pos[1], c="k", s=200, marker="*", label="Weapon and array", zorder=15)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    finish(fig, ax, panel_path(output_path, "map"), f"pipeline summary ({dim}): plan view", provenance)

    # Bearing against time
    fig, ax = new_figure(PANEL_SIZE)
    true_brgs = [math.degrees(math.atan2(ground_truth_fn(t)[1] - cy, ground_truth_fn(t)[0] - cx)) for t in gt_times]
    ax.plot(gt_times, true_brgs, "g-", lw=2, label="True")
    det_t = [d["time"] for d in all_detections if d.get("detected") and "bearing_deg" in d]
    det_brg = [d["bearing_deg"] - 360 if d["bearing_deg"] > 180 else d["bearing_deg"]
               for d in all_detections if d.get("detected") and "bearing_deg" in d]
    if det_t:
        ax.scatter(det_t, det_brg, c="b", s=20, alpha=0.4, label="Detected")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("bearing [deg]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    finish(fig, ax, panel_path(output_path, "bearing"), f"pipeline summary ({dim}): bearing against time "
           f"(mean error {mean_brg:.1f} deg)", provenance)

    # Miss distance against time
    fig, ax = new_figure(PANEL_SIZE)
    fired = [f for f in all_fire_decisions if f.get("miss") is not None]
    shot_times = [f["time"] for f in fired]
    shot_misses = [f["miss"] for f in fired]
    shot_eff_thresh = [max(f.get("pattern_radius", 0.0), hit_threshold) for f in fired]
    if shot_times:
        ax.scatter(shot_times, shot_misses, c=["green" if f.get("hit") else "red" for f in fired], s=60, marker="x",
                   zorder=5)
        ax.scatter([], [], c="green", s=60, marker="x", label="Shot (hit)")
        ax.scatter([], [], c="red", s=60, marker="x", label="Shot (miss)")
        ax.scatter(shot_times, shot_eff_thresh, c="gray", s=30, marker="_", linewidths=2, alpha=0.6, zorder=4,
                   label="Effective threshold")
    ax.axhline(hit_threshold, color="g", ls="--", alpha=0.7, label=f"Point threshold {hit_threshold} m")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("miss distance [m]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    finish(fig, ax, panel_path(output_path, "miss_distance"), f"pipeline summary ({dim}): fire control miss distance, "
           f"hits {n_hits} of {n_shots}, mean miss {mean_miss:.1f} m", provenance)

    # Track error against time
    fig, ax = new_figure(PANEL_SIZE)
    track_errors, te_times = [], []
    for fd in all_fire_decisions:
        if fd.get("est_pos") is not None:
            ep = np.array(fd["est_pos"])
            gt = np.asarray(ground_truth_fn(fd["time"]))
            track_errors.append(float(np.linalg.norm(ep - gt)))
            te_times.append(fd["time"])
    if track_errors:
        ax.plot(te_times, track_errors, "m-", lw=1.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("track position error [m]")
    ax.grid(True, alpha=0.3)
    finish(fig, ax, panel_path(output_path, "track_error"), f"pipeline summary ({dim}): track position error", provenance)

    # Latency
    fig, ax = new_figure(PANEL_SIZE)
    win_times = np.arange(len(wall_times)) * hop_sec
    ax.plot(win_times, wall_times * 1e6, "b-", lw=0.5, alpha=0.6, label="Processing time per window")
    ax.axhline(hop_sec * 1e6, color="r", ls="--", label=f"Real time budget: {hop_sec * 1e3:.1f} ms")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("processing time [us]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    finish(fig, ax, panel_path(output_path, "latency"), f"pipeline summary ({dim}): latency per window "
           f"(mean {wall_times.mean() * 1e6:.0f} us)", provenance)

    summary = (
        f"PIPELINE SUMMARY ({dim})\n"
        f"Detections:    {metrics['n_detections']}/{metrics['n_windows']}\n"
        f"Bearing err:   {mean_brg:.1f} deg\n"
        f"Track states:  {sum(1 for s in all_track_states if s)}\n"
        f"Shots:         {n_shots}\n"
        f"Hits <{hit_threshold}m:     {n_hits} ({100 * n_hits / max(n_shots, 1):.1f}%)\n"
        f"Mean miss:     {mean_miss:.1f} m\n"
        f"Hop cadence:   {hop_sec * 1e3:.1f} ms\n"
        f"Mean latency:  {wall_times.mean() * 1e6:.0f} us\n"
        f"Max latency:   {wall_times.max() * 1e6:.0f} us\n"
        f"RT margin:     {hop_sec / wall_times.mean():.0f}x\n"
    )
    panel_path(output_path, "summary").with_suffix(".txt").write_text(summary)


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
    """RMS weighted against SRP-PHAT bearing, one figure per quantity: ``<stem>_bearing.png``,
    ``_bearing_error.png``, ``_power_map.png``, ``_power_at_cpa.png``."""
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

    def out(panel):
        return panel_path(output_path, panel) if output_path else None

    head = f"beamformer diagnostic ({len(times)} windows)"
    fig, ax = new_figure(PANEL_SIZE)
    ax.plot(times, true_bearings, "k-", lw=2, label="True bearing")
    ax.plot(times, rms_bearings, "r.", ms=4, alpha=0.6, label=f"RMS ({mean_rms_err:.1f} deg)")
    ax.plot(times, srp_bearings, "b.", ms=4, alpha=0.6, label=f"SRP-PHAT ({mean_srp_err:.1f} deg)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("bearing [deg]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    finish(fig, ax, out("bearing"), f"{head}: bearing estimates against time", provenance)

    fig, ax = new_figure(PANEL_SIZE)
    ax.plot(times, rms_errors, "r-", lw=1.5, alpha=0.7, label=f"RMS ({mean_rms_err:.1f} deg)")
    ax.plot(times, srp_errors, "b-", lw=1.5, alpha=0.7, label=f"SRP-PHAT ({mean_srp_err:.1f} deg)")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("bearing error [deg]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    finish(fig, ax, out("bearing_error"), f"{head}: bearing error against time", provenance)

    if srp_power_maps:
        power_map = np.array(srp_power_maps)
        scan_deg = np.degrees(scan_angles)
        row_max = power_map.max(axis=1, keepdims=True)
        row_max[row_max < 1e-12] = 1.0
        power_norm = power_map / row_max
        extent = [scan_deg[0], scan_deg[-1], times[-1], times[0]]
        fig, ax = new_figure(PANEL_SIZE)
        im = ax.imshow(power_norm, aspect="auto", extent=extent, cmap="hot", origin="upper")
        ax.plot(true_bearings, times, "c-", lw=2, label="True bearing")
        ax.set_xlabel("bearing [deg]")
        ax.set_ylabel("time [s]")
        ax.legend()
        fig.colorbar(im, ax=ax, label="normalised power")
        finish(fig, ax, out("power_map"), f"{head}: SRP-PHAT power map", provenance)

        cpa_idx = int(np.argmax(rms_values))
        cpa_power = srp_power_maps[cpa_idx]
        fig, ax = new_figure(PANEL_SIZE)
        ax.plot(scan_deg, cpa_power / cpa_power.max(), "b-", lw=2, label="SRP-PHAT power")
        ax.axvline(true_bearings[cpa_idx], color="k", ls="--", lw=2, label="True")
        ax.axvline(srp_bearings[cpa_idx], color="b", ls=":", lw=2, label="SRP-PHAT estimate")
        ax.set_xlabel("bearing [deg]")
        ax.set_ylabel("normalised power")
        ax.legend()
        ax.grid(True, alpha=0.3)
        finish(fig, ax, out("power_at_cpa"), f"{head}: SRP-PHAT at the closest approach (t = {times[cpa_idx]:.3f} s)",
               provenance)

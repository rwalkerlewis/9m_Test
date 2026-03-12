#!/usr/bin/env python3
"""Analyze raw detection vs ground truth to understand error sources."""
import sys, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from run_full_pipeline import (
    load_simulation, compute_ground_truth, run_mfp_detection,
    apply_rms_range_estimation, run_tracking, run_targeting,
)

sim_dir = Path("/workspace/output/valley_test")
data = load_simulation(sim_dir)
traces, mic_positions, dt, metadata = data["traces"], data["mic_positions"], data["dt"], data["metadata"]
cx, cy = np.mean(mic_positions, axis=0)
gt_fn, src_dur = compute_ground_truth(metadata, 50.0)

mfp = run_mfp_detection(traces, mic_positions, dt)
dets = mfp["detections"]
apply_rms_range_estimation(dets)

print("RAW DETECTIONS vs GROUND TRUTH")
print(f"Array center: ({cx:.2f}, {cy:.2f})")
print(f"{'Time':>6} | {'DetBrg':>8} | {'TruBrg':>8} | {'BrgErr':>6} | {'DetRng':>7} | {'TruRng':>7} | {'RngErr':>6}")
print("-" * 72)
for d in dets:
    if not d["detected"]:
        continue
    t = d["time"]
    gx, gy = gt_fn(t)
    true_brg = math.degrees(math.atan2(gy - cy, gx - cx))
    true_rng = math.hypot(gx - cx, gy - cy)
    det_brg = d["bearing_deg"]
    if det_brg > 180:
        det_brg -= 360
    brg_err = det_brg - true_brg
    if brg_err > 180: brg_err -= 360
    if brg_err < -180: brg_err += 360
    rng_err = d["range"] - true_rng
    print(f"{t:6.3f} | {det_brg:8.1f} | {true_brg:8.1f} | {brg_err:+6.1f} | {d['range']:7.1f} | {true_rng:7.1f} | {rng_err:+6.1f}")

# Now show tracker
print("\n\nTRACKER vs GROUND TRUTH")
track = run_tracking(dets, (cx, cy))
fc = run_targeting(track, gt_fn)

positions = track["positions"]
times = track["times"]
fc_can_fire = fc["can_fire"]
fc_intercepts = fc["intercept_positions"]

print(f"{'Time':>6} | {'TrkX':>7} | {'TrkY':>7} | {'GtX':>7} | {'GtY':>7} | {'PosErr':>6} | {'TrkBrg':>7} | {'TruBrg':>7} | {'BrgErr':>6}")
print("-" * 90)
for i, t in enumerate(times):
    if np.isnan(positions[i, 0]):
        continue
    gx, gy = gt_fn(t)
    tx, ty = positions[i]
    pos_err = math.hypot(tx - gx, ty - gy)
    trk_brg = math.degrees(math.atan2(ty - cy, tx - cx))
    true_brg = math.degrees(math.atan2(gy - cy, gx - cx))
    brg_err = trk_brg - true_brg
    if brg_err > 180: brg_err -= 360
    if brg_err < -180: brg_err += 360
    # Compute aim error: where the projectile would land vs where target actually is
    aim_err = float("nan")
    if fc_can_fire[i]:
        ix, iy = fc_intercepts[i]
        aim_err = math.hypot(ix - gx, iy - gy)
    fire_mark = f" <-- FIRE aim_err={aim_err:.1f}m" if fc_can_fire[i] else ""
    print(f"{t:6.3f} | {tx:7.2f} | {ty:7.2f} | {gx:7.2f} | {gy:7.2f} | {pos_err:6.2f} | {trk_brg:7.1f} | {true_brg:7.1f} | {brg_err:+6.1f}{fire_mark}")

# Summary of fire control
print("\n\nFIRE CONTROL SHOT DETAILS")
shot_idx = [i for i, f in enumerate(fc_can_fire) if f]
for idx in shot_idx:
    t = times[idx]
    tx, ty = positions[idx]
    gx, gy = gt_fn(t)
    ix, iy = fc_intercepts[idx]
    pos_err = math.hypot(tx - gx, ty - gy)
    aim_err = math.hypot(ix - gx, iy - gy)
    trk_brg = math.degrees(math.atan2(ty - cy, tx - cx))
    true_brg = math.degrees(math.atan2(gy - cy, gx - cx))
    trk_rng = math.hypot(tx - 0, ty - 0)  # from weapon at (0,0)
    true_rng = math.hypot(gx - 0, gy - 0)
    print(f"  Shot at t={t:.3f}: tracker=({tx:.1f},{ty:.1f}) gt=({gx:.1f},{gy:.1f}) "
          f"pos_err={pos_err:.1f}m aim_err={aim_err:.1f}m "
          f"trk_brg={trk_brg:.1f}° true_brg={true_brg:.1f}°")

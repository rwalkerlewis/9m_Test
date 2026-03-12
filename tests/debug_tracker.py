#!/usr/bin/env python3
"""Debug tracker vs ground truth positions."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from run_full_pipeline import (
    load_simulation,
    compute_ground_truth,
    run_mfp_detection,
    apply_rms_range_estimation,
    run_tracking,
    run_targeting,
)


def debug_tracker(sim_dir: Path, source_speed: float = 50.0):
    """Compare tracker positions to ground truth."""
    
    data = load_simulation(sim_dir)
    traces = data["traces"]
    mic_positions = data["mic_positions"]
    dt = data["dt"]
    metadata = data["metadata"]
    
    array_center = tuple(np.mean(mic_positions, axis=0))
    cx, cy = array_center
    print(f"Array center: ({cx:.2f}, {cy:.2f})")
    
    ground_truth_fn, src_duration = compute_ground_truth(metadata, source_speed)
    print(f"Source duration: {src_duration:.2f}s")
    print(f"Source start: {ground_truth_fn(0)}")
    print(f"Source end:   {ground_truth_fn(src_duration)}")
    
    # Run pipeline
    mfp_result = run_mfp_detection(traces, mic_positions, dt)
    detections = mfp_result["detections"]
    apply_rms_range_estimation(detections)
    
    track = run_tracking(detections, array_center)
    fire_control = run_targeting(track, ground_truth_fn)
    
    # Get times where fire control said FIRE
    fc_times = fire_control.get("times", np.array([]))
    fc_can_fire = fire_control.get("can_fire", np.array([]))
    fc_intercepts = fire_control.get("intercept_positions", np.array([]).reshape(-1, 2))
    
    track_times = track.get("times", np.array([]))
    track_positions = track.get("positions", np.array([]).reshape(-1, 2))
    
    print("\n" + "=" * 80)
    print("SHOT ANALYSIS")
    print("=" * 80)
    
    for i, (t, can_fire) in enumerate(zip(fc_times, fc_can_fire)):
        if not can_fire:
            continue
            
        # Ground truth at shot time
        gt_x, gt_y = ground_truth_fn(t)
        
        # Tracker position at shot time
        trk_x, trk_y = track_positions[i] if i < len(track_positions) else (float('nan'), float('nan'))
        
        # Intercept point
        int_x, int_y = fc_intercepts[i] if i < len(fc_intercepts) else (float('nan'), float('nan'))
        
        # Compute errors
        track_err = math.hypot(trk_x - gt_x, trk_y - gt_y)
        aim_err = math.hypot(int_x - gt_x, int_y - gt_y)
        
        # True range from array center
        true_range = math.hypot(gt_x - cx, gt_y - cy)
        
        # Bearing
        true_bearing = math.degrees(math.atan2(gt_y - cy, gt_x - cx))
        track_bearing = math.degrees(math.atan2(trk_y - cy, trk_x - cx))
        
        print(f"\nShot at t={t:.3f}s:")
        print(f"  Ground Truth: ({gt_x:7.2f}, {gt_y:7.2f}) range={true_range:.1f}m bearing={true_bearing:.1f}°")
        print(f"  Tracker:      ({trk_x:7.2f}, {trk_y:7.2f}) bearing={track_bearing:.1f}°")
        print(f"  Aim Point:    ({int_x:7.2f}, {int_y:7.2f})")
        print(f"  Track Error:  {track_err:.2f}m")
        print(f"  Aim Error:    {aim_err:.2f}m (miss distance)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path, nargs="?", default=Path("/workspace/output/valley_test"))
    parser.add_argument("--source-speed", type=float, default=50.0)
    args = parser.parse_args()
    debug_tracker(args.sim_dir, args.source_speed)


if __name__ == "__main__":
    main()

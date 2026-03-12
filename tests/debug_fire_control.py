#!/usr/bin/env python3
"""Debug fire control decisions to understand why shots aren't being fired."""

import sys
from pathlib import Path

import numpy as np

# Add src and examples to path
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


def debug_fire_control(sim_dir: Path, source_speed: float = 50.0):
    """Analyze fire control decisions."""
    
    data = load_simulation(sim_dir)
    traces = data["traces"]
    mic_positions = data["mic_positions"]
    dt = data["dt"]
    metadata = data["metadata"]
    
    array_center = tuple(np.mean(mic_positions, axis=0))
    
    # Run pipeline
    mfp_result = run_mfp_detection(traces, mic_positions, dt)
    detections = mfp_result["detections"]
    apply_rms_range_estimation(detections)
    
    ground_truth_fn, src_duration = compute_ground_truth(metadata, source_speed)
    
    track = run_tracking(detections, array_center)
    fire_control = run_targeting(track, ground_truth_fn)
    
    # Compute position uncertainty from tracker covariance
    track_covs = track.get("covariances", [])
    pos_unc = []
    for cov in track_covs:
        if np.any(np.isnan(cov)):
            pos_unc.append(float('nan'))
        else:
            # 2-sigma of position block
            pos_cov = cov[:2, :2]
            eigvals = np.linalg.eigvalsh(pos_cov)
            sigma_max = np.sqrt(max(eigvals.max(), 0.0))
            pos_unc.append(2.0 * sigma_max)
    
    # Analyze decisions
    times = fire_control.get("times", np.array([]))
    can_fire = fire_control.get("can_fire", np.array([]))
    reasons = fire_control.get("reasons", [])
    ranges = fire_control.get("ranges", np.array([]))
    
    pattern_spread_rate = 0.2  # from run_full_pipeline.py
    
    print(f"{'Time':>6} | {'Range':>8} | {'Pattern':>8} | {'PosUnc':>8} | Reason")
    print("-" * 70)
    
    reason_counts = {}
    for i, (t, cf, rng, reason) in enumerate(zip(times, can_fire, ranges, reasons)):
        rng_str = f"{rng:.1f}m" if not np.isnan(rng) else "NaN"
        pattern_diam = rng * pattern_spread_rate if not np.isnan(rng) else float('nan')
        pu = pos_unc[i] if i < len(pos_unc) else float('nan')
        pat_str = f"{pattern_diam:.2f}m" if not np.isnan(pattern_diam) else "NaN"
        pu_str = f"{pu:.2f}m" if not np.isnan(pu) else "NaN"
        print(f"{t:6.3f} | {rng_str:>8} | {pat_str:>8} | {pu_str:>8} | {reason}")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    print()
    print("Summary:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path, nargs="?", default=Path("../output/valley_test"))
    parser.add_argument("--source-speed", type=float, default=50.0)
    args = parser.parse_args()
    debug_fire_control(args.sim_dir, args.source_speed)


if __name__ == "__main__":
    main()

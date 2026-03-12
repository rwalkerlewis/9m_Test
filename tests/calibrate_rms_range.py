#!/usr/bin/env python3
"""Calibrate RMS-to-range relationship from simulation data.

Analyzes detected RMS values against known ground truth positions to find
the optimal reference values for range estimation using the formula:
    range = ref_range * sqrt(ref_rms / measured_rms)

Usage:
    python tests/calibrate_rms_range.py output/valley_test --source-speed 50
"""

import argparse
import math
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
)


def calibrate_rms_range(
    sim_dir: Path,
    source_speed: float = 50.0,
    ref_range: float = 10.0,
) -> dict:
    """Calibrate RMS-to-range parameters from simulation.
    
    Args:
        sim_dir: Path to simulation output
        source_speed: Source velocity in m/s
        ref_range: Reference range for calibration
        
    Returns:
        dict with calibration results
    """
    # Load data
    data = load_simulation(sim_dir)
    traces = data["traces"]
    mic_positions = data["mic_positions"]
    dt = data["dt"]
    metadata = data["metadata"]
    
    cx, cy = np.mean(mic_positions, axis=0)
    ground_truth_fn, src_duration = compute_ground_truth(metadata, source_speed)
    
    # Run MFP
    mfp_result = run_mfp_detection(traces, mic_positions, dt)
    detections = mfp_result["detections"]
    
    # Analyze each detection
    print(f"{'Time':>6} | {'RMS':>10} | {'True Range':>12} | {'Implied Ref':>12}")
    print("-" * 55)
    
    ref_values = []
    data_points = []
    
    for d in detections:
        if not d["detected"]:
            continue
        
        t = d["time"]
        rms = d["window_rms"]
        
        gt_x, gt_y = ground_truth_fn(t)
        true_range = math.hypot(gt_x - cx, gt_y - cy)
        
        # What ref_rms would give correct range?
        # r = ref_r * sqrt(ref_rms / rms)
        # => ref_rms = rms * (r / ref_r)^2
        implied_ref = rms * (true_range / ref_range) ** 2
        ref_values.append(implied_ref)
        data_points.append((t, rms, true_range, implied_ref))
        
        print(f"{t:6.3f} | {rms:10.4f} | {true_range:12.1f}m | {implied_ref:12.4f}")
    
    # Statistics
    median_ref = np.median(ref_values)
    mean_ref = np.mean(ref_values)
    std_ref = np.std(ref_values)
    
    print()
    print("=" * 55)
    print(f"Reference range: {ref_range:.1f}m")
    print(f"Median ref_rms:  {median_ref:.4f}")
    print(f"Mean ref_rms:    {mean_ref:.4f}")
    print(f"Std ref_rms:     {std_ref:.4f}")
    print()
    print("Recommended calibration:")
    print(f"  rms_ref_range = {ref_range:.1f}")
    print(f"  rms_ref_value = {median_ref:.2f}")
    
    return {
        "ref_range": ref_range,
        "median_ref_rms": median_ref,
        "mean_ref_rms": mean_ref,
        "std_ref_rms": std_ref,
        "n_points": len(ref_values),
        "data_points": data_points,
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
        help="Simulation output directory",
    )
    parser.add_argument(
        "--source-speed",
        type=float,
        default=50.0,
        help="Source velocity in m/s (default: 50.0)",
    )
    parser.add_argument(
        "--ref-range",
        type=float,
        default=10.0,
        help="Reference range for calibration (default: 10.0)",
    )
    
    args = parser.parse_args()
    calibrate_rms_range(args.sim_dir, args.source_speed, args.ref_range)


if __name__ == "__main__":
    main()

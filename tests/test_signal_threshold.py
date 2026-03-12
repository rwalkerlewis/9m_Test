#!/usr/bin/env python3
"""
Test script to validate the min_signal_rms fix for the detection pipeline.

This test verifies that:
1. Detections are suppressed when signal RMS is below threshold
2. Tracker doesn't initialize from noise-only windows
3. Fire control points in the correct direction

Usage:
    python tests/test_signal_threshold.py [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.processor import matched_field_process
from acoustic_sim.tracker import run_tracker


def test_signal_threshold(output_dir: Path, verbose: bool = True) -> dict:
    """Test that min_signal_rms correctly filters noise-only windows."""
    
    # Load data
    traces = np.load(output_dir / "traces.npy")
    with open(output_dir / "metadata.json") as f:
        meta = json.load(f)
    
    dt = meta["dt"]
    mic_positions = np.array(meta["receiver_positions"])
    cx, cy = np.mean(mic_positions, axis=0)
    
    results = {"passed": True, "tests": []}
    
    # --- Test 1: Detection without threshold (baseline) ---
    mfp_no_filter = matched_field_process(
        traces, mic_positions, dt,
        sound_speed=343.0,
        fundamental=180.0,
        window_length=0.1,
        detection_threshold=0.2,
        min_signal_rms=0.0,  # No filtering
        range_min=20.0,
        range_max=100.0,
    )
    
    det_no_filter = mfp_no_filter["detections"]
    n_det_no_filter = sum(1 for d in det_no_filter if d["detected"])
    
    # Check first window (should be false positive)
    first_det_no_filter = det_no_filter[0]
    first_rms = first_det_no_filter.get("window_rms", 0)
    first_detected_no_filter = first_det_no_filter["detected"]
    
    test1_pass = first_detected_no_filter  # Without filter, first window SHOULD detect (false positive)
    results["tests"].append({
        "name": "Baseline: First window detects without filter",
        "passed": test1_pass,
        "details": f"First window (t={first_det_no_filter['time']:.2f}s): detected={first_detected_no_filter}, RMS={first_rms:.4f}"
    })
    
    if verbose:
        print(f"Test 1: Baseline detection (no filter)")
        print(f"  Total detections: {n_det_no_filter}/{len(det_no_filter)}")
        print(f"  First window: detected={first_detected_no_filter}, RMS={first_rms:.4f}")
        print(f"  → {'PASS' if test1_pass else 'FAIL'}")
        print()
    
    # --- Test 2: Detection with threshold ---
    mfp_filtered = matched_field_process(
        traces, mic_positions, dt,
        sound_speed=343.0,
        fundamental=180.0,
        window_length=0.1,
        detection_threshold=0.2,
        min_signal_rms=0.01,  # With filtering
        range_min=20.0,
        range_max=100.0,
    )
    
    det_filtered = mfp_filtered["detections"]
    n_det_filtered = sum(1 for d in det_filtered if d["detected"])
    
    # Check first window (should NOT detect)
    first_det_filtered = det_filtered[0]
    first_detected_filtered = first_det_filtered["detected"]
    
    test2_pass = not first_detected_filtered  # With filter, first window should NOT detect
    results["tests"].append({
        "name": "Filtered: First window rejected below RMS threshold",
        "passed": test2_pass,
        "details": f"First window: detected={first_detected_filtered}"
    })
    
    if verbose:
        print(f"Test 2: Filtered detection (min_signal_rms=0.01)")
        print(f"  Total detections: {n_det_filtered}/{len(det_filtered)}")
        print(f"  First window: detected={first_detected_filtered}")
        print(f"  → {'PASS' if test2_pass else 'FAIL'}")
        print()
    
    # --- Test 3: Tracker initialization ---
    track_no_filter = run_tracker(
        det_no_filter,
        sigma_bearing_deg=3.0,
        sigma_range=50.0,
        initial_range_guess=60.0,
        source_level_dB=90.0,
        array_center_x=cx,
        array_center_y=cy,
    )
    
    track_filtered = run_tracker(
        det_filtered,
        sigma_bearing_deg=3.0,
        sigma_range=50.0,
        initial_range_guess=60.0,
        source_level_dB=90.0,
        array_center_x=cx,
        array_center_y=cy,
    )
    
    # Find first valid track position
    pos_no_filter = np.array(track_no_filter["positions"])
    pos_filtered = np.array(track_filtered["positions"])
    
    first_valid_no_filter = None
    for i, p in enumerate(pos_no_filter):
        if not np.isnan(p[0]):
            first_valid_no_filter = (track_no_filter["times"][i], p)
            break
    
    first_valid_filtered = None
    for i, p in enumerate(pos_filtered):
        if not np.isnan(p[0]):
            first_valid_filtered = (track_filtered["times"][i], p)
            break
    
    # True position at t=0.2s (first detection time with filter)
    # Valley scenario: start at (-40,0), arc over array at (0,10)
    # At t=0.2s, drone is at approx (-30, 5.7) based on arc
    true_bearing_at_first = math.degrees(math.atan2(-5.7 - cy, -30 - cx))  # ~192°
    
    test3_pass = first_valid_filtered is not None and first_valid_filtered[0] >= 0.15
    results["tests"].append({
        "name": "Tracker: First valid position delayed until signal present",
        "passed": test3_pass,
        "details": f"First valid track at t={first_valid_filtered[0]:.2f}s" if first_valid_filtered else "No valid track"
    })
    
    if verbose:
        print(f"Test 3: Tracker initialization timing")
        if first_valid_no_filter:
            print(f"  Without filter: t={first_valid_no_filter[0]:.2f}s, pos=({first_valid_no_filter[1][0]:.1f}, {first_valid_no_filter[1][1]:.1f})")
        if first_valid_filtered:
            print(f"  With filter:    t={first_valid_filtered[0]:.2f}s, pos=({first_valid_filtered[1][0]:.1f}, {first_valid_filtered[1][1]:.1f})")
        print(f"  → {'PASS' if test3_pass else 'FAIL'}")
        print()
    
    # --- Test 4: Bearing direction check ---
    if first_valid_filtered:
        init_pos = first_valid_filtered[1]
        init_bearing = math.degrees(math.atan2(init_pos[1] - cy, init_pos[0] - cx))
        
        # True bearing should be ~192° (source is south-west of array)
        # Allow ±30° tolerance
        bearing_error = abs((init_bearing - true_bearing_at_first + 180) % 360 - 180)
        test4_pass = bearing_error < 30
        
        results["tests"].append({
            "name": "Tracker: Initial bearing points toward source",
            "passed": test4_pass,
            "details": f"Init bearing={init_bearing:.1f}°, expected ~{true_bearing_at_first:.1f}°, error={bearing_error:.1f}°"
        })
        
        if verbose:
            print(f"Test 4: Tracker initial bearing direction")
            print(f"  Initial position: ({init_pos[0]:.1f}, {init_pos[1]:.1f})")
            print(f"  Bearing from array: {init_bearing:.1f}°")
            print(f"  Expected (true): ~{true_bearing_at_first:.1f}°")
            print(f"  Error: {bearing_error:.1f}°")
            print(f"  → {'PASS' if test4_pass else 'FAIL'}")
            print()
    else:
        results["tests"].append({
            "name": "Tracker: Initial bearing check",
            "passed": False,
            "details": "No valid track position"
        })
    
    # Overall result
    results["passed"] = all(t["passed"] for t in results["tests"])
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test signal threshold fix")
    parser.add_argument("--output-dir", "-o", type=str, default="output/valley_test",
                       help="FDTD output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("  SIGNAL THRESHOLD FIX VALIDATION")
    print("=" * 60)
    print(f"\nTesting with data from: {output_dir}\n")
    
    results = test_signal_threshold(output_dir, verbose=True)
    
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    for test in results["tests"]:
        status = "✓ PASS" if test["passed"] else "✗ FAIL"
        print(f"  {status}: {test['name']}")
    
    print()
    overall = "ALL TESTS PASSED" if results["passed"] else "SOME TESTS FAILED"
    print(f"  {overall}")
    print("=" * 60)
    
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

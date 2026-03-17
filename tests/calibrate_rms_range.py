#!/usr/bin/env python3
"""Calibrate RMS-to-range relationship from simulation data.

Runs the unified pipeline and reports range estimation metrics.

Usage:
    python tests/calibrate_rms_range.py output/valley_test --source-speed 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from run_pipeline import run_pipeline, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sim_dir", type=Path, nargs="?", default=Path("output/valley_test"))
    parser.add_argument("--source-speed", type=float, default=50.0)
    args = parser.parse_args()

    cfg = load_config(Path(__file__).parent.parent / "examples" / "pipeline.config.json")
    cfg["source"]["speed_mps"] = args.source_speed
    cfg["fire_control"]["max_hits"] = 0  # Run all windows

    results = run_pipeline(args.sim_dir, args.sim_dir, cfg)
    print(f"\nBearing: {results['mean_bearing_error_deg']:.1f} deg")
    print(f"Shots: {results['shots_fired']}, Hits: {results['hits']}")
    sys.exit(0)


if __name__ == "__main__":
    main()

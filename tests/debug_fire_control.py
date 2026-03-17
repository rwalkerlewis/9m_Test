#!/usr/bin/env python3
"""Debug fire control decisions to understand why shots aren't being fired.

Usage:
    python tests/debug_fire_control.py output/valley_test --source-speed 50
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
    cfg["fire_control"]["max_hits"] = 0  # Don't stop early

    results = run_pipeline(args.sim_dir, args.sim_dir, cfg)
    print(f"\nShots: {results['shots_fired']}, Hits: {results['hits']}")
    sys.exit(0)


if __name__ == "__main__":
    main()

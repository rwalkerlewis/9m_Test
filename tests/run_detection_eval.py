#!/usr/bin/env python3
"""
Detection Pipeline Evaluation Script (simplified)

This wraps run_full_pipeline.py for quick evaluation runs with additional
diagnostic output. For the main pipeline, use run_full_pipeline.py directly.

Usage:
    python tests/run_detection_eval.py [--output-dir OUTPUT_DIR]

Example:
    python tests/run_detection_eval.py --output-dir output/valley_test
"""

import argparse
import sys
from pathlib import Path

# Add src and examples to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from run_full_pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("output/valley_test"),
        help="Simulation output directory (default: output/valley_test)",
    )
    parser.add_argument(
        "--source-speed",
        type=float,
        default=50.0,
        help="Source velocity in m/s (default: 50.0)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )
    
    args = parser.parse_args()
    
    # Just use the main pipeline
    run_pipeline(
        args.output_dir,
        args.output_dir,
        source_speed=args.source_speed,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Automated comparison of baseline vs ML-augmented pipelines.

Runs 4 pipeline configurations across 4 simulation scenarios and
produces a unified comparison table showing where ML adds value.

Configurations tested:
1. Baseline (no ML) — signal processing only
2. Source classification only (AcousticClassifier)
3. Source classification + maneuver detection
4. Fusion classification + maneuver detection

Saves results to ``output/comparison_results.json`` and prints a
formatted comparison table.

Usage::

    python examples/run_comparison.py
    python examples/run_comparison.py --max-hits 0  # unlimited shots
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the ML pipeline's run_pipeline directly.
from run_pipeline_ml import load_config, run_pipeline

# ============================================================================
# Scenarios and configurations
# ============================================================================

SCENARIOS = [
    ("valley_test",        Path("output/valley_test")),
    ("valley_3d_test",     Path("output/valley_3d_test")),
    ("isotropic_2D",       Path("output/isotropic_2D")),
    ("erratic_quadcopter", Path("output/erratic_quadcopter")),
]

CONFIGS = [
    {
        "name": "baseline",
        "ml": {
            "enable_source_classification": False,
            "enable_maneuver_detection": False,
            "enable_fusion_classification": False,
        },
    },
    {
        "name": "acoustic_class",
        "ml": {
            "enable_source_classification": True,
            "enable_maneuver_detection": False,
            "enable_fusion_classification": False,
        },
    },
    {
        "name": "class+maneuver",
        "ml": {
            "enable_source_classification": True,
            "enable_maneuver_detection": True,
            "enable_fusion_classification": False,
        },
    },
    {
        "name": "fusion+maneuver",
        "ml": {
            "enable_source_classification": False,
            "enable_maneuver_detection": True,
            "enable_fusion_classification": True,
        },
    },
]


def extract_metrics(result: dict) -> dict:
    """Extract key metrics from a pipeline result dict."""
    return {
        "n_detections": result.get("n_detections", 0),
        "n_windows": result.get("n_windows", 0),
        "detection_rate": result.get("detection_rate", 0),
        "mean_bearing_error_deg": result.get("mean_bearing_error_deg"),
        "shots_fired": result.get("shots_fired", 0),
        "hits": result.get("hits", 0),
        "hit_rate_pct": result.get("hit_rate_pct", 0),
        "mean_miss_m": result.get("mean_miss_m"),
        "min_miss_m": result.get("min_miss_m"),
        "max_miss_m": result.get("max_miss_m"),
        "mean_latency_us": result.get("timing", {}).get("mean_latency_us", 0),
        "class_reject_count": result.get("ml", {}).get("class_reject_count", 0),
        "maneuver_suppress_count": result.get("ml", {}).get(
            "maneuver_suppress_count", 0),
        "n_classified_windows": result.get("ml", {}).get("n_classified_windows", 0),
        "mean_classification_confidence": result.get("ml", {}).get(
            "mean_classification_confidence"),
    }


def print_comparison_table(all_results: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("  COMPARISON RESULTS")
    print("=" * 100)

    for scenario_name, _ in SCENARIOS:
        print(f"\n  Scenario: {scenario_name}")
        print(f"  {'Config':<20s} {'Det':>4s} {'Win':>4s} {'Shots':>5s} "
              f"{'Hits':>4s} {'Hit%':>6s} {'Miss':>6s} "
              f"{'ClsRej':>6s} {'ManSup':>6s} {'Latency':>8s} {'BrgErr':>7s}")
        print(f"  {'-'*90}")

        for config in CONFIGS:
            key = f"{scenario_name}__{config['name']}"
            m = all_results.get(key)
            if m is None:
                print(f"  {config['name']:<20s}  (no data)")
                continue

            miss = m.get("mean_miss_m")
            miss_s = f"{miss:.1f}" if miss is not None and not math.isnan(miss) else "N/A"
            brg = m.get("mean_bearing_error_deg")
            brg_s = f"{brg:.1f}" if brg is not None and not math.isnan(brg) else "N/A"

            print(f"  {config['name']:<20s} "
                  f"{m['n_detections']:>4d} "
                  f"{m['n_windows']:>4d} "
                  f"{m['shots_fired']:>5d} "
                  f"{m['hits']:>4d} "
                  f"{m['hit_rate_pct']:>5.1f}% "
                  f"{miss_s:>6s} "
                  f"{m['class_reject_count']:>6d} "
                  f"{m.get('maneuver_suppress_count', 0):>6d} "
                  f"{m['mean_latency_us']:>7.0f}us "
                  f"{brg_s:>7s}")

    # Summary: aggregate across all scenarios.
    print(f"\n{'=' * 100}")
    print("  AGGREGATE SUMMARY (across all scenarios)")
    print(f"{'=' * 100}")

    for config in CONFIGS:
        total_shots = 0
        total_hits = 0
        total_rejects = 0
        total_suppress = 0
        latencies = []
        miss_values = []
        total_classified = 0

        for scenario_name, _ in SCENARIOS:
            key = f"{scenario_name}__{config['name']}"
            m = all_results.get(key)
            if m is None:
                continue
            total_shots += m["shots_fired"]
            total_hits += m["hits"]
            total_rejects += m["class_reject_count"]
            total_suppress += m.get("maneuver_suppress_count", 0)
            latencies.append(m["mean_latency_us"])
            mm = m.get("mean_miss_m")
            if mm is not None and not math.isnan(mm):
                miss_values.append(mm)
            total_classified += m.get("n_classified_windows", 0)

        agg_hit_rate = 100 * total_hits / max(total_shots, 1)
        agg_miss = sum(miss_values) / max(len(miss_values), 1)
        agg_latency = sum(latencies) / max(len(latencies), 1)

        print(f"\n  {config['name']}:")
        print(f"    Total shots: {total_shots}, Hits: {total_hits}, "
              f"Hit rate: {agg_hit_rate:.1f}%")
        print(f"    Mean miss: {agg_miss:.2f} m")
        print(f"    Class rejects: {total_rejects}, "
              f"Maneuver suppressed: {total_suppress}")
        print(f"    Mean latency: {agg_latency:.0f} us")
        print(f"    Classified windows: {total_classified}")

    # Detection enrichment score.
    print(f"\n{'=' * 100}")
    print("  SUCCESS CRITERIA EVALUATION")
    print(f"{'=' * 100}")

    baseline_key_template = "{}__{}"
    for config in CONFIGS[1:]:  # skip baseline
        improvements = []
        for scenario_name, _ in SCENARIOS:
            bkey = baseline_key_template.format(scenario_name, "baseline")
            mkey = baseline_key_template.format(scenario_name, config["name"])
            b = all_results.get(bkey)
            m = all_results.get(mkey)
            if b is None or m is None:
                continue

            # False engagement: fewer shots = better
            if m["shots_fired"] < b["shots_fired"]:
                improvements.append(f"{scenario_name}: fewer shots "
                                    f"({m['shots_fired']} vs {b['shots_fired']})")

            # Hit rate: higher = better
            if m["hit_rate_pct"] > b["hit_rate_pct"] + 0.1:
                improvements.append(f"{scenario_name}: better hit rate "
                                    f"({m['hit_rate_pct']:.1f}% vs "
                                    f"{b['hit_rate_pct']:.1f}%)")

            # Miss distance: lower = better
            bm = b.get("mean_miss_m")
            mm = m.get("mean_miss_m")
            if (bm is not None and mm is not None
                    and not math.isnan(bm) and not math.isnan(mm)
                    and mm < bm - 0.1):
                improvements.append(f"{scenario_name}: lower miss "
                                    f"({mm:.1f}m vs {bm:.1f}m)")

            # Detection enrichment: classified windows
            if m.get("n_classified_windows", 0) > 0:
                improvements.append(f"{scenario_name}: +{m['n_classified_windows']} "
                                    f"classified windows")

        print(f"\n  {config['name']}:")
        if improvements:
            for imp in improvements:
                print(f"    ✓ {imp}")
        else:
            print(f"    (no improvements over baseline)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max-hits", type=int, default=None,
                        help="Override max_hits for all runs")
    parser.add_argument("--hit-threshold", type=float, default=None,
                        help="Override hit threshold for all runs")
    parser.add_argument("--output", type=Path,
                        default=Path("output/comparison_results.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    base_cfg = load_config(Path("examples/pipeline_ml.config.json"))

    # Apply overrides.
    if args.max_hits is not None:
        base_cfg["fire_control"]["max_hits"] = args.max_hits
    if args.hit_threshold is not None:
        base_cfg["fire_control"]["hit_threshold_m"] = args.hit_threshold

    print("=" * 100)
    print("  ML PIPELINE COMPARISON")
    print("=" * 100)
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"  Configurations: {len(CONFIGS)}")
    print(f"  Total runs: {len(SCENARIOS) * len(CONFIGS)}")
    print(f"  Output: {args.output}")

    all_results = {}
    run_count = 0
    total_runs = len(SCENARIOS) * len(CONFIGS)
    t_start = time.perf_counter()

    for scenario_name, sim_dir in SCENARIOS:
        for config in CONFIGS:
            run_count += 1
            key = f"{scenario_name}__{config['name']}"

            print(f"\n{'#' * 80}")
            print(f"  RUN {run_count}/{total_runs}: "
                  f"{scenario_name} × {config['name']}")
            print(f"{'#' * 80}")

            # Build config for this run.
            cfg = deepcopy(base_cfg)
            cfg["ml"].update(config["ml"])

            # Use a temporary output dir to avoid overwriting.
            out_dir = Path(f"output/comparison/{scenario_name}/{config['name']}")
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = run_pipeline(sim_dir, out_dir, cfg)
                metrics = extract_metrics(result)
                all_results[key] = metrics
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[key] = {"error": str(e)}

    elapsed = time.perf_counter() - t_start

    # Save results.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {args.output}")

    # Print comparison table.
    print_comparison_table(all_results)

    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

"""Systematic parameter-sweep studies for the detection pipeline.

Each ``study_*`` function runs the detection pipeline across a range
of configurations, collects performance metrics, and generates a
comparison plot.  ``run_all_studies`` executes every study.

Studies
=======
1. Array geometry comparison
2. Minimum sensor count
3. Sensor fault robustness (with / without mitigation)
4. Multi-drone detection and threat priority
5. Transient (explosion) robustness
6. Haphazard array placement
7. Echo-prone domain
8. Sensor position errors (with / without calibration)
9. Mixed failure modes (combined stress test)
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np

from acoustic_sim.config import DetectionConfig
from acoustic_sim.detection_main import (
    evaluate_results,
    run_detection,
    run_detection_pipeline,
    simulate_scenario,
)
from acoustic_sim.plotting import plot_study_comparison


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _base_cfg(override: DetectionConfig | None = None) -> DetectionConfig:
    """Return a base config for studies at the correct array scale.

    Uses the DetectionConfig defaults which are already set for a
    0.5 m radius array, dx = 0.05 m, 30 × 30 m domain.
    """
    if override is not None:
        cfg = copy.deepcopy(override)
    else:
        cfg = DetectionConfig()
    # Ensure studies don't run the expensive stationary-source FDTD
    # unless explicitly enabled by the study.
    cfg.stationary_source_enabled = False
    return cfg


def _collect(result: dict) -> dict:
    """Extract key metrics from a pipeline result."""
    return {
        "detection_rate": result.get("detection_rate", 0.0),
        "mean_loc_error": result.get("mean_loc_error", float("nan")),
        "first_shot_miss": result.get("first_shot_miss", float("nan")),
        "first_shot_hit": result.get("first_shot_hit", False),
        "first_shot_pattern": result.get("first_shot_pattern", float("nan")),
        "mean_miss": result.get("mean_miss", float("nan")),
    }


def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Case':<25s} {'Det %':>6s} {'Loc':>6s} {'1st Miss':>8s} {'Pattern':>8s} {'Hit?':>5s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*5}")
    for r in rows:
        det = f"{r['detection_rate']*100:.0f}%"
        loc = f"{r['mean_loc_error']:.1f}m" if np.isfinite(r["mean_loc_error"]) else "N/A"
        miss = f"{r['first_shot_miss']:.2f}m" if np.isfinite(r["first_shot_miss"]) else "N/A"
        pat = f"{r['first_shot_pattern']:.2f}m" if np.isfinite(r["first_shot_pattern"]) else "N/A"
        hit = "YES" if r["first_shot_hit"] else "NO"
        print(f"  {r['label']:<25s} {det:>6s} {loc:>6s} {miss:>8s} {pat:>8s} {hit:>5s}")
    print()


# -----------------------------------------------------------------------
# Study 1: Array geometry
# -----------------------------------------------------------------------

def study_array_geometry(
    base_config: DetectionConfig | None = None,
    output_dir: str = "output/studies/array_geometry",
) -> dict:
    """Compare circular, linear, l_shaped, random, concentric arrays."""
    geometries = ["circular", "linear", "l_shaped", "random", "concentric"]
    rows: list[dict] = []

    for geom in geometries:
        cfg = _base_cfg(base_config)
        cfg.array_type = geom
        cfg.output_dir = os.path.join(output_dir, geom)

        print(f"\n>>> Array geometry study: {geom}")
        result = run_detection_pipeline(cfg)
        m = _collect(result)
        m["label"] = geom
        rows.append(m)

    _print_table(rows, "ARRAY GEOMETRY STUDY")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_study_comparison(
        [r["label"] for r in rows],
        {"Detection Rate": [r["detection_rate"] for r in rows],
         "1st Shot Miss [m]": [r["first_shot_miss"] for r in rows],
         "Loc Error [m]": [r["mean_loc_error"] for r in rows]},
        output_path=os.path.join(output_dir, "comparison.png"),
        title="Array Geometry Comparison",
    )
    return {"rows": rows}


# -----------------------------------------------------------------------
# Study 2: Minimum sensors
# -----------------------------------------------------------------------

def study_min_sensors(
    base_config: DetectionConfig | None = None,
    sensor_counts: tuple[int, ...] = (4, 6, 8, 12, 16, 24),
    output_dir: str = "output/studies/min_sensors",
) -> dict:
    """Sweep n_mics to find minimum acceptable sensor count."""
    rows: list[dict] = []

    for n in sensor_counts:
        cfg = _base_cfg(base_config)
        cfg.n_mics = n
        cfg.output_dir = os.path.join(output_dir, f"n{n}")

        print(f"\n>>> Min-sensor study: n_mics={n}")
        result = run_detection_pipeline(cfg)
        m = _collect(result)
        m["label"] = f"{n} mics"
        rows.append(m)

    _print_table(rows, "MINIMUM SENSOR STUDY")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_study_comparison(
        [r["label"] for r in rows],
        {"Detection Rate": [r["detection_rate"] for r in rows],
         "1st Shot Miss [m]": [r["first_shot_miss"] for r in rows],
         "Loc Error [m]": [r["mean_loc_error"] for r in rows]},
        output_path=os.path.join(output_dir, "comparison.png"),
        title="Sensor Count vs Performance",
    )
    return {"rows": rows}


# -----------------------------------------------------------------------
# Study 3: Sensor faults
# -----------------------------------------------------------------------

def study_sensor_faults(
    base_config: DetectionConfig | None = None,
    fault_fractions: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5),
    output_dir: str = "output/studies/sensor_faults",
) -> dict:
    """Sweep fault fraction with and without robust sensor weighting.

    Runs FDTD **once**, then injects faults post-hoc and re-runs
    detection on the same base traces for each (fraction, mitigate)
    combination.
    """
    from acoustic_sim.noise import inject_sensor_faults

    # Single FDTD run for clean traces.
    cfg = _base_cfg(base_config)
    print("\n>>> Sensor fault study: generating base traces (single FDTD)…")
    scenario = simulate_scenario(cfg)
    clean_traces = scenario["traces"].copy()

    rows: list[dict] = []
    for frac in fault_fractions:
        for mitigate in [False, True]:
            label = f"f={frac:.0%}" + (" +mitig" if mitigate else " raw")
            print(f"\n>>> Sensor fault study: {label} (detection only)")

            # Inject faults on a copy of the clean traces.
            if frac > 0:
                traces, _ = inject_sensor_faults(
                    clean_traces, fault_type="elevated_noise",
                    fault_fraction=frac, seed=cfg.seed + 10,
                )
            else:
                traces = clean_traces.copy()

            det = run_detection(
                traces, scenario["mic_positions"], scenario["dt"],
                sound_speed=cfg.sound_speed,
                weapon_position=cfg.weapon_position,
                enable_sensor_weights=mitigate,
                grid_x_range=cfg.mfp_grid_x_range,
                grid_y_range=cfg.mfp_grid_y_range,
            )
            m = _collect(evaluate_results(
                det, scenario["true_positions"],
                scenario["true_velocities"], scenario["true_times"],
                weapon_position=cfg.weapon_position,
            ))
            m["label"] = label
            rows.append(m)

    _print_table(rows, "SENSOR FAULT STUDY")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_study_comparison(
        [r["label"] for r in rows],
        {"Detection Rate": [r["detection_rate"] for r in rows],
         "1st Shot Miss [m]": [r["first_shot_miss"] for r in rows],
         "Loc Error [m]": [r["mean_loc_error"] for r in rows]},
        output_path=os.path.join(output_dir, "comparison.png"),
        title="Sensor Faults: Raw vs Mitigated",
    )
    return {"rows": rows}


# -----------------------------------------------------------------------
# Study 4: Multi-drone
# -----------------------------------------------------------------------

def study_multi_drone(
    base_config: DetectionConfig | None = None,
    output_dir: str = "output/studies/multi_drone",
) -> dict:
    """Test with 1 and 2 simultaneous drones using multi-peak detection."""
    rows: list[dict] = []

    # Single drone (baseline).
    cfg = _base_cfg(base_config)
    cfg.output_dir = os.path.join(output_dir, "1_drone")
    print("\n>>> Multi-drone study: 1 drone")
    result = run_detection_pipeline(cfg)
    m = _collect(result)
    m["label"] = "1 drone"
    rows.append(m)

    # Two drones via superposition — run second FDTD and add traces.
    # We simulate a second drone by enabling the stationary source as a
    # second drone-like source (tonal, different position).
    cfg2 = _base_cfg(base_config)
    cfg2.max_sources = 2
    cfg2.stationary_source_enabled = True
    cfg2.stationary_source_pos = (-8.0, -6.0)
    cfg2.stationary_source_freq = 150.0  # same as drone to test separation
    cfg2.stationary_source_level_dB = 90.0
    cfg2.output_dir = os.path.join(output_dir, "2_drones")
    print("\n>>> Multi-drone study: 2 drones (drone + stationary-as-drone)")
    result2 = run_detection_pipeline(cfg2)
    m2 = _collect(result2)
    m2["label"] = "2 drones"
    n_tracks = len(result2.get("multi_tracks", []))
    m2["n_tracks"] = n_tracks
    rows.append(m2)

    _print_table(rows, "MULTI-DRONE STUDY")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_study_comparison(
        [r["label"] for r in rows],
        {"Detection Rate": [r["detection_rate"] for r in rows],
         "1st Shot Miss [m]": [r["first_shot_miss"] for r in rows],
         "Loc Error [m]": [r["mean_loc_error"] for r in rows]},
        output_path=os.path.join(output_dir, "comparison.png"),
        title="Multi-Drone Performance",
    )
    return {"rows": rows}


# -----------------------------------------------------------------------
# Study 5: Transient robustness
# -----------------------------------------------------------------------

def study_transient_robustness(
    base_config: DetectionConfig | None = None,
    transient_levels: tuple[float, ...] = (0, 110, 120, 130),
    output_dir: str = "output/studies/transient",
) -> dict:
    """Inject transients, compare with/without blanking.

    Runs FDTD **once**, then injects transients post-hoc.
    """
    from acoustic_sim.noise import inject_transient

    cfg = _base_cfg(base_config)
    print("\n>>> Transient study: generating base traces (single FDTD)…")
    scenario = simulate_scenario(cfg)
    clean_traces = scenario["traces"].copy()

    rows: list[dict] = []
    for level in transient_levels:
        for blank in [False, True]:
            if level == 0 and blank:
                continue
            label = f"{level}dB" + (" +blank" if blank else " raw")
            if level == 0:
                label = "clean"
            print(f"\n>>> Transient study: {label} (detection only)")

            if level > 0:
                traces = inject_transient(
                    clean_traces, scenario["dt"],
                    event_time=0.25,
                    event_pos=(5.0, 5.0),
                    mic_positions=scenario["mic_positions"],
                    level_dB=float(level),
                    sound_speed=cfg.sound_speed,
                    seed=cfg.seed + 20,
                )
            else:
                traces = clean_traces.copy()

            det = run_detection(
                traces, scenario["mic_positions"], scenario["dt"],
                sound_speed=cfg.sound_speed,
                weapon_position=cfg.weapon_position,
                enable_transient_blanking=blank,
                grid_x_range=cfg.mfp_grid_x_range,
                grid_y_range=cfg.mfp_grid_y_range,
            )
            m = _collect(evaluate_results(
                det, scenario["true_positions"],
                scenario["true_velocities"], scenario["true_times"],
                weapon_position=cfg.weapon_position,
            ))
            m["label"] = label
            rows.append(m)

    _print_table(rows, "TRANSIENT ROBUSTNESS STUDY")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_study_comparison(
        [r["label"] for r in rows],
        {"Detection Rate": [r["detection_rate"] for r in rows],
         "1st Shot Miss [m]": [r["first_shot_miss"] for r in rows],
         "Loc Error [m]": [r["mean_loc_error"] for r in rows]},
        output_path=os.path.join(output_dir, "comparison.png"),
        title="Transient Robustness: Raw vs Blanked",
    )
    return {"rows": rows}


# -----------------------------------------------------------------------
# Study 6: Haphazard array
# -----------------------------------------------------------------------

def study_haphazard_array(
    base_config: DetectionConfig | None = None,
    n_trials: int = 3,
    output_dir: str = "output/studies/haphazard",
) -> dict:
    """Compare optimised circular array vs random placements."""
    rows: list[dict] = []

    # Baseline: circular.
    cfg = _base_cfg(base_config)
    cfg.array_type = "circular"
    cfg.output_dir = os.path.join(output_dir, "circular")
    print("\n>>> Haphazard study: circular (optimised)")
    result = run_detection_pipeline(cfg)
    m = _collect(result)
    m["label"] = "circular"
    rows.append(m)

    # Random placements.
    for trial in range(n_trials):
        cfg = _base_cfg(base_config)
        cfg.array_type = "random"
        cfg.seed = 100 + trial
        cfg.output_dir = os.path.join(output_dir, f"random_{trial}")
        print(f"\n>>> Haphazard study: random trial {trial}")
        result = run_detection_pipeline(cfg)
        m = _collect(result)
        m["label"] = f"random_{trial}"
        rows.append(m)

    _print_table(rows, "HAPHAZARD ARRAY STUDY")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_study_comparison(
        [r["label"] for r in rows],
        {"Detection Rate": [r["detection_rate"] for r in rows],
         "1st Shot Miss [m]": [r["first_shot_miss"] for r in rows],
         "Loc Error [m]": [r["mean_loc_error"] for r in rows]},
        output_path=os.path.join(output_dir, "comparison.png"),
        title="Haphazard Array: Circular vs Random",
    )
    return {"rows": rows}


# -----------------------------------------------------------------------
# Study 7: Echo domains
# -----------------------------------------------------------------------

def study_echo_domains(
    base_config: DetectionConfig | None = None,
    output_dir: str = "output/studies/echo",
) -> dict:
    """Compare detection in isotropic vs echo-prone domains."""
    domains = ["isotropic", "echo_canyon", "urban_echo"]
    rows: list[dict] = []

    for dom in domains:
        cfg = _base_cfg(base_config)
        cfg.domain_type = dom
        cfg.output_dir = os.path.join(output_dir, dom)
        print(f"\n>>> Echo study: {dom}")
        result = run_detection_pipeline(cfg)
        m = _collect(result)
        m["label"] = dom
        rows.append(m)

    _print_table(rows, "ECHO DOMAIN STUDY")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_study_comparison(
        [r["label"] for r in rows],
        {"Detection Rate": [r["detection_rate"] for r in rows],
         "1st Shot Miss [m]": [r["first_shot_miss"] for r in rows],
         "Loc Error [m]": [r["mean_loc_error"] for r in rows]},
        output_path=os.path.join(output_dir, "comparison.png"),
        title="Echo Domain Comparison",
    )
    return {"rows": rows}


# -----------------------------------------------------------------------
# Study 8: Position errors
# -----------------------------------------------------------------------

def study_position_errors(
    base_config: DetectionConfig | None = None,
    error_stds: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0),
    output_dir: str = "output/studies/position_error",
) -> dict:
    """Sweep position error magnitude with/without self-calibration.

    Runs FDTD **once**, then perturbs positions post-hoc.
    """
    from acoustic_sim.noise import perturb_mic_positions

    cfg = _base_cfg(base_config)
    print("\n>>> Position error study: generating base traces (single FDTD)…")
    scenario = simulate_scenario(cfg)

    rows: list[dict] = []
    for err in error_stds:
        for calib in [False, True]:
            if err == 0 and calib:
                continue
            label = f"err={err:.0f}m" + (" +calib" if calib else " raw")
            if err == 0:
                label = "perfect"
            print(f"\n>>> Position error study: {label} (detection only)")

            mic_pos = scenario["mic_positions"]
            if err > 0:
                mic_pos = perturb_mic_positions(
                    scenario["mic_positions"], error_std=err,
                    seed=cfg.seed + 30,
                )

            det = run_detection(
                scenario["traces"], mic_pos, scenario["dt"],
                sound_speed=cfg.sound_speed,
                weapon_position=cfg.weapon_position,
                enable_position_calibration=calib,
                grid_x_range=cfg.mfp_grid_x_range,
                grid_y_range=cfg.mfp_grid_y_range,
            )
            m = _collect(evaluate_results(
                det, scenario["true_positions"],
                scenario["true_velocities"], scenario["true_times"],
                weapon_position=cfg.weapon_position,
            ))
            m["label"] = label
            rows.append(m)

    _print_table(rows, "POSITION ERROR STUDY")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_study_comparison(
        [r["label"] for r in rows],
        {"Detection Rate": [r["detection_rate"] for r in rows],
         "1st Shot Miss [m]": [r["first_shot_miss"] for r in rows],
         "Loc Error [m]": [r["mean_loc_error"] for r in rows]},
        output_path=os.path.join(output_dir, "comparison.png"),
        title="Position Error: Raw vs Calibrated",
    )
    return {"rows": rows}


# -----------------------------------------------------------------------
# Study 9: Mixed failure modes
# -----------------------------------------------------------------------

def study_mixed_failures(
    base_config: DetectionConfig | None = None,
    output_dir: str = "output/studies/mixed",
) -> dict:
    """Combined stress test: faults + position errors + transient + echo domain.

    Runs several progressively harder scenarios:
    1. Clean baseline
    2. Sensor faults only (20%)
    3. Faults + position errors (2 m)
    4. Faults + errors + transient (120 dB)
    5. All above + echo canyon domain
    6. All above + haphazard (random) array
    7. All mitigations enabled on scenario 6
    """
    scenarios: list[tuple[str, dict]] = [
        ("clean", {}),
        ("faults_20%", {"inject_faults": True, "fault_fraction": 0.2}),
        ("faults+pos_err", {"inject_faults": True, "fault_fraction": 0.2,
                            "inject_position_error": True, "position_error_std": 2.0}),
        ("faults+err+transient", {"inject_faults": True, "fault_fraction": 0.2,
                                   "inject_position_error": True, "position_error_std": 2.0,
                                   "inject_transient": True, "transient_level_dB": 120.0,
                                   "transient_time": 0.25}),
        ("all+echo_canyon", {"inject_faults": True, "fault_fraction": 0.2,
                              "inject_position_error": True, "position_error_std": 2.0,
                              "inject_transient": True, "transient_level_dB": 120.0,
                              "transient_time": 0.25,
                              "domain_type": "echo_canyon"}),
        ("all+haphazard", {"inject_faults": True, "fault_fraction": 0.2,
                            "inject_position_error": True, "position_error_std": 2.0,
                            "inject_transient": True, "transient_level_dB": 120.0,
                            "transient_time": 0.25,
                            "domain_type": "echo_canyon",
                            "array_type": "random"}),
        ("all+mitigations", {"inject_faults": True, "fault_fraction": 0.2,
                              "inject_position_error": True, "position_error_std": 2.0,
                              "inject_transient": True, "transient_level_dB": 120.0,
                              "transient_time": 0.25,
                              "domain_type": "echo_canyon",
                              "array_type": "random",
                              "enable_sensor_weights": True,
                              "enable_transient_blanking": True,
                              "enable_position_calibration": True}),
    ]

    rows: list[dict] = []
    for name, overrides in scenarios:
        cfg = _base_cfg(base_config)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.output_dir = os.path.join(output_dir, name)

        print(f"\n>>> Mixed study: {name}")
        result = run_detection_pipeline(cfg)
        m = _collect(result)
        m["label"] = name
        rows.append(m)

    _print_table(rows, "MIXED FAILURE MODES STUDY")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_study_comparison(
        [r["label"] for r in rows],
        {"Detection Rate": [r["detection_rate"] for r in rows],
         "1st Shot Miss [m]": [r["first_shot_miss"] for r in rows],
         "Loc Error [m]": [r["mean_loc_error"] for r in rows]},
        output_path=os.path.join(output_dir, "comparison.png"),
        title="Mixed Failure Modes: Progressive Stress Test",
    )
    return {"rows": rows}


# -----------------------------------------------------------------------
# Master runner
# -----------------------------------------------------------------------

def run_all_studies(
    base_config: DetectionConfig | None = None,
    output_dir: str = "output/studies",
) -> dict:
    """Execute all single-FDTD studies and return combined results.

    Only studies that reuse one FDTD run (injecting faults, transients,
    or position errors post-hoc) are included.  Studies that require
    separate FDTD runs per case (array geometry, domain type, sensor
    count) are available individually but not in the batch runner.
    """
    results = {}

    results["sensor_faults"] = study_sensor_faults(
        base_config, output_dir=os.path.join(output_dir, "sensor_faults"))
    results["transient"] = study_transient_robustness(
        base_config, output_dir=os.path.join(output_dir, "transient"))
    results["position_error"] = study_position_errors(
        base_config, output_dir=os.path.join(output_dir, "position_error"))
    results["mixed"] = study_mixed_failures(
        base_config, output_dir=os.path.join(output_dir, "mixed"))

    print("\n" + "=" * 60)
    print("  ALL STUDIES COMPLETE")
    print("=" * 60)

    return results

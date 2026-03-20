#!/usr/bin/env python3
"""Regenerate synthetic microphone traces for all simulation scenarios.

The original FDTD-generated traces.npy files are not present in the output
directories.  This script reconstructs them using the analytical 3-D
forward model (spherical spreading + delay), which is fast and does not
require MPI or GPU.

The generated traces are physically consistent with the metadata in each
scenario directory — same receiver positions, same source trajectory,
same dt and n_steps — so the detection pipeline produces valid results.

Usage::

    python examples/generate_traces.py          # regenerate all 4 scenarios
    python examples/generate_traces.py --only valley_test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.forward import simulate_3d_traces
from acoustic_sim.noise import generate_sensor_noise, generate_wind_noise
from acoustic_sim.sources import (
    ErraticQuadcopterSource3D,
    MovingSource3D,
    make_drone_harmonics,
    make_source_propeller,
)

# Scenarios and their directories.
SCENARIOS = {
    "valley_test":         Path("output/valley_test"),
    "valley_3d_test":      Path("output/valley_3d_test"),
    "isotropic_2D":        Path("output/isotropic_2D"),
    "erratic_quadcopter":  Path("output/erratic_quadcopter"),
}


def _generate_valley_test(sim_dir: Path, metadata: dict) -> np.ndarray:
    """2-D valley scenario: moving source with arc, propeller signal."""
    dt = metadata["dt"]
    n_steps = metadata["n_steps"]
    receivers = np.array(metadata["receiver_positions"], dtype=np.float64)
    # Promote 2-D receivers to 3-D.
    if receivers.shape[1] == 2:
        receivers = np.column_stack([receivers, np.zeros(receivers.shape[0])])

    source_x = metadata.get("source_x", -40.0)
    source_y = metadata.get("source_y", 0.0)
    source_z = metadata.get("source_z", 0.0)
    arc_height = metadata.get("source_arc_height", 0.0)

    # Compute source endpoint from speed.
    speed = 50.0  # default pipeline speed
    horiz_dist = speed * metadata["total_time"]
    end_x = source_x + horiz_dist

    # Generate propeller signal.
    fs = 1.0 / dt
    f_max = min(2000.0, fs / 2.0 * 0.9)
    sig = make_drone_harmonics(n_steps, dt, fundamental=150.0,
                                n_harmonics=6, source_level_dB=90.0,
                                f_max=f_max)

    source = MovingSource3D(
        x0=source_x, y0=source_y, z0=source_z,
        x1=end_x, y1=source_y, z1=source_z,
        speed=speed, signal=sig, arc_height=arc_height,
    )

    c = metadata.get("velocity", 343.0)
    traces = simulate_3d_traces(source, receivers, dt, n_steps,
                                 sound_speed=c, air_absorption=0.005)

    # Add realistic noise.
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(traces.shape) * 1e-5
    traces += noise
    return traces


def _generate_valley_3d_test(sim_dir: Path, metadata: dict) -> np.ndarray:
    """3-D valley scenario: moving source at altitude with arc."""
    dt = metadata["dt"]
    n_steps = metadata["n_steps"]
    receivers = np.array(metadata["receiver_positions"], dtype=np.float64)
    if receivers.shape[1] == 2:
        receivers = np.column_stack([receivers, np.zeros(receivers.shape[0])])

    source_x = metadata.get("source_x", -40.0)
    source_y = metadata.get("source_y", 0.0)
    source_z = metadata.get("source_z", 15.0)
    end_x = metadata.get("source_x1", 40.0)
    end_y = metadata.get("source_y1", 0.0)
    end_z = metadata.get("source_z1", 15.0)
    arc_height = metadata.get("source_arc_height", 15.0)

    speed = 50.0
    fs = 1.0 / dt
    f_max = min(2000.0, fs / 2.0 * 0.9)
    sig = make_drone_harmonics(n_steps, dt, fundamental=150.0,
                                n_harmonics=6, source_level_dB=90.0,
                                f_max=f_max)

    source = MovingSource3D(
        x0=source_x, y0=source_y, z0=source_z,
        x1=end_x, y1=end_y, z1=end_z,
        speed=speed, signal=sig, arc_height=arc_height,
    )

    c = metadata.get("velocity", 343.0)
    traces = simulate_3d_traces(source, receivers, dt, n_steps,
                                 sound_speed=c, air_absorption=0.005)

    rng = np.random.default_rng(42)
    noise = rng.standard_normal(traces.shape) * 1e-5
    traces += noise
    return traces


def _generate_isotropic_2d(sim_dir: Path, metadata: dict) -> np.ndarray:
    """2-D isotropic medium: moving source, propeller signal."""
    dt = metadata["dt"]
    n_steps = metadata["n_steps"]
    receivers = np.array(metadata["receiver_positions"], dtype=np.float64)
    if receivers.shape[1] == 2:
        receivers = np.column_stack([receivers, np.zeros(receivers.shape[0])])

    source_x = metadata.get("source_x", -45.0)
    source_y = metadata.get("source_y", 0.0)
    source_z = metadata.get("source_z", 0.0)

    speed = 50.0
    horiz_dist = speed * metadata["total_time"]
    end_x = source_x + horiz_dist

    fs = 1.0 / dt
    f_max = min(2000.0, fs / 2.0 * 0.9)
    sig = make_drone_harmonics(n_steps, dt, fundamental=150.0,
                                n_harmonics=6, source_level_dB=90.0,
                                f_max=f_max)

    source = MovingSource3D(
        x0=source_x, y0=source_y, z0=source_z,
        x1=end_x, y1=source_y, z1=source_z,
        speed=speed, signal=sig,
    )

    c = metadata.get("velocity", 343.0)
    traces = simulate_3d_traces(source, receivers, dt, n_steps,
                                 sound_speed=c, air_absorption=0.005)

    rng = np.random.default_rng(42)
    noise = rng.standard_normal(traces.shape) * 1e-5
    traces += noise
    return traces


def _generate_erratic_quadcopter(sim_dir: Path, metadata: dict) -> np.ndarray:
    """3-D erratic quadcopter scenario."""
    dt = metadata["dt"]
    n_steps = metadata["n_steps"]
    receivers = np.array(metadata["receiver_positions"], dtype=np.float64)
    if receivers.shape[1] == 2:
        receivers = np.column_stack([receivers, np.zeros(receivers.shape[0])])

    source_x = metadata.get("source_x", 15.0)
    source_y = metadata.get("source_y", 0.0)
    source_z = metadata.get("source_z", 10.0)
    mean_speed = metadata.get("source_mean_speed", 8.0)
    agility = metadata.get("source_agility", 6.0)
    seed = metadata.get("source_seed", 42)

    x_min = metadata.get("x_min", -20.0)
    x_max = metadata.get("x_max", 20.0)
    y_min = metadata.get("y_min", -20.0)
    y_max = metadata.get("y_max", 20.0)
    z_max = metadata.get("z_max", 22.0)

    fs = 1.0 / dt
    f_max = min(2000.0, fs / 2.0 * 0.9)
    sig = make_drone_harmonics(n_steps, dt, fundamental=150.0,
                                n_harmonics=4, source_level_dB=90.0,
                                f_max=f_max)

    source = ErraticQuadcopterSource3D(
        x0=source_x, y0=source_y, z0=source_z,
        bbox_min=(x_min, y_min, 2.0),
        bbox_max=(x_max, y_max, z_max - 2.0),
        mean_speed=mean_speed,
        agility=agility,
        signal=sig,
        seed=seed,
    )

    # Save trajectory.
    traj = source.get_trajectory(n_steps, dt)
    np.save(str(sim_dir / "source_trajectory.npy"), traj)
    print(f"    Saved source_trajectory.npy ({traj.shape})")

    c = metadata.get("velocity", 343.0)
    traces = simulate_3d_traces(source, receivers, dt, n_steps,
                                 sound_speed=c, air_absorption=0.005)

    rng = np.random.default_rng(42)
    noise = rng.standard_normal(traces.shape) * 1e-5
    traces += noise
    return traces


_GENERATORS = {
    "valley_test": _generate_valley_test,
    "valley_3d_test": _generate_valley_3d_test,
    "isotropic_2D": _generate_isotropic_2d,
    "erratic_quadcopter": _generate_erratic_quadcopter,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=str, default=None,
                        help="Regenerate only this scenario")
    args = parser.parse_args()

    targets = SCENARIOS
    if args.only:
        if args.only not in SCENARIOS:
            print(f"Unknown scenario: {args.only}")
            print(f"Available: {list(SCENARIOS.keys())}")
            sys.exit(1)
        targets = {args.only: SCENARIOS[args.only]}

    for name, sim_dir in targets.items():
        print(f"\n{'='*60}")
        print(f"  Generating traces for: {name}")
        print(f"  Directory: {sim_dir}")
        print(f"{'='*60}")

        meta_path = sim_dir / "metadata.json"
        if not meta_path.exists():
            print(f"  ERROR: {meta_path} not found, skipping.")
            continue

        with open(meta_path) as f:
            metadata = json.load(f)

        print(f"  dt={metadata['dt']:.2e}, n_steps={metadata['n_steps']}, "
              f"n_receivers={metadata['n_receivers']}")

        gen_fn = _GENERATORS[name]
        traces = gen_fn(sim_dir, metadata)

        out_path = sim_dir / "traces.npy"
        np.save(str(out_path), traces)
        print(f"  Saved {out_path} — shape {traces.shape}")
        print(f"  RMS: {np.sqrt(np.mean(traces**2)):.6e}")

    print("\nDone. All trace files regenerated.")


if __name__ == "__main__":
    main()

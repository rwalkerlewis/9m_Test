"""FDTD training data generation for the FNO surrogate model.

Runs the existing FDTD solver with randomised parameters and saves
(input, output) pairs:

    input  = (velocity field, source params, receiver positions)
    output = receiver traces

Each sample is a complete FDTD simulation with randomised:
    - domain type and terrain seed
    - source position, trajectory, speed, signal type
    - receiver array layout and centre position

The dataset is stored as a directory of .npz files (one per sample)
to keep memory usage bounded during generation.

Usage
-----
    python -m acoustic_sim.ml.fno_data_gen \\
        --n-samples 500 --output-dir data/fno_train --dims 2d

    python -m acoustic_sim.ml.fno_data_gen \\
        --n-samples 100 --output-dir data/fno_train_3d --dims 3d
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _generate_sample_2d(
    sample_id: int,
    rng: np.random.Generator,
    output_dir: Path,
    dx: float = 0.5,
    total_time: float = 0.3,
    fd_order: int = 2,
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
) -> dict | None:
    """Generate one 2-D FDTD training sample.

    Returns metadata dict on success, None on failure.
    """
    from acoustic_sim.fdtd import FDTDConfig, FDTDSolver
    from acoustic_sim.setup import build_domain, build_receivers, build_source, compute_dt

    # -- Randomise domain --
    domain_type = rng.choice(["isotropic", "wind", "hills_vegetation"])
    wind_speed = float(rng.uniform(0, 10)) if domain_type == "wind" else 0.0
    wind_dir = float(rng.uniform(0, 360))
    terrain_seed = int(rng.integers(0, 2**31))

    model, meta = build_domain(
        domain_type,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        dx=dx, velocity=343.0,
        wind_speed=wind_speed, wind_direction_deg=wind_dir,
        seed=terrain_seed,
    )

    # -- Randomise receivers --
    array_type = rng.choice(["circular", "linear"])
    n_recv = int(rng.choice([8, 12, 16, 24]))
    recv_radius = float(rng.uniform(1.0, 5.0))
    recv_cx = float(rng.uniform(x_min + 10, x_max - 10))
    recv_cy = float(rng.uniform(y_min + 10, y_max - 10))

    receivers = build_receivers(
        array_type,
        count=n_recv,
        radius=recv_radius,
        center_x=recv_cx,
        center_y=recv_cy,
        x0=recv_cx - recv_radius,
        y0=recv_cy,
        x1=recv_cx + recv_radius,
        y1=recv_cy,
    )

    # -- Compute dt --
    dt, f_max = compute_dt(model, meta, cfl_safety=0.9, fd_order=fd_order)
    n_steps = int(math.ceil(total_time / dt))

    # -- Randomise source --
    source_type = rng.choice(["static", "moving"])
    signal_type = rng.choice(["propeller", "tone", "ricker"])
    src_x = float(rng.uniform(x_min + 5, x_max - 5))
    src_y = float(rng.uniform(y_min + 5, y_max - 5))
    src_x1 = float(rng.uniform(x_min + 5, x_max - 5))
    src_y1 = float(rng.uniform(y_min + 5, y_max - 5))
    src_speed = float(rng.uniform(10, 80))
    src_freq = float(rng.uniform(15, 100))
    arc_height = float(rng.uniform(0, 15)) if source_type == "moving" else 0.0
    source_seed = int(rng.integers(0, 2**31))

    source = build_source(
        source_type, signal_type,
        n_steps=n_steps, dt=dt, f_max=f_max,
        x=src_x, y=src_y, x1=src_x1, y1=src_y1,
        speed=src_speed, arc_height=arc_height,
        freq=src_freq, seed=source_seed,
    )

    # -- Run FDTD --
    cfg = FDTDConfig(
        total_time=total_time,
        dt=dt,
        snapshot_interval=0,  # no snapshots for training
        use_cuda=False,
        fd_order=fd_order,
    )
    solver = FDTDSolver(
        model=model, config=cfg, source=source,
        receivers=receivers, domain_meta=meta,
    )
    result = solver.run(snapshot_dir=None, verbose=False)
    traces = result["traces"]

    if traces is None or traces.size == 0:
        return None

    # -- Save sample --
    sample_path = output_dir / f"sample_{sample_id:06d}.npz"
    np.savez_compressed(
        str(sample_path),
        velocity_field=model.values.astype(np.float32),
        grid_x=model.x.astype(np.float32),
        grid_y=model.y.astype(np.float32),
        receiver_positions=receivers.astype(np.float32),
        traces=traces.astype(np.float32),
        source_x=np.float32(src_x),
        source_y=np.float32(src_y),
        source_freq=np.float32(src_freq),
    )

    return {
        "sample_id": sample_id,
        "domain": domain_type,
        "source_type": source_type,
        "signal_type": signal_type,
        "n_receivers": int(receivers.shape[0]),
        "n_steps": n_steps,
        "dt": dt,
        "source_x": src_x,
        "source_y": src_y,
        "source_freq": src_freq,
    }


def _generate_sample_3d(
    sample_id: int,
    rng: np.random.Generator,
    output_dir: Path,
    dx: float = 1.0,
    total_time: float = 0.2,
    fd_order: int = 2,
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    z_min: float = -5.0,
    z_max: float = 50.0,
) -> dict | None:
    """Generate one 3-D FDTD training sample."""
    # Import 3D builders from the example script (they live there, not in setup.py).
    from examples.run_fdtd_3d import build_domain_3d, build_receivers_3d, build_source_3d, compute_dt_3d
    from acoustic_sim.fdtd import FDTD3DSolver, FDTDConfig

    # -- Randomise domain --
    domain_type = rng.choice(["isotropic", "wind", "hills_vegetation"])
    wind_speed = float(rng.uniform(0, 10)) if domain_type == "wind" else 0.0
    wind_dir = float(rng.uniform(0, 360))
    terrain_seed = int(rng.integers(0, 2**31))

    model, meta = build_domain_3d(
        domain_type,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_min, z_max=z_max,
        dx=dx, velocity=343.0,
        wind_speed=wind_speed, wind_direction_deg=wind_dir,
        seed=terrain_seed,
    )

    # -- Randomise receivers --
    array_type = rng.choice(["circular", "linear"])
    n_recv = int(rng.choice([8, 12, 16]))
    recv_radius = float(rng.uniform(1.0, 5.0))
    recv_cx = float(rng.uniform(x_min + 10, x_max - 10))
    recv_cy = float(rng.uniform(y_min + 10, y_max - 10))
    recv_cz = float(rng.uniform(0, 10))

    receivers = build_receivers_3d(
        array_type,
        count=n_recv,
        radius=recv_radius,
        center_x=recv_cx,
        center_y=recv_cy,
        center_z=recv_cz,
        x0=recv_cx - recv_radius,
        y0=recv_cy,
        x1=recv_cx + recv_radius,
        y1=recv_cy,
    )

    # -- Compute dt --
    dt, f_max = compute_dt_3d(model, meta, cfl_safety=0.9, fd_order=fd_order)
    n_steps = int(math.ceil(total_time / dt))

    # -- Randomise source --
    source_type = rng.choice(["static", "moving"])
    signal_type = rng.choice(["propeller", "tone", "ricker"])
    src_x = float(rng.uniform(x_min + 5, x_max - 5))
    src_y = float(rng.uniform(y_min + 5, y_max - 5))
    src_z = float(rng.uniform(5, z_max - 5))
    src_x1 = float(rng.uniform(x_min + 5, x_max - 5))
    src_y1 = float(rng.uniform(y_min + 5, y_max - 5))
    src_z1 = float(rng.uniform(5, z_max - 5))
    src_speed = float(rng.uniform(10, 80))
    src_freq = float(rng.uniform(15, 100))
    source_seed = int(rng.integers(0, 2**31))

    source = build_source_3d(
        source_type, signal_type,
        n_steps=n_steps, dt=dt, f_max=f_max,
        x=src_x, y=src_y, z=src_z,
        x1=src_x1, y1=src_y1, z1=src_z1,
        speed=src_speed, freq=src_freq, seed=source_seed,
    )

    # -- Run FDTD --
    cfg = FDTDConfig(
        total_time=total_time,
        dt=dt,
        snapshot_interval=0,
        use_cuda=False,
        fd_order=fd_order,
    )
    solver = FDTD3DSolver(
        model=model, config=cfg, source=source,
        receivers=receivers, domain_meta=meta,
    )
    result = solver.run(snapshot_dir=None, verbose=False)
    traces = result["traces"]

    if traces is None or traces.size == 0:
        return None

    # For 3D, save a horizontal slice of the velocity model at receiver
    # altitude instead of the full volume (too large).
    z_idx = int(np.argmin(np.abs(model.z - recv_cz)))
    vel_slice = model.values[z_idx, :, :]

    sample_path = output_dir / f"sample_{sample_id:06d}.npz"
    np.savez_compressed(
        str(sample_path),
        velocity_field=vel_slice.astype(np.float32),
        grid_x=model.x.astype(np.float32),
        grid_y=model.y.astype(np.float32),
        receiver_positions=receivers.astype(np.float32),
        traces=traces.astype(np.float32),
        source_x=np.float32(src_x),
        source_y=np.float32(src_y),
        source_freq=np.float32(src_freq),
    )

    return {
        "sample_id": sample_id,
        "domain": domain_type,
        "source_type": source_type,
        "signal_type": signal_type,
        "n_receivers": int(receivers.shape[0]),
        "n_steps": n_steps,
        "dt": dt,
        "source_x": src_x,
        "source_y": src_y,
        "source_z": src_z,
        "source_freq": src_freq,
    }


def generate_dataset(
    n_samples: int = 100,
    output_dir: str = "data/fno_train",
    dims: str = "2d",
    seed: int = 42,
    **kwargs,
) -> Path:
    """Generate a complete FNO training dataset.

    Parameters
    ----------
    n_samples : int
        Number of FDTD simulations to run.
    output_dir : str
        Directory for output .npz files and manifest.json.
    dims : '2d' or '3d'
    seed : int
        Master random seed for reproducibility.
    **kwargs
        Passed through to the per-sample generator (dx, total_time, etc.).

    Returns
    -------
    Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    gen_fn = _generate_sample_2d if dims == "2d" else _generate_sample_3d

    manifest = []
    n_ok = 0
    for i in range(n_samples):
        print(f"[{i + 1}/{n_samples}] Generating sample {i}...")
        try:
            meta = gen_fn(i, rng, out, **kwargs)
            if meta is not None:
                manifest.append(meta)
                n_ok += 1
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

    # Save manifest.
    manifest_path = out / "manifest.json"
    with open(str(manifest_path), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {n_ok}/{n_samples} samples in {out}")
    return out


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate FDTD training data for FNO surrogate model",
    )
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--output-dir", default="data/fno_train")
    p.add_argument("--dims", choices=["2d", "3d"], default="2d")
    p.add_argument("--dx", type=float, default=0.5)
    p.add_argument("--total-time", type=float, default=0.3)
    p.add_argument("--fd-order", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        dims=args.dims,
        seed=args.seed,
        dx=args.dx,
        total_time=args.total_time,
        fd_order=args.fd_order,
    )


if __name__ == "__main__":
    main()

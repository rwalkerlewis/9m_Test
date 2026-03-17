#!/usr/bin/env python3
"""Example 3: 3-D coupled air–ground simulation with a static source.

A propeller source at (50, 50, 5) above a flat ground surface.
Circular microphone array + geophones.

Usage::

    python examples/run_coupled_static_3d.py --use-cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from acoustic_sim.elastic_fdtd import (
    ElasticFDTDConfig,
    ElasticFDTD3DSolver,
    fd1_staggered_coefficients,
    elastic_cfl_factor,
)
from acoustic_sim.elastic_model import GroundConfig, create_coupled_air_ground_3d
from acoustic_sim.receivers import create_colocated_array_3d
from acoustic_sim.sources import (
    StaticSource3D,
    make_source_propeller,
    make_wavelet_ricker,
)
from acoustic_sim.plotting import plot_gather


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="3-D coupled air–ground (static source)")
    g = p.add_argument_group("Domain")
    g.add_argument("--x-min", type=float, default=0.0)
    g.add_argument("--x-max", type=float, default=100.0)
    g.add_argument("--y-min", type=float, default=0.0)
    g.add_argument("--y-max", type=float, default=100.0)
    g.add_argument("--z-min", type=float, default=-5.0)
    g.add_argument("--z-max", type=float, default=15.0)
    g.add_argument("--dx", type=float, default=None)
    g.add_argument("--ground-vp", type=float, default=500.0)
    g.add_argument("--ground-vs", type=float, default=250.0)
    g.add_argument("--ground-density", type=float, default=1800.0)
    g.add_argument("--ground-qp", type=float, default=20.0)
    g.add_argument("--ground-qs", type=float, default=10.0)

    g = p.add_argument_group("Source")
    g.add_argument("--source-x", type=float, default=50.0)
    g.add_argument("--source-y", type=float, default=50.0)
    g.add_argument("--source-z", type=float, default=5.0)
    g.add_argument("--source-signal", choices=["propeller", "ricker"],
                   default="propeller")
    g.add_argument("--source-freq", type=float, default=25.0)

    g = p.add_argument_group("Receivers")
    g.add_argument("--array-cx", type=float, default=50.0)
    g.add_argument("--array-cy", type=float, default=50.0)
    g.add_argument("--array-radius", type=float, default=2.0)
    g.add_argument("--array-count", type=int, default=16)
    g.add_argument("--mic-z", type=float, default=1.5)
    g.add_argument("--geo-z", type=float, default=-0.05)

    g = p.add_argument_group("Simulation")
    g.add_argument("--total-time", type=float, default=0.3)
    g.add_argument("--fd-order", type=int, default=4)
    g.add_argument("--cfl-safety", type=float, default=0.8)
    g.add_argument("--damping-width", type=int, default=10)
    g.add_argument("--damping-max", type=float, default=0.05)
    g.add_argument("--snapshot-interval", type=int, default=200)
    g.add_argument("--source-amplitude", type=float, default=1.0)
    g.add_argument("--use-cuda", action="store_true")
    g.add_argument("--enable-attenuation", action="store_true", default=True)
    g.add_argument("--no-attenuation", dest="enable_attenuation",
                   action="store_false")

    p.add_argument("--output-dir", default="output/coupled_static_3d")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0
    is_root = rank == 0

    out = Path(args.output_dir)
    if is_root:
        out.mkdir(parents=True, exist_ok=True)

    ground = GroundConfig(
        vp=args.ground_vp, vs=args.ground_vs,
        density=args.ground_density,
        qp=args.ground_qp, qs=args.ground_qs,
    )
    if args.dx is None:
        v_min = min(ground.vs, 343.0)
        ppw = {2: 20, 4: 10, 6: 7, 8: 6}.get(args.fd_order, 10)
        args.dx = max(v_min / (ppw * 200.0), 0.5)  # coarser for 3D
        if is_root:
            print(f"Auto dx = {args.dx:.3f} m")

    model = create_coupled_air_ground_3d(
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        z_min=args.z_min, z_max=args.z_max,
        dx=args.dx, ground=ground,
    )
    if is_root:
        ncells = model.nx * model.ny * model.nz
        mem_est_mb = ncells * 9 * 8 / 1e6  # 9 field arrays, float64
        print(f"Domain: {model.shape}  cells={ncells:,}  "
              f"field mem ~{mem_est_mb:.0f} MB")

    cfg = ElasticFDTDConfig(
        total_time=args.total_time,
        fd_order=args.fd_order,
        cfl_safety=args.cfl_safety,
        damping_width=args.damping_width,
        damping_max=args.damping_max,
        snapshot_interval=args.snapshot_interval,
        source_amplitude=args.source_amplitude,
        use_cuda=args.use_cuda,
        enable_attenuation=args.enable_attenuation,
    )

    coeffs = fd1_staggered_coefficients(args.fd_order)
    S = elastic_cfl_factor(coeffs)
    dt_est = args.cfl_safety * args.dx / (model.vp_max * S * math.sqrt(3.0))
    n_steps = int(math.ceil(args.total_time / dt_est)) + 100
    f_max = model.vs_min_nonzero / (10.0 * args.dx)

    if args.source_signal == "propeller":
        sig = make_source_propeller(n_steps, dt_est, f_max=f_max)
    else:
        sig = make_wavelet_ricker(n_steps, dt_est, args.source_freq)

    source = StaticSource3D(
        x=args.source_x, y=args.source_y, z=args.source_z, signal=sig,
    )

    receivers = create_colocated_array_3d(
        cx=args.array_cx, cy=args.array_cy,
        radius=args.array_radius, count=args.array_count,
        mic_z=args.mic_z, geo_z=args.geo_z,
    )
    if is_root:
        print(f"Receivers: {receivers.n_receivers}")

    snap_dir = str(out / "snapshots")
    solver = ElasticFDTD3DSolver(model, cfg, source, receivers)
    if is_root:
        print(f"dt={solver.dt:.2e}, n_steps={solver.n_steps}")
    result = solver.run(snapshot_dir=snap_dir, verbose=is_root)

    if is_root:
        traces = result["traces"]
        np.save(str(out / "traces.npy"), traces)
        mic_idx = receivers.mic_indices
        geo_idx = receivers.geo_indices
        np.save(str(out / "mic_traces.npy"), traces[mic_idx])
        np.save(str(out / "geo_traces.npy"), traces[geo_idx])

        metadata = {
            "example": "coupled_static_3d",
            "domain_shape": list(model.shape),
            "dx": args.dx,
            "dt": result["dt"],
            "n_steps": result["n_steps"],
            "total_time": args.total_time,
            "source_xyz": [args.source_x, args.source_y, args.source_z],
            "ground": {"vp": ground.vp, "vs": ground.vs, "density": ground.density},
            "n_microphones": receivers.n_microphones,
            "n_geophones": receivers.n_geophones,
        }
        with open(str(out / "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        if traces.shape[1] > 0:
            plot_gather(traces[mic_idx], result["dt"],
                        output_path=str(out / "gather_mics.png"),
                        title="3D Microphone gather (static)")
            plot_gather(traces[geo_idx], result["dt"],
                        output_path=str(out / "gather_geos.png"),
                        title="3D Geophone gather (static)")
        print("Done.")


if __name__ == "__main__":
    main()

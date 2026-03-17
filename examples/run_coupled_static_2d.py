#!/usr/bin/env python3
"""Example 1: 2-D coupled air–ground simulation with a static source.

A propeller source sits at 5 m above a flat ground surface.  The elastic
FDTD solver models acoustic propagation in air and seismic wave
propagation in the shallow subsurface.  Microphones record pressure in
air and geophones record vertical particle velocity in the ground.

Usage::

    python examples/run_coupled_static_2d.py --output-dir output/coupled_static_2d
    python examples/run_coupled_static_2d.py --use-cuda   # GPU
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
    ElasticFDTD2DSolver,
    fd1_staggered_coefficients,
    elastic_cfl_factor,
)
from acoustic_sim.elastic_model import (
    GroundConfig,
    create_coupled_air_ground_2d,
)
from acoustic_sim.receivers import create_colocated_array_2d
from acoustic_sim.sources import (
    StaticSource,
    make_source_propeller,
    make_wavelet_ricker,
)
from acoustic_sim.plotting import plot_gather


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="2-D coupled air–ground (static source)")
    g = p.add_argument_group("Domain")
    g.add_argument("--x-min", type=float, default=-100.0)
    g.add_argument("--x-max", type=float, default=100.0)
    g.add_argument("--z-min", type=float, default=-5.0)
    g.add_argument("--z-max", type=float, default=15.0)
    g.add_argument("--dx", type=float, default=None,
                   help="Grid spacing [m]; None → auto from FD order")
    g.add_argument("--ground-vp", type=float, default=500.0)
    g.add_argument("--ground-vs", type=float, default=250.0)
    g.add_argument("--ground-density", type=float, default=1800.0)
    g.add_argument("--ground-qp", type=float, default=20.0)
    g.add_argument("--ground-qs", type=float, default=10.0)

    g = p.add_argument_group("Source")
    g.add_argument("--source-x", type=float, default=50.0)
    g.add_argument("--source-z", type=float, default=5.0)
    g.add_argument("--source-signal", choices=["propeller", "ricker"],
                   default="propeller")
    g.add_argument("--source-freq", type=float, default=25.0,
                   help="Ricker peak frequency [Hz]")

    g = p.add_argument_group("Receivers")
    g.add_argument("--array-cx", type=float, default=0.0)
    g.add_argument("--array-radius", type=float, default=2.0)
    g.add_argument("--array-count", type=int, default=16)
    g.add_argument("--mic-z", type=float, default=1.5)
    g.add_argument("--geo-z", type=float, default=-0.05)

    g = p.add_argument_group("Simulation")
    g.add_argument("--total-time", type=float, default=0.5)
    g.add_argument("--fd-order", type=int, default=4)
    g.add_argument("--cfl-safety", type=float, default=0.8)
    g.add_argument("--damping-width", type=int, default=20)
    g.add_argument("--damping-max", type=float, default=0.05)
    g.add_argument("--snapshot-interval", type=int, default=200)
    g.add_argument("--source-amplitude", type=float, default=1.0)
    g.add_argument("--use-cuda", action="store_true")
    g.add_argument("--enable-attenuation", action="store_true", default=True)
    g.add_argument("--no-attenuation", dest="enable_attenuation",
                   action="store_false")

    p.add_argument("--output-dir", default="output/coupled_static_2d")
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

    # -- Grid spacing (auto if not specified) --
    ground = GroundConfig(
        vp=args.ground_vp, vs=args.ground_vs,
        density=args.ground_density,
        qp=args.ground_qp, qs=args.ground_qs,
    )
    if args.dx is None:
        # Resolve shortest wavelength: Vs in ground or Vp in air
        v_min = min(ground.vs, 343.0)
        ppw = {2: 20, 4: 10, 6: 7, 8: 6}.get(args.fd_order, 10)
        f_max_target = 200.0  # Hz — enough for propeller BPF
        args.dx = v_min / (ppw * f_max_target)
        args.dx = max(args.dx, 0.05)  # don't go too fine
        if is_root:
            print(f"Auto dx = {args.dx:.3f} m  (v_min={v_min}, ppw={ppw})")

    # -- Build domain --
    model = create_coupled_air_ground_2d(
        x_min=args.x_min, x_max=args.x_max,
        z_min=args.z_min, z_max=args.z_max,
        dx=args.dx, ground=ground,
    )
    if is_root:
        print(f"Domain: {model.shape}  dx={args.dx}  "
              f"vp_max={model.vp_max}  vs_min={model.vs_min_nonzero}")

    # -- Build config --
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

    # -- Compute dt and build source signal --
    coeffs = fd1_staggered_coefficients(args.fd_order)
    S = elastic_cfl_factor(coeffs)
    dt_est = args.cfl_safety * args.dx / (model.vp_max * S * math.sqrt(2.0))
    n_steps = int(math.ceil(args.total_time / dt_est)) + 100
    f_max = model.vs_min_nonzero / (10.0 * args.dx)

    if args.source_signal == "propeller":
        sig = make_source_propeller(n_steps, dt_est, f_max=f_max)
    else:
        sig = make_wavelet_ricker(n_steps, dt_est, args.source_freq)

    # -- Build source (2D: x, y where y=z) --
    source = StaticSource(x=args.source_x, y=args.source_z, signal=sig)

    # -- Build receivers --
    receivers = create_colocated_array_2d(
        cx=args.array_cx,
        radius=args.array_radius,
        count=args.array_count,
        mic_z=args.mic_z,
        geo_z=args.geo_z,
    )

    if is_root:
        print(f"Receivers: {receivers.n_receivers} "
              f"({receivers.n_microphones} mics + {receivers.n_geophones} geos)")

    # -- Run solver --
    snap_dir = str(out / "snapshots")
    solver = ElasticFDTD2DSolver(model, cfg, source, receivers)
    if is_root:
        print(f"dt={solver.dt:.2e} s, n_steps={solver.n_steps}, "
              f"fd_order={args.fd_order}")
    result = solver.run(snapshot_dir=snap_dir, verbose=is_root)

    # -- Save outputs --
    if is_root:
        traces = result["traces"]
        np.save(str(out / "traces.npy"), traces)
        print(f"Saved traces {traces.shape}")

        # Separate mic and geo traces
        mic_idx = receivers.mic_indices
        geo_idx = receivers.geo_indices
        np.save(str(out / "mic_traces.npy"), traces[mic_idx])
        np.save(str(out / "geo_traces.npy"), traces[geo_idx])

        metadata = {
            "example": "coupled_static_2d",
            "domain_shape": list(model.shape),
            "dx": args.dx,
            "dt": result["dt"],
            "n_steps": result["n_steps"],
            "total_time": args.total_time,
            "fd_order": args.fd_order,
            "source_x": args.source_x,
            "source_z": args.source_z,
            "source_signal": args.source_signal,
            "ground": {
                "vp": ground.vp, "vs": ground.vs,
                "density": ground.density,
                "qp": ground.qp, "qs": ground.qs,
            },
            "n_microphones": receivers.n_microphones,
            "n_geophones": receivers.n_geophones,
            "receiver_positions": receivers.positions.tolist(),
            "sensor_types": receivers.sensor_types,
            "use_cuda": args.use_cuda,
            "enable_attenuation": args.enable_attenuation,
        }
        with open(str(out / "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Plot gathers
        if traces.shape[1] > 0:
            plot_gather(
                traces[mic_idx], result["dt"],
                output_path=str(out / "gather_mics.png"),
                title="Microphone gather (pressure)",
            )
            plot_gather(
                traces[geo_idx], result["dt"],
                output_path=str(out / "gather_geos.png"),
                title="Geophone gather (vz)",
            )
        print("Done.")


if __name__ == "__main__":
    main()

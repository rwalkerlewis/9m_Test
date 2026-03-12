#!/usr/bin/env python3
"""Run a single FDTD acoustic simulation.

Usage::

    python examples/run_fdtd.py --domain wind --source-type moving \\
        --source-signal propeller --array circular --output-dir output/test

All domain / source / receiver construction is handled by
``acoustic_sim.setup``.  This script is just CLI parsing, wiring,
and output saving.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the package is importable when running from the repo root.
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from acoustic_sim.fdtd import FDTDConfig, FDTDSolver
from acoustic_sim.plotting import plot_domain, plot_gather
from acoustic_sim.setup import build_domain, build_receivers, build_source, compute_dt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FDTD acoustic simulation runner")

    # Domain
    g = p.add_argument_group("Domain")
    g.add_argument("--domain", choices=["isotropic", "wind", "hills_vegetation"],
                   default="isotropic")
    g.add_argument("--velocity", type=float, default=343.0)
    g.add_argument("--dx", type=float, default=0.5)
    g.add_argument("--x-min", type=float, default=-50.0)
    g.add_argument("--x-max", type=float, default=50.0)
    g.add_argument("--y-min", type=float, default=-50.0)
    g.add_argument("--y-max", type=float, default=50.0)
    g.add_argument("--wind-speed", type=float, default=15.0)
    g.add_argument("--wind-dir", type=float, default=45.0)
    g.add_argument("--dirt-velocity", type=float, default=1500.0)
    g.add_argument("--seed", type=int, default=42)

    # Source
    g = p.add_argument_group("Source")
    g.add_argument("--source-type", choices=["static", "moving"], default="static")
    g.add_argument("--source-x", type=float, default=0.0)
    g.add_argument("--source-y", type=float, default=0.0)
    g.add_argument("--source-x1", type=float, default=30.0)
    g.add_argument("--source-y1", type=float, default=0.0)
    g.add_argument("--source-speed", type=float, default=50.0)
    g.add_argument("--source-arc-height", type=float, default=0.0,
                   help="Parabolic arc height [m] added to y (0 = straight line)")

    # Source signal
    g = p.add_argument_group("Source signal")
    g.add_argument("--source-signal",
                   choices=["file", "propeller", "tone", "noise", "ricker"],
                   default="ricker")
    g.add_argument("--source-wav", default="audio/input.wav")
    g.add_argument("--max-seconds", type=float, default=None)
    g.add_argument("--source-freq", type=float, default=25.0)
    g.add_argument("--blade-count", type=int, default=3)
    g.add_argument("--rpm", type=float, default=3600.0)
    g.add_argument("--harmonics", type=int, default=14)

    # Receivers
    g = p.add_argument_group("Receivers")
    g.add_argument("--array", choices=["concentric", "circular", "linear"],
                   default="circular")
    g.add_argument("--receiver-count", type=int, default=16)
    g.add_argument("--receiver-radius", type=float, default=15.0)
    g.add_argument("--receiver-radii", default="10,20,30,40")
    g.add_argument("--receiver-cx", type=float, default=0.0,
                   help="Receiver array centre X (circular/concentric)")
    g.add_argument("--receiver-cy", type=float, default=0.0,
                   help="Receiver array centre Y (circular/concentric)")
    g.add_argument("--receiver-x0", type=float, default=-40.0)
    g.add_argument("--receiver-y0", type=float, default=0.0)
    g.add_argument("--receiver-x1", type=float, default=40.0)
    g.add_argument("--receiver-y1", type=float, default=0.0)

    # Simulation
    g = p.add_argument_group("Simulation")
    g.add_argument("--total-time", type=float, default=0.3)
    g.add_argument("--snapshot-interval", type=int, default=50)
    g.add_argument("--damping-width", type=int, default=40)
    g.add_argument("--damping-max", type=float, default=0.15)
    g.add_argument("--source-amplitude", type=float, default=1.0)
    g.add_argument("--air-absorption", type=float, default=0.005)
    g.add_argument("--use-cuda", action="store_true",
                   help="Use CuPy for GPU acceleration")
    g.add_argument("--fd-order", type=int, default=2,
                   help="FD spatial order of accuracy (2, 4, 6, …)")

    # Output
    p.add_argument("--output-dir", default="output/test")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Determine MPI rank (rank 0 does I/O)
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0
    is_root = rank == 0

    out = Path(args.output_dir)
    if is_root:
        out.mkdir(parents=True, exist_ok=True)

    # -- Build domain -------------------------------------------------------
    model, meta = build_domain(
        args.domain,
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        dx=args.dx, velocity=args.velocity,
        wind_speed=args.wind_speed,
        wind_direction_deg=args.wind_dir,
        dirt_velocity=args.dirt_velocity,
        seed=args.seed,
    )
    if is_root:
        print(f"Domain: {meta.description}  grid={model.shape}  "
              f"c=[{model.c_min:.1f}, {model.c_max:.1f}] m/s")

    # -- Build receivers ----------------------------------------------------
    radii = [float(r) for r in args.receiver_radii.split(",")]
    receivers = build_receivers(
        args.array,
        count=args.receiver_count,
        radius=args.receiver_radius,
        radii=radii,
        x0=args.receiver_x0, y0=args.receiver_y0,
        x1=args.receiver_x1, y1=args.receiver_y1,
        center_x=args.receiver_cx, center_y=args.receiver_cy,
    )
    if is_root:
        print(f"Receivers: {receivers.shape[0]} in '{args.array}' layout")

    # -- Compute dt and build config ----------------------------------------
    dt, f_max = compute_dt(model, meta, cfl_safety=0.9, fd_order=args.fd_order)
    n_steps = int(math.ceil(args.total_time / dt))
    if is_root:
        print(f"dt={dt:.2e} s,  n_steps={n_steps},  f_max={f_max:.1f} Hz")

    cfg = FDTDConfig(
        total_time=args.total_time,
        dt=dt,
        snapshot_interval=args.snapshot_interval,
        damping_width=args.damping_width,
        damping_max=args.damping_max,
        source_amplitude=args.source_amplitude,
        air_absorption=args.air_absorption,
        use_cuda=args.use_cuda,
        fd_order=args.fd_order,
    )

    # -- Build source -------------------------------------------------------
    source = build_source(
        args.source_type, args.source_signal,
        n_steps=n_steps, dt=dt, f_max=f_max,
        x=args.source_x, y=args.source_y,
        x1=args.source_x1, y1=args.source_y1,
        speed=args.source_speed,
        arc_height=args.source_arc_height,
        freq=args.source_freq,
        blade_count=args.blade_count,
        rpm=args.rpm, harmonics=args.harmonics,
        seed=args.seed,
        wav_path=args.source_wav,
        max_seconds=args.max_seconds,
    )

    # -- Domain plot (rank 0 only) ------------------------------------------
    src_xy = np.array([args.source_x, args.source_y])
    src_path = None
    if args.source_type == "moving":
        frac = np.linspace(0.0, 1.0, 60)
        src_path = np.column_stack([
            args.source_x + frac * (args.source_x1 - args.source_x),
            args.source_y + frac * (args.source_y1 - args.source_y)
            + args.source_arc_height * 4.0 * frac * (1.0 - frac),
        ])
    if is_root:
        plot_domain(
            model,
            output_path=str(out / "domain.png"),
            receivers=receivers,
            source_xy=src_xy,
            source_path=src_path,
            attenuation=meta.attenuation,
            wind_vx=meta.wind_vx,
            wind_vy=meta.wind_vy,
            title=f"Domain: {args.domain}",
        )

    # -- Run FDTD -----------------------------------------------------------
    snap_dir = str(out / "snapshots")
    solver = FDTDSolver(
        model=model, config=cfg, source=source,
        receivers=receivers, domain_meta=meta,
    )
    result = solver.run(snapshot_dir=snap_dir, verbose=True)

    # -- Save outputs (rank 0 only) -----------------------------------------
    if is_root:
        traces = result["traces"]
        np.save(str(out / "traces.npy"), traces)
        print(f"Saved traces {traces.shape} to {out / 'traces.npy'}")

        path_slices = result.get("path_slices", [])
        if path_slices:
            slice_matrix = np.zeros((len(path_slices), len(model.x)))
            for i, ps in enumerate(path_slices):
                slice_matrix[i, :] = ps["slice"]
            np.save(str(out / "path_slices.npy"), slice_matrix)

        metadata = {
            "domain": args.domain,
            "source_type": args.source_type,
            "source_signal": args.source_signal,
            "array": args.array,
            "dx": args.dx,
            "dt": result["dt"],
            "n_steps": result["n_steps"],
            "total_time": args.total_time,
            "n_receivers": int(receivers.shape[0]),
            "receiver_positions": receivers.tolist(),
            "source_x": args.source_x,
            "source_y": args.source_y,
            "velocity": args.velocity,
            "wind_vx": meta.wind_vx,
            "wind_vy": meta.wind_vy,
            "use_cuda": args.use_cuda,
            "fd_order": args.fd_order,
        }
        with open(str(out / "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        plot_gather(
            traces, result["dt"],
            output_path=str(out / "gather.png"),
            title=f"Gather: {args.domain} / {args.source_type} / {args.array}",
        )
        print("Done.")


if __name__ == "__main__":
    main()

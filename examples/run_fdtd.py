#!/usr/bin/env python3
"""Run a single FDTD acoustic simulation.

Usage (MPI)::

    mpirun -np 4 python examples/run_fdtd.py \\
        --domain isotropic --source-type static \\
        --source-signal file --source-wav audio/input.wav --max-seconds 0.3 \\
        --array circular --output-dir output/test

Usage (single process)::

    python examples/run_fdtd.py --domain isotropic --source-type static \\
        --source-signal ricker --array linear --output-dir output/quick
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the package is importable even when running from the repo root.
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from acoustic_sim.domains import (
    DomainMeta,
    create_hills_vegetation_domain,
    create_isotropic_domain,
    create_wind_domain,
)
from acoustic_sim.fdtd import FDTDConfig, FDTDSolver
from acoustic_sim.model import VelocityModel
from acoustic_sim.plotting import plot_domain, plot_gather
from acoustic_sim.receivers import (
    create_receiver_circle,
    create_receiver_concentric,
    create_receiver_line,
)
from acoustic_sim.sources import (
    MovingSource,
    StaticSource,
    make_source_from_file,
    make_source_noise,
    make_source_propeller,
    make_source_tone,
    make_wavelet_ricker,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FDTD acoustic simulation runner")

    # Domain ---------------------------------------------------------------
    p.add_argument(
        "--domain",
        choices=["isotropic", "wind", "hills_vegetation"],
        default="isotropic",
    )
    p.add_argument("--velocity", type=float, default=343.0, help="Background velocity [m/s]")
    p.add_argument("--dx", type=float, default=0.5, help="Grid spacing [m]")
    p.add_argument("--x-min", type=float, default=-50.0)
    p.add_argument("--x-max", type=float, default=50.0)
    p.add_argument("--y-min", type=float, default=-50.0)
    p.add_argument("--y-max", type=float, default=50.0)
    # wind
    p.add_argument("--wind-speed", type=float, default=15.0, help="Wind speed [m/s]")
    p.add_argument("--wind-dir", type=float, default=45.0, help="Wind direction [deg]")
    # hills
    p.add_argument("--dirt-velocity", type=float, default=1500.0)
    p.add_argument("--seed", type=int, default=42)

    # Source ----------------------------------------------------------------
    p.add_argument(
        "--source-type", choices=["static", "moving"], default="static",
    )
    p.add_argument("--source-x", type=float, default=0.0)
    p.add_argument("--source-y", type=float, default=0.0)
    p.add_argument("--source-x1", type=float, default=30.0, help="Moving source end x")
    p.add_argument("--source-y1", type=float, default=0.0, help="Moving source end y")
    p.add_argument("--source-speed", type=float, default=50.0, help="Moving source speed [m/s]")

    # Source signal ---------------------------------------------------------
    p.add_argument(
        "--source-signal",
        choices=["file", "propeller", "tone", "noise", "ricker"],
        default="ricker",
    )
    p.add_argument("--source-wav", type=str, default="audio/input.wav")
    p.add_argument("--max-seconds", type=float, default=None, help="Truncate WAV to N seconds")
    p.add_argument("--source-freq", type=float, default=25.0, help="Peak freq for ricker/tone [Hz]")
    p.add_argument("--blade-count", type=int, default=3)
    p.add_argument("--rpm", type=float, default=3600.0)
    p.add_argument("--harmonics", type=int, default=14)

    # Receiver array --------------------------------------------------------
    p.add_argument(
        "--array", choices=["concentric", "circular", "linear"], default="circular",
    )
    p.add_argument("--receiver-count", type=int, default=16)
    p.add_argument("--receiver-radius", type=float, default=15.0)
    p.add_argument("--receiver-radii", type=str, default="10,20,30,40",
                    help="Comma-separated radii for concentric array")
    p.add_argument("--receiver-x0", type=float, default=-40.0)
    p.add_argument("--receiver-y0", type=float, default=0.0)
    p.add_argument("--receiver-x1", type=float, default=40.0)
    p.add_argument("--receiver-y1", type=float, default=0.0)

    # Simulation ------------------------------------------------------------
    p.add_argument("--total-time", type=float, default=0.3, help="Sim duration [s]")
    p.add_argument("--snapshot-interval", type=int, default=50)
    p.add_argument("--damping-width", type=int, default=20)
    p.add_argument("--use-cuda", action="store_true")

    # Output ----------------------------------------------------------------
    p.add_argument("--output-dir", type=str, default="output/test")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    args = parse_args(argv)
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # ---- Domain -----------------------------------------------------------
    kw = dict(
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        dx=args.dx,
    )
    if args.domain == "isotropic":
        model, meta = create_isotropic_domain(velocity=args.velocity, **kw)
    elif args.domain == "wind":
        model, meta = create_wind_domain(
            velocity=args.velocity,
            wind_speed=args.wind_speed,
            wind_direction_deg=args.wind_dir,
            **kw,
        )
    elif args.domain == "hills_vegetation":
        model, meta = create_hills_vegetation_domain(
            air_velocity=args.velocity,
            dirt_velocity=args.dirt_velocity,
            seed=args.seed,
            **kw,
        )
    else:
        raise ValueError(args.domain)

    if rank == 0:
        print(f"Domain: {meta.description}  grid={model.shape}  "
              f"c=[{model.c_min:.1f}, {model.c_max:.1f}] m/s")

    # ---- Receivers --------------------------------------------------------
    if args.array == "circular":
        receivers = create_receiver_circle(
            0.0, 0.0, args.receiver_radius, args.receiver_count,
        )
    elif args.array == "concentric":
        radii = [float(r) for r in args.receiver_radii.split(",")]
        receivers = create_receiver_concentric(
            0.0, 0.0, radii, args.receiver_count,
        )
    elif args.array == "linear":
        receivers = create_receiver_line(
            args.receiver_x0, args.receiver_y0,
            args.receiver_x1, args.receiver_y1,
            args.receiver_count,
        )
    else:
        raise ValueError(args.array)

    if rank == 0:
        print(f"Receivers: {receivers.shape[0]} in '{args.array}' layout")

    # ---- FDTD config & dt ------------------------------------------------
    cfg = FDTDConfig(
        total_time=args.total_time,
        snapshot_interval=args.snapshot_interval,
        damping_width=args.damping_width,
        use_cuda=args.use_cuda,
    )
    # Pre-compute dt so we know n_steps *before* building the source signal.
    c_max = float(np.max(model.values))
    dt = cfg.cfl_safety * args.dx / (c_max * math.sqrt(2.0))
    cfg.dt = dt
    n_steps = int(math.ceil(cfg.total_time / dt))
    f_max = model.c_min / (10.0 * args.dx)

    if rank == 0:
        print(f"dt={dt:.2e} s,  n_steps={n_steps},  f_max={f_max:.1f} Hz")

    # ---- Source signal ----------------------------------------------------
    sig_kind = args.source_signal
    if sig_kind == "ricker":
        sig = make_wavelet_ricker(n_steps, dt, args.source_freq)
    elif sig_kind == "tone":
        sig = make_source_tone(n_steps, dt, args.source_freq)
    elif sig_kind == "noise":
        sig = make_source_noise(n_steps, dt, f_low=5.0, f_high=f_max, seed=args.seed)
    elif sig_kind == "propeller":
        sig = make_source_propeller(
            n_steps, dt, f_max=f_max,
            blade_count=args.blade_count, rpm=args.rpm,
            harmonics=args.harmonics, seed=args.seed,
        )
    elif sig_kind == "file":
        sig = make_source_from_file(
            args.source_wav, n_steps, dt, f_max,
            max_seconds=args.max_seconds,
        )
    else:
        raise ValueError(sig_kind)

    # ---- Source object ----------------------------------------------------
    if args.source_type == "static":
        source = StaticSource(x=args.source_x, y=args.source_y, signal=sig)
    else:
        source = MovingSource(
            x0=args.source_x, y0=args.source_y,
            x1=args.source_x1, y1=args.source_y1,
            speed=args.source_speed, signal=sig,
        )

    # ---- Domain plot (rank 0) --------------------------------------------
    if rank == 0:
        src_xy = np.array([args.source_x, args.source_y])
        # Build source path for moving sources.
        if args.source_type == "moving":
            src_path = np.column_stack([
                np.linspace(args.source_x, args.source_x1, 20),
                np.linspace(args.source_y, args.source_y1, 20),
            ])
        else:
            src_path = None
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

    # ---- Run FDTD ---------------------------------------------------------
    snap_dir = str(out / "snapshots")
    solver = FDTDSolver(
        model=model,
        config=cfg,
        source=source,
        receivers=receivers,
        comm=comm,
        domain_meta=meta,
    )
    result = solver.run(snapshot_dir=snap_dir, verbose=(rank == 0))

    # ---- Save outputs (rank 0) -------------------------------------------
    if rank == 0:
        traces = result["traces"]
        np.save(str(out / "traces.npy"), traces)
        print(f"Saved traces {traces.shape} to {out / 'traces.npy'}")

        # Metadata sidecar.
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
        }
        with open(str(out / "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Gather plot.
        plot_gather(
            traces, result["dt"],
            output_path=str(out / "gather.png"),
            title=f"Gather: {args.domain} / {args.source_type} / {args.array}",
        )

        print("Done.")


if __name__ == "__main__":
    main()

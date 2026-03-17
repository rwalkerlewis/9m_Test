#!/usr/bin/env python3
"""Run a 3-D FDTD simulation with an erratic quadcopter source.

The source follows a random, unpredictable trajectory confined within
a ±20 m domain, mimicking a small drone manoeuvring aggressively.

Usage::

    python examples/run_fdtd_erratic.py --output-dir output/erratic_quadcopter

The script saves:
  - traces.npy        receiver time-series
  - metadata.json     simulation parameters + receiver positions
  - source_trajectory.npy   (n_steps, 3) ground-truth trajectory
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

from acoustic_sim.domains import create_isotropic_domain_3d
from acoustic_sim.fdtd import fd2_coefficients, fd2_cfl_factor, FDTD3DConfig, FDTD3DSolver
from acoustic_sim.receivers import create_receiver_circle_3d
from acoustic_sim.sources import (
    ErraticQuadcopterSource3D,
    make_drone_harmonics,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Erratic quadcopter 3-D FDTD")

    # Domain
    p.add_argument("--domain-half", type=float, default=20.0,
                   help="Half-extent of domain in x/y [m]")
    p.add_argument("--z-min", type=float, default=-2.0)
    p.add_argument("--z-max", type=float, default=22.0)
    p.add_argument("--dx", type=float, default=1.0)
    p.add_argument("--velocity", type=float, default=343.0)

    # Source
    p.add_argument("--source-x0", type=float, default=15.0)
    p.add_argument("--source-y0", type=float, default=0.0)
    p.add_argument("--source-z0", type=float, default=10.0)
    p.add_argument("--mean-speed", type=float, default=8.0,
                   help="Characteristic quadcopter speed [m/s]")
    p.add_argument("--agility", type=float, default=6.0,
                   help="Velocity change rate (higher = more erratic)")
    p.add_argument("--speed-var", type=float, default=4.0)
    p.add_argument("--z-floor", type=float, default=2.0,
                   help="Minimum source altitude [m]")
    p.add_argument("--seed", type=int, default=42)

    # Source signal
    p.add_argument("--fundamental-freq", type=float, default=150.0)
    p.add_argument("--n-harmonics", type=int, default=4)
    p.add_argument("--source-level-dB", type=float, default=90.0)

    # Receivers
    p.add_argument("--receiver-count", type=int, default=16)
    p.add_argument("--receiver-radius", type=float, default=2.0)
    p.add_argument("--receiver-cz", type=float, default=0.0)

    # Simulation
    p.add_argument("--total-time", type=float, default=0.5,
                   help="Simulation duration [s]")
    p.add_argument("--snapshot-interval", type=int, default=100)
    p.add_argument("--damping-width", type=int, default=40)
    p.add_argument("--damping-max", type=float, default=0.15)
    p.add_argument("--source-amplitude", type=float, default=1.0)
    p.add_argument("--air-absorption", type=float, default=0.005)
    p.add_argument("--use-cuda", action="store_true",
                   help="Use CuPy for GPU acceleration")
    p.add_argument("--fd-order", type=int, default=8)

    # Output
    p.add_argument("--output-dir", default="output/erratic_quadcopter")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    half = args.domain_half

    # -- Domain (isotropic air) --------------------------------------------
    model, meta = create_isotropic_domain_3d(
        velocity=args.velocity,
        x_min=-half, x_max=half,
        y_min=-half, y_max=half,
        z_min=args.z_min, z_max=args.z_max,
        dx=args.dx,
    )
    print(f"Domain: isotropic  grid={model.shape}  "
          f"c={model.c_min:.0f} m/s")
    ncells = model.nx * model.ny * model.nz
    mem_mb = ncells * 8 * 2 / 1e6
    print(f"  Grid cells: {ncells:,}  Estimated memory: {mem_mb:.1f} MB")

    # -- Receivers ---------------------------------------------------------
    receivers = create_receiver_circle_3d(
        0.0, 0.0, args.receiver_radius, args.receiver_count,
        z=args.receiver_cz,
    )
    print(f"Receivers: {receivers.shape[0]} circular  "
          f"radius={args.receiver_radius} m  z={args.receiver_cz} m")

    # -- CFL / dt ----------------------------------------------------------
    coeffs = fd2_coefficients(args.fd_order)
    spec_radius = fd2_cfl_factor(coeffs)
    c_max = float(np.max(model.values))
    dt = 0.9 * 2.0 * model.dx / (c_max * math.sqrt(3.0 * spec_radius))
    f_max = float(np.min(model.values)) / (10.0 * model.dx)
    n_steps = int(math.ceil(args.total_time / dt))
    print(f"dt={dt:.2e} s,  n_steps={n_steps},  f_max={f_max:.1f} Hz")

    cfg = FDTD3DConfig(
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

    # -- Source signal ------------------------------------------------------
    sig = make_drone_harmonics(
        n_steps, dt,
        fundamental=args.fundamental_freq,
        n_harmonics=args.n_harmonics,
        source_level_dB=args.source_level_dB,
        f_max=f_max,
    )

    # -- Erratic source ----------------------------------------------------
    source = ErraticQuadcopterSource3D(
        x0=args.source_x0,
        y0=args.source_y0,
        z0=args.source_z0,
        bbox_min=(-half, -half, args.z_floor),
        bbox_max=(half, half, args.z_max - 2.0),
        mean_speed=args.mean_speed,
        agility=args.agility,
        speed_var=args.speed_var,
        signal=sig,
        seed=args.seed,
    )

    # Pre-build and save trajectory.
    traj = source.get_trajectory(n_steps, dt)
    np.save(str(out / "source_trajectory.npy"), traj)
    traj_range = np.linalg.norm(traj - np.array([0, 0, 0]), axis=1)
    print(f"Source trajectory: range to origin "
          f"[{traj_range.min():.1f}, {traj_range.max():.1f}] m")
    print(f"  Start: ({traj[0, 0]:.1f}, {traj[0, 1]:.1f}, {traj[0, 2]:.1f})")
    print(f"  End:   ({traj[-1, 0]:.1f}, {traj[-1, 1]:.1f}, {traj[-1, 2]:.1f})")

    # -- Run 3-D FDTD ------------------------------------------------------
    solver = FDTD3DSolver(
        model=model, config=cfg, source=source,
        receivers=receivers, domain_meta=meta,
    )
    snap_dir = str(out / "snapshots")
    result = solver.run(snapshot_dir=snap_dir, verbose=True)

    # -- Save outputs ------------------------------------------------------
    traces = result["traces"]
    np.save(str(out / "traces.npy"), traces)
    print(f"Saved traces {traces.shape} to {out / 'traces.npy'}")

    metadata = {
        "domain": "isotropic",
        "source_type": "erratic_quadcopter",
        "source_signal": "drone_harmonics",
        "array": "circular",
        "dx": args.dx,
        "dt": result["dt"],
        "n_steps": result["n_steps"],
        "total_time": args.total_time,
        "n_receivers": int(receivers.shape[0]),
        "receiver_positions": receivers.tolist(),
        "source_x": args.source_x0,
        "source_y": args.source_y0,
        "source_z": args.source_z0,
        "source_mean_speed": args.mean_speed,
        "source_agility": args.agility,
        "source_seed": args.seed,
        "x_min": -half,
        "x_max": half,
        "y_min": -half,
        "y_max": half,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "velocity": args.velocity,
        "wind_vx": 0.0,
        "wind_vy": 0.0,
        "wind_vz": 0.0,
        "fd_order": args.fd_order,
        "grid_shape": list(model.shape),
        "source_trajectory_file": "source_trajectory.npy",
    }
    with open(str(out / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {out / 'metadata.json'}")

    # -- Domain + trajectory plot ------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # x-y plan view
    ax = axes[0]
    ax.plot(traj[:, 0], traj[:, 1], "r-", lw=1.2, label="Source path")
    ax.scatter(traj[0, 0], traj[0, 1], c="yellow", s=100, marker="*",
               edgecolors="k", zorder=6, label="Start")
    ax.scatter(traj[-1, 0], traj[-1, 1], c="orange", s=80, marker="s",
               edgecolors="k", zorder=6, label="End")
    ax.scatter(receivers[:, 0], receivers[:, 1], s=20, c="cyan",
               edgecolors="k", linewidths=0.4, zorder=5, label="Receivers")
    ax.set_xlim(-half * 1.05, half * 1.05)
    ax.set_ylim(-half * 1.05, half * 1.05)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Domain (x-y): erratic quadcopter")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # x-z side view
    ax = axes[1]
    ax.plot(traj[:, 0], traj[:, 2], "r-", lw=1.2, label="Source path")
    ax.scatter(traj[0, 0], traj[0, 2], c="yellow", s=100, marker="*",
               edgecolors="k", zorder=6, label="Start")
    ax.scatter(traj[-1, 0], traj[-1, 2], c="orange", s=80, marker="s",
               edgecolors="k", zorder=6, label="End")
    ax.scatter(receivers[:, 0], receivers[:, 2], s=20, c="cyan",
               edgecolors="k", linewidths=0.4, zorder=5, label="Receivers")
    ax.set_xlim(-half * 1.05, half * 1.05)
    ax.set_ylim(args.z_min, args.z_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Domain (x-z): erratic quadcopter")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out / "domain_trajectory.png"), dpi=170)
    plt.close(fig)
    print(f"Saved domain plot to {out / 'domain_trajectory.png'}")

    # -- Gather plot -------------------------------------------------------
    from acoustic_sim.plotting import plot_gather
    plot_gather(
        traces, result["dt"],
        output_path=str(out / "gather.png"),
        title="Gather: erratic quadcopter / drone_harmonics / circular",
    )

    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Estimate memory, resolution, and timestep for 3-D FDTD grid configurations.

Prints a table of candidate grid spacings showing the achievable frequency
resolution, memory footprint, and step count for a given domain and FD order.

Usage::

    python examples/estimate_3d_grid.py
    python examples/estimate_3d_grid.py --dx 0.25 0.3 0.4 0.5
    python examples/estimate_3d_grid.py --z-max 80 --order 4

Expected output: a table of grid configurations to guide the choice of dx
for the 3-D valley FDTD run.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from acoustic_sim.fdtd import fd2_coefficients, fd2_cfl_factor


def estimate(
    dx_values: list[float],
    x_range: float,
    y_range: float,
    z_range: float,
    c: float,
    fd_order: int,
    total_time: float,
    ppw: int,
) -> None:
    """Print a table of grid estimates for each dx."""
    coeffs = fd2_coefficients(fd_order)
    spec_radius = fd2_cfl_factor(coeffs)

    header = (
        f"{'dx':>6s}  {'grid (nz,ny,nx)':>20s}  {'cells':>8s}  {'mem(GB)':>8s}  "
        f"{'f_max(Hz)':>9s}  {'dt(s)':>10s}  {'steps':>7s}"
    )
    print(header)
    print("-" * len(header))

    for dx in sorted(dx_values, reverse=True):
        nx = int(x_range / dx) + 1
        ny = int(y_range / dx) + 1
        nz = int(z_range / dx) + 1
        cells = nx * ny * nz

        # 2 pressure fields + C2 + sigma = 4 float64 arrays
        mem_gb = cells * 8 * 4 / 1e9

        f_max = c / (ppw * dx)

        # CFL limit for 3-D: dt <= 2*dx / (c * sqrt(3 * spec_radius))
        cfl_safety = 0.9
        dt = cfl_safety * 2.0 * dx / (c * math.sqrt(3.0 * spec_radius))
        n_steps = math.ceil(total_time / dt)

        grid_str = f"({nz},{ny},{nx})"
        print(
            f"{dx:6.2f}  {grid_str:>20s}  {cells/1e6:7.1f}M  {mem_gb:8.2f}  "
            f"{f_max:9.0f}  {dt:10.2e}  {n_steps:7d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dx",
        type=float,
        nargs="+",
        default=[0.25, 0.3, 0.4, 0.5, 1.0],
        help="Grid spacings to evaluate (default: 0.25 0.3 0.4 0.5 1.0)",
    )
    parser.add_argument("--x-min", type=float, default=-50.0)
    parser.add_argument("--x-max", type=float, default=50.0)
    parser.add_argument("--y-min", type=float, default=-50.0)
    parser.add_argument("--y-max", type=float, default=50.0)
    parser.add_argument("--z-min", type=float, default=-5.0)
    parser.add_argument("--z-max", type=float, default=50.0)
    parser.add_argument("--velocity", type=float, default=343.0, help="Sound speed (m/s)")
    parser.add_argument("--order", type=int, default=8, help="FD order (default: 8)")
    parser.add_argument("--total-time", type=float, default=1.0, help="Simulation time (s)")
    parser.add_argument(
        "--ppw",
        type=int,
        default=5,
        help="Points per wavelength for f_max estimate (default: 5)",
    )

    args = parser.parse_args()

    x_range = args.x_max - args.x_min
    y_range = args.y_max - args.y_min
    z_range = args.z_max - args.z_min

    print(f"Domain: {x_range:.0f} x {y_range:.0f} x {z_range:.0f} m")
    print(f"FD order: {args.order},  PPW criterion: {args.ppw}")
    print(f"Total time: {args.total_time} s,  c = {args.velocity} m/s")
    print()

    estimate(
        args.dx,
        x_range,
        y_range,
        z_range,
        args.velocity,
        args.order,
        args.total_time,
        args.ppw,
    )

    sys.exit(0)


if __name__ == "__main__":
    main()

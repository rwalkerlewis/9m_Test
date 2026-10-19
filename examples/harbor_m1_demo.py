#!/usr/bin/env python3
"""Harbor M0/M1 demo: environment map, one FDTD source run, DAS gather and wavefield snapshots.

Usage::

    python examples/harbor_m1_demo.py [--out output/harbor] [--time 12] [--shore-loss 0.15]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.water import (  # noqa: E402
    HarborFDTDSolver,
    HarborScenarioConfig,
    build_environment,
    build_layout,
)
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402
from acoustic_sim.sources import StaticSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="output/harbor")
    ap.add_argument("--time", type=float, default=12.0, help="simulated seconds")
    ap.add_argument("--shore-loss", type=float, default=0.15, help="shore_loss_max (0 = perfectly reflecting quays)")
    ap.add_argument("--source", type=float, nargs=2, default=[900.0, 900.0], metavar=("X", "Y"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = HarborScenarioConfig()
    cfg.fdtd.total_time = args.time
    cfg.fdtd.shore_loss_max = args.shore_loss
    cfg.save(out / "scenario_used.json")

    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    fs = cfg.fdtd.fs
    t = np.arange(cfg.fdtd.n_steps) / fs

    # Ricker pulse (peak 40 Hz) followed by silence.
    a = (np.pi * 40.0 * (t - 0.1)) ** 2
    signal = (1 - 2 * a) * np.exp(-a)

    src = StaticSource(args.source[0], args.source[1], signal)
    solver = HarborFDTDSolver(env, cfg.fdtd, src, layout.receivers)
    snap_times = [0.3, 0.6, 1.0, 1.6, 2.4, 4.0]
    snap_steps = [int(s * fs) for s in snap_times if s < args.time]

    t0 = time.time()
    res, fields = solver.run_with_snapshots(snap_steps, verbose=True)
    print(f"FDTD: {solver.n_steps} steps, {time.time() - t0:.1f} s")
    traces = res["traces"]
    parts = layout.split(traces)

    # 1. Map
    fig, ax = plt.subplots(figsize=(9, 7))
    plot_harbor(env, layout, ax=ax, sponge_cells=cfg.fdtd.damping_width)
    ax.plot(*args.source, "r*", ms=14, label="source")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out / "m1_map.png", dpi=100, bbox_inches="tight")

    # 2. Wavefield snapshots
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    x0, x1, y0, y1 = env.extent
    for ax, n in zip(axes.ravel(), snap_steps):
        f = np.where(env.water, fields[n], np.nan)
        v = np.nanmax(np.abs(f)) * 0.5
        ax.imshow(f, origin="lower", extent=(x0, x1, y0, y1), cmap="seismic", vmin=-v, vmax=v)
        ax.imshow(np.where(env.land, 1.0, np.nan), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys", vmin=0, vmax=2)
        ax.plot(*layout.cables[0].positions.T, "-", color="orange", lw=1)
        ax.plot(*args.source, "k*")
        ax.set_title(f"t = {n / fs:.2f} s")
        ax.set_aspect("equal")
    fig.suptitle(f"Pressure snapshots (shore_loss_max = {args.shore_loss})")
    fig.savefig(out / "m1_snapshots.png", dpi=80, bbox_inches="tight")

    # 3. DAS waterfall + hydrophone traces
    das = parts["cable1"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    v = np.abs(das).max() * 0.3
    a1.imshow(das, aspect="auto", origin="lower", cmap="seismic", vmin=-v, vmax=v,
              extent=(0, t[-1], 0, layout.cables[0].length))
    a1.set_xlabel("time [s]")
    a1.set_ylabel("distance along cable [m]")
    a1.set_title("DAS channels (pressure at fiber nodes)")
    hyd = parts["hydrophones"]
    for i, h in enumerate(hyd):
        a2.plot(t, h / np.abs(hyd).max() + 1.5 * i, lw=0.7)
    a2.set_xlabel("time [s]")
    a2.set_yticks([])
    a2.set_title("hydrophones")
    fig.savefig(out / "m1_gather.png", dpi=100, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Harbor M0/M1 demo: environment map, one FDTD source run, DAS gather and wavefield snapshots.

Figures, one panel each, in ``--out`` (default ``output/studies/history/harbor_m1_demo/``): ``m1_map.png``,
``m1_snapshot_<t>s.png`` (one per snapshot time), ``m1_das.png``, ``m1_hydrophones.png``.

Usage::

    python examples/harbor/02_forward_model/harbor_m1_demo.py [--out output/studies/history/harbor_m1_demo] [--time 12] [--shore-loss 0.15]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from acoustic_sim.water import (  # noqa: E402
    HarborFDTDSolver,
    HarborScenarioConfig,
    build_environment,
    build_layout,
)
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402

NAME = "default 5 m harbor scenario"
from acoustic_sim.sources import StaticSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="output/studies/history/harbor_m1_demo")
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
    fig, ax = new_figure((13.0, 10.0))
    plot_harbor(env, layout, ax=ax, sponge_cells=cfg.fdtd.damping_width)
    ax.plot(*args.source, "r*", ms=14, label="source")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=11)
    ax.set_title(f"{NAME}: chart, sensors and the pulse source")
    save_figure(fig, out / "m1_map.png")

    # 2. Wavefield snapshots, one figure each
    x0, x1, y0, y1 = env.extent
    for n in snap_steps:
        f = np.where(env.water, fields[n], np.nan)
        v = np.nanmax(np.abs(f)) * 0.5
        fig, ax = new_figure((13.0, 10.0))
        im = ax.imshow(f, origin="lower", extent=(x0, x1, y0, y1), cmap="seismic", vmin=-v, vmax=v)
        ax.imshow(np.where(env.land, 1.0, np.nan), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys", vmin=0, vmax=2)
        ax.plot(*layout.cables[0].positions.T, "-", color="orange", lw=1.5, label="DAS cable")
        ax.plot(*args.source, "k*", ms=12, label="source")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(loc="upper right", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8, label="pressure [Pa]")
        ax.set_title(f"{NAME}: pressure at t = {n / fs:.2f} s (shore_loss_max = {args.shore_loss})")
        save_figure(fig, out / f"m1_snapshot_{n / fs:.2f}s.png")

    # 3. DAS waterfall and hydrophone traces, one figure each
    das = parts["cable1"]
    fig, ax = new_figure((12.0, 8.0))
    v = np.abs(das).max() * 0.3
    im = ax.imshow(das, aspect="auto", origin="lower", cmap="seismic", vmin=-v, vmax=v,
                   extent=(0, t[-1], 0, layout.cables[0].length))
    fig.colorbar(im, ax=ax, label="pressure [Pa]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance along cable [m]")
    ax.set_title(f"{NAME}: DAS channels (pressure at the fibre nodes)")
    save_figure(fig, out / "m1_das.png")
    hyd = parts["hydrophones"]
    fig, ax = new_figure((12.0, 8.0))
    for i, h in enumerate(hyd):
        ax.plot(t, h / np.abs(hyd).max() + 1.5 * i, lw=0.7, label=f"hydrophone {i}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("pressure over the largest, offset by 1.5 per hydrophone [ratio]")
    ax.legend(fontsize=11)
    ax.set_title(f"{NAME}: hydrophone traces")
    save_figure(fig, out / "m1_hydrophones.png")
    print("wrote", out)


if __name__ == "__main__":
    main()

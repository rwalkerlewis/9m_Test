#!/usr/bin/env python3
"""Harbor M1 variant: compare hard-body vs. soft-body scattering from a static reflector.

Runs the FDTD three times for the same geometry:
  1. Without any scattering body (reference).
  2. With a hard body at (1200, 600) — pressure-release boundary, maximum reflection.
  3. With a soft body at the same position — also pressure-release (same physics, labelled
     "soft" to match the Scatterer API distinction).

Differences relative to the reference run give the isolated scattered fields:
  scattered_hard = with_hard - without
  scattered_soft = with_soft - without

A 3-panel figure is saved:
Figures, one panel each, in ``--out`` (default ``output/studies/history/harbor_m1_scatterer_compare/``):
  ``m1_scatterer_hard.png``: hard scattered DAS gather (seismic colormap, waterfall);
  ``m1_scatterer_soft.png``: soft scattered DAS gather;
  ``m1_scatterer_ratio.png``: ratio in dB per channel per time sample, 20*log10(|hard| / (|soft| + eps)).

Usage::

    python examples/harbor/02_forward_model/harbor_m1_scatterer_compare.py \\
        [--out output/studies/history/harbor_m1_scatterer_compare] [--time 5] [--source 900 900] \\
        [--body 1200 600] [--body-size 10 5] [--shore-loss 0.15]
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
from acoustic_sim.water.scatterer import Scatterer  # noqa: E402
from acoustic_sim.water.vessels import Track  # noqa: E402
from acoustic_sim.sources import StaticSource  # noqa: E402

F0, T_PULSE = 40.0, 0.1      # Ricker peak frequency [Hz] and emission time [s]
SRC_C, BODY_C = "#dc2626", "#f97316"


def static_body(xy: list[float], length: float, width: float, total_time: float, kind: str) -> Scatterer:
    """A stationary scattering body at *xy* (heading +x), with the given *kind*."""
    t = np.array([0.0, total_time])
    p = np.array([xy, xy], dtype=float)
    track = Track(
        id="body", kind="uuv", ais=False,
        t=t, xy=p, speed=np.zeros(2), active=np.ones(2, bool), level_pa=0.0,
    )
    return Scatterer("body", track, length, width, kind=kind, shape="rect")


def run_fdtd(env, cfg, src, receivers, label: str, scatterers: tuple = ()) -> np.ndarray:
    """Run one FDTD job and return the full receiver trace array."""
    solver = HarborFDTDSolver(env, cfg.fdtd, src, receivers, scatterers=scatterers)
    t0 = time.time()
    res = solver.run()
    elapsed = time.time() - t0
    body_desc = f"body={scatterers[0].kind}" if scatterers else "no body"
    print(f"FDTD ({label}, {body_desc}): {solver.n_steps} steps, {elapsed:.1f} s")
    return res["traces"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/studies/history/harbor_m1_scatterer_compare")
    ap.add_argument("--time", type=float, default=5.0, help="simulated seconds")
    ap.add_argument("--shore-loss", type=float, default=0.15)
    ap.add_argument("--source", type=float, nargs=2, default=[900.0, 900.0], metavar=("X", "Y"))
    ap.add_argument("--body", type=float, nargs=2, default=[1200.0, 600.0], metavar=("X", "Y"),
                    help="scatterer body position [m]")
    ap.add_argument("--body-size", type=float, nargs=2, default=[10.0, 5.0], metavar=("LENGTH", "WIDTH"),
                    help="body dimensions [m] (should span ≥ 2 grid cells each)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = HarborScenarioConfig()
    cfg.fdtd.total_time = args.time
    cfg.fdtd.shore_loss_max = args.shore_loss
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    fs = cfg.fdtd.fs
    t = np.arange(cfg.fdtd.n_steps) / fs

    # Ricker pulse source
    a = (np.pi * F0 * (t - T_PULSE)) ** 2
    signal = (1 - 2 * a) * np.exp(-a)
    src = StaticSource(args.source[0], args.source[1], signal)

    hard_body = static_body(args.body, *args.body_size, args.time, kind="hard")
    soft_body = static_body(args.body, *args.body_size, args.time, kind="soft")

    # Three FDTD runs
    traces_without  = layout.split(run_fdtd(env, cfg, src, layout.receivers, "reference"))["cable1"]
    traces_hard     = layout.split(run_fdtd(env, cfg, src, layout.receivers, "hard", (hard_body,)))["cable1"]
    traces_soft     = layout.split(run_fdtd(env, cfg, src, layout.receivers, "soft", (soft_body,)))["cable1"]

    scattered_hard = traces_hard - traces_without
    scattered_soft = traces_soft - traces_without

    # -------------------------------------------------------------------------
    # Figure: 3-panel comparison
    # -------------------------------------------------------------------------
    cab = layout.cables[0]
    s_along = cab.arclength
    ext = (0.0, t[-1], s_along[0], s_along[-1])

    # Shared clip level: 80th percentile of the larger scattered field
    v = max(np.percentile(np.abs(scattered_hard), 98), np.percentile(np.abs(scattered_soft), 98)) * 0.9
    if v < 1e-15:
        v = 1.0   # fallback if both are exactly zero (shouldn't happen)

    # Ratio in dB per channel × time
    eps = np.abs(scattered_soft).max() * 1e-6 + 1e-30
    ratio_db = 20.0 * np.log10(np.abs(scattered_hard) / (np.abs(scattered_soft) + eps))

    S, B = np.array(args.source), np.array(args.body)
    c = float(np.mean(env.model.values[env.water]))
    name = "default 5 m harbor scenario"
    setup = (f"source ({S[0]:.0f}, {S[1]:.0f}) m, body ({B[0]:.0f}, {B[1]:.0f}) m {args.body_size[0]:.0f} x {args.body_size[1]:.0f} m; "
             f"dx {env.dx:.0f} m, fs {fs:.0f} Hz, c {c:.0f} m/s")
    vr = max(np.percentile(np.abs(ratio_db), 95), 1.0)
    for file, data, cmap, lim, label, what in (
            ("m1_scatterer_hard.png", scattered_hard, "seismic", v, "pressure [Pa]", "hard body, scattered field"),
            ("m1_scatterer_soft.png", scattered_soft, "seismic", v, "pressure [Pa]", "soft body, scattered field"),
            ("m1_scatterer_ratio.png", ratio_db, "RdBu_r", vr, "20 log10(|hard| / |soft|) [dB]", "hard over soft scattered field")):
        fig, ax = new_figure((12.0, 8.0))
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=-lim, vmax=lim, extent=ext, interpolation="nearest")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("distance along cable [m]")
        ax.set_title(f"{name}: {what}\n{setup}")
        save_figure(fig, out / file)
        print("wrote", out / file)


if __name__ == "__main__":
    main()

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
  Panel 1: Hard scattered DAS gather (seismic colormap, waterfall)
  Panel 2: Soft scattered DAS gather
  Panel 3: Ratio in dB per channel per time sample: 20*log10(|hard| / (|soft| + eps))

Usage::

    python examples/harbor/02_forward_model/harbor_m1_scatterer_compare.py \\
        [--out output/harbor] [--time 5] [--source 900 900] \\
        [--body 1200 600] [--body-size 10 5] [--shore-loss 0.15]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from acoustic_sim.water import (  # noqa: E402
    HarborFDTDSolver,
    HarborScenarioConfig,
    build_environment,
    build_layout,
)
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402
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
    ap.add_argument("--out", default="output/harbor")
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

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)

    # Panel 1 — hard scattered gather
    im1 = axes[0].imshow(
        scattered_hard, aspect="auto", origin="lower", cmap="seismic",
        vmin=-v, vmax=v, extent=ext, interpolation="nearest",
    )
    axes[0].set_title("Hard body — scattered field", loc="left", fontweight="bold")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("distance along cable [m]")
    plt.colorbar(im1, ax=axes[0], shrink=0.85, label="pressure [Pa]")

    # Panel 2 — soft scattered gather
    im2 = axes[1].imshow(
        scattered_soft, aspect="auto", origin="lower", cmap="seismic",
        vmin=-v, vmax=v, extent=ext, interpolation="nearest",
    )
    axes[1].set_title("Soft body — scattered field", loc="left", fontweight="bold")
    axes[1].set_xlabel("time [s]")
    plt.colorbar(im2, ax=axes[1], shrink=0.85, label="pressure [Pa]")

    # Panel 3 — ratio in dB
    vr = max(np.percentile(np.abs(ratio_db), 95), 1.0)
    im3 = axes[2].imshow(
        ratio_db, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-vr, vmax=vr, extent=ext, interpolation="nearest",
    )
    axes[2].set_title("Ratio: 20·log₁₀(|hard|/|soft|) [dB]", loc="left", fontweight="bold")
    axes[2].set_xlabel("time [s]")
    plt.colorbar(im3, ax=axes[2], shrink=0.85, label="dB")

    S, B = np.array(args.source), np.array(args.body)
    c = float(np.mean(env.model.values[env.water]))
    fig.text(
        0.995, 0.005,
        f"FDTD dx={env.dx:.0f} m, fs={fs:.0f} Hz, shore_loss={args.shore_loss}; "
        f"source ({S[0]:.0f},{S[1]:.0f}) m; body ({B[0]:.0f},{B[1]:.0f}) m "
        f"{args.body_size[0]:.0f}×{args.body_size[1]:.0f} m; c={c:.0f} m/s",
        ha="right", va="bottom", fontsize=8, color="#6b7280",
    )
    fig.savefig(out / "m1_scatterer_compare.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "m1_scatterer_compare.png")


if __name__ == "__main__":
    main()

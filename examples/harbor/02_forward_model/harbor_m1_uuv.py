#!/usr/bin/env python3
"""Harbor M1 variant: one static source, a single pulse, and a static UUV in the water.

Built from ``examples/harbor/02_forward_model/harbor_m1_demo.py``: same environment, same DAS cable, same 40 Hz Ricker pulse from one
static source.  A stationary UUV (a soft, pressure-release body of ``--uuv-size`` metres) is added as an FDTD
scatterer.  The FDTD is run twice, with and without the body; the difference is the field scattered by the UUV,
which is what identifies its reflection in the gather.

Outputs (``--out``, default ``output/studies/history/harbor_m1_uuv/``), one panel each:

* ``m1uuv_map.png``            the scene: land, cable, source, UUV;
* ``m1uuv_gather.png``         the DAS gather recorded with the UUV present;
* ``m1uuv_gather_identified.png``  the same gather with the UUV reflection marked (predicted bistatic arrival
                               source -> UUV -> channel);
* ``m1uuv_scattered.png``      the scattered field alone (run with UUV minus run without).

Usage::

    python examples/harbor/02_forward_model/harbor_m1_uuv.py [--out output/studies/history/harbor_m1_uuv] [--time 5] [--source 900 900]
        [--uuv 1560 980] [--uuv-size 12 6] [--shore-loss 0.15]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from acoustic_sim.water import (  # noqa: E402
    HarborFDTDSolver,
    HarborScenarioConfig,
    build_environment,
    build_layout,
)
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402
from acoustic_sim.water.scatterer import Scatterer  # noqa: E402
from acoustic_sim.water.vessels import Track  # noqa: E402
from acoustic_sim.sources import StaticSource  # noqa: E402

F0, T_PULSE = 40.0, 0.1                     # Ricker peak frequency [Hz] and emission time [s]
SRC_C, UUV_C, CABLE_C = "#dc2626", "#f97316", "#ca8a04"


def static_uuv(xy, length, width, total_time):
    """A stationary soft body at *xy* (heading +x)."""
    t = np.array([0.0, total_time]); p = np.array([xy, xy], dtype=float)
    track = Track(id="uuv", kind="uuv", ais=False, t=t, xy=p, speed=np.zeros(2), active=np.ones(2, bool), level_pa=0.0)   # active: present in the scene
    return Scatterer("uuv", track, length, width, kind="soft", shape="rect")


def run(env, cfg, src, receivers, scatterers=()):
    solver = HarborFDTDSolver(env, cfg.fdtd, src, receivers, scatterers=scatterers)
    t0 = time.time(); res = solver.run(); print(f"FDTD ({'with' if scatterers else 'no'} UUV): {solver.n_steps} steps, {time.time() - t0:.1f} s")
    return res["traces"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/studies/history/harbor_m1_uuv")
    ap.add_argument("--time", type=float, default=5.0, help="simulated seconds")
    ap.add_argument("--shore-loss", type=float, default=0.15, help="shore_loss_max (0 = perfectly reflecting quays)")
    ap.add_argument("--source", type=float, nargs=2, default=[900.0, 900.0], metavar=("X", "Y"))
    ap.add_argument("--uuv", type=float, nargs=2, default=[1560.0, 980.0], metavar=("X", "Y"), help="UUV position (stationary)")
    ap.add_argument("--uuv-size", type=float, nargs=2, default=[12.0, 6.0], metavar=("LENGTH", "WIDTH"),
                    help="soft body size [m]; the 5 m grid cannot represent a real 1 m hull, so this is oversized")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = HarborScenarioConfig()
    cfg.fdtd.total_time = args.time
    cfg.fdtd.shore_loss_max = args.shore_loss
    cfg.save(out / "scenario_used.json")
    env = build_environment(cfg); layout = build_layout(cfg, env)
    fs = cfg.fdtd.fs; t = np.arange(cfg.fdtd.n_steps) / fs
    a = (np.pi * F0 * (t - T_PULSE)) ** 2
    signal = (1 - 2 * a) * np.exp(-a)                                      # one Ricker pulse, then silence
    src = StaticSource(args.source[0], args.source[1], signal)
    uuv = static_uuv(args.uuv, *args.uuv_size, args.time)

    with_uuv = layout.split(run(env, cfg, src, layout.receivers, (uuv,)))["cable1"]
    without = layout.split(run(env, cfg, src, layout.receivers))["cable1"]
    scattered = with_uuv - without                                         # the field of the UUV alone

    cab = layout.cables[0]; rx = cab.positions; s_along = cab.arclength
    c = float(np.mean(env.model.values[env.water]))
    S, U = np.array(args.source), np.array(args.uuv)
    t_direct = T_PULSE + np.linalg.norm(rx - S, axis=1) / c
    t_echo = T_PULSE + (np.linalg.norm(U - S) + np.linalg.norm(rx - U, axis=1)) / c
    env_sc = np.abs(hilbert(scattered, axis=1)); k_pk = env_sc.argmax(axis=1) / fs
    print(f"UUV echo: predicted {t_echo.min():.2f}-{t_echo.max():.2f} s, measured peak {k_pk.min():.2f}-{k_pk.max():.2f} s, "
          f"peak amplitude {np.abs(scattered).max() / np.abs(with_uuv).max():.3f} of the direct arrival")

    ext = (0, t[-1], s_along[0], s_along[-1])
    name = "default 5 m harbor scenario"
    setup = (f"dx {env.dx:.0f} m, fs {fs:.0f} Hz, shore_loss_max {args.shore_loss}; soft rectangular UUV "
             f"{args.uuv_size[0]:.0f} x {args.uuv_size[1]:.0f} m; prediction: straight paths at c = {c:.0f} m/s")

    # 1. Map
    fig, ax = new_figure((13.0, 10.0))
    plot_harbor(env, layout, ax=ax, sponge_cells=cfg.fdtd.damping_width)
    ax.plot(*args.source, "*", color=SRC_C, ms=16, mec="k", label="source (one pulse)")
    ax.plot(*args.uuv, "o", color=UUV_C, ms=10, mec="k", label=f"UUV, stationary ({args.uuv_size[0]:.0f} x {args.uuv_size[1]:.0f} m soft body)")
    ax.plot([S[0], U[0]], [S[1], U[1]], "--", color=UUV_C, lw=1, alpha=0.7, label="source to UUV")
    for k, i in enumerate((0, len(rx) // 2, len(rx) - 1)):
        ax.plot([U[0], rx[i, 0]], [U[1], rx[i, 1]], ":", color=UUV_C, lw=0.8, alpha=0.7, label="UUV to channels" if k == 0 else None)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=11)
    ax.set_title(f"{name}: one source, one pulse, one stationary UUV")
    save_figure(fig, out / "m1uuv_map.png")

    # 2. The gather
    v = np.abs(with_uuv).max() * 0.3
    fig, ax = new_figure((12.0, 8.0))
    im = ax.imshow(with_uuv, aspect="auto", origin="lower", cmap="seismic", vmin=-v, vmax=v, extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="pressure [Pa]")
    ax.set_xlabel("time [s]"); ax.set_ylabel("distance along cable [m]")
    ax.set_title(f"{name}: DAS gather,\none {F0:.0f} Hz pulse from ({S[0]:.0f}, {S[1]:.0f}) m, UUV at ({U[0]:.0f}, {U[1]:.0f}) m")
    save_figure(fig, out / "m1uuv_gather.png")

    # 3. The gather with the UUV reflection identified, and the scattered field alone
    fig, ax = new_figure((12.0, 8.0))
    x0 = np.abs(with_uuv).max() * 0.02                                     # asinh compression: linear below x0, log above
    comp = np.arcsinh(with_uuv / x0); vc = np.abs(comp).max()
    im = ax.imshow(comp, aspect="auto", origin="lower", cmap="seismic", vmin=-vc, vmax=vc, extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=ax, label=f"asinh(pressure / {x0:.2g} Pa)")
    ax.plot(t_direct, s_along, "--", color="k", lw=1.2, alpha=0.6, label="direct arrival (predicted)")
    ax.plot(t_echo, s_along, "--", color=UUV_C, lw=2.2, label="reflection off the UUV (predicted source -> UUV -> channel)")
    j = len(rx) // 3
    ax.annotate("reflection off the UUV", (t_echo[j], s_along[j]), (t_echo[j] + 0.45, s_along[j] + 300), color=UUV_C, fontsize=12,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=UUV_C, lw=2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=UUV_C, alpha=0.95))
    ax.set_xlabel("time [s]"); ax.set_ylabel("distance along cable [m]"); ax.set_xlim(0, min(2.5, t[-1]))
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_title(f"{name}: DAS gather with the UUV, its reflection marked\n{setup}")
    save_figure(fig, out / "m1uuv_gather_identified.png")
    vs = np.abs(scattered).max() * 0.8
    fig, ax = new_figure((12.0, 8.0))
    im = ax.imshow(scattered, aspect="auto", origin="lower", cmap="seismic", vmin=-vs, vmax=vs, extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="pressure [Pa]")
    ax.plot(t_echo, s_along, "--", color=UUV_C, lw=1.5, alpha=0.8, label="reflection off the UUV (predicted)")
    ax.set_xlabel("time [s]"); ax.set_ylabel("distance along cable [m]"); ax.set_xlim(0, min(2.5, t[-1]))
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title(f"{name}: scattered field alone, run with the UUV minus run without\n"
                 f"(colour scale {np.abs(with_uuv).max() / vs:.0f} times finer than the gather)")
    save_figure(fig, out / "m1uuv_scattered.png")
    print("wrote", out)


if __name__ == "__main__":
    main()

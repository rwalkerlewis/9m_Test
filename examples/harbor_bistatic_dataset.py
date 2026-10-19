#!/usr/bin/env python3
"""Bistatic harbor dataset: an AIS ship illuminates a silent UUV body in a cropped, fine grid harbor.

Two cached FDTD runs per radiating vessel (with and without the body); the difference is the
scattered field.  The dataset keeps the direct, scattered and noise components separately so
the processing gain of every later stage can be measured against truth.

Usage::

    python examples/harbor_bistatic_dataset.py [--config examples/harbor_bistatic.scenario.json]
        [--out output/harbor_bistatic] [--coarse] [--time 20] [--workers 2] [--cuda]

``--coarse`` runs the same scenario on a 1 m grid (band to 400 Hz, about 2 min per FDTD run on
two CPU cores) instead of 0.5 m (800 Hz, about 20 min per run); the 1 m by 3 m hull is then one
cell wide, which test A of ``tests/harbor/test_scatterer.py`` shows is a crude body.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.water import (  # noqa: E402
    HarborScenarioConfig,
    assemble_dataset,
    build_environment,
    build_layout,
    build_track,
    render_vessel_gathers,
)
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402
from acoustic_sim.water.signatures import f_max_for_grid  # noqa: E402


def coarsen(cfg: HarborScenarioConfig, dx: float) -> HarborScenarioConfig:
    """Same scenario on grid spacing *dx*: sample rate and noise band scale with the grid."""
    cfg.environment.dx = dx
    cfg.fdtd.fs = 600.0 * 5.0 / dx
    cfg.noise.band_scale = 5.0 / dx
    return cfg


def band_rms(x: np.ndarray, sos: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sosfiltfilt(sos, x, axis=-1) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "harbor_bistatic.scenario.json"))
    ap.add_argument("--out", default="output/harbor_bistatic")
    ap.add_argument("--coarse", action="store_true", help="1 m grid instead of 0.5 m")
    ap.add_argument("--time", type=float, default=None, help="override simulated duration [s]")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cuda", action="store_true", help="run the FDTD on the GPU (the cache key ignores this)")
    args = ap.parse_args()

    cfg = HarborScenarioConfig.load(args.config)
    if args.coarse:
        coarsen(cfg, 1.0)
    if args.time is not None:
        cfg.fdtd.total_time = args.time
    if args.cuda:
        cfg.fdtd.use_cuda = True
    tag = "coarse" if args.coarse else "fine"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "scenario_used.json")

    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    f_max = f_max_for_grid(env.dx)
    print(f"[{tag}] grid {env.nx} x {env.ny} at {env.dx} m, fs {cfg.fdtd.fs:.0f} Hz, band to {f_max:.0f} Hz, "
          f"{layout.n_receivers} channels ({layout.n_samples} sample points), {cfg.fdtd.n_steps} steps")

    t0 = time.time()
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=Path(args.out) / "cache", workers=args.workers)
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.seed, keep_components=True)
    print(f"[{tag}] {len(gathers)} gathers ({', '.join(gathers)}) in {time.time() - t0:.0f} s")
    ds.save(out / "dataset.npz")

    fs = ds.fs
    sos = butter(4, [0.25 * f_max, 0.95 * f_max], btype="bandpass", fs=fs, output="sos")
    live = ~ds.dead
    idx_all = np.arange(layout.n_receivers)
    groups = {name: idx_all[(idx_all >= sl.start) & (idx_all < sl.stop) & live] for name, sl in layout.slices.items()}
    table = {}
    print(f"\n[{tag}] band limited ({0.25 * f_max:.0f} to {0.95 * f_max:.0f} Hz) rms at the sensors, whole run, live channels")
    print(f"  {'component':28s} " + " ".join(f"{n:>12s}" for n in groups) + f" {'best DAS':>12s}")
    for key, comp in ds.components.items():
        gain = ds.gain if key != "noise" else np.ones_like(ds.gain)
        row = [band_rms(comp[g].astype(float) * gain[g, None], sos) for g in groups.values()]
        das = groups[layout.cables[0].name]
        best = max(band_rms(comp[i].astype(float) * gain[i], sos) for i in das) if len(das) else float("nan")
        table[key] = dict(zip(groups, row)) | {"best_das": best}
        print(f"  {key:28s} " + " ".join(f"{v:12.4g}" for v in row) + f" {best:12.4g}")
    d, s = table["direct:ferry"], table["scattered:ferry>uuv1"]
    nz = table["noise"]
    summary = {
        "echo_to_direct_db_median_das": 20 * np.log10(s["cable1"] / d["cable1"]),
        "echo_to_direct_db_best_das": 20 * np.log10(s["best_das"] / d["best_das"]),
        "echo_to_noise_db_median_das": 20 * np.log10(s["cable1"] / nz["cable1"]),
        "echo_to_noise_db_best_das": 20 * np.log10(s["best_das"] / nz["cable1"]),
    }
    print("\n  " + ", ".join(f"{k} {v:+.1f}" for k, v in summary.items()))
    (out / "levels.json").write_text(json.dumps({"table": table, "summary": summary, "f_max": f_max}, indent=2))

    # Map with tracks and the sensors.
    fig, ax = plt.subplots(figsize=(9, 7))
    plot_harbor(env, layout, ax=ax, sponge_cells=cfg.fdtd.damping_width)
    for vid, tr in ds.tracks.items():
        ax.plot(*tr.xy[tr.active].T, "-" if tr.ais else "--", lw=2, label=f"{vid} ({'AIS' if tr.ais else 'silent body'})")
        ax.plot(*tr.xy[0], "o", ms=5, color="k")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Bistatic scenario ({tag}): ship illuminates a {cfg.vessels[1].body_width:g} m x "
                 f"{cfg.vessels[1].body_length:g} m soft body")
    fig.savefig(out / "map.png", dpi=100, bbox_inches="tight")

    # Components on the cable: direct and scattered waterfalls.
    cable = layout.cables[0]
    sl = layout.slices[cable.name]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, key in zip(axes, ("direct:ferry", "scattered:ferry>uuv1", "noise")):
        x = sosfiltfilt(sos, ds.components[key][sl].astype(float), axis=-1)
        v = np.percentile(np.abs(x), 99.5)
        ax.imshow(x, aspect="auto", origin="lower", cmap="seismic", vmin=-v, vmax=v,
                  extent=(0, ds.t[-1], 0, cable.length))
        ax.set_title(f"{key} (rms {np.sqrt(np.mean(x ** 2)):.3g})")
        ax.set_xlabel("time [s]")
    axes[0].set_ylabel("distance along cable [m]")
    fig.savefig(out / "components.png", dpi=100, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

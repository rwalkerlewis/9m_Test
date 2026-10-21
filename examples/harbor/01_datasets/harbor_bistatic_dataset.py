#!/usr/bin/env python3
"""Bistatic harbor dataset: an AIS ship illuminates a silent UUV body in a cropped, fine grid harbor.

Two cached FDTD runs per radiating vessel (with and without the body); the difference is the
scattered field.  The dataset keeps the direct, scattered and noise components separately so
the processing gain of every later stage can be measured against truth.

Usage::

    python examples/harbor/01_datasets/harbor_bistatic_dataset.py [--config examples/configs/harbor_bistatic.scenario.json]
        [--out output/recordings/bistatic] [--coarse] [--figures-only] [--time 20] [--workers 2] [--cuda]

The recording goes to ``--out`` (``--out/coarse/`` with ``--coarse``): ``dataset.npz``, ``levels.json``,
``scenario_used.json``, ``map.png`` and ``components_<component>.png`` (one figure per component on the cable); the FDTD
cache to ``--out/cache`` and ``build.json``.  ``--figures-only`` redraws the figures from the FDTD cache and
``--out/scenario_used.json`` (no FDTD run) and writes nothing else.

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
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figures import component_figures, gathers_from_cache  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402

from acoustic_sim.water.dataset import count_cached_gathers, write_build_info  # noqa: E402
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


def plot_map(cfg, env, layout, tracks, ais_ids, out: Path, rec: str) -> None:
    """The layout and the tracks (one panel)."""
    fig, ax = new_figure((13.0, 10.0))
    plot_harbor(env, layout, ax=ax, sponge_cells=cfg.fdtd.damping_width)
    for (vid, tr), col in zip(tracks.items(), ("#e41a1c", "#984ea3", "#4daf4a", "#a65628")):
        ais = vid in ais_ids
        ax.plot(*tr.xy[tr.active].T, "-" if ais else "--", lw=2.5, color=col, label=f"{vid} ({'AIS' if ais else 'silent body'})")
    starts = np.array([tr.xy[0] for tr in tracks.values()])
    ax.plot(starts[:, 0], starts[:, 1], "o", ms=5, color="k", label="track start")
    ax.plot([], [], ".", color="red", ms=6, label="every 10th cable channel")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=11)
    body = next(v for v in cfg.vessels if v.is_body)
    ax.set_title(f"{rec}: scenario map, a ship illuminates a {body.body_width:g} m x {body.body_length:g} m soft body")
    save_figure(fig, out / "map.png", recording=rec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "harbor_bistatic.scenario.json"))
    ap.add_argument("--out", default="output/recordings/bistatic")
    ap.add_argument("--coarse", action="store_true", help="1 m grid instead of 0.5 m (written to --out/coarse)")
    ap.add_argument("--time", type=float, default=None, help="override simulated duration [s]")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cuda", action="store_true", help="kept for old command lines: the FDTD runs on the GPU by default")
    ap.add_argument("--cpu", action="store_true", help="run the FDTD on the CPU (the cache key ignores this)")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from the FDTD cache and --out/scenario_used.json (no FDTD; writes only figures)")
    args = ap.parse_args()

    root = Path(args.out)
    out = root / "coarse" if args.coarse else root
    rec = root.name + ("_coarse" if args.coarse else "")
    out.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        cfg = HarborScenarioConfig.load(out / "scenario_used.json")
    else:
        cfg = HarborScenarioConfig.load(args.config)
        if args.coarse:
            coarsen(cfg, 1.0)
        if args.time is not None:
            cfg.fdtd.total_time = args.time
        cfg.fdtd.use_cuda = not args.cpu
        cfg.save(out / "scenario_used.json")

    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    f_max = f_max_for_grid(env.dx)
    print(f"[{rec}] grid {env.nx} x {env.ny} at {env.dx} m, fs {cfg.fdtd.fs:.0f} Hz, band to {f_max:.0f} Hz, "
          f"{layout.n_receivers} channels ({layout.n_samples} sample points), {cfg.fdtd.n_steps} steps")

    t0 = time.time()
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    cache = root / "cache"
    keys = [("direct:ferry", "direct_ferry"), ("scattered:ferry>uuv1", "scattered_ferry"), ("noise", "noise")]
    ais_ids = {v.id for v in cfg.vessels if v.has_ais and v.level_pa > 0}
    if args.figures_only:
        gathers = gathers_from_cache(cfg, env, layout, tracks, cache)
        ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.seed, keep_components=True)
        sos = butter(4, [0.25 * f_max, 0.95 * f_max], btype="bandpass", fs=ds.fs, output="sos")
        plot_map(cfg, env, layout, tracks, ais_ids, out, rec)
        component_figures(layout, ds, sos, f_max, keys, out, rec)
        print(f"wrote the figures of {out}")
        return
    n_cached = count_cached_gathers(cache)
    t_fdtd = time.time()
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=cache, workers=args.workers)
    write_build_info(root / "build.json", cfg, env, len(gathers), time.time() - t_fdtd, count_cached_gathers(cache) - n_cached)
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.seed, keep_components=True)
    print(f"[{rec}] {len(gathers)} gathers ({', '.join(gathers)}) in {time.time() - t0:.0f} s")
    ds.save(out / "dataset.npz")
    fs = ds.fs
    sos = butter(4, [0.25 * f_max, 0.95 * f_max], btype="bandpass", fs=fs, output="sos")
    live = ~ds.dead
    idx_all = np.arange(layout.n_receivers)
    groups = {name: idx_all[(idx_all >= sl.start) & (idx_all < sl.stop) & live] for name, sl in layout.slices.items()}
    table = {}
    print(f"\n[{rec}] band limited ({0.25 * f_max:.0f} to {0.95 * f_max:.0f} Hz) rms at the sensors, whole run, live channels")
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

    plot_map(cfg, env, layout, tracks, ais_ids, out, rec)
    component_figures(layout, ds, sos, f_max, keys, out, rec)
    print("wrote", out)


if __name__ == "__main__":
    main()

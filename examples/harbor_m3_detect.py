#!/usr/bin/env python3
"""Harbor M3: detection and localisation of the hidden UUV.

For each sensor configuration (fused DAS + hydrophones, DAS only, hydrophones only):

1. calibrate a CFAR threshold on noise-only mixes (distribution of the per-frame map maximum),
2. sweep the UUV source level (cheap: the cached unit gathers are re-scaled),
3. report Pd, localisation RMSE and false alarms per frame.

Also runs the full scenario (ferry + tug + moored + UUV) without any AIS-aware processing to
show how much the loud known ships hide the UUV.  Everything reads the FDTD cache written by
``examples/harbor_m2_dataset.py``.

Usage::

    python examples/harbor_m3_detect.py [--out output/harbor] [--cal-seeds 4] [--pfa 0.01]
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.harbor import (  # noqa: E402
    HarborScenarioConfig,
    assemble_dataset,
    build_environment,
    build_layout,
    build_track,
    render_vessel_gathers,
)
from acoustic_sim.harbor.detection import (  # noqa: E402
    ProcessingConfig,
    build_grid,
    run_detection,
    threshold_from_maxima,
)
from acoustic_sim.harbor.evaluate import score_detections  # noqa: E402
from acoustic_sim.harbor.plotting import plot_harbor  # noqa: E402

CONFIGS = {"fused": None, "das": ("cable1",), "hydrophones": ("hydrophones",)}
LEVEL_FACTORS = [0.25, 0.5, 1.0, 2.0]
UUV_ONLY = {"ferry": 0.0, "tug": 0.0, "moored": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).parent / "harbor.scenario.json"))
    ap.add_argument("--out", default="output/harbor")
    ap.add_argument("--cal-seeds", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--pfa", type=float, default=0.01, help="per-frame false-alarm probability")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    args = ap.parse_args()

    out = Path(args.out)
    cfg = HarborScenarioConfig.load(args.config)
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=out / "cache")
    uuv_id = next(v.id for v in cfg.vessels if v.kind == "uuv")
    uuv_level = next(v.level_pa for v in cfg.vessels if v.kind == "uuv")
    zero = {v.id: 0.0 for v in cfg.vessels}

    base = ProcessingConfig()
    grid = build_grid(env, base)
    print(f"scan grid: {len(grid)} cells, {layout.n_receivers} receivers")

    results: dict = {"pfa": args.pfa, "uuv_level_pa": uuv_level, "configs": {}}
    runs: dict = {}
    for name in args.configs:
        pc = ProcessingConfig(sensor_groups=CONFIGS[name])
        t0 = time.time()
        maxima = []
        for s in range(args.cal_seeds):
            ds = assemble_dataset(cfg, layout, tracks, gathers, seed=1000 + s, levels=zero, keep_clean=False)
            run = run_detection(ds.traces, layout, ds.fs, env, pc, floor=1e18, grid=grid)
            maxima.append(run.scan.frame_max)
        maxima = np.concatenate(maxima)
        thr = threshold_from_maxima(maxima, args.pfa)
        print(f"[{name}] CFAR threshold {thr:.1f} (noise maxima median {np.median(maxima):.1f}, "
              f"max {maxima.max():.1f}, {time.time() - t0:.0f} s)", flush=True)
        entry = {"threshold": thr, "noise_max_median": float(np.median(maxima)), "noise_max_max": float(maxima.max()), "levels": {}}

        for fac in LEVEL_FACTORS:
            level = uuv_level * fac
            ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.eval_seed,
                                  levels={**UUV_ONLY, uuv_id: level}, keep_clean=False)
            run = run_detection(ds.traces, layout, ds.fs, env, pc, floor=0.5 * thr, grid=grid)
            sc = score_detections(run.dets, ds.tracks, layout, run.wh.times, thr)
            db = 20 * np.log10(level)
            entry["levels"][f"{fac}"] = {
                "level_db_re_1pa_100m": db, "pd": sc.pd(uuv_id), "rmse_m": sc.rmse(uuv_id),
                "fa_per_frame": sc.false_alarms_per_frame,
                "pd_by_range": {f"{a}-{b}": sc.pd(uuv_id, (a, b)) for a, b in [(0, 150), (150, 300), (300, 600), (600, 1e4)]},
            }
            print(f"[{name}] UUV {db:6.1f} dB: Pd {sc.pd(uuv_id):.2f}  RMSE {sc.rmse(uuv_id):6.1f} m  "
                  f"FA/frame {sc.false_alarms_per_frame:.3f}", flush=True)
            runs[(name, fac)] = (ds, run, sc, thr)
        results["configs"][name] = entry

    # Full scenario without AIS-aware processing (fused only)
    if "fused" in args.configs:
        pc = ProcessingConfig()
        thr = results["configs"]["fused"]["threshold"]
        ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.eval_seed, keep_clean=False)
        run = run_detection(ds.traces, layout, ds.fs, env, pc, floor=0.5 * thr, grid=grid)
        sc = score_detections(run.dets, ds.tracks, layout, run.wh.times, thr, targets=list(tracks))
        results["all_vessels_fused"] = {
            vid: {"pd": sc.pd(vid), "rmse_m": sc.rmse(vid)} for vid in tracks
        } | {"fa_per_frame": sc.false_alarms_per_frame}
        print("[all vessels, no suppression] " + ", ".join(
            f"{vid}: Pd {sc.pd(vid):.2f}" if tracks[vid].level_pa > 0 and (tracks[vid].speed > 0.1).any() or vid == uuv_id
            else f"{vid}: n/a" for vid in tracks), f" FA/frame {sc.false_alarms_per_frame:.3f}", flush=True)
        runs[("all", 1.0)] = (ds, run, sc, thr)

    (out / "m3_results.json").write_text(json.dumps(results, indent=2, default=float))

    # ---- figures ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, entry in results["configs"].items():
        db = [v["level_db_re_1pa_100m"] for v in entry["levels"].values()]
        ax[0].plot(db, [v["pd"] for v in entry["levels"].values()], "o-", label=name)
        ax[1].plot(db, [v["rmse_m"] for v in entry["levels"].values()], "o-", label=name)
    ax[0].set(xlabel="UUV source level [dB re 1 Pa @ 100 m]", ylabel="Pd (per frame)", title=f"Detection (Pfa/frame = {args.pfa})", ylim=(0, 1.02))
    ax[1].set(xlabel="UUV source level [dB re 1 Pa @ 100 m]", ylabel="RMSE [m]", title="Localisation error of detections")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.savefig(out / "m3_pd_vs_level.png", dpi=100, bbox_inches="tight")

    for key, fname, title in [(("fused", 1.0), "m3_uuv_only.png", "UUV only (fused sensors)"),
                              (("all", 1.0), "m3_all_vessels.png", "All vessels, no AIS suppression")]:
        if key not in runs:
            continue
        ds, run, sc, thr = runs[key]
        fig, ax = plt.subplots(figsize=(10, 7.5))
        plot_harbor(env, layout, ax=ax)
        for vid, tr in ds.tracks.items():
            if tr.level_pa > 0:
                ax.plot(*tr.xy[tr.active].T, "-" if tr.ais else "--", lw=1.5, label=f"truth {vid}")
        d = run.dets.above(thr)
        sc_ = ax.scatter(*d.xy.T, c=d.t, cmap="viridis", s=18 + 40 * (d.rank == 0), edgecolor="k", linewidths=0.3, zorder=5)
        plt.colorbar(sc_, ax=ax, label="detection time [s]", shrink=0.6, location="left")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        fig.savefig(out / fname, dpi=100, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

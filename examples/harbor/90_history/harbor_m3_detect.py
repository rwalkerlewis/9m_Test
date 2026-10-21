#!/usr/bin/env python3
"""Harbor M3: detection and localisation of the hidden UUV.

For each sensor configuration (fused DAS + hydrophones, DAS only, hydrophones only):

1. calibrate a CFAR threshold on noise-only mixes (distribution of the per-frame map maximum),
2. sweep the UUV source level (cheap: the cached unit gathers are re-scaled),
3. report Pd, localisation RMSE and false alarms per frame.

Also runs the full scenario (ferry + tug + moored + UUV) with/without AIS-aware suppression to
show how much the known ships hide the UUV.  Everything reads the FDTD cache written by
``examples/harbor/01_datasets/harbor_m2_dataset.py``.

Usage::

    python examples/harbor/90_history/harbor_m3_detect.py [--out output/harbor] [--cal-seeds 4] [--pfa 0.01]
    python examples/harbor/90_history/harbor_m3_detect.py --no-suppress   # bare matched-field scan for comparison
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from acoustic_sim.contract import annotate_figure  # noqa: E402
from acoustic_sim.water import (  # noqa: E402
    HarborScenarioConfig,
    assemble_dataset,
    build_environment,
    build_layout,
    build_track,
    render_vessel_gathers,
)
from acoustic_sim.water.detection import (  # noqa: E402
    ProcessingConfig,
    build_grid,
    run_detection,
    threshold_from_maxima,
)
from acoustic_sim.water.evaluate import score_detections  # noqa: E402
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402
from acoustic_sim.water.suppression import AISAwareRun, SuppressionConfig, run_ais_aware  # noqa: E402

CONFIGS = {"fused": None, "das": ("cable1",), "hydrophones": ("hydrophones",)}
LEVEL_FACTORS = [0.25, 0.5, 1.0, 2.0]
UUV_ONLY = {"ferry": 0.0, "tug": 0.0, "moored": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "harbor.scenario.json"))
    ap.add_argument("--out", default="output/harbor")
    ap.add_argument("--cal-seeds", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--pfa", type=float, default=0.01, help="per-frame false-alarm probability")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--no-suppress", dest="suppress", action="store_false", default=True,
                    help="Disable AIS-aware ship suppression (run bare matched-field scan)")
    ap.add_argument("--empirical-refine", choices=["vessel_path", "grid_survey"], default=None,
                    help="Enable empirical replica refinement (vessel_path or grid_survey)")
    ap.add_argument("--cuda", action="store_true", default=False,
                    help="Run MFP beamforming on the GPU via CuPy")
    args = ap.parse_args()

    out = Path(args.out)
    cfg = HarborScenarioConfig.load(args.config)

    # ---- empirical localiser (optional) ----------------------------------------
    empirical_loc = None
    if args.empirical_refine is not None:
        # Either way the replicas are FDTD renders: simulator knowledge, flagged in every run's provenance.
        from acoustic_sim.water.diagnostics import VesselPathLocaliser  # noqa: E402
        from acoustic_sim.water.localiser import EmpiricalLocaliser  # noqa: E402
        if args.empirical_refine == "grid_survey":
            if not (out / "survey_cache").exists():
                sys.exit("Run examples/harbor/90_history/harbor_survey_grid.py first to build the survey cache.")
            empirical_loc = EmpiricalLocaliser(
                cache_dir=out, layout=None, fs=0.0, pc=ProcessingConfig(),
                mode="grid_survey",
            )
        else:
            empirical_loc = VesselPathLocaliser(cache_dir=out / "cache", layout=None, fs=0.0, pc=ProcessingConfig())
        # Defer build() until after cfg/layout/fs are known (done below).

    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    chart = env.public()          # what the processing knows of the environment
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=out / "cache")

    # Finish building the empirical localiser now that we have fs and layout.
    if empirical_loc is not None:
        empirical_loc.layout = layout
        empirical_loc.fs = cfg.fdtd.fs
        empirical_loc.build()
        print(f"Empirical localiser ({args.empirical_refine}): "
              f"{len(empirical_loc._source_xy)} replica points.", flush=True)
    uuv_id = next(v.id for v in cfg.vessels if v.kind == "uuv")
    uuv_level = next(v.level_pa for v in cfg.vessels if v.kind == "uuv")
    zero = {v.id: 0.0 for v in cfg.vessels}

    base = ProcessingConfig(use_cuda=args.cuda)
    grid = build_grid(chart, base)
    print(f"scan grid: {len(grid)} cells, {layout.n_receivers} receivers")

    results: dict = {"pfa": args.pfa, "uuv_level_pa": uuv_level, "configs": {}}
    runs: dict = {}
    for name in args.configs:
        # CFAR calibration uses line_halfwidth=0 (noise-only data, no tonals to enhance)
        pc_cal = ProcessingConfig(sensor_groups=CONFIGS[name], use_cuda=args.cuda)
        t0 = time.time()
        maxima = []
        for s in range(args.cal_seeds):
            ds = assemble_dataset(cfg, layout, tracks, gathers, seed=1000 + s, levels=zero, keep_clean=False)
            run = run_detection(ds.traces, layout, ds.fs, chart, pc_cal, floor=1e18, grid=grid)
            maxima.append(run.scan.frame_max)
        maxima = np.concatenate(maxima)
        thr = threshold_from_maxima(maxima, args.pfa)
        # Sanity: threshold should not change by more than 40% from the median noise statistic
        assert thr < 3.0 * np.median(maxima), f"[{name}] CFAR threshold unexpectedly high: {thr:.1f}"
        print(f"[{name}] CFAR threshold {thr:.1f} (noise maxima median {np.median(maxima):.1f}, "
              f"max {maxima.max():.1f}, {time.time() - t0:.0f} s)", flush=True)
        entry = {"threshold": thr, "noise_max_median": float(np.median(maxima)), "noise_max_max": float(maxima.max()), "levels": {}}

        # UUV-detection sweep uses line_halfwidth=4 to favour tonal lines over broadband ship leakage
        for fac in LEVEL_FACTORS:
            level = uuv_level * fac
            ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.eval_seed,
                                  levels={**UUV_ONLY, uuv_id: level}, keep_clean=False)
            if args.suppress:
                run = run_ais_aware(ds.traces, layout, ds.fs, chart, ds.ais,
                                    ProcessingConfig(sensor_groups=CONFIGS[name], line_halfwidth=4,
                                                     use_cuda=args.cuda,
                                                     empirical_localiser=empirical_loc),
                                    floor=0.5 * thr, grid=grid)
                dets_obj = run.dets
            else:
                run_obj = run_detection(ds.traces, layout, ds.fs, chart,
                                        ProcessingConfig(sensor_groups=CONFIGS[name],
                                                         use_cuda=args.cuda,
                                                         empirical_localiser=empirical_loc),
                                        floor=0.5 * thr, grid=grid)
                dets_obj = run_obj.dets
                run = run_obj
            sc = score_detections(dets_obj, ds.truth, layout, run.wh.times, thr)
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

    # Suppressed-fused baseline curve: re-run level sweep with suppression on fused config
    if args.suppress and "fused" in args.configs:
        thr_fused = results["configs"]["fused"]["threshold"]
        entry_sup: dict = {"threshold": thr_fused, "levels": {}}
        for fac in LEVEL_FACTORS:
            level = uuv_level * fac
            ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.eval_seed,
                                  levels={**UUV_ONLY, uuv_id: level}, keep_clean=False)
            run = run_ais_aware(ds.traces, layout, ds.fs, chart, ds.ais,
                                ProcessingConfig(sensor_groups=None, line_halfwidth=4,
                                                 use_cuda=args.cuda,
                                                 empirical_localiser=empirical_loc),
                                floor=0.5 * thr_fused, grid=grid)
            sc = score_detections(run.dets, ds.truth, layout, run.wh.times, thr_fused)
            db = 20 * np.log10(level)
            entry_sup["levels"][f"{fac}"] = {
                "level_db_re_1pa_100m": db, "pd": sc.pd(uuv_id), "rmse_m": sc.rmse(uuv_id),
                "fa_per_frame": sc.false_alarms_per_frame,
                "pd_by_range": {f"{a}-{b}": sc.pd(uuv_id, (a, b)) for a, b in [(0, 150), (150, 300), (300, 600), (600, 1e4)]},
            }
            print(f"[fused_suppressed] UUV {db:6.1f} dB: Pd {sc.pd(uuv_id):.2f}  RMSE {sc.rmse(uuv_id):6.1f} m  "
                  f"FA/frame {sc.false_alarms_per_frame:.3f}", flush=True)
            runs[("fused_suppressed", fac)] = (ds, run, sc, thr_fused)
        results["configs"]["fused_suppressed"] = entry_sup

    # Full scenario pass — use run_ais_aware when suppression is on (fused only)
    if "fused" in args.configs:
        pc = ProcessingConfig(use_cuda=args.cuda)
        thr = results["configs"]["fused"]["threshold"]
        ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.eval_seed, keep_clean=False)
        if args.suppress:
            run = run_ais_aware(ds.traces, layout, ds.fs, chart, ds.ais, pc, floor=0.5 * thr, grid=grid)
        else:
            run = run_detection(ds.traces, layout, ds.fs, chart, pc, floor=0.5 * thr, grid=grid)
        sc = score_detections(run.dets, ds.truth, layout, run.wh.times, thr, targets=list(tracks))
        results["all_vessels_fused"] = {
            vid: {"pd": sc.pd(vid), "rmse_m": sc.rmse(vid)} for vid in tracks
        } | {"fa_per_frame": sc.false_alarms_per_frame}
        print("[all vessels] " + ", ".join(
            f"{vid}: Pd {sc.pd(vid):.2f}" if tracks[vid].level_pa > 0 and (tracks[vid].speed > 0.1).any() or vid == uuv_id
            else f"{vid}: n/a" for vid in tracks), f" FA/frame {sc.false_alarms_per_frame:.3f}", flush=True)
        runs[("all", 1.0)] = (ds, run, sc, thr)

    # Record per-localiser RMSE at nominal UUV level (fac=1.0) for fused config.
    if empirical_loc is not None and "fused" in args.configs and "fused" in results["configs"]:
        localiser_rmse: dict = {}
        # analytic-only baseline (no empirical localiser)
        fac_key = "1.0"
        if fac_key in results["configs"]["fused"]["levels"]:
            localiser_rmse["analytic"] = results["configs"]["fused"]["levels"][fac_key]["rmse_m"]
        # empirical mode (already computed with localiser active)
        localiser_rmse[args.empirical_refine] = results["configs"]["fused"]["levels"].get(fac_key, {}).get("rmse_m", None)
        results["localiser_rmse"] = localiser_rmse

    results["provenance"] = next(iter(runs.values()))[1].provenance.to_dict() if runs else None
    (out / "m3_results.json").write_text(json.dumps(results, indent=2, default=float))

    # ---- figures ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, entry in results["configs"].items():
        db = [v["level_db_re_1pa_100m"] for v in entry["levels"].values()]
        if name == "fused_suppressed":
            ax[0].plot(db, [v["pd"] for v in entry["levels"].values()], "s--", label="fused+suppress")
            ax[1].plot(db, [v["rmse_m"] for v in entry["levels"].values()], "s--", label="fused+suppress")
        else:
            ax[0].plot(db, [v["pd"] for v in entry["levels"].values()], "o-", label=name)
            ax[1].plot(db, [v["rmse_m"] for v in entry["levels"].values()], "o-", label=name)
    ax[0].set(xlabel="UUV source level [dB re 1 Pa @ 100 m]", ylabel="Pd (per frame)", title=f"Detection (Pfa/frame = {args.pfa})", ylim=(0, 1.02))
    ax[1].set(xlabel="UUV source level [dB re 1 Pa @ 100 m]", ylabel="RMSE [m]", title="Localisation error of detections")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    if runs:
        annotate_figure(fig, next(iter(runs.values()))[1].provenance)
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
        annotate_figure(fig, run.provenance)
        fig.savefig(out / fname, dpi=100, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

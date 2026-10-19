#!/usr/bin/env python3
"""Stage 1c: detection of the silent UUV measured honestly, on development and held out data.

Stage 0b reported the fraction of frames whose map peak lies within 5 m of the UUV, with the
map divided by its own maximum, and a peak over median contrast that turned out to carry no
information.  This script measures a detection probability at a fixed **frame false alarm
rate**: the unnormalised ``coherent_norm="snr"`` map (``map_norm="none"``) is formed for the
same traces **without** the body (direct path and noise only, the identical chain), a
threshold ``T`` is set per configuration so that a fraction ``pfa`` (0.05) of the body free
frames has any cell outside the ship exclusion above ``T``, and ``Pd`` is the fraction of
frames of the run with the body in which the map peak exceeds ``T`` and lies within 5 m of
the UUV truth.  The threshold is set on the development data only.

Data
----
``dev``        the fine bistatic dataset (ferry track 1, noise seed 0): all tuning happened here
``noise``      the same cached gathers mixed with noise seeds 1 to 5 (``assemble_dataset(seed=...)``)
``geometry``   ferry track 2 (``harbor_bistatic_track2.scenario.json``, heading south at x = 100 m,
               a different bistatic angle to the same UUV; two new fine FDTD runs), seeds 0 to 2

Configurations
--------------
1. ``oracle``              truth direct removal, oracle waveform, exact track (stage 0 baseline)
2. ``eca_oracle+doppler``  direct path in the traces, ECA with per sample delays, oracle waveform, exact track
3. ``blind``               best stage 1a reference, stage 1b Kalman track with the low speed fallback, ``eca_doppler``
4. ``blind_stage0``        stage 0 settings: plain ``data`` beam, 4 s straight line track fit, ``eca_doppler``

Also reported per run: median and p90 error, fraction within 5 m, the 1.5 s (three frame)
integrated versions with their own threshold, the realised false alarm rate of the body free
run at ``T`` (for the held out sets this shows whether ``T`` transfers), the ECA depth of the
direct path and loss of the echo (truth components through the same projection, diagnostics
only) and the ship track error.  Every frame with an error over 20 m is classified by where its
peak sits (within 10 m of the ship exclusion edge, of a first order quay image of the ship, of
the direct path fan between the ship and the cable, or elsewhere; ``bistatic_eval.classify_misses``).

Criterion (fixed before the run): configuration 3 reaches ``Pd >= 0.5`` at the 0.05 frame false
alarm rate averaged over the held out noise seeds, and ``Pd >= 0.5`` averaged over the held out
geometry runs.

Usage::

    python examples/harbor_bistatic_detect_eval.py [--cuda] [--workers 3] [--ref-mode snr]
        [--noise-seeds 1,2,3,4,5] [--geom-seeds 0,1,2] [--configs oracle,eca_oracle+doppler,blind,blind_stage0]
        [--out output/harbor_bistatic/eval]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")          # the ECA is BLAS bound; several runs share the cores

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from acoustic_sim.harbor import HarborDataset, HarborScenarioConfig, assemble_dataset, build_environment, build_layout, build_track, render_vessel_gathers  # noqa: E402
from acoustic_sim.harbor.bistatic import BistaticConfig, Truth, run_bistatic, stft  # noqa: E402
from acoustic_sim.harbor.bistatic_eval import MISS_CLASSES, classify_misses, detection_stats, frame_threshold, integrate_frames, miss_counts  # noqa: E402
from acoustic_sim.harbor.dataset import vessel_injection  # noqa: E402
from acoustic_sim.harbor.dispersion import for_scenario  # noqa: E402
from acoustic_sim.harbor.plotting import plot_harbor  # noqa: E402
from acoustic_sim.harbor.signatures import f_max_for_grid  # noqa: E402
from harbor_reference_modes import MODES as REF_MODES  # noqa: E402

PFA = 0.05
RADIUS = 5.0
MISS_ERR = 20.0
CONFIGS = ("oracle", "eca_oracle+doppler", "blind", "blind_stage0")
ORACLE_INPUTS = {
    "oracle": ["direct path removed with the truth component", "oracle ferry waveform", "exact ferry track"],
    "eca_oracle+doppler": ["oracle ferry waveform", "exact ferry track"],
    "blind": [],
    "blind_stage0": [],
}


def config_for(name: str, ref_mode: str, env, cfg, use_cuda: bool) -> tuple[BistaticConfig, bool, bool]:
    """``(BistaticConfig, direct path in the traces, exact track)`` of a configuration."""
    common = dict(win_s=0.5, hop_s=0.25, f_lo=100.0, bin_stride=2, grid_dx=2.0, method="coherent", coherent_norm="snr",
                  map_norm="none", exclusion_radius=25.0, dispersion=for_scenario(env, cfg.fdtd), use_cuda=use_cuda,
                  ais_pos_sigma=cfg.ais.pos_sigma, ais_sog_sigma=cfg.ais.sog_sigma, ais_cog_sigma=cfg.ais.cog_sigma)
    if name == "oracle":
        return BistaticConfig(**common, reference="oracle", removal="none", track_filter="none"), False, True
    if name == "eca_oracle+doppler":
        return BistaticConfig(**common, reference="oracle", removal="eca", eca_doppler=True, track_filter="none"), True, True
    if name == "blind":
        return BistaticConfig(**common, **REF_MODES[ref_mode], removal="eca", eca_doppler=True, track_filter="kalman"), True, False
    if name == "blind_stage0":
        return BistaticConfig(**common, reference="data", ref_mode="beam", removal="eca", eca_doppler=True, track_filter="line",
                              track_smooth_s=4.0), True, False
    raise ValueError(name)


def load_set(path: str, seed: int | None, cache: str) -> HarborDataset:
    """The saved dataset (``seed is None``) or the same gathers mixed with another noise seed."""
    ds = HarborDataset.load(path)
    if seed is None:
        return ds
    env = build_environment(ds.cfg)
    layout = build_layout(ds.cfg, env)
    tracks = {v.id: build_track(v, ds.cfg.fdtd.total_time, env) for v in ds.cfg.vessels}
    gathers = render_vessel_gathers(ds.cfg, env, layout, tracks, cache_dir=cache, workers=1)
    return assemble_dataset(ds.cfg, layout, tracks, gathers, seed=seed, keep_components=True)


def run_job(job: dict) -> dict:
    """One configuration on one dataset with or without the body; returns the map and the diagnostics."""
    name, path, seed, body, ref_mode, use_cuda, cache = (job[k] for k in ("config", "path", "seed", "body", "ref_mode", "cuda", "cache"))
    ds = load_set(path, seed, cache)
    env = build_environment(ds.cfg)
    layout = build_layout(ds.cfg, env)
    f_max = f_max_for_grid(env.dx)
    bc, direct_in, exact = config_for(name, ref_mode, env, ds.cfg, use_cuda)
    gain = ds.gain[:, None]
    direct = gain * ds.components["direct:ferry"].astype(float)
    echo = gain * ds.components["scattered:ferry>uuv1"].astype(float)
    noise = ds.components["noise"].astype(float)
    traces = noise + (echo if body else 0.0) + (direct if direct_in else 0.0)
    ferry = next(v for v in ds.cfg.vessels if v.id == "ferry")
    oracle = ferry.level_pa * vessel_injection(ferry, ds.tracks["ferry"], env, ds.cfg) if bc.reference == "oracle" else None
    sp = stft(traces, layout, ds.fs, bc, f_max)
    comps = {"direct:ferry": direct} if direct_in else {}
    if body and direct_in:
        comps["scattered:ferry>uuv1"] = echo
    truth = Truth(ship_xy=ds.tracks["ferry"].position_at(sp.times) if exact else None, components=comps or None)
    t0 = time.time()
    run = run_bistatic(traces, layout, ds.fs, env, ds.ais, bc, f_max, oracle_signal=oracle, truth=truth)
    cab = layout.slices["cable1"]
    live = run.sp.weights > 0
    exact_xy = ds.tracks["ferry"].position_at(run.sp.times)
    return {
        "config": name, "path": path, "seed": seed, "body": body, "seconds": time.time() - t0, "cfg": ds.cfg.to_dict(),
        "B": run.scan.B.astype(np.float32), "grid": run.grid, "times": run.sp.times,
        "uuv": ds.tracks["uuv1"].position_at(run.sp.times), "ship_used": run.ship.xy, "ship_exact": exact_xy, "live": live,
        "ship_error_median_m": float(np.median(np.hypot(*(run.ship.xy - exact_xy).T))),
        "eca_direct_depth_db": float(np.median(run.eca.removed_db[cab][live[cab]])) if run.eca else None,
        "eca_echo_loss_db": float(np.median(run.eca.echo_lost_db[cab][live[cab]])) if (run.eca and body) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default="output/harbor_bistatic/fine/dataset.npz")
    ap.add_argument("--track2", default="output/harbor_bistatic/fine_track2/dataset.npz")
    ap.add_argument("--cache", default="output/harbor_bistatic/cache")
    ap.add_argument("--out", default="output/harbor_bistatic/eval")
    ap.add_argument("--reference-results", default="output/harbor_reference/results.json", help="stage 1a results (best mode)")
    ap.add_argument("--ref-mode", default=None, help="blind reference mode (default: the best mode of stage 1a)")
    ap.add_argument("--noise-seeds", default="1,2,3,4,5")
    ap.add_argument("--geom-seeds", default="0,1,2")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "w")

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    ref_mode = args.ref_mode
    if ref_mode is None:
        ref_mode = json.loads(Path(args.reference_results).read_text())["best_mode"]
    configs = [c for c in args.configs.split(",") if c]
    noise_seeds = [int(s) for s in args.noise_seeds.split(",") if s]
    geom_seeds = [int(s) for s in args.geom_seeds.split(",") if s]
    sets = [("dev", args.dev, None)] + [(f"noise{s}", args.dev, s) for s in noise_seeds]
    sets += [(f"geometry{s}", args.track2, None if s == 0 else s) for s in geom_seeds]
    results = {
        "criterion": {"text": f"configuration 'blind' reaches Pd >= 0.5 at a frame false alarm rate of {PFA:g} (threshold set on the body free "
                              "development run) averaged over the held out noise seeds, and Pd >= 0.5 averaged over the held out geometry runs",
                      "pd_min": 0.5, "pfa": PFA, "radius_m": RADIUS},
        "blind_reference_mode": ref_mode, "blind_reference_config": REF_MODES[ref_mode],
        "configs": {c: {"oracle_inputs": ORACLE_INPUTS[c]} for c in configs},
        "sets": {name: {"path": path, "seed": seed} for name, path, seed in sets},
        "miss_classes": {"error_min_m": MISS_ERR, "near_m": 10.0, "classes": list(MISS_CLASSES)},
        "runs": {}, "summary": {},
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))         # the criterion is on disk before any run
    say(f"criterion: {results['criterion']['text']}")
    say(f"blind reference mode: {ref_mode} {REF_MODES[ref_mode]}")
    sets = [s for s in sets if Path(s[1]).exists()] or sets
    missing = [s for s in sets if not Path(s[1]).exists()]
    for s in missing:
        say(f"[{s[0]}] {s[1]} missing: skipped")
    sets = [s for s in sets if Path(s[1]).exists()]

    jobs = [dict(config=c, path=path, seed=seed, body=body, ref_mode=ref_mode, cuda=args.cuda, cache=args.cache, set=name)
            for name, path, seed in sets for c in configs for body in (True, False)]
    say(f"{len(jobs)} chains ({len(sets)} sets x {len(configs)} configurations x with / without the body), {args.workers} workers")
    t0 = time.time()
    runs = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for job, res in zip(jobs, pool.map(run_job, jobs)):
            runs[(job["set"], job["config"], job["body"])] = res
            say(f"[{job['set']:9s}] {job['config']:20s} {'body' if job['body'] else 'free'}: {res['seconds']:.0f} s, ship err {res['ship_error_median_m']:.1f} m"
                + (f", ECA direct {res['eca_direct_depth_db']:.1f} dB" if res["eca_direct_depth_db"] is not None else "")
                + (f", echo loss {res['eca_echo_loss_db']:.2f} dB" if res["eca_echo_loss_db"] is not None else ""))
    say(f"all chains in {(time.time() - t0) / 60:.0f} min")

    # Thresholds on the development body free runs, then the statistics of every run.
    thresholds = {}
    for c in configs:
        free = runs[("dev", c, False)]
        B3 = integrate_frames(free["B"], 3)
        thresholds[c] = {"frame": frame_threshold(free["B"].max(axis=0), PFA), "integrated": frame_threshold(B3.max(axis=0), PFA)}
        say(f"threshold {c:20s}: {thresholds[c]['frame']:.4g} (frame), {thresholds[c]['integrated']:.4g} (1.5 s integrated)")
    results["thresholds"] = thresholds
    misses = {}
    for (name, c, body), res in runs.items():
        T = thresholds[c]
        st = detection_stats(res["B"], res["grid"], res["uuv"], T["frame"], RADIUS)
        st3 = detection_stats(integrate_frames(res["B"], 3), res["grid"], res["uuv"], T["integrated"], RADIUS)
        row = {"set": name, "config": c, "body": body, "seconds": res["seconds"], "ship_error_median_m": res["ship_error_median_m"],
               "eca_direct_depth_db": res["eca_direct_depth_db"], "eca_echo_loss_db": res["eca_echo_loss_db"],
               "frame": st.summary(), "integrated": st3.summary()}
        if body:
            if ("env", res["path"]) not in misses:
                cfg_ = HarborScenarioConfig.from_dict(res["cfg"])
                env_ = build_environment(cfg_)
                misses[("env", res["path"])] = (env_, build_layout(cfg_, env_))
            env, layout = misses[("env", res["path"])]
            cls = classify_misses(st.est, st.err, res["ship_used"], env, layout, res["live"], 25.0, MISS_ERR, 10.0)
            row["misses"] = miss_counts(cls)
            row["misses"]["total"] = int(np.sum(st.err > MISS_ERR))
            misses[(name, c)] = (st, cls, res)
        results["runs"][f"{name}/{c}/{'body' if body else 'free'}"] = row
        kind = "body" if body else "free"
        say(f"[{name:9s}] {c:20s} {kind}: Pd {st.pd:.2f} (declared {st.declared:.2f}), within 5 m {st.summary()['within_radius']:.2f}, "
            f"median err {st.summary()['median_error_m']:.1f} m, p90 {st.summary()['p90_error_m']:.0f} m; 1.5 s: Pd {st3.pd:.2f}, within 5 m {st3.summary()['within_radius']:.2f}"
            + (f"; misses {row['misses']}" if body else ""))

    # Summary per configuration: development, held out noise (mean over seeds), held out geometry (mean over seeds).
    groups = {"dev": ["dev"], "noise": [s[0] for s in sets if s[0].startswith("noise")], "geometry": [s[0] for s in sets if s[0].startswith("geometry")]}
    for c in configs:
        results["summary"][c] = {}
        for g, names in groups.items():
            if not names:
                continue
            rows = [results["runs"][f"{n}/{c}/body"] for n in names]
            frees = [results["runs"][f"{n}/{c}/free"] for n in names]
            results["summary"][c][g] = {
                "pd": float(np.mean([r["frame"]["pd"] for r in rows])), "pd_by_set": {n: r["frame"]["pd"] for n, r in zip(names, rows)},
                "declared_free": float(np.mean([r["frame"]["declared"] for r in frees])),
                "within_5m": float(np.mean([r["frame"]["within_radius"] for r in rows])),
                "median_error_m": float(np.mean([r["frame"]["median_error_m"] for r in rows])),
                "integrated_pd": float(np.mean([r["integrated"]["pd"] for r in rows])),
                "integrated_within_5m": float(np.mean([r["integrated"]["within_radius"] for r in rows])),
                "eca_direct_depth_db": float(np.mean([r["eca_direct_depth_db"] for r in rows])) if rows[0]["eca_direct_depth_db"] is not None else None,
                "misses": {k: int(sum(r["misses"][k] for r in rows)) for k in list(MISS_CLASSES) + ["total"]},
            }
        say(f"{c:20s}: " + "  ".join(f"{g} Pd {v['pd']:.2f} (free {v['declared_free']:.2f}, within 5 m {v['within_5m']:.2f}, 1.5 s Pd {v['integrated_pd']:.2f})"
                                     for g, v in results["summary"][c].items()))
    crit = results["summary"].get("blind", {})
    ok_noise = crit.get("noise", {}).get("pd", 0.0) >= 0.5 if "noise" in crit else None
    ok_geom = crit.get("geometry", {}).get("pd", 0.0) >= 0.5 if "geometry" in crit else None
    results["criterion"]["pd_noise"] = crit.get("noise", {}).get("pd")
    results["criterion"]["pd_geometry"] = crit.get("geometry", {}).get("pd")
    results["criterion"]["pass"] = bool(ok_noise and ok_geom)
    say(f"criterion: blind Pd held out noise {results['criterion']['pd_noise']}, held out geometry {results['criterion']['pd_geometry']} "
        f"-> {'PASS' if results['criterion']['pass'] else 'FAIL'}")
    if "blind" in results["summary"]:
        pooled = {k: sum(v["misses"][k] for v in results["summary"]["blind"].values()) for k in MISS_CLASSES}
        results["dominant_miss_class_blind"] = max(pooled, key=pooled.get) if sum(pooled.values()) else None
        say(f"blind misses by class (all sets): {pooled} -> dominant: {results['dominant_miss_class_blind']}")
    (out / "results.json").write_text(json.dumps(results, indent=2))

    # Figure 1: Pd by configuration and data group.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gnames = [g for g in groups if groups[g]]
    w = 0.8 / len(gnames)
    for ax, key, label in ((axes[0], "pd", f"Pd at frame false alarm rate {PFA:g} (5 m)"), (axes[1], "integrated_pd", "Pd, 1.5 s integrated")):
        for j, g in enumerate(gnames):
            vals = [results["summary"][c].get(g, {}).get(key, np.nan) for c in configs]
            x = np.arange(len(configs)) + (j - 0.5 * (len(gnames) - 1)) * w
            ax.bar(x, vals, w, label={"dev": "development", "noise": "held out noise (mean)", "geometry": "held out geometry (mean)"}[g])
            for i, c in enumerate(configs):
                pts = [results["runs"][f"{n}/{c}/body"]["frame" if key == "pd" else "integrated"]["pd"] for n in groups[g]]
                ax.plot(np.full(len(pts), x[i]), pts, "k.", ms=4)
        ax.axhline(0.5, color="r", ls="--", label="criterion")
        ax.set_xticks(np.arange(len(configs)))
        ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Stage 1c: detection probability (blind reference mode: {ref_mode}; dots: individual runs)")
    fig.savefig(out / "pd_by_config.png", dpi=110, bbox_inches="tight")

    # Figure 2: miss locations of the blind configuration, coloured by class.
    if "blind" in configs:
        fig, axes = plt.subplots(1, len(gnames), figsize=(6.5 * len(gnames), 6), squeeze=False)
        colors = dict(zip(MISS_CLASSES, ("C3", "C1", "C0", "0.4")))
        for ax, g in zip(axes[0], gnames):
            first = misses[(groups[g][0], "blind")][2]
            env, layout = misses[("env", first["path"])]
            plot_harbor(env, layout, ax=ax, show_depth=False)
            ax.plot(*first["ship_exact"].T, "-", color="deepskyblue", lw=2, label="ferry track")
            ax.plot(*first["uuv"].T, "-", color="lime", lw=2, label="UUV track")
            for n in groups[g]:
                st, cls, res = misses[(n, "blind")]
                for k in MISS_CLASSES:
                    m = cls == k
                    if m.any():
                        ax.plot(*st.est[m].T, "o", color=colors[k], ms=5, mec="k", mew=0.4, alpha=0.8, label=k if n == groups[g][0] else None)
            counts = results["summary"]["blind"][g]["misses"]
            ax.set_title(f"blind, {g}: " + ", ".join(f"{k} {counts[k]}" for k in MISS_CLASSES) + f" (of {counts['total']} misses)", fontsize=9)
            ax.legend(fontsize=7, loc="lower left")
            if ax.get_legend() and len(ax.get_legend().texts) == 0:
                ax.get_legend().remove()
        fig.suptitle(f"Stage 1c: peaks of the frames with an error over {MISS_ERR:g} m (blind configuration)")
        fig.savefig(out / "misses_map.png", dpi=110, bbox_inches="tight")

    # Figure 3: error against time per configuration.
    fig, axes = plt.subplots(len(configs), len(gnames), figsize=(6 * len(gnames), 3.2 * len(configs)), squeeze=False, sharex=True, sharey=True)
    for i, c in enumerate(configs):
        for j, g in enumerate(gnames):
            ax = axes[i, j]
            for n in groups[g]:
                st, cls, res = misses[(n, c)]
                ax.plot(res["times"], st.err, "-", lw=1, alpha=0.8, label=n)
                ax.plot(res["times"][st.detected], st.err[st.detected], ".", color="k", ms=3)
            ax.axhline(RADIUS, color="r", ls="--", lw=0.8)
            ax.set_yscale("log")
            ax.set_ylim(0.3, 400)
            ax.grid(alpha=0.3, which="both")
            s = results["summary"][c][g]
            ax.set_title(f"{c}, {g}: Pd {s['pd']:.2f}, within 5 m {s['within_5m']:.2f}", fontsize=9)
            if j == 0:
                ax.set_ylabel("error [m]")
            if i == len(configs) - 1:
                ax.set_xlabel("time [s]")
            if len(groups[g]) > 1 and i == 0:
                ax.legend(fontsize=6)
    fig.suptitle("Stage 1c: peak error against time (black dots: declared detections)")
    fig.savefig(out / "error_vs_time.png", dpi=100, bbox_inches="tight")
    say(f"wrote {out}")


if __name__ == "__main__":
    main()

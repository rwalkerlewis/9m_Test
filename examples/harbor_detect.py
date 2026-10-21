#!/usr/bin/env python3
"""Detect objects in a harbor recording and check every detection against where the objects really are.

One pass per chain, everything estimated from the traces (free field replicas at the chart's sound
speed, AIS reports as recorded):

* ``passive``: ``run_ais_aware`` (AIS ships refined, projected out, excluded) and the matched field
  scan; finds what radiates and has no AIS report.
* ``echo``: ``run_bistatic`` with the AIS ship as illuminator (ECA with per sample delays, 4 s track
  fit, coherent map); finds silent bodies by their echo.

Per frame the ``--peaks`` strongest map peaks (at least ``--sep`` m apart) are the detections.  The
truth is read only afterwards, for the check: each detection is labelled with the nearest object
active at that time and its distance; a detection within ``--hit`` m (5) is a hit, one more than
10 m from every object a false alarm.

Outputs in ``--out/<dataset>/``: ``detections.csv`` (chain, t, x, y, score, rank, nearest object,
distance), ``check.csv`` (per object and chain: frames, frames hit by the top peak, by any peak,
median error), ``check.png`` (top: true tracks and detections on the chart; bottom: one frame of each chain with its
map, the AIS ships where the chain refined them, the exclusion zones and the ray paths), ``summary.json``.

Usage::

    python examples/harbor_detect.py --dataset bistatic|multistatic|harbor [--cuda]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Observation, Provenance, Truth  # noqa: E402
from acoustic_sim.water.ais import AISReports  # noqa: E402
from acoustic_sim.water.bistatic import BistaticConfig, exclusion, run_bistatic  # noqa: E402
from acoustic_sim.water.detection import ProcessingConfig, _nms  # noqa: E402
from acoustic_sim.water.plotting import draw_frame_geometry  # noqa: E402
from acoustic_sim.water.propagation import FreeFieldPropagator  # noqa: E402
from acoustic_sim.water.suppression import SuppressionConfig, run_ais_aware  # noqa: E402

DATASETS = {
    # path, band [Hz], chains, passive settings (0.5 m datasets scaled from the 5 m defaults)
    "bistatic": ("output/harbor_bistatic/fine/dataset", (100.0, 760.0), ("echo",), None),
    "multistatic": ("output/harbor_multistatic/fine/dataset", (100.0, 760.0), ("passive", "echo"),
                    dict(pc=dict(win_s=0.5, hop_s=0.25, grid_dx=2.0, grid_margin=5.0, min_range=1.0), edge=40.0,
                         sc=dict(refine_half=12.0, refine_step=1.0, patch_half=0.5, exclusion_radius=25.0))),
    "harbor": ("output/harbor/dataset", (12.0, 78.0), ("passive",), dict(pc=dict(grid_dx=5.0), edge=150.0, sc=dict())),
}


def peaks(B: np.ndarray, grid: np.ndarray, n: int, sep: float) -> list[tuple[int, int, int]]:
    """``(frame, cell, rank)`` of the *n* strongest peaks of every frame, at least *sep* m apart."""
    out = []
    for k in range(B.shape[1]):
        for r, g in enumerate(_nms(B[:, k], grid, sep, n, 0.0)):
            out.append((k, int(g), r))
    return out


def run_passive(obs: Observation, band, spec, cuda: bool):
    env = obs.environment
    x0, x1, y0, y1 = env.extent
    e = spec["edge"]
    prop = FreeFieldPropagator.from_environment(env)
    pc = ProcessingConfig(f_lo=band[0], f_hi=band[1], c=prop.c, propagator=prop, use_cuda=cuda,
                          grid_bounds=(x0 + e, x1 - e, y0 + e, y1 - e), **spec["pc"])
    sc = SuppressionConfig(**spec["sc"])
    run = run_ais_aware(obs.traces, obs.geometry, obs.fs, env, obs.reports, pc, sc, provenance=Provenance(data="forward_model"))
    est, grid = run.ships, run.scanner.grid
    ships = [{vid: est.refined[k, v] for v, vid in enumerate(est.ids) if est.present[k, v]} for k in range(len(run.scan.times))]
    ex = est.exclusion_mask(grid, sc.exclusion_radius) if est.ids else np.zeros(run.scan.B.shape, bool)
    return run.scan.B, grid, run.scan.times, run.provenance, (ships, ex)


def run_echo(obs: Observation, band, cuda: bool, source: str):
    env = obs.environment
    prop = FreeFieldPropagator.from_environment(env)
    rep = obs.reports
    m = rep.vessel == source
    reports = AISReports(rep.vessel[m], rep.t[m], rep.xy[m], rep.sog[m], rep.cog[m])
    bc = BistaticConfig(f_lo=band[0], f_hi=band[1], c=prop.c, propagator=prop, use_cuda=cuda, grid_dx=2.0,
                        coherent_norm="snr", map_scale="none", exclusion_radius=25.0, eca_doppler=True, track_smooth_s=4.0)
    run = run_bistatic(obs.traces, obs.geometry, obs.fs, env, reports, bc, provenance=Provenance(data="forward_model"))
    ships = [{source: run.ship.refined[k]} if run.ship.present[k] else {} for k in range(len(run.sp.times))]
    return run.scan.B, run.grid, run.sp.times, run.provenance, (ships, exclusion(run.grid, run.ship, bc.exclusion_radius))


def check(dets, B, grid, times, truth: Truth, hit: float):
    """Label every detection with the nearest active object; per object hit counts."""
    ids = list(truth.tracks)
    pos = np.stack([truth.tracks[i].position_at(times) for i in ids])            # (O, K, 2)
    act = np.stack([truth.tracks[i].active_at(times) for i in ids])               # (O, K)
    rows = []
    for k, g, r in dets:
        d = np.hypot(*(pos[:, k] - grid[g]).T)
        d[~act[:, k]] = np.inf
        j = int(np.argmin(d))
        rows.append({"t": float(times[k]), "x": float(grid[g, 0]), "y": float(grid[g, 1]), "score": float(B[g, k]), "rank": r,
                     "nearest": ids[j], "distance_m": float(d[j]), "frame": k})
    per = {}
    for j, oid in enumerate(ids):
        frames = int(act[j].sum())
        top = [x for x in rows if x["nearest"] == oid and x["distance_m"] <= hit and x["rank"] == 0]
        anyr = {x["frame"] for x in rows if x["nearest"] == oid and x["distance_m"] <= hit}
        errs = [x["distance_m"] for x in rows if x["nearest"] == oid and x["distance_m"] <= hit]
        tr = truth.tracks[oid]
        per[oid] = {"kind": tr.kind, "radiates": bool(tr.level_pa > 0), "ais": bool(tr.ais), "frames": frames,
                    "top_peak_hit": len(top) / max(frames, 1), "any_peak_hit": len(anyr) / max(frames, 1),
                    "median_error_m": float(np.median(errs)) if errs else None}
    false = sum(1 for x in rows if x["distance_m"] > 10.0)
    return rows, per, false / max(len(times), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--peaks", type=int, default=3, help="peaks per frame reported as detections")
    ap.add_argument("--sep", type=float, default=20.0, help="minimum distance between the peaks of a frame [m]")
    ap.add_argument("--hit", type=float, default=5.0, help="a detection within this distance of an object is a hit [m]")
    ap.add_argument("--out", default="output/harbor_detect")
    args = ap.parse_args()
    path, band, chains, spec = DATASETS[args.dataset]
    out = ROOT / args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    obs = Observation.load(ROOT / f"{path}.observation.npz")
    truth = Truth.load(ROOT / f"{path}.truth.npz")                                # read only for the check
    T = obs.traces.shape[1] / obs.fs

    all_rows, summary, geometry = [], {"dataset": args.dataset, "record_s": T, "chains": {}}, {}
    runs = [("passive", None)] * ("passive" in chains) + [("echo", s) for s in (obs.reports.vessels[:1] if "echo" in chains else [])]
    for chain, src in runs:
        t0 = time.time()
        B, grid, times, prov, (ships, ex) = run_passive(obs, band, spec, args.cuda) if chain == "passive" else run_echo(obs, band, args.cuda, src)
        secs = time.time() - t0
        name = chain if src is None else f"echo ({src} illuminates)"
        dets = peaks(B, grid, args.peaks, args.sep)
        k = len(times) // 2                                                          # a frame for the geometry figure
        geometry[name] = (grid, B[:, k], times[k], ships[k], np.array([grid[g] for kk, g, _ in dets if kk == k]).reshape(-1, 2),
                          ex[:, k], chain == "echo")
        rows, per, fa = check(dets, B, grid, times, truth, args.hit)
        for x in rows:
            x["chain"] = name
        all_rows += rows
        summary["chains"][name] = {"seconds": secs, "seconds_per_second_of_data": secs / T, "frames": len(times),
                                   "false_detections_per_frame": fa, "objects": per, "provenance": prov.to_dict()}
        print(f"\n{name}: {secs:.0f} s for {T:.0f} s of data, {len(times)} frames, {fa:.2f} detections per frame > 10 m from every object")
        print(f"  {'object':10s} {'kind':7s} {'rad':4s} {'AIS':4s} {'top peak hit':>13s} {'any peak hit':>13s} {'median err':>11s}")
        for oid, p in per.items():
            err = "-" if p["median_error_m"] is None else f"{p['median_error_m']:.1f} m"
            print(f"  {oid:10s} {p['kind']:7s} {'y' if p['radiates'] else 'n':4s} {'y' if p['ais'] else 'n':4s} "
                  f"{100 * p['top_peak_hit']:12.0f}% {100 * p['any_peak_hit']:12.0f}% {err:>11s}")

    with open(out / "detections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chain", "t", "x", "y", "score", "rank", "nearest", "distance_m"], extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    with open(out / "check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain", "object", "kind", "radiates", "ais", "frames", "top_peak_hit", "any_peak_hit", "median_error_m"])
        for name, c in summary["chains"].items():
            for oid, p in c["objects"].items():
                w.writerow([name, oid, p["kind"], p["radiates"], p["ais"], p["frames"], f"{p['top_peak_hit']:.3f}",
                            f"{p['any_peak_hit']:.3f}", "" if p["median_error_m"] is None else f"{p['median_error_m']:.2f}"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    env = obs.environment
    names = list(summary["chains"])
    fig, axes = plt.subplots(2, len(names), figsize=(7.5 * len(names), 13.0), squeeze=False)
    x0, x1, y0, y1 = env.extent
    for ax, name in zip(axes[0], names):
        ax.imshow(np.where(env.water, 1.0, 0.0), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys_r", vmin=-0.6, vmax=1.2)
        r = obs.geometry.positions()
        ax.plot(r[:, 0], r[:, 1], ".", color="0.3", ms=1.5)
        d = [x for x in all_rows if x["chain"] == name]
        hit = np.array([[x["x"], x["y"]] for x in d if x["distance_m"] <= args.hit]).reshape(-1, 2)
        far = np.array([[x["x"], x["y"]] for x in d if x["distance_m"] > args.hit]).reshape(-1, 2)
        ax.plot(far[:, 0], far[:, 1], "x", color="tab:red", ms=3, alpha=0.5, label=f"detection > {args.hit:g} m from every object ({len(far)})")
        ax.plot(hit[:, 0], hit[:, 1], "o", color="tab:green", ms=3, label=f"detection within {args.hit:g} m of an object ({len(hit)})")
        for oid, tr in truth.tracks.items():
            p = summary["chains"][name]["objects"][oid]
            moving = np.ptp(tr.xy, axis=0).max() > 0
            ax.plot(tr.xy[:, 0], tr.xy[:, 1], "-" if moving else "s", color="deepskyblue" if tr.level_pa > 0 else "orange",
                    lw=1.5, ms=4, mfc="none")
            ax.annotate(f"{oid} {100 * p['any_peak_hit']:.0f}%", tr.xy[-1], fontsize=7)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"{args.dataset}: {name}\ntracks: blue radiating, orange silent; % = frames with a detection within {args.hit:g} m",
                     fontsize=9)
    for ax, name in zip(axes[1], names):                                             # one frame of each chain, with its geometry
        grid, Bk, tk, rads, pk, exk, is_echo = geometry[name]
        ax.imshow(np.where(env.water, 1.0, 0.0), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys_r", vmin=-0.6, vmax=1.2)
        im = draw_frame_geometry(ax, grid, Bk, obs.geometry, rads, pk, echo=is_echo, exclusion=exk)
        for oid, tr in truth.tracks.items():
            ax.plot(*tr.position_at(tk)[0], "o", ms=10, mfc="none", mec="deepskyblue" if tr.level_pa > 0 else "orange", mew=1.5)
        fig.colorbar(im, ax=ax, shrink=0.7, label="dB re frame peak")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"{name}, t = {tk:.2f} s: map, AIS ships as refined,\nexclusion zones, ray paths; circles: true positions",
                     fontsize=9)
    fig.savefig(out / "check.png", dpi=110, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

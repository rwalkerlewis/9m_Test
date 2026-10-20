#!/usr/bin/env python3
"""Run the passive chain in real time on a recording and check every frame against the true positions.

The traces are fed to ``water.streaming.StreamingPassive`` in chunks of ``--chunk`` seconds, as they
would arrive from the interrogator, with the AIS reports received so far.  Every frame's peaks are
the detections.  The truth is read only for the check afterwards: a detection within ``--hit`` m
(5) of an object is a hit.  Reported: compute time per second of data (below 1 is real time) and
the latency from the last sample of a frame to its detections.

Outputs in ``--out/<dataset>/``: ``detections.csv``, ``check.csv``, ``check.png``, ``summary.json``.

Usage::

    python examples/harbor_stream.py --dataset multistatic|harbor [--cuda]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Observation, Truth  # noqa: E402
from acoustic_sim.water.detection import ProcessingConfig  # noqa: E402
from acoustic_sim.water.propagation import FreeFieldPropagator  # noqa: E402
from acoustic_sim.water.streaming import StreamConfig, StreamingPassive  # noqa: E402
from acoustic_sim.water.suppression import SuppressionConfig  # noqa: E402

DATASETS = {
    # path, band [Hz], ProcessingConfig, edge [m], SuppressionConfig, StreamConfig
    "multistatic": ("output/harbor_multistatic/fine/dataset", (100.0, 760.0),
                    dict(win_s=0.5, hop_s=0.25, grid_dx=2.0, grid_margin=5.0, min_range=1.0, bin_stride=2), 40.0,
                    dict(refine_half=12.0, refine_step=1.0, patch_half=0.5, exclusion_radius=25.0),
                    dict(block_frames=4, history_s=5.0, warmup_s=1.0, interference_every=16)),
    "harbor": ("output/harbor/dataset", (12.0, 78.0), dict(grid_dx=5.0), 150.0, dict(),
               dict(block_frames=1, history_s=30.0, warmup_s=8.0)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--chunk", type=float, default=0.25, help="seconds of data per push")
    ap.add_argument("--hit", type=float, default=5.0)
    ap.add_argument("--out", default="output/harbor_stream")
    args = ap.parse_args()
    path, band, pcd, edge, scd, std = DATASETS[args.dataset]
    out = ROOT / args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    obs = Observation.load(ROOT / f"{path}.observation.npz")
    env = obs.environment
    x0, x1, y0, y1 = env.extent
    prop = FreeFieldPropagator.from_environment(env)
    pc = ProcessingConfig(f_lo=band[0], f_hi=band[1], c=prop.c, propagator=prop, use_cuda=args.cuda,
                          grid_bounds=(x0 + edge, x1 - edge, y0 + edge, y1 - edge), **pcd)
    det = StreamingPassive(obs.geometry, obs.fs, env, pc, SuppressionConfig(**scd), StreamConfig(**std))

    n = int(round(args.chunk * obs.fs))
    frames = []
    for s0 in range(0, obs.traces.shape[1], n):
        t_now = (s0 + n) / obs.fs
        rep = obs.reports
        m = rep.t <= t_now                                       # the reports received by now
        from acoustic_sim.water.ais import AISReports
        frames += det.push(obs.traces[:, s0:s0 + n], AISReports(rep.vessel[m], rep.t[m], rep.xy[m], rep.sog[m], rep.cog[m]))
    T = obs.traces.shape[1] / obs.fs
    blocks = {}
    for f in frames:
        blocks[f.compute_s] = 1
    compute = sum(blocks)
    lat = np.array([f.latency_s for f in frames])

    truth = Truth.load(ROOT / f"{path}.truth.npz")             # read only from here on, for the check
    ids = list(truth.tracks)
    rows = []
    for f in frames:
        pos = np.stack([truth.tracks[i].position_at(f.t)[0] for i in ids])
        for r, (p, s) in enumerate(zip(f.xy, f.score)):
            d = np.hypot(*(pos - p).T)
            j = int(np.argmin(d))
            rows.append({"t": f.t, "x": float(p[0]), "y": float(p[1]), "score": float(s), "rank": r, "nearest": ids[j],
                         "distance_m": float(d[j])})
    per = {}
    for oid in ids:
        hits = [x for x in rows if x["nearest"] == oid and x["distance_m"] <= args.hit]
        tr = truth.tracks[oid]
        per[oid] = {"kind": tr.kind, "radiates": bool(tr.level_pa > 0), "ais": bool(tr.ais), "frames": len(frames),
                    "top_peak_hit": sum(1 for x in hits if x["rank"] == 0) / max(len(frames), 1),
                    "any_peak_hit": len({x["t"] for x in hits}) / max(len(frames), 1),
                    "median_error_m": float(np.median([x["distance_m"] for x in hits])) if hits else None}
    false = sum(1 for x in rows if x["distance_m"] > 10.0) / max(len(frames), 1)
    summary = {"dataset": args.dataset, "record_s": T, "frames": len(frames), "first_frame_t": frames[0].t if frames else None,
               "compute_s": compute, "compute_per_second_of_data": compute / T,
               "latency_s": {"median": float(np.median(lat)), "max": float(lat.max())},
               "false_detections_per_frame": false, "objects": per}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "detections.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "check.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["object", "kind", "radiates", "ais", "frames", "top_peak_hit", "any_peak_hit", "median_error_m"])
        for oid, p in per.items():
            w.writerow([oid, p["kind"], p["radiates"], p["ais"], p["frames"], f"{p['top_peak_hit']:.3f}", f"{p['any_peak_hit']:.3f}",
                        "" if p["median_error_m"] is None else f"{p['median_error_m']:.2f}"])

    print(f"{args.dataset}: {len(frames)} frames from t = {summary['first_frame_t']:.2f} s; compute {compute:.1f} s for {T:.0f} s of data "
          f"({compute / T:.2f} s per s: {'real time' if compute < T else 'NOT real time'}); latency median {np.median(lat):.2f} s, "
          f"max {lat.max():.2f} s; {false:.2f} detections per frame > 10 m from every object")
    print(f"  {'object':14s} {'kind':7s} {'rad':4s} {'AIS':4s} {'top peak hit':>13s} {'any peak hit':>13s} {'median err':>11s}")
    for oid, p in per.items():
        err = "-" if p["median_error_m"] is None else f"{p['median_error_m']:.1f} m"
        print(f"  {oid:14s} {p['kind']:7s} {'y' if p['radiates'] else 'n':4s} {'y' if p['ais'] else 'n':4s} "
              f"{100 * p['top_peak_hit']:12.0f}% {100 * p['any_peak_hit']:12.0f}% {err:>11s}")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(np.where(env.water, 1.0, 0.0), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys_r", vmin=-0.6, vmax=1.2)
    r = obs.geometry.positions()
    ax.plot(r[:, 0], r[:, 1], ".", color="0.3", ms=1.5)
    hit = np.array([[x["x"], x["y"]] for x in rows if x["distance_m"] <= args.hit]).reshape(-1, 2)
    far = np.array([[x["x"], x["y"]] for x in rows if x["distance_m"] > args.hit]).reshape(-1, 2)
    ax.plot(far[:, 0], far[:, 1], "x", color="tab:red", ms=3, alpha=0.5, label=f"> {args.hit:g} m from every object ({len(far)})")
    ax.plot(hit[:, 0], hit[:, 1], "o", color="tab:green", ms=3, label=f"within {args.hit:g} m of an object ({len(hit)})")
    for oid, tr in truth.tracks.items():
        moving = np.ptp(tr.xy, axis=0).max() > 0
        ax.plot(tr.xy[:, 0], tr.xy[:, 1], "-" if moving else "s", color="deepskyblue" if tr.level_pa > 0 else "orange", lw=1.5, ms=4, mfc="none")
        if moving or not oid.startswith("piling"):
            ax.annotate(f"{oid} {100 * per[oid]['any_peak_hit']:.0f}%", tr.xy[-1], fontsize=7)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"{args.dataset}: streaming passive chain, {compute / T:.2f} s compute per s of data, latency median "
                 f"{np.median(lat):.2f} s\n% = frames with a detection within {args.hit:g} m", fontsize=9)
    fig.savefig(out / "check.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The real-time harbor picture: locate every radiator from the traces, track it, map the echoes, check.

The traces are fed in chunks of ``--chunk`` seconds as they would arrive.  No AIS enters the processing:

1. ``StreamingPassive`` (no AIS) locates the radiating objects, ships included: the ``--peaks`` strongest
   peaks of every frame;
2. ``RadiatorTracker`` turns them into causal tracks;
3. ``StreamingEcho`` (``--echo``) removes the direct path of every confirmed track and maps the echoes of
   the silent bodies with every track as an illuminator (the maps summed);
4. the AIS reports are used afterwards only, to label the tracks (``ais_crosscheck``).

The truth is read at the end for the check: a detection within ``--hit`` m (5) of an object is a hit.
Reported: compute time per second of data (below 1 is real time) and the latency of every output.

Outputs in ``--out/<dataset>/``: ``passive.csv``, ``echo.csv``, ``tracks.csv``, ``check.png``, ``summary.json``.

Usage::

    python examples/harbor_stream.py --dataset multistatic|bistatic|simple|harbor [--echo] [--cuda]
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

from acoustic_sim.contract import Observation, Truth  # noqa: E402
from acoustic_sim.water.bistatic import BistaticConfig  # noqa: E402
from acoustic_sim.water.detection import ProcessingConfig  # noqa: E402
from acoustic_sim.water.propagation import FreeFieldPropagator  # noqa: E402
from acoustic_sim.water.streaming import (  # noqa: E402
    EchoStreamConfig, RadiatorTracker, StreamConfig, StreamingEcho, StreamingPassive, TrackerConfig, ais_crosscheck)
from acoustic_sim.water.suppression import SuppressionConfig  # noqa: E402

DATASETS = {
    # path, band [Hz], ProcessingConfig, edge [m], StreamConfig
    "multistatic": ("output/harbor_multistatic/fine/dataset", (100.0, 760.0),
                    dict(win_s=0.5, hop_s=0.25, grid_dx=2.0, grid_margin=5.0, min_range=1.0, bin_stride=2), 40.0,
                    dict(block_frames=8, history_s=5.0, warmup_s=1.0, interference_every=32)),
    "bistatic": ("output/harbor_bistatic/fine/dataset", (100.0, 760.0),
                 dict(win_s=0.5, hop_s=0.25, grid_dx=2.0, grid_margin=5.0, min_range=1.0, bin_stride=2), 40.0,
                 dict(block_frames=8, history_s=5.0, warmup_s=1.0, interference_every=32)),
    "simple": ("output/harbor_simple/fine/dataset", (100.0, 380.0),
               dict(win_s=0.5, hop_s=0.25, grid_dx=2.0, grid_margin=5.0, min_range=1.0, bin_stride=2), 45.0,
               dict(block_frames=8, history_s=5.0, warmup_s=1.0, interference_every=32)),
    "harbor": ("output/harbor/dataset", (12.0, 78.0), dict(grid_dx=5.0), 150.0,
               dict(block_frames=1, history_s=30.0, warmup_s=8.0)),
}
# Tracker per dataset: the defaults suit 0.25 s frames; the 5 m harbor has a frame every 2 s.
TRACKER = {"harbor": dict(gate=15.0, confirm=3, confirm_window_s=8.0, drop_s=8.0, fit_s=20.0, exclusion=60.0)}
# Echo chain settings of stage 0 / stage 1, without the quay image lags (bistatic recording: 77 % of frames
# within 5 m of the UUV without them, 73 % with them, at half the ECA cost).
ECHO = dict(win_s=0.5, hop_s=0.25, grid_dx=2.0, coherent_norm="snr", map_scale="none", exclusion_radius=25.0,
            eca_doppler=True, track_smooth_s=4.0, eca_images=False, eca_gram="prefix", eca_chunk=200)


def check_rows(frames, truth: Truth, chain: str) -> list[dict]:
    ids = list(truth.tracks)
    rows = []
    for f in frames:
        pos = np.stack([truth.tracks[i].position_at(f.t)[0] for i in ids])
        for r, (p, sc) in enumerate(zip(f.xy, f.score)):
            d = np.hypot(*(pos - p).T)
            j = int(np.argmin(d))
            rows.append({"chain": chain, "t": f.t, "x": float(p[0]), "y": float(p[1]), "score": float(sc), "rank": r,
                         "nearest": ids[j], "distance_m": float(d[j])})
    return rows


def per_object(rows, frames, truth, hit):
    out = {}
    times = [f.t for f in frames]
    for oid, tr in truth.tracks.items():
        h = [x for x in rows if x["nearest"] == oid and x["distance_m"] <= hit]
        out[oid] = {"kind": tr.kind, "radiates": bool(tr.level_pa > 0), "frames": len(times),
                    "hit": len({x["t"] for x in h}) / max(len(times), 1),
                    "median_error_m": float(np.median([x["distance_m"] for x in h])) if h else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--echo", action="store_true", help="also run the echo chain on the passive tracks")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--chunk", type=float, default=0.25, help="seconds of data per push")
    ap.add_argument("--peaks", type=int, default=5, help="passive peaks per frame")
    ap.add_argument("--hit", type=float, default=5.0)
    ap.add_argument("--combine", default="sum", choices=("sum", "max", "persistent"),
                    help="echo maps of the illuminators: summed, maximum, or only the most persistent track")
    ap.add_argument("--out", default="output/harbor_stream")
    args = ap.parse_args()
    path, band, pcd, edge, std = DATASETS[args.dataset]
    out = ROOT / args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    obs = Observation.load(ROOT / f"{path}.observation.npz")
    env = obs.environment
    x0, x1, y0, y1 = env.extent
    prop = FreeFieldPropagator.from_environment(env)
    pc = ProcessingConfig(f_lo=band[0], f_hi=band[1], c=prop.c, propagator=prop, use_cuda=args.cuda,
                          grid_bounds=(x0 + edge, x1 - edge, y0 + edge, y1 - edge), **pcd)
    passive = StreamingPassive(obs.geometry, obs.fs, env, pc, SuppressionConfig(), StreamConfig(max_peaks=args.peaks, **std))
    tracker = RadiatorTracker(TrackerConfig(**TRACKER.get(args.dataset, {})))
    echo = None
    if args.echo:
        bc = BistaticConfig(f_lo=band[0], f_hi=band[1], c=prop.c, propagator=prop, use_cuda=args.cuda, **ECHO)
        echo = StreamingEcho(obs.geometry, obs.fs, env, bc, EchoStreamConfig(step_blocks=4, combine="max" if args.combine == "max" else "sum"))

    n = int(round(args.chunk * obs.fs))
    p_frames, e_frames, sent = [], [], {}
    t_passive = t_echo = 0.0
    lat = []
    for s0 in range(0, obs.traces.shape[1], n):
        chunk = obs.traces[:, s0:s0 + n]
        t0 = time.perf_counter()
        new = passive.push(chunk)                                    # no AIS
        for f in new:
            tracker.update(f)
        t_passive += time.perf_counter() - t0
        p_frames += new
        if echo is not None:
            t1 = time.perf_counter()
            conf = tracker.strong()                                       # echoes of loud ships are located too, ~10 dB down
            fixes = {}
            for k, tr in conf.items():
                fixes[k] = [(f[0], f[1]) for f in tr.fixes[sent.get(k, 0):]]
                sent[k] = len(tr.fixes)
            illum = max(conf, key=lambda k: len(conf[k].fixes)) if conf and args.combine == "persistent" else None
            got = echo.push(chunk, fixes, illum)                        # every confirmed radiator is removed
            t_echo += time.perf_counter() - t1
            e_frames += got
        now = (s0 + chunk.shape[1]) / obs.fs
        lat += [now - (f.t + 0.5 * pc.win_s) + (time.perf_counter() - t0) for f in new]
    T = obs.traces.shape[1] / obs.fs
    labels = ais_crosscheck(tracker.confirmed(), obs.reports)          # AIS after the location only

    truth = Truth.load(ROOT / f"{path}.truth.npz")                   # read only from here on, for the check
    prows = check_rows(p_frames, truth, "passive")
    pper = per_object(prows, p_frames, truth, args.hit)
    e_frames = [f for f in e_frames if f.ships]
    erows = check_rows(e_frames, truth, "echo")
    eper = per_object(erows, e_frames, truth, args.hit)
    tracks = []
    for k, tr in tracker.confirmed().items():
        d = {oid: float(np.median([np.hypot(*(truth.tracks[oid].position_at(f[0])[0] - f[1])) for f in tr.fixes])) for oid in truth.tracks}
        nearest = min(d, key=d.get)
        tracks.append({"track": k, "fixes": len(tr.fixes), "first_t": tr.fixes[0][0], "last_t": tr.fixes[-1][0], "score": tr.score,
                       "ais_label": labels.get(k), "true_object": nearest if d[nearest] <= 10.0 else None,
                       "median_error_m": d[nearest]})
    summary = {"dataset": args.dataset, "record_s": T, "ais_used_in_processing": False,
               "compute_per_second_of_data": {"passive": t_passive / T, "echo": t_echo / T, "total": (t_passive + t_echo) / T},
               "passive_latency_s": {"median": float(np.median(lat)), "max": float(np.max(lat))},
               "passive": {"frames": len(p_frames), "objects": pper}, "tracks": tracks,
               "echo": {"frames": len(e_frames), "illuminators": "every confirmed track", "objects": eper} if echo else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for name, rows in (("passive.csv", prows), ("echo.csv", erows)):
        if rows:
            with open(out / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    with open(out / "tracks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tracks[0]) if tracks else ["track"])
        w.writeheader()
        w.writerows(tracks)

    c = summary["compute_per_second_of_data"]
    print(f"{args.dataset}: {T:.0f} s of data; compute per s of data: passive {c['passive']:.2f}, echo {c['echo']:.2f}, total "
          f"{c['total']:.2f} ({'real time' if c['total'] < 1 else 'NOT real time'}); passive latency median "
          f"{np.median(lat):.2f} s; AIS used only to label the tracks")
    print("\npassive (no AIS), frames with a peak within 5 m:  " + ", ".join(
        f"{k} {100 * v['hit']:.0f}%" for k, v in pper.items() if v["radiates"] or v["hit"] > 0))
    print("tracks: " + "; ".join(f"{t['track']} -> {t['true_object']} ({t['median_error_m']:.1f} m), AIS label {t['ais_label']}"
                                 for t in tracks))
    if echo:
        print(f"echo ({len(e_frames)} frames, every track illuminates), frames with a peak within 5 m:  " + ", ".join(
            f"{k} {100 * v['hit']:.0f}%" for k, v in eper.items() if not v["radiates"]))

    fig, axes = plt.subplots(1, 2 if echo else 1, figsize=(15 if echo else 8, 6.5), squeeze=False)
    for ax, (name, rows) in zip(axes[0], [("passive (no AIS) and tracks", prows)] + ([("echo", erows)] if echo else [])):
        ax.imshow(np.where(env.water, 1.0, 0.0), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys_r", vmin=-0.6, vmax=1.2)
        r = obs.geometry.positions()
        ax.plot(r[:, 0], r[:, 1], ".", color="0.3", ms=1.5)
        hitp = np.array([[x["x"], x["y"]] for x in rows if x["distance_m"] <= args.hit]).reshape(-1, 2)
        far = np.array([[x["x"], x["y"]] for x in rows if x["distance_m"] > args.hit]).reshape(-1, 2)
        ax.plot(far[:, 0], far[:, 1], "x", color="tab:red", ms=3, alpha=0.4, label=f"> {args.hit:g} m from every object ({len(far)})")
        ax.plot(hitp[:, 0], hitp[:, 1], "o", color="tab:green", ms=3, label=f"within {args.hit:g} m of an object ({len(hitp)})")
        if name.startswith("passive"):
            for k, tr in tracker.confirmed().items():
                xy = np.array([f[1] for f in tr.fixes])
                ax.plot(xy[:, 0], xy[:, 1], "-", color="k", lw=0.8)
                ax.annotate(f"{k}: {labels.get(k) or 'no AIS'}", xy[-1], fontsize=7, color="k")
        for oid, tr in truth.tracks.items():
            moving = np.ptp(tr.xy, axis=0).max() > 0
            ax.plot(tr.xy[:, 0], tr.xy[:, 1], "-" if moving else "s", color="deepskyblue" if tr.level_pa > 0 else "orange", lw=1.5, ms=4,
                    mfc="none", alpha=0.8)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"{args.dataset}: {name}\ntrue tracks: blue radiating, orange silent; black: passive tracks with their AIS label", fontsize=9)
    fig.savefig(out / "check.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()

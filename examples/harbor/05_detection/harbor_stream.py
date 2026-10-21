#!/usr/bin/env python3
"""The real-time harbor picture on one recording, checked against where the objects really are.

Runs ``acoustic_sim.water.stream`` (the system: passive chain without AIS, radiator tracker, echo chain on the strong
tracks, echo tracker, the measured characteristics of every track, AIS only as a label) on a recording with the
processing config of ``examples/configs/processing/<recording>.json``, the traces fed in chunks as they would arrive.
Then, and only then, the truth is read for the check (``water/scoreboard.py``, the stage 2 definitions):

* every confirmed track is matched to the object nearest over its fixes if within the association radius of the config
  (10 m; 20 m on the 5 m harbor), otherwise it is a false track; false tracks per minute is the false alarm rate;
* Pd per object: the fraction of frames, from the first frame it could have been confirmed, in which a confirmed track
  matched to it is within 5 m (radiator tracks, echo tracks, either);
* diagnostic: the fraction of frames with a map peak within 5 m (passive and echo peaks, as before stage 2);
* on recordings with bodies, the measured characteristics of every object detected in at least 10 frames beside its
  true size and type (``characteristics.json``).

Outputs in ``--out/<name>/``: ``picture.jsonl`` (the operator output), ``summary.json`` (with the ``run`` block),
``characteristics.json``, ``tracks.csv``, ``check_<chain>_tracks.png`` (tracks and peaks over the true tracks),
``check_<chain>_frame_map.png`` (one frame of each chain: its map and peaks), ``check_echo_frame_rays.png`` and
``check_echo_frame_ellipse.png`` (that echo frame's ray paths and equal delay ellipse), ``picture.mp4`` and
``picture_frame.png`` (the operator view; truth drawn only with ``--truth-in-video``).  The default ``--out`` is
``output/reproduce/runs``, the folder ``scripts/reproduce.py`` writes.

Usage::

    python examples/harbor/05_detection/harbor_stream.py --dataset multistatic|bistatic|simple|harbor [--cpu]
        [--observation X.observation.npz --truth X.truth.npz --name multistatic_seed1 --held-out] [--set key=value]
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
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Observation, Provenance, Truth, contract_paths  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water.plotting import HarborPictureView  # noqa: E402
from acoustic_sim.water.scoreboard import peak_hits, score_picture  # noqa: E402
from acoustic_sim.water.stream import load_processing, run  # noqa: E402

CONFIGS = ROOT / "examples" / "configs" / "processing"
RECORDINGS = ("simple", "bistatic", "multistatic", "harbor")


def characteristics_table(records: list[dict], score: dict, truth: Truth, min_frames: int = 10) -> list[dict]:
    """Per object detected (a matched confirmed track within 5 m) in at least *min_frames* frames: its true type and
    size beside the medians of the measured indicators over the frames its tracks were reported in."""
    rows = []
    body = truth.scenario.get("vessels", []) if isinstance(truth.scenario, dict) else []
    size = {v["id"]: (v.get("body_length"), v.get("body_width")) for v in body}
    for oid, o in score["objects"].items():
        hits = o["hits"].get("any", 0)
        if hits < min_frames:
            continue
        tids = set(o["tracks"])
        vals: dict[str, list] = {"size_ratio_db": [], "strength_db": [], "along": [], "across": [], "source_level": [],
                                 "width_major": [], "width_minor": [], "speed": [], "residual": []}
        seen: dict[str, int] = {}
        frames_size: dict[float, float] = {}
        for rec in records:
            for ob in rec["objects"]:
                if not tids & set(ob.get("tracks", {}).values()) or ob["status"] == "tentative":
                    continue
                seen[ob["seen"]] = seen.get(ob["seen"], 0) + 1
                if "echo" in ob:
                    vals["size_ratio_db"].append(ob["echo"]["size_ratio_db"])
                    vals["strength_db"].append(max(ob["echo"]["strength_db"].values()))
                    vals["along"].append(ob["echo"]["extent_m"]["along"])
                    vals["across"].append(ob["echo"]["extent_m"]["across"])
                    frames_size[rec["t"]] = ob["echo"]["size_ratio_db"]
                if "radiator" in ob and ob["radiator"]["source_level_100m_pa"] is not None:
                    vals["source_level"].append(ob["radiator"]["source_level_100m_pa"])
                w = ob["position"].get("halfmax_width_m")
                if w:
                    vals["width_major"].append(w[0])
                    vals["width_minor"].append(w[1])
                if ob["velocity"]["speed"] is not None:
                    vals["speed"].append(ob["velocity"]["speed"])
                if ob["position"]["residual_m"] is not None:
                    vals["residual"].append(ob["position"]["residual_m"])
        tr = truth.tracks[oid]
        L, W = size.get(oid, (None, None))
        med = {k: (float(np.median([x for x in v if x is not None])) if any(x is not None for x in v) else None) for k, v in vals.items()}
        rows.append({"object": oid, "type": tr.kind, "radiates": bool(tr.level_pa > 0), "true_level_pa": float(tr.level_pa),
                     "true_size_m": [L, W] if L else None, "true_speed": float(np.median(tr.speed)),
                     "detected_frames": hits, "seen": seen, "median": med, "size_ratio_per_frame": frames_size})
    return rows


def size_order_check(rows: list[dict], hull: str, uuvs: list[str], piling_prefix: str = "piling") -> dict:
    """Section 6 criterion: on frames where the moored hull, a UUV and a piling are all measured, the fraction in
    which the hull's size ratio is below the piling's (a flatter spectrum)."""
    by = {r["object"]: r["size_ratio_per_frame"] for r in rows}
    h = by.get(hull, {})
    us = [by[u] for u in uuvs if u in by]
    ps = [v for k, v in by.items() if k.startswith(piling_prefix)]
    frames = [t for t in h if any(t in u for u in us) and any(t in p for p in ps)]
    below = [t for t in frames if all(h[t] < p[t] for p in ps if t in p)]
    return {"frames_all_three": len(frames), "hull_below_piling": len(below),
            "fraction": len(below) / len(frames) if frames else None}


def run_recording(name: str, processing: Path, observation: Path, truth_path: Path, out: Path, held_out: bool,
                  use_cuda: bool = True, seed: int | None = None, overrides=(), chunk: float | None = None, video: bool = True,
                  truth_in_video: bool = False, figures: bool = True) -> dict:
    """The picture on one recording, scored; writes everything to *out* and returns the summary."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_processing(processing, overrides)
    obs = Observation.load(observation)
    res = run(obs, cfg, use_cuda=use_cuda, chunk_s=chunk, out=out, provenance=Provenance(data="forward_model", held_out=held_out),
              observation_path=observation, seed=seed)
    pic = res.picture
    truth = Truth.load(truth_path)                                  # read only from here on, for the check
    T = res.record_s
    sc = cfg.scoring
    confirm = {"radiator": cfg.tracker().confirm}
    if cfg.has_echo:
        confirm["echo"] = cfg.echo_tracker().confirm
    score = score_picture(res.records, truth, T, confirm, sc.get("assoc_radius_m", 10.0), sc.get("hit_radius_m", 5.0))
    p_peaks = peak_hits([(f.t, f.xy) for f in pic.p_frames], truth, sc.get("hit_radius_m", 5.0))
    e_frames = [f for f in pic.e_frames if f.ships]
    e_peaks = peak_hits([(f.t, f.xy) for f in e_frames], truth, sc.get("hit_radius_m", 5.0)) if cfg.has_echo else None
    lat = [r["latency_s"] for r in res.records]
    chars = characteristics_table(res.records, score, truth)
    summary = {"recording": name, "record_s": T, "held_out": held_out, "ais_used_in_processing": False,
               "compute_per_second_of_data": res.compute_per_second,
               "latency_s": {"median": float(np.median(lat)), "max": float(np.max(lat))},
               "objects": score["objects"], "tracks": score["tracks"], "false_tracks": score["false_tracks"],
               "definitions": score["definitions"],
               "peaks": {"passive": {"frames": len(pic.p_frames), "objects": p_peaks},
                         "echo": None if e_peaks is None else {"frames": len(e_frames), "objects": e_peaks}},
               "run": dict(res.run, wall_s=res.wall_s)}
    if "moored" in truth.tracks:
        summary["size_order"] = size_order_check(chars, "moored", [k for k in truth.tracks if k.startswith("uuv")])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "characteristics.json").write_text(json.dumps({"recording": name, "run": summary["run"], "objects": chars,
                                                         "size_order": summary.get("size_order")}, indent=2))
    with open(out / "tracks.csv", "w", newline="") as fh:
        rows = score["tracks"]
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["track"])
        w.writeheader()
        w.writerows(rows)
    if figures:
        check_figure(name, obs, truth, pic, score, out)
    if video:
        view = HarborPictureView(obs.environment, obs.geometry, title=name, truth=truth if truth_in_video else None)
        view.records = res.records
        view.save(out / "picture.mp4", frame_png=out / "picture_frame.png")
    return summary


def report(s: dict) -> None:
    c = s["compute_per_second_of_data"]
    print(f"{s['recording']}: {s['record_s']:.0f} s of data; compute per s of data {c['total']:.2f} "
          f"(passive {c['passive']:.2f}, echo {c['echo']:.2f}, characteristics {c['characteristics']:.2f}); latency median "
          f"{s['latency_s']['median']:.2f} s; {'test' if s['held_out'] else 'tuning'} recording")
    for oid, o in s["objects"].items():
        pd = ", ".join(f"{k} {v:.2f}" for k, v in o["pd"].items() if v is not None)
        print(f"  {oid:9s} {o['kind']:7s} Pd {pd or '-'}; tracks {o['tracks']}")
    ft = s["false_tracks"]
    print(f"  false tracks: {ft['count']} ({ft['per_minute']:.1f} per minute)" +
          "".join(f"; {t['track']} at {t['median_xy']}, {t['median_error_m']:.1f} m from {t['nearest_object']}" for t in ft["tracks"]))


def _scan_area(ax, pic) -> None:
    zx, zy = pic.passive.grid[:, 0], pic.passive.grid[:, 1]          # the scan area, so the map fills the image
    ax.set_xlim(zx.min() - 15.0, zx.max() + 15.0)
    ax.set_ylim(zy.min() - 15.0, zy.max() + 15.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (east) [m]")
    ax.set_ylabel("y (north) [m]")


def _chart(ax, obs) -> None:
    """Water and land, the DAS cable and the hydrophones, labelled for the legend."""
    env = obs.environment
    x0, x1, y0, y1 = env.extent
    ax.imshow(np.where(env.water, 1.0, 0.0), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys_r", vmin=-0.6, vmax=1.2)
    lay = obs.geometry
    das = lay.das_positions
    ax.plot(das[:, 0], das[:, 1], ".", color="0.3", ms=2.5, label="DAS channels")
    if lay.n_hydrophones:
        ax.plot(lay.hydrophones[:, 0], lay.hydrophones[:, 1], "^", color="0.3", ms=8, ls="none", label="hydrophones")


def _truth_at(ax, truth: Truth, t: float) -> None:
    for i, (oid, tr) in enumerate(truth.tracks.items()):
        ax.plot(*tr.position_at(t)[0], "o", ms=12, mfc="none", mec="deepskyblue" if tr.level_pa > 0 else "orange", mew=1.8)
    ax.plot([], [], "o", ms=12, mfc="none", mec="deepskyblue", mew=1.8, ls="none", label="true position, radiating (for comparison)")
    ax.plot([], [], "o", ms=12, mfc="none", mec="orange", mew=1.8, ls="none", label="true position, silent (for comparison)")


def check_figure(name: str, obs, truth: Truth, pic, score: dict, out: Path) -> None:
    """``check_<chain>_tracks.png``: the map peaks and the tracks over the true tracks; for one frame of each chain
    ``check_<chain>_frame_map.png`` (its map and peaks), and on the echo chain ``check_echo_frame_rays.png`` (the direct
    and echo ray paths) and ``check_echo_frame_ellipse.png`` (the equal delay ellipse through the strongest peak); one
    plot per image."""
    from acoustic_sim.water.plotting import declutter_labels, draw_frame_ellipse, draw_frame_map, draw_frame_rays
    from acoustic_sim.water.scoreboard import picture_tracks

    matches = {m["track"]: m for m in score["tracks"]}
    has_echo = pic.echo is not None
    tracks = picture_tracks(pic.records)
    panels = [("passive", "passive peaks and radiator tracks", pic.p_frames, "radiator")] + \
        ([("echo", "echo peaks and echo tracks", pic.e_used, "echo")] if has_echo else [])
    for chain, title, frames, kind in panels:
        fig, ax = new_figure((12.0, 10.0))
        _chart(ax, obs)
        pts = np.array([p for f in frames for p in f.xy]).reshape(-1, 2)
        ax.plot(pts[:, 0], pts[:, 1], ".", color="0.5", ms=2, alpha=0.4, label=f"map peaks ({len(pts)})")
        for oid, tr in truth.tracks.items():
            moving = np.ptp(tr.xy, axis=0).max() > 0
            ax.plot(tr.xy[:, 0], tr.xy[:, 1], "-" if moving else "s", color="deepskyblue" if tr.level_pa > 0 else "orange", lw=1.5,
                    ms=5, mfc="none", alpha=0.8)
        ax.plot([], [], "-", color="deepskyblue", lw=1.5, label="true track, radiating")
        ax.plot([], [], "-", color="orange", lw=1.5, label="true track or position, silent")
        anns = []
        for k, t in tracks.items():
            if t["kind"] != kind:
                continue
            xy = np.array([f[1] for f in t["fixes"]])
            m = matches.get(k, {})
            if m.get("false_track"):
                ax.plot(xy[:, 0], xy[:, 1], "--", color="m", lw=1.4)
                anns.append(ax.annotate(f"{k}: false", xy[-1], xytext=(6, 4), textcoords="offset points", color="m",
                                        bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.7)))
            else:
                ax.plot(xy[:, 0], xy[:, 1], "-", color="k", lw=1.0)
                anns.append(ax.annotate(f"{k}: {m.get('true_object')}", xy[-1], xytext=(6, 4), textcoords="offset points",
                                        color="k", bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.7)))
        ax.plot([], [], "-", color="k", lw=1.0, label="confirmed track: matched object")
        ax.plot([], [], "--", color="m", lw=1.4, label="false track")
        _scan_area(ax, pic)
        ax.legend(loc="lower left")
        ax.set_title(f"{name}: {title}")
        declutter_labels(ax, anns)
        save_figure(fig, out / f"check_{chain}_tracks.png", recording=name)
    geo = []
    if pic.p_frames:
        geo.append(("passive", pic.p_frames[len(pic.p_frames) // 2]))
    e_frames = [f for f in pic.e_frames if f.ships]
    if has_echo and e_frames:
        geo.append(("echo", e_frames[len(e_frames) // 2]))
    for chain, f in geo:
        if f.B is None:
            continue
        rads = f.ships if chain == "echo" else {}
        grid = pic.passive.grid if chain == "passive" else pic.echo.grid
        fig, ax = new_figure((12.0, 10.0))
        _chart(ax, obs)
        im = draw_frame_map(ax, grid, f.B, f.xy)
        _truth_at(ax, truth, f.t)
        fig.colorbar(im, ax=ax, shrink=0.7, label="map level [dB re frame peak]")
        _scan_area(ax, pic)
        ax.legend(loc="lower left")
        ax.set_title(f"{name}: {chain} map at t = {f.t:.2f} s and its peaks")
        save_figure(fig, out / f"check_{chain}_frame_map.png", recording=name)
        if chain != "echo":
            continue
        fig, ax = new_figure((12.0, 10.0))
        _chart(ax, obs)
        draw_frame_rays(ax, obs.geometry, rads, f.xy, echo=True)
        _truth_at(ax, truth, f.t)
        _scan_area(ax, pic)
        ax.legend(loc="lower left")
        ax.set_title(f"{name}: ray paths of the echo frame at t = {f.t:.2f} s")
        save_figure(fig, out / "check_echo_frame_rays.png", recording=name)
        fig, ax = new_figure((12.0, 10.0))
        _chart(ax, obs)
        draw_frame_ellipse(ax, obs.geometry, rads, f.xy)
        _truth_at(ax, truth, f.t)
        _scan_area(ax, pic)
        ax.legend(loc="lower left")
        ax.set_title(f"{name}: equal delay ellipse through the strongest echo peak at t = {f.t:.2f} s")
        save_figure(fig, out / "check_echo_frame_ellipse.png", recording=name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=RECORDINGS, help="which processing config (and default recording)")
    ap.add_argument("--processing", default=None, help="processing config (default examples/configs/processing/<dataset>.json)")
    ap.add_argument("--observation", default=None, help="observation file (default: the config's recording)")
    ap.add_argument("--truth", default=None, help="truth file for the check (default: beside the observation)")
    ap.add_argument("--name", default=None, help="output folder name (default: the dataset)")
    ap.add_argument("--held-out", action="store_true", help="a test recording (no parameter was set on it)")
    ap.add_argument("--seed", type=int, default=None, help="noise seed of the recording (recorded in the run block)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a processing parameter (recorded)")
    ap.add_argument("--cpu", action="store_true", help="process on the CPU (default: the GPU, falling back to the CPU)")
    ap.add_argument("--cuda", action="store_true", help="kept for old command lines: the GPU is the default")
    ap.add_argument("--chunk", type=float, default=None, help="seconds of data per push (default: the config's)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--truth-in-video", action="store_true", help="draw the true positions (dashed) in the operator view")
    ap.add_argument("--out", default="output/reproduce/runs")
    a = ap.parse_args()
    processing = Path(a.processing) if a.processing else CONFIGS / f"{a.dataset}.json"
    cfg = json.loads(processing.read_text())
    observation = Path(a.observation) if a.observation else ROOT / cfg["observation"]
    truth = Path(a.truth) if a.truth else contract_paths(observation)[1]
    t0 = time.time()
    s = run_recording(a.name or a.dataset, processing, observation, truth, ROOT / a.out / (a.name or a.dataset), a.held_out,
                      use_cuda=not a.cpu, seed=a.seed, overrides=a.set, chunk=a.chunk, video=not a.no_video,
                      truth_in_video=a.truth_in_video)
    report(s)
    print(f"wrote {ROOT / a.out / (a.name or a.dataset)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

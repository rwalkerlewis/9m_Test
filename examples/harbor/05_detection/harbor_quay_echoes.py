#!/usr/bin/env python3
"""The false echo tracks against the chart (prompt ``run_all_3``, task 2, section 1): a measurement, nothing here feeds
the processing.

The false tracks left after the false tracks study are mostly echo tracks at fixed places on every noise seed, which is
geometry, not noise.  For every confirmed false echo track (``false_tracks.tracks`` of every run's ``summary.json``,
kind ``echo``) of every recording with an echo chain, on both noise models (``output/reproduce/runs/<recording>/`` and
``<recording>_noise2/``), from its fixes in the run's ``picture.jsonl``:

* ``d_image_m``: at every fix, the distance to the nearest first order quay image (``bistatic.clutter_images``, the
  four grid directions, as the ECA's ``eca_images`` used them) of every radiator track confirmed or static in that frame
  (the tracked ships, what the processing has: no truth), then the median over the fixes;
* ``d_land_m``: the distance from the track's median position to the nearest land cell of the chart
  (``ChartEnvironment.distance_to_land``);
* ``d_body_m``: the distance from the median position to the nearest static body's outline (the mask the forward
  model drew, ``scoreboard.ObjectDistance``: a diagnostic from the truth), and that body.

Diagnostics beside the classes (not classes): ``d_ship_m``, the median over the fixes of the distance to the nearest
tracked ship (the echo chain excludes 25 m around each; a residual of its direct path lands just outside); ``d_cable_m``,
the distance of the median position to the nearest DAS channel (a place fixed relative to the cable is geometry too);
``cluster``, the false echo tracks of a family grouped across seeds and noise models (median positions within 15 m of
a cluster's first member), with the number of recordings each cluster appears on (the premise: fixed places on every
seed); and of every quay image used, the fraction in water and the fraction inside the echo chain's scan grid (an
image on land is never scanned, so ``quay_image`` cannot fire for it).

Added after the class counts were seen (the decision was "neither", and the prompt then asks what does account for
them; diagnostics, no threshold is set on them): ``d_grid_edge_m``, the distance of the median position to the
nearest boundary cell of the echo chain's scan grid (a peak there is energy from outside the scanned area); and on the
simple harbor under harbor_v2, ``d_machinery_m``, the distance to the quay machinery of that noise model
(``harbor_noise_model.MACHINE``, a fixed source the old model does not have).

Classes, fixed before any track was measured, the first that applies: ``quay_image`` (``d_image_m`` within 10 m),
``land_edge`` (``d_land_m`` within 10 m), ``body`` (``d_body_m`` within 20 m: an echo of a static body more than the
association radius of 10 m away from its outline, a scoring miss), ``elsewhere``.  ``overlaps`` counts the tracks that
meet more than one condition.

Decision rule, fixed before measuring (prompt): the quay images account for the false echo tracks if at least half of
them (over every recording and both noise models) are ``quay_image``; the land edge accounts for them if at least half
are ``quay_image`` or ``land_edge`` with ``land_edge`` the larger part; otherwise neither does, and the task reports what
does and stops after the measurement.

Outputs in ``output/studies/quay_echoes/``: ``results.json`` (one row per false track, the counts per class and noise
model, the decision), one figure per recording with false echo tracks (``map_<recording>.png``: the chart, the false
tracks of both noise models, the quay images of the tracked ships and the static bodies), ``run.log``.

    python examples/harbor/05_detection/harbor_quay_echoes.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from acoustic_sim.contract import Observation, Truth, contract_paths  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water.bistatic import build_grid, clutter_images  # noqa: E402
from acoustic_sim.water.scoreboard import FIRM, ObjectDistance, picture_tracks  # noqa: E402
from acoustic_sim.water.stream import load_processing, read_picture  # noqa: E402

OUT = ROOT / "output" / "studies" / "quay_echoes"
RUNS = ROOT / "output" / "reproduce" / "runs"
IMAGE_M = 10.0
LAND_M = 10.0
BODY_M = 20.0
CLUSTER_M = 15.0
MACHINE = [(250.0, 375.0)]   # the quay machinery of harbor_v2 on the simple harbor (harbor_noise_model.MACHINE)
CLASSES = ("quay_image", "land_edge", "body", "elsewhere")
CRITERIA = {
    "classes": f"quay_image: median distance to the nearest first order quay image of a tracked ship within {IMAGE_M:g} m; "
               f"land_edge: median position within {LAND_M:g} m of land; body: within {BODY_M:g} m of a static body's "
               "outline; elsewhere; the first that applies",
    "decision": "quay images account for the false echo tracks if at least half are quay_image; the land edge if at "
                "least half are quay_image or land_edge with land_edge the larger part; otherwise neither",
}


def classify(d_image: float | None, d_land: float, d_body: float | None) -> str:
    """The class of a false echo track from its three distances (the first that applies)."""
    if d_image is not None and d_image <= IMAGE_M:
        return "quay_image"
    if d_land <= LAND_M:
        return "land_edge"
    if d_body is not None and d_body <= BODY_M:
        return "body"
    return "elsewhere"


def decide(counts: dict[str, int]) -> str:
    """``"quay_image"``, ``"land_edge"`` or ``"neither"`` from the class counts (the decision rule above)."""
    n = sum(counts.values())
    if n == 0:
        return "neither"
    if counts.get("quay_image", 0) >= 0.5 * n:
        return "quay_image"
    if counts.get("quay_image", 0) + counts.get("land_edge", 0) >= 0.5 * n and counts.get("land_edge", 0) > counts.get("quay_image", 0):
        return "land_edge"
    return "neither"


def ship_positions(records: list[dict]) -> dict[float, list[np.ndarray]]:
    """Per frame time the positions of the radiator tracks confirmed or static in that frame."""
    out: dict[float, list[np.ndarray]] = {}
    for rec in records:
        ps = []
        for o in rec["objects"]:
            tid = o.get("tracks", {}).get("radiator")
            st = o.get("track_state", {}).get(tid) if tid else None
            if st is not None and st["status"] in FIRM:
                ps.append(np.asarray(st["xy"], float))
        out[round(rec["t"], 6)] = ps
    return out


def image_distance(env, ships: dict[float, list[np.ndarray]], fixes: list) -> tuple[float | None, list]:
    """Median over *fixes* ``(t, xy)`` of the distance to the nearest quay image of a tracked ship at that time; and
    the images used (for the figure)."""
    ds, used = [], []
    for t, xy in fixes:
        imgs = [clutter_images(env, s) for s in ships.get(round(t, 6), [])]
        imgs = [i for i in imgs if len(i)]
        if not imgs:
            continue
        im = np.vstack(imgs)
        used.append(im)
        ds.append(float(np.min(np.hypot(*(im - np.asarray(xy, float)[None, :]).T))))
    return (float(np.median(ds)) if ds else None), used


def ship_distance(ships: dict[float, list[np.ndarray]], fixes: list) -> float | None:
    """Median over *fixes* of the distance to the nearest tracked ship at that time."""
    ds = [min(float(np.hypot(*(np.asarray(xy, float) - s))) for s in ships[round(t, 6)])
          for t, xy in fixes if ships.get(round(t, 6))]
    return float(np.median(ds)) if ds else None


def clusters(rows: list[dict]) -> dict[str, dict]:
    """Group the rows of one family by median position (within ``CLUSTER_M`` of a cluster's first member); sets
    ``row["cluster"]`` and returns per cluster its centre and the recordings it appears on."""
    out: dict[str, dict] = {}
    for r in rows:
        xy = np.asarray(r["median_xy"], float)
        hit = next((k for k, c in out.items() if np.hypot(*(xy - c["first_xy"])) <= CLUSTER_M), None)
        if hit is None:
            hit = f"{r['recording'].split('_seed')[0]}:c{len(out) + 1}"
            out[hit] = {"first_xy": xy, "members": []}
        out[hit]["members"].append(f"{r['recording']} ({r['noise_model']}) {r['track']}")
        r["cluster"] = hit
    return {k: {"first_xy": c["first_xy"].tolist(), "n": len(c["members"]),
                "recordings": len({m.split(' ')[0] + m.split(' ')[1] for m in c["members"]}), "members": c["members"]}
            for k, c in out.items()}


def grid_edge(grid: np.ndarray) -> np.ndarray:
    """The boundary cells of a scan grid (fewer than four neighbours one grid step away)."""
    from scipy.spatial import cKDTree

    dx = float(np.min(np.diff(np.unique(grid[:, 0])))) if len(grid) > 1 else 1.0
    tree = cKDTree(grid)
    n = np.array([len(v) - 1 for v in tree.query_ball_point(grid, dx * 1.01)])
    return grid[n < 4]


def measure(name: str, model: str, run_dir: Path, env, truth: Truth, dist: ObjectDistance, layout=None,
            edge: np.ndarray | None = None, machinery: list | None = None) -> list[dict]:
    s = json.loads((run_dir / "summary.json").read_text())
    false = [m for m in s["false_tracks"].get("tracks", []) if m.get("kind") == "echo"]
    if not false:
        return []
    _, records = read_picture(run_dir / "picture.jsonl")
    tracks = picture_tracks(records)
    ships = ship_positions(records)
    static = [k for k, c in dist.cells.items() if c is not None and len(c)]
    rows = []
    for m in false:
        fixes = tracks[m["track"]]["fixes"]
        xy = np.median(np.array([f[1] for f in fixes], float), axis=0)
        d_img, used = image_distance(env, ships, fixes)
        d_land = float(env.distance_to_land(xy)[0])
        db = {k: dist(k, fixes[0][0], xy) for k in static}
        body = min(db, key=db.get) if db else None
        d_body = db[body] if body else None
        d_ship = ship_distance(ships, fixes)
        d_cable = float(np.min(np.hypot(*(layout.das_positions - xy[None, :]).T))) if layout is not None and len(layout.das_positions) else None
        d_edge = float(np.min(np.hypot(*(edge - xy[None, :]).T))) if edge is not None and len(edge) else None
        d_mach = float(min(np.hypot(*(xy - np.asarray(m, float))) for m in machinery)) if machinery else None
        conds = [d_img is not None and d_img <= IMAGE_M, d_land <= LAND_M, d_body is not None and d_body <= BODY_M]
        rows.append({"recording": name, "noise_model": model, "held_out": s["held_out"], "track": m["track"],
                     "median_xy": xy.round(1).tolist(), "fixes": len(fixes), "first_t": fixes[0][0], "last_t": fixes[-1][0],
                     "d_image_m": None if d_img is None else round(d_img, 2), "d_land_m": round(d_land, 2),
                     "nearest_body": body, "d_body_m": None if d_body is None else round(d_body, 2),
                     "nearest_object": m.get("nearest_object"), "median_error_m": round(m["median_error_m"], 2),
                     "d_ship_m": None if d_ship is None else round(d_ship, 2),
                     "d_cable_m": None if d_cable is None else round(d_cable, 2),
                     "d_grid_edge_m": None if d_edge is None else round(d_edge, 2),
                     "d_machinery_m": None if d_mach is None else round(d_mach, 2),
                     "conditions_met": int(sum(conds)), "class": classify(d_img, d_land, d_body),
                     "_images": np.vstack(used) if used else np.empty((0, 2))})
    return rows


def figure(name: str, rows: list[dict], obs, dist: ObjectDistance, path: Path, grid: np.ndarray | None = None,
           machinery: list | None = None) -> None:
    env = obs.environment
    fig, ax = new_figure((12.0, 10.0))
    x0, x1, y0, y1 = env.extent
    ax.imshow(np.where(env.water, 1.0, 0.0), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys_r", vmin=-0.6, vmax=1.2)
    das = obs.geometry.das_positions
    ax.plot(das[:, 0], das[:, 1], ".", color="0.3", ms=2.5, label="DAS channels")
    im = np.vstack([r["_images"] for r in rows]) if rows else np.empty((0, 2))
    if len(im):
        ax.plot(im[:, 0], im[:, 1], ".", color="tab:green", ms=2, alpha=0.3, label="quay images of the tracked ships")
    first = True
    for k, c in dist.cells.items():
        if c is not None and len(c):
            ax.plot(c[:, 0], c[:, 1], "s", color="orange", ms=1.5, label="static bodies (truth)" if first else None)
            first = False
    for model, mk, col in (("old", "o", "tab:red"), ("harbor_v2", "x", "tab:purple")):
        sel = [r for r in rows if r["noise_model"] == model]
        if sel:
            p = np.array([r["median_xy"] for r in sel])
            ax.plot(p[:, 0], p[:, 1], mk, color=col, ms=11, mew=2, mfc="none", ls="none",
                    label=f"false echo tracks, {model} noise model ({len(sel)})")
    if grid is not None and len(grid):
        gx0, gx1, gy0, gy1 = grid[:, 0].min(), grid[:, 0].max(), grid[:, 1].min(), grid[:, 1].max()
        ax.plot([gx0, gx1, gx1, gx0, gx0], [gy0, gy0, gy1, gy1, gy0], "--", color="tab:blue", lw=1.5,
                label="bounds of the echo chain's scan grid")
    if machinery:
        mp = np.asarray(machinery, float)
        ax.plot(mp[:, 0], mp[:, 1], "*", color="k", ms=16, ls="none", label="quay machinery (harbor_v2 only)")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (east) [m]")
    ax.set_ylabel("y (north) [m]")
    ax.set_title(f"{name}: false echo tracks (median positions, every seed) against the quay images and the land"
                 + ("\n(the quay images lie on land beyond the chart's edge where off the axes)" if len(im) else ""))
    ax.legend(loc="best", fontsize=11)
    save_figure(fig, path, recording=name)


def main() -> None:
    import reproduce as R

    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "run.log", "a")
    say = lambda m: (print(m), log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n"), log.flush())  # noqa: E731
    say("measuring the false echo tracks of every recording against the quay images, the land and the static bodies")
    rows: list[dict] = []
    by_family: dict[str, list[dict]] = {}
    ctx = {}
    for r in R.recordings():
        for model, run, ds in (("old", RUNS / r.name, r.dataset), ("harbor_v2", RUNS / f"{r.name}_noise2", r.dataset_noise2)):
            if not (run / "summary.json").exists():
                continue
            if r.family not in ctx:
                obs_p, truth_p = contract_paths(r.dataset.with_suffix(".npz"))
                obs, truth = Observation.load(obs_p), Truth.load(truth_p)
                proc = load_processing(ROOT / "examples" / "configs" / "processing" /
                                       f"{'multistatic' if r.family == 'multistatic_b' else r.family}.json")
                grid = build_grid(obs.environment, proc.bistatic(obs.environment, False)) if proc.has_echo else None
                ctx[r.family] = (obs, truth, ObjectDistance(truth), grid)
            obs, truth, dist, grid = ctx[r.family]
            mach = MACHINE if (model == "harbor_v2" and r.family == "simple") else None
            got = measure(r.name, model, run, obs.environment, truth, dist, obs.geometry, grid_edge(grid) if grid is not None else None, mach)
            rows += got
            by_family.setdefault(r.family, []).extend(got)
            say(f"{r.name} ({model}): {len(got)} false echo tracks: " + ", ".join(f"{g['track']} {g['class']}" for g in got))
    cl = {}
    where = {}
    for fam, rr in by_family.items():
        if not rr:
            continue
        cl.update(clusters(rr))
        obs, grid = ctx[fam][0], ctx[fam][3]
        env = obs.environment
        im = np.vstack([x["_images"] for x in rr if len(x["_images"])]) if any(len(x["_images"]) for x in rr) else np.empty((0, 2))
        g_dx = float(np.min(np.diff(np.unique(grid[:, 0])))) if len(grid) > 1 else 1.0
        in_grid = (np.array([np.min(np.hypot(*(grid - p[None, :]).T)) <= g_dx for p in im]) if len(im) else np.zeros(0, bool))
        where[fam] = {"images": int(len(im)), "in_water": float(np.mean(env.is_water(im))) if len(im) else None,
                      "in_scan_grid": float(np.mean(in_grid)) if len(im) else None}
        say(f"{fam}: {len(im)} quay images at the false tracks' fixes, {where[fam]['in_water']} in water, "
            f"{where[fam]['in_scan_grid']} inside the echo scan grid")
    figs = []
    for fam, rr in by_family.items():
        if rr:
            obs, _, dist, grid = ctx[fam]
            p = OUT / f"map_{fam}.png"
            figure(fam, rr, obs, dist, p, grid, MACHINE if fam == "simple" else None)
            figs.append(p.name)
    counts = {m: {c: sum(1 for x in rows if x["noise_model"] == m and x["class"] == c) for c in CLASSES} for m in ("old", "harbor_v2")}
    total = {c: counts["old"][c] + counts["harbor_v2"][c] for c in CLASSES}
    res = {"criteria": CRITERIA, "radii_m": {"image": IMAGE_M, "land": LAND_M, "body": BODY_M},
           "provenance": {"truth_inputs": ["static body outlines (the class body only)"], "simulator_knowledge": [],
                          "data": "forward_model", "held_out": True},
           "note": "a measurement on the runs reproduce.py made, tuning and test recordings alike; it sets no parameter",
           "clusters": cl, "images_where": where,
           "counts": counts, "counts_total": total, "n": len(rows),
           "overlaps": sum(1 for x in rows if x["conditions_met"] > 1),
           "decision": decide(total), "figures": figs,
           "tracks": [{k: v for k, v in x.items() if not k.startswith("_")} for x in rows]}
    recs_of = {}
    for x in rows:
        recs_of.setdefault(x["recording"].split("_seed")[0], set()).add((x["recording"], x["noise_model"]))
    res["after_the_counts"] = {
        "note": "diagnostics added after the class counts were seen; no threshold is set on them",
        "grid_edge_within_5m": sum(1 for x in rows if x["d_grid_edge_m"] is not None and x["d_grid_edge_m"] <= 5.0),
        "harbor_v2_only_clusters": [k for k, c in cl.items() if all("(harbor_v2)" in m for m in c["members"])],
        "repeated_clusters": {k: {"first_xy": c["first_xy"], "recordings": c["recordings"],
                                  "of_runs_with_false_echo_tracks_in_family": len(recs_of[k.split(":")[0]]),
                                  "median_d_grid_edge_m": float(np.median([x["d_grid_edge_m"] for x in rows if x["cluster"] == k])),
                                  "median_d_land_m": float(np.median([x["d_land_m"] for x in rows if x["cluster"] == k])),
                                  "median_d_ship_m": float(np.median([x["d_ship_m"] for x in rows if x["cluster"] == k and x["d_ship_m"] is not None])),
                                  "first_t_median": float(np.median([x["first_t"] for x in rows if x["cluster"] == k]))}
                              for k, c in cl.items() if c["recordings"] >= 3},
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    say(f"{len(rows)} false echo tracks; classes {total}; decision: {res['decision']}")


if __name__ == "__main__":
    main()

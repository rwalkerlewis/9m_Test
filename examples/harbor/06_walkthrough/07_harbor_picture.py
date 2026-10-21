#!/usr/bin/env python3
"""Step 7, the harbor picture: everything located, and how well.

All outputs together: the radiator tracks (with their AIS label from step 4) and every echo detection
(the 3 strongest peaks of every frame), over the true positions.  The table scores each object: the fraction of
frames with a detection within 5 m of it
(each detection assigned to its nearest object first), and the median error.  Every track is scored too: matched to the object
nearest over its fixes when within 10 m (20 m on the 5 m harbor; ``harbor_stream.match_tracks``), otherwise a false track, drawn dashed
and listed in the table with its median position.  Also the compute time per second of data (below 1: real time).
The scores are also written to ``07_harbor_picture.json``.

    python examples/harbor/06_walkthrough/07_harbor_picture.py [--dataset multistatic] [--cuda]
"""

import json

import numpy as np
import matplotlib.pyplot as plt

import _common as c
from harbor_stream import ASSOC, ASSOC_RADIUS, match_tracks

a = c.args(__doc__)
obs, tru = c.observation(a.dataset), c.truth(a.dataset)
run = c.products(a)
fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
ax = axes[0]
c.chart(ax, obs)
c.truth_overlay(ax, tru, labels=False, alpha=0.6)
R_ASSOC = ASSOC.get(a.dataset, ASSOC_RADIUS)
matches = {m["track"]: m for m in match_tracks(run["tracks"], tru, R_ASSOC)}
false_tracks = [m for m in matches.values() if m["false_track"]]
for k, fx in run["tracks"].items():
    xy = np.array([f[1] for f in fx])
    if matches[k]["false_track"]:
        ax.plot(xy[:, 0], xy[:, 1], "--", color=c.MISS, lw=2.0)
        ax.annotate(f"{k} (unmatched: no object within {R_ASSOC:g} m)", xy[-1], xytext=(5, 4), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", color=c.MISS)
    else:
        ax.plot(xy[:, 0], xy[:, 1], "-", color=c.TRACK, lw=2.5)
        ax.annotate(f"{k} ({run['labels'].get(k) or 'no AIS'})", xy[-1], xytext=(5, 4), textcoords="offset points", fontsize=8.5,
                    fontweight="bold")


def score(frames):
    """Per object: the fraction of frames with a detection within 5 m of it and the median error.  Every detection
    is first assigned to its nearest active object, so a detection of a neighbour (pilings 2 to 5 m apart) does
    not count for this one."""
    ids = list(tru.tracks)
    hits = {oid: [] for oid in ids}
    for f in frames:
        if not len(f[1]):
            continue
        pos = np.stack([tru.tracks[o].position_at(f[0])[0] for o in ids])
        act = np.array([bool(tru.tracks[o].active_at(np.atleast_1d(f[0]))[0]) for o in ids])
        d = np.hypot(f[1][:, None, 0] - pos[None, :, 0], f[1][:, None, 1] - pos[None, :, 1])
        d[:, ~act] = np.inf
        j = d.argmin(axis=1)
        dj = d[np.arange(len(j)), j]
        for o in set(j[dj <= 5.0]):
            hits[ids[o]].append(float(dj[(j == o) & (dj <= 5.0)].min()))
    return {oid: (len(h) / max(len(frames), 1), float(np.median(h)) if h else None) for oid, h in hits.items()}


p_rows = score(run["passive"]["frames"])
e_rows = score(run["echo"]["frames"]) if run["echo"] else {}
if run["echo"]:
    pts = np.array([p for f in run["echo"]["frames"] for p in f[1]]).reshape(-1, 2)
    near = np.array([min(np.hypot(*(tru.tracks[o].position_at(f[0])[0] - p)) for o in tru.tracks if tru.tracks[o].level_pa == 0)
                     for f in run["echo"]["frames"] for p in f[1]])
    ax.plot(pts[near <= 5, 0], pts[near <= 5, 1], "o", color=c.HIT, ms=4, label="echo detection within 5 m of a body")
    ax.plot(pts[near > 5, 0], pts[near > 5, 1], "x", color=c.MISS, ms=3, alpha=0.4, label="echo detection elsewhere")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)
ax.set_title("Located: radiator tracks (black, AIS label in brackets; red dashed: false tracks) and echo detections", fontsize=10)
ax = axes[1]
ax.axis("off")
rows = []
for oid, tr in tru.tracks.items():
    kind = "radiates" if tr.level_pa > 0 else "silent"
    pr, er = p_rows[oid], e_rows.get(oid, (np.nan, None))
    use = pr if tr.level_pa > 0 else er
    rows.append([oid, kind, "yes" if tr.ais else "no", f"{100 * use[0]:.0f} %", "-" if use[1] is None else f"{use[1]:.1f}",
                 "passive" if tr.level_pa > 0 else "echo"])
n_obj = len(rows)
for m in false_tracks:
    x, y = m["median_xy"]
    rows.append([m["track"], "false track", "-", f"at ({x:.0f}, {y:.0f})", f"{m['median_error_m']:.1f} to {m['nearest_object']}",
                 "tracker"])
tab = ax.table(cellText=rows, colLabels=["object", "", "AIS", "frames found", "median err [m]", "by"], loc="upper center",
               cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(9)
tab.scale(1, 1.35)
for (r, col), cell in tab.get_celld().items():
    if r > 0 and col == 3 and r <= n_obj:
        v = float(rows[r - 1][3].rstrip(" %")) / 100
        cell.set_facecolor(plt.cm.RdYlGn(0.15 + 0.7 * v))
    elif r > n_obj:
        cell.set_facecolor("#f4d6d6")
T = run["record_s"]
sp = run["seconds"]
ax.set_title(f"Scored against the true positions (within 5 m); {len(false_tracks)} false track(s) (no object within "
             f"{R_ASSOC:g} m)\ncompute per s of data: passive {sp['passive'] / T:.2f} s, "
             f"echo {sp['echo'] / T:.2f} s (timed in the run that produced the cache)", fontsize=10)
fig.suptitle(f"Step 7 - the harbor picture ({a.dataset})", fontsize=12)
fig.tight_layout()
c.save(fig, c.out_dir(a) / "07_harbor_picture.png", c.NOTE)
(c.out_dir(a) / "07_harbor_picture.json").write_text(json.dumps({
    "dataset": a.dataset, "detect_radius_m": 5.0, "track_association_radius_m": R_ASSOC,
    "objects": {oid: {"by": r[5], "frames_found": (p_rows if r[5] == "passive" else e_rows).get(oid, (None, None))[0],
                      "median_error_m": (p_rows if r[5] == "passive" else e_rows).get(oid, (None, None))[1]}
                for oid, r in zip(tru.tracks, rows[:n_obj])},
    "tracks": list(matches.values()), "false_tracks": {"count": len(false_tracks), "tracks": false_tracks},
    "note": "scored against the truth after the run; the processing reads no truth"}, indent=2))

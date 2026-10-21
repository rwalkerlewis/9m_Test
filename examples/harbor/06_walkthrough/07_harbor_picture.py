#!/usr/bin/env python3
"""Step 7, the harbor picture: everything located, and how well.

All outputs together, as tracks: the radiator tracks (with their AIS label from step 4) and the echo tracks (the echo
peaks of step 6 tracked; the raw peaks faint underneath), over the true positions.  Scored with the stage 2 definitions
(``water/scoreboard.py``, ``score_picture``): every confirmed track is matched to the object nearest over its fixes when
within the association radius of the processing config (10 m; 20 m on the 5 m harbor), otherwise it is a false track
(drawn red dashed and listed in the table with its median position); Pd per object is the fraction of frames, from the
first frame it could have been confirmed, in which a confirmed track matched to it is within 5 m.  Also the compute time
per second of data (below 1: real time).  The scores are also written to ``07_harbor_picture.json``.

    python examples/harbor/06_walkthrough/07_harbor_picture.py [--dataset multistatic]
"""

import json

import numpy as np
import matplotlib.pyplot as plt

import _common as c

from acoustic_sim.water.scoreboard import picture_tracks, score_picture

a = c.args(__doc__)
obs, tru = c.observation(a.dataset), c.truth(a.dataset)
run = c.products(a)
sc = run["scoring"]
score = score_picture(run["records"], tru, run["record_s"], run["confirm"], sc["assoc_radius_m"], sc["hit_radius_m"])
matches = {m["track"]: m for m in score["tracks"]}
tracks = picture_tracks(run["records"])
false_tracks = score["false_tracks"]["tracks"]
F = c.Figures(a, "07_harbor_picture", f"Step 7 - the harbor picture ({a.dataset})", c.NOTE)
fig, ax = F.new("map")
c.chart(ax, obs)
c.zoom(ax, run["passive"]["grid"])
c.truth_overlay(ax, tru, labels=False, alpha=0.6)
if run["echo"]:
    pts = np.array([p for f in run["echo"]["frames"] for p in f[1]]).reshape(-1, 2)
    ax.plot(pts[:, 0], pts[:, 1], "x", color="0.5", ms=3, alpha=0.25, label="echo map peaks (not tracks)")
labels = run["labels"]
for k, t in tracks.items():
    xy = np.array([f[1] for f in t["fixes"]])
    echo = t["kind"] == "echo"
    if matches[k]["false_track"]:
        ax.plot(xy[:, 0], xy[:, 1], "--", color=c.MISS, lw=2.0)
        ax.annotate(f"{k} (false)", xy[-1], xytext=(5, 4), textcoords="offset points", fontsize=8, fontweight="bold", color=c.MISS)
    else:
        ax.plot(xy[:, 0], xy[:, 1], "-" if not echo else ".-", color=c.TRACK if not echo else c.HIT, lw=2.5 if not echo else 1.5, ms=4)
        name = f"{k} ({labels.get(k) or 'no AIS'})" if not echo else f"{k} (echo)"
        ax.annotate(name, xy[-1], xytext=(5, 4), textcoords="offset points", fontsize=8.5, fontweight="bold",
                    color=c.TRACK if not echo else c.HIT)
ax.plot([], [], "-", color=c.TRACK, lw=2.5, label="radiator track")
ax.plot([], [], ".-", color=c.HIT, lw=1.5, label="echo track")
ax.plot([], [], "--", color=c.MISS, lw=2.0, label="false track")
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Located: radiator tracks (AIS label in brackets), echo tracks, false tracks (red dashed)", fontsize=10)
fig, ax = F.new("table", size=(10, 0.9 + 0.33 * (len(score["objects"]) + len(false_tracks) + 2)))
ax.axis("off")
rows = []
for oid, o in score["objects"].items():
    kind = "radiator" if o["radiates"] else "echo"
    pd, err = o["pd"].get(kind), o["median_error_m"].get(kind)
    rows.append([oid, "radiates" if o["radiates"] else "silent", "yes" if o["ais"] else "no",
                 "-" if pd is None else f"{100 * pd:.0f} %", "-" if err is None else f"{err:.1f}", f"{kind} tracks"])
n_obj = len(rows)
for m in false_tracks:
    x, y = m["median_xy"]
    rows.append([m["track"], "false track", "-", f"at ({x:.0f}, {y:.0f})", f"{m['median_error_m']:.1f} to {m['nearest_object']}",
                 f"{m['kind']} tracker"])
tab = ax.table(cellText=rows, colLabels=["object", "", "AIS", "Pd", "median err [m]", "by"], loc="upper center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(9)
tab.scale(1, 1.35)
for (r, col), cell in tab.get_celld().items():
    if r > 0 and col == 3 and r <= n_obj and rows[r - 1][3] != "-":
        v = float(rows[r - 1][3].rstrip(" %")) / 100
        cell.set_facecolor(plt.cm.RdYlGn(0.15 + 0.7 * v))
    elif r > n_obj:
        cell.set_facecolor("#f4d6d6")
T = run["record_s"]
sp = run["seconds"]
ft = score["false_tracks"]
ax.set_title(f"Scored against the true positions: Pd within 5 m from the first frame an object could be confirmed;\n"
             f"{ft['count']} false track(s) ({ft['per_minute']:.1f} per minute; no object within {sc['assoc_radius_m']:g} m); "
             f"compute per s of data: passive {sp['passive'] / T:.2f} s, echo {sp['echo'] / T:.2f} s", fontsize=10)
F.save_all()
(c.out_dir(a) / "07_harbor_picture.json").write_text(json.dumps({
    "dataset": a.dataset, "definitions": score["definitions"], "objects": score["objects"], "tracks": score["tracks"],
    "false_tracks": score["false_tracks"], "run": run["run"],
    "note": "scored against the truth after the run; the processing reads no truth"}, indent=2))

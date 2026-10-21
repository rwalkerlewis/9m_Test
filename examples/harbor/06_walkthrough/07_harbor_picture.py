#!/usr/bin/env python3
"""Step 7, the harbor picture: everything located, and how well.

All outputs together, as tracks: the radiator tracks (with their AIS label from step 4) and the echo tracks (the echo
peaks of step 6 tracked; the raw peaks faint underneath), over the true positions.  Scored with the stage 2 definitions
(``water/scoreboard.py``, ``score_picture``): every confirmed track is matched to the object nearest over its fixes when
within the association radius of the processing config (10 m; 20 m on the 5 m harbor), otherwise it is a false track
(drawn red dashed and listed in the title of the Pd figure with its median position); Pd per object is the fraction of frames, from the
first frame it could have been confirmed, in which a confirmed track matched to it is within 5 m.  Also the compute time
per second of data (below 1: real time).  The scores are also written to ``07_harbor_picture.json``.

    python examples/harbor/06_walkthrough/07_harbor_picture.py [--dataset multistatic]
"""

import json
import textwrap

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
F = c.Figures(a, "07_harbor_picture", f"Step 7: the harbor picture ({a.dataset})", c.NOTE)
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
        ax.annotate(f"{k} (false)", xy[-1], xytext=(5, 4), textcoords="offset points", fontweight="bold", color=c.MISS)
    else:
        ax.plot(xy[:, 0], xy[:, 1], "-" if not echo else ".-", color=c.TRACK if not echo else c.HIT, lw=2.5 if not echo else 1.5, ms=4)
        name = f"{k} ({labels.get(k) or 'no AIS'})" if not echo else f"{k} (echo)"
        ax.annotate(name, xy[-1], xytext=(5, 4), textcoords="offset points", fontweight="bold",
                    color=c.TRACK if not echo else c.HIT)
ax.plot([], [], "-", color=c.TRACK, lw=2.5, label="radiator track")
ax.plot([], [], ".-", color=c.HIT, lw=1.5, label="echo track")
ax.plot([], [], "--", color=c.MISS, lw=2.0, label="false track")
ax.legend(loc="lower left")
ax.set_title("Located: radiator tracks (AIS label in brackets), echo tracks, false tracks (red dashed)")
rows = []
for oid, o in score["objects"].items():
    kind = "radiator" if o["radiates"] else "echo"
    rows.append((oid, kind, o["ais"], o["pd"].get(kind), o["median_error_m"].get(kind)))
fig, ax = F.new("pd", size=(12, max(6.5, 1.8 + 0.5 * len(rows))))
y = np.arange(len(rows))[::-1]
pd = np.array([0.0 if r[3] is None else r[3] for r in rows])
ax.barh(y, pd, color=[plt.cm.RdYlGn(0.15 + 0.7 * v) if r[3] is not None else "0.75" for r, v in zip(rows, pd)], ec="k", lw=0.6)
for yi, r, v in zip(y, rows, pd):
    ax.text(v + 0.01, yi, "not scored" if r[3] is None else
            (f"{100 * v:.0f} %, median error {r[4]:.1f} m" if r[4] is not None else f"{100 * v:.0f} %"), va="center")
ax.set_yticks(y)
ax.set_yticklabels([f"{r[0]} ({r[1]} tracks{', AIS' if r[2] else ''})" for r in rows])
ax.set_xlim(0.0, 1.45)
ax.set_xlabel("Pd: fraction of frames with a matched confirmed track within 5 m")
ax.set_ylabel("object (track kind scored)")
T = run["record_s"]
sp = run["seconds"]
ft = score["false_tracks"]
fl = "; ".join(f"{m['track']} at ({m['median_xy'][0]:.0f}, {m['median_xy'][1]:.0f}) m, {m['median_error_m']:.1f} m from "
               f"{m['nearest_object']}" for m in false_tracks)
ax.set_title(f"Scored against the true positions: {ft['count']} false track(s), {ft['per_minute']:.1f} per minute\n"
             f"compute per s of data: passive {sp['passive'] / T:.2f} s, echo {sp['echo'] / T:.2f} s" +
             ("\n" + textwrap.fill(f"false: {fl}", 95) if fl else ""))
F.save_all()
(c.out_dir(a) / "07_harbor_picture.json").write_text(json.dumps({
    "dataset": a.dataset, "definitions": score["definitions"], "objects": score["objects"], "tracks": score["tracks"],
    "false_tracks": score["false_tracks"], "run": run["run"],
    "note": "scored against the truth after the run; the processing reads no truth"}, indent=2))

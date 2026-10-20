#!/usr/bin/env python3
"""Step 7, the harbor picture: everything located, and how well.

All outputs together: the radiator tracks (with their AIS label from step 4) and every echo detection
(the 3 strongest peaks of every frame), over the true positions.  The table scores each object: the fraction of
frames with a detection within 5 m of it, and the median error.  Also the compute time per second of data
(below 1: real time).

    python examples/walkthrough/07_harbor_picture.py [--dataset multistatic] [--cuda]
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as c

a = c.args(__doc__)
obs, tru = c.observation(a.dataset), c.truth(a.dataset)
run = c.products(a)
fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
ax = axes[0]
c.chart(ax, obs)
c.truth_overlay(ax, tru, labels=False, alpha=0.6)
for k, fx in run["tracks"].items():
    xy = np.array([f[1] for f in fx])
    ax.plot(xy[:, 0], xy[:, 1], "-", color=c.TRACK, lw=2.5)
    ax.annotate(f"{k} ({run['labels'].get(k) or 'no AIS'})", xy[-1], xytext=(5, 4), textcoords="offset points", fontsize=8.5,
                fontweight="bold")


def score(frames, cols):
    rows = {}
    for oid, tr in tru.tracks.items():
        hits = []
        for f in frames:
            p = tr.position_at(f[0])[0]
            if len(f[1]):
                d = np.hypot(*(f[1] - p).T)
                others = [np.min(np.hypot(*(f[1] - tru.tracks[o].position_at(f[0])[0]).T)) for o in tru.tracks if o != oid]
                if d.min() <= 5.0:
                    hits.append(d.min())
        rows[oid] = (len(hits) / max(len(frames), 1), float(np.median(hits)) if hits else None)
    return rows


p_rows = score(run["passive"]["frames"], None)
e_rows = score(run["echo"]["frames"], None) if run["echo"] else {}
if run["echo"]:
    pts = np.array([p for f in run["echo"]["frames"] for p in f[1]]).reshape(-1, 2)
    near = np.array([min(np.hypot(*(tru.tracks[o].position_at(f[0])[0] - p)) for o in tru.tracks if tru.tracks[o].level_pa == 0)
                     for f in run["echo"]["frames"] for p in f[1]])
    ax.plot(pts[near <= 5, 0], pts[near <= 5, 1], "o", color=c.HIT, ms=4, label="echo detection within 5 m of a body")
    ax.plot(pts[near > 5, 0], pts[near > 5, 1], "x", color=c.MISS, ms=3, alpha=0.4, label="echo detection elsewhere")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)
ax.set_title("Located: radiator tracks (black, AIS label in brackets) and echo detections", fontsize=10)
ax = axes[1]
ax.axis("off")
rows = []
for oid, tr in tru.tracks.items():
    kind = "radiates" if tr.level_pa > 0 else "silent"
    pr, er = p_rows[oid], e_rows.get(oid, (np.nan, None))
    use = pr if tr.level_pa > 0 else er
    rows.append([oid, kind, "yes" if tr.ais else "no", f"{100 * use[0]:.0f} %", "-" if use[1] is None else f"{use[1]:.1f}",
                 "passive" if tr.level_pa > 0 else "echo"])
tab = ax.table(cellText=rows, colLabels=["object", "", "AIS", "frames found", "median err [m]", "by"], loc="upper center",
               cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(9)
tab.scale(1, 1.35)
for (r, col), cell in tab.get_celld().items():
    if r > 0 and col == 3:
        v = float(rows[r - 1][3].rstrip(" %")) / 100
        cell.set_facecolor(plt.cm.RdYlGn(0.15 + 0.7 * v))
T = run["record_s"]
sp = run["seconds"]
ax.set_title(f"Scored against the true positions (within 5 m)\ncompute per s of data: passive {sp['passive'] / T:.2f} s, "
             f"echo {sp['echo'] / T:.2f} s (timed in the run that produced the cache)", fontsize=10)
fig.suptitle(f"Step 7 - the harbor picture ({a.dataset})", fontsize=12)
fig.tight_layout()
c.save(fig, c.out_dir(a) / "07_harbor_picture.png", c.NOTE)

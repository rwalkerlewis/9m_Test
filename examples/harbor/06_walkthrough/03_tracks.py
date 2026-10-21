#!/usr/bin/env python3
"""Step 3, tracks: which detections belong together?

Left: every passive peak of the record, coloured by time; most fall on lines (moving radiators), the
rest is clutter.  Right: the tracker joins peaks that follow on in space and time into tracks, confirms a
track after 6 fixes in 2 s, and drops a track that sits next to a stronger one (a sidelobe).  The table
gives each track's distance to the nearest true object.

    python examples/harbor/06_walkthrough/03_tracks.py [--dataset multistatic] [--cuda]
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as c

a = c.args(__doc__)
obs, tru = c.observation(a.dataset), c.truth(a.dataset)
run = c.products(a)
frames = run["passive"]["frames"]
F = c.Figures(a, "03_tracks", f"Step 3 - tracks from the passive detections ({a.dataset})", c.NOTE)
fig, ax = F.new("peaks")
c.chart(ax, obs)
c.zoom(ax, run["passive"]["grid"])
pts = np.array([[t, *p] for t, xy, _, _ in frames for p in xy])
sc = ax.scatter(pts[:, 1], pts[:, 2], c=pts[:, 0], s=6, cmap="plasma")
fig.colorbar(sc, ax=ax, shrink=0.8, label="time [s]")
ax.set_title(f"All passive peaks ({len(pts)} in {len(frames)} frames)", fontsize=10)
fig, ax = F.new("tracks")
c.chart(ax, obs)
c.zoom(ax, run["passive"]["grid"])
c.truth_overlay(ax, tru)
rows = []
for k, fx in run["tracks"].items():
    xy = np.array([f[1] for f in fx])
    ax.plot(xy[:, 0], xy[:, 1], "-", color=c.TRACK, lw=2.2)
    ax.plot(*xy[-1], "o", color=c.TRACK, ms=5)
    ax.annotate(k, xy[-1], xytext=(5, -10), textcoords="offset points", fontsize=9, fontweight="bold")
    d = {o: np.median([np.hypot(*(tru.tracks[o].position_at(f[0])[0] - f[1])) for f in fx]) for o in tru.tracks}
    o = min(d, key=d.get)
    rows.append([k, str(len(fx)), o if d[o] <= 10 else "none", f"{d[o]:.1f}"])
ax.set_title("Confirmed tracks (black) over the true tracks (dashed)", fontsize=10)
fig, ax = F.new("table", size=(8, 0.9 + 0.35 * (len(rows) + 1)))
ax.axis("off")
tab = ax.table(cellText=rows, colLabels=["track", "fixes", "true object", "median error [m]"], loc="upper center",
               cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(8)
F.save_all()

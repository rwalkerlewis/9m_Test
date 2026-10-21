#!/usr/bin/env python3
"""Step 3, tracks: which detections belong together?

First image: every passive peak of the record, coloured by time; most fall on lines (moving radiators), the
rest is clutter.  Second: the tracker joins peaks that follow on in space and time into tracks, confirms a
track after 6 fixes in 2 s, and drops a track that sits next to a stronger one (a sidelobe).  The third
gives each track's distance to the nearest true object.

    python examples/harbor/06_walkthrough/03_tracks.py [--dataset multistatic] [--cuda]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import _common as c

a = c.args(__doc__)
obs, tru = c.observation(a.dataset), c.truth(a.dataset)
run = c.products(a)
frames = run["passive"]["frames"]
F = c.Figures(a, "03_tracks", f"Step 3: tracks from the passive detections ({a.dataset})", c.NOTE)
fig, ax = F.new("peaks")
c.chart(ax, obs)
c.zoom(ax, run["passive"]["grid"])
pts = np.array([[t, *p] for t, xy, _, _ in frames for p in xy])
sc = ax.scatter(pts[:, 1], pts[:, 2], c=pts[:, 0], s=6, cmap="plasma", label="passive peak (colour: time)")
fig.colorbar(sc, ax=ax, shrink=0.8, label="time [s]")
ax.set_title(f"All passive peaks ({len(pts)} in {len(frames)} frames)")
fig, ax = F.new("tracks")
c.chart(ax, obs)
c.zoom(ax, run["passive"]["grid"])
c.truth_overlay(ax, tru)
rows = []
for k, fx in run["tracks"].items():
    xy = np.array([f[1] for f in fx])
    ax.plot(xy[:, 0], xy[:, 1], "-", color=c.TRACK, lw=2.2, label="confirmed track (label at its end)" if not rows else None)
    ax.plot(*xy[-1], "o", color=c.TRACK, ms=5)
    ax.annotate(k, xy[-1], xytext=(5, -10), textcoords="offset points", fontweight="bold")
    d = {o: np.median([np.hypot(*(tru.tracks[o].position_at(f[0])[0] - f[1])) for f in fx]) for o in tru.tracks}
    o = min(d, key=d.get)
    rows.append([k, str(len(fx)), o if d[o] <= 10 else "none", f"{d[o]:.1f}"])
ax.set_title("Confirmed tracks (black) over the true tracks (dashed)")
fig, ax = F.new("errors", size=(12, max(6.5, 1.5 + 0.5 * len(rows))))
y = np.arange(len(rows))[::-1]
err = np.array([float(r[3]) for r in rows])
ax.barh(y, err, color=[c.HIT if r[2] != "none" else c.MISS for r in rows], ec="k", lw=0.6)
for yi, r, e in zip(y, rows, err):
    ax.text(e + 0.3, yi, f"{r[1]} fixes", va="center")
ax.set_yticks(y)
ax.set_yticklabels([f"{r[0]}: {r[2]}" for r in rows])
ax.axvline(10.0, color="0.4", ls="--", lw=1, label="10 m: a track farther from every object is matched to none")

ax.set_xlabel("median distance of the track's fixes to the nearest true object [m]")
ax.set_ylabel("confirmed track: nearest true object")
ax.set_title("Every confirmed track against the nearest true object")
ax.legend(handles=ax.get_legend_handles_labels()[0] + [Patch(color=c.HIT, label="matched to the nearest true object"),
                                                      Patch(color=c.MISS, label="no true object within 10 m")], loc="lower right")
F.save_all()

#!/usr/bin/env python3
"""Step 4, AIS afterwards: which tracks are reporting ships?

The tracks were found from the traces alone.  Now the AIS reports are compared with them: for every
track and reporting ship, the distance between the track and the ship's reported (dead reckoned)
position over time.  A track within 25 m of a ship (median of its last fixes) gets that ship's name; a
track that matches no report is a radiating object without AIS.

    python examples/harbor/06_walkthrough/04_ais_crosscheck.py [--dataset multistatic] [--cuda]
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as c

a = c.args(__doc__)
obs = c.observation(a.dataset)
run = c.products(a)
rep, labels = obs.reports, run["labels"]
F = c.Figures(a, "04_ais_crosscheck", f"Step 4: AIS used only to label what was located ({a.dataset})",
              "AIS is read here for the first time; the tracks did not use it.")
fig, ax = F.new("tracks")
c.chart(ax, obs)
c.zoom(ax, run["passive"]["grid"])
for v in rep.vessels:
    tr = rep.for_vessel(v)
    ax.plot(tr.xy[:, 0], tr.xy[:, 1], "D", ms=6, mfc="none", mec="tab:purple", ls="none",
            label="AIS reports" if v == rep.vessels[0] else None)
    ax.annotate(f"AIS {v}", tr.xy[-1], xytext=(5, 5), textcoords="offset points", color="tab:purple")
for k, fx in run["tracks"].items():
    xy = np.array([f[1] for f in fx])
    lab = labels.get(k)
    col = "tab:purple" if lab else c.TRACK
    ax.plot(xy[:, 0], xy[:, 1], "-", color=col, lw=2.2)
    ax.annotate(f"{k}: {lab or 'no AIS'}", xy[-1], xytext=(5, -12), textcoords="offset points", fontweight="bold", color=col)
ax.plot([], [], "-", color="tab:purple", lw=2.2, label="track labelled by an AIS report")
ax.plot([], [], "-", color=c.TRACK, lw=2.2, label="track with no AIS report")
ax.set_title("Tracks (lines) and AIS reports (diamonds)")
fig, ax = F.new("distance", size=(12, 8))
for k, fx in run["tracks"].items():
    for v in rep.vessels:
        at = rep.for_vessel(v)
        f2 = [f for f in fx if f[0] >= at.t[0]]
        if not f2:
            continue
        d = [np.hypot(*(at.predict(f[0], causal=True)[0] - f[1])) for f in f2]
        ls = "-" if labels.get(k) == v else ":"
        ax.plot([f[0] for f in f2], d, ls, lw=1.8 if ls == "-" else 1.0, label=f"{k} vs AIS {v}")
ax.axhline(25.0, color="k", ls="--", lw=1, label="match: within 25 m")
ax.set_yscale("log")
ax.set_xlabel("time [s]")
ax.set_ylabel("distance track - AIS position [m]")
ax.legend(ncol=2)
ax.set_title("Distance of every track to every reporting ship (solid: the match)")
F.save_all()

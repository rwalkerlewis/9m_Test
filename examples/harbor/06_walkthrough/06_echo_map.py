#!/usr/bin/env python3
"""Step 6, the echo map: where is ship noise being reflected?

On the cleaned traces, every cell of the grid is tested as a reflector: the reference of each tracked
ship, delayed by ship -> cell -> channel, is matched to the data.  A silent body lights up.  The peaks of every frame
feed a second tracker (the echo tracker, peaks next to a radiator track left out); the picture shows its tracks, not
the raw peaks.  Three frames (squares: echo tracks confirmed at that time, with their recent fixes; faint crosses:
the frame's peaks) and the sum of all frames (the full record) with every confirmed echo track, the true bodies
circled.

    python examples/harbor/06_walkthrough/06_echo_map.py [--dataset bistatic] [--cuda]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

import _common as c

a = c.args(__doc__, default="bistatic")
obs, tru = c.observation(a.dataset), c.truth(a.dataset)
run = c.products(a)
grid, frames = run["echo"]["grid"], run["echo"]["frames"]
pick = [frames[int(q * (len(frames) - 1))] for q in (0.2, 0.5, 0.8)]
etracks = run["echo"]["tracks_ever"]
firm = {}                                                       # echo track -> times it was confirmed or static in the picture
for rec in run["records"]:
    for o in rec["objects"]:
        tid = o.get("tracks", {}).get("echo")
        if tid and o.get("track_state", {}).get(tid, {}).get("status") in ("confirmed", "static"):
            firm.setdefault(tid, set()).add(round(rec["t"], 4))


def confirmed_at(k, t):
    return round(t, 4) in firm.get(k, set())



def shown(values):
    """dB over the map's median (its noise level), peaks widened by a 3 cell maximum filter for display."""
    img, ext = c.map_image(grid, 10 * np.log10(np.maximum(values / max(np.median(values[values > 0]), 1e-30), 1e-3)))
    return maximum_filter(np.nan_to_num(img, nan=-30.0), size=3), ext


F = c.Figures(a, "06_echo_map", f"Step 6: echo maps on the cleaned traces ({a.dataset}): silent bodies lit by the tracked ships",
              c.NOTE + " Maps shown with a 3 cell maximum filter.")
for q, (t, xy, score, B, ships) in enumerate(pick):
    fig, ax = F.new(f"{q + 1}")
    c.chart(ax, obs)
    img, ext = shown(B)
    im = ax.imshow(img, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=20, alpha=0.95)
    for oid, tr in tru.tracks.items():
        if tr.level_pa == 0:
            ax.plot(*tr.position_at(t)[0], "o", ms=12, mfc="none", mec="cyan", mew=1.8)
    ax.plot([], [], "o", ms=12, mfc="none", mec="cyan", mew=1.8, ls="none", label="true silent body (for comparison)")
    for j, p in enumerate(ships.values()):
        ax.plot(*p, "s", color=c.RADIATOR, ms=8, mec="w", label="tracked ship (illuminator)" if j == 0 else None)
    for r, p in enumerate(xy):
        ax.plot(*p, "x", color="w", ms=8, mew=1.2, alpha=0.45, label="echo map peak" if r == 0 else None)
    ax.plot([], [], "s-", color="lime", ms=8, mec="k", lw=1.5, label="echo track (last 2 s)")
    for k, fx in etracks.items():
        past = np.array([f[1] for f in fx if t - 2.0 <= f[0] <= t]).reshape(-1, 2)
        if len(past) and confirmed_at(k, t):
            ax.plot(past[:, 0], past[:, 1], "-", color="lime", lw=1.5)
            ax.plot(*past[-1], "s", color="lime", ms=8, mec="k")
            ax.annotate(k, past[-1], xytext=(5, 5), textcoords="offset points", color="lime")
    ax.set_title(f"echo map at t = {t:.2f} s (3 cell maximum filter)")
    c.zoom(ax, grid)
    fig.colorbar(im, ax=ax, shrink=0.8, label="echo map level [dB over the map median, the noise level]")
full = np.sum([f[3] / max(np.median(f[3][f[3] > 0]), 1e-30) for f in frames], axis=0)   # each frame in units of its median
fig, ax = F.new("full_record")
c.chart(ax, obs)
img, ext = shown(full)
im = ax.imshow(img, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=20, alpha=0.95)
c.truth_overlay(ax, tru, labels=True, alpha=0.7)
for k, fx in etracks.items():
    xy = np.array([f[1] for f in fx])
    ax.plot(xy[:, 0], xy[:, 1], ".-", color="lime", lw=1.0, ms=3, label="confirmed echo track" if k == next(iter(etracks)) else None)
    ax.annotate(k, xy[-1], xytext=(5, -9), textcoords="offset points", color="lime")
ax.set_title(f"Full record: sum of {len(frames)} frames, each over its median (3 cell maximum filter)")
c.zoom(ax, grid)
fig.colorbar(im, ax=ax, shrink=0.8, label="echo map level [dB over the map median, the noise level]")
F.save_all()

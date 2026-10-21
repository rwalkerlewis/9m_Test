#!/usr/bin/env python3
"""Step 8, the geometry behind the picture: where the chain thinks things are going, and which paths the sound takes.

First: projected positions.  At three moments each confirmed track's straight line fit over its past fixes is
extended 0.5, 1 and 2 s ahead (open circles; this is how the tracker gates its next fix and how the echo chain
places the illuminator between fixes); filled circles: where the object really was at those times.
Then one frame, in three images: the echo map (or, without an echo chain, the passive map) with its peaks; the
illuminator where the echo chain put it, its direct paths to a few sensors (solid), and for the strongest echo
peak the path illuminator - peak - sensors (dashed); and the ellipse of equal delay through the peak (dotted):
every point of that ellipse returns the echo at the same time, so the map can only place the body along it, and
the array's aperture picks the point.  The true bodies are drawn for comparison only.
Last: how far ahead a projection holds: median distance from the projected to the true position against the
look-ahead time, per matched track.

    python examples/harbor/06_walkthrough/08_geometry.py [--dataset multistatic] [--cuda]
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as c
from acoustic_sim.water.plotting import draw_frame_ellipse, draw_frame_map, draw_frame_rays, draw_projection, projected_positions
from acoustic_sim.water.scoreboard import match_tracks

a = c.args(__doc__)
obs, tru = c.observation(a.dataset), c.truth(a.dataset)
run = c.products(a)
lay = obs.geometry
T = run["record_s"]
matches = {m["track"]: m for m in match_tracks(run["tracks"], tru, run["scoring"]["assoc_radius_m"])}
colors = plt.cm.tab10(np.arange(10))
F = c.Figures(a, "08_geometry", f"Step 8: projected positions and ray paths ({a.dataset})",
              "Processing: traces, sensor positions and the chart only. True positions and bodies are drawn for comparison.")

# Projected positions at three moments.
fig, ax = F.new("projection")
c.chart(ax, obs)
c.zoom(ax, run["passive"]["grid"])
c.truth_overlay(ax, tru, labels=True, alpha=0.5)
horizons = (0.5, 1.0, 2.0)
for q, t_now in enumerate(T * np.array([0.3, 0.55, 0.8])):
    for j, (k, fx) in enumerate(run["tracks"].items()):
        if not any(f[0] <= t_now for f in fx) or fx[-1][0] < t_now - 1.0:
            continue
        col = colors[j % 10]
        proj = draw_projection(ax, fx, t_now, horizons, color=col, label=f"{k}: projected +0.5/1/2 s" if q == 0 else None)
        obj = matches[k]["true_object"]
        if obj is not None and len(proj):
            true_xy = tru.tracks[obj].position_at(t_now + np.asarray(horizons))
            ax.plot(true_xy[:, 0], true_xy[:, 1], "o", color=col, ms=4, alpha=0.9,
                    label=f"{k}: true position of {obj} at those times" if q == 0 else None)
ax.legend(loc="lower left")
ax.set_title("Projected positions at 30, 55 and 80 % of the record\n(open: projected from the track; filled: true)")

# One frame: its map, its ray paths, and (on an echo map) the equal delay ellipse through the strongest peak.
if run["echo"] and run["echo"]["frames"]:
    frames = run["echo"]["frames"]
    t, xy, score, B, ships = frames[len(frames) // 2]
    grid, what, echo = run["echo"]["grid"], "echo map", True
else:
    frames = run["passive"]["frames"]
    t, xy, score, B = frames[len(frames) // 2]
    ships = {k: [f[1] for f in fx if abs(f[0] - t) < 0.3][0] for k, fx in run["tracks"].items() if any(abs(f[0] - t) < 0.3 for f in fx)}
    grid, what, echo = run["passive"]["grid"], "passive map", False
fig, ax = F.new("frame_map")
c.chart(ax, obs)
im = draw_frame_map(ax, grid, B, xy)
c.truth_overlay(ax, tru, t=t, labels=True, alpha=0.7)
c.zoom(ax, grid)
fig.colorbar(im, ax=ax, shrink=0.7, label="map level [dB re frame peak]")
ax.set_title(f"The {what} at t = {t:.2f} s and its peaks")

fig, ax = F.new("ray_paths")
c.chart(ax, obs)
draw_frame_rays(ax, lay, ships, xy, echo=echo)
c.truth_overlay(ax, tru, t=t, labels=True, alpha=0.7)
c.zoom(ax, grid)
ax.set_title(f"Ray paths at t = {t:.2f} s: " + ("illuminators to the sensors, and illuminator to the strongest echo peak "
                                                 "to the sensors" if echo else "each tracked radiator to a few sensors"))

if echo:
    fig, ax = F.new("ellipse")
    c.chart(ax, obs)
    draw_frame_ellipse(ax, lay, ships, xy)
    c.truth_overlay(ax, tru, t=t, labels=True, alpha=0.7)
    c.zoom(ax, grid)
    ax.set_title(f"Equal delay ellipse through the strongest echo peak at t = {t:.2f} s\n(foci: the first illuminator and the "
                 f"array centre)")

# Projection error against look-ahead.
fig, ax = F.new("lookahead", size=(12, 8))
look = np.arange(0.25, 3.01, 0.25)
for j, (k, fx) in enumerate(run["tracks"].items()):
    obj = matches[k]["true_object"]
    if obj is None:
        continue
    errs = []
    for f in fx[6::2]:
        times, proj = projected_positions(fx, f[0], look)
        errs.append(np.hypot(*(proj - tru.tracks[obj].position_at(times)).T))
    if errs:
        ax.plot(look, np.nanmedian(np.array(errs), axis=0), "o-", color=colors[j % 10], ms=3, label=f"{k} ({obj})")
ax.axhline(5.0, color="0.5", ls="--", lw=1, label="5 m (detection radius)")
ax.set_xlabel("look-ahead [s]")
ax.set_ylabel("median distance projected - true [m]")
ax.set_title("How far ahead the projection holds")
ax.legend()
ax.grid(alpha=0.3)
F.save_all()

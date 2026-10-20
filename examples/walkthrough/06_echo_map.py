#!/usr/bin/env python3
"""Step 6, the echo map: where is ship noise being reflected?

On the cleaned traces, every cell of the grid is tested as a reflector: the reference of each tracked
ship, delayed by ship -> cell -> channel, is matched to the data.  A silent body lights up.  Three frames
(crosses: the frame's strongest peaks) and the sum of all frames (the full record), with the true
bodies circled.

    python examples/walkthrough/06_echo_map.py [--dataset bistatic] [--cuda]
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as c

a = c.args(__doc__, default="bistatic")
obs, tru = c.observation(a.dataset), c.truth(a.dataset)
run = c.products(a)
grid, frames = run["echo"]["grid"], run["echo"]["frames"]
pick = [frames[int(q * (len(frames) - 1))] for q in (0.2, 0.5, 0.8)]
fig, axes = plt.subplots(1, 4, figsize=(22, 5.4))
for ax, (t, xy, score, B, ships) in zip(axes[:3], pick):
    c.chart(ax, obs)
    img, ext = c.map_image(grid, 10 * np.log10(np.maximum(B / max(B.max(), 1e-30), 1e-3)))
    im = ax.imshow(img, origin="lower", extent=ext, cmap="inferno", vmin=-15, vmax=0, alpha=0.95)
    for oid, tr in tru.tracks.items():
        if tr.level_pa == 0:
            ax.plot(*tr.position_at(t)[0], "o", ms=12, mfc="none", mec="cyan", mew=1.8)
    for n, p in ships.items():
        ax.plot(*p, "s", color=c.RADIATOR, ms=8, mec="w")
    for r, p in enumerate(xy):
        ax.plot(*p, "x", color="lime" if r == 0 else "w", ms=10, mew=2)
    ax.set_title(f"t = {t:.2f} s (dB re frame peak)\nsquares: tracked ships; cyan: true bodies; x: peaks", fontsize=9)
full = np.sum([f[3] for f in frames], axis=0)
ax = axes[3]
c.chart(ax, obs)
img, ext = c.map_image(grid, 10 * np.log10(np.maximum(full / full.max(), 1e-3)))
ax.imshow(img, origin="lower", extent=ext, cmap="inferno", vmin=-10, vmax=0, alpha=0.95)
c.truth_overlay(ax, tru, labels=True, alpha=0.7)
ax.set_title(f"Full record: sum of {len(frames)} frames", fontsize=9)
fig.colorbar(im, ax=axes, shrink=0.8, label="dB")
fig.suptitle(f"Step 6 - echo maps on the cleaned traces ({a.dataset}): silent bodies lit by the tracked ships", fontsize=12)
c.save(fig, c.out_dir(a) / "06_echo_map.png", c.NOTE)

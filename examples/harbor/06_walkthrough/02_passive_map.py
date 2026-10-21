#!/usr/bin/env python3
"""Step 2, passive location: where does sound come from?

Every 0.25 s the matched field processor steers the array at every cell of the scan grid: a cell where
a source radiates matches the phases across the channels and lights up.  The 5 strongest peaks of each
frame (crosses) are the passive detections; no AIS is used.  Circles: true positions at that time.

    python examples/harbor/06_walkthrough/02_passive_map.py [--dataset multistatic] [--cuda]
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as c

a = c.args(__doc__)
obs, tru = c.observation(a.dataset), c.truth(a.dataset)
run = c.products(a)
grid, frames = run["passive"]["grid"], run["passive"]["frames"]
pick = [frames[int(q * (len(frames) - 1))] for q in (0.2, 0.5, 0.8)]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
for ax, (t, xy, score, B) in zip(axes, pick):
    c.chart(ax, obs)
    img, ext = c.map_image(grid, 10 * np.log10(np.maximum(B / max(B.max(), 1e-30), 1e-4)))
    im = ax.imshow(img, origin="lower", extent=ext, cmap="viridis", vmin=-20, vmax=0, alpha=0.9)
    c.truth_overlay(ax, tru, t=t)
    for r, p in enumerate(xy):
        ax.plot(*p, "x", color="w" if r else "r", ms=11 if r == 0 else 8, mew=2.2)
    ax.set_title(f"t = {t:.2f} s: matched field map (dB re frame peak)\nred x: strongest peak, white x: next 4", fontsize=10)
fig.colorbar(im, ax=axes, shrink=0.8, label="dB")
fig.suptitle(f"Step 2 - passive location without AIS ({a.dataset}): every radiating object, ships included", fontsize=12)
c.save(fig, c.out_dir(a) / "02_passive_map.png", c.NOTE)

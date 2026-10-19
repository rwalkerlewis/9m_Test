#!/usr/bin/env python3
"""Figure of the stage 0 benchmark: wall time per case, original code against the new defaults.

Reads ``output/performance/baseline.json`` and ``output/performance/final.json`` (written by
``scripts/bench_harbor.py``) and writes ``output/performance/timings.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "performance"
LABELS = {
    "fdtd_ferry_2s": "FDTD ferry alone, 2 s",
    "fdtd_ferry_10s": "FDTD ferry alone, 10 s",
    "fdtd_ferry_uuv_2s": "FDTD ferry + UUV body, 2 s",
    "fdtd_ferry_uuv_10s": "FDTD ferry + UUV body, 10 s",
    "fdtd_multistatic_2s": "FDTD multistatic, 10 bodies, 2 s",
    "proc_oracle": "run_bistatic oracle",
    "proc_eca_oracle_doppler": "run_bistatic eca_oracle+doppler",
    "proc_blind_doppler_smooth4": "run_bistatic blind+doppler+smooth4",
    "eca_193_moving_fixed": "eca_stage 193 m, fixed lags",
    "eca_193_moving_doppler_smooth4": "eca_stage 193 m, per sample delays",
}
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"       # categorical slots 1 and 2 of the validated reference palette
INK, INK_2, GRID = "#1f1f1e", "#5f5e58", "#d9d8d2"


def main() -> None:
    base = json.load(open(OUT / "baseline.json"))["cases"]
    final_path = OUT / (sys.argv[1] if len(sys.argv) > 1 else "final.json")
    new = json.load(open(final_path))["cases"]
    names = [n for n in LABELS if n in base and n in new]
    before = np.array([base[n]["wall_s"] for n in names])
    after = np.array([new[n]["wall_s"] for n in names])
    y = np.arange(len(names))
    h = 0.38
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(names) + 1.6))
    ax.barh(y + h / 2, before, h, color=SERIES_1, label="original code", zorder=3)
    ax.barh(y - h / 2, after, h, color=SERIES_2, label="new defaults (fused kernel, device masks, vectorized GPU chain)", zorder=3)
    for yi, b, a in zip(y, before, after):
        ax.text(max(b, a) * 1.08, yi, f"{b / a:.1f}x" if b / a < 10 else f"{b / a:.0f}x", va="center", ha="left", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.3, max(before) * 4)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[n] for n in names], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("wall time [s], Quadro RTX 4000", color=INK_2)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Stage 0 benchmark: original code against the new defaults", loc="left", fontsize=11, color=INK)
    fig.savefig(OUT / "timings.png", dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    print("wrote", OUT / "timings.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Why the batched float32 boat runs of ``multistatic_b`` diverge: the numbers and figures of
``output/studies/float32/batched_divergence.md`` (follow up of task 6, 2026-10-22).

Collects the diagnostic runs made with ``scratch/f32/`` (single60.py, fft_iso.py, fft_events.py, decay.py, pert.py;
their outputs are not committed) into ``batched_divergence.json`` and draws two figures:

* ``divergence_growth_multistatic_b.png``: the largest trace value in every 0.1 s of the boat's two diverged batched
  float32 gathers, the same runs alone in float32 and the committed float64 gathers;
* ``free_decay_multistatic_b.png``: the rms over the receivers of a broadband burst left to decay, float64 and float32
  (the float32 run twice).

These runs are diagnostics with no pass or fail criterion of their own; the criterion they serve is the scattered
component criterion of ``harbor_float32_scattered.py`` (fixed before the first float32 build, unchanged).

    python examples/harbor/02_forward_model/harbor_float32_divergence.py [--scratch scratch/f32] [--out output/studies/float32]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402

REC = ROOT / "output/recordings/multistatic_b/cache"
# The boat's gathers: float32 (diverged, batched) and float64 (committed), without and with the ten bodies.
F32 = {0: "7c5b2a2123b57805cb2d06d7be7de274db3f3077", 10: "047cd5d20735e6dcb4a33dda9b4ea4ad56c4125d"}
F64 = {0: "fde8355d3509f9fd3fa930f193443790cb6e9765", 10: "c47b59aac15ca0a6b1814f26adf759859d778f26"}
FS = 6000.0


def peak_per(g: np.ndarray, width: int) -> np.ndarray:
    a = np.where(np.isfinite(g), np.abs(g), np.inf).max(0)
    n = len(a) // width
    return a[:n * width].reshape(n, width).max(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default="scratch/f32")
    ap.add_argument("--out", default="output/studies/float32")
    a = ap.parse_args()
    sc, out = ROOT / a.scratch, ROOT / a.out
    res: dict = {"note": "diagnostics, no criterion of their own; the criterion is scattered.json's"}

    res["single_runs_60s"] = [json.loads(p.read_text()) for p in sorted((sc / "single60").glob("*_float32.json"))]
    for r in res["single_runs_60s"]:
        r.pop("diff_rms_per_second", None), r.pop("ref_rms_per_second", None)
    res["fft_isolated"] = [json.loads(line) for f in ("fft_iso/matrix_cupy.jsonl", "fft_iso/matrix_torch.jsonl")
                           for line in (sc / f).read_text().splitlines() if line.strip()]
    res["fft_in_solver_shape"] = [json.loads(p.read_text()) for p in sorted((sc / "fft_events").glob("*.json"))]
    res["free_decay"] = {p.stem: json.loads(p.read_text()) for p in sorted((sc / "decay").glob("*.json"))}
    for r in res["free_decay"].values():
        r.pop("decay_rate_per_s", None), r.pop("decay_rate_per_step", None)   # a fit across the jumps, not a rate
    res["batch_of_two_perturbed"] = [json.loads(p.read_text()) for p in sorted((sc / "pert").glob("*.json"))]

    # The diverged gathers: events (the largest value jumps by 10 times its previous 0.5 s) and the channels they hit.
    w = int(0.1 * FS)
    fig, ax = new_figure()
    events = {}
    for nb, colour in ((0, "tab:red"), (10, "tab:orange")):
        g32 = np.load(sc / "blown" / f"gather_{F32[nb]}.npz")["traces"]
        g64 = np.load(REC / f"gather_{F64[nb]}.npz")["traces"]
        m = np.where(np.isfinite(g32), np.abs(g32), np.inf).max(0)
        ev, i = [], int(FS)
        while i < len(m):
            prev = m[i - int(FS) // 2:i].max()
            if np.isfinite(prev) and m[i] > 10 * prev:
                ev.append({"step": i, "t_s": i / FS, "value": float(m[i]), "before": float(prev),
                           "channels_hit_same_step": int((np.abs(g32[:, i]) > 10 * prev).sum())})
                i += int(FS) // 2
            else:
                i += 1
        events[f"bodies_{nb}"] = ev[:6]
        t = (np.arange(len(m) // w) + 0.5) * w / FS
        label = "without the bodies" if nb == 0 else "with the ten bodies"
        ax.semilogy(t, peak_per(g32, w), color=colour, lw=1.2, label=f"float32, batch of two, {label}")
        if nb == 0:
            ax.semilogy(t, peak_per(g64, w), color="k", lw=1.0, label=f"float64, {label}")
        alone = sc / "single60" / f"solver_{int(nb > 0)}_float32.npy"
        if alone.exists():
            ax.semilogy(t, peak_per(np.load(alone), w), color="tab:blue" if nb == 0 else "tab:cyan", lw=1.0, ls="--",
                        label=f"float32, run alone, {label}")
    res["diverged_batch_events"] = events
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Largest |trace| in 0.1 s [Pa]")
    ax.set_title("multistatic_b: the boat's gathers, largest trace value over time")
    ax.legend(loc="upper left")
    save_figure(fig, out / "divergence_growth_multistatic_b.png")

    fig, ax = new_figure()
    for key, colour, label in (("float64_20", "k", "float64, run alone"), ("float32_20", "tab:red", "float32, run alone"),
                               ("float32_20_repeat", "tab:orange", "float32, run alone, the same run again")):
        d = res["free_decay"].get(key)
        if d:
            r = np.asarray(d["rms_per_quarter_second"])
            ax.semilogy((np.arange(len(r)) + 0.5) * 0.25, r, color=colour, lw=1.2, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rms over the receivers in 0.25 s [per unit rms of the burst]")
    ax.set_title("multistatic_b grid: free decay of a broadband burst, float32 and float64")
    ax.legend(loc="upper right")
    save_figure(fig, out / "free_decay_multistatic_b.png")

    (out / "batched_divergence.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(events, indent=1))


if __name__ == "__main__":
    main()

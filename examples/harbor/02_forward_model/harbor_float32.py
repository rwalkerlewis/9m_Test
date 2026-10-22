#!/usr/bin/env python3
"""The FDTD with the k space scheme in float32 against float64, on every recording's scenario (prompt ``float32``,
section 1).

For each recording (one per geometry: ``simple``, ``bistatic``, which ``multistatic`` shares, ``multistatic_b``, the 5 m
``harbor``) the propagation error pulse of ``harbor_propagation_error.py`` runs twice, in float64 and in float32, with
nothing else changed and no cache.  Criterion, fixed before the run: the direct arrival error changes by at most
0.02 ms at every hydrophone; the direct traces agree within 1e-3 relative at the worst sample and 1e-4 at the median
(per receiver the largest difference over the trace's peak; the worst and the median over the receivers).  The scattered
component's criterion (within 1e-3 of its rms) needs the body gathers in both precisions, that is the recordings
rebuilt: not run here.  The time per step and the memory per run are measured on the device the run used.

    python examples/harbor/02_forward_model/harbor_float32.py [--recordings simple ...] [--cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "harbor" / "01_datasets"))

import harbor_propagation_error as pe  # noqa: E402

GEOMETRIES = {"simple": (100.0, 380.0), "bistatic": (100.0, 760.0), "multistatic_b": (100.0, 760.0), "harbor": (12.0, 78.0)}
CRITERION = {"arrival_change_ms": 0.02, "worst_relative": 1e-3, "median_relative": 1e-4}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Per receiver the largest difference of *b* from *a* over *a*'s peak; their worst and median."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    peak = np.max(np.abs(a), axis=1)
    live = peak > 0
    e = np.max(np.abs(b - a), axis=1)[live] / peak[live]
    return {"worst_relative": float(np.max(e)), "median_relative": float(np.median(e)), "receivers": int(live.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", nargs="*", default=list(GEOMETRIES))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="output/studies/float32")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fdtd.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["criterion"] = CRITERION
    res["device"] = "cpu" if a.cpu else "gpu"
    for rec in a.recordings:
        ds = ROOT / "output" / "recordings" / rec / "dataset"
        runs = {}
        for prec in ("float64", "float32"):
            t0 = time.perf_counter()
            r = pe.measure(ds, use_cuda=not a.cpu, band=GEOMETRIES[rec], save=False, precision=prec, use_cache=False, keep_traces=True)
            r["wall_s"] = time.perf_counter() - t0
            runs[prec] = r
        tr64, tr32 = runs["float64"].pop("_traces"), runs["float32"].pop("_traces")
        d_arr = [abs(h32["error_ms"] - h64["error_ms"]) for h64, h32 in zip(runs["float64"]["hydrophones"], runs["float32"]["hydrophones"])]
        cmp = compare(tr64, tr32)
        n = tr64.shape[1]
        row = {"arrival_change_ms": d_arr, "max_arrival_change_ms": float(max(d_arr)), **cmp,
               "per_step_s": {p: runs[p]["wall_s"] / n for p in runs}, "steps": n,
               "passed": bool(max(d_arr) <= CRITERION["arrival_change_ms"] and cmp["worst_relative"] <= CRITERION["worst_relative"]
                              and cmp["median_relative"] <= CRITERION["median_relative"]),
               "float64": {k: v for k, v in runs["float64"].items() if k != "note"},
               "float32": {k: v for k, v in runs["float32"].items() if k != "note"}}
        res.setdefault("recordings", {})[rec] = row
        print(f"{rec}: arrival change at most {row['max_arrival_change_ms']:.4f} ms, traces worst {cmp['worst_relative']:.2e} median "
              f"{cmp['median_relative']:.2e}, {'pass' if row['passed'] else 'FAIL'}; per step {row['per_step_s']['float64'] * 1e3:.2f} ms "
              f"(float64) / {row['per_step_s']['float32'] * 1e3:.2f} ms (float32) on the {res['device']}", flush=True)
        path.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

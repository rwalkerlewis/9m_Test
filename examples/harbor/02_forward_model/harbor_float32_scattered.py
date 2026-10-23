#!/usr/bin/env python3
"""The scattered component of every recording in float32 against float64 (prompt ``float32``, section 1).

Each recording's seed 0 is built twice from the same scenario file, once in float64 (``--built64 <dir>/<rec>``) and
once with only ``fdtd.precision = "float32"`` changed (``--built32 <dir>/<rec>``, the same dataset script and noise
seed).  Compared are the noise free components the truth file holds (``comp:scattered:<illuminator>><bodies>``:
the body gathers less the free gathers of the same run, so the subtraction is done in the run's precision).

Criterion, fixed before the run (the prompt's "the scattered component within 1e-3 of its rms", measured as the
performance work measured it, ``scripts/bench_harbor.py:rel_rms``): for every scattered component of every recording,
the rms of the difference float32 less float64 over every channel and sample is at most 1e-3 of the rms of the
float64 component.  The same ratio of the direct components and the worst channel's ratio are reported beside it
(information, no criterion); a recording whose truth holds no components (``harbor``, one run per vessel with the
bodies in it) has no scattered component, and its noise free sum is compared instead (information).  Writes ``scattered.json`` and ``scattered.md``.

A float32 gather can rarely carry a disturbance from the GPU fault of ``output/studies/float32/gpu_determinism.md``,
which does not repeat.  Rule, fixed before the run: a component over the limit is attributed to that fault only if its
float32 gather, deleted from the cache and built again, gives different traces and is then within the limit; the
rebuilt gather is then compared
and the number of rebuilt gathers is reported.  If the rebuilt gather gives the same traces, or fails again, the
difference is the precision's and the criterion fails.  (The words "and is then within the limit" and "or fails
again" were added after the build, to state how the rule was applied when the rebuilt gathers diverged again; they
make it stricter, and the result is the same either way: fail.)  In the float32 build with the float32 k space runs one at a time
(``runs_alone``, 2026-10-22 evening) no component was over the limit and the rule was not used.

    python examples/harbor/02_forward_model/harbor_float32_scattered.py --built64 <dir> --built32 output/recordings \\
        [--recordings simple bistatic multistatic multistatic_b harbor]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Truth, contract_paths  # noqa: E402

CRITERION = {"scattered_rel_rms": 1e-3}


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / ||b|| over the whole array (``scripts/bench_harbor.py``)."""
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - b) / max(np.linalg.norm(b), 1e-300))


def worst_channel(a: np.ndarray, b: np.ndarray) -> float:
    """The largest per channel ``||a - b|| / ||b||`` over the channels where *b* is not zero."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    nb = np.linalg.norm(b, axis=-1)
    ok = nb > 0
    return float(np.max(np.linalg.norm(a - b, axis=-1)[ok] / nb[ok])) if ok.any() else 0.0


def compare(t64: Truth, t32: Truth) -> dict:
    rows = {}
    if not t64.components:                     # a recording without components (``harbor``): the noise free sum only
        rows["clean"] = {"rel_rms": rel_rms(t32.clean, t64.clean), "worst_channel_rel_rms": worst_channel(t32.clean, t64.clean)}
        return rows
    for key in sorted(t64.components):
        if not key.startswith(("scattered:", "direct:")):
            continue
        a, b = np.asarray(t32.components[key]), np.asarray(t64.components[key])
        rows[key] = {"rel_rms": rel_rms(a, b), "worst_channel_rel_rms": worst_channel(a, b)}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", nargs="*", default=["simple", "bistatic", "multistatic", "multistatic_b", "harbor"])
    ap.add_argument("--built64", required=True, help="folder holding <rec>/dataset.* built in float64")
    ap.add_argument("--built32", default="output/recordings", help="folder holding <rec>/dataset.* built in float32")
    ap.add_argument("--out", default="output/studies/float32")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scattered.json"
    res = {"criterion": CRITERION, "measure": "rms of (float32 - float64) over the rms of float64, every channel and sample",
           "built64": a.built64, "built32": a.built32, "recordings": {}}
    for rec in a.recordings:
        p64 = Path(a.built64) / rec / "dataset.npz"
        p32 = Path(a.built32) / rec / "dataset.npz"
        t64, t32 = Truth.load(contract_paths(p64)[1]), Truth.load(contract_paths(p32)[1])
        if t32.scenario["fdtd"].get("precision") != "float32" or t64.scenario["fdtd"].get("precision", "float64") != "float64":
            raise SystemExit(f"{rec}: the two builds are not float64 and float32")
        rows = compare(t64, t32)
        build = Path(a.built32) / rec / "build.json"
        if build.exists():
            res.setdefault("built32_commit", {})[rec] = json.loads(build.read_text()).get("run", {}).get("git_commit")
        res["recordings"][rec] = rows
        for k, r in rows.items():
            print(f"{rec} {k}: {r['rel_rms']:.2e} (worst channel {r['worst_channel_rel_rms']:.2e})", flush=True)
    sc = [r["rel_rms"] if np.isfinite(r["rel_rms"]) else float("inf")         # a run that diverged fails
          for rows in res["recordings"].values() for k, r in rows.items() if k.startswith("scattered:")]
    res["max_scattered_rel_rms"] = float(max(sc, default=0.0))
    res["passed"] = bool(res["max_scattered_rel_rms"] <= CRITERION["scattered_rel_rms"])
    for rows in res["recordings"].values():                # a diverged component is written as null, not NaN
        for r in rows.values():
            r.update({k: (v if np.isfinite(v) else None) for k, v in r.items()})
    if not np.isfinite(res["max_scattered_rel_rms"]):
        res["max_scattered_rel_rms"] = None
    path.write_text(json.dumps(res, indent=2))
    lines = ["| Recording | Component | rms difference over rms | worst channel |", "|---|---|---|---|"]
    for rec, rows in res["recordings"].items():
        for k, r in rows.items():
            lines.append(f"| {rec} | {k} | {r['rel_rms']:.2e} | {r['worst_channel_rel_rms']:.2e} |")
    lines.append("")
    lines.append(f"Criterion: every scattered component within {CRITERION['scattered_rel_rms']:.0e} of its rms: "
                 f"{'pass' if res['passed'] else 'FAIL'} (largest {res['max_scattered_rel_rms']:.2e}).")
    (out / "scattered.md").write_text("\n".join(lines) + "\n")
    print(f"criterion {'pass' if res['passed'] else 'FAIL'}: largest scattered {res['max_scattered_rel_rms']:.2e}")


if __name__ == "__main__":
    main()

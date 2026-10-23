#!/usr/bin/env python3
"""The criteria of the joint removal of every illuminator on the chain (prompt ``joint_removal``, section 3), from the
summary before the change and the summary after it, on the test recordings.  The criteria were fixed in the prompt
before the chain was run; this script only evaluates them (written before the run's numbers were seen):

1. ``silent_bodies``: at least three of the six pilings and the diver tracked by their echo with Pd at least 0.3 on
   each multistatic test seed (seeds 1 to 3).  Read as: at least three of these seven bodies at 0.3 or more on every
   seed.  The other reading (at least three pilings and the diver, each at 0.3 or more) is reported beside it
   (``alternative_reading``).
2. ``no_fall``: the moored hull and both UUVs not down by more than 3 percentage points (echo Pd) on any multistatic
   test recording (the seeds and ``multistatic_b``).
3. ``compute``: compute per second of data not above its value before on any recording (tuning and test).
4. ``size_order`` (stage 2, evaluated once a piling is tracked): on the frames where the moored hull and a piling both
   have a size indicator, the hull's is below every measured piling's in at least 70 % of them (multistatic recordings,
   from the runs' ``characteristics.json``).

    python examples/harbor/05_detection/harbor_joint_removal_criteria.py BEFORE.json AFTER.json [--runs output/reproduce/runs] [--out criteria.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BODIES = ["piling1", "piling2", "piling3", "piling4", "piling5", "piling6", "diver"]
KEEP = ["moored", "uuv1", "uuv2"]
SEEDS = (1, 2, 3)


def size_order(runs: Path, names: list[str]) -> dict:
    out, n, below = {}, 0, 0
    for name in names:
        p = runs / name / "characteristics.json"
        if not p.exists():
            continue
        by = {r["object"]: r.get("size_ratio_per_frame", {}) for r in json.loads(p.read_text())["objects"]}
        h = by.get("moored", {})
        ps = [v for k, v in by.items() if k.startswith("piling")]
        frames = [t for t in h if any(t in pv for pv in ps)]
        b = [t for t in frames if all(h[t] < pv[t] for pv in ps if t in pv)]
        out[name] = {"frames_both": len(frames), "hull_below_piling": len(b)}
        n += len(frames)
        below += len(b)
    return {"values": out, "frames": n, "fraction": below / n if n else None}


def evaluate(before: dict, after: dict, runs: Path) -> list[dict]:
    rb, ra = before["recordings"], after["recordings"]
    out = []
    v = {f"multistatic_seed{s}": {b: ra[f"multistatic_seed{s}"]["objects"][b]["pd"].get("echo") for b in BODIES} for s in SEEDS}
    n = {k: sum((x or 0.0) >= 0.3 for x in d.values()) for k, d in v.items()}
    alt = {k: sum((d[f"piling{i}"] or 0.0) >= 0.3 for i in range(1, 7)) >= 3 and (d["diver"] or 0.0) >= 0.3 for k, d in v.items()}
    out.append({"id": "silent_bodies", "text": "at least three of the six pilings and the diver tracked with Pd at least 0.3 "
                "on the multistatic seeds (three of the seven bodies on every seed)", "values": v, "at_0.3": n,
                "passed": all(x >= 3 for x in n.values()),
                "alternative_reading": {"text": "three pilings and the diver, each at 0.3 or more, on every seed",
                                        "values": alt, "passed": all(alt.values())}})
    falls, vals = {}, {}
    for name, a in ra.items():
        if not a["held_out"] or a["family"] not in ("multistatic", "multistatic_b"):
            continue
        for oid in KEEP:
            pb, pa = rb[name]["objects"][oid]["pd"].get("echo"), a["objects"][oid]["pd"].get("echo")
            vals[f"{name}/{oid}"] = {"before": pb, "after": pa}
            if pb is not None and (pa or 0.0) < pb - 0.03 - 1e-12:
                falls[f"{name}/{oid}"] = {"before": pb, "after": pa}
    out.append({"id": "no_fall", "text": "the moored hull and both UUVs not down by more than 3 points (multistatic test "
                "recordings)", "values": vals, "falls": falls, "passed": not falls})
    v = {k: {"before": rb[k]["compute_per_second_of_data"]["total"], "after": a["compute_per_second_of_data"]["total"]}
         for k, a in ra.items() if k in rb}
    out.append({"id": "compute", "text": "compute per second of data not above its value before on any recording",
                "values": v, "passed": all(x["after"] <= x["before"] for x in v.values())})
    so = size_order(runs, ["multistatic"] + [f"multistatic_seed{s}" for s in SEEDS] + ["multistatic_b"])
    out.append({"id": "size_order", "text": "the hull below a piling on the size indicator in at least 70 % of the frames "
                "where both are measured", **so,
                "passed": None if not so["frames"] else so["fraction"] >= 0.7,
                "note": None if so["frames"] else "no frame had the hull and a piling both measured: not testable"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--runs", default="output/reproduce/runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = evaluate(json.loads(Path(a.before).read_text()), json.loads(Path(a.after).read_text()), Path(a.runs))
    for c in res:
        state = "not testable" if c["passed"] is None else ("pass" if c["passed"] else "FAIL")
        print(f"{state}: {c['text']}")
    if a.out:
        Path(a.out).write_text(json.dumps({"criteria": res}, indent=2))


if __name__ == "__main__":
    main()

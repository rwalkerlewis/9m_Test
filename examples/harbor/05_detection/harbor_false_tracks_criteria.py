#!/usr/bin/env python3
"""The criteria of the false track change on the chain (prompt ``false_tracks``, section 3), from the summary before the
change and the summary after it, on the test recordings.  The criteria were fixed in the prompt before the chain was
run; this script only evaluates them (written before the run's numbers were seen):

1. ``false_tracks``: at most 2 false tracks per minute on ``multistatic_b`` and at most 3 per minute (one track) on every
   20 s test recording (the simple, bistatic and multistatic seeds; the 5 m harbor's are 120 s and not in it);
2. ``uuv_echo_only``: the bistatic UUV reported as seen by echo only, not both ways, in at least 90 % of the frames it is
   reported in, on every bistatic test seed (from the runs' ``characteristics.json``: the frames its echo track is
   reported in over the frames any of its tracks is);
3. ``boat``: the boat on ``multistatic_b`` tracked with Pd at least 0.8 (radiator tracks);
4. ``no_fall``: no object's Pd (its own track kind) falls by more than 3 percentage points on any test recording;
5. ``compute``: compute per second of data not above its value before on any recording.

    python examples/harbor/05_detection/harbor_false_tracks_criteria.py BEFORE.json AFTER.json [--runs output/reproduce/runs] [--out criteria.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SEEDS = (1, 2, 3)


def uuv_echo_only(runs: Path, name: str) -> float | None:
    p = runs / name / "characteristics.json"
    if not p.exists():
        return None
    for o in json.loads(p.read_text())["objects"]:
        if o["object"] == "uuv1":
            seen = o["seen"]
            n = sum(seen.values())
            return seen.get("echo", 0) / n if n else None
    return None


def evaluate(before: dict, after: dict, runs: Path, runs_before: Path | None = None) -> list[dict]:
    rb, ra = before["recordings"], after["recordings"]
    out = []
    v = {k: a["false_tracks"]["per_minute"] for k, a in ra.items() if a["held_out"] and a["family"] != "harbor"}
    lim = {k: (2.0 if k == "multistatic_b" else 3.0) for k in v}
    out.append({"id": "false_tracks", "text": "false tracks at most 2 per minute on multistatic_b and at most 3 on every 20 s "
                "test recording", "values": v, "before": {k: rb[k]["false_tracks"]["per_minute"] for k in v},
                "limits": lim, "passed": all(v[k] <= lim[k] for k in v)})
    v = {f"bistatic_seed{s}": uuv_echo_only(runs, f"bistatic_seed{s}") for s in SEEDS}
    vb = {k: uuv_echo_only(runs_before, k) for k in v} if runs_before else None
    out.append({"id": "uuv_echo_only", "text": "the bistatic UUV seen by echo only in at least 90 % of its frames",
                "values": v, "before": vb, "passed": all(x is not None and x >= 0.9 for x in v.values())})
    pb, pa = rb["multistatic_b"]["objects"]["boat"]["pd"]["radiator"], ra["multistatic_b"]["objects"]["boat"]["pd"]["radiator"]
    out.append({"id": "boat", "text": "the boat on multistatic_b at least 0.8", "before": pb, "values": pa,
                "passed": pa is not None and pa >= 0.8})
    falls = {}
    for name, a in ra.items():
        if not a["held_out"]:
            continue
        for oid, o in a["objects"].items():
            kind = "radiator" if o["radiates"] else "echo"
            b0, a0 = rb[name]["objects"][oid]["pd"].get(kind), o["pd"].get(kind)
            if b0 is not None and (a0 or 0.0) < b0 - 0.03 - 1e-12:
                falls[f"{name}/{oid}"] = {"before": b0, "after": a0}
    out.append({"id": "no_fall", "text": "no Pd falls by more than 3 percentage points (test recordings)", "values": falls,
                "passed": not falls})
    v = {k: {"before": rb[k]["compute_per_second_of_data"]["total"], "after": a["compute_per_second_of_data"]["total"]}
         for k, a in ra.items() if k in rb}
    out.append({"id": "compute", "text": "compute per second of data not above its value before on any recording",
                "values": v, "passed": all(x["after"] <= x["before"] for x in v.values())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--runs", default="output/reproduce/runs")
    ap.add_argument("--runs-before", default=None, help="the runs before the change (for the UUV's before)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = evaluate(json.loads(Path(a.before).read_text()), json.loads(Path(a.after).read_text()), Path(a.runs),
                   Path(a.runs_before) if a.runs_before else None)
    for c in res:
        print(f"{'pass' if c['passed'] else 'FAIL'}: {c['text']}")
    if a.out:
        Path(a.out).write_text(json.dumps({"criteria": res}, indent=1))


if __name__ == "__main__":
    main()

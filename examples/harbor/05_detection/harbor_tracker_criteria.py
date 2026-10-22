#!/usr/bin/env python3
"""The criteria of the tracker change (prompt ``tracker_uncertainty``, section 4), from the summary before the change
and the summary after it.  The criteria were fixed in the prompt before the test recordings were run; this script only
evaluates them, on the test recordings:

1. false tracks per minute at most half of the value before on every test recording, and at most 2 on ``multistatic_b``;
2. the simple harbor's wreck tracked by its echo with Pd at least 0.9 on every test seed;
3. no object's Pd (its own track kind) falls by more than 3 percentage points;
4. the 95 % ellipse of the tracks contains the true position in 85 to 99 % of the fixes (every fix of a confirmed track
   matched to an object, all test recordings together; each recording's fraction is listed beside).

    python examples/harbor/05_detection/harbor_tracker_criteria.py BEFORE.json AFTER.json [--out criteria.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(before: dict, after: dict) -> list[dict]:
    rb, ra = before["recordings"], after["recordings"]
    test = [k for k, v in ra.items() if v["held_out"]]
    out = []
    v = {k: {"before": rb[k]["false_tracks"]["per_minute"], "after": ra[k]["false_tracks"]["per_minute"]} for k in test}
    ok = all(x["after"] <= 0.5 * x["before"] + 1e-9 for x in v.values()) and v["multistatic_b"]["after"] <= 2.0
    out.append({"id": "false_tracks", "text": "false tracks per minute at most half of the value before on every test recording "
                                              "and at most 2 on multistatic_b", "values": v, "passed": ok})
    v = {f"simple_seed{s}": ra[f"simple_seed{s}"]["objects"]["wreck"]["pd"]["echo"] for s in (1, 2, 3)}
    out.append({"id": "wreck", "text": "the simple harbor's wreck at least 0.9 on every test seed", "values": v,
                "passed": all(x is not None and x >= 0.9 for x in v.values())})
    falls = {}
    for name in test:
        for oid, o in ra[name]["objects"].items():
            kind = "radiator" if o["radiates"] else "echo"
            pb, pa = rb[name]["objects"][oid]["pd"].get(kind), o["pd"].get(kind)
            if pb is not None and pa is not None and pb - pa > 0.03 + 1e-12:
                falls[f"{name}/{oid}"] = {"before": pb, "after": pa}
    out.append({"id": "no_fall", "text": "no object's Pd falls by more than 3 percentage points", "values": falls,
                "passed": not falls})
    per = {k: ra[k].get("ellipse_coverage") for k in test}
    n = sum(x["fixes"] for x in per.values() if x)
    m = sum(x["inside"] for x in per.values() if x)
    frac = m / n if n else None
    out.append({"id": "ellipse", "text": "the 95 % ellipse contains the truth in 85 to 99 % of fixes", "values": per,
                "fixes": n, "inside": m, "fraction": frac, "passed": frac is not None and 0.85 <= frac <= 0.99})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = evaluate(json.loads(Path(a.before).read_text()), json.loads(Path(a.after).read_text()))
    for c in res:
        print(f"{'pass' if c['passed'] else 'FAIL'}: {c['text']}")
    if a.out:
        Path(a.out).write_text(json.dumps({"criteria": res}, indent=2))


if __name__ == "__main__":
    main()

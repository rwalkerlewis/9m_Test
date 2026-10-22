#!/usr/bin/env python3
"""The criteria of the direct path change on the chain (prompt ``direct_path``, section 2), from the summary before
the change and the summary after it.  The criteria were fixed in the prompt before the chain was run; this script
only evaluates them:

1. uuv1 tracked by its echo with Pd at least 0.5 on each multistatic test seed (seeds 1 to 3);
2. at least three of the seven static bodies (the six pilings and the moored hull) tracked on ``multistatic_b``
   (a body counts as tracked when a confirmed echo track is matched to it within 5 m in at least one frame: echo Pd
   above 0);
3. no object's Pd (its own track kind) falls by more than 3 percentage points on any test recording;
4. compute per second of data at most 1.5 on every 0.5 m recording (``bistatic``, ``multistatic`` and their seeds,
   ``multistatic_b``).

    python examples/harbor/05_detection/harbor_direct_path_criteria.py BEFORE.json AFTER.json [--out criteria.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STATIC = ["piling1", "piling2", "piling3", "piling4", "piling5", "piling6", "moored"]


def evaluate(before: dict, after: dict) -> list[dict]:
    rb, ra = before["recordings"], after["recordings"]
    out = []
    v = {f"multistatic_seed{s}": ra[f"multistatic_seed{s}"]["objects"]["uuv1"]["pd"]["echo"] for s in (1, 2, 3)}
    out.append({"id": "uuv1", "text": "uuv1 tracked with Pd at least 0.5 on the multistatic seeds", "values": v,
                "passed": all(x >= 0.5 for x in v.values())})
    v = {b: ra["multistatic_b"]["objects"][b]["pd"]["echo"] for b in STATIC}
    n = sum(x > 0 for x in v.values())
    out.append({"id": "static_bodies", "text": "at least three of the seven static bodies tracked on multistatic_b",
                "values": v, "tracked": n, "passed": n >= 3})
    falls = {}
    for name, a in ra.items():
        if not a["held_out"]:
            continue
        for oid, o in a["objects"].items():
            kind = "radiator" if o["radiates"] else "echo"
            pb, pa = rb[name]["objects"][oid]["pd"].get(kind), o["pd"].get(kind)
            if pb is not None and pa is not None and pb - pa > 0.03 + 1e-12:
                falls[f"{name}/{oid}"] = {"before": pb, "after": pa}
    out.append({"id": "no_fall", "text": "no object's Pd falls by more than 3 percentage points (test recordings)",
                "values": falls, "passed": not falls})
    v = {k: a["compute_per_second_of_data"]["total"] for k, a in ra.items() if a["family"] in ("bistatic", "multistatic", "multistatic_b")}
    out.append({"id": "compute", "text": "compute per second of data at most 1.5 on the 0.5 m recordings", "values": v,
                "passed": all(x <= 1.5 for x in v.values())})
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

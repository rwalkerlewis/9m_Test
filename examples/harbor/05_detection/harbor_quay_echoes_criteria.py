#!/usr/bin/env python3
"""The criteria of the quay echo change on the chain (prompt ``run_all_3``, task 2, section 3), from the summary before
the change and the summary after it.  Fixed in the prompt before the measurement of section 1 was seen and written here
before any chain run with a change:

1. ``false_tracks``: at most 2 false tracks per minute on ``multistatic_b`` and at most 3 per minute on every 20 s test
   recording (the simple, bistatic and multistatic seeds 1 to 3), on the old noise model;
2. ``no_fall``: no object's Pd (its own track kind: radiator for what radiates, echo for a silent body) falls by more
   than 3 percentage points on any test recording, on either noise model;
3. ``compute``: compute per second of data not up by more than 10 percent on any recording (old noise model).

Without *after* the file is written with the values before and ``after`` empty.

    python examples/harbor/05_detection/harbor_quay_echoes_criteria.py BEFORE.json [AFTER.json] [--out criteria.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEXT = {
    "false_tracks": "false tracks at most 2 per minute on multistatic_b and at most 3 per minute on every 20 s test "
                    "recording, old noise model",
    "no_fall": "no object's Pd falls by more than 3 points on either noise model (test recordings)",
    "compute": "compute per second of data not up by more than 10 percent (every recording, old noise model)",
}


def _test_20s(rec: dict) -> list[str]:
    return [k for k, a in rec.items() if a["held_out"] and a["family"] != "harbor"]


def evaluate(before: dict, after: dict | None) -> list[dict]:
    rb = before["recordings"]
    ra = after["recordings"] if after else None
    out = []
    keys = _test_20s(rb)
    lim = {k: (2.0 if k == "multistatic_b" else 3.0) for k in keys}
    vb = {k: rb[k]["false_tracks"]["per_minute"] for k in keys}
    va = {k: ra[k]["false_tracks"]["per_minute"] for k in keys} if ra else None
    out.append({"id": "false_tracks", "text": TEXT["false_tracks"], "limits": lim, "before": vb, "values": va,
                "passed": None if va is None else all(va[k] <= lim[k] for k in keys)})
    falls = {}
    for model in ("recordings", "recordings_noise2"):
        if after is None:
            break
        for name, a in after.get(model, {}).items():
            if not a["held_out"] or name not in before.get(model, {}):
                continue
            for oid, o in a["objects"].items():
                kind = "radiator" if o["radiates"] else "echo"
                b0, a0 = before[model][name]["objects"][oid]["pd"].get(kind), o["pd"].get(kind)
                if b0 is not None and (a0 or 0.0) < b0 - 0.03 - 1e-12:
                    falls[f"{model}/{name}/{oid}"] = {"before": b0, "after": a0}
    out.append({"id": "no_fall", "text": TEXT["no_fall"], "values": falls if after else None,
                "passed": None if after is None else not falls})
    v = {k: {"before": b["compute_per_second_of_data"]["total"],
             "after": ra[k]["compute_per_second_of_data"]["total"] if ra and k in ra else None} for k, b in rb.items()}
    out.append({"id": "compute", "text": TEXT["compute"], "values": v,
                "passed": None if ra is None else all(x["after"] <= 1.10 * x["before"] for x in v.values())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after", nargs="?")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = evaluate(json.loads(Path(a.before).read_text()), json.loads(Path(a.after).read_text()) if a.after else None)
    for c in res:
        print(f"{'-' if c['passed'] is None else 'pass' if c['passed'] else 'FAIL'}: {c['text']}")
    if a.out:
        Path(a.out).write_text(json.dumps({"criteria": res}, indent=1))


if __name__ == "__main__":
    main()

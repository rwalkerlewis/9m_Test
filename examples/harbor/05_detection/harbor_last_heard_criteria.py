#!/usr/bin/env python3
"""The criteria of the last heard state on the chain (prompt ``run_all_3``, task 3, section 2), from the summary before
the change and the summary after it.  Fixed in the prompt and written here before any test recording was run:

1. ``boat``: the boat on ``multistatic_b`` at least 0.8 counting last heard frames inside the ellipse
   (``pd_last_heard``, radiator tracks; old noise model, the recording the prompt names);
2. ``false_tracks``: no rise in false tracks: on every test recording and both noise models, the count after not above
   the count before.

Reported beside them, not criteria: the boat under harbor_v2 and every object's ``pd`` and ``pd_last_heard``.

    python examples/harbor/05_detection/harbor_last_heard_criteria.py BEFORE.json AFTER.json [--out criteria.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(before: dict, after: dict) -> list[dict]:
    out = []
    o = after["recordings"]["multistatic_b"]["objects"]["boat"]
    v = (o.get("pd_last_heard") or {}).get("radiator")
    o2 = after.get("recordings_noise2", {}).get("multistatic_b", {}).get("objects", {}).get("boat", {})
    out.append({"id": "boat", "text": "the boat on multistatic_b at least 0.8 counting last heard frames inside the ellipse",
                "values": v, "pd_without_last_heard": o["pd"]["radiator"],
                "before": before["recordings"]["multistatic_b"]["objects"]["boat"]["pd"]["radiator"],
                "harbor_v2": {"pd_last_heard": (o2.get("pd_last_heard") or {}).get("radiator"), "pd": o2.get("pd", {}).get("radiator")},
                "passed": v is not None and v >= 0.8})
    vals, rise = {}, {}
    for model in ("recordings", "recordings_noise2"):
        for name, a in after.get(model, {}).items():
            if not a["held_out"] or name not in before.get(model, {}):
                continue
            b0, a0 = before[model][name]["false_tracks"]["count"], a["false_tracks"]["count"]
            vals[f"{model}/{name}"] = {"before": b0, "after": a0}
            if a0 > b0:
                rise[f"{model}/{name}"] = {"before": b0, "after": a0}
    out.append({"id": "false_tracks", "text": "no rise in false tracks (every test recording, both noise models)",
                "values": vals, "rises": rise, "passed": not rise})
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
        Path(a.out).write_text(json.dumps({"criteria": res}, indent=1))


if __name__ == "__main__":
    main()

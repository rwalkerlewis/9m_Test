#!/usr/bin/env python3
"""The ECA in mixed precision against double on the stage 0a cases: the table of two ``harbor_cancellation_depth.py``
runs that differ only in ``--eca-precision`` (prompt ``float32``, section 2).

Criterion (fixed in the prompt before any run): the cancellation depth of every case and reference within 0.3 dB of the
double run.  Beside it, the reference and ECA time per second of data of each run, on the device the runs used.  Writes
``<name>.json`` and ``<name>.md``.

    python examples/harbor/05_detection/harbor_eca_precision.py --double <dir> --mixed <dir> --name eca_stage0a_gpu \\
        [--out output/studies/float32]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CRITERION_DB = 0.3


def rows(results: dict) -> dict:
    out = {}
    for case in results["cases"]:
        label = f"{case['mode']} {case['range_min_m']:.0f} m"
        for ref, r in case["rows"].items():
            out[(label, ref)] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--double", required=True, help="output folder of the run with --eca-precision float64")
    ap.add_argument("--mixed", required=True, help="output folder of the run with --eca-precision mixed")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--name", default="eca_stage0a_gpu")
    ap.add_argument("--out", default="output/studies/float32")
    a = ap.parse_args()
    d = rows(json.loads((Path(a.double) / "results.json").read_text()))
    m = rows(json.loads((Path(a.mixed) / "results.json").read_text()))
    table = []
    for key in d:
        if key not in m:
            continue
        table.append({"case": key[0], "reference": key[1], "double_db": d[key]["cable_median_db"], "mixed_db": m[key]["cable_median_db"],
                      "difference_db": m[key]["cable_median_db"] - d[key]["cable_median_db"],
                      "double_s_per_s": d[key]["compute_s_per_s"], "mixed_s_per_s": m[key]["compute_s_per_s"]})
    worst = max(abs(r["difference_db"]) for r in table)
    res = {"criterion_db": CRITERION_DB, "device": a.device, "max_abs_difference_db": worst, "passed": bool(worst <= CRITERION_DB),
           "rows": table}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.json").write_text(json.dumps(res, indent=2))
    lines = ["| Case | Reference | Depth, double [dB] | Depth, mixed [dB] | Difference [dB] "
             f"| Reference and ECA, s per s of data ({a.device}), double | mixed |", "|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['case']} | {r['reference']} | {r['double_db']:.2f} | {r['mixed_db']:.2f} | {r['difference_db']:+.3f} "
                     f"| {r['double_s_per_s']:.2f} | {r['mixed_s_per_s']:.2f} |")
    lines.append("")
    lines.append(f"Criterion: every depth within {CRITERION_DB} dB of the double run: {'pass' if res['passed'] else 'FAIL'} "
                 f"(largest difference {worst:.2e} dB).")
    (out / f"{a.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

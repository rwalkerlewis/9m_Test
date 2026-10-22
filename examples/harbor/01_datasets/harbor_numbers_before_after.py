#!/usr/bin/env python3
"""Every Pd, median error and false track count of two ``output/reproduce/summary.json`` files side by side, with the
propagation error of each recording's forward model beside them (the worst hydrophone), as a markdown table.  "Moved"
marks a number that differs by more than the tolerance of ``scripts/reproduce.py`` (fractions 3 percentage points,
medians 0.5 m, false tracks exactly).

    python examples/harbor/01_datasets/harbor_numbers_before_after.py BEFORE.json AFTER.json [--out table.md] [--compute]

``git show <commit>:output/reproduce/summary.json > before.json`` gives a committed summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL_FRACTION = 0.03
TOL_METRES = 0.5


def worst_propagation_error(numbers: dict, family: str) -> float | None:
    p = numbers.get("propagation_error_ms", {}).get(family)
    return None if p is None else max((h["error_ms"] for h in p["hydrophones"]), key=abs)


def table(before: dict, after: dict, compute: bool = False) -> str:
    """The table; with *compute* the second column is the compute per second of data instead of the propagation
    error."""
    second = "Compute per s of data before / after" if compute else "Propagation error, worst hydrophone [ms] before / after"
    rows = [f"| Recording | {second} | Object (track kind) | Pd before / after | "
            "Median error [m] before / after | Moved |", "|---|---|---|---|---|---|"]
    pct = lambda v: "-" if v is None else f"{100 * v:.0f} %"
    met = lambda v: "-" if v is None else f"{v:.1f}"
    for name, ra in after["recordings"].items():
        rb = before["recordings"].get(name)
        fam = ra["family"]
        pb, pa = worst_propagation_error(before, fam), worst_propagation_error(after, fam)
        prop = f"{'-' if pb is None else f'{pb:+.2f}'} / {'-' if pa is None else f'{pa:+.2f}'}"
        if compute:
            cb = rb["compute_per_second_of_data"]["total"] if rb else None
            prop = f"{'-' if cb is None else f'{cb:.2f}'} / {ra['compute_per_second_of_data']['total']:.2f}"
        for oid, oa in ra["objects"].items():
            kind = "radiator" if oa["radiates"] else "echo"
            ob = rb["objects"].get(oid) if rb else None
            pd_a, er_a = oa["pd"].get(kind), oa["median_error_m"].get(kind)
            pd_b = ob["pd"].get(kind) if ob else None
            er_b = ob["median_error_m"].get(kind) if ob else None
            if not pd_a and not pd_b:
                continue
            moved = []
            if pd_a is not None and pd_b is not None and abs(pd_a - pd_b) > TOL_FRACTION + 1e-12:
                moved.append("Pd")
            if er_a is not None and er_b is not None and abs(er_a - er_b) > TOL_METRES:
                moved.append("error")
            rows.append(f"| {name} | {prop} | {oid} ({kind}) | {pct(pd_b)} / {pct(pd_a)} | {met(er_b)} / {met(er_a)} | "
                        f"{', '.join(moved)} |")
        fa = ra["false_tracks"]
        fb = rb["false_tracks"] if rb else None
        fbs = "-" if fb is None else f"{fb['count']} ({fb['per_minute']:.1f})"
        rows.append(f"| {name} | {prop} | false tracks (per minute) | {fbs} / {fa['count']} ({fa['per_minute']:.1f}) | | "
                    f"{'count' if fb is not None and fb['count'] != fa['count'] else ''} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compute", action="store_true", help="compute per second of data instead of the propagation error")
    a = ap.parse_args()
    md = table(json.loads(Path(a.before).read_text()), json.loads(Path(a.after).read_text()), compute=a.compute)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
    print(md, end="")


if __name__ == "__main__":
    main()

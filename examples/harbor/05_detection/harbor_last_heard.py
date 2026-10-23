#!/usr/bin/env python3
"""The hold time of the last heard state (prompt ``run_all_3``, task 3, section 2), set on the tuning recordings.

An operator wants a ship that stops and goes quiet to stay on the picture: a confirmed radiator track that loses its
fixes is kept as last heard (``TrackerConfig.last_heard_s``, ``streaming.RadiatorTracker``) at its last filtered position
with a growing covariance, reported in ``picture.jsonl`` with ``status: "last_heard"`` and the time of its last fix, and
scored separately (``scoreboard.score_picture``: ``pd_last_heard`` counts a last heard frame as a hit only if the object
is inside the track's 95 % ellipse; ``pd`` as before).

Rule, fixed before any run: the radiator tracker's ``last_heard_s`` is run at 0 (off), 10, 30 and 60 s on the four
tuning recordings (noise seed 0 of ``simple``, ``bistatic``, ``multistatic``, ``harbor``; old noise model); the value
taken is the largest of 10, 30 and 60 s with no more false tracks than at 0 on any tuning recording and no object's
``pd`` (its own track kind) down by more than 3 points against 0; 30 s if two or more qualify and 30 is among them, the
prompt's default; if none qualifies, the state stays off (0).  The echo trackers are not changed.

Criteria on the test recordings, fixed in the prompt before any test run (``harbor_last_heard_criteria.py``): the boat
on ``multistatic_b`` at least 0.8 counting last heard frames inside the ellipse (``pd_last_heard``, radiator tracks);
no rise in false tracks (a last heard track that never regains a fix is false only if it was false when confirmed, which
the scoring's match over the confirmed fixes already does): on every test recording, on both noise models, the count
not above its value before.

Outputs in ``output/studies/last_heard/``: ``tuning.json`` (per value and recording the false tracks, every ``pd`` and
``pd_last_heard``; the value taken), ``run.log``.

    python examples/harbor/05_detection/harbor_last_heard.py [--values 0 10 30 60] [--cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))

import harbor_stream as hs  # noqa: E402

from acoustic_sim.contract import contract_paths  # noqa: E402

OUT = ROOT / "output" / "studies" / "last_heard"
TUNING = ("simple", "bistatic", "multistatic", "harbor")
RULE = ("the largest of 10, 30, 60 s with no more false tracks than at 0 on any tuning recording and no pd down more than "
        "3 points; 30 s if it qualifies with another; off if none qualifies")


def choose(rows: dict) -> float:
    """The value taken by the rule from ``rows[value][recording]`` (summaries)."""
    base = rows[0.0]
    ok = []
    for v in (10.0, 30.0, 60.0):
        if v not in rows:
            continue
        good = True
        for rec, s in rows[v].items():
            b = base[rec]
            if s["false_tracks"]["count"] > b["false_tracks"]["count"]:
                good = False
            for oid, o in s["objects"].items():
                kind = "radiator" if o["radiates"] else "echo"
                x, y = b["objects"][oid]["pd"].get(kind), o["pd"].get(kind)
                if x is not None and (y or 0.0) < x - 0.03 - 1e-12:
                    good = False
        if good:
            ok.append(v)
    if not ok:
        return 0.0
    if 30.0 in ok and len(ok) >= 2:
        return 30.0
    return max(ok)


def main() -> None:
    import reproduce as R

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--values", nargs="*", type=float, default=[0.0, 10.0, 30.0, 60.0])
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "run.log", "a")
    say = lambda m: (print(m, flush=True), log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n"), log.flush())  # noqa: E731
    rows: dict[float, dict] = {}
    recs = {r.name: r for r in R.recordings()}
    for v in a.values:
        for name in TUNING:
            r = recs[name]
            obs, truth = contract_paths(r.dataset.with_suffix(".npz"))
            say(f"{name}: tracker.last_heard_s={v:g}")
            s = hs.run_recording(f"{name}_last_heard_{v:g}", R.CONFIGS / "processing" / f"{r.processing}.json", obs, truth,
                                 OUT / "runs" / f"{name}_{v:g}", False, use_cuda=not a.cpu, seed=r.seed,
                                 overrides=[f"tracker.last_heard_s={v:g}"], video=False, figures=False)
            rows.setdefault(v, {})[name] = s
            say(f"  false tracks {s['false_tracks']['count']}; " + "; ".join(
                f"{k} {o['pd'].get('radiator' if o['radiates'] else 'echo')}/{o.get('pd_last_heard', {}).get('radiator' if o['radiates'] else 'echo')}"
                for k, o in s["objects"].items()))
    taken = choose(rows)
    res = {"rule": RULE, "values": a.values, "taken_s": taken,
           "provenance": {"truth_inputs": [], "simulator_knowledge": [], "data": "forward_model", "held_out": False},
           "runs": {f"{v:g}": {rec: {"false_tracks": s["false_tracks"]["count"],
                                     "pd": {k: o["pd"] for k, o in s["objects"].items()},
                                     "pd_last_heard": {k: o.get("pd_last_heard") for k, o in s["objects"].items()}}
                               for rec, s in rr.items()} for v, rr in rows.items()}}
    (OUT / "tuning.json").write_text(json.dumps(res, indent=2))
    say(f"last_heard_s taken: {taken:g} s")


if __name__ == "__main__":
    main()

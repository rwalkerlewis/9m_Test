#!/usr/bin/env python3
"""The one calibration of the source level estimate, on the simple harbor's tuning recording.

The chain estimates every radiator's source level at 100 m from its received level with cylindrical spreading
(``water/characteristics.py``), up to one scale: the units of the replicas against the units in which the forward model
calibrates its sources.  The simple harbor's ferry (1.0 Pa at 100 m) and boat (0.3 Pa) fix that scale, as a known
source would at sea: the constant is the geometric mean over the two of the true level over the median estimate of the
radiator track matched to it.  It is copied into ``characteristics.level_calibration`` of every processing config and
not changed afterwards; on every other recording the estimate is reported with its error against the truth.

Writes ``output/studies/level_calibration/calibration.json`` (with the ``run`` block).

    python examples/harbor/05_detection/harbor_level_calibration.py [--cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Observation, Provenance, Truth, contract_paths  # noqa: E402
from acoustic_sim.water.scoreboard import picture_tracks, match_tracks  # noqa: E402
from acoustic_sim.water.stream import load_processing, run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="output/studies/level_calibration")
    a = ap.parse_args()
    processing = ROOT / "examples" / "configs" / "processing" / "simple.json"
    cfg = load_processing(processing, ["characteristics.level_calibration=1.0"])
    obs_path = ROOT / cfg.raw["observation"]
    res = run(Observation.load(obs_path), cfg, use_cuda=not a.cpu, provenance=Provenance(held_out=False), observation_path=obs_path)
    truth = Truth.load(contract_paths(obs_path)[1])
    tracks = picture_tracks(res.records)
    matches = match_tracks({k: v["fixes"] for k, v in tracks.items() if v["kind"] == "radiator"}, truth, cfg.scoring["assoc_radius_m"])
    rows = {}
    for m in matches:
        oid = m["true_object"]
        if oid is None or truth.tracks[oid].level_pa <= 0:
            continue
        est = [w["source_level"] for w in res.picture.meas.get(m["track"], {}).get("frames", [])]
        if est:
            rows.setdefault(oid, []).extend(est)
    per = {oid: {"true_pa": truth.tracks[oid].level_pa, "estimate_uncalibrated_median": float(np.median(v)), "frames": len(v)}
           for oid, v in rows.items()}
    K = float(np.exp(np.mean([np.log(p["true_pa"] / p["estimate_uncalibrated_median"]) for p in per.values()])))
    for p in per.values():
        p["estimate_calibrated"] = K * p["estimate_uncalibrated_median"]
        p["error_db"] = 20 * np.log10(p["estimate_calibrated"] / p["true_pa"])
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps({"level_calibration": K, "objects": per, "run": dict(res.run, wall_s=res.wall_s)},
                                                     indent=2))
    print(f"level calibration constant {K:.4f}; " + "; ".join(f"{k}: {p['estimate_calibrated']:.3f} Pa against {p['true_pa']} Pa "
                                                                f"({p['error_db']:+.1f} dB)" for k, p in per.items()))


if __name__ == "__main__":
    main()

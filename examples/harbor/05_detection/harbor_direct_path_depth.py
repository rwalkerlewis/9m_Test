#!/usr/bin/env python3
"""How deep the echo chain removes each ship's direct path on the recordings, per reference mode (a diagnostic from
the truth components).

The system (``acoustic_sim.water.stream``, the recording's processing config with ``echo.bistatic.ref_mode`` set to
each mode given) runs on a recording while the truth's direct component of every ship, scaled by the channel gains,
goes through the same ECA projections as the traces (``stream.run(components=...)``; the processing never reads
them).  Per ship: the median over the live cable channels of the band energy of its direct component before and after
the chain's removals, in dB, over every block the chain cleaned.  Nothing is scored here; ``scripts/reproduce.py``
scores the runs.

    python examples/harbor/05_detection/harbor_direct_path_depth.py --recordings simple bistatic multistatic multistatic_b \\
        --modes beam weighted [--out output/studies/direct_path]
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
from acoustic_sim.water.stream import load_processing, run  # noqa: E402

PROCESSING = {"simple": "simple", "bistatic": "bistatic", "multistatic": "multistatic", "multistatic_b": "multistatic"}


def depth_per_ship(component_energy: dict, live: np.ndarray, cable: slice) -> dict:
    out = {}
    for key, steps in component_energy.items():
        if not key.startswith(("direct:", "scattered:")) or not steps:
            continue
        e0 = np.sum([b for b, _ in steps], axis=0)
        e1 = np.sum([a for _, a in steps], axis=0)
        ok = live.copy()
        ok[:cable.start] = False
        ok[cable.stop:] = False
        ok &= e0 > 0
        d = 10 * np.log10(np.maximum(e0[ok], 1e-300) / np.maximum(e1[ok], 1e-300))
        name = key.split(":", 1)[1] if key.startswith("direct:") else key
        out[name] = {"cable_median_db": float(np.median(d)), "cable_p10_db": float(np.percentile(d, 10)), "steps": len(steps)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", nargs="*", default=["simple", "bistatic", "multistatic", "multistatic_b"])
    ap.add_argument("--modes", nargs="*", default=["beam", "weighted"])
    ap.add_argument("--out", default="output/studies/direct_path")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--precision", default=None, choices=("float64", "mixed"), help="echo.bistatic.eca_precision of every run")
    ap.add_argument("--scattered", action="store_true", help="also pass the scattered components (the echo loss)")
    ap.add_argument("--name", default="recording_depth", help="stem of the output files")
    ap.add_argument("--set", nargs="*", default=[], help="further processing overrides (KEY=VALUE) of every run")
    ap.add_argument("--label", default=None, help="column name of the runs (default: the mode)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.name}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for rec in a.recordings:
        cfg_path = ROOT / "examples" / "configs" / "processing" / f"{PROCESSING[rec]}.json"
        obs_path = ROOT / "output" / "recordings" / rec / "dataset.observation.npz"
        obs = Observation.load(obs_path)
        truth = Truth.load(contract_paths(obs_path)[1])
        gain = np.asarray(truth.gain, dtype=np.float64)[:, None]
        comps = {k: gain * np.asarray(v, dtype=np.float64) for k, v in truth.components.items()
                 if k.startswith("direct:") or (a.scattered and k.startswith("scattered:"))}
        for mode in a.modes:
            cfg = load_processing(cfg_path, [f"echo.bistatic.ref_mode={mode}"]
                                  + ([f"echo.bistatic.eca_precision={a.precision}"] if a.precision else []) + a.set)
            r = run(obs, cfg, use_cuda=not a.cpu, provenance=Provenance(data="forward_model", truth_inputs=["components"]),
                    observation_path=obs_path, components=comps)
            echo = r.picture.echo
            live = np.ones(obs.geometry.n_receivers, bool)
            if truth.dead is not None:
                live &= ~np.asarray(truth.dead, bool)
            cable = obs.geometry.slices[obs.geometry.cables[0].name]
            key = a.label or mode
            res.setdefault(rec, {})[key] = {"ships": depth_per_ship(echo.component_energy, live, cable),
                                             "compute_per_second_of_data": r.compute_per_second, "run": r.run,
                                             "echo_peaks": [[f.t, np.asarray(f.xy).tolist()] for f in r.picture.e_frames]}
            print(rec, key, {k: round(v["cable_median_db"], 1) for k, v in res[rec][key]["ships"].items()},
                  f"compute {r.compute_per_second['total']:.2f} s/s", flush=True)
            path.write_text(json.dumps(res, indent=2, default=str))
    cols = sorted({m for r in res.values() for m in r}, key=lambda m: (m not in ("beam", "joint"), m))
    lines = ["| Recording | Ship | " + " | ".join(f"{m}: median / p10 [dB]" for m in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for rec in res:
        ships = sorted({s for m in res[rec].values() for s in m["ships"]})
        for s in ships:
            cells = [f"{res[rec][m]['ships'][s]['cable_median_db']:.1f} / {res[rec][m]['ships'][s]['cable_p10_db']:.1f}"
                     if m in res[rec] and s in res[rec][m]["ships"] else "-" for m in cols]
            lines.append(f"| {rec} | {s} | " + " | ".join(cells) + " |")
    (out / f"{a.name}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

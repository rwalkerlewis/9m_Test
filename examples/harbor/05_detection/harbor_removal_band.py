#!/usr/bin/env python3
"""The direct path removal on decimated traces (``BistaticConfig.removal_band``, prompt ``run_all_4``, task 1): the
removal depth of every ship on the multistatic recordings with the band on (the configs as they are) and off
(``echo.bistatic.removal_band=null``, the settings of ``main`` before the task).

The removal band of each family is the band its references' spectra occupy, measured in pull request 45
(``output/studies/removal_budget/band.json`` on the ``removal_budget`` branch: 1 % to 99 % of the references' power,
widened to the processing band): simple ``[10, 380]``, bistatic ``[78, 762]``, multistatic ``[100, 760]`` Hz, decimation 3
on each.

The depth of a ship is the median over the live cable channels of the energy its direct component (from the truth, a
diagnostic: ``truth_inputs = ["components"]``; a mixed seed takes the seed 0 recording's direct components, the same
gathers) loses through the chain's projections (``harbor_direct_path_depth.depth_per_ship``).

Criterion, fixed before the run (the task is structural: no number may move beyond the reproduce tolerance): on each of
``multistatic`` (noise seeds 0 to 3) and ``multistatic_b``, every ship's depth with the band within ``DEPTH_TOL_DB``
(0.5 dB) of its depth without it.  The Pd, median errors and false tracks are held to the tolerances of
``scripts/reproduce.py`` by its comparison with the committed ``summary.json`` of ``main``; compute must fall.

Outputs in ``output/studies/removal_band/``: ``results.json`` (the criterion first, then the depths) and ``run.log``.

    python examples/harbor/05_detection/harbor_removal_band.py [--cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))

from acoustic_sim.contract import Observation, Provenance, Truth, contract_paths  # noqa: E402
from acoustic_sim.water.stream import load_processing, run  # noqa: E402

OUT = ROOT / "output" / "studies" / "removal_band"
CONFIGS = ROOT / "examples" / "configs" / "processing"
RECORDINGS = ["multistatic", "multistatic_seed1", "multistatic_seed2", "multistatic_seed3", "multistatic_b"]
DEPTH_TOL_DB = 0.5
CRITERION = (f"every ship's removal depth with the decimated removal band within {DEPTH_TOL_DB} dB of its depth without "
             "it, on multistatic noise seeds 0 to 3 and multistatic_b")


def _say(msg: str) -> None:
    print(msg, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def depth(name: str, overrides: list[str], use_cuda: bool) -> dict:
    """The removal depth per ship [dB] of the chain on recording *name* with *overrides*, and its compute."""
    import harbor_direct_path_depth as dp
    import reproduce as R

    r = next(x for x in R.recordings() if x.name == name)
    obs_p, truth_p = contract_paths(r.dataset.with_suffix(".npz"))
    obs, truth = Observation.load(obs_p), Truth.load(truth_p)
    src = truth if truth.components is not None else Truth.load(contract_paths(ROOT / f"{R.FAMILIES[r.family].dataset}.npz")[1])
    gain = np.asarray(src.gain, dtype=np.float64)[:, None]
    comps = {k: gain * np.asarray(v, dtype=np.float64) for k, v in src.components.items() if k.startswith("direct:")}
    cfg = load_processing(CONFIGS / f"{r.processing}.json", overrides)
    res = run(obs, cfg, use_cuda=use_cuda, provenance=Provenance(data="forward_model", truth_inputs=["components"],
                                                                 held_out=r.held_out),
              observation_path=obs_p, components=comps)
    live = np.ones(obs.geometry.n_receivers, bool)
    if truth.dead is not None:
        live &= ~np.asarray(truth.dead, bool)
    cable = obs.geometry.slices[obs.geometry.cables[0].name]
    d = dp.depth_per_ship(res.picture.echo.component_energy, live, cable)
    return {"depth_db": {k: v["cable_median_db"] for k, v in d.items()},
            "compute_per_second_of_data": res.compute_per_second["total"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    res = {"criterion": {"text": CRITERION, "tolerance_db": DEPTH_TOL_DB, "passed": None},
           "provenance": {"truth_inputs": ["components"], "simulator_knowledge": [], "held_out": "test seeds and multistatic_b"},
           "recordings": {}}
    path.write_text(json.dumps(res, indent=2))
    worst = 0.0
    for name in RECORDINGS:
        off = depth(name, ["echo.bistatic.removal_band=null"], not a.cpu)
        on = depth(name, [], not a.cpu)
        diff = {k: on["depth_db"][k] - off["depth_db"][k] for k in on["depth_db"]}
        worst = max(worst, max(abs(v) for v in diff.values()))
        res["recordings"][name] = {"without_band": off, "with_band": on, "difference_db": diff}
        _say(f"{name}: depth without the band {off['depth_db']}, with it {on['depth_db']}; compute "
             f"{off['compute_per_second_of_data']:.2f} -> {on['compute_per_second_of_data']:.2f} s/s")
        path.write_text(json.dumps(res, indent=2))
    res["criterion"]["largest_difference_db"] = worst
    res["criterion"]["passed"] = bool(worst <= DEPTH_TOL_DB)
    path.write_text(json.dumps(res, indent=2))
    _say(f"{'pass' if res['criterion']['passed'] else 'FAIL'}: {CRITERION} (largest difference {worst:.2f} dB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Which assumed number of the ``harbor_v2`` noise model matters (prompt ``noise_dependence``, section 2): one assumed
parameter at a time over a stated range, everything else at its documented value (``docs/noise.md``), the chain
unchanged, on the multistatic and the bistatic tuning recordings (noise seed 0, mixed from the cached gathers).

The parameters and their values (the documented value in the middle):

* ``shrimp.band_fraction_db``: -50, -40, -30 dB (the click's energy in the simulated band, 10 dB below and above);
* ``shrimp.rate_per_m``: 0.03, 0.3, 3 clicks per second per metre of land edge (10 dB below and above);
* ``machinery.level_pa``: off, 0.1, 0.316 Pa (assumed, 10 dB above).  The machinery needs its own FDTD run per harbor
  and only the simple harbor has one (cached by the noise study); with no FDTD run allowed it is swept on the simple
  harbor's tuning recording instead;
* ``fibre.drift_rms``: 0.0063, 0.02, 0.063 (10 dB below and above);
* ``interrogator.floor_rms``: 0.016, 0.05, 0.16 (10 dB below and above);
* ``interrogator.fade_fraction``: 0.05, 0.10, 0.20 (half and double).

No criterion: a measurement.  For each parameter and value, the Pd of every silent body (its echo tracks) and the false
tracks per minute; one figure per object (Pd against the parameter's setting, one series per parameter) and one per
recording for the false tracks.  Outputs in ``output/studies/noise_model/sweep/``: ``results.json`` (every parameter
value used and every run's numbers), the figures, ``run.log``; the runs under ``runs/``.

    python examples/harbor/05_detection/harbor_noise_sweep.py [--recordings multistatic bistatic] [--cpu]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harbor_noise_model as hnm  # noqa: E402
import harbor_stream as hs  # noqa: E402

from acoustic_sim.contract import Truth, contract_paths  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water import HarborScenarioConfig, assemble_dataset, build_environment, build_layout, build_track  # noqa: E402
from acoustic_sim.water.dataset import machinery_gather  # noqa: E402
from acoustic_sim.water.noise import HarborV2Config  # noqa: E402

PARAMETERS = {
    "shrimp.band_fraction_db": [-50.0, -40.0, -30.0],
    "shrimp.rate_per_m": [0.03, 0.3, 3.0],
    "fibre.drift_rms": [0.02 / np.sqrt(10), 0.02, 0.02 * np.sqrt(10)],
    "interrogator.floor_rms": [0.05 / np.sqrt(10), 0.05, 0.05 * np.sqrt(10)],
    "interrogator.fade_fraction": [0.05, 0.10, 0.20],
}
MACHINERY = ("machinery.level_pa", [None, 0.1, 0.1 * np.sqrt(10)])     # None: off
SETTINGS = ("low", "documented", "high")
MOVED = 0.10                                                          # "moves a silent body by more than 10 points"


def mix_and_run(rec: str, overrides: dict, tag: str, out: Path, use_cuda: bool) -> dict:
    proc, combined = hnm.RECORDINGS[rec]
    base = Truth.load(ROOT / "output" / "recordings" / rec / "dataset.truth.npz")
    cfg = HarborScenarioConfig.from_dict(copy.deepcopy(base.scenario))
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    gathers = hnm.gathers_from_cache(rec, cfg, env, layout, tracks, combined)
    v2 = hnm.v2_config(hnm.COMPONENTS, rec == "simple")
    for key, value in overrides.items():
        comp, field = key.split(".")
        if value is None:
            v2[comp]["enabled"] = False
        else:
            v2[comp][field] = float(value)
    cfg.noise.model = "harbor_v2"
    cfg.noise.harbor_v2 = HarborV2Config.from_dict(v2)
    mach = None
    if rec == "simple" and v2["machinery"]["enabled"]:
        mach = machinery_gather(cfg, env, layout, cache_dir=ROOT / "output" / "studies" / "noise_model" / "simple" / "cache")
    d = out / "runs" / rec / tag
    d.mkdir(parents=True, exist_ok=True)
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=0, keep_clean=False, env=env, machinery=mach)
    obs_p, tru_p = contract_paths(d / "dataset.npz")
    ds.truth.meta = dict(ds.truth.meta, noise_seed=0, mixed_from=f"output/recordings/{rec}", noise_overrides=overrides,
                         propagation_error_ms=base.meta.get("propagation_error_ms"))
    ds.observation.save(obs_p)
    ds.truth.save(tru_p)
    s = hs.run_recording(f"{rec}_{tag}", ROOT / "examples" / "configs" / "processing" / f"{proc}.json", obs_p, tru_p, d, False,
                         use_cuda=use_cuda, seed=0, video=False, figures=False)
    r = hnm.summarise(s)
    r["harbor_v2"] = v2
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", nargs="*", default=["multistatic", "bistatic"])
    ap.add_argument("--out", default="output/studies/noise_model/sweep")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res.update(criterion="none: a measurement (prompt noise_dependence)", seed=0, latitude_deg=hnm.LATITUDE,
               parameters={k: [float(x) for x in v] for k, v in PARAMETERS.items()},
               machinery={"parameter": MACHINERY[0], "values": [None if x is None else float(x) for x in MACHINERY[1]],
                          "recording": "simple", "why": "no machinery gather on the other recordings, and no FDTD run allowed"},
               settings=list(SETTINGS))
    log = open(out / "run.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(time.strftime("%H:%M:%S ") + msg + "\n")
        log.flush()

    jobs = []
    for rec in a.recordings:
        for key, vals in PARAMETERS.items():
            for setting, v in zip(SETTINGS, vals):
                jobs.append((rec, key, setting, v))
    jobs += [("simple", MACHINERY[0], s, v) for s, v in zip(SETTINGS, MACHINERY[1])]
    if not a.plot_only:
        for rec, key, setting, v in jobs:
            runs = res.setdefault("runs", {}).setdefault(rec, {}).setdefault(key, {})
            if setting in runs:
                continue
            if setting == "documented" and any("documented" in p for p in res["runs"][rec].values() if p is not runs):
                runs[setting] = next(p["documented"] for p in res["runs"][rec].values() if "documented" in p)
            else:
                tag = "documented" if setting == "documented" else f"{key.replace('.', '_')}_{setting}"
                runs[setting] = dict(mix_and_run(rec, {key: v}, tag, out, not a.cpu), value=None if v is None else float(v))
            r = runs[setting]
            say(f"{rec} {key} {setting} ({r.get('value')}): false tracks {r['false_tracks_per_minute']:.1f}/min, "
                + ", ".join(f"{o} {x['pd']:.2f}" for o, x in r["objects"].items() if x["pd"] is not None))
            path.write_text(json.dumps(res, indent=1, default=float))
    # Which parameters move a silent body by more than 10 points over their range.
    moved = {}
    for rec, per in res.get("runs", {}).items():
        for key, runs in per.items():
            silent = [o for o, x in next(iter(runs.values()))["objects"].items() if o in SILENT.get(rec, [])]
            for o in silent:
                vals = [runs[s]["objects"][o]["pd"] or 0.0 for s in SETTINGS if s in runs]
                span = max(vals) - min(vals)
                moved.setdefault(rec, {}).setdefault(key, {})[o] = span
    res["span_of_pd"] = moved
    res["moves_more_than_10_points"] = {rec: sorted({k for k, d in per.items() if any(v > MOVED for v in d.values())})
                                        for rec, per in moved.items()}
    res["moves_nothing"] = {rec: sorted({k for k, d in per.items() if all(v <= 0.03 for v in d.values())})
                            for rec, per in moved.items()}
    path.write_text(json.dumps(res, indent=1, default=float))
    figures(res, out)
    say(f"moves a silent body by more than 10 points: {res['moves_more_than_10_points']}; moves nothing (3 points or less): "
        f"{res['moves_nothing']}")


SILENT = {"multistatic": ["uuv1", "uuv2", "diver", "piling1", "piling2", "piling3", "piling4", "piling5", "piling6", "moored"],
          "bistatic": ["uuv1"], "simple": ["uuv1", "wreck"]}


def figures(res: dict, out: Path) -> None:
    x = np.arange(len(SETTINGS))
    labels = ["10 dB below, half, or off", "documented", "10 dB above, or double"]
    for rec, per in res.get("runs", {}).items():
        for o in SILENT.get(rec, []):
            fig, ax = new_figure((12, 8))
            for key, runs in per.items():
                y = [runs[s]["objects"][o]["pd"] if s in runs and runs[s]["objects"][o]["pd"] is not None else np.nan for s in SETTINGS]
                ax.plot(x, 100 * np.array(y, float), "o-", label=key)
            ax.set_xticks(x, labels)
            ax.set_xlabel("setting of the parameter (the others at their documented values)")
            ax.set_ylabel("Pd by its echo [%]")
            ax.set_ylim(-5, 105)
            ax.set_title(f"{rec} (tuning, harbor_v2): {o}, Pd against each assumed noise parameter")
            ax.legend()
            save_figure(fig, out / f"pd_{rec}_{o}.png", recording=rec)
        fig, ax = new_figure((12, 8))
        for key, runs in per.items():
            y = [runs[s]["false_tracks_per_minute"] if s in runs else np.nan for s in SETTINGS]
            ax.plot(x, y, "o-", label=key)
        ax.set_xticks(x, labels)
        ax.set_xlabel("setting of the parameter (the others at their documented values)")
        ax.set_ylabel("false tracks per minute")
        ax.set_title(f"{rec} (tuning, harbor_v2): false tracks against each assumed noise parameter")
        ax.legend()
        save_figure(fig, out / f"false_tracks_{rec}.png", recording=rec)


if __name__ == "__main__":
    main()

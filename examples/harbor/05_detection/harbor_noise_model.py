#!/usr/bin/env python3
"""How much the numbers depend on the noise model: the recordings mixed again with ``noise.model = "harbor_v2"``.

From the cached FDTD gathers of ``simple``, ``bistatic``, ``multistatic`` and ``multistatic_b`` (no FDTD run for
them), noise seeds 0 to 2 are mixed twice, with the old model and with ``harbor_v2`` (every component on: snapping
shrimp, surface agitation, fibre strain noise, interrogator noise, and on the simple harbor the quay machinery, one FDTD
run of a pump on the quay, cached), and the system runs on each with the processing configs exactly as on ``main``
(``harbor_stream.run_recording``).  Then on the simple harbor's seed 0 each component is switched on alone.  The
parameters of every component and their status (measured, derived, assumed) are in ``docs/noise.md``.

No pass or fail criterion: a measurement of how far Pd, median error, false tracks and compute move with the noise
model, and which component moves them.  Outputs in ``output/studies/noise_model/``: ``<recording>/seed<k>/<model>/``
(the run, its figures off), ``results.json``, ``table.md``, ``components.md``, the noise spectra figures, ``run.log``;
the mixed observation and truth files are not committed.

    python examples/harbor/05_detection/harbor_noise_model.py [--recordings simple ...] [--seeds 0 1 2] [--cpu]
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
sys.path.insert(0, str(ROOT / "examples" / "harbor" / "05_detection"))

import harbor_stream as hs  # noqa: E402
from acoustic_sim.contract import Truth, contract_paths  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water import HarborScenarioConfig, assemble_dataset, build_environment, build_layout, build_track  # noqa: E402
from acoustic_sim.water import fdtd_runner, render_vessel_gathers  # noqa: E402
from acoustic_sim.water.dataset import machinery_gather  # noqa: E402
from acoustic_sim.water.noise import HarborV2Config, generate_noise  # noqa: E402

RECORDINGS = {"simple": ("simple", True), "bistatic": ("bistatic", False), "multistatic": ("multistatic", True),
              "multistatic_b": ("multistatic", True)}               # processing config, combined body gathers
COMPONENTS = ("shrimp", "surface", "machinery", "fibre", "interrogator")
LATITUDE = 30.0            # assumed harbor latitude (inside the snapping shrimp's range), so the shrimp component is on
MACHINE = [(250.0, 375.0)]  # one pump 5 m off the simple harbor's quay (its quay is y 380 to 400 m)


def v2_config(components, with_machinery: bool) -> dict:
    c = HarborV2Config().to_dict()
    for k in components:
        if k == "machinery" and not with_machinery:
            continue
        c[k]["enabled"] = True
    c["shrimp"]["latitude_deg"] = LATITUDE
    if with_machinery:
        c["machinery"]["positions"] = [list(p) for p in MACHINE]
    return c


def gathers_from_cache(rec: str, cfg, env, layout, tracks, combined: bool):
    def no_fdtd(*a, **k):
        raise RuntimeError(f"{rec}: a gather is missing from the cache")

    saved = fdtd_runner.HarborFDTDBatch.run, fdtd_runner.HarborFDTDSolver.run
    fdtd_runner.HarborFDTDBatch.run = fdtd_runner.HarborFDTDSolver.run = no_fdtd
    try:
        return render_vessel_gathers(cfg, env, layout, tracks, cache_dir=ROOT / "output" / "recordings" / rec / "cache",
                                     combined_bodies=combined)
    finally:
        fdtd_runner.HarborFDTDBatch.run, fdtd_runner.HarborFDTDSolver.run = saved


def summarise(s: dict) -> dict:
    objs = {o: {"pd": v["pd"].get("radiator" if v["radiates"] else "echo"),
                "median_error_m": v["median_error_m"].get("radiator" if v["radiates"] else "echo")} for o, v in s["objects"].items()}
    return {"objects": objs, "false_tracks_per_minute": s["false_tracks"]["per_minute"],
            "false_tracks": {k: s["false_tracks"][k] for k in ("count", "radiator", "echo")},
            "compute_per_second_of_data": s["compute_per_second_of_data"]["total"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", nargs="*", default=list(RECORDINGS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--no-components", action="store_true", help="skip the one component at a time runs")
    ap.add_argument("--out", default="output/studies/noise_model")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res["criterion"] = "none: a measurement of how far the numbers move with the noise model (prompt ci_and_noise_model)"
    res["assumptions"] = {"latitude_deg": LATITUDE, "machinery_positions": MACHINE,
                          "machinery": "on the simple harbor only (one FDTD run); the other recordings have no machinery gather"}
    log = open(out / "run.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(time.strftime("%H:%M:%S ") + msg + "\n")
        log.flush()

    def run_mix(rec, seed, model, components, tag):
        proc, combined = RECORDINGS[rec]
        base = Truth.load(ROOT / "output" / "recordings" / rec / "dataset.truth.npz")
        cfg = HarborScenarioConfig.from_dict(copy.deepcopy(base.scenario))
        env = build_environment(cfg)
        layout = build_layout(cfg, env)
        tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
        gathers = gathers_from_cache(rec, cfg, env, layout, tracks, combined)
        mach = None
        if model == "harbor_v2":
            cfg.noise.model = "harbor_v2"
            cfg.noise.harbor_v2 = HarborV2Config.from_dict(v2_config(components, rec == "simple"))
            if rec == "simple" and "machinery" in components:
                cfg.fdtd.use_cuda = not a.cpu
                mach = machinery_gather(cfg, env, layout, cache_dir=out / "simple" / "cache")
        d = out / rec / f"seed{seed}" / tag
        d.mkdir(parents=True, exist_ok=True)
        ds = assemble_dataset(cfg, layout, tracks, gathers, seed=seed, keep_clean=False, env=env, machinery=mach)
        obs_p, tru_p = contract_paths(d / "dataset.npz")
        ds.truth.meta = dict(ds.truth.meta, noise_seed=seed, mixed_from=f"output/recordings/{rec}",
                             propagation_error_ms=base.meta.get("propagation_error_ms"))
        ds.observation.save(obs_p)
        ds.truth.save(tru_p)
        s = hs.run_recording(f"{rec}_seed{seed}_{tag}", ROOT / "examples" / "configs" / "processing" / f"{proc}.json", obs_p, tru_p, d,
                             seed > 0 or rec == "multistatic_b", use_cuda=not a.cpu, seed=seed, video=False, figures=False)
        r = summarise(s)
        r["noise_parts"] = ds.meta.get("noise_parts")
        say(f"{rec} seed {seed} {tag}: false tracks {r['false_tracks_per_minute']:.1f}/min, compute {r['compute_per_second_of_data']:.2f}, "
            + ", ".join(f"{o} {v['pd']:.2f}" for o, v in r["objects"].items() if v["pd"]))
        return r

    for rec in a.recordings:
        for seed in a.seeds:
            for model in ("default", "harbor_v2"):
                key = f"{rec}/seed{seed}/{model}"
                if key in res.get("runs", {}):
                    continue
                res.setdefault("runs", {})[key] = run_mix(rec, seed, model, COMPONENTS, model)
                res_path.write_text(json.dumps(res, indent=2, default=str))
    if not a.no_components:
        for comp in COMPONENTS:
            key = f"simple/seed0/only_{comp}"
            if key in res.get("runs", {}):
                continue
            res.setdefault("runs", {})[key] = run_mix("simple", 0, "harbor_v2", (comp,), f"only_{comp}")
            res_path.write_text(json.dumps(res, indent=2, default=str))
    tables(res, out)
    spectra(out)


def _fmt_objects(r):
    return "; ".join(f"{o} {100 * v['pd']:.0f} %" for o, v in r["objects"].items() if v["pd"])


def tables(res: dict, out: Path) -> None:
    runs = res.get("runs", {})
    lines = ["| Recording | Seed | Noise | Pd per object | Median error [m] per object | False tracks per minute | Compute per s of data |",
             "|---|---|---|---|---|---|---|"]
    for key, r in runs.items():
        rec, seed, model = key.split("/")
        if model.startswith("only_"):
            continue
        err = "; ".join(f"{o} {v['median_error_m']:.1f}" for o, v in r["objects"].items() if v["median_error_m"] is not None)
        lines.append(f"| {rec} | {seed[4:]} | {model} | {_fmt_objects(r)} | {err} | {r['false_tracks_per_minute']:.1f} | "
                     f"{r['compute_per_second_of_data']:.2f} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    lines = ["| Components on | Pd per object | False tracks per minute | Compute per s of data |", "|---|---|---|---|"]
    order = ["simple/seed0/default"] + [f"simple/seed0/only_{c}" for c in COMPONENTS] + ["simple/seed0/harbor_v2"]
    names = {"simple/seed0/default": "none (the old model)", "simple/seed0/harbor_v2": "all"}
    for key in order:
        if key in runs:
            r = runs[key]
            lines.append(f"| {names.get(key, key.split('only_')[-1])} | {_fmt_objects(r)} | {r['false_tracks_per_minute']:.1f} | "
                         f"{r['compute_per_second_of_data']:.2f} |")
    (out / "components.md").write_text("\n".join(lines) + "\n")


def spectra(out: Path) -> None:
    """The noise spectrum of a cable channel and of a hydrophone on the simple harbor, old model beside harbor_v2 (every
    component; the machinery from its cached gather), mean over the cable's channels and over the hydrophones."""
    from scipy.signal import welch

    base = Truth.load(ROOT / "output" / "recordings" / "simple" / "dataset.truth.npz")
    cfg = HarborScenarioConfig.from_dict(copy.deepcopy(base.scenario))
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps
    old = generate_noise(layout, fs, n, cfg.noise, np.random.default_rng(1)).noise_only
    cfg.noise.model = "harbor_v2"
    cfg.noise.harbor_v2 = HarborV2Config.from_dict(v2_config(COMPONENTS, True))
    mach = machinery_gather(cfg, env, layout, cache_dir=out / "simple" / "cache")
    new = generate_noise(layout, fs, n, cfg.noise, np.random.default_rng(1), env=env, machinery=mach).noise_only
    cab = layout.slices[layout.cables[0].name]
    hyd = layout.slices["hydrophones"]
    for name, sl in (("cable", cab), ("hydrophone", hyd)):
        fig, ax = new_figure((12, 8))
        for x, label in ((old, "old model"), (new, "harbor_v2, every component")):
            f, p = welch(np.asarray(x[sl], float), fs=fs, nperseg=4096, axis=-1)
            live = np.mean(p, axis=1) > 0
            ax.semilogy(f, np.median(p[live], axis=0), label=label)
        ax.axvspan(100, 380, color="0.9", label="processing band (100 to 380 Hz)")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("noise power spectral density, median over channels [Pa^2/Hz]")
        ax.set_title(f"simple: noise spectrum of the {'cable channels (pressure mode)' if name == 'cable' else 'hydrophones'}, "
                     "old model and harbor_v2")
        ax.legend()
        save_figure(fig, out / f"spectrum_{name}.png", recording="simple")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The direct path removal within the frame budget (prompt ``run_all_3``, task 5).

The compute task measured the direct path removal at about 460 ms per 250 ms frame on the multistatic recording.  Five
changes, each measured alone against the current chain (the processing configs as they are) and then combined:

* ``passes1``: one joint pass instead of two (``echo.bistatic.joint_passes=1``; one pass is the sequential removal
  before the joint solve);
* ``previous``: the noise power of the reference estimate from the residual the chain left in the previous step
  (``echo.bistatic.ref_noise=previous``) instead of a first ECA made for it; an illuminator new in a step keeps its
  first ECA there;
* ``prefix``: the normal equations of the joint solve from prefix sums, the cross terms between illuminators included
  (``echo.bistatic.joint_gram=prefix``);
* ``hop``: the echo chain at a 0.5 s hop instead of 0.25 s (``echo.bistatic.hop_s=0.5``: half the frames, each still
  0.5 s long), with the echo tracker's confirmation count halved to 4 so that it spans the same 3 s
  (``echo.tracker.confirm=4``: at 8 fixes in 3 s a 0.5 s hop could never confirm, which would measure bookkeeping, not
  the change);
* ``band``: the removal on the traces low passed to the band where the ships' energy is and decimated
  (``echo.bistatic.removal_band=[f1, f2]``).  The band is measured, not assumed (``--measure-band``): on each family's
  tuning recording (noise seed 0, old noise model) the current chain runs and the power spectrum of every reference it
  estimates (every illuminator, every block) is summed; ``f1`` and ``f2`` are the frequencies below which 1 % and 99 %
  of that power lie, widened to the processing band ``band_hz`` (the maps read it).  Rule fixed before the spectra
  were seen.  The decimation is the largest divisor of the ECA block with the decimated Nyquist at least 1.25 ``f2``
  (``streaming.removal_decimation``).

Per option, measured on the tuning recordings (noise seed 0 of ``simple``, ``bistatic``, ``multistatic``, old noise
model; ``--options``): the median removal depth over the live cable channels of every ship's direct component on the
multistatic tuning recording (the truth components through the same projections, a diagnostic; the band energy from
``f_lo`` to ``f_hi`` before and after, as ``harbor_direct_path_depth.py``), the compute per second of data and the median
latency of each recording (a run without the components), and the Pd of every silent body by its echo.

Rule for the combination, fixed before any option was measured: an option qualifies if alone it keeps every ship's
depth on the multistatic tuning recording at least 18 dB, keeps every silent body's echo Pd on every tuning recording
within 3 points of the current chain, and lowers the multistatic compute; the combination is every qualifying option;
it is measured on the tuning recordings (``--combination``) and, while it breaks either of the first two conditions,
the qualifying option that saved least alone is dropped and it is measured again.  The test recordings are then run
once with it (``scripts/reproduce.py --update``).

Criteria, fixed in the prompt before any measurement (``--criteria BEFORE AFTER``, ``criteria.json``):

1. ``compute``: compute per second of data at most 1.5 on every 0.5 m recording (the bistatic and multistatic
   recordings and ``multistatic_b``, tuning and test, old noise model);
2. ``latency``: median latency at most 3 s on the same recordings;
3. ``depth``: the removal depth (median over the live cable channels) of every ship on the multistatic recordings
   (``multistatic`` noise seeds 0 to 3 and ``multistatic_b``, ``--depth``; a mixed seed's direct components are its
   seed 0 recording's, the same gathers) at least 18 dB;
4. ``no_fall``: no silent body's Pd (echo tracks) down by more than 3 points on either noise model on the test
   recordings.

Outputs in ``output/studies/removal_budget/``: ``band.json``, ``options.json``, ``combination.json``, ``depth.json``,
``criteria.json``, ``run.log``.  Compute is only meaningful on an idle GPU.

    python examples/harbor/05_detection/harbor_removal_budget.py --measure-band
    python examples/harbor/05_detection/harbor_removal_budget.py --options
    python examples/harbor/05_detection/harbor_removal_budget.py --combination
    python examples/harbor/05_detection/harbor_removal_budget.py --depth          # the configs as they are
    python examples/harbor/05_detection/harbor_removal_budget.py --criteria BEFORE.json AFTER.json
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
from acoustic_sim.water.scoreboard import score_picture  # noqa: E402
from acoustic_sim.water.stream import load_processing, run  # noqa: E402

OUT = ROOT / "output" / "studies" / "removal_budget"
CONFIGS = ROOT / "examples" / "configs" / "processing"
TUNING = ("simple", "bistatic", "multistatic")
FIXED_OPTIONS = {
    "current": [],
    "passes1": ["echo.bistatic.joint_passes=1"],
    "previous": ["echo.bistatic.ref_noise=previous"],
    "prefix": ["echo.bistatic.joint_gram=prefix"],
    "hop": ["echo.bistatic.hop_s=0.5", "echo.tracker.confirm=4"],
}
DEPTH_DB = 18.0
FALL = 0.03


def _say(msg: str) -> None:
    print(msg, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def _recording(name: str):
    import reproduce as R

    return next(r for r in R.recordings() if r.name == name)


def band_option(band: dict) -> dict[str, list[str]]:
    """The ``band`` option per family from ``band.json``."""
    return {fam: [f"echo.bistatic.removal_band=[{b['band_hz'][0]:g}, {b['band_hz'][1]:g}]"] for fam, b in band.items()}


def options(band: dict | None) -> dict[str, dict[str, list[str]]]:
    """Option -> family -> overrides."""
    out = {k: {fam: v for fam in TUNING} for k, v in FIXED_OPTIONS.items()}
    if band:
        out["band"] = band_option(band)
    return out


def cumulative_band(freqs: np.ndarray, power: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> tuple[float, float]:
    """The frequencies below which the fractions *lo* and *hi* of *power* (per bin at *freqs*) lie."""
    c = np.cumsum(power) / max(float(np.sum(power)), 1e-300)
    return float(freqs[np.searchsorted(c, lo)]), float(freqs[min(np.searchsorted(c, hi), len(freqs) - 1)])


def measure_band(use_cuda: bool) -> dict:
    """The ``band`` rule on each family's tuning recording (the module docstring)."""
    from acoustic_sim.water import streaming as SM

    out = {}
    for fam in TUNING:
        r = _recording(fam)
        obs = Observation.load(contract_paths(r.dataset.with_suffix(".npz"))[0])
        cfg = load_processing(CONFIGS / f"{r.processing}.json")
        acc = {}
        orig = SM.StreamingEcho._step

        def step(self, *a, **k):
            b0 = self.next_block
            res = orig(self, *a, **k)
            for b in range(b0, self.next_block):
                for name, ref in self.refd.get(b, {}).items():
                    spec = np.abs(np.fft.rfft(np.asarray(ref, float) * np.hanning(len(ref)))) ** 2
                    acc[name] = acc.get(name, 0.0) + spec
            return res

        SM.StreamingEcho._step = step
        try:
            run(obs, cfg, use_cuda=use_cuda)
        finally:
            SM.StreamingEcho._step = orig
        nb = int(round(cfg.raw["echo"]["bistatic"]["eca_block_s"] * obs.fs))
        freqs = np.fft.rfftfreq(nb, 1.0 / obs.fs)
        total = sum(acc.values())
        f1, f2 = cumulative_band(freqs, total)
        lo, hi = cfg.band
        band = [min(f1, lo), max(f2, hi)]
        out[fam] = {"fs": obs.fs, "f_1pct_hz": f1, "f_99pct_hz": f2, "processing_band_hz": [lo, hi], "band_hz": band,
                    "decimation": SM.removal_decimation(obs.fs, nb, band), "illuminators": sorted(acc)}
        _say(f"band {fam}: references 1 % to 99 % of their power {f1:.0f} to {f2:.0f} Hz; removal band {band}; "
             f"decimation {out[fam]['decimation']}")
    return out


def depth_per_ship(component_energy: dict, live: np.ndarray, cable: slice) -> dict:
    import harbor_direct_path_depth as dp

    return dp.depth_per_ship(component_energy, live, cable)


def run_one(name: str, overrides: list[str], use_cuda: bool, depth: bool = False, noise2: bool = False) -> dict:
    """The chain on recording *name* with *overrides*: its compute, latency and silent bodies' Pd (scored), and with
    *depth* a second run with the truth's direct components through the projections (the depth per ship)."""
    r = _recording(name)
    ds = r.dataset_noise2 if noise2 else r.dataset
    obs_p, truth_p = contract_paths(ds.with_suffix(".npz"))
    obs, truth = Observation.load(obs_p), Truth.load(truth_p)
    cfg = load_processing(CONFIGS / f"{r.processing}.json", overrides)
    res = run(obs, cfg, use_cuda=use_cuda, provenance=Provenance(data="forward_model", held_out=r.held_out), observation_path=obs_p)
    confirm = {"radiator": cfg.tracker().confirm, "echo": cfg.echo_tracker().confirm}
    sc = score_picture(res.records, truth, res.record_s, confirm, cfg.scoring.get("assoc_radius_m", 10.0),
                       cfg.scoring.get("hit_radius_m", 5.0))
    lat = [x["latency_s"] for x in res.records]
    out = {"overrides": overrides, "compute_per_second_of_data": res.compute_per_second["total"],
           "latency_median_s": float(np.median(lat)),
           "silent_pd": {o: v["pd"].get("echo") for o, v in sc["objects"].items() if not v["radiates"]},
           "ships_pd": {o: v["pd"].get("radiator") for o, v in sc["objects"].items() if v["radiates"]},
           "false_tracks": sc["false_tracks"]["count"]}
    if depth:
        src = truth
        if truth.components is None:                       # a mixed seed: the same gathers as the seed 0 recording
            import reproduce as R

            src = Truth.load(contract_paths(ROOT / f"{R.FAMILIES[r.family].dataset}.npz")[1])
        gain = np.asarray(src.gain, dtype=np.float64)[:, None]
        comps = {k: gain * np.asarray(v, dtype=np.float64) for k, v in src.components.items() if k.startswith("direct:")}
        rd = run(obs, cfg, use_cuda=use_cuda, provenance=Provenance(data="forward_model", truth_inputs=["components"]),
                 observation_path=obs_p, components=comps)
        live = np.ones(obs.geometry.n_receivers, bool)
        if truth.dead is not None:
            live &= ~np.asarray(truth.dead, bool)
        cable = obs.geometry.slices[obs.geometry.cables[0].name]
        out["depth_db"] = {k: v["cable_median_db"] for k, v in depth_per_ship(rd.picture.echo.component_energy, live, cable).items()}
    return out


def qualifies(opt: dict, cur: dict) -> bool:
    """The rule for the combination (the module docstring)."""
    if any(v < DEPTH_DB for v in opt["multistatic"]["depth_db"].values()):
        return False
    for fam in TUNING:
        for o, x in cur[fam]["silent_pd"].items():
            y = opt[fam]["silent_pd"].get(o)
            if x is not None and (y or 0.0) < x - FALL - 1e-12:
                return False
    return opt["multistatic"]["compute_per_second_of_data"] < cur["multistatic"]["compute_per_second_of_data"]


def measure_options(names: list[str], use_cuda: bool) -> dict:
    band = json.loads((OUT / "band.json").read_text()) if (OUT / "band.json").exists() else None
    opts = options(band)
    path = OUT / "options.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for k in names:
        res[k] = {}
        for fam in TUNING:
            res[k][fam] = run_one(fam, opts[k][fam], use_cuda, depth=(fam == "multistatic"))
            x = res[k][fam]
            _say(f"{k} on {fam}: compute {x['compute_per_second_of_data']:.2f} s/s, latency {x['latency_median_s']:.1f} s, "
                 f"silent {x['silent_pd']}" + (f", depth {x['depth_db']}" if "depth_db" in x else ""))
            path.write_text(json.dumps(res, indent=2))
    if "current" in res:
        res_q = {k: qualifies(v, res["current"]) for k, v in res.items() if k != "current" and all(f in v for f in TUNING)}
        res["_qualifies"] = res_q
        path.write_text(json.dumps(res, indent=2))
        _say(f"qualify: {res_q}")
    return res


def measure_combination(use_cuda: bool) -> dict:
    """The combination by the rule, measured on the tuning recordings, options dropped while it breaks the rule."""
    res = json.loads((OUT / "options.json").read_text())
    band = json.loads((OUT / "band.json").read_text()) if (OUT / "band.json").exists() else None
    opts = options(band)
    cur = res["current"]
    keep = [k for k, q in res["_qualifies"].items() if q]
    saving = {k: cur["multistatic"]["compute_per_second_of_data"] - res[k]["multistatic"]["compute_per_second_of_data"] for k in keep}
    tried = []
    while keep:
        ov = {fam: [o for k in keep for o in opts[k][fam]] for fam in TUNING}
        got = {fam: run_one(fam, ov[fam], use_cuda, depth=(fam == "multistatic")) for fam in TUNING}
        ok = qualifies(got, cur) or (all(v >= DEPTH_DB for v in got["multistatic"]["depth_db"].values()) and
                                     all((got[f]["silent_pd"].get(o) or 0.0) >= x - FALL - 1e-12
                                         for f in TUNING for o, x in cur[f]["silent_pd"].items() if x is not None))
        tried.append({"options": list(keep), "overrides": ov, "result": got, "passes_rule": ok})
        _say(f"combination {keep}: multistatic compute {got['multistatic']['compute_per_second_of_data']:.2f}, depth "
             f"{got['multistatic']['depth_db']}, rule {'met' if ok else 'broken'}")
        if ok:
            break
        keep.remove(min(keep, key=lambda k: saving[k]))
    out = {"qualifying": res["_qualifies"], "saving_alone": saving, "tried": tried,
           "chosen": tried[-1]["options"] if tried and tried[-1]["passes_rule"] else [],
           "overrides": tried[-1]["overrides"] if tried and tried[-1]["passes_rule"] else {}}
    (OUT / "combination.json").write_text(json.dumps(out, indent=2))
    return out


def measure_depth(names: list[str], use_cuda: bool) -> dict:
    """The depth per ship on the multistatic recordings with the configs as they are (criterion 3)."""
    path = OUT / "depth.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        res[name] = run_one(name, [], use_cuda, depth=True)
        _say(f"depth on {name}: {res[name]['depth_db']}")
        path.write_text(json.dumps(res, indent=2))
    return res


def criteria(before: dict, after: dict, depth: dict) -> list[dict]:
    rec = after["recordings"]
    fine = [k for k, v in rec.items() if v["family"] in ("bistatic", "multistatic", "multistatic_b")]
    out = []
    v = {k: rec[k]["compute_per_second_of_data"]["total"] for k in fine}
    out.append({"id": "compute", "text": "compute per second of data at most 1.5 on every 0.5 m recording", "values": v,
                "before": {k: before["recordings"][k]["compute_per_second_of_data"]["total"] for k in fine if k in before["recordings"]},
                "passed": all(x <= 1.5 for x in v.values())})
    v = {k: rec[k]["latency_s"]["median"] for k in fine}
    out.append({"id": "latency", "text": "median latency at most 3 s on every 0.5 m recording", "values": v,
                "passed": all(x <= 3.0 for x in v.values())})
    v = {k: d["depth_db"] for k, d in depth.items()}
    out.append({"id": "depth", "text": "the removal depth of every ship on the multistatic recordings at least 18 dB",
                "values": v, "passed": bool(v) and all(x >= DEPTH_DB for d in v.values() for x in d.values())})
    falls = {}
    for model in ("recordings", "recordings_noise2"):
        for name, a in after.get(model, {}).items():
            if not a["held_out"] or name not in before.get(model, {}):
                continue
            for oid, o in a["objects"].items():
                if o["radiates"]:
                    continue
                b0, a0 = before[model][name]["objects"][oid]["pd"].get("echo"), o["pd"].get("echo")
                if b0 is not None and (a0 or 0.0) < b0 - FALL - 1e-12:
                    falls[f"{model}/{name}/{oid}"] = {"before": b0, "after": a0}
    out.append({"id": "no_fall", "text": "no silent body's Pd down by more than 3 points on either noise model (test recordings)",
                "values": falls, "passed": not falls})
    return out


def options_table() -> str:
    """``options.md``: per option (and the combination) the multistatic compute and latency, the depth per ship and the
    silent bodies' Pd on the tuning recordings."""
    res = json.loads((OUT / "options.json").read_text())
    comb = json.loads((OUT / "combination.json").read_text())
    rows = {k: v for k, v in res.items() if not k.startswith("_")}
    if comb.get("tried"):
        rows[" + ".join(comb["tried"][-1]["options"])] = comb["tried"][-1]["result"]
    lines = ["| Option | Compute per s of data: simple / bistatic / multistatic | Latency, multistatic [s] | Depth on multistatic: "
             "ferry / tug / boat [dB] | Silent bodies by echo (tuning recordings) | Qualifies |", "|---|---|---|---|---|---|"]
    for k, v in rows.items():
        d = v["multistatic"]["depth_db"]
        sil = "; ".join(f"{f} {o} {100 * x:.0f} %" for f in TUNING for o, x in v[f]["silent_pd"].items() if x)
        q = res.get("_qualifies", {}).get(k)
        lines.append(f"| {k} | {' / '.join(f'{v[f]['compute_per_second_of_data']:.2f}' for f in TUNING)} | "
                     f"{v['multistatic']['latency_median_s']:.1f} | {d.get('ferry', 0):.1f} / {d.get('tug', 0):.1f} / {d.get('boat', 0):.1f} | "
                     f"{sil} | {'-' if q is None else 'yes' if q else 'no'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure-band", action="store_true")
    ap.add_argument("--options", nargs="*", default=None, help="options to measure (default: all)")
    ap.add_argument("--combination", action="store_true")
    ap.add_argument("--depth", nargs="*", default=None,
                    help="recordings (default: multistatic, multistatic_seed1 to 3, multistatic_b)")
    ap.add_argument("--criteria", nargs=2, default=None, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cuda = not a.cpu
    if a.measure_band:
        (OUT / "band.json").write_text(json.dumps(measure_band(cuda), indent=2))
    if a.options is not None:
        names = a.options or list(FIXED_OPTIONS) + ["band"]
        measure_options(names, cuda)
    if a.combination:
        measure_combination(cuda)
    if a.depth is not None:
        measure_depth(a.depth or ["multistatic", "multistatic_seed1", "multistatic_seed2", "multistatic_seed3", "multistatic_b"], cuda)
    if (OUT / "options.json").exists() and (OUT / "combination.json").exists():
        (OUT / "options.md").write_text(options_table())
    if a.criteria:
        before, after = (json.loads(Path(p).read_text()) for p in a.criteria)
        res = criteria(before, after, json.loads((OUT / "depth.json").read_text()))
        for c in res:
            _say(f"{'pass' if c['passed'] else 'FAIL'}: {c['text']}")
        (OUT / "criteria.json").write_text(json.dumps({"criteria": res}, indent=1))


if __name__ == "__main__":
    main()

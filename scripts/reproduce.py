#!/usr/bin/env python3
"""Reproduce every number of ``docs/harbor.md``: build the recordings, run the chain on each, regenerate the walkthrough,
compare against the committed numbers.

Recordings (the processing configs ``examples/configs/processing/<family>.json`` were frozen before any test recording
existed; ``held_out`` is set in every run block):

* tuning: noise seed 0 of ``simple``, ``bistatic``, ``multistatic``, ``harbor`` (the recordings every parameter was set on);
* test: noise seeds 1 to 3 of each, mixed from the cached FDTD gathers (``assemble_dataset(seed=...)``, no FDTD), and
  ``multistatic_b`` (``examples/configs/harbor_multistatic_b.scenario.json``: every track, the pilings and the hull moved,
  same sensors and levels; six FDTD runs), run with the ``multistatic`` config unchanged.

Steps (each skipped when its output exists, unless ``--rerun``): build any missing recording (dataset scripts, their FDTD
caches), measure each geometry's propagation error into its truth files (``harbor_propagation_error.py``), run
``harbor_stream.run_recording`` on every recording (``output/reproduce/runs/<name>/``), regenerate the walkthrough of the
tuning recordings, then write ``output/reproduce/latest.json`` (every number the docs quote, the criteria with pass or
fail, the tables as markdown in ``output/reproduce/tables.md``) and compare it with the committed
``output/reproduce/summary.json``: fraction of frames within 3 percentage points, median errors within 0.5 m, false tracks
exactly, compute time and latency informational.

    python scripts/reproduce.py              # run what is missing, compare, exit 1 on a failed comparison
    python scripts/reproduce.py --check      # only compare latest.json with summary.json
    python scripts/reproduce.py --update     # run, then make latest.json the committed summary.json
    python scripts/reproduce.py --fill-docs  # only fill the docs' reproduce markers from summary.json

Every number of ``docs/harbor.md``, in its tables and its prose, sits in a ``<!-- reproduce:KEY -->`` marker or a
``<!-- reproduce-table:NAME -->`` block (:func:`resolve`); every run fills them from the committed ``summary.json`` and
``--check`` fails when one differs.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "harbor" / "05_detection"))

OUT = ROOT / "output" / "reproduce"
CONFIGS = ROOT / "examples" / "configs"
TEST_SEEDS = (1, 2, 3)

# Tolerances of the comparison with the committed numbers.
TOL_FRACTION = 0.03        # Pd, fractions of frames: 3 percentage points
TOL_METRES = 0.5           # median errors

# Criteria, fixed before the test recordings were built (prompt of stage 2, sections 4 and 6); never changed after.
CRITERIA = [
    {"id": "ships_multistatic_b", "text": "every ship (ferry, tug, boat) tracked with Pd >= 0.8 on multistatic_b (radiator tracks)"},
    {"id": "false_tracks", "text": "false tracks at most 2 per minute on every test recording"},
    {"id": "wreck_echo", "text": "the wreck of the simple harbor found by its echo with Pd >= 0.9 on seeds 1 to 3 (echo tracks)"},
    {"id": "size_order", "text": "on frames where the moored hull, a UUV and a piling are all measured, the size indicator puts "
                                 "the hull below the piling in at least 70 % of them (multistatic recordings, tuning and multistatic_b)"},
]


@dataclass(frozen=True)
class Family:
    name: str
    dataset: str           # <dir>/dataset of the seed 0 recording
    cache: str             # FDTD gather cache
    combined: bool         # gathers rendered with every body at once
    build: list[str]       # command that builds the seed 0 recording


FAMILIES = {
    "simple": Family("simple", "output/recordings/simple/dataset", "output/recordings/simple/cache", True,
                     ["examples/harbor/01_datasets/harbor_multistatic_dataset.py", "--config", "examples/configs/harbor_simple.scenario.json",
                      "--out", "output/recordings/simple"]),
    "bistatic": Family("bistatic", "output/recordings/bistatic/dataset", "output/recordings/bistatic/cache", False,
                       ["examples/harbor/01_datasets/harbor_bistatic_dataset.py", "--out", "output/recordings/bistatic"]),
    "multistatic": Family("multistatic", "output/recordings/multistatic/dataset", "output/recordings/multistatic/cache", True,
                          ["examples/harbor/01_datasets/harbor_multistatic_dataset.py", "--out", "output/recordings/multistatic"]),
    "harbor": Family("harbor", "output/recordings/harbor/dataset", "output/recordings/harbor/cache", False, ["examples/harbor/01_datasets/harbor_m2_dataset.py",
                                                                                                         "--out", "output/recordings/harbor"]),
    "multistatic_b": Family("multistatic_b", "output/recordings/multistatic_b/dataset", "output/recordings/multistatic_b/cache", True,
                            ["examples/harbor/01_datasets/harbor_multistatic_dataset.py", "--config",
                             "examples/configs/harbor_multistatic_b.scenario.json", "--out", "output/recordings/multistatic_b"]),
}


@dataclass(frozen=True)
class Recording:
    name: str
    family: str
    processing: str        # processing config family
    seed: int
    held_out: bool

    @property
    def dataset(self) -> Path:
        base = ROOT / FAMILIES[self.family].dataset
        if self.seed == 0:
            return base
        return base.parent / f"seed{self.seed}" / "dataset"     # output/recordings/<x>/seed<n>/dataset

    @property
    def dataset_noise2(self) -> Path:
        """The same gathers and noise seed mixed with ``harbor_v2``: ``output/recordings/<x>/noise2/[seed<n>/]dataset``."""
        base = (ROOT / FAMILIES[self.family].dataset).parent / "noise2"
        return (base / f"seed{self.seed}" / "dataset") if self.seed else base / "dataset"


def recordings() -> list[Recording]:
    out = [Recording(f, f, f, 0, False) for f in ("simple", "bistatic", "multistatic", "harbor")]
    out += [Recording(f"{f}_seed{s}", f, f, s, True) for f in ("simple", "bistatic", "multistatic", "harbor") for s in TEST_SEEDS]
    out.append(Recording("multistatic_b", "multistatic_b", "multistatic", 0, True))
    return out


def say(msg: str) -> None:
    print(msg, flush=True)
    with open(OUT / "run.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def python(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


# ----------------------------------------------------------------------
# Recordings
# ----------------------------------------------------------------------

def build_seed0(fam: Family) -> None:
    from acoustic_sim.contract import contract_paths

    if contract_paths(ROOT / f"{fam.dataset}.npz")[0].exists():
        return
    say(f"building {fam.name} (FDTD on the GPU)")
    python(*fam.build)


def build_seed(fam: Family, seed: int, path: Path, noise2: bool = False) -> None:
    """The seed 0 recording's gathers mixed with noise seed *seed* (no FDTD; the cache must hold every gather).  With
    *noise2* the noise is the second model, ``harbor_v2`` at its documented levels (``docs/noise.md``; every component
    on, the quay machinery on the simple harbor only, from its cached gather of the noise study)."""
    from acoustic_sim.contract import Truth, contract_paths
    from acoustic_sim.water import HarborScenarioConfig, assemble_dataset, build_environment, build_layout, build_track
    from acoustic_sim.water import fdtd_runner, render_vessel_gathers

    if contract_paths(path.with_suffix(".npz"))[0].exists():
        return
    say(f"mixing {fam.name} seed {seed}{' with harbor_v2' if noise2 else ''} from the cached gathers")
    base = Truth.load(contract_paths(ROOT / f"{fam.dataset}.npz")[1])
    cfg = HarborScenarioConfig.from_dict(base.scenario)
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}

    def no_fdtd(*a, **k):
        raise RuntimeError(f"{fam.name}: an FDTD gather is missing from {fam.cache}; build the seed 0 recording first")

    saved = fdtd_runner.HarborFDTDBatch.run, fdtd_runner.HarborFDTDSolver.run
    fdtd_runner.HarborFDTDBatch.run = fdtd_runner.HarborFDTDSolver.run = no_fdtd
    mach = None
    try:
        gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=ROOT / fam.cache, combined_bodies=fam.combined)
        if noise2:
            import harbor_noise_model as hnm
            from acoustic_sim.water.dataset import machinery_gather
            from acoustic_sim.water.noise import HarborV2Config

            cfg.noise.model = "harbor_v2"
            cfg.noise.harbor_v2 = HarborV2Config.from_dict(hnm.v2_config(hnm.COMPONENTS, fam.name == "simple"))
            if fam.name == "simple":
                mach = machinery_gather(cfg, env, layout, cache_dir=ROOT / "output" / "studies" / "noise_model" / "simple" / "cache")
    finally:
        fdtd_runner.HarborFDTDBatch.run, fdtd_runner.HarborFDTDSolver.run = saved
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=seed, keep_clean=False, env=env, machinery=mach)
    truth = ds.truth
    truth.meta = dict(truth.meta, noise_seed=seed, mixed_from=fam.dataset, noise_model="harbor_v2" if noise2 else "default",
                      propagation_error_ms=base.meta.get("propagation_error_ms"))
    ds.observation.save(contract_paths(path.with_suffix(".npz"))[0])
    truth.save(contract_paths(path.with_suffix(".npz"))[1])


def recording_dir(family: str) -> Path:
    """``output/recordings/<x>`` of a family (where its propagation error figure goes)."""
    return (ROOT / FAMILIES[family].dataset).parent


def propagation_error(fam: Family, band, remeasure: bool) -> dict:
    """The forward model's propagation error of *fam*: measured (one FDTD run of a pulse) only when its
    ``propagation_error.json`` is missing or *remeasure*; otherwise its figure is redrawn from the JSON."""
    from acoustic_sim.contract import Truth, contract_paths

    ds = ROOT / fam.dataset
    out = recording_dir(fam.name)
    js = out / "propagation_error.json"
    script = "examples/harbor/01_datasets/harbor_propagation_error.py"
    args = ("--dataset", str(ds), "--out", str(out), "--band", str(band[0]), str(band[1]))
    if remeasure or not js.exists():
        say(f"propagation error of {fam.name} (FDTD)")
        python(script, *args)
    res = json.loads(js.read_text())
    truth_p = contract_paths(ds.with_suffix(".npz"))[1]
    meta = Truth.load(truth_p).meta.get("propagation_error_ms")
    if meta != res:                                                 # the truth file must carry it too
        python(script, *args)
        res = json.loads(js.read_text())
    else:
        python(script, *args, "--plot-only")
    return res


# ----------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------

def collect(rec: Recording, s: dict) -> dict:
    """The numbers of one recording that the docs quote."""
    objs = {}
    for oid, o in s["objects"].items():
        objs[oid] = {"kind": o["kind"], "radiates": o["radiates"], "pd": o["pd"], "median_error_m": o["median_error_m"],
                     "tracks": o["tracks"]}
    peaks = {chain: (None if v is None else {oid: x["hit"] for oid, x in v["objects"].items()}) for chain, v in s["peaks"].items()}
    out = {"family": rec.family, "seed": rec.seed, "held_out": rec.held_out, "record_s": s["record_s"],
           "compute_per_second_of_data": s["compute_per_second_of_data"], "latency_s": s["latency_s"],
           "false_tracks": {k: s["false_tracks"][k] for k in ("count", "per_minute", "radiator", "echo")},
           "false_track_list": [{"track": t["track"], "median_xy": t["median_xy"], "nearest_object": t["nearest_object"],
                                 "median_error_m": t["median_error_m"]} for t in s["false_tracks"]["tracks"]],
           "objects": objs, "peaks": peaks, "gpu": s["run"]["gpu"]}
    if "ellipse_coverage" in s:
        out["ellipse_coverage"] = s["ellipse_coverage"]
    if "size_order" in s:
        out["size_order"] = s["size_order"]
    return out


def characteristics_rows(name: str) -> list[dict]:
    p = OUT / "runs" / name / "characteristics.json"
    return json.loads(p.read_text())["objects"] if p.exists() else []


def criteria(numbers: dict) -> list[dict]:
    rec = numbers["recordings"]
    out = []
    for c in CRITERIA:
        r = dict(c)
        if c["id"] == "ships_multistatic_b":
            pd = {o: rec["multistatic_b"]["objects"][o]["pd"].get("radiator") for o in ("ferry", "tug", "boat")}
            r.update(values=pd, passed=all(v is not None and v >= 0.8 for v in pd.values()))
        elif c["id"] == "false_tracks":
            v = {k: x["false_tracks"]["per_minute"] for k, x in rec.items() if x["held_out"]}
            r.update(values=v, passed=all(x <= 2.0 for x in v.values()))
        elif c["id"] == "wreck_echo":
            v = {f"simple_seed{s}": rec[f"simple_seed{s}"]["objects"]["wreck"]["pd"].get("echo") for s in TEST_SEEDS}
            r.update(values=v, passed=all(x is not None and x >= 0.9 for x in v.values()))
        elif c["id"] == "size_order":
            v = {k: rec[k].get("size_order") for k in ("multistatic", "multistatic_b")}
            n = sum(x["frames_all_three"] for x in v.values() if x)
            b = sum(x["hull_below_piling"] for x in v.values() if x)
            r.update(values=v, frames=n, fraction=b / n if n else None, passed=bool(n) and b / n >= 0.7,
                     note=None if n else "no frame had the hull, a UUV and a piling all measured: not testable")
        out.append(r)
    return out


def flatten(d, prefix="") -> dict:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}/{k}" if prefix else str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def compare(new: dict, old: dict) -> list[tuple[str, object, object, str]]:
    """Every number of *old* against *new*: ``(key, old, new, "pass" | "FAIL" | "info")``."""
    a, b = flatten(old.get("recordings", {})), flatten(new.get("recordings", {}))
    a.update(flatten(old.get("recordings_noise2", {}), "noise2"))     # the second noise model (harbor_v2), the same rules
    b.update(flatten(new.get("recordings_noise2", {}), "noise2"))
    rows = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if ("compute_per_second" in k or "latency" in k or k.endswith("/gpu") or "/tracks[" in k or "false_track_list" in k
                or k.endswith("ellipse_coverage/fixes") or k.endswith("ellipse_coverage/inside")):
            rows.append((k, x, y, "info"))
            continue
        if isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)) and not isinstance(y, bool):
            if "/false_tracks/count" in k or "/false_tracks/radiator" in k or "/false_tracks/echo" in k:
                ok = x == y
            elif "median_error_m" in k:
                ok = abs(x - y) <= TOL_METRES
            elif "/pd/" in k or "/peaks/" in k or "fraction" in k:
                ok = abs(x - y) <= TOL_FRACTION + 1e-12
            else:
                ok = True if "per_minute" in k or "record_s" in k else abs(x - y) <= 1e-9
            rows.append((k, x, y, "pass" if ok else "FAIL"))
        else:
            rows.append((k, x, y, "pass" if x == y else "FAIL"))
    return rows


def committed_summary() -> dict | None:
    try:
        text = subprocess.run(["git", "show", "HEAD:output/reproduce/summary.json"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout
        return json.loads(text)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        p = OUT / "summary.json"
        return json.loads(p.read_text()) if p.exists() else None


def print_comparison(new: dict, old: dict | None) -> bool:
    if old is None:
        print("no committed output/reproduce/summary.json to compare with")
        return True
    rows = compare(new, old)
    fails = [r for r in rows if r[3] == "FAIL"]
    width = max(len(r[0]) for r in rows)
    for k, x, y, st in rows:
        if st != "pass":
            print(f"{k:{width}s}  {x!s:>12.12} {y!s:>12.12}  {st}")
    print(f"\n{len(rows)} numbers: {sum(r[3] == 'pass' for r in rows)} pass, {len(fails)} FAIL, {sum(r[3] == 'info' for r in rows)} "
          f"informational (tolerances: fractions {100 * TOL_FRACTION:.0f} percentage points, medians {TOL_METRES} m, false tracks exact)")
    return not fails


# ----------------------------------------------------------------------
# Tables for docs/harbor.md
# ----------------------------------------------------------------------

def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.0f} %"


HEADINGS = {
    "realtime": "### Real time, tuning recordings beside test recordings\n\nPd: fraction of frames, from the first frame the "
                "object could have been confirmed, with a confirmed track matched to it within 5 m (radiator tracks for what "
                "radiates, echo tracks for silent bodies); test: the three noise seeds (each value) and multistatic_b. False "
                "tracks: confirmed tracks matched to no object, per minute.",
    "criteria": "### Criteria",
    "characteristics": "### Characteristics against the truth (objects detected in at least 10 frames; the multistatic "
                       "recordings, and the tuning and first test recording of the simple and bistatic harbors, where bodies "
                       "are tracked by their echo)",
    "source_levels": "### Source level estimates against the truth (radiator tracks)",
    "propagation_error": "### Propagation error of the forward model (stationary pulse, arrival minus range over the chart speed)",
}


def tables(numbers: dict) -> str:
    """``tables.md``: every table of :func:`tables_by_name` under its heading."""
    return "\n".join(f"{HEADINGS[k]}\n\n{v}" for k, v in tables_by_name(numbers).items())


def tables_by_name(numbers: dict) -> dict[str, str]:
    """The tables of ``docs/harbor.md`` (markdown, no heading) by the name their ``reproduce-table`` block uses."""
    rec = numbers["recordings"]
    rec2 = numbers.get("recordings_noise2", {})
    out = {}
    lines = ["| Recording | Split | Compute per s of data | Latency (median) | Pd per object, old noise model | "
             "Pd per object, harbor_v2 | False tracks (per minute), old / harbor_v2 |",
             "|---|---|---|---|---|---|---|"]
    for fam in ("simple", "bistatic", "multistatic", "harbor", "multistatic_b"):
        names = [n for n, r in rec.items() if r["family"] == fam]
        for split in ("tuning", "test"):
            sel = [n for n in names if rec[n]["held_out"] == (split == "test")]
            if not sel:
                continue
            first = rec[sel[0]]
            objs, objs2 = [], []
            for oid, o in first["objects"].items():
                kind = "radiator" if o["radiates"] else "echo"
                if kind == "echo" and fam == "harbor":
                    continue
                vals = [rec[n]["objects"][oid]["pd"].get(kind) for n in sel]
                vals2 = [rec2[n]["objects"][oid]["pd"].get(kind) if n in rec2 else None for n in sel]
                if all(v in (None, 0.0) for v in vals + vals2) and not o["radiates"]:
                    continue
                objs.append(f"{oid} {' / '.join(_pct(v) for v in vals)}")
                objs2.append(f"{oid} {' / '.join(_pct(v) for v in vals2)}")
            silent0 = [oid for oid, o in first["objects"].items() if not o["radiates"]
                       and all(rec[n]["objects"][oid]["pd"].get("echo") in (None, 0.0) for n in sel)]
            if silent0 and fam != "harbor":
                objs.append("not tracked: " + ", ".join(silent0))
            comp = " / ".join(f"{rec[n]['compute_per_second_of_data']['total']:.2f}" for n in sel)
            lat = " / ".join(f"{rec[n]['latency_s']['median']:.1f} s" for n in sel)
            ft = " / ".join(f"{rec[n]['false_tracks']['count']} ({rec[n]['false_tracks']['per_minute']:.1f})" for n in sel)
            ft2 = " / ".join(f"{rec2[n]['false_tracks']['count']} ({rec2[n]['false_tracks']['per_minute']:.1f})" if n in rec2 else "-"
                             for n in sel)
            lines.append(f"| {fam} | {split} | {comp} | {lat} | {'; '.join(objs)} | {'; '.join(objs2)} | {ft}; harbor_v2 {ft2} |")
    out["realtime"] = "\n".join(lines) + "\n"
    # The silent bodies under the two noise models: echo Pd, old model beside harbor_v2, every recording.
    lines = ["| Recording | Silent body | Pd, old noise model | Pd, harbor_v2 |", "|---|---|---|---|"]
    for name, r in rec.items():
        if r["family"] == "harbor" or name not in rec2:
            continue
        for oid, o in r["objects"].items():
            if o["radiates"]:
                continue
            x, y = o["pd"].get("echo"), rec2[name]["objects"][oid]["pd"].get("echo")
            if (x or 0.0) > 0 or (y or 0.0) > 0:
                lines.append(f"| {name} | {oid} | {_pct(x)} | {_pct(y)} |")
    out["silent_noise"] = "\n".join(lines) + "\n"
    lines = ["| Criterion | Values | Result |", "|---|---|---|"]
    for c in numbers["criteria"]:
        vals = c.get("values")
        if c["id"] == "false_tracks":
            vals = {k: round(v, 1) for k, v in vals.items()}
        elif c["id"] == "size_order":
            vals = (f"{round(c['fraction'], 2)} of {c['frames']} frames" if c.get("frames") else c["note"])
        else:
            vals = {k: None if v is None else round(v, 2) for k, v in vals.items()}
        result = "pass" if c["passed"] else ("**not met: not testable**" if c.get("note") else "**fail**")
        lines.append(f"| {c['text']} | {vals} | {result} |")
    out["criteria"] = "\n".join(lines) + "\n"
    lines = ["| Recording | Object | True type, size | Seen | Detected frames | Size ratio [dB] | Size indicator | Echo extent along / across [m] | "
              "Source level [Pa] (true) | Half max width [m] | Speed [m/s] (true) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    thr = numbers.get("size_threshold_db", 0.0)
    for name in ("multistatic", "multistatic_b", "simple", "simple_seed1", "bistatic", "bistatic_seed1"):
        for r in numbers["characteristics"].get(name, []):
            m = r["median"]
            size = f"{r['true_size_m'][0]:g} x {r['true_size_m'][1]:g} m" if r["true_size_m"] else "-"
            ratio = m["size_ratio_db"]
            ind = "-" if ratio is None else ("smaller than about a wavelength" if ratio > thr else "about a wavelength or larger")
            ext = "-" if m["along"] is None else f"{m['along']:.1f} / {m['across']:.1f}"
            sl = "-" if m["source_level"] is None else f"{m['source_level']:.2f} ({r['true_level_pa']:g})"
            w = "-" if m["width_major"] is None else f"{m['width_major']:.1f} x {m['width_minor']:.1f}"
            sp = "-" if m["speed"] is None else f"{m['speed']:.1f} ({r['true_speed']:.1f})"
            lines.append(f"| {name} | {r['object']} | {r['type']}, {size} | {', '.join(f'{k} {v}' for k, v in r['seen'].items())} | "
                         f"{r['detected_frames']} | {'-' if ratio is None else f'{ratio:+.1f}'} | {ind} | {ext} | {sl} | {w} | {sp} |")
    out["characteristics"] = "\n".join(lines) + "\n"
    lines = ["| Recording | Object | True [Pa] | Estimate [Pa] | Error [dB] |", "|---|---|---|---|---|"]
    for row in numbers["source_levels"]:
        lines.append(f"| {row['recording']} | {row['object']} | {row['true_pa']:g} | {row['estimate_pa']:.3f} | {row['error_db']:+.1f} |")
    out["source_levels"] = "\n".join(lines) + "\n"
    lines = ["| Recording | Band [Hz] | Chart speed [m/s] | Per hydrophone: error [ms] (direction, range) |", "|---|---|---|---|"]
    for fam, p in numbers["propagation_error_ms"].items():
        per = "; ".join(f"{o['error_ms']:+.2f} ({o['direction_deg']:.0f} deg, {o['range_m']:.0f} m)" for o in p["hydrophones"])
        lines.append(f"| {fam} | {p['band_hz'][0]:.0f}-{p['band_hz'][1]:.0f} | {p['c_chart']:.1f} | {per} |")
    out["propagation_error"] = "\n".join(lines) + "\n"
    return out


def source_levels(numbers: dict) -> list[dict]:
    rows = []
    for name in numbers["recordings"]:
        for r in characteristics_rows(name):
            sl = r["median"]["source_level"]
            if r["radiates"] and sl is not None:
                rows.append({"recording": name, "object": r["object"], "true_pa": r["true_level_pa"], "estimate_pa": sl,
                             "error_db": 20 * np.log10(sl / r["true_level_pa"])})
    return rows


# ----------------------------------------------------------------------
# Numbers in the prose of the docs: <!-- reproduce:KEY -->text<!-- /reproduce -->
# ----------------------------------------------------------------------
#
# KEY is ``[aggregate:][@file:]path[|format]``:
#
# * path: a slash separated path into ``output/reproduce/summary.json`` (``recordings/simple/objects/wreck/pd/echo``);
#   ``*`` and ``?`` match within a key (``recordings/*_seed*/false_tracks/per_minute``; ``?`` also matches a slash
#   inside a key), ``[i]`` indexes a list, ``[*]`` takes every element and ``[field=pattern]`` the elements whose
#   field matches one of the comma separated patterns (``source_levels[recording=simple*][object=ferry,boat]/error_db``);
# * ``@file:``: the path is into another regenerated JSON under ``output/`` instead
#   (``@recordings/simple/build.json:fdtd_wall_s``);
# * aggregate (over every value the path matches, None dropped): ``min``, ``max``, ``absmax`` (the value of largest
#   magnitude), ``range`` ("a to b", one value when equal), ``count``, ``ntrue`` (how many are true), ``sum``;
# * format: ``pct`` (percent, no decimals: "86 %"), ``frac2`` (0.86), ``f0`` ``f1`` ``f2`` ``f3`` (decimals), ``+f1``
#   ``+f2`` (signed), ``absf2`` (magnitude), ``int``, ``db`` (20 log10, one decimal, signed: "+2.4"), ``str`` (default).
#
# Tables: <!-- reproduce-table:NAME --> ... <!-- /reproduce-table --> holds the table NAME of :func:`tables` (or, for
# ``file:<path under output/>``, that generated markdown file) and is rewritten on every run.

DOCS = [ROOT / "docs" / "harbor.md", ROOT / "docs" / "noise.md"]
MARK = re.compile(r"<!-- reproduce:(?P<key>[^>]*?) -->(?P<text>.*?)<!-- /reproduce -->", re.S)
TABLE = re.compile(r"(<!-- reproduce-table:(?P<name>[^>]*?) -->\n)(?P<text>.*?)(<!-- /reproduce-table -->)", re.S)


def _match(node, parts: list[str]) -> list:
    if not parts:
        return [node]
    head, rest = parts[0], parts[1:]
    m = re.fullmatch(r"(.*?)((?:\[[^\]]+\])*)", head)
    name, idx = m.group(1), re.findall(r"\[([^\]]+)\]", m.group(2))
    if not isinstance(node, dict):
        return []
    keys = [k for k in node if fnmatch.fnmatchcase(k, name)] if any(c in name for c in "*?[") else ([name] if name in node else [])
    vals = [node[k] for k in keys]
    for i in idx:
        nxt = []
        for v in vals:
            if "=" in i:                                       # [field=pattern]: the list elements whose field matches
                f, pat = i.split("=", 1)
                nxt += [x for x in (v if isinstance(v, list) else [v])
                        if isinstance(x, dict) and any(fnmatch.fnmatchcase(str(x.get(f)), p) for p in pat.split(","))]
            elif isinstance(v, list):
                nxt += v if i == "*" else ([v[int(i)]] if int(i) < len(v) else [])
        vals = nxt
    out = []
    for v in vals:
        out += _match(v, rest)
    return out


def _fmt(v, fmt: str) -> str:
    if v is None:
        return "-"
    if fmt == "pct":
        return f"{100 * v:.0f} %"
    if fmt == "frac2":
        return f"{v:.2f}"
    if fmt == "int":
        return f"{int(round(v))}"
    if fmt.startswith("absf"):
        return f"{abs(v):.{int(fmt[4:])}f}"
    if fmt == "db":
        return f"{20 * np.log10(v):+.1f}"
    if fmt.startswith("+f"):
        return f"{v:+.{int(fmt[2:])}f}"
    if fmt.startswith("f"):
        return f"{v:.{int(fmt[1:])}f}"
    return str(v)


def resolve(key: str, summary: dict) -> str:
    """The text of the marker *key* (see above) from *summary* or the file the key names."""
    key, _, fmt = key.partition("|")
    agg = None
    m = re.match(r"(min|max|absmax|range|count|ntrue|sum):(.*)", key)
    if m:
        agg, key = m.group(1), m.group(2)
    src = summary
    if key.startswith("@"):
        f, _, key = key[1:].partition(":")
        src = json.loads((ROOT / "output" / f).read_text())
    vals = _match(src, key.split("/"))
    if agg is None:
        if len(vals) != 1:
            raise KeyError(f"reproduce marker {key!r}: {len(vals)} values")
        return _fmt(vals[0], fmt)
    vals = [v for v in vals if v is not None]
    if agg == "count":
        return str(len(vals))
    if agg == "ntrue":
        return str(sum(bool(v) for v in vals))
    if agg == "sum":
        return _fmt(sum(vals), fmt or "int")
    if not vals:
        raise KeyError(f"reproduce marker {key!r}: no values")
    if agg == "absmax":
        return _fmt(max(vals, key=abs), fmt)
    lo, hi = _fmt(min(vals), fmt), _fmt(max(vals), fmt)
    if agg == "min":
        return lo
    if agg == "max":
        return hi
    if lo == hi:
        return lo
    if fmt == "pct":
        return f"{lo[:-2]} to {hi}"
    return f"{lo} to {hi}"


def fill_docs(summary: dict, write: bool) -> list[str]:
    """Every marker and table block of :data:`DOCS` filled from *summary*; written when *write*.  Returns the markers
    whose text in the docs differs from what *summary* gives (empty: the docs are current)."""
    named = tables_by_name(summary)
    stale = []
    for doc in DOCS:
        text = doc.read_text()

        def mark(m):
            new = resolve(m.group("key"), summary)
            if new != m.group("text"):
                stale.append(f"{doc.name}: {m.group('key')}: {m.group('text')!r} -> {new!r}")
            return f"<!-- reproduce:{m.group('key')} -->{new}<!-- /reproduce -->"

        def table(m):
            name = m.group("name")
            if name.startswith("file:"):
                new = (ROOT / "output" / name[5:]).read_text().strip() + "\n"
            else:
                new = named[name].strip() + "\n"
            if new != m.group("text"):
                stale.append(f"{doc.name}: table {name}")
            return m.group(1) + new + m.group(4)

        out = TABLE.sub(table, MARK.sub(mark, text))
        if write and out != text:
            doc.write_text(out)
    return stale


# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only compare output/reproduce/latest.json with the committed summary.json")
    ap.add_argument("--update", action="store_true", help="after the run, copy latest.json to summary.json (commit it)")
    ap.add_argument("--rerun", action="store_true", help="run the chain again on recordings that already have results")
    ap.add_argument("--fill-docs", action="store_true", help="only fill the reproduce markers of the docs from summary.json")
    ap.add_argument("--remeasure", action="store_true", help="measure the propagation error again (one FDTD run per geometry)")
    ap.add_argument("--only", nargs="*", default=None, help="recordings to run (default: all)")
    ap.add_argument("--no-walkthrough", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-noise2", action="store_true", help="skip the run of every recording with the second noise model")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    latest = OUT / "latest.json"
    if a.check:
        ok = print_comparison(json.loads(latest.read_text()), committed_summary())
        stale = fill_docs(json.loads((OUT / "summary.json").read_text()), write=False)
        for line in stale:
            print(f"docs not filled from summary.json: {line}")
        print(f"docs: {len(stale)} markers or tables differ from output/reproduce/summary.json"
              + (" (run scripts/reproduce.py --fill-docs)" if stale else ""))
        return 0 if ok and not stale else 1
    if a.fill_docs:
        stale = fill_docs(json.loads((OUT / "summary.json").read_text()), write=True)
        print(f"filled {len(stale)} markers or tables of {', '.join(d.name for d in DOCS)} from output/reproduce/summary.json")
        return 0

    import harbor_stream as hs

    from acoustic_sim.contract import contract_paths
    from acoustic_sim.water.stream import load_processing

    t_start = time.time()
    recs = [r for r in recordings() if a.only is None or r.name in a.only]
    for fam in {r.family for r in recs}:
        build_seed0(FAMILIES[fam])
    for r in recs:
        if r.seed:
            build_seed(FAMILIES[r.family], r.seed, r.dataset)
    for fam in sorted({r.family for r in recs}):
        band = load_processing(CONFIGS / "processing" / f"{'multistatic' if fam == 'multistatic_b' else fam}.json").band
        propagation_error(FAMILIES[fam], band, a.remeasure)
    for r in recs:
        out = OUT / "runs" / r.name
        if (out / "summary.json").exists() and not a.rerun:
            continue
        say(f"running the chain on {r.name} ({'test' if r.held_out else 'tuning'} recording)")
        obs = contract_paths(r.dataset.with_suffix(".npz"))[0]
        s = hs.run_recording(r.name, CONFIGS / "processing" / f"{r.processing}.json", obs, contract_paths(r.dataset.with_suffix(".npz"))[1],
                             out, r.held_out, use_cuda=not a.cpu, seed=r.seed, video=not a.no_video)
        hs.report(s)
    if not a.no_noise2:                                     # every recording again with the second noise model, harbor_v2
        for r in recs:
            build_seed(FAMILIES[r.family], r.seed, r.dataset_noise2, noise2=True)
            out = OUT / "runs" / f"{r.name}_noise2"
            if (out / "summary.json").exists() and not a.rerun:
                continue
            say(f"running the chain on {r.name} mixed with harbor_v2")
            obs = contract_paths(r.dataset_noise2.with_suffix(".npz"))[0]
            s = hs.run_recording(f"{r.name}_noise2", CONFIGS / "processing" / f"{r.processing}.json", obs,
                                 contract_paths(r.dataset_noise2.with_suffix(".npz"))[1], out, r.held_out, use_cuda=not a.cpu,
                                 seed=r.seed, video=False, figures=False)
            hs.report(s)
    if not a.no_walkthrough:
        for fam in ("simple", "bistatic", "multistatic", "harbor"):
            if a.only is not None and fam not in a.only:
                continue
            cache = ROOT / "output" / "walkthrough" / fam / "cache" / "run.pkl"
            if cache.exists():                                      # the cache ignores code changes: always run again
                cache.unlink()
            say(f"walkthrough figures of {fam}")
            subprocess.run(["bash", "run_all.sh", fam] + (["--cpu"] if a.cpu else []), cwd=ROOT / "examples" / "harbor" / "06_walkthrough",
                           check=True)

    numbers = {"recordings": {}, "characteristics": {}}
    for r in recordings():
        p = OUT / "runs" / r.name / "summary.json"
        if p.exists():
            numbers["recordings"][r.name] = collect(r, json.loads(p.read_text()))
        p = OUT / "runs" / f"{r.name}_noise2" / "summary.json"
        if p.exists():
            numbers.setdefault("recordings_noise2", {})[r.name] = collect(r, json.loads(p.read_text()))
    for name in ("multistatic", "multistatic_b", "simple", "simple_seed1", "bistatic", "bistatic_seed1"):
        numbers["characteristics"][name] = characteristics_rows(name)
    cal = json.loads((ROOT / "output" / "studies" / "level_calibration" / "calibration.json").read_text())
    numbers["level_calibration"] = cal["level_calibration"]
    numbers["size_threshold_db"] = load_processing(CONFIGS / "processing" / "multistatic.json").characteristics().size_threshold_db
    numbers["source_levels"] = source_levels(numbers)
    numbers["propagation_error_ms"] = {fam: json.loads((recording_dir(fam) / "propagation_error.json").read_text())
                                       for fam in FAMILIES if (recording_dir(fam) / "propagation_error.json").exists()}
    if all(k in numbers["recordings"] for k in ("multistatic_b", "multistatic") + tuple(f"simple_seed{s}" for s in TEST_SEEDS)):
        numbers["criteria"] = criteria(numbers)
    else:
        numbers["criteria"] = []
    rec2 = numbers.get("recordings_noise2", {})              # the criteria are evaluated on the old model, reported on both
    if all(k in rec2 for k in ("multistatic_b", "multistatic") + tuple(f"simple_seed{s}" for s in TEST_SEEDS)):
        numbers["criteria_noise2"] = criteria({"recordings": rec2})
    from acoustic_sim.contract import run_info

    numbers["run"] = run_info(wall_s=time.time() - t_start, use_cuda=not a.cpu)
    latest.write_text(json.dumps(numbers, indent=2))
    (OUT / "tables.md").write_text(tables(numbers))
    ok = print_comparison(numbers, committed_summary())
    for c in numbers["criteria"]:
        print(f"criterion {c['id']}: {'pass' if c['passed'] else 'FAIL'} ({c['text']})")
    if a.update:
        shutil.copy(latest, OUT / "summary.json")
        print("updated output/reproduce/summary.json")
    stale = fill_docs(json.loads((OUT / "summary.json").read_text()), write=True)
    print(f"filled {len(stale)} markers or tables of {', '.join(d.name for d in DOCS)} from output/reproduce/summary.json")
    return 0 if ok or a.update else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The sensor layout study: what the simple harbor's system sees with nine layouts of the same scene.

The simple harbor (``examples/configs/harbor_simple.scenario.json``: a 500 x 400 m basin, 1 m grid, 100 to 380 Hz; a
ferry, a boat, a silent UUV and a wreck) with only the sensors changed (``examples/configs/layout/<variant>.scenario.json``,
built with ``harbor_multistatic_dataset.py --out output/studies/layout/<variant>``, the scheme and precision of the
scenario file at the time: the k space scheme, float64).  Noise seeds 1 and 2 are mixed from the cached gathers.  The
system runs on every variant and seed with ``examples/configs/processing/simple.json`` unchanged and is scored as on
``main`` (``harbor_stream.run_recording``: Pd per object from confirmed tracks within 5 m, median error, false tracks per
minute).  Per object the coverage quantities of ``water/coverage.py`` at its median position are recorded, the maps of
every variant and of every scenario on ``main`` are drawn (one figure per quantity per layout, the true tracks drawn:
this is a design tool), and the rank correlation of each quantity with Pd over every variant and object is computed.

No pass or fail criterion: the study measures (prompt ``sensor_layout_study``).  A variant that fails because a
setting assumes the loop is reported, not tuned.

    python examples/harbor/07_layout_study/run_layout_study.py --cuda [--variants A B ...] [--seeds 0 1 2] [--maps-only]
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
sys.path.insert(0, str(ROOT / "examples" / "harbor" / "05_detection"))

import harbor_stream as hs  # noqa: E402
from acoustic_sim.contract import Truth, contract_paths  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water import HarborScenarioConfig, assemble_dataset, build_environment, build_layout, build_track  # noqa: E402
from acoustic_sim.water import fdtd_runner, render_vessel_gathers  # noqa: E402
from acoustic_sim.water.coverage import aperture, predicted_resolution, shadowing, strain_coverage  # noqa: E402
from acoustic_sim.water.plotting import grid_image  # noqa: E402

VARIANTS = {
    "A": "one straight leg along the north quay",
    "B": "the same leg, 4 hydrophones at the basin corners",
    "C": "an L: north quay and east side",
    "D": "a U: north, east, south",
    "E": "the current loop",
    "F": "the current loop, 4 hydrophones at the corners (the main layout in strain mode)",
    "G": "the current loop, 8 hydrophones: corners and mid sides",
    "H": "a zigzag of 45 degree segments across the north half",
    "P": "the current loop in pressure mode, 4 hydrophones (the main layout as it is)",
}
MAIN = {"simple": "harbor_simple", "bistatic": "harbor_bistatic", "multistatic": "harbor_multistatic",
        "multistatic_b": "harbor_multistatic_b", "harbor": "harbor"}
QUANTITIES = {
    "strain_fraction": ("cable channels that see the cell", "fraction"),
    "strain_weighted": ("cable channels that see the cell, spreading weighted", "fraction"),
    "aperture_deg": ("aperture of the sensors that see the cell", "degrees"),
    "shadowing": ("sensors with a straight water path", "fraction"),
    "resolution_major_m": ("predicted resolution, major axis", "m"),
}
SHORT = {"A": "straight leg", "B": "straight leg, 4 hydrophones", "C": "L", "D": "U", "E": "loop", "F": "loop, 4 hydrophones",
         "G": "loop, 8 hydrophones", "H": "zigzag", "P": "pressure loop, 4 hydrophones"}
OUT = ROOT / "output" / "studies" / "layout"
PROC = ROOT / "examples" / "configs" / "processing" / "simple.json"


def mix_seed(v: str, seed: int) -> tuple[Path, Path]:
    """The observation and truth of variant *v*, noise seed *seed* (seed 0: the build; others mixed from the cache)."""
    d = OUT / v
    if seed == 0:
        return contract_paths(d / "dataset.npz")
    obs_p, tru_p = contract_paths(d / f"seed{seed}" / "dataset.npz")
    if obs_p.exists():
        return obs_p, tru_p
    base = Truth.load(contract_paths(d / "dataset.npz")[1])
    cfg = HarborScenarioConfig.from_dict(base.scenario)
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    tracks = {x.id: build_track(x, cfg.fdtd.total_time, env) for x in cfg.vessels}

    def no_fdtd(*a, **k):
        raise RuntimeError(f"variant {v}: a gather is missing from its cache")

    saved = fdtd_runner.HarborFDTDBatch.run, fdtd_runner.HarborFDTDSolver.run
    fdtd_runner.HarborFDTDBatch.run = fdtd_runner.HarborFDTDSolver.run = no_fdtd
    try:
        gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=d / "cache", combined_bodies=True)
    finally:
        fdtd_runner.HarborFDTDBatch.run, fdtd_runner.HarborFDTDSolver.run = saved
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=seed, keep_clean=False, env=env)
    obs_p.parent.mkdir(parents=True, exist_ok=True)
    ds.truth.meta = dict(ds.truth.meta, noise_seed=seed, mixed_from=f"output/studies/layout/{v}")
    ds.observation.save(obs_p)
    ds.truth.save(tru_p)
    return obs_p, tru_p


def coverage_at(layout, env, pts: np.ndarray, freq: float, c: float, half: float, dx: float) -> dict:
    frac, weighted = strain_coverage(layout, pts)
    res = predicted_resolution(layout, pts, freq, c, half=half, dx=dx)
    return {"strain_fraction": frac, "strain_weighted": weighted, "aperture_deg": aperture(layout, pts),
            "shadowing": shadowing(layout, pts, env), "resolution_major_m": res[:, 0], "resolution_minor_m": res[:, 1]}


def maps(name: str, title: str, cfg_path: Path, out: Path, band, cell: float, res_cell: float, half: float, dx: float) -> None:
    """The coverage maps of one layout: one figure per quantity, the true tracks drawn."""
    cfg = HarborScenarioConfig.load(cfg_path)
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    chart = env.public()
    x0, x1, y0, y1 = env.extent
    margin = cfg.fdtd.damping_width * env.dx
    xs = np.arange(x0 + margin, x1 - margin + 1e-6, cell)
    ys = np.arange(y0 + margin, y1 - margin + 1e-6, cell)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts[env.is_water(pts, margin=1.0)]
    freq, c = 0.5 * (band[0] + band[1]), chart.c_nominal
    frac, weighted = strain_coverage(layout, pts)
    vals = {"strain_fraction": frac, "strain_weighted": weighted, "aperture_deg": aperture(layout, pts),
            "shadowing": shadowing(layout, pts, env)}
    rx = np.arange(x0 + margin, x1 - margin + 1e-6, res_cell)
    ry = np.arange(y0 + margin, y1 - margin + 1e-6, res_cell)
    rgx, rgy = np.meshgrid(rx, ry)
    rpts = np.column_stack([rgx.ravel(), rgy.ravel()])
    rpts = rpts[env.is_water(rpts, margin=1.0)]
    res = predicted_resolution(layout, rpts, freq, c, half=half, dx=dx)
    tracks = {x.id: build_track(x, cfg.fdtd.total_time, env) for x in cfg.vessels}
    out.mkdir(parents=True, exist_ok=True)
    for q, (label, unit) in QUANTITIES.items():
        fig, ax = new_figure((12, 9))
        ax.imshow(np.where(env.water, np.nan, 1.0), origin="lower", extent=env.extent, cmap="Greys", vmin=0, vmax=1.5)
        if q == "resolution_major_m":
            img, ext = grid_image(rpts, np.minimum(res[:, 0], 2 * half))
            vmax = 2 * half
        else:
            img, ext = grid_image(pts, vals[q])
            vmax = 360.0 if q == "aperture_deg" else 1.0
        im = ax.imshow(img, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(f"{label} [{unit}]" + (f" (at most the {2 * half:g} m window)" if q == "resolution_major_m" else ""))
        for cable in layout.cables:
            ax.plot(*cable.positions.T, "-", color="w", lw=2.5, label=f"DAS cable ({cable.das_mode})")
        if layout.n_hydrophones:
            ax.plot(*layout.hydrophones.T, "^", color="w", mec="k", ms=10, label="hydrophones")
        for i, (oid, tr) in enumerate(tracks.items()):
            moving = np.ptp(tr.xy, axis=0).max() > 0
            ax.plot(*tr.xy.T, "--" if moving else "s", color="tab:red" if tr.level_pa > 0 else "tab:orange", lw=1.8, ms=7,
                    label=("true track, radiating" if tr.level_pa > 0 else "true track or position, silent") if i in (0, 2) else None)
        ax.set_xlim(x0 + margin, x1 - margin)
        ax.set_ylim(y0 + margin, y1 - margin)
        ax.set_aspect("equal")
        ax.set_xlabel("x (east) [m]")
        ax.set_ylabel("y (north) [m]")
        ax.set_title(f"{name} ({title}): {label}")
        ax.legend(loc="lower left")
        save_figure(fig, out / f"coverage_{q}.png", recording=name)


def spearman(x, y) -> float | None:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return None
    return float(spearmanr(x[ok], y[ok]).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--cuda", action="store_true", help="kept for the prompt's command line: the GPU is the default")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--maps-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "run.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(time.strftime("%H:%M:%S ") + msg + "\n")
        log.flush()

    # Coverage maps: every variant, and every scenario on main.
    for v in a.variants:
        maps(v, SHORT[v], ROOT / "examples" / "configs" / "layout" / f"{v}.scenario.json", OUT / v / "maps", (100.0, 380.0),
             5.0, 20.0, 40.0, 1.0)
        say(f"maps of {v}")
    for name, stem in MAIN.items():
        band = (12.0, 78.0) if name == "harbor" else ((100.0, 380.0) if name == "simple" else (100.0, 760.0))
        big = name == "harbor"
        maps(name, "layout on main", ROOT / "examples" / "configs" / f"{stem}.scenario.json", OUT / "main" / name / "maps", band,
             50.0 if big else 5.0, 100.0 if big else 20.0, 200.0 if big else 40.0, 5.0 if big else 1.0)
        say(f"maps of the {name} layout on main")
    if a.maps_only:
        return

    path = OUT / "summary.json"
    summ = json.loads(path.read_text()) if path.exists() else {}
    summ["criterion"] = "none: a measurement (prompt sensor_layout_study)"
    summ["build"] = ("harbor_multistatic_dataset.py with each variant's scenario file: the k space scheme and float64, as the "
                     "scenario files had them when the study was run")
    runs = summ.setdefault("runs", {})
    for v in a.variants:
        cfg = HarborScenarioConfig.load(ROOT / "examples" / "configs" / "layout" / f"{v}.scenario.json")
        env = build_environment(cfg)
        layout = build_layout(cfg, env)
        cable_len = sum(c.length for c in layout.cables)
        for seed in a.seeds:
            key = f"{v}/seed{seed}"
            if key in runs:
                continue
            obs_p, tru_p = mix_seed(v, seed)
            try:
                s = hs.run_recording(f"layout_{v}_seed{seed}", PROC, obs_p, tru_p, OUT / v / "runs" / f"seed{seed}", True,
                                     use_cuda=not a.cpu, seed=seed, video=False, figures=seed == 0)
            except Exception as ex:                                           # a variant the processing cannot run: a finding
                runs[key] = {"failed": f"{type(ex).__name__}: {ex}"}
                say(f"{key}: FAILED {ex}")
                path.write_text(json.dumps(summ, indent=2))
                continue
            truth = Truth.load(tru_p)
            objs = {}
            for oid, o in s["objects"].items():
                kind = "radiator" if o["radiates"] else "echo"
                tr = truth.tracks[oid]
                med = np.median(tr.xy, axis=0)
                cov = coverage_at(layout, env, med[None, :], 240.0, env.public().c_nominal, 40.0, 1.0)
                objs[oid] = {"radiates": o["radiates"], "pd": o["pd"].get(kind), "median_error_m": o["median_error_m"].get(kind),
                             "median_xy": med.tolist(), "coverage": {k: float(val[0]) for k, val in cov.items()}}
            runs[key] = {"variant": v, "seed": seed, "cable_m": cable_len, "channels": layout.n_das,
                         "hydrophones": layout.n_hydrophones, "objects": objs,
                         "false_tracks_per_minute": s["false_tracks"]["per_minute"],
                         "compute_per_second_of_data": s["compute_per_second_of_data"]["total"]}
            say(f"{key}: false tracks {s['false_tracks']['per_minute']:.1f}/min, "
                + ", ".join(f"{k} {x['pd']:.2f}" for k, x in objs.items() if x["pd"] is not None))
            path.write_text(json.dumps(summ, indent=2))
    report(summ)
    path.write_text(json.dumps(summ, indent=2))


def report(summ: dict) -> None:
    """The table (mean and spread over seeds) and the rank correlations, into ``summary.json`` and ``table.md``."""
    runs = {k: r for k, r in summ["runs"].items() if "failed" not in r}
    by_v: dict[str, list] = {}
    for r in runs.values():
        by_v.setdefault(r["variant"], []).append(r)
    objects = sorted({o for r in runs.values() for o in r["objects"]})
    lines = ["| Variant | Cable [m] | Channels | Hydrophones | " + " | ".join(f"{o}: Pd (min to max) / median error [m]" for o in objects)
             + " | False tracks per minute | Compute per s of data |", "|---|---|---|---|" + "---|" * len(objects) + "---|---|"]
    table = {}
    for v in VARIANTS:
        rs = by_v.get(v)
        if not rs:
            failed = [k for k, r in summ["runs"].items() if k.startswith(v + "/") and "failed" in r]
            if failed:
                lines.append(f"| {v} | failed: {summ['runs'][failed[0]]['failed']} |")
            continue
        row = {"cable_m": rs[0]["cable_m"], "channels": rs[0]["channels"], "hydrophones": rs[0]["hydrophones"], "objects": {}}
        cells = []
        for o in objects:
            pd = [r["objects"][o]["pd"] or 0.0 for r in rs]
            er = [r["objects"][o]["median_error_m"] for r in rs if r["objects"][o]["median_error_m"] is not None]
            row["objects"][o] = {"pd_mean": float(np.mean(pd)), "pd_min": float(np.min(pd)), "pd_max": float(np.max(pd)),
                                 "median_error_m": float(np.median(er)) if er else None}
            cells.append(f"{100 * np.mean(pd):.0f} % ({100 * np.min(pd):.0f} to {100 * np.max(pd):.0f}) / "
                         + (f"{np.median(er):.1f}" if er else "-"))
        ft = [r["false_tracks_per_minute"] for r in rs]
        cp = [r["compute_per_second_of_data"] for r in rs]
        row.update(false_tracks_per_minute=[float(np.min(ft)), float(np.max(ft))], compute=float(np.median(cp)))
        table[v] = row
        lines.append(f"| {v} | {row['cable_m']:.0f} | {row['channels']} | {row['hydrophones']} | " + " | ".join(cells)
                     + f" | {np.min(ft):.1f} to {np.max(ft):.1f} | {np.median(cp):.2f} |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    # Rank correlation of each quantity at the objects with Pd: over every variant, seed and object, and per kind.
    corr = {}
    for kind, sel in (("all", lambda o: True), ("radiating", lambda o: o["radiates"]), ("silent", lambda o: not o["radiates"])):
        corr[kind] = {}
        for q in list(QUANTITIES) + ["resolution_minor_m"]:
            x, y = [], []
            for r in runs.values():
                for o in r["objects"].values():
                    if sel(o):
                        x.append(o["coverage"][q])
                        y.append(o["pd"] or 0.0)
            corr[kind][q] = {"spearman": spearman(x, y), "n": len(x)}
    summ["table"] = table
    summ["rank_correlation_with_pd"] = corr
    lines = ["| Quantity at the object | Spearman with Pd, every object | radiating | silent |", "|---|---|---|---|"]
    for q in list(QUANTITIES) + ["resolution_minor_m"]:
        f = lambda k: "-" if corr[k][q]["spearman"] is None else f"{corr[k][q]['spearman']:+.2f} (n {corr[k][q]['n']})"
        lines.append(f"| {q} | {f('all')} | {f('radiating')} | {f('silent')} |")
    (OUT / "correlation.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

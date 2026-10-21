#!/usr/bin/env python3
"""Harbor picture, stage 1: the scoreboard, and the chains of ``main`` measured on it.

Goal of the program: from the recorded traces, the sensor positions, a chart with a constant sound
speed and the AIS reports, detect and locate every object in the harbor.  This stage measures the
chains that exist today on the scoreboard of ``water/scoreboard.py`` (definitions in
``docs/architecture.md``, Scoreboard); it changes no processing algorithm.

Chains and configurations
-------------------------
``passive`` (the radiating objects; the AIS ships are each scored on a run with their own reports
withheld, so the locator has to find them): ``run_ais_aware`` (AIS ships refined, projected out and
excluded), the matched field scan, then the occupancy filter.  Two maps are scored: ``occupancy`` (the
filter's posterior per frame times the number of cells, the chain's output) and ``scan`` (the matched
field statistic it consumes).  Neither threshold is guaranteed to give 5 % on the full scene: it is set on
runs with one object removed, so whatever that object adds beyond the false alarm radius (the sidelobes of a
ship in the scan map) appears only in the full scene, and the occupancy posterior is renormalised to sum 1
every frame, so the mass the objects hold in the full scene is taken from the background (and a weak
object's Pd depends on how loud the others are).  Every scoreboard therefore reports the rate its threshold
gives on the full scene (``false_alarm_frame_rate``, frames with at least one false alarm) beside Pd.
Configuration ``all_estimated`` (free field replicas at the chart's sound speed); ``--propagator simulator``
gives the same chain with the FDTD dispersion corrected
replicas (simulator knowledge, labelled ``simulator_propagator``).

``echo`` (the silent bodies; each AIS ship of the dataset as the illuminator in turn, its reports
the only ones the chain reads): ``run_bistatic`` with the coherent map normalised per cell
(``coherent_norm="snr"``) and not divided by its maximum (``map_scale="none"``, so one threshold
applies across frames, runs and datasets), ECA with per sample delays and the refined track fitted
over 4 s.  Configurations: ``all_estimated`` (the observation's traces, the data reference, the AIS
refined track, free field replicas); ``true_waveform_exact_track`` (given the true waveform and the
exact track); ``direct_removed_exactly`` (the illuminator's direct path removed with its truth
component, true waveform, exact track).  The last two are given truth inputs, but they are not upper
limits: the true waveform is the emitted signal, while the replicas use the chart's sound speed and the
FDTD propagates with its own direction dependent numerical phase speed, so the echo arrives 0.5 to 3 ms
off the true waveform and the free field matched filter loses the UUV (``docs/harbor.md``, stage 1); the
data reference is estimated from the traces and carries the same propagation as the echo.

Data split, fixed before any run: the noise seed 0 mix of each dataset is the tuning set, seeds 1 to
5 (mixed from the cached FDTD gathers with ``assemble_dataset(seed=...)``) the test set.  Every
parameter is set below before the test set is run; the threshold comes from object free runs of the
tuning set (the object's contribution subtracted with its truth components, everything else kept)
and is frozen.  On the multistatic dataset the scattering of the ten bodies is one combined gather
per illuminator, so the object free run of the echo chain removes every body at once.

Criteria, fixed before the run (test set, everything estimated, free field replicas):

1. passive: ferry, tug and boat of the multistatic dataset each Pd >= 0.9 at the 5 % threshold;
2. echo, bistatic dataset: Pd of the UUV >= 0.5;
3. echo, multistatic dataset, ferry as illuminator: >= 3 of the 7 static bodies detected on the
   full record map (median over the test seeds).

Outputs in ``--out``: ``scoreboard_<dataset>_<chain>_<config>.png`` / ``.json`` (tuning and test side
by side), ``..._frame.png`` (one test frame with its threshold contour, exclusion zones, peaks and ray paths),
``pd_summary.png``, ``misses_<dataset>.png``, ``results.json``, ``run.log``.

Usage::

    python examples/harbor/90_history/harbor_picture_stage1.py [--dataset all|bistatic|multistatic|harbor]
        [--chain all|passive|echo] [--propagator free_field|simulator] [--cuda] [--workers 1]
        [--split both|tuning|test] [--out output/harbor_picture/stage1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Provenance, annotate_figure  # noqa: E402
from acoustic_sim.water import HarborDataset, assemble_dataset, build_environment, build_layout, build_track  # noqa: E402
from acoustic_sim.water import fdtd_runner  # noqa: E402
from acoustic_sim.water.ais import AISReports  # noqa: E402
from acoustic_sim.water.bistatic import BistaticConfig, Diagnostics, clutter_images, eca_stage, exclusion, run_bistatic  # noqa: E402
from acoustic_sim.water.dataset import render_vessel_gathers  # noqa: E402
from acoustic_sim.water.detection import ProcessingConfig  # noqa: E402
from acoustic_sim.water.diagnostics import SimulatorAwarePropagator, true_illuminator_waveform  # noqa: E402
from acoustic_sim.water.occupancy import OccupancyConfig, OccupancyGrid  # noqa: E402
from acoustic_sim.water.plotting import draw_frame_geometry  # noqa: E402
from acoustic_sim.water.propagation import FreeFieldPropagator  # noqa: E402
from acoustic_sim.water.scoreboard import ChainMaps, Scoreboard, plot_map, score, set_threshold  # noqa: E402
from acoustic_sim.water.suppression import SuppressionConfig, run_ais_aware  # noqa: E402

TUNING_SEED = 0
TEST_SEEDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Spec:
    path: str
    cache: str
    combined: bool
    band: tuple[float, float]
    chains: tuple[str, ...]
    sources: tuple[str, ...] = ()


DATASETS = {
    "bistatic": Spec("output/harbor_bistatic/fine/dataset.npz", "output/harbor_bistatic/cache", False, (100.0, 760.0), ("echo",), ("ferry",)),
    "multistatic": Spec("output/harbor_multistatic/fine/dataset.npz", "output/harbor_multistatic/cache", True, (100.0, 760.0),
                        ("passive", "echo"), ("ferry", "tug")),
    "harbor": Spec("output/harbor/dataset.npz", "output/harbor/cache", False, (12.0, 78.0), ("passive",)),
}
ECHO_CONFIGS = {
    # name: (direct path in the traces, reference, exact track)
    "all_estimated": (True, "data", False),
    "true_waveform_exact_track": (True, "true_waveform", True),
    "direct_removed_exactly": (False, "true_waveform", True),
}
PASSIVE_MAPS = ("occupancy", "scan")
CRITERIA = [
    {"id": 1, "text": "passive chain, multistatic dataset: ferry, tug and boat each Pd >= 0.9 at the 5 % threshold",
     "dataset": "multistatic", "chain": "passive", "config": "all_estimated/occupancy", "objects": ["ferry", "tug", "boat"], "pd_min": 0.9},
    {"id": 2, "text": "echo chain, bistatic dataset: Pd of the UUV >= 0.5",
     "dataset": "bistatic", "chain": "echo", "config": "ferry/all_estimated", "objects": ["uuv1"], "pd_min": 0.5},
    {"id": 3, "text": "echo chain, multistatic dataset, ferry as illuminator: >= 3 of the 7 static bodies detected on the full record map",
     "dataset": "multistatic", "chain": "echo", "config": "ferry/all_estimated", "static_min": 3},
]
EXPECTATION = ("criterion 1 passes; 2 and 3 fail, because the direct path of a moving ship is removed by 8 to 19 dB when its "
               "waveform is estimated from the traces and the echo sits 16 to 18 dB below it")

# Parameters, fixed on the tuning set before the test set is run.
ECHO = dict(win_s=0.5, hop_s=0.25, bin_stride=2, grid_dx=2.0, grid_margin=5.0, edge=40.0, method="coherent", coherent_norm="snr",
            map_scale="none", exclusion_radius=25.0, eca_doppler=True, track_smooth_s=4.0, refine_half=10.0, refine_step=1.0)
PASSIVE = {
    # 0.5 m datasets: lengths scaled from the 5 m defaults by the band (wavelengths 2 to 15 m against 19 to 125 m)
    "multistatic": dict(pc=dict(win_s=0.5, hop_s=0.25, bin_stride=2, grid_dx=2.0, grid_margin=5.0, min_range=1.0, edge=40.0),
                        sc=dict(refine_half=12.0, refine_step=1.0, patch_half=0.5, exclusion_radius=25.0),
                        occ=dict(speed_mps=5.0)),
    # 5 m dataset: the defaults of ProcessingConfig / SuppressionConfig, with a 5 m scan grid so the 5 m detection radius can be met
    "harbor": dict(pc=dict(grid_dx=5.0, edge=150.0), sc=dict(), occ=dict(speed_mps=5.0)),
}
FA_EXCLUSION_RADIUS = {"echo": ECHO["exclusion_radius"]}
MISS_CLASSES = ("ship exclusion edge", "quay image of the ship", "direct path ship to cable", "elsewhere")


def log_to(path: Path):
    f = open(path, "a")

    def say(msg: str) -> None:
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()
    return say


# ----------------------------------------------------------------------
# Data: the tuning and test mixes from the cached gathers
# ----------------------------------------------------------------------

def _no_fdtd(*a, **k):
    raise RuntimeError("an FDTD gather is missing from the cache (stage 1 builds nothing; pass --allow-fdtd to run it)")


class Mixer:
    """The dataset's cached unit gathers, mixed with any noise seed."""

    def __init__(self, name: str, allow_fdtd: bool = False) -> None:
        spec = DATASETS[name]
        self.name, self.spec = name, spec
        ds = HarborDataset.load(ROOT / spec.path)
        self.disk_traces = ds.traces
        self.cfg = ds.cfg
        self.env = build_environment(self.cfg)
        self.layout = build_layout(self.cfg, self.env)
        self.tracks = {v.id: build_track(v, self.cfg.fdtd.total_time, self.env) for v in self.cfg.vessels}
        if not allow_fdtd:
            fdtd_runner.HarborFDTDBatch.run = _no_fdtd
            fdtd_runner.HarborFDTDSolver.run = _no_fdtd
        self.gathers = render_vessel_gathers(self.cfg, self.env, self.layout, self.tracks, cache_dir=ROOT / spec.cache,
                                             combined_bodies=spec.combined)

    def mix(self, seed: int):
        ds = assemble_dataset(self.cfg, self.layout, self.tracks, self.gathers, seed=seed, keep_components=True, env=self.env)
        return ds.observation, ds.truth


_MIXERS: dict[str, Mixer] = {}


def mixer(name: str, allow_fdtd: bool = False) -> Mixer:
    if name not in _MIXERS:
        _MIXERS[name] = Mixer(name, allow_fdtd)
    return _MIXERS[name]


def reports_of(rep: AISReports, keep=None, drop=None) -> AISReports:
    """The reports of the vessels in *keep* (or all but *drop*)."""
    m = np.ones(len(rep), bool)
    if keep is not None:
        m &= np.isin(rep.vessel, list(keep))
    if drop is not None:
        m &= ~np.isin(rep.vessel, list(drop))
    return AISReports(rep.vessel[m], rep.t[m], rep.xy[m], rep.sog[m], rep.cog[m])


def without(obs, truth, prefixes) -> np.ndarray:
    """The observed traces with the truth components whose keys start with any of *prefixes* subtracted
    (the gain scaled contribution of those objects: noise and everything else stay as recorded)."""
    x = np.asarray(obs.traces, dtype=np.float64).copy()
    gain = truth.gain[:, None]
    for key, comp in truth.components.items():
        if key != "noise" and key.startswith(tuple(prefixes)):
            x -= gain * comp.astype(np.float64)
    return x


def radiators(truth) -> list[str]:
    return [k for k, t in truth.tracks.items() if t.level_pa > 0]


def bodies(truth) -> list[str]:
    return list(truth.meta.get("bodies", []))


# ----------------------------------------------------------------------
# One run of a chain
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    dataset: str
    chain: str
    config: str             # echo: "<source>/<config>"; passive: "all_estimated"
    seed: int
    target: str | None      # passive: the AIS ship whose reports are withheld (None: all reports)
    free: tuple[str, ...]   # objects removed from the traces (object free run), () for the full run
    propagator: str
    cuda: bool

    @property
    def held_out(self) -> bool:
        return self.seed != TUNING_SEED

    @property
    def label(self) -> str:
        parts = [self.dataset, self.chain, self.config, f"seed{self.seed}"]
        if self.target:
            parts.append(f"without_{self.target}_ais")
        if self.free:
            parts.append("free_of_" + "+".join(self.free) if len(self.free) < 4 else "free_of_bodies")
        return ":".join(parts)


def _propagator(kind: str, env, truth):
    return SimulatorAwarePropagator.from_truth(truth) if kind == "simulator" else FreeFieldPropagator.from_environment(env)


def _other_ship_exclusion(grid, times, reports: AISReports, skip: str, radius: float) -> np.ndarray:
    """Cells within *radius* of the AIS predicted position of every reporting ship other than *skip* (batch scoring:
    like ``bistatic.ship_positions`` it marks a ship from 60 s before its first report)."""
    mask = np.zeros((len(grid), len(times)), bool)
    for vid in reports.vessels:
        if vid == skip:
            continue
        tr = reports.for_vessel(vid)
        pos = tr.predict(times)
        live = (times >= tr.t[0] - 60.0) & (tr.age(times) <= 60.0)
        for k in np.flatnonzero(live):
            mask[:, k] |= np.hypot(*(grid - pos[k]).T) < radius
    return mask


def classify_misses(B, grid, times, ship_xy, present, truth_xy, env, layout, R) -> list[dict]:
    """The frame's highest cell outside the exclusion when it is more than 20 m from every UUV, by class."""
    cab = layout.das_positions
    out = []
    for k in range(B.shape[1]):
        if not present[k] or B[:, k].max() <= 0:
            continue
        p = grid[int(np.argmax(B[:, k]))]
        d_uuv = float(np.min(np.hypot(*(truth_xy[:, k] - p).T)))
        if d_uuv <= 20.0:
            continue
        s = ship_xy[k]
        d_ship = float(np.hypot(*(p - s)))
        imgs = clutter_images(env, s)
        # distance from p to the segments ship -> every cable channel
        v = cab - s
        t = np.clip(((p - s) @ v.T) / np.maximum(np.sum(v * v, axis=1), 1e-12), 0.0, 1.0)
        d_path = float(np.min(np.hypot(*(s + t[:, None] * v - p).T)))
        if R <= d_ship <= R + 10.0:
            cls = MISS_CLASSES[0]
        elif len(imgs) and np.min(np.hypot(*(imgs - p).T)) <= 10.0:
            cls = MISS_CLASSES[1]
        elif d_path <= 10.0:
            cls = MISS_CLASSES[2]
        else:
            cls = MISS_CLASSES[3]
        out.append({"frame": k, "t": float(times[k]), "xy": p.tolist(), "class": cls, "uuv_distance_m": d_uuv})
    return out


def run_job(job: Job, allow_fdtd: bool = False) -> dict:
    """Run one chain configuration on one mix; returns the maps (``ChainMaps``) and diagnostics."""
    t0 = time.time()
    mx = mixer(job.dataset, allow_fdtd)
    obs, truth = mx.mix(job.seed)
    env, layout = obs.environment, obs.geometry
    spec = mx.spec
    prop = _propagator(job.propagator, env, truth)
    out: dict = {"job": job, "diag": {}}
    if job.chain == "echo":
        src, cfg_name = job.config.split("/")
        direct_in, reference, exact = ECHO_CONFIGS[cfg_name]
        reports = reports_of(obs.reports, keep=[src])
        given: list[str] = []
        traces = np.asarray(obs.traces, dtype=np.float64)
        if not direct_in:
            traces = without(obs, truth, [f"direct:{src}"])
            given += ["direct_removal", "components"]
        if job.free:
            traces = without(obs, truth, ([] if direct_in else [f"direct:{src}"]) + ["scattered:"])
            given = sorted(set(given) | {"components"})
        bc = BistaticConfig(f_lo=spec.band[0], f_hi=spec.band[1], c=prop.c, reference=reference, removal="eca" if direct_in else "none",
                            propagator=prop, use_cuda=job.cuda, **ECHO)
        waveform = true_illuminator_waveform(truth, src) if reference == "true_waveform" else None
        nper, hop = int(round(bc.win_s * obs.fs)), int(round(bc.hop_s * obs.fs))
        times = (np.arange(0, traces.shape[1] - nper + 1, hop) + nper / 2.0) / obs.fs
        diag = Diagnostics(exact_track=truth.tracks[src].position_at(times) if exact else None)
        run = run_bistatic(traces, layout, obs.fs, env, reports, bc, true_waveform=waveform, diagnostics=diag,
                           provenance=Provenance(truth_inputs=given, held_out=job.held_out))
        ex = exclusion(run.grid, run.ship, bc.exclusion_radius) | _other_ship_exclusion(run.grid, run.sp.times, obs.reports, src,
                                                                                        bc.exclusion_radius)
        out["maps"] = {"": ChainMaps(run.grid, run.sp.times, run.scan.B, ex, run.provenance, job.free, label=job.label)}
        out["ship_error_m"] = float(np.median(np.hypot(*(run.ship.refined - truth.tracks[src].position_at(run.sp.times)).T)))
        if not job.free and direct_in:
            # Diagnostic: the source's direct and scattered components through the same ECA (after the run).
            gain = truth.gain[:, None]
            d = gain * truth.components[f"direct:{src}"].astype(np.float64)
            e = gain * sum(c.astype(np.float64) for k, c in truth.components.items() if k.startswith(f"scattered:{src}>"))
            outs, _ = eca_stage(traces, layout, obs.fs, env, run.sp, run.ship, bc, waveform, [d, e])
            live = (run.sp.weights > 0) & (np.arange(layout.n_receivers) < layout.n_das)
            depth = 10 * np.log10(np.sum(d ** 2, axis=1) / np.maximum(np.sum(outs[1] ** 2, axis=1), 1e-30))
            loss = 10 * np.log10(np.sum(e ** 2, axis=1) / np.maximum(np.sum(outs[2] ** 2, axis=1), 1e-30))
            out["diag"]["cancellation"] = {
                "direct_depth_db_median": float(np.median(depth[live])), "direct_depth_db_p10": float(np.percentile(depth[live], 10)),
                "echo_loss_db_median": float(np.median(loss[live])), "channels": "live DAS channels",
                "provenance": run.provenance.with_flags(["components"]).to_dict(),
                "note": "diagnostic: truth components through the same projection after the run; not part of the processing"}
        if not job.free:
            moving = [b for b in bodies(truth) if np.ptp(truth.tracks[b].xy, axis=0).max() > 0 and truth.tracks[b].kind == "uuv"]
            uuv_xy = np.stack([truth.tracks[b].position_at(run.sp.times) for b in moving])
            out["misses"] = classify_misses(np.where(ex, 0.0, run.scan.B), run.grid, run.sp.times, run.ship.refined, run.ship.present,
                                            uuv_xy, env, layout, bc.exclusion_radius)
            out["miss_frames"] = int(np.sum(run.ship.present))
    else:
        p = PASSIVE[job.dataset]
        pcd = dict(p["pc"])
        edge = pcd.pop("edge")
        x0, x1, y0, y1 = env.extent
        pc = ProcessingConfig(f_lo=spec.band[0], f_hi=spec.band[1], c=prop.c, propagator=prop, use_cuda=job.cuda,
                              grid_bounds=(x0 + edge, x1 - edge, y0 + edge, y1 - edge), **pcd)
        sc = SuppressionConfig(**p["sc"])
        reports = reports_of(obs.reports, drop=[job.target] if job.target else None)
        traces = np.asarray(obs.traces, dtype=np.float64)
        given = []
        if job.free:
            traces = without(obs, truth, [f"direct:{o}" for o in job.free] + [f"scattered:{o}>" for o in job.free])
            given = ["components"]
        run = run_ais_aware(traces, layout, obs.fs, env, reports, pc, sc, floor=1e18,
                            provenance=Provenance(truth_inputs=given, held_out=job.held_out))
        grid, times = run.scanner.grid, run.scan.times
        ex = run.ships.exclusion_mask(grid, sc.exclusion_radius) if run.ships.ids else np.zeros(run.scan.B.shape, bool)
        occ = OccupancyGrid(grid, OccupancyConfig(use_cuda=job.cuda, **p["occ"]))
        hist = occ.run(run.scan.B, times, snapshot_frames=tuple(range(len(times))))
        P = np.stack([hist.snapshots[k] for k in range(len(times))], axis=1) * len(grid)
        out["maps"] = {"scan": ChainMaps(grid, times, run.scan.B, ex, run.provenance, job.free, label=job.label + ":scan"),
                       "occupancy": ChainMaps(grid, times, P, ex, run.provenance, job.free, label=job.label + ":occupancy")}
        out["ships"] = {vid: float(np.mean(run.ships.significant[:, v])) for v, vid in enumerate(run.ships.ids)}
    out["seconds"] = time.time() - t0
    return out


# ----------------------------------------------------------------------
# Job lists
# ----------------------------------------------------------------------

def config_names(dataset: str, chain: str, truth_like) -> list[str]:
    if chain == "echo":
        return [f"{src}/{c}" for src in DATASETS[dataset].sources for c in ECHO_CONFIGS]
    return ["all_estimated"]


def jobs_for(dataset: str, chain: str, propagator: str, cuda: bool, seeds, truth) -> list[Job]:
    jobs = []
    rad = radiators(truth)
    ais_ships = [k for k in rad if truth.tracks[k].ais]
    for cfg in config_names(dataset, chain, truth):
        for seed in seeds:
            if chain == "echo":
                jobs.append(Job(dataset, chain, cfg, seed, None, (), propagator, cuda))
                if seed == TUNING_SEED:
                    jobs.append(Job(dataset, chain, cfg, seed, None, tuple(bodies(truth)), propagator, cuda))
            else:
                jobs.append(Job(dataset, chain, cfg, seed, None, (), propagator, cuda))
                for s in ais_ships:
                    jobs.append(Job(dataset, chain, cfg, seed, s, (), propagator, cuda))
                if seed == TUNING_SEED:
                    for o in rad:
                        jobs.append(Job(dataset, chain, cfg, seed, o if o in ais_ships else None, (o,), propagator, cuda))
    return jobs


# ----------------------------------------------------------------------
# Scoring and figures
# ----------------------------------------------------------------------

def scoreboards(dataset: str, chain: str, cfg: str, mapname: str, results: dict, truths: dict, seeds_test) -> dict:
    """Threshold from the tuning set's object free runs; tuning and test scoreboards."""
    def get(seed, target=None, free=()):
        return results[(cfg, seed, target, tuple(free))]["maps"][mapname]

    truth0 = truths[TUNING_SEED]
    free_runs = [r["maps"][mapname] for (c, s, t, f), r in results.items() if c == cfg and s == TUNING_SEED and f]
    thr = set_threshold(free_runs, truth0)
    if chain == "echo":
        targets = bodies(truth0)
    else:
        targets = radiators(truth0)
    ais_ships = [k for k in targets if truth0.tracks[k].ais]
    pilings = [b for b in targets if b.startswith("piling")] or None
    out = {"threshold": thr}
    for split, seeds in (("tuning", (TUNING_SEED,)), ("test", tuple(seeds_test))):
        if not seeds or any((cfg, s, None, ()) not in results for s in seeds):
            continue
        maps = {o: [get(s, o if (chain == "passive" and o in ais_ships) else None) for s in seeds] for o in targets}
        sb = score(maps, [truths[s] for s in seeds], thr, primary=[get(s) for s in seeds], pilings=pilings,
                   label=f"{dataset} / {chain} / {cfg}{'' if mapname == '' else ' / ' + mapname} / {split} set")
        sb.meta = {"split": split, "seeds": list(seeds)}
        out[split] = sb
    return out


def plot_frame(key: str, run: ChainMaps, T: float, truth, reports: AISReports, chain: str, src: str | None, env, layout,
               path: Path) -> None:
    """One test frame of a scoreboard's map with its geometry: the threshold contour, the exclusion zones, the local
    maxima above ``T`` (detections near an object, false alarms elsewhere), the AIS ships at their predicted
    positions with their direct ray paths and, for the echo chain, the path illuminator - strongest peak - sensors
    and the equal delay ellipse through it.  The frame is the middle one of the record."""
    from acoustic_sim.water.scoreboard import local_maxima

    K = len(run.times)
    B = np.where(run.exclusion, 0.0, run.B)
    k = K // 2
    t = float(run.times[k])
    rads = {}
    for vid in reports.vessels:
        tr = reports.for_vessel(vid)
        if tr.t[0] - 60.0 <= t and tr.age(np.array([t]))[0] <= 60.0 and (chain == "passive" or vid == src):
            rads[vid] = tr.predict(np.array([t]))[0]
    pk = local_maxima(B[:, k], run.grid) & (B[:, k] > T)
    order = np.argsort(B[pk, k])[::-1]
    peaks = run.grid[pk][order]
    below = not len(peaks)
    if below:                                                  # nothing above T: show the strongest cell, marked as such
        peaks = run.grid[[int(np.argmax(B[:, k]))]]
    fig, ax = plt.subplots(figsize=(10, 8))
    x0, x1, y0, y1 = env.extent
    ax.imshow(np.where(env.water, 1.0, 0.0), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys_r", vmin=-0.6, vmax=1.2)
    im = draw_frame_geometry(ax, run.grid, run.B[:, k], layout, rads, peaks, echo=chain == "echo",
                             threshold=T, exclusion=run.exclusion[:, k])
    for oid, tr in truth.tracks.items():
        ax.plot(*tr.position_at(t)[0], "o", ms=11, mfc="none", mec="deepskyblue" if tr.level_pa > 0 else "orange", mew=1.6)
        ax.annotate(oid, tr.position_at(t)[0], xytext=(5, -10), textcoords="offset points", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="dB re frame peak")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"{key}, test seed, t = {t:.2f} s: " + ("no local maximum above T = {:.4g}; the strongest cell is marked".format(T)
                 if below else f"{len(peaks)} local maxima above T = {T:.4g}") + "\n"
                 f"AIS ships at their predicted positions; circles: true positions (scoring side)", fontsize=10)
    annotate_figure(fig, run.provenance)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _after(m: ChainMaps, t0: float) -> ChainMaps:
    keep = m.times >= t0
    return ChainMaps(m.grid, m.times[keep], m.B[:, keep], m.exclusion[:, keep], m.provenance, m.removed, m.present,
                     m.label + ":after_first_ais_report")


def after_first_report_diagnostic(cfg: str, mapname: str, results: dict, truths: dict, reports: dict, seeds_test) -> dict:
    """Diagnostic, not a criterion: the passive scoreboard on the frames after every reporting ship of a
    run has sent its first AIS report (before it the chain neither projects out nor excludes a ship)."""
    def t_first(seed, target):
        rep = reports[seed]
        return max(rep.for_vessel(v).t[0] for v in rep.vessels if v != target)

    truth0 = truths[TUNING_SEED]
    free = [_after(r["maps"][mapname], t_first(s, t)) for (c, s, t, f), r in results.items() if c == cfg and s == TUNING_SEED and f]
    thr = set_threshold(free, truth0)
    targets = radiators(truth0)
    ais_ships = [k for k in targets if truth0.tracks[k].ais]
    out = {"note": "diagnostic: frames after the first AIS report of every reporting ship of the run; threshold from the same "
                   "frames of the object free runs; not part of any criterion", "threshold": thr.to_dict()}
    for split, seeds in (("tuning", (TUNING_SEED,)), ("test", tuple(seeds_test))):
        if not seeds or any((cfg, s, None, ()) not in results for s in seeds):
            continue
        maps = {o: [_after(results[(cfg, s, o if o in ais_ships else None, ())]["maps"][mapname], t_first(s, o if o in ais_ships else None))
                    for s in seeds] for o in targets}
        prim = [_after(results[(cfg, s, None, ())]["maps"][mapname], t_first(s, None)) for s in seeds]
        sb = score(maps, [truths[s] for s in seeds], thr, primary=prim)
        n_all = sum(len(results[(cfg, s, None, ())]["maps"][mapname].times) for s in seeds)
        out[split] = {"pd": {o.id: o.pd for o in sb.per_object},
                      "median_error_m": {o.id: (o.median_error_m if np.isfinite(o.median_error_m) else None) for o in sb.per_object},
                      "false_alarm_rate": sb.false_alarm_rate, "frames_kept": sb.n_frames, "frames_all": n_all}
    return out


def argmax_diagnostic(cfg: str, results: dict, truths: dict, seeds) -> dict:
    """Diagnostic: distance from every frame's highest cell (outside the exclusion zones) to the nearest moving UUV,
    over the frames with an illuminator, whatever its level (the scoreboard counts only peaks above the threshold)."""
    out = {}
    for split, sd in (("tuning", [s for s in seeds if s == TUNING_SEED]), ("test", [s for s in seeds if s != TUNING_SEED])):
        d = []
        for s in sd:
            m = results[(cfg, s, None, ())]["maps"][""]
            tru = truths[s]
            uuvs = [b for b in bodies(tru) if tru.tracks[b].kind == "uuv" and np.ptp(tru.tracks[b].xy, axis=0).max() > 0]
            B = np.where(m.exclusion, 0.0, m.B)
            for k in np.flatnonzero(B.max(axis=0) > 0):
                p = m.grid[int(np.argmax(B[:, k]))]
                d.append(min(float(np.hypot(*(tru.tracks[u].position_at(m.times[k])[0] - p))) for u in uuvs))
        if d:
            d = np.array(d)
            out[split] = {"frames": int(len(d)), "median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90)),
                          "within_5m": float(np.mean(d <= 5.0)), "within_20m": float(np.mean(d <= 20.0))}
    return out


def plot_scoreboard_pair(sbs: dict, path: Path, title: str) -> None:
    """Tuning and test tables side by side."""
    splits = [s for s in ("tuning", "test") if s in sbs]
    rows = max(len(sbs[s].per_object) for s in splits)
    fig, axes = plt.subplots(1, len(splits), figsize=(9.5 * len(splits), 1.4 + 0.34 * (rows + 3)), squeeze=False)
    for ax, split in zip(axes[0], splits):
        sb: Scoreboard = sbs[split]
        ax.axis("off")
        cols = ["object", "kind", "rad.", "AIS", "moving", "frames", "Pd", "med. err", "p90 err", "full record"]
        cells = [[o.id, o.kind, "y" if o.radiates else "n", "y" if o.ais else "n", "y" if o.moving else "n", str(o.frames),
                  f"{o.pd:.2f}", "-" if not np.isfinite(o.median_error_m) else f"{o.median_error_m:.1f}",
                  "-" if not np.isfinite(o.p90_error_m) else f"{o.p90_error_m:.1f}",
                  "-" if o.full_record_detected is None else f"{o.full_record_detected:.2f}"] for o in sb.per_object]
        tab = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.5)
        tab.scale(1.0, 1.3)
        for (r, c), cell in tab.get_celld().items():
            if r > 0 and c == 6:
                cell.set_facecolor(plt.cm.RdYlGn(0.15 + 0.7 * sb.per_object[r - 1].pd))
        sep = (f"\npilings resolved on the full record map: {sb.separation['resolved_per_run']}" if sb.separation else "")
        ax.set_title(f"{split} set (seeds {sb.meta['seeds']}): T = {sb.threshold.T:.4g}, full record T = {sb.threshold.T_full:.4g}\n"
                     f"false alarms {sb.false_alarm_rate:.2f} per frame ({sb.false_alarms} in {sb.n_frames} frames), frames with a "
                     f"false alarm {100 * sb.false_alarm_frame_rate:.0f} % (threshold set for {100 * sb.threshold.pfa:.0f} %){sep}", fontsize=9)
        ax.text(0.5, -0.02, sb.provenance.label(), transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
                color="0.25" if sb.provenance.all_estimated else "firebrick")
    h = fig.get_size_inches()[1]
    fig.suptitle(title + "\nPd over active frames; errors [m] over detected frames; full record: fraction of runs whose full record map "
                 "detects the static object", fontsize=10, y=1.0 + 0.75 / h, va="bottom")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_pd_summary(summary: list[dict], path: Path) -> None:
    """One figure: every object, every configuration, tuning and test side by side (rows), Pd in the cells."""
    panels: dict[str, list[dict]] = {}
    for e in summary:
        panels.setdefault(f"{e['dataset']} / {e['chain']}", []).append(e)
    heights = [2 * len(v) for v in panels.values()]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 0.42 * sum(heights) + 1.3 * len(panels)), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    for ax, (name, entries) in zip(axes[:, 0], panels.items()):
        objects = list(dict.fromkeys(o for e in entries for s_ in ("tuning", "test") for o in e["pd"].get(s_, {})))
        rows, labels = [], []
        for e in entries:
            for split in ("tuning", "test"):
                rows.append([e["pd"].get(split, {}).get(o, np.nan) for o in objects])
                labels.append(f"{e['config']}  ({split})")
        M = np.array(rows, dtype=float)
        ax.imshow(np.ma.masked_invalid(M), cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                ax.text(c, r, "-" if not np.isfinite(M[r, c]) else f"{M[r, c]:.2f}", ha="center", va="center", fontsize=8,
                        fontweight="bold" if labels[r].endswith("(test)") else "normal")
        for r in range(1, M.shape[0], 2):
            ax.axhline(r + 0.5, color="w", lw=2.5)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks(np.arange(len(objects)))
        ax.set_xticklabels(objects, fontsize=8.5)
        ax.xaxis.tick_top()
        ax.set_title(name, fontsize=10, loc="left")
    fig.suptitle("Harbor picture stage 1: Pd per object at the 5 % threshold (tuning set: seed 0; test set, bold: seeds 1 to 5)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_misses(dataset: str, misses: dict, env, layout, truth, path: Path) -> None:
    configs = list(misses)
    fig, axes = plt.subplots(1, len(configs), figsize=(6.2 * len(configs), 5.2), squeeze=False)
    colours = dict(zip(MISS_CLASSES, ("tab:red", "tab:orange", "tab:blue", "0.4")))
    x0, x1, y0, y1 = env.extent
    for ax, cfg in zip(axes[0], configs):
        ax.imshow(np.where(env.water, 1.0, 0.0), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys_r", vmin=-0.6, vmax=1.2)
        ax.plot(layout.das_positions[:, 0], layout.das_positions[:, 1], ".", color="k", ms=1.5)
        for oid, tr in truth.tracks.items():
            ax.plot(tr.xy[:, 0], tr.xy[:, 1], "-" if np.ptp(tr.xy, axis=0).max() > 0 else "o", color="lime" if tr.level_pa == 0 else "deepskyblue",
                    lw=1.2, ms=3)
        m = misses[cfg]["frames"]
        counts = {c: 0 for c in MISS_CLASSES}
        for e in m:
            counts[e["class"]] += 1
        for c in MISS_CLASSES:
            pts = np.array([e["xy"] for e in m if e["class"] == c]).reshape(-1, 2)
            ax.plot(pts[:, 0], pts[:, 1], "o", color=colours[c], ms=3.5, alpha=0.7, label=f"{c}: {counts[c]}")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{cfg}\n{len(m)} of {misses[cfg]['frames_total']} frames peak > 20 m from every UUV", fontsize=9)
    fig.suptitle(f"{dataset}: where the echo chain's frame peak goes when it misses the UUV (all seeds)", fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

PARAMS_KEY = hashlib.sha1(json.dumps({"echo": ECHO, "passive": PASSIVE, "bands": {k: v.band for k, v in DATASETS.items()}},
                                     sort_keys=True).encode()).hexdigest()[:12]


def cached_run(job: Job, cache: Path, allow_fdtd: bool = False) -> dict:
    """:func:`run_job`, with its result kept in *cache* (maps as float32) under a key of the job and the parameters."""
    key = hashlib.sha1(f"{job}|{PARAMS_KEY}".encode()).hexdigest()[:16]
    path = cache / f"{job.dataset}_{job.chain}_{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            r = pickle.load(f)
        r["cached"] = True
        return r
    r = run_job(job, allow_fdtd)
    for m in r["maps"].values():
        m.B = m.B.astype(np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    return r


def _work(args):
    job, cache, allow = args
    return cached_run(job, cache, allow)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    ap.add_argument("--chain", default="all", choices=("all", "passive", "echo"))
    ap.add_argument("--propagator", default="free_field", choices=("free_field", "simulator"))
    ap.add_argument("--cuda", action="store_true", help="GPU for every chain that supports it (falls back to the CPU)")
    ap.add_argument("--workers", type=int, default=1, help="runs in parallel (processes)")
    ap.add_argument("--split", default="both", choices=("both", "tuning", "test"))
    ap.add_argument("--out", default="output/harbor_picture/stage1")
    ap.add_argument("--allow-fdtd", action="store_true", help="run FDTD for gathers missing from the cache (never needed here)")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    say = log_to(out / "run.log")
    say(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv[1:])}")
    for c in CRITERIA:
        say(f"criterion {c['id']} (fixed before the run): {c['text']}")
    say(f"expectation written before the run: {EXPECTATION}")
    res_path = out / "results.json"
    results_json = json.loads(res_path.read_text()) if res_path.exists() else {}
    results_json.setdefault("criteria", CRITERIA)
    results_json["expectation"] = EXPECTATION
    results_json["split"] = {"tuning": [TUNING_SEED], "test": list(TEST_SEEDS)}
    results_json["parameters"] = {"echo": ECHO, "passive": PASSIVE, "bands_hz": {k: v.band for k, v in DATASETS.items()},
                                  "scoreboard": {"detect_radius_m": 5.0, "false_alarm_radius_m": 10.0, "pfa": 0.05}}
    results_json.setdefault("entries", {})
    seeds = {"both": (TUNING_SEED, *TEST_SEEDS), "tuning": (TUNING_SEED,), "test": TEST_SEEDS}[args.split]
    prop_tag = "" if args.propagator == "free_field" else "_simulator_propagator"

    for dataset in ([*DATASETS] if args.dataset == "all" else [args.dataset]):
        mx = mixer(dataset, args.allow_fdtd)
        obs0, truth0 = mx.mix(TUNING_SEED)
        same = float(np.max(np.abs(obs0.traces.astype(np.float64) - mx.disk_traces.astype(np.float64))))
        say(f"[{dataset}] seed 0 remix against the dataset on disk: max |diff| {same:.3g} "
            f"(traces rms {float(np.sqrt(np.mean(mx.disk_traces.astype(np.float64) ** 2))):.3g})")
        mixes = {s: (mx.mix(s) if s != TUNING_SEED else (obs0, truth0)) for s in seeds}
        truths = {s: m[1] for s, m in mixes.items()}
        reports = {s: m[0].reports for s, m in mixes.items()}
        del mixes
        for chain in (DATASETS[dataset].chains if args.chain == "all" else [args.chain]):
            if chain not in DATASETS[dataset].chains:
                continue
            jobs = jobs_for(dataset, chain, args.propagator, args.cuda, seeds, truth0)
            say(f"[{dataset}/{chain}] {len(jobs)} runs, propagator {args.propagator}")
            results: dict = {}
            t0 = time.time()
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    it = pool.map(_work, [(j, out / "cache", args.allow_fdtd) for j in jobs])
                    for j, r in zip(jobs, it):
                        results[(j.config, j.seed, j.target, j.free)] = r
                        say(f"  {j.label}: {r['seconds']:.0f} s")
            else:
                for j in jobs:
                    r = cached_run(j, out / "cache", args.allow_fdtd)
                    results[(j.config, j.seed, j.target, j.free)] = r
                    extra = ""
                    if "cancellation" in r["diag"]:
                        extra = f", direct path removed by {r['diag']['cancellation']['direct_depth_db_median']:.1f} dB (diagnostic)"
                    if "ship_error_m" in r:
                        extra += f", illuminator track error {r['ship_error_m']:.1f} m"
                    say(f"  {j.label}: {r['seconds']:.0f} s{extra}{' (cached)' if r.get('cached') else ''}")
            say(f"[{dataset}/{chain}] runs done in {time.time() - t0:.0f} s")

            misses: dict = {}
            for cfg in config_names(dataset, chain, truth0):
                for mapname in (("",) if chain == "echo" else PASSIVE_MAPS):
                    name = cfg.replace("/", "_") + prop_tag + ("" if mapname == "" else "_" + mapname)
                    key = f"{dataset}/{chain}/{cfg}{prop_tag}" + ("" if mapname == "" else "/" + mapname)
                    sbs = scoreboards(dataset, chain, cfg, mapname, results, truths, [s for s in seeds if s != TUNING_SEED])
                    entry = {"dataset": dataset, "chain": chain, "config": cfg + prop_tag, "map": mapname or "coherent",
                             "propagator": args.propagator, "threshold": sbs["threshold"].to_dict()}
                    for split in ("tuning", "test"):
                        if split in sbs:
                            entry[split] = sbs[split].to_dict()
                            entry[split]["provenance"] = sbs[split].provenance.to_dict()
                    if chain == "passive":
                        entry["diagnostic_after_first_ais_report"] = after_first_report_diagnostic(
                            cfg, mapname, results, truths, reports, [s for s in seeds if s != TUNING_SEED])
                        entry["frames_before_first_ais_report"] = {
                            str(s): int(np.sum(results[(cfg, s, None, ())]["maps"][mapname].times
                                               < max(reports[s].for_vessel(v).t[0] for v in reports[s].vessels))) for s in seeds}
                    if chain == "echo":
                        entry["diagnostic_frame_argmax_to_uuv"] = argmax_diagnostic(cfg, results, truths, seeds)
                        canc = {split: [results[(cfg, s, None, ())]["diag"].get("cancellation") for s in sd if (cfg, s, None, ()) in results]
                                for split, sd in (("tuning", (TUNING_SEED,)), ("test", TEST_SEEDS))}
                        entry["cancellation_diagnostic"] = canc
                        entry["illuminator_track_error_m"] = {
                            split: [results[(cfg, s, None, ())]["ship_error_m"] for s in sd if (cfg, s, None, ()) in results]
                            for split, sd in (("tuning", (TUNING_SEED,)), ("test", TEST_SEEDS))}
                        src = cfg.split("/")[0]
                        fr = []
                        for s in seeds:
                            others = [o for o in radiators(truths[s]) if o != src]
                            for e in results[(cfg, s, None, ())]["misses"]:
                                e = dict(e, seed=s)
                                if others:
                                    e["nearest_other_radiator_m"] = min(
                                        float(np.hypot(*(truths[s].tracks[o].position_at(e["t"])[0] - e["xy"]))) for o in others)
                                fr.append(e)
                        misses[cfg] = {"frames": fr, "frames_total": int(sum(results[(cfg, s, None, ())]["miss_frames"] for s in seeds)),
                                       "counts": {c: sum(1 for e in fr if e["class"] == c) for c in MISS_CLASSES}}
                        near = [e["nearest_other_radiator_m"] for e in fr if "nearest_other_radiator_m" in e]
                        entry["misses"] = {"counts": misses[cfg]["counts"], "frames_total": misses[cfg]["frames_total"],
                                           "within_25m_of_another_radiator": int(sum(1 for d in near if d <= 25.0)) if near else None,
                                           "note": "frames with an illuminator whose highest cell is more than 20 m from every UUV, all seeds"}
                    results_json["entries"][key] = entry
                    (out / f"scoreboard_{dataset}_{chain}_{name}.json").write_text(json.dumps(entry, indent=2))
                    plot_scoreboard_pair(sbs, out / f"scoreboard_{dataset}_{chain}_{name}.png", key)
                    if "test" in sbs:
                        plot_map(sbs["test"], truth0, obs0.environment, out / f"scoreboard_{dataset}_{chain}_{name}_map.png", obs0.geometry,
                                 title=f"{key}: test set Pd")
                        s1 = [s for s in seeds if s != TUNING_SEED][0]
                        plot_frame(key, results[(cfg, s1, None, ())]["maps"][mapname], sbs["threshold"].T, truths[s1], reports[s1], chain,
                                   cfg.split("/")[0] if chain == "echo" else None, obs0.environment, obs0.geometry,
                                   out / f"scoreboard_{dataset}_{chain}_{name}_frame.png")
                    for split in ("tuning", "test"):
                        if split in sbs:
                            sb = sbs[split]
                            say(f"[{key}] {split}: T {sb.threshold.T:.4g}, FA {sb.false_alarm_rate:.2f}/frame, frames with a false alarm "
                                f"{100 * sb.false_alarm_frame_rate:.0f} % (threshold set for {100 * sb.threshold.pfa:.0f} %), Pd " +
                                ", ".join(f"{o.id} {o.pd:.2f}" for o in sb.per_object) +
                                (f", pilings resolved {sb.separation['resolved_per_run']}" if sb.separation else ""))
                    res_path.write_text(json.dumps(results_json, indent=2))
            if misses:
                plot_misses(dataset, misses, obs0.environment, obs0.geometry, truth0, out / f"misses_{dataset}{prop_tag}.png")

    # Criteria and the summary figure from everything in results.json.
    ent = results_json["entries"]
    verdicts = []
    for c in CRITERIA:
        key = f"{c['dataset']}/{c['chain']}/{c['config']}"
        e = ent.get(key, {}).get("test")
        if e is None:
            verdicts.append({**c, "result": "not run"})
            continue
        rows = {r["id"]: r for r in e["per_object"]}
        if "pd_min" in c:
            pds = {o: rows[o]["pd"] for o in c["objects"]}
            ok = all(v >= c["pd_min"] for v in pds.values())
            verdicts.append({**c, "pd": pds, "result": "PASS" if ok else "FAIL"})
        else:
            statics = [r for r in e["per_object"] if not r["moving"]]
            per_seed = [int(sum(r["full_record_hits"][i] for r in statics)) for i in range(len(e["seeds"]))]
            ok = float(np.median(per_seed)) >= c["static_min"]
            verdicts.append({**c, "static_detected_per_seed": per_seed, "median": float(np.median(per_seed)),
                             "static": {r["id"]: r["full_record_detected"] for r in statics}, "result": "PASS" if ok else "FAIL"})
        say(f"criterion {c['id']}: {verdicts[-1]['result']}")
    results_json["verdicts"] = verdicts
    summary = []
    for key, e in ent.items():
        summary.append({"dataset": e["dataset"], "chain": e["chain"] if e["chain"] == "passive" else f"echo, {e['config'].split('/')[0]} as source",
                        "config": e["config"].split("/")[-1] + ("" if e["chain"] == "echo" else f" / {e['map']} map"),
                        "pd": {s: {r["id"]: r["pd"] for r in e[s]["per_object"]} for s in ("tuning", "test") if s in e}})
    if summary:
        plot_pd_summary(summary, out / "pd_summary.png")
    res_path.write_text(json.dumps(results_json, indent=2))
    say(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark and accuracy harness for the harbor forward model and processing chain.

Every performance stage is measured twice with this script: wall time, and accuracy against
the outputs of a reference run saved earlier (``--save-reference``, run once with the
unmodified code).  The fixed cases are

FDTD (the fine bistatic scenario, 0.5 m grid, 6 kHz):
    ``fdtd_ferry_2s``, ``fdtd_ferry_10s``          the moving ferry alone
    ``fdtd_ferry_uuv_2s``, ``fdtd_ferry_uuv_10s``  the same with the 1 m x 3 m UUV body
    ``fdtd_multistatic_2s``                        the multistatic scene, ferry with all 10 bodies

Processing (the fine bistatic dataset and the stage 0a 193 m moving case):
    ``proc_oracle``, ``proc_eca_oracle_doppler``, ``proc_blind_doppler_smooth4``
                                                   ``run_bistatic`` as in ``harbor_bistatic_ablation.py``
    ``eca_193_moving_fixed``, ``eca_193_moving_doppler_smooth4``
                                                   ``eca_stage`` alone (fixed lags; per sample delays with a 4 s track fit)

Usage::

    python scripts/bench_harbor.py --tag baseline --cuda --save-reference
    python scripts/bench_harbor.py --tag stage1_fused64 --cuda --kernel fused
    python scripts/bench_harbor.py --tag stage1_fused32 --cuda --kernel fused --precision float32 \
        --ref-dir output/performance/runs/stage1_fused64

Timings and comparison metrics go to ``output/performance/<tag>.json`` (committed); the outputs
of a run go to ``output/performance/runs/<tag>/`` or, with ``--save-reference``, to
``output/performance/reference/`` (both gitignored through ``*.npz``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.water import HarborDataset, HarborScenarioConfig, build_environment, build_layout, build_track  # noqa: E402
from acoustic_sim.water.bistatic import BistaticConfig, ReplicaModel, eca_stage, run_bistatic, ship_positions, stft  # noqa: E402
try:
    from acoustic_sim.water.bistatic import _xp as _bistatic_xp  # noqa: E402
except ImportError:  # the original code
    _bistatic_xp = None
from acoustic_sim.water.dataset import assemble_dataset, make_scatterer, render_vessel_gathers, vessel_injection  # noqa: E402
from acoustic_sim.water.dispersion import for_scenario  # noqa: E402

try:  # the sensing contract API (propagator, diagnostics)
    from acoustic_sim.water.bistatic import Diagnostics as _Diagnostics  # noqa: E402
    from acoustic_sim.water.diagnostics import SimulatorAwarePropagator  # noqa: E402

    def replica_options(env, fdtd_cfg) -> dict:
        return {"propagator": SimulatorAwarePropagator.from_scenario(env, fdtd_cfg)}

    def diagnostics_option(d) -> dict:
        return {"diagnostics": d}
except ImportError:  # older commits (dispersion=, truth=)
    from acoustic_sim.water.bistatic import Truth as _Diagnostics  # noqa: E402

    def replica_options(env, fdtd_cfg) -> dict:
        return {"dispersion": for_scenario(env, fdtd_cfg)}

    def diagnostics_option(d) -> dict:
        return {"truth": d}
from acoustic_sim.water.fdtd_runner import HarborFDTDSolver, simulate_source  # noqa: E402
from acoustic_sim.water.noise import generate_noise  # noqa: E402
from acoustic_sim.water.signatures import f_max_for_grid  # noqa: E402
from acoustic_sim.water.vessels import VesselConfig  # noqa: E402
from acoustic_sim.sources import CustomTrajectorySource  # noqa: E402

BISTATIC_SCENARIO = ROOT / "examples" / "harbor_bistatic.scenario.json"
MULTISTATIC_SCENARIO = ROOT / "examples" / "harbor_multistatic.scenario.json"
DATA_ROOT = Path(__import__("os").environ.get("BENCH_DATA_ROOT", ROOT))      # where output/harbor_bistatic and output/harbor_cancel live
FINE_DATASET = DATA_ROOT / "output" / "harbor_bistatic" / "fine" / "dataset.npz"
CANCEL_CACHE = DATA_ROOT / "output" / "harbor_cancel" / "cache"

FDTD_CASES = {
    # name: (scenario, total_time, bodies)
    "fdtd_ferry_2s": (BISTATIC_SCENARIO, 2.0, "none"),
    "fdtd_ferry_10s": (BISTATIC_SCENARIO, 10.0, "none"),
    "fdtd_ferry_uuv_2s": (BISTATIC_SCENARIO, 2.0, "uuv1"),
    "fdtd_ferry_uuv_10s": (BISTATIC_SCENARIO, 10.0, "uuv1"),
    "fdtd_multistatic_2s": (MULTISTATIC_SCENARIO, 2.0, "all"),
}
# (run with bodies, run without): their difference is the scattered field.
SCATTER_PAIRS = [
    ("fdtd_ferry_uuv_2s", "fdtd_ferry_2s"),
    ("fdtd_ferry_uuv_10s", "fdtd_ferry_10s"),
    ("fdtd_multistatic_2s", "fdtd_ferry_2s"),
]
PROC_CASES = {
    # name: (ablation configuration, eca_doppler, track_smooth_s)
    "proc_oracle": ("oracle", False, 0.0),
    "proc_eca_oracle_doppler": ("eca_oracle", True, 0.0),
    "proc_blind_doppler_smooth4": ("blind", True, 4.0),
}
ECA_CASES = {
    # name: (eca_doppler, track_smooth_s)
    "eca_193_moving_fixed": (False, 0.0),
    "eca_193_moving_doppler_smooth4": (True, 4.0),
}
ABLATION_CONFIGS = {
    # name: (direct path in the traces, reference, exact track)   (examples/harbor_bistatic_ablation.py)
    "oracle": (False, "oracle", True),
    "eca_oracle": (True, "oracle", True),
    "eca_data": (True, "data", True),
    "blind": (True, "data", False),
}
ALL_CASES = list(FDTD_CASES) + list(PROC_CASES) + list(ECA_CASES)

# Criteria fixed in the task description.
CRITERIA = {
    "fused64_vs_reference_rel_l2": 1e-12,
    "fused32_direct_rel_l2_median": 1e-4,
    "fused32_direct_rel_l2_max": 1e-3,
    "fused32_scattered_rel_rms": 1e-3,
    "fused32_eca_depth_db": 0.5,
    "fused32_estimate_frames_within_one_cell": 0.95,
    "masks_rel_l2": 1e-10,
    "batched_rel_l2": 1e-12,
    "processing_rel_l2": 1e-10,
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_example(name: str):
    """Import ``examples/<name>.py`` as a module (its ``main`` is not run)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gpu_info(use_cuda: bool) -> dict:
    info = {"requested": use_cuda, "available": False}
    if not use_cuda:
        return info
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        free, total = cp.cuda.runtime.memGetInfo()
        info.update(available=True, name=props["name"].decode(), cupy=cp.__version__,
                    cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), memory_total_mb=total / 2 ** 20,
                    memory_free_mb=free / 2 ** 20, compute_capability=f"{props['major']}.{props['minor']}")
    except Exception as e:  # pragma: no cover
        info["error"] = repr(e)
    return info


def sync_gpu() -> None:
    try:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def apply_solver_options(hcfg, args) -> None:
    """Set the solver options of this stage on a ``HarborFDTDConfig`` (only the fields that exist)."""
    names = {f.name for f in fields(hcfg)}
    for opt in ("precision", "kernel", "boundary", "body_masks", "cpml_width"):
        val = getattr(args, opt, None)
        if val is not None:
            if opt not in names:
                raise SystemExit(f"--{opt.replace('_', '-')} needs HarborFDTDConfig.{opt}, which this checkout lacks")
            setattr(hcfg, opt, val)


def rel_l2_per_channel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relative L2 error of every row of *a* against the reference row of *b* (rows with a zero reference are dropped)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(b, axis=-1)
    num = np.linalg.norm(a - b, axis=-1)
    keep = den > 0
    return num[keep] / den[keep]


def summarize_rel(err: np.ndarray) -> dict:
    err = np.asarray(err, dtype=float)
    if err.size == 0:
        return {"median": None, "max": None, "n": 0}
    return {"median": float(np.median(err)), "max": float(np.max(err)), "n": int(err.size)}


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / ||b|| over the whole array."""
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - b) / max(np.linalg.norm(b), 1e-300))


# ----------------------------------------------------------------------
# FDTD cases
# ----------------------------------------------------------------------

def run_fdtd_case(name: str, args) -> tuple[dict, dict]:
    scenario, total_time, bodies = FDTD_CASES[name]
    cfg = HarborScenarioConfig.load(scenario)
    cfg.fdtd.total_time = total_time
    cfg.fdtd.use_cuda = args.cuda
    apply_solver_options(cfg.fdtd, args)
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    tracks = {v.id: build_track(v, total_time, env) for v in cfg.vessels}
    ferry = next(v for v in cfg.vessels if v.id == "ferry")
    signal = vessel_injection(ferry, tracks["ferry"], env, cfg)
    tr = tracks["ferry"]
    source = CustomTrajectorySource(times=tr.t, positions=tr.xy, signal=signal)
    if bodies == "none":
        scatterers = []
    elif bodies == "uuv1":
        scatterers = [make_scatterer(next(v for v in cfg.vessels if v.id == "uuv1"), tracks["uuv1"])]
    else:
        scatterers = [make_scatterer(v, tracks[v.id]) for v in cfg.vessels if v.is_body]
    sync_gpu()
    t0 = time.perf_counter()
    solver = HarborFDTDSolver(env, cfg.fdtd, source, layout.sample_points, scatterers=scatterers)
    out = solver.run(verbose=False)
    sync_gpu()
    wall = time.perf_counter() - t0
    traces = np.asarray(layout.collapse(out["traces"].T).T, dtype=np.float64)
    info = {
        "kind": "fdtd", "scenario": scenario.name, "total_time_s": total_time, "bodies": bodies,
        "n_bodies": len(scatterers), "n_steps": int(out["n_steps"]), "grid": [int(env.ny), int(env.nx)],
        "n_sample_points": int(layout.n_samples), "n_channels": int(layout.n_receivers),
        "wall_s": wall, "ms_per_step": 1e3 * wall / out["n_steps"],
        "solver": {k: getattr(solver, k) for k in ("kernel", "precision") if hasattr(solver, k)},
    }
    return {"traces": traces}, info


# ----------------------------------------------------------------------
# Processing cases (the ablation configurations on the fine dataset)
# ----------------------------------------------------------------------

def rerender_fine_dataset(args, cache_dir: Path) -> tuple[HarborDataset, float]:
    """The fine bistatic dataset rebuilt with the current solver options (same scenario, tracks and
    noise seed as ``examples/harbor_bistatic_dataset.py``), so the processing criteria can be
    evaluated on float32 or CPML or batched forward modelling."""
    cfg = HarborScenarioConfig.load(BISTATIC_SCENARIO)
    cfg.fdtd.use_cuda = args.cuda
    apply_solver_options(cfg.fdtd, args)
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    sync_gpu()
    t0 = time.perf_counter()
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=cache_dir)
    sync_gpu()
    wall = time.perf_counter() - t0
    ds = assemble_dataset(cfg, layout, tracks, gathers, keep_components=True)
    return ds, wall

def bistatic_options(args) -> dict:
    """Extra ``BistaticConfig`` fields of the processing stage (only those that exist)."""
    names = {f.name for f in fields(BistaticConfig)}
    out = {}
    for opt, field in (("proc_cuda", "use_cuda"), ("impl", "impl")):
        val = getattr(args, opt, None)
        if val is not None:
            if field not in names:
                raise SystemExit(f"--{opt.replace('_', '-')} needs BistaticConfig.{field}, which this checkout lacks")
            out[field] = val
    return out


def run_proc_case(name: str, ds: HarborDataset, env, layout, f_max: float, args) -> tuple[dict, dict]:
    config, doppler, smooth = PROC_CASES[name]
    direct_in, reference, exact = ABLATION_CONFIGS[config]
    cfg = ds.cfg
    comps = ds.components
    gain = ds.gain[:, None]
    direct = gain * comps["direct:ferry"].astype(float)
    echo = gain * comps["scattered:ferry>uuv1"].astype(float)
    noise = comps["noise"].astype(float)
    traces = echo + noise + (direct if direct_in else 0.0)
    ferry = next(v for v in cfg.vessels if v.id == "ferry")
    oracle = ferry.level_pa * vessel_injection(ferry, ds.tracks["ferry"], env, cfg) if reference == "oracle" else None
    bc = BistaticConfig(win_s=0.5, hop_s=0.25, f_lo=100.0, bin_stride=2, grid_dx=2.0, reference=reference,
                        removal="eca" if direct_in else "none", method="coherent", coherent_norm="snr",
                        exclusion_radius=25.0, **replica_options(env, cfg.fdtd), eca_doppler=doppler,
                        track_smooth_s=smooth if doppler else 0.0, **bistatic_options(args))
    sp = stft(traces, layout, ds.fs, bc, f_max)
    truth = _Diagnostics(ship_xy=ds.tracks["ferry"].position_at(sp.times) if exact else None,
                  components={"direct:ferry": direct, "scattered:ferry>uuv1": echo} if direct_in else None)
    sync_gpu()
    t0 = time.perf_counter()
    run = run_bistatic(traces, layout, ds.fs, env, ds.ais, bc, f_max, oracle_signal=oracle, **diagnostics_option(truth))
    sync_gpu()
    wall = time.perf_counter() - t0
    B, grid, times = run.scan.B, run.grid, run.sp.times
    est = grid[np.argmax(B, axis=0)]
    uuv = ds.tracks["uuv1"].position_at(times)
    err = np.hypot(*(est - uuv).T)
    live = run.sp.weights[layout.slices["cable1"]] > 0
    outputs = {"B": B, "Z": run.Z, "grid": grid, "times": times, "est": est, "ship_refined": run.ship.refined,
               "ref": run.ref, "grid_dx": np.float64(bc.grid_dx)}
    if run.eca is not None:
        outputs["eca_removed_db"] = run.eca.removed_db
        outputs["eca_echo_lost_db"] = run.eca.echo_lost_db
    if run.reference is not None:
        outputs["waveform"] = run.reference
    info = {
        "kind": "processing", "config": config, "eca_doppler": doppler, "track_smooth_s": smooth, "wall_s": wall,
        "frames": int(len(times)), "grid_cells": int(len(grid)), "bins": int(len(sp.freqs)), "channels": int(layout.n_receivers),
        "median_error_m": float(np.median(err)), "within_5m": float(np.mean(err <= 5.0)),
        "eca_direct_depth_db": float(np.median(run.eca.removed_db[layout.slices["cable1"]][live])) if run.eca else None,
        "eca_echo_loss_db": float(np.median(run.eca.echo_lost_db[layout.slices["cable1"]][live])) if run.eca else None,
        "bistatic_options": bistatic_options(args),
    }
    return outputs, info


# ----------------------------------------------------------------------
# ECA alone on the stage 0a 193 m moving case
# ----------------------------------------------------------------------

def run_eca_case(name: str, args) -> tuple[dict, dict]:
    from scipy.signal import butter, sosfiltfilt

    doppler, smooth = ECA_CASES[name]
    cd = load_example("harbor_cancellation_depth")
    cfg = HarborScenarioConfig.load(BISTATIC_SCENARIO)
    cfg.fdtd.total_time = 10.0
    cfg.fdtd.use_cuda = args.cuda
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps
    f_max = f_max_for_grid(env.dx)
    ferry = next(v for v in cfg.vessels if v.kind == "ship")
    path_len = cd.SPEED * cfg.fdtd.total_time
    edge = cfg.fdtd.damping_width * env.dx + 15.0
    p = cd.pick_position(env, layout, 200.0, path_len, edge=edge)
    v = VesselConfig("src_moving_200", "ship", [p.tolist(), (p + [path_len, 0.0]).tolist()], speed=cd.SPEED,
                     level_pa=ferry.level_pa, ais=True, seed=ferry.seed)
    tr = build_track(v, cfg.fdtd.total_time, env)
    sig = vessel_injection(v, tr, env, cfg)
    apply_solver_options(cfg.fdtd, args) if args.rerender else None
    cache = (Path(args.out) / f"cache_{args.tag}") if args.rerender else CANCEL_CACHE
    sync_gpu()
    t0 = time.perf_counter()
    g = simulate_source(env, layout, cfg.fdtd, sig, trajectory=(tr.t, tr.xy), cache_dir=cache)
    sync_gpu()
    t_gather = time.perf_counter() - t0
    direct = v.level_pa * g.traces.astype(np.float64)
    noise = generate_noise(layout, fs, n, cfg.noise, np.random.default_rng(cfg.seed))
    truth_meas = noise.gain[:, None] * direct
    traces = noise.apply(direct).astype(np.float64)
    ais = cd.ais_from_start(v.id, tr, "moving", cfg.ais, np.random.default_rng(sum(map(ord, v.id))))
    bc = BistaticConfig(win_s=0.5, hop_s=0.25, f_lo=100.0, removal="eca", reference="data",
                        **replica_options(env, cfg.fdtd), eca_doppler=doppler, track_smooth_s=smooth,
                        **bistatic_options(args))
    sp = stft(traces, layout, fs, bc, f_max)
    model = ReplicaModel(layout, bc.c, **({"propagator": bc.propagator} if hasattr(bc, "propagator") else {"dispersion": bc.dispersion}), **({"xp": _bistatic_xp(bc)} if _bistatic_xp else {}))
    model.weights = sp.weights
    sync_gpu()
    t0 = time.perf_counter()
    ship = ship_positions(sp, ais, model, bc, vessel=v.id)
    sync_gpu()
    t_ship = time.perf_counter() - t0
    t0 = time.perf_counter()
    outs, waveform = eca_stage(traces, layout, fs, env, sp, ship, bc, None, [truth_meas])
    sync_gpu()
    t_eca = time.perf_counter() - t0
    sos = butter(4, [bc.f_lo, 0.95 * f_max], btype="bandpass", fs=fs, output="sos")
    e0 = cd.band_energy(truth_meas, sos)
    e1 = cd.band_energy(outs[1], sos)
    depth = 10 * np.log10(np.maximum(e0, 1e-30) / np.maximum(e1, 1e-30))
    live = sp.weights > 0
    cab = layout.slices[layout.cables[0].name]
    outputs = {"clean": outs[0], "clean_direct": outs[1], "waveform": waveform, "depth_db": depth,
               "ship_refined": ship.refined, "live": live}
    info = {
        "kind": "eca", "eca_doppler": doppler, "track_smooth_s": smooth, "range_min_m": float(cd.nearest_channel_range(layout, tr.xy).min()),
        "gather_s": t_gather, "gather_from_cache": t_gather < 20.0, "rerendered": bool(args.rerender),
        "ship_positions_s": t_ship, "eca_stage_s": t_eca,
        "wall_s": t_eca, "cable_median_depth_db": float(np.median(depth[cab][live[cab]])),
        "position_error_median_m": float(np.median(np.hypot(*(ship.refined - tr.position_at(sp.times)).T))),
        "bistatic_options": bistatic_options(args),
    }
    return outputs, info


# ----------------------------------------------------------------------
# Comparison against a saved run
# ----------------------------------------------------------------------

def load_outputs(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def compare_case(name: str, out: dict, ref: dict, out_all: dict, ref_all: dict) -> dict:
    """Accuracy of *out* against *ref* in the metrics the stage criteria use."""
    cmp: dict = {}
    if name in FDTD_CASES:
        err = rel_l2_per_channel(out["traces"], ref["traces"])
        cmp["direct_rel_l2"] = summarize_rel(err)
        cmp["rel_rms"] = rel_rms(out["traces"], ref["traces"])
        for with_b, without in SCATTER_PAIRS:
            if name == with_b and without in out_all and without in ref_all:
                sc_out = out["traces"] - out_all[without]["traces"]
                sc_ref = ref["traces"] - ref_all[without]["traces"]
                cmp["scattered_rel_rms"] = rel_rms(sc_out, sc_ref)
                cmp["scattered_rel_l2"] = summarize_rel(rel_l2_per_channel(sc_out, sc_ref))
                cmp["scattered_to_direct_rms"] = float(np.linalg.norm(sc_ref) / np.linalg.norm(ref["traces"]))
    elif name in PROC_CASES:
        cmp["B_rel_rms"] = rel_rms(out["B"], ref["B"])
        cmp["B_rel_l2_per_frame"] = summarize_rel(rel_l2_per_channel(out["B"].T, ref["B"].T))
        if "Z" in out and "Z" in ref and out["Z"].shape == ref["Z"].shape:
            cmp["Z_rel_rms"] = rel_rms(out["Z"], ref["Z"])
        cmp["ref_rel_rms"] = rel_rms(out["ref"], ref["ref"])
        d = np.hypot(*(out["est"] - ref["est"]).T)
        dx = float(ref["grid_dx"])
        cmp["estimate"] = {"identical": float(np.mean(d == 0.0)), "within_one_cell": float(np.mean(d <= 1.0001 * dx)),
                           "within_one_cell_diagonal": float(np.mean(d <= 1.0001 * np.sqrt(2.0) * dx)),
                           "max_shift_m": float(d.max()), "grid_dx": dx}
        cmp["ship_refined_max_shift_m"] = float(np.hypot(*(out["ship_refined"] - ref["ship_refined"]).T).max())
        if "eca_removed_db" in out and "eca_removed_db" in ref:
            da = out["eca_removed_db"] - ref["eca_removed_db"]
            ok = np.isfinite(da)
            cmp["eca_depth_db"] = {"median_abs_diff": float(np.median(np.abs(da[ok]))), "max_abs_diff": float(np.abs(da[ok]).max()),
                                   "median_depth_diff": float(np.median(out["eca_removed_db"][ok]) - np.median(ref["eca_removed_db"][ok]))}
        if "waveform" in out and "waveform" in ref:
            cmp["waveform_rel_rms"] = rel_rms(out["waveform"], ref["waveform"])
    elif name in ECA_CASES:
        cmp["clean_rel_l2"] = summarize_rel(rel_l2_per_channel(out["clean"], ref["clean"]))
        cmp["clean_direct_rel_l2"] = summarize_rel(rel_l2_per_channel(out["clean_direct"], ref["clean_direct"]))
        cmp["waveform_rel_rms"] = rel_rms(out["waveform"], ref["waveform"])
        live = ref["live"].astype(bool)
        da = (out["depth_db"] - ref["depth_db"])[live]
        cmp["depth_db"] = {"median_abs_diff": float(np.median(np.abs(da))), "max_abs_diff": float(np.abs(da).max()),
                           "median_depth_diff": float(np.median(out["depth_db"][live]) - np.median(ref["depth_db"][live]))}
        cmp["ship_refined_max_shift_m"] = float(np.hypot(*(out["ship_refined"] - ref["ship_refined"]).T).max())
    return cmp


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="name of the run (output/performance/<tag>.json)")
    ap.add_argument("--cases", default="all", help="comma separated case names, or 'all', 'fdtd', 'proc', 'eca'")
    ap.add_argument("--cuda", action="store_true", help="FDTD on the GPU")
    ap.add_argument("--out", default=str(ROOT / "output" / "performance"))
    ap.add_argument("--save-reference", action="store_true", help="save the outputs as the reference (output/performance/reference)")
    ap.add_argument("--ref-dir", default=None, help="compare against the outputs in this directory (default: the reference)")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the FDTD cases and keep the best wall time")
    ap.add_argument("--rerender", action="store_true",
                    help="rebuild the fine dataset and the stage 0a gather with the current solver options before the processing cases")
    # Solver options of the later stages (passed through to HarborFDTDConfig when the field exists).
    ap.add_argument("--precision", default=None, choices=[None, "float64", "float32"])
    ap.add_argument("--kernel", default=None, choices=[None, "reference", "fused", "auto"])
    ap.add_argument("--boundary", default=None, choices=[None, "sponge", "cpml"])
    ap.add_argument("--cpml-width", dest="cpml_width", type=int, default=None)
    ap.add_argument("--body-masks", dest="body_masks", default=None, choices=[None, "host", "device", "auto"])
    # Processing options (BistaticConfig fields of stage 5).
    ap.add_argument("--proc-cuda", dest="proc_cuda", action="store_const", const=True, default=None, help="processing chain on the GPU")
    ap.add_argument("--impl", default=None, choices=[None, "reference", "vectorized"], help="processing implementation")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / ("reference" if args.save_reference else f"runs/{args.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(args.ref_dir) if args.ref_dir else out_root / "reference"

    cases: list[str] = []
    for sel in args.cases.split(","):
        if not sel:
            continue
        if sel == "all":
            cases += ALL_CASES
        elif sel in ("fdtd", "proc", "eca"):
            cases += [c for c in ALL_CASES if c.startswith(sel)]
        elif sel in ALL_CASES:
            cases.append(sel)
        else:
            raise SystemExit(f"unknown case {sel!r}; choose from {ALL_CASES} or the groups fdtd, proc, eca, all")

    json_path = out_root / f"{args.tag}.json"
    previous = json.loads(json_path.read_text()) if json_path.exists() else None
    results = {
        "tag": args.tag, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": _git_rev(),
        "host": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
                 "cpu_count": __import__("os").cpu_count()},
        "gpu": gpu_info(args.cuda),
        "options": {k: v for k, v in vars(args).items() if k not in ("tag", "cases", "out", "save_reference", "ref_dir", "no_compare")},
        "criteria": CRITERIA,
        "reference_dir": None if args.save_reference else str(ref_dir),
        "cases": {} if previous is None else previous.get("cases", {}),     # a rerun of some cases keeps the others
    }

    def say(msg: str) -> None:
        print(msg, flush=True)

    say(f"[bench] tag {args.tag}, GPU: {results['gpu'].get('name', 'none')}, cases: {', '.join(cases)}")
    outputs_all: dict[str, dict] = {}
    refs_all: dict[str, dict] = {}
    ds = env_ds = layout_ds = f_max_ds = None
    for name in cases:
        if name in FDTD_CASES:
            best_info, best_out = None, None
            for _ in range(max(1, args.repeat)):
                out, info = run_fdtd_case(name, args)
                if best_info is None or info["wall_s"] < best_info["wall_s"]:
                    best_info, best_out = info, out
            out, info = best_out, best_info
        elif name in PROC_CASES:
            if ds is None:
                if args.rerender:
                    ds, t_render = rerender_fine_dataset(args, out_root / f"cache_{args.tag}")
                    results["rerender"] = {"fine_dataset_render_s": t_render}
                    say(f"[bench] fine dataset rerendered with the current solver options in {t_render:.1f} s")
                elif not (FINE_DATASET.exists() or FINE_DATASET.with_name("dataset.observation.npz").exists()):
                    say(f"[bench] {FINE_DATASET} missing, skipping {name}")
                    continue
                else:
                    ds = HarborDataset.load(FINE_DATASET)
                env_ds = build_environment(ds.cfg)
                layout_ds = build_layout(ds.cfg, env_ds)
                f_max_ds = f_max_for_grid(env_ds.dx)
            out, info = run_proc_case(name, ds, env_ds, layout_ds, f_max_ds, args)
            info["rerendered"] = bool(args.rerender)
        else:
            out, info = run_eca_case(name, args)
        np.savez(run_dir / f"{name}.npz", **{k: v for k, v in out.items() if v is not None})
        outputs_all[name] = out
        if not args.save_reference and not args.no_compare:
            ref = load_outputs(ref_dir / f"{name}.npz")
            if ref is None:
                info["compare"] = {"error": f"no reference output {ref_dir / (name + '.npz')}"}
            else:
                refs_all[name] = ref
                # The scattered field pairs need the run without the body; load it from disk if it was not run now.
                for with_b, without in SCATTER_PAIRS:
                    if name == with_b and without not in outputs_all:
                        o = load_outputs(run_dir / f"{without}.npz")
                        r = load_outputs(ref_dir / f"{without}.npz")
                        if o is not None and r is not None:
                            outputs_all[without], refs_all[without] = o, r
                info["compare"] = compare_case(name, out, ref, outputs_all, refs_all)
        results["cases"][name] = info
        extra = ""
        if info["kind"] == "fdtd":
            extra = f"{info['n_steps']} steps, {info['ms_per_step']:.3f} ms/step, {info['n_bodies']} bodies"
        elif info["kind"] == "processing":
            extra = f"median err {info['median_error_m']:.1f} m, within 5 m {100 * info['within_5m']:.0f} %" + (
                f", ECA {info['eca_direct_depth_db']:.1f} dB" if info["eca_direct_depth_db"] is not None else "")
        else:
            extra = f"depth {info['cable_median_depth_db']:.1f} dB, ship_positions {info['ship_positions_s']:.1f} s"
        cmp = info.get("compare")
        cmp_txt = ""
        if cmp and "error" not in cmp:
            key = next((k for k in ("direct_rel_l2", "clean_rel_l2", "B_rel_rms") if k in cmp), None)
            if key:
                val = cmp[key]
                cmp_txt = f", vs ref {key} " + (f"median {val['median']:.2e} max {val['max']:.2e}" if isinstance(val, dict) else f"{val:.2e}")
            if "scattered_rel_rms" in cmp:
                cmp_txt += f", scattered rel rms {cmp['scattered_rel_rms']:.2e}"
            if "estimate" in cmp:
                cmp_txt += f", estimates within one cell {100 * cmp['estimate']['within_one_cell']:.0f} %"
            if "eca_depth_db" in cmp:
                cmp_txt += f", ECA depth diff {cmp['eca_depth_db']['median_depth_diff']:+.2f} dB"
            if "depth_db" in cmp:
                cmp_txt += f", depth diff {cmp['depth_db']['median_depth_diff']:+.2f} dB"
        say(f"[bench] {name:32s} {info['wall_s']:9.2f} s  {extra}{cmp_txt}")
        json_path.write_text(json.dumps(results, indent=2, default=_json_default))
    total = sum(c["wall_s"] for c in results["cases"].values())
    results["total_wall_s"] = total
    json_path.write_text(json.dumps(results, indent=2, default=_json_default))
    say(f"[bench] total {total:.1f} s, wrote {json_path} and {run_dir}")


def _git_rev() -> str | None:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o)}")


if __name__ == "__main__":
    main()

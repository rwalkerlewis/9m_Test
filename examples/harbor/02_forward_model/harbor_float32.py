#!/usr/bin/env python3
"""The FDTD with the k space scheme in float32 against float64, on every recording's scenario (prompt ``float32``,
section 1).

For each recording (one per geometry: ``simple``, ``bistatic``, which ``multistatic`` shares, ``multistatic_b``, the 5 m
``harbor``) the propagation error pulse of ``harbor_propagation_error.py`` runs twice, in float64 and in float32, with
nothing else changed and no cache.  Criterion, fixed before the run: the direct arrival error changes by at most
0.02 ms at every hydrophone; the direct traces agree within 1e-3 relative at the worst sample and 1e-4 at the median
(per receiver the largest difference over the trace's peak; the worst and the median over the receivers).  The scattered
component's criterion (within 1e-3 of its rms) needs the body gathers in both precisions, that is the recordings
rebuilt: ``harbor_float32_scattered.py``.  ``per_step_s`` here is the wall time of the whole measurement over its steps.
The results go to ``fdtd.json`` (CPU) or ``fdtd_gpu.json`` (GPU).

With ``--bench`` (GPU) the time per step and the memory per run are measured instead, on each recording's own grid: one
run of the first illuminator over ``--bench-seconds`` of its track, without bodies and with every body of the scenario
present (the device masks), in each precision; each case runs once to warm up (kernel compilation, FFT plans) and is
then timed over ``solver.run()`` alone, synchronised; the memory is the peak of CuPy's pool (fields, FFT work area,
masks, trace buffer), the pool and the FFT plan cache emptied before each run.  No criterion: a measurement.  Writes ``fdtd_gpu_bench.json``.

    python examples/harbor/02_forward_model/harbor_float32.py [--recordings simple ...] [--cpu] [--bench]
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
sys.path.insert(0, str(ROOT / "examples" / "harbor" / "01_datasets"))

import harbor_propagation_error as pe  # noqa: E402

from acoustic_sim.contract import Truth, contract_paths  # noqa: E402

GEOMETRIES = {"simple": (100.0, 380.0), "bistatic": (100.0, 760.0), "multistatic_b": (100.0, 760.0), "harbor": (12.0, 78.0)}
CRITERION = {"arrival_change_ms": 0.02, "worst_relative": 1e-3, "median_relative": 1e-4}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Per receiver the largest difference of *b* from *a* over *a*'s peak; their worst and median."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    peak = np.max(np.abs(a), axis=1)
    live = peak > 0
    e = np.max(np.abs(b - a), axis=1)[live] / peak[live]
    return {"worst_relative": float(np.max(e)), "median_relative": float(np.median(e)), "receivers": int(live.sum())}


def bench(rec: str, seconds: float, repeats: int = 2) -> dict:
    """Time per step and peak device memory of one run of *rec*'s first illuminator, with and without its bodies, in
    both precisions (GPU, fused kernel)."""
    import cupy as cp

    from acoustic_sim.water import HarborScenarioConfig, build_environment, build_layout
    from acoustic_sim.water.dataset import make_scatterer, vessel_injection
    from acoustic_sim.water.fdtd_runner import HarborFDTDSolver, _fused_available, _make_source
    from acoustic_sim.water.vessels import build_track

    ds = ROOT / "output" / "recordings" / rec / "dataset"
    cfg = HarborScenarioConfig.from_dict(Truth.load(contract_paths(ds.with_suffix(".npz"))[1]).scenario)
    cfg.fdtd.use_cuda = True
    full = cfg.fdtd.total_time
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    tracks = {v.id: build_track(v, full, env) for v in cfg.vessels}
    src = next(v for v in cfg.vessels if v.level_pa > 0)
    bodies = tuple(make_scatterer(v, tracks[v.id]) for v in cfg.vessels if v.is_body and v.id != src.id)
    cfg.fdtd.total_time = seconds
    tr = tracks[src.id]
    pool = cp.get_default_memory_pool()
    out = {"grid": [int(env.ny), int(env.nx)], "dx": float(env.dx), "fs": float(cfg.fdtd.fs), "steps": int(cfg.fdtd.n_steps),
           "sample_points": int(layout.n_samples), "bodies": len(bodies), "source": src.id, "cases": {}}
    for prec in ("float64", "float32"):
        cfg.fdtd.precision = prec
        assert _fused_available(cfg.fdtd)
        signal = vessel_injection(src, tr, env, cfg)
        trajectory = (tr.t, tr.xy) if len(src.waypoints) > 1 else None
        position = None if trajectory else (float(src.waypoints[0][0]), float(src.waypoints[0][1]))
        for label, scs in (("no bodies", ()), ("bodies", bodies)):
            if label == "bodies" and not scs:
                continue
            times, peaks = [], []
            for k in range(repeats + 1):
                cp.fft.config.get_plan_cache().clear()
                pool.free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                solver = HarborFDTDSolver(env, cfg.fdtd, _make_source(signal, position, trajectory), layout.sample_points,
                                          scatterers=scs)
                cp.cuda.Device().synchronize()
                t0 = time.perf_counter()
                solver.run(verbose=False)
                cp.cuda.Device().synchronize()
                dt = time.perf_counter() - t0
                peaks.append(int(pool.total_bytes()))
                del solver
                if k > 0:                                              # the first run warms up
                    times.append(dt)
            row = {"ms_per_step": 1e3 * float(np.median(times)) / cfg.fdtd.n_steps, "runs_ms_per_step": [1e3 * t / cfg.fdtd.n_steps for t in times],
                   "pool_peak_mb": max(peaks[1:]) / 2 ** 20}
            out["cases"][f"{prec}, {label}"] = row
            print(f"{rec} {prec} {label}: {row['ms_per_step']:.3f} ms per step, pool {row['pool_peak_mb']:.0f} MB", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", nargs="*", default=list(GEOMETRIES))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="output/studies/float32")
    ap.add_argument("--bench", action="store_true", help="time per step and memory per run on the GPU (no criterion)")
    ap.add_argument("--bench-seconds", type=float, default=2.0)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.bench:
        import cupy as cp

        path = out / "fdtd_gpu_bench.json"
        res = json.loads(path.read_text()) if path.exists() else {}
        res["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        res["note"] = ("one run of the first illuminator, warmed up, solver.run() timed alone (median of the timed runs); "
                       "memory: peak of CuPy's pool over the run (the desktop held about 1.3 GB of the 8 GB card besides)")
        for rec in a.recordings:
            res.setdefault("recordings", {})[rec] = bench(rec, a.bench_seconds)
            path.write_text(json.dumps(res, indent=2))
        return
    path = out / ("fdtd.json" if a.cpu else "fdtd_gpu.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    res["criterion"] = CRITERION
    res["device"] = "cpu" if a.cpu else "gpu"
    for rec in a.recordings:
        ds = ROOT / "output" / "recordings" / rec / "dataset"
        runs = {}
        for prec in ("float64", "float32"):
            t0 = time.perf_counter()
            r = pe.measure(ds, use_cuda=not a.cpu, band=GEOMETRIES[rec], save=False, precision=prec, use_cache=False, keep_traces=True)
            r["wall_s"] = time.perf_counter() - t0
            runs[prec] = r
        tr64, tr32 = runs["float64"].pop("_traces"), runs["float32"].pop("_traces")
        d_arr = [abs(h32["error_ms"] - h64["error_ms"]) for h64, h32 in zip(runs["float64"]["hydrophones"], runs["float32"]["hydrophones"])]
        cmp = compare(tr64, tr32)
        n = tr64.shape[1]
        row = {"arrival_change_ms": d_arr, "max_arrival_change_ms": float(max(d_arr)), **cmp,
               "per_step_s": {p: runs[p]["wall_s"] / n for p in runs}, "steps": n,
               "passed": bool(max(d_arr) <= CRITERION["arrival_change_ms"] and cmp["worst_relative"] <= CRITERION["worst_relative"]
                              and cmp["median_relative"] <= CRITERION["median_relative"]),
               "float64": {k: v for k, v in runs["float64"].items() if k != "note"},
               "float32": {k: v for k, v in runs["float32"].items() if k != "note"}}
        res.setdefault("recordings", {})[rec] = row
        print(f"{rec}: arrival change at most {row['max_arrival_change_ms']:.4f} ms, traces worst {cmp['worst_relative']:.2e} median "
              f"{cmp['median_relative']:.2e}, {'pass' if row['passed'] else 'FAIL'}; per step {row['per_step_s']['float64'] * 1e3:.2f} ms "
              f"(float64) / {row['per_step_s']['float32'] * 1e3:.2f} ms (float32) on the {res['device']}", flush=True)
        path.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

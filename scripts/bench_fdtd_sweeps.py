#!/usr/bin/env python3
"""Stages 2 and 3: time per step against body count, and throughput against batch size.

Both sweeps use the fine bistatic scenario grid (0.5 m, 6 kHz) with the ferry as the source.

``--bodies``  time per step for 0, 1 and 10 bodies (the multistatic scene's bodies) with the
              reference kernel, the fused kernel with host masks (``body_factor`` per step) and the
              fused kernel with device masks.
``--batch``   runs per hour for B copies of the ferry run (different signals: the ferry signature
              scaled and delayed per copy) in one :class:`HarborFDTDBatch`, B = 1, 2, 4, ...; every
              batched run is checked against its single run (1e-12 relative).

Usage::

    python scripts/bench_fdtd_sweeps.py --cuda --bodies --batch [--time 2] [--out output/performance]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.harbor import HarborScenarioConfig, build_environment, build_layout, build_track  # noqa: E402
from acoustic_sim.harbor.dataset import make_scatterer, vessel_injection  # noqa: E402
from acoustic_sim.harbor.fdtd_runner import HarborFDTDBatch, HarborFDTDSolver, batch_size  # noqa: E402
from acoustic_sim.sources import CustomTrajectorySource  # noqa: E402

MULTISTATIC = ROOT / "examples" / "harbor_multistatic.scenario.json"


def sync() -> None:
    import cupy as cp

    cp.cuda.Stream.null.synchronize()


def setup(total_time: float, cuda: bool, **kw):
    cfg = HarborScenarioConfig.load(MULTISTATIC)
    cfg.fdtd.total_time = total_time
    cfg.fdtd.use_cuda = cuda
    for k, v in kw.items():
        setattr(cfg.fdtd, k, v)
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    tracks = {v.id: build_track(v, total_time, env) for v in cfg.vessels}
    ferry = next(v for v in cfg.vessels if v.id == "ferry")
    signal = vessel_injection(ferry, tracks["ferry"], env, cfg)
    source = CustomTrajectorySource(times=tracks["ferry"].t, positions=tracks["ferry"].xy, signal=signal)
    bodies = [make_scatterer(v, tracks[v.id]) for v in cfg.vessels if v.is_body]
    return cfg, env, layout, source, bodies


def timed_run(env, cfg, source, layout, bodies):
    sync()
    t0 = time.perf_counter()
    solver = HarborFDTDSolver(env, cfg.fdtd, source, layout.sample_points, scatterers=bodies)
    out = solver.run(verbose=False)
    sync()
    return out["traces"], time.perf_counter() - t0, solver


def sweep_bodies(total_time: float, cuda: bool) -> dict:
    res = {"total_time_s": total_time, "configs": {}}
    variants = {"reference": dict(kernel="reference"), "fused_host_masks": dict(kernel="fused", body_masks="host"),
                "fused_device_masks": dict(kernel="fused", body_masks="device")}
    for name, kw in variants.items():
        cfg, env, layout, source, bodies = setup(total_time, cuda, **kw)
        for nb in (0, 1, 10):
            traces, wall, solver = timed_run(env, cfg, source, layout, bodies[:nb])
            key = f"{name}/{nb}"
            res["configs"][key] = {"kernel": solver.kernel, "body_masks": solver.body_masks, "n_bodies": nb, "wall_s": wall,
                                   "ms_per_step": 1e3 * wall / cfg.fdtd.n_steps, "n_steps": cfg.fdtd.n_steps}
            print(f"[sweep] {name:20s} {nb:2d} bodies: {wall:7.2f} s, {1e3 * wall / cfg.fdtd.n_steps:.3f} ms/step")
    return res


def sweep_batch(total_time: float, cuda: bool, max_b: int) -> dict:
    cfg, env, layout, source, bodies = setup(total_time, cuda, kernel="fused")
    n = cfg.fdtd.n_steps
    rng = np.random.default_rng(0)
    def job(i):
        sig = np.roll(source.signal, 7 * i) * (1.0 + 0.1 * i)                  # distinct signals per copy
        src = CustomTrajectorySource(times=source.times, positions=source.positions + np.array([0.0, 2.0 * i]), signal=sig)
        return src, [bodies[i % len(bodies)]] if i % 2 else []
    singles = {}
    res = {"total_time_s": total_time, "n_steps": n, "batches": {}}
    b_mem = batch_size(max_b, env, cfg.fdtd, layout.n_samples, max_batch=max_b)
    res["batch_size_from_memory"] = b_mem
    B = 1
    while B <= min(max_b, b_mem):
        jobs = [job(i) for i in range(B)]
        sync()
        t0 = time.perf_counter()
        out = HarborFDTDBatch(env, cfg.fdtd, jobs, layout.sample_points).run(verbose=False)["traces"]
        sync()
        wall = time.perf_counter() - t0
        worst = 0.0
        for i in (0, B - 1):
            if i not in singles:
                singles[i] = timed_run(env, cfg, jobs[i][0], layout, jobs[i][1])[0]
            num = np.linalg.norm(out[i] - singles[i], axis=1)
            den = np.linalg.norm(singles[i], axis=1)
            worst = max(worst, float(np.max(num[den > 0] / den[den > 0])))
        res["batches"][str(B)] = {"B": B, "wall_s": wall, "ms_per_step_per_run": 1e3 * wall / n / B,
                                  "runs_per_hour": 3600.0 * B / wall, "max_rel_l2_vs_single": worst,
                                  "pass_1e-12": bool(worst < 1e-12)}
        print(f"[sweep] batch B={B:3d}: {wall:7.2f} s, {3600 * B / wall:8.1f} runs/h ({total_time:g} s runs), max rel vs single {worst:.1e}")
        B *= 2
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--bodies", action="store_true")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--time", type=float, default=2.0)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "output" / "performance"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fdtd_sweeps.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    if args.bodies:
        results["bodies"] = sweep_bodies(args.time, args.cuda)
        path.write_text(json.dumps(results, indent=2))
    if args.batch:
        results["batch"] = sweep_batch(args.time, args.cuda, args.max_batch)
        path.write_text(json.dumps(results, indent=2))
    print(f"[sweep] wrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Is the GPU deterministic?  The two float32 faults found on the Quadro RTX 4000 as one script that runs on any card.

    python scripts/gpu_determinism.py                         # N = 20 processes, M = 20 000 batched transforms
    python scripts/gpu_determinism.py --n 40 --m 100000 --out output/studies/gpu_determinism/<card>

Needs only the repository and a CUDA card with CuPy (no recording, no FDTD cache). Three measurements:

(a) **Runs in fresh processes.** N fresh processes each run the same k space FDTD job (the 601 x 801 grid of
    ``examples/configs/harbor_bistatic.scenario.json``: its water, land mask and sponge, a 400 Hz Ricker at the
    hydrophones' centroid, the four hydrophones recorded, ``--seconds`` of simulated time, 5 s by default), once in
    float32 and once in float64 (``--n64`` processes, 5 by default: on the Quadro a float64 run takes four times as
    long and float64 was never affected), through the repository's solver (``HarborFDTDSolver``, the fused CUDA step). Each
    process's traces are compared with the median over all processes of the same precision, sample by sample (so a
    faulty first process cannot make every other one look faulty): the largest difference over the trace's peak and
    the first step at which it appears. ``--no-waits`` adds a float32 set run with the k space step
    as it was before the waits of ``FusedFDTD.step`` went in (a copy of that step without them, in this script), which
    is the condition in which the fault was found (about one fresh process in ten with a large difference).
(b) **Batched inverse FFTs.** M calls of CuPy's ``irfft2`` over a batch of two identical spectra of the 601 x 801 grid,
    in float32 and in float64, each output compared with the unbatched ``irfft2`` of the same spectrum: a call is
    *corrupted* when a member differs from it by 1e-4 of its peak or more (members of a correct batch differ from the
    unbatched result by about 1e-6 in float32, by round off in float64).
(c) **The versions** (device, compute capability, driver, CUDA runtime, cuFFT, CuPy) in the header of ``results.json``.

Criterion (fixed before the first run, written into ``results.json``): the card is *deterministic* when, in float32
and in float64, no process of (a) differs from the median of the processes by 1e-4 of the peak or more and no call of (b) is corrupted.
What the result means for the Quadro (``docs/gpu_check.md``): on the Quadro, (a) without the waits and (b) in float32
were not deterministic (about 1 process in 10, 19 calls in 100 000) and float64 always was. If another card with the
same code and the same CuPy is deterministic with ``--no-waits`` and ``--m 100000``, the faults belong to the Quadro or
its driver; if it shows them too, they belong to the code, CuPy or cuFFT.

Outputs in ``--out`` (default ``output/studies/gpu_determinism/``): ``results.json`` and ``run.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
SCENARIO = ROOT / "examples" / "configs" / "harbor_bistatic.scenario.json"
LARGE = 1e-4                                       # a difference of 1e-4 of the peak or more is large (as in the study)
CRITERION = ("deterministic when, in float32 and in float64, no process of (a) with the solver as committed differs from the median of the processes by 1e-4 of the "
             "trace's peak or more and no batched irfft2 call of (b) has a member differing from the unbatched result "
             "by 1e-4 of its peak or more")


def versions() -> dict:
    import cupy as cp
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    drv = cp.cuda.runtime.driverGetVersion()
    rt = cp.cuda.runtime.runtimeGetVersion()
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        smi = None
    return {"device": name, "compute_capability": f"{props['major']}.{props['minor']}",
            "memory_gb": round(props["totalGlobalMem"] / 2**30, 1), "driver": smi,
            "cuda_driver_api": f"{drv // 1000}.{drv % 1000 // 10}", "cuda_runtime": f"{rt // 1000}.{rt % 1000 // 10}",
            "cufft": cp.cuda.cufft.getVersion(), "cupy": cp.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "host": platform.node()}


# ---------------------------------------------------------------- (a) one k space run, in a child process

def _step_without_waits(self, p, pp, n: int):
    """``FusedFDTD.step`` for the k space scheme as it was before the waits (the condition the fault was found in)."""
    from acoustic_sim.solver import fdtd_cuda as fc
    col = n - 1 - self.col0 if n >= 1 and self.sampled_to == n - 1 else -1
    if self.cpml is not None:
        self.psi_kernel(self.grid, fc._BLOCK, (p, self.psix, self.psiy, *self.cp_prof, np.int32(self.ny), np.int32(self.nx)))
    xp = self.xp
    spec = xp.fft.rfft2(p, axes=(-2, -1))
    self.q[...] = xp.fft.irfft2(spec * self.k_kernel, s=(self.ny, self.nx), axes=(-2, -1))
    self.step_kernel(
        self.grid, fc._BLOCK,
        (p, pp, self.c2, self.sigma, self.mask, self.src_iy, self.src_ix, self.src_w,
         self.r_iy, self.r_ix, self.r_wx, self.r_wy, self.trace_buf,
         np.int32(self.n_recv), np.int32(col), np.int32(self.chunk),
         self.psix, self.psiy, self.zetax, self.zetay, *self.cp_prof, self.q,
         np.int32(n), np.int32(self.n_steps), np.int32(self.ny), np.int32(self.nx)),
    )
    if col >= 0:
        self.sampled_to = n
        self._flush()
    for hook in self.post_hooks:
        hook(pp, n)
    return pp, p


def child(precision: str, seconds: float, waits: bool, out: Path) -> None:
    from acoustic_sim.sources import StaticSource
    from acoustic_sim.water import HarborScenarioConfig, build_environment
    from acoustic_sim.water.fdtd_runner import HarborFDTDSolver

    if not waits:
        from acoustic_sim.solver import fdtd_cuda
        fdtd_cuda.FusedFDTD.step = _step_without_waits
    cfg = HarborScenarioConfig.from_dict(json.loads(SCENARIO.read_text()))
    cfg.fdtd.use_cuda = True
    cfg.fdtd.precision = precision
    cfg.fdtd.time_scheme = "kspace"
    cfg.fdtd.total_time = seconds
    env = build_environment(cfg)
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps
    t = np.arange(n) / fs
    a = (np.pi * 400.0 * (t - 1.5 / 400.0)) ** 2
    sig = (1 - 2 * a) * np.exp(-a)
    hyd = np.asarray(cfg.hydrophones, float)
    pos = hyd.mean(0)
    solver = HarborFDTDSolver(env, cfg.fdtd, StaticSource(x=float(pos[0]), y=float(pos[1]), signal=sig), hyd)
    shape = tuple(int(s) for s in env.water.shape)
    t0 = time.time()
    traces = np.asarray(solver.run(verbose=False)["traces"], dtype=np.float64)
    np.save(out, traces)
    print(json.dumps({"grid": shape, "steps": int(n), "wall_s": round(time.time() - t0, 2)}), flush=True)


def runs(precision: str, n_proc: int, seconds: float, waits: bool, log) -> dict:
    """*n_proc* fresh processes of :func:`child`, each compared with the median over all of them (sample by sample), so
    that one faulty process, the first included, shows as one process and not as all the others."""
    traces, walls, info = [], [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(n_proc):
            path = Path(tmp) / f"run{k}.npy"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child", precision, "--seconds", str(seconds),
                   "--child-out", str(path)] + ([] if waits else ["--no-waits"])
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"child {k} ({precision}) failed:\n{res.stderr[-3000:]}")
            info = json.loads(res.stdout.strip().splitlines()[-1])
            traces.append(np.load(path))
            walls.append(round(time.time() - t0, 1))
            log(f"  (a) {precision}{'' if waits else ' without waits'} process {k:2d} done in {walls[-1]} s")
    stack = np.stack(traces)
    ref = np.median(stack, axis=0)
    peak = float(np.abs(ref).max())
    rows = []
    for k, tr in enumerate(stack):
        dev = np.abs(tr - ref).max(axis=0) / peak
        d = float(dev.max())
        first = int(np.argmax(dev > 0)) if d > 0 else -1
        large_from = int(np.argmax(dev >= LARGE)) if d >= LARGE else -1
        rows.append({"process": k, "largest": d, "first_step": first, "first_large_step": large_from, "wall_s": walls[k]})
        log(f"  (a) {precision}{'' if waits else ' without waits'} process {k:2d}: largest difference from the median "
            f"{d:.2e} of the peak, first at step {first}, first large at step {large_from}")
    ds = np.array([r["largest"] for r in rows])
    return {"precision": precision, "waits": waits, "processes": n_proc, "seconds": seconds,
            "grid": info.get("grid"), "steps": info.get("steps"),
            "identical": int((ds == 0).sum()), "small": int(((ds > 0) & (ds < LARGE)).sum()),
            "large": int((ds >= LARGE).sum()), "largest": float(ds.max()) if ds.size else 0.0, "per_process": rows}


# ---------------------------------------------------------------- (b) batched inverse FFTs

def batched(precision: str, m: int, shape=(601, 801), batch: int = 2, log=print) -> dict:
    import cupy as cp
    ny, nx = shape
    rng = np.random.default_rng(0)
    x = cp.asarray(rng.standard_normal(shape).astype(precision))
    spec1 = cp.fft.rfft2(x)
    ref = cp.fft.irfft2(spec1, s=shape)
    scale = float(cp.abs(ref).max())
    spec = cp.ascontiguousarray(cp.broadcast_to(spec1, (batch,) + spec1.shape))
    cp.cuda.Device().synchronize()
    devs, bad_at = [], []
    worst = cp.zeros(m, dtype=cp.float64)
    t0 = time.time()
    for i in range(m):
        y = cp.fft.irfft2(spec, s=shape, axes=(-2, -1))
        worst[i] = cp.abs(y - ref).max()
        if (i + 1) % 5000 == 0:
            cp.cuda.Device().synchronize()
            log(f"  (b) {precision}: {i + 1} of {m} calls, {time.time() - t0:.0f} s")
    w = cp.asnumpy(worst) / scale
    bad = np.flatnonzero(w >= LARGE)
    normal = w[w < LARGE]
    return {"precision": precision, "calls": m, "batch": batch, "grid": list(shape), "corrupted": int(bad.size),
            "corrupted_at": [int(i) for i in bad[:20]], "largest": float(w.max()),
            "normal_largest": float(normal.max()) if normal.size else None, "wall_s": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=20, help="fresh processes per precision in (a)")
    ap.add_argument("--m", type=int, default=20000, help="batched irfft2 calls per precision in (b)")
    ap.add_argument("--seconds", type=float, default=5.0, help="simulated time of each run of (a)")
    ap.add_argument("--n64", type=int, default=5, help="fresh processes in float64 in (a) (a float64 run is four to eight "
                    "times slower on cards with a reduced float64 rate)")
    ap.add_argument("--no-waits", action="store_true", help="also run (a) in float32 without the k space waits")
    ap.add_argument("--skip-runs", action="store_true", help="only (b) and (c)")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "studies" / "gpu_determinism")
    ap.add_argument("--child", choices=("float32", "float64"), help=argparse.SUPPRESS)
    ap.add_argument("--child-out", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        child(a.child, a.seconds, not a.no_waits, a.child_out)
        return 0

    a.out.mkdir(parents=True, exist_ok=True)
    logf = open(a.out / "run.log", "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    results = {"criterion": CRITERION, "large_threshold": LARGE, "versions": versions(),
               "args": {"n": a.n, "n64": a.n64, "m": a.m, "seconds": a.seconds, "no_waits": a.no_waits},
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (a.out / "results.json").write_text(json.dumps(results, indent=1))     # the criterion is written before any run
    log(f"versions: {json.dumps(results['versions'])}")
    t0 = time.time()
    if not a.skip_runs:
        results["runs"] = []
        sets = [("float32", a.n, True), ("float64", a.n64, True)] + ([("float32", a.n, False)] if a.no_waits else [])
        for prec, n, waits in sets:
            results["runs"].append(runs(prec, n, a.seconds, waits, log))
            r = results["runs"][-1]
            log(f"(a) {prec}{'' if waits else ' without waits'}: {r['identical']} identical, {r['small']} small, "
                f"{r['large']} large (largest {r['largest']:.2e}) in {n} processes against their median")
            (a.out / "results.json").write_text(json.dumps(results, indent=1))
    results["batched"] = []
    for prec in ("float32", "float64"):
        results["batched"].append(batched(prec, a.m, log=log))
        b = results["batched"][-1]
        log(f"(b) {prec}: {b['corrupted']} corrupted in {b['calls']} batched calls (largest {b['largest']:.2e}, "
            f"correct calls at most {b['normal_largest']:.2e})")
    with_waits = [r for r in results.get("runs", []) if r["waits"]]
    results["deterministic"] = bool(all(r["large"] == 0 for r in with_waits)
                                    and all(b["corrupted"] == 0 for b in results["batched"]))
    results["deterministic_without_waits"] = next((r["large"] == 0 for r in results.get("runs", []) if not r["waits"]), None)
    results["wall_s"] = round(time.time() - t0, 1)
    (a.out / "results.json").write_text(json.dumps(results, indent=1))
    log(f"deterministic (criterion above): {results['deterministic']}; without the waits: "
        f"{results['deterministic_without_waits']}; {results['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

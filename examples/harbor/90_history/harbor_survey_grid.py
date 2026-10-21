#!/usr/bin/env python3
"""Pre-compute per-cell FDTD transfer functions for the harbor grid survey.

Fires a Ricker 40 Hz ``StaticSource`` from every water node of the scan grid,
records all-receiver traces via ``HarborFDTDSolver``, and saves each cell's
traces to ``<out>/survey_cache/{ix}_{iy}.npz`` (fields: ``traces`` (N, T),
``source_xy`` (2,)).

This is the optional pre-computation step for the grid-survey empirical
localiser mode in ``harbor_m3_detect.py --empirical-refine grid_survey``.
Run once and reuse; existing cells are skipped (resume-safe).

Usage::

    python examples/harbor/90_history/harbor_survey_grid.py [--grid-dx 20] [--out output/harbor] \\
                                          [--time 6] [--workers 4]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from acoustic_sim.water import (  # noqa: E402
    HarborFDTDSolver,
    HarborScenarioConfig,
    build_environment,
    build_layout,
)
from acoustic_sim.water.detection import ProcessingConfig, build_grid  # noqa: E402


# ---------------------------------------------------------------------------
# Per-cell worker (must be importable at the top level for multiprocessing)
# ---------------------------------------------------------------------------

def _run_cell(args):
    """Run FDTD for one grid cell and return ``(ix, iy, x, y, traces)``.

    Called in a subprocess; all imports are done inside the function so that
    the worker can be pickled cleanly.
    """
    ix, iy, x, y, env_pkl, hcfg_pkl, layout_pkl, total_time = args

    import numpy as np
    from acoustic_sim.water.fdtd_runner import HarborFDTDSolver
    from acoustic_sim.sources import StaticSource

    env, hcfg, layout = env_pkl, hcfg_pkl, layout_pkl
    fs = hcfg.fs
    n = int(round(total_time * fs))

    # Ricker pulse centred at 40 Hz
    t = np.arange(n) / fs
    a = (np.pi * 40.0 * (t - 0.1)) ** 2
    signal = (1.0 - 2.0 * a) * np.exp(-a)
    # Pad or trim to match n_steps of hcfg
    n_steps = hcfg.n_steps
    if len(signal) < n_steps:
        signal = np.pad(signal, (0, n_steps - len(signal)))
    else:
        signal = signal[:n_steps]

    src = StaticSource(float(x), float(y), signal)
    solver = HarborFDTDSolver(env, hcfg, src, layout.sample_points)
    result = solver.run(verbose=False)
    traces = result["traces"]  # (n_samples, n_steps)
    if layout.has_strain:
        traces = layout.collapse(traces.T).T
    return ix, iy, x, y, traces.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "harbor.scenario.json"))
    ap.add_argument("--grid-dx", type=float, default=20.0, help="scan grid spacing [m]")
    ap.add_argument("--out", default="output/harbor")
    ap.add_argument("--time", type=float, default=6.0, help="simulated seconds per cell")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    survey_dir = out / "survey_cache"
    survey_dir.mkdir(parents=True, exist_ok=True)

    # Load scenario
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = HarborScenarioConfig.load(cfg_path)
    else:
        cfg = HarborScenarioConfig()
    # Override simulation time for the shorter per-cell runs
    cfg.fdtd.total_time = args.time

    env = build_environment(cfg)
    layout = build_layout(cfg, env)

    pc = ProcessingConfig(grid_dx=args.grid_dx)
    grid = build_grid(env, pc)   # (G, 2) water cells

    # Build list of cells that still need processing
    jobs = []
    for gi, (x, y) in enumerate(grid):
        ix = int(round(x / args.grid_dx))
        iy = int(round(y / args.grid_dx))
        out_path = survey_dir / f"{ix}_{iy}.npz"
        if out_path.exists():
            continue
        jobs.append((ix, iy, x, y, env, cfg.fdtd, layout, args.time))

    total = len(grid)
    done_already = total - len(jobs)
    print(f"Grid: {total} cells total, {done_already} already cached, {len(jobs)} to compute.")

    if not jobs:
        print("All cells already computed — nothing to do.")
        return

    t0 = time.time()
    done_count = done_already

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_cell, j): j for j in jobs}
            for fut in as_completed(futures):
                try:
                    ix, iy, x, y, traces = fut.result()
                except Exception as exc:
                    j = futures[fut]
                    print(f"  ERROR cell ({j[2]:.0f}, {j[3]:.0f}): {exc}", file=sys.stderr)
                    continue
                out_path = survey_dir / f"{ix}_{iy}.npz"
                np.savez_compressed(out_path, traces=traces, source_xy=np.array([x, y]))
                done_count += 1
                elapsed = time.time() - t0
                print(f"[{done_count}/{total}] cell ({x:.0f}, {y:.0f})  "
                      f"{elapsed:.0f}s elapsed", flush=True)
    else:
        for j in jobs:
            try:
                ix, iy, x, y, traces = _run_cell(j)
            except Exception as exc:
                print(f"  ERROR cell ({j[2]:.0f}, {j[3]:.0f}): {exc}", file=sys.stderr)
                continue
            out_path = survey_dir / f"{ix}_{iy}.npz"
            np.savez_compressed(out_path, traces=traces, source_xy=np.array([x, y]))
            done_count += 1
            elapsed = time.time() - t0
            print(f"[{done_count}/{total}] cell ({x:.0f}, {y:.0f})  "
                  f"{elapsed:.0f}s elapsed", flush=True)

    print(f"Done. {done_count - done_already} cells written to {survey_dir}  "
          f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()

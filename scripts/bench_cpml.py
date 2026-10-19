#!/usr/bin/env python3
"""Stage 4: boundary reflection of the sponge and of the CPML, against angle and frequency.

A point source 40 m from one edge of an open water domain (the fine scenario's 0.5 m grid, 6 kHz,
order 4), receivers 40 m from that edge at offsets along it (incidence 0 to 82 degrees).  A
reference run on a domain large enough that no boundary is reached within the window gives the
incident field alone; the reflection is the difference.  Reported per receiver: the broadband
peak ratio [dB] and the spectral ratio in bands of the processing band (100 to 760 Hz).

Criterion (fixed in the task): a CPML thickness exists whose reflection is at least 10 dB below
the current sponge (60 cells, damping_max 0.15) across the band and at all tested angles.

Usage::

    python scripts/bench_cpml.py --cuda [--out output/performance] [--widths 10,15,20,30]
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

from acoustic_sim.harbor import HarborFDTDConfig, HarborFDTDSolver, SensorLayout, uniform_water_environment  # noqa: E402
from acoustic_sim.sources import StaticSource  # noqa: E402

DX, FS, C = 0.5, 6000.0, 1500.0
BANDS = [(100.0, 200.0), (200.0, 400.0), (400.0, 600.0), (600.0, 760.0)]
OFFSETS = np.array([0.0, 40.0, 80.0, 160.0, 320.0, 560.0])      # along the boundary; angle = atan(offset / 80)
STANDOFF = 40.0                                                   # source and receivers this far from the edge
T = 0.5
HARBOR_GRID = (601, 801)                                          # fine bistatic scenario (ny, nx)


def ricker(n: int, f0: float, delay: float) -> np.ndarray:
    t = np.arange(n) / FS
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def band_ratio_db(diff: np.ndarray, ref: np.ndarray) -> list[float]:
    F = np.fft.rfftfreq(ref.shape[-1], 1.0 / FS)
    D, R = np.abs(np.fft.rfft(diff)) ** 2, np.abs(np.fft.rfft(ref)) ** 2
    return [float(10 * np.log10(D[(F >= lo) & (F < hi)].sum() / R[(F >= lo) & (F < hi)].sum())) for lo, hi in BANDS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "output" / "performance"))
    ap.add_argument("--widths", default="10,15,20,30")
    ap.add_argument("--kernel", default="auto")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Domain: the measured edge is x = W; everything else is >= 400 m away from source and receivers.
    W, H = 700.0, 1400.0
    y0 = 420.0
    src = np.array([W - STANDOFF, y0])
    rec = np.column_stack([np.full_like(OFFSETS, W - STANDOFF), y0 + OFFSETS])
    angles = np.degrees(np.arctan(OFFSETS / (2 * STANDOFF)))
    shift = 600.0
    env = uniform_water_environment(W, H, DX, C)
    env_big = uniform_water_environment(W + 2 * shift, H + 2 * shift, DX, C)
    base = dict(fs=FS, total_time=T, fd_order=4, background_absorption=0.0, shore_loss_max=0.0, use_cuda=args.cuda, kernel=args.kernel)
    n = HarborFDTDConfig(**base).n_steps
    sig = ricker(n, 350.0, 0.02)

    def run(e, cfg, s, r):
        import_sync()
        t0 = time.perf_counter()
        solver = HarborFDTDSolver(e, cfg, StaticSource(s[0], s[1], sig), SensorLayout(hydrophones=r).sample_points)
        tr = solver.run(verbose=False)["traces"]
        import_sync()
        return tr, time.perf_counter() - t0, solver

    print(f"[cpml] domain {env.nx} x {env.ny}, reference {env_big.nx} x {env_big.ny}, {n} steps, angles {angles.round(0)}")
    ref, t_ref, s_ref = run(env_big, HarborFDTDConfig(**{**base, "damping_width": 60}), src + shift, rec + shift)
    print(f"[cpml] reference {t_ref:.1f} s (kernel {s_ref.kernel})")
    configs = {"sponge60": dict(damping_width=60, damping_max=0.15)}
    widths = [int(w) for w in args.widths.split(",") if w]
    for w in widths:
        configs[f"cpml{w}"] = dict(boundary="cpml", cpml_width=w)
        configs[f"cpml{w}_m3"] = dict(boundary="cpml", cpml_width=w, cpml_order=3)
        configs[f"cpml{w}_a314"] = dict(boundary="cpml", cpml_width=w, cpml_alpha_max=np.pi * 100.0)
    results = {"angles_deg": angles.tolist(), "offsets_m": OFFSETS.tolist(), "standoff_m": STANDOFF, "bands_hz": BANDS,
               "grid": [env.ny, env.nx], "reference_grid": [env_big.ny, env_big.nx], "n_steps": n, "fs": FS, "dx": DX,
               "gpu": gpu_name(args.cuda), "configs": {}}
    for name, kw in configs.items():
        tr, wall, solver = run(env, HarborFDTDConfig(**{**base, **kw}), src, rec)
        diff = tr - ref
        peak_db = (20 * np.log10(np.abs(diff).max(axis=1) / np.abs(ref).max(axis=1))).tolist()
        bands_db = [band_ratio_db(diff[i], ref[i]) for i in range(len(rec))]
        results["configs"][name] = {"options": kw, "wall_s": wall, "ms_per_step": 1e3 * wall / n, "kernel": solver.kernel,
                                    "peak_db": peak_db, "band_db": bands_db}
        print(f"[cpml] {name:14s} {wall:6.1f} s  peak dB by angle {np.round(peak_db, 1)}  worst band {max(max(b) for b in bands_db):6.1f} dB")
    sponge = results["configs"]["sponge60"]
    passing = {}
    for name, r in results["configs"].items():
        if name == "sponge60":
            continue
        margin_peak = min(s - c for s, c in zip(sponge["peak_db"], r["peak_db"]))
        margin_band = min(s - c for sb, cb in zip(sponge["band_db"], r["band_db"]) for s, c in zip(sb, cb))
        r["margin_db"] = {"peak": margin_peak, "band": margin_band}
        r["pass"] = bool(margin_peak >= 10.0 and margin_band >= 10.0)
        if r["pass"]:
            passing[name] = r["options"]["cpml_width"]
    results["criterion"] = {"text": "a CPML thickness with reflection >= 10 dB below the 60 cell sponge in every band and at every angle",
                            "pass": bool(passing), "passing_configs": passing}
    if passing:
        w_min = min(passing.values())
        ny, nx = HARBOR_GRID
        saved = 2 * (60 - w_min)
        results["cells_saved_fine_harbor"] = {"sponge_cells": 60, "cpml_cells": w_min, "grid_now": [ny, nx],
                                              "grid_with_cpml": [ny - saved, nx - saved],
                                              "fraction_saved": 1.0 - (ny - saved) * (nx - saved) / (ny * nx)}
    (out / "cpml_reflection.json").write_text(json.dumps(results, indent=2))
    print(f"[cpml] criterion {'PASS' if passing else 'FAIL'}: {passing}; wrote {out / 'cpml_reflection.json'}")
    try:
        plot(results, out / "cpml_reflection.png")
    except Exception as e:  # pragma: no cover
        print(f"[cpml] plot skipped: {e!r}")


def import_sync() -> None:
    try:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def gpu_name(use_cuda: bool) -> str | None:
    if not use_cuda:
        return None
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        return None


def plot(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles = results["angles_deg"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, r in results["configs"].items():
        style = "k-o" if name == "sponge60" else ("-s" if "_m3" in name else ("--^" if "_a314" in name else "-o"))
        axes[0].plot(angles, r["peak_db"], style, label=name, lw=2 if name == "sponge60" else 1)
        bands = np.array(r["band_db"])             # (angles, bands)
        centres = [0.5 * (lo + hi) for lo, hi in results["bands_hz"]]
        axes[1].plot(centres, bands.max(axis=0), style, label=name, lw=2 if name == "sponge60" else 1)
    axes[0].set_xlabel("incidence angle [deg]"); axes[0].set_ylabel("peak reflection [dB re incident]")
    axes[0].set_title("Broadband peak of the reflected field")
    axes[1].set_xlabel("band centre [Hz]"); axes[1].set_ylabel("worst angle reflected energy [dB re incident]")
    axes[1].set_title("Reflected energy per band (worst angle)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Boundary reflection, {results['dx']} m grid, {results['fs']:.0f} Hz, order 4, source and receivers {results['standoff_m']:g} m from the edge")
    fig.savefig(path, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()

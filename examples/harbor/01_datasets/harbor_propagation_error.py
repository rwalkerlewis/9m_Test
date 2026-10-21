#!/usr/bin/env python3
"""The forward model's own propagation error on one recording: arrival time at each hydrophone against range over the
chart's sound speed.

The simulated water propagates the direct paths faster than the chart's single sound speed, by an amount that depends
on direction (``docs/harbor.md``): the FDTD scheme's dispersion at 2 to 4 cells per wavelength at the top of the band.
Measured here from the truth side: one FDTD run of a stationary Ricker pulse at a ship position (the first radiating
vessel at mid record), the same grid, solver and sensors as the recording; at each hydrophone the arrival time is the
lag of the peak of the envelope of the band limited cross correlation between the recorded pulse and the free field
prediction at the chart's speed (the group arrival; the nearest correlation peak, the phase arrival, is kept too, but it is
ambiguous by a period where the pulse is dispersed) (``green2d`` at the chart's nominal speed, the processing's own replica), the window gated around the
direct arrival so no reflection enters.  A negative error is an early arrival (faster than the chart).

The result goes into the truth file's ``meta`` as ``propagation_error_ms`` (per hydrophone: position, range, direction
from the source, error) and the figure ``propagation_error.png`` (error against direction).  The forward model is not
changed.

Usage::

    python examples/harbor/01_datasets/harbor_propagation_error.py --dataset output/harbor_multistatic/fine/dataset
        [--out output/harbor_multistatic] [--cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Truth, contract_paths  # noqa: E402
from acoustic_sim.sources import make_wavelet_ricker  # noqa: E402
from acoustic_sim.water import HarborScenarioConfig, build_environment, build_layout, build_track, simulate_source  # noqa: E402
from acoustic_sim.water.propagation import green2d  # noqa: E402
from acoustic_sim.water.signatures import f_max_for_grid  # noqa: E402

UPSAMPLE = 16


def arrival_lag(measured: np.ndarray, predicted: np.ndarray, fs: float, t0: float, t1: float, period: float) -> tuple[float, float]:
    """Lag [s] of *measured* behind *predicted* inside the window [t0, t1]: the peak of the envelope of their cross
    correlation (FFT upsampled by ``UPSAMPLE``; the group arrival, free of cycle skips) and the correlation peak nearest
    to it, within half a *period* (the phase arrival).  Returns ``(phase lag, envelope lag)``."""
    a, b = int(max(t0, 0.0) * fs), int(t1 * fs)
    x, y = measured[a:b], predicted[a:b]
    n = 2 * len(x)
    xc = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(y, n)), n * UPSAMPLE)
    xc = np.fft.fftshift(xc)
    lags = (np.arange(n * UPSAMPLE) - n * UPSAMPLE // 2) / (UPSAMPLE * fs)
    env = np.abs(hilbert(xc))
    i = int(np.argmax(env))
    near = np.abs(lags - lags[i]) <= 0.5 * period
    j = np.flatnonzero(near)[int(np.argmax(xc[near]))]
    return float(lags[j]), float(lags[i])


def measure(dataset: Path, use_cuda: bool = True, band: tuple[float, float] | None = None) -> dict:
    obs_p, truth_p = contract_paths(dataset.with_suffix(".npz"))
    truth = Truth.load(truth_p)
    cfg = HarborScenarioConfig.from_dict(truth.scenario)
    cfg.fdtd.use_cuda = use_cuda
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps
    f_hi = band[1] if band else f_max_for_grid(cfg.environment.dx)
    f_lo = band[0] if band else f_hi / 7.6
    src = next(v for v in cfg.vessels if v.level_pa > 0)
    tr = build_track(src, cfg.fdtd.total_time, env)
    pos = tr.position_at(0.5 * cfg.fdtd.total_time)[0]
    sig = make_wavelet_ricker(n, 1.0 / fs, 0.5 * (f_lo + f_hi))
    g = simulate_source(env, layout, cfg.fdtd, sig, position=(float(pos[0]), float(pos[1])),
                        cache_dir=dataset.parent.parent / "cache" if (dataset.parent.parent / "cache").exists() else None)
    c = env.public().c_nominal
    hyd = np.asarray(cfg.hydrophones, float)
    rows = layout.slices["hydrophones"]
    sos = butter(4, (f_lo, f_hi), btype="bandpass", fs=fs, output="sos")
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    k = 2.0 * np.pi * f / c
    t_peak = 1.5 / (0.5 * (f_lo + f_hi))                         # the Ricker's centre
    out = []
    for i, (h, row) in enumerate(zip(hyd, range(rows.start, rows.stop))):
        r = float(np.hypot(*(h - pos)))
        P = S * np.where(f > 0, green2d(np.maximum(k * r, 1e-9)), 0.0)
        pred = sosfiltfilt(sos, np.fft.irfft(P, n))
        meas = sosfiltfilt(sos, g.traces[row].astype(float))
        t_arr = t_peak + r / c
        lag, lag_env = arrival_lag(meas, pred, fs, t_arr - 4.0 / f_lo, t_arr + 4.0 / f_lo, 2.0 / (f_lo + f_hi))
        out.append({"hydrophone": i, "xy": h.tolist(), "range_m": r,
                    "direction_deg": float(np.degrees(np.arctan2(h[0] - pos[0], h[1] - pos[1])) % 360.0),
                    "error_ms": 1e3 * lag_env, "phase_error_ms": 1e3 * lag})
    e = [o["error_ms"] for o in out]
    res = {"source_xy": pos.tolist(), "source": src.id, "band_hz": [f_lo, f_hi], "c_chart": c, "hydrophones": out,
           "median_ms": float(np.median(e)), "min_ms": float(np.min(e)), "max_ms": float(np.max(e)),
           "note": "arrival (peak of the cross correlation envelope) at the hydrophone minus range / chart speed; negative: faster than the chart; phase_error_ms: the correlation peak nearest to it, ambiguous by a period where the pulse is dispersed"}
    truth.meta["propagation_error_ms"] = res
    truth.save(truth_p)
    return res


def figure(res: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    h = res["hydrophones"]
    ax.axhline(0.0, color="0.5", lw=1)
    ax.plot([o["direction_deg"] for o in h], [o["error_ms"] for o in h], "o", ms=9, color="#1f77b4", label="envelope (group) arrival")
    ax.plot([o["direction_deg"] for o in h], [o["phase_error_ms"] for o in h], "s", ms=7, mfc="none", color="#d62728",
            label="nearest phase peak (ambiguous by a period)")
    ax.legend(fontsize=9)
    for o in h:
        ax.annotate(f"hydrophone {o['hydrophone'] + 1}, {o['range_m']:.0f} m", (o["direction_deg"], o["error_ms"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlim(0, 360)
    ax.set_xlabel("direction from the source [degrees clockwise from north]")
    ax.set_ylabel("arrival minus range / chart speed [ms]")
    ax.set_title(f"{title}: propagation error of the forward model, stationary pulse at ({res['source_xy'][0]:.0f}, "
                 f"{res['source_xy'][1]:.0f}) m,\n{res['band_hz'][0]:.0f}-{res['band_hz'][1]:.0f} Hz, chart speed "
                 f"{res['c_chart']:.1f} m/s (negative: faster than the chart)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="<dir>/dataset (the .observation.npz / .truth.npz pair)")
    ap.add_argument("--out", default=None, help="folder of propagation_error.png and .json (default: the dataset's recording folder)")
    ap.add_argument("--band", type=float, nargs=2, default=None, help="band [Hz] (default: the grid's, f_max / 7.6 to f_max)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    ds = Path(a.dataset)
    out = Path(a.out) if a.out else (ds.parent.parent if ds.parent.name == "fine" else ds.parent)
    res = measure(ds, use_cuda=not a.cpu, band=tuple(a.band) if a.band else None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "propagation_error.json").write_text(json.dumps(res, indent=2))
    figure(res, out / "propagation_error.png", out.name)
    print(f"{out.name}: " + ", ".join(f"hydrophone {o['hydrophone'] + 1} {o['error_ms']:+.2f} ms ({o['direction_deg']:.0f} deg, "
                                      f"{o['range_m']:.0f} m)" for o in res["hydrophones"]))


if __name__ == "__main__":
    main()

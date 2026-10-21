#!/usr/bin/env python3
"""The forward model's own propagation error on one recording: arrival time at each hydrophone against range over the
chart's sound speed.

Measured from the truth side: one FDTD run of a stationary Ricker pulse at a ship position (the first radiating vessel
at mid record), the same grid, solver and sensors as the recording; at each hydrophone the arrival time is the lag of
the peak of the envelope of the band limited cross correlation between the recorded pulse and the free field prediction
at the chart's speed (``green2d`` at the chart's nominal speed, the processing's own replica; the group arrival; the
nearest correlation peak, the phase arrival, is kept too, but it is ambiguous by a period where the pulse is dispersed),
the window gated around the direct arrival.  A negative error is an early arrival (faster than the chart).  Beside it,
``straight_path_ms``: the travel time along the straight line through the scenario's sound speed map minus range over
the chart's speed, what the water itself contributes (refraction aside).

The result goes into the truth file's ``meta`` as ``propagation_error_ms``, ``propagation_error.json`` and the figure
``propagation_error.png`` in the recording's folder.  Variants (``--scheme``, ``--uniform-speed``, ``--dx``) describe
the scenario changed, not the recording, and are written to ``output/studies/propagation_error/<recording>/`` as
``propagation_error_<variant>.json`` and ``.png``, never into the truth.  ``--plot-only`` redraws the figures of a
recording and of its variants from their JSON (no FDTD run).

Criterion (fixed before the k space recordings were built, task ``forward_fidelity``): the direct path arrival error
at every hydrophone within 0.2 ms on the 0.5 m and 1 m recordings, within 0.1 ms on the simple harbor (whose water has
exactly the chart's speed); the 5 m harbor has no criterion (reported).  ``criterion_ms`` and ``passed`` are in the
result (``None`` where there is no criterion).

Usage::

    python examples/harbor/01_datasets/harbor_propagation_error.py --dataset output/recordings/multistatic/dataset
        [--out output/recordings/multistatic] [--cpu] [--plot-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Truth, contract_paths  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
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


def criterion_ms(cfg: HarborScenarioConfig) -> float | None:
    """The criterion of the module docstring for *cfg*: 0.1 ms on the simple harbor, 0.2 ms on the other 0.5 m and 1 m
    grids, None on coarser grids."""
    if cfg.environment.dx > 1.0:
        return None
    return 0.1 if cfg.environment.kind == "simple" else 0.2


def straight_path_time(env, a: np.ndarray, b: np.ndarray, step: float | None = None) -> float:
    """Travel time [s] from *a* to *b* along the straight line through the environment's sound speed map (the slowness
    sampled at nearest grid nodes every *step* m, default a fifth of a cell)."""
    step = step or 0.2 * env.dx
    r = float(np.hypot(*(np.asarray(b, float) - np.asarray(a, float))))
    n = max(int(np.ceil(r / step)), 1)
    s = (np.arange(n) + 0.5) / n
    xy = np.asarray(a, float)[None, :] + s[:, None] * (np.asarray(b, float) - np.asarray(a, float))[None, :]
    c = env.model.values
    ix = np.clip(np.rint((xy[:, 0] - env.model.x[0]) / env.dx).astype(int), 0, c.shape[1] - 1)
    iy = np.clip(np.rint((xy[:, 1] - env.model.y[0]) / env.dx).astype(int), 0, c.shape[0] - 1)
    return float(np.sum(r / n / c[iy, ix]))


def measure(dataset: Path, use_cuda: bool = True, band: tuple[float, float] | None = None, scheme: str | None = None,
            uniform_speed: bool = False, dx: float | None = None, save: bool = True) -> dict:
    """The propagation error of *dataset*'s forward model.  *scheme* overrides the scenario's ``time_scheme``,
    *uniform_speed* sets ``shallow_c_drop = 0`` (the water everywhere at ``c0``, the chart's speed too) and *dx* runs the
    same scenario on another grid spacing (the sample rate scaled with it, the sponge kept in metres); with any of them the
    result describes a variant of the scenario, not the recording, and is not written into the truth file."""
    obs_p, truth_p = contract_paths(dataset.with_suffix(".npz"))
    truth = Truth.load(truth_p)
    cfg = HarborScenarioConfig.from_dict(truth.scenario)
    total = cfg.fdtd.total_time
    cfg.fdtd.use_cuda = use_cuda
    if scheme is not None:
        cfg.fdtd.time_scheme = scheme
    if uniform_speed:
        cfg.environment.shallow_c_drop = 0.0
    if dx is not None:
        r = cfg.environment.dx / dx
        cfg.fdtd.damping_width = int(round(cfg.fdtd.damping_width * r))
        cfg.fdtd.fs *= r
        cfg.environment.dx = dx
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    f_hi = band[1] if band else f_max_for_grid(cfg.environment.dx)
    f_lo = band[0] if band else f_hi / 7.6
    src = next(v for v in cfg.vessels if v.level_pa > 0)
    tr = build_track(src, total, env)
    pos = tr.position_at(0.5 * total)[0]
    # The pulse run only needs to outlast the direct arrival at the farthest hydrophone (and its gate).
    hyd = np.asarray(cfg.hydrophones, float)
    c_min = float(np.min(env.model.values[env.water]))
    cfg.fdtd.total_time = min(total, 1.5 / (0.5 * (f_lo + f_hi)) + float(np.max(np.hypot(*(hyd - pos).T))) / c_min + 10.0 / f_lo)
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps
    sig = make_wavelet_ricker(n, 1.0 / fs, 0.5 * (f_lo + f_hi))
    g = simulate_source(env, layout, cfg.fdtd, sig, position=(float(pos[0]), float(pos[1])),
                        cache_dir=dataset.parent / "cache" if (dataset.parent / "cache").exists() else None)
    c = env.public().c_nominal
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
                    "error_ms": 1e3 * lag_env, "phase_error_ms": 1e3 * lag,
                    "straight_path_ms": 1e3 * (straight_path_time(env, pos, h) - r / c)})
    e = [o["error_ms"] for o in out]
    res = {"source_xy": pos.tolist(), "source": src.id, "band_hz": [f_lo, f_hi], "c_chart": c, "time_scheme": cfg.fdtd.time_scheme,
           "uniform_speed": uniform_speed, "dx": cfg.environment.dx, "fs": fs, "hydrophones": out,
           "median_ms": float(np.median(e)), "min_ms": float(np.min(e)), "max_ms": float(np.max(e)),
           "note": "arrival (peak of the cross correlation envelope) at the hydrophone minus range / chart speed; negative: faster than the chart; phase_error_ms: the correlation peak nearest to it, ambiguous by a period where the pulse is dispersed; straight_path_ms: the travel time along the straight line through the scenario's sound speed map minus range / chart speed (what the water itself contributes, refraction aside)"}
    lim = criterion_ms(cfg)
    res["criterion_ms"] = lim
    res["passed"] = None if lim is None else bool(max(abs(x) for x in e) <= lim)
    if save and scheme is None and not uniform_speed and dx is None:
        truth.meta["propagation_error_ms"] = res
        truth.save(truth_p)
    return res


def figure(res: dict, path: Path, recording: str, variant: str = "") -> None:
    """The arrival error at every hydrophone against its direction from the source (one panel)."""
    fig, ax = new_figure((12.0, 8.0))
    h = res["hydrophones"]
    ax.axhline(0.0, color="0.5", lw=1, label="no error")
    ax.plot([o["direction_deg"] for o in h], [o["error_ms"] for o in h], "o", ms=10, color="#1f77b4", label="envelope (group) arrival")
    ax.plot([o["direction_deg"] for o in h], [o["phase_error_ms"] for o in h], "s", ms=8, mfc="none", color="#d62728",
            label="nearest phase peak (ambiguous by a period)")
    if all("straight_path_ms" in o for o in h):
        ax.plot([o["direction_deg"] for o in h], [o["straight_path_ms"] for o in h], "x", ms=10, mew=2, color="#2ca02c",
                label="straight path through the sound speed map")
    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    order = np.argsort([o["direction_deg"] for o in h])
    for k, i in enumerate(order):                           # alternate above and below so neighbours do not overlap
        o = h[i]
        right = o["direction_deg"] > 270.0                  # labels near the right edge go to the left of the point
        ax.annotate(f"hydrophone {o['hydrophone'] + 1}, {o['range_m']:.0f} m", (o["direction_deg"], o["error_ms"]),
                    xytext=(-8 if right else 8, 10 if k % 2 == 0 else -18), textcoords="offset points", fontsize=11,
                    ha="right" if right else "left")
    ax.set_xlim(0, 360)
    ax.set_xlabel("direction from the source [degrees clockwise from north]")
    ax.set_ylabel("arrival minus range / chart speed [ms]")
    ax.set_title(f"{recording}{variant}: propagation error, pulse at ({res['source_xy'][0]:.0f}, {res['source_xy'][1]:.0f}) m, "
                 f"{res['band_hz'][0]:.0f} to {res['band_hz'][1]:.0f} Hz,\nchart speed {res['c_chart']:.1f} m/s "
                 f"(negative: faster than the chart)")
    ax.grid(alpha=0.3)
    save_figure(fig, path, recording=recording)


def variant_name(stem: str) -> str:
    """The variant of a ``propagation_error_<variant>`` file stem in words (empty for the recording itself)."""
    parts = stem.removeprefix("propagation_error").strip("_").split("_")
    words = []
    for p in filter(None, parts):
        if p == "uniform":
            words.append("uniform speed")
        elif p.startswith("dx"):
            words.append(f"{p[2:]} m grid")
        else:
            words.append(p)
    return "".join(f", {w}" for w in words)


def plot_only(ds: Path) -> None:
    """Redraw the figures of the recording of *ds* and of its variants from their JSON."""
    rec = ds.parent.name
    files = [ds.parent / "propagation_error.json"] + sorted((ROOT / "output" / "studies" / "propagation_error" / rec).glob("propagation_error*.json"))
    for js in files:
        if js.exists():
            figure(json.loads(js.read_text()), js.with_suffix(".png"), rec, variant_name(js.stem))
            print(f"wrote {js.with_suffix('.png')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="<dir>/dataset (the .observation.npz / .truth.npz pair)")
    ap.add_argument("--out", default=None, help="folder of propagation_error.png and .json (default: the dataset's recording folder)")
    ap.add_argument("--band", type=float, nargs=2, default=None, help="band [Hz] (default: the grid's, f_max / 7.6 to f_max)")
    ap.add_argument("--scheme", default=None, choices=("leapfrog", "lw4", "kspace"),
                    help="time scheme instead of the scenario's (a variant: written as propagation_error_<scheme>.json, not into the truth)")
    ap.add_argument("--uniform-speed", action="store_true",
                    help="water everywhere at the scenario's c0 (a variant: propagation_error_uniform.json, not into the truth)")
    ap.add_argument("--dx", type=float, default=None,
                    help="grid spacing [m] instead of the scenario's (a variant: propagation_error_dx<dx>.json, not into the truth)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figures from the JSON of the recording and its variants (no FDTD)")
    a = ap.parse_args()
    ds = Path(a.dataset)
    if a.plot_only:
        plot_only(ds)
        return
    variant = bool(a.scheme or a.uniform_speed or a.dx)
    out = Path(a.out) if a.out else (ROOT / "output" / "studies" / "propagation_error" / ds.parent.name if variant else ds.parent)
    res = measure(ds, use_cuda=not a.cpu, band=tuple(a.band) if a.band else None, scheme=a.scheme, uniform_speed=a.uniform_speed,
                  dx=a.dx)
    out.mkdir(parents=True, exist_ok=True)
    stem = ("propagation_error" + (f"_{a.scheme}" if a.scheme else "") + ("_uniform" if a.uniform_speed else "")
            + (f"_dx{a.dx:g}" if a.dx else ""))
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2))
    figure(res, out / f"{stem}.png", ds.parent.name, variant_name(stem))
    print(f"{ds.parent.name}: " + ", ".join(f"hydrophone {o['hydrophone'] + 1} {o['error_ms']:+.2f} ms ({o['direction_deg']:.0f} deg, "
                                      f"{o['range_m']:.0f} m)" for o in res["hydrophones"]))


if __name__ == "__main__":
    main()

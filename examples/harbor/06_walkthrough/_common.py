"""Shared by the walkthrough scripts: one real-time run of the harbor chain, kept for the figures.

:func:`products` runs the system of ``examples/harbor/05_detection/harbor_stream.py`` once on a recording (the passive chain
without AIS, the tracker, the echo chain on the strong tracks, the AIS labels afterwards) and keeps what
the figures need: every passive and echo map, every track, the cleaned traces.  The result is cached in
``output/walkthrough/<dataset>/cache/run.pkl``; delete it (or pass ``--rerun``) to run again.

The truth (:func:`truth`) is read only to draw where the objects really are, for comparison; nothing in
the processing reads it.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "harbor" / "05_detection"))

import harbor_stream as hs  # noqa: E402

from acoustic_sim.contract import Observation, Truth  # noqa: E402
from acoustic_sim.water.bistatic import BistaticConfig  # noqa: E402
from acoustic_sim.water.detection import ProcessingConfig  # noqa: E402
from acoustic_sim.water.propagation import FreeFieldPropagator  # noqa: E402
from acoustic_sim.water.streaming import (  # noqa: E402
    EchoStreamConfig, RadiatorTracker, StreamConfig, StreamingEcho, StreamingPassive, TrackerConfig, ais_crosscheck)
from acoustic_sim.water.suppression import SuppressionConfig  # noqa: E402

# Colours used by every figure.
RADIATOR, SILENT, TRACK, HIT, MISS, SENSOR = "#1f77b4", "#ff7f0e", "#111111", "#2ca02c", "#d62728", "#555555"


def args(doc: str, default: str = "multistatic") -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=default, choices=list(hs.DATASETS))
    ap.add_argument("--cuda", action="store_true", help="GPU (falls back to the CPU)")
    ap.add_argument("--rerun", action="store_true", help="run the chain again instead of using the cached run")
    ap.add_argument("--out", default="output/walkthrough")
    return ap.parse_args()


def out_dir(a: argparse.Namespace) -> Path:
    d = ROOT / a.out / a.dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def observation(dataset: str) -> Observation:
    return Observation.load(ROOT / f"{hs.DATASETS[dataset][0]}.observation.npz")


def truth(dataset: str) -> Truth:
    """Where the objects really are: for drawing and checking only."""
    return Truth.load(ROOT / f"{hs.DATASETS[dataset][0]}.truth.npz")


def products(a: argparse.Namespace) -> dict:
    """The run's products (cached): passive frames and maps, tracks, AIS labels, echo frames and maps,
    cleaned traces, timings."""
    cache = out_dir(a) / "cache" / "run.pkl"
    if cache.exists() and not a.rerun:
        with open(cache, "rb") as f:
            return pickle.load(f)
    path, band, pcd, edge, std = hs.DATASETS[a.dataset]
    obs = observation(a.dataset)
    env = obs.environment
    x0, x1, y0, y1 = env.extent
    prop = FreeFieldPropagator.from_environment(env)
    pc = ProcessingConfig(f_lo=band[0], f_hi=band[1], c=prop.c, propagator=prop, use_cuda=a.cuda,
                          grid_bounds=(x0 + edge, x1 - edge, y0 + edge, y1 - edge), **pcd)
    passive = StreamingPassive(obs.geometry, obs.fs, env, pc, SuppressionConfig(), StreamConfig(max_peaks=5, **std), keep_maps=True)
    tracker = RadiatorTracker(TrackerConfig(**hs.TRACKER.get(a.dataset, {})))
    echo = None
    if a.dataset != "harbor":
        bc = BistaticConfig(f_lo=band[0], f_hi=band[1], c=prop.c, propagator=prop, use_cuda=a.cuda, **hs.ECHO)
        echo = StreamingEcho(obs.geometry, obs.fs, env, bc, EchoStreamConfig(step_blocks=4), keep_maps=True, keep_clean=True)
    n = int(round(0.25 * obs.fs))
    p_frames, e_frames, sent, ever = [], [], {}, {}
    t_p = t_e = 0.0
    for s0 in range(0, obs.traces.shape[1], n):
        chunk = obs.traces[:, s0:s0 + n]
        t0 = time.perf_counter()
        new = passive.push(chunk)
        for f in new:
            tracker.update(f)
        t_p += time.perf_counter() - t0
        p_frames += new
        for k, tr in tracker.confirmed().items():
            ever[k] = tr
        if echo is not None:
            t1 = time.perf_counter()
            strong = tracker.strong()
            fixes = {}
            for k, tr in strong.items():
                fixes[k] = [(f[0], f[1]) for f in tr.fixes[sent.get(k, 0):]]
                sent[k] = len(tr.fixes)
            e_frames += echo.push(chunk, fixes)
            t_e += time.perf_counter() - t1
    final = tracker.confirmed()
    T = obs.traces.shape[1] / obs.fs
    out = {
        "dataset": a.dataset, "record_s": T, "fs": obs.fs,
        "passive": {"grid": passive.grid, "frames": [(f.t, f.xy, f.score, f.B) for f in p_frames]},
        "tracks": {k: [(f[0], f[1], f[2]) for f in tr.fixes] for k, tr in final.items()},
        "tracks_ever": {k: [(f[0], f[1], f[2]) for f in tr.fixes] for k, tr in ever.items()},
        "strong": sorted(set(sent)),
        "labels": ais_crosscheck(final, obs.reports),
        "echo": None if echo is None else {"grid": echo.grid, "frames": [(f.t, f.xy, f.score, f.B, f.ships) for f in e_frames if f.ships],
                                           "clean": np.concatenate([echo.kept_clean[b] for b in sorted(echo.kept_clean)], axis=1)
                                           if echo.kept_clean else None,
                                           "clean_start": min(echo.kept_clean) * echo.nb if echo.kept_clean else 0},
        "seconds": {"passive": t_p, "echo": t_e},
        "pc": {"win_s": pc.win_s, "hop_s": pc.hop_s, "band": band},
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def chart(ax, obs: Observation, sensors: bool = True) -> None:
    """Water (white) and land (grey), the DAS cable and hydrophones."""
    env = obs.environment
    x0, x1, y0, y1 = env.extent
    ax.imshow(np.where(env.water, 1.0, 0.0), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys_r", vmin=-0.8, vmax=1.0)
    if sensors:
        lay = obs.geometry
        das = lay.das_positions
        ax.plot(das[:, 0], das[:, 1], "-", color=SENSOR, lw=2.0, label="DAS cable")
        if lay.n_hydrophones:
            ax.plot(lay.hydrophones[:, 0], lay.hydrophones[:, 1], "^", color=SENSOR, ms=7, label="hydrophones")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def truth_overlay(ax, tru: Truth, t: float | None = None, labels: bool = True, alpha: float = 0.9) -> None:
    """True tracks (dashed) or positions at *t* (circles): blue radiating, orange silent."""
    for oid, tr in tru.tracks.items():
        col = RADIATOR if tr.level_pa > 0 else SILENT
        if t is None:
            moving = np.ptp(tr.xy, axis=0).max() > 0
            if moving:
                ax.plot(tr.xy[:, 0], tr.xy[:, 1], "--", color=col, lw=1.2, alpha=alpha)
                p = tr.xy[-1]
            else:
                p = tr.xy[0]
                ax.plot(*p, "s", color=col, ms=4, mfc="none", alpha=alpha)
        else:
            p = tr.position_at(t)[0]
            ax.plot(*p, "o", ms=11, mfc="none", mec=col, mew=1.6, alpha=alpha)
        if labels and not (oid.startswith("piling") and oid != "piling1"):
            ax.annotate(oid if not oid.startswith("piling") else "pilings", p, xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=col)


def map_image(grid: np.ndarray, values: np.ndarray):
    """Scattered grid values to an image (NaN outside the grid) and its extent."""
    xs, ys = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    dx = float(np.min(np.diff(xs)))
    img = np.full((len(ys), len(xs)), np.nan)
    img[np.rint((grid[:, 1] - ys[0]) / dx).astype(int), np.rint((grid[:, 0] - xs[0]) / dx).astype(int)] = values
    return img, (xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dx / 2, ys[-1] + dx / 2)


def save(fig, path: Path, note: str = "") -> None:
    if note:
        fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=7.5, color="0.35")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


NOTE = "Processing: traces, sensor positions and the chart only (AIS only labels tracks). Dashed / circles: true positions, for comparison."

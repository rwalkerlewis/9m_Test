"""Shared by the walkthrough scripts: one real-time run of the harbor chain, kept for the figures.

:func:`products` runs the system (``acoustic_sim.water.stream``, as ``examples/harbor/05_detection/harbor_stream.py``: the
passive chain without AIS, the radiator tracker, the echo chain on the strong tracks, the echo tracker, the
characteristics, AIS labels) once on a recording with its processing config (``examples/configs/processing/<dataset>.json``)
and keeps what the figures need: every passive and echo map, every track, the cleaned traces, the picture records.  The
result is cached in ``output/walkthrough/<dataset>/cache/run.pkl``; delete it (or pass ``--rerun``) to run again.

The truth (:func:`truth`) is read only to draw where the objects really are, for comparison; nothing in
the processing reads it.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Observation, Truth, contract_paths  # noqa: E402
from acoustic_sim.water.streaming import ais_crosscheck  # noqa: E402
from acoustic_sim.water.stream import load_processing, run  # noqa: E402

CONFIGS = ROOT / "examples" / "configs" / "processing"
DATASETS = ("simple", "bistatic", "multistatic", "harbor")

# Colours used by every figure.
RADIATOR, SILENT, TRACK, HIT, MISS, SENSOR = "#1f77b4", "#ff7f0e", "#111111", "#2ca02c", "#d62728", "#555555"


def args(doc: str, default: str = "multistatic") -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=default, choices=DATASETS)
    ap.add_argument("--cpu", action="store_true", help="process on the CPU (default: the GPU, falling back to the CPU)")
    ap.add_argument("--cuda", action="store_true", help="kept for old command lines: the GPU is the default")
    ap.add_argument("--rerun", action="store_true", help="run the chain again instead of using the cached run")
    ap.add_argument("--out", default="output/walkthrough")
    return ap.parse_args()


def out_dir(a: argparse.Namespace) -> Path:
    d = ROOT / a.out / a.dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def config(dataset: str):
    return load_processing(CONFIGS / f"{dataset}.json")


def observation_path(dataset: str) -> Path:
    return ROOT / config(dataset).raw["observation"]


def observation(dataset: str) -> Observation:
    return Observation.load(observation_path(dataset))


def truth(dataset: str) -> Truth:
    """Where the objects really are: for drawing and checking only."""
    return Truth.load(contract_paths(observation_path(dataset))[1])


def products(a: argparse.Namespace) -> dict:
    """The run's products (cached): passive frames and maps, tracks, AIS labels, echo frames and maps, echo tracks,
    cleaned traces, the picture records, timings."""
    cache = out_dir(a) / "cache" / "run.pkl"
    if cache.exists() and not a.rerun:
        with open(cache, "rb") as f:
            return pickle.load(f)
    cfg = config(a.dataset)
    obs = observation(a.dataset)
    res = run(obs, cfg, use_cuda=not a.cpu, observation_path=observation_path(a.dataset), keep_frames=True, keep_clean=True)
    pic = res.picture
    final = pic.tracker.confirmed()
    echo = pic.echo
    tracks_ever = {k: [(f[0], f[1], f[2]) for f in fx] for k, fx in pic.ever.items() if k.startswith("r")}
    out = {
        "dataset": a.dataset, "record_s": res.record_s, "fs": obs.fs,
        "passive": {"grid": pic.passive.grid, "frames": [(f.t, f.xy, f.score, f.B) for f in pic.p_frames]},
        "tracks": {k: [(f[0], f[1], f[2]) for f in tr.fixes] for k, tr in final.items()},
        "tracks_ever": tracks_ever,
        "strong": sorted(pic.sent),
        "labels": ais_crosscheck(final, obs.reports),
        "echo": None if echo is None else {
            "grid": echo.grid, "frames": [(f.t, f.xy, f.score, f.B, f.ships) for f in pic.e_frames if f.ships],
            "used": [(f.t, f.xy) for f in pic.e_used],
            "tracks_ever": {k: [(f[0], f[1], f[2]) for f in fx] for k, fx in pic.ever.items() if k.startswith("e")},
            "clean": np.concatenate([echo.kept_clean[b] for b in sorted(echo.kept_clean)], axis=1) if echo.kept_clean else None,
            "clean_start": min(echo.kept_clean) * echo.nb if echo.kept_clean else 0},
        "records": res.records,
        "seconds": dict(pic.seconds),
        "pc": {"win_s": pic.pc.win_s, "hop_s": pic.pc.hop_s, "band": cfg.band},
        "scoring": cfg.scoring,
        "confirm": {"radiator": cfg.tracker().confirm, **({"echo": cfg.echo_tracker().confirm} if cfg.has_echo else {})},
        "run": res.run,
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


def zoom(ax, grid: np.ndarray, margin: float = 15.0) -> None:
    """Limit a map's axes to its scan grid (plus *margin* m), so the map fills the image."""
    ax.set_xlim(grid[:, 0].min() - margin, grid[:, 0].max() + margin)
    ax.set_ylim(grid[:, 1].min() - margin, grid[:, 1].max() + margin)


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


class Figures:
    """One plot per image: :meth:`new` gives the axes of a new figure saved by :meth:`save_all` as
    ``<step>_<name>.png``, each headed by the step's title."""

    def __init__(self, a: argparse.Namespace, step: str, title: str, note: str = "") -> None:
        self.dir, self.step, self.title, self.note = out_dir(a), step, title, note
        self.figs: list[tuple[str, object]] = []

    def new(self, name: str, size=(12.0, 10.0)):
        fig, ax = plt.subplots(figsize=size)
        self.figs.append((name, fig))
        return fig, ax

    def save_all(self) -> None:
        for name, fig in self.figs:
            fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
            fig.suptitle(self.title, fontsize=12, y=0.995)
            save(fig, self.dir / f"{self.step}_{name}.png", self.note)


def save(fig, path: Path, note: str = "") -> None:
    if note:
        fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=7.5, color="0.35")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


NOTE = "Processing: traces, sensor positions and the chart only (AIS only labels tracks). Dashed / circles: true positions, for comparison."

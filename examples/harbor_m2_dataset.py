#!/usr/bin/env python3
"""Harbor M2: build the labelled dataset for the default scenario and plot it.

Runs one cached FDTD job per vessel (in parallel), mixes them with noise and AIS,
saves the dataset and writes overview figures plus a received-level table.

Usage::

    python examples/harbor_m2_dataset.py [--config examples/harbor.scenario.json]
        [--out output/harbor] [--workers 4] [--seed 0] [--time 120]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.harbor import (  # noqa: E402
    HarborScenarioConfig,
    assemble_dataset,
    build_environment,
    build_layout,
    build_track,
    render_vessel_gathers,
)
from acoustic_sim.harbor.plotting import plot_harbor  # noqa: E402


def band_rms(x: np.ndarray, sos: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sosfiltfilt(sos, x, axis=-1) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).parent / "harbor.scenario.json"))
    ap.add_argument("--out", default="output/harbor")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--time", type=float, default=None, help="override simulated duration [s]")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = HarborScenarioConfig.load(args.config)
    if args.time is not None:
        cfg.fdtd.total_time = args.time
    env = build_environment(cfg)
    layout = build_layout(cfg, env)

    t0 = time.time()
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=out / "cache", workers=args.workers)
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.seed)
    print(f"simulated {len(cfg.vessels)} vessels in {time.time() - t0:.0f} s")
    ds.save(out / "dataset.npz")

    fs, t = ds.fs, ds.t
    sos = butter(4, [15.0, 75.0], btype="bandpass", fs=fs, output="sos")
    cable = layout.cables[0]
    das_sl = layout.slices[cable.name]
    live = ~ds.dead

    # Received level (15-75 Hz rms, whole run, live sensors) per vessel and sensor group vs noise.
    noise_only = ds.traces - ds.gain[:, None] * ds.clean
    groups = {name: np.flatnonzero(live & (np.arange(layout.n_receivers) >= sl.start) & (np.arange(layout.n_receivers) < sl.stop))
              for name, sl in layout.slices.items()}
    print("\nband-limited (15-75 Hz) rms pressure [Pa] at the sensors")
    print(f"  {'source':10s} " + " ".join(f"{n:>12s}" for n in groups))
    for vid, tr in ds.tracks.items():
        row = [band_rms(tr.level_pa * gathers[vid].traces[idx].astype(float) * ds.gain[idx, None], sos) for idx in groups.values()]
        print(f"  {vid:10s} " + " ".join(f"{r:12.4f}" for r in row))
    print(f"  {'noise':10s} " + " ".join(f"{band_rms(noise_only[idx], sos):12.4f}" for idx in groups.values()))
    print()

    # 1. Map with truth tracks and AIS reports
    fig, ax = plt.subplots(figsize=(10, 7.5))
    plot_harbor(env, layout, ax=ax)
    for vid, tr in ds.tracks.items():
        ax.plot(*tr.xy[tr.active].T, "--" if not tr.ais else "-", lw=2,
                label=f"{vid} ({tr.kind}, {'AIS' if tr.ais else 'hidden'})")
        if tr.ais and len(ds.ais):
            r = ds.ais.for_vessel(vid)
            if len(r):
                ax.plot(*r.xy.T, "k.", ms=3)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out / "m2_scenario.png", dpi=100, bbox_inches="tight")

    # 2. DAS waterfall as 1 s rms in dB: clean, noisy, band-passed noisy
    def rms_db(data: np.ndarray, win: int = int(fs)) -> np.ndarray:
        n = data.shape[1] // win
        return 10 * np.log10(np.mean(data[:, : n * win].reshape(len(data), n, win) ** 2, axis=2) + 1e-14)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    das_noisy = ds.traces[das_sl]
    das_filt = sosfiltfilt(sos, das_noisy, axis=1)
    panels = [(ds.clean[das_sl], "clean (FDTD sum)"), (das_noisy, "noisy"), (das_filt, "noisy, 15-75 Hz")]
    vmax = rms_db(das_filt).max()
    for ax, (data, title) in zip(axes, panels):
        im = ax.imshow(rms_db(data), aspect="auto", origin="lower", cmap="magma", vmin=vmax - 35, vmax=vmax,
                       extent=(0, t[-1], 0, cable.length))
        ax.set_title(f"DAS: {title} [dB re 1 Pa^2]")
        ax.set_xlabel("time [s]")
    plt.colorbar(im, ax=axes, shrink=0.8)
    axes[0].set_ylabel("distance along cable [m]")
    fig.savefig(out / "m2_das.png", dpi=90, bbox_inches="tight")

    # 3. Hydrophone spectrograms
    nh = layout.n_hydrophones
    fig, axes = plt.subplots(1, nh, figsize=(4.5 * nh, 4.5), sharey=True)
    hyd = ds.traces[layout.slices["hydrophones"]]
    for i, ax in enumerate(np.atleast_1d(axes)):
        f, tt, S = spectrogram(hyd[i], fs=fs, nperseg=1024, noverlap=768)
        db = 10 * np.log10(S + 1e-14)
        m = (f > 10) & (f < 80)
        ax.pcolormesh(tt, f, db, shading="auto", vmin=np.percentile(db[m], 2), vmax=np.percentile(db[m], 99.8))
        ax.set_ylim(0, 100)
        ax.set_title(f"hydrophone {i}")
        ax.set_xlabel("time [s]")
    np.atleast_1d(axes)[0].set_ylabel("frequency [Hz]")
    fig.savefig(out / "m2_hydrophones.png", dpi=90, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

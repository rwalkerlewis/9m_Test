#!/usr/bin/env python3
"""Harbor M2: build the labelled dataset for the default scenario and plot it.

Runs one cached FDTD job per vessel (in parallel), mixes them with noise and AIS,
saves the dataset and writes overview figures plus a received-level table.

Figures, one panel each, in ``--out``: ``map.png`` (tracks and AIS reports), ``das_clean.png``, ``das_noisy.png``,
``das_noisy_band.png`` (the cable as 1 s rms), ``hydrophone_<i>_spectrogram.png``.  ``--figures-only`` redraws them
from the FDTD cache (no FDTD run; the scenario is the one in the recording's truth file) and writes nothing else.

Usage::

    python examples/harbor/01_datasets/harbor_m2_dataset.py [--config examples/configs/harbor.scenario.json]
        [--out output/recordings/harbor] [--workers 4] [--seed 0] [--time 120] [--figures-only]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figures import gathers_from_cache  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402

from acoustic_sim.water.dataset import count_cached_gathers, write_build_info  # noqa: E402
from acoustic_sim.water import (  # noqa: E402
    HarborScenarioConfig,
    assemble_dataset,
    build_environment,
    build_layout,
    build_track,
    render_vessel_gathers,
)
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402


def band_rms(x: np.ndarray, sos: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sosfiltfilt(sos, x, axis=-1) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "harbor.scenario.json"))
    ap.add_argument("--out", default="output/recordings/harbor")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--time", type=float, default=None, help="override simulated duration [s]")
    ap.add_argument("--cuda", action="store_true", help="kept for old command lines: the FDTD runs on the GPU by default")
    ap.add_argument("--cpu", action="store_true", help="run the FDTD on the CPU")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from the FDTD cache and the scenario in the truth file (no FDTD; writes only figures)")
    args = ap.parse_args()

    out = Path(args.out)
    rec = out.name
    out.mkdir(parents=True, exist_ok=True)
    if args.figures_only:                           # the scenario the recording was built with, from its truth file
        from acoustic_sim.contract import Truth, contract_paths

        cfg = HarborScenarioConfig.from_dict(Truth.load(contract_paths(out / "dataset.npz")[1]).scenario)
    else:
        cfg = HarborScenarioConfig.load(args.config)
    if args.time is not None and not args.figures_only:
        cfg.fdtd.total_time = args.time
    cfg.fdtd.use_cuda = not args.cpu
    env = build_environment(cfg)
    layout = build_layout(cfg, env)

    t0 = time.time()
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    if args.figures_only:
        gathers = gathers_from_cache(cfg, env, layout, tracks, out / "cache")
        ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.seed)
        figures(ds, env, layout, out, rec)
        return
    n_cached = count_cached_gathers(out / "cache")
    t_fdtd = time.time()
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=out / "cache", workers=args.workers)
    write_build_info(out / "build.json", cfg, env, len(gathers), time.time() - t_fdtd,
                     count_cached_gathers(out / "cache") - n_cached)
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.seed)
    print(f"simulated {len(cfg.vessels)} vessels in {time.time() - t0:.0f} s")
    ds.save(out / "dataset.npz")

    fs = ds.fs
    sos = butter(4, [15.0, 75.0], btype="bandpass", fs=fs, output="sos")
    live = ~ds.dead

    # Received level (15-75 Hz rms, whole run, live sensors) per vessel and sensor group vs noise.
    noise_only = ds.traces - ds.gain[:, None] * ds.clean
    groups = {name: np.flatnonzero(live & (np.arange(layout.n_receivers) >= sl.start) & (np.arange(layout.n_receivers) < sl.stop))
              for name, sl in layout.slices.items()}
    print("\nband-limited (15-75 Hz) rms pressure [Pa] at the sensors")
    print(f"  {'source':10s} " + " ".join(f"{n:>12s}" for n in groups))
    for vid, tr in ds.tracks.items():
        if tr.level_pa <= 0 or vid not in gathers:
            continue  # passive scatterers have no direct-path gather
        row = [band_rms(tr.level_pa * gathers[vid].traces[idx].astype(float) * ds.gain[idx, None], sos) for idx in groups.values()]
        print(f"  {vid:10s} " + " ".join(f"{r:12.4f}" for r in row))
    print(f"  {'noise':10s} " + " ".join(f"{band_rms(noise_only[idx], sos):12.4f}" for idx in groups.values()))
    print()

    figures(ds, env, layout, out, rec)
    print("wrote", out)


def figures(ds, env, layout, out: Path, rec: str) -> None:
    """The map, the cable as 1 s rms (clean, noisy, noisy in the band) and each hydrophone's spectrogram, one figure
    each."""
    fs, t = ds.fs, ds.t
    sos = butter(4, [15.0, 75.0], btype="bandpass", fs=fs, output="sos")
    cable = layout.cables[0]
    das_sl = layout.slices[cable.name]

    fig, ax = new_figure((13.0, 10.0))
    plot_harbor(env, layout, ax=ax)
    first = True
    for vid, tr in ds.tracks.items():
        xy = tr.xy[tr.active]
        label = f"{vid} ({tr.kind}, {'AIS' if tr.ais else 'no AIS'}{', stationary' if np.ptp(xy, axis=0).max() < 1.0 else ''})"
        if np.ptp(xy, axis=0).max() < 1.0:                     # a stationary object: a marker, not a line
            ax.plot(*xy[0], "s", ms=9, mec="k", label=label)
        else:
            ax.plot(*xy.T, "--" if not tr.ais else "-", lw=2, label=label)
        if tr.ais and len(ds.ais):
            r = ds.ais.for_vessel(vid)
            if len(r):
                ax.plot(*r.xy.T, "k.", ms=3, label="AIS reports" if first else None)
                first = False
    ax.plot([], [], ".", color="red", ms=6, label="every 10th cable channel")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=11)
    ax.set_title(f"{rec}: scenario map, true tracks and AIS reports")
    save_figure(fig, out / "map.png", recording=rec)

    def rms_db(data: np.ndarray, win: int = int(fs)) -> np.ndarray:
        n = data.shape[1] // win
        return 10 * np.log10(np.mean(data[:, : n * win].reshape(len(data), n, win) ** 2, axis=2) + 1e-14)

    das_noisy = ds.traces[das_sl]
    das_filt = sosfiltfilt(sos, das_noisy, axis=1)
    vmax = rms_db(das_filt).max()
    for name, data, title in (("das_clean", ds.clean[das_sl], "clean (FDTD sum)"), ("das_noisy", das_noisy, "noisy"),
                              ("das_noisy_band", das_filt, "noisy, 15 to 75 Hz")):
        fig, ax = new_figure((12.0, 8.0))
        im = ax.imshow(rms_db(data), aspect="auto", origin="lower", cmap="magma", vmin=vmax - 35, vmax=vmax,
                       extent=(0, t[-1], 0, cable.length))
        fig.colorbar(im, ax=ax, label="1 s rms pressure [dB re 1 Pa squared]")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("distance along cable [m]")
        ax.set_title(f"{rec}: DAS cable, {title}")
        save_figure(fig, out / f"{name}.png", recording=rec)

    hyd = ds.traces[layout.slices["hydrophones"]]
    for i in range(layout.n_hydrophones):
        f, tt, S = spectrogram(hyd[i], fs=fs, nperseg=1024, noverlap=768)
        db = 10 * np.log10(S + 1e-14)
        m = (f > 10) & (f < 80)
        fig, ax = new_figure((12.0, 8.0))
        pc = ax.pcolormesh(tt, f, db, shading="auto", vmin=np.percentile(db[m], 2), vmax=np.percentile(db[m], 99.8))
        fig.colorbar(pc, ax=ax, label="power spectral density [dB re 1 Pa squared per Hz]")
        ax.set_ylim(0, 100)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(f"{rec}: hydrophone {i} spectrogram")
        save_figure(fig, out / f"hydrophone_{i}_spectrogram.png", recording=rec)


if __name__ == "__main__":
    main()

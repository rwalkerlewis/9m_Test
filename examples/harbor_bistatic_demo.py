#!/usr/bin/env python3
"""Bistatic demo: a silent UUV located from the ferry's echo on the DAS cable (pitch figures).

Idealised, and said so on every figure: the ferry's direct path is removed with the dataset's
truth component (perfect cancellation), the ferry waveform and position are known (oracle
reference, exact track).  What remains is real: an FDTD simulated echo of a 1 m x 3 m soft body
on a strain DAS cable and four hydrophones, in noise, processed with the coherent bistatic
matched filter of ``water/bistatic.py``.

Usage::

    python examples/harbor_bistatic_demo.py [--dataset output/harbor_bistatic/coarse/dataset.npz]
        [--out output/harbor_bistatic/demo] [--noise 1.0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.water import HarborDataset, build_environment, build_layout  # noqa: E402
from acoustic_sim.water.bistatic import BistaticConfig, Truth, run_bistatic  # noqa: E402
from acoustic_sim.water.dataset import vessel_injection  # noqa: E402
from acoustic_sim.water.dispersion import for_scenario  # noqa: E402
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402
from acoustic_sim.water.signatures import f_max_for_grid  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="output/harbor_bistatic/coarse/dataset.npz")
    ap.add_argument("--out", default="output/harbor_bistatic/demo")
    ap.add_argument("--noise", type=float, default=1.0, help="noise scale relative to the dataset (1 = as simulated)")
    ap.add_argument("--direct", type=float, default=0.0, help="fraction of the direct path left in (0 = ideal cancellation)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = HarborDataset.load(args.dataset)
    cfg = ds.cfg
    env = build_environment(cfg)
    layout = build_layout(cfg, env)            # the saved header drops the strain mode; rebuild from the config
    f_max = f_max_for_grid(env.dx)
    comps = ds.components
    gain = ds.gain[:, None]
    traces = (gain * (comps["scattered:ferry>uuv1"].astype(float) + args.direct * comps["direct:ferry"].astype(float))
              + args.noise * comps["noise"].astype(float))
    ferry = next(v for v in cfg.vessels if v.id == "ferry")
    oracle = ferry.level_pa * vessel_injection(ferry, ds.tracks["ferry"], env, cfg)

    bc = BistaticConfig(win_s=0.5, hop_s=0.25, f_lo=100.0, bin_stride=2, grid_dx=2.0, reference="oracle",
                        removal="none", method="coherent", coherent_norm="snr", exclusion_radius=25.0, dispersion=for_scenario(env, cfg.fdtd))
    t0 = time.time()
    # Frame times are needed for the exact ship track: run the STFT geometry once via a dry config.
    from acoustic_sim.water.bistatic import stft
    sp = stft(traces, layout, ds.fs, bc, f_max)
    ship_xy = ds.tracks["ferry"].position_at(sp.times)
    run = run_bistatic(traces, layout, ds.fs, env, ds.ais, bc, f_max, oracle_signal=oracle, truth=Truth(ship_xy=ship_xy))
    print(f"chain: {time.time() - t0:.0f} s, {len(run.grid)} cells, {len(sp.times)} frames, {len(sp.freqs)} bins")

    # Per frame estimate = map peak; truth = UUV track.
    B, grid, times = run.scan.B, run.grid, run.sp.times
    uuv = ds.tracks["uuv1"].position_at(times)
    est = grid[np.argmax(B, axis=0)]
    err = np.hypot(*(est - uuv).T)
    # Three frame (1.5 s) integration of the maps: the target moves 3 m in that time, well inside a cell.
    B3 = np.zeros_like(B)
    for k in range(B.shape[1]):
        B3[:, k] = B[:, max(0, k - 1):k + 2].sum(axis=1)
    est3 = grid[np.argmax(B3, axis=0)]
    err3 = np.hypot(*(est3 - uuv).T)
    # Peak to background contrast per frame (dB): peak over the median of the map.
    contrast = 10 * np.log10(np.max(B, axis=0) / np.maximum(np.median(B, axis=0), 1e-30))
    summary = {
        "frames": int(len(times)), "window_s": bc.win_s, "band_hz": [bc.f_lo, 0.95 * f_max], "grid_m": bc.grid_dx,
        "median_error_m": float(np.median(err)), "p90_error_m": float(np.percentile(err, 90)),
        "within_5m": float(np.mean(err <= 5.0)), "within_10m": float(np.mean(err <= 10.0)),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))), "contrast_db_median": float(np.median(contrast)),
        "integrated_1p5s": {"median_error_m": float(np.median(err3)), "p90_error_m": float(np.percentile(err3, 90)),
                            "within_5m": float(np.mean(err3 <= 5.0)), "rmse_m": float(np.sqrt(np.mean(err3 ** 2)))},
        "noise_scale": args.noise, "direct_fraction": args.direct,
        "assumptions": ["direct path removed with the truth component", "oracle ferry waveform", "exact ferry track"],
    }
    print(json.dumps(summary, indent=2))
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "demo_maps.npz", B=B.astype(np.float32), grid=grid, times=times, est=est, uuv=uuv, ship=ship_xy)

    tag = "ideal direct path cancellation, known ferry waveform and track"
    xs, ys = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    ix = np.rint((grid[:, 0] - xs[0]) / bc.grid_dx).astype(int)
    iy = np.rint((grid[:, 1] - ys[0]) / bc.grid_dx).astype(int)

    def image(k):
        img = np.full((len(ys), len(xs)), np.nan)
        img[iy, ix] = 10 * np.log10(np.maximum(B[:, k], 1e-12))
        return img

    # Figure 1: six frames of the bistatic map.
    picks = np.linspace(2, len(times) - 3, 6).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, k in zip(axes.ravel(), picks):
        plot_harbor(env, layout, ax=ax, show_depth=False)
        im = ax.imshow(image(k), origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="magma", vmin=-20, vmax=0,
                       alpha=0.9, zorder=2)
        ax.plot(*ship_xy[k], "s", color="deepskyblue", ms=9, mec="k", label="ferry (AIS)", zorder=5)
        ax.plot(*uuv[k], "o", mfc="none", mec="lime", mew=2, ms=14, label="UUV truth", zorder=5)
        ax.plot(*est[k], "x", color="w", ms=10, mew=2, label="estimate", zorder=6)
        ax.set_title(f"t = {times[k]:.2f} s, error {err[k]:.1f} m")
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
        if ax is axes[0, 0]:
            ax.legend(loc="lower left", fontsize=7, framealpha=0.7)
        else:
            ax.get_legend().remove()
    fig.colorbar(im, ax=axes, shrink=0.6, label="coherent bistatic map [dB re peak]")
    fig.suptitle(f"Silent UUV located from the ferry's echo on a strain DAS cable (FDTD, {bc.f_lo:.0f}-{0.95 * f_max:.0f} Hz)\n"
                 f"{tag}", fontsize=12)
    fig.savefig(out / "demo_maps.png", dpi=110, bbox_inches="tight")

    # Figure 2: tracks and error over time.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    plot_harbor(env, layout, ax=a1, show_depth=False)
    a1.plot(*ship_xy.T, "-", color="deepskyblue", lw=3, label="ferry track (AIS illuminator)")
    a1.plot(*uuv.T, "-", color="lime", lw=3, label="UUV truth (silent, 1 m x 3 m body)")
    sc = a1.scatter(*est.T, c=times, cmap="viridis", s=28, edgecolor="k", lw=0.4, zorder=6, label="per frame estimate")
    a1.legend(loc="lower left", fontsize=8, framealpha=0.8)
    a1.set_title("Estimates per 0.5 s frame over the 20 s run")
    fig.colorbar(sc, ax=a1, shrink=0.7, label="time [s]")
    a2.plot(times, np.minimum(err, 14.5), "o-", ms=4, color="0.6", label=f"single 0.5 s frame: median {summary['median_error_m']:.1f} m, "
            f"{100 * summary['within_5m']:.0f} % within 5 m")
    a2.plot(times, np.minimum(err3, 14.5), "o-", ms=4, color="k", label=f"1.5 s integration: median {np.median(err3):.1f} m, "
            f"{100 * np.mean(err3 <= 5):.0f} % within 5 m, rmse {np.sqrt(np.mean(err3 ** 2)):.1f} m")
    a2.axhline(5.0, color="r", ls="--", label="5 m")
    a2.set_xlabel("time [s]"); a2.set_ylabel("position error [m] (clipped at 15)")
    a2.set_ylim(0, 15)
    a2.set_title("Position error per frame")
    a2.legend(fontsize=9, loc="upper right")
    fig.suptitle(f"Bistatic localisation of a silent UUV: {tag}", fontsize=12)
    fig.savefig(out / "demo_track.png", dpi=110, bbox_inches="tight")

    # Figure 3: echo and noise on the cable (what the processor sees).
    sl = layout.slices["cable1"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    sos = butter(4, [bc.f_lo, 0.95 * f_max], btype="bandpass", fs=ds.fs, output="sos")
    for ax, (key, x) in zip(axes, (("echo of the ferry from the UUV alone (truth component)", gain * comps["scattered:ferry>uuv1"].astype(float)),
                                   ("echo + noise: what the processor sees", traces))):
        y = sosfiltfilt(sos, x[sl], axis=-1)
        v = np.percentile(np.abs(sosfiltfilt(sos, traces[sl], axis=-1)), 99.5)
        ax.imshow(y[:, ::10], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, extent=(0, ds.t[-1], 0, layout.cables[0].length),
                  interpolation="nearest")
        ax.set_title(key); ax.set_xlabel("time [s]")
    axes[0].set_ylabel("distance along cable [m]")
    fig.suptitle("Strain DAS cable, 100 to 380 Hz, same colour scale: the echo is 12 dB below the noise on the median channel", fontsize=12)
    fig.savefig(out / "demo_cable.png", dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

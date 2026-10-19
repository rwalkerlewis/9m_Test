#!/usr/bin/env python3
"""Video of the single-pulse gather of ``examples/harbor_m1_uuv.py`` arriving on the cable, with the reflections off
the stationary UUV highlighted as they appear.

Same scene and FDTD runs as ``harbor_m1_uuv.py`` (one static source, one 40 Hz Ricker pulse, one stationary soft-body
UUV; one run with the body, one without).  The record is replayed in slow motion: ``--record`` simulated seconds
over ``--seconds`` of video.  As the time cursor advances,

* the gather fills in (asinh-compressed amplitude so the weak arrivals show);
* the energy scattered by the UUV (run with UUV minus run without, envelope above ``--tint-level`` of its peak)
  is tinted orange where it has already arrived, so every reflection off the UUV is marked, including the quay
  echoes it re-radiates later;
* the predicted first reflection (source -> UUV -> channel, straight paths) is drawn as it arrives and labelled.

Outputs (``--out``): m1uuv_gather.mp4, m1uuv_gather.gif, m1uuv_gather_poster.png and frames_gather/ (every 30th frame).

Usage::

    python examples/harbor_m1_uuv_video.py [--out output/harbor_m1_uuv] [--seconds 10] [--record 2.5] [--fps 30]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.water import (  # noqa: E402
    HarborFDTDSolver,
    HarborScenarioConfig,
    build_environment,
    build_layout,
)
from acoustic_sim.water.scatterer import Scatterer  # noqa: E402
from acoustic_sim.water.vessels import Track  # noqa: E402
from acoustic_sim.sources import StaticSource  # noqa: E402

# Constants shared with harbor_m1_uuv.py (inlined to avoid sibling-file import).
F0, T_PULSE = 40.0, 0.1                     # Ricker peak frequency [Hz] and emission time [s]
UUV_C = "#f97316"


def static_uuv(xy, length, width, total_time):
    """A stationary soft body at *xy* (heading +x)."""
    t = np.array([0.0, total_time])
    p = np.array([xy, xy], dtype=float)
    track = Track(id="uuv", kind="uuv", ais=False, t=t, xy=p,
                  speed=np.zeros(2), active=np.ones(2, bool), level_pa=0.0)
    return Scatterer("uuv", track, length, width, kind="soft", shape="rect")


def run(env, cfg, src, receivers, scatterers=()):
    """Run the FDTD solver and return traces."""
    solver = HarborFDTDSolver(env, cfg.fdtd, src, receivers, scatterers=scatterers)
    t0 = time.time()
    res = solver.run()
    print(f"FDTD ({'with' if scatterers else 'no'} UUV): {solver.n_steps} steps, {time.time() - t0:.1f} s")
    return res["traces"]

INK, MUTED = "#111827", "#6b7280"


def ease(u):
    u = np.clip(u, 0, 1); return u * u * (3 - 2 * u)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/harbor_m1_uuv")
    ap.add_argument("--seconds", type=float, default=10.0, help="video length [s]")
    ap.add_argument("--record", type=float, default=2.5, help="simulated seconds of record replayed")
    ap.add_argument("--hold", type=float, default=1.0, help="seconds the finished gather is held at the end")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--shore-loss", type=float, default=0.15)
    ap.add_argument("--source", type=float, nargs=2, default=[900.0, 900.0], metavar=("X", "Y"))
    ap.add_argument("--uuv", type=float, nargs=2, default=[1560.0, 980.0], metavar=("X", "Y"))
    ap.add_argument("--uuv-size", type=float, nargs=2, default=[12.0, 6.0], metavar=("LENGTH", "WIDTH"))
    ap.add_argument("--tint-level", type=float, default=0.2, help="scattered envelope above this fraction of its peak is tinted")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); t_start = time.time()

    # ---- the two FDTD runs (the record is simulated a little past the replay window) ----
    cfg = HarborScenarioConfig(); cfg.fdtd.total_time = args.record + 0.2; cfg.fdtd.shore_loss_max = args.shore_loss
    env = build_environment(cfg); layout = build_layout(cfg, env)
    fs = cfg.fdtd.fs; t = np.arange(cfg.fdtd.n_steps) / fs
    a = (np.pi * F0 * (t - T_PULSE)) ** 2; signal = (1 - 2 * a) * np.exp(-a)
    src = StaticSource(args.source[0], args.source[1], signal)
    uuv = static_uuv(args.uuv, *args.uuv_size, cfg.fdtd.total_time)
    with_uuv = layout.split(run(env, cfg, src, layout.receivers, (uuv,)))["cable1"]
    without = layout.split(run(env, cfg, src, layout.receivers))["cable1"]
    scattered = with_uuv - without
    cab = layout.cables[0]; rx = cab.positions; s_along = cab.arclength; nch = len(rx)
    c = float(np.mean(env.model.values[env.water])); S, U = np.array(args.source), np.array(args.uuv)
    t_direct = T_PULSE + np.linalg.norm(rx - S, axis=1) / c
    t_echo = T_PULSE + (np.linalg.norm(U - S) + np.linalg.norm(rx - U, axis=1)) / c

    x0 = np.abs(with_uuv).max() * 0.02
    comp = np.arcsinh(with_uuv / x0); vc = np.abs(comp).max()
    env_sc = np.abs(hilbert(scattered, axis=1)); mask = env_sc > args.tint_level * env_sc.max()
    tint = np.zeros(mask.shape + (4,)); tint[..., :3] = matplotlib.colors.to_rgb(UUV_C); tint[mask, 3] = 0.45

    # ---- figure ----
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes([0.06, 0.10, 0.91, 0.74])
    ext = (0, t[-1], s_along[0], s_along[-1])
    blank = np.full_like(comp, np.nan)
    h_g = ax.imshow(blank, aspect="auto", origin="lower", cmap="seismic", vmin=-vc, vmax=vc, extent=ext, interpolation="nearest", zorder=1)
    h_t = ax.imshow(np.zeros_like(tint), aspect="auto", origin="lower", extent=ext, interpolation="nearest", zorder=2)
    cursor = ax.axvline(0, color=INK, lw=1.5, zorder=5)
    l_dir = ax.plot([], [], "--", color="#374151", lw=1.2, alpha=0.7, zorder=4, label="direct arrival (predicted)")[0]
    l_echo = ax.plot([], [], "--", color="#7c2d12", lw=2.2, zorder=4, label="first reflection off the UUV (predicted: source -> UUV -> channel)")[0]
    ping = ax.plot([], [], "o", color=UUV_C, ms=11, mec="k", zorder=6)[0]
    ax.fill([], [], color=UUV_C, alpha=0.45, label="energy scattered by the UUV (run with UUV minus run without)")
    ax.set_xlim(0, args.record); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("time [s]", fontsize=12); ax.set_ylabel("distance along cable [m]", fontsize=12)
    fig.legend(loc="upper right", bbox_to_anchor=(0.97, 0.935), fontsize=10, frameon=False)
    j = np.argmin(t_echo); j = int(np.clip(j, 10, nch - 11))
    lab = ax.annotate("reflection off the UUV", (t_echo[j], s_along[j]), (t_echo[j] + 0.35, s_along[j] + 280), color=UUV_C, fontsize=16,
                      fontweight="bold", arrowprops=dict(arrowstyle="->", color=UUV_C, lw=2.5), zorder=7,
                      bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=UUV_C, lw=1.5)); lab.set_visible(False)
    k_dir = int(np.argmin(t_direct))
    lab_d = ax.annotate("direct arrival", (t_direct[k_dir], s_along[k_dir]), (t_direct[k_dir] + 0.18, s_along[k_dir] - 230), color="#374151",
                        fontsize=12, arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5), zorder=7,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9ca3af")); lab_d.set_visible(False)
    title = fig.text(0.06, 0.955, "", fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.905, f"One {F0:.0f} Hz pulse from ({S[0]:.0f}, {S[1]:.0f}) m; stationary UUV at ({U[0]:.0f}, {U[1]:.0f}) m; "
             f"replayed at {args.seconds / args.record:.0f}x slow motion", fontsize=12.5, color=MUTED, va="top")
    fig.text(0.97, 0.015, f"FDTD, dx = {env.dx:.0f} m, fs = {fs:.0f} Hz, shore_loss_max = {args.shore_loss}; soft rectangular UUV body "
             f"{args.uuv_size[0]:.0f} x {args.uuv_size[1]:.0f} m; amplitude asinh-compressed", ha="right", va="bottom", fontsize=9, color=MUTED)

    # ---- frames ----
    nfr = int(round(args.seconds * args.fps)); n_play = int(round((args.seconds - args.hold) * args.fps))
    t_now_all = np.minimum(args.record, args.record * np.arange(nfr) / max(1, n_play - 1))
    fdir = out / "_gather"; fdir.mkdir(exist_ok=True); keep = out / "frames_gather"; keep.mkdir(exist_ok=True)
    for f in keep.glob("*.png"): f.unlink()
    t1 = time.time(); t_first = float(t_echo.min()); t_dfirst = float(t_direct.min())
    for i, tn in enumerate(t_now_all):
        n = int(tn * fs)
        shown = blank.copy(); shown[:, :n] = comp[:, :n]; h_g.set_data(shown)
        tt = np.zeros_like(tint); tt[:, :n] = tint[:, :n]; h_t.set_data(tt)
        cursor.set_xdata([tn, tn])
        ok = t_direct <= tn; l_dir.set_data(t_direct[ok], s_along[ok])
        ok = t_echo <= tn; l_echo.set_data(t_echo[ok], s_along[ok])
        # the ping: where the first UUV echo is arriving right now
        hit = np.abs(t_echo - tn) < 0.5 / fs * 3
        if tn >= t_first and tn <= t_echo.max() and ok.any():
            kk = np.argsort(np.abs(t_echo - tn))[:2]; ping.set_data(t_echo[kk], s_along[kk]); ping.set_alpha(1.0)
        else: ping.set_data([], [])
        for L, t0, dur in ((lab, t_first, 0.4), (lab_d, t_dfirst, 0.3)):
            if tn >= t0:
                L.set_visible(True); L.set_alpha(float(ease((tn - t0) / (dur * args.record / args.seconds * 4))))
            else: L.set_visible(False)
        title.set_text(f"DAS record arriving on the cable, t = {tn:4.2f} s")
        fig.savefig(fdir / f"f{i:05d}.png", dpi=args.dpi, facecolor="white")
        if i % 30 == 0 or i == nfr - 1: shutil.copy(fdir / f"f{i:05d}.png", keep / f"gather_{i:05d}.png")
    plt.close(fig); print(f"{nfr} frames in {time.time() - t1:.0f} s", flush=True)
    mp4 = out / "m1uuv_gather.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(fdir / "f%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)], check=True)
    shutil.copy(fdir / f"f{nfr - 1:05d}.png", out / "m1uuv_gather_poster.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    "fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(out / "m1uuv_gather.gif")], check=True)
    shutil.rmtree(fdir)
    print(f"wrote {mp4} ({args.seconds:.0f} s, {args.fps} fps), m1uuv_gather.gif, m1uuv_gather_poster.png in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()

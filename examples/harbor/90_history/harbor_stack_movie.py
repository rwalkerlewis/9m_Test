#!/usr/bin/env python3
"""
Ten second movie: the echoes of a silent UUV are stacked pulse by pulse until it stands out unambiguously.

Scene, data and detector are those of ``examples/harbor/90_history/harbor_presentation.py`` (one ship radiating a train of broadband
pulses, one silent stopped UUV, the DAS cable of ``output/harbor/dataset.npz``; analytic propagation, exact removal
of the ship's direct arrival, generous target strength) with the noise raised so that one pulse alone shows nothing
and the stack is what reveals the UUV.  Each pulse's echo, back-projected over the harbor, is one noisy image; the
ship moves, the UUV does not, so only the UUV adds up in phase from pulse to pulse.

The map is normalised to its own peak every frame (the viewer sees the speckle sink and one spot stay); the
readouts give pulses stacked and peak-to-background contrast, and the detection is marked once it is unambiguous.

Usage::

    python examples/harbor/90_history/harbor_stack_movie.py [--out output/harbor_domain] [--seconds 10] [--fps 30] [--noise 3]
"""
import argparse, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "harbor" / "90_history"))
import harbor_presentation as hp  # noqa: E402
from harbor_presentation import Scene, ricker, encode, C, FS, T, UUV, CROP  # noqa: E402

BG, PANEL, INK, MUTED = "#0b1020", "#0f1730", "#f3f4f6", "#9ca3af"
SHIP_C, UUV_C, CABLE_C, LOCK_C = "#38bdf8", "#f97316", "#facc15", "#22d3ee"
LAND = ListedColormap(["#2b3345"])
VMIN_DB = -14.0                      # display floor re the frame's own peak
LOCK_DB = 20.0                       # contrast (peak over median) at which the detection is called


def per_pulse_images(sc, resid):
    """Complex back-projection of each pulse separately, (npulses, npts_water)."""
    w = ricker(np.arange(-0.06, 0.06 + 1 / FS, 1 / FS))
    mf = np.array([np.convolve(tr, w[::-1], mode="same") for tr in resid])
    env = hilbert(mf, axis=1)
    pts = sc.grid[sc.water]
    rxg = np.linalg.norm(pts[:, None, :] - sc.rx[None], axis=2)
    lxg = sc.los(pts, sc.rx)
    out = []
    for k in range(len(sc.tk)):
        rs = np.linalg.norm(pts - sc.sk[k], axis=1); ls = sc.los(sc.sk[k], pts)[0]
        idx = (sc.tk[k] + (rs[:, None] + rxg) / C) * FS
        i0 = np.clip(idx.astype(int), 0, len(sc.t) - 2); fr = idx - i0
        v = env[np.arange(sc.n)[None, :], i0] * (1 - fr) + env[np.arange(sc.n)[None, :], i0 + 1] * fr
        v[idx >= len(sc.t) - 1] = 0
        out.append(ls * np.sum(v * lxg, axis=1))
    return np.array(out)


def ease(u):
    """Smoothstep, 0..1 -> 0..1."""
    u = np.clip(u, 0, 1); return u * u * (3 - 2 * u)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "output" / "harbor" / "dataset.npz"))
    ap.add_argument("--out", default=str(ROOT / "output" / "harbor_domain"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--noise", type=float, default=3.0, help="noise std / median echo peak on the cable")
    ap.add_argument("--cache", default="", help="npz to cache the per-pulse images in (speeds up re-renders)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); t_start = time.time()

    hp.NOISE_REL = args.noise
    sc = Scene(args.dataset)
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists() and float(np.load(cache)["noise"]) == args.noise:
        I = np.load(cache)["I"]
    else:
        direct, echo, noise = sc.synth(np.random.default_rng(3))
        resid = (direct + echo + noise) - direct                   # exact direct-path removal (the cheat)
        print(f"{len(sc.tk)} pulses; synthesising per-pulse images", flush=True)
        I = per_pulse_images(sc, resid)
        if cache: np.savez(cache, I=I, noise=args.noise)
    npulse = len(I); cum = np.cumsum(I, axis=0); pts = sc.grid[sc.water]
    full = np.abs(cum[-1]) ** 2; est = pts[int(np.argmax(full))]; err = float(np.linalg.norm(est - UUV))
    print(f"final estimate {est}, error {err:.1f} m, {time.time() - t_start:.0f} s", flush=True)

    # ---- pacing: hold 1 frame, ramp to all frames, hold with the lock-on ----
    nfr = int(round(args.seconds * args.fps)); times = np.arange(nfr) / args.fps
    t_hold0, t_ramp1 = 0.8, args.seconds - 2.2
    stack = 1 + (npulse - 1) * ease((times - t_hold0) / (t_ramp1 - t_hold0))   # fractional frames stacked

    def image_at(s):
        n = int(np.floor(s)); fr = s - n
        img = cum[n - 1] if n >= 1 else 0
        if n < npulse and fr > 0: img = img + fr * I[n]
        return np.abs(img) ** 2

    # ---- figure: the map centred and large, readouts either side ----
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.weight": "bold", "axes.labelweight": "bold", "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#334155",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    MH = 0.835; MW = MH * 9 / 16; MX, MY = 0.5 - MW / 2, 0.08
    axm = fig.add_axes([MX, MY, MW, MH]); axm.set_facecolor(PANEL)
    x0, x1, y0, y1 = CROP
    him = axm.imshow(np.full(sc.gshape, np.nan), origin="lower", extent=CROP, cmap="inferno", vmin=VMIN_DB, vmax=0, zorder=2, interpolation="bicubic")
    axm.imshow(np.where(sc.land_crop, 1.0, np.nan), origin="lower", extent=CROP, cmap=LAND, vmin=0, vmax=1, zorder=3)
    axm.plot(*sc.rx.T, "-", color=CABLE_C, lw=3, zorder=4, label="DAS cable, 125 channels")
    axm.plot(*sc.ship(np.linspace(-30, T + 30, 200)).T, ":", color=SHIP_C, lw=1.5, alpha=0.6, zorder=5, label="ship track (AIS)")
    emit = axm.scatter([], [], s=26, color=SHIP_C, alpha=0.9, zorder=6, label="ship position, each frame")
    ship_m = axm.plot([], [], "^", ms=16, color=SHIP_C, mec="k", zorder=7, label="ship now")[0]
    lock_m = axm.plot([], [], "+", color=LOCK_C, ms=36, mew=4, zorder=8, label="detection")[0]
    lock_ring = axm.plot([], [], "o", ms=54, mfc="none", mec=LOCK_C, mew=2.5, zorder=8)[0]
    truth_m = axm.plot([], [], "o", ms=20, mfc="none", mec="w", mew=2, zorder=8, label="UUV, truth")[0]
    axm.set_xlim(x0, x1); axm.set_ylim(y0, y1); axm.set_aspect("equal")
    axm.set_xlabel("x (m)", fontsize=12); axm.set_ylabel("y (m)", fontsize=12); axm.tick_params(labelsize=11)
    lock_txt = axm.annotate("", est, est + np.array([70, 110]), color=LOCK_C, fontweight="bold", fontsize=18, zorder=9,
                            arrowprops=dict(arrowstyle="-", color=LOCK_C, lw=2), ha="left"); lock_txt.set_visible(False)
    fig.text(0.5, 0.965, "Stacking the echoes: a silent UUV emerges", fontsize=28, fontweight="bold", color=INK, va="top", ha="center")

    # left column: frames stacked; right column: contrast
    XL, XR, W = 0.035, 0.775, 0.175
    fig.text(XL, 0.60, "FRAMES STACKED", fontsize=13, color=MUTED, va="bottom", fontweight="bold")
    n_txt = fig.text(XL, 0.50, "", fontsize=46, fontweight="bold", color=INK, va="bottom")
    bar_ax = fig.add_axes([XL, 0.455, W, 0.026]); bar_ax.set_xlim(0, npulse); bar_ax.set_ylim(0, 1); bar_ax.axis("off")
    bar_ax.add_patch(FancyBboxPatch((0, 0.1), npulse, 0.8, boxstyle="square,pad=0", fc="#1f2a44", ec="none"))
    bar = bar_ax.barh(0.5, 0, height=0.8, color=UUV_C, left=0)[0]
    for k in range(1, npulse): bar_ax.plot([k, k], [0.1, 0.9], color=BG, lw=1.4)
    fig.text(XR, 0.60, "PEAK ABOVE\nBACKGROUND", fontsize=13, color=MUTED, va="bottom", fontweight="bold", linespacing=1.3)
    c_txt = fig.text(XR, 0.50, "", fontsize=46, fontweight="bold", color=INK, va="bottom")
    v_txt = fig.text(XR, 0.43, "", fontsize=15, fontweight="bold", color=LOCK_C, va="bottom", linespacing=1.3)
    cax = fig.add_axes([XR, 0.265, W, 0.018])
    cb = fig.colorbar(him, cax=cax, orientation="horizontal"); cb.outline.set_edgecolor("#334155")
    cax.tick_params(colors=MUTED, labelsize=10, length=2)
    fig.text(XR, 0.295, "ECHO ENERGY\ndB re frame peak", fontsize=11, color=MUTED, va="bottom", fontweight="bold", linespacing=1.3)
    h, l = axm.get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(XL - 0.012, 0.045), ncol=1, fontsize=11, frameon=False, labelcolor=INK,
               handletextpad=0.6, labelspacing=1.0, markerscale=0.5)
    fig.text(0.985, 0.012, "idealised analytic simulation", ha="right", va="bottom", fontsize=9, color=MUTED)

    fdir = out / "_reveal"; fdir.mkdir(exist_ok=True); keep = out / "frames_reveal"; keep.mkdir(exist_ok=True)
    for f in keep.glob("*.png"): f.unlink()
    t1 = time.time(); lock_frame = None
    for i, (t, s) in enumerate(zip(times, stack)):
        img = image_at(s); n_shown = min(npulse, int(np.ceil(s)))
        pk = img.max(); con = 10 * np.log10(pk / np.median(img[img > 0]))
        Z = np.full(len(sc.grid), np.nan); Z[sc.water] = 10 * np.log10(np.maximum(img, 1e-30) / pk)
        him.set_data(Z.reshape(sc.gshape))
        k_cur = n_shown - 1; ship_m.set_data([sc.sk[k_cur, 0]], [sc.sk[k_cur, 1]]); emit.set_offsets(sc.sk[:n_shown])
        bar.set_width(s); n_txt.set_text(f"{n_shown} / {npulse}"); c_txt.set_text(f"{con:.0f} dB")
        locked = con >= LOCK_DB and s >= 0.5 * npulse
        if locked and lock_frame is None: lock_frame = i
        if locked:
            p = pts[int(np.argmax(img))]; a = float(ease((i - lock_frame) / (0.6 * args.fps)))
            lock_m.set_data([p[0]], [p[1]]); lock_m.set_alpha(a); lock_ring.set_data([p[0]], [p[1]]); lock_ring.set_alpha(a)
            lock_ring.set_markersize(54 - 22 * a); truth_m.set_data([UUV[0]], [UUV[1]]); truth_m.set_alpha(0.9 * a)
            lock_txt.set_text("UUV"); lock_txt.set_visible(True); lock_txt.set_alpha(a); lock_txt.xy = p; lock_txt.set_position(p + np.array([70, 110]))
            v_txt.set_text(f"UUV detected\n{np.linalg.norm(p - UUV):.0f} m from truth"); v_txt.set_alpha(a); c_txt.set_color(LOCK_C)
        else:
            for m in (lock_m, lock_ring, truth_m): m.set_data([], [])
            lock_txt.set_visible(False); v_txt.set_text(""); c_txt.set_color(INK)
        fig.savefig(fdir / f"f{i:05d}.png", dpi=args.dpi, facecolor=BG)
        if i % 30 == 0 or i == nfr - 1: shutil.copy(fdir / f"f{i:05d}.png", keep / f"reveal_{i:05d}.png")
    plt.close(fig); print(f"{nfr} frames in {time.time() - t1:.0f} s; lock at frame {lock_frame}", flush=True)
    mp4 = out / "uuv_reveal.mp4"; encode(fdir, "f%05d.png", args.fps, mp4)
    shutil.copy(fdir / f"f{nfr - 1:05d}.png", out / "uuv_reveal_poster.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    "fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(out / "uuv_reveal.gif")], check=True)
    shutil.rmtree(fdir)
    print(f"wrote {mp4} ({args.seconds:.0f} s, {args.fps} fps), uuv_reveal.gif, uuv_reveal_poster.png in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()

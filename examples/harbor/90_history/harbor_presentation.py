#!/usr/bin/env python3
"""
Presentation demo: one ship, one silent stopped UUV, a DAS cable, in the harbor of ``output/recordings/harbor/dataset.npz``.

Built to be shown, not to be a fair test.  Deliberate simplifications (stated on the figures):

* analytic 2-D propagation (amplitude sqrt(100 m / r), delay r / c, c = 1500 m/s), land opaque by line of sight,
  no reflections;
* the ship radiates a train of broadband pulses (40 Hz Ricker, every 0.6 +- 0.2 s) so wavefronts and echoes can be seen;
* the UUV is a point scatterer with a generous target strength (-10 dB re 1 m);
* the ship's direct arrival is removed exactly (known signature and AIS track), noise is modest.

Detection: matched filter + envelope on each channel of the cleaned DAS data, then back-projection of every pulse's
echo over the harbor (delay ship -> x -> channel).  The stopped UUV adds up in one place; everything else does not.

Outputs (``output/studies/history/harbor_presentation``), one panel per figure: scene.png, gather_raw.png,
gather_ship_removed.png, detection.png, detection_zoom.png, buildup_<n>_pulses.png, storyboard_<step>.png; wavefield.mp4,
das_gather.mp4 (two panels sharing their axes) and every 10th frame of both videos in frames_wavefield/ and frames_das/.

Usage::

    python examples/harbor/90_history/harbor_presentation.py [--out output/studies/history/harbor_presentation] [--fps 30]
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import ListedColormap
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water import HarborDataset, build_environment  # noqa: E402

C, R_REF = 1500.0, 100.0
FS = 500.0
T = 12.0                                   # simulated seconds
T0 = 50.0                                  # start time on the dataset's ferry track (ship just inside the breakwater)
PULSE_DT, F0 = 0.6, 40.0                   # pulse train: 40 Hz Ricker every 0.6 s on average
PULSE_JITTER = 0.2                         # +- s; irregular spacing stops echoes of one pulse aligning with the next
UUV = np.array([1560.0, 980.0])            # silent, stopped
TS = -10.0                                 # dB re 1 m (generous)
NOISE_REL = 0.15                           # noise std / median echo peak on the cable
CROP = (900.0, 2000.0, 350.0, 1450.0)      # x0, x1, y0, y1 of the pictures
DX = 5.0                                   # picture / image grid
FRAME_DT = 0.02                            # simulated seconds per video frame

SHIP_C, UUV_C, CABLE_C = "#38bdf8", "#f97316", "#facc15"
INK, MUTED = "#111827", "#6b7280"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
REC = "harbor presentation"                 # every title names the scene


def ricker(u, f=F0):
    a = (np.pi * f * u) ** 2
    return (1 - 2 * a) * np.exp(-a)


class Scene:
    def __init__(self, dataset):
        ds = HarborDataset.load(dataset)
        self.env = build_environment(ds.cfg)
        self.track = ds.tracks["ferry"]
        cab = ds.layout.cables[0]
        self.rx = cab.positions; self.s = cab.arclength; self.n = len(self.rx)
        self.t = np.arange(int(T * FS)) / FS
        jit = np.random.default_rng(1).uniform(-PULSE_JITTER, PULSE_JITTER, 64)
        self.tk = (0.2 + np.cumsum(np.r_[0, PULSE_DT + jit]))                 # emission times, irregular
        self.tk = self.tk[self.tk < T - 0.5]
        self.sk = self.ship(self.tk)
        x0, x1, y0, y1 = CROP
        xs = np.arange(x0 + DX / 2, x1, DX); ys = np.arange(y0 + DX / 2, y1, DX)
        X, Y = np.meshgrid(xs, ys); self.grid = np.c_[X.ravel(), Y.ravel()]; self.gshape = X.shape
        self.water = self.env.is_water(self.grid)
        land = self.env.land; i0, j0 = int(y0 / self.env.dx), int(x0 / self.env.dx)
        self.land_crop = land[i0:int(y1 / self.env.dx), j0:int(x1 / self.env.dx)]

    def ship(self, t):
        return self.track.position_at(T0 + np.atleast_1d(t))

    def los(self, a, b, step=6.0):
        a = np.atleast_2d(a); b = np.atleast_2d(b); land = self.env.land; dx = self.env.dx
        ns = int(np.ceil(np.hypot(*np.ptp(np.vstack([a, b]), axis=0)) / step)) + 2
        s = np.linspace(0.02, 0.98, ns); out = np.ones((len(a), len(b)), bool)
        chunk = max(1, 3_000_000 // (len(b) * ns))
        for i in range(0, len(a), chunk):
            aa = a[i:i + chunk]
            p = aa[:, None, None, :] + (b[None, :, None, :] - aa[:, None, None, :]) * s[None, None, :, None]
            ix = np.clip((p[..., 0] / dx).astype(int), 0, land.shape[1] - 1)
            iy = np.clip((p[..., 1] / dx).astype(int), 0, land.shape[0] - 1)
            out[i:i + len(aa)] = ~land[iy, ix].any(axis=2)
        return out

    # ---------- data on the cable ----------
    def synth(self, rng):
        """Direct and echo arrivals on every channel, (n, nt) each, plus noise."""
        direct = np.zeros((self.n, len(self.t))); echo = np.zeros_like(direct)
        lu = self.los(UUV, self.rx)[0]; ru = np.linalg.norm(self.rx - UUV, axis=1)
        self.arr_echo = []
        for tk, sk in zip(self.tk, self.sk):
            r = np.linalg.norm(self.rx - sk, axis=1); l = self.los(sk, self.rx)[0]
            direct += (np.sqrt(R_REF / np.maximum(r, 1)) * l)[:, None] * ricker(self.t[None] - tk - r[:, None] / C)
            rsu = np.linalg.norm(sk - UUV); lsu = float(self.los(sk, UUV)[0, 0])
            a = np.sqrt(R_REF / rsu) * lsu * 10 ** (TS / 20) * np.sqrt(1 / np.maximum(ru, 1)) * lu
            arr = tk + (rsu + ru) / C; self.arr_echo.append(np.where(a > 0, arr, np.nan))
            echo += a[:, None] * ricker(self.t[None] - arr[:, None])
        peak = np.median(np.abs(echo).max(axis=1)[np.abs(echo).max(axis=1) > 0])
        spec = np.fft.rfft(rng.standard_normal(direct.shape), axis=1); f = np.fft.rfftfreq(len(self.t), 1 / FS)
        spec[:, (f < 10) | (f > 100)] = 0; noise = np.fft.irfft(spec, len(self.t), axis=1)
        noise *= NOISE_REL * peak / noise.std()
        return direct, echo, noise

    # ---------- detection ----------
    def backproject(self, resid, pulses=None):
        """Coherent back-projection: |sum_k sum_i a_i(t_k + (|s_k - x| + |x - r_i|) / c)|^2, a_i the analytic signal of
        the matched-filtered channel i.  Noise averages out, the stopped UUV's echoes add in phase."""
        w = ricker(np.arange(-0.06, 0.06 + 1 / FS, 1 / FS))
        mf = np.array([np.convolve(tr, w[::-1], mode="same") for tr in resid])
        env = hilbert(mf, axis=1)                                          # analytic signal: stacked coherently
        pts = self.grid[self.water]
        if not hasattr(self, "_rxg"):
            self._rxg = np.linalg.norm(pts[:, None, :] - self.rx[None], axis=2)        # (npts, n)
            self._lxg = self.los(pts, self.rx)
        img = np.zeros(len(pts), complex); pulses = range(len(self.tk)) if pulses is None else pulses
        for k in pulses:
            rs = np.linalg.norm(pts - self.sk[k], axis=1); ls = self.los(self.sk[k], pts)[0]
            idx = (self.tk[k] + (rs[:, None] + self._rxg) / C) * FS
            i0 = np.clip(idx.astype(int), 0, len(self.t) - 2); fr = idx - i0
            v = env[np.arange(self.n)[None, :], i0] * (1 - fr) + env[np.arange(self.n)[None, :], i0 + 1] * fr
            v[idx >= len(self.t) - 1] = 0
            img += ls * np.sum(v * self._lxg, axis=1)
        out = np.full(len(self.grid), np.nan); out[self.water] = np.abs(img) ** 2
        return out


# ---------- drawing helpers ----------
LAND = ListedColormap(["#9ca3af"])


def base_map(sc, ax, land_alpha=1.0):
    x0, x1, y0, y1 = CROP
    ax.imshow(np.where(sc.land_crop, 1.0, np.nan), origin="lower", extent=CROP, cmap=LAND, vmin=0, vmax=1, alpha=land_alpha, zorder=3)
    ax.plot(*sc.rx.T, "-", color=CABLE_C, lw=2.5, zorder=4, label="DAS cable (125 ch)")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xlabel("x (east) [m]"); ax.set_ylabel("y (north) [m]")


def mark(ax, ship_xy, uuv=True, trail=None, big=False):
    if trail is not None: ax.plot(*trail.T, "-", color=SHIP_C, lw=1.5, alpha=0.8, zorder=5)
    ax.plot(*ship_xy, "^", ms=13 if big else 10, color=SHIP_C, mec="k", zorder=6, label="ship (AIS)")
    if uuv: ax.plot(*UUV, "o", ms=13 if big else 10, mfc="none", mec=UUV_C, mew=2.5, zorder=6, label="UUV (silent, stopped)")


def caveat(fig, text="Analytic simulation, idealised for illustration: pulsed ship source, exact direct-path removal, strong UUV echo"):
    if fig._suptitle is not None:                          # video frames: below their running title
        fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=11, color=MUTED)
    else:                                                  # one panel figures: as the last line of the title
        ax = fig.axes[0]; ax.set_title(ax.get_title() + "\n" + text, fontweight="bold", color=INK)


def save(fig, path):
    save_figure(fig, path, recording=REC); print("wrote", path.name, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "output" / "recordings" / "harbor" / "dataset.npz"))
    ap.add_argument("--out", default=str(ROOT / "output/studies/history" / "harbor_presentation"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-video", action="store_true", help="figures only")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); t_start = time.time()
    sc = Scene(args.dataset); rng = np.random.default_rng(3)
    direct, echo, noise = sc.synth(rng)
    raw = direct + echo + noise
    resid = raw - direct                                           # exact direct-path removal (the cheat)
    print(f"{len(sc.tk)} pulses, ship {sc.sk[0].round()} -> {sc.sk[-1].round()}, UUV {UUV}", flush=True)

    # ---------------- detection ----------------
    img = sc.backproject(resid)
    w = np.isfinite(img); k = np.nanargmax(img); est = sc.grid[k]
    contrast = 10 * np.log10(img[k] / np.nanmedian(img))
    err = float(np.linalg.norm(est - UUV))
    print(f"detection: estimate {est}, error {err:.1f} m, peak {contrast:.1f} dB over the median", flush=True)
    imgs_build = {n: sc.backproject(resid, range(n)) for n in (1, 3, 8, len(sc.tk))}

    # ---------------- figure: scene ----------------
    fig, ax = new_figure((12, 11))
    ax.set_facecolor("#dbeafe"); base_map(sc, ax)
    full = sc.ship(np.linspace(-30, T + 30, 200)); ax.plot(*full.T, ":", color=SHIP_C, lw=1.5, label="ship route")
    trail = sc.ship(np.linspace(0, T, 50)); ax.plot(*trail.T, "-", color=SHIP_C, lw=4, label=f"ship track during the {T:.0f} s shown")
    mark(ax, sc.sk[0], big=True)
    ax.annotate("UUV\nsilent, stopped", UUV, UUV + [60, 90], color=UUV_C, fontweight="bold", fontsize=12, arrowprops=dict(arrowstyle="-", color=UUV_C))
    ax.annotate("ship\npulsed broadband source", sc.sk[0], sc.sk[0] + [110, -150], color="#0369a1", fontweight="bold", fontsize=12,
                arrowprops=dict(arrowstyle="-", color="#0369a1"))
    ax.annotate("DAS cable\n125 channels, 10 m", sc.rx[50], sc.rx[50] + [-260, -200], color="#a16207", fontweight="bold", fontsize=12,
                arrowprops=dict(arrowstyle="-", color="#a16207"))
    ax.set_title(f"{REC}: the scene, one ship, one silent UUV, one fiber optic cable", fontweight="bold", color=INK)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9); caveat(fig)
    save(fig, out / "scene.png")

    # ---------------- figures: gather (raw, then the ship's sound removed) ----------------
    dist = sc.s; ext = [0, T, dist[0], dist[-1]]
    v1 = np.percentile(np.abs(raw), 99.5); v2 = 0.7 * np.nanmax(np.abs(echo))
    fig, ax = new_figure((12, 9))
    h = ax.imshow(raw, aspect="auto", origin="lower", extent=ext, cmap="RdBu_r", vmin=-v1, vmax=v1, interpolation="nearest")
    ax.set_title(f"{REC}, 1. what the cable records (the ship's own sound dominates)", fontweight="bold", color=INK)
    ax.set_xlabel("time [s]"); ax.set_ylabel("distance along cable [m]")
    fig.colorbar(h, ax=ax, label="pressure [model units]"); caveat(fig)
    save(fig, out / "gather_raw.png")
    fig, ax = new_figure((12, 9))
    h = ax.imshow(resid, aspect="auto", origin="lower", extent=ext, cmap="RdBu_r", vmin=-v2, vmax=v2, interpolation="nearest")
    ax.set_title(f"{REC}, 2. ship's sound removed (known signature + AIS track), shown {v1 / v2:.0f}x brighter:\n"
                 "the curved echoes of each pulse off the UUV remain", fontweight="bold", color=INK)
    k = len(sc.arr_echo) // 2; j = int(np.nanargmin(sc.arr_echo[k]))
    ax.annotate("echo from the UUV", (sc.arr_echo[k][j], dist[j]), (sc.arr_echo[k][j] + 1.2, dist[j] + 200), color=UUV_C,
                fontweight="bold", fontsize=12, arrowprops=dict(arrowstyle="->", color=UUV_C, lw=2))
    ax.set_xlabel("time [s]"); ax.set_ylabel("distance along cable [m]")
    fig.colorbar(h, ax=ax, label="pressure [model units]"); caveat(fig)
    save(fig, out / "gather_ship_removed.png")

    # ---------------- figures: detection ----------------
    def show_img(ax, im, title, zoom=False):
        ax.set_facecolor("#0b1020")
        Z = 10 * np.log10(np.maximum(im, 1e-30) / np.nanmax(im)).reshape(sc.gshape)
        h = ax.imshow(Z, origin="lower", extent=CROP, cmap="inferno", vmin=-25, vmax=0, zorder=2)
        base_map(sc, ax)
        ax.plot(*sc.ship(np.linspace(0, T, 50)).T, "-", color=SHIP_C, lw=2, zorder=5, label="ship track")
        ax.plot(*UUV, "o", ms=16, mfc="none", mec="w", mew=2, zorder=6, label="UUV truth")
        if zoom:
            ax.set_xlim(UUV[0] - 150, UUV[0] + 150); ax.set_ylim(UUV[1] - 150, UUV[1] + 150)
        ax.set_title(f"{REC}, {title}", fontweight="bold", color=INK)
        return h
    for zoom, name, title in ((False, "detection.png", f"3. echoes from all {len(sc.tk)} pulses stacked over the harbor: one bright spot"),
                              (True, "detection_zoom.png", f"zoom: UUV found {err:.0f} m from the truth,\npeak {contrast:.0f} dB above the background")):
        fig, ax = new_figure((12, 11))
        h = show_img(ax, img, title, zoom=zoom)
        ax.plot(*est, "+", color="#22d3ee", ms=22, mew=3, zorder=7, label="detection")
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
        fig.colorbar(h, ax=ax, shrink=0.8, label="stacked echo energy [dB re peak]"); caveat(fig)
        save(fig, out / name)

    # ---------------- figures: build-up ----------------
    for n, im in imgs_build.items():
        fig, ax = new_figure((12, 11))
        kk = np.nanargmax(im); c = 10 * np.log10(im[kk] / np.nanmedian(im))
        h = show_img(ax, im, f"{n} pulse{'s' if n > 1 else ''} stacked ({n * PULSE_DT:.1f} s): peak {c:.0f} dB\n"
                             "(the UUV emerges as more echoes are stacked; the ship moves, the UUV does not)")
        ax.plot(*sc.grid[kk], "+", color="#22d3ee", ms=18, mew=2.5, zorder=7, label="strongest cell")
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
        fig.colorbar(h, ax=ax, shrink=0.8, label="stacked echo energy [dB re peak]"); caveat(fig)
        save(fig, out / f"buildup_{n}_pulses.png")

    if args.no_video:
        print(f"detection error {err:.1f} m, contrast {contrast:.1f} dB ({time.time() - t_start:.0f} s)"); return

    # ---------------- wavefield video (two maps of the same area, read together) ----------------
    pts = sc.grid; lu = sc.los(UUV, pts)[0]; ru = np.linalg.norm(pts - UUV, axis=1)
    pulses = []
    for tk, sk in zip(sc.tk, sc.sk):
        r = np.linalg.norm(pts - sk, axis=1); rsu = np.linalg.norm(sk - UUV); lsu = float(sc.los(sk, UUV)[0, 0])
        pulses.append(dict(tk=tk, r=r, a=np.sqrt(R_REF / np.maximum(r, 1)) * sc.los(sk, pts)[0],
                           te=tk + rsu / C, ae=np.sqrt(R_REF / rsu) * lsu * 10 ** (TS / 20) * np.sqrt(1 / np.maximum(ru, 1)) * lu))
    rmax = np.hypot(CROP[1] - CROP[0], CROP[3] - CROP[2])
    def field(t):
        d = np.zeros(len(pts)); e = np.zeros(len(pts))
        for p in pulses:
            if 0 <= t - p["tk"] <= rmax / C + 0.1: d += p["a"] * ricker(t - p["tk"] - p["r"] / C)
            if 0 <= t - p["te"] <= rmax / C + 0.1: e += p["ae"] * ricker(t - p["te"] - ru / C)
        return d.reshape(sc.gshape), e.reshape(sc.gshape)
    times = np.arange(0, T, FRAME_DT)
    comp = lambda x, v: np.sign(x) * np.sqrt(np.abs(x) / v)            # square-root compression for display
    vd = 0.6; ve = np.sqrt(R_REF / 400) * 10 ** (TS / 20) * np.sqrt(1 / 150)
    keep = out / "frames_wavefield"; keep.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.5), sharex=True, sharey=True)
    ims = []
    for ax, title in zip(axs, ("total pressure: the ship's pulses dominate", f"UUV echo only, shown {vd / ve:,.0f}x amplified")):
        ax.set_facecolor("#0b1020")
        ims.append(ax.imshow(np.zeros(sc.gshape), origin="lower", extent=CROP, cmap="RdBu_r", vmin=-1, vmax=1, zorder=2))
        base_map(sc, ax); ax.set_title(f"{REC}: {title}", fontweight="bold", color=INK)
    ship_m = [ax.plot([], [], "^", ms=12, color=SHIP_C, mec="k", zorder=6, label="ship (AIS)")[0] for ax in axs]
    trail_m = [ax.plot([], [], "-", color=SHIP_C, lw=1.5, zorder=5)[0] for ax in axs]
    for ax in axs: ax.plot(*UUV, "o", ms=12, mfc="none", mec=UUV_C, mew=2.5, zorder=6, label="UUV (silent, stopped)")
    axs[0].legend(loc="lower right", fontsize=11, framealpha=0.9)
    ttl = fig.suptitle("", fontweight="bold", fontsize=15, color=INK, x=0.05, ha="left"); caveat(fig)
    t1 = time.time()
    writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    with writer.saving(fig, str(out / "wavefield.mp4"), dpi=90):
        for i, t in enumerate(times):
            d, e = field(t)
            ims[0].set_data(np.clip(comp(d + e, vd), -1, 1)); ims[1].set_data(np.clip(comp(e, ve), -1, 1))
            s = sc.ship(t)[0]; tr = sc.ship(np.linspace(max(0, t - 6), t, 20))
            for m, l in zip(ship_m, trail_m): m.set_data([s[0]], [s[1]]); l.set_data(*tr.T)
            ttl.set_text(f"Wavefield, t = {t:5.2f} s    (the ship pulses about every {PULSE_DT} s; the UUV re-radiates each pulse)")
            writer.grab_frame()
            if i % 10 == 0: save_figure(fig, keep / f"wavefield_{i:05d}.png", allow_two_panels=True, recording=REC, close=False)
    plt.close(fig); print(f"wavefield frames: {len(times)} in {time.time() - t1:.0f} s; wrote wavefield.mp4", flush=True)

    # ---------------- DAS gather video (raw and ship removed, sharing the cable axis) ----------------
    keep = out / "frames_das"; keep.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.5), sharey=True)
    gims = []
    for ax, (data, v, title) in zip(axs, ((raw, v1, "raw DAS: the ship dominates"), (resid, v2, f"ship removed, {v1 / v2:.0f}x brighter: UUV echoes"))):
        gims.append(ax.imshow(np.full_like(data, np.nan), aspect="auto", origin="lower", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v,
                              interpolation="nearest"))
        ax.set_xlabel("time [s]"); ax.set_ylabel("distance along cable [m]"); ax.set_xlim(0, T)
        ax.set_title(f"{REC}: {title}", fontweight="bold", color=INK)
        gims.append(ax.axvline(0, color="k", lw=1.2, label="now"))
    axs[0].legend(loc="upper right", fontsize=11)
    ttl = fig.suptitle("", fontweight="bold", fontsize=15, color=INK, x=0.04, ha="left"); caveat(fig)
    t1 = time.time()
    writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    with writer.saving(fig, str(out / "das_gather.mp4"), dpi=90):
        for i, t in enumerate(times):
            n = int(t * FS)
            for (img_h, cur), data in zip(((gims[0], gims[1]), (gims[2], gims[3])), (raw, resid)):
                shown = np.full_like(data, np.nan); shown[:, :n] = data[:, :n]; img_h.set_data(shown); cur.set_xdata([t, t])
            ttl.set_text(f"DAS recording as it arrives, t = {t:5.2f} s")
            writer.grab_frame()
            if i % 10 == 0: save_figure(fig, keep / f"das_{i:05d}.png", allow_two_panels=True, recording=REC, close=False)
    plt.close(fig); print(f"DAS frames: {len(times)} in {time.time() - t1:.0f} s; wrote das_gather.mp4", flush=True)

    # ---------------- storyboard: one figure per step ----------------
    t_echo = pulses[len(pulses) // 2]["te"] + 0.18                    # an echo ring well clear of the UUV
    d, e = field(t_echo)
    fig, ax = new_figure((12, 11))
    ax.set_facecolor("#dbeafe"); base_map(sc, ax); mark(ax, sc.sk[0], trail=sc.ship(np.linspace(0, T, 50)), big=True)
    ax.set_title(f"{REC}, 1. ship (AIS) and a silent, stopped UUV", fontweight="bold"); ax.legend(loc="lower right", fontsize=11)
    caveat(fig); save(fig, out / "storyboard_1_scene.png")
    fig, ax = new_figure((12, 11))
    ax.set_facecolor("#0b1020")
    h = ax.imshow(np.clip(comp(e, ve), -1, 1), origin="lower", extent=CROP, cmap="RdBu_r", vmin=-1, vmax=1, zorder=2); base_map(sc, ax)
    mark(ax, sc.ship(t_echo)[0]); ax.set_title(f"{REC}, 2. the UUV re-radiates each pulse (echo only, {vd / ve:.0f}x amplified)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    fig.colorbar(h, ax=ax, shrink=0.8, label="pressure, square root compressed [re display scale]")
    caveat(fig); save(fig, out / "storyboard_2_echo.png")
    fig, ax = new_figure((12, 9))
    h = ax.imshow(resid, aspect="auto", origin="lower", extent=ext, cmap="RdBu_r", vmin=-v2, vmax=v2, interpolation="nearest")
    ax.set_xlabel("time [s]"); ax.set_ylabel("distance along cable [m]")
    ax.set_title(f"{REC}, 3. the cable hears the echoes once the ship's own sound is removed", fontweight="bold")
    fig.colorbar(h, ax=ax, label="pressure [model units]")
    caveat(fig); save(fig, out / "storyboard_3_gather.png")
    fig, ax = new_figure((12, 11))
    h = show_img(ax, img, f"4. stacking the echoes finds the UUV\n{err:.0f} m error, peak {contrast:.0f} dB above the background")
    ax.plot(*est, "+", color="#22d3ee", ms=24, mew=3, zorder=7, label="detection")
    ax.legend(loc="lower right", fontsize=11)
    fig.colorbar(h, ax=ax, shrink=0.8, label="stacked echo energy [dB re peak]")
    caveat(fig); save(fig, out / "storyboard_4_detection.png")

    res = {"pulses": int(len(sc.tk)), "pulse_interval_s": PULSE_DT, "pulse_jitter_s": PULSE_JITTER, "ricker_hz": F0, "duration_s": T, "uuv_xy": UUV.tolist(),
           "uuv_ts_db": TS, "noise_over_median_echo_peak": NOISE_REL, "estimate_xy": est.tolist(), "error_m": err,
           "peak_over_median_db": contrast, "frames": int(len(times)), "fps": args.fps, "runtime_s": round(time.time() - t_start)}
    (out / "results.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

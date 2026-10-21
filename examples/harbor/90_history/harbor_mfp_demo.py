#!/usr/bin/env python3
"""
Simplified harbor matched field processing demo (analytic forward model).

Constant sound speed, flat 2-D harbor channel, two DAS fiber runs (cos^2 axial directivity), one AIS
ferry that is both the loud interferer and the illuminator, three fixed pier scatterers, and a hidden
UUV.  The replica model is the forward model, so the processing works by construction: this is the
reference the FDTD harbor chain (``src/acoustic_sim/water``, ``bistatic.py``) is measured against.

Two ways to find the UUV, each in the chain AIS -> refine ferry -> project ferry + piers out -> scan:

* passive, motor ON: Bartlett at the UUV's 400 Hz thruster line;
* bistatic, motor OFF: coherent matched filter of the ferry's echo off the UUV, with the ferry's own
  spectrum estimated by beamforming at the refined ferry position.

Figures, a 20 s track, an occupancy filter and Pd against target level go to ``output/studies/history/harbor_mfp_demo``
(``results.json`` holds the numbers).

Usage::

    python examples/harbor/90_history/harbor_mfp_demo.py [--out output/studies/history/harbor_mfp_demo] [--trials 20] [--windows 40]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]

# ---------------- geometry ----------------
C = 1500.0                       # m/s
LX, LY = 1500.0, 300.0           # channel length, width (m)
DEPTH = 16.0
GRID = 10.0
gx = np.arange(GRID/2, LX, GRID); gy = np.arange(GRID/2, LY, GRID)
GX, GY = np.meshgrid(gx, gy)     # (ny, nx)
cells = np.c_[GX.ravel(), GY.ravel()]           # (ncell, 2)
NC = len(cells)

# fiber 1: diagonal channel crossing; fiber 2: short run along the north seawall. 2 m channel spacing
f0, f1 = np.array([500.0, 0.0]), np.array([1000.0, 300.0])
s0, s1 = np.array([1050.0, 300.0]), np.array([1350.0, 300.0])
def line(a, b):
    n = int(np.linalg.norm(b - a) / 2.0)
    pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
    return pts, np.tile((b - a) / np.linalg.norm(b - a), (n, 1))
c1, t1 = line(f0, f1); c2, t2 = line(s0, s1)
chan = np.vstack([c1, c2]); tang = np.vstack([t1, t2]); nch = len(chan); n1 = len(c1)

# clutter: fixed scatterers on pier faces (x, y, TS dB)
clutter = np.array([[300.0, 0.0, 0.0], [900.0, 300.0, 0.0], [1200.0, 300.0, 0.0]])

# ---------------- replica model ----------------
def spread(r):
    r = np.maximum(r, 1.0)
    return np.where(r < DEPTH, 1.0 / r, 1.0 / (np.sqrt(DEPTH) * np.sqrt(r)))

def replica_tables(pts, ch=None):
    """pts (n,2) -> delay (n,nch), gain (n,nch) including DAS cos^2 directivity and spreading.
    ch selects a subset of channels (default all)."""
    xc, tc = (chan, tang) if ch is None else (chan[ch], tang[ch])
    d = pts[:, None, :] - xc[None, :, :]
    r = np.linalg.norm(d, axis=2)
    cosang = np.einsum('ijk,jk->ij', d, tc) / np.maximum(r, 1e-6)
    gain = (cosang**2) * spread(r)
    return r / C, gain

D_cell, G_cell = replica_tables(cells)

# ---------------- band and snapshots ----------------
FS = 2000.0
TSNAP = 0.5                       # snapshot length (s)
NSNAP = 11                        # 3 s window, 50 % overlap
freqs = np.arange(40.0, 800.0, 20.0)   # 38 bins
NF = len(freqs)
FI_TONE = [int(np.argmin(np.abs(freqs - 400)))]   # UUV thruster line

def db2amp(db): return 10 ** (db / 20)

def ferry_spec(f):   # source level per Hz, dB re 1 uPa at 1 m, broadband with blade rate lines
    base = 160.0 - 6 * np.log2(np.maximum(f, 100) / 100)
    lines = np.zeros_like(f)
    for h in range(1, 20):
        lines += 12 * np.exp(-0.5 * ((f - 12 * h) / 2.0) ** 2)
    return base + lines

def uuv_spec(f):     # tonal thruster, 400 Hz and harmonic
    s = np.full_like(f, -np.inf)
    for fh, lv in ((400.0, 135.0), (800.0, 115.0)):
        s = np.logaddexp(s * np.log(10) / 20, (lv - 60 * np.abs(f - fh) / 20) * np.log(10) / 20) * 20 / np.log(10)
    return s

NOISE_DB = 65.0  # harbor ambient per Hz, quiet night
UUV_TS = -20.0
UUV_LINE_DB = 135.0   # 400 Hz thruster line level of uuv_spec

def synth(ferry_xy, uuv_xy, uuv_motor, snap_seed, ts=UUV_TS, motor_db=0.0, fr=None, nsnap=NSNAP, ch=None, parts=False):
    """Data d[nsnap, nf, nch] (complex pressure per Hz) for one window with the geometry frozen.
    ts: UUV target strength (dB, -inf for no UUV echo); motor_db: offset on the UUV thruster spectrum.
    parts=True returns the components (ferry, clutter, echo, motor, noise) instead of their sum."""
    fr = freqs if fr is None else fr
    r = np.random.default_rng(snap_seed)
    w = 2j * np.pi * fr
    def H(pt):                                        # point source -> channels, (nf, nch)
        D, G = replica_tables(np.atleast_2d(pt), ch)
        return G[0][None, :] * np.exp(-w[:, None] * D[0][None, :])
    src = db2amp(ferry_spec(fr))[None, :] * np.exp(1j * r.uniform(0, 2 * np.pi, (nsnap, len(fr))))
    def echo(pt, tsdb):                               # bistatic echo: ferry -> pt -> channels
        rs = np.linalg.norm(ferry_xy - pt)
        return (src * spread(rs) * db2amp(tsdb) * np.exp(-w * rs / C))[:, :, None] * H(pt)[None]
    p = {"ferry": src[:, :, None] * H(ferry_xy)[None],
         "clutter": sum(echo(c[:2], c[2]) for c in clutter)}
    zero = np.zeros_like(p["ferry"])
    p["echo"] = echo(uuv_xy, ts) if np.isfinite(ts) else zero
    if uuv_motor:
        Su = db2amp(uuv_spec(fr) + motor_db)[None, :] * np.exp(1j * r.uniform(0, 2 * np.pi, (nsnap, len(fr))))
        p["motor"] = Su[:, :, None] * H(uuv_xy)[None]
    else:
        p["motor"] = zero
    p["noise"] = db2amp(NOISE_DB) * (r.standard_normal(zero.shape) + 1j * r.standard_normal(zero.shape)) / np.sqrt(2)
    return p if parts else sum(p.values())

# ---------------- processors ----------------
def steer(D, fidx, sign):
    """Yield (fi, exp(sign 2 pi i freqs[fi] D)); by recurrence when the bins are evenly spaced (much cheaper than exp)."""
    fidx = list(fidx); f = freqs[fidx]
    if len(f) > 2 and np.allclose(np.diff(f), f[1] - f[0]):
        E = np.exp(sign * 2j * np.pi * f[0] * D); step = np.exp(sign * 2j * np.pi * (f[1] - f[0]) * D)
        for fi in fidx:
            yield fi, E
            E = E * step
    else:
        for fi in fidx:
            yield fi, np.exp(sign * 2j * np.pi * freqs[fi] * D)

def bartlett(d, D, G, fidx=None):
    """Bartlett MFP: sum_f sum_snap |w^H d|^2 / |w|^2.  d (NSNAP,NF,nch). returns (ncell,)"""
    fidx = range(NF) if fidx is None else fidx
    P = np.zeros(len(D)); wn = np.sum(G**2, axis=1)
    for fi, E in steer(D, fidx, 1):
        W = G * E                                            # conj replica (ncell,nch)
        y = W @ d[:, fi, :].T                                # (ncell, NSNAP)
        P += np.sum(np.abs(y) ** 2, axis=1) / wn
    return P / (len(fidx) * d.shape[0])

def mvdr(d, D, G, fidx=None, load=0.05):
    fidx = range(NF) if fidx is None else fidx
    P = np.zeros(len(D)); m = d.shape[2]; wn = np.sum(G**2, axis=1)
    for fi, E in steer(D, fidx, -1):
        X = d[:, fi, :]                                      # (NSNAP,nch)
        K = X.T @ X.conj() / X.shape[0]
        K += load * np.trace(K).real / m * np.eye(m)
        Ki = np.linalg.inv(K)
        W = G * E                                            # replica (ncell,nch)
        q = np.einsum("ij,jk,ik->i", W.conj(), Ki, W).real
        P += wn / q
    return P / len(fidx)

def bistatic(d_raw, d_res, D, G, s_pt, pts=cells, fidx=None, coherent=True):
    """Bistatic MFP, coherent over frequency (coherent=False: power summed over frequency, which has no
    delay ambiguity from the 20 Hz bin spacing but a broader focal spot).
    Replica for cell x = (illuminator spectrum, measured by beamforming at s) x path s->x x path x->channel.
    d_raw: data with illuminator present (for the reference), d_res: data with direct path and clutter removed.
    Normalised by the expected noise power of each cell, so the map is an SNR map."""
    fidx = range(NF) if fidx is None else fidx
    Ds, Gs = replica_tables(s_pt[None]); Ds, Gs = Ds[0], Gs[0]
    rsx = np.linalg.norm(pts - s_pt, axis=1)                                      # |s - x| per cell
    tsx = rsx / C; gsx = spread(rsx)
    Z = np.zeros((len(D), d_raw.shape[0]), complex); Pi = np.zeros(len(D)); norm = np.zeros(len(D))
    for fi, E in steer(D, fidx, 1):
        ws = Gs * np.exp(-2j * np.pi * freqs[fi] * Ds)
        r = (d_raw[:, fi, :] @ ws.conj()) / np.sum(Gs**2)                      # (NSNAP,) source spectrum estimate
        W = G * E                                                               # conj replica x->channel (ncell,nch)
        y = W @ d_res[:, fi, :].T                                               # (ncell,NSNAP)
        path = gsx * np.exp(-2j * np.pi * freqs[fi] * tsx)                      # s->x (ncell,)
        z = y * np.conj(path)[:, None] * np.conj(r)[None, :]
        if coherent: Z += z
        else: Pi += np.sum(np.abs(z) ** 2, axis=1)
        norm += np.sum(G**2, axis=1) * (gsx**2) * np.mean(np.abs(r) ** 2)
    if not coherent: return Pi / (norm * d_raw.shape[0])
    return np.sum(np.abs(Z) ** 2, axis=1) / (norm * d_raw.shape[0]) / len(fidx)

def patch(center, half, step):
    xs = np.arange(center[0] - half, center[0] + half + 1e-6, step)
    ys = np.arange(center[1] - half, center[1] + half + 1e-6, step)
    XX, YY = np.meshgrid(xs, ys); return np.c_[XX.ravel(), YY.ravel()]

def refine(d, guess, half=16.0):
    """Fine Bartlett search around an AIS guess: 2 m over +-half, then 0.25 m over +-2 m."""
    best = np.asarray(guess, float)
    for h, step in ((half, 2.0), (2.0, 0.25)):
        pts = patch(best, h, step)
        D, G = replica_tables(pts)
        best = pts[np.argmax(bartlett(d, D, G, fidx=range(0, NF, 3)))]
    return best

def project_out(d, pts_list, half=0.75, n=3):
    """Jointly remove everything coherent with replicas in small patches around each point in pts_list
    (one subspace projection per frequency; joint so the removals do not interfere)."""
    pts = []
    for pt in np.atleast_2d(pts_list):
        xs = np.linspace(pt[0] - half, pt[0] + half, n); ys = np.linspace(pt[1] - half, pt[1] + half, n)
        XX, YY = np.meshgrid(xs, ys); pts.append(np.c_[XX.ravel(), YY.ravel()])
    pts = np.vstack(pts)
    D, G = replica_tables(pts)
    out = d.copy()
    for fi in range(NF):
        W = (G * np.exp(-2j * np.pi * freqs[fi] * D)).T          # (nch, npts)
        Q, _ = np.linalg.qr(W)
        out[:, fi, :] -= (d[:, fi, :] @ Q.conj()) @ Q.T
    return out

def todb(P): return 10 * np.log10(np.maximum(P, 1e-30))

EXCL = 60.0   # cells this close to the refined ferry are not searched (residual of the illuminator)

def peaks(P, k, sep, pts=cells):
    """Points of the k largest values of P at least sep metres apart."""
    out = []
    for i in np.argsort(P)[::-1][:3000]:
        if all(np.linalg.norm(pts[i] - q) > sep for q in out): out.append(pts[i])
        if len(out) == k: break
    return out

_FINE = []
def fine_tables(step=2.5, chunk=12000):
    """2.5 m grid replica tables in chunks (float32), built on first use: the single bin passive map has
    fringes a few metres apart that the 10 m grid misses."""
    if not _FINE:
        xs = np.arange(step / 2, LX, step); ys = np.arange(step / 2, LY, step)
        XX, YY = np.meshgrid(xs, ys); pts = np.c_[XX.ravel(), YY.ravel()]
        for i in range(0, len(pts), chunk):
            D, G = replica_tables(pts[i:i + chunk]); _FINE.append((pts[i:i + chunk], D.astype(np.float32), G.astype(np.float32)))
    return _FINE

def process(d, ais, mode, ncand=8):
    """AIS -> refined ferry -> ferry and stored piers projected out -> map, peak, contrast.
    mode 'passive' (Bartlett at 400 Hz) or 'bistatic' (coherent echo matched filter).
    The 10 m map undersamples the focal spot (wavelength 2 to 4 m; coherent bistatic range resolution c / 760 Hz
    = 2 m) and the coherent bistatic map has delay grating lobes every c / 20 Hz = 75 m of path, so the map only
    proposes candidates: bistatic, the ncand largest peaks of it and of the incoherent map (which has no grating
    lobes); passive, those of a 2.5 m map.  Each candidate is searched at 1 m, climbed at 1 m until it stops moving, then refined at 0.25 m.
    The statistic is the best peak over the median of the 10 m map."""
    ref = refine(d, ais)
    dres = project_out(d, np.vstack([ref, clutter[:, :2]]))
    def scan(pts, D=None, G=None, coherent=True):
        D, G = replica_tables(pts) if D is None else (D, G)
        v = bartlett(dres, D, G, fidx=FI_TONE) if mode == "passive" else bistatic(d, dres, D, G, ref, pts, coherent=coherent)
        v[np.linalg.norm(pts - ref, axis=1) < EXCL] = 0
        return v
    m = np.linalg.norm(cells - ref, axis=1) < EXCL
    P = scan(cells, D_cell, G_cell)
    bg = np.median(P[~m]); P[m] = bg
    if mode == "bistatic":
        cand = peaks(P, ncand, GRID) + peaks(scan(cells, D_cell, G_cell, coherent=False), ncand, GRID)
    else:
        F = fine_tables(); pf = np.vstack([f[0] for f in F])
        cand = peaks(np.concatenate([scan(*f) for f in F]), ncand, GRID, pf)
    best, est = -np.inf, None
    for c in cand:
        pts = patch(c, 0.7 * GRID, 1.0); v = scan(pts)
        e = pts[np.argmax(v)]
        for _ in range(10):                                  # climb: the focal spot may sit outside the patch
            pts = patch(e, 3.0, 1.0); v = scan(pts); e_new = pts[np.argmax(v)]
            if np.allclose(e_new, e): break
            e = e_new
        if v.max() > best: best, est = v.max(), e
    pts = patch(est, 1.0, 0.25); v = scan(pts)
    return {"ref": ref, "P": P, "est": pts[np.argmax(v)], "contrast": float(todb(max(best, v.max()) / bg))}

# ---------------- plotting helpers ----------------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
INK, MUTED = "#1f2937", "#6b7280"
FERRY_C, UUV_C = "#1d4ed8", "#ff7f0e"

REC = "analytic channel"          # every title names the scene


def harbor_axes(ax, title):
    ax.set_facecolor("#dbe4ee")
    ax.add_patch(Rectangle((0, 0), LX, LY, fill=False, lw=1.2, ec=INK))
    ax.plot(c1[:, 0], c1[:, 1], color="#111111", lw=2.2, label="DAS fiber"); ax.plot(c2[:, 0], c2[:, 1], color="#111111", lw=2.2)
    for cx, cy, _ in clutter:
        ax.plot(cx, cy, "s", ms=7, mfc="#9ca3af", mec=INK, label="fixed scatterer (pier)")
    ax.set_xlim(-20, LX + 20); ax.set_ylim(-30, LY + 30); ax.set_aspect("equal")
    ax.set_xlabel("x (east) [m]"); ax.set_ylabel("y (north) [m]"); ax.set_title(f"{REC}: {title}")
    ax.tick_params(colors=MUTED)

def show_map(ax, P, title, ferry, uuv, refined=None, ais=None, est=None, vmin=-30, cmap="viridis"):
    shape = GY.shape if len(P) == NC else (int(round(LY / 2.5)), int(round(LX / 2.5)))   # 10 m or 2.5 m grid
    Z = todb(P).reshape(shape); Z -= Z.max()
    im = ax.imshow(Z, origin="lower", extent=[0, LX, 0, LY], cmap=cmap, vmin=vmin, vmax=0, aspect="equal")
    ax.plot(c1[:, 0], c1[:, 1], color="white", lw=1.5, label="DAS fiber"); ax.plot(c2[:, 0], c2[:, 1], color="white", lw=1.5)
    for cx, cy, _ in clutter: ax.plot(cx, cy, "s", ms=6, mfc="none", mec="white", label="pier")
    ax.plot(*ferry, "^", ms=10, mfc="none", mec="white", mew=1.5, label="ferry (true)")
    ax.plot(*uuv, "o", ms=12, mfc="none", mec=UUV_C, mew=1.8, label="UUV (true)")
    if ais is not None: ax.plot(*ais, "x", ms=8, color="white", mew=1.5, label="ferry AIS")
    if refined is not None: ax.plot(*refined, "+", ms=12, color="#ffd60a", mew=2, label="ferry refined")
    if est is not None: ax.plot(*est, "x", ms=9, color="#ff2d55", mew=2, label="UUV estimate")
    ax.set_xlim(0, LX); ax.set_ylim(0, LY)
    ax.set_title(f"{REC}: {title}"); ax.set_xlabel("x (east) [m]"); ax.set_ylabel("y (north) [m]")
    ax.tick_params(colors=MUTED)
    return im

def dedup_legend(ax, **kw):
    h, l = ax.get_legend_handles_labels(); seen = {}
    for hh, ll in zip(h, l): seen.setdefault(ll, hh)
    kw.setdefault("frameon", False)
    ax.legend(seen.values(), seen.keys(), fontsize=11, **kw)

def save(fig, out, name):
    save_figure(fig, Path(out) / name, recording=REC); print("wrote", name)

def map_figure():
    """A figure for a map of the 1500 m x 300 m channel (wide and short)."""
    return new_figure((15, 6))

# ---------------- scene ----------------
ferry0 = np.array([600.0, 150.0]); ferry_v = np.array([4.0, 0.0])      # 8 kn down channel
uuv0 = np.array([850.0, 40.0]);   uuv_v = np.array([0.0, 1.0])         # 2 kn across
AIS_SIGMA = 6.0                                                        # m, per axis
ASSUMPTIONS = ["analytic forward model: the replica is exact (constant c, no multipath, known DAS directivity)",
               "geometry frozen inside each 3 s window (11 x 0.5 s snapshots)",
               "pier scatterer positions known (stored clutter map) and projected out",
               "ferry position from AIS (6 m rms per axis) refined by a Bartlett search on the data",
               "ferry waveform estimated by beamforming at the refined ferry position (estimated from the traces)"]


def fig_scene(out, T):
    fig, ax = map_figure()
    harbor_axes(ax, "1500 m x 300 m channel, constant c = 1500 m/s, 16 m deep, two DAS fiber runs")
    t = np.linspace(0, T, 50)
    ax.plot(ferry0[0] + ferry_v[0] * t, ferry0[1] + ferry_v[1] * t, "-", color=FERRY_C, lw=2, label=f"ferry track, 8 kn, AIS ({T:.0f} s)")
    ax.plot(*ferry0, "^", ms=9, color=FERRY_C, label="ferry start")
    ax.plot(uuv0[0] + uuv_v[0] * t, uuv0[1] + uuv_v[1] * t, "-", color=UUV_C, lw=2, label="UUV track, 2 kn, hidden")
    ax.plot(*uuv0, "o", ms=8, color=UUV_C, label="UUV start")
    ax.text(10, LY - 22, f"fiber 1: {n1} channels, {np.linalg.norm(f1-f0):.0f} m crossing\n"
            f"fiber 2: {nch-n1} channels, {np.linalg.norm(s1-s0):.0f} m along the seawall\n2 m channel spacing, cos$^2$ axial directivity",
            fontsize=11, color=INK, va="top")
    dedup_legend(ax, loc="lower right", ncol=2)
    save(fig, out, "map.png")


def fig_gather(out):
    """2 s of DAS data at full resolution (the total, the UUV echo alone and echo + noise: one figure each), and the
    levels per channel."""
    T = 2.0; fr = np.arange(40.0, 800.0, 1 / T); nfft = int(FS * T)
    p = synth(ferry0, uuv0, True, 3, fr=fr, nsnap=1, parts=True)
    def ts(x):
        spec = np.zeros((nfft // 2 + 1, nch), complex); spec[np.rint(fr * T).astype(int)] = x[0]
        return np.fft.irfft(spec, n=nfft, axis=0)
    tot = sum(p[k] for k in ("ferry", "clutter", "echo", "noise"))
    lev = {k: 10 * np.log10(np.mean(np.abs(v[0]) ** 2, axis=0) + 1e-30) for k, v in p.items()}
    v = np.percentile(np.abs(ts(tot)), 99); ve = np.percentile(np.abs(ts(p["echo"] + p["noise"])), 99)
    for x, s, name, tag in ((tot, v, "motor OFF, everything on the fiber", "total"),
                            (p["echo"], ve, f"UUV echo alone (scale x{v / ve:.0f})", "echo"),
                            (p["echo"] + p["noise"], ve, "UUV echo + noise, ferry and piers perfectly removed (echo scale)", "echo_noise")):
        fig, ax = new_figure((12, 8))
        im = ax.imshow(ts(x).T, aspect="auto", cmap="RdBu_r", vmin=-s, vmax=s, extent=[0, T, 0, nch], origin="lower", interpolation="nearest")
        ax.axhline(n1, color="k", lw=0.8, ls="--", label="fiber 1 | fiber 2")
        ax.set_xlabel("time [s]"); ax.set_ylabel("DAS channel (fiber 1 | fiber 2)"); ax.tick_params(colors=MUTED)
        ax.set_title(f"{REC}: 2 s of DAS data, {name}")
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax, label="strain rate [model units]")
        save(fig, out, f"gather_{tag}.png")
    fig, ax = new_figure((13, 8))
    for k, col, lab in (("ferry", FERRY_C, "ferry direct"), ("clutter", "#6b7280", "pier echoes"), ("echo", UUV_C, "UUV echo (TS -20 dB)"),
                        ("motor", "#c2410c", "UUV thruster (motor ON)"), ("noise", "k", "ambient noise")):
        ax.plot(lev[k], color=col, lw=1.3 if k != "noise" else 0.8, ls="--" if k == "motor" else "-", label=lab)
    ax.axvline(n1, color="k", lw=0.8, ls="--")
    gap = np.median(lev["noise"] - lev["echo"])
    ax.set_xlim(0, nch); ax.set_xlabel("DAS channel"); ax.set_ylabel("mean level per Hz [dB]"); ax.tick_params(colors=MUTED)
    ax.set_title(f"{REC}: levels per channel, 40 to 800 Hz: the UUV echo is {gap:.0f} dB below the noise on the median channel\n"
                 f"and {np.median(lev['ferry'] - lev['echo']):.0f} dB below the ferry (dips: cos$^2$ broadside nulls)")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    save(fig, out, "gather_levels.png")
    return {"echo_below_noise_db_median_channel": float(gap), "echo_below_ferry_db_median_channel": float(np.median(lev["ferry"] - lev["echo"])),
            "thruster_vs_noise_db_median_channel": float(np.median(lev["motor"] - lev["noise"]))}


def fig_snapshots(out):
    """Impulse response of the scene: envelope of a ferry pulse and of the echoes it raises, at three instants
    (one figure per instant and content, same dB scale)."""
    fr = np.arange(40.0, 800.0, 0.5); S = np.hanning(len(fr) + 2)[1:-1]; S /= S.sum()
    tp = np.arange(-0.05, 0.05, 1e-4); a = np.exp(2j * np.pi * tp[:, None] * fr[None, :]) @ S   # analytic pulse
    def pulse(t):                                       # a(t), zero outside the tabulated support
        return np.interp(t, tp, a.real, left=0, right=0) + 1j * np.interp(t, tp, a.imag, left=0, right=0)
    xs = np.arange(0, LX + 1, 2.0); ys = np.arange(0, LY + 1, 2.0); XX, YY = np.meshgrid(xs, ys); pts = np.c_[XX.ravel(), YY.ravel()]
    rf = np.linalg.norm(pts - ferry0, axis=1)
    scat = [np.array([*uuv0, UUV_TS]), *clutter]
    times = (0.12, 0.24, 0.36)
    rows = [[], []]
    for t in times:
        e = np.zeros(len(pts), complex)
        for sx, sy, tsdb in scat:
            rs = np.linalg.norm(ferry0 - [sx, sy]); rx = np.linalg.norm(pts - [sx, sy], axis=1)
            e += spread(rs) * db2amp(tsdb) * spread(rx) * pulse(t - (rs + rx) / C)
        dfield = spread(rf) * pulse(t - rf / C)
        rows[0].append(np.abs(dfield + e).reshape(XX.shape)); rows[1].append(np.abs(e).reshape(XX.shape))
    ref = max(r.max() for r in rows[0])
    for i, (lab, tag) in enumerate((("everything", "all"), ("echoes only", "echoes"))):
        for j, t in enumerate(times):
            fig, ax = map_figure()
            im = ax.imshow(20 * np.log10(rows[i][j] / ref + 1e-12), origin="lower", extent=[0, LX, 0, LY], cmap="magma", vmin=-80, vmax=0)
            ax.plot(c1[:, 0], c1[:, 1], color="w", lw=0.8, label="DAS fiber"); ax.plot(c2[:, 0], c2[:, 1], color="w", lw=0.8)
            ax.plot(*ferry0, "^", ms=7, mfc="none", mec="#60a5fa", mew=1.5, label="ferry")
            ax.plot(*uuv0, "o", ms=9, mfc="none", mec=UUV_C, mew=1.5, label="UUV (TS -20 dB)")
            for cx, cy, _ in clutter: ax.plot(cx, cy, "s", ms=5, mfc="none", mec="w", label="pier (TS 0 dB)")
            ax.set_xlim(0, LX); ax.set_ylim(0, LY)
            ax.set_xlabel("x (east) [m]"); ax.set_ylabel("y (north) [m]")
            ax.set_title(f"{REC}: impulse response at t = {t:.2f} s, {lab}\n(a 40 to 800 Hz pulse from the ferry; same dB scale in every figure)")
            dedup_legend(ax, loc="upper right", labelcolor="white")
            fig.colorbar(im, ax=ax, label="envelope [dB re direct peak]", shrink=0.45)
            save(fig, out, f"snapshots_{tag}_{t:.2f}s.png")


def fig_single_window(out, rng):
    """One window: passive maps (ferry, then the UUV line) and bistatic maps (with and without the piers), one figure each."""
    ferry, uuv = ferry0.copy(), uuv0.copy()
    ais = ferry + rng.normal(0, AIS_SIGMA, 2)
    d_on = synth(ferry, uuv, True, 1); d_off = synth(ferry, uuv, False, 1)
    ref = refine(d_on, ais)
    F = fine_tables(); pf = np.vstack([f[0] for f in F])
    def fine(fn, ref_pt):                               # map on the 2.5 m grid for display, illuminator masked
        P = np.concatenate([fn(*f) for f in F]); m = np.linalg.norm(pf - ref_pt, axis=1) < EXCL
        P[m] = np.median(P[~m]); return P
    res = {"ais_error_m": float(np.linalg.norm(ais - ferry)), "refined_error_m": float(np.linalg.norm(ref - ferry))}

    def one(P, title, name, label, **kw):
        fig, ax = map_figure()
        im = show_map(ax, P, title, ferry, uuv, **kw)
        dedup_legend(ax, loc="upper right", labelcolor="white")
        fig.colorbar(im, ax=ax, label=label, shrink=0.45)
        save(fig, out, name)

    one(bartlett(d_on, D_cell, G_cell),
        f"passive Bartlett, 40 to 800 Hz: the ferry dominates\nAIS error {res['ais_error_m']:.1f} m, refined {res['refined_error_m']:.2f} m",
        "passive_maps_broadband.png", "map [dB re peak]", refined=ref, ais=ais)
    pas = process(d_on, ais, "passive")
    dres = project_out(d_on, np.vstack([pas["ref"], clutter[:, :2]]))
    one(fine(lambda x, D, G: bartlett(dres, D, G, FI_TONE), pas["ref"]),
        f"passive Bartlett at 400 Hz, ferry and piers projected out (motor ON)\nUUV error {np.linalg.norm(pas['est'] - uuv):.1f} m",
        "passive_maps_400hz.png", "map [dB re peak]", est=pas["est"])
    res["passive"] = {"error_m": float(np.linalg.norm(pas["est"] - uuv)), "contrast_db": pas["contrast"]}

    ref_off = refine(d_off, ais)
    m = np.linalg.norm(cells - ref_off, axis=1) < EXCL
    dres = project_out(d_off, ref_off)
    P_raw = fine(lambda x, D, G: bistatic(d_off, dres, D, G, ref_off, x), ref_off)
    k = int(np.argmax(P_raw))
    bis = process(d_off, ais, "bistatic")
    dres = project_out(d_off, np.vstack([bis["ref"], clutter[:, :2]]))
    P_b = fine(lambda x, D, G: bistatic(d_off, dres, D, G, bis["ref"], x), bis["ref"])
    one(P_raw, f"bistatic (coherent, 40 to 800 Hz), motor OFF, ferry removed\nstrongest echo is a pier at "
        f"({pf[k][0]:.0f}, {pf[k][1]:.0f}) m", "bistatic_maps_ferry_removed.png", "SNR map [dB re peak]", refined=ref_off)
    one(P_b, f"bistatic, pier clutter also removed: UUV error {np.linalg.norm(bis['est'] - uuv):.1f} m,\n"
        f"peak {bis['contrast']:.0f} dB over the map median", "bistatic_maps_piers_removed.png", "SNR map [dB re peak]",
        refined=bis["ref"], est=bis["est"])
    res["bistatic"] = {"error_m": float(np.linalg.norm(bis["est"] - uuv)), "contrast_db": bis["contrast"],
                       "peak_before_pier_removal_xy": pf[k].tolist()}
    return res


def fig_bartlett_vs_mvdr(out):
    """Bartlett and MVDR on the ferry, with the exact replica and with a 0.7 % sound speed error in the replica (one
    figure per processor and replica).  MVDR needs more snapshots than channels: a 44 channel sub-array (every 10th
    channel) and 60 snapshots (0.05 s each in the same 3 s window)."""
    ch = np.arange(0, nch, 10); fsub = range(0, NF, 4)
    d = synth(ferry0, uuv0, False, 5, nsnap=60, ch=ch)
    pts = patch(ferry0, 40.0, 0.5); D, G = replica_tables(pts, ch); n = int(round(np.sqrt(len(pts))))
    ext = [pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()]
    c_mis = 1490.0
    res = {"mismatch_c_m_s": c_mis}
    for i, (Dr, tag) in enumerate(((D, "exact replica"), (D * C / c_mis, f"c = {c_mis:.0f} m/s replica"))):
        for j, (fn, name) in enumerate(((bartlett, "Bartlett"), (mvdr, "MVDR, 5 % loading"))):
            P = fn(d, Dr, G, fsub)
            if i == 0: res.setdefault("_peak", {})[name] = P.max()
            loss = todb(P.max() / res["_peak"][name])
            Z = todb(P).reshape(n, n); Z -= Z.max()
            fig, ax = new_figure((11, 9.5))
            im = ax.imshow(Z, origin="lower", extent=ext, cmap="viridis", vmin=-20, vmax=0)
            ax.contour(Z, levels=[-3], colors="w", linewidths=1, extent=ext)
            ax.plot([], [], "-", color="w", lw=1, label="-3 dB contour")
            ax.plot(*ferry0, "^", ms=10, mfc="none", mec="white", mew=1.5, label="ferry (true)")
            pk = pts[np.argmax(P)]; ax.plot(*pk, "x", ms=8, color="#ff2d55", mew=2, label="peak")
            area = float(np.sum(Z > -3)) * 0.25; err = float(np.linalg.norm(pk - ferry0))
            key = ("bartlett" if j == 0 else "mvdr") + ("" if i == 0 else "_mismatch")
            res[key] = {"area_3db_m2": area, "peak_error_m": err, "peak_loss_db": float(loss)}
            a_txt = f"{area:.0f} m$^2$" if area >= 1 else "< 1 m$^2$"
            ax.set_title(f"{REC}: ferry focal spot, {name}, {tag}\n-3 dB area {a_txt}, peak error {err:.1f} m, peak {loss:+.1f} dB "
                         "(44 channels, 60 snapshots)")
            ax.set_xlabel("x (east) [m]"); ax.set_ylabel("y (north) [m]"); ax.tick_params(colors=MUTED)
            ax.legend(loc="upper right", facecolor="0.3", labelcolor="white")
            fig.colorbar(im, ax=ax, label="map [dB re own peak]")
            save(fig, out, f"bartlett_vs_mvdr_{key}.png")
    del res["_peak"]
    return res


def run_track(out, nw, dt, rng):
    """nw windows, hop dt: passive (motor ON) and bistatic (motor OFF) per window, occupancy over the bistatic maps."""
    O = np.full(NC, 1.0 / NC); snaps = {}; rows = []
    for k in range(nw):
        t = k * dt; ferry = ferry0 + ferry_v * t; uuv = uuv0 + uuv_v * t
        ais = ferry + rng.normal(0, AIS_SIGMA, 2)
        pas = process(synth(ferry, uuv, True, 1000 + k), ais, "passive")
        bis = process(synth(ferry, uuv, False, 100 + k), ais, "bistatic")
        Pb = bis["P"]; m = np.linalg.norm(cells - bis["ref"], axis=1) < EXCL
        # likelihood ratio: map power relative to the noise distribution of the map
        z = todb(Pb); z = (z - np.median(z[~m])) / (1.4826 * np.median(np.abs(z[~m] - np.median(z[~m]))))
        L = np.exp(np.clip(0.15 * z, -3, 4))
        # predict: diffuse with 1 m/s max speed
        O = gaussian_filter(O.reshape(GY.shape), sigma=1.0 * dt / GRID + 0.3).ravel()
        O = O * L; O /= O.sum()
        occ = cells[np.argmax(O)]
        rows.append({"t": t, "ferry": ferry, "uuv": uuv, "ref": bis["ref"], "pas": pas["est"], "bis": bis["est"], "occ": occ,
                     "pas_c": pas["contrast"], "bis_c": bis["contrast"]})
        if k in (0, 4, 9, nw - 1): snaps[k] = (O.copy(), ferry.copy(), uuv.copy())
        print(f"  window {k + 1}/{nw}: passive {np.linalg.norm(pas['est'] - uuv):5.1f} m, bistatic {np.linalg.norm(bis['est'] - uuv):5.1f} m, "
              f"occupancy {np.linalg.norm(occ - uuv):5.1f} m")
    A = lambda key: np.array([r[key] for r in rows])
    t, uuv, ferry = A("t"), A("uuv"), A("ferry")
    err = {k: np.linalg.norm(A(k) - uuv, axis=1) for k in ("pas", "bis", "occ")}
    ferr = np.linalg.norm(A("ref") - ferry, axis=1)

    for k, (Ok, fk, uk) in snaps.items():
        fig, ax = map_figure()
        Z = Ok.reshape(GY.shape); Z = Z / Z.max()
        im = ax.imshow(Z, origin="lower", extent=[0, LX, 0, LY], cmap="magma", vmin=0, vmax=1)
        ax.plot(c1[:, 0], c1[:, 1], color="white", lw=1, label="DAS fiber"); ax.plot(c2[:, 0], c2[:, 1], color="white", lw=1)
        ax.plot(uuv[:k + 1, 0], uuv[:k + 1, 1], "-", color=UUV_C, lw=1.2, label="UUV track so far")
        ax.plot(*uk, "o", ms=9, mfc="none", mec=UUV_C, mew=1.8, label="UUV (true)")
        ax.plot(*fk, "^", ms=8, mfc="none", mec="white", mew=1.2, label="ferry (true)")
        ax.set_title(f"{REC}: occupancy over the bistatic maps after window {k+1} (t = {k * dt:.1f} s), motor OFF\n"
                     f"MAP error {err['occ'][k]:.1f} m (10 m cells, 1 m/s diffusion)")
        ax.set_xlim(0, LX); ax.set_ylim(0, LY); ax.tick_params(colors=MUTED)
        ax.set_xlabel("x (east) [m]"); ax.set_ylabel("y (north) [m]")
        dedup_legend(ax, loc="upper right", labelcolor="white")
        fig.colorbar(im, ax=ax, label="occupancy [re peak]", shrink=0.45)
        save(fig, out, f"occupancy_window{k + 1}.png")

    fig, a1 = new_figure((13, 9))
    pad = 60
    lo = np.minimum(uuv.min(0), np.vstack([A("pas"), A("bis")]).min(0)) - pad; hi = np.maximum(uuv.max(0), np.vstack([A("pas"), A("bis")]).max(0)) + pad
    a1.set_facecolor("#dbe4ee"); a1.plot(c1[:, 0], c1[:, 1], color="#111111", lw=2, label="DAS fiber")
    a1.plot(*uuv.T, "-", color=UUV_C, lw=3, label="UUV truth")
    a1.plot(*A("pas").T, "o", ms=4, mfc="none", mec="#7c3aed", label="passive, 400 Hz (motor ON)")
    a1.plot(*A("bis").T, "x", ms=5, color="#047857", label="bistatic (motor OFF)")
    a1.plot(*A("occ").T, "-", color="k", lw=1, label="occupancy MAP (bistatic)")
    a1.set_xlim(lo[0], hi[0]); a1.set_ylim(max(lo[1], 0), min(hi[1], LY)); a1.set_aspect("equal")
    a1.set_xlabel("x (east) [m]"); a1.set_ylabel("y (north) [m]"); a1.tick_params(colors=MUTED); a1.legend(frameon=False, loc="upper right")
    a1.set_title(f"{REC}: UUV estimates per window ({dt:.1f} s hop, 3 s windows)")
    save(fig, out, "tracks_map.png")
    fig, a2 = new_figure((13, 8))
    for k, col, lab in (("pas", "#7c3aed", "passive"), ("bis", "#047857", "bistatic"), ("occ", "k", "occupancy MAP")):
        a2.plot(t, np.minimum(err[k], 29.5), "o-", ms=3, lw=1, color=col, label=f"{lab}: median {np.median(err[k]):.1f} m, "
                f"{100 * np.mean(err[k] <= 10):.0f} % within 10 m")
    a2.plot(t, np.minimum(ferr, 29.5), ":", color=FERRY_C, lw=1, label=f"ferry refined vs truth: median {np.median(ferr):.2f} m")
    a2.axhline(10, color="r", ls="--", lw=0.8, label="10 m")
    a2.set_ylim(0, 30); a2.set_xlabel("time [s]"); a2.set_ylabel("position error [m], clipped at 30"); a2.tick_params(colors=MUTED)
    a2.legend(frameon=False, loc="upper right")
    a2.set_title(f"{REC}: position error per window (occupancy: 10 m cells)")
    save(fig, out, "tracks_error.png")
    summ = lambda e: {"median_m": float(np.median(e)), "p90_m": float(np.percentile(e, 90)), "within_10m": float(np.mean(e <= 10)),
                      "rmse_m": float(np.sqrt(np.mean(e ** 2)))}
    return {"windows": nw, "hop_s": dt, "passive": summ(err["pas"]), "bistatic": summ(err["bis"]), "occupancy_map": summ(err["occ"]),
            "occupancy_final_error_m": float(err["occ"][-1]), "ferry_refined_error_median_m": float(np.median(ferr)),
            "contrast_db_median": {"passive": float(np.median(A("pas_c"))), "bistatic": float(np.median(A("bis_c")))}}


def run_pd(out, ntrial, rng, pfa=0.05, gate=15.0):
    """Pd against UUV level from random scenes.  The threshold on the peak contrast is set from scenes with no UUV
    (Pfa per window); a detection also needs the peak within `gate` m of the UUV.  The passive sweep leaves the UUV's
    echo of the ferry out (thruster only), so it measures the passive mechanism alone.  Per mode: the Pd curve
    (``pd_vs_level_<mode>.png``) and where the scenes were (``pd_vs_level_<mode>_scenes.png``)."""
    scenes = []
    for i in range(2 * ntrial):
        ferry = ferry0 + ferry_v * rng.uniform(0, 20)
        while True:
            uuv = np.array([rng.uniform(100, LX - 100), rng.uniform(10, LY - 10)])
            if np.linalg.norm(uuv - ferry) > 100: break
        scenes.append((ferry, uuv, ferry + rng.normal(0, AIS_SIGMA, 2), 5000 + i))
    sweeps = {"bistatic": {"levels": np.arange(-60.0, -9.0, 5.0), "label": "UUV target strength [dB]", "default": UUV_TS},
              "passive": {"levels": np.arange(85.0, 141.0, 5.0), "label": "UUV 400 Hz line level [dB re 1 uPa at 1 m, per Hz]", "default": UUV_LINE_DB}}
    res = {"pfa": pfa, "gate_m": gate, "trials_per_level": ntrial, "noise_trials": len(scenes)}
    for mode, sw in sweeps.items():
        t0 = time.time()
        def one(sc, level):
            ferry, uuv, ais, seed = sc
            if mode == "bistatic": d = synth(ferry, uuv, False, seed, ts=level)
            else: d = synth(ferry, uuv, level is not None, seed, ts=-np.inf, motor_db=0.0 if level is None else level - UUV_LINE_DB)
            return process(d, ais, mode)
        h0 = np.array([one(sc, -np.inf if mode == "bistatic" else None)["contrast"] for sc in scenes])
        thr = float(np.quantile(h0, 1 - pfa))
        pd, med = [], []
        for lv in sw["levels"]:
            r = [one(sc, lv) for sc in scenes[:ntrial]]
            e = np.array([np.linalg.norm(x["est"] - sc[1]) for x, sc in zip(r, scenes)])
            det = (np.array([x["contrast"] for x in r]) > thr) & (e <= gate)
            pd.append(float(det.mean())); med.append(float(np.median(e[det])) if det.any() else None)
            if lv == sw["default"]: det_default = det
        print(f"  {mode}: threshold {thr:.1f} dB, Pd {np.round(pd, 2).tolist()} ({time.time() - t0:.0f} s)")
        res[mode] = {"threshold_contrast_db": thr, "noise_contrast_db_median": float(np.median(h0)),
                     "levels": sw["levels"].tolist(), "pd": pd, "median_error_detected_m": med, "default_level": sw["default"]}
        what = "motor OFF, ferry echo" if mode == "bistatic" else "motor ON, 400 Hz line, no echo"
        fig, ax = new_figure((12, 8))
        ax.plot(sw["levels"], pd, "o-", color=INK, label="Pd")
        ax.axvline(sw["default"], color=UUV_C, ls="--", lw=1, label=f"scene default ({sw['default']:.0f})")
        ax.axhline(pfa, color=MUTED, ls=":", lw=1, label=f"Pfa {pfa:.2f} (threshold {thr:.1f} dB over median)")
        ax.set_ylim(-0.03, 1.03); ax.set_xlabel(sw["label"]); ax.set_ylabel("Pd (fraction of scenes)"); ax.tick_params(colors=MUTED)
        ax.set_title(f"{REC}, {mode} ({what}): Pd against UUV level, {ntrial} random scenes per level\n"
                     f"(peak contrast above the no UUV threshold and within {gate:.0f} m)")
        ax.legend(frameon=False, loc="center right")
        save(fig, out, f"pd_vs_level_{mode}.png")
        # where the scenes were, detected or missed at the default level
        fig, axm = map_figure()
        axm.set_facecolor("#dbe4ee"); axm.plot(c1[:, 0], c1[:, 1], color="#111111", lw=2, label="DAS fiber")
        axm.plot(c2[:, 0], c2[:, 1], color="#111111", lw=2)
        for cx, cy, _ in clutter: axm.plot(cx, cy, "s", ms=6, mfc="#9ca3af", mec=INK, label="pier")
        for (ferry, uuv, _, _), ok in zip(scenes, det_default):
            axm.plot([ferry[0], uuv[0]], [ferry[1], uuv[1]], "-", color=MUTED, lw=0.5)
            axm.plot(*ferry, "^", ms=5, color=FERRY_C, label="ferry")
            axm.plot(*uuv, "o" if ok else "X", ms=7, color="#047857" if ok else "#dc2626", mec="k", mew=0.4,
                     label="UUV detected" if ok else "UUV missed")
        axm.set_xlim(0, LX); axm.set_ylim(0, LY); axm.set_aspect("equal"); axm.tick_params(colors=MUTED)
        axm.set_xlabel("x (east) [m]"); axm.set_ylabel("y (north) [m]")
        axm.set_title(f"{REC}, {mode}: scenes at the default level, {det_default.sum()} detected, {(~det_default).sum()} missed")
        dedup_legend(axm, loc="upper right", frameon=True)
        save(fig, out, f"pd_vs_level_{mode}_scenes.png")
        res[mode]["missed_at_default_uuv_xy"] = [sc[1].tolist() for sc, ok in zip(scenes, det_default) if not ok]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "output/studies/history" / "harbor_mfp_demo"))
    ap.add_argument("--windows", type=int, default=40, help="windows in the track run (0.5 s hop)")
    ap.add_argument("--trials", type=int, default=20, help="random scenes per level in the Pd sweep (0 skips it)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed); t0 = time.time()
    dt = 0.5
    res = {"scene": {"channel_m": [LX, LY], "depth_m": DEPTH, "c_m_s": C, "channels": nch, "band_hz": [float(freqs[0]), float(freqs[-1])],
                     "bins": NF, "snapshots": NSNAP, "snapshot_s": TSNAP, "grid_m": GRID, "noise_db_per_hz": NOISE_DB,
                     "uuv_ts_db": UUV_TS, "uuv_line_db": UUV_LINE_DB, "ais_sigma_m": AIS_SIGMA},
           "assumptions": ASSUMPTIONS}
    fig_scene(out, args.windows * dt)
    res["levels"] = fig_gather(out)
    fig_snapshots(out)
    res["single_window"] = fig_single_window(out, rng)
    res["resolution"] = fig_bartlett_vs_mvdr(out)
    print("track run")
    res["track"] = run_track(out, args.windows, dt, rng)
    if args.trials:
        print("Pd sweep")
        res["pd"] = run_pd(out, args.trials, rng)
    res["runtime_s"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("single_window", "track")}, indent=2))
    print(f"done in {res['runtime_s']:.0f} s ->", out)


if __name__ == "__main__":
    main()

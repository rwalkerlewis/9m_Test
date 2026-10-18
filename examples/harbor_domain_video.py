#!/usr/bin/env python3
"""
Videos of the wavefield and of the DAS gather in the harbor domain of ``output/harbor/dataset.npz``.

A copy of ``examples/harbor_mfp_domain.py`` (the analytic matched-field test in the dataset's domain) with movies:
the same scene (ferry, tug and moored ship with AIS, the hidden UUV, their 120 s tracks and source signatures from
the dataset's generator, the 125 channel cable and 4 hydrophones) and the same propagation (2-D free field,
amplitude sqrt(100 m / r), delay r / c, c = 1500 m/s, land opaque by line of sight, no reflections), now in the time
domain:

* ``wavefield.mp4``: the pressure over the whole harbor (2.5 m grid), one large panel;
* ``das_gather.mp4``: the 125 cable channels (signals x the dataset's DAS gains + the dataset's own noise, dead
  channels flat, 12 to 78 Hz), a scrolling window, one large panel;
* every Nth frame of both as PNG (``frames_wavefield/``, ``frames_das/``).

Scenario ``passive`` (default, the dataset's): the UUV radiates its 0.03 Pa at 100 m signature.  ``bistatic``: the UUV
is silent and scatters the three ships (TS re 1 m, 2-D).  ``--process`` also runs the matched-field chain of the
original example into the same output directory (about an hour).

Usage::

    python examples/harbor_domain_video.py [--t0 40] [--duration 15] [--every 10] [--scenario passive]
        [--out output/harbor_domain_video] [--process]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from acoustic_sim.harbor import HarborDataset, build_environment  # noqa: E402
from acoustic_sim.harbor.plotting import plot_harbor  # noqa: E402
from acoustic_sim.harbor.signatures import R_REF, f_max_for_grid, make_reference_signal  # noqa: E402

C = 1500.0
TSNAP, HOP, NSNAP = 1.0, 0.5, 5          # 1 s Hann snapshots, 50 % overlap: 3 s windows, 1 Hz bins
F_LO, F_HI = 12.0, 78.0                  # the dataset's band
GRID = 10.0
EXCL = 60.0                              # not searched this close to a ship (its residual)
AZ_DB_ABOVE = 10 ** 0.6                   # plane-wave interference: azimuth scan peaks 6 dB above its median
RANK_REL = 0.03                          # projection subspace: singular values above 3 % of the largest per point
SHIPS = ("ferry", "tug", "moored")
UUV = "uuv1"
UUV_TS = -20.0
# stored clutter: pier heads and breakwater heads at the entrance (water side), TS 0 dB
CLUTTER = np.array([[815.0, 1172.0], [1515.0, 1192.0], [1262.0, 520.0], [1440.0, 520.0]])
INK, MUTED = "#1f2937", "#6b7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


class Domain:
    """Geometry, signals and noise of the dataset, and the analytic forward model on it."""

    def __init__(self, path):
        self.ds = ds = HarborDataset.load(path)
        self.env = build_environment(ds.cfg)
        self.rx = ds.layout.receivers; self.m = len(self.rx); self.n_das = ds.layout.n_das
        self.fs = ds.fs; self.n = ds.traces.shape[1]; self.T = self.n / self.fs
        self.nseg = int(round(TSNAP * self.fs)); self.win = np.hanning(self.nseg + 2)[1:-1]
        f = np.fft.rfftfreq(self.nseg, 1 / self.fs); self.bins = np.where((f >= F_LO) & (f <= F_HI))[0]
        self.freqs = f[self.bins]; self.nf = len(self.freqs)
        t = np.arange(self.n) / self.fs
        f_max = min(f_max_for_grid(self.env.dx), 0.45 * self.fs)
        self.sig = {v.id: v.level_pa * make_reference_signal(v.kind, ds.tracks[v.id].speed_at(t), ds.tracks[v.id].active_at(t),
                                                           self.fs, v.seed, f_max) for v in ds.cfg.vessels}
        self.level = {v.id: v.level_pa for v in ds.cfg.vessels}
        self.noise = ds.traces.astype(float) - ds.gain[:, None] * ds.clean.astype(float)
        self.gain, self.dead = ds.gain, ds.dead
        xs = np.arange(GRID / 2, 2400, GRID); ys = np.arange(GRID / 2, 1800, GRID)
        X, Y = np.meshgrid(xs, ys); allc = np.c_[X.ravel(), Y.ravel()]
        self.water = self.env.is_water(allc); self.cells = allc[self.water]; self.shape = X.shape
        self.ais = {v: ds.ais.for_vessel(v) for v in SHIPS}

    # --- propagation ---
    def los(self, a, b, step=8.0):
        """(len(a), len(b)) True where the straight path a -> b stays in water (land is opaque)."""
        a = np.atleast_2d(a); b = np.atleast_2d(b); land = self.env.land; dx = self.env.dx
        out = np.ones((len(a), len(b)), bool)
        ns = int(np.ceil(np.hypot(*(np.ptp(np.vstack([a, b]), axis=0))) / step)) + 2
        s = np.linspace(0.02, 0.98, ns)                          # ends excluded: sensors and sources sit near walls
        chunk = max(1, 4_000_000 // (len(b) * ns))
        for i in range(0, len(a), chunk):
            aa = a[i:i + chunk]
            p = aa[:, None, None, :] + (b[None, :, None, :] - aa[:, None, None, :]) * s[None, None, :, None]
            ix = np.clip((p[..., 0] / dx).astype(int), 0, land.shape[1] - 1)
            iy = np.clip((p[..., 1] / dx).astype(int), 0, land.shape[0] - 1)
            out[i:i + len(aa)] = ~land[iy, ix].any(axis=2)
        return out

    def tables(self, pts, body=False):
        """Delay (n, m) and gain (n, m) from points to the receivers; body=True: 2-D spreading re 1 m."""
        r = np.linalg.norm(pts[:, None, :] - self.rx[None], axis=2)
        g = np.sqrt((1.0 if body else R_REF) / np.maximum(r, 1.0)) * self.los(pts, self.rx)
        return r / C, g

    # --- signals ---
    def snap_times(self, tc):
        return tc + (np.arange(NSNAP) - (NSNAP - 1) / 2) * HOP

    def stft(self, x, tc):
        """(NSNAP, nf, ...) spectra of x (n, ...) or (..., n) at the snapshots of the window centred at tc."""
        out = []
        for ts in self.snap_times(tc):
            i0 = int(round((ts - TSNAP / 2) * self.fs))
            seg = x[..., i0:i0 + self.nseg] * self.win
            out.append(np.fft.rfft(seg, axis=-1)[..., self.bins] / self.win.sum())
        out = np.array(out)                                       # (NSNAP, ..., nf)
        return np.moveaxis(out, -1, 1)

    def truth(self, vid, t):
        return self.ds.tracks[vid].position_at(np.atleast_1d(t))[0]

    def synth(self, tc, uuv_level=None, ts=-np.inf, parts=False):
        """Data (NSNAP, nf, m) for the window centred at tc.  uuv_level: radiated level [Pa at 100 m] (None: the
        dataset level, 0: silent); ts: UUV target strength [dB re 1 m] for the echoes of the three ships."""
        w = 2j * np.pi * self.freqs
        def H(pt, body=False):
            D, G = self.tables(np.atleast_2d(pt), body); return G[0][None] * np.exp(-w[:, None] * D[0][None])
        S = {v: self.stft(self.sig[v], tc) for v in self.sig}       # (NSNAP, nf)
        uuv = self.truth(UUV, tc)
        p = {"ships": sum(S[v][:, :, None] * H(self.truth(v, tc))[None] for v in SHIPS)}
        lv = self.level[UUV] if uuv_level is None else uuv_level
        p["uuv"] = (lv / self.level[UUV]) * S[UUV][:, :, None] * H(uuv)[None]
        Hb = H(uuv, body=True); p["echo"] = np.zeros_like(p["ships"]); p["clutter"] = np.zeros_like(p["ships"])
        for v in SHIPS:
            xv = self.truth(v, tc)
            for pt, a, key in [(uuv, 10 ** (ts / 20), "echo")] + [(c, 1.0, "clutter") for c in CLUTTER]:
                if a == 0: continue
                rs = np.linalg.norm(xv - pt); lo = float(self.los(xv, pt)[0, 0])
                p[key] += (S[v] * np.sqrt(R_REF / max(rs, 1.0)) * lo * a * np.exp(-w * rs / C))[:, :, None] * \
                          (Hb if key == "echo" else H(pt, body=True))[None]
        clean = p["ships"] + p["uuv"] + p["echo"] + p["clutter"]
        p["noise"] = self.stft(self.noise, tc)
        d = self.gain[None, None, :] * clean + p["noise"]
        d[:, :, self.dead] = 0.0
        return (d, p) if parts else d


def steer(D, freqs, sign):
    """exp(sign 2 pi i f D) for evenly spaced freqs, by recurrence."""
    E = np.exp(sign * 2j * np.pi * freqs[0] * D); step = np.exp(sign * 2j * np.pi * (freqs[1] - freqs[0]) * D)
    for fi in range(len(freqs)):
        yield fi, E
        E = E * step


def patch(c, half, step):
    xs = np.arange(c[0] - half, c[0] + half + 1e-6, step); ys = np.arange(c[1] - half, c[1] + half + 1e-6, step)
    X, Y = np.meshgrid(xs, ys); return np.c_[X.ravel(), Y.ravel()]


def peaks(P, pts, k, sep):
    out = []
    for i in np.argsort(P)[::-1][:5000]:
        if P[i] <= 0: break
        if all(np.linalg.norm(pts[i] - q) > sep for q in out): out.append(pts[i])
        if len(out) == k: break
    return out


class Processor:
    """Whitened Bartlett / bistatic matched field processing of one window."""

    def __init__(self, dom, floor):
        self.dom = dom; self.fr = dom.freqs
        self.isig0 = np.where(floor > 0, 1 / np.sqrt(np.maximum(floor, 1e-30)), 0.0)   # (nf, m), 0 on dead channels
        self.isig = self.isig0.copy(); self.g = np.ones(dom.m); self.good = floor.max(0) > 0
        self.az = np.empty(0)                                     # azimuths of stationary far-field interference
        self.Dc, self.Gc = dom.tables(dom.cells)

    def calibrate(self, centers, iters=2, **kw):
        """Per-channel gains from the AIS ships (loud sources at known, refined positions): regress each channel's
        data on the fitted ship model over a few windows; data are divided by the gains.  Then re-whiten: the noise
        floor is re-estimated from the residual with the ships and the stored clutter projected out."""
        for _ in range(iters):
            num = np.zeros(self.dom.m); den = np.zeros(self.dom.m)
            for tc in centers:
                x = self.dom.synth(tc, **kw) / self.g
                _, ships = self.ships_for(x, tc)
                refs = self.references(x, ships); D, G = self.dom.tables(np.array(ships))
                for fi, E in steer(D, self.fr, -1):
                    model = refs[:, :, fi].T @ (G * E)                       # (NSNAP, m)
                    w = self.isig[fi] ** 2
                    num += np.sum(np.real(np.conj(model) * x[:, fi, :]), axis=0) * w
                    den += np.sum(np.abs(model) ** 2, axis=0) * w
            upd = np.where(self.good & (den > 0), num / np.where(den > 0, den, 1), 1.0)
            self.g = self.g * upd; self.g /= np.median(self.g[self.good]); self.g[~self.good] = 1.0
            self.isig = self.isig0 * self.g[None, :]
        # re-whiten: the floor of the raw data is ships, not noise; take it from the residual with ships and clutter removed
        R, X, S = [], [], []
        for tc in centers:
            x = self.dom.synth(tc, **kw) / self.g
            _, ships = self.ships_for(x, tc)
            R.append(self.project(x, np.vstack([ships, CLUTTER]))); X.append(x); S.append(ships)
        fl = np.median(np.abs(np.concatenate(R, axis=0)) ** 2, axis=0); fl[:, ~self.good] = 0
        self.isig = np.where(fl > 0, 1 / np.sqrt(np.maximum(fl, 1e-30)), 0.0)
        # distant shipping: plane waves that stay in the residual, found by a far-field azimuth scan.  The scan uses
        # replicas projected like the data (a^H P x / |P a|^2), because a ship's subspace can hold part of a plane
        # wave (the ferry runs along the cable, which makes it look like an endfire plane wave to the far channels).
        azg = np.radians(np.arange(0.0, 360.0, 0.5)); num = np.zeros(len(azg)); den = np.zeros(len(azg))
        for x, ships in zip(X, S):
            res, Qs = self.project(x, np.vstack([ships, CLUTTER]), return_q=True)
            for fi, Q in enumerate(Qs):
                A = self.plane_waves(azg, fi); Ap = A - Q @ (Q.conj().T @ A)
                y = Ap.conj().T @ (res[:, fi, :] * self.isig[fi]).T
                num += np.sum(np.abs(y) ** 2, axis=1); den += np.sum(np.abs(Ap) ** 2, axis=0) * x.shape[0]
        Bz = num / np.maximum(den, 1e-30)
        self.az_scan = (np.degrees(azg), Bz)
        loc = [i for i in np.argsort(Bz)[::-1] if Bz[i] >= Bz[i - 1] and Bz[i] >= Bz[(i + 1) % len(Bz)]]
        picks = []
        for i in loc:
            if Bz[i] < AZ_DB_ABOVE * np.median(Bz) or len(picks) == 4: break
            if all(abs((azg[i] - a + np.pi) % (2 * np.pi) - np.pi) > np.radians(3) for a in picks): picks.append(azg[i])
        self.az = np.array(picks)
        return self.g

    def plane_waves(self, az, fi):
        """Whitened plane-wave replicas (m, len(az)) arriving from the azimuths az (source in direction az)."""
        u = np.c_[np.cos(az), np.sin(az)]
        return np.exp(2j * np.pi * self.fr[fi] * (self.dom.rx @ u.T) / C) * self.isig[fi][:, None]

    def ships_for(self, x, tc):
        """AIS prediction of each ship, refined successively on the data with the other ships removed."""
        ais = {v: self.dom.ais[v].predict(tc)[0] for v in SHIPS}
        ships = []
        for v in SHIPS:
            xr = self.project(x, ships) if ships else x
            ships.append(self.refine(xr, ais[v]))
        return ais, ships

    def bartlett_bins(self, x, D, G):
        """Whitened Bartlett per bin, (nf, npts)."""
        out = np.empty((len(self.fr), len(D)))
        for fi, E in steer(D, self.fr, 1):
            Gw = G * self.isig[fi]; W = Gw * E
            y = W @ (x[:, fi, :] * self.isig[fi]).T
            out[fi] = np.sum(np.abs(y) ** 2, axis=1) / np.maximum(np.sum(Gw ** 2, axis=1), 1e-30)
        return out

    def refine(self, x, guess, half=24.0):
        best = np.asarray(guess, float)
        for h, s in ((half, 3.0), (3.0, 0.5)):
            pts = patch(best, h, s); pts = pts[self.dom.env.is_water(pts)]
            best = pts[np.argmax(self.bartlett_bins(x, *self.dom.tables(pts))[::3].sum(0))]
        return best

    def project(self, x, pts_list, half=1.5, rel=RANK_REL, return_q=False):
        """Remove, per bin, the whitened subspace of replicas in small patches around each point.  At 12 to 78 Hz the
        patch replicas are nearly collinear: the subspace keeps only singular values above rel x the largest of each
        point, so a point costs a few dimensions instead of nine.  The DAS common mode (one signal on every cable
        channel, part of the dataset's noise) is removed with it."""
        pl = np.atleast_2d(pts_list); npp = len(patch(pl[0], half, half))
        pts = np.vstack([patch(p, half, half) for p in pl])
        D, G = self.dom.tables(pts); out = x.copy(); Qs = []
        for fi, E in steer(D, self.fr, -1):
            Wm = ((G * E) * self.isig[fi]).T                                        # whitened replicas (m, npts)
            Us = []
            for j in range(len(pl)):
                U, sv, _ = np.linalg.svd(Wm[:, j * npp:(j + 1) * npp], full_matrices=False)
                if sv[0] > 0: Us.append(U[:, sv > rel * sv[0]])
            for a in self.az:                                                         # stationary plane waves
                U, sv, _ = np.linalg.svd(self.plane_waves(a + np.radians([-0.5, 0, 0.5]), fi), full_matrices=False)
                Us.append(U[:, sv > rel * sv[0]])
            cm = self.isig[fi] * (np.arange(self.dom.m) < self.dom.n_das)              # DAS common mode, whitened
            Us.append((cm / np.linalg.norm(cm))[:, None])
            Q, _ = np.linalg.qr(np.hstack(Us))                                      # joint, so removals do not interfere
            if return_q: Qs.append(Q)
            xw = x[:, fi, :] * self.isig[fi]
            out[:, fi, :] = np.where(self.isig[fi] > 0, (xw - (xw @ Q.conj()) @ Q.T) / np.where(self.isig[fi] > 0, self.isig[fi], 1), 0)
        return (out, Qs) if return_q else out

    def references(self, x, ships):
        """Joint least-squares source spectra of the ships at their refined positions, (nship, NSNAP, nf)."""
        D, G = self.dom.tables(np.array(ships)); r = np.empty((len(ships), x.shape[0], len(self.fr)), complex)
        for fi, E in steer(D, self.fr, -1):
            Hs = ((G * E) * self.isig[fi]).T
            r[:, :, fi] = np.linalg.lstsq(Hs, (x[:, fi, :] * self.isig[fi]).T, rcond=None)[0]
        return r

    def bistatic(self, x, res, ships, refs, pts, D=None, G=None):
        """Sum over illuminators of the coherent (and incoherent) SNR-normalised echo matched filter."""
        if D is None: D, G = self.dom.tables(pts, body=True)
        else: G = G / np.sqrt(R_REF)                              # body leg is re 1 m
        coh = np.zeros(len(pts)); inc = np.zeros(len(pts))
        for v, s in enumerate(ships):
            rs = np.linalg.norm(pts - s, axis=1); gsx = np.sqrt(R_REF / np.maximum(rs, 1.0)) * self.dom.los(s, pts)[0]
            Z = np.zeros((len(pts), x.shape[0]), complex); Pi = np.zeros(len(pts)); norm = np.zeros(len(pts))
            for fi, E in steer(D, self.fr, 1):
                Gw = G * self.isig[fi]; y = (Gw * E) @ (res[:, fi, :] * self.isig[fi]).T
                z = y * (gsx * np.exp(2j * np.pi * self.fr[fi] * rs / C))[:, None] * np.conj(refs[v, :, fi])[None]
                Z += z; Pi += np.sum(np.abs(z) ** 2, axis=1)
                norm += np.sum(Gw ** 2, axis=1) * gsx ** 2 * np.mean(np.abs(refs[v, :, fi]) ** 2)
            ok = norm > 0; nz = np.where(ok, norm, 1) * x.shape[0]
            coh += np.where(ok, np.sum(np.abs(Z) ** 2, axis=1) / nz / len(self.fr), 0); inc += np.where(ok, Pi / nz, 0)
        return coh / len(ships), inc / len(ships)

    def passive(self, res, pts, D=None, G=None):
        if D is None: D, G = self.dom.tables(pts)
        B = self.bartlett_bins(res, D, G)
        return B

    def run(self, x, tc, mode, ncand=8):
        dom = self.dom
        x = x / self.g                                            # gain calibration (identity before calibrate())
        ais, ships = self.ships_for(x, tc)
        res = self.project(x, np.vstack([ships, CLUTTER]))
        excl = np.min([np.linalg.norm(dom.cells - s, axis=1) for s in ships], axis=0) < EXCL
        if mode == "passive":
            B = self.passive(res, dom.cells, self.Dc, self.Gc)
            med = np.median(B[:, ~excl], axis=1, keepdims=True); P = (B / med).mean(0)
            def score(pts):
                B = self.passive(res, pts); return (B / med).mean(0)
            maps = [P]
        else:
            refs = self.references(x, ships)
            P, Pinc = self.bistatic(x, res, ships, refs, dom.cells, self.Dc, self.Gc)
            def score(pts):
                return self.bistatic(x, res, ships, refs, pts)[0]
            maps = [P, Pinc]
        for M in maps: M[excl] = 0
        bg = np.median(P[~excl])
        cand = [c for M in maps for c in peaks(M, dom.cells, ncand, 3 * GRID)]
        def sc(pts):
            pts = pts[dom.env.is_water(pts)]
            pts = pts[np.min([np.linalg.norm(pts - s, axis=1) for s in ships], axis=0) >= EXCL]
            return pts, (score(pts) if len(pts) else np.array([]))
        best, est = -np.inf, None
        for c in cand:
            pts, v = sc(patch(c, 1.2 * GRID, 2.0))
            if not len(v): continue
            e = pts[np.argmax(v)]
            for _ in range(10):
                pts, v = sc(patch(e, 6.0, 2.0)); e_new = pts[np.argmax(v)]
                if np.allclose(e_new, e): break
                e = e_new
            if v.max() > best: best, est = v.max(), e
        pts, v = sc(patch(est, 2.0, 0.5))
        if len(v) and v.max() >= best: best, est = v.max(), pts[np.argmax(v)]
        return {"ships": np.array(ships), "ais": np.array([ais[v] for v in SHIPS]), "P": P, "est": est,
                "contrast": float(10 * np.log10(best / bg)), "res": res}


def noise_floor(dom, centers, **kw):
    """Per channel and bin noise floor: median over the run of |X|^2 (data derived), (nf, m)."""
    X = np.concatenate([np.abs(dom.synth(tc, **kw)) ** 2 for tc in centers[::2]], axis=0)
    fl = np.median(X, axis=0); fl[:, dom.dead] = 0
    q = np.median(fl[fl > 0]); fl[fl < 1e-6 * q] = 0                    # channel QC
    return fl


def image(dom, P):
    img = np.full(dom.water.shape, np.nan); img[dom.water] = P; return img.reshape(dom.shape)


def show(dom, ax, P, title, tc, est=None, ships=None, ais=None, vmin=-25):
    plot_harbor(dom.env, dom.ds.layout, ax=ax, show_depth=False)
    Z = 10 * np.log10(np.maximum(image(dom, P), 1e-30)); Z -= np.nanmax(Z)
    im = ax.imshow(Z, origin="lower", extent=dom.env.extent, cmap="viridis", vmin=vmin, vmax=0, zorder=2, alpha=0.95)
    for v in SHIPS: ax.plot(*dom.truth(v, tc), "^", ms=9, mfc="none", mec="w", mew=1.5, zorder=5)
    if ais is not None: ax.plot(*ais.T, "x", color="w", ms=7, mew=1.5, zorder=5, label="ships AIS")
    if ships is not None: ax.plot(*ships.T, "+", color="#ffd60a", ms=12, mew=2, zorder=6, label="ships refined")
    ax.plot(*CLUTTER.T, "s", ms=6, mfc="none", mec="w", zorder=5, label="stored clutter")
    ax.plot(*dom.truth(UUV, tc), "o", ms=14, mfc="none", mec="#ff7f0e", mew=2, zorder=6, label="UUV truth")
    if est is not None: ax.plot(*est, "x", color="#ff2d55", ms=10, mew=2.5, zorder=7, label="UUV estimate")
    ax.set_title("")                                              # plot_harbor's centred title
    ax.set_title(title, loc="left", color=INK, fontweight="bold", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    return im


def save(fig, out, name):
    fig.savefig(out / name, bbox_inches="tight"); plt.close(fig); print("wrote", name, flush=True)


def processing(args):
    """The matched-field chain of harbor_mfp_domain.py (maps, tracks, Pd), unchanged."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    dom = Domain(args.dataset)
    centers = np.arange(TSNAP / 2 + (NSNAP - 1) / 2 * HOP + 0.01, dom.T - 2.0, (NSNAP + 1) * HOP)   # 3 s hop
    res = {"dataset": os.path.relpath(args.dataset, ROOT), "band_hz": [float(dom.freqs[0]), float(dom.freqs[-1])],
           "bins": dom.nf, "snapshots": NSNAP, "snapshot_s": TSNAP, "windows": len(centers), "grid_m": GRID,
           "water_cells": int(len(dom.cells)), "uuv_level_pa": dom.level[UUV], "uuv_ts_db": UUV_TS,
           "assumptions": ["analytic 2-D free-field propagation (sqrt(100 m / r), r / c, c = 1500 m/s), land opaque by line of "
                           "sight, no reflection, diffraction or depth dependence; the replica is this model",
                           "geometry frozen inside each 3 s window (5 x 1 s snapshots); per-snapshot source spectra from the "
                           "dataset's signature generator", "noise, DAS gains and dead channels from the dataset",
                           "pier and breakwater heads are point scatterers (TS 0 dB) at known positions, projected out",
                           "ships: AIS (dataset reports, interpolated) refined by Bartlett on the data, projected out",
                           "DAS gains calibrated on the AIS ships; noise floor re-estimated from the cleaned residual; DAS common "
                           "mode and the plane waves found by a far-field azimuth scan projected out"]}
    print(f"{len(dom.cells)} water cells, {dom.m} receivers, {dom.nf} bins, {len(centers)} windows", flush=True)

    # ---------- map ----------
    fig, ax = plt.subplots(figsize=(10, 7.5))
    plot_harbor(dom.env, dom.ds.layout, ax=ax)
    for v, col in zip(SHIPS + (UUV,), ("#1d4ed8", "#16a34a", "#7c3aed", "#ff7f0e")):
        tr = dom.ds.tracks[v]; ax.plot(*tr.xy.T, "--" if v == UUV else "-", color=col, lw=2, label=f"{v} ({'AIS' if v in SHIPS else 'hidden'})")
        ax.plot(*tr.xy[0], "o", color=col, ms=7, mec="k", zorder=5)                 # start (the moored ship has no track)
    ax.plot(*dom.ds.ais.xy.T, ".", color="k", ms=3, label="AIS reports")
    ax.plot(*CLUTTER.T, "s", ms=7, mfc="#9ca3af", mec=INK, label="stored clutter (TS 0 dB)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax.set_title(f"Domain of {res['dataset']}: 125 ch cable + 4 hydrophones, 120 s, analytic propagation", fontsize=10)
    save(fig, out, "map.png")

    # ---------- analytic vs FDTD levels ----------
    fdtd = np.array([np.mean(np.abs(dom.stft(dom.ds.clean.astype(float), tc)) ** 2, axis=(0, 1)) for tc in centers])
    ana = []
    for tc in centers:
        _, p = dom.synth(tc, parts=True); ana.append(np.mean(np.abs(p["ships"] + p["uuv"]) ** 2, axis=(0, 1)))
    ana = np.array(ana); noi = np.array([np.mean(np.abs(dom.stft(dom.noise, tc)) ** 2, axis=(0, 1)) for tc in centers])
    db = lambda x: 10 * np.log10(np.maximum(x, 1e-30))
    fig = plt.figure(figsize=(14, 7.5)); gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.4, wspace=0.08)
    lo, hi = np.percentile(db(fdtd), 5), np.percentile(db(fdtd), 99.5)
    for j, (A, name) in enumerate(((fdtd, "FDTD clean field (dataset)"), (ana, "Analytic field (this test)"))):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(db(A).T, aspect="auto", origin="lower", extent=[centers[0], centers[-1], 0, dom.m], cmap="magma", vmin=lo, vmax=hi)
        ax.axhline(dom.n_das, color="w", lw=0.8, ls="--"); ax.set_xlabel("time (s)"); ax.tick_params(colors=MUTED)
        ax.set_title(name + ", 12 to 78 Hz, all vessels", loc="left", color=INK, fontweight="bold", fontsize=10)
        if j == 0: ax.set_ylabel("receiver (cable | hydrophones)")
        else: ax.set_yticklabels([])
    fig.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.01, label="mean level per bin (dB)")
    ax = fig.add_subplot(gs[1, :])
    ax.plot(db(fdtd.mean(0)), color=INK, label="FDTD clean"); ax.plot(db(ana.mean(0)), color="#ff7f0e", label="analytic")
    ax.plot(np.where(dom.dead, np.nan, db(noi.mean(0))), color=MUTED, lw=0.8, label="dataset noise (dead channels omitted)")
    good = ~dom.dead; diff = db(ana.mean(0))[good] - db(fdtd.mean(0))[good]
    ax.axvline(dom.n_das, color="k", lw=0.8, ls="--"); ax.set_xlim(0, dom.m); ax.set_xlabel("receiver"); ax.set_ylabel("dB")
    ax.set_title(f"Run-average level per receiver: analytic - FDTD median {np.median(diff):+.1f} dB, "
                 f"spread (IQR) {np.subtract(*np.percentile(diff, [75, 25])):.1f} dB", loc="left", color=INK, fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.tick_params(colors=MUTED)
    save(fig, out, "analytic_vs_fdtd.png")
    res["analytic_minus_fdtd_db"] = {"median": float(np.median(diff)), "iqr": float(np.subtract(*np.percentile(diff, [75, 25])))}

    # ---------- processors ----------
    procs = {}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.2)); res["calibration"] = {}
    for (mode, kw), col in zip((("passive", {}), ("bistatic", {"uuv_level": 0.0, "ts": UUV_TS})), ("#7c3aed", "#047857")):
        t1 = time.time(); pr = Processor(dom, noise_floor(dom, centers, **kw)); pr.calibrate(centers[::5], **kw); procs[mode] = pr
        ratio = pr.g[pr.good] / dom.gain[pr.good]; ratio /= np.median(ratio)
        res["calibration"][mode] = {"gain_error_rms": float(np.std(ratio)), "true_gain_scatter_rms": float(np.std(dom.gain[pr.good])),
                                    "interference_azimuths_deg": np.degrees(pr.az).tolist(),
                                    "true_shipping_azimuths_deg": dom.ds.meta["shipping_azimuth_deg"], "seconds": round(time.time() - t1)}
        print(f"  calibrated {mode}: gain error {np.std(ratio):.3f} rms, interference at {np.round(np.degrees(pr.az), 1)} deg", flush=True)
        a1.plot(dom.gain[pr.good], pr.g[pr.good] / np.median(pr.g[pr.good] / dom.gain[pr.good]), "o", ms=3, color=col,
                label=f"{mode}: error {100 * np.std(ratio):.1f} % rms")
        az, B = pr.az_scan; a2.plot(az, 10 * np.log10(B / np.median(B)), color=col, lw=1, label=f"{mode}: picked {np.round(np.degrees(pr.az), 1)}")
    lim = [dom.gain[~dom.dead].min() * 0.95, dom.gain[~dom.dead].max() * 1.05]; a1.plot(lim, lim, "k:", lw=0.8)
    a1.set_xlabel("true DAS gain (dataset)"); a1.set_ylabel("estimated gain"); a1.legend(frameon=False, fontsize=8); a1.tick_params(colors=MUTED)
    a1.set_title(f"Gain calibration from the AIS ships (true scatter {100 * np.std(dom.gain[~dom.dead]):.0f} %)", loc="left", color=INK, fontweight="bold", fontsize=10)
    for k, a in enumerate(dom.ds.meta["shipping_azimuth_deg"]): a2.axvline(a, color="#dc2626", ls="--", lw=0.8, label="true shipping azimuth" if k == 0 else None)
    a2.axhline(6, color=MUTED, ls=":", lw=0.8, label="pick threshold (6 dB)")
    a2.set_xlabel("azimuth (deg, source direction)"); a2.set_ylabel("dB over median"); a2.legend(frameon=False, fontsize=8); a2.tick_params(colors=MUTED)
    a2.set_title("Far-field interference scan of the residual (projected replicas)", loc="left", color=INK, fontweight="bold", fontsize=10)
    save(fig, out, "calibration.png")

    # ---------- one window ----------
    tc = centers[len(centers) // 3]
    one = {}
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, (mode, kw) in zip(axs, (("passive", {}), ("bistatic", {"uuv_level": 0.0, "ts": UUV_TS}))):
        r = procs[mode].run(dom.synth(tc, **kw), tc, mode)
        err = float(np.linalg.norm(r["est"] - dom.truth(UUV, tc)))
        one[mode] = {"t_s": float(tc), "error_m": err, "contrast_db": r["contrast"],
                     "ship_refined_error_m": np.linalg.norm(r["ships"] - np.array([dom.truth(v, tc) for v in SHIPS]), axis=1).tolist(),
                     "ship_ais_error_m": np.linalg.norm(r["ais"] - np.array([dom.truth(v, tc) for v in SHIPS]), axis=1).tolist()}
        lab = "passive, UUV radiating 0.03 Pa at 100 m (motor ON)" if mode == "passive" else f"bistatic, silent UUV TS {UUV_TS:.0f} dB, 3 illuminators"
        im = show(dom, ax, r["P"], f"{lab}\nt = {tc:.1f} s: error {err:.1f} m, peak {r['contrast']:.1f} dB over median",
                  tc, est=r["est"], ships=r["ships"], ais=r["ais"])
    fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01, label="map, dB rel. peak (10 m grid)")
    save(fig, out, "maps.png")
    res["single_window"] = one

    # ---------- track over the run ----------
    rows = {m: [] for m in procs}
    for k, tc in enumerate(centers):
        for mode, kw in (("passive", {}), ("bistatic", {"uuv_level": 0.0, "ts": UUV_TS})):
            r = procs[mode].run(dom.synth(tc, **kw), tc, mode)
            rows[mode].append((tc, r["est"], r["contrast"], r["ships"]))
        print(f"  window {k + 1}/{len(centers)} t {tc:5.1f}: passive {np.linalg.norm(rows['passive'][-1][1] - dom.truth(UUV, tc)):6.1f} m "
              f"{rows['passive'][-1][2]:5.1f} dB, bistatic {np.linalg.norm(rows['bistatic'][-1][1] - dom.truth(UUV, tc)):6.1f} m "
              f"{rows['bistatic'][-1][2]:5.1f} dB", flush=True)
    truth = np.array([dom.truth(UUV, tc) for tc in centers])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.1, 1]})
    plot_harbor(dom.env, dom.ds.layout, ax=a1, show_depth=False)
    a1.plot(*dom.ds.tracks[UUV].xy.T, "-", color="#ff7f0e", lw=3, label="UUV truth")
    tr = {}
    for mode, col, mk in (("passive", "#7c3aed", "o"), ("bistatic", "#047857", "x")):
        est = np.array([r[1] for r in rows[mode]]); e = np.linalg.norm(est - truth, axis=1); tr[mode] = e
        a1.plot(*est.T, mk, color=col, ms=5, mfc="none" if mk == "o" else col, label=f"{mode} estimate")
        a2.plot(centers, np.minimum(e, 99), mk + "-", color=col, ms=4, lw=1, mfc="none" if mk == "o" else col,
                label=f"{mode}: median {np.median(e):.1f} m, {100 * np.mean(e <= 20):.0f} % within 20 m")
    a1.legend(loc="upper left", fontsize=8); a1.set_title("UUV estimates per 3 s window", fontsize=10)
    a2.axhline(20, color="r", ls="--", lw=0.8); a2.set_ylim(0, 100); a2.set_xlabel("time (s)"); a2.set_ylabel("error (m), clipped at 100")
    a2.legend(frameon=False, fontsize=8); a2.set_title("Position error per window", loc="left", color=INK, fontweight="bold", fontsize=10)
    a2.tick_params(colors=MUTED)
    save(fig, out, "tracks.png")
    res["track"] = {m: {"median_m": float(np.median(e)), "p90_m": float(np.percentile(e, 90)), "within_20m": float(np.mean(e <= 20)),
                        "contrast_db_median": float(np.median([r[2] for r in rows[m]]))} for m, e in tr.items()}
    res["track"]["ship_refined_error_median_m"] = {
        v: float(np.median([np.linalg.norm(r[3][i] - dom.truth(v, r[0])) for r in rows["bistatic"]])) for i, v in enumerate(SHIPS)}

    # ---------- Pd against level ----------
    if args.pd_windows:
        sub = centers[np.linspace(0, len(centers) - 1, min(args.pd_windows, len(centers))).astype(int)]
        sweeps = {"passive": (np.arange(-55.0, -19.0, 5.0), lambda lv: {"uuv_level": 10 ** (lv / 20)}, {"uuv_level": 0.0},
                              20 * np.log10(dom.level[UUV]), "UUV radiated level (dB re 1 Pa at 100 m)"),
                  "bistatic": (np.arange(-40.0, 1.0, 5.0), lambda lv: {"uuv_level": 0.0, "ts": lv}, {"uuv_level": 0.0, "ts": -np.inf},
                               UUV_TS, "UUV target strength (dB re 1 m, 2-D)")}
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.2)); res["pd"] = {"windows": len(sub), "pfa": 0.05, "gate_m": 20.0}
        for ax, (mode, (levels, kw, kw0, default, xlabel)) in zip(axs, sweeps.items()):
            t1 = time.time()
            h0 = np.array([procs[mode].run(dom.synth(tc, **kw0), tc, mode)["contrast"] for tc in sub])
            thr = float(np.quantile(h0, 0.95)); pd = []
            for lv in levels:
                det = []
                for tc in sub:
                    r = procs[mode].run(dom.synth(tc, **kw(lv)), tc, mode)
                    det.append(r["contrast"] > thr and np.linalg.norm(r["est"] - dom.truth(UUV, tc)) <= 20.0)
                pd.append(float(np.mean(det)))
            print(f"  {mode}: threshold {thr:.1f} dB, Pd {np.round(pd, 2).tolist()} ({time.time() - t1:.0f} s)", flush=True)
            res["pd"][mode] = {"levels": levels.tolist(), "pd": pd, "threshold_db": thr, "noise_contrast_db_median": float(np.median(h0)),
                               "default": float(default)}
            ax.plot(levels, pd, "o-", color=INK); ax.axvline(default, color="#ff7f0e", ls="--", lw=1, label=f"scenario ({default:.1f})")
            ax.axhline(0.05, color=MUTED, ls=":", lw=1, label=f"Pfa 0.05 (threshold {thr:.1f} dB)")
            ax.set_ylim(-0.03, 1.03); ax.set_xlabel(xlabel); ax.set_ylabel("Pd"); ax.tick_params(colors=MUTED)
            ax.set_title(f"{mode.capitalize()}, {len(sub)} windows along the UUV track", loc="left", color=INK, fontweight="bold", fontsize=10)
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle("Pd against UUV level in the dataset domain (peak above the no-UUV threshold and within 20 m)", color=INK, fontsize=10)
        save(fig, out, "pd_vs_level.png")

    res["runtime_s"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("analytic_minus_fdtd_db", "single_window", "track")}, indent=2))
    print(f"done in {res['runtime_s']:.0f} s -> {out}")



# ======================================================================================================================
# Videos
# ======================================================================================================================

VID_DX = 2.5                    # wavefield grid [m]
LOS_REFRESH = 0.5               # recompute moving sources' line of sight every this many seconds
UP = 4                          # signal upsampling for fractional delays


class TimeDomain:
    """Time-domain version of the Domain forward model: the field anywhere, the traces on the receivers."""

    def __init__(self, dom, scenario, ts=UUV_TS):
        from scipy.signal import resample_poly
        self.dom, self.scenario, self.ts = dom, scenario, ts
        self.fsu = dom.fs * UP
        self.sig = {v: resample_poly(x, UP, 1) for v, x in dom.sig.items()}
        if scenario == "bistatic": self.sig[UUV] = np.zeros_like(self.sig[UUV])
        self.tu = np.arange(len(self.sig[UUV])) / self.fsu
        self._los = {}

    def s(self, v, t):
        """Source signature of vessel v at times t (Pa at 100 m), zero outside the record."""
        return np.interp(t, self.tu, self.sig[v], left=0.0, right=0.0)

    def los(self, key, a, pts, t):
        """Line of sight from a (moving) point to pts, cached per vessel and LOS_REFRESH interval."""
        k = (key, int(t // LOS_REFRESH), len(pts))
        if k not in self._los:
            if len(self._los) > 64: self._los.clear()
            self._los[k] = np.concatenate([self.dom.los(a, pts[i:i + 8000])[0] for i in range(0, len(pts), 8000)])
        return self._los[k]

    def pressure(self, pts, t, ret_parts=False):
        """Pressure at pts at time t: every vessel's direct field (+ the UUV's echoes of the ships, bistatic)."""
        dom = self.dom; p = np.zeros(len(pts)); uuv = dom.truth(UUV, t)
        for v in list(SHIPS) + [UUV]:
            if v == UUV and self.scenario == "bistatic": continue
            xv = dom.truth(v, t); r = np.linalg.norm(pts - xv, axis=1)
            p += np.sqrt(R_REF / np.maximum(r, 1.0)) * self.los(v, xv, pts, t) * self.s(v, t - r / C)
        if self.scenario == "bistatic":
            ru = np.linalg.norm(pts - uuv, axis=1); lu = self.los("uuv", uuv, pts, t)
            for v in SHIPS:
                xv = dom.truth(v, t); rs = np.linalg.norm(xv - uuv); ls = float(self.los(v + ">uuv", xv, uuv[None], t)[0])
                if ls == 0: continue
                p += (np.sqrt(R_REF / rs) * 10 ** (self.ts / 20) * np.sqrt(1 / np.maximum(ru, 1.0)) * lu
                      * self.s(v, t - (rs + ru) / C))
        return p

    def traces(self, t0, t1):
        """Cable channels over [t0, t1): gain x field + the dataset's noise, dead channels flat, 12 to 78 Hz."""
        from scipy.signal import butter, sosfiltfilt
        dom = self.dom; fs = dom.fs; pad = 1.0
        t = np.arange(int(round((t0 - pad) * fs)), int(round((t1 + pad) * fs))) / fs
        rx = dom.rx[:dom.n_das]; X = np.zeros((len(rx), len(t)))
        for j in range(0, len(t), int(fs * LOS_REFRESH)):                   # geometry refreshed every LOS_REFRESH
            tt = t[j:j + int(fs * LOS_REFRESH)]
            X[:, j:j + len(tt)] = np.array([self.pressure(rx, ti) for ti in tt]).T
        i0 = int(round((t0 - pad) * fs))
        X = dom.gain[:dom.n_das, None] * X + dom.noise[:dom.n_das, i0:i0 + len(t)]
        X[dom.dead[:dom.n_das]] = 0.0
        X = sosfiltfilt(butter(4, [F_LO, F_HI], btype="band", fs=fs, output="sos"), X, axis=1)
        k = slice(int(pad * fs), len(t) - int(pad * fs))
        return t[k], X[:, k]


def encode(frames_dir, fps, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "f%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
                   check=True)


def vessel_marks(ax, dom, t, trail=8.0, labels=True, color_uuv="#f97316"):
    """Ship and UUV markers with short trails; returns the artists so they can be updated."""
    art = []
    cols = {"ferry": "#38bdf8", "tug": "#4ade80", "moored": "#c084fc", UUV: color_uuv}
    for v in list(SHIPS) + [UUV]:
        tr = dom.truth(v, t); trl = dom.ds.tracks[v].position_at(np.linspace(max(0, t - trail), t, 20))
        l, = ax.plot(*trl.T, "-", color=cols[v], lw=2, alpha=0.9, zorder=6)
        m, = ax.plot(*tr, "^" if v in SHIPS else "o", ms=14 if v in SHIPS else 15, color=cols[v],
                     mfc=cols[v] if v in SHIPS else "none", mec="k" if v in SHIPS else cols[v], mew=1.5 if v in SHIPS else 3, zorder=7)
        txt = ax.text(*(tr + [25, 25]), f"{v} (AIS)" if v in SHIPS else "UUV (hidden)", color=cols[v], fontsize=13,
                      fontweight="bold", zorder=8, path_effects=[patheffects.withStroke(linewidth=3, foreground="k")]) if labels else None
        art.append((v, l, m, txt))
    return art


def update_marks(art, dom, t, trail=8.0):
    for v, l, m, txt in art:
        tr = dom.truth(v, t); trl = dom.ds.tracks[v].position_at(np.linspace(max(0, t - trail), t, 20))
        l.set_data(*trl.T); m.set_data([tr[0]], [tr[1]])
        if txt is not None: txt.set_position(tr + [25, 25])


def video_wavefield(dom, td, args, out):
    env = dom.env; x0, x1, y0, y1 = env.extent
    xs = np.arange(x0 + VID_DX / 2, x1, VID_DX); ys = np.arange(y0 + VID_DX / 2, y1, VID_DX)
    X, Y = np.meshgrid(xs, ys); pts = np.c_[X.ravel(), Y.ravel()]; water = env.is_water(pts)
    times = np.arange(args.t0, args.t0 + args.duration, args.frame_dt)
    vmax = args.wave_vmax
    fdir = out / "_wf"; fdir.mkdir(exist_ok=True); keep = out / "frames_wavefield"; keep.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(16, 12.6)); ax = fig.add_axes([0.05, 0.05, 0.83, 0.86]); cax = fig.add_axes([0.9, 0.2, 0.015, 0.55])
    ax.set_facecolor("#9ca3af")
    im = ax.imshow(np.zeros(X.shape), origin="lower", extent=env.extent, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="bilinear", zorder=2)
    ax.imshow(np.where(env.land, 1.0, np.nan), origin="lower", extent=env.extent, cmap=ListedColormap(["#9ca3af"]), vmin=0, vmax=1, zorder=3)
    cab = dom.ds.layout.cables[0].positions
    ax.plot(*cab.T, "-", color="#facc15", lw=3, zorder=5, label="DAS cable (125 ch)")
    ax.plot(*dom.rx[dom.n_das:].T, "v", ms=12, color="#facc15", mec="k", zorder=5, label="hydrophone")
    art = vessel_marks(ax, dom, times[0])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xlabel("x (m)", fontsize=13); ax.set_ylabel("y (m)", fontsize=13)
    ax.tick_params(labelsize=11); ax.legend(loc="lower left", fontsize=12, framealpha=0.85)
    tick = np.array([-1, -0.5, -0.1, 0, 0.1, 0.5, 1]) * vmax
    cb = fig.colorbar(im, cax=cax); cb.set_ticks(np.sign(tick) * np.sqrt(np.abs(tick) / vmax)); cb.set_ticklabels([f"{v:+.2g}" for v in tick])
    cb.set_label("pressure (Pa), square-root colour scale", fontsize=12)
    ttl = fig.text(0.05, 0.95, "", fontsize=20, fontweight="bold", color=INK)
    sub = ("UUV radiating its 0.03 Pa signature (about 30 dB below the ferry)" if td.scenario == "passive"
           else f"UUV silent, scattering the ships (TS {td.ts:.0f} dB)")
    fig.text(0.05, 0.905, f"Analytic 2-D propagation in the dataset harbor, 12 to 78 Hz signatures from the dataset; land blocks sound, "
             f"no reflections.\n{sub}", fontsize=12, color=MUTED)
    t1 = time.time()
    for i, t in enumerate(times):
        p = np.full(len(pts), np.nan); p[water] = td.pressure(pts[water], t)
        im.set_data(np.clip(np.sign(p) * np.sqrt(np.abs(p) / vmax), -1, 1).reshape(X.shape))
        update_marks(art, dom, t); ttl.set_text(f"Harbor wavefield   t = {t:6.2f} s")
        fig.savefig(fdir / f"f{i:05d}.png", dpi=args.dpi)
        if i % args.every == 0: shutil.copy(fdir / f"f{i:05d}.png", keep / f"wavefield_{i:05d}_t{t:06.2f}.png")
        if i % 50 == 0: print(f"  wavefield frame {i}/{len(times)} ({time.time() - t1:.0f} s)", flush=True)
    plt.close(fig)
    encode(fdir, args.fps, out / "wavefield.mp4"); shutil.rmtree(fdir); print("wrote wavefield.mp4", flush=True)


def video_das(dom, td, args, out):
    win = args.das_window
    t_all, Xd = td.traces(args.t0 - win, args.t0 + args.duration)
    dist = dom.ds.layout.cables[0].arclength
    v = np.percentile(np.abs(Xd[~dom.dead[:dom.n_das]]), 99)
    times = np.arange(args.t0, args.t0 + args.duration, args.frame_dt)
    fdir = out / "_das"; fdir.mkdir(exist_ok=True); keep = out / "frames_das"; keep.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(18, 10.5)); ax = fig.add_axes([0.07, 0.08, 0.76, 0.8]); cax = fig.add_axes([0.955, 0.2, 0.012, 0.55])
    ds = int(max(1, round(dom.fs / 300)))                                   # keep at least 300 samples per second on screen
    im = ax.imshow(Xd[:, :1], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="bilinear",
                   extent=[-win, 0, dist[0], dist[-1]])
    cb = fig.colorbar(im, cax=cax); cb.set_label("pressure on the fiber (Pa), 12 to 78 Hz", fontsize=12)
    ax.set_xlabel("time before now (s)", fontsize=14); ax.set_ylabel("distance along the cable (m)", fontsize=14); ax.tick_params(labelsize=12)
    ax.set_xlim(-win, 0); ax.set_ylim(dist[0], dist[-1])
    # where each vessel is closest to the cable: moving markers on the right edge
    cab = dom.ds.layout.cables[0].positions
    cols = {"ferry": "#0284c7", "tug": "#16a34a", "moored": "#9333ea", UUV: "#ea580c"}
    mk = {}
    for vv in list(SHIPS) + [UUV]:
        m, = ax.plot([], [], "<", ms=16, color=cols[vv], mec="k", clip_on=False, zorder=6)
        tx = ax.text(0, 0, "", color=cols[vv], fontsize=13, fontweight="bold", va="center", clip_on=False)
        mk[vv] = (m, tx)
    ttl = fig.text(0.07, 0.94, "", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.07, 0.905, f"DAS cable, {dom.n_das} channels at 10 m: analytic field of all vessels x the dataset's channel gains + the dataset's "
             f"noise (dead channels flat).  Arrows: closest point of each vessel.", fontsize=12, color=MUTED)
    t1 = time.time()
    for i, t in enumerate(times):
        j1 = int(round((t - t_all[0]) * dom.fs)); j0 = max(0, j1 - int(win * dom.fs))
        im.set_data(Xd[:, j0:j1:ds]); im.set_extent([t_all[j0] - t, t_all[j1 - 1] - t, dist[0], dist[-1]])
        for vv, (m, tx) in mk.items():
            k = int(np.argmin(np.linalg.norm(cab - dom.truth(vv, t), axis=1)))
            m.set_data([1.012], [dist[k]]); m.set_transform(ax.get_yaxis_transform())
            tx.set_position((1.03, dist[k])); tx.set_transform(ax.get_yaxis_transform())
            tx.set_text(f"{vv if vv in SHIPS else 'UUV'} {np.linalg.norm(cab[k] - dom.truth(vv, t)):.0f} m")
        ttl.set_text(f"DAS recording, last {win:.0f} s   t = {t:6.2f} s")
        fig.savefig(fdir / f"f{i:05d}.png", dpi=args.dpi)
        if i % args.every == 0: shutil.copy(fdir / f"f{i:05d}.png", keep / f"das_{i:05d}_t{t:06.2f}.png")
        if i % 50 == 0: print(f"  DAS frame {i}/{len(times)} ({time.time() - t1:.0f} s)", flush=True)
    plt.close(fig)
    encode(fdir, args.fps, out / "das_gather.mp4"); shutil.rmtree(fdir); print("wrote das_gather.mp4", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "output" / "harbor" / "dataset.npz"))
    ap.add_argument("--out", default=str(ROOT / "output" / "harbor_domain_video"))
    ap.add_argument("--scenario", choices=("passive", "bistatic"), default="passive")
    ap.add_argument("--t0", type=float, default=40.0, help="start time on the dataset's tracks (s)")
    ap.add_argument("--duration", type=float, default=15.0, help="simulated seconds")
    ap.add_argument("--frame-dt", type=float, default=0.025, help="simulated seconds per frame")
    ap.add_argument("--fps", type=int, default=40, help="40 fps x 0.025 s per frame: real time, 15 s movie by default")
    ap.add_argument("--every", type=int, default=10, help="save every Nth frame as PNG")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--wave-vmax", type=float, default=2.0, help="pressure at the ends of the colour scale (Pa)")
    ap.add_argument("--das-window", type=float, default=6.0, help="seconds shown in the DAS video")
    ap.add_argument("--only", choices=("wavefield", "das"), default=None)
    ap.add_argument("--process", action="store_true", help="also run the matched-field chain (about an hour)")
    ap.add_argument("--pd-windows", type=int, default=10, help="with --process: windows per level in the Pd sweep")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); t_start = time.time()
    dom = Domain(args.dataset); td = TimeDomain(dom, args.scenario)
    print(f"{args.scenario}: t = {args.t0} to {args.t0 + args.duration} s, {int(round(args.duration / args.frame_dt))} frames", flush=True)
    if args.only in (None, "wavefield"): video_wavefield(dom, td, args, out)
    if args.only in (None, "das"): video_das(dom, td, args, out)
    if args.process: processing(args)
    print(f"done in {time.time() - t_start:.0f} s -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Helmholtz forward model of the bistatic harbor scene: wavefields, gathers, passive and
bistatic matched field maps, sparse inversion and a validation over several target positions.

Same scene as the FDTD bistatic run (``examples/harbor_bistatic.scenario.json``: the 400 m x
300 m crop of the harbor with the north quay and the east pier, the strain DAS cable, four
hydrophones, the ferry as illuminator and the silent UUV), but the forward model is the
frequency domain solver of ``harbor/helmholtz.py`` and the replicas are its Green's functions
by reciprocity, so the processor knows every quay reflection, pier shadow and the shore loss.

Data model per 3 s window (eleven 0.5 s snapshots, geometry frozen within the window):
ferry direct path with a random phase per snapshot and bin, the UUV echo in the Born
approximation (point scatterer of target strength ``--ts`` dB), optionally the UUV's own tonal
lines (motor on), and white noise per channel set from the ferry's direct path level on that
sensor type (``--snr`` dB).  Data and replicas are whitened by the noise level per channel, so
the strain DAS channels and the hydrophones are combined on equal terms.

Usage::

    python examples/harbor_helmholtz_demo.py [--scenario examples/harbor_bistatic.scenario.json]
        [--out output/harbor_helmholtz] [--grid-dx 1.0] [--df 20] [--no-gathers]

Cold run: about 12 min for the replica library and 10 min for the gathers on two CPU cores;
reruns read both from ``<out>/cache``.
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.harbor import HarborScenarioConfig, build_environment, build_layout, build_track  # noqa: E402
from acoustic_sim.harbor.bistatic import BistaticConfig, build_grid  # noqa: E402
from acoustic_sim.harbor.detection import green2d  # noqa: E402
from acoustic_sim.harbor.helmholtz import (  # noqa: E402
    HelmholtzConfig,
    HelmholtzSolver,
    bartlett,
    bistatic_coherent,
    free_field_replicas,
    project_out,
    replica_library,
    sparse_image,
)
from acoustic_sim.harbor.plotting import plot_harbor  # noqa: E402

N_SNAP = 11
WIN_S = 3.0
UUV_LINES = (140.0, 220.0, 300.0)      # motor on: thruster lines [Hz], on the 20 Hz bin grid
SPARSE_ITER = 80


def db(x, floor=1e-30):
    return 10 * np.log10(np.maximum(x, floor))


def ferry_spectrum(f):
    """Relative ferry source amplitude per bin: broadband falling 6 dB per octave above 100 Hz."""
    return 10 ** ((-6 * np.log2(np.maximum(f, 100.0) / 100.0)) / 20)


def scatter_coef(f, ts_db, c):
    """Born coefficient so the scattered field 1 m from the body is ``10^(TS/20)`` of the incident."""
    return 10 ** (ts_db / 20) / np.abs(green2d(2 * np.pi * f / c * 1.0))


class Scene:
    """Everything the synthetic data and the processors need, from one replica library."""

    def __init__(self, args):
        self.args = args
        self.cfg = HarborScenarioConfig.load(args.scenario)
        self.env = build_environment(self.cfg)
        self.layout = build_layout(self.cfg, self.env)
        self.hcfg = HelmholtzConfig.from_fdtd(self.cfg.fdtd, f_lo=100.0, f_hi=380.0, df=args.df)
        self.solver = HelmholtzSolver(self.env, self.hcfg)
        self.freqs = self.hcfg.freqs
        self.c = self.cfg.environment.c0
        T = self.cfg.fdtd.total_time
        ferry = next(v for v in self.cfg.vessels if v.id == "ferry")
        uuv = next(v for v in self.cfg.vessels if v.kind == "uuv")
        self.t_win = np.arange(WIN_S / 2 + 0.5, T - WIN_S / 2 + 1e-6, WIN_S)       # window centres
        self.ferry_xy = build_track(ferry, T, self.env).position_at(self.t_win)
        self.uuv_xy = build_track(uuv, T, self.env).position_at(self.t_win)
        self.grid = build_grid(self.env, BistaticConfig(grid_dx=args.grid_dx, edge=40.0, grid_margin=5.0))
        # Validation targets: the UUV track at every window plus random off grid points.
        rng = np.random.default_rng(3)
        x0, x1, y0, y1 = self.env.extent
        cand = np.column_stack([rng.uniform(x0 + 50, x1 - 50, 400), rng.uniform(y0 + 50, y1 - 50, 400)])
        cand = cand[self.env.is_water(cand, margin=15.0)]
        far = np.min(np.hypot(cand[:, None, 0] - self.ferry_xy[None, :, 0], cand[:, None, 1] - self.ferry_xy[None, :, 1]), axis=1) > 40
        self.rand_xy = cand[far][: len(self.t_win)]
        self.extra = np.vstack([self.ferry_xy, self.uuv_xy, self.rand_xy])
        self.n_win = len(self.t_win)
        cache = Path(args.out) / "cache"
        t0 = time.time()
        self.lib = replica_library(self.solver, self.layout, self.grid, extra=self.extra, sources=self.ferry_xy,
                                   cache_dir=cache)
        print(f"replica library: {len(self.freqs)} bins, {len(self.grid)} cells, {time.time() - t0:.0f} s")
        self._noise_std = None

    # indices into extra
    def i_ferry(self, k):
        return k

    def i_uuv(self, k):
        return self.n_win + k

    def i_rand(self, j):
        return 2 * self.n_win + j

    @property
    def noise_std(self):
        """Per channel noise std: the median ferry direct level on that sensor type, ``--snr`` dB down."""
        if self._noise_std is None:
            amp = np.abs(self.lib.E[:, : self.n_win, :] * ferry_spectrum(self.freqs)[:, None, None])
            lvl = np.median(amp, axis=(0, 1))
            std = np.empty(self.layout.n_receivers)
            for name, sl in self.layout.slices.items():
                std[sl] = np.median(lvl[sl]) * 10 ** (-self.args.snr / 20)
            self._noise_std = std
        return self._noise_std

    def synth(self, k, target, motor=False, echo=True, seed=0, noise=True, ts=None):
        """``(N_SNAP, n_f, n_ch)`` whitened data for window *k*, target = extra index or None."""
        rng = np.random.default_rng(seed)
        nf, nch = len(self.freqs), self.layout.n_receivers
        Sf = ferry_spectrum(self.freqs)
        Ef = self.lib.E[:, self.i_ferry(k), :]
        ts = self.args.ts if ts is None else ts
        d = np.zeros((N_SNAP, nf, nch), complex)
        for s in range(N_SNAP):
            src = Sf * np.exp(1j * rng.uniform(0, 2 * np.pi, nf))
            P = src[:, None] * Ef
            if target is not None and echo:
                inc = self.lib.incident_extra(k, target)
                P += (src * inc * scatter_coef(self.freqs, ts, self.c))[:, None] * self.lib.E[:, target, :]
            if target is not None and motor:
                tone = np.zeros(nf)
                for fl in UUV_LINES:
                    tone[np.argmin(np.abs(self.freqs - fl))] = 10 ** (-self.args.motor_db / 20)
                P += (tone * np.exp(1j * rng.uniform(0, 2 * np.pi, nf)))[:, None] * self.lib.E[:, target, :]
            if noise:
                P += self.noise_std * (rng.standard_normal((nf, nch)) + 1j * rng.standard_normal((nf, nch))) / np.sqrt(2)
            d[s] = P
        return d / self.noise_std

    def white(self, G):
        return G / self.noise_std

    def reference(self, d, k):
        """Illuminator spectrum per snapshot and bin: beam steered at the ferry (AIS position)."""
        w = self.white(self.lib.E[:, self.i_ferry(k), :])                     # (n_f, n_ch)
        return np.einsum("sfc,fc->sf", d, w.conj()) / np.sum(np.abs(w) ** 2, axis=1)[None]

    def remove_ferry(self, d, k):
        """Project out the ferry's replica at its position (exact position: idealised)."""
        return project_out(d, [self.white(self.lib.E[:, self.i_ferry(k), :])])

    def ferry_mask(self, k, radius=25.0):
        return np.hypot(*(self.grid - self.ferry_xy[k]).T) > radius


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------

def grid_image(sc, values):
    gx, gy = sc.grid[:, 0], sc.grid[:, 1]
    dx = sc.args.grid_dx
    xs = np.arange(gx.min(), gx.max() + dx / 2, dx)
    ys = np.arange(gy.min(), gy.max() + dx / 2, dx)
    img = np.full((len(ys), len(xs)), np.nan)
    img[np.rint((gy - ys[0]) / dx).astype(int), np.rint((gx - xs[0]) / dx).astype(int)] = values
    return img, (xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dx / 2, ys[-1] + dx / 2)


def show_map(ax, sc, P, title, k=None, truth=None, vmin=-20, est=True, cmap="magma"):
    plot_harbor(sc.env, sc.layout, ax=ax, show_depth=False)
    Pn = np.where(np.isfinite(P), P, 0)
    z = db(Pn / max(Pn.max(), 1e-30))
    img, ext = grid_image(sc, z)
    im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=0, zorder=2, interpolation="nearest")
    if k is not None:
        ax.plot(*sc.ferry_xy[k], "s", color="deepskyblue", mec="k", ms=9, zorder=5, label="ferry (illuminator)")
    if truth is not None:
        ax.plot(*truth, "o", mfc="none", mec="lime", mew=2, ms=14, zorder=5, label="UUV truth")
    if est:
        ax.plot(*sc.grid[np.argmax(Pn)], "x", color="w", ms=10, mew=2, zorder=6, label="map peak")
    ax.set_title(title, fontsize=10)
    x0, x1, y0, y1 = sc.env.extent
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    return im


def metrics(sc, P, truth, valid, near=3.0, away=10.0):
    """Error of the map peak [m] and contrast [dB]: best value within *near* m of the truth over
    the strongest value more than *away* m from it."""
    P = np.where(valid, P, 0.0)
    r = np.hypot(*(sc.grid - truth).T)
    err = float(r[np.argmax(P)])
    pk = P[r <= near].max() if np.any(r <= near) else 0.0
    off = P[r > away].max()
    return err, float(db(pk) - db(off)) if off > 0 else float("inf")


def fig_fields(sc, out):
    k = 0
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, f in zip(axes, (140.0, 300.0)):
        u = sc.solver.solve(f, sc.ferry_xy[k:k + 1], full=True)[0]
        v = np.percentile(np.abs(u[sc.env.water]), 99)
        plot_harbor(sc.env, sc.layout, ax=ax, show_depth=False)
        ax.imshow(np.where(sc.env.water, u.real, np.nan), origin="lower", extent=sc.env.extent, cmap="RdBu_r",
                  vmin=-v, vmax=v, zorder=1)
        ax.plot(*sc.ferry_xy[k], "s", color="deepskyblue", mec="k", ms=9, zorder=5)
        ax.plot(*sc.uuv_xy[k], "o", mfc="none", mec="lime", mew=2, ms=12, zorder=5)
        w = sc.hcfg.npml * sc.env.dx
        x0, x1, y0, y1 = sc.env.extent
        ax.plot([x0 + w, x1 - w, x1 - w, x0 + w, x0 + w], [y0 + w, y0 + w, y1 - w, y1 - w, y0 + w], "k:", lw=0.8)
        ax.set_title(f"Re p at {f:.0f} Hz from the ferry (square); UUV circled; dotted: PML edge", fontsize=10)
        ax.get_legend().remove()
    fig.suptitle("Helmholtz field in the bistatic harbor crop: quay and pier reflections, shore loss, pier shadow "
                 f"({sc.hcfg.stencil} stencil, {sc.env.dx} m grid)", fontsize=11)
    fig.savefig(out / "h1_fields.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def gathers(sc, out):
    """Time domain cable gathers from a dense frequency sweep (2.5 Hz steps, 0.4 s period)."""
    f = np.arange(100.0, 380.0 + 1e-6, 2.5)
    key = out / "cache" / f"gathers_{sc.solver.fingerprint()}_{sc.layout.fingerprint()[:8]}.npz"
    k = 0
    if key.exists():
        z = np.load(key)
        Df, Du, inc = z["Df"], z["Du"], z["inc"]
    else:
        pts, inv = np.unique(np.round(sc.layout.sample_points, 6), axis=0, return_inverse=True)
        S = sc.solver.interp_matrix(np.vstack([pts, sc.uuv_xy[k:k + 1]]))
        Df = np.zeros((len(f), sc.layout.n_receivers), complex)
        Du = np.zeros_like(Df)
        inc = np.zeros(len(f), complex)
        t0 = time.time()
        for i, fi in enumerate(f):
            lu = sc.solver.factor(fi)
            v = sc.solver.solve(fi, np.vstack([sc.ferry_xy[k], sc.uuv_xy[k]]), S, lu=lu)   # (2, n_pts + 1)
            Df[i] = sc.layout.collapse(v[0, :-1][inv.ravel()][None])[0]
            Du[i] = sc.layout.collapse(v[1, :-1][inv.ravel()][None])[0]
            inc[i] = v[0, -1]
            if i % 20 == 0:
                print(f"  gathers {fi:6.1f} Hz ({i + 1}/{len(f)}) {time.time() - t0:5.0f} s", flush=True)
        np.savez(key, Df=Df, Du=Du, inc=inc)
    # Source pulse: Tukey tapered flat band, zero phase at t = 0.05 s.
    from scipy.signal.windows import tukey
    W = tukey(len(f), 0.3) * np.exp(-1j * 2 * np.pi * f * 0.05)
    fs, T = 2000.0, 0.4
    nt = int(fs * T)
    bins = np.rint(f * T).astype(int)

    def to_time(D):
        spec = np.zeros((nt // 2 + 1, D.shape[1]), complex)
        # the package convention (H0^(2), np.fft forward transform) inverts directly
        spec[bins] = D * W[:, None]
        return np.fft.irfft(spec, n=nt, axis=0)

    b = scatter_coef(f, sc.args.ts, sc.c)
    direct = to_time(Df)
    echo = to_time((b * inc)[:, None] * Du)
    sl = sc.layout.slices[sc.layout.cables[0].name]
    t = np.arange(nt) / fs
    s = sc.layout.cables[0].arclength
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    v1 = np.percentile(np.abs(direct[:, sl]), 99.5)
    v2 = np.percentile(np.abs(echo[:, sl]), 99.5)
    ratio = db(v2 ** 2 / v1 ** 2)
    for ax, (title, x, v) in zip(axes, (("ferry direct path (plus quay and pier multiples)", direct, v1),
                                        (f"UUV echo alone (TS {sc.args.ts:.0f} dB), own scale: {ratio:.0f} dB re direct", echo, v2),
                                        ("direct + echo, direct scale", direct + echo, v1))):
        ax.imshow(x[:, sl].T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                  extent=(t[0], t[-1], s[0], s[-1]), interpolation="nearest")
        ax.set_title(title, fontsize=10); ax.set_xlabel("time [s]")
    axes[0].set_ylabel("distance along cable [m] (bend at 280 m)")
    fig.suptitle(f"Strain DAS gathers from the Helmholtz model, {f[0]:.0f}-{f[-1]:.0f} Hz pulse, ferry at "
                 f"({sc.ferry_xy[k][0]:.0f}, {sc.ferry_xy[k][1]:.0f}) m", fontsize=11)
    fig.savefig(out / "h2_gathers.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="examples/harbor_bistatic.scenario.json")
    ap.add_argument("--out", default="output/harbor_helmholtz")
    ap.add_argument("--grid-dx", type=float, default=1.0)
    ap.add_argument("--df", type=float, default=20.0)
    ap.add_argument("--ts", type=float, default=-20.0, help="UUV target strength [dB] (Born point scatterer)")
    ap.add_argument("--snr", type=float, default=40.0, help="ferry direct over noise per channel per bin [dB]")
    ap.add_argument("--motor-db", type=float, default=20.0, help="UUV thruster lines below the ferry at 100 Hz [dB]")
    ap.add_argument("--no-gathers", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)

    sc = Scene(args)
    summary = {"scenario": args.scenario, "grid_dx_m": args.grid_dx, "freqs_hz": [float(sc.freqs[0]), float(sc.freqs[-1]), args.df],
               "stencil": sc.hcfg.stencil, "model_dx_m": sc.env.dx, "n_cells": int(len(sc.grid)),
               "ts_db": args.ts, "snr_db": args.snr, "snapshots": N_SNAP, "window_s": WIN_S}

    fig_fields(sc, out)
    if not args.no_gathers:
        gathers(sc, out)

    Gh = sc.white(sc.lib.G)
    Gff = sc.white(free_field_replicas(sc.layout, sc.grid, sc.freqs, sc.c))
    kmid = sc.n_win // 2
    tu = sc.i_uuv(kmid)
    truth = sc.uuv_xy[kmid]
    valid = sc.ferry_mask(kmid)

    # ---------- passive: motor on, ferry projected out ----------
    d = sc.synth(kmid, tu, motor=True, echo=False, seed=1)
    d_res = sc.remove_ferry(d, kmid)
    line_bins = [int(np.argmin(np.abs(sc.freqs - fl))) for fl in UUV_LINES]
    Ph = bartlett(d_res[:, line_bins], Gh[line_bins]); Ph[~valid] = 0
    Pf = bartlett(d_res[:, line_bins], Gff[line_bins]); Pf[~valid] = 0
    eh, ch = metrics(sc, Ph, truth, valid)
    ef, cf = metrics(sc, Pf, truth, valid)
    summary["passive_motor_on"] = {"helmholtz": {"error_m": eh, "contrast_db": ch}, "free_field": {"error_m": ef, "contrast_db": cf}}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    im = show_map(axes[0], sc, Ph, f"Helmholtz replica: error {eh:.1f} m, contrast {ch:.1f} dB", kmid, truth)
    show_map(axes[1], sc, Pf, f"Free field replica on the same data: error {ef:.1f} m, contrast {cf:.1f} dB", kmid, truth)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.8)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Bartlett power [dB re peak]")
    fig.suptitle(f"Passive MFP, UUV motor on (lines at {', '.join(f'{x:.0f}' for x in UUV_LINES)} Hz), ferry direct path "
                 "projected out", fontsize=11)
    fig.savefig(out / "h3_passive.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---------- bistatic: motor off ----------
    def bistatic_run(k, target, seed, use_ff=False, noise=True, echo=True):
        d = sc.synth(k, target, echo=echo, seed=seed, noise=noise)
        ref = sc.reference(d, k)
        d_res = sc.remove_ferry(d, k)
        if use_ff:
            kk = 2 * np.pi * sc.freqs / sc.c
            r = np.maximum(np.hypot(*(sc.grid - sc.ferry_xy[k]).T), 1.0)
            Gin = np.stack([green2d(kf * r) for kf in kk])
            G = Gff
        else:
            Gin, G = sc.lib.incident_grid(k), Gh
        return d_res, ref, Gin, G

    d_res, ref, Gin, G = bistatic_run(kmid, tu, 2)
    Bh = bistatic_coherent(d_res, ref, Gin, G); Bh[~valid] = 0
    d_ff, ref_ff, Gin_ff, G_ff = bistatic_run(kmid, tu, 2, use_ff=True)
    Bf = bistatic_coherent(d_ff, ref_ff, Gin_ff, G_ff); Bf[~valid] = 0
    d_nn, ref_nn, _, _ = bistatic_run(kmid, tu, 2, noise=False)
    Bn = bistatic_coherent(d_nn, ref_nn, Gin, G); Bn[~valid] = 0
    t0 = time.time()
    M = sparse_image(d_res, ref, Gin, G, mask=valid, n_iter=SPARSE_ITER, lam_frac=0.35) ** 2
    t_sparse = time.time() - t0
    res = {}
    for name, P in (("bartlett_helmholtz", Bh), ("bartlett_free_field", Bf), ("bartlett_helmholtz_no_noise", Bn),
                    ("sparse_helmholtz", M)):
        e, c = metrics(sc, P, truth, valid)
        res[name] = {"error_m": e, "contrast_db": c}
    res["sparse_nonzero_cells"] = int(np.sum(M > 0)); res["sparse_time_s"] = t_sparse
    summary["bistatic_motor_off"] = res
    print(json.dumps(res, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    im = show_map(axes[0, 0], sc, Bh, f"Coherent bistatic, Helmholtz replica: error {res['bartlett_helmholtz']['error_m']:.1f} m, "
                  f"contrast {res['bartlett_helmholtz']['contrast_db']:.1f} dB", kmid, truth)
    show_map(axes[0, 1], sc, Bf, f"Same data, free field replica and incident field: error {res['bartlett_free_field']['error_m']:.0f} m",
             kmid, truth)
    show_map(axes[1, 0], sc, Bn, "Helmholtz replica, noise switched off: the floor is the echo's own sidelobes", kmid, truth)
    Mn = np.where(M > 0, M, M[M > 0].min() * 1e-3 if np.any(M > 0) else 1e-30)
    show_map(axes[1, 1], sc, Mn, f"Sparse L1 inversion (FISTA): {res['sparse_nonzero_cells']} nonzero cells, "
             f"error {res['sparse_helmholtz']['error_m']:.1f} m", kmid, truth)
    axes[0, 0].legend(loc="lower left", fontsize=8, framealpha=0.8)
    fig.colorbar(im, ax=axes, shrink=0.6, label="[dB re peak]")
    fig.suptitle(f"Silent UUV from the ferry's echo, t = {sc.t_win[kmid]:.1f} s window, TS {args.ts:.0f} dB, "
                 f"{sc.freqs[0]:.0f}-{sc.freqs[-1]:.0f} Hz, {N_SNAP} snapshots", fontsize=11)
    fig.savefig(out / "h4_bistatic.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---------- validation over many target positions ----------
    rows = []
    targets = [(k, sc.i_uuv(k), "UUV track") for k in range(sc.n_win)] + \
              [(j % sc.n_win, sc.i_rand(j), "random") for j in range(len(sc.rand_xy))]
    maps = []
    for n, (k, ti, kind) in enumerate(targets):
        tr = sc.extra[ti]
        vk = sc.ferry_mask(k)
        d_res, ref, Gin, G = bistatic_run(k, ti, 100 + n)
        B = bistatic_coherent(d_res, ref, Gin, G); B[~vk] = 0
        Mv = sparse_image(d_res, ref, Gin, G, mask=vk, n_iter=SPARSE_ITER, lam_frac=0.35) ** 2
        eb, cb = metrics(sc, B, tr, vk)
        es, cs = metrics(sc, Mv, tr, vk)
        rows.append({"kind": kind, "window": int(k), "t_s": float(sc.t_win[k]), "truth": [float(tr[0]), float(tr[1])],
                     "ferry": [float(v) for v in sc.ferry_xy[k]], "bartlett_error_m": eb, "bartlett_contrast_db": cb,
                     "sparse_error_m": es, "sparse_contrast_db": cs, "sparse_nonzero": int(np.sum(Mv > 0))})
        maps.append((k, tr, B, Mv))
        print(f"  {kind:9s} t={sc.t_win[k]:5.1f} s truth ({tr[0]:6.1f},{tr[1]:6.1f}): Bartlett {eb:5.1f} m {cb:5.1f} dB | "
              f"sparse {es:5.1f} m {cs:6.1f} dB", flush=True)
    # null run: no target
    d_res, ref, Gin, G = bistatic_run(kmid, None, 999)
    M0 = sparse_image(d_res, ref, Gin, G, mask=valid, n_iter=SPARSE_ITER, lam_frac=0.35) ** 2
    typical = np.median([m[3].max() for m in maps])
    summary["validation"] = rows
    summary["null_run"] = {"sparse_nonzero": int(np.sum(M0 > 0)), "max_over_typical_target_peak": float(M0.max() / typical)}
    ok_b = np.mean([r["bartlett_error_m"] <= 2.0 for r in rows])
    ok_s = np.mean([r["sparse_error_m"] <= 2.0 for r in rows])
    summary["validation_fraction_within_2m"] = {"bartlett": float(ok_b), "sparse": float(ok_s)}

    n = len(maps)
    picks = list(range(0, n, max(1, n // 4)))[:4]
    fig = plt.figure(figsize=(17, 14))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8])
    for col, i in enumerate(picks):
        k, tr, B, Mv = maps[i]
        a = fig.add_subplot(gs[0, col]); show_map(a, sc, B, f"Bartlett #{i}: {rows[i]['bartlett_error_m']:.1f} m, "
                                                         f"{rows[i]['bartlett_contrast_db']:.1f} dB", k, tr)
        b = fig.add_subplot(gs[1, col])
        Mn = np.where(Mv > 0, Mv, Mv[Mv > 0].min() * 1e-3 if np.any(Mv > 0) else 1e-30)
        show_map(b, sc, Mn, f"Sparse #{i}: {rows[i]['sparse_error_m']:.1f} m, {min(rows[i]['sparse_contrast_db'], 99):.0f} dB", k, tr)
    ax = fig.add_subplot(gs[2, :3])
    idx = np.arange(n)
    cb_ = np.array([r["bartlett_contrast_db"] for r in rows]); cs_ = np.clip([r["sparse_contrast_db"] for r in rows], -10, 60)
    ax.bar(idx - 0.2, cb_, 0.4, label="Bartlett", color="0.6")
    ax.bar(idx + 0.2, cs_, 0.4, label="sparse (clipped at 60 dB)", color="tab:red")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(idx, [f"#{i}\n{r['kind'][:6]}\n{r['sparse_error_m']:.0f} m" for i, r in enumerate(rows)], fontsize=7)
    ax.set_ylabel("contrast [dB]: target over strongest cell > 10 m away")
    ax.legend(fontsize=9)
    ax.set_title(f"All {n} targets (labels: sparse error). Within 2 m: Bartlett {100 * ok_b:.0f} %, sparse {100 * ok_s:.0f} %", fontsize=10)
    a0 = fig.add_subplot(gs[2, 3])
    M0n = np.where(M0 > 0, M0 / typical, 1e-6)
    plot_harbor(sc.env, sc.layout, ax=a0, show_depth=False)
    img, ext = grid_image(sc, db(M0n))
    a0.imshow(img, origin="lower", extent=ext, cmap="magma", vmin=-20, vmax=0, zorder=2)
    a0.plot(*sc.ferry_xy[kmid], "s", color="deepskyblue", mec="k", ms=9, zorder=5)
    a0.set_title(f"Null run, no UUV: sparse max {db(M0.max() / typical):.0f} dB re a typical target", fontsize=10)
    a0.get_legend().remove()
    fig.suptitle("Validation: coherent bistatic Bartlett versus sparse L1 inversion with the Helmholtz forward operator", fontsize=12)
    fig.savefig(out / "h5_validation.png", dpi=105, bbox_inches="tight"); plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

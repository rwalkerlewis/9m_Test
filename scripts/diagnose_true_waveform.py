#!/usr/bin/env python3
"""Why the stage 1 echo run given the true waveform and the exact track maps the UUV worse than the run with
everything estimated (``docs/harbor.md``, stage 1).

Seed 0 mix of the bistatic dataset.  **Reads the truth** (the true waveform, the exact track, the truth
components): a diagnostic, not part of any processing chain.

1. ``components``: for each reference (data / true waveform) and track (refined / exact), the ECA applied to
   the traces and to the ferry's direct and the UUV's echo truth components, each output mapped with that
   configuration's reference; the reference against the true waveform (delay, coherence).
2. ``lags``: the recorded direct component at each hydrophone against the true waveform propagated in the free
   field at the chart's sound speed (cross correlation per 1 s block).
3. ``dispersion``: the same lags predicted by the FDTD scheme's dispersion model (simulator knowledge), and the
   true waveform map with the dispersion corrected replicas (``--maps``).

Usage::

    python scripts/diagnose_true_waveform.py [components] [lags] [dispersion] [--maps] [--cuda]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "harbor" / "05_detection"))
import harbor_picture_stage1 as s1  # noqa: E402
from acoustic_sim.water.bistatic import (BistaticConfig, Diagnostics, ReplicaModel, _xp, bistatic_map, build_grid,  # noqa: E402
                                         eca_stage, exclusion, reference_spectrum, run_bistatic, ship_positions, stft,
                                         waveform_spectra)
from acoustic_sim.water.diagnostics import SimulatorAwarePropagator, true_illuminator_waveform  # noqa: E402
from acoustic_sim.water.propagation import FreeFieldPropagator  # noqa: E402

cuda = "--cuda" in sys.argv
parts = [p for p in ("components", "lags", "dispersion") if p in sys.argv] or ["components", "lags", "dispersion"]
mx = s1.mixer("bistatic")
obs, truth = mx.mix(0)


def components():
    mx = s1.mixer("bistatic")
    obs, truth = mx.mix(0)
    env, layout = obs.environment, obs.geometry
    prop = FreeFieldPropagator.from_environment(env)
    gain = truth.gain[:, None]
    d = gain * truth.components["direct:ferry"].astype(np.float64)
    e = gain * sum(c.astype(np.float64) for k, c in truth.components.items() if k.startswith("scattered:ferry>"))
    traces = np.asarray(obs.traces, dtype=np.float64)
    rest = traces - d - e
    wf = true_illuminator_waveform(truth, "ferry")
    reports = s1.reports_of(obs.reports, keep=["ferry"])

    for name, ref_kind, exact in [("all_estimated", "data", False), ("true_waveform_exact_track", "true_waveform", True),
                                  ("data_ref_exact_track", "data", True), ("true_waveform_refined_track", "true_waveform", False)]:
        bc = BistaticConfig(f_lo=100.0, f_hi=760.0, c=prop.c, reference=ref_kind, removal="eca", propagator=prop, use_cuda=cuda, **s1.ECHO)
        sp = stft(traces, layout, obs.fs, bc)
        model = ReplicaModel(layout, bc.c, propagator=prop, xp=_xp(bc))
        model.weights = sp.weights
        ex_track = truth.tracks["ferry"].position_at(sp.times) if exact else None
        ship = ship_positions(sp, reports, model, bc, truth=ex_track)
        grid = build_grid(env, bc)
        outs, waveform = eca_stage(traces, layout, obs.fs, env, sp, ship, bc, wf if ref_kind == "true_waveform" else None, [d, e, rest])
        ref = reference_spectrum(sp, ship, model, bc, waveform=waveform, fs=obs.fs)
        uuv = truth.tracks["uuv1"].position_at(sp.times)
        near = np.hypot(grid[:, None, 0] - uuv[None, :, 0], grid[:, None, 1] - uuv[None, :, 1]) <= 5.0
        excl = exclusion(grid, ship, bc.exclusion_radius)
        live = (sp.weights > 0) & (np.arange(layout.n_receivers) < layout.n_das)
        dep = 10 * np.log10(np.sum(d ** 2, 1) / np.maximum(np.sum(outs[1] ** 2, 1), 1e-30))[live]
        loss = 10 * np.log10(np.sum(e ** 2, 1) / np.maximum(np.sum(outs[2] ** 2, 1), 1e-30))[live]
        print(f"\n== {name}: direct depth median {np.median(dep):.1f} dB, echo loss median {np.median(loss):.2f} dB, "
              f"ship error {np.median(np.hypot(*(ship.refined - truth.tracks['ferry'].position_at(sp.times)).T)):.2f} m")
        # Reference against the true waveform: complex ratio per bin, averaged over frames.
        rw = waveform_spectra(wf[:traces.shape[1]], obs.fs, sp)
        ratio = np.sum(ref * np.conj(rw), axis=1) / np.maximum(np.sum(np.abs(rw) ** 2, axis=1), 1e-30)
        coh = np.abs(np.sum(ref * np.conj(rw), axis=1)) / np.sqrt(np.sum(np.abs(ref) ** 2, 1) * np.sum(np.abs(rw) ** 2, 1))
        ph = np.unwrap(np.angle(ratio))
        slope = np.polyfit(sp.freqs, ph, 1)
        print(f"   reference vs true waveform: |ratio| median {np.median(np.abs(ratio)):.3g}, coherence median {np.median(coh):.3f}, "
              f"phase slope {slope[0]:.4g} rad/Hz (delay {-slope[0] / (2 * np.pi) * 1e3:.3f} ms), phase intercept {slope[1]:.2f} rad, "
              f"phase residual std {np.std(ph - np.polyval(slope, sp.freqs)):.2f} rad")
        res = {}
        for lab, x in zip(["all", "direct_residual", "echo", "rest"], outs):
            clean = stft(x, layout, obs.fs, bc)
            clean.weights = sp.weights
            B, _ = bistatic_map(clean, ship, ref, grid, model, bc)
            B = np.asarray(B)
            B[excl] = 0.0
            res[lab] = B
        B = res["all"]
        at_uuv = np.array([B[near[:, k], k].max() for k in range(B.shape[1])])
        pk = grid[B.argmax(0)]
        hit = np.hypot(*(pk - uuv).T) <= 5.0
        print(f"   full map: peak within 5 m of the UUV in {hit.mean():.2f} of frames; median map at UUV {np.median(at_uuv):.3g}, median max {np.median(B.max(0)):.3g}")
        for lab in ("direct_residual", "echo", "rest"):
            Bl = res[lab]
            a = np.array([Bl[near[:, k], k].max() for k in range(Bl.shape[1])])
            print(f"   {lab:16s} alone: median max {np.median(Bl.max(0)):.3g}, median at UUV {np.median(a):.3g}, peak on UUV {np.mean(np.hypot(*(grid[Bl.argmax(0)] - uuv).T) <= 5):.2f}")
        # Where the full map peaks when it misses
        dship = np.hypot(*(pk - ship.refined).T)
        print(f"   misses: distance of the peak to the ship median {np.median(dship[~hit]) if (~hit).any() else np.nan:.1f} m")


def lags():
    env, layout, fs = obs.environment, obs.geometry, obs.fs
    c = env.c_nominal
    wf = true_illuminator_waveform(truth, "ferry")
    n = obs.traces.shape[1]
    wf = wf[:n]
    d = truth.components["direct:ferry"].astype(np.float64)
    f = np.fft.rfftfreq(n, 1 / fs)
    band = (f >= 100) & (f <= 760)
    k = 2 * np.pi * np.maximum(f, 1e-9) / c
    H = np.where(band, np.sqrt(2 / (np.pi * k)) * np.exp(1j * np.pi / 4), 0)
    wf_f = np.fft.irfft(np.fft.rfft(wf) * H, n)
    t = np.arange(n) / fs
    ship = truth.tracks["ferry"].position_at(t)
    U = 8
    up = resample(wf_f, n * U)
    for h in range(layout.n_das, layout.n_receivers):
        r = np.hypot(*(ship - layout.receivers[h]).T)
        pos = (np.arange(n) - r / c * fs) * U
        pred = np.interp(pos, np.arange(n * U), up, left=0, right=0) / np.sqrt(r)
        obs_h = np.fft.irfft(np.fft.rfft(d[h]) * band, n)
        out = []
        for b0 in range(int(1 * fs), n - int(1 * fs), int(1 * fs)):
            a, z = b0, b0 + int(1 * fs)
            x, y = obs_h[a:z], pred[a:z]
            L = int(0.01 * fs)
            xc = np.array([np.dot(x[L:-L], np.roll(y, s)[L:-L]) for s in range(-L, L + 1)])
            j = int(np.argmax(np.abs(xc)))
            # parabolic refinement
            if 0 < j < len(xc) - 1:
                y0, y1, y2 = np.abs(xc[j - 1:j + 2])
                dj = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            else:
                dj = 0.0
            lag = (j - L + dj) / fs * 1e3
            out.append((float(np.mean(r[a:z])), lag, np.sign(xc[j]), np.max(np.abs(xc)) / np.sqrt(np.dot(x, x) * np.dot(y, y))))
        out = np.array(out)
        print(f"hyd {h}: range {out[:,0].min():.0f}-{out[:,0].max():.0f} m, lag of the record behind the free field prediction "
              f"{out[:,1].min():.2f} to {out[:,1].max():.2f} ms (median {np.median(out[:,1]):.2f}), sign {np.median(out[:,2]):+.0f}, peak corr {np.median(out[:,3]):.2f}")
        A = np.column_stack([np.ones(len(out)), out[:, 0]])
        coef, *_ = np.linalg.lstsq(A, out[:, 1], rcond=None)
        print(f"      lag = {coef[0]:.3f} ms + {coef[1]*1e3:.3f} ms/km * range")


def dispersion():
    env, layout, fs = obs.environment, obs.geometry, obs.fs
    sim = SimulatorAwarePropagator.from_truth(truth)
    free = FreeFieldPropagator.from_environment(env)
    f = np.arange(100.0, 761.0, 2.0)
    w = 2 * np.pi * f
    t = np.arange(1.0, 19.0, 1.0)
    ship = truth.tracks["ferry"].position_at(t)
    wf = true_illuminator_waveform(truth, "ferry")
    W = np.abs(np.fft.rfft(wf[:int(fs)]))
    fw = np.fft.rfftfreq(int(fs), 1 / fs)
    wt = np.interp(f, fw, W) ** 2
    for h in range(layout.n_das, layout.n_receivers):
        p = layout.receivers[h]
        dx, dy = ship[:, 0] - p[0], ship[:, 1] - p[1]
        r = np.hypot(dx, dy)
        theta = np.arctan2(np.abs(dy), np.abs(dx))
        kn = np.asarray(sim.wavenumbers(f, theta[None, :]))[:, 0, :] if np.ndim(sim.wavenumbers(f, theta[None, :])) == 3 else np.asarray(sim.wavenumbers(f, theta[None, :]))
        kn = kn.reshape(len(f), -1)
        dphi = (kn - (w / free.c)[:, None]) * r[None, :]                                  # extra phase of the scheme [rad]
        tg = np.array([np.polyfit(w, dphi[:, j], 1, w=np.sqrt(wt))[0] for j in range(len(t))]) * 1e3
        print(f"hyd {h}: predicted group delay of the scheme vs free field {tg.min():.2f} to {tg.max():.2f} ms (median {np.median(tg):.2f}), "
              f"angle {np.degrees(theta).min():.0f}-{np.degrees(theta).max():.0f} deg")

    if "--maps" in sys.argv:
        reports = s1.reports_of(obs.reports, keep=["ferry"])
        for name, ref, prop in [("true_waveform_exact_track / free field", "true_waveform", free),
                                ("true_waveform_exact_track / simulator aware", "true_waveform", sim),
                                ("all_estimated / simulator aware", "data", sim)]:
            bc = BistaticConfig(f_lo=100.0, f_hi=760.0, c=prop.c, reference=ref, removal="eca", propagator=prop, use_cuda=cuda, **s1.ECHO)
            nper, hop = int(round(bc.win_s * fs)), int(round(bc.hop_s * fs))
            times = (np.arange(0, obs.traces.shape[1] - nper + 1, hop) + nper / 2.0) / fs
            diag = Diagnostics(exact_track=truth.tracks["ferry"].position_at(times) if ref == "true_waveform" else None)
            run = run_bistatic(obs.traces, layout, fs, env, reports, bc, true_waveform=wf if ref == "true_waveform" else None, diagnostics=diag)
            B, grid = run.scan.B, run.grid
            uuv = truth.tracks["uuv1"].position_at(run.sp.times)
            pk = grid[B.argmax(0)]
            e = np.hypot(*(pk - uuv).T)
            near = np.hypot(grid[:, None, 0] - uuv[None, :, 0], grid[:, None, 1] - uuv[None, :, 1]) <= 8.0
            at = np.array([grid[near[:, k]][np.argmax(B[near[:, k], k])] for k in range(B.shape[1])])
            print(f"{name}: peak within 5 m of the UUV in {np.mean(e <= 5):.2f} of frames, median map max {np.median(B.max(0)):.3g}, "
                  f"local peak near the UUV offset median {np.median(np.hypot(*(at - uuv).T)):.1f} m")


for p in parts:
    print(f"\n##### {p}")
    globals()[p]()

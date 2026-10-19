#!/usr/bin/env python3
"""Stage 1a: blind reference modes scored in the stage 0a cancellation depth framework.

Stage 0a (``harbor_cancellation_depth.py``) found the blind ``data`` reference (delay and sum
beam of the strain cable) limited the ECA to 9 dB at 193 m against 34 dB with the true
waveform, because the strain response falls as frequency squared and the low end of the band
is noise dominated in the beam.  This script reuses its six cached FDTD cases (the ferry
signature alone at nearest channel ranges of about 193, 99 and 29 m, stationary and moving at
4 m/s) and scores the reference modes of ``BistaticConfig``:

``data``         stage 0: plain beam of every live channel (``ref_mode="beam"``)
``hydrophones``  plain beam of the four pressure hydrophones (``ref_channels="hydrophones"``)
``equalised``    every strain channel divided by its modelled response (floored) before the beam
``snr``          maximum ratio combiner: channel and frequency weights from a first ECA pass
``mrc``          model based maximum ratio combiner: modelled response over the residual power of a first pass
``iterated``     beam, ECA, map, the modelled echo of the strongest peak subtracted, beam again
``oracle``       the true waveform (not blind; context only)

Stationary cases run without Doppler; moving cases run with ``eca_doppler`` and the stage 1b
Kalman track (``track_filter="kalman"``, low speed fallback).  No oracle input enters the blind
modes.  For every mode and case: ECA depth of the direct component (median and 10th percentile
over the live cable channels, 100 Hz to 0.95 f_max) and the coherence of the reference with
the true waveform against frequency.

Criterion (fixed before the run): at least one blind mode reaches a median depth of 20 dB for
the stationary source at both 193 m and 99 m.  The best mode, used in the rest of stage 1, is
the blind mode with the highest mean median depth over the six cases (rule fixed before the
run, written to ``results.json`` first).

Usage::

    python examples/harbor_reference_modes.py [--cuda] [--out output/harbor_reference]
        [--cache output/harbor_cancel/cache] [--modes data,hydrophones,equalised,snr,mrc,iterated,oracle]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")          # the ECA is BLAS bound; several runs share the cores

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, coherence

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from acoustic_sim.harbor import HarborScenarioConfig, build_environment, build_layout, build_track  # noqa: E402
from acoustic_sim.harbor.bistatic import BistaticConfig, ReplicaModel, ShipTrack, eca_stage, ship_positions, stft  # noqa: E402
from acoustic_sim.harbor.dataset import vessel_injection  # noqa: E402
from acoustic_sim.harbor.device import get_device  # noqa: E402
from acoustic_sim.harbor.dispersion import for_scenario  # noqa: E402
from acoustic_sim.harbor.noise import generate_noise  # noqa: E402
from acoustic_sim.harbor.signatures import f_max_for_grid  # noqa: E402
from acoustic_sim.harbor.vessels import VesselConfig  # noqa: E402
from harbor_cancellation_depth import RANGES, SPEED, _run, ais_from_start, band_energy, nearest_channel_range, pick_position  # noqa: E402

CRITERION_DB = 20.0
MODES = {
    # name: BistaticConfig overrides
    "data": dict(reference="data", ref_mode="beam"),
    "hydrophones": dict(reference="data", ref_mode="beam", ref_channels="hydrophones"),
    "equalised": dict(reference="data", ref_mode="equalised"),
    "snr": dict(reference="data", ref_mode="snr"),
    "mrc": dict(reference="data", ref_mode="mrc"),
    "iterated": dict(reference="data", ref_mode="iterated"),
    "oracle": dict(reference="oracle"),
}
BLIND = ("data", "hydrophones", "equalised", "snr", "mrc", "iterated")


def build_cases(cfg, env, layout, total_time: float):
    path_len = SPEED * total_time
    edge = cfg.fdtd.damping_width * env.dx + 15.0
    ferry = next(v for v in cfg.vessels if v.kind == "ship")
    cases = []
    for r in RANGES:
        p = pick_position(env, layout, r, path_len, edge=edge)
        for mode in ("stationary", "moving"):
            wp = [p.tolist()] if mode == "stationary" else [p.tolist(), (p + [path_len, 0.0]).tolist()]
            v = VesselConfig(f"src_{mode}_{int(r)}", "ship", wp, speed=0.0 if mode == "stationary" else SPEED,
                             level_pa=ferry.level_pa, ais=True, seed=ferry.seed)
            cases.append((r, mode, v))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "harbor_bistatic.scenario.json"))
    ap.add_argument("--out", default="output/harbor_reference")
    ap.add_argument("--cache", default="output/harbor_cancel/cache")
    ap.add_argument("--time", type=float, default=10.0, help="simulated duration [s] (stage 0a: 10)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--cuda", action="store_true", help="GPU for the FDTD (cached) and the replicas")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = [m for m in args.modes.split(",") if m]

    cfg = HarborScenarioConfig.load(args.config)
    cfg.fdtd.total_time = args.time
    if args.cuda:
        cfg.fdtd.use_cuda = True
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps
    f_max = f_max_for_grid(env.dx)
    cases = build_cases(cfg, env, layout, args.time)
    log = open(out / "run.log", "w")

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    results = {
        "criterion": {"text": f"at least one blind mode reaches a median ECA depth of {CRITERION_DB:g} dB over the live cable channels "
                              "for the stationary source at both 193 m and 99 m", "depth_db_min": CRITERION_DB,
                      "cases": ["stationary 193 m", "stationary 99 m"], "blind_modes": list(BLIND)},
        "best_mode_rule": "blind mode with the highest mean of the cable median depth over the six cases (moving cases with eca_doppler and the Kalman track)",
        "modes": {m: MODES[m] for m in modes}, "band_hz": [100.0, 0.95 * f_max], "simulated_time_s": args.time,
        "track": {"filter": "kalman", "accel_m_s2": 0.05, "meas_sigma_m": 1.0, "fallback_speed_m_s": 0.3},
        "cases": [],
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))       # criterion and rule on disk before any run
    say(f"criterion: {results['criterion']['text']}")
    say(f"best mode rule: {results['best_mode_rule']}")
    say(f"grid {env.nx} x {env.ny} at {env.dx} m, fs {fs:.0f} Hz, {layout.n_receivers} channels, band 100 to {0.95 * f_max:.0f} Hz")

    jobs, tracks = [], {}
    for r, mode, v in cases:
        tr = build_track(v, cfg.fdtd.total_time, env)
        tracks[v.id] = tr
        sig = vessel_injection(v, tr, env, cfg)
        pos = tuple(v.waypoints[0]) if mode == "stationary" else None
        traj = None if mode == "stationary" else (tr.t, tr.xy)
        jobs.append(dict(env=env, layout=layout, cfg=cfg.fdtd, signal=sig, position=pos, trajectory=traj, cache_dir=args.cache))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        gathers = list(pool.map(_run, jobs))
    say(f"{len(gathers)} FDTD gathers (cached) in {time.time() - t0:.0f} s")

    base = BistaticConfig(win_s=0.5, hop_s=0.25, f_lo=100.0, removal="eca", dispersion=for_scenario(env, cfg.fdtd),
                          track_filter="kalman", ais_pos_sigma=cfg.ais.pos_sigma, ais_sog_sigma=cfg.ais.sog_sigma,
                          ais_cog_sigma=cfg.ais.cog_sigma, use_cuda=args.cuda)
    sos = butter(4, [base.f_lo, 0.95 * f_max], btype="bandpass", fs=fs, output="sos")
    noise = generate_noise(layout, fs, n, cfg.noise, np.random.default_rng(cfg.seed))
    cab = layout.slices[layout.cables[0].name]
    coh = {}
    for (r, mode, v), g in zip(cases, gathers):
        tr = tracks[v.id]
        direct = v.level_pa * g.traces.astype(np.float64)
        truth_meas = noise.gain[:, None] * direct
        traces = noise.apply(direct).astype(np.float64)
        ais = ais_from_start(v.id, tr, mode, cfg.ais, np.random.default_rng(sum(map(ord, v.id))))
        bc0 = BistaticConfig(**{**base.__dict__, "eca_doppler": mode == "moving"})
        sp = stft(traces, layout, fs, bc0, f_max)
        model = ReplicaModel(layout, bc0.c, dispersion=bc0.dispersion, device=get_device(args.cuda))
        model.weights = sp.weights
        oracle = v.level_pa * vessel_injection(v, tr, env, cfg)
        live = sp.weights > 0
        e0 = band_energy(truth_meas, sos)
        ship = ship_positions(sp, ais, model, bc0, vessel=v.id)
        row = {"range_target_m": r, "mode": mode, "range_min_m": float(nearest_channel_range(layout, tr.xy).min()),
               "eca_doppler": mode == "moving",
               "track_error_median_m": None, "refined_error_median_m": float(np.median(np.hypot(*(ship.refined - tr.position_at(sp.times)).T)))}
        for name in modes:
            b = BistaticConfig(**{**bc0.__dict__, **MODES[name]})
            t1 = time.time()
            ship_m = ShipTrack(ship.id, ship.predicted.copy(), ship.refined.copy(), ship.present.copy(), ship.score.copy())   # the track fit happens in eca_stage
            outs, wf = eca_stage(traces, layout, fs, env, sp, ship_m, b, oracle if name == "oracle" else None, [truth_meas],
                                 ais=ais, model=model, f_max=f_max)
            e1 = band_energy(outs[1], sos)
            depth = 10 * np.log10(np.maximum(e0, 1e-30) / np.maximum(e1, 1e-30))
            dc = depth[cab][live[cab]]
            f_c, c = coherence(wf, oracle[: len(wf)], fs=fs, nperseg=int(0.5 * fs))
            band = (f_c >= b.f_lo) & (f_c <= 0.95 * f_max)
            coh[(r, mode, name)] = (f_c[band], c[band])
            if row["track_error_median_m"] is None and ship_m.smoothed is not None:
                row["track_error_median_m"] = float(np.median(np.hypot(*(ship_m.smoothed - tr.position_at(sp.times)).T)))
            row[name] = {"cable_median_db": float(np.median(dc)), "cable_p10_db": float(np.percentile(dc, 10)),
                         "coherence_band_mean": float(np.mean(c[band])), "seconds": time.time() - t1}
            say(f"{mode:10s} {row['range_min_m']:5.0f} m  {name:12s} median {row[name]['cable_median_db']:5.1f} dB  p10 {row[name]['cable_p10_db']:5.1f} dB  "
                f"coherence {row[name]['coherence_band_mean']:.2f}  ({row[name]['seconds']:.0f} s)")
        results["cases"].append(row)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    # Criterion and the best mode.
    stat = {c["range_target_m"]: c for c in results["cases"] if c["mode"] == "stationary"}
    passing = [m for m in modes if m in BLIND and all(stat[r][m]["cable_median_db"] >= CRITERION_DB for r in (200.0, 100.0))]
    means = {m: float(np.mean([c[m]["cable_median_db"] for c in results["cases"]])) for m in modes if m in BLIND}
    best = max(means, key=means.get)
    results["criterion"]["pass"] = bool(passing)
    results["criterion"]["passing_modes"] = passing
    results["mean_depth_by_mode_db"] = means
    results["best_mode"] = best
    results["best_mode_config"] = MODES[best]
    (out / "results.json").write_text(json.dumps(results, indent=2))
    say(f"criterion: {'PASS' if passing else 'FAIL'} (passing modes: {passing or 'none'})")
    say("mean cable median depth over the six cases: " + ", ".join(f"{m} {v:.1f} dB" for m, v in means.items()))
    say(f"best blind mode: {best}")

    # Figure 1: depth by mode.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {m: f"C{i}" for i, m in enumerate(modes)}
    for ax, mode in zip(axes, ("stationary", "moving")):
        rows = [c for c in results["cases"] if c["mode"] == mode]
        rr = [c["range_min_m"] for c in rows]
        for m in modes:
            ax.plot(rr, [c[m]["cable_median_db"] for c in rows], "o-", color=colors[m], label=f"{m}: median")
            ax.plot(rr, [c[m]["cable_p10_db"] for c in rows], "o:", color=colors[m], alpha=0.5, label=f"{m}: p10")
        ax.axhline(CRITERION_DB, color="r", ls="--", label="criterion (stationary)")
        ax.set_xlabel("nearest channel range [m]")
        ax.set_title(f"{mode} source" + ("" if mode == "stationary" else f", {SPEED:g} m/s, Doppler ECA with the Kalman track"))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ECA depth of the direct path, live cable channels [dB]")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Stage 1a: reference modes, fine grid, 100 to {0.95 * f_max:.0f} Hz (best blind mode: {best})")
    fig.savefig(out / "depth_by_mode.png", dpi=110, bbox_inches="tight")

    # Figure 2: coherence with the true waveform against frequency.
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    for j, r in enumerate(RANGES):
        for i, mode in enumerate(("stationary", "moving")):
            ax = axes[i, j]
            for m in modes:
                f_c, c = coh[(r, mode, m)]
                ax.plot(f_c, c, color=colors[m], label=m, lw=1.2)
            row = next(c for c in results["cases"] if c["mode"] == mode and c["range_target_m"] == r)
            ax.set_title(f"{mode}, {row['range_min_m']:.0f} m")
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("frequency [Hz]")
    axes[0, 0].set_ylabel("coherence with the true waveform")
    axes[1, 0].set_ylabel("coherence with the true waveform")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Stage 1a: magnitude squared coherence of each reference with the true ship waveform (0.5 s segments)")
    fig.savefig(out / "coherence_by_mode.png", dpi=110, bbox_inches="tight")
    say(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 1b: the Kalman track for the Doppler ECA, scored on the stage 0a cases.

Stage 0 found that ``eca_doppler`` with the exact track reaches 21 to 25 dB on a ship at
4 m/s, that the AIS refined track reaches 15 to 23 dB only after a 4 s straight line fit
(``track_smooth_s``, tuned on one dataset), and that the per frame jitter of the refined
position, read as velocity, drops even a stationary source from 37 dB to 6 dB.  This script
replaces the line fit by the constant velocity Kalman filter and smoother of
``acoustic_sim.harbor.track`` (AIS reports with the scenario's error statistics plus the refined
positions at 1 m, process noise 0.05 m/s^2, all fixed before the run) with the low speed
fallback (blocks slower than 0.3 m/s use fixed lags at the smoothed position).

All runs use the **oracle waveform**, so the track is isolated from the reference.  Tracks:

``oracle_exact``   exact track (stationary: fixed lags; moving: per sample delays) - the target
``oracle``         AIS refined positions, fixed lags, no Doppler (stage 0a; the stationary target)
``kalman``         AIS refined track, Kalman smoother and fallback, ``eca_doppler`` in every case
``line4``          AIS refined track, 4 s line fit, ``eca_doppler`` (stage 0)
``raw``            AIS refined track unfiltered, ``eca_doppler`` (stage 0 finding 6)

Criterion (fixed before the run): ``kalman`` reaches a cable median depth within 3 dB of
``oracle_exact`` for every moving case and within 3 dB of ``oracle`` for every stationary case.

Usage::

    python examples/harbor_track_eval.py [--cuda] [--out output/harbor_track] [--cache output/harbor_cancel/cache]
"""

from __future__ import annotations

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
from scipy.signal import butter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from acoustic_sim.harbor import HarborScenarioConfig, build_environment, build_layout, build_track  # noqa: E402
from acoustic_sim.harbor.bistatic import BistaticConfig, ReplicaModel, ShipTrack, eca_stage, fit_track, ship_positions, stft  # noqa: E402
from acoustic_sim.harbor.dataset import vessel_injection  # noqa: E402
from acoustic_sim.harbor.device import get_device  # noqa: E402
from acoustic_sim.harbor.dispersion import for_scenario  # noqa: E402
from acoustic_sim.harbor.noise import generate_noise  # noqa: E402
from acoustic_sim.harbor.signatures import f_max_for_grid  # noqa: E402
from harbor_cancellation_depth import RANGES, SPEED, _run, ais_from_start, band_energy, nearest_channel_range  # noqa: E402
from harbor_reference_modes import build_cases  # noqa: E402

MARGIN_DB = 3.0
TRACKS = {
    # name: (exact track, BistaticConfig overrides); eca_doppler for the exact track follows the case
    "oracle_exact": (True, dict()),
    "oracle": (False, dict(eca_doppler=False, track_filter="none")),
    "kalman": (False, dict(eca_doppler=True, track_filter="kalman")),
    "line4": (False, dict(eca_doppler=True, track_filter="line", track_smooth_s=4.0)),
    "raw": (False, dict(eca_doppler=True, track_filter="none", fallback_speed=0.0)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "harbor_bistatic.scenario.json"))
    ap.add_argument("--out", default="output/harbor_track")
    ap.add_argument("--cache", default="output/harbor_cancel/cache")
    ap.add_argument("--time", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

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
        "criterion": {"text": f"kalman within {MARGIN_DB:g} dB of oracle_exact (cable median depth) for every moving case and within "
                              f"{MARGIN_DB:g} dB of oracle (fixed lags, refined positions) for every stationary case; oracle waveform throughout",
                      "margin_db": MARGIN_DB},
        "track": {"filter": "kalman", "accel_m_s2": 0.05, "meas_sigma_m": 1.0, "gate_sigma": 5.0, "fallback_speed_m_s": 0.3,
                  "ais_sigma": [cfg.ais.pos_sigma, cfg.ais.sog_sigma, cfg.ais.cog_sigma]},
        "tracks": {k: v[1] | {"exact_track": v[0]} for k, v in TRACKS.items()}, "cases": [],
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    say(f"criterion: {results['criterion']['text']}")

    jobs, tracks = [], {}
    for r, mode, v in cases:
        tr = build_track(v, cfg.fdtd.total_time, env)
        tracks[v.id] = tr
        sig = vessel_injection(v, tr, env, cfg)
        jobs.append(dict(env=env, layout=layout, cfg=cfg.fdtd, signal=sig, position=tuple(v.waypoints[0]) if mode == "stationary" else None,
                         trajectory=None if mode == "stationary" else (tr.t, tr.xy), cache_dir=args.cache))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        gathers = list(pool.map(_run, jobs))

    base = BistaticConfig(win_s=0.5, hop_s=0.25, f_lo=100.0, removal="eca", reference="oracle", dispersion=for_scenario(env, cfg.fdtd),
                          ais_pos_sigma=cfg.ais.pos_sigma, ais_sog_sigma=cfg.ais.sog_sigma, ais_cog_sigma=cfg.ais.cog_sigma, use_cuda=args.cuda)
    sos = butter(4, [base.f_lo, 0.95 * f_max], btype="bandpass", fs=fs, output="sos")
    noise = generate_noise(layout, fs, n, cfg.noise, np.random.default_rng(cfg.seed))
    cab = layout.slices[layout.cables[0].name]
    errors = {}
    for (r, mode, v), g in zip(cases, gathers):
        tr = tracks[v.id]
        direct = v.level_pa * g.traces.astype(np.float64)
        truth_meas = noise.gain[:, None] * direct
        traces = noise.apply(direct).astype(np.float64)
        ais = ais_from_start(v.id, tr, mode, cfg.ais, np.random.default_rng(sum(map(ord, v.id))))
        sp = stft(traces, layout, fs, base, f_max)
        model = ReplicaModel(layout, base.c, dispersion=base.dispersion, device=get_device(args.cuda))
        model.weights = sp.weights
        oracle = v.level_pa * vessel_injection(v, tr, env, cfg)
        live = sp.weights > 0
        e0 = band_energy(truth_meas, sos)
        exact = tr.position_at(sp.times)
        refined = ship_positions(sp, ais, model, base, vessel=v.id)
        row = {"range_target_m": r, "mode": mode, "range_min_m": float(nearest_channel_range(layout, tr.xy).min())}
        errors[(r, mode)] = {"t": sp.times, "ais": np.hypot(*(refined.predicted - exact).T), "refined": np.hypot(*(refined.refined - exact).T)}
        for name, (use_exact, kw) in TRACKS.items():
            b = BistaticConfig(**{**base.__dict__, **kw})
            if use_exact:
                b = BistaticConfig(**{**b.__dict__, "eca_doppler": mode == "moving"})
                ship = ship_positions(sp, ais, model, b, vessel=v.id, truth=exact)
            else:
                ship = ShipTrack(refined.id, refined.predicted.copy(), refined.refined.copy(), refined.present.copy(), refined.score.copy())
            ship = fit_track(ship, sp.times, ais, b)
            t1 = time.time()
            outs, _ = eca_stage(traces, layout, fs, env, sp, ship, b, oracle, [truth_meas], ais=ais)
            depth = 10 * np.log10(np.maximum(e0, 1e-30) / np.maximum(band_energy(outs[1], sos), 1e-30))
            dc = depth[cab][live[cab]]
            err = np.hypot(*(ship.xy - exact).T)
            speed = np.hypot(*ship.velocity.T) if ship.velocity is not None else np.zeros(len(sp.times))
            errors[(r, mode)][name] = err
            row[name] = {"cable_median_db": float(np.median(dc)), "cable_p10_db": float(np.percentile(dc, 10)),
                         "track_error_median_m": float(np.median(err)), "track_error_max_m": float(err.max()),
                         "speed_median_m_s": float(np.median(speed)), "eca_doppler": b.eca_doppler, "seconds": time.time() - t1}
            say(f"{mode:10s} {row['range_min_m']:5.0f} m  {name:12s} depth {row[name]['cable_median_db']:5.1f} dB (p10 {row[name]['cable_p10_db']:5.1f})  "
                f"track err {row[name]['track_error_median_m']:4.1f} m (max {row[name]['track_error_max_m']:4.1f})  speed {row[name]['speed_median_m_s']:.2f} m/s  ({row[name]['seconds']:.0f} s)")
        target = row["oracle_exact" if mode == "moving" else "oracle"]["cable_median_db"]
        row["target_db"] = target
        row["kalman_minus_target_db"] = row["kalman"]["cable_median_db"] - target
        row["pass"] = bool(row["kalman_minus_target_db"] >= -MARGIN_DB)
        say(f"  -> kalman {row['kalman']['cable_median_db']:.1f} dB against the target {target:.1f} dB: {'PASS' if row['pass'] else 'FAIL'}")
        results["cases"].append(row)
        (out / "results.json").write_text(json.dumps(results, indent=2))
    results["criterion"]["pass"] = bool(all(c["pass"] for c in results["cases"]))
    (out / "results.json").write_text(json.dumps(results, indent=2))
    say(f"criterion: {'PASS' if results['criterion']['pass'] else 'FAIL'} ({sum(c['pass'] for c in results['cases'])} of {len(results['cases'])} cases)")

    # Figure 1: track error against time per case.
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j, r in enumerate(RANGES):
        for i, mode in enumerate(("stationary", "moving")):
            ax = axes[i, j]
            e = errors[(r, mode)]
            ax.plot(e["t"], e["ais"], color="0.6", lw=1, label="AIS prediction")
            ax.plot(e["t"], e["refined"], color="C0", lw=1, label="refined (per frame)")
            for name, c in (("line4", "C2"), ("kalman", "C3")):
                ax.plot(e["t"], e[name], color=c, lw=1.5, label=name)
            row = next(c for c in results["cases"] if c["mode"] == mode and c["range_target_m"] == r)
            ax.set_title(f"{mode}, {row['range_min_m']:.0f} m: kalman {row['kalman']['cable_median_db']:.1f} dB, target {row['target_db']:.1f} dB")
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            if i == 1:
                ax.set_xlabel("time [s]")
    axes[0, 0].set_ylabel("track error [m]")
    axes[1, 0].set_ylabel("track error [m]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Stage 1b: ship track error against time (AIS, per frame refinement, 4 s line fit, Kalman smoother)")
    fig.savefig(out / "track_error.png", dpi=110, bbox_inches="tight")

    # Figure 2: depth table.
    names = list(TRACKS)
    fig, ax = plt.subplots(figsize=(12, 5))
    cases_sorted = results["cases"]
    w = 0.8 / len(names)
    for k, name in enumerate(names):
        ax.bar(np.arange(len(cases_sorted)) + (k - 0.5 * (len(names) - 1)) * w, [c[name]["cable_median_db"] for c in cases_sorted], w, label=name)
    ax.set_xticks(np.arange(len(cases_sorted)))
    ax.set_xticklabels([f"{c['mode']}\n{c['range_min_m']:.0f} m" for c in cases_sorted], fontsize=8)
    ax.set_ylabel("ECA depth, cable median [dB]")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax.set_title("Stage 1b: cancellation depth by track (oracle waveform; moving cases with per sample delays)")
    fig.savefig(out / "depth_by_track.png", dpi=110, bbox_inches="tight")
    say(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 0a: direct path cancellation depth of ECA against range, stationary and moving source.

One broadband ship source (the ferry's signature) runs alone through the fine bistatic harbor
(0.5 m grid, strain DAS, quays) at three ranges from the nearest cable channel, first
stationary and then moving at 4 m/s parallel to the first cable leg.  Each gather is mixed
with the scenario noise and processed by the ECA stage of ``harbor/bistatic.py``; the
noise free direct component goes through the same projection, so the energy it loses is the
cancellation depth of each channel.  Three references are compared:

``data``         delay and sum beam at the AIS refined position (blind)
``oracle``       the true source waveform, AIS refined position
``oracle_exact`` the true source waveform and the exact source position

Criterion (fixed before the run): median depth over the live cable channels, in the
processing band, with the ``data`` reference, at least 20 dB.  Interpretation (fixed before
the run): low at every range for the stationary source points at the reference or removal
method; low only at the shortest range points at near field channels; stationary passes but
moving fails points at the moving source representation or the ECA block length.

The crop is 400 m x 300 m with a 30 m sponge, so the longest range available is about
200 m; the ranges are 200, 100 and 30 m (the docs proposed 400, 150 and 30 m on the full
harbor).

Usage::

    python examples/harbor_cancellation_depth.py [--time 10] [--workers 2] [--out output/harbor_cancel]
        [--cuda] [--gate 50] [--doppler]

``--gate`` excludes channels closer than that to the source from the statistics; ``--doppler``
switches the ECA to per sample delays (``BistaticConfig.eca_doppler``).  Both write their
outputs with a suffix next to the baseline, which keeps its criterion and results.
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

from acoustic_sim.harbor import HarborScenarioConfig, build_environment, build_layout, build_track  # noqa: E402
from acoustic_sim.harbor.ais import AISConfig, AISReports  # noqa: E402
from acoustic_sim.harbor.bistatic import BistaticConfig, ReplicaModel, eca_stage, ship_positions, stft  # noqa: E402
from acoustic_sim.harbor.dataset import vessel_injection  # noqa: E402
from acoustic_sim.harbor.dispersion import for_scenario  # noqa: E402
from acoustic_sim.harbor.fdtd_runner import simulate_source  # noqa: E402
from acoustic_sim.harbor.noise import generate_noise  # noqa: E402
from acoustic_sim.harbor.plotting import plot_harbor  # noqa: E402
from acoustic_sim.harbor.signatures import f_max_for_grid  # noqa: E402
from acoustic_sim.harbor.vessels import VesselConfig  # noqa: E402

RANGES = (200.0, 100.0, 30.0)
SPEED = 4.0
CRITERION_DB = 20.0
REFERENCES = ("data", "oracle", "oracle_exact")


def nearest_channel_range(layout, pts: np.ndarray) -> np.ndarray:
    das = layout.das_positions
    return np.min(np.hypot(pts[:, None, 0] - das[None, :, 0], pts[:, None, 1] - das[None, :, 1]), axis=1)


def pick_position(env, layout, target: float, path_len: float, edge: float = 45.0, margin: float = 20.0) -> np.ndarray:
    """Start point whose whole path (start to start + (path_len, 0)) is in open water, at least
    *edge* m from the domain boundary (outside the sponge), and whose nearest channel range over
    the path is closest to *target*; among equally good paths the one centred on the first
    cable leg wins (a tie break, so the sources sit over the middle of the leg, not at an edge)."""
    from scipy.ndimage import distance_transform_edt

    x0, x1, y0, y1 = env.extent
    dist = distance_transform_edt(env.water) * env.dx          # once; env.is_water(margin=...) recomputes it per call
    leg = layout.cables[0].waypoints[:2]
    x_mid = 0.5 * (leg[0, 0] + leg[1, 0])
    best, best_err = None, np.inf
    for x in np.arange(x0 + edge, x1 - edge - path_len, 2.0):
        for y in np.arange(y0 + edge, y1 - edge, 2.0):
            path = np.column_stack([np.linspace(x, x + path_len, 21), np.full(21, y)])
            if not np.all(dist[env._index(path)] >= margin):
                continue
            r = nearest_channel_range(layout, path)
            err = abs(r.min() - target) + 0.5 * (r.max() - r.min())   # prefer paths at a constant range
            err += 1e-3 * abs(x + 0.5 * path_len - x_mid)              # tie break: centred on the first leg
            if err < best_err:
                best, best_err = np.array([x, y]), err
    if best is None:
        raise RuntimeError(f"no open water position for range {target} m")
    return best


def ais_from_start(vid: str, tr, mode: str, ais_cfg: AISConfig, rng: np.random.Generator) -> AISReports:
    """AIS reports with the scenario's position, speed and course errors, the first one at t = 0.

    A real illuminator has reported long before the analysis window, so every frame has a
    prediction; :func:`generate_ais` would start the first report up to one interval late.
    """
    interval = ais_cfg.interval_moving if mode == "moving" else ais_cfg.interval_stationary
    t = np.arange(0.0, tr.t[-1] + 1e-9, interval)
    xy = tr.position_at(t) + ais_cfg.pos_sigma * rng.standard_normal((len(t), 2))
    vel = tr.position_at(t + 0.5) - tr.position_at(t - 0.5)                 # direction only (clamped at the ends)
    sog = tr.speed_at(t) + (ais_cfg.sog_sigma * rng.standard_normal(len(t)) if mode == "moving" else 0.0)
    cog = np.arctan2(vel[:, 1], vel[:, 0]) + ais_cfg.cog_sigma * rng.standard_normal(len(t))
    return AISReports(vessel=np.array([vid] * len(t)), t=t, xy=xy, sog=sog, cog=cog)


def band_energy(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return np.sum(sosfiltfilt(sos, x, axis=-1) ** 2, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "harbor_bistatic.scenario.json"))
    ap.add_argument("--out", default="output/harbor_cancel")
    ap.add_argument("--cache", default="output/harbor_cancel/cache")
    ap.add_argument("--time", type=float, default=10.0, help="simulated duration [s]")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gate", type=float, default=0.0, help="exclude channels closer than this [m] to the source from the statistics")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--doppler", action="store_true", help="eca_doppler=True (per sample delays following the ship); outputs get a _doppler suffix")
    ap.add_argument("--smooth", type=float, default=0.0, help="track_smooth_s for the refined track (with --doppler); adds a _smooth<s> suffix")
    ap.add_argument("--coarse", action="store_true", help="1 m grid (for quick checks only)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = HarborScenarioConfig.load(args.config)
    if args.coarse:
        cfg.environment.dx = 1.0
        cfg.fdtd.fs = 3000.0
        cfg.noise.band_scale = 5.0
    cfg.fdtd.total_time = args.time
    if args.cuda:
        cfg.fdtd.use_cuda = True
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps
    f_max = f_max_for_grid(env.dx)
    ferry = next(v for v in cfg.vessels if v.kind == "ship")
    path_len = SPEED * args.time
    edge = cfg.fdtd.damping_width * env.dx + 15.0          # the sponge is damping_width cells wide

    cases = []
    for r in RANGES:
        p = pick_position(env, layout, r, path_len, edge=edge)
        for mode in ("stationary", "moving"):
            wp = [p.tolist()] if mode == "stationary" else [p.tolist(), (p + [path_len, 0.0]).tolist()]
            v = VesselConfig(f"src_{mode}_{int(r)}", "ship", wp, speed=0.0 if mode == "stationary" else SPEED,
                             level_pa=ferry.level_pa, ais=True, seed=ferry.seed)
            cases.append((r, mode, v))
    log = open(out / ("run.log" if not args.doppler else "run_doppler" + (f"_smooth{args.smooth:g}" if args.smooth else "") + ".log"), "w")

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"grid {env.nx} x {env.ny} at {env.dx} m, fs {fs:.0f} Hz, {n} steps, {layout.n_receivers} channels, band to {f_max:.0f} Hz")
    say(f"criterion: median depth over live cable channels with the data reference >= {CRITERION_DB} dB")

    # FDTD runs (cached), all six in a pool.
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    tracks = {}
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
    say(f"{len(gathers)} FDTD runs in {time.time() - t0:.0f} s")

    bc = BistaticConfig(win_s=0.5, hop_s=0.25, f_lo=100.0, removal="eca", dispersion=for_scenario(env, cfg.fdtd), eca_doppler=args.doppler,
                        track_smooth_s=args.smooth)
    sos = butter(4, [bc.f_lo, 0.95 * f_max], btype="bandpass", fs=fs, output="sos")
    noise = generate_noise(layout, fs, n, cfg.noise, np.random.default_rng(cfg.seed))
    cab = layout.slices[layout.cables[0].name]
    hyd = layout.slices.get("hydrophones", slice(layout.n_das, layout.n_receivers))
    results = []
    per_channel = {}
    for (r, mode, v), g in zip(cases, gathers):
        tr = tracks[v.id]
        direct = v.level_pa * g.traces.astype(np.float64)
        truth_meas = noise.gain[:, None] * direct
        traces = noise.apply(direct).astype(np.float64)
        ais = ais_from_start(v.id, tr, mode, cfg.ais, np.random.default_rng(sum(map(ord, v.id))))
        sp = stft(traces, layout, fs, bc, f_max)
        model = ReplicaModel(layout, bc.c, dispersion=bc.dispersion)
        model.weights = sp.weights
        oracle = v.level_pa * vessel_injection(v, tr, env, cfg)
        live = sp.weights > 0
        ch_range = np.min(np.hypot(*(layout.receivers[:, None, :] - tr.position_at(tr.t)[None]).transpose(2, 0, 1)), axis=1)
        keep = live & (ch_range >= args.gate)
        e0 = band_energy(truth_meas, sos)
        row = {"range_target_m": r, "mode": mode, "start_xy": v.waypoints[0],
               "range_min_m": float(nearest_channel_range(layout, tr.xy).min()),
               "range_max_m": float(nearest_channel_range(layout, tr.xy).max())}
        for ref in REFERENCES:
            b = BistaticConfig(**{**bc.__dict__, "reference": "data" if ref == "data" else "oracle"})
            ship = ship_positions(sp, ais, model, b, vessel=v.id, truth=tr.position_at(sp.times) if ref == "oracle_exact" else None)
            outs, _ = eca_stage(traces, layout, fs, env, sp, ship, b, oracle if ref != "data" else None, [truth_meas])
            e1 = band_energy(outs[1], sos)
            depth = 10 * np.log10(np.maximum(e0, 1e-30) / np.maximum(e1, 1e-30))
            dc = depth[cab][keep[cab]]
            dh = depth[hyd][keep[hyd]]
            pos_err = float(np.median(np.hypot(*(ship.refined - tr.position_at(sp.times)).T)))
            row[ref] = {"cable_median_db": float(np.median(dc)), "cable_p10_db": float(np.percentile(dc, 10)),
                        "hydrophone_median_db": float(np.median(dh)) if len(dh) else None,
                        "position_error_median_m": pos_err}
            per_channel[(r, mode, ref)] = (ch_range, depth, keep)
        row["pass"] = bool(row["data"]["cable_median_db"] >= CRITERION_DB)
        results.append(row)
        say(f"{mode:10s} range {row['range_min_m']:6.1f} m: " + "  ".join(
            f"{ref} {row[ref]['cable_median_db']:5.1f} dB (p10 {row[ref]['cable_p10_db']:5.1f}, pos err {row[ref]['position_error_median_m']:4.1f} m)"
            for ref in REFERENCES) + f"  -> {'PASS' if row['pass'] else 'FAIL'}")

    summary = {
        "criterion": f"median ECA depth over live cable channels, {bc.f_lo:.0f} to {0.95 * f_max:.0f} Hz, data reference >= {CRITERION_DB} dB",
        "interpretation_fixed_in_advance": [
            "low at every range for the stationary source: reference or removal method",
            "low only at the shortest range: near field channels",
            "stationary passes, moving fails: moving source representation or ECA block length",
        ],
        "simulated_time_s": args.time, "gate_m": args.gate, "eca_block_s": bc.eca_block_s,
        "eca_halfwidth_s": bc.eca_halfwidth_s, "speed_m_s": SPEED, "cases": results,
        "assumptions": ["single source alone (no echo, no other ships)", "AIS reports every interval from t = 0, no dropouts, scenario position error", "scenario noise model"],
    }
    sfx = ("" if args.gate == 0 else f"_gate{int(args.gate)}") + ("_doppler" if args.doppler else "") + (f"_smooth{args.smooth:g}" if args.smooth else "")
    summary["eca_doppler"] = args.doppler
    summary["track_smooth_s"] = args.smooth
    (out / f"results{sfx}.json").write_text(json.dumps(summary, indent=2))

    # Figure 1: depth against range.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, mode in zip(axes, ("stationary", "moving")):
        for ref, c in zip(REFERENCES, ("C0", "C1", "C2")):
            rows = [x for x in results if x["mode"] == mode]
            rr = [x["range_min_m"] for x in rows]
            ax.plot(rr, [x[ref]["cable_median_db"] for x in rows], "o-", color=c, label=f"{ref}: median")
            ax.plot(rr, [x[ref]["cable_p10_db"] for x in rows], "o:", color=c, alpha=0.6, label=f"{ref}: 10th percentile")
        ax.axhline(CRITERION_DB, color="r", ls="--", label="criterion")
        ax.set_xlabel("nearest channel range [m]")
        ax.set_title(f"{mode} source" + ("" if mode == "stationary" else f", {SPEED:g} m/s parallel to the cable"))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ECA cancellation depth, live cable channels [dB]")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Direct path cancellation by ECA, fine grid, {bc.f_lo:.0f} to {0.95 * f_max:.0f} Hz" + (f", channels within {args.gate:g} m excluded" if args.gate else "")
                 + (", per sample delays (eca_doppler)" if args.doppler else "") + (f", track smoothed over {args.smooth:g} s" if args.smooth else ""))
    fig.savefig(out / f"depth_vs_range{sfx}.png", dpi=110, bbox_inches="tight")

    # Figure 2: depth per channel against its range, data reference.
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for j, r in enumerate(RANGES):
        for i, mode in enumerate(("stationary", "moving")):
            ax = axes[i, j]
            for ref, c in zip(("data", "oracle_exact"), ("C0", "C2")):
                rng_, depth, keep = per_channel[(r, mode, ref)]
                m = keep.copy()
                m[hyd] = False
                ax.plot(rng_[m], depth[m], ".", color=c, ms=3, label=ref)
            ax.axhline(CRITERION_DB, color="r", ls="--")
            ax.set_title(f"{mode}, target range {r:g} m")
            ax.set_xlabel("channel to source range [m]")
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("depth [dB]"); axes[1, 0].set_ylabel("depth [dB]")
    axes[0, 0].legend()
    fig.savefig(out / f"depth_per_channel{sfx}.png", dpi=110, bbox_inches="tight")

    # Figure 3: source positions.
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_harbor(env, layout, ax=ax, show_depth=False, sponge_cells=cfg.fdtd.damping_width)
    for r, mode, v in cases:
        xy = tracks[v.id].xy
        ax.plot(*xy.T, "-" if mode == "moving" else "", marker="o" if mode == "stationary" else None, lw=2, label=f"{mode} {r:g} m")
    ax.legend(fontsize=8)
    ax.set_title("Source positions for the cancellation depth test")
    fig.savefig(out / "positions.png", dpi=100, bbox_inches="tight")
    say(f"wrote {out}")


def _run(kw):
    return simulate_source(**kw)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Direct path removal: the reference estimators of ``BistaticConfig.ref_mode`` on the stage 0a cases.

One broadband ship source (the ferry's signature) runs alone through the bistatic harbor (0.5 m grid, strain DAS,
quays, the k space scheme of the recordings) at three ranges from the nearest cable channel (about 193, 99 and 29 m),
first stationary and then moving at 4 m/s parallel to the first cable leg.  Each gather is mixed with the scenario
noise and its direct path removed by the ECA of the echo chain, with the chain's own settings (the ``echo.bistatic``
block of ``examples/configs/processing/multistatic.json``: per sample delays, the track fitted over 4 s); the ship
position is the AIS prediction refined on the traces (no truth).  The noise free direct component goes through the
same projection, so the energy it loses is the cancellation depth of each channel (a diagnostic, from the truth).

Rows: every ``ref_mode`` (``beam``, ``weighted``, ``weighted_cable``, ``hydrophones``, ``iterated``, ``joint``;
everything estimated from the traces) and, given the truth, the true waveform with the refined track and the true
waveform with the exact track.  Per row and case: median and 10th percentile depth over the live cable channels, the
coherence of the reference with the true waveform against frequency (a diagnostic, from the truth), and the compute
time of the reference and the ECA per second of data.  In these cases there is no body, so ``iterated`` has no echo
to subtract and is the beam again.

Criterion, fixed before the run (prompt ``direct_path``, section 1): at least one mode with no truth input reaches a
median depth of 20 dB on the stationary source at about 193 m and at about 99 m, and 15 dB on the moving source at
about 193 m with the tracked position.  The mode with no truth input of the highest mean median depth over the six
cases becomes the default ``ref_mode`` of every processing config.

The FDTD runs are cached in ``output/studies/direct_path/cache/`` (six runs of 10 s).  Outputs in
``output/studies/direct_path/``: ``results.json`` (the criterion first, then every number), ``depth_table.md``,
``run.log`` and the figures.

Usage::

    python examples/harbor/05_detection/harbor_cancellation_depth.py [--time 10] [--cpu] [--out output/studies/direct_path]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import butter, coherence, sosfiltfilt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Provenance  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water import HarborScenarioConfig, build_environment, build_layout, build_track  # noqa: E402
from acoustic_sim.water.ais import AISReports  # noqa: E402
from acoustic_sim.water.ais_sim import AISConfig  # noqa: E402
from acoustic_sim.water.bistatic import REF_MODES, ReplicaModel, eca_stage, ship_positions, stft  # noqa: E402
from acoustic_sim.water.dataset import vessel_injection  # noqa: E402
from acoustic_sim.water.fdtd_runner import simulate_sources  # noqa: E402
from acoustic_sim.water.noise import generate_noise  # noqa: E402
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402
from acoustic_sim.water.signatures import f_max_for_grid  # noqa: E402
from acoustic_sim.water.stream import load_processing  # noqa: E402
from acoustic_sim.water.vessels import VesselConfig  # noqa: E402

RANGES = (200.0, 100.0, 30.0)          # targets; the open water positions give about 193, 99 and 29 m
SPEED = 4.0
TRUTH_ROWS = ("true_waveform", "true_waveform_exact_track")
CRITERION = {
    "text": "at least one mode with no truth input reaches a median depth over the live cable channels of 20 dB on the "
            "stationary source at about 193 m and at about 99 m, and 15 dB on the moving source at about 193 m with the "
            "tracked position; the mode with no truth input of the highest mean median depth over the six cases becomes "
            "the default ref_mode of every processing config",
    "stationary_db": 20.0, "stationary_ranges": (200.0, 100.0), "moving_db": 15.0, "moving_ranges": (200.0,),
}


def nearest_channel_range(layout, pts: np.ndarray) -> np.ndarray:
    das = layout.das_positions
    return np.min(np.hypot(pts[:, None, 0] - das[None, :, 0], pts[:, None, 1] - das[None, :, 1]), axis=1)


def pick_position(env, layout, target: float, path_len: float, edge: float = 45.0, margin: float = 20.0) -> np.ndarray:
    """Start point whose whole path (start to start + (path_len, 0)) is in open water, at least *edge* m from the
    domain boundary (outside the sponge), and whose nearest channel range over the path is closest to *target*;
    among equally good paths the one centred on the first cable leg wins."""
    from scipy.ndimage import distance_transform_edt

    x0, x1, y0, y1 = env.extent
    dist = distance_transform_edt(env.water) * env.dx
    leg = layout.cables[0].waypoints[:2]
    x_mid = 0.5 * (leg[0, 0] + leg[1, 0])
    best, best_err = None, np.inf
    for x in np.arange(x0 + edge, x1 - edge - path_len, 2.0):
        for y in np.arange(y0 + edge, y1 - edge, 2.0):
            path = np.column_stack([np.linspace(x, x + path_len, 21), np.full(21, y)])
            if not np.all(dist[env._index(path)] >= margin):
                continue
            r = nearest_channel_range(layout, path)
            err = abs(r.min() - target) + 0.5 * (r.max() - r.min())
            err += 1e-3 * abs(x + 0.5 * path_len - x_mid)
            if err < best_err:
                best, best_err = np.array([x, y]), err
    if best is None:
        raise RuntimeError(f"no open water position for range {target} m")
    return best


def ais_from_start(vid: str, tr, mode: str, ais_cfg: AISConfig, rng: np.random.Generator) -> AISReports:
    """AIS reports with the scenario's position, speed and course errors, the first one at t = 0."""
    interval = ais_cfg.interval_moving if mode == "moving" else ais_cfg.interval_stationary
    t = np.arange(0.0, tr.t[-1] + 1e-9, interval)
    xy = tr.position_at(t) + ais_cfg.pos_sigma * rng.standard_normal((len(t), 2))
    vel = tr.position_at(t + 0.5) - tr.position_at(t - 0.5)
    sog = np.hypot(*vel.T) + (ais_cfg.sog_sigma * rng.standard_normal(len(t)) if mode == "moving" else 0.0)
    cog = np.arctan2(vel[:, 1], vel[:, 0]) + ais_cfg.cog_sigma * rng.standard_normal(len(t))
    return AISReports(vessel=np.array([vid] * len(t)), t=t, xy=xy, sog=sog, cog=cog)


def band_energy(x: np.ndarray, sos: np.ndarray, trim: int) -> np.ndarray:
    return np.sum(sosfiltfilt(sos, x, axis=-1)[:, trim:-trim] ** 2, axis=1)


def coherence_curve(ref: np.ndarray, true: np.ndarray, fs: float, band) -> tuple[np.ndarray, np.ndarray]:
    f, c = coherence(ref, true, fs=fs, nperseg=int(0.5 * fs))
    m = (f >= band[0]) & (f <= band[1])
    return f[m], c[m]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "examples" / "configs" / "harbor_bistatic.scenario.json"))
    ap.add_argument("--processing", default=str(ROOT / "examples" / "configs" / "processing" / "multistatic.json"))
    ap.add_argument("--out", default="output/studies/direct_path")
    ap.add_argument("--time", type=float, default=10.0, help="simulated duration [s]")
    ap.add_argument("--cpu", action="store_true", help="FDTD and processing on the CPU")
    ap.add_argument("--modes", nargs="*", default=list(REF_MODES) + list(TRUTH_ROWS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"

    cfg = HarborScenarioConfig.load(args.config)
    cfg.fdtd.total_time = args.time
    cfg.fdtd.use_cuda = not args.cpu
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps
    f_max = f_max_for_grid(env.dx)
    proc = load_processing(Path(args.processing))
    bc0 = proc.bistatic(env.public(), use_cuda=not args.cpu)
    band = (bc0.f_lo, bc0.f_hi)
    ferry = next(v for v in cfg.vessels if v.kind == "ship")
    path_len = SPEED * args.time
    edge = cfg.fdtd.damping_width * env.dx + 15.0

    log = open(out / "run.log", "w")

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    # The criterion goes into results.json before anything is measured.
    summary = {"criterion": CRITERION, "status": "running", "provenance": {
        **{m: Provenance(truth_inputs=["components"], simulator_knowledge=[]).to_dict() for m in REF_MODES},
        "true_waveform": Provenance(truth_inputs=["waveform", "components"], simulator_knowledge=[]).to_dict(),
        "true_waveform_exact_track": Provenance(truth_inputs=["waveform", "track", "components"], simulator_knowledge=[]).to_dict()}}
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    say(f"criterion: {CRITERION['text']}")
    say(f"grid {env.nx} x {env.ny} at {env.dx} m, fs {fs:.0f} Hz, {n} steps, {layout.n_receivers} channels, "
        f"time scheme {cfg.fdtd.time_scheme}, processing band {band[0]:.0f} to {band[1]:.0f} Hz")

    cases = []
    for r in RANGES:
        p = pick_position(env, layout, r, path_len, edge=edge)
        for mode in ("stationary", "moving"):
            wp = [p.tolist()] if mode == "stationary" else [p.tolist(), (p + [path_len, 0.0]).tolist()]
            v = VesselConfig(f"src_{mode}_{int(r)}", "ship", wp, speed=0.0 if mode == "stationary" else SPEED,
                             level_pa=ferry.level_pa, ais=True, seed=ferry.seed)
            cases.append((r, mode, v))
    jobs, tracks = [], {}
    for r, mode, v in cases:
        tr = build_track(v, cfg.fdtd.total_time, env)
        tracks[v.id] = tr
        sig = vessel_injection(v, tr, env, cfg)
        jobs.append(dict(env=env, layout=layout, cfg=cfg.fdtd, signal=sig,
                         position=tuple(v.waypoints[0]) if mode == "stationary" else None,
                         trajectory=None if mode == "stationary" else (tr.t, tr.xy), cache_dir=cache))
    t0 = time.time()
    gathers = simulate_sources(env, layout, cfg.fdtd, jobs, cache_dir=cache)
    say(f"{len(gathers)} FDTD runs (cached in {cache}) in {time.time() - t0:.0f} s")

    sos = butter(4, [band[0], band[1]], btype="bandpass", fs=fs, output="sos")
    trim = int(0.5 * fs)
    noise = generate_noise(layout, fs, n, cfg.noise, np.random.default_rng(cfg.seed))
    cab = layout.slices[layout.cables[0].name]
    rows = []
    curves = {}
    for (r, mode, v), g in zip(cases, gathers):
        tr = tracks[v.id]
        direct = v.level_pa * g.traces.astype(np.float64)
        truth_meas = noise.gain[:, None] * direct
        traces = noise.apply(direct).astype(np.float64)
        ais = ais_from_start(v.id, tr, mode, cfg.ais, np.random.default_rng(sum(map(ord, v.id))))
        sp = stft(traces, layout, fs, bc0)
        model = ReplicaModel(layout, bc0.c, propagator=bc0.propagator)
        model.weights = sp.weights
        given = v.level_pa * vessel_injection(v, tr, env, cfg)
        live = sp.weights > 0
        e0 = band_energy(truth_meas, sos, trim)
        rng_ = nearest_channel_range(layout, tr.xy)
        case = {"range_target_m": r, "mode": mode, "start_xy": v.waypoints[0], "range_min_m": float(rng_.min()),
                "range_max_m": float(rng_.max()), "rows": {}}
        ship_est = ship_positions(sp, ais, model, bc0, vessel=v.id)
        ship_exact = ship_positions(sp, ais, model, bc0, vessel=v.id, truth=tr.position_at(sp.times))
        pos_err = float(np.median(np.hypot(*(ship_est.refined - tr.position_at(sp.times)).T)))
        case["position_error_median_m"] = pos_err
        for row in args.modes:
            if row in TRUTH_ROWS:
                b = replace(bc0, reference="true_waveform", ref_mode="beam")
                ship, wf = (ship_exact if row == "true_waveform_exact_track" else ship_est), given
            else:
                b = replace(bc0, ref_mode=row)
                ship, wf = ship_est, None
            t1 = time.perf_counter()
            outs, ref = eca_stage(traces, layout, fs, env.public(), sp, ship, b, wf, [truth_meas])
            dt = time.perf_counter() - t1
            depth = 10 * np.log10(np.maximum(e0, 1e-30) / np.maximum(band_energy(outs[1], sos, trim), 1e-30))
            dc = depth[cab][live[cab]]
            f, c = coherence_curve(ref, given, fs, band)
            curves[(r, mode, row)] = (f, c)
            case["rows"][row] = {
                "cable_median_db": float(np.median(dc)), "cable_p10_db": float(np.percentile(dc, 10)),
                "coherence_median": float(np.median(c)),
                "coherence_by_band": {f"{lo:.0f}-{hi:.0f}": float(np.median(c[(f >= lo) & (f < hi)]))
                                      for lo, hi in ((100, 200), (200, 400), (400, band[1] + 1))},
                "compute_s_per_s": dt / args.time,
            }
            say(f"{mode:10s} {case['range_min_m']:6.1f} m  {row:26s} median {np.median(dc):5.1f} dB  p10 {np.percentile(dc, 10):5.1f} dB  "
                f"coherence {np.median(c):.3f}  compute {dt / args.time:.2f} s/s")
        rows.append(case)

    # Criterion and the default.
    estimated = [m for m in args.modes if m not in TRUTH_ROWS]

    def get(rt, mode, row):
        return next(c for c in rows if c["range_target_m"] == rt and c["mode"] == mode)["rows"][row]["cable_median_db"]

    passing = [m for m in estimated
               if all(get(rt, "stationary", m) >= CRITERION["stationary_db"] for rt in CRITERION["stationary_ranges"])
               and all(get(rt, "moving", m) >= CRITERION["moving_db"] for rt in CRITERION["moving_ranges"])]
    means = {m: float(np.mean([c["rows"][m]["cable_median_db"] for c in rows])) for m in args.modes}
    best = max(estimated, key=lambda m: means[m])
    summary.update(status="done", passed=bool(passing), passing_modes=passing, mean_median_db=means, best_mode=best,
                   cases=rows, band_hz=list(band), simulated_time_s=args.time, speed_m_s=SPEED,
                   time_scheme=cfg.fdtd.time_scheme, chain_settings={k: getattr(bc0, k) for k in (
                       "eca_block_s", "eca_halfwidth_s", "eca_images", "eca_doppler", "track_smooth_s", "eca_gram")},
                   assumptions=["single source alone (no echo, no other ships)", "AIS reports every interval from t = 0",
                                "scenario noise model", "iterated: no body in these cases, so it is the beam again"],
                   coherence={f"{c[0]:.0f} m {c[1]} {c[2]}": [v[0].tolist(), v[1].tolist()] for c, v in curves.items()})
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    say(f"criterion {'PASS' if passing else 'FAIL'}: passing modes {passing}; best mode by the mean over the six cases: {best} "
        f"({means[best]:.1f} dB)")

    # The depth table.
    hdr = [f"{c['mode']} {c['range_min_m']:.0f} m" for c in rows]
    lines = ["| Reference | Truth given | " + " | ".join(hdr) + " | Mean of the medians [dB] | Compute per s of data |",
             "|---|---|" + "---|" * len(hdr) + "---|---|"]
    for m in args.modes:
        given = {"true_waveform": "waveform", "true_waveform_exact_track": "waveform, track"}.get(m, "none")
        cells = [f"{c['rows'][m]['cable_median_db']:.1f} / {c['rows'][m]['cable_p10_db']:.1f}" for c in rows]
        comp = np.median([c["rows"][m]["compute_s_per_s"] for c in rows])
        lines.append(f"| {m} | {given} | " + " | ".join(cells) + f" | {means[m]:.1f} | {comp:.2f} |")
    (out / "depth_table.md").write_text("\n".join(lines) + "\n")

    # Figures: median depth per case, one series per row; coherence against frequency per case.
    labels = [f"{c['mode'][:4]} {c['range_min_m']:.0f} m" for c in rows]
    for stat, name in (("cable_median_db", "median"), ("cable_p10_db", "10th percentile")):
        fig, ax = new_figure((13, 8))
        for m in args.modes:
            ax.plot(range(len(rows)), [c["rows"][m][stat] for c in rows], "o-" if m not in TRUTH_ROWS else "s--", label=m)
        ax.axhline(CRITERION["stationary_db"], color="k", ls=":", label="criterion, stationary (20 dB)")
        ax.axhline(CRITERION["moving_db"], color="0.5", ls=":", label="criterion, moving (15 dB)")
        ax.set_xticks(range(len(rows)), labels)
        ax.set_xlabel("case (source mode, nearest channel range)")
        ax.set_ylabel(f"{name} depth over live cable channels [dB]")
        ax.set_title(f"bistatic harbor, stage 0a cases: {name} direct path removal depth per reference")
        ax.legend(ncol=2)
        save_figure(fig, out / f"depth_{name.split()[0]}.png", recording="bistatic")
    for c in rows:
        fig, ax = new_figure((12, 8))
        for m in args.modes:
            f, coh = curves[(c["range_target_m"], c["mode"], m)]
            ax.plot(f, coh, "-" if m not in TRUTH_ROWS else "--", label=m)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("coherence of the reference with the true waveform")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"bistatic harbor, {c['mode']} source at {c['range_min_m']:.0f} m: reference coherence "
                     f"(a diagnostic from the truth)")
        ax.legend(ncol=2)
        save_figure(fig, out / f"coherence_{c['mode']}_{c['range_target_m']:.0f}m.png", recording="bistatic")
    fig, ax = new_figure((12, 9))
    plot_harbor(env, layout, ax=ax, show_depth=False, sponge_cells=cfg.fdtd.damping_width)
    for r, mode, v in cases:
        xy = tracks[v.id].xy
        ax.plot(*xy.T, "-" if mode == "moving" else "", marker="o" if mode == "stationary" else None, lw=2, label=f"{mode} {r:g} m")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlabel("x (east) [m]")
    ax.set_ylabel("y (north) [m]")
    ax.set_title("bistatic harbor: source positions of the direct path removal cases")
    save_figure(fig, out / "positions.png", recording="bistatic")
    say(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 0b: the bistatic demo with its truth inputs removed one at a time.

The demo (``harbor_bistatic_demo.py``) locates the silent UUV from the ferry's echo with three
truth inputs: the direct path removed with the truth component, the true ferry waveform and
the exact ferry track.  This script runs ``run_bistatic`` on the coarse and the fine dataset in
four configurations that change one assumption at a time:

1. ``direct_removed_exactly``          direct path removed exactly, true waveform, exact track (the
                                       demo; on the coarse dataset with ``--propagator simulator`` it
                                       must reproduce ``demo/demo_summary.json``)
2. ``true_waveform_exact_track``       direct path left in the traces and removed by ECA, true
                                       waveform, exact track
3. ``estimated_waveform_exact_track``  ECA with the data reference (beam at the ship), exact track
4. ``all_estimated``                   ECA with the data reference, AIS track with Bartlett
                                       refinement: everything estimated from the traces

For every configuration and dataset: median and p90 position error, fraction of frames within
5 m, median echo contrast (map peak over the map median, dB), the same after a 1.5 s (three
frame) integration, and the ECA depth of the direct component and loss of the echo component
(a diagnostic: the truth components go through the same projection in a separate
``eca_stage`` call after the run, so they never enter the processing and the run's provenance is
unchanged).

Criterion (fixed before the run): configuration 4 on the fine dataset reaches a median echo
contrast of at least 10 dB and at least half of the frames within 5 m.  If it fails, the
configuration between 1 and 4 that loses the most contrast is the step to work on.

``all_estimated`` reads the observation's traces and uses ``FreeFieldPropagator`` and nothing else.
Configurations 1 to 3 assemble their input from the truth components (that is what they are for) and
read their other truth inputs from the dataset's ``Truth``; their ``provenance`` says so
(``truth_inputs`` includes ``"components"``).  The default replicas are the free field ones;
``--propagator simulator`` gives configurations 1 to 3 the FDTD dispersion corrected replicas
(simulator knowledge, flagged; that is how stage 0b was first run, ``output/harbor_bistatic/ablation/``).
The band is a stated processing choice: 100 to 380 Hz on the 1 m coarse dataset, 100 to 760 Hz on the
0.5 m fine one.

``--doppler`` adds the same configurations 2 to 4 with ``eca_doppler=True`` (per sample delays
following the moving ship inside each ECA block, added in stage 0a); they are reported as
variants and are not part of the criterion.

Usage::

    python examples/harbor_bistatic_ablation.py [--fine output/harbor_bistatic/fine/dataset.npz]
        [--coarse output/harbor_bistatic/coarse/dataset.npz] [--out output/harbor_bistatic/ablation_freefield]
        [--propagator {free_field,simulator}]
        [--doppler] [--configs direct_removed_exactly,true_waveform_exact_track,estimated_waveform_exact_track,all_estimated]
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

from acoustic_sim.contract import Provenance, dataset_exists  # noqa: E402
from acoustic_sim.water import HarborDataset  # noqa: E402
from acoustic_sim.water.bistatic import BistaticConfig, Diagnostics, eca_stage, run_bistatic, stft  # noqa: E402
from acoustic_sim.water.diagnostics import SimulatorAwarePropagator, true_illuminator_waveform  # noqa: E402
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402
from acoustic_sim.water.propagation import FreeFieldPropagator  # noqa: E402

CONFIGS = {
    # name: (direct path in the traces, reference, exact track)
    "direct_removed_exactly": (False, "true_waveform", True),
    "true_waveform_exact_track": (True, "true_waveform", True),
    "estimated_waveform_exact_track": (True, "data", True),
    "all_estimated": (True, "data", False),
}
TRUTH_INPUTS = {
    "direct_removed_exactly": ["direct path removed with the truth component", "true ferry waveform", "exact ferry track"],
    "true_waveform_exact_track": ["true ferry waveform", "exact ferry track"],
    "estimated_waveform_exact_track": ["exact ferry track"],
    "all_estimated": [],
}
BAND_HZ = {"coarse": (100.0, 380.0), "fine": (100.0, 760.0)}     # stated processing band per dataset
C_NOMINAL = 1495.0
CRITERION = {"config": "all_estimated", "dataset": "fine", "contrast_db_median_min": 10.0, "within_5m_min": 0.5}
METRICS = ("median_error_m", "p90_error_m", "within_5m", "contrast_db_median", "int_median_error_m", "int_within_5m",
           "eca_direct_depth_db", "eca_echo_loss_db")
METRIC_LABELS = {
    "median_error_m": "median error [m]", "p90_error_m": "p90 error [m]", "within_5m": "fraction within 5 m",
    "contrast_db_median": "median contrast [dB]", "int_median_error_m": "1.5 s integrated median error [m]",
    "int_within_5m": "1.5 s integrated fraction within 5 m", "eca_direct_depth_db": "ECA depth of the direct path [dB]",
    "eca_echo_loss_db": "ECA loss of the echo [dB]",
}


def run_config(name: str, obs, truth, band: tuple[float, float], doppler: bool = False, smooth: float = 0.0,
               propagator: str = "free_field", use_cuda: bool = False) -> tuple[dict, object, np.ndarray, np.ndarray]:
    """One configuration on one dataset: summary dict, the run, the UUV truth and the estimates."""
    direct_in, reference, exact = CONFIGS[name]
    env, layout = obs.environment, obs.geometry
    comps = truth.components
    gain = truth.gain[:, None]
    direct = gain * comps["direct:ferry"].astype(float)
    echo = gain * comps["scattered:ferry>uuv1"].astype(float)
    noise = comps["noise"].astype(float)
    if name == "all_estimated":
        traces, given_inputs = np.asarray(obs.traces, dtype=float), []
        prop = FreeFieldPropagator(C_NOMINAL)
    else:
        traces = echo + noise + (direct if direct_in else 0.0)
        given_inputs = ["components"] + ([] if direct_in else ["direct_removal"])
        prop = SimulatorAwarePropagator.from_truth(truth) if propagator == "simulator" else FreeFieldPropagator(C_NOMINAL)
    given = true_illuminator_waveform(truth, "ferry") if reference == "true_waveform" else None
    bc = BistaticConfig(win_s=0.5, hop_s=0.25, f_lo=band[0], f_hi=band[1], bin_stride=2, grid_dx=2.0, c=C_NOMINAL,
                        reference=reference, removal="eca" if direct_in else "none", method="coherent", coherent_norm="snr",
                        exclusion_radius=25.0, eca_doppler=doppler, track_smooth_s=smooth if doppler else 0.0,
                        propagator=prop, use_cuda=use_cuda)
    sp = stft(traces, layout, obs.fs, bc)
    diag = Diagnostics(exact_track=truth.tracks["ferry"].position_at(sp.times) if exact else None)
    t0 = time.time()
    run = run_bistatic(traces, layout, obs.fs, env, obs.reports, bc, true_waveform=given, diagnostics=diag,
                       provenance=Provenance(truth_inputs=given_inputs))
    elapsed = time.time() - t0
    B, grid, times = run.scan.B, run.grid, run.sp.times
    uuv = truth.tracks["uuv1"].position_at(times)
    est = grid[np.argmax(B, axis=0)]
    err = np.hypot(*(est - uuv).T)
    B3 = np.zeros_like(B)
    for k in range(B.shape[1]):
        B3[:, k] = B[:, max(0, k - 1):k + 2].sum(axis=1)
    est3 = grid[np.argmax(B3, axis=0)]
    err3 = np.hypot(*(est3 - uuv).T)
    contrast = 10 * np.log10(np.max(B, axis=0) / np.maximum(np.median(B, axis=0), 1e-30))
    ship_err = np.hypot(*(run.ship.refined - truth.tracks["ferry"].position_at(times)).T)
    live = run.sp.weights[layout.slices["cable1"]] > 0
    summary = {
        "config": name, "doppler": doppler, "track_smooth_s": smooth if doppler else 0.0, "truth_inputs": TRUTH_INPUTS[name],
        "provenance": run.provenance.to_dict(), "frames": int(len(times)),
        "band_hz": [bc.f_lo, bc.f_hi], "chain_s": elapsed, "propagator": type(prop).__name__,
        "median_error_m": float(np.median(err)), "p90_error_m": float(np.percentile(err, 90)),
        "within_5m": float(np.mean(err <= 5.0)), "within_10m": float(np.mean(err <= 10.0)),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))), "contrast_db_median": float(np.median(contrast)),
        "integrated_1p5s": {"median_error_m": float(np.median(err3)), "p90_error_m": float(np.percentile(err3, 90)),
                            "within_5m": float(np.mean(err3 <= 5.0)), "rmse_m": float(np.sqrt(np.mean(err3 ** 2)))},
        "int_median_error_m": float(np.median(err3)), "int_within_5m": float(np.mean(err3 <= 5.0)),
        "ship_position_error_median_m": float(np.median(ship_err)),
        "eca_direct_depth_db": None, "eca_echo_loss_db": None,
    }
    if direct_in:
        # Diagnostic (truth components through the same projection, after the run): ECA depth and echo loss.
        outs, _ = eca_stage(traces, layout, obs.fs, env, run.sp, run.ship, bc, given, [direct, echo])
        cab = layout.slices["cable1"]
        depth = 10 * np.log10(np.sum(direct ** 2, axis=1) / np.maximum(np.sum(outs[1] ** 2, axis=1), 1e-30))
        loss = 10 * np.log10(np.sum(echo ** 2, axis=1) / np.maximum(np.sum(outs[2] ** 2, axis=1), 1e-30))
        summary["eca_direct_depth_db"] = float(np.median(depth[cab][live]))
        summary["eca_echo_loss_db"] = float(np.median(loss[cab][live]))
        summary["eca_diagnostic_provenance"] = run.provenance.with_flags(["components"]).to_dict()
    return summary, run, uuv, est


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fine", default="output/harbor_bistatic/fine/dataset.npz")
    ap.add_argument("--coarse", default="output/harbor_bistatic/coarse/dataset.npz")
    ap.add_argument("--demo-summary", default="output/harbor_bistatic/demo/demo_summary.json")
    ap.add_argument("--out", default=None, help="default: output/harbor_bistatic/ablation_freefield (free field) or "
                    "output/harbor_bistatic/ablation (--propagator simulator)")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--doppler", action="store_true", help="also run configurations 2 to 4 with eca_doppler=True")
    ap.add_argument("--smooth", type=float, default=0.0, help="track_smooth_s of the Doppler variants (local straight line fit of the refined track)")
    ap.add_argument("--propagator", choices=("free_field", "simulator"), default="free_field",
                    help="replicas of configurations 1 to 3: free field (default) or FDTD dispersion corrected (simulator "
                         "knowledge); all_estimated always uses the free field")
    ap.add_argument("--cuda", action="store_true", help="run the chain on the GPU (maps within 1e-12 of the CPU)")
    ap.add_argument("--frame", type=int, default=40, help="frame index of maps_by_config.png")
    args = ap.parse_args()
    out = Path(args.out or ("output/harbor_bistatic/ablation" if args.propagator == "simulator" else "output/harbor_bistatic/ablation_freefield"))
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "w")

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    names = [c for c in args.configs.split(",") if c]
    variants = [(c, False) for c in names] + ([(c, True) for c in names if c != "direct_removed_exactly"] if args.doppler else [])
    results = {"criterion": {**CRITERION, "text": "configuration 'all_estimated' on the fine dataset: median echo contrast >= 10 dB and >= 50 % of frames within 5 m"},
               "configs": {c: {"direct_in_traces": v[0], "reference": v[1], "exact_track": v[2], "truth_inputs": TRUTH_INPUTS[c]} for c, v in CONFIGS.items()},
               "propagator": args.propagator, "datasets": {}, "metrics": {}}
    (out / "results.json").write_text(json.dumps(results, indent=2))       # the criterion is on disk before any run
    say(f"criterion: {results['criterion']['text']}")
    maps = {}
    for tag, path in (("coarse", args.coarse), ("fine", args.fine)):
        if not dataset_exists(path):
            say(f"[{tag}] {path} missing, skipped")
            continue
        ds = HarborDataset.load(path)
        obs, truth = ds.observation, ds.truth
        env, layout = obs.environment, obs.geometry
        band = BAND_HZ[tag]
        say(f"[{tag}] {path}: {env.dx} m chart, fs {obs.fs:.0f} Hz, band {band[0]:.0f} to {band[1]:.0f} Hz, {layout.n_receivers} channels")
        results["datasets"][tag] = {"path": path, "dx": env.dx, "fs": obs.fs, "band_hz": list(band)}
        results["metrics"][tag] = {}
        for name, dop in variants:
            key = name + (("+doppler" + (f"+smooth{args.smooth:g}" if args.smooth else "")) if dop else "")
            summary, run, uuv, est = run_config(name, obs, truth, band, doppler=dop, smooth=args.smooth, propagator=args.propagator,
                                                use_cuda=args.cuda)
            results["metrics"][tag][key] = summary
            maps[tag, key] = (run, uuv, est, env, layout)
            eca = (f", ECA direct {summary['eca_direct_depth_db']:.1f} dB, echo loss {summary['eca_echo_loss_db']:.2f} dB"
                   if summary["eca_direct_depth_db"] is not None else "")
            say(f"[{tag}] {key:20s} median {summary['median_error_m']:5.1f} m, p90 {summary['p90_error_m']:6.1f} m, "
                f"within 5 m {100 * summary['within_5m']:3.0f} %, contrast {summary['contrast_db_median']:5.1f} dB, "
                f"1.5 s: median {summary['int_median_error_m']:5.1f} m, within 5 m {100 * summary['int_within_5m']:3.0f} %, "
                f"ship err {summary['ship_position_error_median_m']:.1f} m{eca} ({summary['chain_s']:.0f} s)")
            if (tag == "coarse" and name == "direct_removed_exactly" and not dop and args.propagator == "simulator"
                    and Path(args.demo_summary).exists()):
                demo = json.loads(Path(args.demo_summary).read_text())
                same = all(abs(demo[k] - summary[k]) < 1e-6 for k in ("median_error_m", "p90_error_m", "within_5m", "contrast_db_median"))
                results["reproduces_demo_summary"] = bool(same)
                say(f"[coarse] direct_removed_exactly configuration reproduces demo_summary.json: {same} "
                    f"(demo median {demo['median_error_m']:.2f} m, contrast {demo['contrast_db_median']:.2f} dB)")
            (out / "results.json").write_text(json.dumps(results, indent=2))
    crit = results["metrics"].get("fine", {}).get("all_estimated")
    if crit is not None:
        ok = crit["contrast_db_median"] >= CRITERION["contrast_db_median_min"] and crit["within_5m"] >= CRITERION["within_5m_min"]
        results["criterion"]["pass"] = bool(ok)
        say(f"criterion: all_estimated on fine: contrast {crit['contrast_db_median']:.1f} dB, within 5 m {100 * crit['within_5m']:.0f} % -> {'PASS' if ok else 'FAIL'}")
        order = [k for k in results["metrics"]["fine"] if "+doppler" not in k]
        drops = {f"{a} -> {b}": results["metrics"]["fine"][a]["contrast_db_median"] - results["metrics"]["fine"][b]["contrast_db_median"]
                 for a, b in zip(order[:-1], order[1:])}
        results["contrast_loss_by_step_fine_db"] = drops
        say("contrast lost per step (fine): " + ", ".join(f"{k} {v:+.1f} dB" for k, v in drops.items()))
    (out / "results.json").write_text(json.dumps(results, indent=2))

    # Bar figure per metric.
    tags = list(results["metrics"])
    keys = [k for k in results["metrics"][tags[0]]]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, m in zip(axes.ravel(), METRICS):
        w = 0.8 / max(len(tags), 1)
        for j, tag in enumerate(tags):
            vals = [results["metrics"][tag][k].get(m) for k in keys]
            vals = [np.nan if v is None else v for v in vals]
            ax.bar(np.arange(len(keys)) + (j - 0.5 * (len(tags) - 1)) * w, vals, w, label=tag)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS[m], fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        if m == "contrast_db_median":
            ax.axhline(CRITERION["contrast_db_median_min"], color="r", ls="--", label="criterion")
        if m == "within_5m":
            ax.axhline(CRITERION["within_5m_min"], color="r", ls="--", label="criterion")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Bistatic ablation: truth inputs removed one at a time (coherent map, 0.5 s frames)\n"
                 f"replicas: {'FDTD dispersion corrected (simulator knowledge)' if args.propagator == 'simulator' else 'free field (nothing beyond the chart)'}",
                 fontsize=12)
    fig.savefig(out / "metrics_by_config.png", dpi=110, bbox_inches="tight")

    # Maps: one frame per configuration and dataset.
    rows = tags
    fig, axes = plt.subplots(len(rows), len(keys), figsize=(4.6 * len(keys), 4.2 * len(rows)), squeeze=False)
    for i, tag in enumerate(rows):
        for j, key in enumerate(keys):
            ax = axes[i, j]
            run, uuv, est, env, layout = maps[tag, key]
            B, grid, times = run.scan.B, run.grid, run.sp.times
            k = min(args.frame, len(times) - 1)
            xs, ys = np.unique(grid[:, 0]), np.unique(grid[:, 1])
            ix = np.rint((grid[:, 0] - xs[0]) / 2.0).astype(int)
            iy = np.rint((grid[:, 1] - ys[0]) / 2.0).astype(int)
            img = np.full((len(ys), len(xs)), np.nan)
            img[iy, ix] = 10 * np.log10(np.maximum(B[:, k], 1e-12))
            plot_harbor(env, layout, ax=ax, show_depth=False)
            ax.imshow(img, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="magma", vmin=-20, vmax=0, alpha=0.9, zorder=2)
            ax.plot(*run.ship.refined[k], "s", color="deepskyblue", ms=8, mec="k", zorder=5)
            ax.plot(*uuv[k], "o", mfc="none", mec="lime", mew=2, ms=14, zorder=5)
            ax.plot(*est[k], "x", color="w", ms=10, mew=2, zorder=6)
            s = results["metrics"][tag][key]
            prov = Provenance.from_dict(s["provenance"])
            ax.set_title(f"{tag}: {key} [{'all estimated' if prov.all_estimated else 'not all estimated'}]\nt = {times[k]:.1f} s, err {np.hypot(*(est[k] - uuv[k])):.1f} m, contrast {s['contrast_db_median']:.1f} dB", fontsize=9)
            ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
            if ax.get_legend():
                ax.get_legend().remove()
    fig.suptitle("Bistatic maps by configuration (dB re frame peak; square: ship position used, circle: UUV truth, x: estimate)", fontsize=11)
    fig.savefig(out / "maps_by_config.png", dpi=100, bbox_inches="tight")
    say(f"wrote {out}")


if __name__ == "__main__":
    main()

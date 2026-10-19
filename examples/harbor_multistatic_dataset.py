#!/usr/bin/env python3
"""Stage 0c: multi illuminator, multi body harbor dataset for the later multistatic stages.

Same 0.5 m crop and sensor layout as the bistatic scenario, with three illuminators (the AIS
ferry, an AIS tug crossing its geometry, a small radiating boat without AIS), three silent
moving bodies (two UUVs, a diver) and seven static bodies (a row of six 1 m pilings and a
moored 6 m x 30 m hull by the north quay).  Every radiating vessel is run once with all the
bodies present and once with none (``render_vessel_gathers(combined_bodies=True)``), so the
scattered component of each illuminator includes the multiple scattering between the bodies.
The dataset keeps the direct, scattered and noise components separately and writes the truth
of every object per frame.

Outputs in ``--out/<fine|coarse>/``: ``dataset.npz``, ``truth.json`` (per object: kind, radiates,
reports AIS, body size, and per frame position and velocity), ``map.png``, ``components.png``,
``levels.json``, ``scenario_used.json`` and ``run.log``.  ``ferry_only_map.png`` is the oracle
bistatic chain (direct path removed with the truth component, oracle ferry waveform, exact
ferry track) on the ferry's scattered component alone, with the truth of every body marked, to
confirm that the echoes of the bodies appear; it is a check of the dataset, not a result.

Usage::

    python examples/harbor_multistatic_dataset.py [--config examples/harbor_multistatic.scenario.json]
        [--out output/harbor_multistatic] [--coarse] [--time 20] [--workers 2] [--cuda]
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

from acoustic_sim.water import (  # noqa: E402
    HarborScenarioConfig,
    assemble_dataset,
    build_environment,
    build_layout,
    build_track,
    render_vessel_gathers,
)
from acoustic_sim.water.bistatic import BistaticConfig, Truth, run_bistatic, stft  # noqa: E402
from acoustic_sim.water.dataset import COMBINED, scattered_key, vessel_injection  # noqa: E402
from acoustic_sim.water.dispersion import for_scenario  # noqa: E402
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402
from acoustic_sim.water.signatures import f_max_for_grid  # noqa: E402

TYPE_STYLE = {
    "illuminator (AIS)": dict(color="deepskyblue", ls="-"),
    "illuminator (no AIS)": dict(color="orange", ls="-"),
    "silent moving body": dict(color="lime", ls="--"),
    "static body": dict(color="magenta", ls=""),
}


def coarsen(cfg: HarborScenarioConfig, dx: float) -> HarborScenarioConfig:
    """Same scenario on grid spacing *dx*: sample rate and noise band scale with the grid, and
    the sponge keeps its width in metres (``damping_width`` is in cells; the bistatic script
    leaves it, so its coarse sponge is 60 m)."""
    cfg.fdtd.damping_width = int(round(cfg.fdtd.damping_width * cfg.environment.dx / dx))
    cfg.environment.dx = dx
    cfg.fdtd.fs = 600.0 * 5.0 / dx
    cfg.noise.band_scale = 5.0 / dx
    return cfg


def object_type(v) -> str:
    if v.level_pa > 0:
        return "illuminator (AIS)" if v.has_ais else "illuminator (no AIS)"
    return "static body" if v.kind == "static" or len(v.waypoints) == 1 else "silent moving body"


def band_rms(x: np.ndarray, sos: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sosfiltfilt(sos, x, axis=-1) ** 2)))


def body_outline(v, xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Polygon of the body footprint at *xy* with unit *heading* (for the map)."""
    a, b = 0.5 * v.body_length, 0.5 * v.body_width
    n = np.array([-heading[1], heading[0]])
    if v.body_shape == "ellipse":
        th = np.linspace(0, 2 * np.pi, 33)
        return xy + np.outer(a * np.cos(th), heading) + np.outer(b * np.sin(th), n)
    return np.array([xy + a * heading + b * n, xy - a * heading + b * n, xy - a * heading - b * n, xy + a * heading - b * n, xy + a * heading + b * n])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "harbor_multistatic.scenario.json"))
    ap.add_argument("--out", default="output/harbor_multistatic")
    ap.add_argument("--coarse", action="store_true", help="1 m grid instead of 0.5 m")
    ap.add_argument("--time", type=float, default=None, help="override simulated duration [s]")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cuda", action="store_true", help="run the FDTD on the GPU (the cache key ignores this)")
    ap.add_argument("--hop", type=float, default=0.25, help="frame hop of the truth file [s]")
    ap.add_argument("--no-check", action="store_true", help="skip the ferry only bistatic check map")
    args = ap.parse_args()

    cfg = HarborScenarioConfig.load(args.config)
    if args.coarse:
        coarsen(cfg, 1.0)
    if args.time is not None:
        cfg.fdtd.total_time = args.time
    if args.cuda:
        cfg.fdtd.use_cuda = True
    tag = "coarse" if args.coarse else "fine"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "scenario_used.json")
    log = open(out / "run.log", "w")

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    f_max = f_max_for_grid(env.dx)
    say(f"[{tag}] grid {env.nx} x {env.ny} at {env.dx} m, fs {cfg.fdtd.fs:.0f} Hz, band to {f_max:.0f} Hz, "
        f"{layout.n_receivers} channels ({layout.n_samples} sample points), {cfg.fdtd.n_steps} steps, "
        f"{sum(v.level_pa > 0 for v in cfg.vessels)} illuminators, {sum(v.is_body for v in cfg.vessels)} bodies")

    t0 = time.time()
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=Path(args.out) / "cache", workers=args.workers,
                                    combined_bodies=True)
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=args.seed, keep_components=True)
    say(f"[{tag}] {len(gathers)} gathers ({', '.join(gathers)}) in {time.time() - t0:.0f} s")
    ds.save(out / "dataset.npz")

    # Truth per object and frame.
    fs = ds.fs
    frames = np.arange(0.25, cfg.fdtd.total_time - 0.25 + 1e-9, args.hop)
    truth = {"frame_times_s": frames.tolist(), "objects": {}}
    for v in cfg.vessels:
        tr = tracks[v.id]
        pos = tr.position_at(frames)
        vel = (tr.position_at(frames + 0.25) - tr.position_at(frames - 0.25)) / 0.5
        truth["objects"][v.id] = {
            "type": object_type(v), "kind": v.kind, "radiates": bool(v.level_pa > 0), "level_pa": v.level_pa,
            "reports_ais": bool(v.has_ais and v.level_pa > 0), "is_body": bool(v.is_body),
            "body": {"length": v.body_length, "width": v.body_width, "shape": v.body_shape, "kind": v.body_kind} if v.is_body else None,
            "position_m": pos.tolist(), "velocity_m_s": vel.tolist(), "active": tr.active_at(frames).tolist(),
        }
    (out / "truth.json").write_text(json.dumps(truth, indent=1))

    # Levels on the sensors per component.
    sos = butter(4, [0.25 * f_max, 0.95 * f_max], btype="bandpass", fs=fs, output="sos")
    live = ~ds.dead
    idx_all = np.arange(layout.n_receivers)
    groups = {name: idx_all[(idx_all >= sl.start) & (idx_all < sl.stop) & live] for name, sl in layout.slices.items()}
    table = {}
    say(f"\n[{tag}] band limited ({0.25 * f_max:.0f} to {0.95 * f_max:.0f} Hz) rms at the sensors, whole run, live channels")
    say(f"  {'component':28s} " + " ".join(f"{n:>12s}" for n in groups) + f" {'best DAS':>12s}")
    for key, comp in ds.components.items():
        gain = ds.gain if key != "noise" else np.ones_like(ds.gain)
        row = [band_rms(comp[g].astype(float) * gain[g, None], sos) for g in groups.values()]
        das = groups[layout.cables[0].name]
        best = max(band_rms(comp[i].astype(float) * gain[i], sos) for i in das) if len(das) else float("nan")
        table[key] = dict(zip(groups, row)) | {"best_das": best}
        say(f"  {key:28s} " + " ".join(f"{v:12.4g}" for v in row) + f" {best:12.4g}")
    summary = {}
    for v in cfg.vessels:
        if v.level_pa <= 0:
            continue
        d, s = table["direct:" + v.id], table["scattered:" + scattered_key(v.id, COMBINED)]
        summary[v.id] = {
            "echo_to_direct_db_median_das": 20 * np.log10(s["cable1"] / d["cable1"]),
            "echo_to_noise_db_median_das": 20 * np.log10(s["cable1"] / table["noise"]["cable1"]),
            "echo_to_noise_db_best_das": 20 * np.log10(s["best_das"] / table["noise"]["cable1"]),
        }
        say(f"  {v.id}: " + ", ".join(f"{k} {x:+.1f}" for k, x in summary[v.id].items()))
    (out / "levels.json").write_text(json.dumps({"table": table, "summary": summary, "f_max": f_max,
                                                 "bodies": [v.id for v in cfg.vessels if v.is_body]}, indent=2))

    # Map: layout, every track and static body labelled by type.
    fig, ax = plt.subplots(figsize=(10, 7.5))
    plot_harbor(env, layout, ax=ax, sponge_cells=cfg.fdtd.damping_width)
    seen = set()
    for v in cfg.vessels:
        tr = tracks[v.id]
        typ = object_type(v)
        st = TYPE_STYLE[typ]
        label = typ if typ not in seen else None
        seen.add(typ)
        if typ == "static body":
            xy, hd = tr.xy[0], np.array([1.0, 0.0])
            poly = body_outline(v, xy, hd)
            ax.fill(poly[:, 0], poly[:, 1], color=st["color"], alpha=0.8, label=label, zorder=5)
        else:
            xy = tr.xy[tr.active]
            ax.plot(*xy.T, ls=st["ls"], color=st["color"], lw=2, label=label, zorder=5)
            ax.plot(*xy[0], "o", ms=5, color="k", zorder=6)
            if v.is_body:
                hd = xy[-1] - xy[0]
                hd = hd / max(np.hypot(*hd), 1e-9)
                poly = body_outline(v, xy[-1], hd)
                ax.plot(poly[:, 0], poly[:, 1], color=st["color"], lw=1, zorder=6)
        if v.id.startswith("piling") and v.id != "piling1":
            continue
        ax.annotate("pilings 1-6" if v.id == "piling1" else v.id, tr.xy[0], textcoords="offset points", xytext=(4, 4), fontsize=8, zorder=7)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Multistatic scenario ({tag}): {sum(v.level_pa > 0 for v in cfg.vessels)} illuminators, "
                 f"{sum(v.is_body and v.level_pa == 0 and len(v.waypoints) > 1 for v in cfg.vessels)} silent moving bodies, "
                 f"{sum(v.is_body and len(v.waypoints) == 1 for v in cfg.vessels)} static bodies")
    fig.savefig(out / "map.png", dpi=110, bbox_inches="tight")

    # Components on the cable per illuminator.
    cable = layout.cables[0]
    sl = layout.slices[cable.name]
    illum = [v for v in cfg.vessels if v.level_pa > 0]
    fig, axes = plt.subplots(2, len(illum) + 1, figsize=(5 * (len(illum) + 1), 8), sharey=True, squeeze=False)
    v_noise = np.percentile(np.abs(sosfiltfilt(sos, ds.components["noise"][sl].astype(float), axis=-1)), 99.5)
    for j, v in enumerate(illum):
        for i, key in enumerate(("direct:" + v.id, "scattered:" + scattered_key(v.id, COMBINED))):
            x = sosfiltfilt(sos, ds.components[key][sl].astype(float), axis=-1)
            vv = np.percentile(np.abs(x), 99.5)
            axes[i, j].imshow(x[:, ::10], aspect="auto", origin="lower", cmap="seismic", vmin=-vv, vmax=vv, extent=(0, ds.t[-1], 0, cable.length))
            axes[i, j].set_title(f"{key} (rms {np.sqrt(np.mean(x ** 2)):.3g})", fontsize=9)
            axes[i, j].set_xlabel("time [s]")
    x = sosfiltfilt(sos, ds.components["noise"][sl].astype(float), axis=-1)
    axes[0, -1].imshow(x[:, ::10], aspect="auto", origin="lower", cmap="seismic", vmin=-v_noise, vmax=v_noise, extent=(0, ds.t[-1], 0, cable.length))
    axes[0, -1].set_title(f"noise (rms {np.sqrt(np.mean(x ** 2)):.3g})", fontsize=9)
    axes[1, -1].axis("off")
    axes[0, 0].set_ylabel("distance along cable [m]"); axes[1, 0].set_ylabel("distance along cable [m]")
    fig.suptitle(f"Cable components per illuminator, {0.25 * f_max:.0f} to {0.95 * f_max:.0f} Hz (each panel at its own scale)", fontsize=11)
    fig.savefig(out / "components.png", dpi=100, bbox_inches="tight")
    say(f"wrote {out}")
    if args.no_check:
        return

    # Ferry only oracle check: do the bodies' echoes appear?
    comps = ds.components
    gain = ds.gain[:, None]
    traces = gain * comps["scattered:" + scattered_key("ferry", COMBINED)].astype(float) + comps["noise"].astype(float)
    ferry = next(v for v in cfg.vessels if v.id == "ferry")
    oracle = ferry.level_pa * vessel_injection(ferry, tracks["ferry"], env, cfg)
    bc = BistaticConfig(win_s=0.5, hop_s=0.25, f_lo=100.0, bin_stride=2, grid_dx=2.0, reference="oracle", removal="none",
                        method="coherent", coherent_norm="snr", exclusion_radius=25.0, dispersion=for_scenario(env, cfg.fdtd))
    sp = stft(traces, layout, fs, bc, f_max)
    ship_xy = tracks["ferry"].position_at(sp.times)
    t0 = time.time()
    run = run_bistatic(traces, layout, fs, env, ds.ais, bc, f_max, oracle_signal=oracle, truth=Truth(ship_xy=ship_xy))
    say(f"[{tag}] ferry only oracle chain: {time.time() - t0:.0f} s, {len(run.grid)} cells, {len(sp.times)} frames")
    B, grid, times = run.scan.B, run.grid, run.sp.times
    xs, ys = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    ix = np.rint((grid[:, 0] - xs[0]) / bc.grid_dx).astype(int)
    iy = np.rint((grid[:, 1] - ys[0]) / bc.grid_dx).astype(int)
    bodies = [v for v in cfg.vessels if v.is_body]
    integrated = B.sum(axis=1)
    integrated /= integrated.max()
    picks = np.linspace(2, len(times) - 3, 5).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    panels = [(k, B[:, k], f"t = {times[k]:.2f} s") for k in picks] + [(None, integrated, f"sum over all {len(times)} frames")]
    check = {}
    for ax, (k, b, title) in zip(axes.ravel(), panels):
        img = np.full((len(ys), len(xs)), np.nan)
        img[iy, ix] = 10 * np.log10(np.maximum(b / b.max(), 1e-12))
        plot_harbor(env, layout, ax=ax, show_depth=False)
        im = ax.imshow(img, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="magma", vmin=-20, vmax=0, alpha=0.9, zorder=2)
        kk = k if k is not None else len(times) // 2
        ax.plot(*ship_xy[kk], "s", color="deepskyblue", ms=9, mec="k", zorder=5, label="ferry (exact)")
        for v in bodies:
            p = tracks[v.id].position_at(times[kk])[0]
            ax.plot(*p, "o", mfc="none", mec="lime" if v.level_pa == 0 and len(v.waypoints) > 1 else "magenta", mew=1.5, ms=10, zorder=5)
            if k is None:
                # Local map value at every body against the map median: is the echo visible?
                d = np.hypot(*(grid - p).T)
                check[v.id] = float(10 * np.log10(max(b[d <= 4.0].max() if np.any(d <= 4.0) else 0.0, 1e-12) / np.median(b)))
        ax.set_title(title, fontsize=10)
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
        if ax.get_legend():
            ax.get_legend().remove()
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.6, label="coherent map [dB re peak]")
    fig.suptitle("Ferry only, oracle setup (truth direct removal, oracle waveform, exact track): circles are the bodies "
                 "(green moving, magenta static)", fontsize=11)
    fig.savefig(out / "ferry_only_map.png", dpi=100, bbox_inches="tight")
    say("[ferry only] map value within 4 m of each body over the map median, summed map: " + ", ".join(f"{k} {v:+.1f} dB" for k, v in check.items()))
    lv = json.loads((out / "levels.json").read_text())
    lv["ferry_only_oracle_check_db"] = check
    lv["ferry_only_oracle_inputs"] = ["direct path removed with the truth component", "oracle ferry waveform", "exact ferry track"]
    (out / "levels.json").write_text(json.dumps(lv, indent=2))


if __name__ == "__main__":
    main()

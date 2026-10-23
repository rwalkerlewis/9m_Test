#!/usr/bin/env python3
"""What a silent body's echo can give (prompt ``run_all_3``, task 3, section 1): the echo to noise ratio of every
silent body on the multistatic recordings, from the analytic scattering of a cylinder, before any echo chain chases it.

Nothing here feeds the processing; the levels come from the truth (a diagnostic): the illuminator's direct arrival and
the noise as recorded on every channel.  Per recording (``multistatic``, ``multistatic_b``, noise seed 0), per
illuminator (every radiating ship) and per frame of the echo chain (``win_s``, ``hop_s`` of its config) and band
(``band_hz``, the echo chain's FFT bins):

* the illuminator's level per bin and frame: its direct component of the truth (``direct:<ship>``) on every channel,
  its band power fitted, over the channels, by the analytic direct field (the 2D Green's function at the channel's
  sample points from the true ship position, through ``SensorLayout.collapse``: the strain channels' second difference);
* the echo of each body per channel: the incident field at the body (the same Green's function and level) times the
  far field of an infinite pressure release cylinder (partial waves ``c_n = -J_n(ka) / H_n(ka)``, ``|n| <= ka + 10``),
  at the channel's sample points, through the same ``collapse``.  Radius: 0.5 m for a piling (1 m across), the radius
  of the circle of the body's area (``sqrt(L W / pi)``) for the diver, the UUVs and the moored hull (a check: the UUVs
  and the hull are tracked, so their numbers calibrate the budget);
* the interference per channel: the noise component of the truth (``noise``, measured: a diagnostic) plus the
  illuminator's direct arrival 20 dB down (the removal depth measured on the multistatic recordings);
* per frame the echo to interference ratio over the band on the best channel, and on the best 10 channels (ranked by
  their own ratio) summed coherently: ``(sum sqrt(E_i))^2 / sum N_i`` (the echo adds in amplitude, the interference in
  power).  No time bandwidth gain is counted: these are the ratios of one frame's band power.

Criterion for the conclusion, fixed before the budget was computed (prompt): if no piling reaches 0 dB on the coherent
sum of its best 10 channels in any frame, the piling criterion of stage 2 is retired as unreachable in this scenario;
if some piling does, the frames and illuminators where it does are reported, and the budget becomes the target the next
echo chain task is measured against.

Check of the model against the forward model: the analytic echoes of every body, summed in power, against the
scattered component of the truth (``scattered:<ship>>bodies``, every body at once) on every channel, the median of the
ratio in dB over channels and frames (``calibration_db``).

Outputs in ``output/studies/echo_budget/``: ``results.json`` (per recording, illuminator and body the best channel and
best 10 ratios: maximum and median over the frames, the frames at or above 0 dB, the conclusion), ``budget.md`` (the
table), one figure per recording (``budget_<recording>.png``: the coherent 10 channel ratio of every body against time,
the best illuminator per frame), ``run.log``.

    python examples/harbor/05_detection/harbor_echo_budget.py [--recordings multistatic multistatic_b]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import hankel2, jv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

REMOVAL_DB = 20.0
N_BEST = 10
CRITERION = ("if no piling reaches 0 dB on the coherent sum of its best 10 channels in any frame, the piling criterion of "
             "stage 2 is retired as unreachable in this scenario; if some piling does, report in which frames and which "
             "illuminator, and the budget becomes the target of the next echo chain task")


# ---- the cylinder ----------------------------------------------------------------------------

def soft_coefficients(ka: float) -> tuple[np.ndarray, np.ndarray]:
    """Partial wave orders and coefficients ``c_n = -J_n(ka) / H_n(ka)`` of a pressure release cylinder (the
    convention of :func:`form_function`; ``H_n = H_n^(2)`` with the time factor ``exp(i w t)``, ``|n| <= ka + 10``)."""
    n_max = max(10, int(np.ceil(ka + 10)))
    n = np.arange(-n_max, n_max + 1)
    return n, -jv(n, ka) / hankel2(n, ka)


def form_function(ka: float, phi: np.ndarray) -> np.ndarray:
    """Far field form function ``f(phi) = sum_n c_n exp(i n phi)`` of a pressure release cylinder for a unit plane
    wave (``phi`` from the propagation direction, 0 forward, pi back); the scattered field at range ``r`` is
    ``f(phi) sqrt(2 / (pi k r)) exp(-i (k r - pi / 4))``."""
    n, c = soft_coefficients(ka)
    return np.exp(1j * np.outer(np.atleast_1d(phi), n)) @ c


def echo_width(k: float, a: float, phi) -> np.ndarray:
    """The 2D echo width ``sigma(phi) = (4 / k) |f(phi)|^2`` [m] (``lim 2 pi r |p_s|^2 / |p_i|^2``)."""
    return 4.0 / k * np.abs(form_function(k * a, phi)) ** 2


def green(k: float, r: np.ndarray) -> np.ndarray:
    """``H0^(2)(k r)``, the 2D point source of the forward model up to a constant."""
    return hankel2(0, k * np.maximum(r, 1e-3))


# ---- the budget ------------------------------------------------------------------------------

def stft_power(x: np.ndarray, fs: float, win_s: float, hop_s: float, band: tuple[float, float]):
    """Power per channel, band bin and frame ``(N, F, K)`` of *x* (N, n), Hann window; the bins and frame times."""
    nper, hop = int(round(win_s * fs)), int(round(hop_s * fs))
    w = np.hanning(nper)
    f = np.fft.rfftfreq(nper, 1.0 / fs)
    sel = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    starts = np.arange(0, x.shape[1] - nper + 1, hop)
    out = np.empty((x.shape[0], len(sel), len(starts)))
    for j, s in enumerate(starts):
        out[:, :, j] = np.abs(np.fft.rfft(x[:, s:s + nper] * w, axis=1)[:, sel]) ** 2
    return out, f[sel], (starts + nper / 2.0) / fs


def fields(layout, ship: np.ndarray, body: np.ndarray, k: float, a: float | None):
    """Channel values (N,) of the direct field of a unit source at *ship* and, with radius *a*, of the echo of a
    pressure release cylinder of radius *a* at *body*."""
    pts = layout.sample_points
    rd = np.hypot(*(pts - ship[None, :]).T)
    direct = layout.collapse(green(k, rd)[None, :])[0]
    if a is None:
        return direct, None
    r_sb = float(np.hypot(*(body - ship)))
    inc = green(k, np.array([r_sb]))[0]
    d_in = (body - ship) / max(r_sb, 1e-9)
    v = pts - body[None, :]
    r_bp = np.maximum(np.hypot(*v.T), 1e-3)
    phi = np.arccos(np.clip((v @ d_in) / r_bp, -1.0, 1.0))
    sc = inc * form_function(k * a, phi) * np.sqrt(2.0 / (np.pi * k * r_bp)) * np.exp(-1j * (k * r_bp - np.pi / 4.0))
    return direct, layout.collapse(sc[None, :])[0]


def coherent_best(echo: np.ndarray, interf: np.ndarray, n_best: int) -> tuple[float, float]:
    """(best channel ratio, coherent ratio of the *n_best* channels with the highest ratios) [dB] from band powers per
    channel *echo* and *interf* (N,)."""
    r = echo / np.maximum(interf, 1e-300)
    order = np.argsort(r)[::-1][:n_best]
    coh = np.sum(np.sqrt(echo[order])) ** 2 / max(np.sum(interf[order]), 1e-300)
    return float(10 * np.log10(max(r[order[0]], 1e-300))), float(10 * np.log10(max(coh, 1e-300)))


def radius(v: dict) -> float:
    if v["id"].startswith("piling"):
        return 0.5
    return float(np.sqrt(float(v["body_length"]) * float(v.get("body_width") or v["body_length"]) / np.pi))


def budget(name: str, obs, truth, cfg, say) -> dict:
    layout, fs = obs.geometry, obs.fs
    bc = cfg.raw["echo"]["bistatic"]
    band = cfg.band
    comps = truth.components
    noise_p, freqs, times = stft_power(comps["noise"], fs, bc["win_s"], bc["hop_s"], band)
    ships = [k for k, tr in truth.tracks.items() if tr.level_pa > 0]
    bodies = [v for v in truth.scenario["vessels"] if v.get("body_length") and v["id"] in truth.tracks]
    stride = int(bc.get("bin_stride", 1))                     # the echo chain's bins
    noise_p, freqs = noise_p[:, ::stride], freqs[::stride]
    ks = 2 * np.pi * freqs / float(obs.environment.c_nominal)
    out = {"band_hz": list(band), "frames": len(times), "illuminators": {}, "calibration_db": {}}
    for sh in ships:
        dir_p = stft_power(comps[f"direct:{sh}"], fs, bc["win_s"], bc["hop_s"], band)[0][:, ::stride]
        sc_key = f"scattered:{sh}>bodies"
        sc_p = stft_power(comps[sc_key], fs, bc["win_s"], bc["hop_s"], band)[0][:, ::stride] if sc_key in comps else None
        active = truth.tracks[sh].active_at(times)
        s_xy = truth.tracks[sh].position_at(times)
        E = {v["id"]: np.zeros((layout.n_receivers, len(times))) for v in bodies}
        for j in range(len(times)):
            if not active[j]:
                continue
            b_xy = {v["id"]: truth.tracks[v["id"]].position_at(times[j:j + 1])[0] for v in bodies}
            for i, k in enumerate(ks):
                d0 = None
                for v in bodies:
                    d, e = fields(layout, s_xy[j], b_xy[v["id"]], k, radius(v))
                    if d0 is None:
                        d0 = np.abs(d) ** 2
                        level = np.sum(dir_p[:, i, j]) / max(np.sum(d0), 1e-300)     # the measured level of the bin
                    E[v["id"]][:, j] += level * np.abs(e) ** 2
        interf = noise_p.sum(axis=1) + dir_p.sum(axis=1) * 10 ** (-REMOVAL_DB / 10.0)    # (N, K)
        rows = {}
        for v in bodies:
            bid = v["id"]
            best, coh = np.full(len(times), -np.inf), np.full(len(times), -np.inf)
            for j in np.flatnonzero(active):
                best[j], coh[j] = coherent_best(E[bid][:, j], interf[:, j], N_BEST)
            ok = np.isfinite(coh)
            rows[bid] = {"kind": v.get("kind"), "radius_m": radius(v),
                         "best_channel_db": {"max": float(best[ok].max()), "median": float(np.median(best[ok]))},
                         "best10_coherent_db": {"max": float(coh[ok].max()), "median": float(np.median(coh[ok]))},
                         "frames_at_0db": [float(t) for t in times[ok & (coh >= 0.0)]],
                         "series_db": np.where(ok, coh, np.nan).round(2).tolist()}
        if sc_p is not None:
            tot = sum(E.values())
            act = np.flatnonzero(active)
            m = sc_p.sum(axis=1)[:, act]
            ratio = 10 * np.log10(np.maximum(tot[:, act], 1e-300) / np.maximum(m, 1e-300))
            out["calibration_db"][sh] = {"median": float(np.median(ratio)), "p10": float(np.percentile(ratio, 10)),
                                         "p90": float(np.percentile(ratio, 90))}
        out["illuminators"][sh] = rows
        say(f"{name}, {sh}: " + ", ".join(f"{b} {r['best10_coherent_db']['max']:.1f}" for b, r in rows.items()) +
            (f"; analytic over FDTD echoes {out['calibration_db'][sh]['median']:+.1f} dB" if sh in out["calibration_db"] else ""))
    out["times"] = times.round(3).tolist()
    return out


def conclusion(results: dict) -> dict:
    """The criterion: every (recording, illuminator, piling, frame) at or above 0 dB on the coherent 10 channel sum."""
    hits = [{"recording": rec, "illuminator": sh, "piling": b, "frames": r["frames_at_0db"]}
            for rec, res in results.items() for sh, rows in res["illuminators"].items() for b, r in rows.items()
            if b.startswith("piling") and r["frames_at_0db"]]
    return {"criterion": CRITERION, "piling_reaches_0db": bool(hits), "where": hits,
            "verdict": "the budget is the target of the next echo chain task" if hits else
                       "the piling criterion of stage 2 is retired as unreachable in this scenario"}


def table(results: dict) -> str:
    """Per recording and illuminator: the coherent 10 channel ratio, maximum over the frames, of the pilings (their
    range), the diver, the UUVs and the hull; the frames in which some piling reaches 0 dB; the analytic echoes over the
    forward model's (median, dB)."""
    lines = ["| Recording | Illuminator | Pilings [dB] | Diver [dB] | UUV 1 / UUV 2 [dB] | Moored hull [dB] | Frames with a piling at 0 dB | Analytic over FDTD echoes [dB] |",
             "|---|---|---|---|---|---|---|---|"]
    for rec, res in results.items():
        for sh, rows in res["illuminators"].items():
            pil = [r["best10_coherent_db"]["max"] for b, r in rows.items() if b.startswith("piling")]
            frames = set()
            for b, r in rows.items():
                if b.startswith("piling"):
                    frames |= set(r["frames_at_0db"])
            g = lambda b: f"{rows[b]['best10_coherent_db']['max']:+.1f}" if b in rows else "-"  # noqa: E731
            cal = res["calibration_db"].get(sh, {}).get("median")
            lines.append(f"| {rec} | {sh} | {min(pil):+.1f} to {max(pil):+.1f} | {g('diver')} | {g('uuv1')} / {g('uuv2')} | "
                         f"{g('moored')} | {len(frames)} of {res['frames']} | {'-' if cal is None else f'{cal:+.1f}'} |")
    return "\n".join(lines) + "\n"


def figure(name: str, res: dict, path: Path) -> None:
    from acoustic_sim.plotting import new_figure, save_figure

    fig, ax = new_figure((12.0, 8.0))
    t = np.array(res["times"])
    bodies = list(next(iter(res["illuminators"].values())))
    for b in bodies:
        best = np.nanmax(np.array([res["illuminators"][sh][b]["series_db"] for sh in res["illuminators"]], float), axis=0)
        ax.plot(t, best, "-" if b.startswith("piling") else "--", lw=1.6, label=b)
    ax.axhline(0.0, color="k", lw=1.0, ls=":", label="0 dB (the criterion)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("echo to noise and residual, best 10 channels coherent [dB]")
    ax.set_title(f"{name}: echo budget of every silent body (best illuminator per frame)")
    ax.legend(loc="best", fontsize=11, ncol=2)
    save_figure(fig, path, recording=name)


def main() -> None:
    from acoustic_sim.contract import Observation, Truth, contract_paths
    from acoustic_sim.water.stream import load_processing

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recordings", nargs="*", default=["multistatic", "multistatic_b"])
    ap.add_argument("--out", default=str(ROOT / "output" / "studies" / "echo_budget"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "a")
    say = lambda m: (print(m), log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n"), log.flush())  # noqa: E731
    results = {}
    for name in a.recordings:
        obs_p, truth_p = contract_paths(ROOT / "output" / "recordings" / name / "dataset.npz")
        obs, truth = Observation.load(obs_p), Truth.load(truth_p)
        cfg = load_processing(ROOT / "examples" / "configs" / "processing" / "multistatic.json")
        results[name] = budget(name, obs, truth, cfg, say)
        figure(name, results[name], out / f"budget_{name}.png")
    concl = conclusion(results)
    res = {"criterion": CRITERION, "removal_depth_db": REMOVAL_DB, "n_best": N_BEST,
           "provenance": {"truth_inputs": ["direct components (the illuminators' levels)", "noise component (the floor)",
                                           "true positions of the ships and bodies"],
                          "simulator_knowledge": ["2D Green's function", "pressure release cylinder"],
                          "data": "forward_model", "held_out": False},
           "recordings": results, "conclusion": concl}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "budget.md").write_text(table(results))
    say(concl["verdict"])


if __name__ == "__main__":
    main()

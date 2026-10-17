#!/usr/bin/env python3
"""Harbor bistatic link budget: a silent UUV lit by an AIS ship.

Analytic 2-D (plan view) budget for detecting a non-radiating UUV from the energy of a
known, AIS-reporting ship that it scatters toward a DAS cable and a few hydrophones.
Nothing is simulated here; every term is a closed-form 2-D expression, so the script
answers "could the processing chain ever see this echo?" before any FDTD run.

Model (all cylindrical spreading, consistent with the FDTD point source of this package)

* Ship free-field pressure at range r: ``L_ship * |H0(k r)| / |H0(k R_REF)|`` per frequency,
  with ``R_REF = 100 m`` (source levels in ``signatures.py`` are rms at 100 m).  The far-field
  form ``sqrt(2 / (pi k r))`` is used only when ``k r > 8``, the exact Hankel function otherwise.
* UUV: equivalent infinite cylinder of radius ``a`` (0.5 m and 1.0 m for the 1 m by 3 m body),
  soft (pressure release, Dirichlet) or rigid (Neumann).  Partial-wave coefficients
  ``-J_n(ka) / H_n(ka)`` (soft) and ``-J_n'(ka) / H_n'(ka)`` (rigid), summed over
  ``|n| <= N = ka + 10`` (at least 10).  Far-field scattered pressure for a unit plane wave:
  ``sqrt(2 / (pi k r)) exp(i (k r - pi/4)) f(phi)`` with ``f(phi) = sum_n c_n exp(i n phi)``,
  ``phi`` measured from the incident propagation direction (0 = forward, pi = backscatter).
* Echo at a receiver: incident pressure at the UUV times ``sqrt(2 / (pi k r_uuv_rx)) |f(phi)|``.
* Band: 100 Hz to ``f_max_for_grid(dx)`` (800 Hz at dx = 0.5 m, 400 Hz at dx = 1.0 m) in
  10 Hz bins.  The ship band level is 1 Pa rms at 100 m for both spectra: flat per Hz, or
  falling 6 dB per octave (power ~ f^-2) above 100 Hz.

Noise scaling rule (the ``NoiseConfig`` numbers are rms over the full band at fs = 600 Hz):

* white instrument noise (DAS channel noise 0.05 Pa, DAS common mode 0.03 Pa, hydrophone self
  noise 0.005 Pa) is scaled by ``sqrt(B / 300)`` with B the analysis bandwidth and 300 Hz the
  Nyquist band at fs = 600;
* acoustic ambient noise (0.02 Pa rms over 5 to 120 Hz) keeps the spectral density of that band,
  ``0.02**2 / 115`` Pa^2/Hz, so its in-band power is that density times B;
* the distant-shipping term (8 to 70 Hz) lies below the band and is left out.

Per receiver and frequency bin the script forms the direct ship level, the echo level and the
noise level, sums over the band, and reports (median over all receivers, and at the best
receiver = the one with the highest echo-to-direct ratio) the direct level, echo level, noise
level, echo-to-direct and echo-to-noise ratios, the processing gain ``10 log10(B T)`` for
T = 1, 2, 4 s, and the required direct-path cancellation
``(direct - echo) - processing gain + 10 dB margin`` clipped at 0.

The ship and the UUV are each sampled at 5 positions along their paths.  All 25 pairs are
saved in ``budget.json``; the printed table, the summary lines and the figure use the
synchronous pairing (both vessels at the same fraction of their path, time axis = UUV time,
the ship covering its 120 m run in the same interval), flagged ``"synchronous": true``.

Usage::

    python examples/harbor_bistatic_budget.py [--out output/harbor_bistatic] [--dx 0.5 1.0]
        [--ship-level 1.0] [--radii 0.5 1.0] [--margin-db 10] [--t-ref 2.0]
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
from scipy.special import h1vp, hankel1, jv, jvp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acoustic_sim.harbor.noise import NoiseConfig  # noqa: E402
from acoustic_sim.harbor.signatures import R_REF, f_max_for_grid  # noqa: E402

C = 1500.0
FS0 = 600.0                 # sampling rate the NoiseConfig levels are defined at
NYQ0 = FS0 / 2.0
AMBIENT_BAND = (5.0, 120.0)  # band of the ambient_rms figure in noise.py
MODELS = ("soft", "rigid")
SPECTRA = ("flat", "rolloff6")
T_PROC = (1.0, 2.0, 4.0)
# dataviz reference palette, categorical slots 1 to 4 in fixed order
COLORS = {("soft", 0.5): "#2a78d6", ("soft", 1.0): "#eb6834", ("rigid", 0.5): "#1baf7a", ("rigid", 1.0): "#eda100"}


# ---- geometry -------------------------------------------------------------------------------

def build_receivers(args) -> tuple[np.ndarray, list[str], np.ndarray]:
    xs = np.arange(args.das_x0, args.das_x1 + 1e-9, args.das_spacing)
    das = np.column_stack([xs, np.full_like(xs, args.das_y)])
    hyd = np.array(args.hydrophones, dtype=float).reshape(-1, 2)
    names = [f"das{i:03d}(x={x:.0f})" for i, x in enumerate(xs)] + [f"hyd{i}({x:.0f},{y:.0f})" for i, (x, y) in enumerate(hyd)]
    is_das = np.r_[np.ones(len(xs), bool), np.zeros(len(hyd), bool)]
    return np.vstack([das, hyd]), names, is_das


def path_samples(p0, p1, speed, n):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    frac = np.linspace(0.0, 1.0, n)
    xy = p0[None, :] + frac[:, None] * (p1 - p0)[None, :]
    t = frac * np.linalg.norm(p1 - p0) / speed
    return xy, t, frac


# ---- acoustics ------------------------------------------------------------------------------

def green_mag(k: np.ndarray, r: np.ndarray) -> np.ndarray:
    """|H0(k r)|: far-field sqrt form when k r > 8, exact Hankel otherwise.  Broadcasts."""
    kr = k * r
    out = np.empty(np.broadcast(k, r).shape)
    far = kr > 8.0
    out[far] = np.sqrt(2.0 / (np.pi * kr[far]))
    out[~far] = np.abs(hankel1(0, kr[~far]))
    return out


def cylinder_coefficients(ka: float, model: str) -> np.ndarray:
    n_max = max(10, int(np.ceil(ka + 10)))
    n = np.arange(-n_max, n_max + 1)
    if model == "soft":
        c = -jv(n, ka) / hankel1(n, ka)
    elif model == "rigid":
        c = -jvp(n, ka) / h1vp(n, ka)
    else:
        raise ValueError(model)
    return n, c


def form_function(k: np.ndarray, a: float, model: str, phi: np.ndarray) -> np.ndarray:
    """|f(phi)|^2 for every (frequency, angle): shape (n_k, n_phi)."""
    out = np.empty((len(k), len(phi)))
    for j, kj in enumerate(k):
        n, c = cylinder_coefficients(kj * a, model)
        out[j] = np.abs(np.exp(1j * np.outer(phi, n)) @ c) ** 2
    return out


def ship_weights(f: np.ndarray, spectrum: str) -> np.ndarray:
    """Fraction of the band power (1 Pa^2 total at R_REF) in each 10 Hz bin."""
    if spectrum == "flat":
        w = np.ones_like(f)
    elif spectrum == "rolloff6":
        w = (f[0] / f) ** 2          # amplitude -6 dB per octave -> power ~ f^-2
    else:
        raise ValueError(spectrum)
    return w / w.sum()


def noise_power(is_das: np.ndarray, bandwidth: float, cfg: NoiseConfig) -> np.ndarray:
    ambient_density = cfg.ambient_rms ** 2 / (AMBIENT_BAND[1] - AMBIENT_BAND[0])
    white_scale = bandwidth / NYQ0
    das = ambient_density * bandwidth + (cfg.das_noise_rms ** 2 + cfg.das_common_mode_rms ** 2) * white_scale
    hyd = ambient_density * bandwidth + cfg.hydrophone_self_rms ** 2 * white_scale
    return np.where(is_das, das, hyd)


def db(p: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(p, 1e-30))


# ---- the budget -----------------------------------------------------------------------------

def evaluate(ship_xy, uuv_xy, rx, names, f, w, level, a, model, noise_p, pg, margin):
    k = 2.0 * np.pi * f / C
    r_ship_rx = np.linalg.norm(rx - ship_xy, axis=1)
    r_uuv_rx = np.linalg.norm(rx - uuv_xy, axis=1)
    r_ship_uuv = np.linalg.norm(uuv_xy - ship_xy)
    g_ref = green_mag(k, np.full_like(k, R_REF))

    # power per (frequency, receiver)
    direct = (level ** 2 * w)[:, None] * (green_mag(k[:, None], r_ship_rx[None, :]) / g_ref[:, None]) ** 2
    incident = level ** 2 * w * (green_mag(k, np.full_like(k, r_ship_uuv)) / g_ref) ** 2

    d_in = (uuv_xy - ship_xy) / r_ship_uuv
    d_out = (rx - uuv_xy) / r_uuv_rx[:, None]
    phi = np.arccos(np.clip(d_out @ d_in, -1.0, 1.0))
    f2 = form_function(k, a, model, phi)                                  # (n_f, n_rx)
    f2_ref = form_function(k, a, model, np.array([0.0, np.pi]))           # forward, back
    echo = incident[:, None] * (2.0 / (np.pi * k[:, None] * r_uuv_rx[None, :])) * f2

    direct_b, echo_b = direct.sum(0), echo.sum(0)
    edr = db(echo_b) - db(direct_b)
    enr = db(echo_b) - db(noise_p)
    req = {f"T{T:g}s": np.maximum(db(direct_b) - db(echo_b) - pg[T] + margin, 0.0) for T in T_PROC}
    best = int(np.argmax(edr))
    strength_bistatic = db((w[:, None] * f2).sum(0))      # spectrum-weighted |f|^2 per receiver

    def pack(sel):
        return {
            "direct_db": float(sel(db(direct_b))), "echo_db": float(sel(db(echo_b))), "noise_db": float(sel(db(noise_p))),
            "echo_to_direct_db": float(sel(edr)), "echo_to_noise_db": float(sel(enr)),
            "scattering_strength_bistatic_db": float(sel(strength_bistatic)),
            "required_cancellation_db": {key: float(sel(v)) for key, v in req.items()},
        }

    return {
        "range_ship_uuv_m": float(r_ship_uuv),
        "scattering_strength_forward_db": float(db((w * f2_ref[:, 0]).sum())),
        "scattering_strength_back_db": float(db((w * f2_ref[:, 1]).sum())),
        "median": pack(np.median),
        "best": pack(lambda v: v[best]) | {"receiver": names[best], "range_uuv_rx_m": float(r_uuv_rx[best]),
                                            "bistatic_angle_deg": float(np.degrees(phi[best]))},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/harbor_bistatic")
    ap.add_argument("--dx", type=float, nargs="+", default=[0.5, 1.0], help="grid spacings [m]; f_max = f_max_for_grid(dx), fs = 600*5/dx")
    ap.add_argument("--f-min", type=float, default=100.0)
    ap.add_argument("--df", type=float, default=10.0, help="integration step [Hz]")
    ap.add_argument("--domain", type=float, nargs=2, default=[400.0, 300.0])
    ap.add_argument("--das-y", type=float, default=60.0)
    ap.add_argument("--das-x0", type=float, default=20.0)
    ap.add_argument("--das-x1", type=float, default=330.0)
    ap.add_argument("--das-spacing", type=float, default=2.0)
    ap.add_argument("--hydrophones", type=float, nargs="+", default=[30, 30, 370, 30, 30, 250, 200, 20], help="x y pairs")
    ap.add_argument("--ship-level", type=float, default=1.0, help="ship rms pressure at 100 m [Pa], 2-D free field")
    ap.add_argument("--ship-path", type=float, nargs=4, default=[140, 200, 260, 200], help="x0 y0 x1 y1")
    ap.add_argument("--ship-speed", type=float, default=4.0)
    ap.add_argument("--uuv-path", type=float, nargs=4, default=[240, 100, 170, 150], help="x0 y0 x1 y1")
    ap.add_argument("--uuv-speed", type=float, default=2.0)
    ap.add_argument("--n-pos", type=int, default=5, help="positions per path")
    ap.add_argument("--radii", type=float, nargs="+", default=[0.5, 1.0], help="equivalent cylinder radii [m]")
    ap.add_argument("--margin-db", type=float, default=10.0)
    ap.add_argument("--t-ref", type=float, default=2.0, choices=T_PROC, help="integration time for the summary and figure")
    ap.add_argument("--stop-db", type=float, default=40.0, help="stop criterion on the required cancellation")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    noise_cfg = NoiseConfig()

    rx, names, is_das = build_receivers(args)
    ship_xy, ship_t, frac = path_samples(args.ship_path[:2], args.ship_path[2:], args.ship_speed, args.n_pos)
    uuv_xy, uuv_t, _ = path_samples(args.uuv_path[:2], args.uuv_path[2:], args.uuv_speed, args.n_pos)
    print(f"receivers: {int(is_das.sum())} DAS channels + {int((~is_das).sum())} hydrophones; "
          f"ship {args.ship_speed} m/s over {ship_t[-1]:.0f} s, UUV {args.uuv_speed} m/s over {uuv_t[-1]:.0f} s")

    results = {
        "scenario": {
            "domain_m": args.domain, "das": {"y": args.das_y, "x0": args.das_x0, "x1": args.das_x1, "spacing": args.das_spacing,
                                             "n_channels": int(is_das.sum())},
            "hydrophones": np.array(args.hydrophones).reshape(-1, 2).tolist(),
            "ship": {"level_pa_rms_at_100m": args.ship_level, "path": args.ship_path, "speed_mps": args.ship_speed,
                     "positions": ship_xy.tolist(), "times_s": ship_t.tolist()},
            "uuv": {"path": args.uuv_path, "speed_mps": args.uuv_speed, "positions": uuv_xy.tolist(), "times_s": uuv_t.tolist(),
                    "models": list(MODELS), "radii_m": args.radii},
            "noise": {"ambient_rms_pa": noise_cfg.ambient_rms, "ambient_band_hz": AMBIENT_BAND, "das_noise_rms_pa": noise_cfg.das_noise_rms,
                      "das_common_mode_rms_pa": noise_cfg.das_common_mode_rms, "hydrophone_self_rms_pa": noise_cfg.hydrophone_self_rms,
                      "rule": "white instrument noise scaled by sqrt(B/300); ambient keeps its 5-120 Hz spectral density"},
            "margin_db": args.margin_db, "processing_times_s": list(T_PROC), "sound_speed_mps": C,
        },
        "grids": {},
        "cases": [],
        "summary": [],
    }

    hdr = (f"{'dx':>4} {'spec':>8} {'model':>5} {'a':>4} {'pos':>3} {'t_uuv':>5} {'r_s-u':>5} | "
           f"{'Direct':>6} {'Echo':>6} {'Noise':>6} {'EDR':>6} {'ENR':>6} | {'Direct':>6} {'Echo':>6} {'Noise':>6} {'EDR':>6} {'ENR':>6} "
           f"{'req1':>5} {'req2':>5} {'req4':>5} {'fwd':>5} {'back':>5} best receiver")
    print("\nlevels in dB re 1 Pa (band), ratios in dB; left block median over receivers, right block best receiver;"
          " req1/2/4 = required cancellation for T = 1/2/4 s at the best receiver; fwd/back = 2-D scattering strength 10log10|f|^2")
    print(hdr)
    print("-" * len(hdr))

    track = {}   # (dx, spectrum, model, a) -> list over synchronous positions of case dicts
    for dx in args.dx:
        f_max = f_max_for_grid(dx)
        fs = FS0 * 5.0 / dx
        f = np.arange(args.f_min, f_max + 1e-9, args.df)
        bandwidth = f_max - args.f_min
        pg = {T: 10.0 * np.log10(bandwidth * T) for T in T_PROC}
        noise_p = noise_power(is_das, bandwidth, noise_cfg)
        results["grids"][f"{dx:g}"] = {"dx_m": dx, "fs_hz": fs, "f_min_hz": args.f_min, "f_max_hz": f_max, "bandwidth_hz": bandwidth,
                                       "n_bins": int(len(f)), "processing_gain_db": {f"T{T:g}s": pg[T] for T in T_PROC},
                                       "noise_db_das": float(db(noise_p[is_das][0])), "noise_db_hydrophone": float(db(noise_p[~is_das][0]))}
        for spectrum in SPECTRA:
            w = ship_weights(f, spectrum)
            for model in MODELS:
                for a in args.radii:
                    key = (dx, spectrum, model, a)
                    track[key] = []
                    for i in range(args.n_pos):
                        for j in range(args.n_pos):
                            ev = evaluate(ship_xy[i], uuv_xy[j], rx, names, f, w, args.ship_level, a, model, noise_p, pg, args.margin_db)
                            case = {"dx_m": dx, "spectrum": spectrum, "model": model, "radius_m": a,
                                    "ship_index": i, "ship_xy": ship_xy[i].tolist(), "ship_t_s": float(ship_t[i]),
                                    "uuv_index": j, "uuv_xy": uuv_xy[j].tolist(), "uuv_t_s": float(uuv_t[j]),
                                    "synchronous": i == j, "track_fraction": float(frac[i]) if i == j else None} | ev
                            results["cases"].append(case)
                            if i == j:
                                track[key].append(case)
                                m, b = ev["median"], ev["best"]
                                rq = b["required_cancellation_db"]
                                print(f"{dx:4.1f} {spectrum:>8} {model:>5} {a:4.1f} {i:3d} {uuv_t[i]:5.1f} {ev['range_ship_uuv_m']:5.0f} | "
                                      f"{m['direct_db']:6.1f} {m['echo_db']:6.1f} {m['noise_db']:6.1f} {m['echo_to_direct_db']:6.1f} {m['echo_to_noise_db']:6.1f} | "
                                      f"{b['direct_db']:6.1f} {b['echo_db']:6.1f} {b['noise_db']:6.1f} {b['echo_to_direct_db']:6.1f} {b['echo_to_noise_db']:6.1f} "
                                      f"{rq['T1s']:5.1f} {rq['T2s']:5.1f} {rq['T4s']:5.1f} {ev['scattering_strength_forward_db']:5.1f} "
                                      f"{ev['scattering_strength_back_db']:5.1f} {b['receiver']} ({b['bistatic_angle_deg']:.0f} deg)")

    # ---- summary -------------------------------------------------------------------------
    tkey = f"T{args.t_ref:g}s"
    print(f"\nSummary (synchronous track, best receiver, T = {args.t_ref:g} s, margin {args.margin_db:g} dB):")
    for dx in args.dx:
        for model in MODELS:
            for a in args.radii:
                parts = []
                entry = {"dx_m": dx, "model": model, "radius_m": a, "t_ref_s": args.t_ref, "spectra": {}}
                for spectrum in SPECTRA:
                    cases = track[(dx, spectrum, model, a)]
                    req = np.array([c["best"]["required_cancellation_db"][tkey] for c in cases])
                    req_med = np.array([c["median"]["required_cancellation_db"][tkey] for c in cases])
                    ib, iw = int(np.argmin(req)), int(np.argmax(req))
                    entry["spectra"][spectrum] = {
                        "best_case_db": float(req[ib]), "best_case_t_s": cases[ib]["uuv_t_s"],
                        "worst_case_db": float(req[iw]), "worst_case_t_s": cases[iw]["uuv_t_s"],
                        "median_receiver_best_case_db": float(req_med.min()), "median_receiver_worst_case_db": float(req_med.max()),
                    }
                    parts.append(f"{spectrum}: best {req[ib]:.1f} dB (t={cases[ib]['uuv_t_s']:.0f} s), worst {req[iw]:.1f} dB "
                                 f"(t={cases[iw]['uuv_t_s']:.0f} s), median receiver {req_med.min():.1f} to {req_med.max():.1f} dB")
                line = f"dx={dx:g} m, {model} cylinder a={a:g} m: required cancellation " + "; ".join(parts)
                entry["line"] = line
                results["summary"].append(entry)
                print("  " + line)

    # ---- stop criterion ------------------------------------------------------------------
    req_all = np.array([c["best"]["required_cancellation_db"]["T4s"] for c in results["cases"] if c["synchronous"]])
    req_min_by_model = {m: min(c["best"]["required_cancellation_db"]["T4s"] for c in results["cases"] if c["synchronous"] and c["model"] == m)
                       for m in MODELS}
    exceeds = all(v > args.stop_db for v in req_min_by_model.values())
    sync = [c for c in results["cases"] if c["synchronous"]]
    req_hard = max(c["median"]["required_cancellation_db"]["T1s"] for c in sync)
    enr_best = np.array([c["best"]["echo_to_noise_db"] for c in sync])
    enr_med = np.array([c["median"]["echo_to_noise_db"] for c in sync])
    stop_sentence = (
        f"Stop criterion (required cancellation above {args.stop_db:g} dB for both target models at every point): "
        f"{'MET' if exceeds else 'NOT met'}; at the best receiver with T = 4 s the lowest value along the track is "
        + ", ".join(f"{v:.1f} dB ({m})" for m, v in req_min_by_model.items())
        + f" and the highest is {req_all.max():.1f} dB; the least favourable figure (median receiver, T = 1 s) is {req_hard:.1f} dB."
    )
    soft_all = all(c["best"]["echo_to_noise_db"] > 0 for c in sync if c["model"] == "soft")
    rigid_all = all(c["best"]["echo_to_noise_db"] > 0 for c in sync if c["model"] == "rigid")
    always = ", ".join(m for m, ok in (("soft", soft_all), ("rigid", rigid_all)) if ok) or "neither model"
    noise_sentence = (
        f"Before any cancellation the echo stands above the noise at the best receiver in {int((enr_best > 0).sum())} of {len(sync)} "
        f"synchronous cases (echo-to-noise {enr_best.min():.1f} to {enr_best.max():.1f} dB) and at the median receiver in "
        f"{int((enr_med > 0).sum())} of {len(sync)} ({enr_med.min():.1f} to {enr_med.max():.1f} dB); it is above the noise at the best "
        f"receiver in every case for {always}."
    )
    results["stop_criterion"] = {"threshold_db": args.stop_db, "met": bool(exceeds), "min_required_db_by_model_T4s": req_min_by_model,
                                 "max_required_db_T4s": float(req_all.max()), "max_required_db_median_T1s": float(req_hard),
                                 "echo_to_noise_best_receiver_db_range": [float(enr_best.min()), float(enr_best.max())],
                                 "echo_to_noise_median_receiver_db_range": [float(enr_med.min()), float(enr_med.max())],
                                 "echo_above_noise_best_receiver_all_cases": {"soft": soft_all, "rigid": rigid_all},
                                 "sentences": [stop_sentence, noise_sentence]}
    print("\n" + stop_sentence)
    print(noise_sentence)

    (out / "budget.json").write_text(json.dumps(results, indent=1, default=float))

    # ---- figure --------------------------------------------------------------------------
    fig, axes = plt.subplots(len(args.dx), len(SPECTRA), figsize=(5.5 * len(SPECTRA), 3.8 * len(args.dx)), squeeze=False, sharex=True, sharey=True)
    for r, dx in enumerate(args.dx):
        for c_, spectrum in enumerate(SPECTRA):
            ax = axes[r, c_]
            for model in MODELS:
                for a in args.radii:
                    cases = track[(dx, spectrum, model, a)]
                    t = [c["uuv_t_s"] for c in cases]
                    col = COLORS.get((model, a), "#52514e")
                    ax.plot(t, [c["best"]["required_cancellation_db"][tkey] for c in cases], "-o", color=col, lw=2, ms=5,
                            label=f"{model} a={a:g} m, best receiver")
                    ax.plot(t, [c["median"]["required_cancellation_db"][tkey] for c in cases], "--", color=col, lw=1.2,
                            label=f"{model} a={a:g} m, median receiver")
            ax.axhline(args.stop_db, color="#52514e", lw=1, ls=":")
            ax.text(t[0], args.stop_db + 0.8, f"{args.stop_db:g} dB stop criterion", fontsize=8, color="#52514e")
            g = results["grids"][f"{dx:g}"]
            ax.set_title(f"dx = {dx:g} m ({args.f_min:.0f} to {g['f_max_hz']:.0f} Hz), ship spectrum {spectrum}", fontsize=10)
            ax.grid(alpha=0.3)
            if r == len(args.dx) - 1:
                ax.set_xlabel("time along UUV crossing [s] (ship at the same fraction of its run)")
            if c_ == 0:
                ax.set_ylabel(f"required cancellation [dB], T = {args.t_ref:g} s")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Bistatic budget: direct-path cancellation needed for the UUV echo (margin {args.margin_db:g} dB)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "budget.png", dpi=110)
    print(f"\nwrote {out / 'budget.json'} and {out / 'budget.png'} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()

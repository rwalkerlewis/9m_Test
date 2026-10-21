#!/usr/bin/env python3
"""Stage 6: phase accuracy and wall reflection of the time stepping schemes (prototypes, opt in).

Schemes: the current leapfrog (order 4 in space), the fourth order in time modified equation
update (``time_scheme = "lw4"``) and the k-space corrected pseudospectral update
(``time_scheme = "kspace"``), each on the 0.5 m and the 1 m grid at the harbor's sample rate for
that grid (6 kHz, 3 kHz).

1. Stationary source in open water, receivers on the grid axis and on the diagonal at 100 to
   600 m (all on grid nodes, arrivals time gated).  The transfer function of every receiver is divided by the analytic 2-D Green's
   function ``H0^(2)(k r)`` and referred to the 100 m axis receiver (the injection factor cancels);
   the phase of that ratio is the dispersion error, per bin of the band.  For the current scheme
   the same is done with the dispersion corrected wavenumber of ``dispersion.py`` (the replica
   correction the processing uses), which is the number to beat.
2. A straight pressure release wall (land half plane on a grid node line) with the source 60 m
   away: the reflected part (run with the wall minus run without) over the direct part is
   compared with the image solution ``-H0(k r_image) / H0(k r_direct)``, normal and oblique.

Criterion (fixed in the task): a scheme on the 1 m grid matches or beats the current 0.5 m scheme
with dispersion correction (at most 0.1 rad of phase error at 400 to 600 m across the band, axis
and diagonal) with a wall reflection error no worse than the current scheme's.

Usage::

    python scripts/bench_timestep.py --cuda [--out output/studies/performance]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import hankel2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.water import HarborFDTDConfig, HarborFDTDSolver, SensorLayout, uniform_water_environment  # noqa: E402
from acoustic_sim.water.dispersion import DispersionModel, replica_wavenumbers  # noqa: E402
from acoustic_sim.water.environment import _finish_environment  # noqa: E402
from acoustic_sim.sources import StaticSource  # noqa: E402

C = 1500.0
BAND_FINE = (100.0, 760.0)          # the fine (0.5 m) scenario's processing band
BAND_COARSE = (100.0, 380.0)        # the band a 1 m grid serves with the current scheme
# Receivers on grid nodes of both grids (integer coordinates): axis offsets, and diagonal offsets (d, -d)
# whose ranges d sqrt(2) are 100.4, 199.4, 400.2 and 599.6 m; the source is on a node too, so neither
# the injection nor the sampling interpolates.  600 m is the largest range the 1023 m domain allows
# with every path at least 275 m from the edges; the arrivals are time gated as well.
AXIS_OFF = np.array([100.0, 200.0, 400.0, 600.0])
DIAG_OFF = np.array([71.0, 141.0, 283.0, 424.0])
RANGES = AXIS_OFF
FAR = RANGES >= 400.0
SRC = np.array([200.0, 700.0])
T = 0.6
GATE_PRE, GATE_POST, GATE_TAPER = 0.03, 0.10, 0.01      # s around the direct arrival (stationary case)
WALL_GATE = 0.45                                        # s: direct and wall reflected arrivals, before the domain edges answer
SCHEMES = {
    # name: (time_scheme, dx, fs)
    "leapfrog_0.5m": ("leapfrog", 0.5, 6000.0),
    "lw4_0.5m": ("lw4", 0.5, 6000.0),
    "kspace_0.5m": ("kspace", 0.5, 6000.0),
    "leapfrog_1m": ("leapfrog", 1.0, 3000.0),
    "lw4_1m": ("lw4", 1.0, 3000.0),
    "kspace_1m": ("kspace", 1.0, 3000.0),
}


def ricker(n: int, fs: float, f0: float = 350.0, delay: float = 0.02) -> np.ndarray:
    t = np.arange(n) / fs
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def domain_width(dx: float) -> float:
    """Width giving a power of two cell count (the k-space FFT) : 2048 cells at 0.5 m, 1024 at 1 m."""
    return (2048 if dx <= 0.5 else 1024) * dx - dx


def config(scheme: str, dx: float, fs: float, cuda: bool, kernel: str) -> HarborFDTDConfig:
    return HarborFDTDConfig(fs=fs, total_time=T, fd_order=4, damping_width=int(round(30.0 / dx)), damping_max=0.15,
                            background_absorption=0.0, shore_loss_max=0.0, use_cuda=cuda, kernel=kernel, time_scheme=scheme)


def run(env, cfg, src, rec):
    sync()
    t0 = time.perf_counter()
    solver = HarborFDTDSolver(env, cfg, StaticSource(src[0], src[1], ricker(cfg.n_steps, cfg.fs)), SensorLayout(hydrophones=rec).sample_points)
    tr = solver.run(verbose=False)["traces"]
    sync()
    return tr, time.perf_counter() - t0, solver


def sync() -> None:
    try:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def band_bins(f: np.ndarray, S: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (f >= band[0]) & (f <= band[1]) & (np.abs(S) > 1e-2 * np.abs(S).max())


def gate(traces: np.ndarray, fs: float, t_arr: np.ndarray, pre: float, post: float, taper: float = GATE_TAPER) -> np.ndarray:
    """Cosine tapered window ``[t_arr - pre, t_arr + post]`` per receiver (removes later arrivals)."""
    t = np.arange(traces.shape[1]) / fs
    out = np.zeros_like(traces)
    for i, ta in enumerate(t_arr):
        a, b = ta - pre, ta + post
        w = np.clip((t - a) / taper, 0.0, 1.0) * np.clip((b - t) / taper, 0.0, 1.0)
        out[i] = traces[i] * (0.5 - 0.5 * np.cos(np.pi * w))
    return out


def phase_errors(traces: np.ndarray, sig: np.ndarray, fs: float, rec: np.ndarray, k_of: callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase [rad] and amplitude [dB] of ``H(f, r) / G(f, r)`` referred to receiver 0, per (receiver, bin); and the bin frequencies.

    *k_of(f, theta)* gives the wavenumber used in ``G = H0^(2)(k r)`` for the bins *f* and the
    propagation angle *theta* of each receiver.  The traces are gated around the direct arrival."""
    n = traces.shape[1]
    f = np.fft.rfftfreq(n, 1.0 / fs)
    r_all = np.hypot(*(rec - SRC).T)
    traces = gate(traces, fs, 0.02 + r_all / C, GATE_PRE, GATE_POST)
    P = np.fft.rfft(traces, axis=1)
    S = np.fft.rfft(sig)
    d = rec - SRC
    r = np.hypot(*d.T)
    theta = np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0]))
    k = k_of(f, theta)                                   # (n_rec, n_bins)
    G = hankel2(0, k * r[:, None])
    E = P / (S[None, :] * G)
    rel = E / E[0:1, :]
    return np.angle(rel), 20 * np.log10(np.abs(rel)), f


def k_free(f, theta):
    return np.broadcast_to(2 * np.pi * f[None, :] / C, (len(theta), len(f)))


def k_corrected(model: DispersionModel):
    def k_of(f, theta):
        good = f > 0
        out = np.full((len(theta), len(f)), np.nan)
        kk = replica_wavenumbers(model, f[good], C, theta[:, None], np)      # (bins, n_rec, 1)
        out[:, good] = kk[:, :, 0].T
        return out
    return k_of


def stationary_case(name: str, cuda: bool, kernel: str) -> dict:
    scheme, dx, fs = SCHEMES[name]
    width = domain_width(dx)
    env = uniform_water_environment(width, width, dx, C)
    axis = SRC + np.column_stack([AXIS_OFF, np.zeros_like(AXIS_OFF)])
    diag = SRC + np.column_stack([DIAG_OFF, -DIAG_OFF])
    rec = np.vstack([axis, diag])
    cfg = config(scheme, dx, fs, cuda, kernel)
    traces, wall, solver = run(env, cfg, SRC, rec)
    sig = ricker(cfg.n_steps, fs)
    out = {"scheme": scheme, "dx": dx, "fs": fs, "grid": [env.ny, env.nx], "n_steps": cfg.n_steps, "wall_s": wall,
           "ms_per_step": 1e3 * wall / cfg.n_steps, "kernel": solver.kernel, "finite": bool(np.isfinite(traces).all()),
           "peak": float(np.abs(traces).max())}
    if not out["finite"]:
        return out
    variants = {"free": k_free}
    if scheme == "leapfrog":
        variants["corrected"] = k_corrected(DispersionModel(dx, 1.0 / fs, C, 4))
    S = np.fft.rfft(sig)
    for vname, k_of in variants.items():
        ph, amp, f = phase_errors(traces, sig, fs, rec, k_of)
        res = {}
        for bname, band in (("fine", BAND_FINE), ("coarse", BAND_COARSE)):
            sel = band_bins(f, S, band)
            res[bname] = {
                "max_phase_rad_axis_far": float(np.nanmax(np.abs(ph[:4][FAR][:, sel]))),
                "max_phase_rad_diag_far": float(np.nanmax(np.abs(ph[4:][FAR][:, sel]))),
                "max_phase_rad_axis_all": float(np.nanmax(np.abs(ph[:4][:, sel]))),
                "max_phase_rad_diag_all": float(np.nanmax(np.abs(ph[4:][:, sel]))),
                "max_amp_db_far": float(np.nanmax(np.abs(np.vstack([amp[:4][FAR], amp[4:][FAR]])[:, sel]))),
                "phase_rad_per_receiver_axis": [float(np.nanmax(np.abs(ph[i][sel]))) for i in range(4)],
                "phase_rad_per_receiver_diag": [float(np.nanmax(np.abs(ph[4 + i][sel]))) for i in range(4)],
            }
        out[vname] = res
    return out


def wall_case(name: str, cuda: bool, kernel: str) -> dict:
    """Straight pressure release wall at x = x_w (a node line), source 60 m away, receivers between."""
    scheme, dx, fs = SCHEMES[name]
    width = domain_width(dx)
    x_w = 700.0
    poly = [np.array([(x_w - 0.01, -100.0), (width + 200.0, -100.0), (width + 200.0, width + 200.0), (x_w - 0.01, width + 200.0)])]
    env_wall = _finish_environment(poly, width, width, dx, C, 0.0, "wall")
    env_open = uniform_water_environment(width, width, dx, C)
    src = np.array([x_w - 60.0, 500.0])
    rec = np.array([[x_w - 10.0, 500.0], [x_w - 20.0, 500.0], [x_w - 40.0, 500.0],       # normal incidence
                    [x_w - 20.0, 600.0], [x_w - 20.0, 800.0]])                         # oblique (about 58 and 76 degrees)
    cfg = config(scheme, dx, fs, cuda, kernel)
    tw, wall_t, _ = run(env_wall, cfg, src, rec)
    to, _, _ = run(env_open, cfg, src, rec)
    sig = ricker(cfg.n_steps, fs)
    n = tw.shape[1]
    t = np.arange(n) / fs
    win = (0.5 - 0.5 * np.cos(np.pi * np.clip((WALL_GATE - t) / 0.02, 0.0, 1.0)))[None, :]   # drop the domain edge arrivals
    tw, to = tw * win, to * win
    f = np.fft.rfftfreq(n, 1.0 / fs)
    S = np.fft.rfft(sig)
    Pw, Po = np.fft.rfft(tw, axis=1), np.fft.rfft(to, axis=1)
    refl = Pw - Po
    img = np.array([2 * x_w - src[0], src[1]])
    r_dir = np.hypot(*(rec - src).T)
    r_img = np.hypot(*(rec - img).T)
    k = 2 * np.pi * f / C
    analytic = -hankel2(0, k[None, :] * r_img[:, None]) / hankel2(0, k[None, :] * r_dir[:, None])
    ratio = (refl / Po) / analytic
    out = {"wall_s": wall_t, "finite": bool(np.isfinite(tw).all())}
    for bname, band in (("fine", BAND_FINE), ("coarse", BAND_COARSE)):
        sel = band_bins(f, S, band)
        out[bname] = {"max_phase_rad": float(np.nanmax(np.abs(np.angle(ratio[:, sel])))),
                      "max_amp_db": float(np.nanmax(np.abs(20 * np.log10(np.abs(ratio[:, sel]))))),
                      "per_receiver_phase_rad": [float(np.nanmax(np.abs(np.angle(ratio[i, sel])))) for i in range(len(rec))],
                      "per_receiver_amp_db": [float(np.nanmax(np.abs(20 * np.log10(np.abs(ratio[i, sel]))))) for i in range(len(rec))]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--kernel", default="auto")
    ap.add_argument("--out", default=str(ROOT / "output" / "studies" / "performance"))
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [s for s in args.schemes.split(",") if s]
    results = {"ranges_m": RANGES.tolist(), "band_fine_hz": BAND_FINE, "band_coarse_hz": BAND_COARSE, "stationary": {}, "wall": {}}
    for name in names:
        st = stationary_case(name, args.cuda, args.kernel)
        results["stationary"][name] = st
        if st["finite"]:
            free = st["free"]["fine"]
            msg = f"free k: axis {free['max_phase_rad_axis_far']:.3f} rad, diag {free['max_phase_rad_diag_far']:.3f} rad (400-600 m, 100-760 Hz)"
            if "corrected" in st:
                c = st["corrected"]["fine"]
                msg += f"; corrected: axis {c['max_phase_rad_axis_far']:.3f}, diag {c['max_phase_rad_diag_far']:.3f} rad"
            print(f"[timestep] {name:14s} {st['wall_s']:6.1f} s ({st['ms_per_step']:.3f} ms/step, {st['grid'][0]}x{st['grid'][1]})  {msg}")
        else:
            print(f"[timestep] {name:14s} unstable (non finite traces)")
        w = wall_case(name, args.cuda, args.kernel)
        results["wall"][name] = w
        if w["finite"]:
            print(f"[timestep] {name:14s} wall: phase {w['fine']['max_phase_rad']:.3f} rad, amplitude {w['fine']['max_amp_db']:.2f} dB (100-760 Hz); "
                  f"coarse band phase {w['coarse']['max_phase_rad']:.3f} rad")
        (out / "timestep_schemes.json").write_text(json.dumps(results, indent=2, default=float))
    # Criterion.
    base = results["stationary"].get("leapfrog_0.5m")
    verdict = {"text": "a scheme on the 1 m grid within 0.1 rad at 400-600 m over 100-760 Hz (axis and diagonal) with a wall "
                       "reflection error no worse than the current 0.5 m scheme's", "pass": False, "passing": []}
    if base is not None and base["finite"]:
        ref_phase = max(base["corrected"]["fine"]["max_phase_rad_axis_far"], base["corrected"]["fine"]["max_phase_rad_diag_far"])
        ref_wall = results["wall"]["leapfrog_0.5m"]["fine"]["max_phase_rad"]
        verdict["current_0.5m_corrected_phase_rad"] = ref_phase
        verdict["current_0.5m_wall_phase_rad"] = ref_wall
        for name in names:
            if not name.endswith("_1m"):
                continue
            st = results["stationary"][name]
            w = results["wall"][name]
            if not (st["finite"] and w["finite"]):
                continue
            ph = max(st["free"]["fine"]["max_phase_rad_axis_far"], st["free"]["fine"]["max_phase_rad_diag_far"])
            ok = ph <= min(0.1, max(ref_phase, 0.1)) and w["fine"]["max_phase_rad"] <= ref_wall * 1.0001
            verdict.setdefault("candidates", {})[name] = {"phase_rad": ph, "wall_phase_rad": w["fine"]["max_phase_rad"], "pass": bool(ok),
                                                           "cost_ratio_vs_0.5m": st["wall_s"] / base["wall_s"]}
            if ok:
                verdict["passing"].append(name)
        verdict["pass"] = bool(verdict["passing"])
    results["criterion"] = verdict
    (out / "timestep_schemes.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"[timestep] criterion {'PASS' if verdict['pass'] else 'FAIL'}: {verdict.get('candidates')}")


if __name__ == "__main__":
    main()

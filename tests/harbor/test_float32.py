"""The k space FDTD in float32 against float64 on the tiny CI scenario (CPU): the traces agree within 1e-3 of each
receiver's peak at the worst sample and 1e-4 at the median receiver, and the direct arrival at every hydrophone is the
same to a fraction of a sample (the propagation error unchanged)."""

from __future__ import annotations

import copy

import numpy as np

from acoustic_sim.water import HarborScenarioConfig, build_environment, build_layout
from acoustic_sim.water.fdtd_runner import simulate_source

from .test_ci_end_to_end import SCENARIO


def _run(precision: str):
    d = copy.deepcopy(SCENARIO)
    d["fdtd"].update(time_scheme="kspace", precision=precision, total_time=0.3, use_cuda=False)
    cfg = HarborScenarioConfig.from_dict(d)
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    n, fs = cfg.fdtd.n_steps, cfg.fdtd.fs
    t = np.arange(n) / fs
    f0 = 200.0
    a = (np.pi * f0 * (t - 1.5 / f0)) ** 2
    sig = (1 - 2 * a) * np.exp(-a)
    g = simulate_source(env, layout, cfg.fdtd, sig, position=(25.0, 30.0))
    return g.traces.astype(np.float64), layout, fs


def test_float32_kspace_traces_match_float64():
    a, layout, fs = _run("float64")
    b, _, _ = _run("float32")
    peak = np.max(np.abs(a), axis=1)
    e = np.max(np.abs(b - a), axis=1) / peak
    assert np.max(e) < 1e-3, np.max(e)
    assert np.median(e) < 1e-4, np.median(e)
    hyd = layout.slices["hydrophones"]
    up = 16
    for row in range(hyd.start, hyd.stop):
        x, y = np.fft.irfft(np.fft.rfft(a[row]), up * a.shape[1]), np.fft.irfft(np.fft.rfft(b[row]), up * b.shape[1])
        lag = np.argmax(np.correlate(y, x, "full")) - (len(x) - 1)
        assert abs(lag) / (up * fs) * 1e3 < 0.02                       # the arrival moves by less than 0.02 ms

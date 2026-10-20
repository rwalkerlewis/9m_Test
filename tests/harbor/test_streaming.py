"""The real time passive chain (``water/streaming.py``): it finds a radiating source, and it is causal."""

from __future__ import annotations

import numpy as np

from acoustic_sim.contract import ChartEnvironment
from acoustic_sim.water.ais import AISReports
from acoustic_sim.water.detection import ProcessingConfig
from acoustic_sim.water.streaming import StreamConfig, StreamingPassive

from ..test_contract import _field_recording


def _setup():
    rec = _field_recording()
    chart = ChartEnvironment(water=np.ones((len(rec["y"]), len(rec["x"])), bool), depth=np.full((len(rec["y"]), len(rec["x"])), 15.0),
                             x=rec["x"], y=rec["y"], c_nominal=rec["c"])
    pc = ProcessingConfig(win_s=0.5, hop_s=0.25, f_lo=80.0, f_hi=450.0, c=rec["c"], grid_dx=5.0, grid_margin=0.0,
                          grid_bounds=(20.0, 380.0, 20.0, 380.0), min_range=1.0)
    stc = StreamConfig(block_frames=2, history_s=1.5, warmup_s=1.5, interference_every=4, max_peaks=2)
    none = AISReports(np.array([], dtype=str), np.empty(0), np.empty((0, 2)), np.empty(0), np.empty(0))
    return rec, chart, pc, stc, none


def _stream(traces, rec, chart, pc, stc, reports, chunk):
    det = StreamingPassive(rec["layout"], rec["fs"], chart, pc, stc=stc)
    out = []
    for s0 in range(0, traces.shape[1], chunk):
        out += det.push(traces[:, s0:s0 + chunk], reports)
    return out


def test_streaming_finds_the_ship_without_its_ais_report():
    """With no AIS report the ship is an unknown radiating source: the top peak of most frames is on it
    (the batch chain, ``run_ais_aware`` on the whole record, has 73 % of its frames within 10 m here)."""
    rec, chart, pc, stc, none = _setup()
    frames = _stream(rec["traces"], rec, chart, pc, stc, none, chunk=500)
    assert len(frames) >= 5
    ship = np.array([150.0, 250.0])
    err = np.array([np.hypot(*(f.xy[0] - (ship + [4.0 * f.t, 0.0]))) for f in frames])
    assert np.mean(err <= 10.0) >= 0.6, err
    assert all(f.latency_s >= f.compute_s for f in frames)


def test_streaming_is_causal():
    """Changing the data after time t changes no frame whose block ended before t; the chunk size does not matter."""
    rec, chart, pc, stc, none = _setup()
    a = _stream(rec["traces"], rec, chart, pc, stc, none, chunk=500)
    cut = int(2.0 * rec["fs"])
    other = rec["traces"].copy()
    other[:, cut:] = np.random.default_rng(3).standard_normal(other[:, cut:].shape)
    b = _stream(other, rec, chart, pc, stc, none, chunk=137)
    nper, hop = int(pc.win_s * rec["fs"]), int(pc.hop_s * rec["fs"])
    n_safe = 0
    for fa, fb in zip(a, b):
        k = int(round((fa.t * rec["fs"] - nper / 2) / hop))
        block_stop = (k // stc.block_frames * stc.block_frames + stc.block_frames - 1) * hop + nper
        if block_stop <= cut:
            np.testing.assert_allclose(fa.xy, fb.xy)
            np.testing.assert_allclose(fa.score, fb.score, rtol=1e-5)
            n_safe += 1
    assert n_safe >= 2


def _echo(traces, rec, chart, chunk):
    """Frames of the streaming echo chain, each with the number of samples received when it was output."""
    from acoustic_sim.water.bistatic import BistaticConfig
    from acoustic_sim.water.streaming import EchoStreamConfig, StreamingEcho

    bc = BistaticConfig(c=rec["c"], f_lo=80.0, f_hi=450.0, grid_dx=5.0, edge=20.0, exclusion_radius=30.0, coherent_norm="snr",
                        map_scale="none", eca_doppler=True, track_smooth_s=1.0, eca_images=False)
    det = StreamingEcho(rec["layout"], rec["fs"], chart, bc, "ship", EchoStreamConfig(step_blocks=1, history_s=0.5, lookahead_s=0.4))
    out = []
    for s0 in range(0, traces.shape[1], chunk):
        out += [(f, min(s0 + chunk, traces.shape[1])) for f in det.push(traces[:, s0:s0 + chunk], rec["reports"])]
    return out


def test_streaming_echo_finds_the_silent_body():
    """The echo of the silent body, lit by the AIS ship: the top peak of the frames is on the body (the batch
    ``run_bistatic`` on the same recording has its peak within 8 m in at least 40 % of the frames)."""
    rec, chart, *_ = _setup()
    frames = [f for f, _ in _echo(rec["traces"], rec, chart, 500) if f.ships]
    assert len(frames) >= 3
    err = np.array([np.hypot(*(f.xy[0] - rec["body"])) for f in frames])
    assert np.mean(err <= 8.0) >= 0.5, err


def test_streaming_echo_is_causal():
    """Changing the data after sample *cut* changes no frame output before that sample arrived."""
    rec, chart, *_ = _setup()
    a = _echo(rec["traces"], rec, chart, 400)
    cut = int(2.4 * rec["fs"])
    other = rec["traces"].copy()
    other[:, cut:] = np.random.default_rng(5).standard_normal(other[:, cut:].shape) * other.std()
    b = _echo(other, rec, chart, 400)
    safe = [(fa, fb) for (fa, na), (fb, _) in zip(a, b) if na <= cut]
    assert len(safe) >= 2 and len(safe) < len(a)
    for fa, fb in safe:
        np.testing.assert_array_equal(fa.xy, fb.xy)
        np.testing.assert_array_equal(fa.score, fb.score)


def test_fast_replicas_match_green2d_on_the_gpu():
    """``green2d_bins`` (phase recurrence) and the streaming echo kernel (channel replicas from the range table)
    against ``green2d`` and ``SensorLayout.collapse``."""
    import pytest

    from ._gpu import HAS_GPU

    if not HAS_GPU:
        pytest.skip("no GPU")
    import cupy as cp

    from acoustic_sim.water.bistatic import BistaticConfig
    from acoustic_sim.water.propagation import green2d, green2d_bins
    from acoustic_sim.water.streaming import StreamingEcho

    rec, chart, *_ = _setup()
    rng = np.random.default_rng(0)
    r = cp.asarray(rng.uniform(1.0, 400.0, (300, 50)))
    ks = 2 * np.pi * np.arange(80.0, 450.0, 4.0) / rec["c"]
    for k, g in zip(ks, green2d_bins(r, ks, cp)):
        ref = green2d(k * r, cp)
        assert float(cp.max(cp.abs(g - ref)) / cp.max(cp.abs(ref))) < 1e-12
    bc = BistaticConfig(c=rec["c"], f_lo=80.0, f_hi=450.0, grid_dx=5.0, edge=20.0, coherent_norm="snr", map_scale="none",
                        eca_doppler=True, eca_images=False, use_cuda=True)
    det = StreamingEcho(rec["layout"], rec["fs"], chart, bc, "ship")
    w = cp.asarray(rng.uniform(0.5, 1.5, rec["layout"].n_receivers).astype(np.float32))
    for k, A in zip(det.ks, det._channel_greens(w)):
        ref = rec["layout"].collapse(green2d(k * det.r_samples, cp), cp) * w[None, :]
        assert float(cp.max(cp.abs(A - ref)) / cp.max(cp.abs(ref))) < 2e-5

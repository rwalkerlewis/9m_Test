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

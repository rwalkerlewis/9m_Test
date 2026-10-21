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
    """Frames of the streaming echo chain, each with the number of samples received when it was output.  The
    ship's position fixes are the input a tracker would give (here its track every 0.25 s)."""
    from acoustic_sim.water.bistatic import BistaticConfig
    from acoustic_sim.water.streaming import EchoStreamConfig, StreamingEcho

    bc = BistaticConfig(c=rec["c"], f_lo=80.0, f_hi=450.0, grid_dx=5.0, edge=20.0, exclusion_radius=30.0, coherent_norm="snr",
                        map_scale="none", eca_doppler=True, track_smooth_s=1.0, eca_images=False)
    det = StreamingEcho(rec["layout"], rec["fs"], chart, bc, EchoStreamConfig(step_blocks=1, history_s=0.5, lookahead_s=0.4))
    out = []
    for s0 in range(0, traces.shape[1], chunk):
        t1 = min(s0 + chunk, traces.shape[1]) / rec["fs"]
        t_fix = np.arange(0.0, t1, 0.25)
        t_fix = t_fix[t_fix > (s0 / rec["fs"]) - 0.25] if s0 else t_fix
        fixes = {"ship": [(t, np.array([150.0 + 4.0 * t, 250.0])) for t in t_fix if t <= t1 - 0.25]}
        out += [(f, min(s0 + chunk, traces.shape[1])) for f in det.push(traces[:, s0:s0 + chunk], fixes)]
    return out


def test_streaming_echo_finds_the_silent_body():
    """The echo of the silent body, lit by the ship: the top peak of the frames is on the body (the batch
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


def test_tracker_confirms_moving_radiators_and_ais_labels_them_afterwards():
    """Two radiators and random clutter: two confirmed tracks, the clutter and a sidelobe next to the stronger
    radiator are not confirmed, only the stronger one is strong enough for the echo chain (the weaker scores 20 %),
    and the AIS cross check labels the one that reports."""
    from acoustic_sim.water.streaming import FrameResult, RadiatorTracker, ais_crosscheck

    rng = np.random.default_rng(1)
    tracker = RadiatorTracker()
    for k in range(60):
        t = 0.25 * k
        a, b = np.array([100.0 + 3.0 * t, 200.0]), np.array([300.0, 100.0 + 1.5 * t])
        pts = [a + rng.normal(0, 1, 2), b + rng.normal(0, 1, 2), a + [0.0, 20.0], rng.uniform(0, 400, 2)]
        tracker.update(FrameResult(t, np.array(pts), np.array([100.0, 20.0, 40.0, 30.0]), {}, 0.0, 0.0))
    conf = tracker.confirmed()
    assert len(conf) == 2
    ends = sorted(tr.fixes[-1][1][0] for tr in conf.values())
    assert abs(ends[0] - 144.0) < 5 and abs(ends[1] - 300.0) < 5
    assert list(tracker.strong()) == [max(conf, key=lambda k: conf[k].score)]
    k = 12
    reports = AISReports(np.array(["ferry"] * k), np.arange(k) * 1.25, np.array([[100.0 + 3.0 * 1.25 * i, 200.0] for i in range(k)]),
                         np.full(k, 3.0), np.zeros(k))
    labels = ais_crosscheck(conf, reports)
    assert sorted(labels.values(), key=str) == [None, "ferry"]


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
    det = StreamingEcho(rec["layout"], rec["fs"], chart, bc)
    w = cp.asarray(rng.uniform(0.5, 1.5, rec["layout"].n_receivers).astype(np.float32))
    for k, A in zip(det.ks, det._channel_greens(w)):
        ref = rec["layout"].collapse(green2d(k * det.r_samples, cp), cp) * w[None, :]
        assert float(cp.max(cp.abs(A - ref)) / cp.max(cp.abs(ref))) < 2e-5


def test_a_channel_zeroed_by_qc_contributes_nothing_from_the_first_block():
    """A channel 100 times too loud is zeroed by the channel QC: from the first output block on, the frames (and the
    re-whitening floor of that channel) are those of the same recording with the channel silent.  The warmup frames
    are stored without channel weights and the current weights are applied when they are used."""
    rec, chart, pc, stc, none = _setup()
    j = 3
    loud, dead = rec["traces"].copy(), rec["traces"].copy()
    loud[j] *= 100.0
    dead[j] = 0.0
    out = []
    for tr in (loud, dead):
        det = StreamingPassive(rec["layout"], rec["fs"], chart, pc, stc=stc)
        frames = []
        for s0 in range(0, tr.shape[1], 500):
            frames += det.push(tr[:, s0:s0 + 500], none)
        out.append(frames)
    assert len(out[0]) == len(out[1]) >= 5
    for fa, fb in zip(*out):
        np.testing.assert_allclose(fa.xy, fb.xy)
        np.testing.assert_allclose(fa.score, fb.score, rtol=1e-5)


def test_one_sound_speed_a_free_field_propagator_must_agree_with_the_config():
    """The bistatic replicas take their wavenumber from the config's ``c``: a free field propagator with another
    speed is refused (``propagation.resolve``) by the batch chain, its replica model and the streaming echo chain."""
    import pytest

    from acoustic_sim.water.bistatic import BistaticConfig, ReplicaModel, run_bistatic
    from acoustic_sim.water.propagation import FreeFieldPropagator
    from acoustic_sim.water.streaming import StreamingEcho

    rec, chart, *_ = _setup()
    other = FreeFieldPropagator(rec["c"] + 20.0)
    bc = BistaticConfig(c=rec["c"], propagator=other, f_lo=80.0, f_hi=450.0, grid_dx=5.0, edge=20.0, coherent_norm="snr",
                        map_scale="none", eca_doppler=True, eca_images=False)
    with pytest.raises(ValueError, match="disagrees"):
        ReplicaModel(rec["layout"], bc.c, propagator=other)
    with pytest.raises(ValueError, match="disagrees"):
        run_bistatic(rec["traces"], rec["layout"], rec["fs"], chart, rec["reports"], bc)
    with pytest.raises(ValueError, match="disagrees"):
        StreamingEcho(rec["layout"], rec["fs"], chart, bc)


def test_streaming_passive_uses_only_the_ais_reports_already_received():
    """Given every report of the record at every push (the future ones too), a ship is refined, projected out and
    excluded only in frames at or after its first report (the batch ``bistatic.ship_positions`` looks 60 s ahead)."""
    rec, chart, pc, stc, _ = _setup()
    rep = rec["reports"]
    m = rep.t >= 2.0                                           # first report at 2.0 s
    reports = AISReports(rep.vessel[m], rep.t[m], rep.xy[m], rep.sog[m], rep.cog[m])
    frames = _stream(rec["traces"], rec, chart, pc, stc, reports, chunk=500)
    before = [f for f in frames if f.t < 2.0]
    after = [f for f in frames if f.t >= 2.0]
    assert before and after
    assert all("ship" not in f.ships for f in before)
    assert all("ship" in f.ships for f in after)


def test_end_to_end_passive_tracker_echo_locates_the_silent_body_without_truth():
    """The path ``examples/harbor_stream.py`` runs, on a synthetic 8 s scene (one ship, one silent scatterer, no AIS):
    ``StreamingPassive`` -> ``RadiatorTracker`` -> ``StreamingEcho`` with the strong tracks as illuminators.  Nothing
    from the truth goes in.  The ship becomes the one strong track (within 3 m), and the top echo peak is within 8 m
    of the scatterer in at least half of the mapped frames, and the sum of the frame maps (each over its median, as
    walkthrough step 6) peaks within 5 m of it."""
    from acoustic_sim.water.bistatic import BistaticConfig
    from acoustic_sim.water.streaming import EchoStreamConfig, RadiatorTracker, StreamingEcho

    from ..test_contract import _field_recording

    rec = _field_recording(8.0)                                 # measured: 62 % of the frames within 8 m, record sum on the body
    _, chart, pc, stc, none = _setup()
    passive = StreamingPassive(rec["layout"], rec["fs"], chart, pc, stc=stc)
    tracker = RadiatorTracker()
    bc = BistaticConfig(c=rec["c"], f_lo=80.0, f_hi=450.0, grid_dx=5.0, edge=20.0, exclusion_radius=30.0, coherent_norm="snr",
                        map_scale="none", eca_doppler=True, track_smooth_s=1.0, eca_images=False)
    echo = StreamingEcho(rec["layout"], rec["fs"], chart, bc, EchoStreamConfig(step_blocks=1, history_s=0.5, lookahead_s=0.4),
                         keep_maps=True)
    sent, e_frames = {}, []
    for s0 in range(0, rec["traces"].shape[1], 500):
        chunk = rec["traces"][:, s0:s0 + 500]
        for f in passive.push(chunk, none):
            tracker.update(f)
        fixes = {}
        for k, tr in tracker.strong().items():
            fixes[k] = [(f[0], f[1]) for f in tr.fixes[sent.get(k, 0):]]
            sent[k] = len(tr.fixes)
        e_frames += [f for f in echo.push(chunk, fixes) if f.ships]
    strong = tracker.strong()
    assert len(strong) == 1
    tr = next(iter(strong.values()))
    err = [np.hypot(*(f[1] - (np.array([150.0, 250.0]) + [4.0 * f[0], 0.0]))) for f in tr.fixes[-8:]]
    assert np.median(err) <= 3.0, err
    assert len(e_frames) >= 8
    d = np.array([np.hypot(*(f.xy[0] - rec["body"])) for f in e_frames])
    assert np.mean(d <= 8.0) >= 0.5, d
    S = sum(f.B / np.median(f.B[f.B > 0]) for f in e_frames)
    assert np.hypot(*(echo.grid[np.argmax(S)] - rec["body"])) <= 5.0


def _ship_map(mode: str | None, exclusion: float):
    """Streaming passive maps of the field recording with the ship's AIS reports (``mode``) or without (``None``)."""
    from acoustic_sim.water.suppression import SuppressionConfig

    rec, chart, pc, stc, none = _setup()
    sc = SuppressionConfig(mode=mode or "adaptive", exclusion_radius=exclusion, refine_half=6.0, refine_step=1.0, min_significance=2.0)
    det = StreamingPassive(rec["layout"], rec["fs"], chart, pc, sc, stc, keep_maps=True)
    frames = []
    for s0 in range(0, rec["traces"].shape[1], 500):
        frames += det.push(rec["traces"][:, s0:s0 + 500], rec["reports"] if mode else none)
    ship = np.array([150.0, 250.0])
    at_ship = np.array([f.B[np.hypot(*(det.grid - (ship + [4.0 * f.t, 0.0])).T) <= 6.0].max() for f in frames])
    return frames, det.grid, at_ship


def test_streaming_ship_projection_adaptive_and_model_remove_the_reporting_ship():
    """With its AIS reports the ship is projected out of the streaming passive maps: the adaptive projection
    (``StreamingPassive._adaptive``) and the replica patch projection (``project_out_ships``) each take the map at
    the ship down by more than 10 dB against the run without reports (no exclusion zone, so the map is seen)."""
    _, _, free = _ship_map(None, 0.0)
    for mode in ("adaptive", "model"):
        frames, _, cleaned = _ship_map(mode, 0.0)
        assert all("ship" in f.ships for f in frames)
        assert np.median(cleaned) < 0.1 * np.median(free), (mode, np.median(cleaned), np.median(free))


def test_streaming_exclusion_zone_from_the_ais_reports():
    """The cells within ``exclusion_radius`` of the refined position of a reporting ship are zeroed and no peak is
    reported there."""
    frames, grid, _ = _ship_map("adaptive", 30.0)
    for f in frames:
        p = f.ships["ship"]
        near = np.hypot(*(grid - p).T) < 30.0
        assert np.all(f.B[near] == 0.0)
        assert all(np.hypot(*(xy - p)) >= 30.0 for xy, s in zip(f.xy, f.score) if s > 0)

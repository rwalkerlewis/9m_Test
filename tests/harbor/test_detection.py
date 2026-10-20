"""M3: detection, localisation and scoring."""

import numpy as np
import pytest
from scipy.special import hankel2

from acoustic_sim.water import (
    Cable,
    HarborFDTDConfig,
    HarborScenarioConfig,
    NoiseConfig,
    SensorLayout,
    VesselConfig,
    assemble_dataset,
    build_track,
    build_synthetic_harbor,
    render_vessel_gathers,
    uniform_water_environment,
)
from acoustic_sim.water.detection import (
    DetectionSet,
    ProcessingConfig,
    build_grid,
    channel_weights,
    green2d,
    remove_common_mode,
    run_detection,
    threshold_from_maxima,
)
from acoustic_sim.water.evaluate import associate, score_detections
from acoustic_sim.contract import Truth

FS = 600.0


# ----------------------------- unit pieces -----------------------------

def test_green2d_matches_hankel_and_is_continuous():
    kr = np.array([0.5, 3.0, 7.9, 8.1, 50.0, 400.0])
    g = green2d(kr)
    assert np.allclose(g[:3], hankel2(0, kr[:3]))
    assert np.allclose(g[3:], hankel2(0, kr[3:]), rtol=2e-3, atol=1e-4)
    assert np.allclose(np.abs(g[1:]), np.abs(hankel2(0, kr[1:])), rtol=2e-3)


def test_threshold_from_maxima_extrapolates_exponential_tail():
    rng = np.random.default_rng(0)
    maxima = rng.exponential(scale=2.0, size=4000)
    for pfa in (0.1, 0.01, 0.001):
        thr = threshold_from_maxima(maxima, pfa)
        assert thr == pytest.approx(-2.0 * np.log(pfa), rel=0.15)
    assert threshold_from_maxima(maxima, 0.001) > threshold_from_maxima(maxima, 0.1)


def _layout():
    cable = Cable("c", np.array([[200.0, 300.0], [700.0, 300.0]]), channel_spacing=25.0)
    return SensorLayout(cables=[cable], hydrophones=np.array([[300.0, 500.0], [600.0, 500.0]]))


def test_channel_weights_flag_dead_and_noisy_channels_and_respect_groups():
    layout = _layout()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((layout.n_receivers, 2000))
    x[3] = 0.0           # dead
    x[5] *= 50.0         # abnormally loud
    w = channel_weights(x, layout, ProcessingConfig())
    assert w[3] == 0 and w[5] == 0 and w[0] == 1 and np.all(w[layout.n_das:] == 1)
    only_h = channel_weights(x, layout, ProcessingConfig(sensor_groups=("hydrophones",)))
    assert only_h[: layout.n_das].sum() == 0 and only_h[layout.n_das:].sum() == layout.n_hydrophones
    only_das = channel_weights(x, layout, ProcessingConfig(sensor_groups=("c",)))
    assert only_das[layout.n_das:].sum() == 0 and only_das[0] == 1


def test_common_mode_removal_only_on_das():
    layout = _layout()
    rng = np.random.default_rng(1)
    cm = rng.standard_normal(1000)
    x = rng.standard_normal((layout.n_receivers, 1000)) * 0.1
    x[: layout.n_das] += cm
    w = np.ones(layout.n_receivers)
    y = remove_common_mode(x, layout, w)
    assert np.std(y[: layout.n_das]) < 0.2
    assert np.array_equal(y[layout.n_das:], x[layout.n_das:])


def test_build_grid_is_water_with_margin():
    env = build_synthetic_harbor()
    pc = ProcessingConfig(grid_margin=30.0)
    grid = build_grid(env, pc)
    assert env.is_water(grid, margin=30.0).all()
    assert grid[:, 0].min() >= 150 and grid[:, 0].max() <= 2250
    assert 4000 < len(grid) < 8000


# ----------------------------- scoring -----------------------------

def _truth_track(kind="uuv", ais=False, level=1.0):
    from acoustic_sim.water import Track

    t = np.arange(0, 21.0, 1.0)
    xy = np.column_stack([100.0 + 10.0 * t, np.full_like(t, 200.0)])
    return Track("v", kind, ais, t, xy, np.full_like(t, 10.0), np.ones_like(t, bool), level_pa=level)


def _dets(rows):
    """rows: (frame, t, x, y, score)"""
    a = np.array(rows, dtype=float).reshape(-1, 5)
    xy = a[:, 2:4]
    return DetectionSet(a[:, 0].astype(int), a[:, 1], xy, xy.copy(), a[:, 4], np.zeros(len(a), int), [np.empty((0, 2))] * len(a))


def test_associate_gates_by_distance_and_activity():
    tr = _truth_track()
    d = _dets([(0, 5.0, 150.0 + 3.0, 200.0, 50), (1, 6.0, 400.0, 600.0, 50), (2, 25.0, 100.0, 200.0, 50)])
    labels, dist = associate(d, {"v": tr}, gate=50.0)
    assert labels == ["v", None, None]          # far away; target inactive at t=25
    assert dist[0] == pytest.approx(3.0, abs=0.1)


def test_score_pd_rmse_false_alarms_and_threshold():
    tr = _truth_track()
    ais_ship = _truth_track(kind="ship", ais=True)
    ais_ship.id = "ship"
    ais_ship.xy = ais_ship.xy + np.array([0.0, 500.0])
    layout = SensorLayout(hydrophones=np.array([[0.0, 0.0], [500.0, 0.0]]))
    times = np.array([2.0, 4.0, 6.0, 8.0])
    d = _dets([
        (0, 2.0, 120.0 + 4.0, 200.0, 40),        # hit, 4 m error
        (1, 4.0, 140.0, 200.0 + 6.0, 40),        # hit, 6 m error
        (2, 6.0, 700.0, 900.0, 40),              # false alarm
        (3, 8.0, 180.0 + 5.0, 700.0, 40),        # near the ship, not the UUV -> "other"
        (3, 8.0, 180.0, 200.0, 5),               # below threshold, ignored
    ])
    sc = score_detections(d, Truth(tracks={"v": tr, "ship": ais_ship}), layout, times, threshold=10.0, targets=["v"], gate=50.0)
    assert sc.pd("v") == pytest.approx(0.5)
    assert sc.rmse("v") == pytest.approx(np.sqrt((16 + 36) / 2))
    assert sc.n_false == 1 and sc.n_other == 1
    assert sc.false_alarms_per_frame == pytest.approx(0.25)
    assert sc.pd("v", (0, 1e6)) == 0.5 and np.isnan(sc.pd("v", (1e6, 2e6)))
    assert score_detections(d, Truth(tracks={"v": tr, "ship": ais_ship}), layout, times, threshold=1.0, targets=["v"]).pd("v") == 0.75


def test_absent_vessels_are_not_associable():
    tr = _truth_track(level=0.0)
    d = _dets([(0, 2.0, 120.0, 200.0, 40)])
    layout = SensorLayout(hydrophones=np.array([[0.0, 0.0]]))
    sc = score_detections(d, Truth(tracks={"v": tr}), layout, np.array([2.0]), threshold=10.0)
    assert sc.n_false == 1 and sc.rows == []


# ----------------------------- successive cancellation (synthetic) -----------------------------

def _synthetic_whitened(amps, positions, n_frames=6, seed=0):
    """Whitened data built directly from free-field replicas: unit complex noise + tones at *positions*."""
    from acoustic_sim.water.detection import MatchedFieldScanner, Whitened

    rng = np.random.default_rng(seed)
    layout = SensorLayout(
        cables=[Cable("c", np.array([[200.0, 200.0], [500.0, 200.0], [500.0, 500.0]]), channel_spacing=15.0)],
        hydrophones=np.array([[300.0, 450.0], [700.0, 250.0]]),
    )
    pc = ProcessingConfig(interference_rank_max=0, grid_bounds=(150, 850, 120, 650), min_separation=60.0, bin_stride=1)
    freqs = np.arange(20.0, 61.0, 1.0)
    n, f = layout.n_receivers, len(freqs)
    wh = Whitened(freqs=freqs, times=np.arange(n_frames) * 2.0 + 2.0,
                  Z=np.zeros((n, f, n_frames), np.complex64), sigma=np.ones((n, f)),
                  weights=np.ones(n), scan_bins=np.arange(f))
    scanner = MatchedFieldScanner(layout, build_grid(uniform_water_environment(1000, 750, 5.0), pc), pc)
    Z = (rng.standard_normal((n, f, n_frames)) + 1j * rng.standard_normal((n, f, n_frames))) / np.sqrt(2.0)
    for amp, xy in zip(amps, positions):
        r = scanner._geometry(np.atleast_2d(xy))
        for i in range(f):
            rep = scanner._replicas(r, wh, i)[0]
            s_ = amp * (rng.standard_normal(n_frames) + 1j * rng.standard_normal(n_frames)) / np.sqrt(2.0)
            Z[:, i, :] += rep[:, None] * s_[None, :]
    wh.Z = Z.astype(np.complex64)
    return scanner, wh


def test_successive_cancellation_keeps_independent_sources_and_drops_sidelobes():
    from acoustic_sim.water.detection import extract_detections

    strong, weak = np.array([350.0, 330.0]), np.array([650.0, 520.0])
    for amps, pos, expect in [([12.0], [strong], 1), ([12.0, 5.0], [strong, weak], 2)]:
        scanner, wh = _synthetic_whitened(amps, pos)
        scan = scanner.scan(wh)
        dets = extract_detections(scanner, wh, scan, floor=60.0)
        per_frame = np.bincount(dets.frame, minlength=len(wh.times))
        assert np.all(per_frame == expect), (amps, per_frame)
        for p, rank in zip(pos, range(expect)):
            sel = dets.xy[dets.rank == rank]
            assert len(sel) == len(wh.times)
            assert np.max(np.hypot(*(sel - p).T)) < 15.0
        assert np.all(dets.score[dets.rank == 0] > dets.score[dets.rank == 1].max(initial=0.0)) or expect == 1


def test_match_frame_deflation_removes_the_source_and_its_sidelobes_but_not_others():
    from acoustic_sim.water.detection import ScanResult

    p1, p2 = np.array([350.0, 330.0]), np.array([650.0, 520.0])
    # Two sources: deflating one leaves the other intact.
    scanner, wh = _synthetic_whitened([12.0, 12.0], [p1, p2])
    scan = ScanResult(scanner.grid, wh.times, np.zeros((len(scanner.grid), len(wh.times))), {}, np.zeros(0))
    bins = wh.scan_bins
    plain = scanner.match_frame(wh, scan, np.array([p1, p2]), 2, bins)
    after = scanner.match_frame(wh, scan, p2[None], 2, bins, deflate_xy=p1[None])
    assert plain.mean() > 10.0
    assert after[:, 0].sum() > 0.7 * plain[:, 1].sum()
    # One source: after deflating it, nothing is left anywhere near it (sidelobe region).
    scanner, wh = _synthetic_whitened([12.0], [p1])
    scan = ScanResult(scanner.grid, wh.times, np.zeros((len(scanner.grid), len(wh.times))), {}, np.zeros(0))
    near = p1 + np.array([[60.0, 0.0], [0.0, 80.0], [-70.0, -40.0], [90.0, 30.0]])
    before = scanner.match_frame(wh, scan, near, 2, bins).sum(axis=0)
    after = scanner.match_frame(wh, scan, near, 2, bins, deflate_xy=p1[None]).sum(axis=0)
    assert np.all(after < 0.5 * before) and after.max() < 1.5 * len(bins)   # ~ noise level (Exp(1) per bin)


# ----------------------------- end to end -----------------------------

@pytest.fixture(scope="module")
def open_water():
    env = uniform_water_environment(1000, 750, 5.0, 1500.0)
    cable = Cable("cable1", np.array([[250.0, 200.0], [500.0, 200.0], [500.0, 500.0], [750.0, 520.0]]), channel_spacing=15.0)
    layout = SensorLayout(cables=[cable], hydrophones=np.array([[300.0, 450.0], [700.0, 250.0], [450.0, 600.0]]))
    cfg = HarborScenarioConfig(
        fdtd=HarborFDTDConfig(fs=FS, total_time=32.0),
        # Must move: a stationary source is (by design) rejected like any persistent interference.
        vessels=[VesselConfig("uuv", "uuv", [[300, 380], [480, 380]], speed=5.5, level_pa=0.1, seed=3)],
        noise=NoiseConfig(n_shipping=2, shipping_rms=0.2, ambient_rms=0.03, das_dead_fraction=0.05),
        seed=1,
    )
    tracks = {v.id: build_track(v, 32.0, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks)
    pc = ProcessingConfig(grid_bounds=(150, 850, 120, 650), min_separation=60.0)
    return env, layout, cfg, tracks, gathers, pc


def test_moving_source_is_detected_and_localised(open_water):
    env, layout, cfg, tracks, gathers, pc = open_water
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=2)
    run = run_detection(ds.traces, layout, FS, env, pc, floor=20.0)
    top = run.dets.above(20.0)
    sel = top.rank == 0
    assert sel.sum() >= 0.8 * len(run.wh.times)
    truth = tracks["uuv"].position_at(top.t[sel])
    err = np.hypot(*(top.xy[sel] - truth).T)
    assert np.median(err) < 15.0 and np.percentile(err, 90) < 40.0
    # Each detection carries its strongest matched-field lines, sorted, inside the scanned band.
    for ln in top.lines:
        assert 1 <= len(ln) <= pc.n_lines
        assert np.all(np.diff(ln[:, 1]) <= 0) and ln[:, 0].min() >= pc.f_lo and ln[:, 0].max() <= pc.f_hi
    assert np.allclose(run.wh.weights[ds.dead], 0.0)
    # Sidelobes are cancelled: essentially one detection per frame for a single target.
    assert len(top) <= 1.3 * len(run.wh.times)


def test_noise_only_scores_stay_far_below_target(open_water):
    env, layout, cfg, tracks, gathers, pc = open_water
    ds_t = assemble_dataset(cfg, layout, tracks, gathers, seed=3)
    ds_n = assemble_dataset(cfg, layout, tracks, gathers, seed=3, levels={"uuv": 0.0})
    t = run_detection(ds_t.traces, layout, FS, env, pc, floor=1e18)
    n = run_detection(ds_n.traces, layout, FS, env, pc, floor=1e18)
    assert np.median(t.scan.frame_max) > 5 * n.scan.frame_max.max()


def test_interference_projection_suppresses_distant_shipping(open_water):
    env, layout, cfg, tracks, gathers, pc = open_water
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=4, levels={"uuv": 0.0})
    on = run_detection(ds.traces, layout, FS, env, pc, floor=1e18)
    off = run_detection(ds.traces, layout, FS, env, ProcessingConfig(**{**pc.__dict__, "interference_rank_max": 0}), floor=1e18)
    assert np.median(on.scan.frame_max) < 0.3 * np.median(off.scan.frame_max)
    assert on.scan.ranks.mean() >= 1.5          # two plane waves -> rank about 2


def test_stationary_source_is_rejected_as_persistent_interference(open_water):
    """Documented limitation: anything that stays put (moored ship, bottomed UUV) is projected out."""
    env, layout, cfg, tracks, gathers, pc = open_water
    still = VesselConfig("uuv", "uuv", [[400, 330]], level_pa=0.1, seed=3)
    cfg2 = HarborScenarioConfig(fdtd=cfg.fdtd, vessels=[still], noise=cfg.noise, seed=1)
    tr2 = {"uuv": build_track(still, 32.0, env)}
    g2 = render_vessel_gathers(cfg2, env, layout, tr2)
    ds = assemble_dataset(cfg2, layout, tr2, g2, seed=2)
    run = run_detection(ds.traces, layout, FS, env, pc, floor=1e18)
    assert np.median(run.scan.frame_max) < 20.0


def test_das_only_and_hydrophone_only_configurations_run(open_water):
    env, layout, cfg, tracks, gathers, pc = open_water
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=5)
    run = run_detection(ds.traces, layout, FS, env, ProcessingConfig(**{**pc.__dict__, "sensor_groups": ("cable1",)}), floor=15.0)
    best = run.dets.xy[run.dets.rank == 0]
    truth = tracks["uuv"].position_at(run.dets.t[run.dets.rank == 0])
    assert len(best) > 0.5 * len(run.wh.times)
    assert np.median(np.hypot(*(best - truth).T)) < 25.0
    hyd = run_detection(ds.traces, layout, FS, env, ProcessingConfig(**{**pc.__dict__, "sensor_groups": ("hydrophones",)}), floor=1e18)
    assert hyd.wh.weights[: layout.n_das].sum() == 0

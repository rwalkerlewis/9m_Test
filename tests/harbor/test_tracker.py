"""The tracker of the harbor picture (``streaming.RadiatorTracker``, constant velocity Kalman filter per track) and the
scoring of static bodies by their outline, on inputs with known answers: a straight track with known measurement noise
(the state error against the filter's covariance, the gate's acceptance), a turning track, the sidelobe test on a
synthetic two source map, the echo coherence, the static decision, the outline distance."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_sim.contract import Track, Truth
from acoustic_sim.water.scoreboard import ObjectDistance, match_tracks, static_body_cells
from acoustic_sim.water.stream import ellipse95, spectral_coherence
from acoustic_sim.water.streaming import FrameResult, RadiatorTracker, TrackerConfig, fix_covariance

SIGMA = 1.5          # measurement noise per axis [m] (the covariance floor of these frames: no map)


def _frames(path, sigma=SIGMA, dt=0.25, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    for k, p in enumerate(path):
        pts = [np.asarray(p) + sigma * rng.standard_normal(2)]
        if extra is not None:
            pts += list(extra(k, rng))
        yield FrameResult(k * dt, np.array(pts), np.linspace(100.0, 10.0, len(pts)), {}, 0.0, 0.0)


def test_straight_track_error_against_covariance_and_gate():
    """A ship at 3 m/s with fixes of known noise: one track; the normalised position error squared averages 2 (two
    degrees of freedom) within its sampling error; the 0.99 gate takes at least 98 % of the fixes."""
    cfg = TrackerConfig(accel=0.05, meas_floor=SIGMA, gate_probability=0.99, confirm=4, confirm_window_s=2.0, drop_s=2.0)
    n = 400
    t = 0.25 * np.arange(n)
    truth = np.column_stack([100.0 + 3.0 * t, 200.0 - 1.0 * t])
    tracker = RadiatorTracker(cfg)
    nees = []
    for k, f in enumerate(_frames(truth)):
        tracker.update(f)
        if k >= 20:
            tr = next(iter(tracker.tracks.values()))
            d = tr.x[:2] - truth[k]
            nees.append(float(d @ np.linalg.solve(tr.P[:2, :2], d)))
    assert len(tracker.tracks) == 1
    tr = next(iter(tracker.tracks.values()))
    assert len(tr.fixes) >= 0.98 * n
    assert 1.0 <= np.mean(nees) <= 2.6, np.mean(nees)            # 2 when consistent; below when the process noise exceeds a straight path
    assert np.mean(np.array(nees) <= -2 * np.log(0.05)) >= 0.9                     # 95 % ellipse holds the truth
    np.testing.assert_allclose(tr.x[2:], [3.0, -1.0], atol=0.3)


def test_turning_track_is_kept():
    """A UUV on a circle of 40 m at 1.5 m/s (0.06 m/s^2): one track through the turn with the echo tracks' process
    noise (0.2 m/s^2)."""
    cfg = TrackerConfig(accel=0.2, meas_floor=SIGMA, confirm=4)
    n = 500
    w = 1.5 / 40.0
    t = 0.25 * np.arange(n)
    path = np.column_stack([200 + 40 * np.cos(w * t), 200 + 40 * np.sin(w * t)])
    tracker = RadiatorTracker(cfg)
    for f in _frames(path, seed=3):
        tracker.update(f)
    conf = tracker.confirmed()
    assert len(conf) == 1 and not tracker.lost
    assert len(next(iter(conf.values())).fixes) >= 0.97 * n


def test_static_decision():
    """A stationary body becomes static after ``static_s``; one moving at 1 m/s does not."""
    cfg = TrackerConfig(meas_floor=SIGMA, confirm=4, static_s=10.0)
    for speed, static in ((0.0, True), (1.0, False)):
        tracker = RadiatorTracker(cfg)
        t = 0.25 * np.arange(80)
        for f in _frames(np.column_stack([100 + speed * t, 100 + 0 * t]), seed=5):
            tracker.update(f)
        tr = next(iter(tracker.tracks.values()))
        assert tr.static is static, (speed, tr.x)


def _gauss(grid, c, w):
    return np.exp(-0.5 * np.sum((grid - c) ** 2, axis=1) / w ** 2)


def test_sidelobe_of_a_confirmed_track_is_explained_by_its_predicted_map():
    """Two sources, A strong and B weak, and A's sidelobe 25 m from A: with A's predicted map (its main lobe and the
    sidelobe at half its peak) the sidelobe starts no track, while B, which A's map does not reach, is confirmed."""
    x, y = np.meshgrid(np.arange(0, 200, 2.0), np.arange(0, 200, 2.0))
    grid = np.column_stack([x.ravel(), y.ravel()])
    a, b, side = np.array([60.0, 100.0]), np.array([150.0, 60.0]), np.array([60.0, 125.0])
    cfg = TrackerConfig(meas_floor=1.0, confirm=4, explain_fraction=0.5, shape_radius=10.0)
    tracker = RadiatorTracker(cfg, grid=grid)
    rng = np.random.default_rng(2)
    for k in range(40):
        t = 0.25 * k
        pat_a = _gauss(grid, a, 3.0) + 0.6 * _gauss(grid, side, 3.0)                    # A's map alone, 1 at A
        B = 100.0 * pat_a + 30.0 * _gauss(grid, b, 3.0)
        pts = np.array([a, side, b]) + 0.3 * rng.standard_normal((3, 2))
        conf = {kk: tr for kk, tr in tracker.tracks.items() if tr.confirmed}
        pats = {kk: pat_a for kk in conf}
        cells = grid[[int(np.argmin(np.hypot(*(grid - p).T))) for p in (a, side, b)]]
        R = np.array([[1.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]])               # the array's response between them
        tracker.update(FrameResult(t, pts, np.array([100.0, 60.0, 30.0]), {}, 0.0, 0.0, B=B, patterns=pats,
                                   pair_cells=cells, pair_R=R))
    conf = tracker.confirmed()
    ends = sorted(np.round(tr.x[:2]).tolist() for tr in conf.values())
    assert len(conf) == 2, ends
    assert any(np.hypot(*(np.array(e) - a)) < 3 for e in ends) and any(np.hypot(*(np.array(e) - b)) < 3 for e in ends)
    assert any(np.hypot(*(xy - side)) < 3 for _, xy, _ in tracker.explained)


def test_fix_covariance_from_the_half_maximum_widths():
    """A peak 24 m long north to south and 4 m wide: the covariance's major axis is north, its standard deviations
    the widths over 2.355 (with the floor added)."""
    cfg = TrackerConfig(meas_scale=1.0, meas_floor=0.5)
    R = fix_covariance([24.0, 4.0], 0.0, cfg)
    np.testing.assert_allclose(R, np.diag([(4 / 2.355) ** 2 + 0.25, (24 / 2.355) ** 2 + 0.25]), atol=1e-9)
    e = ellipse95(np.diag([1.0, 4.0]))
    assert abs(e["major_deg"]) < 1e-6 or abs(e["major_deg"] - 180) < 1e-6
    np.testing.assert_allclose([e["major_m"], e["minor_m"]], [2 * np.sqrt(5.991465) * 2, 2 * np.sqrt(5.991465)], rtol=1e-5)


def test_echo_coherence():
    """An echo (the illuminator's spectra delayed and scaled, plus a little noise) is coherent with it; an independent
    source is not."""
    rng = np.random.default_rng(4)
    K, F = 8, 120
    s = rng.standard_normal((K, F)) + 1j * rng.standard_normal((K, F))
    delay = np.exp(-2j * np.pi * np.linspace(100, 700, F) * 0.013)
    echo = 0.1 * s * delay + 0.01 * (rng.standard_normal((K, F)) + 1j * rng.standard_normal((K, F)))
    other = rng.standard_normal((K, F)) + 1j * rng.standard_normal((K, F))
    assert spectral_coherence(echo, s) > 0.95
    assert spectral_coherence(other, s) < 0.3


def _truth_with_hull():
    t = np.array([0.0, 20.0])
    hull = Track("moored", "ship", False, t, np.array([[250.0, 260.0], [250.0, 260.0]]), np.zeros(2), np.zeros(2, bool), 0.0)
    uuv = Track("uuv", "uuv", False, t, np.array([[100.0, 100.0], [120.0, 100.0]]), np.ones(2), np.ones(2, bool), 0.0)
    scenario = {"environment": {"dx": 0.5},
                "vessels": [{"id": "moored", "waypoints": [[250.0, 260.0]], "body_length": 30.0, "body_width": 6.0, "body_shape": "rect"},
                            {"id": "uuv", "waypoints": [[100.0, 100.0], [120.0, 100.0]], "body_length": 3.0, "body_width": 1.0}]}
    return Truth(tracks={"moored": hull, "uuv": uuv}, scenario=scenario)


def test_outline_scoring_of_a_static_body():
    """A track 2 m off the west end of a 30 m hull is 2 m from its outline and 17 m from its centre: matched to the
    hull within 10 m; a moving body keeps its centre."""
    truth = _truth_with_hull()
    cells = static_body_cells(truth, "moored")
    assert cells is not None and np.ptp(cells[:, 0]) == pytest.approx(30.0, abs=1.0) and np.ptp(cells[:, 1]) == pytest.approx(6.0, abs=1.0)
    assert static_body_cells(truth, "uuv") is None
    dist = ObjectDistance(truth)
    p = np.array([233.0, 260.0])
    assert dist("moored", 5.0, p) == pytest.approx(2.0, abs=0.6)
    assert dist("moored", 5.0, [250.0, 260.0]) == 0.0
    assert dist("uuv", 10.0, [110.0, 103.0]) == pytest.approx(3.0, abs=1e-9)
    m = match_tracks({"e1": [(t, p) for t in (1.0, 2.0, 3.0)]}, truth, 10.0)[0]
    assert m["true_object"] == "moored" and m["median_error_m"] < 3.0

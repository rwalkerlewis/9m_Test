"""M2: vessel tracks and AIS."""

import numpy as np
import pytest

from acoustic_sim.harbor import (
    AISConfig,
    AISTrack,
    VesselConfig,
    build_synthetic_harbor,
    build_track,
    default_vessels,
    generate_ais,
    uniform_water_environment,
)


@pytest.fixture(scope="module")
def harbor():
    return build_synthetic_harbor()


def test_default_vessels_stay_in_water(harbor):
    for v in default_vessels():
        track = build_track(v, 120.0, harbor)
        assert track.xy.shape == (241, 2)
        assert harbor.is_water(track.xy[track.active], margin=15.0).all()


def test_constant_speed_and_endpoints():
    env = uniform_water_environment(1000, 750, 5.0)
    v = VesselConfig("a", "ship", [[100, 300], [500, 300]], speed=5.0)
    tr = build_track(v, 60.0, env)
    assert np.allclose(tr.xy[0], [100, 300], atol=1.0)
    assert np.allclose(tr.speed[10:-10], 5.0, atol=0.05)
    # 60 s * 5 m/s = 300 m travelled
    assert tr.xy[-1, 0] == pytest.approx(400.0, abs=3.0)


def test_stationary_vessel_and_activity_window():
    env = uniform_water_environment(1000, 750, 5.0)
    moored = build_track(VesselConfig("m", "ship", [[400, 400]]), 20.0, env)
    assert np.all(moored.speed == 0.0) and np.all(moored.xy == [400, 400])
    v = VesselConfig("a", "uuv", [[100, 300], [500, 300]], speed=4.0, t_start=10.0, t_end=30.0)
    tr = build_track(v, 40.0, env)
    assert not tr.active[0] and tr.active[30] and not tr.active[-1]
    assert np.all(tr.speed[~tr.active] == 0.0)
    assert np.allclose(tr.xy[0], [100, 300], atol=0.01)  # waits at the start until t_start
    assert not tr.ais


def test_wander_is_bounded_smooth_and_reproducible():
    env = uniform_water_environment(1000, 750, 5.0)
    kw = dict(waypoints=[[100, 375], [900, 375]], speed=3.0, wander_sigma=10.0, seed=3)
    a = build_track(VesselConfig("w", "uuv", **kw), 200.0, env)
    b = build_track(VesselConfig("w", "uuv", **kw), 200.0, env)
    assert np.array_equal(a.xy, b.xy)
    lateral = a.xy[:, 1] - 375.0
    assert 2.0 < lateral.std() < 25.0
    assert a.speed[a.active].max() < 4.5  # smooth: no spurious speed from jitter


def test_track_leaving_water_is_rejected(harbor):
    bad = VesselConfig("bad", "ship", [[1000, 1000], [1000, 1600]], speed=5.0)
    with pytest.raises(ValueError, match="leaves the water"):
        build_track(bad, 120.0, harbor)


def test_vessel_config_validation():
    with pytest.raises(ValueError):
        VesselConfig("x", "whale", [[0, 0]])
    with pytest.raises(ValueError):
        VesselConfig("x", "ship", [[0, 0], [10, 0]], speed=0.0)


# --------------------------- AIS ---------------------------------

def _tracks():
    env = uniform_water_environment(1000, 750, 5.0)
    return {
        "ship": build_track(VesselConfig("ship", "ship", [[100, 300], [900, 300]], speed=5.0), 120.0, env),
        "moored": build_track(VesselConfig("moored", "ship", [[500, 500]]), 120.0, env),
        "uuv": build_track(VesselConfig("uuv", "uuv", [[100, 600], [900, 600]], speed=2.0), 120.0, env),
    }


def test_ais_only_for_ais_vessels_and_rates():
    cfg = AISConfig(dropout_prob=0.0, outage_prob=0.0)
    ais = generate_ais(_tracks(), cfg, np.random.default_rng(0))
    assert ais.vessels == ["moored", "ship"]
    n_ship = len(ais.for_vessel("ship"))
    n_moored = len(ais.for_vessel("moored"))
    assert abs(n_ship - 120 / 5.0) <= 2
    assert abs(n_moored - 120 / 20.0) <= 2


def test_ais_position_error_statistics():
    cfg = AISConfig(dropout_prob=0.0, outage_prob=0.0, pos_sigma=8.0)
    tracks = _tracks()
    err = []
    for seed in range(20):
        ais = generate_ais(tracks, cfg, np.random.default_rng(seed)).for_vessel("ship")
        truth = tracks["ship"].position_at(ais.t)
        err.append(ais.xy - truth)
    err = np.concatenate(err)
    assert err.std() == pytest.approx(8.0, rel=0.15)
    assert abs(err.mean()) < 1.0


def test_ais_dropouts_and_outages_reduce_reports():
    tracks = _tracks()
    full = generate_ais(tracks, AISConfig(dropout_prob=0.0, outage_prob=0.0), np.random.default_rng(1))
    lossy = generate_ais(tracks, AISConfig(dropout_prob=0.4, outage_prob=0.05), np.random.default_rng(1))
    assert len(lossy.for_vessel("ship")) < 0.8 * len(full.for_vessel("ship"))


def test_ais_predict_interpolation_and_causal_dead_reckoning():
    cfg = AISConfig(dropout_prob=0.0, outage_prob=0.0, pos_sigma=2.0)
    tracks = _tracks()
    ais = generate_ais(tracks, cfg, np.random.default_rng(2)).for_vessel("ship")
    t = np.linspace(ais.t[1], ais.t[-1], 200)
    truth = tracks["ship"].position_at(t)
    assert np.hypot(*(ais.predict(t) - truth).T).max() < 12.0
    causal = ais.predict(t, causal=True)
    assert np.hypot(*(causal - truth).T).max() < 40.0  # within one 5 s report interval
    assert np.all(ais.age(t) <= 6.0) and np.isinf(ais.age(-1.0)[0])


def test_ais_predict_extrapolates_beyond_last_report():
    ais = AISTrack("x", t=np.array([0.0, 5.0]), xy=np.array([[0.0, 0.0], [25.0, 0.0]]),
                   sog=np.array([5.0, 5.0]), cog=np.array([0.0, 0.0]))
    assert np.allclose(ais.predict(2.5), [[12.5, 0.0]])             # interpolation
    assert np.allclose(ais.predict(9.0), [[45.0, 0.0]])             # dead reckoning past the last report
    assert np.allclose(ais.predict(3.0, causal=True), [[15.0, 0.0]])  # from the latest earlier report only
    assert np.allclose(ais.predict(-2.0, causal=True), [[-10.0, 0.0]])
    assert ais.age(7.0)[0] == pytest.approx(2.0)

    turning = AISTrack("y", t=np.array([0.0]), xy=np.array([[0.0, 0.0]]), sog=np.array([2.0]), cog=np.array([np.pi / 2]))
    assert np.allclose(turning.predict(10.0), [[0.0, 20.0]], atol=1e-9)

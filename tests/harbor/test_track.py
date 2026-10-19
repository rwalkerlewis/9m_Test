"""Constant velocity Kalman smoother on a synthetic straight track with known noise."""

from __future__ import annotations

import numpy as np

from acoustic_sim.harbor.ais import AISReports
from acoustic_sim.harbor.track import TrackConfig, kalman_track


def _straight(v=(4.0, 0.0), x0=(100.0, 200.0), T=20.0, hop=0.25):
    t = np.arange(0.0, T + 1e-9, hop)
    xy = np.asarray(x0)[None, :] + np.outer(t, v)
    return t, xy


def _ais(t_rep, truth_t, truth_xy, v, rng, pos_sigma=8.0, sog_sigma=0.2, cog_sigma=0.035):
    xy = np.column_stack([np.interp(t_rep, truth_t, truth_xy[:, 0]), np.interp(t_rep, truth_t, truth_xy[:, 1])])
    xy = xy + pos_sigma * rng.standard_normal(xy.shape)
    sog = np.hypot(*v) + sog_sigma * rng.standard_normal(len(t_rep))
    cog = np.arctan2(v[1], v[0]) + cog_sigma * rng.standard_normal(len(t_rep))
    return AISReports(vessel=np.array(["ship"] * len(t_rep)), t=t_rep, xy=xy, sog=sog, cog=cog)


def test_smoother_beats_the_raw_refinement_on_a_straight_track():
    rng = np.random.default_rng(1)
    v = (4.0, 0.0)
    t, truth = _straight(v)
    refined = truth + 1.0 * rng.standard_normal(truth.shape)
    ais = _ais(np.arange(0.0, 20.0, 5.0), t, truth, v, rng)
    fit = kalman_track(t, refined, np.ones(len(t), bool), ais, "ship", TrackConfig(accel=0.05, meas_sigma=1.0))
    err_raw = np.hypot(*(refined - truth).T)
    err_fit = np.hypot(*(fit.xy - truth).T)
    assert np.sqrt(np.mean(err_fit ** 2)) < 0.4 * np.sqrt(np.mean(err_raw ** 2)), (err_fit.mean(), err_raw.mean())
    assert np.all(np.abs(fit.speed - 4.0) < 0.3), fit.speed
    assert fit.used.all()
    # The posterior sigma is consistent with the realised error (within a factor 2 on the rms).
    assert 0.5 < np.sqrt(np.mean(err_fit ** 2)) / np.sqrt(np.mean(fit.sigma_xy ** 2) * 2) < 2.0


def test_outlier_refinement_is_gated_and_ais_alone_works():
    rng = np.random.default_rng(2)
    v = (-3.0, 2.0)
    t, truth = _straight(v, x0=(250.0, 100.0))
    refined = truth + 1.0 * rng.standard_normal(truth.shape)
    refined[30] += np.array([40.0, -30.0])                      # one refinement jumped to a sidelobe
    ais = _ais(np.arange(1.0, 20.0, 5.0), t, truth, v, rng)
    fit = kalman_track(t, refined, np.ones(len(t), bool), ais, "ship")
    assert not fit.used[30] and fit.used.sum() == len(t) - 1
    assert np.hypot(*(fit.xy[30] - truth[30])) < 2.0
    only_ais = kalman_track(t, None, None, ais, "ship")
    err = np.hypot(*(only_ais.xy - truth).T)
    assert np.median(err) < 8.0 and np.all(np.abs(only_ais.speed - np.hypot(*v)) < 0.5)


def test_stationary_ship_has_near_zero_speed():
    rng = np.random.default_rng(3)
    t, truth = _straight((0.0, 0.0), x0=(150.0, 150.0))
    refined = truth + 1.0 * rng.standard_normal(truth.shape)
    ais = _ais(np.array([0.0]), t, truth, (0.0, 0.0), rng)
    fit = kalman_track(t, refined, np.ones(len(t), bool), ais, "ship")
    assert np.all(fit.speed < 0.1), fit.speed.max()
    assert np.all(np.hypot(*(fit.xy - truth).T) < 0.6)


def test_output_times_and_absent_frames():
    rng = np.random.default_rng(4)
    v = (2.0, 1.0)
    t, truth = _straight(v)
    present = np.ones(len(t), bool)
    present[20:40] = False                                      # the ship is not refined in these frames
    refined = truth + rng.standard_normal(truth.shape)
    ais = _ais(np.arange(0.0, 20.0, 5.0), t, truth, v, rng)
    out_t = np.linspace(0.0, 20.0, 7)
    fit = kalman_track(t, refined, present, ais, "ship", out_times=out_t)
    assert fit.xy.shape == (7, 2) and np.allclose(fit.t, out_t)
    truth_out = np.asarray((100.0, 200.0))[None, :] + np.outer(out_t, v)
    assert np.all(np.hypot(*(fit.xy - truth_out).T) < 2.0)

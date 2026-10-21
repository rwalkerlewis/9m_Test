"""The scoreboard on synthetic maps whose peaks are placed by hand (``water/scoreboard.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_sim.contract import Provenance, Track, Truth
from acoustic_sim.water.scoreboard import ChainMaps, local_maxima, score, set_threshold

K = 20
TIMES = np.arange(K) * 0.5
XS = np.arange(0.0, 101.0, 1.0)
GX, GY = np.meshgrid(XS, XS)
GRID = np.column_stack([GX.ravel(), GY.ravel()])


def _cell(x, y):
    return int(np.flatnonzero((GRID[:, 0] == x) & (GRID[:, 1] == y))[0])


def _track(oid, kind, xy0, xy1, level=0.0, ais=False):
    t = np.linspace(0.0, TIMES[-1], 50)
    xy = np.column_stack([np.linspace(xy0[0], xy1[0], 50), np.linspace(xy0[1], xy1[1], 50)])
    speed = np.full(50, np.hypot(*(np.subtract(xy1, xy0))) / t[-1])
    return Track(oid, kind, ais, t, xy, speed, np.ones(50, bool), level_pa=level)


@pytest.fixture()
def truth():
    return Truth(tracks={
        "static": _track("static", "static", (30, 30), (30, 30)),
        "mover": _track("mover", "uuv", (60, 20), (60, 80)),
        "ship": _track("ship", "ship", (90, 10), (90, 10), level=1.0, ais=True),
    })


def _exclusion():
    ex = np.zeros((len(GRID), K), bool)
    ex[np.hypot(*(GRID - [90, 10]).T) < 8.0] = True
    return ex


def test_local_maxima_on_a_masked_grid():
    keep = np.hypot(*(GRID - 50).T) < 40                      # a round harbor: cells outside are land
    g = GRID[keep]
    v = np.zeros(len(g))
    i, j = np.flatnonzero((g == [50, 50]).all(1))[0], np.flatnonzero((g == [51, 50]).all(1))[0]
    v[i], v[j] = 3.0, 2.0
    m = local_maxima(v, g)
    assert m[i] and not m[j]


def test_threshold_at_the_five_percent_quantile(truth):
    B = np.zeros((len(GRID), K))
    for k in range(K):
        B[_cell(80, 80), k] = k + 1.0                         # the frame statistic is k + 1
        B[_cell(90, 12), k] = 100.0                           # inside the ship exclusion: ignored
    free = ChainMaps(GRID, TIMES, B, _exclusion(), removed=("static", "mover"), label="free")
    thr = set_threshold([free], truth)
    assert thr.T == 19.0                                      # exceeded in 1 of 20 frames (5 %)
    assert np.sum(thr.frame_maxima > thr.T) == 1
    assert thr.T_full == pytest.approx(np.sum(np.arange(1, K + 1)))

    # A peak near an object still in the scene is not part of the object free statistic.
    B2 = B.copy()
    B2[_cell(31, 30), :] = 50.0
    free2 = ChainMaps(GRID, TIMES, B2, _exclusion(), removed=("mover",), label="free2")
    assert set_threshold([free2], truth).T == 19.0


def test_pd_false_alarms_and_errors_by_construction(truth):
    T = 19.0
    free = np.zeros((len(GRID), K))
    free[_cell(80, 80), :] = np.arange(1, K + 1)
    thr = set_threshold([ChainMaps(GRID, TIMES, free, _exclusion(), removed=("static", "mover"))], truth)
    assert thr.T == T

    B = np.zeros((len(GRID), K))
    mover = truth.tracks["mover"].position_at(TIMES)
    for k in range(K):
        if k < 10:
            B[_cell(31, 30), k] = 50.0                         # static: detected in half the frames, 1 m off
        x, y = np.rint(mover[k]).astype(int)
        B[_cell(x, y + 2), k] = 10.0 if k == 3 else 30.0        # moving: 2 m off, below T in frame 3
        if k < 5:
            B[_cell(10, 90), k] = 25.0                         # far from every object: a false alarm
        B[_cell(91, 10), k] = 80.0                             # inside the ship exclusion: nothing
    B[_cell(38, 30), 5] = 25.0                                 # 8 m from the static body: neither
    run = ChainMaps(GRID, TIMES, B, _exclusion(), Provenance(held_out=True), label="full")
    sb = score({"static": run, "mover": run}, truth, thr)
    st, mv = sb.get("static"), sb.get("mover")
    assert (st.frames, st.detected, st.pd) == (K, 10, 0.5)
    assert st.median_error_m == pytest.approx(1.0) and not st.moving
    assert st.full_record_detected == 1.0 and st.full_record_error_m == pytest.approx(1.0)      # 500 against T_full 210
    assert (mv.detected, mv.pd) == (K - 1, (K - 1) / K)
    assert mv.median_error_m == pytest.approx(2.0, abs=0.75) and mv.moving
    assert sb.false_alarms == 5 and sb.false_alarm_rate == pytest.approx(5 / K)
    d = sb.to_dict()
    assert d["provenance"]["held_out"] and d["threshold"]["T"] == T


def test_one_peak_cannot_detect_two_pilings():
    tracks = {f"p{i}": _track(f"p{i}", "static", (40 + 5 * i, 50), (40 + 5 * i, 50)) for i in range(2)}
    truth = Truth(tracks=tracks)
    B = np.zeros((len(GRID), K))
    B[_cell(42, 50), :] = 10.0                                 # 2 m from p0, 3 m from p1
    run = ChainMaps(GRID, TIMES, B)
    noise = np.zeros_like(B)
    noise[_cell(90, 90), :] = 1.0                              # the object free background: T = 1
    free = ChainMaps(GRID, TIMES, noise, removed=("p0", "p1"))
    sb = score({"p0": run, "p1": run}, truth, set_threshold([free], truth), pilings=["p0", "p1"])
    assert sb.get("p0").pd == 1.0 and sb.get("p1").pd == 0.0
    assert sb.separation == {"pilings": 2, "resolved_per_run": [1], "resolved_median": 1.0}


def test_runs_of_several_seeds_are_pooled(truth):
    """Two realisations: Pd and the false alarm rate are pooled over the frames of both."""
    free = np.zeros((len(GRID), K))
    free[_cell(80, 80), :] = np.arange(1, K + 1)
    thr = set_threshold([ChainMaps(GRID, TIMES, free, _exclusion(), removed=("static", "mover"))], truth)
    runs = []
    for n_hit in (K, 0):
        B = np.zeros((len(GRID), K))
        B[_cell(30, 30), :n_hit] = 50.0
        B[_cell(10, 90), 0] = 30.0                              # one false alarm per run
        runs.append(ChainMaps(GRID, TIMES, B, _exclusion()))
    sb = score({"static": runs}, [truth, truth], thr)
    st = sb.get("static")
    assert (st.frames, st.detected, st.pd) == (2 * K, K, 0.5)
    assert st.full_record_hits == [True, False] and st.full_record_detected == 0.5
    assert sb.false_alarms == 2 and sb.n_frames == 2 * K


# ----------------------------- calibration -----------------------------

N_CAL = 2000                                                  # frames per run
G_CAL = np.column_stack([a.ravel() for a in np.meshgrid(np.arange(40.0), np.arange(40.0))])
OBJ_CAL = {"a": (10.0, 10.0), "b": (30.0, 28.0)}


def _cal_truth():
    t = np.linspace(0.0, N_CAL * 0.25, 50)
    return Truth(tracks={k: Track(k, "static", False, t, np.tile(xy, (50, 1)), np.zeros(50), np.ones(50, bool))
                         for k, xy in OBJ_CAL.items()})


def _cal_run(seed, present, normalise=False):
    """Exp(1) noise on every cell and frame plus a bump of 20 at each object in *present*.  ``normalise``: the map
    is ``exp(B / 2)`` renormalised to sum 1 every frame and times the number of cells, as the occupancy posterior."""
    rng = np.random.default_rng(seed)
    B = rng.exponential(1.0, (len(G_CAL), N_CAL))
    for k in present:
        B[np.hypot(*(G_CAL - OBJ_CAL[k]).T) <= 1.0] += 20.0
    if normalise:
        E = np.exp(0.5 * B)
        B = E / E.sum(axis=0) * len(G_CAL)
    removed = tuple(k for k in OBJ_CAL if k not in present)
    return ChainMaps(G_CAL, np.arange(N_CAL) * 0.25, B, removed=removed, label=f"seed{seed}:{'+'.join(present)}")


def test_threshold_holds_its_false_alarm_rate_on_an_independent_noise_run_of_the_scene():
    """T set on the one object removed runs of a scene (noise seeds 1, 2); on the full scene with an independent
    noise realisation (seed 3) the fraction of frames with a false alarm is the 5 % target within 0.02 (the binomial
    standard error over 2000 frames is 0.005, that of the threshold quantile over 4000 frames about 0.004)."""
    truth = _cal_truth()
    thr = set_threshold([_cal_run(1, ("b",)), _cal_run(2, ("a",))], truth)
    full = _cal_run(3, ("a", "b"))
    sb = score({"a": full, "b": full}, truth, thr)
    assert abs(sb.false_alarm_frame_rate - 0.05) <= 0.02, sb.false_alarm_frame_rate
    assert sb.get("a").pd == 1.0 and sb.get("b").pd == 1.0


def test_a_map_normalised_every_frame_does_not_hold_its_false_alarm_rate():
    """The same scene through a per frame normalisation (the occupancy posterior): the second object takes mass
    from the background, so on the full scene far fewer frames than the 5 % set on the object free runs have a
    false alarm.  The scoreboard reports the rate it measures, not the target."""
    truth = _cal_truth()
    thr = set_threshold([_cal_run(1, ("b",), True), _cal_run(2, ("a",), True)], truth)
    full = _cal_run(3, ("a", "b"), True)
    sb = score({"a": full, "b": full}, truth, thr)
    assert sb.false_alarm_frame_rate < 0.02, sb.false_alarm_frame_rate
    assert sb.to_dict()["false_alarm_frame_rate"] == sb.false_alarm_frame_rate


def test_a_flat_maximum_is_one_peak_and_one_false_alarm(truth):
    """A plateau of 3 x 3 equal cells is one local maximum, so one flat false peak is one false alarm."""
    B = np.zeros((len(GRID), K))
    plateau = (np.abs(GRID[:, 0] - 80) <= 1) & (np.abs(GRID[:, 1] - 80) <= 1)
    B[plateau, :] = 5.0
    m = local_maxima(B[:, 0], GRID)
    assert m[plateau].sum() == 1
    free = np.zeros((len(GRID), K))
    free[_cell(70, 70), :] = np.arange(1, K + 1)
    thr = set_threshold([ChainMaps(GRID, TIMES, free, _exclusion(), removed=("static", "mover"))], truth)
    assert thr.T == 19.0
    B[_cell(70, 70), :] = 1.0                                 # a peak below T keeps the zero background from being a maximum there
    sb = score({"static": ChainMaps(GRID, TIMES, B * 5.0, _exclusion())}, truth, thr)
    assert sb.false_alarms == K and sb.false_alarm_frame_rate == 1.0


def test_threshold_refuses_object_free_runs_without_a_positive_peak(truth):
    empty = ChainMaps(GRID, TIMES, np.zeros((len(GRID), K)), _exclusion(), removed=("static", "mover"))
    with pytest.raises(ValueError, match="no noise statistic"):
        set_threshold([empty], truth)
    land = ChainMaps(GRID, TIMES, np.full((len(GRID), K), -np.inf), _exclusion(), removed=("static", "mover"))
    with pytest.raises(ValueError, match="no noise statistic"):
        set_threshold([land], truth)

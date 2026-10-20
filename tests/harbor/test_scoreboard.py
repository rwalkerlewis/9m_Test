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
    free = ChainMaps(GRID, TIMES, np.zeros_like(B), removed=("p0", "p1"))
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

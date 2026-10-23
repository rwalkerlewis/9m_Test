"""The last heard state (``TrackerConfig.last_heard_s``): a confirmed track that loses its fixes is kept at its last
filtered position with a growing covariance, comes back to confirmed on a fix in its gate, and is dropped after the hold
time; the scoring counts its frames as hits only when the object is inside its 95 % ellipse."""

from __future__ import annotations

import numpy as np

from acoustic_sim.water.streaming import FrameResult, RadiatorTracker, TrackerConfig

BASE = dict(meas_floor=1.0, confirm=6, confirm_window_s=2.0, drop_s=2.0, accel=0.05)


def _run(cfg, seen, pos, t_end=40.0):
    trk = RadiatorTracker(cfg)
    states = []
    for t in np.round(np.arange(0.25, t_end, 0.25), 2):
        xy = [pos(t)] if t in seen else []
        trk.update(FrameResult(float(t), np.array(xy).reshape(-1, 2), np.ones(len(xy)), {}, 0.0, 0.0))
        tr = trk.tracks.get("r1")
        states.append((float(t), None if tr is None else ("last_heard" if tr.last_heard else "confirmed" if tr.confirmed else
                                                          "tentative"), tr))
    return trk, states


def test_a_track_that_loses_its_fixes_is_kept_then_dropped():
    """A ship moving east at 2 m/s heard until 10 s: with ``last_heard_s = 8`` its track is last heard from 12 s (2 s
    without a fix) to 18 s, at its last filtered position (not coasting on), the covariance growing; then it is lost."""
    seen = set(np.round(np.arange(0.25, 10.01, 0.25), 2))
    pos = lambda t: np.array([100.0 + 2.0 * t, 50.0])  # noqa: E731
    trk, states = _run(TrackerConfig(last_heard_s=8.0, **BASE), seen, pos, 20.0)
    st = {t: s for t, s, _ in states}
    assert st[10.0] == "confirmed" and st[12.25] == "last_heard" and st[17.75] == "last_heard"
    assert st[18.25] is None and "r1" in trk.lost
    held = [tr for t, s, tr in states if s == "last_heard"]
    xy0 = held[0].predict(12.5)
    assert np.allclose(held[0].predict(17.5), xy0)                  # stays where it was last heard
    assert abs(xy0[0] - pos(10.0)[0]) < 1.0
    p1, p2 = held[0].predicted(12.5)[1][:2, :2], held[0].predicted(17.5)[1][:2, :2]
    assert np.trace(p2) > np.trace(p1)                              # the covariance grows


def test_off_by_default_and_heard_again():
    """Off (``last_heard_s = 0``) the track is dropped after 2 s as before; on, a fix in its gate at 16 s makes it
    confirmed again (the same track, no new one)."""
    seen = set(np.round(np.arange(0.25, 10.01, 0.25), 2)) | {16.0, 16.25}
    pos = lambda t: np.array([120.0, 50.0])  # noqa: E731     a ship that stopped and went quiet
    trk, states = _run(TrackerConfig(**BASE), seen, pos, 17.0)
    assert {t: s for t, s, _ in states}[12.25] is None
    trk, states = _run(TrackerConfig(last_heard_s=30.0, **BASE), seen, pos, 17.0)
    st = {t: s for t, s, _ in states}
    assert st[15.75] == "last_heard" and st[16.0] == "confirmed"
    assert set(trk.tracks) == {"r1"}


def test_scoring_counts_last_heard_inside_the_ellipse_only():
    """Two frames held by a last heard track: the object inside its ellipse in one, outside in the other; Pd without
    last heard 1 of 3, with 2 of 3."""
    from acoustic_sim.contract import Track, Truth
    from acoustic_sim.water.scoreboard import score_picture

    t = np.array([0.0, 10.0])
    truth = Truth(tracks={"boat": Track(id="boat", kind="ship", ais=False, t=t, xy=np.array([[0.0, 0.0], [0.0, 0.0]]),
                                        speed=np.zeros(2), active=np.ones(2, bool), level_pa=1.0, meta={})})

    def rec(tt, status, xy, cov):
        st = {"status": status, "xy": xy, "fix_t": 0.0, "fix_xy": [0.0, 0.0], "detected": status == "confirmed", "cov": cov}
        return {"t": tt, "chains": {"passive": True}, "objects": [{"id": "r1", "tracks": {"radiator": "r1"},
                                                                   "track_state": {"r1": st}, "status": status}]}
    records = [rec(1.0, "confirmed", [0.5, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
               rec(2.0, "last_heard", [3.0, 0.0], [[25.0, 0.0], [0.0, 25.0]]),
               rec(3.0, "last_heard", [30.0, 0.0], [[25.0, 0.0], [0.0, 25.0]])]
    sc = score_picture(records, truth, 3.0, {"radiator": 1})
    o = sc["objects"]["boat"]
    assert abs(o["pd"]["radiator"] - 1 / 3) < 1e-9
    assert abs(o["pd_last_heard"]["radiator"] - 2 / 3) < 1e-9
    assert sc["false_tracks"]["count"] == 0

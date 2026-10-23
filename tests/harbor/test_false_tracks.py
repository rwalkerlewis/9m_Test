"""False tracks (prompt ``false_tracks``): the classification of the peaks that start radiator tracks (a diagnostic of
``harbor_false_tracks.py``), the echo test by the predicted map ratio (``TrackerConfig.echo_test = "ratio"``) and the
spectral coherence it replaced, and a confirmed track that keeps a ship which slows to a stop and is then seen only
now and then (``accel``, ``drop_confirmed_s``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from acoustic_sim.water.stream import spectral_coherence
from acoustic_sim.water.streaming import FrameResult, RadiatorTracker, TrackerConfig

ROOT = Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location("harbor_false_tracks",
                                                  ROOT / "examples" / "harbor" / "05_detection" / "harbor_false_tracks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_peak_classes_on_a_known_scene():
    """A radiator at (100, 100), a silent body at (200, 100), a ship at (100, 300) whose nearest channel is (100, 200)."""
    m = _script()
    rad, sil = np.array([100.0, 100.0]), np.array([200.0, 100.0])
    seg = {"ship": (np.array([100.0, 300.0]), np.array([100.0, 200.0]))}

    def cls(xy, ratio=None):
        xy = np.asarray(xy, float)
        return m.classify(xy, ratio, {"ship_a": float(np.hypot(*(xy - rad)))}, {"body": float(np.hypot(*(xy - sil)))}, seg)[0]

    assert cls([102.0, 101.0]) == "radiator"
    assert cls([203.0, 98.0]) == "silent_body"
    assert cls([300.0, 300.0], ratio=0.4) == "sidelobe_near_miss"
    assert cls([300.0, 300.0], ratio=0.6) == "elsewhere"                # explained already: not a near miss
    assert cls([103.0, 250.0]) == "direct_path"
    assert cls([300.0, 300.0]) == "elsewhere"
    assert cls([102.0, 101.0], ratio=0.4) == "radiator"                 # the first class that applies


def _grid():
    x = np.arange(0.0, 101.0, 2.0)
    gx, gy = np.meshgrid(x, x)
    return np.column_stack([gx.ravel(), gy.ravel()])


def test_echo_ratio_from_the_predicted_map():
    """A confirmed track at (20, 20) with its peak 100 in the data map; a peak of 10 at (80, 80) where the track's
    predicted map is 0.02 of its own peak: the ratio is 0.02 * 100 / 10 = 0.2, recorded at each fix with
    ``echo_test = "ratio"``."""
    grid = _grid()
    B = np.full(len(grid), 1.0)
    src, at = np.array([20.0, 20.0]), np.array([80.0, 80.0])
    B[np.argmin(np.hypot(*(grid - src).T))] = 100.0
    B[np.argmin(np.hypot(*(grid - at).T))] = 10.0
    cells = np.array([src, at])
    R = np.array([[1.0, 0.02], [0.02, 1.0]])                            # response at row cell to a source at column cell
    cfg = TrackerConfig(confirm=2, confirm_window_s=1.0, echo_test="ratio", explain_fraction=0.5, meas_floor=1.0)
    trk = RadiatorTracker(cfg, grid=grid)
    for k in range(4):
        t = 0.25 * k
        f = FrameResult(t, np.array([src, at]), np.array([100.0, 10.0]), {}, 0.0, 0.0, B=B, pair_cells=cells, pair_R=R)
        trk.update(f)
    big = next(tr for tr in trk.tracks.values() if np.allclose(tr.fixes[-1][1], src))
    small = next(tr for tr in trk.tracks.values() if np.allclose(tr.fixes[-1][1], at))
    assert big.confirmed
    assert abs(trk.ratio(f, big, at, 10.0, f.t) - 0.2) < 1e-12
    assert small.ratios[big.id] and np.allclose(small.ratios[big.id], 0.2)


def test_spectral_coherence_of_an_echo_and_of_an_independent_source():
    """An echo (a fixed complex gain per frequency) is coherent with its illuminator; an independent source sits near
    1 / K for K frames."""
    rng = np.random.default_rng(0)
    K, F = 8, 200
    il = rng.standard_normal((K, F)) + 1j * rng.standard_normal((K, F))
    echo = il * (0.1 * np.exp(1j * rng.uniform(0, 2 * np.pi, F)))[None, :]
    other = rng.standard_normal((K, F)) + 1j * rng.standard_normal((K, F))
    assert spectral_coherence(echo, il) > 0.999
    assert abs(spectral_coherence(other, il) - 1.0 / K) < 0.05


def _halting(cfg: TrackerConfig) -> tuple[RadiatorTracker, list]:
    """A ship at 4.5 m/s north east that slows uniformly to a stop between 30 s and 39 s at (250, 220), seen every
    frame until 36.5 s and then only at 37, 41, 44.5, 46.5, 48.25 s and every frame from 49.5 s (the pattern of the boat
    on ``multistatic_b``)."""
    v = 4.5 * np.array([-0.47, 0.88])
    stop = np.array([250.0, 220.0])

    def pos(t):
        if t >= 39.0:
            return stop
        if t <= 30.0:
            return stop - v * (30.0 - t) - v * 4.5           # 4.5 s of uniform deceleration over 9 s covers 4.5 * v * 1
        dt = 39.0 - t
        return stop - v * dt ** 2 / (2 * 9.0)
    seen = set(np.round(np.arange(0.25, 36.5, 0.25), 2)) | {37.0, 41.0, 44.5, 46.5, 48.25} | set(np.round(np.arange(49.5, 60.0, 0.25), 2))
    trk = RadiatorTracker(cfg)
    log = []
    for t in np.round(np.arange(0.25, 60.0, 0.25), 2):
        xy = [pos(t)] if t in seen else []
        trk.update(FrameResult(float(t), np.array(xy).reshape(-1, 2), np.ones(len(xy)), {}, 0.0, 0.0))
        firm = [tr for tr in trk.tracks.values() if tr.confirmed]
        log.append((float(t), min((float(np.hypot(*(tr.predict(t) - pos(t)))) for tr in firm), default=np.inf)))
    return trk, log


def test_a_stopping_ship_seen_now_and_then_keeps_its_track():
    """A confirmed track dropped after 2 s without a fix (as a tentative one) is lost in the gaps after the stop; kept
    for 4.5 s (``drop_confirmed_s``) it holds the stopped ship in at least 90 % of the frames after 36.5 s (measured:
    0.51 against 0.99).  More process noise does not help: at 0.3 m/s^2 the coasting prediction drifts off the stopped
    ship (0.49 even with 4.5 s)."""
    base = dict(meas_floor=1.0, confirm=6, confirm_window_s=2.0, drop_s=2.0)
    frac = {}
    for name, cfg in (("old", TrackerConfig(accel=0.05, **base)),
                      ("new", TrackerConfig(accel=0.05, drop_confirmed_s=4.5, **base)),
                      ("noisy", TrackerConfig(accel=0.3, drop_confirmed_s=4.5, **base))):
        _, log = _halting(cfg)
        after = [d for t, d in log if t >= 36.5]
        frac[name] = float(np.mean(np.array(after) <= 5.0))
    assert frac["new"] >= 0.9, frac
    assert frac["old"] < 0.6, frac
    assert frac["noisy"] < frac["new"], frac

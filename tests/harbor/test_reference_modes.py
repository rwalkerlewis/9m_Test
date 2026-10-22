"""The reference estimators of ``BistaticConfig.ref_mode`` (``bistatic.estimate_reference``) on an analytic free
field case with a known waveform: a stationary ship, a strain cable of two legs (one end on, one broadside) and four
hydrophones, the traces computed exactly from ``collapse(G)`` per frequency (the response the weighted modes model),
noise of a different level on every channel.  Each mode recovers the waveform (coherence) and removes the direct
path (ECA depth); the weighted modes beat the beam; the joint solve recovers a known waveform and known scales from
noise free blocks; the iterated mode subtracts a body's modelled echo."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, coherence, sosfiltfilt

from acoustic_sim.water import Cable, SensorLayout
from acoustic_sim.water.bistatic import (
    BistaticConfig,
    REF_MODES,
    _joint_reference,
    channel_response,
    eca_remove,
    estimate_reference,
    make_reference,
)
from acoustic_sim.water.propagation import green2d

FS = 3000.0
T = 4.0
C = 1500.0
SHIP = np.array([170.0, 230.0])
BAND = (100.0, 600.0)


def _propagate(s: np.ndarray, layout: SensorLayout, src: np.ndarray) -> np.ndarray:
    """Traces (N, n) of the waveform *s* radiated at *src*: ``collapse(G(k |p - src|))`` per rfft bin."""
    n = len(s)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    S = np.fft.rfft(s)
    r = np.hypot(*(layout.sample_points - src[None, :]).T)
    X = np.zeros((layout.n_receivers, len(f)), dtype=complex)
    for i in range(1, len(f)):
        k = 2.0 * np.pi * f[i] / C
        X[:, i] = layout.collapse(green2d(k * r, np), np) * S[i]
    return np.fft.irfft(X, n, axis=1)


@pytest.fixture(scope="module")
def scene():
    cable = Cable("cable1", np.array([[60.0, 100.0], [300.0, 100.0], [300.0, 300.0]]), channel_spacing=4.0,
                  das_mode="strain", gauge_length=2.0)
    layout = SensorLayout(cables=[cable], hydrophones=np.array([[40.0, 40.0], [340.0, 40.0], [40.0, 340.0], [340.0, 340.0]]))
    n = int(T * FS)
    rng = np.random.default_rng(11)
    sos = butter(4, [80.0, 700.0], btype="bandpass", fs=FS, output="sos")
    s = sosfiltfilt(sos, rng.standard_normal(n))
    tt = np.arange(n) / FS
    s *= np.clip((tt - 0.4) / 0.1, 0.0, 1.0) * np.clip((T - 0.4 - tt) / 0.1, 0.0, 1.0)
    s /= s.std()
    direct = _propagate(s, layout, SHIP)
    # Noise: a level per channel kind, and a tenth of the cable channels ten times noisier.
    cab = layout.slices["cable1"]
    level = np.full(layout.n_receivers, 0.3 * np.median(np.std(direct[cab], axis=1)))
    hyd = layout.slices["hydrophones"]
    level[hyd] = 0.05 * np.median(np.std(direct[hyd], axis=1))
    noisy = rng.choice(np.arange(cab.start, cab.stop), size=(cab.stop - cab.start) // 10, replace=False)
    level[noisy] *= 10.0
    noise = level[:, None] * rng.standard_normal(direct.shape)
    return layout, s, direct, noise


def _bc(mode: str) -> BistaticConfig:
    return BistaticConfig(f_lo=BAND[0], f_hi=BAND[1], c=C, eca_doppler=True, eca_images=False, ref_mode=mode,
                          eca_gram="prefix", eca_chunk=400)


def _run(scene, mode: str, bodies=None, extra=None):
    layout, s, direct, noise = scene
    traces = direct + noise + (0.0 if extra is None else extra)
    n = traces.shape[1]
    bc = _bc(mode)
    nb = int(round(bc.eca_block_s * FS))
    ship_t = np.repeat(SHIP[None, :], n, axis=0)
    ship_b = np.repeat(SHIP[None, :], int(np.ceil(n / nb)), axis=0)
    chans = np.arange(layout.n_receivers)
    ref = estimate_reference(traces, layout, FS, ship_b, bc, chans, ship_t, bodies=bodies)
    out = eca_remove([traces, direct], ref, layout, FS, ship_b, [], bc.eca_block_s, bc, ship_t=ship_t)
    return ref, out, bc


def _coherence(ref: np.ndarray, s: np.ndarray) -> float:
    """Median magnitude squared coherence of the reference with the true waveform over the band (0.5 s segments)."""
    f, c = coherence(ref, s, fs=FS, nperseg=int(0.5 * FS))
    m = (f >= BAND[0]) & (f <= BAND[1])
    return float(np.median(c[m]))


def _depth(direct: np.ndarray, cleaned: np.ndarray, layout: SensorLayout) -> float:
    sos = butter(4, BAND, btype="bandpass", fs=FS, output="sos")
    cab = layout.slices["cable1"]
    e0 = np.sum(sosfiltfilt(sos, direct[cab], axis=1)[:, 3000:-3000] ** 2, axis=1)
    e1 = np.sum(sosfiltfilt(sos, cleaned[cab], axis=1)[:, 3000:-3000] ** 2, axis=1)
    return float(np.median(10 * np.log10(e0 / e1)))


@pytest.mark.parametrize("mode, min_coh, min_depth", [
    ("beam", 0.99, 20.0),
    ("weighted", 0.999, 28.0),
    ("weighted_cable", 0.999, 25.0),
    ("hydrophones", 0.999, 28.0),
    ("iterated", 0.99, 20.0),            # no body given: the beam again
    ("joint", 0.999, 30.0),
])
def test_each_mode_recovers_the_waveform_and_removes_the_direct_path(scene, mode, min_coh, min_depth):
    layout, s, direct, _ = scene
    ref, out, _ = _run(scene, mode)
    coh = _coherence(ref, s)
    depth = _depth(direct, out[1], layout)
    assert coh >= min_coh, (mode, coh)
    assert depth >= min_depth, (mode, depth)


def test_weighting_beats_the_beam(scene):
    """The broadside leg and the noisy channels add noise to the beam; the maximum ratio combination leaves them out:
    a higher coherence and a deeper removal."""
    layout, s, direct, _ = scene
    got = {}
    for m in ("beam", "weighted"):
        ref, out, _ = _run(scene, m)
        got[m] = (_coherence(ref, s), _depth(direct, out[1], layout))
    assert got["weighted"][0] > got["beam"][0], got
    assert got["weighted"][1] > got["beam"][1] + 5.0, got


def test_every_mode_is_tested():
    assert set(REF_MODES) == {"beam", "weighted", "weighted_cable", "hydrophones", "iterated", "joint"}


def test_channel_response_matches_the_propagated_traces(scene):
    """The modelled response of a channel, times the source spectrum and the delay, is the channel's spectrum."""
    layout, s, direct, _ = scene
    n = len(s)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    i = np.flatnonzero((f >= 200.0) & (f <= 400.0))[::50]
    chans = np.array([3, 40, layout.n_receivers - 1])
    A = channel_response(layout, SHIP, f[i], _bc("weighted"), chans)                                  # (F, 3)
    r = np.hypot(*(layout.receivers[chans] - SHIP).T)
    k = 2.0 * np.pi * f[i] / C
    pred = A * np.exp(-1j * k[:, None] * r[None, :]) * np.fft.rfft(s)[i][:, None]
    got = np.fft.rfft(direct[chans], axis=1)[:, i].T
    np.testing.assert_allclose(pred, got, rtol=1e-6, atol=1e-9 * np.abs(got).max())


def test_joint_recovers_a_known_waveform_and_scales():
    """Noise free blocks ``Y_i = a_i s``: the alternating least squares returns ``s`` and ``a`` up to one scale."""
    rng = np.random.default_rng(2)
    n, nb = 3000, 1500
    s = rng.standard_normal(n)
    a = rng.uniform(0.2, 3.0, size=12) * rng.choice([-1.0, 1.0], size=12)
    Y = a[:, None] * s[None, :]
    bc = _bc("joint")
    init = s + 0.5 * rng.standard_normal(n)
    est = _joint_reference(Y, np.ones(12), init, nb, 3, bc)
    for lo in (0, nb):
        e, t = est[lo:lo + nb], s[lo:lo + nb]
        assert abs(np.corrcoef(e, t)[0, 1]) > 1 - 1e-10
        g = (e @ t) / (t @ t)
        a_est = (Y[:, lo:lo + nb] @ e) / (e @ e)
        np.testing.assert_allclose(a_est * g, a, rtol=1e-8)


def test_iterated_subtracts_the_modelled_echo_of_a_body(scene):
    """An echo off a body at a known position: the iterated reference with that body is closer to the true
    waveform than the beam on the same traces (the echo is a second, delayed copy in the beam)."""
    layout, s, direct, noise = scene
    body = np.array([200.0, 140.0])
    d_sb = np.hypot(*(body - SHIP)) / C
    delayed = np.interp(np.arange(len(s)) / FS - d_sb, np.arange(len(s)) / FS, s, left=0.0, right=0.0)
    echo = 3.0 * _propagate(delayed, layout, body)
    beam = _coherence(_run(scene, "beam", extra=echo)[0], s)
    it = _coherence(_run(scene, "iterated", bodies=[body], extra=echo)[0], s)
    assert it > beam, (beam, it)


def test_beam_mode_is_make_reference(scene):
    layout, s, direct, noise = scene
    traces = direct + noise
    n = traces.shape[1]
    bc = _bc("beam")
    ship_t = np.repeat(SHIP[None, :], n, axis=0)
    ship_b = np.repeat(SHIP[None, :], 8, axis=0)
    chans = np.arange(layout.n_receivers)
    a = estimate_reference(traces, layout, FS, ship_b, bc, chans, ship_t)
    b = make_reference(traces, layout, FS, ship_b, bc.eca_block_s, bc, chans, ship_t=ship_t)
    np.testing.assert_array_equal(a, b)

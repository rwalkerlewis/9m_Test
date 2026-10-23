"""The direct paths of several illuminators removed in one solve (``bistatic.remove_direct_paths``,
``removal_order = "joint"``) on an analytic free field case with two known waveforms: two stationary ships, a strain
cable of two legs and four hydrophones, the traces computed exactly from ``collapse(G)`` per frequency, noise on every
channel.  With both ships present the joint removal recovers each waveform and removes each ship as deep as when that
ship is alone in the traces; the alternating estimate is no worse than the single pass on noise free traces; one
illuminator gives exactly the sequential result."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import butter, coherence, sosfiltfilt

from acoustic_sim.water import Cable, SensorLayout
from acoustic_sim.water.bistatic import BistaticConfig, remove_direct_paths
from acoustic_sim.water.propagation import green2d

FS = 3000.0
T = 4.0
C = 1500.0
SHIPS = (np.array([170.0, 230.0]), np.array([250.0, 160.0]))
LEVELS = (1.0, 0.5)
BAND = (100.0, 600.0)


def _propagate(s: np.ndarray, layout: SensorLayout, src: np.ndarray) -> np.ndarray:
    n = len(s)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    S = np.fft.rfft(s)
    r = np.hypot(*(layout.sample_points - src[None, :]).T)
    X = np.zeros((layout.n_receivers, len(f)), dtype=complex)
    for i in range(1, len(f)):
        X[:, i] = layout.collapse(green2d(2.0 * np.pi * f[i] / C * r, np), np) * S[i]
    return np.fft.irfft(X, n, axis=1)


@pytest.fixture(scope="module")
def scene():
    cable = Cable("cable1", np.array([[60.0, 100.0], [300.0, 100.0], [300.0, 300.0]]), channel_spacing=4.0,
                  das_mode="strain", gauge_length=2.0)
    layout = SensorLayout(cables=[cable], hydrophones=np.array([[40.0, 40.0], [340.0, 40.0], [40.0, 340.0], [340.0, 340.0]]))
    n = int(T * FS)
    rng = np.random.default_rng(5)
    sos = butter(4, [80.0, 700.0], btype="bandpass", fs=FS, output="sos")
    tt = np.arange(n) / FS
    taper = np.clip((tt - 0.4) / 0.1, 0.0, 1.0) * np.clip((T - 0.4 - tt) / 0.1, 0.0, 1.0)
    waves, directs = [], []
    for ship, level in zip(SHIPS, LEVELS):
        s = sosfiltfilt(sos, rng.standard_normal(n)) * taper
        s *= level / s.std()
        waves.append(s)
        directs.append(_propagate(s, layout, ship))
    # The noise of test_reference_modes: a level per channel kind, a tenth of the cable channels ten times noisier.
    cab, hyd = layout.slices["cable1"], layout.slices["hydrophones"]
    level = np.full(layout.n_receivers, 0.3 * np.median(np.std(directs[0][cab], axis=1)))
    level[hyd] = 0.05 * np.median(np.std(directs[0][hyd], axis=1))
    noisy = rng.choice(np.arange(cab.start, cab.stop), size=(cab.stop - cab.start) // 10, replace=False)
    level[noisy] *= 10.0
    noise = level[:, None] * rng.standard_normal(directs[0].shape)
    return layout, waves, directs, noise


def _bc(order: str, passes: int = 2) -> BistaticConfig:
    return BistaticConfig(f_lo=BAND[0], f_hi=BAND[1], c=C, eca_doppler=True, eca_images=False, ref_mode="joint",
                          eca_gram="prefix", eca_chunk=400, removal_order=order, joint_passes=passes)


def _ships(n: int, bc: BistaticConfig, which) -> list:
    nb = int(round(bc.eca_block_s * FS))
    return [(np.repeat(SHIPS[k][None, :], int(np.ceil(n / nb)), axis=0), np.repeat(SHIPS[k][None, :], n, axis=0)) for k in which]


def _depth(direct: np.ndarray, cleaned: np.ndarray, layout: SensorLayout) -> float:
    sos = butter(4, BAND, btype="bandpass", fs=FS, output="sos")
    cab = layout.slices["cable1"]
    e0 = np.sum(sosfiltfilt(sos, direct[cab], axis=1)[:, 3000:-3000] ** 2, axis=1)
    e1 = np.sum(sosfiltfilt(sos, cleaned[cab], axis=1)[:, 3000:-3000] ** 2, axis=1)
    return float(np.median(10 * np.log10(e0 / e1)))


def _coherence(ref: np.ndarray, s: np.ndarray) -> float:
    f, c = coherence(ref, s, fs=FS, nperseg=int(0.5 * FS))
    m = (f >= BAND[0]) & (f <= BAND[1])
    return float(np.median(c[m]))


def _remove(scene, which, order: str, passes: int = 2, noise: bool = True):
    layout, waves, directs, nz = scene
    traces = sum(directs[k] for k in which) + (nz if noise else 0.0)
    bc = _bc(order, passes)
    chans = np.arange(layout.n_receivers)
    _, comps, refs = remove_direct_paths(traces, [directs[k] for k in which], layout, FS, _ships(traces.shape[1], bc, which),
                                         chans, bc)
    return [_depth(directs[k], c, layout) for k, c in zip(which, comps)], refs


@pytest.fixture(scope="module")
def lone(scene):
    return [_remove(scene, [k], "sequential")[0][0] for k in range(2)]


def test_joint_removes_each_ship_as_deep_as_alone_and_recovers_each_waveform(scene, lone):
    """Four passes: measured 0.7 and 0.3 dB short of alone (37.3 and 39.3 dB); with the configs' two passes this case
    is 5.5 and 2.4 dB short (the louder ship converges slowly; the two ship FDTD case is within 3 dB at two)."""
    depths, refs = _remove(scene, [0, 1], "joint", passes=4)
    for k in range(2):
        assert abs(depths[k] - lone[k]) <= 1.0, (k, depths[k], lone[k])
        # Each reference against its own ship's true waveform, with the other ship in the traces.
        assert _coherence(refs[k], scene[1][k]) >= 0.999, (k, _coherence(refs[k], scene[1][k]))


def test_sequential_removal_falls_short_of_the_lone_depth(scene, lone):
    """The mechanism the joint removal acts on: removed one at a time, the second ship's reference was formed with
    the first's residual in the traces and the first's with the second's direct sound in them."""
    depths, _ = _remove(scene, [0, 1], "sequential")
    assert min(lone[k] - depths[k] for k in range(2)) > 1.0, (depths, lone)


def test_alternating_estimate_deeper_than_one_pass_on_noise_free_traces(scene):
    """One solve on the beams of the raw traces against the passes that re-estimate each reference without the other
    ship (measured: 14.9 and 20.3 dB against 36.3 and 41.4 at two passes)."""
    one, _ = _remove(scene, [0, 1], "joint", passes=0, noise=False)
    alt, _ = _remove(scene, [0, 1], "joint", passes=2, noise=False)
    for k in range(2):
        assert alt[k] >= one[k] + 10.0, (k, alt, one)
        assert alt[k] >= 30.0, (k, alt)


def test_one_illuminator_is_the_sequential_result(scene):
    layout, waves, directs, nz = scene
    traces = directs[0] + nz
    chans = np.arange(layout.n_receivers)
    bc = _bc("sequential")
    a = remove_direct_paths(traces, [], layout, FS, _ships(traces.shape[1], bc, [0]), chans, bc)[0]
    b = remove_direct_paths(traces, [], layout, FS, _ships(traces.shape[1], bc, [0]), chans, replace(bc, removal_order="joint"))[0]
    np.testing.assert_array_equal(a, b)


def _stream_depths(scene, order: str) -> list[float]:
    """The streaming echo chain (``StreamingEcho``) on the two ship traces (measured: sequential 26.8 and 17.4 dB, joint
    29.4 and 34.0), both ships given as radiators (fixes every
    0.25 s, as the passive tracks give them), the first the illuminator; the depth of each ship's direct path over the
    cleaned blocks, from the truth components passed through the same projections (a diagnostic)."""
    from acoustic_sim.contract import ChartEnvironment
    from acoustic_sim.water.streaming import EchoStreamConfig, StreamingEcho

    layout, waves, directs, nz = scene
    traces = directs[0] + directs[1] + nz
    x = np.arange(0.0, 401.0, 5.0)
    chart = ChartEnvironment(water=np.ones((len(x), len(x)), bool), depth=np.full((len(x), len(x)), 15.0), x=x, y=x, c_nominal=C)
    bc = replace(_bc(order), grid_dx=10.0, edge=20.0, exclusion_radius=30.0, coherent_norm="snr", map_scale="none",
                 track_smooth_s=1.0)
    det = StreamingEcho(layout, FS, chart, bc, EchoStreamConfig(step_blocks=2, history_s=1.0, lookahead_s=0.4))
    det.components = {"a": directs[0], "b": directs[1]}
    frames, chunk = [], 600
    for s0 in range(0, traces.shape[1], chunk):
        t1 = min(s0 + chunk, traces.shape[1]) / FS
        t_fix = [t for t in np.arange(0.0, t1, 0.25) if t > s0 / FS - 0.25]
        fixes = {name: [(t, SHIPS[k]) for t in t_fix] for k, name in enumerate(("a", "b"))}
        frames += det.push(traces[:, s0:s0 + chunk], fixes, illuminator="a" if s0 == 0 else None)
    assert len(frames) >= 4
    cab = layout.slices["cable1"]
    out = []
    for name in ("a", "b"):
        e0 = np.sum([b for b, _ in det.component_energy[name]], axis=0)[cab]
        e1 = np.sum([a for _, a in det.component_energy[name]], axis=0)[cab]
        out.append(float(np.median(10 * np.log10(e0 / e1))))
    return out


def test_the_streaming_chain_removes_both_ships_deeper_with_the_joint_removal(scene):
    seq = _stream_depths(scene, "sequential")
    joint = _stream_depths(scene, "joint")
    assert min(joint) >= min(seq) + 3.0, (joint, seq)

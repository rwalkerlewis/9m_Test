"""Stage 1 additions to the bistatic chain on the analytic scene of ``test_bistatic.py``: the
reference modes, the low speed fallback of the Doppler ECA, the unnormalised map, the GPU replicas
and the detection statistics."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_sim.harbor import Cable, SensorLayout, uniform_water_environment
from acoustic_sim.harbor.ais import AISReports
from acoustic_sim.harbor.bistatic import (
    BistaticConfig,
    ReplicaModel,
    ShipTrack,
    Truth,
    _band_taper,
    bistatic_map,
    build_grid,
    eca_stage,
    equaliser_weights,
    fit_track,
    make_reference,
    oracle_spectra,
    reference_spectrum,
    run_bistatic,
    ship_positions,
    snr_weights,
    stft,
)
from acoustic_sim.harbor.bistatic_eval import (
    classify_misses,
    detection_stats,
    frame_threshold,
    integrate_frames,
    miss_counts,
)
from acoustic_sim.harbor.device import get_device
from tests.harbor._gpu import HAS_GPU
from tests.harbor.test_bistatic import BODY, C, F_MAX, FS, SHIP, T, _depth_db, scene  # noqa: F401


def _coherence(ref: np.ndarray, truth: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """Magnitude squared coherence of two waveforms averaged over the band (Welch, 0.5 s segments)."""
    from scipy.signal import coherence

    f, c = coherence(ref, truth, fs=fs, nperseg=int(0.5 * fs))
    m = (f >= f_lo) & (f <= f_hi)
    return float(np.mean(c[m]))


@pytest.mark.parametrize("mode", ["beam", "equalised", "snr", "mrc", "iterated"])
def test_reference_modes_cancel_the_direct_path_and_match_the_waveform(scene, mode):
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(reference="data", removal="eca", c=C, refine=False, ref_mode=mode, grid_dx=5.0, edge=20.0, f_hi=450.0)
    sp = stft(traces, layout, FS, bc, F_MAX)
    model = ReplicaModel(layout, bc.c)
    model.weights = sp.weights
    ship = ship_positions(sp, ais, model, bc)
    outs, wf = eca_stage(traces, layout, FS, env, sp, ship, bc, None, [direct, echo], ais=ais, model=model, f_max=F_MAX)
    depth, loss = _depth_db(direct, outs[1]), _depth_db(echo, outs[2])
    assert np.median(depth) > 12.0, (mode, depth)
    assert np.median(loss) < 1.0, (mode, loss)
    assert _coherence(wf, s[: len(wf)], FS, 100.0, 450.0) > 0.9, mode


def test_hydrophone_reference_uses_only_the_hydrophones(scene):
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(reference="data", removal="eca", c=C, refine=False, ref_channels="hydrophones")
    sp = stft(traces, layout, FS, bc, F_MAX)
    model = ReplicaModel(layout, bc.c)
    model.weights = sp.weights
    ship = ship_positions(sp, ais, model, bc)
    outs, wf = eca_stage(traces, layout, FS, env, sp, ship, bc, None, [direct], ais=ais)
    assert np.median(_depth_db(direct, outs[1])) > 12.0
    # The same beam built by hand from the two hydrophones.
    hyd = np.arange(layout.n_das, layout.n_receivers)
    nb = int(round(bc.eca_block_s * FS))
    ship_b = np.repeat(SHIP[None], int(np.ceil(traces.shape[1] / nb)), axis=0)
    assert np.allclose(wf, make_reference(traces, layout, FS, ship_b, bc.eca_block_s, bc, hyd))


def test_equaliser_weights_floor_and_plain_beam_limit(scene):
    """The equaliser table has the band taper, the floor caps the gain on weak channels, and a
    model whose response is only the range delay reduces the weights to the plain mean."""
    env, layout, *_ = scene
    bc = BistaticConfig(c=C, ref_floor=0.2, f_lo=100.0)
    model = ReplicaModel(layout, bc.c)
    chans = np.arange(layout.n_receivers)
    freqs, W = equaliser_weights(model, layout, chans, SHIP[None], bc, 450.0)
    assert W.shape == (1, len(chans), len(freqs))
    assert np.all(W[:, :, freqs < 100.0 - bc.ref_taper_hz] == 0) and np.all(W[:, :, freqs > 450.0 + bc.ref_taper_hz] == 0)
    mid = np.argmin(np.abs(freqs - 300.0))
    gain = np.abs(W[0, :, mid]) * len(chans)
    # Pressure channels: |a| = |H0(kr)| ~ 1 / sqrt(r); the gain is the inverse, capped by the floor.
    assert gain.max() <= 1.0 / (bc.ref_floor * np.abs(model(SHIP[None], 2 * np.pi * 300.0 / C)[0]).max()) * 1.0001
    assert np.all(_band_taper(np.array([50.0, 100.0, 300.0, 450.0, 480.0]), 100.0, 450.0, 20.0) == [0.0, 1.0, 1.0, 1.0, 0.0])


def test_snr_weights_favour_the_cleaner_channel():
    """Two channels with the same signal power and noise powers 1 and 4: the maximum ratio
    weights are in the ratio 4:1 and sum to a unit signal gain."""
    bc = BistaticConfig(f_lo=100.0, f_hi=400.0)
    freqs = np.linspace(100.0, 400.0, 16)
    times = np.array([0.25, 0.75])
    X = np.sqrt(2.0) * np.ones((2, len(freqs), 2), dtype=complex)
    R = np.ones((2, len(freqs), 2), dtype=complex)
    R[1] *= 2.0
    from acoustic_sim.harbor.bistatic import Spectra

    sp_x = Spectra(X, freqs, times, np.ones(2), 2000.0, 0.5)
    sp_r = Spectra(R, freqs, times, np.ones(2), 2000.0, 0.5)
    f_w, W = snr_weights(sp_x, sp_r, np.array([0, 1]), np.array([0.5]), bc)
    w = W[0, :, 1:-1].real
    assert np.allclose(w[0] / w[1], 4.0)
    # unit gain: sum_i sqrt(S_i) w_i = 1
    assert np.allclose(np.sqrt(2.0) * w.sum(axis=0), 1.0)
    assert W[0, :, 0].sum() == 0 and W[0, :, -1].sum() == 0


def test_low_speed_fallback_protects_a_stationary_ship(scene):
    """Stationary ship, refined positions jittering by 1 m: the per sample delays read the jitter
    as velocity (stage 0 finding 6) unless the Kalman track and the fallback hold the position."""
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(reference="oracle", removal="eca", c=C, refine=False, eca_doppler=True)
    sp = stft(traces, layout, FS, bc, F_MAX)
    K = len(sp.times)
    jitter = SHIP[None, :] + np.random.default_rng(9).standard_normal((K, 2))
    present = np.ones(K, bool)
    depths = {}
    for name, kw in (("jitter", dict(track_filter="none", fallback_speed=0.0)),
                     ("kalman_fallback", dict(track_filter="kalman", fallback_speed=0.3)),
                     ("fixed", dict(eca_doppler=False, track_filter="none"))):
        b = BistaticConfig(**{**bc.__dict__, **kw})
        ship = ShipTrack("ship", jitter, jitter.copy(), present, np.ones(K))
        outs, _ = eca_stage(traces, layout, FS, env, sp, ship, b, s, [direct], ais=ais)
        depths[name] = float(np.median(_depth_db(direct, outs[1])))
    assert depths["kalman_fallback"] > depths["jitter"] + 6.0, depths
    assert depths["kalman_fallback"] > depths["fixed"] - 3.0, depths


def test_fit_track_modes_and_exact_track(scene):
    env, layout, traces, *_ , ais = scene
    bc = BistaticConfig(c=C, refine=False)
    sp = stft(traces, layout, FS, bc, F_MAX)
    K = len(sp.times)
    jitter = SHIP[None, :] + np.random.default_rng(1).standard_normal((K, 2))
    ship = fit_track(ShipTrack("ship", jitter, jitter.copy(), np.ones(K, bool), np.ones(K)), sp.times, ais, bc)
    assert ship.smoothed.shape == (K, 2) and np.all(np.hypot(*(ship.smoothed - SHIP).T) < 0.5)
    assert np.all(np.hypot(*ship.velocity.T) < 0.3)
    exact = fit_track(ShipTrack("ship", jitter, jitter.copy(), np.ones(K, bool), np.ones(K), exact=True), sp.times, ais, bc)
    assert np.array_equal(exact.xy, jitter)
    none = fit_track(ShipTrack("ship", jitter, jitter.copy(), np.ones(K, bool), np.ones(K)), sp.times, ais,
                     BistaticConfig(c=C, refine=False, track_filter="none"))
    assert np.array_equal(none.xy, jitter)


def test_unnormalised_map_option(scene):
    env, layout, traces, direct, echo, s, ais = scene
    kw = dict(reference="oracle", removal="eca", method="coherent", coherent_norm="snr", c=C, grid_dx=5.0, edge=20.0,
              exclusion_radius=30.0, refine=False)
    truth = Truth(ship_xy=None)
    runs = {}
    for norm in ("max", "none"):
        bc = BistaticConfig(**kw, map_norm=norm)
        sp = stft(traces, layout, FS, bc, F_MAX)
        truth = Truth(ship_xy=np.repeat(SHIP[None], len(sp.times), axis=0))
        runs[norm] = run_bistatic(traces, layout, FS, env, ais, bc, F_MAX, oracle_signal=s, truth=truth)
    Bn, Bm = runs["none"].scan.B, runs["max"].scan.B
    assert Bn.max() != pytest.approx(1.0) or Bm.max() == pytest.approx(1.0)
    assert Bm.max() == pytest.approx(1.0)
    assert np.allclose(Bm, Bn / Bn.max(), rtol=1e-6, atol=1e-9)
    # The body is found in both.
    est = runs["none"].grid[np.argmax(Bn, axis=0)]
    assert np.median(np.hypot(*(est - BODY).T)) < 15.0


@pytest.mark.skipif(not HAS_GPU, reason="CuPy / GPU not available")
def test_gpu_replicas_match_the_cpu(scene):
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(reference="oracle", removal="none", method="coherent", c=C, grid_dx=5.0, edge=20.0, refine=True)
    sp = stft(traces, layout, FS, bc, F_MAX)
    grid = build_grid(env, bc)
    maps, tracks = {}, {}
    for cuda in (False, True):
        model = ReplicaModel(layout, bc.c, device=get_device(cuda))
        model.weights = sp.weights
        a = model.asnumpy(model(grid[:50], 2 * np.pi * 300.0 / C))
        p = model.asnumpy(model.path(grid[:50], np.vstack([SHIP, SHIP + 3.0]), 2 * np.pi * 300.0 / C))
        assert a.shape == (50, layout.n_receivers) and p.shape == (50, 2)
        if cuda:
            assert np.allclose(a, a_cpu, rtol=2e-4, atol=1e-6 * np.abs(a_cpu).max())
            assert np.allclose(p, p_cpu, rtol=2e-4, atol=1e-6 * np.abs(p_cpu).max())
        else:
            a_cpu, p_cpu = a, p
        ship = ship_positions(sp, ais, model, bc)
        ref = reference_spectrum(sp, ship, model, bc, oracle=oracle_spectra(s, FS, sp))
        maps[cuda], _ = bistatic_map(sp, ship, ref, grid, model, bc)
        tracks[cuda] = ship.refined
    assert np.allclose(tracks[True], tracks[False])
    c = np.corrcoef(maps[True].ravel(), maps[False].ravel())[0, 1]
    assert c > 0.9999, c


def test_frame_threshold_and_detection_stats():
    rng = np.random.default_rng(0)
    maxima = rng.standard_normal(100)
    T = frame_threshold(maxima, 0.05)
    assert np.sum(maxima > T) == 5
    assert frame_threshold(maxima, 0.0) == maxima.max()
    grid = np.array([[x, y] for x in range(0, 50, 5) for y in range(0, 50, 5)], dtype=float)
    K = 4
    B = np.zeros((len(grid), K))
    truth = np.array([[10.0, 10.0]] * K)
    B[np.argmin(np.hypot(*(grid - [10.0, 10.0]).T)), 0] = 3.0     # detected (above, within)
    B[np.argmin(np.hypot(*(grid - [12.0, 10.0]).T)), 1] = 0.5     # within but below the threshold
    B[np.argmin(np.hypot(*(grid - [40.0, 40.0]).T)), 2] = 5.0     # above but far
    B[np.argmin(np.hypot(*(grid - [10.0, 10.0]).T)), 3] = 1.0     # exactly at the threshold: not above
    st = detection_stats(B, grid, truth, threshold=1.0, radius=5.0)
    assert st.detected.tolist() == [True, False, False, False]
    assert st.pd == 0.25 and st.declared == 0.5
    assert st.summary()["within_radius"] == 0.75
    B3 = integrate_frames(B, 3)
    assert np.allclose(B3[:, 1], B[:, 0] + B[:, 1] + B[:, 2]) and np.allclose(B3[:, 0], B[:, 0] + B[:, 1])


def test_classify_misses():
    env = uniform_water_environment(400.0, 400.0, 5.0, C)
    env.land[:, -5:] = True                                   # a quay along the east edge (x >= 380)
    env.water[:, -5:] = False
    cable = Cable("cable1", np.array([[100.0, 100.0], [300.0, 100.0]]), channel_spacing=10.0)
    layout = SensorLayout(cables=[cable])
    ship = np.array([200.0, 300.0])
    K = 6
    est = np.array([
        [200.0, 272.0],      # 28 m from the ship: 3 m inside the 25 m exclusion edge -> ship_edge
        [555.0, 300.0],      # image of the ship in the quay at x = 380: (560, 300) -> quay_image (within 10 m)
        [200.0, 200.0],      # on the segment ship -> channel (200, 100) -> direct_path
        [60.0, 250.0],       # elsewhere
        [201.0, 299.0],      # small error: not a miss
        [240.0, 250.0],      # on the segment ship -> channel (280, 100)? distance of the point to that line
    ])
    truth = np.repeat(ship[None] + [0.0, -100.0], K, axis=0)   # (200, 200) is the truth: frame 2 is not a miss
    err = np.hypot(*(est - truth).T)
    live = np.ones(layout.n_receivers, bool)
    cls = classify_misses(est, err, np.repeat(ship[None], K, axis=0), env, layout, live, exclusion_radius=25.0)
    assert cls[0] == "ship_edge" and cls[1] == "quay_image" and cls[2] == "" and cls[3] == "elsewhere"
    assert cls[4] == "direct_path"                               # 1 m from the ship: on the apex of the fan
    counts = miss_counts(cls)
    assert counts["ship_edge"] == 1 and counts["quay_image"] == 1 and sum(counts.values()) == int(np.sum(err > 20.0))
    # The fan check: a point on the ship -> channel (280, 100) segment, 80 m off the truth.
    seg = ship + 0.5 * (np.array([280.0, 100.0]) - ship)
    cls2 = classify_misses(seg[None], np.array([80.0]), ship[None], env, layout, live, exclusion_radius=25.0)
    assert cls2[0] == "direct_path"

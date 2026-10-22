"""Passive replicas on a strain DAS cable: the Green's function at the sample points, collapsed to channels.

The traces of a strain cable are the second difference of the field over the gauge length
(``SensorLayout.collapse``); the passive scanner, the streaming chain and the ship suppression must use
the same model, as the echo chain does.  The old replica (the Green's function at the channel centres)
is kept here only to show the difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_sim.water import Cable, SensorLayout, uniform_water_environment
from acoustic_sim.water.detection import (
    MatchedFieldScanner,
    ProcessingConfig,
    Whitened,
    build_grid,
    green2d,
    preprocess,
)
from acoustic_sim.water.streaming import StreamConfig, StreamingPassive
from acoustic_sim.water.suppression import _replicas as ship_replicas

from ._gpu import DEVICES, HAS_GPU

C, FS = 1495.0, 2000.0
SOURCE = np.array([170.0, 150.0])


def _layout(mode: str) -> SensorLayout:
    cable = Cable("cable1", np.array([[100.0, 100.0], [260.0, 100.0], [260.0, 200.0]]), 8.0, mode, 8.0)
    return SensorLayout(cables=[cable], hydrophones=np.array([[60.0, 200.0], [300.0, 80.0]]))


def _whitened(layout: SensorLayout, rng) -> Whitened:
    n, freqs = layout.n_receivers, np.arange(80.0, 400.0, 8.0)
    return Whitened(freqs=freqs, times=np.array([1.0]), Z=np.zeros((n, len(freqs), 1), np.complex64),
                    sigma=rng.uniform(0.5, 2.0, (n, len(freqs))), weights=(rng.uniform(size=n) > 0.1).astype(float),
                    scan_bins=np.arange(len(freqs)))


def _cfg(**kw) -> ProcessingConfig:
    return ProcessingConfig(win_s=0.5, hop_s=0.25, f_lo=80.0, f_hi=400.0, c=C, grid_dx=4.0, grid_margin=0.0, min_range=1.0,
                            grid_bounds=(120.0, 240.0, 120.0, 190.0), bin_stride=1, interference_rank_max=0, **kw)


def _old_replica(layout, xy, wh, i, cfg):
    """The replica before the fix: ``green2d`` at the channel centres, ignoring the strain mode."""
    r = np.maximum(np.hypot(*(xy[:, None, :] - layout.receivers[None, :, :]).transpose(2, 0, 1)), cfg.min_range)
    return green2d(2 * np.pi * wh.freqs[i] / C * r) / wh.sigma[None, :, i] * wh.weights[None, :]


def test_scanner_replica_is_the_collapsed_greens_function_at_the_sample_points():
    rng = np.random.default_rng(0)
    xy = rng.uniform(120.0, 240.0, (30, 2))
    for mode in ("strain", "pressure"):
        layout, cfg = _layout(mode), _cfg()
        wh = _whitened(layout, rng)
        scanner = MatchedFieldScanner(layout, xy, cfg)
        pts = layout.sample_points
        r = np.maximum(np.hypot(*(xy[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)), cfg.min_range)
        for i in (0, 17, len(wh.freqs) - 1):
            k = 2 * np.pi * wh.freqs[i] / C
            ref = layout.collapse(green2d(k * r)) / wh.sigma[None, :, i] * wh.weights[None, :]
            got = scanner._replicas(scanner._geometry(xy), wh, i)
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)
            # The ship suppression builds the same replicas.
            sup = ship_replicas(scanner.rec, xy, wh, np.array([i]), cfg, layout)[0]
            np.testing.assert_allclose(sup, ref, rtol=1e-12, atol=0)
            if mode == "pressure":                    # pressure layouts: exactly the replica of before
                np.testing.assert_array_equal(got, _old_replica(layout, xy, wh, i, cfg).astype(got.dtype))


def _point_source_traces(layout: SensorLayout, T: float = 3.0, seed: int = 4) -> np.ndarray:
    """A stationary broadband point source at ``SOURCE``, the field sampled at the sample points and collapsed."""
    rng = np.random.default_rng(seed)
    n = int(T * FS)
    f = np.fft.rfftfreq(n, 1 / FS)
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[(f < 80.0) | (f > 400.0)] = 0.0
    pts = layout.sample_points
    r = np.hypot(*(pts - SOURCE).T)
    sampled = np.fft.irfft(spec[None, :] * np.exp(-2j * np.pi * f[None, :] * r[:, None] / C) / np.sqrt(r)[:, None], n)
    traces = layout.collapse(sampled.T).T
    return traces + 0.3 * traces.std() * rng.standard_normal(traces.shape)


def test_strain_passive_map_peaks_at_the_source_and_above_the_old_replica():
    layout, cfg = _layout("strain"), _cfg(common_mode=False)
    env = uniform_water_environment(400, 300, 4.0, C)
    wh = preprocess(_point_source_traces(layout), layout, FS, cfg)
    scanner = MatchedFieldScanner(layout, build_grid(env, cfg), cfg)
    B = scanner.scan(wh).B.sum(axis=1)
    peak = scanner.grid[int(np.argmax(B))]
    assert np.hypot(*(peak - SOURCE)) <= cfg.grid_dx, peak

    class OldScanner(MatchedFieldScanner):
        def __init__(self, *a):
            super().__init__(*a)
            self.rec = np.asarray(layout.receivers, dtype=float)

        def _replicas(self, geom, wh, i):
            return (green2d(2 * np.pi * wh.freqs[i] / C * geom[0]) / wh.sigma[None, :, i] * wh.weights_d[None, :]).astype(np.complex128)

    B_old = OldScanner(layout, scanner.grid, cfg).scan(wh).B.sum(axis=1)
    assert B.max() > 2.0 * B_old.max(), (B.max(), B_old.max())


@pytest.mark.parametrize("mode", ["strain", "pressure"])
@pytest.mark.parametrize("cuda", DEVICES)
def test_streaming_fast_replicas_match_the_scanner(mode, cuda):
    """``StreamingPassive._greens`` (phase recurrence; the fused CUDA kernel on the GPU) against the scanner's
    replicas, on a strain and a pressure cable."""
    layout = _layout(mode)
    env = uniform_water_environment(400, 300, 4.0, C)
    pc = _cfg(use_cuda=cuda)
    sp = StreamingPassive(layout, FS, env, pc, stc=StreamConfig(block_frames=2))
    assert sp._fast
    xp = sp.dev.xp
    rng = np.random.default_rng(2)
    n = layout.n_receivers
    sigma = rng.uniform(0.5, 2.0, (n, len(sp.freqs)))
    w = (rng.uniform(size=n) > 0.1).astype(float)
    wh = Whitened(sp.freqs, np.array([1.0]), np.zeros((n, len(sp.freqs), 1), np.complex64), xp.asarray(sigma), w, sp.scan_bins, sp.dev)
    gen = sp._greens()
    gen.send(None)
    tol = 2e-5 if cuda else 1e-6
    for i in sp.scan_bins:
        A = gen.send(xp.asarray(w / sigma[:, i]))
        ref = sp.scanner._replicas(sp._geom, wh, int(i))
        err = float(xp.max(xp.abs(A - ref)) / xp.max(xp.abs(ref)))
        assert err < tol, (int(i), err)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy / GPU not available")
def test_gpu_strain_passive_map_matches_the_cpu():
    layout = _layout("strain")
    env = uniform_water_environment(400, 300, 4.0, C)
    traces = _point_source_traces(layout)
    maps = []
    for cuda in (False, True):
        cfg = _cfg(common_mode=False, use_cuda=cuda)
        wh = preprocess(traces, layout, FS, cfg)
        maps.append(MatchedFieldScanner(layout, build_grid(env, cfg), cfg).scan(wh).B)
    np.testing.assert_allclose(maps[1], maps[0], rtol=2e-3, atol=2e-3 * maps[0].max())

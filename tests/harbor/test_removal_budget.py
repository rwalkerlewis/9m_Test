"""The direct path removal within the frame budget (prompt ``run_all_3``, task 5): the prefix sum Gram between two
references against the matrix product, the decimation of the removal band, and the decimation round trip."""

from __future__ import annotations

import numpy as np

from acoustic_sim.water.bistatic import _gram_cross_prefix, _gram_prefix
from acoustic_sim.water.streaming import _decimate, _interpolate, removal_decimation


def _windows(base, start, block):
    return np.stack([np.stack([base[g, s:s + block] for s in start[g]], axis=1) for g in range(base.shape[0])])   # (Ng, block, L)


def test_cross_prefix_gram_equals_the_matrix_product():
    rng = np.random.default_rng(0)
    Ng, n, block = 3, 400, 300
    bp, bq = rng.standard_normal((Ng, n)), rng.standard_normal((Ng, n))
    sp = np.stack([np.arange(20, 37)[::-1] + g for g in range(Ng)])                 # contiguous lags, one offset per channel
    sq = np.stack([np.arange(60, 73) + 3 * g for g in range(Ng)])
    Sp, Sq = _windows(bp, sp, block), _windows(bq, sq, block)
    G = _gram_cross_prefix(bp, sp, bq, sq, block)
    ref = np.matmul(Sp.transpose(0, 2, 1), Sq)
    assert np.allclose(G, ref, rtol=1e-9, atol=1e-9 * np.abs(ref).max())
    # and the full joint Gram assembled from the blocks equals the product of the stacked copies
    S = np.concatenate([Sp, Sq], axis=2)
    full = np.concatenate([np.concatenate([_gram_prefix(bp, sp, block), G], axis=2),
                           np.concatenate([_gram_cross_prefix(bq, sq, bp, sp, block), _gram_prefix(bq, sq, block)], axis=2)], axis=1)
    assert np.allclose(full, np.matmul(S.transpose(0, 2, 1), S), rtol=1e-9, atol=1e-9 * np.abs(full).max())


def test_removal_decimation():
    assert removal_decimation(6000.0, 3000, None) == 1
    assert removal_decimation(6000.0, 3000, [100.0, 760.0]) == 3          # 1000 Hz Nyquist >= 950
    assert removal_decimation(6000.0, 3000, [100.0, 800.0]) == 3          # 1000 >= 1000
    assert removal_decimation(6000.0, 3000, [100.0, 900.0]) == 2


def test_decimation_round_trip_keeps_the_band():
    """A tone at 300 Hz sampled at 6000 Hz, decimated by 3 and brought back: the same tone away from the ends, to the
    filter's ripple (about -50 dB)."""
    fs, n = 6000.0, 6000
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 300.0 * t)[None, :]
    y = _decimate(x, 3)
    assert y.shape[1] == n // 3
    assert np.allclose(y[0, 100:-100], x[0, ::3][100:-100], atol=5e-3)
    z = _interpolate(y, 3, n)
    assert z.shape[1] == n
    assert np.allclose(z[0, 300:-300], x[0, 300:-300], atol=5e-3)

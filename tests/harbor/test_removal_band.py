"""The direct path removal on decimated traces (``BistaticConfig.removal_band``, prompt ``run_all_4``, task 1): the
decimation of the removal band and the decimation round trip."""

from __future__ import annotations

import numpy as np

from acoustic_sim.water.streaming import _decimate, _interpolate, removal_decimation


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

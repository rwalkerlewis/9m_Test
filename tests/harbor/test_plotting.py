"""The geometry helpers behind the result figures (``water/plotting.py``)."""

import numpy as np

from acoustic_sim.water.plotting import bistatic_ellipse, projected_positions


def test_every_point_of_the_equal_delay_ellipse_has_the_bistatic_range_of_the_body():
    s, r, b = np.array([10.0, 20.0]), np.array([150.0, -30.0]), np.array([90.0, 80.0])
    e = bistatic_ellipse(s, r, b)
    L = np.hypot(*(b - s)) + np.hypot(*(b - r))
    np.testing.assert_allclose(np.hypot(*(e - s).T) + np.hypot(*(e - r).T), L, rtol=1e-12)


def test_projected_positions_extend_the_line_fitted_to_the_past_fixes():
    fixes = [(t, np.array([1.0 + 3.0 * t, 2.0 - 1.0 * t]), 1.0) for t in np.arange(0.0, 5.0, 0.25)]
    times, xy = projected_positions(fixes, 4.0, (0.5, 2.0))
    np.testing.assert_allclose(times, [4.5, 6.0])
    np.testing.assert_allclose(xy, [[1.0 + 3.0 * 4.5, 2.0 - 4.5], [1.0 + 18.0, 2.0 - 6.0]], atol=1e-9)
    _, one = projected_positions(fixes[:1], 4.0, (1.0,))           # fewer than 3 fixes: the last fix
    np.testing.assert_allclose(one, [fixes[0][1]])

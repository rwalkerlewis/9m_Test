"""``water/coverage.py`` on layouts with known answers: a straight strain cable along x (coverage 1 on its axis, 0
broadside), a wall that shadows half the sensors, and the cross range resolution of a short straight array against the
textbook beam width ``0.886 lambda R / L``."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_sim.water import Cable, SensorLayout, uniform_water_environment
from acoustic_sim.water.coverage import aperture, predicted_resolution, shadowing, strain_coverage


def _straight(mode="strain"):
    cable = Cable("c", np.array([[400.0, 500.0], [600.0, 500.0]]), channel_spacing=4.0, das_mode=mode, gauge_length=2.0)
    return SensorLayout(cables=[cable], hydrophones=np.empty((0, 2)))


def test_straight_strain_cable_sees_along_its_axis_not_broadside():
    lay = _straight()
    frac, weighted = strain_coverage(lay, np.array([[2000.0, 500.0], [500.0, 2500.0]]))
    assert frac[0] == pytest.approx(1.0) and weighted[0] == pytest.approx(1.0)
    assert frac[1] == pytest.approx(0.0) and weighted[1] == pytest.approx(0.0)
    near = strain_coverage(lay, np.array([[500.0, 550.0]]))[0][0]  # 50 m off the middle: the channels beyond +-45 degrees
    expected = np.mean(np.abs(lay.receivers[:, 0] - 500.0) > 50.0)
    assert near == pytest.approx(expected, abs=0.02)


def test_pressure_cable_and_hydrophones_see_every_direction():
    lay = _straight("pressure")
    frac, _ = strain_coverage(lay, np.array([[500.0, 2500.0]]))
    assert frac[0] == pytest.approx(1.0)


def test_aperture_of_the_sensors_that_see_a_cell():
    """Broadside to a strain cable nothing on the cable sees the cell; two hydrophones east and west of it span 180
    degrees."""
    cable = Cable("c", np.array([[400.0, 500.0], [600.0, 500.0]]), channel_spacing=4.0, das_mode="strain", gauge_length=2.0)
    lay = SensorLayout(cables=[cable], hydrophones=np.array([[300.0, 2500.0], [700.0, 2500.0]]))
    assert aperture(lay, np.array([[500.0, 2500.0]]))[0] == pytest.approx(180.0, abs=1e-6)
    lay2 = SensorLayout(cables=[cable], hydrophones=np.empty((0, 2)))
    assert aperture(lay2, np.array([[500.0, 2500.0]]))[0] == 0.0


def test_a_wall_shadows_half_the_sensors():
    env = uniform_water_environment(400.0, 400.0, 1.0, 1500.0)
    env.water[150:250, 195:205] = False                             # a wall at x 195 to 205 m, y 150 to 250 m
    lay = SensorLayout(cables=[], hydrophones=np.array([[300.0, 180.0], [300.0, 220.0], [100.0, 180.0], [100.0, 220.0]]))
    cell = np.array([[120.0, 200.0]])                              # west of the wall: the two east sensors are behind it
    assert shadowing(lay, cell, env)[0] == pytest.approx(0.5)
    assert shadowing(lay, np.array([[200.0, 350.0]]), env)[0] == pytest.approx(1.0)


def test_cross_range_resolution_of_a_short_line_array():
    """A 100 m pressure line array, 1 m channels, 300 Hz, a source 200 m broadside: the response's width across the
    range direction is the half power beam width times the range, 0.886 lambda R / L (about 8.9 m)."""
    cable = Cable("c", np.array([[450.0, 500.0], [550.0, 500.0]]), channel_spacing=1.0, das_mode="pressure")
    lay = SensorLayout(cables=[cable], hydrophones=np.empty((0, 2)))
    c, f, R, L = 1500.0, 300.0, 200.0, 100.0
    w = predicted_resolution(lay, np.array([[500.0, 500.0 + R]]), f, c, half=40.0, dx=0.5)[0]
    expected = 0.886 * (c / f) * R / L
    assert w[1] == pytest.approx(expected, rel=0.15)
    assert abs(w[2] - 0.0) < 5.0 or abs(w[2] - 180.0) < 5.0     # the major axis along the range direction (north)

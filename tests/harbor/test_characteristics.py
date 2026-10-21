"""The measured characteristics (``water/characteristics.py``) on synthetic inputs with known answers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import get_window

from acoustic_sim.water.ais import AISReports
from acoustic_sim.water.characteristics import (
    LARGE,
    SMALL,
    ais_match,
    band_rms,
    beam_gain,
    channel_replicas,
    echo_strength,
    kinematics,
    line_persistence,
    peak_shape,
    received_level,
    size_class,
    source_level_100m,
    source_spectrum,
    tonal_lines,
)
from acoustic_sim.water.propagation import green2d
from acoustic_sim.water.sensors import Cable, SensorLayout

C = 1500.0


def _layout(mode: str = "pressure") -> SensorLayout:
    return SensorLayout(cables=[Cable("c", np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 150.0]]), 4.0, mode, 4.0)],
                        hydrophones=np.array([[-20.0, 100.0], [220.0, 160.0]]))


def test_kinematics_of_a_straight_track():
    """Fixes on a line at 3 m/s heading 60 degrees (from north): velocity, speed and heading exact, residual zero; with
    1 m of scatter the speed standard error is the least squares one."""
    t = np.arange(0.0, 6.0, 0.25)
    v = 3.0 * np.array([np.sin(np.radians(60.0)), np.cos(np.radians(60.0))])
    fixes = [(ti, np.array([10.0, 20.0]) + v * ti) for ti in t]
    k = kinematics(fixes, t[-1], 4.0)
    assert k["speed"] == pytest.approx(3.0)
    assert k["heading_deg"] == pytest.approx(60.0)
    assert k["residual_m"] == pytest.approx(0.0, abs=1e-9)
    assert k["window_s"] == [pytest.approx(t[-1] - 4.0), t[-1]]
    assert np.allclose(k["xy"], np.array([10.0, 20.0]) + v * t[-1])
    rng = np.random.default_rng(0)
    se = []
    for _ in range(300):
        noisy = [(ti, p + rng.normal(0, 1.0, 2)) for ti, p in fixes]
        se.append(kinematics(noisy, t[-1], 4.0))
    m = t >= t[-1] - 4.0
    expected = 1.0 / np.sqrt(np.sum((t[m] - t[m].mean()) ** 2))       # slope standard error for unit scatter
    assert np.median([s["speed_se"] for s in se]) == pytest.approx(expected, rel=0.1)
    assert np.std([s["velocity"][0] for s in se]) == pytest.approx(expected, rel=0.15)


def test_peak_shape_of_an_elliptical_gaussian():
    """A Gaussian peak with full widths at half maximum 12 m and 4 m, major axis 30 degrees from north: the widths, the
    direction, and the length along and across a direction."""
    x = np.arange(-40.0, 40.1, 1.0)
    gx, gy = np.meshgrid(x, x)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    th = np.radians(30.0)
    u, w = np.array([np.sin(th), np.cos(th)]), np.array([np.cos(th), -np.sin(th)])
    a, b = grid @ u, grid @ w
    s = lambda fwhm: fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))     # noqa: E731
    B = np.exp(-0.5 * (a / s(12.0)) ** 2 - 0.5 * (b / s(4.0)) ** 2)
    out = peak_shape(B, grid, [0.0, 0.0], 30.0, direction=u)
    assert out["widths_m"][0] == pytest.approx(12.0, rel=0.08)
    assert out["widths_m"][1] == pytest.approx(4.0, rel=0.15)
    assert out["major_deg"] == pytest.approx(30.0, abs=3.0)
    assert out["along_m"] == pytest.approx(12.0, abs=2.0)
    assert out["across_m"] == pytest.approx(4.0, abs=2.0)


def test_band_rms_is_parseval():
    fs, nper = 3000.0, 1500
    t = np.arange(nper) / fs
    x = 0.7 * np.cos(2 * np.pi * 200.0 * t) + 0.2 * np.cos(2 * np.pi * 310.0 * t)
    win = get_window("hann", nper)
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(nper, 1 / fs)
    band = (f >= 100) & (f <= 380)
    assert band_rms(X[band], nper, win) == pytest.approx(np.sqrt(0.7 ** 2 / 2 + 0.2 ** 2 / 2), rel=1e-3)


@pytest.mark.parametrize("mode", ["pressure", "strain"])
def test_source_spectrum_received_and_source_level(mode):
    """Channel spectra of a source of known spectrum at a known position (the chain's replicas, strain or pressure):
    the least squares spectrum is the source's; the received level is the analytic field's band rms at the range; the
    source level scales by sqrt(r / 100)."""
    lay = _layout(mode)
    freqs = np.arange(100.0, 380.0, 2.0)
    rng = np.random.default_rng(1)
    s = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
    xy = np.array([90.0, 70.0])
    X = (channel_replicas(lay, xy, freqs, C) * s[:, None]).T             # (N, F)
    w = np.ones(lay.n_receivers)
    w[3] = 0.0
    X[3] = 1e6                                                           # a channel the QC removed does not count
    est = source_spectrum(X, w, lay, xy, freqs, C)
    np.testing.assert_allclose(est, s, rtol=1e-9)
    nper, win = 1500, get_window("hann", 1500)
    rl = received_level(est, freqs, C, 120.0, nper, win)
    assert rl == pytest.approx(band_rms(s * green2d(2 * np.pi * freqs / C * 120.0), nper, win))
    assert source_level_100m(rl, 400.0, 1.5) == pytest.approx(1.5 * rl * 2.0)


def test_tonal_lines_over_a_flat_floor():
    freqs = np.arange(100.0, 380.0, 2.0)
    p = np.ones(len(freqs))
    p[30], p[80] = 10.0, 100.0                                           # 10 dB and 20 dB lines
    lines = tonal_lines(p, freqs, 3, 6.0, 10)
    assert [ln["f_hz"] for ln in lines] == [freqs[80], freqs[30]]
    assert [round(ln["level_db"]) for ln in lines] == [20, 10]
    assert line_persistence([[80], [79], [30], [81, 30], []], 80, 1) == 3


def test_echo_strength_and_size_ratio():
    """Cleaned spectra holding only an echo of known strength (sigma times the illuminator's field at the body, a
    source there), and the data reference of the illuminator (its source spectrum times the beam's gain): the strength
    in dB; with sigma rising 6 dB over the band the size ratio is about +4 dB (upper over lower third) and the body is
    "smaller than about a wavelength"."""
    lay = _layout("strain")
    freqs = np.arange(100.0, 760.0, 4.0)
    rng = np.random.default_rng(2)
    S = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
    ill, body = np.array([40.0, 120.0]), np.array([130.0, 60.0])
    w = np.ones(lay.n_receivers)
    ref = S * beam_gain(lay, w, ill, freqs, C)
    path = green2d(2 * np.pi * freqs / C * np.hypot(*(body - ill)))
    for sig, ratio in ((np.full(len(freqs), 0.1), 0.0), (0.1 * 10 ** (6.0 / 20 * (freqs - 100) / 660), 4.0)):
        X = (channel_replicas(lay, body, freqs, C) * (sig * path * S)[:, None]).T
        e = echo_strength(X, w, lay, body, ill, ref, freqs, C)
        assert e["lower_third_db"] < e["strength_db"] < e["upper_third_db"] or ratio == 0.0
        if ratio == 0.0:
            assert e["strength_db"] == pytest.approx(-20.0, abs=1e-6)
            assert e["size_ratio_db"] == pytest.approx(0.0, abs=1e-6)
        else:
            assert e["size_ratio_db"] == pytest.approx(ratio, abs=0.3)
            assert size_class(e["size_ratio_db"], 0.0) == SMALL
    assert size_class(-1.0, 0.0) == LARGE


def test_ais_match_identity_and_distance():
    """Reports of a ship 3 m north of a track: the track takes its identity with the distance at the latest fix; reports
    received after the frame are not used; a far ship is no report."""
    t = np.arange(0.0, 10.0, 0.25)
    fixes = [(ti, np.array([100.0 + 2.0 * ti, 50.0]), 1.0) for ti in t]
    tr = np.arange(0.0, 10.0, 2.0)
    rep = AISReports(np.array(["ferry"] * len(tr) + ["far"] * len(tr)), np.r_[tr, tr],
                     np.r_[np.column_stack([100.0 + 2.0 * tr, np.full(len(tr), 53.0)]), np.column_stack([tr, np.full(len(tr), 400.0)])],
                     np.r_[np.full(len(tr), 2.0), np.zeros(len(tr))], np.zeros(2 * len(tr)))
    m = ais_match(fixes, rep, t[-1], 25.0)
    assert m["id"] == "ferry" and m["distance_m"] == pytest.approx(3.0, abs=1e-6)
    late = AISReports(rep.vessel, rep.t + 20.0, rep.xy, rep.sog, rep.cog)
    assert ais_match(fixes, late, t[-1], 25.0) == {"id": None}

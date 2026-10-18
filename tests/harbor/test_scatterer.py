"""Moving scattering bodies, strain mode DAS and the cropped environment.

Test A compares the FDTD scattered field of a soft cylinder with the partial wave series.
Test B measures what a *moving* body radiates on its own: with a hard cell mask every cell
hop is an impulse; with the smooth mask it is not.  Test C checks the cos^2 directivity of
the strain channels.  All runs are small (a few seconds each on the CPU).
"""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_sim.harbor import HarborFDTDConfig, HarborFDTDSolver, SensorLayout, simulate_source, uniform_water_environment
from acoustic_sim.harbor.cylinder import line_source_scattered, plane_wave_far_field_amplitude
from acoustic_sim.harbor.environment import build_synthetic_harbor
from acoustic_sim.harbor.scatterer import Scatterer, body_factor
from acoustic_sim.harbor.sensors import Cable
from acoustic_sim.harbor.vessels import Track
from acoustic_sim.sources import StaticSource

DX = 0.5
C = 1500.0
FS = 600.0 * 5.0 / DX


def _cfg(total_time: float) -> HarborFDTDConfig:
    return HarborFDTDConfig(fs=FS, total_time=total_time, fd_order=4, damping_width=60, damping_max=0.15,
                            background_absorption=0.0, shore_loss_width=0.0, shore_loss_max=0.0)


def _still_track(name: str, xy: np.ndarray, total_time: float) -> Track:
    t = np.array([0.0, total_time])
    return Track(name, "uuv", False, t, np.vstack([xy, xy]), np.zeros(2), np.ones(2, bool))


def _gate(x: np.ndarray, t: np.ndarray, t_arr: np.ndarray, pre: float, post: float) -> np.ndarray:
    w = np.zeros_like(x)
    for i in range(len(x)):
        w[i, (t > t_arr[i] - pre) & (t < t_arr[i] + post)] = 1.0
    return x * w


# ----------------------------------------------------------------------
# Test A: stationary soft cylinder against the partial wave series
# ----------------------------------------------------------------------

def test_soft_cylinder_matches_partial_wave_series():
    """Scattered / incident ratio at 16 receivers, ka = 0.5, 1 and 2 (>= 12 cells per wavelength):
    within 1 dB and 0.15 rad.  The same body at ka = 3, 4 and 6 (8, 6 and 4 cells per
    wavelength) is off by 1.5, 2.3 and 6.6 dB, so a scatterer needs about 10 cells per
    wavelength, finer than the 3.75 the propagation itself was validated at."""
    env = uniform_water_environment(200.0, 160.0, DX)
    total = 0.30
    cfg = _cfg(total)
    n = cfg.n_steps
    t = np.arange(n) / FS
    f0 = 450.0
    arg = (np.pi * f0 * (t - 0.01)) ** 2
    sig = (1 - 2 * arg) * np.exp(-arg)                       # Ricker, energy 100 to 800 Hz
    src = (40.0, 80.0)
    centre = np.array([100.0, 80.0])
    a = 2.0                                                  # 4 cells radius
    ang = np.radians(np.arange(0, 360, 45))
    rec = np.vstack([centre + r * np.c_[np.cos(ang), np.sin(ang)] for r in (30.0, 50.0)])
    layout = SensorLayout(cables=[], hydrophones=rec)
    body = Scatterer("cyl", _still_track("cyl", centre, total), 2 * a, 2 * a, "soft", shape="ellipse")

    direct = simulate_source(env, layout, cfg, sig, position=src).traces.astype(float)
    scattered = simulate_source(env, layout, cfg, sig, position=src, scatterers=[body]).traces.astype(float) - direct

    t_dir = np.hypot(*(rec - np.array(src)).T) / C
    t_sc = (np.hypot(*(centre - np.array(src))) + np.hypot(*(rec - centre).T)) / C
    D = np.fft.rfft(_gate(direct, t, t_dir, 0.006, 0.03), axis=1)
    S = np.fft.rfft(_gate(scattered, t, t_sc, 0.006, 0.03), axis=1)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    for ka in (0.5, 1.0, 2.0):
        i = int(np.argmin(np.abs(f - ka * C / (2 * np.pi * a))))
        k = 2 * np.pi * f[i] / C
        inc, sc = line_source_scattered(k, a, centre, np.array(src), rec, "soft")
        ratio_fd = np.conj(S[:, i] / D[:, i])                 # FFT sign -> exp(-i w t) convention
        ratio_an = sc / inc
        amp_db = 20 * np.log10(np.abs(ratio_fd) / np.abs(ratio_an))
        phase = np.angle(ratio_fd / ratio_an)
        assert np.max(np.abs(amp_db)) < 1.0, (ka, amp_db)
        assert np.max(np.abs(phase)) < 0.15, (ka, phase)


def test_partial_wave_optical_theorem_sanity():
    """|f(0)| grows with ka and the soft far field is the strongest in the forward direction."""
    k, a = 2.0 * np.pi * 300.0 / C, 1.0
    phi = np.linspace(0, np.pi, 181)
    f_soft = np.abs(plane_wave_far_field_amplitude(k, a, phi, "soft"))
    f_rigid = np.abs(plane_wave_far_field_amplitude(k, a, phi, "rigid"))
    assert np.argmax(f_soft) == 0
    assert f_soft.max() > f_rigid.max()          # ka ~ 1.3: the soft body scatters more


# ----------------------------------------------------------------------
# Test B: what a moving body radiates
# ----------------------------------------------------------------------

@pytest.mark.parametrize("smooth, limit_db", [(True, -40.0), (False, -40.0)])
def test_moving_body_out_of_band_radiation(smooth: bool, limit_db: float):
    """A 1 m x 3 m hull crossing a 200 Hz tone field at 2.5 m/s.  The scattered field's energy
    more than 40 Hz away from the tone, relative to the energy at the tone, must be below
    -40 dB.  The smooth mask passes (about -60 dB); the hard cell mask fails (about -30 dB:
    every cell hop radiates a click).  Both numbers are asserted so the comparison is kept."""
    env = uniform_water_environment(120.0, 100.0, DX)
    total = 1.0
    cfg = _cfg(total)
    n = cfg.n_steps
    t = np.arange(n) / FS
    f0 = 200.0
    sig = np.sin(2 * np.pi * f0 * t) * np.clip(t / 0.05, 0, 1)
    src = (25.0, 50.0)
    centre0 = np.array([65.0, 50.0])
    ang = np.radians([60, 120, 240, 300])
    rec = centre0 + 25.0 * np.c_[np.cos(ang), np.sin(ang)]
    layout = SensorLayout(cables=[], hydrophones=rec)
    v = 2.5
    xy = np.array([centre0 + [0, -v * total / 2], centre0 + [0, v * total / 2]])
    track = Track("b", "uuv", False, np.array([0.0, total]), xy, np.full(2, v), np.ones(2, bool))
    body = Scatterer("b", track, 3.0, 1.0, "soft", smooth=smooth)
    direct = simulate_source(env, layout, cfg, sig, position=src).traces.astype(float)
    sc = (simulate_source(env, layout, cfg, sig, position=src, scatterers=[body]).traces.astype(float) - direct)
    sc = sc[:, int(0.3 * FS):]
    S = np.abs(np.fft.rfft(sc * np.hanning(sc.shape[1]), axis=1)) ** 2
    ff = np.fft.rfftfreq(sc.shape[1], 1.0 / FS)
    inband = np.abs(ff - f0) <= 15.0
    outband = (np.abs(ff - f0) > 40.0) & (ff > 20.0) & (ff < 1200.0)
    ratio_db = 10 * np.log10(S[:, outband].sum(axis=1) / S[:, inband].sum(axis=1))
    if smooth:
        assert np.max(ratio_db) < limit_db, ratio_db
    else:
        assert np.max(ratio_db) > limit_db, ratio_db     # the hard mask does radiate; documented, not used


# ----------------------------------------------------------------------
# Test C: strain mode directivity
# ----------------------------------------------------------------------

def test_strain_channels_follow_cos_squared():
    """Far point sources at 0, 45 and 90 degrees to a straight fiber.  The strain channel
    (second difference over the gauge g) over the pressure channel must follow
    4 sin^2(k g cos(theta) / 4) within 0.5 dB at 0 and 45 degrees, and be at least 30 dB down
    at 90 degrees."""
    env = uniform_water_environment(200.0, 200.0, DX)
    total = 0.25
    cfg = _cfg(total)
    n = cfg.n_steps
    t = np.arange(n) / FS
    f0 = 200.0
    sig = np.sin(2 * np.pi * f0 * t) * np.clip(t / 0.03, 0, 1)
    g = 4.0
    cable = Cable("c", np.array([[90.0, 100.0], [110.0, 100.0]]), channel_spacing=4.0, das_mode="strain", gauge_length=g)
    layout = SensorLayout(cables=[cable], hydrophones=np.empty((0, 2)))
    k = 2 * np.pi * f0 / C
    r = 80.0
    out = {}
    for theta_deg in (0.0, 45.0, 90.0):
        th = np.radians(theta_deg)
        sxy = np.array([100.0 - r * np.cos(th), 100.0 - r * np.sin(th)])
        solver = HarborFDTDSolver(env, cfg, StaticSource(sxy[0], sxy[1], sig), layout.sample_points)
        raw = solver.run()["traces"]
        strain = layout.collapse(raw.T).T
        pressure = raw[1::3]
        sl = slice(int(0.15 * FS), n)                      # steady state
        measured = np.sqrt(np.mean(strain[:, sl] ** 2, axis=1)) / np.sqrt(np.mean(pressure[:, sl] ** 2, axis=1))
        # Angle between the fiber (+x) and the source direction, per channel.
        d = sxy[None, :] - layout.receivers
        cos_th = np.abs(d[:, 0]) / np.hypot(d[:, 0], d[:, 1])
        out[theta_deg] = (measured, 4 * np.sin(0.25 * k * g * cos_th) ** 2)
    for theta_deg in (0.0, 45.0):
        measured, expected = out[theta_deg]
        err_db = 20 * np.log10(measured / expected)
        assert np.max(np.abs(err_db)) < 0.5, (theta_deg, err_db)
    assert np.max(out[90.0][0] / out[0.0][0]) < 10 ** (-30 / 20)


def test_collapse_is_identity_for_pressure_layouts():
    cable = Cable("c", np.array([[0.0, 0.0], [100.0, 0.0]]), channel_spacing=10.0)
    layout = SensorLayout(cables=[cable], hydrophones=np.array([[50.0, 50.0]]))
    x = np.random.default_rng(0).standard_normal((layout.n_samples, 7))
    assert layout.sample_points.shape == layout.receivers.shape
    assert np.array_equal(layout.collapse(x.T).T, x)


def test_strain_layout_sample_points_and_fingerprint():
    wp = np.array([[0.0, 0.0], [100.0, 0.0]])
    p = SensorLayout(cables=[Cable("c", wp, 10.0)], hydrophones=np.empty((0, 2)))
    s = SensorLayout(cables=[Cable("c", wp, 10.0, das_mode="strain", gauge_length=2.0)], hydrophones=np.empty((0, 2)))
    assert s.n_samples == 3 * s.n_receivers
    assert np.allclose(s.sample_points[1::3], s.receivers)
    assert np.allclose(s.sample_points[2::3] - s.sample_points[0::3], [[2.0, 0.0]] * s.n_receivers)
    assert p.fingerprint() != s.fingerprint()


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def test_body_factor_covers_the_body_and_nothing_else():
    x = np.arange(0, 50.0, DX)
    y = np.arange(0, 40.0, DX)
    track = _still_track("b", np.array([25.0, 20.0]), 1.0)
    body = Scatterer("b", track, 6.0, 2.0, "soft")
    rows, cols, factor = body_factor(body, np.array([25.0, 20.0]), np.array([1.0, 0.0]), x, y, DX)
    X, Y = np.meshgrid(x[cols], y[rows])
    inside = (np.abs(X - 25.0) < 2.5) & (np.abs(Y - 20.0) < 0.5)
    outside = (np.abs(X - 25.0) > 3.5) | (np.abs(Y - 20.0) > 1.5)
    assert np.all(factor[inside] == 0.0)
    assert np.all(factor[outside] == 1.0)
    hard = Scatterer("b", track, 6.0, 2.0, "soft", smooth=False)
    _, _, fh = body_factor(hard, np.array([25.0, 20.0]), np.array([1.0, 0.0]), x, y, DX)
    assert set(np.unique(fh)) <= {0.0, 1.0}
    absorbing = Scatterer("b", track, 6.0, 2.0, "absorbing", strength=0.3)
    _, _, fa = body_factor(absorbing, np.array([25.0, 20.0]), np.array([1.0, 0.0]), x, y, DX)
    assert np.isclose(fa.min(), 0.7)


def test_scatterer_changes_cache_key(tmp_path):
    env = uniform_water_environment(60.0, 60.0, 1.0)
    cfg = HarborFDTDConfig(fs=3000.0, total_time=0.02, damping_width=10)
    layout = SensorLayout(cables=[], hydrophones=np.array([[40.0, 30.0]]))
    sig = np.zeros(cfg.n_steps)
    sig[5] = 1.0
    body = Scatterer("b", _still_track("b", np.array([30.0, 30.0]), 0.02), 4.0, 4.0, "soft")
    simulate_source(env, layout, cfg, sig, position=(10.0, 30.0), cache_dir=tmp_path)
    simulate_source(env, layout, cfg, sig, position=(10.0, 30.0), cache_dir=tmp_path, scatterers=[body])
    assert len(list(tmp_path.glob("gather_*.npz"))) == 2


def test_cropped_harbor_keeps_shoreline_and_pier():
    full = build_synthetic_harbor(5.0)
    crop = (1150.0, 1550.0, 1200.0, 1500.0)
    part = build_synthetic_harbor(5.0, crop=crop)
    assert part.extent == (0.0, 400.0, 0.0, 300.0)
    # Land in the crop equals the full harbor's land on the same cells.
    ix0, iy0 = int(crop[0] / 5.0), int(crop[2] / 5.0)
    assert np.array_equal(part.land, full.land[iy0:iy0 + part.ny, ix0:ix0 + part.nx])
    assert part.land[:, 70:76].any()      # the pier at harbor x 1500-1530
    assert part.land[-1].mean() > 0.9 and part.land[-2].mean() > 0.7    # the north shore (the quay dips near x = 1250)

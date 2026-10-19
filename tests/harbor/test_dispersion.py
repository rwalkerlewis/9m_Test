"""Dispersion-corrected replica: closed-form properties, and agreement with the FDTD's own phase."""

import numpy as np
import pytest
from scipy.special import hankel2

from acoustic_sim.solver.fdtd import fd2_coefficients
from acoustic_sim.water import HarborFDTDConfig, SensorLayout, simulate_source, uniform_water_environment
from acoustic_sim.water.dispersion import (
    N_THETA,
    DispersionModel,
    for_scenario,
    replica_wavenumbers,
    wavenumber_pairs,
)

C, DX, FS = 1495.0, 5.0, 600.0


def test_low_frequency_limit_symmetry_and_anisotropy():
    m = DispersionModel(DX, 1.0 / FS, C, 4)
    k = m.wavenumber([5.0, 20.0, 60.0], [0.0, np.pi / 4, np.pi / 2])
    iso = 2 * np.pi * np.array([5.0, 20.0, 60.0]) / C
    rel = k / iso[:, None] - 1.0
    assert np.all(np.abs(rel[0]) < 3e-4)                       # tends to w / c at low frequency
    assert np.allclose(k[:, 0], k[:, 2])                       # symmetric under theta -> pi/2 - theta
    assert np.all(rel < 0)                                      # the grid is slow (leapfrog + FD both lag)
    assert -0.007 < rel[2, 0] < -0.003 and rel[2, 1] < -0.010  # at 60 Hz: ~0.4 % on axis, ~1.3 % on the diagonal


def test_solution_satisfies_the_dispersion_relation():
    m = DispersionModel(DX, 1.0 / FS, C, 4)
    f, theta = np.array([12.0, 40.0, 78.0]), np.linspace(0.0, np.pi / 2, 7)
    k = m.wavenumber(f, theta)
    coef = fd2_coefficients(4)

    def lam(kappa):
        return -(coef[0] + 2.0 * (coef[1] * np.cos(kappa * DX) + coef[2] * np.cos(2 * kappa * DX))) / DX ** 2

    lhs = C ** 2 * (lam(k * np.cos(theta)) + lam(k * np.sin(theta)))
    rhs = (2.0 / m.dt * np.sin(np.pi * f * m.dt)) ** 2
    assert np.allclose(lhs, rhs[:, None], rtol=1e-9)


def test_higher_order_and_finer_step_reduce_the_error():
    base = DispersionModel(DX, 1.0 / FS, C, 4).wavenumber([60.0], [np.pi / 4])[0, 0]
    iso = 2 * np.pi * 60.0 / C
    fine = DispersionModel(DX, 1.0 / (2 * FS), C, 8).wavenumber([60.0], [np.pi / 4])[0, 0]
    assert abs(fine / iso - 1) < 0.4 * abs(base / iso - 1)      # measured: 0.41 % vs 1.33 %, about 3x smaller


def test_pair_interpolation_and_isotropic_fallback():
    m = DispersionModel(DX, 1.0 / FS, C, 4)
    freqs = [20.0, 60.0]
    table = np.asarray(m.table(freqs))
    assert table.shape == (2, N_THETA)
    theta = np.array([[0.0, np.pi / 2, np.pi / 4], [np.pi / 8, 0.01, 1.5]])
    k = wavenumber_pairs(table, theta, np)
    assert k.shape == (2, 2, 3)
    assert np.allclose(k[:, 0, 0], table[:, 0]) and np.allclose(k[:, 0, 1], table[:, -1])
    assert np.allclose(k[:, 0, 2], table[:, (N_THETA - 1) // 2])
    iso = replica_wavenumbers(None, freqs, C, theta, np, np.float64)
    assert iso.shape == (2, 2, 3) and np.allclose(iso[1], 2 * np.pi * 60.0 / C)
    disp = replica_wavenumbers(m, freqs, C, theta, np, np.float64)
    assert np.allclose(disp, k)


def test_for_scenario_uses_water_sound_speed_and_scheme_parameters():
    env = uniform_water_environment(500, 400, 5.0, 1490.0)
    model = for_scenario(env, HarborFDTDConfig(fs=800.0, fd_order=6))
    assert (model.dx, model.dt, model.c, model.order) == (5.0, 1.0 / 800.0, 1490.0, 6)


def test_replica_phase_follows_the_fdtd_while_a_single_sound_speed_does_not():
    """60 Hz, 300 m on the grid diagonal: the isotropic replica is ~0.8 rad off; the numerical one is not."""
    env = uniform_water_environment(1000, 750, DX, C)
    cfg = HarborFDTDConfig(fs=FS, total_time=3.0, use_cuda=False)
    src = np.array([500.0, 375.0])
    r = np.array([100.0, 200.0, 300.0])
    axis = src + np.column_stack([r, np.zeros_like(r)])
    diag = src + np.column_stack([r, r]) / np.sqrt(2.0)
    layout = SensorLayout(hydrophones=np.vstack([axis, diag]))
    t = np.arange(cfg.n_steps) / FS
    a = (np.pi * 35.0 * (t - 0.15)) ** 2
    signal = (1 - 2 * a) * np.exp(-a)
    g = simulate_source(env, layout, cfg, signal, position=tuple(src))
    P, S = np.fft.rfft(g.traces.astype(np.float64), axis=1), np.fft.rfft(signal)
    k_bin = int(round(60.0 * cfg.total_time))
    f = np.fft.rfftfreq(cfg.n_steps, 1.0 / FS)[k_bin]
    d = layout.hydrophones - src
    rng = np.hypot(*d.T)
    theta = np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0]))[None, :]
    k_num = replica_wavenumbers(DispersionModel(DX, 1.0 / FS, C, 4), [f], C, theta, np, np.float64)[0, 0]
    phase = {}
    for name, k in (("iso", 2 * np.pi * f / C), ("num", k_num)):
        e = P[:, k_bin] / (S[k_bin] * hankel2(0, k * rng))
        phase[name] = np.angle(e / e[0])           # relative to the 100 m on-axis receiver
    assert abs(phase["num"][5]) < 0.15 and abs(phase["num"][2]) < 0.15
    assert abs(phase["iso"][5]) > 0.4 > 2 * abs(phase["num"][5])

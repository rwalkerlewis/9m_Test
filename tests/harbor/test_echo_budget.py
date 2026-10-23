"""The echo budget (``harbor_echo_budget.py``): the pressure release cylinder's partial wave series against what it
must satisfy (energy conservation per order, the geometric backscatter width at large ka, the small ka limit), and the
best channel and coherent sums on known inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location("harbor_echo_budget",
                                                  ROOT / "examples" / "harbor" / "05_detection" / "harbor_echo_budget.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_soft_cylinder_conserves_energy_per_order():
    """A lossless scatterer's coefficients satisfy |c_n|^2 = -Re c_n (with c_n = -J_n / H_n: J^2 / (J^2 + Y^2))."""
    m = _script()
    for ka in (0.05, 0.7, 3.0, 25.0):
        _, c = m.soft_coefficients(ka)
        assert np.allclose(np.abs(c) ** 2, -c.real, atol=1e-12)


def test_backscatter_width_tends_to_geometric_optics():
    """At large ka the backscatter echo width of a pressure release cylinder tends to pi a."""
    m = _script()
    a = 1.0
    for ka in (60.0, 120.0):
        k = ka / a
        sigma = float(m.echo_width(k, a, np.pi)[0])
        assert abs(sigma / (np.pi * a) - 1.0) < 0.05, (ka, sigma)


def test_small_soft_cylinder_scatters_isotropically():
    """At small ka only n = 0 matters: f is the same in every direction, |f| = |J0 / H0| ~ 1 / |1 - (2 i / pi) (ln(ka / 2)
    + gamma)| (the small argument forms of J0 and Y0)."""
    m = _script()
    ka = 0.01
    f = m.form_function(ka, np.linspace(0, np.pi, 7))
    assert np.allclose(np.abs(f), np.abs(f[0]), rtol=1e-3)
    approx = 1.0 / abs(1.0 - 2j / np.pi * (np.log(ka / 2) + np.euler_gamma))
    assert abs(abs(f[0]) / approx - 1.0) < 0.01


def test_coherent_best_on_known_powers():
    """Ten equal channels of echo 1 and interference 1 give 0 dB alone and 10 dB summed coherently; the ranking
    takes the best channels."""
    m = _script()
    e = np.r_[np.ones(10), np.full(5, 1e-6)]
    n = np.ones(15)
    best, coh = m.coherent_best(e, n, 10)
    assert abs(best) < 1e-9 and abs(coh - 10.0) < 1e-9
    best, coh = m.coherent_best(np.r_[4.0, np.ones(9)], np.ones(10), 10)
    assert abs(best - 10 * np.log10(4.0)) < 1e-9
    assert abs(coh - 10 * np.log10((2.0 + 9.0) ** 2 / 10.0)) < 1e-9

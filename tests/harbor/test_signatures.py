"""M2: signatures and level calibration."""

import numpy as np
import pytest

from acoustic_sim.harbor import HarborFDTDConfig, SensorLayout, simulate_source, uniform_water_environment
from acoustic_sim.harbor.signatures import (
    R_REF,
    InjectionCalibration,
    activity_gate,
    make_reference_signal,
    shaped_noise,
    ship_reference,
    to_injection,
    tonal_stack,
    uuv_reference,
)

FS = 600.0


def peak_freq(x, fs=FS, band=(5, 100)):
    f = np.fft.rfftfreq(len(x), 1 / fs)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    m = (f >= band[0]) & (f <= band[1])
    return f[m][np.argmax(p[m])]


def test_shaped_noise_rms_and_band():
    x = shaped_noise(60000, FS, np.random.default_rng(0), 10.0, 80.0, slope=0.7)
    assert np.sqrt(np.mean(x ** 2)) == pytest.approx(1.0)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    p = np.abs(np.fft.rfft(x)) ** 2
    assert p[(f < 7) | (f > 90)].sum() / p.sum() < 0.01


def test_tonal_stack_frequency_and_harmonic_limit():
    n = 60000
    x = tonal_stack(np.full(n, 20.0), FS, [0.0, -3.0, -6.0, -9.0, -12.0], np.random.default_rng(1))
    assert np.sqrt(np.mean(x ** 2)) == pytest.approx(1.0)
    assert peak_freq(x) == pytest.approx(20.0, abs=0.05)
    f = np.fft.rfftfreq(n, 1 / FS)
    p = np.abs(np.fft.rfft(x)) ** 2
    assert p[f > 78].sum() / p.sum() < 1e-3  # 4th harmonic (80 Hz) dropped, 5th too


def test_ship_and_uuv_have_distinct_spectra():
    n = 60000
    speed = np.full(n, 5.0)
    ship = ship_reference(speed, FS, np.random.default_rng(2))
    uuv = uuv_reference(np.full(n, 2.0), FS, np.random.default_rng(3))
    assert 15 < peak_freq(ship, band=(8, 40)) < 30
    assert 29 < peak_freq(uuv) < 42
    def tonal_fraction(x, k=40):  # power in the strongest k bins (Hann-windowed)
        p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        return np.sort(p)[-k:].sum() / p.sum()

    assert tonal_fraction(uuv) > 0.8
    assert tonal_fraction(ship) < 0.65


def test_ship_level_grows_with_speed():
    n = 60000
    lo = ship_reference(np.full(n, 0.0), FS, np.random.default_rng(4))
    hi = ship_reference(np.full(n, 5.0), FS, np.random.default_rng(4))
    assert np.std(lo) < 0.5 * np.std(hi)


def test_activity_gate():
    active = np.zeros(6000, dtype=bool)
    active[2000:4000] = True
    g = activity_gate(active, FS)
    assert g[0] == 0.0 and g[-1] == 0.0 and g[3000] == pytest.approx(1.0)
    sig = make_reference_signal("ship", np.full(6000, 5.0), active, FS, seed=1)
    assert np.all(sig[:1500] == 0.0) and np.std(sig[2500:3500]) > 0


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        make_reference_signal("whale", np.ones(10), np.ones(10, bool), FS, 0)


@pytest.mark.parametrize("kind,speed", [("ship", 5.0), ("uuv", 2.0)])
def test_injection_calibration_gives_unit_rms_at_100m(kind, speed):
    """A unit-level reference signal produces ~1 Pa rms at 100 m in free field."""
    env = uniform_water_environment(1000, 750, 5.0, 1500.0)
    cfg = HarborFDTDConfig(fs=FS, total_time=14.0)
    n = cfg.n_steps
    ref = make_reference_signal(kind, np.full(n, speed), np.ones(n, bool), FS, seed=5)
    sig = to_injection(ref, FS, InjectionCalibration(dx=5.0, dt=cfg.dt, c=1500.0))
    src = (500.0, 375.0)
    layout = SensorLayout(hydrophones=np.array([[src[0] + R_REF, src[1]], [src[0], src[1] - R_REF]]))
    g = simulate_source(env, layout, cfg, sig, position=src)
    t = g.t
    rms = np.sqrt(np.mean(g.traces[:, t > 2.0] ** 2, axis=1))
    gain = 0.3 + 0.7 * min(speed / 5.0, 1.5) if kind == "ship" else 0.5 + 0.5 * min(speed / 2.0, 1.5)
    assert np.allclose(rms / gain, 1.0, atol=0.2)


def test_f_max_follows_grid_spacing():
    """The usable band is 80 Hz on the 5 m grid and scales inversely with dx."""
    from acoustic_sim.harbor.signatures import F_MAX, f_max_for_grid

    assert F_MAX == 80.0
    assert f_max_for_grid(5.0) == 80.0
    assert np.isclose(f_max_for_grid(0.5), 800.0)
    assert np.isclose(f_max_for_grid(2.5, c=1500.0), 160.0)


def test_injection_band_follows_f_max():
    """to_injection on a fine grid keeps energy above 80 Hz; on the 5 m grid it does not."""
    n = 6000
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(n)
    fine = to_injection(ref, FS, InjectionCalibration(dx=0.5, dt=1.0 / FS, c=1500.0))
    coarse = to_injection(ref, FS, InjectionCalibration(dx=5.0, dt=1.0 / FS, c=1500.0))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    hi = f > 100.0
    assert np.sum(np.abs(np.fft.rfft(coarse))[hi] ** 2) < 1e-12 * np.sum(np.abs(np.fft.rfft(coarse)) ** 2)
    assert np.sum(np.abs(np.fft.rfft(fine))[hi] ** 2) > 0.1 * np.sum(np.abs(np.fft.rfft(fine)) ** 2)

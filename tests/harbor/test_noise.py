"""M2: noise and sensor effects."""

import numpy as np
import pytest

from acoustic_sim.water import Cable, NoiseConfig, SensorLayout, generate_noise

FS = 600.0
N = 12000


@pytest.fixture(scope="module")
def layout():
    cable = Cable("c", np.array([[200.0, 300.0], [700.0, 300.0]]), channel_spacing=10.0)
    return SensorLayout(cables=[cable], hydrophones=np.array([[300.0, 500.0], [600.0, 500.0]]))


def rms(x):
    return np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2, axis=-1))


def test_reproducible_and_seed_dependent(layout):
    cfg = NoiseConfig()
    a = generate_noise(layout, FS, N, cfg, np.random.default_rng(0))
    b = generate_noise(layout, FS, N, cfg, np.random.default_rng(0))
    c = generate_noise(layout, FS, N, cfg, np.random.default_rng(1))
    assert np.array_equal(a.noise_only, b.noise_only)
    assert not np.allclose(a.noise_only, c.noise_only)


def test_levels_ambient_only(layout):
    cfg = NoiseConfig(shipping_rms=0.0, das_noise_rms=0.0, das_common_mode_rms=0.0, hydrophone_self_rms=0.0,
                      das_gain_sigma=0.0, das_dead_fraction=0.0, ambient_rms=0.02)
    r = generate_noise(layout, FS, N, cfg, np.random.default_rng(0))
    assert np.allclose(rms(r.noise_only), 0.02, rtol=0.05)


def test_levels_instrument_only(layout):
    cfg = NoiseConfig(ambient_rms=0.0, shipping_rms=0.0, das_noise_rms=0.05, das_common_mode_rms=0.0,
                      hydrophone_self_rms=0.005, das_gain_sigma=0.0, das_dead_fraction=0.0)
    r = generate_noise(layout, FS, N, cfg, np.random.default_rng(0))
    out = r.noise_only
    assert np.allclose(rms(out[: layout.n_das]), 0.05, rtol=0.1)
    assert np.allclose(rms(out[layout.n_das:]), 0.005, rtol=0.1)


def test_distant_shipping_is_coherent_across_cable_ambient_is_not(layout):
    quiet = dict(das_noise_rms=0.0, das_common_mode_rms=0.0, hydrophone_self_rms=0.0, das_gain_sigma=0.0, das_dead_fraction=0.0)
    ship = generate_noise(layout, FS, N, NoiseConfig(ambient_rms=0.0, shipping_rms=0.02, n_shipping=1, **quiet), np.random.default_rng(0))
    amb = generate_noise(layout, FS, N, NoiseConfig(ambient_rms=0.02, shipping_rms=0.0, **quiet), np.random.default_rng(0))

    def best_lag_corr(x, i, j, max_lag=200):
        a, b = x[i].astype(float), x[j].astype(float)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        xc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), len(a))
        return max(np.abs(xc[:max_lag]).max(), np.abs(xc[-max_lag:]).max())

    assert best_lag_corr(ship.noise_only, 10, 11) > 0.9
    assert best_lag_corr(ship.noise_only, 0, 40) > 0.9   # plane wave: still coherent 400 m along the cable
    assert best_lag_corr(amb.noise_only, 10, 11) < 0.2


def test_distant_shipping_arrival_delay_matches_azimuth(layout):
    """A single plane wave from azimuth phi reaches channel i earlier by (x_i - x_0) . u / c."""
    cfg = NoiseConfig(ambient_rms=0.0, shipping_rms=0.02, n_shipping=1, shipping_azimuth_deg=(270.0, 270.0),
                      das_noise_rms=0.0, das_common_mode_rms=0.0, hydrophone_self_rms=0.0, das_gain_sigma=0.0,
                      das_dead_fraction=0.0)
    r = generate_noise(layout, FS, 60000, cfg, np.random.default_rng(0))
    out = r.noise_only
    # Source straight south (u = (0, -1)); the hydrophones are at y = 500, the cable at y = 300:
    # the cable is nearer the southern source, so it hears the wave first by 200 m / c.
    a, b = out[0], out[layout.n_das]
    xc = np.correlate(b, a, mode="full")
    lag = (np.argmax(xc) - (len(a) - 1)) / FS
    assert lag == pytest.approx(200.0 / 1500.0, abs=2.0 / FS)


def test_gain_and_dead_channels(layout):
    cfg = NoiseConfig(das_gain_sigma=0.2, das_dead_fraction=0.1)
    r = generate_noise(layout, FS, N, cfg, np.random.default_rng(0))
    assert r.dead.sum() == round(0.1 * layout.n_das) and not r.dead[layout.n_das:].any()
    assert np.all(r.noise_only[r.dead] == 0.0)
    assert np.all(r.gain[layout.n_das:] == 1.0)
    assert 0.1 < np.std(np.log(r.gain[: layout.n_das])) < 0.35


def test_common_mode_is_identical_on_all_das_channels(layout):
    cfg = NoiseConfig(ambient_rms=0.0, shipping_rms=0.0, das_noise_rms=0.0, hydrophone_self_rms=0.0,
                      das_common_mode_rms=0.03, das_gain_sigma=0.0, das_dead_fraction=0.0)
    out = generate_noise(layout, FS, N, cfg, np.random.default_rng(0)).noise_only
    assert np.allclose(out[0], out[20], atol=1e-6)
    assert np.all(out[layout.n_das:] == 0.0)


def test_apply_passes_signal_through_gain(layout):
    cfg = NoiseConfig(ambient_rms=0.0, shipping_rms=0.0, das_noise_rms=0.0, das_common_mode_rms=0.0,
                      hydrophone_self_rms=0.0, das_gain_sigma=0.2, das_dead_fraction=0.0)
    r = generate_noise(layout, FS, N, cfg, np.random.default_rng(0))
    clean = np.ones((layout.n_receivers, N))
    out = r.apply(clean)
    assert np.allclose(out[:, 0], r.gain, rtol=1e-5)

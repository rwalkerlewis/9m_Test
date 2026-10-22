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


# ----------------------------------------------------------------------
# harbor_v2 components (docs/noise.md)
# ----------------------------------------------------------------------

from scipy.signal import welch  # noqa: E402

from acoustic_sim.water import uniform_water_environment  # noqa: E402
from acoustic_sim.water.noise import (  # noqa: E402
    FibreConfig,
    HarborV2Config,
    InterrogatorConfig,
    ShrimpConfig,
    SurfaceConfig,
    fibre_noise,
    interrogator_noise,
    land_edge_points,
    shrimp_noise,
    surface_noise,
)

FS2 = 3000.0
N2 = 30000


@pytest.fixture(scope="module")
def layout2():
    cable = Cable("c", np.array([[50.0, 100.0], [350.0, 100.0]]), channel_spacing=4.0, das_mode="pressure")
    return SensorLayout(cables=[cable], hydrophones=np.array([[100.0, 200.0], [300.0, 200.0]]))


def _slope(x, fs, lo, hi):
    f, p = welch(x, fs=fs, nperseg=4096, axis=-1)
    p = np.mean(np.atleast_2d(p), axis=0)
    m = (f >= lo) & (f <= hi)
    return np.polyfit(np.log10(f[m]), np.log10(p[m]), 1)[0], f, p


def test_surface_noise_has_the_knudsen_slope_and_level(layout2):
    cfg = SurfaceConfig(enabled=True, sea_state=2, slope=1.7, knee_hz=300.0, variation_db=0.0)
    x = surface_noise(layout2, FS2, N2, cfg, np.random.default_rng(1))
    slope, f, p = _slope(x[:20], FS2, 400.0, 1400.0)
    assert slope == pytest.approx(-1.7, abs=0.1)
    level = 10 * np.log10(np.interp(1000.0, f, p) / 1e-12)
    assert level == pytest.approx(cfg.levels_1khz_db[2], abs=1.0)
    flat, _, _ = _slope(x[:20], FS2, 50.0, 250.0)                # below the knee
    assert abs(flat) < 0.3


def test_surface_noise_level_wanders_over_minutes(layout2):
    cfg = SurfaceConfig(enabled=True, variation_db=3.0, variation_s=60.0)
    n = int(600 * 200.0)
    x = surface_noise(layout2, 200.0, n, cfg, np.random.default_rng(2))[0]
    per_10s = 10 * np.log10(np.mean(x.reshape(-1, 2000) ** 2, axis=1))
    assert 1.5 < np.std(per_10s) < 5.0


def _harbor_with_quay():
    env = uniform_water_environment(400.0, 300.0, 1.0, 1500.0)
    env.water[280:, :] = False                                     # a quay along the north edge
    return env


def test_shrimp_clicks_poisson_heavy_tailed_and_from_the_edge(layout2):
    env = _harbor_with_quay()
    pts = land_edge_points(env, 5.0)
    assert len(pts) and np.all(np.abs(pts[:, 1] - 279.5) < 1.0)    # the water side of the quay
    cfg = ShrimpConfig(enabled=True, rate_per_m=0.5, latitude_deg=30.0)
    x, info = shrimp_noise(layout2, FS2, N2, cfg, np.random.default_rng(3), env)
    mean = cfg.rate_per_m * info["edge_length_m"] * N2 / FS2
    assert abs(info["clicks"] - mean) < 4 * np.sqrt(mean)
    h = x[-1]                                                      # a hydrophone
    k = np.mean(h ** 4) / np.mean(h ** 2) ** 2
    assert k > 10.0                                                # heavy tailed (3 for a Gaussian)


def test_shrimp_click_arrives_after_range_over_c(layout2):
    env = _harbor_with_quay()
    cfg = ShrimpConfig(enabled=True, rate_per_m=1e-9, latitude_deg=30.0)
    from acoustic_sim.water import noise as nz

    class OneClick:                                                # the generator, with exactly one click drawn
        def __init__(self, g):
            self.g = g

        def poisson(self, lam):
            return 1

        def uniform(self, lo, hi, size=None):                    # the click at 0.1 s
            return np.full(size, 0.1)

        def __getattr__(self, name):
            return getattr(self.g, name)

    x, info = nz.shrimp_noise(layout2, FS2, 3000, cfg, OneClick(np.random.default_rng(4)), env)
    assert info["clicks"] == 1
    arrivals = np.array([np.flatnonzero(r)[0] for r in x if np.any(r)])
    rec = layout2.receivers[[i for i, r in enumerate(x) if np.any(r)]]
    # the source is the one point that fits every arrival: the differences of arrival match differences of range
    pts = land_edge_points(env, cfg.edge_spacing_m)
    best = min(pts, key=lambda p: np.std(arrivals / FS2 - np.hypot(*(rec - p).T) / 1500.0))
    resid = arrivals / FS2 - np.hypot(*(rec - best).T) / 1500.0
    assert np.ptp(resid) <= 1.5 / FS2


def test_shrimp_off_outside_their_latitudes_or_without_one():
    assert ShrimpConfig(enabled=True, latitude_deg=55.0).off_reason().startswith("latitude 55")
    assert ShrimpConfig(enabled=True).off_reason().startswith("the scenario gives no latitude")
    assert ShrimpConfig(enabled=True, latitude_deg=-30.0).off_reason() is None


def test_fibre_traffic_correlated_along_the_cable_with_a_delay(layout2):
    """The ground motion of a vehicle reaches a channel 20 m further along the cable 20 m / 300 m/s later, and is felt
    only near the vehicle (envelope of the correlation length moving at the vehicle's speed)."""
    cfg = FibreConfig(enabled=True, drift_rms=0.0, traffic_rms=1.0, traffic_speed=300.0, vehicle_speed=15.0,
                      vehicles_per_min=30.0, corr_length_m=40.0, traffic_band_hz=(3.0, 30.0))
    fs, n = 500.0, 60000
    x = fibre_noise(layout2, fs, n, cfg, np.random.default_rng(5))
    a, b = x[20], x[25]                                            # 20 m apart
    lags = np.arange(-60, 61)
    cc = np.array([np.dot(a[60:-60], b[60 + l:len(b) - 60 + l]) for l in lags]) / np.sqrt(np.dot(a, a) * np.dot(b, b))
    lag = lags[int(np.argmax(cc))] / fs
    assert lag == pytest.approx(20.0 / 300.0, abs=2.5 / fs)
    assert cc.max() > 0.6
    e20 = np.convolve(x[20] ** 2, np.ones(500) / 500, "same")
    e70 = np.convolve(x[70] ** 2, np.ones(500) / 500, "same")
    peak20, peak70 = np.argmax(e20), np.argmax(e70)
    assert (peak70 - peak20) / fs == pytest.approx(200.0 / 15.0, abs=4.0) or np.corrcoef(e20, e70)[0, 1] < 0.9


def test_interrogator_noise_rises_at_low_frequency_and_fades(layout2):
    cfg = InterrogatorConfig(enabled=True, floor_rms=0.05, corner_hz=50.0, slope=1.0, fade_fraction=0.05, fade_s=1.0,
                             fade_factor=10.0)
    x, fade = interrogator_noise(layout2, FS2, N2, cfg, np.random.default_rng(6))
    assert fade.mean() == pytest.approx(0.05, abs=0.02)
    clean = np.where(fade, np.nan, x)
    f, p = welch(np.nan_to_num(clean[~fade.any(axis=1)]), fs=FS2, nperseg=4096, axis=-1)
    p = p.mean(axis=0)
    low, high = np.interp(5.0, f, p), np.interp(500.0, f, p)
    assert low / high == pytest.approx((1 + 50 / 5.0) / (1 + 50 / 500.0), rel=0.35)
    faded = x[fade].std() / x[~fade].std()
    assert faded == pytest.approx(10.0, rel=0.3)


def test_harbor_v2_config_round_trips_and_default_model_unchanged(layout):
    d = {"model": "harbor_v2", "harbor_v2": {"surface": {"enabled": True, "sea_state": 3}, "shrimp": {"enabled": False}}}
    cfg = NoiseConfig(**d)
    assert isinstance(cfg.harbor_v2, HarborV2Config) and cfg.harbor_v2.surface.sea_state == 3
    old = generate_noise(layout, FS, N, NoiseConfig(), np.random.default_rng(0))
    v2_off = generate_noise(layout, FS, N, NoiseConfig(model="harbor_v2"), np.random.default_rng(0))
    assert np.array_equal(old.noise_only, v2_off.noise_only)       # every component off: the old model, bit for bit

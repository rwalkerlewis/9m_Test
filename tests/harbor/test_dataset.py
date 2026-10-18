"""M2: end-to-end scenario simulation and dataset mixing."""

import numpy as np
import pytest

from acoustic_sim.harbor.scenario import default_vessels
from acoustic_sim.harbor import (
    Cable,
    HarborDataset,
    HarborFDTDConfig,
    HarborScenarioConfig,
    NoiseConfig,
    SensorLayout,
    VesselConfig,
    assemble_dataset,
    build_track,
    render_vessel_gathers,
    simulate_scenario,
    uniform_water_environment,
)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    env = uniform_water_environment(1000, 750, 5.0, 1500.0)
    cable = Cable("cable1", np.array([[250.0, 250.0], [750.0, 250.0]]), channel_spacing=25.0)
    layout = SensorLayout(cables=[cable], hydrophones=np.array([[300.0, 500.0], [700.0, 500.0]]))
    cfg = HarborScenarioConfig(
        fdtd=HarborFDTDConfig(fs=600.0, total_time=6.0),
        vessels=[
            VesselConfig("ship", "ship", [[300, 400], [700, 400]], speed=40.0, level_pa=1.0, seed=1),
            VesselConfig("moored", "ship", [[500, 550]], level_pa=0.3, seed=2),
            VesselConfig("uuv", "uuv", [[400, 450]], level_pa=0.1, seed=3),
        ],
        noise=NoiseConfig(das_dead_fraction=0.1),
        seed=5,
    )
    cache = tmp_path_factory.mktemp("cache")
    tracks = {v.id: build_track(v, 6.0, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=cache)
    return env, layout, cfg, tracks, gathers, cache


def test_clean_is_level_weighted_sum_of_unit_gathers(setup):
    env, layout, cfg, tracks, gathers, _ = setup
    ds = assemble_dataset(cfg, layout, tracks, gathers)
    expected = sum(v.level_pa * gathers[v.id].traces.astype(np.float64) for v in cfg.vessels)
    assert np.allclose(ds.clean, expected, rtol=1e-4, atol=1e-6 * np.abs(expected).max())
    assert ds.traces.shape == (layout.n_receivers, cfg.fdtd.n_steps)


def test_level_override_and_removal_without_new_fdtd(setup):
    env, layout, cfg, tracks, gathers, cache = setup
    n_files = len(list(cache.glob("gather_*.npz")))
    full = assemble_dataset(cfg, layout, tracks, gathers)
    no_uuv = assemble_dataset(cfg, layout, tracks, gathers, levels={"uuv": 0.0})
    loud_uuv = assemble_dataset(cfg, layout, tracks, gathers, levels={"uuv": 0.2})
    uuv_part = 0.1 * gathers["uuv"].traces
    assert np.allclose(full.clean - no_uuv.clean, uuv_part, rtol=1e-3, atol=1e-6 * np.abs(uuv_part).max())
    assert np.allclose(loud_uuv.clean - full.clean, uuv_part, rtol=1e-3, atol=1e-6 * np.abs(uuv_part).max())
    assert loud_uuv.tracks["uuv"].level_pa == 0.2
    assert len(list(cache.glob("gather_*.npz"))) == n_files


def test_cache_reuse(setup):
    env, layout, cfg, tracks, gathers, cache = setup
    assert len(list(cache.glob("gather_*.npz"))) == 3
    again = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=cache)
    assert len(list(cache.glob("gather_*.npz"))) == 3
    assert np.array_equal(again["ship"].traces, gathers["ship"].traces)


def test_noise_seeded_and_ais_for_ships_only(setup):
    env, layout, cfg, tracks, gathers, _ = setup
    a = assemble_dataset(cfg, layout, tracks, gathers, seed=1)
    b = assemble_dataset(cfg, layout, tracks, gathers, seed=1)
    c = assemble_dataset(cfg, layout, tracks, gathers, seed=2)
    assert np.array_equal(a.traces, b.traces) and not np.allclose(a.traces, c.traces)
    assert a.ais.vessels == ["moored", "ship"]
    assert a.unknown_ids == ["uuv"]
    assert a.dead.sum() == round(0.1 * layout.n_das)
    assert np.all(a.traces[a.dead] == 0.0)


def test_removing_a_ship_removes_its_ais(setup):
    env, layout, cfg, tracks, gathers, _ = setup
    ds = assemble_dataset(cfg, layout, tracks, gathers, levels={"ship": 0.0})
    assert ds.ais.vessels == ["moored"]


def test_fixed_noise_realisation_isolates_signal_change(setup):
    from acoustic_sim.harbor import generate_noise

    env, layout, cfg, tracks, gathers, _ = setup
    noise = generate_noise(layout, cfg.fdtd.fs, cfg.fdtd.n_steps, cfg.noise, np.random.default_rng(9))
    a = assemble_dataset(cfg, layout, tracks, gathers, noise=noise, levels={"uuv": 0.0})
    b = assemble_dataset(cfg, layout, tracks, gathers, noise=noise, levels={"uuv": 0.1})
    live = ~a.dead
    diff = (b.traces - a.traces)[live]
    expected = (noise.gain[:, None] * 0.1 * gathers["uuv"].traces)[live]
    assert np.allclose(diff, expected, rtol=1e-2, atol=1e-5)


def test_save_load_roundtrip(setup, tmp_path):
    env, layout, cfg, tracks, gathers, _ = setup
    ds = assemble_dataset(cfg, layout, tracks, gathers)
    path = tmp_path / "ds.npz"
    ds.save(path)
    back = HarborDataset.load(path)
    assert np.array_equal(back.traces, ds.traces) and np.array_equal(back.clean, ds.clean)
    assert back.fs == ds.fs and back.layout.n_receivers == layout.n_receivers
    assert set(back.tracks) == set(ds.tracks) and back.tracks["uuv"].ais is False
    assert np.array_equal(back.tracks["ship"].xy, ds.tracks["ship"].xy)
    assert len(back.ais) == len(ds.ais) and np.array_equal(back.ais.t, ds.ais.t)
    assert np.array_equal(back.dead, ds.dead)
    assert back.cfg.to_dict() == ds.cfg.to_dict()


def test_simulate_scenario_end_to_end(setup):
    env, layout, cfg, _, _, cache = setup
    ds = simulate_scenario(cfg, env=env, layout=layout, cache_dir=cache)
    assert ds.traces.shape[0] == layout.n_receivers and np.isfinite(ds.traces).all()
    assert ds.tracks["uuv"].level_pa == 0.1


def test_scenario_config_roundtrip(tmp_path):
    cfg = HarborScenarioConfig()
    cfg.noise.ambient_rms = 0.03
    cfg.ais.pos_sigma = 12.0
    cfg.seed = 17
    path = tmp_path / "cfg.json"
    cfg.save(path)
    back = HarborScenarioConfig.load(path)
    assert back.to_dict() == cfg.to_dict()
    assert [v.id for v in back.vessels] == [v.id for v in default_vessels()]
    assert back.ais.outage_reports == (3, 6)

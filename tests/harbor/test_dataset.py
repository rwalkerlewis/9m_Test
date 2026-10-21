"""M2: end-to-end scenario simulation and dataset mixing."""

import numpy as np
import pytest

from acoustic_sim.water.scenario import default_vessels
from acoustic_sim.water import (
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
    from acoustic_sim.water import generate_noise

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


def test_combined_bodies_gather(tmp_path):
    """``combined_bodies=True``: one scattered gather per source with every body present.  With
    one body it is the per body gather (same FDTD run); with two it differs from the sum of the
    per body gathers (multiple scattering and the bodies' joint mask), and the mixer uses it."""
    env = uniform_water_environment(500, 400, 5.0, 1500.0)
    cable = Cable("cable1", np.array([[100.0, 100.0], [400.0, 100.0]]), channel_spacing=25.0)
    layout = SensorLayout(cables=[cable], hydrophones=np.array([[250.0, 300.0]]))
    body = dict(level_pa=0.0, body_length=15.0, body_width=10.0, body_kind="soft")
    vessels = [
        VesselConfig("ship", "ship", [[150, 300]], level_pa=1.0, seed=1),
        VesselConfig("b1", "uuv", [[250, 200]], **body, seed=2),
        VesselConfig("b2", "static", [[262, 200]], **body, seed=3),
    ]
    cfg = HarborScenarioConfig(fdtd=HarborFDTDConfig(fs=600.0, total_time=2.0), vessels=vessels, seed=5)
    tracks = {v.id: build_track(v, 2.0, env) for v in cfg.vessels}
    one = HarborScenarioConfig(fdtd=cfg.fdtd, vessels=vessels[:2], seed=5)
    per_body = render_vessel_gathers(one, env, layout, tracks, cache_dir=tmp_path)
    combined_one = render_vessel_gathers(one, env, layout, tracks, cache_dir=tmp_path, combined_bodies=True)
    assert set(combined_one) == {"ship", "ship>bodies"}
    assert np.array_equal(combined_one["ship>bodies"].traces, per_body["ship>b1"].traces)
    both = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=tmp_path)
    combined = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=tmp_path, combined_bodies=True)
    assert set(both) == {"ship", "ship>b1", "ship>b2"} and set(combined) == {"ship", "ship>bodies"}
    single_sum = both["ship>b1"].traces.astype(float) + both["ship>b2"].traces.astype(float)
    joint = combined["ship>bodies"].traces.astype(float)
    assert np.isfinite(joint).all() and np.abs(joint).max() > 0
    assert not np.allclose(joint, single_sum, rtol=1e-3, atol=1e-6 * np.abs(joint).max())
    ds = assemble_dataset(cfg, layout, tracks, combined, keep_components=True)
    assert set(ds.components) == {"direct:ship", "scattered:ship>bodies", "noise"}
    expected = combined["ship"].traces.astype(float) + joint
    assert np.allclose(ds.clean, expected, rtol=1e-4, atol=1e-6 * np.abs(expected).max())
    with pytest.raises(ValueError):
        assemble_dataset(cfg, layout, tracks, combined, scatter_gain={"b1": 0.0})


def test_build_info_counts_runs_and_cached_gathers(setup, tmp_path):
    """``build.json``: the cached gathers counted before and after a render give the FDTD runs; a render read wholly from
    the cache counts none."""
    import json

    from acoustic_sim.water.dataset import count_cached_gathers, write_build_info

    env, layout, cfg, tracks, gathers, cache = setup
    assert count_cached_gathers(None) == 0 and count_cached_gathers(tmp_path / "missing") == 0
    n = count_cached_gathers(cache)
    assert n == len(gathers)
    again = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=cache)
    assert count_cached_gathers(cache) - n == 0 and set(again) == set(gathers)
    info = write_build_info(tmp_path / "build.json", cfg, env, len(gathers), 12.0, 3, note="x")
    assert json.loads((tmp_path / "build.json").read_text()) == info
    assert info["fdtd_runs"] == 3 and info["n_steps"] == cfg.fdtd.n_steps and info["grid"] == [env.nx, env.ny]
    assert info["fdtd_wall_s_per_run_per_simulated_s"] == pytest.approx(12.0 / 3 / 6.0) and info["note"] == "x"
    assert write_build_info(tmp_path / "b.json", cfg, env, 0, 0.1, 0)["fdtd_wall_s_per_run_per_simulated_s"] is None

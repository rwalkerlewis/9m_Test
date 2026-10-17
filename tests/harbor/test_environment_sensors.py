"""M0: environment, sensor layout and scenario-config tests."""

import json

import numpy as np
import pytest

from acoustic_sim.harbor import (
    Cable,
    HarborScenarioConfig,
    SensorLayout,
    build_environment,
    build_layout,
    build_synthetic_harbor,
    uniform_water_environment,
)


@pytest.fixture(scope="module")
def harbor():
    return build_synthetic_harbor()


def test_harbor_features(harbor):
    assert harbor.model.values.shape == (361, 481)
    water = lambda x, y: bool(harbor.is_water([[x, y]])[0])
    assert water(1000, 1000)          # basin
    assert water(1350, 520)           # entrance gap
    assert not water(800, 539)        # breakwater
    assert not water(815, 1300)       # pier
    assert not water(1200, 1700)      # north land
    assert water(1000, 300)           # open sea
    assert 0.5 < harbor.water.mean() < 0.8


def test_margin_and_bounds(harbor):
    assert harbor.is_water([[1000, 1000]], margin=50)[0]
    assert not harbor.is_water([[1000, 1000]], margin=2000)[0]
    assert not harbor.is_water([[-10, 100]])[0]
    assert harbor.distance_to_land([[815, 1300 - 20]])[0] <= 25.0


def test_sound_speed_in_water_range(harbor):
    c = harbor.model.values[harbor.water]
    assert 1480.0 <= c.min() and c.max() <= 1500.0


def test_uniform_environment_has_no_land():
    env = uniform_water_environment(500, 400, 5.0)
    assert env.water.all()
    assert np.allclose(env.model.values, 1500.0)


def test_cable_spacing_and_tangents():
    cable = Cable("c", np.array([[0, 0], [100, 0], [100, 100]]), channel_spacing=10.0)
    assert cable.length == pytest.approx(200.0)
    assert cable.n_channels == 21
    pos = cable.positions
    assert np.allclose(np.hypot(*np.diff(pos, axis=0).T), 10.0)
    t = cable.tangents
    assert np.allclose(t[0], [1, 0]) and np.allclose(t[-1], [0, 1])
    assert np.allclose(np.hypot(*t.T), 1.0)


def test_cable_rejects_bad_input():
    with pytest.raises(ValueError):
        Cable("c", np.array([[0, 0]]))
    with pytest.raises(ValueError):
        Cable("c", np.array([[0, 0], [1, 0]]), channel_spacing=0)


def test_layout_slices_and_split():
    layout = SensorLayout(
        cables=[Cable("a", np.array([[0, 0], [50, 0]]), 10.0), Cable("b", np.array([[0, 10], [30, 10]]), 10.0)],
        hydrophones=np.array([[5.0, 5.0]]),
    )
    assert layout.n_das == 6 + 4 and layout.n_receivers == 11
    assert layout.receivers.shape == (11, 2)
    traces = np.arange(11 * 3, dtype=float).reshape(11, 3)
    parts = layout.split(traces)
    assert parts["a"].shape == (6, 3) and parts["b"].shape == (4, 3) and parts["hydrophones"].shape == (1, 3)
    assert np.array_equal(parts["hydrophones"][0], traces[10])
    with pytest.raises(ValueError):
        layout.split(traces[:5])


def test_layout_validate_flags_land(harbor):
    bad = SensorLayout(hydrophones=np.array([[1000.0, 1000.0], [1200.0, 1700.0]]))
    with pytest.raises(ValueError, match="on land"):
        bad.validate(harbor)


def test_default_scenario_layout_is_in_water(harbor):
    cfg = HarborScenarioConfig()
    layout = build_layout(cfg, harbor)  # validates
    assert layout.n_das > 100 and layout.n_hydrophones == 4


def test_config_json_roundtrip(tmp_path):
    cfg = HarborScenarioConfig()
    cfg.fdtd.total_time = 30.0
    cfg.fdtd.shore_loss_max = 0.3
    path = tmp_path / "scenario.json"
    cfg.save(path)
    json.loads(path.read_text())
    again = HarborScenarioConfig.load(path)
    assert again.to_dict() == cfg.to_dict()
    assert build_environment(again).model.values.shape == (361, 481)


def test_config_ignores_missing_sections():
    cfg = HarborScenarioConfig.from_dict({"fdtd": {"total_time": 5.0}})
    assert cfg.fdtd.total_time == 5.0 and len(cfg.cables) == 1 and len(cfg.hydrophones) == 4

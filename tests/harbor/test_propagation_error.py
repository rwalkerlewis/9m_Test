"""``harbor_propagation_error.py``: the straight path travel time and the criterion, on inputs with known answers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from acoustic_sim.water import HarborScenarioConfig, uniform_water_environment

_P = Path(__file__).resolve().parents[2] / "examples" / "harbor" / "01_datasets" / "harbor_propagation_error.py"
_spec = importlib.util.spec_from_file_location("harbor_propagation_error", _P)
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)


def test_straight_path_time_uniform_and_two_speeds():
    env = uniform_water_environment(200.0, 100.0, 1.0, 1500.0)
    a, b = np.array([20.0, 50.0]), np.array([180.0, 50.0])
    assert pe.straight_path_time(env, a, b) == pytest.approx(160.0 / 1500.0, rel=1e-12)
    # East half at 1400 m/s: 80 m at each speed (the boundary on a node column, a fifth of a cell of rounding).
    x0 = env.model.x[0]
    env.model.values[:, np.flatnonzero(env.model.x - x0 >= 100.0)] = 1400.0
    expect = 80.0 / 1500.0 + 80.0 / 1400.0
    assert pe.straight_path_time(env, a, b) == pytest.approx(expect, abs=1.0 * (1 / 1400 - 1 / 1500))


def test_criterion_by_grid_and_kind():
    cfg = HarborScenarioConfig()
    cfg.environment.dx = 0.5
    assert pe.criterion_ms(cfg) == 0.2
    cfg.environment.dx, cfg.environment.kind = 1.0, "simple"
    assert pe.criterion_ms(cfg) == 0.1
    cfg.environment.dx, cfg.environment.kind = 5.0, "synthetic"
    assert pe.criterion_ms(cfg) is None

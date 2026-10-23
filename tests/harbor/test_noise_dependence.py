"""The second noise model of ``scripts/reproduce.py`` and the sweep of its assumed numbers (prompt
``noise_dependence``): the ``harbor_v2`` mix at its documented levels is reproducible from the same seed (and differs
with another), the sweep's documented values are the model's, and its ``results.json`` carries no criterion and every
parameter value used."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from acoustic_sim.water import HarborScenarioConfig, build_environment, build_layout
from acoustic_sim.water.noise import HarborV2Config, generate_noise

from .test_ci_end_to_end import SCENARIO

ROOT = Path(__file__).resolve().parents[2]
DET = ROOT / "examples" / "harbor" / "05_detection"


def _module(name: str):
    sys.path.insert(0, str(DET))
    spec = importlib.util.spec_from_file_location(name, DET / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _noise(seed: int) -> np.ndarray:
    hnm = _module("harbor_noise_model")
    cfg = HarborScenarioConfig.from_dict(SCENARIO)
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    cfg.noise.model = "harbor_v2"
    cfg.noise.harbor_v2 = HarborV2Config.from_dict(hnm.v2_config(hnm.COMPONENTS, False))
    n = 3000
    return generate_noise(layout, cfg.fdtd.fs, n, cfg.noise, np.random.default_rng(seed), env=env).noise_only


def test_the_harbor_v2_mix_is_reproducible_from_its_seed():
    a, b, c = _noise(1), _noise(1), _noise(2)
    np.testing.assert_array_equal(a, b)
    assert np.abs(a - c).max() > 0.1 * np.abs(a).max()


def test_the_sweep_records_every_value_and_no_criterion(tmp_path):
    sw = _module("harbor_noise_sweep")
    doc = HarborV2Config()
    for key, vals in sw.PARAMETERS.items():
        comp, field = key.split(".")
        assert len(vals) == 3 and np.isclose(vals[1], getattr(getattr(doc, comp), field)), key
        assert vals[0] < vals[1] < vals[2]
    assert np.isclose(sw.MACHINERY[1][1], doc.machinery.level_pa) and sw.MACHINERY[1][0] is None
    sys.argv = ["harbor_noise_sweep.py", "--plot-only", "--out", str(tmp_path)]
    sw.main()
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["criterion"].startswith("none")
    assert res["parameters"] == {k: [float(x) for x in v] for k, v in sw.PARAMETERS.items()}
    assert res["machinery"]["values"][0] is None

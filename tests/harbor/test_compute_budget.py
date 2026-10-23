"""The compute budget (prompt ``compute_budget``): every stage of the harbor picture within its per frame budget on the
tiny CI scenario (``ci_processing.json``, ``budget``), so a change that doubles a stage's cost fails here; and the
reductions that must not change what the chain computes (the noise spectra on the device, the beam taken from the
aligned channels)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from acoustic_sim.water import HarborScenarioConfig, simulate_scenario
from acoustic_sim.water.bistatic import BistaticConfig, _aligned_rows, make_reference, residual_psd
from acoustic_sim.water.profile import PROFILE, over_budget, reference_ms
from acoustic_sim.water.stream import load_processing, run

from .test_ci_end_to_end import CONFIG, SCENARIO
from .test_joint_removal import scene  # noqa: F401  (the analytic two ship scene)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("budget")
    return simulate_scenario(HarborScenarioConfig.from_dict(SCENARIO), cache_dir=d / "cache", out=d / "ci.npz")


def test_every_stage_within_its_budget_on_the_tiny_scenario(tiny):
    budget = json.loads(CONFIG.read_text())["budget"]
    scale = reference_ms() / budget["reference_ms"]
    PROFILE.start()
    try:
        res = run(tiny.observation, load_processing(CONFIG), use_cuda=False)
    finally:
        rep = PROFILE.stop()
    over = over_budget(rep, len(res.picture.p_frames), budget["ms_per_frame"], scale=scale)
    assert not over, (scale, over)
    assert set(budget["ms_per_frame"]) <= set(rep["seconds"]), set(budget["ms_per_frame"]) - set(rep["seconds"])


def test_a_doubled_stage_is_caught():
    rep = {"seconds": {"passive: map": 2.0, "echo: map": 1.0}}
    assert over_budget(rep, 10, {"passive: map": 150.0, "echo: map": 150.0}) == {"passive: map": {"ms_per_frame": 200.0,
                                                                                                 "budget_ms": 150.0}}
    assert over_budget(rep, 10, {"passive: map": 150.0}, scale=2.0) == {}


def test_the_beam_from_the_aligned_channels_is_make_reference(scene):  # noqa: F811
    """The beam ``estimate_reference`` now takes as the mean of the aligned channels equals ``make_reference``."""
    layout, waves, directs, nz = scene
    x = directs[0] + directs[1] + nz
    n = x.shape[1]
    bc = BistaticConfig(c=1500.0, eca_doppler=True, eca_images=False)
    ship_t = np.repeat(np.array([[170.0, 230.0]]), n, axis=0)
    ship_b = ship_t[:: int(round(bc.eca_block_s * 3000.0))]
    chans = np.arange(layout.n_receivers)
    a = make_reference(x, layout, 3000.0, ship_b, bc.eca_block_s, bc, chans, ship_t=ship_t)
    Y = _aligned_rows(x[chans], layout.receivers[chans], ship_t, 3000.0, bc)
    b = np.asarray(Y).mean(axis=0)
    assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(a))


def test_noise_spectra_on_the_device_match_the_host():
    from ._gpu import HAS_GPU

    if not HAS_GPU:
        pytest.skip("no GPU")
    import cupy as cp

    rng = np.random.default_rng(1)
    r = rng.standard_normal((40, 6000)) * np.linspace(1, 3, 40)[:, None]
    f = np.arange(2.0, 0.49 * 3000.0, 2.0)
    a = residual_psd(r, 3000.0, f, 0.5)
    b = residual_psd(r, 3000.0, f, 0.5, xp=cp)
    assert np.max(np.abs(a - b) / a) < 1e-9

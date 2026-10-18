"""The bistatic chain on a small analytic (free field, exact delay) case: ``eca_stage`` against
the inline code it replaced, cancellation depth and echo preservation, and ``run_bistatic``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from acoustic_sim.harbor import Cable, SensorLayout, uniform_water_environment
from acoustic_sim.harbor.ais import AISReports
from acoustic_sim.harbor.bistatic import (
    BistaticConfig,
    ReplicaModel,
    Truth,
    _fractional_delay,
    clutter_images,
    eca_remove,
    eca_stage,
    make_reference,
    run_bistatic,
    ship_positions,
    stft,
)

FS = 2000.0
T = 3.0
C = 1495.0
F_MAX = 500.0
SHIP = np.array([150.0, 250.0])
BODY = np.array([260.0, 180.0])     # off the ship to channel lines: every echo is > 40 m (27 ms) after the direct arrival


@pytest.fixture(scope="module")
def scene():
    env = uniform_water_environment(400.0, 400.0, 5.0, C)
    cable = Cable("cable1", np.array([[100.0, 100.0], [210.0, 100.0]]), channel_spacing=10.0)
    layout = SensorLayout(cables=[cable], hydrophones=np.array([[60.0, 200.0], [300.0, 80.0]]))
    n = int(T * FS)
    rng = np.random.default_rng(3)
    sos = butter(4, [80.0, 450.0], btype="bandpass", fs=FS, output="sos")
    s = sosfiltfilt(sos, rng.standard_normal(n))
    tt = np.arange(n) / FS
    s *= np.clip((tt - 0.3) / 0.1, 0.0, 1.0) * np.clip((T - 0.3 - tt) / 0.1, 0.0, 1.0)   # silent ends: the circular delay must not wrap
    s /= s.std()
    rec = layout.receivers
    r_d = np.hypot(*(rec - SHIP).T)
    r_e = np.hypot(*(BODY - SHIP)) + np.hypot(*(rec - BODY).T)
    direct = _fractional_delay(np.repeat(s[None], len(rec), axis=0), r_d / C, FS) / np.sqrt(r_d)[:, None]
    echo = 0.2 * _fractional_delay(np.repeat(s[None], len(rec), axis=0), r_e / C, FS) / np.sqrt(r_e)[:, None]
    noise = 0.002 * rng.standard_normal(direct.shape)
    traces = direct + echo + noise
    k = int(T / 0.5)
    t_ais = np.arange(k) * 0.5
    ais = AISReports(vessel=np.array(["ship"] * k), t=t_ais, xy=np.repeat(SHIP[None], k, axis=0), sog=np.zeros(k), cog=np.zeros(k))
    return env, layout, traces, direct, echo, s, ais


def _depth_db(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.sum(before ** 2, axis=1) / np.maximum(np.sum(after ** 2, axis=1), 1e-30))


def _old_inline(traces, layout, fs, env, sp, ship, bc, oracle_signal, comps):
    """The ECA block of ``run_bistatic`` before it was factored out (verbatim)."""
    nb = int(round(bc.eca_block_s * fs))
    n_blocks = int(np.ceil(traces.shape[1] / nb))
    t_blocks = (np.arange(n_blocks) + 0.5) * bc.eca_block_s
    ship_b = np.column_stack([np.interp(t_blocks, sp.times, ship.refined[:, 0]), np.interp(t_blocks, sp.times, ship.refined[:, 1])])
    if bc.ref_channels == "all":
        chans = np.flatnonzero(sp.weights > 0)
    else:
        sl = layout.slices[bc.ref_channels]
        chans = np.arange(sl.start, sl.stop)[sp.weights[sl] > 0]
    waveform = make_reference(traces, layout, fs, ship_b, bc.eca_block_s, bc, chans, oracle_signal)
    images = [clutter_images(env, sxy) for sxy in ship_b] if bc.eca_images else []
    return eca_remove([traces] + comps, waveform, layout, fs, ship_b, images, bc.eca_block_s, bc), waveform


@pytest.mark.parametrize("reference", ["oracle", "data"])
def test_eca_stage_matches_old_inline_code(scene, reference):
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(reference=reference, removal="eca", c=C, refine=False)
    sp = stft(traces, layout, FS, bc, F_MAX)
    model = ReplicaModel(layout, bc.c)
    model.weights = sp.weights
    ship = ship_positions(sp, ais, model, bc)
    oracle = s if reference == "oracle" else None
    outs, wf = eca_stage(traces, layout, FS, env, sp, ship, bc, oracle, [direct, echo])
    outs_old, wf_old = _old_inline(traces, layout, FS, env, sp, ship, bc, oracle, [direct, echo])
    assert np.array_equal(wf, wf_old)
    assert len(outs) == 3 and all(np.array_equal(a, b) for a, b in zip(outs, outs_old))


@pytest.mark.parametrize("reference, min_depth_db", [("oracle", 25.0), ("data", 12.0)])
def test_eca_attenuates_direct_and_keeps_echo(scene, reference, min_depth_db):
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(reference=reference, removal="eca", c=C, refine=False)
    sp = stft(traces, layout, FS, bc, F_MAX)
    model = ReplicaModel(layout, bc.c)
    model.weights = sp.weights
    ship = ship_positions(sp, ais, model, bc)
    outs, _ = eca_stage(traces, layout, FS, env, sp, ship, bc, s if reference == "oracle" else None, [direct, echo])
    depth, loss = _depth_db(direct, outs[1]), _depth_db(echo, outs[2])
    assert np.median(depth) > min_depth_db, depth
    assert np.median(loss) < 1.0, loss
    # The cleaned measurement has lost the direct path but not the echo.
    assert np.sum(outs[0] ** 2) < 0.1 * np.sum(traces ** 2)


def test_run_bistatic_end_to_end_reports_eca_depth_and_finds_the_body(scene):
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(reference="oracle", removal="eca", method="coherent", c=C, grid_dx=5.0, edge=20.0,
                        exclusion_radius=30.0, refine=False)
    sp = stft(traces, layout, FS, bc, F_MAX)
    truth = Truth(ship_xy=np.repeat(SHIP[None], len(sp.times), axis=0), components={"direct:ship": direct, "scattered:ship>body": echo})
    run = run_bistatic(traces, layout, FS, env, ais, bc, F_MAX, oracle_signal=s, truth=truth)
    assert run.eca is not None
    assert np.median(run.eca.removed_db) > 25.0
    assert np.median(run.eca.echo_lost_db) < 1.0
    assert run.scan.B.shape == (len(run.grid), len(run.sp.times))
    est = run.grid[np.argmax(run.scan.B, axis=0)]
    err = np.hypot(*(est - BODY).T)
    assert np.median(err) < 15.0, err

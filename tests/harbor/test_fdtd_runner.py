"""M1: FDTD per-source runner validation (physics, linearity, cache, shoreline)."""

import numpy as np
import pytest
from scipy.special import hankel1

from acoustic_sim.water import (
    Cable,
    HarborFDTDConfig,
    HarborFDTDSolver,
    SensorLayout,
    cfl_dt,
    simulate_source,
    uniform_water_environment,
)
from acoustic_sim.water.environment import _finish_environment, segment_polygon
from acoustic_sim.sources import StaticSource

C0 = 1500.0
DX = 5.0
FS = 600.0


def ricker(cfg, f0, delay):
    t = np.arange(cfg.n_steps) / cfg.fs - delay
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def lag_seconds(ref, sig, fs):
    """Delay of *sig* relative to *ref* from the cross-correlation peak (parabolic interpolation)."""
    xc = np.correlate(sig, ref, mode="full")
    k = int(np.argmax(xc))
    y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
    k_f = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return (k_f - (len(ref) - 1)) / fs


@pytest.fixture(scope="module")
def env():
    return uniform_water_environment(1000, 750, DX, C0)


def test_amplitude_matches_2d_green_function(env):
    """Steady-state tone amplitude follows (dx/(c dt))^2 |H0(kr)| / 4 (cylindrical spreading)."""
    cfg = HarborFDTDConfig(fs=FS, total_time=1.2)
    f0 = 40.0
    t = np.arange(cfg.n_steps) / FS
    sig = np.clip(t / 0.2, 0, 1) * np.sin(2 * np.pi * f0 * t)
    src = (500.0, 375.0)
    ranges = np.array([50, 80, 110, 140, 170, 200.0])
    layout = SensorLayout(hydrophones=np.column_stack([src[0] + ranges, np.full_like(ranges, src[1])]))

    g = simulate_source(env, layout, cfg, sig, position=src)
    amp = np.abs(g.traces[:, t > 0.9]).max(axis=1)
    pred = (DX / (C0 / FS)) ** 2 / 4 * np.abs(hankel1(0, 2 * np.pi * f0 / C0 * ranges))
    assert np.allclose(amp / pred, 1.0, atol=0.06)


def test_hyperbolic_moveout_on_das_line(env):
    """Arrival delays along a straight cable match (sqrt(d^2 + s^2) - d) / c."""
    cfg = HarborFDTDConfig(fs=FS, total_time=1.0)
    src = (500.0, 375.0)
    offset = 120.0
    cable = Cable("line", np.array([[350.0, src[1] + offset], [650.0, src[1] + offset]]), channel_spacing=15.0)
    layout = SensorLayout(cables=[cable])
    g = simulate_source(env, layout, cfg, ricker(cfg, 30.0, 0.05), position=src)

    pos = cable.positions
    r = np.hypot(pos[:, 0] - src[0], pos[:, 1] - src[1])
    ref = int(np.argmin(r))
    measured = np.array([lag_seconds(g.traces[ref], g.traces[i], FS) for i in range(len(r))])
    expected = (r - r[ref]) / C0
    assert np.max(np.abs(measured - expected)) < 0.004


def test_linearity(env):
    cfg = HarborFDTDConfig(fs=FS, total_time=0.6)
    layout = SensorLayout(hydrophones=np.array([[600.0, 375.0], [450.0, 450.0]]))
    sig = ricker(cfg, 30.0, 0.05)
    g1 = simulate_source(env, layout, cfg, sig, position=(500.0, 375.0))
    g2 = simulate_source(env, layout, cfg, -2.5 * sig, position=(500.0, 375.0))
    assert np.allclose(g2.traces, -2.5 * g1.traces, rtol=1e-4, atol=1e-7 * np.abs(g1.traces).max())


def test_constant_trajectory_equals_static(env):
    cfg = HarborFDTDConfig(fs=FS, total_time=0.5)
    layout = SensorLayout(hydrophones=np.array([[600.0, 375.0]]))
    sig = ricker(cfg, 30.0, 0.05)
    g_static = simulate_source(env, layout, cfg, sig, position=(500.0, 375.0))
    traj = (np.array([0.0, 10.0]), np.array([[500.0, 375.0], [500.0, 375.0]]))
    g_traj = simulate_source(env, layout, cfg, sig, trajectory=traj)
    assert np.allclose(g_static.traces, g_traj.traces)


def test_moving_source_envelope_peaks_at_cpa(env):
    """A source passing a hydrophone is loudest around the closest-approach time."""
    cfg = HarborFDTDConfig(fs=FS, total_time=8.0)
    t = np.arange(cfg.n_steps) / FS
    sig = np.sin(2 * np.pi * 40.0 * t)
    traj = (np.array([0.0, 8.0]), np.array([[300.0, 375.0], [700.0, 375.0]]))  # 50 m/s, CPA at t=4 s
    layout = SensorLayout(hydrophones=np.array([[500.0, 425.0]]))
    g = simulate_source(env, layout, cfg, sig, trajectory=traj)

    win = int(0.1 * FS)
    env_amp = np.sqrt(np.convolve(g.traces[0] ** 2, np.ones(win) / win, mode="same"))
    sel = (t > 0.5) & (t < 7.5)
    w = env_amp[sel] ** 8
    centroid = np.sum(t[sel] * w) / np.sum(w)
    assert abs(centroid - (4.0 + 50.0 / C0)) < 0.3


def test_cache_roundtrip(env, tmp_path):
    cfg = HarborFDTDConfig(fs=FS, total_time=0.4)
    layout = SensorLayout(hydrophones=np.array([[600.0, 375.0]]))
    sig = ricker(cfg, 30.0, 0.05)
    a = simulate_source(env, layout, cfg, sig, position=(500.0, 375.0), cache_dir=tmp_path)
    assert len(list(tmp_path.glob("gather_*.npz"))) == 1
    b = simulate_source(env, layout, cfg, sig, position=(500.0, 375.0), cache_dir=tmp_path)
    assert len(list(tmp_path.glob("gather_*.npz"))) == 1
    assert np.array_equal(a.traces, b.traces) and b.fs == FS
    simulate_source(env, layout, cfg, 2 * sig, position=(500.0, 375.0), cache_dir=tmp_path)
    simulate_source(env, layout, cfg, sig, position=(510.0, 375.0), cache_dir=tmp_path)
    assert len(list(tmp_path.glob("gather_*.npz"))) == 3


def test_input_validation(env):
    cfg = HarborFDTDConfig(fs=FS, total_time=0.3)
    layout = SensorLayout(hydrophones=np.array([[600.0, 375.0]]))
    sig = np.zeros(cfg.n_steps)
    with pytest.raises(ValueError, match="exactly one"):
        simulate_source(env, layout, cfg, sig)
    with pytest.raises(ValueError, match="signal must have shape"):
        simulate_source(env, layout, cfg, sig[:-1], position=(500.0, 375.0))
    with pytest.raises(ValueError, match="in water"):
        simulate_source(env, layout, cfg, sig, position=(5000.0, 375.0))
    with pytest.raises(ValueError, match="CFL"):
        simulate_source(env, layout, HarborFDTDConfig(fs=300.0, total_time=0.3), np.zeros(90), position=(500.0, 375.0))


def test_cfl_limit_value(env):
    assert cfl_dt(env, 4) == pytest.approx(2.0 * DX / (C0 * np.sqrt(2.0 * 16.0 / 3.0)))
    assert 1.0 / FS < cfl_dt(env, 4)




def _pulse(cfg, f0=30.0, delay=0.05):
    return ricker(cfg, f0, delay)


def _wall_env(x0, y0, x1, y1, thickness):
    return _finish_environment([segment_polygon((x0, y0), (x1, y1), thickness)], 1000.0, 750.0, DX, C0, 0.0, "wall")


def test_wall_casts_acoustic_shadow(env):
    cfg = HarborFDTDConfig(fs=FS, total_time=0.8)
    layout = SensorLayout(hydrophones=np.array([[470.0, 375.0]]))
    wall = _wall_env(350.0, 300.0, 350.0, 450.0, 20.0)
    shadowed = simulate_source(wall, layout, cfg, _pulse(cfg), position=(250.0, 375.0))
    free = simulate_source(env, layout, cfg, _pulse(cfg), position=(250.0, 375.0))
    assert np.isfinite(shadowed.traces).all()
    assert np.abs(shadowed.traces).max() < 0.6 * np.abs(free.traces).max()


def test_land_cells_have_zero_pressure():
    wall = _wall_env(350.0, 300.0, 350.0, 450.0, 20.0)
    cfg = HarborFDTDConfig(fs=FS, total_time=0.5)
    layout = SensorLayout(hydrophones=np.array([[470.0, 375.0]]))
    solver = HarborFDTDSolver(wall, cfg, StaticSource(250.0, 375.0, _pulse(cfg)), layout.receivers)
    solver.run(verbose=False)
    assert np.abs(solver.p_now[wall.land]).max() == 0.0
    assert np.abs(solver.p_now[wall.water]).max() > 0.0


def test_shore_loss_reflection_coefficient():
    """Measured wall reflection with shore loss vs a lossless wall matches R ~ 0.5."""
    wall = _wall_env(700.0, 100.0, 700.0, 650.0, 30.0)
    layout = SensorLayout(hydrophones=np.array([[450.0, 400.0]]))
    ratio = {}
    for smax in (0.0, 0.15):
        cfg = HarborFDTDConfig(fs=FS, total_time=1.0, shore_loss_max=smax)
        t = np.arange(cfg.n_steps) / FS
        g = simulate_source(wall, layout, cfg, _pulse(cfg), position=(450.0, 375.0)).traces[0]
        ratio[smax] = np.abs(g[(t > 0.25) & (t < 0.55)]).max() / np.abs(g[t < 0.2]).max()
    assert 0.35 < ratio[0.15] / ratio[0.0] < 0.65


def test_shore_loss_damps_reverberation():
    frame = [
        segment_polygon((200.0, 200.0), (700.0, 200.0), 20.0),
        segment_polygon((700.0, 200.0), (700.0, 550.0), 20.0),
        segment_polygon((700.0, 550.0), (200.0, 550.0), 20.0),
        segment_polygon((200.0, 550.0), (200.0, 200.0), 20.0),
    ]
    box = _finish_environment(frame, 1000.0, 750.0, DX, C0, 0.0, "box")
    layout = SensorLayout(hydrophones=np.array([[500.0, 300.0]]))
    late = {}
    for smax in (0.0, 0.15):
        cfg = HarborFDTDConfig(fs=FS, total_time=2.5, shore_loss_max=smax)
        t = np.arange(cfg.n_steps) / FS
        g = simulate_source(box, layout, cfg, _pulse(cfg), position=(400.0, 400.0)).traces[0]
        late[smax] = np.sum(g[t > 1.5] ** 2)
    assert late[0.15] < 0.25 * late[0.0]

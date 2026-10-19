"""Performance paths of the harbor forward model against the reference implementations.

Fused CUDA kernel against the array code (both precisions), device body masks against
``body_factor``, batched runs against single runs, the CPML boundary, and the stable cache key.
The GPU tests skip without CuPy."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_sim.fdtd import FDTDConfig
from acoustic_sim.harbor import Cable, HarborFDTDConfig, HarborFDTDSolver, SensorLayout, uniform_water_environment
from acoustic_sim.harbor.environment import _finish_environment, segment_polygon
from acoustic_sim.harbor.fdtd_runner import _cache_key, _key_fields
from acoustic_sim.harbor.scatterer import Scatterer, body_factor
from acoustic_sim.harbor.vessels import Track
from acoustic_sim.sources import CustomTrajectorySource, StaticSource

from ._gpu import HAS_GPU

gpu = pytest.mark.skipif(not HAS_GPU, reason="CuPy / GPU not available")

DX, C, FS = 0.5, 1500.0, 6000.0
TOTAL = 0.3


def _cfg(**kw) -> HarborFDTDConfig:
    base = dict(fs=FS, total_time=TOTAL, fd_order=4, damping_width=30, use_cuda=True)
    base.update(kw)
    return HarborFDTDConfig(**base)


def _ricker(n: int, f0: float = 400.0, delay: float = 0.01) -> np.ndarray:
    t = np.arange(n) / FS
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _layout() -> SensorLayout:
    cable = Cable("c", np.array([[20.0, 80.0], [100.0, 80.0]]), channel_spacing=4.0, das_mode="strain", gauge_length=2.0)
    return SensorLayout(cables=[cable], hydrophones=np.array([[90.0, 30.0], [30.0, 30.0]]))


def _track(name: str, p0, p1, total: float = TOTAL) -> Track:
    return Track(name, "uuv", False, np.array([0.0, total]), np.array([p0, p1], dtype=float), np.full(2, 2.5), np.ones(2, bool))


def _rel_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(a - b, axis=1)
    den = np.linalg.norm(b, axis=1)
    keep = den > 0
    return num[keep] / den[keep]


def _run(env, cfg, source, layout, bodies=()):
    solver = HarborFDTDSolver(env, cfg, source, layout.sample_points, scatterers=list(bodies))
    out = solver.run(verbose=False)
    return out["traces"], solver


@pytest.fixture(scope="module")
def scene():
    env = _finish_environment([segment_polygon((60.0, 20.0), (60.0, 70.0), 4.0)], 120.0, 100.0, DX, C, 0.0, "wall")
    n = _cfg().n_steps
    sig = _ricker(n)
    moving = CustomTrajectorySource(times=np.array([0.0, TOTAL]), positions=np.array([[30.0, 50.0], [31.0, 50.0]]), signal=sig)
    body = Scatterer("b", _track("b", [80.0, 50.0], [81.0, 50.4]), 3.0, 1.0, "soft")
    return env, moving, body, _layout()


# ----------------------------------------------------------------------
# Fused kernel against the reference
# ----------------------------------------------------------------------

@gpu
def test_fused_float64_matches_reference(scene):
    """Same traces to 1e-12 relative per channel (measured: identical to the last bit)."""
    env, src, body, layout = scene
    ref, s_ref = _run(env, _cfg(), src, layout, [body])
    fused, s_f = _run(env, _cfg(kernel="fused", body_masks="host"), src, layout, [body])
    assert s_ref.kernel == "reference" and s_f.kernel == "fused"
    assert np.isfinite(fused).all() and np.abs(fused).max() > 0
    assert _rel_l2(fused, ref).max() < 1e-12


@gpu
def test_fused_float32_close_to_float64(scene):
    """float32 fields: direct traces within 1e-4 (median) and 1e-3 (max) relative of the float64 run."""
    env, src, body, layout = scene
    ref, _ = _run(env, _cfg(kernel="fused"), src, layout)
    f32, s = _run(env, _cfg(kernel="fused", precision="float32"), src, layout)
    assert s.p_now.dtype == np.float32 and f32.dtype == np.float64
    err = _rel_l2(f32, ref)
    assert np.median(err) < 1e-4 and err.max() < 1e-3, (np.median(err), err.max())


@gpu
def test_fused_keeps_land_at_zero_and_snapshots(scene):
    env, src, body, layout = scene
    import cupy as cp

    solver = HarborFDTDSolver(env, _cfg(kernel="fused"), src, layout.sample_points)
    out, fields = solver.run_with_snapshots([50, 1000])
    assert float(cp.abs(solver.p_now[cp.asarray(env.land)]).max()) == 0.0
    assert float(cp.abs(solver.p_now[cp.asarray(env.water)]).max()) > 0.0
    assert set(fields) == {50, 1000} and fields[50].shape == (env.ny, env.nx)
    ref, _ = _run(env, _cfg(), src, layout)
    assert _rel_l2(out["traces"], ref).max() < 1e-12


def test_fused_request_falls_back_to_reference_on_cpu(scene):
    env, src, body, layout = scene
    cfg = _cfg(kernel="fused", use_cuda=False)
    solver = HarborFDTDSolver(env, cfg, src, layout.sample_points)
    assert solver.kernel == "reference" and not solver.is_cuda


def test_config_validation():
    with pytest.raises(ValueError, match="precision"):
        FDTDConfig(precision="half")
    with pytest.raises(ValueError, match="kernel"):
        FDTDConfig(kernel="magic")
    with pytest.raises(ValueError, match="body_masks"):
        HarborFDTDConfig(body_masks="cpu")


def test_reference_float32_runs_on_cpu(scene):
    """The reference path accepts float32 fields too (CPU), and the traces stay float64."""
    env, src, body, layout = scene
    cfg = _cfg(use_cuda=False, precision="float32", total_time=0.05)
    traces, solver = _run(env, cfg, StaticSource(30.0, 50.0, _ricker(cfg.n_steps)), layout)
    assert solver.p_now.dtype == np.float32 and traces.dtype == np.float64 and np.isfinite(traces).all()


# ----------------------------------------------------------------------
# Cache key
# ----------------------------------------------------------------------

def test_cache_key_ignores_kernel_and_depends_on_precision():
    env = uniform_water_environment(60.0, 60.0, 1.0)
    layout = SensorLayout(cables=[], hydrophones=np.array([[40.0, 30.0]]))
    sig = np.zeros(HarborFDTDConfig(fs=3000.0, total_time=0.02).n_steps)
    keys = {}
    for name, kw in {"default": {}, "fused": {"kernel": "fused"}, "device": {"body_masks": "device"}, "cuda": {"use_cuda": True},
                     "f32": {"precision": "float32"}}.items():
        cfg = HarborFDTDConfig(fs=3000.0, total_time=0.02, damping_width=10, **kw)
        keys[name] = _cache_key(env, layout, cfg, sig, np.array([10.0, 30.0]), None)
    assert keys["default"] == keys["fused"] == keys["device"] == keys["cuda"]
    assert keys["f32"] != keys["default"]
    # The fields that enter the key for a default config are exactly the pre-existing ones (old caches stay valid).
    assert set(_key_fields(HarborFDTDConfig())) == {"fs", "total_time", "fd_order", "damping_width", "damping_max",
                                                    "background_absorption", "shore_loss_width", "shore_loss_max"}


# ----------------------------------------------------------------------
# Device body masks
# ----------------------------------------------------------------------

@gpu
def test_device_masks_match_body_factor():
    """Every shape and kind, a body partly outside the grid: factors agree to 1e-12 (measured 1e-16)."""
    import cupy as cp

    from acoustic_sim.harbor.fdtd_cuda_bodies import DeviceBodyMasks, pose_table

    env = uniform_water_environment(120.0, 100.0, DX)
    x, y = env.model.x, env.model.y
    bodies = [Scatterer("rect", _track("r", [70.0, 50.0], [71.0, 50.4]), 3.0, 1.0, "soft"),
              Scatterer("ell", _track("e", [40.0, 30.0], [40.0, 30.0]), 2.0, 2.0, "soft", shape="ellipse"),
              Scatterer("abs", _track("a", [90.0, 70.0], [92.0, 69.0]), 4.0, 1.5, "absorbing", strength=0.3),
              Scatterer("hard", _track("h", [20.0, 20.0], [21.0, 22.0]), 3.0, 1.0, "soft", smooth=False),
              Scatterer("edge", _track("x", [118.0, 50.0], [119.5, 50.0]), 3.0, 1.0, "soft")]
    n, dt = _cfg().n_steps, 1.0 / FS
    dm = DeviceBodyMasks([bodies], x, y, DX, n, dt, np.float64)
    assert not dm.sequential.any()
    t = np.arange(n) * dt
    for sc in bodies:
        xy, hd, act = pose_table(sc, t)
        for k in (0, 7, n - 1):
            c, h = sc.pose_at(k * dt)
            assert np.array_equal(xy[k], c) and np.array_equal(hd[k], h)
    for step in (0, 1, n // 3, n - 1):
        field = cp.asnumpy(dm.factor_field(0, step))
        ref = np.ones_like(field)
        for sc in bodies:
            res = body_factor(sc, *sc.pose_at(step * dt), x, y, DX)
            if res is not None:
                rows, cols, f = res
                ref[rows, cols] *= f
        assert (ref < 1).sum() > 50
        assert np.abs(field - ref).max() < 1e-12


@gpu
def test_overlapping_bodies_fall_back_to_sequential_launches():
    """Two bodies sharing cells are applied one after the other (the reference order), still exact."""
    import cupy as cp

    from acoustic_sim.harbor.fdtd_cuda_bodies import DeviceBodyMasks

    env = uniform_water_environment(60.0, 60.0, DX)
    x, y = env.model.x, env.model.y
    a = Scatterer("a", _track("a", [30.0, 30.0], [30.0, 30.0]), 4.0, 2.0, "absorbing", strength=0.4)
    b = Scatterer("b", _track("b", [31.0, 30.5], [31.0, 30.5]), 3.0, 3.0, "absorbing", strength=0.25, shape="ellipse")
    dm = DeviceBodyMasks([[a, b]], x, y, DX, 10, 1.0 / FS, np.float64)
    assert dm.sequential.all()
    field = cp.asnumpy(dm.factor_field(0, 3))
    ref = np.ones_like(field)
    for sc in (a, b):
        rows, cols, f = body_factor(sc, *sc.pose_at(3.0 / FS), x, y, DX)
        ref[rows, cols] *= f
    assert (ref < 0.6).any()                                    # both factors act on the shared cells
    assert np.abs(field - ref).max() < 1e-12


@gpu
def test_device_masks_reproduce_host_masks_in_traces(scene):
    env, src, body, layout = scene
    host, s_h = _run(env, _cfg(kernel="fused", body_masks="host"), src, layout, [body])
    dev, s_d = _run(env, _cfg(kernel="fused", body_masks="device"), src, layout, [body])
    assert s_h.body_masks == "host" and s_d.body_masks == "device"
    assert _rel_l2(dev, host).max() < 1e-10
    none, _ = _run(env, _cfg(kernel="fused"), src, layout)
    assert np.linalg.norm(dev - none) > 1e-6 * np.linalg.norm(none)   # the body does scatter


# ----------------------------------------------------------------------
# Batched runs
# ----------------------------------------------------------------------

@gpu
def test_batched_runs_match_single_runs(scene, tmp_path):
    """Four jobs (static and moving sources, zero to two bodies) in one batch: each equals its single run to 1e-12."""
    from acoustic_sim.harbor import simulate_source, simulate_sources
    from acoustic_sim.harbor.fdtd_runner import HarborFDTDBatch, batch_size

    env, moving, body, layout = scene
    cfg = _cfg(kernel="fused")
    n = cfg.n_steps
    other = Scatterer("e", _track("e", [40.0, 30.0], [40.0, 30.0]), 2.0, 2.0, "soft", shape="ellipse")
    jobs = [(StaticSource(30.0, 50.0, _ricker(n)), []),
            (moving, [body]),
            (StaticSource(50.0, 30.0, 0.5 * _ricker(n, 300.0, 0.02)), [other, body]),
            (StaticSource(30.0, 50.0, _ricker(n)), [body])]
    out = HarborFDTDBatch(env, cfg, jobs, layout.sample_points).run(verbose=False)["traces"]
    assert out.shape == (4, layout.n_samples, n)
    for b, (src, sc) in enumerate(jobs):
        single, _ = _run(env, cfg, src, layout, sc)
        assert _rel_l2(out[b], single).max() < 1e-12, b
    assert np.linalg.norm(out[0] - out[3]) > 0            # the body changes job 3 relative to job 0
    assert batch_size(6, env, cfg, layout.n_samples) >= 1
    # simulate_sources: cached and uncached jobs mixed; the cached gathers equal simulate_source's.
    traj = (np.array([0.0, TOTAL]), np.array([[30.0, 40.0], [31.0, 40.5]]))
    jd = [dict(signal=_ricker(n), position=(30.0, 50.0)), dict(signal=_ricker(n, 300.0, 0.02), trajectory=traj, scatterers=[body])]
    g1 = simulate_sources(env, layout, cfg, jd, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("gather_*.npz"))) == 2
    g2 = simulate_sources(env, layout, cfg, jd + [dict(signal=0.5 * _ricker(n), position=(50.0, 30.0))], cache_dir=tmp_path)
    assert len(list(tmp_path.glob("gather_*.npz"))) == 3
    assert all(np.array_equal(a.traces, b.traces) for a, b in zip(g1, g2))
    single = simulate_source(env, layout, cfg, jd[1]["signal"], trajectory=traj, scatterers=[body])
    assert np.array_equal(single.traces, g1[1].traces)


def test_simulate_sources_runs_sequentially_on_cpu(scene, tmp_path):
    from acoustic_sim.harbor import simulate_source, simulate_sources

    env, moving, body, layout = scene
    cfg = _cfg(use_cuda=False, total_time=0.05)
    sig = _ricker(cfg.n_steps)
    jobs = [dict(signal=sig, position=(30.0, 50.0)), dict(signal=-sig, position=(30.0, 50.0))]
    g = simulate_sources(env, layout, cfg, jobs, cache_dir=tmp_path)
    ref = simulate_source(env, layout, cfg, sig, position=(30.0, 50.0))
    assert np.array_equal(g[0].traces, ref.traces) and np.allclose(g[1].traces, -ref.traces)
    assert len(list(tmp_path.glob("gather_*.npz"))) == 2


# ----------------------------------------------------------------------
# CPML boundary
# ----------------------------------------------------------------------

def _reflection_db(env_small, env_big, shift, src, rec, cfg_small, cfg_big):
    """Peak of (small domain trace - large domain trace) over the peak of the large domain trace, per receiver [dB]."""
    sig = _ricker(cfg_small.n_steps, 150.0, 0.03)
    small = HarborFDTDSolver(env_small, cfg_small, StaticSource(src[0], src[1], sig), rec).run(verbose=False)["traces"]
    big = HarborFDTDSolver(env_big, cfg_big, StaticSource(src[0] + shift, src[1] + shift, sig), rec + shift).run(verbose=False)["traces"]
    return 20 * np.log10(np.abs(small - big).max(axis=1) / np.abs(big).max(axis=1))


@pytest.fixture(scope="module")
def cpml_scene():
    """1 m grid: a 160 m x 320 m box with the source 40 m from the right edge and 160 m from the
    others (whose reflections arrive after the 0.15 s window), and a domain large enough that no
    reflection reaches the receivers at all."""
    dx, fs, c = 1.0, 3000.0, 1500.0
    small = uniform_water_environment(160.0, 320.0, dx, c)
    shift = 320.0
    big = uniform_water_environment(160.0 + 2 * shift, 320.0 + 2 * shift, dx, c)
    src = np.array([120.0, 160.0])
    rec = np.array([[120.0, 160.0 + o] for o in (0.0, 20.0, 50.0)])      # 0, 14 and 32 degrees from normal
    base = dict(fs=fs, total_time=0.15, fd_order=4, background_absorption=0.0, shore_loss_max=0.0, use_cuda=False)
    return small, big, shift, src, rec, base


def test_cpml_reflection_well_below_the_sponge(cpml_scene):
    """12 CPML cells reflect at least 10 dB less than the 30 cell sponge at every tested angle, and below -45 dB."""
    small, big, shift, src, rec, base = cpml_scene
    ref_cfg = HarborFDTDConfig(**{**base, "damping_width": 40})
    sponge = _reflection_db(small, big, shift, src, rec, HarborFDTDConfig(**{**base, "damping_width": 30, "damping_max": 0.15}), ref_cfg)
    cpml = _reflection_db(small, big, shift, src, rec, HarborFDTDConfig(**{**base, "boundary": "cpml", "cpml_width": 12}), ref_cfg)
    assert np.all(cpml < sponge - 10.0), (sponge, cpml)
    assert np.all(cpml < -45.0), cpml
    assert np.all(sponge < -15.0), sponge          # the sponge does absorb; the test geometry is sane


def test_cpml_config_and_damping():
    with pytest.raises(ValueError, match="boundary"):
        FDTDConfig(boundary="pml")
    with pytest.raises(ValueError, match="boundary"):
        HarborFDTDConfig(boundary="pml")
    env = uniform_water_environment(60.0, 50.0, 1.0)
    cfg = HarborFDTDConfig(fs=3000.0, total_time=0.01, boundary="cpml", cpml_width=8, shore_loss_max=0.0, use_cuda=False)
    solver = HarborFDTDSolver(env, cfg, StaticSource(30.0, 25.0, np.zeros(cfg.n_steps)), np.array([[40.0, 25.0]]))
    sigma = np.asarray(solver.sigma)
    assert np.allclose(sigma, cfg.background_absorption)              # no sponge ramp
    assert solver._cpml is not None and np.count_nonzero(np.asarray(solver._cpml["mx"])) == 16
    with pytest.raises(ValueError, match="cpml_width"):
        HarborFDTDSolver(env, HarborFDTDConfig(fs=3000.0, total_time=0.01, boundary="cpml", cpml_width=40, use_cuda=False),
                         StaticSource(30.0, 25.0, np.zeros(30)), np.array([[40.0, 25.0]]))


@gpu
def test_fused_cpml_matches_reference_cpml(scene):
    env, src, body, layout = scene
    for order in (4, 8):
        kw = dict(boundary="cpml", cpml_width=12, cpml_alpha_max=200.0, fd_order=order)
        ref, _ = _run(env, HarborFDTDConfig(**{**_cfg().__dict__, **kw}), src, layout, [body])
        fused, s = _run(env, HarborFDTDConfig(**{**_cfg().__dict__, **kw, "kernel": "fused"}), src, layout, [body])
        assert s.kernel == "fused" and s.boundary == "cpml"
        assert _rel_l2(fused, ref).max() < 1e-12, order

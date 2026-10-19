"""The vectorized processing chain (``BistaticConfig.impl = "vectorized"``) against the reference loops.

Analytic free field scene (as ``test_bistatic.py``): a ship moving at 4 m/s, an echo from a silent
body, noise.  On the CPU the vectorized ECA, refinement, maps and motion integration must agree
with the reference to 1e-10 relative (measured: identical for the refinement, the maps and the
motion integration; 1e-12 for the ECA whose normal equations are ill conditioned).  The GPU
variants (CuPy, double precision) skip without a GPU."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from acoustic_sim.water import Cable, SensorLayout, uniform_water_environment
from acoustic_sim.water.ais import AISReports
from acoustic_sim.water.bistatic import (
    BistaticConfig,
    ReplicaModel,
    Truth,
    _fractional_delay,
    _xp,
    bistatic_map,
    build_grid,
    eca_stage,
    motion_integrate,
    reference_spectrum,
    run_bistatic,
    ship_positions,
    stft,
)
from acoustic_sim.water.dispersion import DispersionModel

from ._gpu import HAS_GPU

FS, T, C, F_MAX = 2000.0, 3.0, 1495.0, 500.0
SHIP = np.array([150.0, 250.0])
BODY = np.array([260.0, 180.0])
DEVICES = [pytest.param(False, id="cpu"),
           pytest.param(True, id="gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="CuPy / GPU not available"))]
# Dispersion of a resolved scheme (the fine 0.5 m grid); the scene itself is analytic, the model only
# exercises the direction dependent wavenumber path.
DISPERSION = DispersionModel(0.5, 1.0 / 6000.0, C, 4)


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
    s *= np.clip((tt - 0.3) / 0.1, 0.0, 1.0) * np.clip((T - 0.3 - tt) / 0.1, 0.0, 1.0)
    s /= s.std()
    rec = layout.receivers
    ship_t = SHIP[None, :] + np.outer(tt, [4.0, 0.0])
    direct = np.stack([np.interp(tt - np.hypot(*(ship_t - r).T) / C, tt, s, left=0.0, right=0.0) / np.sqrt(np.hypot(*(SHIP - r)))
                       for r in rec])
    r_e = np.hypot(*(BODY - SHIP)) + np.hypot(*(rec - BODY).T)
    echo = 0.2 * _fractional_delay(np.repeat(s[None], len(rec), axis=0), r_e / C, FS) / np.sqrt(r_e)[:, None]
    traces = direct + echo + 0.002 * rng.standard_normal(direct.shape)
    k = int(T / 0.5)
    t_ais = np.arange(k) * 0.5
    ais = AISReports(vessel=np.array(["ship"] * k), t=t_ais, xy=SHIP[None, :] + np.outer(t_ais, [4.0, 0.0]), sog=np.full(k, 4.0), cog=np.zeros(k))
    return env, layout, traces, direct, echo, s, ais


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _model(layout, bc, sp):
    m = ReplicaModel(layout, bc.c, dispersion=bc.dispersion, xp=_xp(bc))
    m.weights = sp.weights
    return m


def test_impl_validation():
    with pytest.raises(ValueError, match="impl"):
        BistaticConfig(impl="fast")
    assert BistaticConfig().impl == "vectorized" and not BistaticConfig().use_cuda


@pytest.mark.parametrize("cuda", DEVICES)
@pytest.mark.parametrize("reference, doppler", [("oracle", False), ("data", False), ("oracle", True), ("data", True)])
def test_refinement_and_eca_match_reference(scene, cuda, reference, doppler):
    env, layout, traces, direct, echo, s, ais = scene
    res = {}
    for impl in ("reference", "vectorized"):
        bc = BistaticConfig(reference=reference, removal="eca", c=C, refine=True, eca_doppler=doppler,
                            track_smooth_s=2.0 if doppler else 0.0, impl=impl, use_cuda=cuda, dispersion=DISPERSION)
        sp = stft(traces, layout, FS, bc, F_MAX)
        model = _model(layout, bc, sp)
        ship = ship_positions(sp, ais, model, bc)
        outs, wf = eca_stage(traces, layout, FS, env, sp, ship, bc, s if reference == "oracle" else None, [direct, echo])
        res[impl] = (ship, outs, wf)
    (r_ship, r_outs, r_wf), (v_ship, v_outs, v_wf) = res["reference"], res["vectorized"]
    assert np.array_equal(v_ship.refined, r_ship.refined) and np.array_equal(v_ship.present, r_ship.present)
    assert _rel(v_wf, r_wf) < 1e-12
    # The normal equations are ill conditioned (loading 1e-14 of the mean diagonal), so a different
    # Cholesky (cuSOLVER) moves the solution within the regularisation: 1e-10 on the CPU (SciPy's
    # factorisation, as the reference), 1e-9 on the GPU.
    tol = 1e-9 if cuda else 1e-10
    for a, b in zip(v_outs, r_outs):
        assert _rel(a, b) < tol
    depth = [10 * np.log10(np.sum(direct ** 2, axis=1) / np.sum(o[1] ** 2, axis=1)) for o in (r_outs, v_outs)]
    assert np.abs(depth[1] - depth[0]).max() < 1e-6


@pytest.mark.parametrize("cuda", DEVICES)
def test_maps_match_reference(scene, cuda):
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(reference="oracle", removal="eca", c=C, refine=False, grid_dx=5.0, edge=20.0, exclusion_radius=30.0,
                        impl="reference", dispersion=DISPERSION)
    sp = stft(traces, layout, FS, bc, F_MAX)
    grid = build_grid(env, bc)
    truth = Truth(ship_xy=SHIP[None, :] + np.outer(sp.times, [4.0, 0.0]))
    runs = {impl: run_bistatic(traces, layout, FS, env, ais, BistaticConfig(**{**bc.__dict__, "impl": impl, "use_cuda": cuda}),
                               F_MAX, oracle_signal=s, truth=truth) for impl in ("reference", "vectorized")}
    assert _rel(runs["vectorized"].scan.B, runs["reference"].scan.B) < 1e-10
    assert _rel(runs["vectorized"].Z, runs["reference"].Z) < 1e-10
    assert np.array_equal(runs["vectorized"].ref, runs["reference"].ref)
    est = [r.grid[np.argmax(r.scan.B, axis=0)] for r in runs.values()]
    assert np.array_equal(est[0], est[1])
    clean, ship, ref = runs["reference"].clean, runs["reference"].ship, runs["reference"].ref
    for method in ("bartlett", "bartlett_phat", "mvdr", "music", "coherent"):
        b_ref = BistaticConfig(**{**bc.__dict__, "method": method})
        b_vec = BistaticConfig(**{**bc.__dict__, "method": method, "impl": "vectorized", "use_cuda": cuda})
        B1, _ = bistatic_map(clean, ship, ref, grid, _model(layout, b_ref, sp), b_ref)
        B2, _ = bistatic_map(clean, ship, ref, grid, _model(layout, b_vec, sp), b_vec)
        assert B2.shape == B1.shape and _rel(B2, B1) < 1e-10, method
    # Data reference spectrum (beam at the ship), the branch without a time domain waveform.
    b_data = BistaticConfig(**{**bc.__dict__, "reference": "data", "removal": "none"})
    r_ref = reference_spectrum(sp, ship, _model(layout, b_data, sp), b_data)
    b_vec = BistaticConfig(**{**b_data.__dict__, "impl": "vectorized", "use_cuda": cuda})
    r_vec = reference_spectrum(sp, ship, _model(layout, b_vec, sp), b_vec)
    assert _rel(r_vec, r_ref) < 1e-10


@pytest.mark.parametrize("cuda", DEVICES)
def test_motion_integrate_matches_reference(scene, cuda):
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(reference="oracle", removal="eca", c=C, refine=False, grid_dx=5.0, edge=20.0, exclusion_radius=30.0,
                        impl="reference", dispersion=DISPERSION)
    sp = stft(traces, layout, FS, bc, F_MAX)
    run = run_bistatic(traces, layout, FS, env, ais, bc, F_MAX, oracle_signal=s,
                       truth=Truth(ship_xy=SHIP[None, :] + np.outer(sp.times, [4.0, 0.0])))
    speeds = np.array([(vx, vy) for vx in np.linspace(-3, 3, 5) for vy in np.linspace(-3, 3, 5)])
    for coherent in (True, False):
        Z = run.Z if coherent else np.abs(run.Z)
        b1, p1 = motion_integrate(Z, run.grid, sp.times, speeds, coherent, impl="reference")
        b2, p2 = motion_integrate(Z, run.grid, sp.times, speeds, coherent, impl="vectorized", use_cuda=cuda)
        assert _rel(b2, b1) < 1e-10 and _rel(p2, p1) < 1e-10
        assert np.argmax(p2) == np.argmax(p1)

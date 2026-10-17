"""Helmholtz forward model: accuracy against the analytic Green's function, reciprocity,
pressure release land, the replica library and the frequency domain processors."""

import numpy as np
import pytest

from acoustic_sim.harbor import Cable, SensorLayout, uniform_water_environment
from acoustic_sim.harbor.detection import green2d
from acoustic_sim.harbor.environment import _finish_environment, segment_polygon
from acoustic_sim.harbor.helmholtz import (
    HelmholtzConfig,
    HelmholtzSolver,
    bartlett,
    bistatic_coherent,
    nested_dissection,
    project_out,
    replica_library,
    sparse_image,
)

C0 = 1500.0
F = 200.0
K = 2 * np.pi * F / C0


def lossless(**kw):
    return HelmholtzConfig(shore_loss_max=0.0, background_absorption=0.0, **kw)


@pytest.fixture(scope="module")
def open_water():
    env = uniform_water_environment(width=160.0, height=160.0, dx=0.5, c0=C0)
    return env, HelmholtzSolver(env, lossless())


@pytest.fixture(scope="module")
def walled():
    """Open water with a thick breakwater across the middle, leaving a gap at the east end."""
    polys = [segment_polygon((0.0, 80.0), (110.0, 80.0), 6.0)]
    env = _finish_environment(polys, 160.0, 160.0, 0.5, C0, 0.0, "walled")
    return env, HelmholtzSolver(env, HelmholtzConfig(shore_loss_width=10.0, shore_loss_max=0.05))


def test_nested_dissection_is_a_permutation():
    idx = -np.ones((30, 40), int)
    water = np.ones((30, 40), bool)
    water[10:12, 5:30] = False
    idx[water] = np.arange(water.sum())
    p = nested_dissection(idx, leaf=16)
    assert np.array_equal(np.sort(p), np.arange(water.sum()))


@pytest.mark.parametrize("stencil, tol", [("9pt", 0.08), ("5pt", 0.6)])
def test_open_water_matches_hankel(open_water, stencil, tol):
    env, _ = open_water
    s = HelmholtzSolver(env, lossless(stencil=stencil))
    src = np.array([[80.25, 80.25]])
    r = np.array([10.0, 25.0, 40.0, 50.0])
    for ang in (0.0, np.pi / 4):
        pts = src + r[:, None] * np.array([np.cos(ang), np.sin(ang)])
        u = s.solve(F, src, s.interp_matrix(pts))[0]
        g = green2d(K * r)
        assert np.max(np.abs(np.angle(u / g))) < tol
        assert np.allclose(np.abs(u / g), 1.0, atol=0.06)


def test_9pt_beats_5pt(open_water):
    env, s9 = open_water
    s5 = HelmholtzSolver(env, lossless(stencil="5pt"))
    src = np.array([[80.25, 80.25]])
    pt = np.array([[130.25, 80.25]])
    e9 = abs(np.angle(s9.solve(F, src, s9.interp_matrix(pt))[0, 0] / green2d(np.array([K * 50.0]))[0]))
    e5 = abs(np.angle(s5.solve(F, src, s5.interp_matrix(pt))[0, 0] / green2d(np.array([K * 50.0]))[0]))
    assert e9 < 0.25 * e5


def test_residual_is_small(walled):
    env, s = walled
    A = s.operator(F)
    lu = s.factor(F)
    b = np.zeros(s.n, complex)
    b[s.index[100, 120]] = 1.0
    u = lu.solve(b)
    assert np.linalg.norm(A @ u - b) < 1e-8 * np.linalg.norm(b)


def test_operator_is_symmetric(walled):
    _, s = walled
    A = s.operator(F)
    assert abs(A - A.T).max() < 1e-12 * abs(A).max()


def test_reciprocity_with_walls(walled):
    _, s = walled
    a = np.array([[30.3, 40.7]])
    b = np.array([[60.9, 130.1]])        # behind the breakwater
    ab = s.solve(F, a, s.interp_matrix(b))[0, 0]
    ba = s.solve(F, b, s.interp_matrix(a))[0, 0]
    assert abs(ab - ba) < 1e-8 * abs(ab)


def test_land_is_pressure_release_and_shadows(walled):
    env, s = walled
    src = np.array([[40.0, 40.0]])
    u = s.solve(F, src, full=True)[0]
    assert np.all(u[env.land] == 0)
    # Directly behind the wall versus the same range in the open direction.
    behind = np.abs(s.solve(F, src, s.interp_matrix(np.array([[40.0, 110.0]])))[0, 0])
    beside = np.abs(s.solve(F, src, s.interp_matrix(np.array([[110.0, 40.0]])))[0, 0])
    assert behind < 0.3 * beside


@pytest.fixture(scope="module")
def library(walled):
    env, s = walled
    layout = SensorLayout(cables=[Cable("c", np.array([[30.0, 30.0], [130.0, 30.0]]), 2.0, "strain", 2.0)],
                          hydrophones=np.array([[30.0, 130.0]]))
    xs = np.arange(45.0, 116.0, 2.0)
    ys = np.arange(45.0, 116.0, 2.0)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    grid = grid[env.is_water(grid, margin=4.0)]
    extra = np.array([[71.3, 55.9], [120.0, 60.0]])
    src = np.array([[120.0, 60.0]])
    freqs = np.array([150.0, 200.0, 250.0])
    lib = replica_library(s, layout, grid, extra=extra, sources=src, freqs=freqs, verbose=False)
    return env, s, layout, lib


def test_library_matches_direct_solve(library):
    _, s, layout, lib = library
    # Forward: source at the extra point, field at the sample points, collapsed to channels.
    fi = 1
    vals = s.solve(lib.freqs[fi], lib.extra[:1], s.interp_matrix(layout.sample_points))[0]
    ch = layout.collapse(vals[None, :])[0]
    assert np.allclose(lib.E[fi, 0], ch, rtol=1e-4, atol=1e-6 * np.abs(ch).max())
    # Illuminator field at the extra point equals a direct solve.
    direct = s.solve(lib.freqs[fi], lib.sources, s.interp_matrix(lib.extra[:1]))[0, 0]
    assert np.isclose(lib.incident_extra(0, 0)[fi], direct, rtol=1e-4)


def test_project_out_removes_span():
    rng = np.random.default_rng(0)
    v1 = rng.standard_normal((2, 20)) + 1j * rng.standard_normal((2, 20))
    v2 = rng.standard_normal((2, 3, 20)) + 1j * rng.standard_normal((2, 3, 20))
    d = (2.0 * v1[None] + np.einsum("m,fmc->fc", np.array([1.0, -1j, 0.5]), v2)[None]).repeat(3, axis=0)
    out = project_out(d, [v1, v2])
    assert np.abs(out).max() < 1e-10 * np.abs(d).max()


def _echo_data(lib, cell, n_snap=4, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    nf = len(lib.freqs)
    ref = np.exp(1j * rng.uniform(0, 2 * np.pi, (n_snap, nf)))
    Gin = lib.incident_grid(0)
    d = ref[:, :, None] * (Gin[:, cell][:, None] * lib.G[:, cell, :])[None]
    d = d + noise * np.abs(d).mean() * (rng.standard_normal(d.shape) + 1j * rng.standard_normal(d.shape))
    return d, ref, Gin


def test_bistatic_and_sparse_locate_a_point_scatterer(library):
    _, _, _, lib = library
    cell = int(np.argmin(np.hypot(*(lib.grid - [75.0, 65.0]).T)))
    d, ref, Gin = _echo_data(lib, cell, noise=0.05)
    B = bistatic_coherent(d, ref, Gin, lib.G)
    assert np.argmax(B) == cell
    m = sparse_image(d, ref, Gin, lib.G, n_iter=60, lam_frac=0.3)
    assert np.argmax(m) == cell
    far = np.hypot(*(lib.grid - lib.grid[cell]).T) > 6.0
    assert m[far].max() < 0.1 * m[cell]


def test_bartlett_passive_source(library):
    _, _, _, lib = library
    cell = int(np.argmin(np.hypot(*(lib.grid - [90.0, 100.0]).T)))
    rng = np.random.default_rng(1)
    amp = np.exp(1j * rng.uniform(0, 2 * np.pi, (5, len(lib.freqs))))
    d = amp[:, :, None] * lib.G[:, cell, :][None]
    assert np.argmax(bartlett(d, lib.G)) == cell

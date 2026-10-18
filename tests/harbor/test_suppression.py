"""AIS-aware suppression, re-whitening and the occupancy filter (CPU, and GPU when available)."""

import numpy as np
import pytest

from acoustic_sim.harbor import AISReports, Cable, SensorLayout, uniform_water_environment
from acoustic_sim.harbor.detection import (
    MatchedFieldScanner,
    ProcessingConfig,
    Whitened,
    build_grid,
    rewhiten,
)
from acoustic_sim.harbor.device import get_device
from acoustic_sim.harbor.occupancy import OccupancyConfig, OccupancyGrid
from acoustic_sim.harbor.suppression import (
    SuppressionConfig,
    predict_ships,
    project_out_adaptive,
    project_out_ships,
    refine_ships,
    run_ais_aware,
)

from ._gpu import DEVICES

FREQS = np.arange(20.0, 61.0, 1.0)
N_FRAMES = 14
TARGET0, TARGET_V = np.array([650.0, 520.0]), np.array([0.0, -6.0])   # weak mover
SHIP0, SHIP_V = np.array([260.0, 300.0]), np.array([4.0, 3.0])         # loud ship, metres per frame


def ship_xy(k):
    return SHIP0 + SHIP_V * k


def target_xy(k):
    return TARGET0 + TARGET_V * k


def make_scene(use_cuda, ship_amp=300.0, target_amp=7.0, multipath=False, seed=0, ship_present=True):
    """Whitened free-field data: unit noise + one loud moving ship + one weak moving target.

    With ``multipath`` the ship's channel is its direct path plus an equally strong static
    multipath vector (unknown to any replica model), as in a reverberant harbor: a free-field
    replica can remove only about half of the ship's energy.
    """
    rng = np.random.default_rng(seed)
    layout = SensorLayout(
        cables=[Cable("c", np.array([[200.0, 200.0], [500.0, 200.0], [500.0, 500.0]]), channel_spacing=15.0)],
        hydrophones=np.array([[300.0, 450.0], [700.0, 250.0]]),
    )
    pc = ProcessingConfig(interference_rank_max=0, grid_bounds=(150, 850, 120, 650), min_separation=60.0,
                          bin_stride=1, f_lo=20.0, f_hi=60.0, use_cuda=use_cuda, grid_dx=20.0)
    dev = get_device(use_cuda)
    n, f = layout.n_receivers, len(FREQS)
    times = 1.0 + np.arange(N_FRAMES)
    wh = Whitened(freqs=FREQS, times=times, Z=dev.xp.zeros((n, f, N_FRAMES), np.complex64),
                  sigma=dev.xp.ones((n, f)), weights=np.ones(n), scan_bins=np.arange(f), dev=dev)
    scanner = MatchedFieldScanner(layout, build_grid(uniform_water_environment(1000, 750, 5.0), pc), pc)

    def cg(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

    Z = cg(n, f, N_FRAMES)
    h_static = cg(n, f)
    for k in range(N_FRAMES):
        for amp, xy, is_ship in ((ship_amp, ship_xy(k), True), (target_amp, target_xy(k), False)):
            if is_ship and not ship_present:
                continue
            for i in range(f):
                rep = dev.asnumpy(scanner._replicas(scanner._geometry(xy[None]), wh, i))[0]
                chan = rep
                if is_ship and multipath:
                    chan = np.linalg.norm(rep) * (rep / np.linalg.norm(rep) + h_static[:, i] / np.linalg.norm(h_static[:, i]))
                Z[:, i, k] += amp * cg(1)[0] * chan
    wh.Z = dev.xp.asarray(Z.astype(np.complex64))
    # AIS reports every 2 frames with 8 m position error
    ks = np.arange(0, N_FRAMES, 2)
    pos = np.array([ship_xy(k) for k in ks]) + rng.normal(0, 8.0, (len(ks), 2))
    ais = AISReports(vessel=np.array(["ship"] * len(ks)), t=times[ks], xy=pos,
                     sog=np.full(len(ks), float(np.hypot(*SHIP_V))), cog=np.full(len(ks), float(np.arctan2(*SHIP_V[::-1]))))
    return layout, pc, wh, scanner, ais


@pytest.mark.parametrize("cuda", DEVICES)
def test_refinement_beats_ais_and_flags_significance(cuda):
    layout, pc, wh, scanner, ais = make_scene(cuda)
    sc = SuppressionConfig()
    est = refine_ships(wh, layout, predict_ships(wh, ais, sc), pc, sc)
    truth = np.array([ship_xy(k) for k in range(N_FRAMES)])
    ok = est.significant[:, 0]
    assert ok.all() and np.all(est.significance[:, 0] > 50)
    e_ais = np.hypot(*(est.predicted[:, 0] - truth).T)
    e_ref = np.hypot(*(est.refined[:, 0] - truth).T)
    assert np.median(e_ref) < 3.0 < np.median(e_ais)
    # A ship far below the noise is reported (AIS) but not significant, so it is not projected.
    layout, pc, wh, scanner, ais = make_scene(cuda, ship_amp=0.05, target_amp=0.0)
    est = refine_ships(wh, layout, predict_ships(wh, ais, sc), pc, sc)
    assert not est.significant.any() and est.present.all()
    assert np.allclose(est.refined, est.predicted)


@pytest.mark.parametrize("cuda", DEVICES)
def test_model_projection_removes_a_free_field_ship_and_leaves_the_target(cuda):
    layout, pc, wh, scanner, ais = make_scene(cuda)
    wh = rewhiten(wh, pc)          # what the pipeline sees: a noise floor estimated from ship-contaminated data
    sc = SuppressionConfig(mode="model")
    est = refine_ships(wh, layout, predict_ships(wh, ais, sc), pc, sc)
    clean = rewhiten(project_out_ships(wh, layout, est, pc, sc), pc)
    dev = wh.dev
    # The ship dominated the original floor; after cleaning it is back to ~ the noise level.
    assert float(dev.xp.median(wh.sigma)) > 20.0 * float(dev.xp.median(clean.sigma))
    assert 0.5 < float(dev.xp.mean(dev.xp.abs(clean.Z[:, :, :]) ** 2)) < 2.0
    B = scanner.scan(clean).B
    grid = scanner.grid
    for k in range(2, N_FRAMES - 2):
        assert np.hypot(*(grid[np.argmax(B[:, k])] - target_xy(k))) < 40.0, k
    raw = scanner.scan(wh).B
    hits = [np.hypot(*(grid[np.argmax(raw[:, k])] - target_xy(k))) < 40.0 for k in range(N_FRAMES)]
    assert np.mean(hits) < 0.2      # without suppression the ship's sidelobes win


@pytest.mark.parametrize("cuda", DEVICES)
def test_adaptive_projection_handles_a_multipath_ship_the_model_cannot(cuda):
    layout, pc, wh, scanner, ais = make_scene(cuda, multipath=True)
    wh = rewhiten(wh, pc)
    grid = scanner.grid

    def hit_rate(mode):
        # The rank must cover how far the ship's direct path rotates over the window; a controlled
        # sweep in this scene gives hit rates 0.0 (window 3, rank 2) and 0.9 (window 2, rank 3).
        sc = SuppressionConfig(mode=mode, ranks_per_ship=3, adaptive_window=2)
        est = refine_ships(wh, layout, predict_ships(wh, ais, sc), pc, sc)
        clean = wh
        if mode in ("model", "both"):
            clean = project_out_ships(clean, layout, est, pc, sc)
        if mode in ("adaptive", "both"):
            clean = project_out_adaptive(clean, est, sc)
        B = scanner.scan(rewhiten(clean, pc)).B
        return np.mean([np.hypot(*(grid[np.argmax(B[:, k])] - target_xy(k))) < 40.0 for k in range(2, N_FRAMES - 2)])

    assert hit_rate("model") < 0.3
    assert hit_rate("adaptive") > 0.8


@pytest.mark.parametrize("cuda", DEVICES)
def test_pipeline_on_synthetic_traces_excludes_ship_zone(cuda):
    layout, pc, wh, scanner, ais = make_scene(cuda)
    est = refine_ships(wh, layout, predict_ships(wh, ais, SuppressionConfig()), pc, SuppressionConfig())
    mask = est.exclusion_mask(scanner.grid, 100.0)
    assert mask.shape == (len(scanner.grid), N_FRAMES)
    k = 5
    near = np.hypot(*(scanner.grid - est.refined[k, 0]).T) < 100.0
    assert np.array_equal(mask[:, k], near) and near.any() and not near.all()


def test_invalid_mode_rejected():
    layout, pc, wh, scanner, ais = make_scene(False)
    with pytest.raises(ValueError, match="mode"):
        run_ais_aware(np.zeros((layout.n_receivers, 4000)), layout, 600.0, uniform_water_environment(1000, 750, 5.0),
                      ais, pc, SuppressionConfig(mode="bogus"))


# ----------------------------- occupancy -----------------------------

def _grid_20():
    xs, ys = np.meshgrid(np.arange(0, 1000.0, 20.0), np.arange(0, 600.0, 20.0))
    g = np.column_stack([xs.ravel(), ys.ravel()])
    return g[~((g[:, 0] > 400) & (g[:, 0] < 500) & (g[:, 1] > 100))]       # a land block


def _maps(grid, truth, rng, spike=20.0, peak=30.0):
    """Per-frame maps: a blob on the target plus one random false spike each frame."""
    B = np.zeros((len(grid), len(truth)), dtype=np.float32)
    for k, xy in enumerate(truth):
        B[:, k] += peak * np.exp(-0.5 * (np.hypot(*(grid - xy).T) / 25.0) ** 2) * (rng.random() < 0.5)
        B[rng.integers(len(grid)), k] += spike
    return B


@pytest.mark.parametrize("cuda", DEVICES)
def test_occupancy_tracks_a_marginal_target_through_false_peaks(cuda):
    rng = np.random.default_rng(0)
    grid = _grid_20()
    times = np.arange(40) * 1.0
    truth = np.column_stack([200.0 + 6.0 * times, np.full_like(times, 300.0)])
    B = _maps(grid, truth, rng)
    occ = OccupancyGrid(grid, OccupancyConfig(speed_mps=8.0, use_cuda=cuda))
    h = occ.run(B, times, snapshot_frames=(30,))
    err = np.hypot(*(h.xy - truth).T)
    assert np.median(err[10:]) < 30.0 and np.mean(err[10:] < 60.0) > 0.8
    assert h.contrast[-1] > 50.0
    assert abs(h.snapshots[30].sum() - 1.0) < 1e-4
    # no single frame's argmax is reliable (half the frames have no blob at all)
    assert np.mean([np.hypot(*(grid[np.argmax(B[:, k])] - truth[k])) < 60 for k in range(40)]) < 0.7


@pytest.mark.parametrize("cuda", DEVICES)
def test_occupancy_is_flat_without_evidence_and_never_leaks_into_land(cuda):
    grid = _grid_20()
    occ = OccupancyGrid(grid, OccupancyConfig(use_cuda=cuda))
    h = occ.run(np.zeros((len(grid), 10), np.float32), np.arange(10.0))
    assert np.allclose(h.contrast, 1.0, atol=0.05)
    assert occ.mask2d.sum() == len(grid)
    p = occ.dev.asnumpy(occ.p)
    assert np.isclose(p.sum(), 1.0, atol=1e-4) and p.min() > 0


@pytest.mark.parametrize("cuda", DEVICES)
def test_occupancy_reacquires_after_the_target_jumps(cuda):
    rng = np.random.default_rng(1)
    grid = _grid_20()
    times = np.arange(30) * 1.0
    truth = np.vstack([np.tile([200.0, 300.0], (15, 1)), np.tile([800.0, 100.0], (15, 1))])
    B = _maps(grid, truth, rng, peak=40.0)
    h = OccupancyGrid(grid, OccupancyConfig(speed_mps=4.0, leak=0.05, use_cuda=cuda)).run(B, times)
    assert np.hypot(*(h.xy[12] - truth[12])) < 40.0
    assert np.hypot(*(h.xy[-1] - truth[-1])) < 40.0


# ----------------------------- GPU / CPU parity -----------------------------

@pytest.mark.parametrize("cuda", [pytest.param(True, id="gpu", marks=pytest.mark.skipif(
    not __import__("tests.harbor._gpu", fromlist=["HAS_GPU"]).HAS_GPU, reason="no GPU"))])
def test_gpu_matches_cpu_on_the_full_ship_suppression_chain(cuda):
    out = {}
    for use_cuda in (False, True):
        layout, pc, wh, scanner, ais = make_scene(use_cuda, multipath=True)
        sc = SuppressionConfig(mode="both")
        est = refine_ships(wh, layout, predict_ships(wh, ais, sc), pc, sc)
        clean = rewhiten(project_out_adaptive(project_out_ships(wh, layout, est, pc, sc), est, sc), pc)
        out[use_cuda] = (est, scanner.scan(clean).B)
    (e0, B0), (e1, B1) = out[False], out[True]
    assert np.array_equal(e0.significant, e1.significant)
    assert np.allclose(e0.refined, e1.refined, atol=1e-6)
    assert np.corrcoef(B0.ravel(), B1.ravel())[0, 1] > 0.999
    assert np.mean(B0.argmax(0) == B1.argmax(0)) >= 0.9

"""The bistatic chain on a small analytic (free field, exact delay) case: ``eca_stage`` against
the inline code it replaced, cancellation depth and echo preservation, and ``run_bistatic``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from acoustic_sim.water import Cable, SensorLayout, uniform_water_environment
from acoustic_sim.water.ais import AISReports
from acoustic_sim.water.bistatic import (
    BistaticConfig,
    ReplicaModel,
    ShipTrack,
    Diagnostics,
    _fractional_delay,
    clutter_images,
    eca_remove,
    eca_stage,
    make_reference,
    run_bistatic,
    ship_positions,
    smooth_track,
    stft,
)

FS = 2000.0
T = 3.0
C = 1495.0
F_MAX = 500.0
F_HI = 0.95 * F_MAX          # the band top these tests were written with (the old default 0.95 f_max)
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


def _old_inline(traces, layout, fs, env, sp, ship, bc, true_waveform, comps):
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
    waveform = make_reference(traces, layout, fs, ship_b, bc.eca_block_s, bc, chans, true_waveform)
    images = [clutter_images(env, sxy) for sxy in ship_b] if bc.eca_images else []
    return eca_remove([traces] + comps, waveform, layout, fs, ship_b, images, bc.eca_block_s, bc), waveform


@pytest.mark.parametrize("reference", ["true_waveform", "data"])
def test_eca_stage_matches_old_inline_code(scene, reference):
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(f_hi=F_HI, reference=reference, removal="eca", c=C, refine=False)
    sp = stft(traces, layout, FS, bc)
    model = ReplicaModel(layout, bc.c)
    model.weights = sp.weights
    ship = ship_positions(sp, ais, model, bc)
    given = s if reference == "true_waveform" else None
    outs, wf = eca_stage(traces, layout, FS, env, sp, ship, bc, given, [direct, echo])
    outs_old, wf_old = _old_inline(traces, layout, FS, env, sp, ship, bc, given, [direct, echo])
    assert np.array_equal(wf, wf_old)
    assert len(outs) == 3 and all(np.array_equal(a, b) for a, b in zip(outs, outs_old))


@pytest.mark.parametrize("reference, min_depth_db", [("true_waveform", 25.0), ("data", 12.0)])
def test_eca_attenuates_direct_and_keeps_echo(scene, reference, min_depth_db):
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(f_hi=F_HI, reference=reference, removal="eca", c=C, refine=False)
    sp = stft(traces, layout, FS, bc)
    model = ReplicaModel(layout, bc.c)
    model.weights = sp.weights
    ship = ship_positions(sp, ais, model, bc)
    outs, _ = eca_stage(traces, layout, FS, env, sp, ship, bc, s if reference == "true_waveform" else None, [direct, echo])
    depth, loss = _depth_db(direct, outs[1]), _depth_db(echo, outs[2])
    assert np.median(depth) > min_depth_db, depth
    assert np.median(loss) < 1.0, loss
    # The cleaned measurement has lost the direct path but not the echo.
    assert np.sum(outs[0] ** 2) < 0.1 * np.sum(traces ** 2)


def test_run_bistatic_end_to_end_reports_eca_depth_and_finds_the_body(scene):
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(f_hi=F_HI, reference="true_waveform", removal="eca", method="coherent", c=C, grid_dx=5.0, edge=20.0,
                        exclusion_radius=30.0, refine=False)
    sp = stft(traces, layout, FS, bc)
    truth = Diagnostics(exact_track=np.repeat(SHIP[None], len(sp.times), axis=0), components={"direct:ship": direct, "scattered:ship>body": echo})
    run = run_bistatic(traces, layout, FS, env, ais, bc, true_waveform=s, diagnostics=truth)
    assert run.eca is not None
    assert np.median(run.eca.removed_db) > 25.0
    assert np.median(run.eca.echo_lost_db) < 1.0
    assert run.scan.B.shape == (len(run.grid), len(run.sp.times))
    est = run.grid[np.argmax(run.scan.B, axis=0)]
    err = np.hypot(*(est - BODY).T)
    assert np.median(err) < 15.0, err



def test_map_scale_none_returns_the_map_before_the_division_by_its_maximum(scene):
    """``map_scale="none"`` (the scoreboard's maps) is the default map times the run's largest value."""
    env, layout, traces, direct, echo, s, ais = scene
    bc = BistaticConfig(f_hi=F_HI, reference="true_waveform", removal="none", method="coherent", coherent_norm="snr", c=C,
                        grid_dx=10.0, edge=20.0, exclusion_radius=30.0, refine=False)
    runs = {m: run_bistatic(traces - direct, layout, FS, env, ais, BistaticConfig(**{**bc.__dict__, "map_scale": m}), true_waveform=s)
            for m in ("max", "none")}
    B_max, B_none = runs["max"].scan.B, runs["none"].scan.B
    assert B_max.max() == pytest.approx(1.0) and B_none.max() > 0
    np.testing.assert_allclose(B_none / B_none.max(), B_max, rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError):
        BistaticConfig(map_scale="frame")


def test_doppler_eca_follows_a_moving_ship(scene):
    """A ship at 4 m/s: the per block fixed lags leave the delay drift inside a block, the per
    sample delays (``eca_doppler``) follow it, for both the true waveform and the data reference."""
    env, layout, _, _, _, s, _ = scene
    n = len(s)
    tt = np.arange(n) / FS
    ship_t = SHIP[None, :] + np.outer(tt, [4.0, 0.0])
    rec = layout.receivers
    direct = np.stack([np.interp(tt - np.hypot(*(ship_t - r).T) / C, tt, s, left=0.0, right=0.0) / np.sqrt(np.hypot(*(SHIP - r)))
                       for r in rec])
    traces = direct + 0.002 * np.random.default_rng(5).standard_normal(direct.shape)
    k = int(T / 0.5)
    t_ais = np.arange(k) * 0.5
    ais = AISReports(vessel=np.array(["ship"] * k), t=t_ais, xy=SHIP[None, :] + np.outer(t_ais, [4.0, 0.0]),
                     sog=np.full(k, 4.0), cog=np.zeros(k))
    depths = {}
    for reference in ("true_waveform", "data"):
        for doppler in (False, True):
            bc = BistaticConfig(f_hi=F_HI, reference=reference, removal="eca", c=C, refine=False, eca_doppler=doppler)
            sp = stft(traces, layout, FS, bc)
            model = ReplicaModel(layout, bc.c)
            model.weights = sp.weights
            ship = ship_positions(sp, ais, model, bc)
            outs, _ = eca_stage(traces, layout, FS, env, sp, ship, bc, s if reference == "true_waveform" else None, [direct])
            depths[reference, doppler] = float(np.median(_depth_db(direct, outs[1])))
    assert depths["true_waveform", False] < 20.0, depths
    assert depths["true_waveform", True] > 30.0, depths
    assert depths["data", True] > depths["data", False] + 6.0, depths


def test_track_smoothing_removes_refinement_jitter(scene):
    """Per frame position jitter of 1 m on a 4 m/s track costs the Doppler ECA most of its depth;
    a 2 s local straight line fit (``track_smooth_s``) recovers it."""
    env, layout, _, _, _, s, _ = scene
    n = len(s)
    tt = np.arange(n) / FS
    ship_t = SHIP[None, :] + np.outer(tt, [4.0, 0.0])
    rec = layout.receivers
    direct = np.stack([np.interp(tt - np.hypot(*(ship_t - r).T) / C, tt, s, left=0.0, right=0.0) / np.sqrt(np.hypot(*(SHIP - r)))
                       for r in rec])
    traces = direct + 0.002 * np.random.default_rng(5).standard_normal(direct.shape)
    bc = BistaticConfig(f_hi=F_HI, reference="true_waveform", removal="eca", c=C, refine=False, eca_doppler=True)
    sp = stft(traces, layout, FS, bc)
    exact = SHIP[None, :] + np.outer(sp.times, [4.0, 0.0])
    jitter = exact + np.random.default_rng(8).standard_normal(exact.shape)
    present = np.ones(len(sp.times), bool)
    depths = {}
    for name, smooth in (("jitter", 0.0), ("smoothed", 2.0)):
        b = BistaticConfig(**{**bc.__dict__, "track_smooth_s": smooth})
        ship = ShipTrack("ship", jitter, jitter.copy(), present, np.ones(len(sp.times)))
        outs, _ = eca_stage(traces, layout, FS, env, sp, ship, b, s, [direct])
        depths[name] = float(np.median(_depth_db(direct, outs[1])))
    assert depths["smoothed"] > depths["jitter"] + 6.0, depths
    assert depths["smoothed"] > 15.0, depths
    assert np.allclose(smooth_track(sp.times, exact, present, 2.0), exact, atol=1e-9)


# ----------------------------------------------------------------------
# The ECA in mixed precision (BistaticConfig.eca_precision = "mixed")
# ----------------------------------------------------------------------

def test_qr_solve_matches_the_loaded_normal_equations_on_a_well_conditioned_block():
    """On a well conditioned block (independent columns) the unloaded QR in double precision and the loaded normal
    equations give the same projection to single precision."""
    from acoustic_sim.water.bistatic import _project_out_batched, _project_out_qr

    rng = np.random.default_rng(4)
    S = rng.standard_normal((3, 600, 12))
    segs = rng.standard_normal((3, 600, 2))
    a = _project_out_batched(S.copy(), segs.copy())
    b = _project_out_qr(S.astype(np.float32), segs.astype(np.float32))
    assert b.dtype == np.float32
    np.testing.assert_allclose(b, a, rtol=0, atol=2e-5 * np.abs(segs).max())


@pytest.mark.parametrize("reference", ["data", "true_waveform"])
def test_mixed_precision_eca_matches_the_double_one(scene, reference):
    """The analytic case with a moving ship (per sample delays): the mixed precision ECA removes the direct path within
    0.3 dB of the double precision one and loses the echo within 0.1 dB of it."""
    env, layout, _, _, echo, s, _ = scene
    n = len(s)
    tt = np.arange(n) / FS
    ship_t = SHIP[None, :] + np.outer(tt, [4.0, 0.0])
    rec = layout.receivers
    direct = np.stack([np.interp(tt - np.hypot(*(ship_t - r).T) / C, tt, s, left=0.0, right=0.0) / np.sqrt(np.hypot(*(SHIP - r)))
                       for r in rec])
    traces = direct + echo + 0.002 * np.random.default_rng(5).standard_normal(direct.shape)
    k = int(T / 0.5)
    t_ais = np.arange(k) * 0.5
    ais = AISReports(vessel=np.array(["ship"] * k), t=t_ais, xy=SHIP[None, :] + np.outer(t_ais, [4.0, 0.0]),
                     sog=np.full(k, 4.0), cog=np.zeros(k))
    got = {}
    for prec in ("float64", "mixed"):
        bc = BistaticConfig(f_hi=F_HI, reference=reference, removal="eca", c=C, refine=False, eca_doppler=True, eca_images=False,
                            eca_precision=prec)
        sp = stft(traces, layout, FS, bc)
        model = ReplicaModel(layout, bc.c)
        model.weights = sp.weights
        ship = ship_positions(sp, ais, model, bc)
        outs, _ = eca_stage(traces, layout, FS, env, sp, ship, bc, s if reference == "true_waveform" else None, [direct, echo])
        got[prec] = (float(np.median(_depth_db(direct, outs[1]))), float(np.median(_depth_db(echo, outs[2]))))
    assert got["float64"][0] > 15.0, got                          # a deep removal to compare
    assert abs(got["mixed"][0] - got["float64"][0]) <= 0.3, got
    assert abs(got["mixed"][1] - got["float64"][1]) <= 0.1, got

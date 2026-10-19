"""Snapshot writers moved from air.plotting to solver.snapshots: old paths and solver output."""

import inspect
import re

import numpy as np

import acoustic_sim
import acoustic_sim.air.plotting as plotting
import acoustic_sim.solver.fdtd as fdtd
import acoustic_sim.solver.snapshots as snapshots
from acoustic_sim.model import create_uniform_model, create_uniform_model_3d
from acoustic_sim.solver.fdtd import FDTD3DConfig, FDTD3DSolver, FDTDConfig, FDTDSolver
from acoustic_sim.sources import StaticSource, StaticSource3D, make_wavelet_ricker


def test_legacy_snapshot_imports_are_the_solver_functions():
    assert plotting.save_snapshot is snapshots.save_snapshot
    assert plotting.save_snapshot_3d is snapshots.save_snapshot_3d
    assert plotting._to_db_spl is snapshots._to_db_spl
    assert acoustic_sim.save_snapshot is snapshots.save_snapshot
    assert acoustic_sim.save_snapshot_3d is snapshots.save_snapshot_3d


def test_shared_solver_does_not_import_air_domain():
    for module in (fdtd, snapshots):
        source = inspect.getsource(module)
        assert not re.search(r"^\s*(from|import)\s+acoustic_sim\.air\b", source, re.M), module.__name__


def test_fdtd_2d_run_writes_snapshots(tmp_path):
    model = create_uniform_model(-5, 5, -5, 5, 0.5, 343.0)
    src = StaticSource(x=0.0, y=0.0, signal=make_wavelet_ricker(400, 1e-4, 300.0))
    cfg = FDTDConfig(total_time=0.004, damping_width=4, snapshot_interval=10)
    FDTDSolver(model, cfg, src, np.array([[2.0, 0.0]])).run(snapshot_dir=str(tmp_path), verbose=False)
    assert (tmp_path / "snapshot_000000.png").is_file()


def test_fdtd_3d_run_writes_snapshots(tmp_path):
    model = create_uniform_model_3d(-3, 3, -3, 3, 0, 6, 0.5, 343.0)
    src = StaticSource3D(x=0.0, y=0.0, z=3.0, signal=make_wavelet_ricker(400, 1e-4, 300.0))
    cfg = FDTD3DConfig(total_time=0.002, damping_width=3, snapshot_interval=5)
    FDTD3DSolver(model, cfg, src, np.array([[2.0, 0.0, 3.0]])).run(
        snapshot_dir=str(tmp_path), verbose=False)
    assert sorted(p.name for p in tmp_path.glob("snapshot3d_*.png"))[:1] == ["snapshot3d_000000.png"]

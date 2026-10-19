from importlib import import_module

from acoustic_sim.air import DetectionConfig
from acoustic_sim.solver import FDTDSolver
from acoustic_sim.water import HarborFDTDSolver


def test_water_solver_extends_shared_fdtd_solver():
    assert issubclass(HarborFDTDSolver, FDTDSolver)


def test_legacy_air_and_water_imports_resolve_to_canonical_modules():
    assert import_module("acoustic_sim.processor") is import_module("acoustic_sim.air.processor")
    assert import_module("acoustic_sim.detection.bearing") is import_module("acoustic_sim.air.detection.bearing")
    assert import_module("acoustic_sim.harbor.fdtd_runner") is import_module("acoustic_sim.water.fdtd_runner")
    assert DetectionConfig.__module__ == "acoustic_sim.air.config"

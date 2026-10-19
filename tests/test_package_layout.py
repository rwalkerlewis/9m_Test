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


def test_legacy_harbor_aliases_are_lazy_and_identical():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import acoustic_sim.harbor.fdtd_runner as legacy\n"
        "import acoustic_sim.harbor\n"
        "from acoustic_sim.harbor import scenario\n"
        "import acoustic_sim.water.fdtd_runner as canonical\n"
        "import acoustic_sim.water.scenario\n"
        "assert legacy is canonical\n"
        "assert acoustic_sim.harbor.fdtd_runner is canonical\n"
        "assert scenario is acoustic_sim.water.scenario\n"
        "assert canonical.__spec__.name == 'acoustic_sim.water.fdtd_runner'\n"
        "assert acoustic_sim.harbor.HarborFDTDSolver is canonical.HarborFDTDSolver\n"
        # Optional-dependency modules are not imported until requested.
        "assert 'acoustic_sim.water.classifier' not in sys.modules\n"
        "try:\n"
        "    import acoustic_sim.harbor.no_such_module\n"
        "except ModuleNotFoundError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('missing legacy submodule imported')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_root_facade_exports_resolve():
    import acoustic_sim
    import acoustic_sim.air.detection_main as detection_main

    missing = [name for name in acoustic_sim.__all__ if not hasattr(acoustic_sim, name)]
    assert not missing
    assert acoustic_sim.run_detection is detection_main.run_detection
    assert acoustic_sim.FDTDSolver is import_module("acoustic_sim.solver").FDTDSolver


def test_root_legacy_module_aliases_are_canonical_modules():
    import acoustic_sim

    pairs = {
        **{name: f"air.{name}" for name in (
            "config", "domains", "fire_control", "forward", "noise", "processor",
            "plotting", "setup", "tracker", "validate", "detection_main", "studies",
        )},
        "detection": "air.detection",
        **{f"detection.{name}": f"air.detection.{name}"
           for name in ("bearing", "engine", "ranging", "tracking")},
        "fdtd": "solver.fdtd",
        "fdtd_cuda": "solver.fdtd_cuda",
    }
    for legacy, canonical in pairs.items():
        module = import_module(f"acoustic_sim.{canonical}")
        assert import_module(f"acoustic_sim.{legacy}") is module, legacy
        if "." not in legacy and legacy not in ("detection_main", "studies"):
            assert getattr(acoustic_sim, legacy) is module, legacy


def test_detection_main_runs_as_module_without_runpy_warning():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-m", "acoustic_sim.air.detection_main", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "usage" in result.stdout.lower()
    assert "RuntimeWarning" not in result.stderr


def test_moved_functions_keep_their_old_import_paths():
    bistatic = import_module("acoustic_sim.water.bistatic")
    motion = import_module("acoustic_sim.water.motion")
    plotting = import_module("acoustic_sim.air.plotting")
    pipeline_plots = import_module("acoustic_sim.air.pipeline_plots")

    for name in ("motion_integrate", "_motion_integrate_reference", "_motion_integrate_vectorized"):
        assert getattr(bistatic, name) is getattr(motion, name), name
    assert import_module("acoustic_sim.harbor.bistatic").motion_integrate is motion.motion_integrate
    for name in ("plot_radial_engagement", "plot_pipeline_summary", "plot_beamformer_diagnostic"):
        assert getattr(plotting, name) is getattr(pipeline_plots, name), name

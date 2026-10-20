"""Harbor DAS + hydrophone detection, localisation, tracking and classification toy model."""

from acoustic_sim.water.ais import AISReports, AISTrack
from acoustic_sim.water.ais_sim import AISConfig, generate_ais
from acoustic_sim.water.dataset import (
    HarborDataset,
    assemble_dataset,
    render_vessel_gathers,
    simulate_scenario,
)
from acoustic_sim.water.environment import (
    HarborEnvironment,
    build_synthetic_harbor,
    uniform_water_environment,
)
from acoustic_sim.water.fdtd_runner import (
    Gather,
    HarborFDTDConfig,
    HarborFDTDSolver,
    cfl_dt,
    simulate_source,
    simulate_sources,
)
from acoustic_sim.water.noise import NoiseConfig, NoiseRealisation, generate_noise
from acoustic_sim.water.scenario import (
    HarborScenarioConfig,
    build_environment,
    build_layout,
    default_vessels,
)
from acoustic_sim.water.sensors import Cable, SensorLayout
from acoustic_sim.water.vessels import Track, VesselConfig, build_track

__all__ = [
    "AISConfig",
    "AISReports",
    "AISTrack",
    "Cable",
    "Gather",
    "HarborDataset",
    "HarborEnvironment",
    "HarborFDTDConfig",
    "HarborFDTDSolver",
    "HarborScenarioConfig",
    "NoiseConfig",
    "NoiseRealisation",
    "SensorLayout",
    "Track",
    "VesselConfig",
    "assemble_dataset",
    "build_environment",
    "build_layout",
    "build_synthetic_harbor",
    "build_track",
    "cfl_dt",
    "default_vessels",
    "generate_ais",
    "generate_noise",
    "render_vessel_gathers",
    "simulate_scenario",
    "simulate_source",
    "simulate_sources",
    "uniform_water_environment",
]

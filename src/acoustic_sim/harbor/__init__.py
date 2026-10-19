"""Harbor DAS + hydrophone detection, localisation, tracking and classification toy model."""

from acoustic_sim.harbor.ais import AISConfig, AISReports, AISTrack, generate_ais
from acoustic_sim.harbor.dataset import (
    HarborDataset,
    assemble_dataset,
    render_vessel_gathers,
    simulate_scenario,
)
from acoustic_sim.harbor.environment import (
    HarborEnvironment,
    build_synthetic_harbor,
    uniform_water_environment,
)
from acoustic_sim.harbor.fdtd_runner import (
    Gather,
    HarborFDTDConfig,
    HarborFDTDSolver,
    cfl_dt,
    simulate_source,
    simulate_sources,
)
from acoustic_sim.harbor.noise import NoiseConfig, NoiseRealisation, generate_noise
from acoustic_sim.harbor.scenario import (
    HarborScenarioConfig,
    build_environment,
    build_layout,
    default_vessels,
)
from acoustic_sim.harbor.sensors import Cable, SensorLayout
from acoustic_sim.harbor.vessels import Track, VesselConfig, build_track

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

"""acoustic_sim — 2D acoustic simulation on user-defined velocity models.

Extended with passive acoustic drone detection, tracking, and fire
control capabilities using matched field processing.
"""

from acoustic_sim.model import (
    VelocityModel,
    add_circle_anomaly,
    add_rectangle_anomaly,
    create_checkerboard_model,
    create_gradient_model,
    create_layered_model,
    create_uniform_model,
    create_valley_model,
    model_from_array,
)
from acoustic_sim.sampling import check_cfl, check_spatial_sampling, suggest_dx
from acoustic_sim.solver import solve_helmholtz
from acoustic_sim.receivers import (
    create_receiver_circle,
    create_receiver_concentric,
    create_receiver_custom,
    create_receiver_l_shaped,
    create_receiver_line,
    create_receiver_random,
)
from acoustic_sim.io import load_json, load_model, model_from_json, save_model
from acoustic_sim.plotting import (
    plot_beam_power,
    plot_detection_domain,
    plot_detection_gather,
    plot_domain,
    plot_gather,
    plot_tracking,
    plot_velocity_model,
    plot_vespagram,
    plot_wavefield,
    save_snapshot,
)
from acoustic_sim.backend import get_backend
from acoustic_sim.sources import (
    CircularOrbitSource,
    CustomTrajectorySource,
    EvasiveSource,
    FigureEightSource,
    LoiterApproachSource,
    MovingSource,
    StaticSource,
    inject_source,
    load_wav_mono,
    make_drone_harmonics,
    make_source_from_file,
    make_source_noise,
    make_source_propeller,
    make_source_tone,
    make_stationary_tonal,
    make_wavelet_ricker,
    prepare_source_signal,
    source_velocity_at,
)
from acoustic_sim.domains import (
    DomainMeta,
    create_hills_vegetation_domain,
    create_isotropic_domain,
    create_wind_domain,
)
from acoustic_sim.fdtd import FDTDConfig, FDTDSolver
from acoustic_sim.setup import build_domain, build_receivers, build_source, compute_dt
from acoustic_sim.config import DetectionConfig
from acoustic_sim.noise import add_all_noise, generate_sensor_noise, generate_wind_noise
from acoustic_sim.processor import (
    apply_filter_bank,
    compute_beam_power,
    compute_travel_times,
    create_filter_bank,
    detect_stationary,
    matched_field_process,
)
from acoustic_sim.tracker import KalmanTracker, run_tracker
from acoustic_sim.fire_control import (
    compute_engagement,
    compute_lead,
    pattern_diameter,
    pellet_velocity_at_range,
    run_fire_control,
    time_of_flight,
)
from acoustic_sim.validate import (
    check_amplitude,
    check_energy,
    check_localization,
    check_snr,
    check_travel_times,
    run_all_checks,
)
from acoustic_sim.detection_main import run_detection_pipeline

__all__ = [
    # ── Model ──
    "VelocityModel",
    "add_circle_anomaly",
    "add_rectangle_anomaly",
    "create_checkerboard_model",
    "create_gradient_model",
    "create_layered_model",
    "create_uniform_model",
    "create_valley_model",
    "model_from_array",
    # ── Sampling ──
    "check_cfl",
    "check_spatial_sampling",
    "suggest_dx",
    # ── Solver ──
    "solve_helmholtz",
    # ── Receivers ──
    "create_receiver_circle",
    "create_receiver_concentric",
    "create_receiver_custom",
    "create_receiver_l_shaped",
    "create_receiver_line",
    "create_receiver_random",
    # ── I/O ──
    "load_json",
    "load_model",
    "model_from_json",
    "save_model",
    # ── Plotting ──
    "plot_beam_power",
    "plot_detection_domain",
    "plot_detection_gather",
    "plot_domain",
    "plot_gather",
    "plot_tracking",
    "plot_velocity_model",
    "plot_vespagram",
    "plot_wavefield",
    "save_snapshot",
    # ── Backend ──
    "get_backend",
    # ── Sources ──
    "CircularOrbitSource",
    "CustomTrajectorySource",
    "EvasiveSource",
    "FigureEightSource",
    "LoiterApproachSource",
    "MovingSource",
    "StaticSource",
    "inject_source",
    "load_wav_mono",
    "make_drone_harmonics",
    "make_source_from_file",
    "make_source_noise",
    "make_source_propeller",
    "make_source_tone",
    "make_stationary_tonal",
    "make_wavelet_ricker",
    "prepare_source_signal",
    "source_velocity_at",
    # ── Domains ──
    "DomainMeta",
    "create_hills_vegetation_domain",
    "create_isotropic_domain",
    "create_wind_domain",
    # ── FDTD ──
    "FDTDConfig",
    "FDTDSolver",
    # ── Setup ──
    "build_domain",
    "build_receivers",
    "build_source",
    "compute_dt",
    # ── Detection config ──
    "DetectionConfig",
    # ── Noise ──
    "add_all_noise",
    "generate_sensor_noise",
    "generate_wind_noise",
    # ── Processor ──
    "apply_filter_bank",
    "compute_beam_power",
    "compute_travel_times",
    "create_filter_bank",
    "detect_stationary",
    "matched_field_process",
    # ── Tracker ──
    "KalmanTracker",
    "run_tracker",
    # ── Fire control ──
    "compute_engagement",
    "compute_lead",
    "pattern_diameter",
    "pellet_velocity_at_range",
    "run_fire_control",
    "time_of_flight",
    # ── Validate ──
    "check_amplitude",
    "check_energy",
    "check_localization",
    "check_snr",
    "check_travel_times",
    "run_all_checks",
    # ── Pipeline ──
    "run_detection_pipeline",
]

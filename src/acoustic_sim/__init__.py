"""acoustic_sim — 2D/3D acoustic simulation on user-defined velocity models.

Extended with passive acoustic drone detection, tracking, and fire
control capabilities using matched field processing.  Includes a
comprehensive study framework for robustness analysis.
"""

from acoustic_sim.model import (
    VelocityModel,
    VelocityModel3D,
    add_circle_anomaly,
    add_rectangle_anomaly,
    create_checkerboard_model,
    create_gradient_model,
    create_layered_model,
    create_uniform_model,
    create_valley_model,
    create_uniform_model_3d,
    create_layered_z_model_3d,
    model_3d_from_array,
    model_from_array,
)
from acoustic_sim.sampling import check_cfl, check_spatial_sampling, suggest_dx
from acoustic_sim.solver import solve_helmholtz
from acoustic_sim.receivers import (
    create_receiver_circle,
    create_receiver_circle_3d,
    create_receiver_concentric,
    create_receiver_custom,
    create_receiver_custom_3d,
    create_receiver_l_shaped,
    create_receiver_l_shaped_3d,
    create_receiver_line,
    create_receiver_line_3d,
    create_receiver_log_spiral,
    create_receiver_nested_circular,
    create_receiver_nested_circular_3d,
    create_receiver_random,
    create_receiver_random_disk,
    create_receiver_random_disk_3d,
    print_array_diagnostics,
)
from acoustic_sim.io import load_json, load_model, model_from_json, save_model
from acoustic_sim.air.plotting import (
    plot_beam_power,
    plot_detection_domain,
    plot_detection_gather,
    plot_domain,
    plot_gather,
    plot_multi_track,
    plot_polar_beam_power,
    plot_study_comparison,
    plot_tracking,
    plot_velocity_model,
    plot_vespagram,
    plot_wavefield,
    save_snapshot,
    # 3D
    plot_3d_trajectory,
    plot_altitude_vs_time,
    plot_tracking_3d,
    plot_kinematic_scatter,
    save_snapshot_3d,
)
from acoustic_sim.backend import get_backend
from acoustic_sim.sources import (
    CircularOrbitSource,
    CircularOrbitSource3D,
    CustomTrajectorySource,
    CustomTrajectorySource3D,
    ErraticQuadcopterSource3D,
    EvasiveSource,
    EvasiveSource3D,
    FigureEightSource,
    FigureEightSource3D,
    LoiterApproachSource,
    LoiterApproachSource3D,
    MovingSource,
    MovingSource3D,
    StaticSource,
    StaticSource3D,
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
    source_velocity_at_3d,
)
from acoustic_sim.air.domains import (
    DomainMeta,
    DomainMeta3D,
    create_echo_canyon_domain,
    create_hills_vegetation_domain,
    create_isotropic_domain,
    create_urban_echo_domain,
    create_wind_domain,
    create_isotropic_domain_3d,
    create_wind_domain_3d,
    create_ground_layer_domain_3d,
    create_hills_vegetation_domain_3d,
)
from acoustic_sim.solver.fdtd import FDTD3DConfig, FDTD3DSolver, FDTDConfig, FDTDSolver
from acoustic_sim.air.setup import build_domain, build_receivers, build_source, compute_dt
from acoustic_sim.air.config import DetectionConfig, sound_speed_from_temperature
from acoustic_sim.air.noise import (
    add_all_noise,
    generate_sensor_noise,
    generate_wind_noise,
    inject_sensor_faults,
    inject_transient,
    perturb_mic_positions,
)
from acoustic_sim.air.processor import (
    blank_transients,
    broadband_weighted_sum,
    build_polar_grid,
    calibrate_positions,
    compute_csdm,
    compute_sensor_weights,
    compute_steering_vectors,
    compute_travel_times_polar,
    conventional_beam_power,
    detect_stationary,
    find_peaks_polar,
    matched_field_process,
    matched_field_process_3d,
    mvdr_beam_power,
    polar_to_cartesian,
    select_harmonic_bins,
)
from acoustic_sim.air.tracker import (
    EKFTracker,
    EKFTracker3D,
    MultiTargetTracker,
    MultiTargetTracker3D,
    run_multi_tracker,
    run_multi_tracker_3d,
    run_tracker,
    run_tracker_3d,
)
from acoustic_sim.air.fire_control import (
    compute_engagement,
    compute_engagement_3d,
    compute_lead,
    compute_lead_3d,
    compute_miss_distance,
    compute_miss_distance_3d,
    pattern_diameter,
    pellet_velocity_at_range,
    prioritize_threats,
    run_fire_control,
    run_fire_control_3d,
    run_multi_fire_control,
    time_of_flight,
)
from acoustic_sim.air.validate import (
    check_amplitude,
    check_energy,
    check_localization,
    check_snr,
    check_travel_times,
    run_all_checks,
)
from acoustic_sim.air.forward import (
    simulate_3d_traces,
    simulate_scenario_3d,
)
__all__ = [
    # ── Model ──
    "VelocityModel",
    "VelocityModel3D",
    "add_circle_anomaly",
    "add_rectangle_anomaly",
    "create_checkerboard_model",
    "create_gradient_model",
    "create_layered_model",
    "create_uniform_model",
    "create_valley_model",
    "model_from_array",
    "create_uniform_model_3d",
    "create_layered_z_model_3d",
    "model_3d_from_array",
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
    "create_receiver_log_spiral",
    "create_receiver_nested_circular",
    "create_receiver_random",
    "create_receiver_random_disk",
    "print_array_diagnostics",
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
    "plot_multi_track",
    "plot_polar_beam_power",
    "plot_study_comparison",
    "plot_tracking",
    "plot_velocity_model",
    "plot_vespagram",
    "plot_wavefield",
    "save_snapshot",
    "plot_3d_trajectory",
    "plot_altitude_vs_time",
    "plot_tracking_3d",
    "plot_kinematic_scatter",
    "save_snapshot_3d",
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
    "source_velocity_at_3d",
    "StaticSource3D",
    "MovingSource3D",
    "CircularOrbitSource3D",
    "FigureEightSource3D",
    "LoiterApproachSource3D",
    "EvasiveSource3D",
    "CustomTrajectorySource3D",
    # ── Domains ──
    "DomainMeta",
    "DomainMeta3D",
    "create_echo_canyon_domain",
    "create_hills_vegetation_domain",
    "create_isotropic_domain",
    "create_urban_echo_domain",
    "create_wind_domain",
    "create_isotropic_domain_3d",
    "create_wind_domain_3d",
    "create_ground_layer_domain_3d",
    "create_hills_vegetation_domain_3d",
    # ── FDTD ──
    "FDTDConfig",
    "FDTD3DConfig",
    "FDTDSolver",
    "FDTD3DSolver",
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
    "inject_sensor_faults",
    "inject_transient",
    "perturb_mic_positions",
    # ── Processor ──
    "blank_transients",
    "broadband_weighted_sum",
    "build_polar_grid",
    "calibrate_positions",
    "compute_csdm",
    "compute_sensor_weights",
    "compute_steering_vectors",
    "compute_travel_times_polar",
    "conventional_beam_power",
    "detect_stationary",
    "find_peaks_polar",
    "matched_field_process",
    "matched_field_process_3d",
    "mvdr_beam_power",
    "polar_to_cartesian",
    "select_harmonic_bins",
    # ── Tracker ──
    "EKFTracker",
    "EKFTracker3D",
    "MultiTargetTracker",
    "MultiTargetTracker3D",
    "run_multi_tracker",
    "run_multi_tracker_3d",
    "run_tracker",
    "run_tracker_3d",
    # ── Fire control ──
    "compute_engagement",
    "compute_engagement_3d",
    "compute_lead",
    "compute_lead_3d",
    "compute_miss_distance",
    "compute_miss_distance_3d",
    "pattern_diameter",
    "pellet_velocity_at_range",
    "prioritize_threats",
    "run_fire_control",
    "run_fire_control_3d",
    "run_multi_fire_control",
    "time_of_flight",
    # ── Validate ──
    "check_amplitude",
    "check_energy",
    "check_localization",
    "check_snr",
    "check_travel_times",
    "run_all_checks",
    # ── Pipeline ──
    "evaluate_results",
    "evaluate_results_3d",
    "run_detection",
    "run_detection_3d",
    "run_detection_pipeline",
    "simulate_scenario",
    "simulate_3d_traces",
    "simulate_scenario_3d",
    # ── Studies ──
    "run_all_studies",
    "study_array_geometry",
    "study_echo_domains",
    "study_haphazard_array",
    "study_min_sensors",
    "study_mixed_failures",
    "study_multi_drone",
    "study_position_errors",
    "study_sensor_faults",
    "study_transient_robustness",
]

_LAZY_AIR_EXPORTS = {
    "evaluate_results": "detection_main",
    "evaluate_results_3d": "detection_main",
    "run_detection": "detection_main",
    "run_detection_3d": "detection_main",
    "run_detection_pipeline": "detection_main",
    "simulate_scenario": "detection_main",
    "run_all_studies": "studies",
    "study_array_geometry": "studies",
    "study_echo_domains": "studies",
    "study_haphazard_array": "studies",
    "study_min_sensors": "studies",
    "study_mixed_failures": "studies",
    "study_multi_drone": "studies",
    "study_position_errors": "studies",
    "study_sensor_faults": "studies",
    "study_transient_robustness": "studies",
}


def __getattr__(name: str):
    try:
        module_name = _LAZY_AIR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(_importlib.import_module(f".air.{module_name}", __name__), name)
    globals()[name] = value
    return value


# Keep the pre-domain package paths importable while callers migrate to air/ and water/.
import importlib as _importlib
import sys as _sys

_air_modules = (
    "config",
    "domains",
    "fire_control",
    "forward",
    "noise",
    "processor",
    "plotting",
    "setup",
    "tracker",
    "validate",
)
for _module_name in _air_modules:
    _module = _importlib.import_module(
        f".air.{_module_name}", __name__
    )
    _sys.modules[f"{__name__}.{_module_name}"] = _module
    setattr(_sys.modules[__name__], _module_name, _module)
_air_detection = _importlib.import_module(".air.detection", __name__)
_sys.modules[f"{__name__}.detection"] = _air_detection
setattr(_sys.modules[__name__], "detection", _air_detection)
for _module_name in ("bearing", "engine", "ranging", "tracking"):
    _sys.modules[f"{__name__}.detection.{_module_name}"] = _importlib.import_module(
        f".air.detection.{_module_name}", __name__
    )
for _module_name in ("fdtd", "fdtd_cuda"):
    _module = _importlib.import_module(f".solver.{_module_name}", __name__)
    _sys.modules[f"{__name__}.{_module_name}"] = _module
    setattr(_sys.modules[__name__], _module_name, _module)

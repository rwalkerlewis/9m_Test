"""Detection pipeline configuration.

All tuneable parameters for the drone detection, tracking, and fire
control pipeline.  No magic numbers anywhere else in the code.

Default scenario
================
13-element nested circular array (0.5 m radius) at the centre of a
1000 × 1000 m domain.  Drone with 150 Hz fundamental and 6 harmonics
(up to 900 Hz) on a loiter-and-approach trajectory.  Frequency-domain
MVDR beamforming on a polar grid.  Extended Kalman Filter with
bearing-primary measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def sound_speed_from_temperature(t_celsius: float = 20.0) -> float:
    """Speed of sound from temperature: c = 331.3 * sqrt(1 + T/273.15)."""
    return 331.3 * math.sqrt(1.0 + t_celsius / 273.15)


@dataclass
class DetectionConfig:
    """All configurable parameters for the detection / tracking / fire-control pipeline."""

    # ── Domain ──────────────────────────────────────────────────────────
    domain_type: str = "isotropic"
    x_min: float = 0.0
    x_max: float = 1000.0
    y_min: float = 0.0
    y_max: float = 1000.0
    dx: float = 0.05
    temperature_celsius: float = 20.0
    sound_speed: float = 343.0       # overridden by temperature if set
    wind_speed: float = 0.0
    wind_direction_deg: float = 0.0
    dirt_velocity: float = 1500.0
    seed: int = 42

    # ── FDTD ────────────────────────────────────────────────────────────
    total_time: float = 2.0
    fd_order: int = 2
    damping_width: int = 40
    damping_max: float = 0.15
    air_absorption: float = 0.005
    snapshot_interval: int = 0

    # ── Array ───────────────────────────────────────────────────────────
    array_type: str = "nested_circular"
    n_mics: int = 13
    array_radius: float = 0.5
    array_inner_radius: float = 0.15
    array_center_x: float = 500.0
    array_center_y: float = 500.0
    array_spacing: float = 0.2
    mic_positions: list[tuple[float, float]] | None = None
    sample_rate: float = 4000.0

    # ── Drone source ────────────────────────────────────────────────────
    source_signal: str = "drone_harmonics"
    fundamental_freq: float = 150.0
    n_harmonics: int = 6
    harmonic_amplitudes: list[float] = field(
        default_factory=lambda: [1.0, 0.6, 0.35, 0.2, 0.12, 0.08],
    )
    source_level_dB: float = 90.0

    # ── Trajectory ──────────────────────────────────────────────────────
    trajectory_type: str = "loiter_approach"
    drone_speed: float = 15.0

    # Linear
    source_start: tuple[float, float] = (200.0, 500.0)
    source_end: tuple[float, float] = (800.0, 500.0)

    # Circular orbit
    orbit_center: tuple[float, float] = (500.0, 200.0)
    orbit_radius: float = 100.0
    orbit_start_angle: float = 0.0

    # Figure-eight
    fig8_center: tuple[float, float] = (500.0, 300.0)
    fig8_x_amp: float = 80.0
    fig8_y_amp: float = 40.0
    fig8_x_freq: float = 0.03
    fig8_y_freq: float = 0.06
    fig8_phase_offset: float = 1.5708

    # Loiter-approach (default scenario from spec)
    loiter_orbit_center: tuple[float, float] = (500.0, 200.0)
    loiter_orbit_radius: float = 100.0
    loiter_orbit_duration: float = 30.0
    loiter_approach_target: tuple[float, float] = (500.0, 500.0)

    # Evasive
    evasive_start: tuple[float, float] = (200.0, 300.0)
    evasive_heading: float = 0.0
    evasive_speed_var: float = 2.0
    evasive_heading_var: float = 0.3

    # ── Noise ───────────────────────────────────────────────────────────
    wind_noise_enabled: bool = True
    wind_noise_level_dB: float = 55.0
    wind_corner_freq: float = 15.0
    wind_correlation_length: float = 3.0

    stationary_source_enabled: bool = True
    stationary_source_pos: tuple[float, float] = (600.0, 400.0)
    stationary_source_freq: float = 60.0
    stationary_source_level_dB: float = 75.0
    stationary_source_n_harmonics: int = 4

    sensor_noise_enabled: bool = True
    sensor_noise_level_dB: float = 40.0

    # ── Matched field processor ─────────────────────────────────────────
    # Polar grid
    mfp_azimuth_spacing_deg: float = 1.0
    mfp_range_min: float = 20.0
    mfp_range_max: float = 500.0
    mfp_range_spacing: float = 5.0
    # Legacy Cartesian (used only if polar is disabled)
    mfp_grid_spacing: float = 5.0
    mfp_grid_x_range: tuple[float, float] | None = None
    mfp_grid_y_range: tuple[float, float] | None = None

    mfp_window_length: float = 0.2       # seconds
    mfp_window_overlap: float = 0.5
    mfp_n_subwindows: int = 4
    mfp_detection_threshold: float = 0.25
    mfp_min_signal_rms: float = 0.01  # Minimum RMS to consider detection valid
    mfp_harmonic_bandwidth: float = 10.0  # Hz half-width
    mfp_stationary_history: int = 10
    mfp_stationary_cv_threshold: float = 0.15
    mfp_diagonal_loading: float = 0.01   # epsilon for MVDR

    # ── Tracker (EKF) ──────────────────────────────────────────────────
    tracker_process_noise_std: float = 2.0   # m/s²
    tracker_sigma_bearing_deg: float = 3.0   # degrees
    tracker_sigma_range: float = 100.0       # metres
    tracker_initial_range_guess: float = 200.0
    tracker_measurement_noise_std: float = 5.0  # legacy, not used by EKF

    # ── Fire control ────────────────────────────────────────────────────
    weapon_position: tuple[float, float] = (500.0, 500.0)
    muzzle_velocity: float = 400.0
    pellet_decel: float = 1.5
    pattern_spread_rate: float = 0.025
    lead_max_iterations: int = 5
    range_uncertainty_fire_threshold: float = 50.0

    # ── Threat priority ─────────────────────────────────────────────────
    priority_w_range: float = 1.0
    priority_w_closing: float = 2.0
    priority_w_quality: float = 0.5

    # ── Multi ───────────────────────────────────────────────────────────
    max_sources: int = 1
    min_source_separation_m: float = 20.0
    tracker_gate_threshold: float = 30.0
    tracker_max_missed: int = 5
    n_drones: int = 1
    drone_configs: list[dict] | None = None

    # ── Robustness ──────────────────────────────────────────────────────
    enable_sensor_weights: bool = False
    sensor_fault_threshold: float = 10.0
    enable_transient_blanking: bool = False
    transient_subwindow_ms: float = 5.0
    transient_threshold_factor: float = 10.0
    enable_position_calibration: bool = False
    position_calibration_max_lag_m: float = 2.0

    # ── Fault / transient / position error injection ────────────────────
    inject_faults: bool = False
    fault_type: str = "elevated_noise"
    fault_fraction: float = 0.2
    fault_level_dB: float = 100.0
    fault_sensors: list[int] | None = None
    inject_transient: bool = False
    transient_time: float = 15.0
    transient_pos: tuple[float, float] = (550.0, 450.0)
    transient_level_dB: float = 130.0
    transient_duration_ms: float = 10.0
    inject_position_error: bool = False
    position_error_std: float = 0.01  # metres (14mm tolerance from spec)

    # ── CUDA ────────────────────────────────────────────────────────────
    use_cuda: bool = False

    # ── Output ──────────────────────────────────────────────────────────
    output_dir: str = "output/detection"

    def __post_init__(self) -> None:
        """Compute sound speed from temperature if not explicitly set."""
        self.sound_speed = sound_speed_from_temperature(self.temperature_celsius)

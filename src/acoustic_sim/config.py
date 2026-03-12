"""Detection pipeline configuration.

All tuneable parameters for the drone detection, tracking, and fire
control pipeline are collected here in a single :class:`DetectionConfig`
dataclass.  No magic numbers should appear anywhere else in the
detection code — everything references this config.

Default array scale
===================
The microphone array uses sub-metre spacing consistent with acoustic
beamforming at drone rotor frequencies (150 Hz fundamental, λ ≈ 2.3 m).
A circular array of radius 0.5 m with 16 elements gives ~0.2 m
inter-element spacing — well below the spatial Nyquist limit.

The FDTD grid spacing (dx = 0.05 m) resolves both the array geometry
and all four drone harmonics up to 600 Hz (f_max = 686 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectionConfig:
    """All configurable parameters for the detection / tracking / fire-control pipeline."""

    # ── Domain ──────────────────────────────────────────────────────────
    domain_type: str = "isotropic"
    x_min: float = -15.0
    x_max: float = 15.0
    y_min: float = -15.0
    y_max: float = 15.0
    dx: float = 0.05
    sound_speed: float = 343.0
    wind_speed: float = 0.0
    wind_direction_deg: float = 0.0
    dirt_velocity: float = 1500.0
    seed: int = 42

    # ── FDTD ────────────────────────────────────────────────────────────
    total_time: float = 0.5
    fd_order: int = 2
    damping_width: int = 40
    damping_max: float = 0.15
    air_absorption: float = 0.005
    snapshot_interval: int = 0

    # ── Array ───────────────────────────────────────────────────────────
    array_type: str = "circular"
    n_mics: int = 16
    array_radius: float = 0.5
    array_center_x: float = 0.0
    array_center_y: float = 0.0
    array_spacing: float = 0.2
    mic_positions: list[tuple[float, float]] | None = None

    # ── Drone source ────────────────────────────────────────────────────
    source_signal: str = "drone_harmonics"
    fundamental_freq: float = 150.0
    n_harmonics: int = 4
    harmonic_amplitudes: list[float] = field(
        default_factory=lambda: [1.0, 0.6, 0.3, 0.15],
    )
    source_level_dB: float = 90.0

    # ── Trajectory ──────────────────────────────────────────────────────
    trajectory_type: str = "linear"
    drone_speed: float = 15.0

    # Linear — source crosses at ~8 m from array centre
    source_start: tuple[float, float] = (-12.0, 8.0)
    source_end: tuple[float, float] = (12.0, 8.0)

    # Circular orbit
    orbit_center: tuple[float, float] = (0.0, 8.0)
    orbit_radius: float = 6.0
    orbit_start_angle: float = 0.0

    # Figure-eight (Lissajous)
    fig8_center: tuple[float, float] = (0.0, 6.0)
    fig8_x_amp: float = 6.0
    fig8_y_amp: float = 3.0
    fig8_x_freq: float = 0.3
    fig8_y_freq: float = 0.6
    fig8_phase_offset: float = 1.5708  # π/2

    # Loiter-approach
    loiter_orbit_center: tuple[float, float] = (0.0, 10.0)
    loiter_orbit_radius: float = 5.0
    loiter_orbit_duration: float = 1.0
    loiter_approach_target: tuple[float, float] = (0.0, 2.0)

    # Evasive
    evasive_start: tuple[float, float] = (-12.0, 8.0)
    evasive_heading: float = 0.0
    evasive_speed_var: float = 2.0
    evasive_heading_var: float = 0.3

    # ── Noise ───────────────────────────────────────────────────────────
    wind_noise_enabled: bool = True
    wind_noise_level_dB: float = 60.0
    wind_corner_freq: float = 15.0
    wind_correlation_length: float = 3.0

    stationary_source_enabled: bool = True
    stationary_source_pos: tuple[float, float] = (8.0, -8.0)
    stationary_source_freq: float = 60.0
    stationary_source_level_dB: float = 70.0
    stationary_source_n_harmonics: int = 4

    sensor_noise_enabled: bool = True
    sensor_noise_level_dB: float = 30.0

    # ── Matched field processor ─────────────────────────────────────────
    mfp_grid_spacing: float = 1.0
    mfp_grid_x_range: tuple[float, float] = (-14.0, 14.0)
    mfp_grid_y_range: tuple[float, float] = (-14.0, 14.0)
    mfp_window_length: float = 0.05      # seconds
    mfp_window_overlap: float = 0.5
    mfp_detection_threshold: float = 0.15
    mfp_harmonic_bandwidth: float = 20.0  # Hz half-width of each bandpass
    mfp_stationary_history: int = 10      # number of past windows
    mfp_stationary_cv_threshold: float = 0.2

    # ── Tracker (Kalman) ────────────────────────────────────────────────
    tracker_process_noise_std: float = 3.0   # m/s²
    tracker_measurement_noise_std: float = 2.0  # metres (≈ grid spacing)

    # ── Fire control ────────────────────────────────────────────────────
    weapon_position: tuple[float, float] = (0.0, 0.0)
    muzzle_velocity: float = 400.0        # m/s
    pellet_decel: float = 1.5             # m/s per metre of travel
    pattern_spread_rate: float = 0.025    # 1 m diameter per 40 m range
    lead_max_iterations: int = 5

    # ── Robustness / mitigation ─────────────────────────────────────────
    enable_sensor_weights: bool = False
    sensor_fault_threshold: float = 10.0
    enable_transient_blanking: bool = False
    transient_subwindow_ms: float = 5.0
    transient_threshold_factor: float = 10.0
    enable_position_calibration: bool = False
    position_calibration_max_lag_m: float = 2.0

    # ── Multi-source ────────────────────────────────────────────────────
    max_sources: int = 1
    min_source_separation_m: float = 5.0
    tracker_gate_threshold: float = 10.0
    tracker_max_missed: int = 5

    # ── Threat priority weights ─────────────────────────────────────────
    priority_w_range: float = 1.0
    priority_w_closing: float = 2.0
    priority_w_quality: float = 0.5

    # ── Multi-drone ─────────────────────────────────────────────────────
    n_drones: int = 1
    drone_configs: list[dict] | None = None

    # ── Sensor faults (injection) ───────────────────────────────────────
    inject_faults: bool = False
    fault_type: str = "elevated_noise"
    fault_fraction: float = 0.2
    fault_level_dB: float = 100.0
    fault_sensors: list[int] | None = None

    # ── Transient injection ─────────────────────────────────────────────
    inject_transient: bool = False
    transient_time: float = 0.25
    transient_pos: tuple[float, float] = (5.0, 5.0)
    transient_level_dB: float = 130.0
    transient_duration_ms: float = 10.0

    # ── Position errors ─────────────────────────────────────────────────
    inject_position_error: bool = False
    position_error_std: float = 0.1       # metres — small at array scale

    # ── CUDA acceleration ──────────────────────────────────────────────
    use_cuda: bool = False

    # ── Output ──────────────────────────────────────────────────────────
    output_dir: str = "output/detection"

# Configuration

Two independent configuration systems exist: one for the SRP-PHAT
engagement pipeline (`pipeline.config.json`) and one for the MFP/EKF
library pipeline (`DetectionConfig` dataclass).

---

## Pipeline Configuration (`pipeline.config.json`)

Used by `examples/run_pipeline.py`.  JSON file with six sections
(five core + optional `ml`).  CLI flags override individual values.

### `detection`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window_length_s` | float | 0.1 | analysis window length in seconds |
| `window_overlap` | float | 0.75 | fractional overlap between windows |
| `min_signal_rms` | float | 5e-5 | RMS threshold for detection (Pa) |

### `beamformer`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `method` | str | `"srp_phat"` | Bearing estimator: `srp_phat`, `music`, or `mvdr` |
| `n_bearings` | int | 360 | Number of candidate look directions |
| `freq_lo_hz` | float | 100.0 | Lower band edge (Hz) |
| `freq_hi_hz` | float | 2000.0 | Upper band edge (Hz) |
| `max_sources` | int | 1 | Maximum number of sources to detect per window |

### `ranging`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `method` | str | `"auto"` | Range method: `rms`, `tdoa`, `bearing_rate`, or `auto` |
| `n_range_bins` | int | 80 | Grid size for TDOA range search |
| `auto_cpa_threshold_m` | float | 3.0 | Auto mode: use TDOA when CPA ≤ this; else RMS |

When `method` is `"auto"`, the pipeline selects TDOA for nearfield
scenarios (ground-truth CPA ≤ threshold) and RMS otherwise.

### `tracking`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `min_detections` | int | 5 | detections needed before WLS fit |
| `max_history` | int | 20 | maximum number of detections in sliding window |
| `ema_alpha` | float | 0.35 | bearing EMA smoothing factor |
| `rms_fire_gate_frac` | float | 0.20 | fire gate: window RMS must exceed this fraction of peak RMS |
| `range_min_m` | float | 5.0 | minimum clamped range estimate (m) |
| `range_max_m` | float | 100.0 | maximum clamped range estimate (m) |
| `rms_ref_range_m` | float | 10.0 | reference range for RMS calibration (m) |
| `covariance_floor` | float | 0.5 | minimum per-axis position uncertainty |
| `covariance_cap` | float | 1.0 | maximum per-axis position uncertainty |

### `fire_control`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `muzzle_velocity_mps` | float | 400.0 | pellet muzzle velocity (m/s) |
| `pellet_decel_mps2` | float | 1.5 | velocity loss per metre of travel |
| `pattern_spread_rate` | float | 0.3 | pattern diameter per metre of range |

> **Note on `pattern_spread_rate` defaults:** Three defaults exist in
> the codebase: `DetectionConfig` uses 0.025, the `pattern_diameter()`
> function signature defaults to 0.025, and `pipeline.config.json`
> overrides to 0.3.  The pipeline config value (0.3) is what the
> production pipeline uses.  The function-level default (0.025)
> applies only when calling `pattern_diameter()` directly without
> specifying a spread rate.
| `max_engagement_range_m` | float | 500.0 | maximum engagement range (m) |
| `max_position_uncertainty_m` | float | 0.0 | position uncertainty gate (0 = disabled) |
| `class_label` | str | "fixed_wing" | assumed source class for engagement rules |
| `class_confidence` | float | 0.9 | assumed classification confidence |
| `hit_threshold_m` | float | 2.0 | miss distance for scoring a hit (m) |
| `max_hits` | int | 3 | stop pipeline after this many hits (0 = unlimited) |

### `source`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `speed_mps` | float | 50.0 | assumed source speed for ground-truth trajectory reconstruction |
| `altitude_estimate_m` | float\|null | null | assumed z coordinate; null reads from metadata `source_z` |

### `ml` (optional)

All ML features are disabled by default.  When disabled, the pipeline
produces identical results to a signal-processing-only baseline.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_source_classification` | bool | false | Enable CNN source type classification gate |
| `enable_maneuver_detection` | bool | false | Enable maneuver-adaptive tracking |
| `enable_fusion_classification` | bool | false | Enable fusion (acoustic + kinematic) classifier |
| `enable_fno_surrogate` | bool | false | Enable FNO data augmentation path |
| `enable_anomaly_detection` | bool | false | Enable CVAE anomaly detection |
| `classification_checkpoint` | str | `"output/models/acoustic_classifier.pt"` | Path to acoustic classifier weights |
| `maneuver_checkpoint` | str | `"output/models/maneuver_classifier.pt"` | Path to maneuver classifier weights |
| `fusion_checkpoint` | str | `"output/models/fusion_classifier.pt"` | Path to fusion classifier weights |
| `anomaly_checkpoint` | str | `"output/models/anomaly_detector.pt"` | Path to CVAE anomaly detector weights |
| `anomaly_threshold_file` | str | `"output/models/anomaly_threshold.json"` | Path to anomaly threshold calibration |
| `anomaly_override_confidence_threshold` | float | 0.7 | CNN confidence below this + anomaly novel = NOVEL_THREAT |
| `classification_confidence_threshold` | float | 0.7 | P(non-drone) must exceed this to reject a detection |
| `maneuver_window_size` | int | 20 | Number of kinematic history steps for maneuver classifier |
| `reject_non_drone_classes` | bool | true | Whether to suppress fire on non-drone classifications |
| `maneuver_aware_tracking` | bool | true | Whether maneuver class adjusts tracker covariance |

### CLI overrides

```
python examples/run_pipeline.py SIM_DIR \
    [--config PATH]          # JSON config file
    [--output-dir DIR]       # output directory (default: SIM_DIR)
    [--source-speed FLOAT]   # override source.speed_mps
    [--hit-threshold FLOAT]  # override fire_control.hit_threshold_m
    [--max-hits INT]         # override fire_control.max_hits
    [--enable-classification]    # enable source classification gate
    [--enable-maneuver]          # enable maneuver detection
    [--enable-fusion]            # enable fusion classification
    [--enable-anomaly]           # enable CVAE anomaly detection
    [--classification-threshold FLOAT]  # override ml.classification_confidence_threshold
    [--maneuver-window INT]      # override ml.maneuver_window_size
```

---

## DetectionConfig (MFP/EKF Library Pipeline)

Defined in `src/acoustic_sim/config.py`.  A `@dataclass` with ~80
parameters used by `detection_main.py`.  Grouped below by function.

### Domain

| Field | Type | Default | Units |
|-------|------|---------|-------|
| `domain_type` | str | `"isotropic"` | — |
| `x_min`, `x_max` | float | 0.0, 1000.0 | m |
| `y_min`, `y_max` | float | 0.0, 1000.0 | m |
| `dx` | float | 0.05 | m |
| `temperature_celsius` | float | 20.0 | °C |
| `sound_speed` | float | 343.0 | m/s |
| `wind_speed` | float | 0.0 | m/s |
| `wind_direction_deg` | float | 0.0 | ° |
| `seed` | int | 42 | — |

### FDTD

| Field | Type | Default | Units |
|-------|------|---------|-------|
| `total_time` | float | 2.0 | s |
| `fd_order` | int | 2 | — |
| `damping_width` | int | 40 | cells |
| `damping_max` | float | 0.15 | — |
| `air_absorption` | float | 0.005 | — |
| `snapshot_interval` | int | 0 | steps |

### Array

| Field | Type | Default | Units |
|-------|------|---------|-------|
| `array_type` | str | `"nested_circular"` | — |
| `n_mics` | int | 13 | — |
| `array_radius` | float | 0.5 | m |
| `array_inner_radius` | float | 0.15 | m |
| `array_center_x`, `array_center_y` | float | 500.0 | m |
| `array_spacing` | float | 0.2 | m |
| `mic_positions` | list\|None | None | m |
| `sample_rate` | float | 4000.0 | Hz |

### Source Signal

| Field | Type | Default | Units |
|-------|------|---------|-------|
| `source_signal` | str | `"drone_harmonics"` | — |
| `fundamental_freq` | float | 150.0 | Hz |
| `n_harmonics` | int | 6 | — |
| `harmonic_amplitudes` | list | [1.0, 0.6, 0.35, 0.2, 0.12, 0.08] | — |
| `source_level_dB` | float | 90.0 | dB SPL |

### Trajectory

| Field | Type | Default | Units |
|-------|------|---------|-------|
| `trajectory_type` | str | `"loiter_approach"` | — |
| `drone_speed` | float | 15.0 | m/s |
| `source_start`, `source_end` | tuple | (200,500), (800,500) | m |
| `orbit_center`, `orbit_radius` | tuple, float | (500,200), 100.0 | m |

Additional fields exist for figure-eight, loiter-approach, and evasive
trajectory parameters.

### Noise

| Field | Type | Default | Units |
|-------|------|---------|-------|
| `wind_noise_enabled` | bool | True | — |
| `wind_noise_level_dB` | float | 55.0 | dB SPL |
| `wind_corner_freq` | float | 15.0 | Hz |
| `wind_correlation_length` | float | 3.0 | m |
| `sensor_noise_enabled` | bool | True | — |
| `sensor_noise_level_dB` | float | 40.0 | dB SPL |
| `stationary_source_enabled` | bool | True | — |

### MFP

| Field | Type | Default | Units |
|-------|------|---------|-------|
| `mfp_azimuth_spacing_deg` | float | 1.0 | ° |
| `mfp_range_min`, `mfp_range_max` | float | 20.0, 500.0 | m |
| `mfp_range_spacing` | float | 5.0 | m |
| `mfp_window_length` | float | 0.2 | s |
| `mfp_window_overlap` | float | 0.5 | — |
| `mfp_detection_threshold` | float | 0.25 | — |
| `mfp_min_signal_rms` | float | 0.01 | Pa |
| `mfp_diagonal_loading` | float | 0.01 | — |

### Tracker (EKF)

| Field | Type | Default | Units |
|-------|------|---------|-------|
| `tracker_process_noise_std` | float | 2.0 | m/s² |
| `tracker_sigma_bearing_deg` | float | 3.0 | ° |
| `tracker_sigma_range` | float | 100.0 | m |
| `tracker_initial_range_guess` | float | 200.0 | m |

### Fire Control

| Field | Type | Default | Units |
|-------|------|---------|-------|
| `weapon_position` | tuple | (500, 500) | m |
| `muzzle_velocity` | float | 400.0 | m/s |
| `pellet_decel` | float | 1.5 | m/s/m |
| `pattern_spread_rate` | float | 0.025 | m/m |
| `lead_max_iterations` | int | 5 | — |

### Robustness

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_sensor_weights` | bool | False | enable fault-aware sensor weighting |
| `enable_transient_blanking` | bool | False | enable transient detection & blanking |
| `enable_position_calibration` | bool | False | enable TDOA self-calibration |
| `inject_faults` | bool | False | inject sensor faults into traces |
| `inject_transient` | bool | False | inject broadband transient |
| `inject_position_error` | bool | False | perturb mic positions |

---

## FDTD Runner Configuration

`examples/run_fdtd.py` and `run_fdtd_3d.py` accept all parameters via
CLI flags.  Key arguments:

| Flag | Description |
|------|-------------|
| `--domain` | `isotropic`, `wind`, `hills_vegetation` |
| `--source-type` | `static`, `moving` |
| `--source-signal` | `file`, `propeller`, `tone`, `noise`, `ricker` |
| `--array` | `circular`, `concentric`, `linear`, ... |
| `--total-time` | simulation duration in seconds |
| `--dx` | grid spacing in metres |
| `--fd-order` | spatial FD order: 2, 4, 6, 8 |
| `--use-cuda` | enable GPU acceleration |
| `--output-dir` | output directory |

3-D adds: `--z-min`, `--z-max`, `--source-z`, `--source-z1`.

---

## Domain JSON Configs

Example configs in `examples/` define velocity models:

- `domain.example.json` — gradient model with circle + rectangle anomalies
- `layered.example.json` — layered model with anomaly
- `valley.example.json` — valley model with randomised hills

Loaded via `acoustic-sim --model-file path.json`.

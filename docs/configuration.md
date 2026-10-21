# Configuration

Two independent configuration systems exist: one for the SRP-PHAT
engagement pipeline (`pipeline.config.json`) and one for the MFP/EKF
library pipeline (`DetectionConfig` dataclass).

The separate harbor application uses a third configuration model,
`HarborScenarioConfig`, typically loaded from
`examples/configs/harbor.scenario.json`. Its FDTD/data-processing settings are
described below and in [Harbor](harbor.md).

---

## Pipeline Configuration (`pipeline.config.json`)

Used by `examples/air/02_pipeline/run_pipeline.py`, which merges the JSON file over built-in defaults
(`_DEFAULTS` in `run_pipeline.py`) and then applies CLI overrides.  Sections: `detection`,
`beamformer`, `ranging`, `tracking`, `fire_control`, `source`, and optional `ml`.

The **Default** columns below give the effective value with the shipped
`examples/configs/pipeline.config.json` (loaded unless `--config` points elsewhere).  Where the shipped
file and the built-in default differ, both are given.  Keys absent from the shipped file
(`beamformer.method`, `beamformer.max_sources`, all of `ranging`, the `anomaly_*` keys) come from
the built-in defaults.

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
| `pattern_spread_rate` | float | 0.3 (built-in 0.2) | pattern diameter per metre of range |

| `max_engagement_range_m` | float | 500.0 | maximum engagement range (m) |
| `max_position_uncertainty_m` | float | 0.0 | position uncertainty gate (0 = disabled) |
| `class_label` | str | `"fixed_wing"` (built-in `"quadcopter"`) | assumed source class for engagement rules |
| `class_confidence` | float | 0.9 | assumed classification confidence |
| `hit_threshold_m` | float | 2.0 | miss distance for scoring a hit (m) |
| `max_hits` | int | 3 | stop pipeline after this many hits (0 = unlimited) |

> **`pattern_spread_rate` defaults:** the production pipeline uses the shipped config value
> (0.3).  `DetectionConfig` (MFP/EKF library path) and the `pattern_diameter()` function
> signature both default to 0.025; the function default applies only when `pattern_diameter()`
> is called without a spread rate.

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
| `fno_checkpoint` | str | `"checkpoints/fno_demo/fno_final.pt"` | Path to FNO weights (written by `examples/air/03_ml/run_fno.py`; not shipped) |
| `enable_anomaly_detection` | bool | false | Enable CVAE anomaly detection |
| `classification_checkpoint` | str | `"output/models/acoustic_classifier.pt"` | Path to acoustic classifier weights |
| `maneuver_checkpoint` | str | `"output/models/maneuver_classifier.pt"` | Path to maneuver classifier weights |
| `fusion_checkpoint` | str | `"output/models/fusion_classifier.pt"` | Path to fusion classifier weights |
| `anomaly_checkpoint` | str | `"output/models/anomaly_detector.pt"` | Path to CVAE anomaly detector weights (not shipped; train with `src/acoustic_sim/ml/anomaly_training.py`) |
| `anomaly_threshold_file` | str | `"output/models/anomaly_threshold.json"` | Path to anomaly threshold calibration |
| `anomaly_override_confidence_threshold` | float | 0.7 | CNN confidence below this + anomaly novel = NOVEL_THREAT |
| `classification_confidence_threshold` | float | 0.95 (built-in 0.7) | P(non-drone) must exceed this to reject a detection |
| `maneuver_window_size` | int | 20 | Number of kinematic history steps for maneuver classifier |
| `reject_non_drone_classes` | bool | true | Whether to suppress fire on non-drone classifications |
| `maneuver_aware_tracking` | bool | true | Whether maneuver class adjusts tracker covariance |

### CLI overrides

```
python examples/air/02_pipeline/run_pipeline.py SIM_DIR \
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

## Harbor Scenario (`HarborScenarioConfig`)

The harbor scenario JSON has top-level `environment`, `fdtd`, `cables`, `hydrophones`, `vessels`,
`ais`, `noise`, and `seed` fields. Load/save it with
`HarborScenarioConfig.load(path)` / `.save(path)`. Coordinates are metres in the scenario's
local frame; when `environment.crop` is set to `[x0, x1, y0, y1]`, the crop becomes a new domain
whose origin is `(x0, y0)`, and sensor/vessel coordinates must be expressed in that local frame.

| Section | Main fields | Purpose |
|---------|-------------|---------|
| `environment` | `dx`, `c0`, `shallow_c_drop`, `crop` | Plan-view grid spacing, base sound speed, shallow-water effective-speed variation, and optional crop |
| `fdtd` | `fs`, `total_time`, `fd_order`, `damping_width`, `damping_max`, `background_absorption`, `shore_loss_width`, `shore_loss_max`, `use_cuda` | Sampling rate/duration, stencil, outer sponge, background damping, shoreline damping, and GPU selection |
| `fdtd` (performance options, defaults shown) | `precision="float64"`, `kernel="auto"`, `body_masks="auto"`, `boundary="sponge"` (or `"cpml"`, with `cpml_width`, `cpml_order`, `cpml_alpha_max`), `time_scheme="leapfrog"` | GPU precision, fused CUDA kernel selection, scatterer mask placement, absorbing boundary and time-stepping scheme; see [Harbor](harbor.md) for measured effects |
| `cables` | `name`, `waypoints`, `channel_spacing`, `das_mode`, `gauge_length` | Polyline cable layout; `das_mode` is `pressure` or `strain` |
| `hydrophones` | array of `[x, y]` points | Discrete sensor positions |
| `vessels` | `id`, `kind`, `waypoints`, `speed`, `t_start`, `t_end`, `level_pa`, `ais`, `wander_sigma`, `seed`, body fields | Source/target trajectories, radiated level, AIS status and optional scattering body |
| `ais` | report intervals, error, jitter, dropout/outage probabilities | Synthetic AIS reporting behaviour |
| `noise` | ambient/shipping levels, receiver noise, gain scatter, dead channels, seed-related settings | Acoustic background and sensor imperfections |

`HarborFDTDConfig` uses `dt = 1 / fs`; the runner validates stability against the FDTD CFL
limit. The default 5 m grid is suitable for the low-frequency harbor scenario, not for resolving
a metre-scale hull. The bistatic example uses a separate 0.5 m scenario (or 1 m with
`--coarse`) and changes `fs`/analysis bandwidth with grid spacing; even the coarse grid only
crudely represents its 1 m × 3 m body.

For detector settings, `ProcessingConfig` controls STFT duration/hop, frequency band, scan grid,
channel QC, common-mode removal, interference projection, peak extraction, optional CUDA, and
the replica model (`propagator`; `None` is the free field, `SimulatorAwarePropagator` the
FDTD dispersion correction, simulator knowledge). `SuppressionConfig` controls AIS ship refinement and model/adaptive
projection. Bistatic-specific controls are in `BistaticConfig`: `reference` (`data` or
`true_waveform`), `removal` (`eca`, `patch`, `adaptive`, `patch+adaptive`, `none`), `method` (`bartlett`,
`bartlett_phat`, `coherent`, `mvdr`, `music`), scan-grid resolution and `propagator`. These
dataclasses and their defaults are documented in their module docstrings and summarized in
[Harbor](harbor.md).

---

## DetectionConfig (MFP/EKF Library Pipeline)

Defined in `src/acoustic_sim/air/config.py`. A `@dataclass` with 121
fields used by `air/detection_main.py`. The main ones are grouped below by function.

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

`examples/air/01_simulation/run_fdtd.py` and `run_fdtd_3d.py` accept all parameters via
CLI flags.  Key arguments:

| Flag | Description |
|------|-------------|
| `--domain` | `isotropic`, `wind`, `hills_vegetation` (3-D adds `ground_layer`) |
| `--source-type` | `static`, `moving` |
| `--source-signal` | `file`, `propeller`, `tone`, `noise`, `ricker` |
| `--array` | `circular`, `concentric`, `linear` (3-D adds `l_shaped`, `random`) |
| `--total-time` | simulation duration in seconds |
| `--dx` | grid spacing in metres |
| `--fd-order` | spatial FD order: 2, 4, 6, 8 |
| `--use-cuda` | enable GPU acceleration |
| `--output-dir` | output directory |

3-D adds: `--z-min`, `--z-max`, `--source-z`, `--source-z1`, `--receiver-cz`, and `--field-plane-z` (save a horizontal slice for `extract_array_traces.py`).

---

## Domain JSON Configs

Example configs in `examples/` define velocity models:

- `domain.example.json` — gradient model with circle + rectangle anomalies
- `layered.example.json` — layered model with anomaly
- `valley.example.json` — valley model with randomised hills

Loaded via `acoustic-sim --model-file path.json`.

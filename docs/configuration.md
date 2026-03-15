# Configuration Reference

This document provides a complete reference for every configurable parameter in `acoustic-sim`. Parameters are grouped by functional category and documented with their type, default value, units, and physical meaning.

---

## Table of Contents

1. [DetectionConfig Reference](#1-detectionconfig-reference)
2. [FDTDConfig Reference](#2-fdtdconfig-reference)
3. [JSON Config Format](#3-json-config-format)
4. [CLI Argument Reference](#4-cli-argument-reference)

---

## 1. DetectionConfig Reference

**Source:** `src/acoustic_sim/config.py`

`DetectionConfig` is a Python `@dataclass` that centralises all tuneable parameters for the drone detection, tracking, and fire-control pipeline. No magic numbers appear anywhere else in the code.

### 1.1 Domain Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `domain_type` | `str` | `"isotropic"` | — | Domain builder: `isotropic`, `wind`, `hills_vegetation`, `echo_canyon`, `urban_echo` |
| `x_min` | `float` | `0.0` | m | Western boundary of the domain |
| `x_max` | `float` | `1000.0` | m | Eastern boundary |
| `y_min` | `float` | `0.0` | m | Southern boundary |
| `y_max` | `float` | `1000.0` | m | Northern boundary |
| `dx` | `float` | `0.05` | m | Grid spacing. Determines maximum resolvable frequency: `f_max = c_min / (10 × dx)` |
| `temperature_celsius` | `float` | `20.0` | °C | Ambient temperature. Used to compute `sound_speed` via `c = 331.3 × √(1 + T/273.15)` |
| `sound_speed` | `float` | `343.0` | m/s | Overridden by `temperature_celsius` in `__post_init__` |
| `wind_speed` | `float` | `0.0` | m/s | Wind speed (used only when `domain_type = "wind"`) |
| `wind_direction_deg` | `float` | `0.0` | ° | Meteorological wind direction (degrees CW from +y = north) |
| `dirt_velocity` | `float` | `1500.0` | m/s | Sound speed in solid material (hills, canyon walls) |
| `seed` | `int` | `42` | — | Random seed for reproducibility |

### 1.2 FDTD Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `total_time` | `float` | `2.0` | s | Simulation duration |
| `fd_order` | `int` | `2` | — | Spatial FD stencil order (2, 4, 6, 8). Higher orders have less numerical dispersion but wider stencils |
| `damping_width` | `int` | `40` | cells | Sponge-layer absorbing boundary thickness |
| `damping_max` | `float` | `0.15` | — | Peak damping coefficient at domain edge |
| `air_absorption` | `float` | `0.005` | — | Background damping applied everywhere |
| `snapshot_interval` | `int` | `0` | steps | Save wavefield snapshot every N steps (0 = disabled) |

### 1.3 Array Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `array_type` | `str` | `"nested_circular"` | — | Array geometry: `nested_circular`, `circular`, `linear`, `concentric`, `l_shaped`, `log_spiral`, `random_disk`, `random`, `custom` |
| `n_mics` | `int` | `13` | — | Number of microphones |
| `array_radius` | `float` | `0.5` | m | Outer radius of the array |
| `array_inner_radius` | `float` | `0.15` | m | Inner ring radius (nested circular) |
| `array_center_x` | `float` | `500.0` | m | Array centre x-coordinate |
| `array_center_y` | `float` | `500.0` | m | Array centre y-coordinate |
| `array_spacing` | `float` | `0.2` | m | Element spacing (for linear/L-shaped arrays) |
| `mic_positions` | `list[tuple] \| None` | `None` | m | Custom microphone positions (overrides array_type if not None) |
| `sample_rate` | `float` | `4000.0` | Hz | Nominal sample rate (informational; actual rate is `1/dt`) |

### 1.4 Drone Source Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `source_signal` | `str` | `"drone_harmonics"` | — | Signal type: `drone_harmonics`, `propeller`, `ricker`, `tone`, `noise`, `file` |
| `fundamental_freq` | `float` | `150.0` | Hz | Fundamental BPF of the drone rotor |
| `n_harmonics` | `int` | `6` | — | Number of harmonics in the source signal |
| `harmonic_amplitudes` | `list[float]` | `[1.0, 0.6, 0.35, 0.2, 0.12, 0.08]` | — | Relative amplitude of each harmonic |
| `source_level_dB` | `float` | `90.0` | dB SPL | Source level at 1 m distance (re 20 µPa) |

### 1.5 Trajectory Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `trajectory_type` | `str` | `"loiter_approach"` | — | Trajectory: `linear`, `circular`, `figure_eight`, `loiter_approach`, `evasive` |
| `drone_speed` | `float` | `15.0` | m/s | Source movement speed |

#### Linear Trajectory

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `source_start` | `tuple[float, float]` | `(200.0, 500.0)` | m | Start position (x, y) |
| `source_end` | `tuple[float, float]` | `(800.0, 500.0)` | m | End position (x, y) |

#### Circular Orbit

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `orbit_center` | `tuple[float, float]` | `(500.0, 200.0)` | m | Orbit centre |
| `orbit_radius` | `float` | `100.0` | m | Orbit radius |
| `orbit_start_angle` | `float` | `0.0` | rad | Starting angle (0 = +x axis) |

#### Figure-Eight (Lissajous)

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `fig8_center` | `tuple[float, float]` | `(500.0, 300.0)` | m | Trajectory centre |
| `fig8_x_amp` | `float` | `80.0` | m | x-axis amplitude |
| `fig8_y_amp` | `float` | `40.0` | m | y-axis amplitude |
| `fig8_x_freq` | `float` | `0.03` | Hz | x-axis frequency |
| `fig8_y_freq` | `float` | `0.06` | Hz | y-axis frequency |
| `fig8_phase_offset` | `float` | `1.5708` | rad | y-axis phase offset (π/2) |

#### Loiter-Approach

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `loiter_orbit_center` | `tuple[float, float]` | `(500.0, 200.0)` | m | Orbit centre during loiter phase |
| `loiter_orbit_radius` | `float` | `100.0` | m | Orbit radius during loiter |
| `loiter_orbit_duration` | `float` | `30.0` | s | Duration of loiter phase |
| `loiter_approach_target` | `tuple[float, float]` | `(500.0, 500.0)` | m | Target position for approach |

#### Evasive

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `evasive_start` | `tuple[float, float]` | `(200.0, 300.0)` | m | Start position |
| `evasive_heading` | `float` | `0.0` | rad | Initial heading |
| `evasive_speed_var` | `float` | `2.0` | m/s | Speed perturbation std |
| `evasive_heading_var` | `float` | `0.3` | rad/s | Heading perturbation std |

### 1.6 Noise Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `wind_noise_enabled` | `bool` | `True` | — | Enable spatially correlated wind noise |
| `wind_noise_level_dB` | `float` | `55.0` | dB SPL | Wind noise RMS level |
| `wind_corner_freq` | `float` | `15.0` | Hz | Wind noise spectral corner frequency |
| `wind_correlation_length` | `float` | `3.0` | m | Spatial coherence decay length |
| `stationary_source_enabled` | `bool` | `True` | — | Enable stationary coherent noise source |
| `stationary_source_pos` | `tuple[float, float]` | `(600.0, 400.0)` | m | Position of stationary source |
| `stationary_source_freq` | `float` | `60.0` | Hz | Fundamental frequency |
| `stationary_source_level_dB` | `float` | `75.0` | dB SPL | Source level at 1 m |
| `stationary_source_n_harmonics` | `int` | `4` | — | Number of harmonics |
| `sensor_noise_enabled` | `bool` | `True` | — | Enable sensor self-noise |
| `sensor_noise_level_dB` | `float` | `40.0` | dB SPL | Sensor noise floor |

### 1.7 Matched Field Processor Parameters

#### Polar Grid

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `mfp_azimuth_spacing_deg` | `float` | `1.0` | ° | Azimuth grid resolution |
| `mfp_range_min` | `float` | `20.0` | m | Minimum search range |
| `mfp_range_max` | `float` | `500.0` | m | Maximum search range |
| `mfp_range_spacing` | `float` | `5.0` | m | Range grid resolution |

#### Window and Detection

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `mfp_window_length` | `float` | `0.2` | s | Processing window length |
| `mfp_window_overlap` | `float` | `0.5` | — | Overlap fraction (0.5 = 50%) |
| `mfp_n_subwindows` | `int` | `4` | — | Sub-windows for CSDM averaging |
| `mfp_detection_threshold` | `float` | `0.25` | — | Normalised beam-power threshold for detection |
| `mfp_min_signal_rms` | `float` | `0.01` | Pa | Minimum window RMS to attempt detection |
| `mfp_harmonic_bandwidth` | `float` | `10.0` | Hz | Half-width for harmonic bin selection |
| `mfp_stationary_history` | `int` | `10` | windows | Number of past maps for stationary detection |
| `mfp_stationary_cv_threshold` | `float` | `0.15` | — | Coefficient of variation threshold for stationary rejection |
| `mfp_diagonal_loading` | `float` | `0.01` | — | Diagonal loading fraction for MVDR (ε) |

#### Legacy Cartesian (unused in current code)

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `mfp_grid_spacing` | `float` | `5.0` | m | Cartesian grid spacing |
| `mfp_grid_x_range` | `tuple \| None` | `None` | m | Cartesian x search range |
| `mfp_grid_y_range` | `tuple \| None` | `None` | m | Cartesian y search range |

### 1.8 Tracker (EKF) Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `tracker_process_noise_std` | `float` | `2.0` | m/s² | Acceleration noise std for process model |
| `tracker_sigma_bearing_deg` | `float` | `3.0` | ° | Bearing measurement noise std |
| `tracker_sigma_range` | `float` | `100.0` | m | Range measurement noise std |
| `tracker_initial_range_guess` | `float` | `200.0` | m | Initial range for first detection |

### 1.9 Fire Control Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `weapon_position` | `tuple[float, float]` | `(500.0, 500.0)` | m | Weapon location (typically co-located with array) |
| `muzzle_velocity` | `float` | `400.0` | m/s | Pellet muzzle velocity |
| `pellet_decel` | `float` | `1.5` | m/s/m | Velocity loss per metre of travel |
| `pattern_spread_rate` | `float` | `0.025` | m/m | Pattern diameter per metre of range |
| `lead_max_iterations` | `int` | `5` | — | Max iterations for lead-angle convergence |
| `range_uncertainty_fire_threshold` | `float` | `50.0` | m | Max 2-σ uncertainty to engage |

### 1.10 Threat Priority Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `priority_w_range` | `float` | `1.0` | — | Weight for inverse range |
| `priority_w_closing` | `float` | `2.0` | — | Weight for closing speed |
| `priority_w_quality` | `float` | `0.5` | — | Weight for inverse uncertainty |

### 1.11 Multi-Target Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `max_sources` | `int` | `1` | — | Maximum simultaneous sources to detect |
| `min_source_separation_m` | `float` | `20.0` | m | Minimum separation for multi-source detection |
| `tracker_gate_threshold` | `float` | `30.0` | m | Gating distance for data association |
| `tracker_max_missed` | `int` | `5` | — | Consecutive misses before track deletion |
| `n_drones` | `int` | `1` | — | Number of drones (informational) |
| `drone_configs` | `list[dict] \| None` | `None` | — | Per-drone config overrides (future use) |

### 1.12 Robustness Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `enable_sensor_weights` | `bool` | `False` | — | Enable median-power sensor fault detection |
| `sensor_fault_threshold` | `float` | `10.0` | — | Power ratio threshold (10× median = fault) |
| `enable_transient_blanking` | `bool` | `False` | — | Enable impulsive sub-window blanking |
| `transient_subwindow_ms` | `float` | `5.0` | ms | Sub-window size for transient detection |
| `transient_threshold_factor` | `float` | `10.0` | — | Energy ratio for transient detection |
| `enable_position_calibration` | `bool` | `False` | — | Enable cross-correlation TDOA self-calibration |
| `position_calibration_max_lag_m` | `float` | `2.0` | m | Maximum position correction from calibration |

### 1.13 Fault / Transient / Position Error Injection

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `inject_faults` | `bool` | `False` | — | Inject sensor faults into traces |
| `fault_type` | `str` | `"elevated_noise"` | — | Fault mode: `elevated_noise`, `dropout`, `spikes`, `dc_offset` |
| `fault_fraction` | `float` | `0.2` | — | Fraction of sensors to fault |
| `fault_level_dB` | `float` | `100.0` | dB SPL | Fault noise / spike level |
| `fault_sensors` | `list[int] \| None` | `None` | — | Specific sensor indices to fault |
| `inject_transient` | `bool` | `False` | — | Inject broadband transient event |
| `transient_time` | `float` | `15.0` | s | Time of transient event |
| `transient_pos` | `tuple[float, float]` | `(550.0, 450.0)` | m | Position of transient source |
| `transient_level_dB` | `float` | `130.0` | dB SPL | Transient source level at 1 m |
| `transient_duration_ms` | `float` | `10.0` | ms | Transient pulse duration |
| `inject_position_error` | `bool` | `False` | — | Add Gaussian errors to mic positions |
| `position_error_std` | `float` | `0.01` | m | Per-axis position error std |

### 1.14 CUDA and Output

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `use_cuda` | `bool` | `False` | — | Use CuPy for GPU acceleration |
| `output_dir` | `str` | `"output/detection"` | — | Directory for output files |

---

## 2. FDTDConfig Reference

**Source:** `src/acoustic_sim/fdtd.py`

`FDTDConfig` controls the FDTD solver directly (used by `run_fdtd.py`).

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `total_time` | `float` | `0.3` | s | Simulation duration |
| `dt` | `float \| None` | `None` | s | Timestep. `None` = auto-compute from CFL |
| `cfl_safety` | `float` | `0.9` | — | Fraction of CFL limit for auto-dt (must be < 1) |
| `damping_width` | `int` | `40` | cells | Sponge-layer thickness |
| `damping_max` | `float` | `0.15` | — | Peak sponge damping coefficient |
| `air_absorption` | `float` | `0.005` | — | Background damping everywhere |
| `snapshot_interval` | `int` | `50` | steps | Save snapshot every N steps (0 = disabled) |
| `source_amplitude` | `float` | `1.0` | Pa | Peak source amplitude multiplier |
| `use_cuda` | `bool` | `False` | — | Use CuPy for GPU acceleration |
| `fd_order` | `int` | `2` | — | Spatial FD order (2, 4, 6, 8, …) |

**Auto-dt formula:**

```
dt = cfl_safety × 2·dx / ((c_max + |v_wind|) × √(2·ρ_stencil))
```

where `ρ_stencil` is the spectral radius of the FD stencil at Nyquist frequency.

---

## 3. JSON Config Format

JSON configuration files define velocity models for the Helmholtz solver. They are loaded by `io.model_from_json()`.

### 3.1 Schema

```json
{
    "bounds": {
        "x_min": -20.0,
        "x_max": 20.0,
        "y_min": -20.0,
        "y_max": 20.0
    },
    "dx": 0.4,
    "type": "uniform | layered | gradient | checkerboard | valley",
    "background_velocity": 343.0,

    // Type-specific fields:
    "layers": [{"y": -10.0, "velocity": 360.0}, ...],     // layered
    "v_bottom": 360.0, "v_top": 320.0,                     // gradient
    "cell_size": 4.0, "perturbation": 20.0,                // checkerboard
    "dirt_velocity": 1500.0, "seed": 42,                    // valley
    "hill_south_y": -20.0, "hill_north_y": 20.0,           // valley
    "hill_peak_height": 18.0, "hill_base_width": 60.0,     // valley
    "saddle_x": 0.0, "saddle_width": 12.0,                 // valley
    "saddle_depth_frac": 0.55,                              // valley

    "anomalies": [
        {
            "type": "circle",
            "center": [8.0, -5.0],
            "radius": 3.5,
            "velocity": 290.0
        },
        {
            "type": "rectangle",
            "x_min": -4.0, "x_max": 4.0,
            "y_min": 8.0, "y_max": 14.0,
            "velocity": 310.0
        }
    ]
}
```

### 3.2 Field Descriptions

| Field | Required | Default | Description |
|---|---|---|---|
| `bounds.x_min/x_max/y_min/y_max` | No | ±20 m | Domain extent |
| `dx` | No | 0.4 m | Grid spacing |
| `type` | No | `"uniform"` | Model type |
| `background_velocity` | No | 343.0 m/s | Default velocity |
| `layers` | For `layered` | — | List of `{y, velocity}` layer boundaries |
| `v_bottom`, `v_top` | For `gradient` | 360, 330 | Bottom and top velocities |
| `cell_size` | For `checkerboard` | 4.0 m | Checker cell size |
| `perturbation` | For `checkerboard` | 20.0 m/s | Velocity perturbation |
| `dirt_velocity` | For `valley` | 1500 m/s | Hill material velocity |
| `seed` | For `valley` | 42 | Random seed for hill profiles |
| `anomalies` | No | `[]` | List of circular/rectangular velocity anomalies |

### 3.3 Examples

**Gradient with anomalies** (`examples/domain.example.json`):
```json
{
    "bounds": {"x_min": -20.0, "x_max": 20.0, "y_min": -20.0, "y_max": 20.0},
    "dx": 0.4,
    "type": "gradient",
    "v_bottom": 360.0,
    "v_top": 320.0,
    "anomalies": [
        {"type": "circle", "center": [8.0, -5.0], "radius": 3.5, "velocity": 290.0},
        {"type": "rectangle", "x_min": -4.0, "x_max": 4.0, "y_min": 8.0, "y_max": 14.0, "velocity": 310.0}
    ]
}
```

**Layered with anomaly** (`examples/layered.example.json`):
```json
{
    "bounds": {"x_min": -20.0, "x_max": 20.0, "y_min": -20.0, "y_max": 20.0},
    "dx": 0.4,
    "type": "layered",
    "background_velocity": 343.0,
    "layers": [
        {"y": -10.0, "velocity": 360.0},
        {"y": 0.0, "velocity": 343.0},
        {"y": 10.0, "velocity": 320.0}
    ],
    "anomalies": [
        {"type": "circle", "center": [5.0, -5.0], "radius": 3.0, "velocity": 280.0}
    ]
}
```

**Valley** (`examples/valley.example.json`):
```json
{
    "bounds": {"x_min": -50.0, "x_max": 50.0, "y_min": -50.0, "y_max": 50.0},
    "dx": 0.5,
    "type": "valley",
    "background_velocity": 343.0,
    "dirt_velocity": 1500.0,
    "seed": 42,
    "hill_south_y": -20.0,
    "hill_north_y": 20.0,
    "hill_peak_height": 18.0,
    "saddle_width": 12.0,
    "saddle_depth_frac": 0.55
}
```

---

## 4. CLI Argument Reference

### 4.1 Helmholtz CLI (`acoustic-sim`)

```
acoustic-sim [OPTIONS]
```

#### Model Source (mutually exclusive)

| Argument | Description |
|---|---|
| `--model-file PATH` | JSON config file |
| `--model-npz PATH` | Pre-built .npz model |
| `--model-preset {uniform,layered,gradient,checkerboard,valley}` | Built-in preset |

#### Domain Geometry

| Argument | Default | Description |
|---|---|---|
| `--x-min` | -20.0 | Western boundary [m] |
| `--x-max` | 20.0 | Eastern boundary [m] |
| `--y-min` | -20.0 | Southern boundary [m] |
| `--y-max` | 20.0 | Northern boundary [m] |
| `--dx` | 0.4 | Grid spacing [m] |
| `--bg-velocity` | 343.0 | Background wave speed [m/s] |

#### Source

| Argument | Default | Description |
|---|---|---|
| `--source-x` | 0.0 | Source x position [m] |
| `--source-y` | 0.0 | Source y position [m] |

#### Receivers

| Argument | Default | Description |
|---|---|---|
| `--receiver-type {line,circle}` | `circle` | Array geometry |
| `--receiver-count` | 16 | Number of receivers |
| `--receiver-radius` | 0.2 | Circle radius [m] |
| `--receiver-x0/y0/x1/y1` | -15/0/15/0 | Line endpoints [m] |

#### Helmholtz

| Argument | Default | Description |
|---|---|---|
| `--frequency` | 480.0 | Source frequency [Hz] |
| `--min-ppw` | 10.0 | Minimum points per wavelength |

#### Valley-Specific

| Argument | Default | Description |
|---|---|---|
| `--hill-south-y` | -20.0 | Southern ridge y-centre [m] |
| `--hill-north-y` | 20.0 | Northern ridge y-centre [m] |
| `--hill-peak-height` | 18.0 | Max ridge height [m] |
| `--saddle-width` | 12.0 | Saddle notch width [m] |
| `--saddle-depth` | 0.55 | Saddle depth fraction |
| `--dirt-velocity` | 1500.0 | Hill material velocity [m/s] |
| `--seed` | 42 | Random seed |

#### Output

| Argument | Default | Description |
|---|---|---|
| `--velocity-plot` | `velocity_model.png` | Velocity model plot path |
| `--field-plot` | `wavefield.png` | Wavefield plot path |
| `--save-model-path` | None | Save model to .npz |

### 4.2 FDTD CLI (`examples/run_fdtd.py`)

```
python examples/run_fdtd.py [OPTIONS]
```

#### Domain

| Argument | Default | Description |
|---|---|---|
| `--domain {isotropic,wind,hills_vegetation}` | `isotropic` | Domain type |
| `--velocity` | 343.0 | Background velocity [m/s] |
| `--dx` | 0.5 | Grid spacing [m] |
| `--x-min/max` | ±50 | Domain bounds [m] |
| `--y-min/max` | ±50 | Domain bounds [m] |
| `--wind-speed` | 15.0 | Wind speed [m/s] |
| `--wind-dir` | 45.0 | Wind direction [°] |
| `--dirt-velocity` | 1500.0 | Hill material velocity [m/s] |
| `--seed` | 42 | Random seed |

#### Source

| Argument | Default | Description |
|---|---|---|
| `--source-type {static,moving}` | `static` | Source motion |
| `--source-x/y` | 0/0 | Source position [m] |
| `--source-x1/y1` | 30/0 | Moving source endpoint [m] |
| `--source-speed` | 50.0 | Moving source speed [m/s] |
| `--source-arc-height` | 0.0 | Parabolic arc height [m] |

#### Source Signal

| Argument | Default | Description |
|---|---|---|
| `--source-signal {file,propeller,tone,noise,ricker}` | `ricker` | Signal type |
| `--source-wav` | `audio/input.wav` | WAV file path |
| `--max-seconds` | None | Truncate audio [s] |
| `--source-freq` | 25.0 | Frequency [Hz] |
| `--blade-count` | 3 | Propeller blades |
| `--rpm` | 3600 | Rotor RPM |
| `--harmonics` | 14 | Number of harmonics |

#### Receivers

| Argument | Default | Description |
|---|---|---|
| `--array {concentric,circular,linear}` | `circular` | Array type |
| `--receiver-count` | 16 | Number of receivers |
| `--receiver-radius` | 15.0 | Array radius [m] |
| `--receiver-radii` | `"10,20,30,40"` | Concentric ring radii |
| `--receiver-cx/cy` | 0/0 | Array centre [m] |
| `--receiver-x0/y0/x1/y1` | -40/0/40/0 | Linear endpoints [m] |

#### Simulation

| Argument | Default | Description |
|---|---|---|
| `--total-time` | 0.3 | Duration [s] |
| `--snapshot-interval` | 50 | Steps between snapshots |
| `--damping-width` | 40 | Sponge width [cells] |
| `--damping-max` | 0.15 | Peak sponge damping |
| `--source-amplitude` | 1.0 | Source amplitude [Pa] |
| `--air-absorption` | 0.005 | Background damping |
| `--use-cuda` | off | GPU acceleration |
| `--fd-order` | 2 | FD spatial order |
| `--output-dir` | `output/test` | Output directory |

### 4.3 Detection Pipeline CLI (`detection_main.py`)

```
python src/acoustic_sim/detection_main.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--trajectory` | `linear` | Trajectory: linear, circular, figure_eight, loiter_approach, evasive |
| `--domain` | `isotropic` | Domain: isotropic, wind, hills_vegetation, echo_canyon, urban_echo |
| `--total-time` | 0.5 | Duration [s] |
| `--dx` | 0.05 | Grid spacing [m] |
| `--x-min/max` | ±15 | Domain bounds [m] |
| `--y-min/max` | ±15 | Domain bounds [m] |
| `--n-mics` | 16 | Microphone count |
| `--array-type` | `circular` | Array geometry |
| `--array-radius` | 0.5 | Array radius [m] |
| `--drone-speed` | 15.0 | Drone speed [m/s] |
| `--source-level-dB` | 90.0 | Source level [dB SPL] |
| `--fundamental-freq` | 150.0 | Fundamental [Hz] |
| `--no-noise` | off | Disable all noise |
| `--no-stationary` | off | Disable stationary source |
| `--output-dir` | `output/detection` | Output directory |
| `--grid-spacing` | 1.0 | MFP grid spacing [m] |
| `--detection-threshold` | 0.15 | Detection threshold |
| `--snapshot-interval` | 0 | FDTD snapshot interval |

---

## 5. 3D Detection Pipeline Parameters

**Source:** `src/acoustic_sim/detection_main_3d.py` → `run_detection_3d()`

The 3D detection pipeline accepts all the same parameters as the 2D pipeline (MFP, tracker, fire control) plus additional parameters for the z-dimension search grid and ML integration. These are keyword arguments to `run_detection_3d()`.

### 5.1 Z-Grid Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `z_min` | `float` | `0.0` | m | Minimum z-value in the search grid |
| `z_max` | `float` | `200.0` | m | Maximum z-value in the search grid |
| `z_spacing` | `float` | `10.0` | m | Z-grid resolution. Smaller values give finer altitude estimation at higher computational cost |

When `z_min == z_max == 0`, the 3D MFP degenerates to a single z-slice and produces results identical to the 2D processor.

### 5.2 ML Classifier Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `acoustic_model` | `nn.Module \| None` | `None` | — | Pre-trained `AcousticClassifier` for source classification. `None` = disabled |
| `fusion_model` | `nn.Module \| None` | `None` | — | Pre-trained `FusionClassifier` for acoustic+kinematic classification. `None` = disabled |
| `maneuver_model` | `nn.Module \| None` | `None` | — | Pre-trained `ManeuverClassifier` for maneuver detection. `None` = disabled |
| `confidence_threshold` | `float` | `0.7` | — | Minimum classification confidence to accept a label. Below this, the label is overridden to `"unknown"` |
| `kinematic_buffer_size` | `int` | `50` | detections | Number of detections accumulated before switching from acoustic-only to fusion classification |
| `maneuver_buffer_size` | `int` | `20` | tracker steps | Minimum tracker history length before maneuver classification is attempted |

**Classifier precedence:** If both `acoustic_model` and `fusion_model` are provided, the pipeline uses acoustic-only classification until `kinematic_buffer_size` detections have been accumulated (providing enough tracker history for kinematic features), then switches to fusion classification.

### 5.3 3D Fire Control Class-Based Parameters

These parameters control class-based engagement rules in `compute_engagement_3d()`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_label` | `str` | `"unknown"` | Source classification from the ML classifier |
| `class_confidence` | `float` | `0.0` | Classification confidence score |
| `confidence_threshold` | `float` | `0.7` | Minimum confidence for class-based engagement rules to apply |
| `maneuver_class` | `str` | `"steady"` | Current maneuver state from the maneuver classifier |

**Threat classification rules:**
- Threat classes: `quadcopter`, `hexacopter`, `fixed_wing` → engagement permitted
- Non-threat classes: `bird`, `ground_vehicle`, `unknown` → engagement suppressed (`can_fire = False`)
- Low confidence: Engagement suppressed if `class_confidence < confidence_threshold` and `class_label ≠ "unknown"`

---

## 6. 3D Forward Model Parameters

**Source:** `src/acoustic_sim/forward_3d.py`

### 6.1 Analytical Forward Model (`simulate_3d_traces`)

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `source` | object | — | — | 3D source with `position_at(step, dt) → (x, y, z)` and `signal` |
| `mic_positions` | `ndarray` | — | m | `(n_mics, 3)` or `(n_mics, 2)` (z=0 assumed) |
| `dt` | `float` | — | s | Timestep |
| `n_steps` | `int` | — | — | Number of simulation steps |
| `sound_speed` | `float` | `343.0` | m/s | Propagation velocity |
| `air_absorption` | `float` | `0.005` | 1/m | Exponential absorption coefficient |
| `enable_ground_reflection` | `bool` | `False` | — | Add ground-reflected image source |
| `ground_reflection_coeff` | `float` | `-0.9` | — | Reflection coefficient (negative = phase flip) |
| `ground_z` | `float` | `0.0` | m | Ground plane z-coordinate |

### 6.2 FDTD-Based 3D Forward Model (`simulate_3d_traces_fdtd`)

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `source` | object | — | — | 3D source object |
| `mic_positions` | `ndarray` | — | m | Microphone positions |
| `dt` | `float \| None` | `None` | s | Timestep. `None` = auto from CFL |
| `total_time` | `float` | `1.0` | s | Simulation duration |
| `sound_speed` | `float` | `343.0` | m/s | Uniform velocity |
| `dx` | `float` | `1.0` | m | Grid spacing |
| `domain_margin` | `float` | `20.0` | m | Extra padding beyond source/receiver extent |
| `z_min` | `float` | `-5.0` | m | Domain z-minimum |
| `z_max` | `float` | `120.0` | m | Domain z-maximum |
| `damping_width` | `int` | `10` | cells | Sponge-layer width |
| `fd_order` | `int` | `2` | — | Spatial FD order |
| `air_absorption` | `float` | `0.005` | — | Background damping |
| `source_amplitude` | `float` | `1.0` | Pa | Peak source amplitude |
| `verbose` | `bool` | `True` | — | Print progress |

### 6.3 Full 3D Scenario (`simulate_scenario_3d`)

In addition to the forward-model parameters, `simulate_scenario_3d()` accepts noise parameters:

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `sources` | `list` | — | — | List of 3D source objects |
| `wind_noise_enabled` | `bool` | `True` | — | Enable spatially correlated wind noise |
| `wind_noise_level_dB` | `float` | `55.0` | dB SPL | Wind noise RMS level |
| `wind_corner_freq` | `float` | `15.0` | Hz | Wind noise spectral corner |
| `wind_correlation_length` | `float` | `3.0` | m | Spatial coherence decay length |
| `sensor_noise_enabled` | `bool` | `True` | — | Enable sensor self-noise |
| `sensor_noise_level_dB` | `float` | `40.0` | dB SPL | Sensor noise floor |
| `seed` | `int` | `42` | — | Random seed |

---

## 7. ML Training Parameters

**Source:** `src/acoustic_sim/ml/training.py`, `src/acoustic_sim/ml/data_generation.py`

### 7.1 Data Generation Parameters

#### Classification Dataset (`generate_classification_dataset`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_samples_per_class` | `int` | `200` | Samples generated per source class (total = 6 × 200 = 1200) |
| `dt` | `float` | `1/4000` | Sample timestep (4 kHz effective sample rate) |
| `window_duration` | `float` | `0.5` | Duration of each sample in seconds |
| `sound_speed` | `float` | `343.0` | Propagation velocity for forward model |
| `seed` | `int` | `42` | Random seed for reproducibility |

#### Maneuver Dataset (`generate_maneuver_dataset`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_samples_per_class` | `int` | `400` | Samples generated per maneuver class (total = 6 × 400 = 2400) |
| `window_size` | `int` | `20` | Number of tracker time steps per segment |
| `dt_tracker` | `float` | `0.1` | Time between consecutive tracker updates [s] |
| `seed` | `int` | `42` | Random seed |

### 7.2 Mel Spectrogram Parameters (`compute_mel_spectrogram`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_fft` | `int` | `512` | FFT window size (frequency resolution = fs / n_fft) |
| `hop_length` | `int` | `128` | Hop between consecutive frames (time resolution = hop / fs) |
| `n_mels` | `int` | `64` | Number of mel filterbank bands |
| `f_min` | `float` | `20.0` | Lowest mel filter frequency [Hz] |
| `f_max` | `float \| None` | `None` | Highest mel filter frequency (None = Nyquist) |

### 7.3 Training Hyperparameters

#### Single-Input Classifier (`train_classifier`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_epochs` | `int` | `50` | Number of training epochs |
| `lr` | `float` | `1e-3` | Adam learning rate |
| `batch_size` | `int` | `32` | Mini-batch size |
| `verbose` | `bool` | `True` | Print progress every 10 epochs |

#### Fusion Classifier (`train_fusion_classifier`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_epochs` | `int` | `50` | Number of training epochs |
| `lr` | `float` | `5e-4` | Lower learning rate to protect pre-trained acoustic branch |
| `batch_size` | `int` | `32` | Mini-batch size |

### 7.4 Source Classes

| Index | Class Name | Description |
|---|---|---|
| 0 | `quadcopter` | 4-rotor multi-rotor drone |
| 1 | `hexacopter` | 6-rotor multi-rotor drone |
| 2 | `fixed_wing` | Fixed-wing aircraft with single propeller |
| 3 | `bird` | Avian target |
| 4 | `ground_vehicle` | Ground-based motorised vehicle |
| 5 | `unknown` | Ambiguous or unclassifiable target |

### 7.5 Maneuver Classes

| Index | Class Name | Description |
|---|---|---|
| 0 | `steady` | Constant velocity straight-line flight |
| 1 | `turning` | Circular arc at constant speed |
| 2 | `accelerating` | Linear trajectory with changing speed |
| 3 | `diving` | Steep descent |
| 4 | `evasive` | Rapidly varying heading and speed |
| 5 | `hovering` | Near-zero velocity station-keeping |

---

*Next: [Study Methodology & Results](studies.md) — Detailed study documentation, including notes on 3D extension.*

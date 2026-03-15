# Code Architecture & Module Reference

This document describes the package structure, data flow, design principles, and provides a detailed reference for every module in the `acoustic-sim` codebase.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Package Structure](#2-package-structure)
3. [Data Flow](#3-data-flow)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Module Reference — 2D Core](#5-module-reference)
   - [config.py](#51-configpy)
   - [model.py](#52-modelpy)
   - [sampling.py](#53-samplingpy)
   - [solver.py](#54-solverpy)
   - [backend.py](#55-backendpy)
   - [sources.py](#56-sourcespy)
   - [domains.py](#57-domainspy)
   - [fdtd.py](#58-fdtdpy)
   - [receivers.py](#59-receiverspy)
   - [io.py](#510-iopy)
   - [plotting.py](#511-plottingpy)
   - [noise.py](#512-noisepy)
   - [processor.py](#513-processorpy)
   - [tracker.py](#514-trackerpy)
   - [fire_control.py](#515-fire_controlpy)
   - [validate.py](#516-validatepy)
   - [setup.py](#517-setuppy)
   - [detection_main.py](#518-detection_mainpy)
   - [studies.py](#519-studiespy)
   - [cli.py](#520-clipy)
   - [\_\_init\_\_.py](#521-__init__py)
   - [\_\_main\_\_.py](#522-__main__py)
6. [Module Reference — 3D Extension](#6-module-reference--3d-extension)
   - [model_3d.py](#61-model_3dpy)
   - [domains_3d.py](#62-domains_3dpy)
   - [sources_3d.py](#63-sources_3dpy)
   - [receivers_3d.py](#64-receivers_3dpy)
   - [forward_3d.py](#65-forward_3dpy)
   - [fdtd_3d.py](#66-fdtd_3dpy)
   - [processor_3d.py](#67-processor_3dpy)
   - [tracker_3d.py](#68-tracker_3dpy)
   - [fire_control_3d.py](#69-fire_control_3dpy)
   - [plotting_3d.py](#610-plotting_3dpy)
   - [detection_main_3d.py](#611-detection_main_3dpy)
7. [Module Reference — ML Classification](#7-module-reference--ml-classification)
   - [ml/features.py](#71-mlfeaturesspy)
   - [ml/acoustic_classifier.py](#72-mlacoustic_classifierpy)
   - [ml/fusion_classifier.py](#73-mlfusion_classifierpy)
   - [ml/maneuver_classifier.py](#74-mlmaneuver_classifierpy)
   - [ml/training.py](#75-mltrainingpy)
   - [ml/data_generation.py](#76-mldata_generationpy)
8. [Example Scripts](#8-example-scripts)

---

## 1. Design Principles

### 1.1 Separation of Concerns

The codebase enforces a strict **information barrier** between simulation and detection:

- **Stage 1 (Simulation)** generates synthetic sensor data using full knowledge of the environment (velocity model, source position, domain geometry).
- **Stage 2 (Detection)** processes sensor data using **only** what a real sensor system would have: pressure traces, microphone positions, sample rate, and local sound speed. It has **no access** to the velocity model, source trajectory, or simulation parameters.
- **Stage 3 (Evaluation)** compares detection output to ground truth, but is never invoked during detection.

This ensures that detection algorithms are physically honest — they cannot "cheat" by using simulation metadata.

### 1.2 Composability

Every module exposes functions that accept plain data types (numpy arrays, floats, lists) rather than tightly coupled objects. This allows:

- Running the MFP on real sensor data (not just FDTD traces)
- Swapping array geometries, noise models, or trackers independently
- Reusing FDTD traces across many detection parameter sweeps (as the studies do)

### 1.3 Physical Units

All quantities are in SI units throughout:

- Pressure: Pascals (Pa)
- Distance: metres (m)
- Time: seconds (s)
- Frequency: Hertz (Hz)
- Velocity: metres per second (m/s)
- Angles: radians (internally), degrees (user-facing display)

The sole exception is dB SPL, which is always referenced to `p_ref = 20 µPa`.

---

## 2. Package Structure

```
src/acoustic_sim/
├── __init__.py              # Public API (re-exports all 2D symbols)
├── __main__.py              # python -m acoustic_sim
├── cli.py                   # Helmholtz CLI entry point
├── config.py                # DetectionConfig dataclass
│
│  ── 2D Simulation Core ──
├── model.py                 # VelocityModel + creation helpers
├── sampling.py              # Spatial sampling & CFL checks
├── solver.py                # Helmholtz (frequency-domain) solver
├── backend.py               # NumPy / CuPy abstraction
├── sources.py               # Source signals & trajectory classes
├── domains.py               # Domain builders (5 types)
├── fdtd.py                  # 2D FDTD solver (MPI + CUDA)
├── receivers.py             # Receiver array geometry factories
├── io.py                    # JSON / NPZ I/O
├── plotting.py              # All 2D visualisation functions
│
│  ── 2D Detection Pipeline ──
├── noise.py                 # Post-hoc noise generators
├── processor.py             # 2D Matched field processor (MVDR)
├── tracker.py               # 2D EKF tracker & multi-target tracker
├── fire_control.py          # 2D ballistics & engagement logic
├── validate.py              # Automated sanity checks
├── setup.py                 # High-level builders
├── detection_main.py        # 2D three-stage pipeline orchestrator
├── studies.py               # Parametric study framework
│
│  ── 3D Extension ──
├── model_3d.py              # VelocityModel3D + creation helpers
├── domains_3d.py            # 3D domain builders (isotropic, wind, ground-layer)
├── sources_3d.py            # 3D source trajectories (7 types with altitude)
├── receivers_3d.py          # 3D receiver array wrappers
├── forward_3d.py            # Analytical 3D forward model + FDTD bridge
├── fdtd_3d.py               # 3D FDTD solver (MPI z-slab decomposition)
├── processor_3d.py          # 3D Matched field processor (polar+z grid)
├── tracker_3d.py            # 3D EKF tracker (6-state) & multi-target
├── fire_control_3d.py       # 3D fire control (az+el lead, class-based rules)
├── plotting_3d.py           # 3D visualisation (trajectory, altitude, tracking)
├── detection_main_3d.py     # 3D pipeline orchestrator with ML integration
│
│  ── ML Classification ──
└── ml/
    ├── __init__.py          # ML package init
    ├── features.py          # Mel spectrogram & kinematic feature extraction
    ├── acoustic_classifier.py  # CNN for acoustic source classification
    ├── fusion_classifier.py    # Two-branch acoustic+kinematic fusion
    ├── maneuver_classifier.py  # 1D CNN for maneuver detection
    ├── training.py          # Training & evaluation loops
    └── data_generation.py   # Physics-based training data synthesis
```

### Dependency Graph — 2D Pipeline

Arrows indicate "imports from". The graph flows from low-level utilities at the bottom to high-level orchestrators at the top:

```
                     studies.py
                        │
                  detection_main.py
                 /    |    |     \
            setup.py  |  noise.py validate.py
           /  |  \    |    |
    domains  recv  sources  |
       |      |      |     |
     model  model  model   |
       |                   |
    sampling            processor.py
       |               /        \
     model        tracker.py  fire_control.py
                     |
                  (numpy, scipy)
                     |
                  backend.py
                     |
                fdtd.py ──── plotting.py
                  |
               model.py
```

### Dependency Graph — 3D Extension

The 3D modules mirror the 2D structure with `_3d` suffixes. Each 3D module imports from its 2D counterpart where possible:

```
                detection_main_3d.py
               /    |     |     \
     forward_3d   proc_3d  tracker_3d  fire_control_3d
       |    \        |         |              |
    fdtd_3d  \   processor.py  |         fire_control.py
       |      \      |         |
    model_3d   \  noise.py     |
       |        \              |
    domains_3d  sources_3d   plotting_3d
                   |
                sources.py  (signal generators)
```

### Dependency Graph — ML Subsystem

```
        detection_main_3d.py  (imports classifiers at runtime)
               |
    ┌──────────┼──────────┐
    │          │          │
 acoustic   fusion    maneuver
 classifier classifier classifier
    │          │          │
    └──────────┼──────────┘
               │
          training.py
               │
        data_generation.py ──── features.py
               │
    sources_3d.py + forward_3d.py + noise.py
```

Key dependencies:
- `model.py` is the foundation — everything depends on `VelocityModel`
- `fdtd.py` depends on `model.py`, `backend.py`, `sources.py`, `domains.py`
- `processor.py` is self-contained (no FDTD dependency)
- `detection_main.py` orchestrates the 2D pipeline; `detection_main_3d.py` orchestrates the 3D pipeline
- `studies.py` wraps `detection_main.py` in parameter sweeps
- 3D modules import FD coefficients and config from 2D (`fdtd_3d` imports from `fdtd`)
- ML modules are optional — `detection_main_3d.py` checks for `None` models and runs without ML
- `data_generation.py` depends on the 3D forward model for physics-based training data

---

## 3. Data Flow

### 3.1 Simulation Data Flow

```
DetectionConfig
    │
    ▼
build_domain() ──► VelocityModel + DomainMeta
    │
build_receivers() ──► mic_positions (n_mics, 2)
    │
compute_dt() ──► dt, f_max
    │
build_source() ──► Source object (with signal + trajectory)
    │
FDTDSolver.run() ──► traces (n_mics, n_steps), dt
    │
add_all_noise() ──► noisy_traces
    │
[optional] inject_sensor_faults() ──► faulted_traces
[optional] inject_transient() ──► traces with transient
[optional] perturb_mic_positions() ──► reported_positions
```

### 3.2 Detection Data Flow

```
SENSOR OBSERVABLES ONLY:
    traces, mic_positions, dt, sound_speed, weapon_position
        │
        ▼
matched_field_process()
    │  ► detections: list of {time, bearing, range, coherence, detected, beam_power_map}
    │  ► multi_detections: list of lists (multi-peak)
    │  ► azimuths, ranges, grid_x, grid_y, sensor_weights, filtered_traces
    │
    ▼
run_tracker() / run_multi_tracker()
    │  ► track: {times, positions, velocities, covariances, ...}
    │  ► multi_tracks: list of tracks
    │
    ▼
run_fire_control() / run_multi_fire_control()
    │  ► fire_control: {times, aim_bearings, lead_angles, tofs, can_fire, ranges, intercepts, reasons}
    │
    ▼
[returned to caller — no ground truth access]
```

### 3.3 Evaluation Data Flow

```
detection_output + ground_truth (true_positions, true_times)
    │
    ▼
evaluate_results()
    │  ► detection_rate, mean_loc_error, first_shot_miss, etc.
    │
compute_miss_distance()
    │  ► miss_distances, would_hit, first_shot_hit
```

### 3.4 3D Simulation Data Flow

```
Source list (3D sources with altitude)
    │
    ▼
simulate_scenario_3d() or simulate_scenario_3d_fdtd()
    │  ► For each source: simulate_3d_traces() or FDTD3DSolver.run()
    │  ► Sum all source traces
    │  ► Add wind noise + sensor noise
    │  ► Return: traces, mic_positions, true_positions (3D), true_velocities (3D)
```

### 3.5 3D Detection Data Flow

```
SENSOR OBSERVABLES ONLY:
    traces, mic_positions (3D), dt, sound_speed, weapon_position (3D)
        │
        ▼
matched_field_process_3d()
    │  ► detections with bearing, range, z, coherence
    │  ► 3D beam_power_map (n_az × n_range × n_gz)
    │
    ▼
[optional] _classify_detections()
    │  ► class_label, class_confidence (from acoustic or fusion model)
    │
    ▼
run_tracker_3d()
    │  ► track: positions (N,3), velocities (N,3), covariances (N,6,6)
    │
    ▼
[optional] _detect_maneuvers()
    │  ► maneuver_class, maneuver_history
    │
    ▼
run_fire_control_3d()
    │  ► aim_bearings, aim_elevations, lead_angles (az+el), can_fire, reasons
    │  ► class-based and maneuver-based engagement rules applied
```

### 3.6 ML Training Data Flow

```
generate_classification_dataset()
    │  ► For each class × n_samples:
    │     generate_source_signal() → MovingSource3D → simulate_3d_traces()
    │     → add noise → beamform → store (signal, label)
    │  ► prepare_acoustic_data() → mel spectrograms → (X_train, y_train)
    │
    ▼
train_classifier(AcousticClassifier, X_train, y_train, ...)
    │  ► trained acoustic model
    │
    ▼
compute_kinematic_features() → (X_kinematic)
    │
    ▼
train_fusion_classifier(FusionClassifier, X_acoustic, X_kinematic, y, ...)
    │  ► trained fusion model

generate_maneuver_dataset()
    │  ► For each maneuver class × n_samples:
    │     generate trajectory segment → add tracker noise → (features, label)
    │
    ▼
train_classifier(ManeuverClassifier, X_train, y_train, ...)
    │  ► trained maneuver model
```

---

## 4. Pipeline Architecture

### 4.1 The Three-Stage Pipeline

The function `run_detection_pipeline()` in `detection_main.py` orchestrates the complete flow:

```
Stage 1: simulate_scenario(config)
    - Builds domain, receivers, source from config
    - Runs FDTD (one or two times: drone + optional stationary source)
    - Adds noise (wind + sensor)
    - Returns: {traces, mic_positions, dt, sound_speed, true_positions, ...}

Stage 2: run_detection(traces, mic_positions, dt, sound_speed, ...)
    - MFP → tracker → fire control
    - Uses ONLY sensor observables
    - Returns: {mfp_result, track, multi_tracks, fire_control, ...}

Stage 3: evaluate_results(detection_output, true_positions, ...)
    - Compares detections to ground truth
    - Computes detection rate, localisation error, miss distance
    - Returns metrics dict
```

Post-processing:
- Sanity checks (validate.py)
- Plots (plotting.py): detection_domain, detection_gather, beam_power, tracking, vespagram, polar_beam_power

### 4.3 The 3D Pipeline

The 3D pipeline (`detection_main_3d.py`) follows the same three-stage structure but with 3D modules and optional ML integration:

```
Stage 1: simulate_scenario_3d(sources, mic_positions, ...)
    - Uses analytical forward model (1/r + optional ground reflection)
    - OR simulate_scenario_3d_fdtd() for wave-equation simulation
    - Adds noise (wind + sensor)
    - Returns: {traces, mic_positions(3D), true_positions(3D), true_velocities(3D), ...}

Stage 2: run_detection_3d(traces, mic_positions, dt, ..., acoustic_model, fusion_model, maneuver_model)
    - 3D MFP → optional source classification → 3D tracker → optional maneuver detection → 3D fire control
    - ML models are optional: None = baseline mode (no classification)
    - Class label and maneuver state feed into fire-control engagement rules
    - Returns: {mfp_result, track, fire_control, class_label, maneuver_class, ...}

Stage 3: evaluate_results_3d(detection_output, true_positions, true_velocities, ...)
    - 3D localisation error: Euclidean distance in (x, y, z)
    - 3D miss distance computation
    - Returns metrics dict
```

Post-processing:
- 3D plots (plotting_3d.py): 3D trajectory, altitude vs time, 6-panel tracking display, kinematic scatter

### 4.2 One-FDTD-Many-Detections Pattern

The study framework exploits the pipeline separation: a single expensive FDTD run generates base traces, which are then reused across many detection configurations:

```python
scenario = simulate_scenario(base_config)  # ~minutes
for each parameter variation:
    modified_traces = inject_faults(scenario["traces"])  # ~milliseconds
    detection = run_detection(modified_traces, ...)      # ~seconds
    metrics = evaluate_results(detection, ...)
```

This pattern is used by `study_sensor_faults`, `study_transient_robustness`, `study_position_errors`, and `study_mixed_failures`.

---

## 5. Module Reference

### 5.1 `config.py`

**Purpose:** Centralised configuration for the entire detection pipeline. All tuneable parameters are defined here — no magic numbers elsewhere.

**Key Class:**

#### `DetectionConfig`

A `@dataclass` with ~100 fields covering domain, FDTD, array, source, trajectory, noise, MFP, tracker, fire control, robustness, and output parameters. See [Configuration Reference](configuration.md) for the complete field listing.

**Key Function:**

#### `sound_speed_from_temperature(t_celsius: float) -> float`

Computes the speed of sound from temperature using the formula `c = 331.3 × √(1 + T/273.15)`.

- `t_celsius`: Temperature in degrees Celsius (default 20.0)
- Returns: Sound speed in m/s (343.2 at 20°C)

**Design Note:** `DetectionConfig.__post_init__` auto-computes `sound_speed` from `temperature_celsius`.

---

### 5.2 `model.py`

**Purpose:** The `VelocityModel` dataclass and factory functions for creating velocity fields.

#### `VelocityModel`

Fields:
- `x: np.ndarray` — 1D array of x cell-centre coordinates
- `y: np.ndarray` — 1D array of y cell-centre coordinates
- `values: np.ndarray` — 2D array `[ny, nx]` of wave speeds in m/s
- `dx: float` — Grid spacing in x direction
- `dy: float` — Grid spacing in y direction

Properties: `nx`, `ny`, `shape`, `extent` (for matplotlib `imshow`), `c_min`, `c_max`

Methods:
- `velocity_at(px, py) -> float` — Nearest-neighbour velocity look-up at an arbitrary point

#### Factory Functions

| Function | Description |
|---|---|
| `create_uniform_model(x_min, x_max, y_min, y_max, dx, velocity)` | Constant velocity everywhere |
| `create_layered_model(..., layers, background)` | Horizontal layers with specified velocities |
| `create_gradient_model(..., v_bottom, v_top)` | Linear vertical gradient |
| `create_checkerboard_model(..., cell_size, v_base, perturbation)` | Alternating velocity perturbations |
| `create_valley_model(...)` | Two ridges with a valley, randomised hill profiles, optional saddle |
| `model_from_array(values, x_min, x_max, y_min, y_max)` | Wrap an existing 2D array |

#### Anomaly Injection

| Function | Description |
|---|---|
| `add_circle_anomaly(model, cx, cy, radius, velocity)` | Circular region with altered velocity |
| `add_rectangle_anomaly(model, x0, x1, y0, y1, velocity)` | Rectangular region with altered velocity |

Both return a **new** `VelocityModel` (non-mutating).

---

### 5.3 `sampling.py`

**Purpose:** Grid resolution and stability checks.

#### `check_spatial_sampling(model, frequency_hz, min_ppw=10.0) -> dict`

Verifies `λ_min / dx ≥ min_ppw`. Returns `{valid, min_wavelength, ppw, required_ppw, max_dx, message}`.

#### `check_cfl(model, dt) -> dict`

Verifies `c_max × dt / dx ≤ 1/√2`. Returns `{valid, courant, limit, message}`.

#### `suggest_dx(c_min, frequency_hz, min_ppw=10.0) -> float`

Returns the maximum allowable `dx` for a given frequency and minimum velocity.

---

### 5.4 `solver.py`

**Purpose:** 2D Helmholtz (frequency-domain) solver.

#### `solve_helmholtz(model, source_xy, frequency_hz, damping_width=None) -> np.ndarray`

Assembles and solves the sparse Helmholtz system.

- `model`: VelocityModel
- `source_xy`: Source position `[x, y]`
- `frequency_hz`: Driving frequency
- `damping_width`: Sponge width (auto-chosen if None)
- Returns: `np.ndarray` shape `(ny, nx)` — pressure magnitude `|P(x,y)|`

Raises `RuntimeError` if the solution contains non-finite values.

---

### 5.5 `backend.py`

**Purpose:** Abstraction layer for NumPy/CuPy interchangeability.

#### `get_backend(use_cuda=False) -> tuple[ModuleType, bool]`

Returns `(xp, is_cuda)` where `xp` is either `numpy` or `cupy`. Falls back to NumPy with a warning if CuPy is unavailable. The FDTD solver uses `xp` for all array operations.

---

### 5.6 `sources.py`

**Purpose:** Source signal generators and trajectory classes.

#### WAV File Loading

`load_wav_mono(path: str) -> tuple[np.ndarray, int]` — Load a WAV file, convert to mono float32 in [-1, 1], return `(audio, sample_rate)`. Handles PCM int16, int32, uint8, and float formats.

#### Signal Generators

| Function | Description | Key Parameters |
|---|---|---|
| `make_wavelet_ricker(n_steps, dt, f0)` | Ricker (Mexican-hat) wavelet | Peak frequency `f0` |
| `make_source_propeller(n_steps, dt, f_max, ...)` | Synthetic rotor noise | BPF harmonics, modulation, broadband |
| `make_source_tone(n_steps, dt, frequency_hz)` | Pure sine wave | Single frequency |
| `make_source_noise(n_steps, dt, f_low, f_high)` | Band-limited coloured noise | Passband limits |
| `make_source_from_file(path, n_steps, dt, f_max)` | WAV file, filtered & resampled | LP filter at `f_max` |
| `make_drone_harmonics(n_steps, dt, ...)` | Drone harmonic series | Fundamental, amplitudes, SPL |
| `make_stationary_tonal(n_steps, dt, ...)` | Stationary source harmonics | Base freq, broadband level |

#### Signal Conditioning

`prepare_source_signal(raw, fs_audio, dt_sim, f_max)` — Low-pass filters at `f_max`, resamples from audio rate to simulation rate using rational polyphase resampling (falling back to FFT resampling if factors are too large).

#### Trajectory Classes

All trajectory classes expose `position_at(step, dt) -> (x, y)`.

| Class | Trajectory | Key Parameters |
|---|---|---|
| `StaticSource` | Fixed position | `x`, `y` |
| `MovingSource` | Linear A→B with optional parabolic arc | Start/end points, speed, `arc_height` |
| `CircularOrbitSource` | Circular orbit | Centre, radius, angular velocity |
| `FigureEightSource` | Lissajous curve | Centre, amplitudes, frequencies, phase |
| `LoiterApproachSource` | Orbit then linear approach | Orbit params, approach target |
| `EvasiveSource` | Random-walk heading | Mean speed/heading + variances |
| `CustomTrajectorySource` | User-supplied waypoints | Time/position arrays, linear interp |

#### Utility Functions

- `source_velocity_at(source, step, dt) -> (vx, vy)` — Central finite-difference velocity for any source class
- `inject_source(field, sx, sy, amplitude, x_arr, y_arr, dx, dy, ...)` — Bilinear source injection into a pressure field

---

### 5.7 `domains.py`

**Purpose:** Domain builders that return `(VelocityModel, DomainMeta)` pairs.

#### `DomainMeta`

Fields:
- `wind_vx: float`, `wind_vy: float` — Wind velocity components
- `attenuation: np.ndarray | None` — Per-cell damping coefficients
- `description: str` — Human-readable description

#### Domain Builders

| Function | Description | Velocity Structure |
|---|---|---|
| `create_isotropic_domain(...)` | Uniform, no wind | Constant `c` |
| `create_wind_domain(..., wind_speed, wind_direction_deg)` | Uniform + wind | Constant `c`, wind metadata |
| `create_hills_vegetation_domain(...)` | Valley with ridges + vegetation | Air + dirt + attenuation zones |
| `create_echo_canyon_domain(...)` | Parallel high-impedance walls | Air + 2000 m/s walls |
| `create_urban_echo_domain(...)` | Rectangular buildings | Air + 2500 m/s blocks |

---

### 5.8 `fdtd.py`

**Purpose:** 2D FDTD acoustic wave-equation solver with MPI and CUDA.

#### `FDTDConfig`

Configuration dataclass. Fields:
- `total_time: float` — Simulation duration [s]
- `dt: float | None` — Timestep (None = auto from CFL)
- `cfl_safety: float` — Fraction of CFL limit (default 0.9)
- `damping_width: int` — Sponge thickness in cells
- `damping_max: float` — Peak sponge damping
- `air_absorption: float` — Background damping everywhere
- `snapshot_interval: int` — Save snapshot every N steps (0 = off)
- `source_amplitude: float` — Peak source pressure [Pa]
- `use_cuda: bool` — Use CuPy for GPU
- `fd_order: int` — Spatial FD order (2, 4, 6, 8, ...)

#### `FDTDSolver`

Constructor: `FDTDSolver(model, config, source, receivers, domain_meta=None)`

Key internal state:
- `p_now`, `p_prev`: Current and previous pressure fields (local sub-domain)
- `C2`: Squared Courant number array
- `sigma`: Damping array
- `traces`: Receiver time-series (rank 0 only)

Key methods:
- `run(snapshot_dir=None, verbose=True) -> dict` — Run the full simulation, return `{traces, dt, n_steps}`

Internal methods:
- `_step(n)` — Single timestep: halo exchange → Laplacian → update → boundary → inject → sample
- `_halo_exchange()` — MPI ghost-row exchange
- `_inject(p, n)` — Bilinear source injection
- `_sample_receivers(n)` — Bilinear sampling + MPI gather
- `_gather_field()` — Collect full field on rank 0 for snapshots

#### Utility Functions

- `fd2_coefficients(order) -> np.ndarray` — Central FD coefficients for d²/dx²
- `fd2_cfl_factor(coeffs) -> float` — Spectral radius at Nyquist

---

### 5.9 `receivers.py`

**Purpose:** Microphone array geometry factories.

All functions return `np.ndarray` of shape `(N, 2)` containing `(x, y)` positions.

| Function | Geometry | Parameters |
|---|---|---|
| `create_receiver_nested_circular(cx, cy, inner_radius, outer_radius, n_inner, n_outer)` | Centre + inner ring + outer ring | Default: 13 elements (1+4+8) |
| `create_receiver_circle(cx, cy, radius, count)` | Single ring | N equally spaced |
| `create_receiver_line(x_start, y_start, x_end, y_end, count)` | Straight line | Linearly spaced |
| `create_receiver_concentric(cx, cy, radii, counts_per_ring)` | Multiple rings | List of radii/counts |
| `create_receiver_l_shaped(n1, n2, spacing, origin_x, origin_y)` | L-shaped | Two perpendicular arms |
| `create_receiver_log_spiral(count, radius, cx, cy)` | Golden-angle spiral | Maximum baseline diversity |
| `create_receiver_random_disk(count, radius, cx, cy, seed)` | Random within disk | Rejection sampling |
| `create_receiver_random(count, x_min, x_max, y_min, y_max, seed)` | Random within box | Uniform sampling |
| `create_receiver_custom(positions)` | User-specified | List of (x,y) tuples |

#### `print_array_diagnostics(positions, sound_speed=343.0) -> dict`

Computes and prints:
- Number of elements and unique baselines
- Min/max baseline lengths
- Spatial aliasing frequencies
- Angular resolution at key frequencies (300, 600, 1000, 1500 Hz)

---

### 5.10 `io.py`

**Purpose:** JSON configuration loading and velocity model persistence.

| Function | Description |
|---|---|
| `load_json(path) -> dict \| None` | Load a JSON file |
| `save_model(model, path)` | Save VelocityModel to `.npz` |
| `load_model(path) -> VelocityModel` | Load VelocityModel from `.npz` |
| `model_from_json(cfg) -> VelocityModel` | Build model from JSON config dict |

The JSON format supports model types (`uniform`, `layered`, `gradient`, `checkerboard`, `valley`), bounds, anomalies, and all relevant parameters. See [Configuration Reference](configuration.md#3-json-config-format).

---

### 5.11 `plotting.py`

**Purpose:** All visualisation functions. Uses matplotlib with the `Agg` backend (no display required).

#### Simulation Plots

| Function | Description |
|---|---|
| `plot_velocity_model(model, ...)` | Velocity field with receivers/source overlays |
| `plot_wavefield(model, field, ...)` | Helmholtz pressure magnitude |
| `plot_domain(model, ..., attenuation, wind_vx, wind_vy, source_path)` | Domain with vegetation overlay, wind arrow, source trajectory |
| `plot_gather(traces, dt, ...)` | Receiver traces as dB SPL image |
| `save_snapshot(model, field, step, ...)` | Single FDTD wavefield frame |

#### Detection Plots

| Function | Description |
|---|---|
| `plot_detection_domain(model, receivers, source_positions, ...)` | Overview with mics, trajectory, weapon |
| `plot_detection_gather(traces, filtered_traces, dt, ...)` | Two-panel raw/filtered gather |
| `plot_beam_power(results, true_positions, grid_x, grid_y, ...)` | Multi-panel beam-power snapshots |
| `plot_polar_beam_power(results, azimuths, ranges, ...)` | Polar beam-power maps |
| `plot_tracking(track, true_positions, true_times, fire_control, weapon_pos, ...)` | Four-panel: bearing, range, lead, engagement |
| `plot_vespagram(traces, mic_positions, dt, ...)` | Beam power vs. slowness and time |

#### Study Plots

| Function | Description |
|---|---|
| `plot_study_comparison(labels, metrics, ...)` | Bar charts comparing cases |
| `plot_multi_track(tracks, true_positions_list, ...)` | Multiple tracks on spatial plot |

All functions write PNG files at 150–170 DPI and close the figure.

---

### 5.12 `noise.py`

**Purpose:** Post-hoc noise generators for microphone traces.

See [Algorithms — Noise Generation](algorithms.md#6-noise-generation) for detailed algorithms.

| Function | Description |
|---|---|
| `generate_wind_noise(mic_positions, n_samples, dt, ...)` | Spatially correlated, spectrally shaped wind noise |
| `generate_sensor_noise(n_mics, n_samples, dt, ...)` | Uncorrelated white Gaussian |
| `add_all_noise(traces, stationary_traces, mic_positions, dt, ...)` | Sum drone + stationary + wind + sensor noise |
| `inject_sensor_faults(traces, fault_type, ...)` | Inject faults (noise, dropout, spikes, DC) |
| `inject_transient(traces, dt, event_time, event_pos, ...)` | Broadband impulse with propagation |
| `perturb_mic_positions(true_positions, error_std, ...)` | Gaussian position errors |

---

### 5.13 `processor.py`

**Purpose:** Frequency-domain MVDR matched field processor.

See [Algorithms — Matched Field Processor](algorithms.md#3-matched-field-processor) for the full pipeline.

#### Grid and Steering

| Function | Description |
|---|---|
| `build_polar_grid(azimuth_spacing_deg, range_min, range_max, range_spacing)` | Returns `(azimuths_rad, ranges_m)` |
| `polar_to_cartesian(azimuths, ranges, center_x, center_y)` | Convert polar to `(gx, gy)` arrays |
| `compute_travel_times_polar(gx, gy, mic_positions, sound_speed)` | Returns `(n_az, n_range, n_mics)` travel times |
| `compute_steering_vectors(freqs, travel_times)` | Returns `(n_freq, n_az, n_range, n_mics)` complex steering |

#### CSDM and Beamforming

| Function | Description |
|---|---|
| `select_harmonic_bins(n_fft, dt, fundamental, n_harmonics, bandwidth)` | FFT bin indices for harmonics |
| `compute_csdm(traces, dt, window_start, window_length, freq_bins, n_subwindows)` | CSDM averaged over sub-windows |
| `mvdr_beam_power(csdm, steering, epsilon)` | MVDR beam power with diagonal loading |
| `conventional_beam_power(csdm, steering)` | Delay-and-sum beam power |
| `broadband_weighted_sum(beam_powers, freqs)` | Frequency-weighted broadband sum |

#### Robustness

| Function | Description |
|---|---|
| `compute_sensor_weights(traces, fault_threshold)` | Median-power fault detection |
| `blank_transients(traces, dt, subwindow_ms, threshold_factor)` | Zero out impulsive sub-windows |
| `detect_stationary(history, cv_threshold)` | Identify stationary grid points from beam-power history |
| `calibrate_positions(traces, reported_positions, dt, sound_speed, max_lag_m)` | Cross-correlation TDOA calibration |

#### Peak Finding

| Function | Description |
|---|---|
| `find_peaks_polar(bpm, azimuths, ranges, threshold, max_sources, min_sep_deg)` | Peak finding with sub-grid interpolation |

#### Main Entry Point

`matched_field_process(traces, mic_positions, dt, ...)` — Runs the full MFP pipeline. Returns dict with `detections`, `multi_detections`, `azimuths`, `ranges`, `grid_x`, `grid_y`, `sensor_weights`, `calibrated_positions`, `selected_freqs`, `filtered_traces`.

---

### 5.14 `tracker.py`

**Purpose:** Extended Kalman Filter tracker and multi-target tracker.

See [Algorithms — EKF Tracker](algorithms.md#4-extended-kalman-filter-tracker).

#### `EKFTracker`

Constructor: `EKFTracker(process_noise_std, sigma_bearing, sigma_range, initial_range_guess, source_level_estimate)`

Methods:
- `initialise_from_bearing(bearing, range_est, center_x, center_y)` — Set initial state
- `predict(dt)` — Constant-velocity prediction
- `update(bearing, range_est, amplitude, center_x, center_y)` — EKF measurement update
- `get_state() -> np.ndarray` — [x, y, vx, vy]
- `get_position() -> (float, float)`
- `get_velocity() -> (float, float)`
- `get_covariance() -> np.ndarray` — 4×4 covariance matrix
- `get_range_uncertainty(cx, cy) -> float` — 1-σ radial uncertainty

#### `MultiTargetTracker`

Constructor: `MultiTargetTracker(process_noise_std, sigma_bearing_deg, sigma_range, initial_range_guess, gate_threshold, max_missed, source_level_dB)`

Methods:
- `set_array_center(cx, cy)` — Set array centre for measurement model
- `update(detections, t)` — Process new detections (predict, associate, update, birth/death)
- `get_tracks() -> list[dict]` — Confirmed tracks
- `get_all_tracks() -> list[dict]` — All tracks (including unconfirmed)

#### High-Level Functions

| Function | Description |
|---|---|
| `run_tracker(detections, ...)` | Run single-target EKF on detection list |
| `run_multi_tracker(multi_detections, times, ...)` | Run multi-target tracker on multi-peak detections |

---

### 5.15 `fire_control.py`

**Purpose:** Shotgun ballistics and engagement logic.

See [Algorithms — Fire Control](algorithms.md#5-fire-control).

#### Ballistics

| Function | Description |
|---|---|
| `time_of_flight(range_m, muzzle_velocity, decel)` | Pellet TOF with drag |
| `pellet_velocity_at_range(range_m, muzzle_velocity, decel)` | Pellet speed at range |
| `pattern_diameter(range_m, spread_rate)` | Shot pattern diameter |

#### Lead and Engagement

| Function | Description |
|---|---|
| `compute_lead(target_pos, target_vel, weapon_pos, ...)` | Iterative lead-angle solution |
| `compute_engagement(target_pos, target_vel, target_cov, weapon_pos, ...)` | Engagement envelope decision |

#### High-Level Functions

| Function | Description |
|---|---|
| `run_fire_control(track, ...)` | FC solution at every tracked timestep |
| `compute_miss_distance(fire_control, true_positions, true_times, ...)` | Miss distance against ground truth |
| `prioritize_threats(tracks, weapon_pos, ...)` | Score and rank multiple targets |
| `run_multi_fire_control(tracks, ...)` | FC for multiple targets in priority order |

---

### 5.16 `validate.py`

**Purpose:** Five automated sanity checks.

| Function | Check | Criterion |
|---|---|---|
| `check_amplitude(traces, max_pressure)` | Peak pressure | ≤ 200 Pa |
| `check_snr(traces, filtered_traces, mic_positions, source_positions)` | SNR at closest mic | > 0 dB |
| `check_travel_times(mic_positions, sound_speed, dt)` | Travel-time table | Error < 1 sample |
| `check_localization(traces, dt, mic_positions, true_pos, ...)` | Bearing to known source | Error < 10° |
| `check_energy(traces, dt, source_level_dB, mic_positions, source_positions)` | Total received energy | Within 2 orders of magnitude |
| `run_all_checks(...)` | All five | All pass |

---

### 5.17 `setup.py`

**Purpose:** High-level builder functions that map plain parameters to domain/receiver/source objects. Used by both `run_fdtd.py` and `detection_main.py`.

| Function | Description |
|---|---|
| `build_domain(domain, ...)` | Map domain name → VelocityModel + DomainMeta |
| `build_receivers(array, ...)` | Map array name → receiver positions |
| `compute_dt(model, meta, cfl_safety, fd_order)` | Compute `(dt, f_max)` from CFL |
| `build_source(source_type, signal_type, ...)` | Map type names → Source object with signal |

---

### 5.18 `detection_main.py`

**Purpose:** Three-stage pipeline orchestrator with CLI.

#### Pipeline Functions

| Function | Description |
|---|---|
| `simulate_scenario(config)` | Stage 1: FDTD → sensor observables + ground truth |
| `run_detection(traces, mic_positions, dt, ...)` | Stage 2: MFP → tracker → fire control (sensor-only) |
| `evaluate_results(detection_output, true_positions, ...)` | Stage 3: Compare to ground truth |
| `run_detection_pipeline(config)` | All three stages + plots + reporting |

#### CLI

`detection_main.py` can be run as a script: `python src/acoustic_sim/detection_main.py [options]`

Arguments: `--trajectory`, `--domain`, `--total-time`, `--dx`, domain bounds, array params, noise toggles, `--output-dir`.

---

### 5.19 `studies.py`

**Purpose:** Nine parametric robustness studies.

| Function | What It Varies |
|---|---|
| `study_array_geometry(...)` | Array type: circular, linear, l_shaped, random, concentric |
| `study_min_sensors(...)` | n_mics: 4, 6, 8, 12, 16, 24 |
| `study_sensor_faults(...)` | Fault fraction: 0–50%, with/without mitigation |
| `study_multi_drone(...)` | 1 vs 2 simultaneous sources |
| `study_transient_robustness(...)` | Transient level: 0–130 dB, with/without blanking |
| `study_haphazard_array(...)` | Circular vs 3 random placements |
| `study_echo_domains(...)` | Isotropic vs canyon vs urban |
| `study_position_errors(...)` | Position error: 0–5 m, with/without calibration |
| `study_mixed_failures(...)` | Progressive combination of all failure modes |
| `run_all_studies(...)` | Runs sensor faults, transient, position errors, and mixed |

Each study function:
1. Generates base FDTD traces (once, or per-case for geometry/domain studies)
2. Sweeps parameters, running detection on modified traces
3. Collects metrics (detection rate, localisation error, miss distance)
4. Prints a summary table
5. Generates a comparison bar chart

---

### 5.20 `cli.py`

**Purpose:** Helmholtz solver CLI entry point, registered as `acoustic-sim` console script.

Provides `parse_args()` and `main()`. Supports:
- Model source: file (JSON), npz, or preset
- Domain geometry: bounds, dx, velocity
- Source: position
- Receivers: line or circle
- Helmholtz parameters: frequency, min PPW
- Valley-specific parameters
- Output: velocity plot, wavefield plot, model save

---

### 5.21 `__init__.py`

**Purpose:** Public API surface. Re-exports all symbols from all modules so users can write:

```python
from acoustic_sim import FDTDSolver, DetectionConfig, run_detection_pipeline
```

The `__all__` list contains 120+ symbols, organised by module.

---

### 5.22 `__main__.py`

**Purpose:** Enables `python -m acoustic_sim` by invoking `cli.main()`.

---

## 6. Module Reference — 3D Extension

### 6.1 `model_3d.py`

**Purpose:** 3D velocity model dataclass and factory functions. Mirrors `model.py` with a z-dimension.

#### `VelocityModel3D`

Fields:
- `x, y, z: np.ndarray` — 1D arrays of cell-centre coordinates
- `values: np.ndarray` — 3D array `[nz, ny, nx]` of wave speeds in m/s
- `dx, dy, dz: float` — Grid spacings

Properties: `nx`, `ny`, `nz`, `shape`, `extent_xy` (for x-y slice imshow), `extent_xz` (for x-z slice imshow), `c_min`, `c_max`

Methods:
- `velocity_at(px, py, pz) -> float` — Nearest-neighbour velocity look-up at arbitrary (x, y, z) point

#### Factory Functions

| Function | Description |
|---|---|
| `create_uniform_model_3d(x_min, x_max, y_min, y_max, z_min, z_max, dx, velocity)` | Constant velocity everywhere |
| `create_layered_z_model_3d(..., layers, background)` | Horizontal layers defined by z-boundaries |
| `model_3d_from_array(values, x_min, x_max, y_min, y_max, z_min, z_max)` | Wrap an existing 3D numpy array |

---

### 6.2 `domains_3d.py`

**Purpose:** 3D domain builders returning `(VelocityModel3D, DomainMeta3D)` pairs.

#### `DomainMeta3D`

Fields:
- `wind_vx, wind_vy, wind_vz: float` — Wind velocity components (adds `wind_vz` vs 2D)
- `attenuation: np.ndarray | None` — `(nz, ny, nx)` damping coefficients
- `description: str`

#### Domain Builders

| Function | Description | Velocity Structure |
|---|---|---|
| `create_isotropic_domain_3d(...)` | Uniform, no wind | Constant `c` |
| `create_wind_domain_3d(..., wind_speed, wind_direction_deg, wind_vz)` | Uniform + 3-component wind | Constant `c`, wind metadata |
| `create_ground_layer_domain_3d(..., air_velocity, ground_velocity, ground_z)` | Air above, high-velocity material below | Air + ground at z-boundary |

---

### 6.3 `sources_3d.py`

**Purpose:** 3D source trajectory classes. All expose `position_at(step, dt) -> (x, y, z)`.

#### Trajectory Classes

| Class | Trajectory | Key Additions vs 2D |
|---|---|---|
| `StaticSource3D` | Fixed position (x, y, z) | z-coordinate |
| `MovingSource3D` | Linear A→B in 3D | z interpolates linearly; arc_height on y |
| `CircularOrbitSource3D` | Circular orbit at constant altitude | `altitude` parameter (default 50 m) |
| `FigureEightSource3D` | Lissajous with optional z-oscillation | `z_amplitude`, `z_freq` parameters |
| `LoiterApproachSource3D` | Orbit then descend on approach | `orbit_altitude`, `approach_target_z`, `descent_rate` |
| `EvasiveSource3D` | Random-walk heading + altitude variation | `mean_altitude`, `z_variance` parameters |
| `CustomTrajectorySource3D` | User-supplied (t, x, y, z) waypoints | 3-column position array with linear interpolation |

#### Utility Functions

- `source_velocity_at_3d(source, step, dt, eps_steps=1) -> (vx, vy, vz)` — Central finite-difference velocity in 3D

---

### 6.4 `receivers_3d.py`

**Purpose:** 3D receiver array wrappers. All functions return `(N, 3)` arrays.

#### Helper

- `_to_3d(positions_2d, z=0.0) -> ndarray` — Convert `(N, 2)` to `(N, 3)` with configurable z (scalar or per-element array)

#### Wrapper Functions

| Function | Wraps | Extra Parameter |
|---|---|---|
| `create_receiver_circle_3d(cx, cy, radius, count, z)` | `create_receiver_circle` | `z: float | ndarray` |
| `create_receiver_nested_circular_3d(...)` | `create_receiver_nested_circular` | `z` |
| `create_receiver_line_3d(...)` | `create_receiver_line` | `z` |
| `create_receiver_l_shaped_3d(...)` | `create_receiver_l_shaped` | `z` |
| `create_receiver_random_disk_3d(...)` | `create_receiver_random_disk` | `z` |
| `create_receiver_custom_3d(positions)` | Direct | Takes `(x, y, z)` tuples |

---

### 6.5 `forward_3d.py`

**Purpose:** Analytical 3D forward model and FDTD bridge. Generates synthetic microphone traces without or with wave-equation solving.

#### Core Functions

| Function | Description |
|---|---|
| `simulate_3d_traces(source, mic_positions, dt, n_steps, ...)` | Sample-by-sample 1/r propagation with optional ground reflection |
| `simulate_3d_traces_vectorized(source, mic_positions, dt, n_steps, ...)` | Same physics, block-based for speed |
| `simulate_scenario_3d(sources, mic_positions, dt, n_steps, ...)` | Multi-source + wind/sensor noise assembly |
| `simulate_3d_traces_fdtd(source, mic_positions, dt, ...)` | Auto-construct 3D domain and run `FDTD3DSolver` |
| `simulate_scenario_3d_fdtd(sources, mic_positions, ...)` | Multi-source FDTD + noise |

**Design note:** `simulate_3d_traces_fdtd` automatically determines domain bounds from the source trajectory and receiver positions (with configurable margin), creates an isotropic `VelocityModel3D`, and runs the full 3D FDTD.

---

### 6.6 `fdtd_3d.py`

**Purpose:** 3D FDTD acoustic wave-equation solver with MPI z-slab decomposition.

#### `FDTD3DConfig`

Alias for `FDTDConfig` (re-exported from `fdtd.py`). Same fields apply to 3D.

#### `FDTD3DSolver`

Constructor: `FDTD3DSolver(model, config, source, receivers, domain_meta=None)`

Key internal state:
- `p_now`, `p_prev`: `(pad_nz, ny, nx)` pressure fields (local sub-domain)
- `C2`: Squared Courant number array (3D)
- `sigma`: 6-face sponge damping array (3D)
- `traces`: Receiver time-series (rank 0 only)

Key methods:
- `run(snapshot_dir=None, snapshot_z_index=None, verbose=True) -> dict` — Full 3D simulation
- `_step(n)` — Single timestep: halo exchange → 3D Laplacian → update → boundary → inject → sample
- `_halo_exchange()` — MPI z-slab ghost exchange
- `_inject(p, n)` — Trilinear source injection (8 cells)
- `_sample_receivers(n)` — Trilinear sampling + MPI gather
- `_build_damping_global(nz, ny, nx)` — Construct 6-face sponge array
- `_gather_field()` — Collect full 3D field on rank 0

Key differences from 2D `FDTDSolver`:
- Decomposition along z-axis (instead of y-axis in 2D)
- Trilinear interpolation (8 cells instead of bilinear 4 cells)
- CFL factor uses `√3` instead of `√2`
- 3D Laplacian: centre weight `3·c₀` (instead of `2·c₀`)
- Snapshots save a single z-slice as `.npy`

---

### 6.7 `processor_3d.py`

**Purpose:** 3D matched field processor extending the 2D polar-grid processor with a z-dimension.

#### Grid and Steering

| Function | Description |
|---|---|
| `build_3d_grid(azimuth_spacing_deg, range_min, range_max, range_spacing, z_min, z_max, z_spacing)` | Returns `(azimuths_rad, ranges_m, z_values_m)` |
| `compute_travel_times_3d(gx, gy, gz, mic_positions, sound_speed)` | Returns `(n_az, n_range, n_gz, n_mics)` |
| `compute_steering_vectors_3d(freqs, travel_times)` | Returns `(n_freq, n_az, n_range, n_gz, n_mics)` complex |

#### Beamforming

| Function | Description |
|---|---|
| `mvdr_beam_power_3d(csdm, steering, epsilon)` | MVDR over 3D grid, returns `(n_freq, n_az, n_range, n_gz)` |
| `broadband_weighted_sum_3d(beam_powers, freqs)` | Returns `(n_az, n_range, n_gz)` |

#### Peak Finding

| Function | Description |
|---|---|
| `find_peaks_3d(bpm, azimuths, ranges, z_values, ...)` | 3D peak finding with exclusion zones; delegates to 2D for single z-slice |

#### Main Entry Point

`matched_field_process_3d(traces, mic_positions, dt, ...)` — Full 3D MFP pipeline. Returns dict with `detections` (including `z` field), `multi_detections`, `azimuths`, `ranges`, `z_values`, `grid_x`, `grid_y`, `sensor_weights`, `filtered_traces`.

---

### 6.8 `tracker_3d.py`

**Purpose:** 3D Extended Kalman Filter tracker with 6-state model.

#### `EKFTracker3D`

Constructor: `EKFTracker3D(process_noise_std, sigma_bearing, sigma_range, sigma_elevation, initial_range_guess, source_level_estimate)`

Methods:
- `initialise_from_detection(bearing, range_est, z_est, center_x, center_y)` — Set initial state with anisotropic 3D covariance
- `predict(dt)` — 6-state constant-velocity prediction
- `update(bearing, range_est, amplitude, z_est, center_x, center_y)` — 3D EKF measurement update
- `set_process_noise_multiplier(multiplier)` — Adaptive process noise from maneuver classifier
- `get_state() -> ndarray` — [x, y, z, vx, vy, vz]
- `get_position() -> (float, float, float)`
- `get_velocity() -> (float, float, float)`
- `get_covariance() -> ndarray` — 6×6 covariance matrix
- `get_range_uncertainty(cx, cy) -> float` — 1-σ radial uncertainty in 3D

#### `MultiTargetTracker3D`

Constructor: `MultiTargetTracker3D(process_noise_std, sigma_bearing_deg, sigma_range, initial_range_guess, gate_threshold, max_missed, source_level_dB)`

Methods: `set_array_center`, `update`, `get_tracks`, `get_all_tracks` — Same interface as 2D, but with 3D Euclidean gating.

#### High-Level Functions

| Function | Description |
|---|---|
| `run_tracker_3d(detections, ...)` | Run single-target 3D EKF |
| `run_multi_tracker_3d(multi_detections, times, ...)` | Run multi-target 3D tracker |

---

### 6.9 `fire_control_3d.py`

**Purpose:** 3D fire control with azimuth + elevation lead angles and class-based engagement rules.

#### Lead and Engagement

| Function | Description |
|---|---|
| `compute_lead_3d(target_pos, target_vel, weapon_pos, ...)` | 3D iterative lead: returns `aim_bearing`, `aim_elevation`, `lead_angle_az`, `lead_angle_el`, `intercept_pos(3D)`, `tof` |
| `compute_engagement_3d(target_pos, target_vel, target_cov, weapon_pos, ..., class_label, class_confidence, maneuver_class)` | 3D engagement envelope with class-based and maneuver-based rules |

#### High-Level Functions

| Function | Description |
|---|---|
| `run_fire_control_3d(track, ...)` | FC solution at every tracked timestep; includes `aim_elevations`, `lead_angles_el` |
| `compute_miss_distance_3d(fire_control, true_positions, true_times, ...)` | 3D miss distance against ground truth |
| `prioritize_threats_3d(tracks, weapon_pos, ...)` | 3D threat scoring and ranking |

---

### 6.10 `plotting_3d.py`

**Purpose:** 3D visualisation utilities.

| Function | Description |
|---|---|
| `plot_3d_trajectory(true_positions, estimated_positions, mic_positions, weapon_pos, ...)` | 3D matplotlib trajectory plot with start/end markers |
| `plot_altitude_vs_time(times, true_z, estimated_z, ...)` | Altitude profile over time |
| `plot_tracking_3d(track, true_positions, true_times, fire_control, weapon_pos, ...)` | 6-panel display: bearing, range, altitude, lead angle (az+el), engagement, maneuver/class |
| `plot_kinematic_scatter(features_by_class, feature_names, ...)` | 2D scatter of kinematic features coloured by class |

---

### 6.11 `detection_main_3d.py`

**Purpose:** 3D pipeline orchestrator with ML integration.

#### Pipeline Functions

| Function | Description |
|---|---|
| `run_detection_3d(traces, mic_positions, dt, ..., acoustic_model, fusion_model, maneuver_model)` | Stage 2: 3D MFP → classification → tracker → maneuver detection → fire control (sensor-only) |
| `evaluate_results_3d(detection_output, true_positions, true_velocities, true_times, ...)` | Stage 3: 3D localisation error, 3D miss distance |

#### Internal Functions

| Function | Description |
|---|---|
| `_classify_detections(detections, filtered_traces, dt, sample_rate, acoustic_model, fusion_model, confidence_threshold)` | Run acoustic or fusion classifier on detected windows; weighted majority vote |
| `_detect_maneuvers(track, maneuver_model, buffer_size)` | Run maneuver classifier on sliding window of tracker state history |

**Design note:** ML models are passed as optional parameters. When `None`, the pipeline runs in baseline mode identical to the 2D pipeline's detection quality. This ensures backward compatibility and allows incremental adoption of ML features.

---

## 7. Module Reference — ML Classification

### 7.1 `ml/features.py`

**Purpose:** Feature extraction for ML classification.

| Function | Description |
|---|---|
| `compute_mel_spectrogram(signal, sample_rate, n_fft, hop_length, n_mels, f_min, f_max)` | Log-mel spectrogram `(n_mels, n_time_frames)` from 1D signal |
| `compute_kinematic_features(positions, velocities, dt)` | 14-dimensional kinematic feature vector from tracker history |

Internal:
- `_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)` — Triangular mel filterbank matrix `(n_mels, n_fft//2+1)`

---

### 7.2 `ml/acoustic_classifier.py`

**Purpose:** Small CNN for acoustic source classification.

#### `AcousticClassifier(nn.Module)`

Constructor: `AcousticClassifier(n_classes=6)`

Architecture: Conv2d(1→16) → BN → ReLU → Conv2d(16→32) → BN → ReLU → MaxPool2d(2) → Conv2d(32→64) → BN → ReLU → GAP → Linear(64→n_classes)

Methods:
- `forward(x) -> Tensor` — Input `(batch, 1, n_mels, n_time)`, output `(batch, n_classes)` logits
- `get_embedding(x) -> Tensor` — 64-dimensional embedding before the final FC layer

---

### 7.3 `ml/fusion_classifier.py`

**Purpose:** Two-branch acoustic+kinematic fusion classifier and baselines.

#### `KinematicBranch(nn.Module)`

Constructor: `KinematicBranch(n_features=14, embed_dim=32)` — Two-layer MLP (14→32→32)

#### `FusionClassifier(nn.Module)`

Constructor: `FusionClassifier(n_classes=6, n_kinematic_features=14)`

Methods:
- `forward(mel_spec, kinematic) -> Tensor` — Two inputs: `(batch, 1, n_mels, n_time)` + `(batch, 14)`, output `(batch, n_classes)`
- `load_acoustic_weights(acoustic_model)` — Copy conv/BN weights from pre-trained AcousticClassifier

#### `KinematicOnlyClassifier(nn.Module)`

Constructor: `KinematicOnlyClassifier(n_classes=6, n_features=14)` — Baseline MLP (14→32→32→n_classes)

---

### 7.4 `ml/maneuver_classifier.py`

**Purpose:** 1D CNN for maneuver state classification.

#### `ManeuverClassifier(nn.Module)`

Constructor: `ManeuverClassifier(n_classes=6)`

Architecture: Conv1d(6→32, k=5, pad=2) → ReLU → Conv1d(32→64, k=5, pad=2) → ReLU → GAP → Linear(64→n_classes)

Methods:
- `forward(x) -> Tensor` — Input `(batch, 6, N)` (6 state features × N timesteps), output `(batch, n_classes)` logits

---

### 7.5 `ml/training.py`

**Purpose:** Training and evaluation loops for all classifiers.

| Function | Description |
|---|---|
| `prepare_acoustic_data(signals, labels, sample_rate, ...)` | Convert raw signals to mel-spectrogram tensors `(N, 1, n_mels, n_time)` |
| `train_classifier(model, X_train, y_train, X_val, y_val, ...)` | Generic single-input training loop (Adam + CrossEntropy) |
| `train_fusion_classifier(model, X_acoustic_train, X_kinematic_train, y_train, ...)` | Two-input training loop for FusionClassifier |
| `evaluate_classifier(model, X_test, y_test, class_names)` | Confusion matrix, per-class precision/recall/F1, overall accuracy |
| `evaluate_fusion_classifier(model, X_acoustic, X_kinematic, y_test, class_names)` | Two-input evaluation |

---

### 7.6 `ml/data_generation.py`

**Purpose:** Physics-based training data synthesis for source classification and maneuver detection.

#### Constants

- `SOURCE_CLASSES = ["quadcopter", "hexacopter", "fixed_wing", "bird", "ground_vehicle", "unknown"]`
- `MANEUVER_CLASSES = ["steady", "turning", "accelerating", "diving", "evasive", "hovering"]`
- `CLASS_TO_IDX`, `IDX_TO_CLASS`, `MANEUVER_TO_IDX` — Index mappings

#### Signal Generators

| Function | Description |
|---|---|
| `_make_multi_rotor_signal(n_steps, dt, n_rotors, fundamental, ...)` | Multi-rotor harmonics with per-rotor frequency spread and beat modulation |
| `_make_bird_signal(n_steps, dt, wing_beat_freq, ...)` | Wing-beat Gaussian pulses + narrowband vocalisations |
| `_make_ground_vehicle_signal(n_steps, dt, engine_fundamental, ...)` | Engine harmonics + bandpass tire noise + pink rumble |
| `generate_source_signal(class_name, n_steps, dt, rng)` | Dispatch to class-specific generator with randomised parameters |

#### Dataset Generators

| Function | Description |
|---|---|
| `generate_classification_dataset(n_samples_per_class, dt, ...)` | Full classification dataset: signals + labels + metadata |
| `generate_maneuver_dataset(n_samples_per_class, window_size, ...)` | Labeled tracker-state segments for maneuver detection |

---

## 8. Example Scripts

### `examples/run_fdtd.py`

Single FDTD run with full CLI control. Builds domain/receivers/source from arguments, runs the solver, saves traces + metadata + gather plot + domain plot + snapshots.

### `examples/run_all_examples.py`

Orchestrates all 18 combinations of `{static, moving} × {isotropic, wind, hills_vegetation} × {concentric, circular, linear}` via `subprocess.call` with `mpirun`.

### `examples/run_full_pipeline.py`

End-to-end detection and targeting pipeline that loads pre-computed FDTD data, runs MFP detection, EKF tracking, and fire control, then produces comprehensive evaluation plots including a radial engagement view.

### `examples/run_valley.sh`

Shell script to run an FDTD simulation in the valley domain with pre-configured parameters.

### `examples/run_wind_circular.sh`

Shell script to run an FDTD simulation with wind domain and circular orbit source.

---

*Next: [Usage Guide](usage.md) — How to install, run, and use the code, including the 3D pipeline and ML classification.*

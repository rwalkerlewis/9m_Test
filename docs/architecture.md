# Code Architecture & Module Reference

This document describes the package structure, data flow, design principles, and provides a detailed reference for every module in the `acoustic-sim` codebase.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Package Structure](#2-package-structure)
3. [Data Flow](#3-data-flow)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Module Reference](#5-module-reference)
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
6. [Example Scripts](#6-example-scripts)

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
├── __init__.py          # Public API (re-exports all symbols)
├── __main__.py          # python -m acoustic_sim
├── cli.py               # Helmholtz CLI entry point
├── config.py            # DetectionConfig dataclass
├── model.py             # VelocityModel + creation helpers
├── sampling.py          # Spatial sampling & CFL checks
├── solver.py            # Helmholtz (frequency-domain) solver
├── backend.py           # NumPy / CuPy abstraction
├── sources.py           # Source signals & trajectory classes
├── domains.py           # Domain builders (5 types)
├── fdtd.py              # FDTD solver (MPI + CUDA)
├── receivers.py         # Receiver array geometry factories
├── io.py                # JSON / NPZ I/O
├── plotting.py          # All visualisation functions
├── noise.py             # Post-hoc noise generators
├── processor.py         # Matched field processor (MVDR)
├── tracker.py           # EKF tracker & multi-target tracker
├── fire_control.py      # Ballistics & engagement logic
├── validate.py          # Automated sanity checks
├── setup.py             # High-level builders
├── detection_main.py    # Three-stage pipeline orchestrator
└── studies.py           # Parametric study framework
```

### Dependency Graph

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

Key dependencies:
- `model.py` is the foundation — everything depends on `VelocityModel`
- `fdtd.py` depends on `model.py`, `backend.py`, `sources.py`, `domains.py`
- `processor.py` is self-contained (no FDTD dependency)
- `detection_main.py` orchestrates everything
- `studies.py` wraps `detection_main.py` in parameter sweeps

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

## 6. Example Scripts

### `examples/run_fdtd.py`

Single FDTD run with full CLI control. Builds domain/receivers/source from arguments, runs the solver, saves traces + metadata + gather plot + domain plot + snapshots.

### `examples/run_all_examples.py`

Orchestrates all 18 combinations of `{static, moving} × {isotropic, wind, hills_vegetation} × {concentric, circular, linear}` via `subprocess.call` with `mpirun`.

### `examples/run_full_pipeline.py`

End-to-end detection and targeting pipeline that loads pre-computed FDTD data, runs MFP detection, EKF tracking, and fire control, then produces comprehensive evaluation plots including a radial engagement view.

---

*Next: [Usage Guide](usage.md) — How to install, run, and use the code.*

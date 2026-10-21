# API Reference

Reference to the public classes, functions, and common entry points in
`acoustic_sim`, organised by source file under `src/acoustic_sim/` (shared solvers in
`solver/`, the drone pipeline in `air/`, the harbor pipeline in `water/`).  New code should import
from `acoustic_sim.air`, `acoustic_sim.water` and `acoustic_sim.solver` or their submodules; the
former top-level module paths (e.g. `acoustic_sim.processor`, `acoustic_sim.fdtd`) and
`acoustic_sim.harbor.*` remain importable as compatibility aliases.
Signatures list parameters in order with their defaults; long signatures end in `...` (see the
docstring for the remaining keyword parameters).

## Contents

- **Shared core:** [`backend.py`](#backendpy), [`cli.py`](#clipy), [`io.py`](#iopy), [`model.py`](#modelpy), [`receivers.py`](#receiverspy), [`sampling.py`](#samplingpy), [`sources.py`](#sourcespy)
- **Shared solvers (`solver/`):** [`solver/fdtd.py`](#solverfdtdpy), [`solver/snapshots.py`](#solversnapshotspy), [`solver/fdtd_cuda.py`](#solverfdtd_cudapy), [`solver/domain_meta.py`](#solverdomain_metapy), [`solver/helmholtz.py`](#solverhelmholtzpy)
- **Air / drone pipeline (`air/`):** [`air/config.py`](#airconfigpy), [`air/detection_main.py`](#airdetection_mainpy), [`air/domains.py`](#airdomainspy), [`air/fire_control.py`](#airfire_controlpy), [`air/forward.py`](#airforwardpy), [`air/noise.py`](#airnoisepy), [`air/plotting.py`](#airplottingpy), [`air/pipeline_plots.py`](#airpipeline_plotspy), [`air/processor.py`](#airprocessorpy), [`air/setup.py`](#airsetuppy), [`air/studies.py`](#airstudiespy), [`air/tracker.py`](#airtrackerpy), [`air/validate.py`](#airvalidatepy), [`air/detection/bearing.py`](#airdetectionbearingpy), [`air/detection/engine.py`](#airdetectionenginepy), [`air/detection/ranging.py`](#airdetectionrangingpy), [`air/detection/tracking.py`](#airdetectiontrackingpy)
- **Water / harbor (`water/`):** [`water/`](#water--water-based-harbor-sensing)
- **ML (`ml/`, optional `[ml]` extra):** [`ml/acoustic_classifier.py`](#mlacoustic_classifierpy), [`ml/data_generation.py`](#mldata_generationpy), [`ml/features.py`](#mlfeaturespy), [`ml/fusion_classifier.py`](#mlfusion_classifierpy), [`ml/maneuver_classifier.py`](#mlmaneuver_classifierpy), [`ml/training.py`](#mltrainingpy), [`ml/anomaly_detector.py`](#mlanomaly_detectorpy), [`ml/anomaly_integration.py`](#mlanomaly_integrationpy), [`ml/anomaly_training.py`](#mlanomaly_trainingpy), [`ml/fno.py`](#mlfnopy), [`ml/fno_data_gen.py`](#mlfno_data_genpy), [`ml/fno_training.py`](#mlfno_trainingpy), [`ml/fno_inference.py`](#mlfno_inferencepy)

---

## backend.py

*NumPy / CuPy backend abstraction.*

### `get_backend(use_cuda=False)`

Return `(xp, is_cuda)` where `xp` is `numpy` or `cupy`.  With `use_cuda=True` and a working
CuPy + GPU it returns `(cupy, True)`; otherwise it warns and returns `(numpy, False)`.

---

## cli.py

*Command-line interface for acoustic_sim.*

### `parse_args(argv=None)`

Build and return an `argparse.Namespace` for the frequency-domain (Helmholtz) CLI.

### `main(argv=None)`

Entry point of `acoustic-sim` and `python -m acoustic_sim`.  Builds or loads a 2-D velocity
model (`--model-file`, `--model-npz` or `--model-preset`), checks spatial sampling, places
receivers, solves the Helmholtz equation at `--frequency` and writes the velocity-model and
wavefield plots.  FDTD runs and the detection pipeline have their own entry points
(`examples/air/01_simulation/run_fdtd.py`, `examples/air/02_pipeline/run_pipeline.py`, `python -m acoustic_sim.air.detection_main`).

---

## air/config.py

*Detection pipeline configuration.*

### `sound_speed_from_temperature(t_celsius=20.0)`

Speed of sound from temperature:

$$c = 331.3\,\sqrt{1 + T/273.15}$$

Returns speed in m/s.

### class `DetectionConfig`

All configurable parameters of the legacy MFP/EKF pipeline (`air/detection_main.py`).  A plain
dataclass with about 120 fields; the main groups:

| Group | Key fields |
|-------|-----------|
| Domain / FDTD | `domain_type`, `x_min`, `x_max`, `y_min`, `y_max`, `dx`, `temperature_celsius`, `sound_speed`, `wind_speed`, `wind_direction_deg`, `total_time`, `fd_order`, `damping_width` |
| Array | `array_type`, `n_mics`, `array_radius`, `array_center_x`, `array_center_y`, `mic_positions` |
| Source | `source_signal`, `fundamental_freq`, `n_harmonics`, `source_level_dB`, `trajectory_type`, `drone_speed`, `source_start`, `source_end`, plus orbit / figure-eight / loiter / evasive parameters |
| Noise | `wind_noise_enabled`, `wind_noise_level_dB`, `wind_corner_freq`, `sensor_noise_enabled`, `sensor_noise_level_dB`, `stationary_source_*` |
| Beamformer (MFP) | `mfp_azimuth_spacing_deg`, `mfp_range_min`, `mfp_range_max`, `mfp_window_length`, `mfp_detection_threshold`, `mfp_diagonal_loading`, ... |
| Tracker | `tracker_process_noise_std`, `tracker_sigma_bearing_deg`, `tracker_sigma_range`, `tracker_gate_threshold`, `tracker_max_missed` |
| Fire control | `weapon_position`, `muzzle_velocity`, `pellet_decel`, `pattern_spread_rate`, `lead_max_iterations`, `priority_w_*` |
| Multi-target | `max_sources`, `min_source_separation_m`, `n_drones`, `drone_configs` |
| Robustness | `enable_sensor_weights`, `enable_transient_blanking`, `enable_position_calibration`, `inject_faults`, `fault_type`, `fault_fraction`, `inject_transient`, `inject_position_error` |
| Run | `use_cuda`, `output_dir`, `seed` |

The 3-D pipeline functions take their geometry from their arguments rather than from this
dataclass.

---

## air/detection_main.py

*End-to-end passive acoustic drone detection, tracking, and fire
control.*  This is the **legacy MFP/EKF pipeline**; the current
production pipeline is `examples/air/02_pipeline/run_pipeline.py` (SRP-PHAT).

### `simulate_scenario(config=None)`

Run the 2-D FDTD to produce synthetic sensor data.  Builds the domain, source and receiver array
from `config`, runs the solver (plus a second run for an optional stationary interferer) and adds
noise.  Returns a dict with the sensor observables (`traces`, `mic_positions`, `dt`,
`sound_speed`, `weapon_position`), the ground truth (`true_positions`, `true_velocities`,
`true_times`) and `model`, `config`.

### `run_detection(traces, mic_positions, dt, sound_speed=343.0, weapon_position=(500.0, 500.0), ...)`

Detection / tracking / fire-control — takes **only sensor data** (no ground truth).  Runs
broadband MFP → EKF tracker → fire-control solution.  Returns a dict with keys `mfp_result`,
`track`, `multi_tracks`, `fire_control`, `fire_control_multi`.

### `evaluate_results(detection_output, true_positions, true_velocities, true_times, weapon_position=(0.0, 0.0), pattern_spread_rate=0.025)`

Compare detection output to ground truth.  Returns `detection_rate`, `mean_loc_error`,
`n_detected`, `n_windows`, the first-shot miss / hit / pattern / time, `mean_miss` and the full
`miss_result`.

### `run_detection_pipeline(config=None)`

Run the full pipeline: simulate → detect → evaluate → plot.  Returns a dict with `scenario`,
`detection_output`, `metrics`, `config` and flat convenience keys (`detection_rate`,
`mean_loc_error`, `first_shot_*`, `mean_miss`, `track`, `fire_control`, ...).

### `parse_args(argv=None)`

CLI argument parser for the detection pipeline.

### `main(argv=None)`

Entry point for `python -m acoustic_sim.air.detection_main`.

### `run_detection_3d(traces, mic_positions, dt, sound_speed=343.0, weapon_position=(0.0, 0.0, 0.0), ...)`

3-D detection / tracking / fire-control — takes **only sensor data**.  Uses the 3-D MFP
beamformer, 3-D EKF tracker and 3-D ballistic lead, with optional ML source and maneuver
classification.  Returns `mfp_result`, `track`, `multi_tracks`, `fire_control`, `class_label`,
`class_confidence`, `classification_history`, `maneuver_class`, `maneuver_history`.

### `evaluate_results_3d(detection_output, true_positions, true_velocities, true_times, weapon_position=(0.0, 0.0, 0.0), pattern_spread_rate=0.025)`

Compare 3-D detection output to ground truth.

---

## air/domains.py

*Domain builders for FDTD simulations.*  Each builder returns
`(VelocityModel, DomainMeta)` or `(VelocityModel3D, DomainMeta3D)`.

### classes `DomainMeta`, `DomainMeta3D`

Defined in `solver/domain_meta.py` and re-exported here.  Extra per-cell physics that supplement
the velocity model:

| Attribute | Default | Description |
|-----------|---------|-------------|
| `wind_vx`, `wind_vy` (`wind_vz` in 3-D) | `0.0` | Wind components (m/s); scalar or per-cell array |
| `attenuation` | `None` | Per-cell absorption coefficient, or `None` |
| `description` | `""` | Free-text label |

### 2-D domain builders

#### `create_isotropic_domain(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, dx=0.5, velocity=343.0)`

Uniform velocity, no wind, no attenuation.

#### `create_wind_domain(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, dx=0.5, velocity=343.0, wind_speed=15.0, wind_direction_deg=45.0)`

Uniform velocity with a constant wind field.  `wind_direction_deg` is
measured counter-clockwise from east.

#### `create_hills_vegetation_domain(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, dx=0.5, air_velocity=343.0, ...)`

2-D slice with two ridges, a valley, and vegetation zones.  Dirt (high
velocity) forms the hills; vegetation applies frequency-dependent
attenuation in a strip above the surface.

#### `create_echo_canyon_domain(x_min=-100.0, x_max=100.0, y_min=-100.0, y_max=100.0, dx=0.2, air_velocity=343.0, ...)`

Domain with two parallel walls forming a canyon.  High-velocity walls
produce strong reflections.

#### `create_urban_echo_domain(x_min=-100.0, x_max=100.0, y_min=-100.0, y_max=100.0, dx=0.2, air_velocity=343.0, ...)`

Domain with rectangular buildings that produce complex multipath.
Buildings are placed randomly using `seed`.

### 3-D domain builders

#### `create_isotropic_domain_3d(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, z_min=0.0, z_max=100.0, dx=1.0, velocity=343.0)`

Uniform velocity, no wind, no attenuation.

#### `create_wind_domain_3d(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, z_min=0.0, z_max=100.0, ...)`

Uniform velocity with a constant 3-D wind field.

#### `create_ground_layer_domain_3d(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, z_min=-10.0, z_max=100.0, ...)`

3-D domain with air above and ground (high velocity) below `ground_z`.

#### `create_hills_vegetation_domain_3d(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, z_min=-5.0, z_max=50.0, ...)`

3-D valley between two ridges, extruded from the 2-D model.

---

## solver/fdtd.py

*Second-order (pressure-only) 2-D and 3-D FDTD acoustic solver: arbitrary even FD order, sponge or
CPML boundaries, optional MPI domain decomposition, CuPy and a fused CUDA kernel.*  The air
forward models use it directly; `water/fdtd_runner.py` subclasses it for the harbor.

### `fd2_coefficients(order)`

Central finite-difference coefficients for $d^2f/dx^2$ at even accuracy `order` (2, 4, 6, 8, …).
Returns a 1-D array of length `order // 2 + 1`.

### `fd1_coefficients(order)`

Central coefficients for the first derivative (used by the CPML boundary).

### `fd2_cfl_factor(coeffs)`

1-D spectral radius of the second-derivative stencil at Nyquist; used to compute the CFL limit.

### class `FDTDConfig`

User-tunable simulation parameters (dataclass).  Option constants: `PRECISIONS`, `KERNELS`,
`BOUNDARIES`, `TIME_SCHEMES`.

| Field | Default | Description |
|-------|---------|-------------|
| `total_time` | `0.3` | Simulation duration (s) |
| `dt` | `None` | Time step (s); `None` = `cfl_safety` × CFL limit |
| `cfl_safety` | `0.9` | Fraction of the CFL limit |
| `damping_width` | `40` | Sponge layer width (cells) |
| `damping_max` | `0.15` | Peak sponge damping at the edge |
| `air_absorption` | `0.005` | Background damping everywhere |
| `snapshot_interval` | `50` | Save a snapshot every N steps (0 = off) |
| `source_amplitude` | `1.0` | Peak source pressure (Pa) |
| `use_cuda` | `True` | Run on the GPU with CuPy (falls back to the CPU with a warning without one) |
| `fd_order` | `2` | Spatial FD order (2, 4, 6, 8, …) |
| `precision` | `"float64"` | `"float64"` or `"float32"` fields (traces are always float64) |
| `kernel` | `"auto"` | `"reference"` array code, `"fused"` single CUDA kernel per step (`solver/fdtd_cuda.py`), or `"auto"` (fused when available) |
| `boundary` | `"sponge"` | `"sponge"` or `"cpml"` (single process) |
| `cpml_width`, `cpml_order`, `cpml_r0`, `cpml_alpha_max` | `15`, `2`, `1e-6`, `0.0` | CPML layer parameters |
| `time_scheme` | `"leapfrog"` | `"leapfrog"`, or the prototypes `"lw4"` (4th order in time) and `"kspace"` (pseudospectral); prototypes are single process, no CPML |

### class `FDTDSolver`

#### `__init__(model, config, source, receivers, domain_meta=None)`

Build the 2-D solver: decompose the grid across MPI ranks (when mpi4py is present), allocate the
pressure fields, the stencil coefficients and the CFL-limited time step.  After construction `dt`
and `n_steps` hold the time step (s) and number of steps.

#### `run(snapshot_dir=None, verbose=True)`

Run the simulation.  Returns (on rank 0) a dict with `traces` (`(n_receivers, n_steps)`), `dt`,
`n_steps` and `path_slices`; other ranks return empty traces.  With `snapshot_dir`, writes a
snapshot PNG every `snapshot_interval` steps (`solver/snapshots.py`).

### class `FDTD3DSolver`

#### `__init__(model, config, source, receivers, domain_meta=None)`

Same contract as the 2-D solver on a `VelocityModel3D`.

#### `run(snapshot_dir=None, snapshot_z_index=None, snapshot_y_index=None, verbose=True, field_plane_z=None, field_plane_subsample=4)`

Run the 3-D simulation.  Returns a dict with `traces`, `dt`, `n_steps` and, when `field_plane_z`
is set, `field_plane`, `field_plane_x`, `field_plane_y`, `field_plane_z`.

| Parameter | Description |
|-----------|-------------|
| `snapshot_z_index`, `snapshot_y_index` | Slices for the two-panel snapshots (default: middle) |
| `field_plane_z` | Record the x-y pressure plane at this height |
| `field_plane_subsample` | Spatial subsampling of the recorded plane |

---

## solver/snapshots.py

*Wavefield snapshot PNGs written by the FDTD solvers; re-exported by `air/plotting.py`.*

### `save_snapshot(model, field, step, output_dir, receivers=None, source_xy=None, db_range=60.0, title=None)`

Write `{output_dir}/snapshot_{step:06d}.png`: the 2-D field in dB SPL re 20 µPa over `db_range`.

### `save_snapshot_3d(field_3d, step, output_dir, *, extent_xy, extent_xz, z_index=None, y_index=None, receivers=None, source_xyz=None, db_range=60.0, title=None)`

Write `{output_dir}/snapshot3d_{step:06d}.png`: an x-y slice and an x-z slice of the 3-D field.

---

## solver/fdtd_cuda.py

*Fused CUDA time step (`FDTDConfig.kernel = "fused"`, the default `"auto"` on a GPU): Laplacian,
leapfrog update, damping, pressure-release mask, source injection and receiver sampling in one
`cupy.RawKernel` launch per step, with a leading batch dimension for independent runs.  The
float64 kernel reproduces the reference path to rounding.*  Used internally by `FDTDSolver`;
`FusedFDTD` and `compile_step` are the entry points for code that drives the kernel directly.

---

## solver/domain_meta.py

Defines `DomainMeta` and `DomainMeta3D` (see `air/domains.py`).

---

## air/fire_control.py

*Fire-control solution for a 12-gauge shotgun engagement.*

All ballistic functions model constant deceleration:
$r = v_0 t - \tfrac{1}{2}at^2$ with default $v_0 = 400\,\text{m/s}$,
$a = 1.5\,\text{m/s}^2$.

### `time_of_flight(range_m, muzzle_velocity=400.0, decel=1.5)`

Solve $r = v_0 t - \tfrac{1}{2}at^2$ for $t$ (positive root).

### `pellet_velocity_at_range(range_m, muzzle_velocity=400.0, decel=1.5)`

Return pellet speed at `range_m` (m/s).

### `pattern_diameter(range_m, spread_rate=0.025)`

Shotgun pattern diameter at `range_m` (m).  Uses linear spread:
$d = \text{spread\_rate} \times r$.

### `compute_lead(target_pos, target_vel, weapon_pos, muzzle_velocity=400.0, decel=1.5, max_iter=5)`

Iterative lead-angle solution: predicts where the target will be when the pellets arrive,
iterating up to `max_iter` times.  Returns a dict with `aim_bearing`, `lead_angle`,
`intercept_pos`, `tof`, `converged`.

### `compute_engagement(target_pos, target_vel, target_cov, weapon_pos, muzzle_velocity=400.0, decel=1.5, ...)`

Decide whether to fire: computes the lead, then checks pellet reach, the optional maximum
engagement range, position uncertainty against the limit and the pattern size.  Returns a dict
with the lead solution plus `can_fire`, `range`, `pattern_diam`, `position_uncertainty`,
`crossing_speed` and `reason` (`FIRE`, `OUT_OF_RANGE`, `TOO_FAR`, `UNCERTAINTY_TOO_HIGH`,
`MAX_RANGE_EXCEEDED`).

### `run_fire_control(track, *, weapon_position=(0.0, 0.0), muzzle_velocity=400.0, pellet_decel=1.5, pattern_spread_rate=0.025, ...)`

Compute fire-control solution at every tracked time step.  Takes the
output of `run_tracker()` and appends engagement decisions to each
frame.

### `compute_miss_distance(fire_control, true_positions, true_times, weapon_position=(0.0, 0.0), pattern_spread_rate=0.025)`

Compute how far each shot would miss the actual target.  Compares the
predicted aim point (at the time of pellet arrival) against the true
target position interpolated from `true_positions` / `true_times`.

### `prioritize_threats(tracks, weapon_pos, time_idx=-1, w_range=1.0, w_closing=2.0, w_quality=0.5)`

Score and rank tracked targets by threat priority.  The score is a
weighted combination of range, closing rate, and track-quality factors.

### `run_multi_fire_control(tracks, *, weapon_position=(0.0, 0.0), muzzle_velocity=400.0, pellet_decel=1.5, pattern_spread_rate=0.025, ...)`

Fire-control solution for multiple targets, sorted by threat priority.

### 3-D variants

| Function | Notes |
|----------|-------|
| `compute_lead_3d(...)` | Same algorithm in 3-D |
| `compute_engagement_3d(...)` | Adds `class_label`, `class_confidence`, `confidence_threshold`, `maneuver_class`; can hold fire on non-threat classes |
| `run_fire_control_3d(track, *, weapon_position=(0.0, 0.0, 0.0), muzzle_velocity=400.0, pellet_decel=1.5, pattern_spread_rate=0.025, ...)` | 3-D fire-control at every time step |
| `compute_miss_distance_3d(...)` | 3-D miss distances |
| `prioritize_threats_3d(...)` | 3-D threat scoring |

### Trajectory and CPA analysis

### `projectile_path(weapon_pos, aim_bearing, aim_elevation, muzzle_velocity, decel, tof, n_points=50)`

3-D pellet trajectory from the weapon along the aim direction.  Returns `(x, y, z)` arrays of
`n_points` points with ballistic velocity decay.

### `find_cpa(weapon_pos, aim_dir, muzzle_velocity, pellet_decel, target_position_fn, t_fire, n_samples=200)`

Closest point of approach between the projectile and the target trajectory
(`target_position_fn(t)`) over the flight.  Returns a dict with `miss`, `cpa_range`,
`cpa_time`, `cpa_pos`, `gt_at_cpa`, `in_range` and related fields.

### `compute_bearing_rate(bearing_history, t_now, bearing_rad, window_s=0.15)`

Append `(t_now, bearing_rad)` to the history and return `(rate_deg_per_s, bearing_history)`
over the last `window_s` seconds (`inf` until there is enough history).  Used for bearing-rate
fire gating.

---

## air/forward.py

*Analytical 3-D forward model for synthetic microphone traces, plus FDTD-backed variants.*  The
analytical model uses point-source spherical spreading ($1/r$ and travel-time delay), optional
air absorption, optional ground reflection (image source) and vegetation attenuation zones; noise
comes from `air/noise.py`.

### `simulate_3d_traces(source, mic_positions, dt, n_steps, sound_speed=343.0, air_absorption=0.005, ...)`

Generate synthetic microphone traces from a 3-D source.  Time-domain
delay-and-attenuate per sample.

### `simulate_3d_traces_vectorized(...)`

Vectorized version using frequency-domain delay application.  Much
faster for large `n_steps`.

### `simulate_scenario_3d(sources, mic_positions, dt, n_steps, sound_speed=343.0, air_absorption=0.005, ...)`

Run full 3-D scenario: forward model + wind/sensor noise.  Accepts
multiple sources and all noise parameters.

### `simulate_3d_traces_fdtd(source, mic_positions, dt, n_steps=None, total_time=1.0, sound_speed=343.0, ...)`

Generate microphone traces using the 3-D FDTD solver.  Automatically
builds a domain around the source/receiver geometry.

### `simulate_scenario_3d_fdtd(sources, mic_positions, total_time=1.0, sound_speed=343.0, dx=1.0, domain_margin=20.0, ...)`

Run full 3-D scenario using the FDTD forward model + noise.

---

## io.py

*I/O helpers — JSON config loading, velocity-model persistence.*

### `load_json(path)`

Load and return a JSON file, or `None` if `path` is falsy.

### `save_model(model, path)`

Persist a velocity model to a `.npz` file (velocity grid + extent).

### `load_model(path)`

Load a velocity model from a `.npz` file.

### `model_from_json(cfg)`

Build a velocity model from a JSON configuration dict.  `"type"` is `"uniform"` (default),
`"layered"`, `"gradient"`, `"checkerboard"` or `"valley"`; optional `"anomalies"` (circle or
rectangle) are applied afterwards.

---

## model.py

*Velocity model dataclass and creation helpers (2-D and 3-D).*

### class `VelocityModel`

2-D velocity model on a regular grid.

| Attribute | Type | Description |
|-----------|------|-------------|
| `x`, `y` | `ndarray` | Grid coordinates (m) |
| `values` | `ndarray (ny, nx)` | Velocity grid in m/s |
| `dx`, `dy` | `float` | Grid spacing (m) |

#### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `nx` | `int` | Number of x grid points |
| `ny` | `int` | Number of y grid points |
| `shape` | `(ny, nx)` | Grid dimensions |
| `extent` | `(x_min, x_max, y_min, y_max)` | For matplotlib `imshow` |
| `c_min` | `float` | Minimum velocity in grid |
| `c_max` | `float` | Maximum velocity in grid |

#### `velocity_at(px, py)`

Nearest-neighbour velocity look-up at an arbitrary `(x, y)` point.

### 2-D model builders

#### `create_uniform_model(x_min, x_max, y_min, y_max, dx, velocity=343.0)`

Constant-velocity model.

#### `create_layered_model(x_min, x_max, y_min, y_max, dx, layers, background=343.0)`

Horizontally layered model.  `layers` is a list of `(y_top, velocity)`; each entry sets the
velocity below `y_top` (entries sorted by `y_top`), `background` applies above the last.

#### `create_gradient_model(x_min, x_max, y_min, y_max, dx, v_bottom=360.0, v_top=330.0)`

Linear velocity gradient from `v_bottom` (y_min) to `v_top` (y_max).

#### `create_checkerboard_model(x_min, x_max, y_min, y_max, dx, cell_size=4.0, v_base=343.0, perturbation=20.0)`

Checkerboard velocity perturbation (useful for resolution tests).

#### `model_from_array(values, x_min, x_max, y_min, y_max)`

Wrap an existing 2-D numpy array as a `VelocityModel`.

#### `add_circle_anomaly(model, cx, cy, radius, velocity)`

Return a copy of `model` with a circular region overwritten.

#### `add_rectangle_anomaly(model, x0, x1, y0, y1, velocity)`

Return a copy of `model` with a rectangular region overwritten.

#### `create_valley_model(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, dx=0.5, air_velocity=343.0, ...)`

Valley between two ridges with a saddle notch (hills of `dirt_velocity` in air); `seed`
controls the surface roughness.

### class `VelocityModel3D`

3-D velocity model on a regular grid.

| Attribute | Type | Description |
|-----------|------|-------------|
| `x`, `y`, `z` | `ndarray` | Grid coordinates (m) |
| `values` | `ndarray (nz, ny, nx)` | Velocity grid in m/s |
| `dx`, `dy`, `dz` | `float` | Grid spacing (m) |

#### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `nx`, `ny`, `nz` | `int` | Grid counts |
| `shape` | `(nz, ny, nx)` | Grid dimensions |
| `extent_xy` | `(4-tuple)` | x-y slice extent for `imshow` |
| `extent_xz` | `(4-tuple)` | x-z slice extent for `imshow` |
| `c_min`, `c_max` | `float` | Velocity range |

#### `velocity_at(px, py, pz)`

Nearest-neighbour velocity look-up at an arbitrary `(x, y, z)` point.

### 3-D model builders

#### `create_uniform_model_3d(x_min, x_max, y_min, y_max, z_min, z_max, dx, velocity=343.0)`

Constant-velocity 3-D model.

#### `create_layered_z_model_3d(x_min, x_max, y_min, y_max, z_min, z_max, ...)`

3-D model with horizontal layers: `layers` is a list of `(z_boundary, velocity)`, each setting the
velocity below `z_boundary`.

#### `model_3d_from_array(values, x_min, x_max, y_min, y_max, z_min, z_max)`

Wrap an existing 3-D numpy array as a `VelocityModel3D`.

---

## water/ — Water-Based Harbor Sensing

The water package (`acoustic_sim.water`; legacy alias `acoustic_sim.harbor`) is a separate
2-D harbor application. `acoustic_sim.water` re-exports the scenario, environment, sensor,
simulation and dataset API: `HarborScenarioConfig`, `HarborEnvironment`, `HarborFDTDConfig`,
`HarborFDTDSolver`, `HarborDataset`, `Cable`, `SensorLayout`, `VesselConfig`, `Track`,
`AISConfig`, `AISReports`, `AISTrack`, `NoiseConfig`, `NoiseRealisation`, `Gather` and the helper
functions below (`build_*`, `simulate_*`, `render_vessel_gathers`, `assemble_dataset`, `cfl_dt`,
`generate_ais`, `generate_noise`, ...).  Processing code is imported from its submodule.

### Scenario and environment

* `HarborScenarioConfig` — JSON-serializable top-level scenario. Use `.load(path)`,
  `.from_dict(data)`, `.to_dict()`, and `.save(path)`.
* `build_environment(cfg)` — create the synthetic harbor from environment settings.
* `build_layout(cfg, env=None, margin=10.0)` — create the cables and hydrophones; if an
  environment is supplied, validate that all FDTD sample points are in water and outside the
  shore margin.
* `default_vessels()` — default AIS, UUV, diver, and body configurations.
* `build_synthetic_harbor(dx=5.0, c0=1500.0, shallow_c_drop=10.0, crop=None)` —
  build the default harbor or a local-frame rectangular crop.
* `uniform_water_environment(width=1000.0, height=750.0, dx=5.0, c0=1500.0)` —
  open-water validation environment.

### Sensors, simulation, and datasets

* `Cable(name, waypoints, channel_spacing=10.0, das_mode="pressure", gauge_length=10.0)` —
  cable channels along a polyline. Pressure mode samples one point per channel; strain mode
  samples three points and collapses them with an unnormalised second difference.
* `SensorLayout(cables=[], hydrophones=...)` — combined channel ordering, sample points,
  `.split(traces)`, and `.collapse(sampled)`; `validate_layout(layout, env, margin=...)` and
  `layout_fingerprint(layout)` (the FDTD cache key) are in `acoustic_sim.water.sensor_sampling`
  (forward side).
* `HarborFDTDConfig` — sample rate, duration, FD stencil, outer sponge, background and shoreline
  damping, and CUDA selection.
* `HarborFDTDSolver(env, hcfg, source, receivers, scatterers=())` — harbor FDTD solver with
  land mask and optional moving scatterers.
* `simulate_source(env, layout, cfg, signal, *, position=None, trajectory=None, cache_dir=None,
  verbose=False, scatterers=())` — one source gather, optionally read/written through the
  simulation cache.
* `simulate_sources(env, layout, cfg, jobs, *, cache_dir=None, verbose=False, batch=None)` —
  several sources through the same harbor, batched on the GPU (`HarborFDTDBatch`) when the fused
  kernel is available.  `cfl_dt(env, ...)` gives the solver's stable time step.
* `HarborDataset` — scenario configuration, layout, sample rate, noisy traces, truth tracks,
  AIS, and optional clean/gain/dead-channel/component arrays. `.observation` and `.truth` give
  the two sides of the sensing contract; `.save("<name>.npz")` writes `<name>.observation.npz` and
  `<name>.truth.npz`, and `.load(path)` reads that pair or an old single `<name>.npz`
  (`python -m acoustic_sim.contract split <name>.npz` converts one).
* `render_vessel_gathers(cfg, env, layout, tracks, cache_dir=None, workers=1,
  include_silent=False, combined_bodies=False)` — generate direct and per-body scattered
  unit-level gathers.
* `assemble_dataset(cfg, layout, tracks, gathers, seed=None, levels=None, noise=None,
  keep_clean=True, scatter_gain=None, keep_components=False, env=None, out=None)` — mix cached fields with
  noise/AIS; `levels` and `scatter_gain` enable parameter sweeps without new FDTD solves.
* `simulate_scenario(cfg, env=None, layout=None, cache_dir=None, workers=1, seed=None,
  levels=None, keep_components=False, out=None)` — end-to-end convenience wrapper (`out` writes the two files).

### Passive detection and bistatic processing

* `ProcessingConfig` — passive STFT band/window, channel QC, grid, interference rejection,
  extraction, GPU and the replica model (`propagator`: `None` is the free field).
* `build_grid(env, cfg, edge=150.0)` / `run_detection(traces, layout, fs, env, cfg, floor=0.0,
  grid=None, provenance=None)` from `acoustic_sim.water.detection` — construct the passive scan grid and run
  matched-field detection.
* `threshold_from_maxima(maxima, pfa, tail_quantile=0.7)` — estimate a CFAR score threshold
  from noise-only per-frame map maxima.
* `SuppressionConfig` and `run_ais_aware(traces, layout, fs, env, ais, pc, sc=None, floor=1e18, grid=None,
  provenance=None)` from
  `acoustic_sim.water.suppression` — refine reported ship positions, project ship energy out of
  the data, re-whiten, and rerun the passive scan.
* `BistaticConfig` — bistatic STFT and band (`f_lo`, `f_hi`: a processing choice, default 100 to
  760 Hz), reference, direct-path removal, processor, map scale (`map_scale="none"` returns the map
  before the division by its maximum), grid, peak and replica model (`propagator`) settings.
* `build_grid(env, bc)` / `run_bistatic(traces, layout, fs, env, ais, bc, grid=None,
  true_waveform=None, diagnostics=None, floor=0.0, provenance=None)` from
  `acoustic_sim.water.bistatic` — form bistatic scan maps and extracted peaks; `Diagnostics(exact_track,
  components)` holds the truth inputs, and `run.provenance` lists every one used.  Its stages are public too: `stft`, `ship_positions`,
  `eca_stage` (time-domain direct-path removal), `bistatic_map`, `extract_peaks`, and
  `ReplicaModel`.
* `motion_integrate(Z, grid, times, speeds, coherent=True, impl="vectorized", use_cuda=False)`
  from `acoustic_sim.water.motion` (also importable from `water.bistatic`) — integrate per-frame
  maps along constant-velocity hypotheses (synthetic aperture); returns the best map and the
  per-velocity peaks.
* `StreamingPassive(layout, fs, env, pc, sc=None, stc=StreamConfig())` from `acoustic_sim.water.streaming` —
  the passive chain in real time: `push(samples, reports)` takes the next chunk of traces `(N, n)` and the
  AIS reports received so far and returns a `FrameResult` (time, peaks, scores, ship positions, compute
  time, latency) for every completed frame; only past data are used.
* `RadiatorTracker(TrackerConfig())` — causal tracks from the passive peaks: `update(frame)`, `confirmed()`,
  `strong()` (the tracks the echo chain uses); `ais_crosscheck(tracks, reports)` labels tracks with AIS ships.
* `StreamingEcho(layout, fs, env, bc, stc=EchoStreamConfig())` — the echo chain in real time:
  `push(samples, radiators={id: [(t, xy), ...]}, illuminator=None)` removes the direct path of every live
  radiator and maps the echoes with each (or only *illuminator*) as the illuminator (`bc`: data reference,
  `removal="eca"`, `eca_doppler=True`, coherent map, free field propagator; `eca_gram="prefix"` and
  `eca_chunk` for speed).
* `green2d_bins(r, ks, xp)` from `acoustic_sim.water.propagation` — `green2d(k r)` for evenly spaced
  wavenumbers by phase recurrence (GPU), used by the batch coherent map and the streaming chains.
* `OccupancyConfig` / `OccupancyGrid(grid, cfg=None)` — track-before-detect over map frames;
  `.run(B, times, snapshot_frames=())` returns `OccupancyHistory`.
* `FreeFieldPropagator(c)` from `acoustic_sim.water.propagation` — the free field replica model (nothing beyond the chart);
  `SimulatorAwarePropagator.from_truth(truth)` from `acoustic_sim.water.diagnostics` — FDTD-specific
  numerical-dispersion correction (simulator knowledge). Use it for simulated data only, not as a
  real-water model.
* `score_detections(dets, truth, layout, frame_times, threshold, targets=None, gate=100.0,
  present=None)` and `truth_labels(dets, truth, gate=100.0)` from `acoustic_sim.water.evaluate` —
  associate detections to the tracks of the `Truth` and report probability of detection,
  localization error, false alarms, and other-vessel detections; the labels a classifier is scored
  against.
* `ChainMaps`, `set_threshold(free_runs, truth, pfa=0.05)`, `score(maps, truth, threshold, primary=None,
  pilings=None)` and `plot_map` from `acoustic_sim.water.scoreboard` — the scoreboard every stage of the
  harbor picture is measured on (definitions: [Architecture, Scoreboard](architecture.md#scoreboard)).

### Other water modules

| Module | Main symbols |
|--------|--------------|
| `water/signatures.py` | `make_reference_signal(kind, speed, active, fs, seed, f_max=...)`, `ship_reference`, `uuv_reference`, `diver_reference` — unit-level source signatures and level calibration |
| `water/vessels.py` | `VesselConfig`, `build_track(vcfg, total_time, env=None, ...)` — vessel settings and ground-truth tracks (`Track` lives in `acoustic_sim.contract`) |
| `water/ais.py`, `water/ais_sim.py` | `AISReports`, `AISTrack` — AIS reports and prediction (processing side); `AISConfig`, `generate_ais(tracks, cfg, rng)` — noisy synthetic reports (forward side) |
| `water/noise.py` | `NoiseConfig`, `NoiseRealisation`, `generate_noise(layout, fs, n_steps, cfg, rng)` — ambient, shipping and instrument noise, channel gains and faults |
| `water/scatterer.py`, `water/fdtd_cuda_bodies.py` | `Scatterer`, `make_scatterer(vcfg, track)`; `DeviceBodyMasks` — moving bodies in the harbor FDTD (host or on-device masks) |
| `water/cylinder.py` | `line_source_scattered`, `partial_wave_coefficients` — analytic cylinder scattering used for validation |
| `water/dispersion.py` | `DispersionModel`, `replica_wavenumbers`, `for_scenario` — the FDTD scheme's numerical wavenumber (forward side) |
| `water/diagnostics.py` | `SimulatorAwarePropagator`, `VesselPathLocaliser`, `true_illuminator_waveform` — forward model knowledge on the inference path, flagged in the provenance |
| `water/localiser.py` | `EmpiricalLocaliser` — analytic MFP first pass refined with survey replicas |
| `water/classifier.py` | `HarborClassifier.load(cnn_path, cvae_path, device="cpu")` — CNN label + CVAE anomaly score (needs `[ml]`) |
| `water/device.py` | `get_device(use_cuda=False)`, `Device` — CPU / CuPy selection and dtype policy |
| `water/plotting.py` | `plot_harbor(env, layout=None, ax=None, show_depth=True, sponge_cells=None)` — harbor map |

The `run_bistatic` truth options and the FDTD demonstration's truth-based direct-path removal
are experimental controls, not inputs a deployed system has, and the run's `provenance` says so.
`acoustic_sim.contract` (`Observation`, `Truth`, `Provenance`, `Observation.from_arrays`) is
described in [Architecture](architecture.md#sensing-contract). See [Harbor](harbor.md) for definitions,
workflow examples, and model limitations.

---

## air/noise.py

*Post-hoc noise generators for microphone traces.*

### `generate_wind_noise(mic_positions, n_samples, dt, level_dB=60.0, corner_freq=15.0, correlation_length=3.0, seed=42)`

Spatially correlated, spectrally shaped wind noise.  Low-pass filtered
at `corner_freq` Hz with exponential spatial correlation decaying over
`correlation_length` metres.

### `generate_sensor_noise(n_mics, n_samples, dt, level_dB=30.0, seed=43)`

Uncorrelated white Gaussian sensor self-noise.  Each microphone channel
gets independent noise at `level_dB` dB SPL.

### `add_all_noise(traces, stationary_traces, mic_positions, dt, *, wind_enabled=True, ...)`

Sum drone traces, optional stationary traces, and post-hoc noise.
Convenience wrapper that calls `generate_wind_noise` +
`generate_sensor_noise` and adds them to the input traces.

### `inject_sensor_faults(traces, fault_type='elevated_noise', fault_sensors=None, fault_fraction=0.2, fault_level_dB=100.0, spike_rate=0.01, seed=44)`

Inject faults into `fault_sensors` (or a random `fault_fraction` of channels).  Returns
`(traces, fault_sensors)`.

| `fault_type` | Effect |
|-------------|--------|
| `elevated_noise` | Add white noise at `fault_level_dB` |
| `dropout` | Zero the channel |
| `spikes` | Random ± spikes at `spike_rate` |
| `dc_offset` | Constant offset at `fault_level_dB` |

### `inject_transient(traces, dt, event_time, event_pos, mic_positions, level_dB=130.0, ...)`

Inject a broadband transient (explosion / gunshot) into traces at
`event_time` seconds from position `event_pos`.  Properly delays the
impulse to each microphone by distance.

### `perturb_mic_positions(true_positions, error_std=2.0, seed=77)`

Add Gaussian position errors to microphone coordinates.  Returns a
perturbed copy; the original is not modified.

---

## air/plotting.py

*Plotting utilities for velocity models, wavefields, detection and tracking.*  All functions
write PNG files to `output_path` (Agg backend); none display interactively.  The module also
re-exports `save_snapshot` / `save_snapshot_3d` (`solver/snapshots.py`) and the engagement plots
of `air/pipeline_plots.py`.

### 2-D plots

| Function | Description |
|----------|-------------|
| `plot_velocity_model(model, output_path='velocity_model.png', receivers=None, source_xy=None, title='Velocity Model')` | Velocity field with optional mic/source overlays |
| `plot_wavefield(model, field, output_path='wavefield.png', receivers=None, source_xy=None, title='Helmholtz wavefield')` | Helmholtz pressure magnitude (dB) |
| `plot_gather(traces, dt, output_path='gather.png', title='Receiver Gather', db_range=60.0, cmap='inferno')` | Receiver traces as dB SPL gather (time vs channel) |
| `plot_domain(model, output_path='domain.png', receivers=None, source_xy=None, source_path=None, attenuation=None, ...)` | Velocity model with source trajectory, attenuation, and wind overlays |

### Detection pipeline plots

| Function | Description |
|----------|-------------|
| `plot_detection_domain(model, receivers, source_positions, weapon_pos=None, stationary_pos=None, output_path='detection_domain.png', title='Detection Domain')` | Domain overview with mics, drone trajectory, and weapon |
| `plot_detection_gather(traces, filtered_traces, dt, output_path='detection_gather.png', title='Receiver Gather', db_range=60.0)` | Two-panel gather: raw (left) and filtered (right) |
| `plot_beam_power(results, true_positions, grid_x, grid_y, output_path='beam_power.png', n_panels=9)` | Multi-panel beam-power / coherence snapshots |
| `plot_tracking(track, true_positions, true_times, fire_control, weapon_pos, output_path='tracking.png')` | Four-panel tracking and fire-control display (range, bearing, position, engagement) |
| `plot_vespagram(traces, mic_positions, dt, output_path='vespagram.png', sound_speed=343.0, slowness_range=None, n_slowness=101, title='Vespagram')` | Beam power as a function of slowness and time |
| `plot_study_comparison(labels, metrics, output_path='study_comparison.png', title='Study Comparison')` | Bar-chart comparison of metrics across study cases |
| `plot_multi_track(tracks, true_positions_list=None, output_path='multi_track.png', title='Multi-Target Tracking')` | Spatial plot of multiple tracks + true trajectories |
| `plot_polar_beam_power(results, azimuths, ranges, true_positions, array_center=(500.0, 500.0), output_path='polar_beam_power.png', n_panels=6)` | Multi-panel polar beam-power maps |

### 3-D plots

| Function | Description |
|----------|-------------|
| `plot_3d_trajectory(true_positions, estimated_positions=None, mic_positions=None, weapon_pos=None, title='3D Trajectory', output_path='trajectory_3d.png')` | 3-D source trajectory with optional estimated track |
| `plot_altitude_vs_time(times, true_z, estimated_z=None, estimated_times=None, title='Altitude vs Time', output_path='altitude_time.png')` | Altitude over time for true and estimated |
| `plot_tracking_3d(track, true_positions, true_times, fire_control, weapon_pos, output_path='tracking_3d.png', ...)` | Six-panel 3-D tracking + fire-control display with altitude and maneuver state |
| `plot_kinematic_scatter(features_by_class, feature_names, title='Kinematic Feature Scatter', output_path='kinematic_scatter.png')` | 2-D scatter of kinematic features coloured by class |

---

## air/pipeline_plots.py

*Figures of the SRP-PHAT engagement pipeline (`examples/air/02_pipeline/run_pipeline.py`); also importable from
`acoustic_sim.air.plotting`.*

| Function | Description |
|----------|-------------|
| `plot_radial_engagement(fire_decisions, ground_truth_fn, source_duration, weapon_pos, is_3d, cfg_fc, output_path=None)` | Polar plot of fire decisions with per-shot effective threshold markers; colours by hit/miss |
| `plot_pipeline_summary(all_detections, all_fire_decisions, all_track_states, wall_times, ground_truth_fn, source_duration, array_center, weapon_pos, ...)` | Six-panel summary: bearing, range, position track, fire decisions, RMS, and metrics |
| `plot_beamformer_diagnostic(traces, mic_positions, dt, ground_truth_fn, source_duration, array_center, beamformer, cfg_det, output_path=None)` | Four-panel beamformer diagnostic: power spectrum, bearing vs time, polar power map, SNR |

---

## air/processor.py

*Frequency-domain matched field processor (MFP) with MVDR beamforming.*

### Grid construction

#### `build_polar_grid(azimuth_spacing_deg=1.0, range_min=20.0, range_max=500.0, range_spacing=5.0)`

Build the polar search grid.  Returns `(azimuths, ranges)`; convert with
`polar_to_cartesian`.

#### `polar_to_cartesian(azimuths, ranges, center_x=0.0, center_y=0.0)`

Convert the polar grid to Cartesian coordinate meshes `(gx, gy)`.

### Travel times & steering

#### `compute_travel_times_polar(gx, gy, mic_positions, sound_speed=343.0)`

Travel times from each grid point to each microphone.  Returns
`(n_grid_points, n_mics)`.

#### `compute_steering_vectors(freqs, travel_times)`

Complex steering vectors: $a_{ij} = e^{-2\pi i f_j \tau_i}$.

#### `select_harmonic_bins(n_fft, dt, fundamental, n_harmonics, bandwidth)`

Return FFT bin indices for each harmonic $\pm$ bandwidth.

### CSDM & beamforming

#### `compute_csdm(traces, dt, window_start, window_length, freq_bins, n_subwindows=4)`

Cross-spectral density matrix averaged over `n_subwindows`
sub-windows.  Returns `(n_freq, n_mics, n_mics)` complex array.

#### `mvdr_beam_power(csdm, steering, epsilon=0.01)`

MVDR (Capon) beam power: $P = 1 / (a^H C^{-1} a)$.  Diagonal loading
controlled by `epsilon`.

#### `conventional_beam_power(csdm, steering)`

Conventional (delay-and-sum) beam power: $P = a^H C a$.

#### `broadband_weighted_sum(beam_powers, freqs)`

Frequency-weighted broadband sum: $w(f) = (f / f_\text{max})^2$.

### Peak detection & sensor health

#### `find_peaks_polar(bpm, azimuths, ranges, threshold=0.25, max_sources=5, min_sep_deg=10.0)`

Find peaks in the polar beam-power map with sub-grid (parabolic) interpolation.  Returns a list
of dicts with `bearing` (rad), `bearing_deg`, `range`, `coherence`, ...

#### `compute_sensor_weights(traces, fault_threshold=10.0)`

Per-sensor reliability weights.  Sensors with anomalous RMS are
down-weighted.

#### `blank_transients(traces, dt, subwindow_ms=5.0, threshold_factor=10.0)`

Zero out impulsive sub-windows (transient suppression).

#### `detect_stationary(history, cv_threshold=0.15)`

Identify stationary grid points from beam-power history (low coefficient
of variation).

#### `calibrate_positions(traces, reported_positions, dt, sound_speed=343.0, max_lag_m=2.0)`

Cross-correlation TDOA self-calibration.  Adjusts reported microphone
positions to match observed inter-channel delays.

### Full MFP pipeline

#### `matched_field_process(traces, mic_positions, dt, *, sound_speed=343.0, azimuth_spacing_deg=1.0, ...)`

Broadband frequency-domain MVDR matched field processing: grid → steering → CSDM → MVDR → peaks
for each time window, with optional sensor weighting, transient blanking, position calibration
and stationary-interferer rejection.  Returns a dict with `detections`, `multi_detections`,
`azimuths`, `ranges`, `grid_x`, `grid_y`, `sensor_weights`, `calibrated_positions`,
`selected_freqs` and `filtered_traces`.

### 3-D variants

Same interfaces as the 2-D functions on an azimuth × range × z grid (`build_3d_grid` returns
`(azimuths, ranges, z_values)`).

| Function | Description |
|----------|-------------|
| `build_3d_grid(...)` | Build polar + z search grid |
| `compute_travel_times_3d(gx, gy, gz, mic_positions, sound_speed=343.0)` | Travel times from 3-D grid points |
| `compute_steering_vectors_3d(freqs, travel_times)` | Complex steering vectors for 3-D grid |
| `mvdr_beam_power_3d(csdm, steering, epsilon=0.01)` | MVDR beam power for 3-D steering |
| `broadband_weighted_sum_3d(beam_powers, freqs)` | Frequency-weighted broadband sum |
| `find_peaks_3d(bpm, azimuths, ranges, z_values, threshold=0.25, max_sources=5, ...)` | 3-D peak finding |
| `matched_field_process_3d(traces, mic_positions, dt, *, sound_speed=343.0, azimuth_spacing_deg=1.0, ...)` | Full 3-D MFP pipeline |

---

## receivers.py

*Receiver (microphone array) geometry helpers.*  Every function returns
an `(N, 2)` or `(N, 3)` numpy array of microphone positions.

### 2-D array geometries

#### `create_receiver_nested_circular(cx=0.0, cy=0.0, inner_radius=0.15, outer_radius=0.5, n_inner=4, n_outer=8)`

Nested circular array: 1 centre + inner ring + outer ring.  Returns
`(1 + n_inner + n_outer, 2)`.

#### `create_receiver_line(x_start, y_start, x_end, y_end, count)`

Line of receivers.  Returns shape `(count, 2)`.

#### `create_receiver_circle(cx, cy, radius, count)`

Single-ring circular array.  Returns shape `(count, 2)`.

#### `create_receiver_concentric(cx, cy, radii, counts_per_ring=12)`

Concentric rings.  `radii` is a list of ring radii; `counts_per_ring`
can be an int (same for all rings) or a list.  Returns `(N_total, 2)`.

#### `create_receiver_l_shaped(n1, n2, spacing, origin_x=0.0, origin_y=0.0)`

L-shaped array.  Returns `(n1 + n2 - 1, 2)`.

#### `create_receiver_log_spiral(count=13, radius=0.5, cx=0.0, cy=0.0)`

Logarithmic (golden-angle) spiral for maximum baseline diversity.

#### `create_receiver_random_disk(count=13, radius=0.5, cx=0.0, cy=0.0, seed=42)`

Random positions within a disk.  Returns `(count, 2)`.

#### `create_receiver_random(count, x_min, x_max, y_min, y_max, seed=42)`

Random positions within a bounding box.  Returns `(count, 2)`.

#### `create_receiver_custom(positions)`

Wrap a list of `(x, y)` tuples as an `(N, 2)` array.

### Array diagnostics

#### `print_array_diagnostics(positions, sound_speed=343.0)`

Print and return a diagnostic summary of the array: `n_elements`, `n_baselines`, `min_baseline`,
`max_baseline`, the spatial-aliasing frequencies of the shortest and longest baselines, and the
diffraction-limited `resolutions` at key frequencies.

### 3-D array geometries

All append a constant `z` coordinate.

| Function | Returns |
|----------|---------|
| `create_receiver_l_shaped_3d(n1, n2, spacing, origin_x=0.0, origin_y=0.0, z=0.0)` | `(n1+n2-1, 3)` |
| `create_receiver_circle_3d(cx, cy, radius, count, z=0.0)` | `(count, 3)` |
| `create_receiver_nested_circular_3d(cx=0.0, cy=0.0, inner_radius=0.15, outer_radius=0.5, n_inner=4, n_outer=8, z=0.0)` | `(1+n_inner+n_outer, 3)` |
| `create_receiver_line_3d(x_start, y_start, x_end, y_end, count, z=0.0)` | `(count, 3)` |
| `create_receiver_random_disk_3d(count=13, radius=0.5, cx=0.0, cy=0.0, z=0.0, seed=42)` | `(count, 3)` |
| `create_receiver_custom_3d(positions)` | `(N, 3)` |

---

## sampling.py

*Spatial-sampling and CFL stability checks.*

### `check_spatial_sampling(model, frequency_hz, min_ppw=10.0)`

Check that the grid resolves the shortest wavelength with at least `min_ppw` points per
wavelength at `frequency_hz`.  Returns a dict with `valid`, `min_wavelength`, `ppw`,
`required_ppw`, `max_dx`, `message`.

### `check_cfl(model, dt)`

CFL stability check for an explicit time-domain scheme.  Returns a dict with `valid`,
`courant`, `limit`, `message`.

### `suggest_dx(c_min, frequency_hz, min_ppw=10.0)`

Return the maximum allowable `dx` for a given frequency and `c_min`:

$$\Delta x \le \frac{c_\min}{f \cdot \text{ppw}}$$

---

## air/setup.py

*Simulation setup helpers.*  High-level convenience builders used by the
CLI and detection pipeline.

### `build_domain(domain='isotropic', *, x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, ...)`

Build a velocity model and domain metadata for `domain` in `isotropic`, `wind`,
`hills_vegetation`, `echo_canyon`, `urban_echo`.  Returns `(model, domain_meta)`.

### `build_receivers(array='circular', *, count=16, radius=15.0, radii=(10.0, 20.0, 30.0, 40.0), x0=-40.0, ...)`

Build a receiver array for `array` in `circular`, `concentric`, `linear`, `l_shaped`,
`random`, `nested_circular`, `log_spiral`, `random_disk`, `custom`.  Returns `(n_recv, 2)`.

### `compute_dt(model, meta, cfl_safety=0.9, fd_order=2)`

Return `(dt, f_max)` from CFL and spatial sampling constraints.

### `build_source(source_type='static', signal_type='ricker', *, n_steps, dt, f_max, ...)`

Build a source object with its signal.  `source_type` selects the trajectory class
(`static`, `moving`, `circular_orbit`, `figure_eight`, `loiter_approach`, `evasive`,
`custom_trajectory`); `signal_type` selects the waveform (`ricker`, `tone`, `noise`,
`propeller`, `file`, `drone_harmonics`, `stationary_tonal`).

---

## solver/helmholtz.py

*2-D Helmholtz (frequency-domain) solver.*

### `solve_helmholtz(model, source_xy, frequency_hz, damping_width=None)`

Solve the 2-D Helmholtz equation on the velocity-model grid.  Builds the sparse system
$(\nabla^2 + k^2) p = s$ with an absorbing damping layer of `damping_width` cells (default about 8 %
of the smaller dimension) and solves it with `scipy.sparse.linalg.spsolve`.

Returns `field` with shape `(ny, nx)` (complex pressure).

---

## sources.py

*Source signal generators and source-position helpers for FDTD.*

### Signal generators

The `make_*` generators return a 1-D numpy array of length `n_steps`.

#### `load_wav_mono(path)`

Load a WAV file and return `(mono_float32, sample_rate)`.

#### `prepare_source_signal(raw, fs_audio, dt_sim, f_max)`

Low-pass filter and resample an audio signal for the FD simulation.
Applies a Butterworth filter at `f_max`, then resamples from `fs_audio`
to `1/dt_sim`.

#### `make_wavelet_ricker(n_steps, dt, f0)`

Ricker (Mexican-hat) wavelet centred at $t_0 = 1.5/f_0$.

#### `make_source_propeller(n_steps, dt, f_max=None, blade_count=3, rpm=3600.0, harmonics=14, ...)`

Synthetic propeller / rotor noise.  Sum of blade-pass harmonics with
amplitude modulation (Doppler-like) plus broadband turbulent self-noise.

#### `make_source_tone(n_steps, dt, frequency_hz=40.0)`

Pure sine-wave source at `frequency_hz`.

#### `make_source_noise(n_steps, dt, f_low=5.0, f_high=60.0, seed=42)`

Band-limited coloured noise source.

#### `make_source_from_file(path, n_steps, dt, f_max, max_seconds=None)`

Load a WAV file and prepare it for FD injection.

#### `make_drone_harmonics(n_steps, dt, fundamental=150.0, n_harmonics=4, harmonic_amplitudes=None, source_level_dB=90.0, f_max=None)`

Synthetic drone rotor signal as a sum of harmonics.  Each harmonic
rolls off as $1/(n+1)$ unless explicit `harmonic_amplitudes` are given.
Normalised to `source_level_dB` SPL re 20 µPa.

#### `make_stationary_tonal(n_steps, dt, base_freq=60.0, n_harmonics=4, source_level_dB=70.0, broadband_level=0.1, f_max=None, seed=99)`

Tonal signal for a stationary coherent noise source (e.g. generator,
HVAC).

### 2-D trajectory classes

Dataclasses exposing `position_at(step, dt) → (x, y)` and a `signal` array.

| Class | Fields | Motion |
|-------|--------|--------|
| `StaticSource` | `x, y, signal` | Fixed position |
| `MovingSource` | `x0, y0, x1, y1, speed, signal, arc_height=0.0` | Straight line at `speed`, optional parabolic arc |
| `CircularOrbitSource` | `cx, cy, radius, angular_velocity, start_angle, signal` | Constant angular velocity orbit |
| `FigureEightSource` | `cx, cy, x_amp, y_amp, x_freq, y_freq, phase_offset, signal` | Lissajous figure-eight |
| `LoiterApproachSource` | `orbit_cx, orbit_cy, orbit_radius, orbit_duration, approach_target_x, approach_target_y, approach_speed, signal` | Orbit at standoff, then straight approach |
| `EvasiveSource` | `x0, y0, heading, mean_speed, speed_var, heading_var, signal, seed=42` | Random-walk heading and speed |
| `CustomTrajectorySource` | `times, positions, signal` | Linear interpolation of user waypoints |

### 3-D trajectory classes

All expose `position_at(step, dt) → (x, y, z)`.  Same motion models as 2-D with an added
z-coordinate (e.g. `altitude=50.0` for the orbit classes, `z0`/`z1` for `MovingSource3D`).

| Class | Description |
|-------|-------------|
| `StaticSource3D` | Fixed position |
| `MovingSource3D` | Linear trajectory from `(x0,y0,z0)` to `(x1,y1,z1)` |
| `CircularOrbitSource3D` | Circular orbit at constant altitude |
| `FigureEightSource3D` | Figure-eight with optional z oscillation |
| `LoiterApproachSource3D` | Loiter then descend on approach |
| `EvasiveSource3D` | Random-walk heading with altitude variation |
| `ErraticQuadcopterSource3D` | Ornstein–Uhlenbeck mean-reverting 3-D trajectory confined to a bounding box; fields: `x0, y0, z0, bbox_min, bbox_max, mean_speed, agility, speed_var, signal, seed` |
| `CustomTrajectorySource3D` | User-supplied `(t, x, y, z)` arrays |

### Utility functions

#### `source_velocity_at(source, step, dt, eps_steps=1)`

Compute instantaneous 2-D velocity via central finite differences.

#### `inject_source(field, sx, sy, amplitude, x_arr, y_arr, ...)`

Inject `amplitude` into FDTD `field` at `(sx, sy)` using bilinear
interpolation weights (sub-cell accuracy).

#### `source_velocity_at_3d(source, step, dt, eps_steps=1)`

Compute instantaneous 3-D velocity via central finite differences.

---

## air/studies.py

*Systematic parameter-sweep studies for the detection pipeline.*  Each
function runs a sweep over one variable and records detection/tracking
metrics.

### `study_array_geometry(base_config=None, output_dir='output/studies/array_geometry')`

Compare circular, linear, L-shaped, random, concentric arrays.

### `study_min_sensors(base_config=None, sensor_counts=(4, 6, 8, 12, 16, 24), output_dir='output/studies/min_sensors')`

Sweep `n_mics` to find minimum acceptable sensor count.

### `study_sensor_faults(base_config=None, fault_fractions=(0.0, 0.1, 0.2, 0.3, 0.5), output_dir='output/studies/sensor_faults')`

Sweep fault fraction with and without robust sensor weighting.

### `study_multi_drone(base_config=None, output_dir='output/studies/multi_drone')`

Test with 1 and 2 simultaneous drones using multi-peak detection.

### `study_transient_robustness(base_config=None, transient_levels=(0, 110, 120, 130), output_dir='output/studies/transient')`

Inject transients, compare with/without blanking.

### `study_haphazard_array(base_config=None, n_trials=3, output_dir='output/studies/haphazard')`

Compare optimised circular array vs random placements.

### `study_echo_domains(base_config=None, output_dir='output/studies/echo')`

Compare detection in isotropic vs echo-prone domains.

### `study_position_errors(base_config=None, error_stds=(0.0, 1.0, 2.0, 5.0), output_dir='output/studies/position_error')`

Sweep position error magnitude with/without self-calibration.

### `study_mixed_failures(base_config=None, output_dir='output/studies/mixed')`

Combined stress test: faults + position errors + transient + echo
domain.

### `run_all_studies(base_config=None, output_dir='output/studies')`

Execute all nine studies and return combined results.

---

## air/tracker.py

*Extended Kalman Filter tracker for bearing-primary measurements.*

### class `EKFTracker`

Extended Kalman Filter with bearing / range / amplitude measurements.

State vector: $\mathbf{x} = [x, y, v_x, v_y]^T$.  Measurement model
uses bearing (from array centre), range (inverse-square-law), and
amplitude.

#### Constructor

```python
EKFTracker(
    process_noise_std=2.0,        # m/s² — drives Q matrix
    sigma_bearing=0.0524,         # rad — bearing measurement noise (~3°)
    sigma_range=100.0,            # m — range measurement noise
    initial_range_guess=200.0,    # m — initial range before first update
    source_level_estimate=0.632,  # linear amplitude at 1 m
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `initialise_from_bearing(bearing, range_est=None, center_x=0.0, center_y=0.0)` | Place initial state at bearing from array centre |
| `predict(dt)` | Constant-velocity prediction |
| `update(bearing, range_est, amplitude, center_x=0.0, center_y=0.0)` | EKF measurement update |
| `get_state()` | Return full 4-element state vector |
| `get_position()` | Return `(x, y)` |
| `get_velocity()` | Return `(vx, vy)` |
| `get_covariance()` | Return 4×4 covariance matrix |
| `get_range_uncertainty(cx=0.0, cy=0.0)` | 1-σ range uncertainty from covariance |

### `run_tracker(detections, *, process_noise_std=2.0, sigma_bearing_deg=3.0, sigma_range=100.0, initial_range_guess=200.0, ...)`

Run the EKF tracker on MFP detections.  Takes a list of per-window
detection dicts and returns a track dict with position/velocity/covariance
time series.

### class `MultiTargetTracker`

Multi-target EKF tracker with nearest-neighbour data association.

#### Constructor

```python
MultiTargetTracker(
    process_noise_std=2.0,
    sigma_bearing_deg=3.0,
    sigma_range=100.0,
    initial_range_guess=200.0,
    gate_threshold=30.0,   # Mahalanobis distance gate
    max_missed=5,          # frames before track deletion
    source_level_dB=90.0,
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `set_array_center(cx, cy)` | Set array centre for bearing computation |
| `update(detections, t)` | Associate detections, update/create/delete tracks |
| `get_tracks()` | Return active tracks only |
| `get_all_tracks()` | Return all tracks (including deleted) |

### `run_multi_tracker(multi_detections, times, *, process_noise_std=2.0, sigma_bearing_deg=3.0, sigma_range=100.0, ...)`

Run multi-target EKF tracker on multi-peak detections.

### class `EKFTracker3D`

Extended Kalman Filter for 3-D tracking.

State vector: $\mathbf{x} = [x, y, z, v_x, v_y, v_z]^T$.

#### Constructor

```python
EKFTracker3D(
    process_noise_std=2.0,
    sigma_bearing=0.0524,
    sigma_range=100.0,
    sigma_elevation=0.1,       # rad — elevation measurement noise
    initial_range_guess=200.0,
    source_level_estimate=0.632,
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `initialise_from_detection(bearing, range_est=None, z_est=0.0, center_x=0.0, center_y=0.0)` | Place initial state from a detection |
| `predict(dt)` | Constant-velocity prediction in 3-D |
| `update(bearing, range_est, amplitude, z_est=0.0, center_x=0.0, center_y=0.0)` | EKF measurement update |
| `set_process_noise_multiplier(multiplier)` | Scale process noise (from maneuver detector) |
| `get_state()` | Return 6-element state vector |
| `get_position()` | Return `(x, y, z)` |
| `get_velocity()` | Return `(vx, vy, vz)` |
| `get_covariance()` | Return 6×6 covariance matrix |
| `get_range_uncertainty(cx=0.0, cy=0.0)` | 1-σ range uncertainty |

### class `MultiTargetTracker3D`

Multi-target 3-D EKF tracker with nearest-neighbour data association.
Same interface as `MultiTargetTracker` with 3-D state vectors.

#### Constructor

```python
MultiTargetTracker3D(
    process_noise_std=2.0,
    sigma_bearing_deg=3.0,
    sigma_range=100.0,
    initial_range_guess=200.0,
    gate_threshold=30.0,
    max_missed=5,
    source_level_dB=90.0,
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `set_array_center(cx, cy)` | Set array centre |
| `update(detections, t)` | Associate, update, create, delete tracks |
| `get_tracks()` | Active tracks |
| `get_all_tracks()` | All tracks |

### `run_tracker_3d(detections, *, process_noise_std=2.0, sigma_bearing_deg=3.0, sigma_range=100.0, initial_range_guess=200.0, ...)`

Run the 3-D EKF tracker on MFP detections.

### `run_multi_tracker_3d(multi_detections, times, *, process_noise_std=2.0, sigma_bearing_deg=3.0, sigma_range=100.0, ...)`

Run multi-target 3-D EKF tracker on multi-peak detections.

---

## air/validate.py

*Automated sanity checks for the detection pipeline.*  Each check
returns `(passed: bool, message: str)`.

### `check_amplitude(traces, max_pressure=200.0)`

**Check 1:** No instantaneous pressure exceeds `max_pressure` Pa.

### `check_snr(traces, filtered_traces, mic_positions, source_positions)`

**Check 2:** SNR > 0 dB at the closest microphone after bandpass
filtering.

### `check_travel_times(mic_positions, sound_speed, dt)`

**Check 3:** Travel-time table matches distance / $c$ within 1 sample.

### `check_localization(traces, dt, mic_positions, true_pos, sound_speed=343.0, grid_spacing=5.0)`

**Check 4:** Stationary source localises to within acceptable error via
a coarse grid search.

### `check_energy(traces, dt, source_level_dB, mic_positions, source_positions)`

**Check 5:** Total received energy within order of magnitude of
expected (based on `source_level_dB` and 1/r² spreading).

### `run_all_checks(traces, filtered_traces, dt, mic_positions, source_positions, source_level_dB=90.0, sound_speed=343.0)`

Run all five sanity checks, print the results and return `True` if all passed.

---

## ml/acoustic_classifier.py

*Acoustic source classifier using a simple CNN on mel spectrograms.*
Requires `torch`.

### class `AcousticClassifier`

Small CNN for acoustic source classification.

Architecture: three 3×3 Conv2d blocks (1→16→32→64 channels) → global average pool →
Linear(64→n_classes).

| Method | Description |
|--------|-------------|
| `__init__(n_classes=6)` | Build the network |
| `forward(x)` | Forward pass: mel spectrogram → class logits |
| `get_embedding(x)` | Get the 64-dim acoustic embedding (before FC layer) |

Source classes (default 6): quadcopter, hexacopter, fixed_wing, bird,
ground_vehicle, unknown.

---

## ml/data_generation.py

*Training data generation for source classification and maneuver
detection.*

### `generate_source_signal(class_name, n_steps, dt, rng)`

Generate a randomised signal for a source class.  Returns `(signal, params)`.

### `generate_classification_dataset(n_samples_per_class=200, dt=1.0 / 4000, window_duration=0.5, sound_speed=343.0, seed=42)`

Generate a classification training set.  Returns a dict with `signals`, `labels`,
`class_names`, `params`, `snr_dbs`, `dt`.

### `generate_maneuver_dataset(n_samples_per_class=400, window_size=20, dt_tracker=0.1, seed=42)`

Generate labelled track segments for maneuver detection.  Returns a dict with `features`
`(N, window_size, 6)` (`x, y, z, vx, vy, vz`), `labels`, `class_names`.

Maneuver classes: steady, turning, accelerating, diving, evasive,
hovering.

---

## ml/features.py

*Feature extraction for ML classification.*

### `compute_mel_spectrogram(signal, sample_rate, n_fft=512, hop_length=128, n_mels=64, f_min=20.0, f_max=None)`

Log-power mel spectrogram of a 1-D signal, `(n_mels, n_frames)`.  Pure numpy (Hann-windowed
frames and a built-in mel filterbank; no librosa dependency).

### `compute_kinematic_features(positions, velocities, dt=0.1)`

Compute a 14-element kinematic feature vector from tracker positions and velocities `(N, 3)`
(all zeros for fewer than 3 samples).  **Pure numpy — no torch dependency.**

| Index | Feature |
|-------|---------|
| 0–2 | Speed mean, std, min |
| 3–4 | Mean absolute heading rate, heading-rate std |
| 5–6 | Curvature mean, std |
| 7–10 | Altitude mean, std, climb-rate std, fraction with \|z\| < 1 m |
| 11 | Hover fraction (speed < 1 m/s) |
| 12 | Lag-1 autocorrelation of the heading rate |
| 13 | Reserved (0) |

---

## ml/fusion_classifier.py

*Kinematic fusion classifier — two-branch network.*  Requires `torch`.

### class `KinematicBranch`

MLP for kinematic features: Linear(14→32) → ReLU → Linear(32→32) → ReLU (`n_features=14`,
`embed_dim=32`).

### class `FusionClassifier`

Two-branch acoustic + kinematic fusion classifier: the `AcousticClassifier` convolution stack
(64-dim embedding) and a `KinematicBranch` (32-dim), concatenated (96) → Linear(96→64) →
Linear(64→n_classes).

| Method | Description |
|--------|-------------|
| `__init__(n_classes=6, n_kinematic_features=14)` | Build both branches + fusion head |
| `forward(mel_spec, kinematic)` | Forward pass: dual input → class logits |
| `load_acoustic_weights(acoustic_model)` | Initialize acoustic branch from a trained `AcousticClassifier` |

### class `KinematicOnlyClassifier`

Kinematic-only baseline: Linear(14→32) → Linear(32→32) → Linear(32→n_classes) with ReLUs.

---

## ml/maneuver_classifier.py

*Maneuver detection classifier using a 1-D CNN.*  Requires `torch`.

### class `ManeuverClassifier`

1-D CNN for maneuver state classification.

Architecture: Conv1d(6→32, k=5) → Conv1d(32→64, k=5) → global average pool →
Linear(64→n_classes).

Input: `(batch, 6, window_size)` — 6 channels are `(x, y, z, vx, vy, vz)`.

| Method | Description |
|--------|-------------|
| `__init__(n_classes=6)` | Build the network |
| `forward(x)` | Forward pass: kinematic window → maneuver logits |

Maneuver classes: steady, turning, accelerating, diving, evasive,
hovering.

---

## ml/training.py

*Training loops for all ML classifiers.*  Requires `torch`.

### `prepare_acoustic_data(signals, labels, sample_rate, n_fft=512, hop_length=128, n_mels=64)`

Convert raw signals to mel spectrogram tensors.  Returns
`(X_tensor, y_tensor)` ready for DataLoader.

### `train_classifier(model, X_train, y_train, X_val, y_val, n_epochs=50, ...)`

Training loop for single-input classifiers (`AcousticClassifier`, `ManeuverClassifier`):
Adam + cross-entropy.  Returns a history dict with `train_loss`, `val_loss`, `val_acc`.

### `train_fusion_classifier(model, X_acoustic_train, X_kinematic_train, y_train, X_acoustic_val, X_kinematic_val, ...)`

Training loop for the two-input `FusionClassifier`; returns the same history dict.

### `evaluate_classifier(model, X_test, y_test, class_names)`

Evaluate a single-input classifier.  Returns a dict with `confusion_matrix`, `accuracy`,
`per_class` (precision, recall, F1), `predictions`, `probabilities`.

### `evaluate_fusion_classifier(model, X_acoustic, X_kinematic, y_test, class_names)`

Evaluate the fusion classifier.  Same metrics as `evaluate_classifier`.

---

## ml/anomaly_detector.py

*Convolutional variational autoencoder (CVAE) for novel-source detection on mel spectrograms.*
Requires `torch`.

### class `ConvVAE(latent_dim=12)`

| Method | Description |
|--------|-------------|
| `encode(x)` | Mel spectrogram `(B, 1, n_mels, T)` → `(mu, log_var)` |
| `reparameterize(mu, log_var)` | Sample a latent vector |
| `decode(z, output_size=None)` | Latent → reconstructed spectrogram |
| `forward(x)` | Returns `(recon, mu, log_var)` |
| `compute_anomaly_score(x)` | `(score, latent)` for one spectrogram; the score combines reconstruction error and KL divergence |

### `vae_loss(recon_x, x, mu, log_var, kl_weight=1.0)`

MSE reconstruction + weighted KL divergence; returns `(total_loss, recon_loss, kl_loss)`.

---

## ml/anomaly_integration.py

*Pipeline wrapper around a trained `ConvVAE`.*  Requires `torch`.

### dataclass `AnomalyResult`

`is_novel`, `reconstruction_error`, `threshold`, `latent_vector`.

### class `AnomalyDetector(model_path, threshold_path, latent_dim=12, sample_rate=8000.0, n_fft=512, hop_length=128, n_mels=64, device='cpu')`

Loads the weights and the threshold JSON (`threshold_3sigma`).

| Method | Description |
|--------|-------------|
| `process_frame(audio_frame, sample_rate=None)` | Raw audio → mel spectrogram → `AnomalyResult` |
| `process_mel_spectrogram(mel_spec)` | Pre-computed spectrogram → `AnomalyResult` (`is_novel` when the score exceeds the threshold) |

---

## ml/anomaly_training.py

*Training of the CVAE on the known threat classes (`KNOWN_THREAT_CLASSES` = quadcopter,
hexacopter, fixed_wing).*  Requires `torch`.

| Function | Description |
|----------|-------------|
| `generate_anomaly_training_data(n_samples_per_class=300, dt=1.0 / 4000, window_duration=0.5, sound_speed=343.0, n_fft=512, hop_length=128, ...)` | Returns `spectrograms`, `class_names`, `sample_rate` |
| `prepare_anomaly_tensors(spectrograms)` | Padded `(N, 1, n_mels, max_time)` tensor |
| `train_anomaly_detector(model, X_train, X_val, n_epochs=100, lr=0.001, batch_size=32, ...)` | Training loop; returns a history dict |
| `calibrate_threshold(model, X_val)` | Returns `mean`, `std`, `threshold_3sigma` of validation scores |

CLI: `python -m acoustic_sim.ml.anomaly_training [--quick] [--output-dir output/models]` writes
`anomaly_detector.pt` and `anomaly_threshold.json`.

---

## air/detection/bearing.py

*Pluggable bearing (DOA) estimation algorithms.*

### dataclass `BearingDetection`

Single bearing estimate: `bearing_rad`, `power`.  Property:
`bearing_deg`.

### dataclass `BearingResult`

Collection of bearing detections with optional power spectrum and
source-count estimate.  Fields: `detections`, `spectrum`,
`bearings_rad`, `n_sources_estimated`.

### class `BearingEstimator` (ABC)

Abstract base class.  Subclasses must implement
`estimate(segment, max_sources) → BearingResult`.

### class `SRPBeamformer(mic_positions, fs, window_samples, c=343.0, n_bearings=360, freq_lo=100.0, ...)`

Steered Response Power with Phase Transform.

| Method | Description |
|--------|-------------|
| `__init__(mic_positions, fs, window_samples, c=343.0, n_bearings=360, freq_lo=100.0, ...)` | Pre-compute steering matrix |
| `estimate(segment, max_sources=1)` | SRP-PHAT → `BearingResult` |

### class `MUSICEstimator(mic_positions, fs, window_samples, c=343.0, n_bearings=360, freq_lo=100.0, ...)`

MUSIC subspace DOA with MDL source-count estimation.

| Method | Description |
|--------|-------------|
| `__init__(mic_positions, fs, window_samples, c=343.0, n_bearings=360, freq_lo=100.0, ...)` | Init |
| `estimate(segment, max_sources=1)` | MUSIC pseudo-spectrum → `BearingResult` |

### class `MVDRBeamformer(mic_positions, dt, sound_speed=343.0, azimuth_spacing_deg=1.0, range_min=20.0, range_max=500.0, ...)`

MVDR (Capon) beamformer on a polar grid with harmonic selection.

| Method | Description |
|--------|-------------|
| `__init__(mic_positions, dt, sound_speed=343.0, azimuth_spacing_deg=1.0, range_min=20.0, range_max=500.0, ...)` | Init |
| `estimate(segment, max_sources=1)` | MVDR → `BearingResult` |

### Factory functions

- `available_bearing_methods() → list[str]` — returns `["srp_phat", "music", "mvdr"]`
- `create_bearing_estimator(method, **kwargs) → BearingEstimator`

---

## air/detection/engine.py

*Streaming detection engine that chains bearing, range, and tracking.*

### dataclass `WindowDetection`

Per-window detection result.  Fields: `time`, `detected`, `window_rms`,
`bearings`, `bearing_rad`, `bearing_deg`, `range_m`, `x`, `y`, `z`,
`track` (`TrackState` or None), `n_sources`, `bearing_method`.

### class `DetectionEngine`

| Method | Description |
|--------|-------------|
| `__init__(mic_positions, fs, window_samples, bearing_method='srp_phat', range_method='rms', max_sources=1, ...)` | Create engine with chosen estimators |
| `calibrate_range(peak_rms, cpa_distance)` | Set RMS reference for range estimation |
| `reset()` | Clear tracker and smoother state |
| `process_window(segment, t_center) → WindowDetection` | Process one window through the full chain |

---

## air/detection/ranging.py

*Pluggable range estimation algorithms.*

### dataclass `RangeEstimate`

Single range estimate: `range_m`, `uncertainty_m`.

### class `RangeEstimator` (ABC)

Abstract base class.  Subclasses must implement
`estimate(segment, bearing_rad) → RangeEstimate`.

### class `RMSRangeEstimator(ref_range=10.0, ref_rms=None, range_min=5.0, range_max=100.0)`

Inverse-square-law range from RMS amplitude.

| Method | Description |
|--------|-------------|
| `__init__(ref_range=10.0, ref_rms=None, range_min=5.0, range_max=100.0)` | Init |
| `calibrate(peak_rms, cpa_distance)` | Set reference from CPA |
| `is_calibrated()` | Whether a reference RMS is set |
| `estimate(segment, bearing_rad=None)` | RMS → range |

### class `TDOARangeEstimator(mic_positions, fs, window_samples, c=343.0, range_min=2.0, range_max=200.0, ...)`

GCC-PHAT TDOA multilateration with bearing-constrained grid search.

| Method | Description |
|--------|-------------|
| `__init__(mic_positions, fs, window_samples, c=343.0, range_min=2.0, range_max=200.0, ...)` | Init |
| `estimate(segment, bearing_rad=None)` | TDOA → range |

### class `BearingRateRangeEstimator(source_speed=50.0, hop_sec=0.025, ema_alpha=0.3, range_min=2.0, range_max=200.0, min_rate_dps=5.0)`

Kinematic range from angular rate of change: $R \approx v/|\dot{\theta}|$.

| Method | Description |
|--------|-------------|
| `__init__(source_speed=50.0, hop_sec=0.025, ema_alpha=0.3, range_min=2.0, range_max=200.0, min_rate_dps=5.0)` | Init |
| `reset()` | Clear state |
| `estimate(segment, bearing_rad=None)` | Bearing rate → range |

### Factory functions

- `available_range_methods() → list[str]` — returns `["rms", "tdoa", "bearing_rate", "nearfield"]`
  (`"nearfield"` is a backward-compatible alias of `"tdoa"`)
- `create_range_estimator(method='rms', **kwargs) → RangeEstimator`

---

## air/detection/tracking.py

*Causal tracking (no Kalman filter).*

### dataclass `TrackState`

6-DOF track state: `x0, y0, z0, vx, vy, vz, t_ref, res_x, res_y,
res_z, n_det`.

| Method | Description |
|--------|-------------|
| `position_at(t) → ndarray` | Extrapolate position to time $t$ |
| `velocity → ndarray` | Return velocity vector |
| `covariance_6x6(floor=0.5, cap=1.0) → ndarray` | Diagonal covariance from residuals |

### class `CausalWLSTracker`

Weighted-least-squares constant-velocity fit over a sliding window.

| Method | Description |
|--------|-------------|
| `__init__(min_detections=5, max_history=20)` | Init |
| `reset()` | Clear detections |
| `n_detections → int` | Property: count of stored detections |
| `add_detection(t, x, y, z, rms)` | Append a detection |
| `fit() → TrackState \| None` | Solve WLS; returns None if too few detections |

### class `EMABearingSmoother`

Exponential moving average on the unit circle.

| Method | Description |
|--------|-------------|
| `__init__(alpha=0.35)` | Init |
| `reset()` | Clear state |
| `update(bearing_rad) → float` | Return smoothed bearing |

---

## ml/fno.py

*Fourier Neural Operator surrogate for FDTD.*  Requires `torch`.

### class `SpectralConv2d(in_channels, out_channels, modes1, modes2)`

Truncated Fourier-space convolution layer.  Retains the lowest
`modes1 × modes2` Fourier modes and applies a learnable complex weight.

### class `FNOBlock2d(width, modes1, modes2)`

Single FNO block: `SpectralConv2d` + pointwise `Conv2d(1×1)` +
residual skip + GELU activation.

### class `TraceDecoder(latent_dim, n_time_steps, hidden=256)`

MLP that maps latent features at receiver locations to time-domain
traces.  Output: `(batch, n_recv, n_time_steps)`.

### class `AcousticFNO(modes1=16, modes2=16, width=64, n_layers=4, n_time_steps=1000, in_channels=4)`

Full FNO model.  Input: a 4-channel field (normalised velocity model, source $x/y$ Gaussian
blobs, frequency encoding), built by `prepare_fno_input`.

| Method | Description |
|--------|-------------|
| `__init__(modes1=16, modes2=16, width=64, n_layers=4, n_time_steps=1000, in_channels=4)` | Build model |
| `forward(field, receiver_positions, grid_x, grid_y)` | Field → traces |

### `prepare_fno_input(velocity_field, grid_x, grid_y, source_x, source_y, source_freq, freq_scale=500.0, sigma=2.0)`

Build the 4-channel FNO input; `encode_source_on_grid(grid_x, grid_y, source_x, source_y, sigma=2.0)`
gives the Gaussian source blob.

---

## ml/fno_data_gen.py

*FDTD training data pipeline for FNO.*  Requires `torch`.

### `generate_dataset(n_samples=100, output_dir='data/fno_train', dims='2d', seed=42, **kwargs)`

Randomise the domain, source and receiver layout per sample, run the 2-D or 3-D FDTD solver and
save one `.npz` per sample.

CLI: `python -m acoustic_sim.ml.fno_data_gen --n-samples 500 --output-dir data/fno_train [--dims 3d]`

---

## ml/fno_training.py

*FNO training loop.*  Requires `torch`.

### class `FNODataset(data_dir, max_receivers=24, max_time_steps=0)`

Lazy `.npz` loader with receiver/trace padding.

### `train_fno(model, train_loader, val_loader=None, n_epochs=100, lr=0.001, weight_decay=1e-05, checkpoint_dir='checkpoints', verbose=True)`

Training loop: relative $L^2$ loss, masked receivers, cosine LR
schedule, gradient clipping.

CLI: `python -m acoustic_sim.ml.fno_training --data-dir data/fno_train --epochs 200`

---

## ml/fno_inference.py

*FNO inference wrapper.*  Requires `torch`.

### class `FNOForwardModel`

Drop-in replacement for FDTD.  Loads a trained checkpoint.

| Method | Description |
|--------|-------------|
| `__init__(checkpoint, modes=16, width=64, n_layers=4, n_time_steps=1000, device=None)` | Load model |
| `predict(velocity_field, grid_x, grid_y, receiver_positions, source_x, source_y, source_freq) → traces` | Forward pass |

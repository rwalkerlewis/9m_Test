# API Reference

Complete listing of every public class, method, and function in the
`acoustic_sim` package.  Organised by source file under
`src/acoustic_sim/`.  Line numbers are clickable references into the
source.

---

## backend.py

*NumPy / CuPy backend abstraction.*

### `get_backend(use_cuda: bool = False)` — L16

Return `(xp, is_cuda)` where `xp` is `numpy` or `cupy`.  When
`use_cuda=True` and CuPy is available, all array operations run on the
GPU.  Falls back silently to NumPy when CuPy is not installed.

---

## cli.py

*Command-line interface for acoustic_sim.*

### `parse_args(argv=None)` — L23

Build and return an `argparse.Namespace` for the Helmholtz / FDTD CLI.

### `main(argv=None)` — L112

Top-level entry point (`python -m acoustic_sim`).  Dispatches to the
Helmholtz solver, FDTD forward model, or detection pipeline depending on
`--mode`.

---

## config.py

*Detection pipeline configuration.*

### `sound_speed_from_temperature(t_celsius: float = 20.0)` — L21

Speed of sound from temperature:

$$c = 331.3\,\sqrt{1 + T/273.15}$$

Returns speed in m/s.

### class `DetectionConfig` — L27

All configurable parameters for the detection / tracking / fire-control
pipeline.  A plain dataclass with ~60 fields grouped into:

| Group | Key fields |
|-------|-----------|
| Domain / model | `grid_x_min`, `grid_x_max`, `dx`, `velocity`, `domain_type` |
| Source | `source_type`, `source_speed`, `source_altitude`, `n_sources` |
| Array | `array_type`, `array_x`, `array_y`, `n_mics`, `mic_radius` |
| Signal | `signal_type`, `fundamental`, `source_level_dB`, `total_time` |
| Noise | `wind_noise_level_dB`, `sensor_noise_level_dB`, `wind_corner_freq` |
| Beamformer | `mfp_azimuth_spacing`, `mfp_range_min/max`, `n_harmonics` |
| Tracker | `tracker_process_noise`, `sigma_bearing_deg`, `sigma_range` |
| Fire control | `muzzle_velocity`, `pellet_decel`, `spread_rate`, `weapon_x/y` |
| Faults | `fault_type`, `fault_fraction`, `inject_transient` |
| 3-D extras | `z_min`, `z_max`, `ground_z`, `enable_ground_reflection` |

---

## detection_main.py

*End-to-end passive acoustic drone detection, tracking, and fire
control.*  This is the **legacy MFP/EKF pipeline**; the current
production pipeline is `examples/run_pipeline.py` (SRP-PHAT).

### `simulate_scenario(config=None)` — L46

Run FDTD (or analytical forward model) to produce synthetic sensor data.
Builds the domain, source, and receiver array from `config`, runs the
solver, adds noise, and returns `(traces, mic_positions, dt, metadata)`.

### `run_detection(traces, mic_positions, dt, sound_speed=343.0, weapon_position=...)` — L206

Detection / tracking / fire-control — takes **only sensor data** (no
ground truth).  Runs broadband MFP → EKF tracker → fire-control
solution.  Returns a dict with keys `detections`, `track`, `fire_control`,
`multi_tracks`, `multi_fire_control`.

### `evaluate_results(detection_output, true_positions, true_velocities, true_times, weapon_position=..., pattern_spread_rate=0.025)` — L370

Compare detection output to ground truth.  Computes bearing error,
position error, miss distance, and hit/miss statistics.

### `run_detection_pipeline(config=None)` — L430

Run the full pipeline: simulate → detect → evaluate → plot.
Orchestrates `simulate_scenario`, `run_detection`, `evaluate_results`,
and all plotting calls.

### `parse_args(argv=None)` — L674

CLI argument parser for the detection pipeline.

### `main(argv=None)` — L727

Entry point for `python -m acoustic_sim.detection_main`.

### `run_detection_3d(traces, mic_positions, dt, sound_speed=343.0, weapon_position=...)` — L754

3-D detection / tracking / fire-control — takes **only sensor data**.
Uses 3-D MFP beamformer, 3-D EKF tracker, and 3-D ballistic lead.

### `evaluate_results_3d(detection_output, true_positions, true_velocities, true_times, weapon_position=..., pattern_spread_rate=0.025)` — L1079

Compare 3-D detection output to ground truth.

---

## domains.py

*Domain builders for FDTD simulations.*  Each builder returns
`(VelocityModel, DomainMeta)` or `(VelocityModel3D, DomainMeta3D)`.

### class `DomainMeta` — L22

Extra per-cell physics that supplement the 2-D velocity model.

| Attribute | Type | Description |
|-----------|------|-------------|
| `attenuation` | `ndarray` or `None` | Per-cell absorption coefficient (Np/m) |
| `wind_vx` | `ndarray` or `float` | East–west wind component (m/s) |
| `wind_vy` | `ndarray` or `float` | North–south wind component (m/s) |

### class `DomainMeta3D` — L277

Extra per-cell physics for a 3-D domain.

| Attribute | Type | Description |
|-----------|------|-------------|
| `attenuation` | `ndarray` or `None` | Per-cell absorption (Np/m) |
| `wind_vx` | `ndarray` or `float` | Wind x-component |
| `wind_vy` | `ndarray` or `float` | Wind y-component |
| `wind_vz` | `ndarray` or `float` | Wind z-component |

### 2-D domain builders

#### `create_isotropic_domain(x_min, x_max=50.0, y_min, y_max=50.0, dx=0.5, velocity=343.0)` — L35

Uniform velocity, no wind, no attenuation.

#### `create_wind_domain(x_min, x_max=50.0, y_min, y_max=50.0, dx=0.5, velocity=343.0, wind_speed=15.0, wind_direction_deg=45.0)` — L53

Uniform velocity with a constant wind field.  `wind_direction_deg` is
measured counter-clockwise from east.

#### `create_hills_vegetation_domain(x_min, x_max=50.0, y_min, y_max=50.0, dx=0.5, air_velocity=343.0, dirt_velocity=1500.0, veg_thickness=4.0, veg_attenuation=0.15, seed=42, hill_south_y, hill_north_y=20.0, hill_peak_height=18.0, hill_base_width=60.0)` — L113

2-D slice with two ridges, a valley, and vegetation zones.  Dirt (high
velocity) forms the hills; vegetation applies frequency-dependent
attenuation in a strip above the surface.

#### `create_echo_canyon_domain(x_min, x_max=100.0, y_min, y_max=100.0, dx=0.2, air_velocity=343.0, wall_velocity=2000.0, canyon_y_south, canyon_y_north=60.0, canyon_wall_thickness=5.0)` — L179

Domain with two parallel walls forming a canyon.  High-velocity walls
produce strong reflections.

#### `create_urban_echo_domain(x_min, x_max=100.0, y_min, y_max=100.0, dx=0.2, air_velocity=343.0, building_velocity=2500.0, n_buildings=4, building_size=15.0, seed=42)` — L221

Domain with rectangular buildings that produce complex multipath.
Buildings are placed randomly using `seed`.

### 3-D domain builders

#### `create_isotropic_domain_3d(x_min, x_max=50.0, y_min, y_max=50.0, z_min=0.0, z_max=100.0, dx=1.0, velocity=343.0)` — L291

Uniform velocity, no wind, no attenuation.

#### `create_wind_domain_3d(x_min, x_max=50.0, y_min, y_max=50.0, z_min=0.0, z_max=100.0, dx=1.0, velocity=343.0, wind_speed=15.0, wind_direction_deg=45.0, wind_vz=0.0)` — L312

Uniform velocity with a constant 3-D wind field.

#### `create_ground_layer_domain_3d(x_min, x_max=50.0, y_min, y_max=50.0, z_min, z_max=100.0, dx=1.0, air_velocity=343.0, ground_velocity=1500.0, ground_z=0.0)` — L346

3-D domain with air above and ground (high velocity) below `ground_z`.

#### `create_hills_vegetation_domain_3d(x_min, x_max=50.0, y_min, y_max=50.0, z_min, z_max=50.0, dx=1.0, air_velocity=343.0, dirt_velocity=1500.0, veg_thickness=4.0, veg_attenuation=0.15, seed=42, hill_south_y, hill_north_y=20.0, hill_peak_height=18.0, hill_base_width=60.0)` — L378

3-D valley between two ridges, extruded from the 2-D model.

---

## fdtd.py

*2-D and 3-D FDTD acoustic wave-equation solver with MPI + CUDA
support.*

### `fd2_coefficients(order: int)` — L79

Central finite-difference coefficients for $d^2f/dx^2$ at accuracy
`order` (2, 4, 6, or 8).  Returns a 1-D numpy array of length
$\lfloor\text{order}/2\rfloor + 1$.

### `fd2_cfl_factor(coeffs)` — L106

Return the 1-D spectral radius of the stencil at Nyquist.  Used
internally to compute the maximum stable time step.

### class `FDTDConfig` — L126

User-tunable simulation parameters (dataclass).

| Field | Default | Description |
|-------|---------|-------------|
| `total_time` | `1.0` | Simulation duration (s) |
| `fd_order` | `2` | Spatial FD accuracy (2, 4, 6, 8) |
| `cfl_safety` | `0.9` | CFL safety margin |
| `damping_width` | `20` | Absorbing-boundary layer width (cells) |
| `damping_alpha` | `0.015` | Absorbing-boundary strength |
| `use_cuda` | `False` | Use CuPy GPU acceleration |
| `source_amplitude` | `1.0` | Source injection scale factor |
| `snapshot_interval` | `50` | Save wavefield every N steps |

### class `FDTDSolver` — L188

2-D FDTD acoustic solver with MPI domain decomposition and optional CUDA
acceleration.

#### `__init__(model, config, source, receivers, domain_meta=None)`  — L210

Build the solver.  Decomposes the grid across MPI ranks, allocates
pressure and velocity fields, pre-computes FD stencil coefficients and
CFL-limited `dt`.

**Stored attributes** (available after construction):

| Attribute | Description |
|-----------|-------------|
| `dt` | Actual time step (s) |
| `n_steps` | Total number of time steps |
| `f_max` | Maximum resolvable frequency (Hz) |

#### `run(snapshot_dir=None, verbose=True)` — L553

Run the full simulation.  Returns `(traces, dt)` where `traces` is
`(n_receivers, n_steps)`.  If `snapshot_dir` is set, saves wavefield PNGs
at the configured interval.

### class `FDTD3DSolver` — L627

3-D FDTD acoustic solver with MPI domain decomposition and optional CUDA
acceleration.

#### `__init__(model, config, source, receivers, domain_meta=None)` — L645

Same contract as the 2-D solver but operates on `VelocityModel3D`.

#### `run(snapshot_dir=None, snapshot_z_index=None, snapshot_y_index=None, verbose=True, field_plane_z=None, field_plane_subsample=4)` — L1046

Run the full 3-D simulation.  Returns `(traces, dt)`.

Extra parameters:

| Parameter | Description |
|-----------|-------------|
| `snapshot_z_index` | z-slice for wavefield snapshots |
| `snapshot_y_index` | y-slice for wavefield snapshots |
| `field_plane_z` | If set, record the full x-y field plane at this z every snapshot |
| `field_plane_subsample` | Spatial sub-sampling factor for field planes (saves memory) |

---

## fire_control.py

*Fire-control solution for a 12-gauge shotgun engagement.*

All ballistic functions model constant deceleration:
$r = v_0 t - \tfrac{1}{2}at^2$ with default $v_0 = 400\,\text{m/s}$,
$a = 1.5\,\text{m/s}^2$.

### `time_of_flight(range_m, muzzle_velocity=400.0, decel=1.5)` — L28

Solve $r = v_0 t - \tfrac{1}{2}at^2$ for $t$ (positive root).

### `pellet_velocity_at_range(range_m, muzzle_velocity=400.0, decel=1.5)` — L68

Return pellet speed at `range_m` (m/s).

### `pattern_diameter(range_m, spread_rate=0.025)` — L78

Shotgun pattern diameter at `range_m` (m).  Uses linear spread:
$d = \text{spread\_rate} \times r$.

### `compute_lead(target_pos, target_vel, weapon_pos, muzzle_velocity=400.0, decel=1.5, max_iter=5)` — L93

Compute iterative lead-angle solution.  Predicts where the target will
be when the pellets arrive, iterating up to `max_iter` times.  Returns
`(aim_point, tof)`.

### `compute_engagement(target_pos, target_vel, target_cov, weapon_pos, muzzle_velocity=400.0, decel=1.5, spread_rate=0.025, max_iter=5, max_position_uncertainty=0.0, max_engagement_range=0.0)` — L163

Determine whether engagement is feasible.  Checks range limits,
position-uncertainty limits, and pellet energy at target range.  Returns
a dict with `feasible`, `aim_point`, `tof`, `pattern_diameter`,
`p_hit`, `reason`.

### `run_fire_control(track: dict)` — L264

Compute fire-control solution at every tracked time step.  Takes the
output of `run_tracker()` and appends engagement decisions to each
frame.

### `compute_miss_distance(fire_control, true_positions, true_times, weapon_position, pattern_spread_rate=0.025)` — L379

Compute how far each shot would miss the actual target.  Compares the
predicted aim point (at the time of pellet arrival) against the true
target position interpolated from `true_positions` / `true_times`.

### `prioritize_threats(tracks, weapon_pos, time_idx, w_range=1.0, w_closing=2.0, w_quality=0.5)` — L484

Score and rank tracked targets by threat priority.  The score is a
weighted combination of range, closing rate, and track-quality factors.

### `run_multi_fire_control(tracks)` — L553

Fire-control solution for multiple targets, sorted by threat priority.

### 3-D variants

| Function | Line | Notes |
|----------|------|-------|
| `compute_lead_3d(...)` | 604 | Same algorithm in 3-D |
| `compute_engagement_3d(...)` | 675 | Adds `class_label`, `class_confidence`, `confidence_threshold`, `maneuver_class` — suppresses fire on `bird`/`unknown` below confidence |
| `run_fire_control_3d(track)` | 778 | 3-D fire-control at every time step |
| `compute_miss_distance_3d(...)` | 881 | 3-D miss distances |
| `prioritize_threats_3d(...)` | 960 | 3-D threat scoring |

### Trajectory and CPA analysis

### `projectile_path(weapon_pos, aim_bearing, aim_elevation, muzzle_velocity, decel, tof, n_points=50)`

Generate 3-D pellet trajectory from weapon position along aim direction.
Returns `(xs, ys, zs)` arrays of $n$ evenly spaced points with ballistic
velocity decay.

### `find_cpa(weapon_pos, aim_dir, muzzle_velocity, pellet_decel, target_position_fn, t_fire, n_samples=200)`

Evaluate the closest point of approach between the projectile path and
the target trajectory over the time-of-flight window.  Returns a dict
with `miss_distance_m`, `cpa_time`, `cpa_pellet_pos`, `cpa_target_pos`,
`pattern_radius_at_cpa`.

### `compute_bearing_rate(bearing_history, t_now, bearing_rad, window_s=0.15)`

Compute the angular rate of change of bearing from a sliding history
window.  Returns `(rate_rad_s, updated_history)`.  Used for
bearing-rate fire gating.

---

## forward.py

*Analytical 3-D forward model for generating synthetic microphone
traces.*  Implements spherical spreading ($1/r$), frequency-dependent
air absorption, and optional ground reflection (image source).

### `simulate_3d_traces(source, mic_positions, dt, n_steps, sound_speed=343.0, air_absorption=0.005, enable_ground_reflection=False, ground_reflection_coeff, ground_z=0.0)` — L31

Generate synthetic microphone traces from a 3-D source.  Time-domain
delay-and-attenuate per sample.

### `simulate_3d_traces_vectorized(...)` — L148

Vectorized version using frequency-domain delay application.  Much
faster for large `n_steps`.

### `simulate_scenario_3d(sources, mic_positions, dt, n_steps, sound_speed=343.0, ...)` — L235

Run full 3-D scenario: forward model + wind/sensor noise.  Accepts
multiple sources and all noise parameters.

### `simulate_3d_traces_fdtd(source, mic_positions, dt, n_steps=None, total_time=1.0, sound_speed=343.0, dx=1.0, domain_margin=20.0, z_min, z_max=120.0, damping_width=10, fd_order=2, air_absorption=0.005, source_amplitude=1.0, verbose=True)` — L347

Generate microphone traces using the 3-D FDTD solver.  Automatically
builds a domain around the source/receiver geometry.

### `simulate_scenario_3d_fdtd(sources, mic_positions, total_time=1.0, sound_speed=343.0, dx=1.0, ...)` — L459

Run full 3-D scenario using the FDTD forward model + noise.

---

## io.py

*I/O helpers — JSON config loading, velocity-model persistence.*

### `load_json(path)` — L23

Load and return a JSON file, or `None` if `path` is falsy.

### `save_model(model: VelocityModel, path: str)` — L34

Persist a velocity model to a `.npz` file (velocity grid + extent).

### `load_model(path: str)` — L42

Load a velocity model from a `.npz` file.

### `model_from_json(cfg)` — L54

Build a velocity model from a JSON configuration dict.  Supports
`"uniform"`, `"layered"`, `"gradient"`, `"checkerboard"`, and
`"valley"` model types.

---

## model.py

*Velocity model dataclass and creation helpers (2-D and 3-D).*

### class `VelocityModel` — L11

2-D velocity model on a regular grid.

| Attribute | Type | Description |
|-----------|------|-------------|
| `values` | `ndarray (ny, nx)` | Velocity grid in m/s |
| `x_min`, `x_max` | `float` | Horizontal extent (m) |
| `y_min`, `y_max` | `float` | Vertical extent (m) |
| `dx` | `float` | Grid spacing (m) |

#### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `nx` | `int` | Number of x grid points |
| `ny` | `int` | Number of y grid points |
| `shape` | `(ny, nx)` | Grid dimensions |
| `extent` | `(x_min, x_max, y_min, y_max)` | For matplotlib `imshow` |
| `c_min` | `float` | Minimum velocity in grid |
| `c_max` | `float` | Maximum velocity in grid |

#### `velocity_at(px: float, py: float)` — L64

Nearest-neighbour velocity look-up at an arbitrary `(x, y)` point.

### 2-D model builders

#### `create_uniform_model(x_min, x_max, y_min, y_max, dx, velocity=343.0)` — L76

Constant-velocity model.

#### `create_layered_model(x_min, x_max, y_min, y_max, dx, layers, background=343.0)` — L91

Horizontally layered model.  `layers` is a list of
`(y_start, y_end, velocity)` tuples.

#### `create_gradient_model(x_min, x_max, y_min, y_max, dx, v_bottom=360.0, v_top=330.0)` — L121

Linear velocity gradient from `v_bottom` (y_min) to `v_top` (y_max).

#### `create_checkerboard_model(x_min, x_max, y_min, y_max, dx, cell_size=4.0, v_base=343.0, perturbation=20.0)` — L138

Checkerboard velocity perturbation (useful for resolution tests).

#### `model_from_array(values, x_min, x_max, y_min, y_max)` — L158

Wrap an existing 2-D numpy array as a `VelocityModel`.

#### `add_circle_anomaly(model, cx, cy, radius, velocity)` — L191

Return a copy of `model` with a circular region overwritten.

#### `add_rectangle_anomaly(model, x0, x1, y0, y1, velocity)` — L208

Return a copy of `model` with a rectangular region overwritten.

#### `create_valley_model(x_min, x_max=50.0, y_min, y_max=50.0, dx=0.5, air_velocity=343.0, dirt_velocity=1500.0, seed=42, hill_south_y, hill_north_y=20.0, hill_peak_height=18.0, hill_base_width=60.0, saddle_x=0.0, saddle_width=12.0, saddle_depth_frac=0.55)` — L255

Create a valley between two ridges with a saddle notch.  Uses Perlin-like
noise seeded by `seed` for surface roughness.

### class `VelocityModel3D` — L318

3-D velocity model on a regular grid.

| Attribute | Type | Description |
|-----------|------|-------------|
| `values` | `ndarray (nz, ny, nx)` | Velocity grid in m/s |
| `x_min` .. `z_max` | `float` | 3-D extent (m) |
| `dx` | `float` | Grid spacing (m) |

#### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `nx`, `ny`, `nz` | `int` | Grid counts |
| `shape` | `(nz, ny, nx)` | Grid dimensions |
| `extent_xy` | `(4-tuple)` | x-y slice extent for `imshow` |
| `extent_xz` | `(4-tuple)` | x-z slice extent for `imshow` |
| `c_min`, `c_max` | `float` | Velocity range |

#### `velocity_at(px, py, pz)` — L383

Nearest-neighbour velocity look-up at an arbitrary `(x, y, z)` point.

### 3-D model builders

#### `create_uniform_model_3d(x_min, x_max, y_min, y_max, z_min, z_max, dx, velocity=343.0)` — L396

Constant-velocity 3-D model.

#### `create_layered_z_model_3d(x_min, x_max, y_min, y_max, z_min, z_max, dx, layers, background=343.0)` — L414

3-D model with horizontal layers defined by z boundaries.

#### `model_3d_from_array(values, x_min, x_max, y_min, y_max, z_min, z_max)` — L447

Wrap an existing 3-D numpy array as a `VelocityModel3D`.

---

## noise.py

*Post-hoc noise generators for microphone traces.*

### `generate_wind_noise(mic_positions, n_samples, dt, level_dB=60.0, corner_freq=15.0, correlation_length=3.0, seed=42)` — L26

Spatially correlated, spectrally shaped wind noise.  Low-pass filtered
at `corner_freq` Hz with exponential spatial correlation decaying over
`correlation_length` metres.

### `generate_sensor_noise(n_mics, n_samples, dt, level_dB=30.0, seed=43)` — L101

Uncorrelated white Gaussian sensor self-noise.  Each microphone channel
gets independent noise at `level_dB` dB SPL.

### `add_all_noise(traces, stationary_traces, mic_positions, dt)` — L133

Sum drone traces, optional stationary traces, and post-hoc noise.
Convenience wrapper that calls `generate_wind_noise` +
`generate_sensor_noise` and adds them to the input traces.

### `inject_sensor_faults(traces, fault_type='elevated_noise', fault_sensors=None, fault_fraction=0.2, fault_level_dB=100.0, spike_rate=0.01, seed=44)` — L196

Inject faults into selected sensors.

| `fault_type` | Effect |
|-------------|--------|
| `elevated_noise` | Add high-level white noise to selected channels |
| `dead` | Zero out the channel entirely |
| `intermittent` | Random spike artifacts at `spike_rate` |

### `inject_transient(traces, dt, event_time, event_pos, mic_positions, level_dB=130.0, duration_ms=10.0, sound_speed=343.0, seed=55)` — L260

Inject a broadband transient (explosion / gunshot) into traces at
`event_time` seconds from position `event_pos`.  Properly delays the
impulse to each microphone by distance.

### `perturb_mic_positions(true_positions, error_std=2.0, seed=77)` — L335

Add Gaussian position errors to microphone coordinates.  Returns a
perturbed copy; the original is not modified.

---

## plotting.py

*Plotting utilities for velocity models and wavefields.*  All functions
write PNG files to `output_path`; none display interactively.

### 2-D plots

| Function | Line | Description |
|----------|------|-------------|
| `plot_velocity_model(model, output_path, receivers, source_xy, title)` | 17 | Velocity field with optional mic/source overlays |
| `plot_wavefield(model, field, output_path, receivers, source_xy, title)` | 59 | Helmholtz pressure magnitude (dB) |
| `plot_gather(traces, dt, output_path, title, db_range, cmap)` | 120 | Receiver traces as dB SPL gather (time vs channel) |
| `save_snapshot(model, field, step, output_dir, receivers, source_xy, db_range, title)` | 171 | Single numbered wavefield snapshot (PNG) |
| `plot_domain(model, output_path, receivers, source_xy, source_path, attenuation, wind_vx, wind_vy, title)` | 227 | Velocity model with source trajectory, attenuation, and wind overlays |

### Detection pipeline plots

| Function | Line | Description |
|----------|------|-------------|
| `plot_detection_domain(model, receivers, source_positions, weapon_pos, stationary_pos, output_path, title)` | 346 | Domain overview with mics, drone trajectory, and weapon |
| `plot_detection_gather(traces, filtered_traces, dt, output_path, title, db_range)` | 408 | Two-panel gather: raw (left) and filtered (right) |
| `plot_beam_power(results, true_positions, grid_x, grid_y, output_path, n_panels)` | 452 | Multi-panel beam-power / coherence snapshots |
| `plot_tracking(track, true_positions, true_times, fire_control, weapon_pos, output_path)` | 535 | Four-panel tracking and fire-control display (range, bearing, position, engagement) |
| `plot_vespagram(traces, mic_positions, dt, output_path, sound_speed, slowness_range, n_slowness, title)` | 615 | Beam power as a function of slowness and time |
| `plot_study_comparison(labels, metrics, output_path, title)` | 706 | Bar-chart comparison of metrics across study cases |
| `plot_multi_track(tracks, true_positions_list, output_path, title)` | 744 | Spatial plot of multiple tracks + true trajectories |
| `plot_polar_beam_power(results, azimuths, ranges, true_positions, array_center, output_path, n_panels)` | 787 | Multi-panel polar beam-power maps |

### 3-D plots

| Function | Line | Description |
|----------|------|-------------|
| `plot_3d_trajectory(true_positions, estimated_positions, mic_positions, weapon_pos, title, output_path)` | 866 | 3-D source trajectory with optional estimated track |
| `plot_altitude_vs_time(times, true_z, estimated_z, estimated_times, title, output_path)` | 927 | Altitude over time for true and estimated |
| `plot_tracking_3d(track, true_positions, true_times, fire_control, weapon_pos, output_path, maneuver_labels, class_label, class_confidence)` | 954 | Six-panel 3-D tracking + fire-control display with altitude and maneuver state |
| `plot_kinematic_scatter(features_by_class, feature_names, title, output_path)` | 1087 | 2-D scatter of kinematic features coloured by class |
| `save_snapshot_3d(field_3d, step, output_dir)` | 1132 | Two-panel (x-y + x-z) wavefield snapshot |

### Pipeline engagement plots

| Function | Description |
|----------|-------------|
| `plot_radial_engagement(fire_decisions, ground_truth_fn, source_duration, weapon_pos, is_3d, cfg_fc, output_path)` | Polar plot of fire decisions with per-shot effective threshold markers; colours by hit/miss |
| `plot_pipeline_summary(all_detections, all_fire_decisions, all_track_states, wall_times, ground_truth_fn, source_duration, array_center, weapon_pos, is_3d, hop_sec, hit_threshold, metrics, output_path)` | Six-panel pipeline summary: bearing, range, position track, fire decisions, RMS, and metrics |
| `plot_beamformer_diagnostic(traces, mic_positions, dt, ground_truth_fn, source_duration, array_center, beamformer, cfg_det, output_path)` | Four-panel beamformer diagnostic: power spectrum, bearing vs time, polar power map, SNR |

---

## processor.py

*Frequency-domain matched field processor (MFP) with MVDR beamforming.*

### Grid construction

#### `build_polar_grid(azimuth_spacing_deg=1.0, range_min=20.0, range_max=500.0, range_spacing=5.0)` — L54

Build polar search grid.  Returns `(azimuths, ranges, gx, gy)` where
`gx`, `gy` are 2-D Cartesian coordinate meshes.

#### `polar_to_cartesian(azimuths, ranges, center_x=0.0, center_y=0.0)` — L69

Convert polar grid to Cartesian coordinate arrays.

### Travel times & steering

#### `compute_travel_times_polar(gx, gy, mic_positions, sound_speed=343.0)` — L86

Travel times from each grid point to each microphone.  Returns
`(n_grid_points, n_mics)`.

#### `compute_steering_vectors(freqs, travel_times)` — L111

Complex steering vectors: $a_{ij} = e^{-2\pi i f_j \tau_i}$.

#### `select_harmonic_bins(n_fft, dt, fundamental, n_harmonics, bandwidth)` — L135

Return FFT bin indices for each harmonic $\pm$ bandwidth.

### CSDM & beamforming

#### `compute_csdm(traces, dt, window_start, window_length, freq_bins, n_subwindows=4)` — L155

Cross-spectral density matrix averaged over `n_subwindows`
sub-windows.  Returns `(n_freq, n_mics, n_mics)` complex array.

#### `mvdr_beam_power(csdm, steering, epsilon=0.01)` — L212

MVDR (Capon) beam power: $P = 1 / (a^H C^{-1} a)$.  Diagonal loading
controlled by `epsilon`.

#### `conventional_beam_power(csdm, steering)` — L262

Conventional (delay-and-sum) beam power: $P = a^H C a$.

#### `broadband_weighted_sum(beam_powers, freqs)` — L278

Frequency-weighted broadband sum: $w(f) = (f / f_\text{max})^2$.

### Peak detection & sensor health

#### `find_peaks_polar(bpm, azimuths, ranges, threshold=0.25, max_sources=5, min_sep_deg=10.0)` — L309

Find peaks in the polar beam-power map with sub-grid interpolation.
Returns a list of dicts with `bearing`, `range`, `amplitude`.

#### `compute_sensor_weights(traces, fault_threshold=10.0)` — L366

Per-sensor reliability weights.  Sensors with anomalous RMS are
down-weighted.

#### `blank_transients(traces, dt, subwindow_ms=5.0, threshold_factor=10.0)` — L382

Zero out impulsive sub-windows (transient suppression).

#### `detect_stationary(history, cv_threshold=0.15)` — L407

Identify stationary grid points from beam-power history (low coefficient
of variation).

#### `calibrate_positions(traces, reported_positions, dt, sound_speed=343.0, max_lag_m=2.0)` — L421

Cross-correlation TDOA self-calibration.  Adjusts reported microphone
positions to match observed inter-channel delays.

### Full MFP pipeline

#### `matched_field_process(traces, mic_positions, dt)` — L477

Broadband frequency-domain MVDR matched field processing.  Runs the
complete MFP pipeline: grid → steering → CSDM → MVDR → peaks for
each time window.

### 3-D variants

All have the same interface as their 2-D counterparts but operate on
3-D grids (azimuth × range × z).

| Function | Line | Description |
|----------|------|-------------|
| `build_3d_grid(...)` | 704 | Build polar + z search grid |
| `compute_travel_times_3d(gx, gy, gz, mic_positions, sound_speed)` | 725 | Travel times from 3-D grid points |
| `compute_steering_vectors_3d(freqs, travel_times)` | 773 | Complex steering vectors for 3-D grid |
| `mvdr_beam_power_3d(csdm, steering, epsilon)` | 792 | MVDR beam power for 3-D steering |
| `broadband_weighted_sum_3d(beam_powers, freqs)` | 839 | Frequency-weighted broadband sum |
| `find_peaks_3d(bpm, azimuths, ranges, z_values, ...)` | 859 | 3-D peak finding |
| `matched_field_process_3d(traces, mic_positions, dt)` | 944 | Full 3-D MFP pipeline |

---

## receivers.py

*Receiver (microphone array) geometry helpers.*  Every function returns
an `(N, 2)` or `(N, 3)` numpy array of microphone positions.

### 2-D array geometries

#### `create_receiver_nested_circular(cx=0.0, cy=0.0, inner_radius=0.15, outer_radius=0.5, n_inner=4, n_outer=8)` — L31

Nested circular array: 1 centre + inner ring + outer ring.  Returns
`(1 + n_inner + n_outer, 2)`.

#### `create_receiver_line(x_start, y_start, x_end, y_end, count)` — L59

Line of receivers.  Returns shape `(count, 2)`.

#### `create_receiver_circle(cx, cy, radius, count)` — L71

Single-ring circular array.  Returns shape `(count, 2)`.

#### `create_receiver_concentric(cx, cy, radii, counts_per_ring=12)` — L83

Concentric rings.  `radii` is a list of ring radii; `counts_per_ring`
can be an int (same for all rings) or a list.  Returns `(N_total, 2)`.

#### `create_receiver_l_shaped(n1, n2, spacing, origin_x=0.0, origin_y=0.0)` — L96

L-shaped array.  Returns `(n1 + n2 - 1, 2)`.

#### `create_receiver_log_spiral(count=13, radius=0.5, cx=0.0, cy=0.0)` — L112

Logarithmic (golden-angle) spiral for maximum baseline diversity.

#### `create_receiver_random_disk(count=13, radius=0.5, cx=0.0, cy=0.0, seed=42)` — L134

Random positions within a disk.  Returns `(count, 2)`.

#### `create_receiver_random(count, x_min, x_max, y_min, y_max, seed=42)` — L152

Random positions within a bounding box.  Returns `(count, 2)`.

#### `create_receiver_custom(positions)` — L164

Wrap a list of `(x, y)` tuples as an `(N, 2)` array.

### Array diagnostics

#### `print_array_diagnostics(positions, sound_speed=343.0)` — L175

Print and return a diagnostic summary of array geometry.  Reports:
inter-element spacing (min, max, mean), aperture diameter, angular
coverage, spatial aliasing frequency, and Rayleigh resolution.

### 3-D array geometries

All append a constant `z` coordinate.

| Function | Line | Returns |
|----------|------|---------|
| `create_receiver_l_shaped_3d(n1, n2, spacing, origin_x, origin_y, z)` | 257 | `(n1+n2-1, 3)` |
| `create_receiver_circle_3d(cx, cy, radius, count, z)` | 267 | `(count, 3)` |
| `create_receiver_nested_circular_3d(cx, cy, inner_radius, outer_radius, n_inner, n_outer, z)` | 276 | `(1+n_inner+n_outer, 3)` |
| `create_receiver_line_3d(x_start, y_start, x_end, y_end, count, z)` | 288 | `(count, 3)` |
| `create_receiver_random_disk_3d(count, radius, cx, cy, z, seed)` | 297 | `(count, 3)` |
| `create_receiver_custom_3d(positions)` | 307 | `(N, 3)` |

---

## sampling.py

*Spatial-sampling and CFL stability checks.*

### `check_spatial_sampling(model, frequency_hz, min_ppw=10.0)` — L12

Verify that the grid resolves the shortest wavelength.  Requires at
least `min_ppw` points per wavelength at `frequency_hz`.  Returns
`(ok, actual_ppw)`.

### `check_cfl(model, dt)` — L61

CFL stability check for an explicit time-domain scheme.  Returns
`(ok, courant_number)`.

### `suggest_dx(c_min, frequency_hz, min_ppw=10.0)` — L90

Return the maximum allowable `dx` for a given frequency and `c_min`:

$$\Delta x \le \frac{c_\min}{f \cdot \text{ppw}}$$

---

## setup.py

*Simulation setup helpers.*  High-level convenience builders used by the
CLI and detection pipeline.

### `build_domain(domain='isotropic')` — L58

Build a velocity model and domain metadata.  Accepts domain name strings
(`isotropic`, `wind`, `hills_vegetation`, `echo_canyon`, `urban_echo`,
`valley`) and returns `(model, domain_meta)`.

### `build_receivers(array='circular')` — L111

Build a receiver array.  Accepts array name strings (`circular`, `line`,
`l_shaped`, `nested`, `concentric`, `log_spiral`, `random_disk`,
`random`).  Returns shape `(n_recv, 2)`.

### `compute_dt(model, meta, cfl_safety=0.9, fd_order=2)` — L166

Return `(dt, f_max)` from CFL and spatial sampling constraints.

### `build_source(source_type='static', signal_type='ricker')` — L193

Build a source object with its signal.  `source_type` selects the
trajectory class (`static`, `moving`, `circular`, `figure_eight`,
`loiter`, `evasive`, `custom`); `signal_type` selects the waveform
(`ricker`, `propeller`, `tone`, `noise`, `drone`, `wav`).

---

## solver.py

*2-D Helmholtz (frequency-domain) solver.*

### `solve_helmholtz(model, source_xy, frequency_hz, damping_width=None)` — L12

Solve the 2-D Helmholtz equation on the velocity-model grid.  Builds a
sparse system $(\nabla^2 + k^2) p = s$ with PML absorbing boundaries and
solves via `scipy.sparse.linalg.spsolve`.

Returns `field` with shape `(ny, nx)` (complex pressure).

---

## sources.py

*Source signal generators and source-position helpers for FDTD.*

### Signal generators

All return a 1-D numpy array of length `n_steps`.

#### `load_wav_mono(path: str)` — L56

Load a WAV file and return `(mono_float32, sample_rate)`.

#### `prepare_source_signal(raw, fs_audio, dt_sim, f_max)` — L76

Low-pass filter and resample an audio signal for the FD simulation.
Applies a Butterworth filter at `f_max`, then resamples from `fs_audio`
to `1/dt_sim`.

#### `make_wavelet_ricker(n_steps, dt, f0)` — L132

Ricker (Mexican-hat) wavelet centred at $t_0 = 1.5/f_0$.

#### `make_source_propeller(n_steps, dt, f_max=None, blade_count=3, rpm=3600.0, harmonics=14, mod_depth=0.25, broadband_level=0.12, seed=42)` — L150

Synthetic propeller / rotor noise.  Sum of blade-pass harmonics with
amplitude modulation (Doppler-like) plus broadband turbulent self-noise.

#### `make_source_tone(n_steps, dt, frequency_hz=40.0)` — L198

Pure sine-wave source at `frequency_hz`.

#### `make_source_noise(n_steps, dt, f_low=5.0, f_high=60.0, seed=42)` — L208

Band-limited coloured noise source.

#### `make_source_from_file(path, n_steps, dt, f_max, max_seconds=None)` — L228

Load a WAV file and prepare it for FD injection.

#### `make_drone_harmonics(n_steps, dt, fundamental=150.0, n_harmonics=4, harmonic_amplitudes=None, source_level_dB=90.0, f_max=None)` — L252

Synthetic drone rotor signal as a sum of harmonics.  Each harmonic
rolls off as $1/(n+1)$ unless explicit `harmonic_amplitudes` are given.
Normalised to `source_level_dB` SPL re 20 µPa.

#### `make_stationary_tonal(n_steps, dt, base_freq=60.0, n_harmonics=4, source_level_dB=70.0, broadband_level=0.1, f_max=None, seed=99)` — L322

Tonal signal for a stationary coherent noise source (e.g. generator,
HVAC).

### 2-D trajectory classes

All expose `position_at(step, dt) → (x, y)`.

#### class `StaticSource` — L370

Fixed position.  Constructor: `StaticSource(x, y, signal)`.

#### class `MovingSource` — L382

Linear trajectory with optional parabolic arc.  Constructor:
`MovingSource(x0, y0, x1, y1, total_time, signal, arc_height=0.0)`.

#### class `CircularOrbitSource` — L412

Circular orbit.  Constructor:
`CircularOrbitSource(cx, cy, radius, period, signal, phase0=0.0)`.

#### class `FigureEightSource` — L446

Lissajous figure-eight.  Constructor:
`FigureEightSource(cx, cy, rx, ry, period, signal, phase0=0.0)`.

#### class `LoiterApproachSource` — L482

Orbit at standoff, then transition to linear approach.  Constructor
includes `loiter_time`, `approach_time`, standoff radius and final
position.

#### class `EvasiveSource` — L546

Random-walk heading overlaid on a general course.  Constructor includes
`heading_std`, `turn_rate`, base speed, and `seed`.

#### class `CustomTrajectorySource` — L618

User-supplied trajectory as `(t, x, y)` arrays.  Interpolates linearly
between waypoints.

### 3-D trajectory classes

All expose `position_at(step, dt) → (x, y, z)`.  Same motion models as
2-D with an added z-coordinate.

| Class | Line | Description |
|-------|------|-------------|
| `StaticSource3D` | 720 | Fixed position |
| `MovingSource3D` | 732 | Linear trajectory from `(x0,y0,z0)` to `(x1,y1,z1)` |
| `CircularOrbitSource3D` | 765 | Circular orbit at constant altitude |
| `FigureEightSource3D` | 787 | Figure-eight with optional z oscillation |
| `LoiterApproachSource3D` | 817 | Loiter then descend on approach |
| `EvasiveSource3D` | 869 | Random-walk heading with altitude variation |
| `ErraticQuadcopterSource3D` | — | Ornstein–Uhlenbeck mean-reverting 3-D trajectory confined to a bounding box; fields: `x0, y0, z0, bbox_min, bbox_max, mean_speed, agility, speed_var, signal, seed` |
| `CustomTrajectorySource3D` | 923 | User-supplied `(t, x, y, z)` arrays |

### Utility functions

#### `source_velocity_at(source, step, dt, eps_steps=1)` — L648

Compute instantaneous 2-D velocity via central finite differences.

#### `inject_source(field, sx, sy, amplitude, x_arr, y_arr, dx, dy, ix_offset=0, iy_offset=0)` — L675

Inject `amplitude` into FDTD `field` at `(sx, sy)` using bilinear
interpolation weights (sub-cell accuracy).

#### `source_velocity_at_3d(source, step, dt, eps_steps=1)` — L944

Compute instantaneous 3-D velocity via central finite differences.

---

## studies.py

*Systematic parameter-sweep studies for the detection pipeline.*  Each
function runs a sweep over one variable and records detection/tracking
metrics.

### `study_array_geometry(base_config=None, output_dir='output/studies/array_geometry')` — L90

Compare circular, linear, L-shaped, random, concentric arrays.

### `study_min_sensors(base_config=None, sensor_counts=..., output_dir='output/studies/min_sensors')` — L127

Sweep `n_mics` to find minimum acceptable sensor count.

### `study_sensor_faults(base_config=None, fault_fractions=..., output_dir='output/studies/sensor_faults')` — L164

Sweep fault fraction with and without robust sensor weighting.

### `study_multi_drone(base_config=None, output_dir='output/studies/multi_drone')` — L232

Test with 1 and 2 simultaneous drones using multi-peak detection.

### `study_transient_robustness(base_config=None, transient_levels=..., output_dir='output/studies/transient')` — L284

Inject transients, compare with/without blanking.

### `study_haphazard_array(base_config=None, n_trials=3, output_dir='output/studies/haphazard')` — L357

Compare optimised circular array vs random placements.

### `study_echo_domains(base_config=None, output_dir='output/studies/echo')` — L405

Compare detection in isotropic vs echo-prone domains.

### `study_position_errors(base_config=None, error_stds=..., output_dir='output/studies/position_error')` — L441

Sweep position error magnitude with/without self-calibration.

### `study_mixed_failures(base_config=None, output_dir='output/studies/mixed')` — L507

Combined stress test: faults + position errors + transient + echo
domain.

### `run_all_studies(base_config=None, output_dir='output/studies')` — L584

Execute all nine studies and return combined results.

---

## tracker.py

*Extended Kalman Filter tracker for bearing-primary measurements.*

### class `EKFTracker` — L36

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

| Method | Line | Description |
|--------|------|-------------|
| `initialise_from_bearing(bearing, range_est, center_x, center_y)` | 77 | Place initial state at bearing from array centre |
| `predict(dt)` | 114 | Constant-velocity prediction |
| `update(bearing, range_est, amplitude, center_x, center_y)` | 171 | EKF measurement update |
| `get_state()` | 206 | Return full 4-element state vector |
| `get_position()` | 209 | Return `(x, y)` |
| `get_velocity()` | 212 | Return `(vx, vy)` |
| `get_covariance()` | 215 | Return 4×4 covariance matrix |
| `get_range_uncertainty(cx, cy)` | 218 | 1-σ range uncertainty from covariance |

### `run_tracker(detections)` — L232

Run the EKF tracker on MFP detections.  Takes a list of per-window
detection dicts and returns a track dict with position/velocity/covariance
time series.

### class `MultiTargetTracker` — L396

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

| Method | Line | Description |
|--------|------|-------------|
| `set_array_center(cx, cy)` | 422 | Set array centre for bearing computation |
| `update(detections, t)` | 426 | Associate detections, update/create/delete tracks |
| `get_tracks()` | 480 | Return active tracks only |
| `get_all_tracks()` | 483 | Return all tracks (including deleted) |

### `run_multi_tracker(multi_detections, times)` — L487

Run multi-target EKF tracker on multi-peak detections.

### class `EKFTracker3D` — L523

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

| Method | Line | Description |
|--------|------|-------------|
| `initialise_from_detection(bearing, range_est, z_est, center_x, center_y)` | 556 | Place initial state from a detection |
| `predict(dt)` | 596 | Constant-velocity prediction in 3-D |
| `update(bearing, range_est, amplitude, z_est, center_x, center_y)` | 665 | EKF measurement update |
| `set_process_noise_multiplier(multiplier)` | 699 | Scale process noise (from maneuver detector) |
| `get_state()` | 707 | Return 6-element state vector |
| `get_position()` | 710 | Return `(x, y, z)` |
| `get_velocity()` | 713 | Return `(vx, vy, vz)` |
| `get_covariance()` | 716 | Return 6×6 covariance matrix |
| `get_range_uncertainty(cx, cy)` | 719 | 1-σ range uncertainty |

### class `MultiTargetTracker3D` — L902

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

| Method | Line | Description |
|--------|------|-------------|
| `set_array_center(cx, cy)` | 928 | Set array centre |
| `update(detections, t)` | 932 | Associate, update, create, delete tracks |
| `get_tracks()` | 989 | Active tracks |
| `get_all_tracks()` | 992 | All tracks |

### `run_tracker_3d(detections)` — L733

Run the 3-D EKF tracker on MFP detections.

### `run_multi_tracker_3d(multi_detections, times)` — L996

Run multi-target 3-D EKF tracker on multi-peak detections.

---

## validate.py

*Automated sanity checks for the detection pipeline.*  Each check
returns `(passed: bool, message: str)`.

### `check_amplitude(traces, max_pressure=200.0)` — L25

**Check 1:** No instantaneous pressure exceeds `max_pressure` Pa.

### `check_snr(traces, filtered_traces, mic_positions, source_positions)` — L40

**Check 2:** SNR > 0 dB at the closest microphone after bandpass
filtering.

### `check_travel_times(mic_positions, sound_speed, dt)` — L71

**Check 3:** Travel-time table matches distance / $c$ within 1 sample.

### `check_localization(traces, dt, mic_positions, true_pos, sound_speed=343.0, grid_spacing=5.0)` — L109

**Check 4:** Stationary source localises to within acceptable error via
a coarse grid search.

### `check_energy(traces, dt, source_level_dB, mic_positions, source_positions)` — L158

**Check 5:** Total received energy within order of magnitude of
expected (based on `source_level_dB` and 1/r² spreading).

### `run_all_checks(traces, filtered_traces, dt, mic_positions, source_positions, source_level_dB=90.0, sound_speed=343.0)` — L194

Run all five sanity checks and print results.  Returns a list of
`(name, passed, message)` tuples.

---

## ml/acoustic_classifier.py

*Acoustic source classifier using a simple CNN on mel spectrograms.*
Requires `torch`.

### class `AcousticClassifier` — L19

Small CNN for acoustic source classification.

Architecture: Conv2d(1→16) → Conv2d(16→32) → Conv2d(32→64) →
AdaptiveAvgPool → FC(64→n_classes).

| Method | Line | Description |
|--------|------|-------------|
| `__init__(n_classes=6)` | 22 | Build the network |
| `forward(x)` | 33 | Forward pass: mel spectrogram → class logits |
| `get_embedding(x)` | 53 | Get the 64-dim acoustic embedding (before FC layer) |

Source classes (default 6): quadcopter, hexacopter, fixed_wing, bird,
ground_vehicle, unknown.

---

## ml/data_generation.py

*Training data generation for source classification and maneuver
detection.*

### `generate_source_signal(class_name, n_steps, dt, rng)` — L160

Generate a source signal for the given class.  Maps class names to
signal generators with class-specific parameters (e.g. quadcopter →
4 blades at 5400 RPM, fixed_wing → propeller at 2400 RPM).

### `generate_classification_dataset(n_samples_per_class=200, dt, window_duration=0.5, sound_speed=343.0, seed=42)` — L233

Generate a full classification training dataset.  For each sample:
generates a signal, propagates to a random microphone position, and adds
noise.  Returns `(signals, labels, class_names)`.

### `generate_maneuver_dataset(n_samples_per_class=400, window_size=20, dt_tracker=0.1, seed=42)` — L331

Generate labelled track segments for maneuver detection.  Simulates
kinematic trajectories for each maneuver class and extracts 6-channel
windows `(x, y, z, vx, vy, vz)` of length `window_size`.

Maneuver classes: steady, turning, accelerating, diving, evasive,
hovering.

---

## ml/features.py

*Feature extraction for ML classification.*

### `compute_mel_spectrogram(signal, sample_rate, n_fft=512, hop_length=128, n_mels=64, f_min=20.0, f_max=None)` — L15

Compute log-mel spectrogram of a 1-D signal.  Returns a 2-D numpy array
`(n_mels, n_frames)`.  Uses `scipy.signal.stft` + custom mel filterbank
(no librosa dependency).

### `compute_kinematic_features(positions, velocities, dt=0.1)` — L111

Compute 14-dimensional kinematic feature vector from tracker output.
**Pure numpy — no torch dependency.**

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `mean_speed` | Average speed over window |
| 1 | `std_speed` | Speed variability |
| 2 | `max_speed` | Peak speed |
| 3 | `mean_accel` | Average acceleration magnitude |
| 4 | `max_accel` | Peak acceleration |
| 5 | `mean_heading_rate` | Average heading change rate (rad/s) |
| 6 | `max_heading_rate` | Peak heading change rate |
| 7 | `curvature` | Path curvature (1/m) |
| 8 | `hover_fraction` | Fraction of time nearly stationary |
| 9 | `altitude_mean` | Mean z-coordinate (3-D only) |
| 10 | `altitude_std` | Altitude variability |
| 11 | `climb_rate_mean` | Mean vertical speed |
| 12 | `climb_rate_max` | Peak vertical speed |
| 13 | `range_rate` | Range change rate from origin |

---

## ml/fusion_classifier.py

*Kinematic fusion classifier — two-branch network.*  Requires `torch`.

### class `KinematicBranch` — L14

MLP for kinematic features: Linear(14→64) → ReLU → Linear(64→32) →
ReLU.

### class `FusionClassifier` — L28

Two-branch acoustic + kinematic fusion classifier.  Acoustic branch
reuses `AcousticClassifier` backbone (64-dim embedding); kinematic
branch is `KinematicBranch` (32-dim).  Concatenated 96-dim vector →
FC(96→64) → FC(64→n_classes).

| Method | Line | Description |
|--------|------|-------------|
| `__init__(n_classes=6, n_kinematic_features=14)` | 31 | Build both branches + fusion head |
| `forward(mel_spec, kinematic)` | 47 | Forward pass: dual input → class logits |
| `load_acoustic_weights(acoustic_model)` | 77 | Initialize acoustic branch from a trained `AcousticClassifier` |

### class `KinematicOnlyClassifier` — L87

Kinematic-only classifier (baseline for comparison).
Linear(14→128) → ReLU → Linear(128→64) → ReLU → Linear(64→n_classes).

---

## ml/maneuver_classifier.py

*Maneuver detection classifier using a 1-D CNN.*  Requires `torch`.

### class `ManeuverClassifier` — L17

1-D CNN for maneuver state classification.

Architecture: Conv1d(6→32) → Conv1d(32→64) → AdaptiveAvgPool →
FC(64→n_classes).

Input: `(batch, 6, window_size)` — 6 channels are `(x, y, z, vx, vy, vz)`.

| Method | Line | Description |
|--------|------|-------------|
| `__init__(n_classes=6)` | 20 | Build the network |
| `forward(x)` | 26 | Forward pass: kinematic window → maneuver logits |

Maneuver classes: steady, turning, accelerating, diving, evasive,
hovering.

---

## ml/training.py

*Training loops for all ML classifiers.*  Requires `torch`.

### `prepare_acoustic_data(signals, labels, sample_rate, n_fft=512, hop_length=128, n_mels=64)` — L20

Convert raw signals to mel spectrogram tensors.  Returns
`(X_tensor, y_tensor)` ready for DataLoader.

### `train_classifier(model, X_train, y_train, X_val, y_val, n_epochs=50, lr=0.001, batch_size=32, verbose=True)` — L47

Generic training loop for single-input classifiers (`AcousticClassifier`
or `ManeuverClassifier`).  Uses Adam optimiser + cross-entropy loss.
Returns `(train_losses, val_losses)`.

### `train_fusion_classifier(model, X_acoustic_train, X_kinematic_train, y_train, X_acoustic_val, X_kinematic_val, y_val, n_epochs=50, lr=0.0005, batch_size=32, verbose=True)` — L109

Training loop for the two-input `FusionClassifier`.

### `evaluate_classifier(model, X_test, y_test, class_names)` — L173

Evaluate a single-input classifier and return metrics.  Computes
per-class precision, recall, F1, and overall accuracy.

### `evaluate_fusion_classifier(model, X_acoustic, X_kinematic, y_test, class_names)` — L221

Evaluate the fusion classifier.  Same metrics as `evaluate_classifier`.

---

## detection/bearing.py

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

### class `SRPBeamformer(BearingEstimator)`

Steered Response Power with Phase Transform.

| Method | Description |
|--------|-------------|
| `__init__(mic_positions, fs, window_samples, c=343.0, n_bearings=360, freq_lo=100.0, freq_hi=2000.0, min_peak_sep_deg=15.0, secondary_threshold=0.3)` | Pre-compute steering matrix |
| `estimate(segment, max_sources=1)` | SRP-PHAT → `BearingResult` |

### class `MUSICEstimator(BearingEstimator)`

MUSIC subspace DOA with MDL source-count estimation.

| Method | Description |
|--------|-------------|
| `__init__(mic_positions, fs, window_samples, c=343.0, n_bearings=360, freq_lo=100.0, freq_hi=2000.0, n_subbands=0, diagonal_loading=0.01, min_peak_sep_deg=10.0)` | Init |
| `estimate(segment, max_sources=1)` | MUSIC pseudo-spectrum → `BearingResult` |

### class `MVDRBeamformer(BearingEstimator)`

MVDR (Capon) beamformer on a polar grid with harmonic selection.

| Method | Description |
|--------|-------------|
| `__init__(mic_positions, dt, sound_speed=343.0, azimuth_spacing_deg=1.0, range_min=20.0, range_max=500.0, range_spacing=5.0, fundamental=150.0, n_harmonics=6, harmonic_bandwidth=10.0, ...)` | Init |
| `estimate(segment, max_sources=1)` | MVDR → `BearingResult` |

### Factory functions

- `available_bearing_methods() → list[str]` — returns `["srp_phat", "music", "mvdr"]`
- `create_bearing_estimator(method, **kwargs) → BearingEstimator`

---

## detection/engine.py

*Streaming detection engine that chains bearing, range, and tracking.*

### dataclass `WindowDetection`

Per-window detection result.  Fields: `time`, `detected`, `window_rms`,
`bearings`, `bearing_rad`, `bearing_deg`, `range_m`, `x`, `y`, `z`,
`track` (`TrackState` or None), `n_sources`, `bearing_method`.

### class `DetectionEngine`

| Method | Description |
|--------|-------------|
| `__init__(mic_positions, fs, window_samples, bearing_method="srp_phat", range_method="rms", max_sources=1, min_signal_rms=5e-5, ema_alpha=0.35, source_z_estimate=0.0, c=343.0, bearing_kwargs=None, range_kwargs=None, tracker_min_detections=5, tracker_max_history=20)` | Create engine with chosen estimators |
| `calibrate_range(peak_rms, cpa_distance)` | Set RMS reference for range estimation |
| `reset()` | Clear tracker and smoother state |
| `process_window(segment, t_center) → WindowDetection` | Process one window through the full chain |

---

## detection/ranging.py

*Pluggable range estimation algorithms.*

### dataclass `RangeEstimate`

Single range estimate: `range_m`, `uncertainty_m`.

### class `RangeEstimator` (ABC)

Abstract base class.  Subclasses must implement
`estimate(segment, bearing_rad) → RangeEstimate`.

### class `RMSRangeEstimator(RangeEstimator)`

Inverse-square-law range from RMS amplitude.

| Method | Description |
|--------|-------------|
| `__init__(ref_range=10.0, ref_rms=None, range_min=5.0, range_max=100.0)` | Init |
| `calibrate(peak_rms, cpa_distance)` | Set reference from CPA |
| `estimate(segment, bearing_rad=None)` | RMS → range |

### class `TDOARangeEstimator(RangeEstimator)`

GCC-PHAT TDOA multilateration with bearing-constrained grid search.

| Method | Description |
|--------|-------------|
| `__init__(mic_positions, fs, window_samples, c=343.0, range_min=2.0, range_max=200.0, n_range_bins=80, freq_lo=100.0, freq_hi=2000.0)` | Init |
| `estimate(segment, bearing_rad)` | TDOA → range |

### class `BearingRateRangeEstimator(RangeEstimator)`

Kinematic range from angular rate of change: $R \approx v/|\dot{\theta}|$.

| Method | Description |
|--------|-------------|
| `__init__(source_speed=50.0, hop_sec=0.025, ema_alpha=0.30, range_min=2.0, range_max=200.0, min_rate_dps=5.0)` | Init |
| `reset()` | Clear state |
| `estimate(segment, bearing_rad)` | Bearing rate → range |

### Factory functions

- `available_range_methods() → list[str]` — returns `["rms", "tdoa", "bearing_rate", "nearfield"]`
- `create_range_estimator(method, **kwargs) → RangeEstimator`

---

## detection/tracking.py

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

### class `SpectralConv2d(nn.Module)`

Truncated Fourier-space convolution layer.  Retains the lowest
`modes1 × modes2` Fourier modes and applies a learnable complex weight.

### class `FNOBlock2d(nn.Module)`

Single FNO block: `SpectralConv2d` + pointwise `Conv2d(1×1)` +
residual skip + GELU activation.

### class `TraceDecoder(nn.Module)`

MLP that maps latent features at receiver locations to time-domain
traces.  Output: `(batch, n_recv, n_time_steps)`.

### class `AcousticFNO(nn.Module)`

Full FNO model.  Input: 4-channel field (normalised velocity model,
source $x/y$ Gaussian blobs, frequency encoding).

| Method | Description |
|--------|-------------|
| `__init__(modes1, modes2, width, n_layers, n_recv, n_time_steps)` | Build model |
| `forward(x, receiver_positions)` | Field → traces |

---

## ml/fno_data_gen.py

*FDTD training data pipeline for FNO.*  Requires `torch`.

### `generate_sample(sample_id, output_dir, ...)`

Randomise domain type, source parameters, receiver layout; run
`FDTDSolver` or `FDTD3DSolver`; save `.npz` per sample.

CLI: `python -m acoustic_sim.ml.fno_data_gen --n-samples 500 --output-dir data/fno_train`

---

## ml/fno_training.py

*FNO training loop.*  Requires `torch`.

### class `FNODataset(Dataset)`

Lazy `.npz` loader with receiver/trace padding.

### `train_fno(data_dir, epochs=200, ...)`

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
| `__init__(checkpoint_path, device="cpu")` | Load model |
| `predict(velocity_field, grid, receivers, source) → traces` | Forward pass |

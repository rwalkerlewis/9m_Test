# Architecture

## Package Organisation

All production code lives in `src/acoustic_sim/`. The package has two sides and one contract
between them: the **forward side** generates synthetic data by solving the acoustic wave equation
(FDTD, Helmholtz or an analytic propagator) in air or in water, and the **processing side**
detects, locates, tracks and classifies unknown objects in data it cannot tell from a recording.
`contract.py` is the boundary. Air and water are two forward models feeding one
detect–locate–track–classify chain; `ml/` contains optional PyTorch models, and `harbor/` and a few
root modules are compatibility import paths only (see
[Compatibility import paths](#compatibility-import-paths)).

## Sensing contract

### The principle

The **forward side** knows everything: the scenario, every track, every source waveform, the grid,
the time step, the gains, the dead channels.

The **processing side** consumes either real sensor data or forward model data treated as if it
were real, and it cannot tell the difference. It knows only what a deployed system would know: the
recorded traces, the sample rate, the sensor geometry, publicly available environment information
(land, bathymetry, a nominal sound speed; in air the wind and terrain a met report and a map give)
and, in water, the AIS reports.

Anything the processing side reads that is not in that list is a **diagnostic**. Diagnostics are
allowed (they are how processing gain is measured), but every result they touch says so in the
data (its [provenance](#provenance)), not only in the prose.

### Two data classes, two files

`acoustic_sim/contract.py` (domain neutral; it imports no forward side module):

* `Observation`: `traces` (n_receivers, n_samples), `fs`, `geometry` (anything with `positions()`
  and `n_receivers`: the water `SensorLayout` with its cables, DAS mode, gauge length and
  hydrophones, or the air `MicArray`), `environment` (public knowledge only: `ChartEnvironment` in
  water, a land / water raster, depth and a nominal sound speed; `AirEnvironment` in air, nominal
  sound speed, wind, terrain type, area), `reports` (AIS reports in water, `None` in air), `t0`, and
  `meta` (free text about the recording: instrument, units, band; never scenario parameters).
* `Truth`: `tracks` (`Track`, also a contract type), `components` (direct and scattered field per
  vessel, noise), `gain`, `dead`, `clean`, `scenario` (the full scenario config including the solver
  config), `fdtd` (the solver config used, with the derived `dt`) and `meta`.

`save` / `load` write each as one `.npz` with a JSON header: `<name>.observation.npz` and
`<name>.truth.npz`. The observation header holds no solver configuration, grid spacing, time step,
scenario, track, source level or component (`tests/test_contract.py` checks the keys).
`HarborDataset` stays the forward side convenience object: `.observation` and `.truth` build the
two objects, `HarborDataset.save` (and `assemble_dataset(..., out=...)`, `simulate_scenario(...,
out=...)`) write the two files, and `HarborDataset.load` reads the pair as well as an old single
`dataset.npz`, so existing caches and outputs stay usable.
`python -m acoustic_sim.contract split <dataset.npz>` writes the pair from an old file.

In air, `examples/air/02_pipeline/run_pipeline.py` splits a simulation directory (`traces.npy` +
`metadata.json`) the same way: the observation gets the traces, the sample rate, the microphone
positions, the nominal sound speed and the wind; the source trajectory and the solver parameters
are truth.

### Import rule

No processing side module imports a forward side module, at module level or inside a function.
`tests/test_contract.py` walks both lists with `ast`, resolves relative imports and names
re-exported from a package `__init__`, and fails with the file and line of every violation.

| Side | Modules |
|------|---------|
| Processing | `water/stream.py`, `water/streaming.py`, `water/characteristics.py`, `water/detection.py`, `water/suppression.py`, `water/occupancy.py`, `water/bistatic.py`, `water/motion.py`, `water/localiser.py`, `water/classifier.py`, `water/propagation.py`, `water/sensors.py`, `water/ais.py`; the scoring modules `water/evaluate.py` and `water/scoreboard.py` (they read the `Truth`; no other processing module imports them); `air/detection/*`, `air/processor.py`, `air/tracker.py`, `air/fire_control.py`; the `ml/` inference paths (`features.py`, `acoustic_classifier.py`, `fusion_classifier.py`, `maneuver_classifier.py`, `anomaly_detector.py`, `anomaly_integration.py`) |
| Forward | `solver/*`, `model.py`, `sources.py`; `water/fdtd_runner.py`, `water/fdtd_cuda_bodies.py`, `water/scatterer.py`, `water/dispersion.py`, `water/signatures.py`, `water/vessels.py`, `water/noise.py`, `water/dataset.py`, `water/scenario.py`, `water/environment.py`, `water/ais_sim.py`, `water/sensor_sampling.py`; `air/forward.py`, `air/domains.py`, `air/noise.py`, `air/setup.py`; `ml/data_generation.py`, `ml/fno*` |
| Either | `water/diagnostics.py`, `air/diagnostics.py` (diagnostics may import both sides; library code on the processing side never imports them) |

Shared modules on neither list (`contract.py`, `water/device.py`, `water/plotting.py`, `backend.py`)
import no forward module either.  The AIS reports and their predictor (`water/ais.py`) are processing
side and the synthetic report generator is forward side (`water/ais_sim.py`); the sensor layout, the
strain gauge sample points of the instrument model and `collapse` are processing side
(`water/sensors.py`: the replicas need them) and the solver side check of the sample points and the
FDTD cache key are forward side (`water/sensor_sampling.py`).  Every module `contract._KINDS` names
(the classes an observation file is read into) is processing side; the test checks it.
The rule is static: the package `__init__` files still import the forward modules eagerly (the
root facade re-exports both sides), so loading the processing side at run time loads the forward
code too; it is never called.

A shared type moves to `contract.py` (`Track`); a forward dependency becomes an injected object
(the [propagator](#propagator)); a function that is really a diagnostic moves to
`water/diagnostics.py` or `air/diagnostics.py`.

### Propagator

Replicas are an injected object. `contract.Propagator` is the protocol:
`greens(sources_xy, receivers_xy, freqs)` returns the field `(F, P, N)` at the receivers of unit
sources, and `path(points_xy, source_xy, freqs)` the propagation factor `(F, P)` from one source to
many points (the illuminator to cell leg of the bistatic model). A propagator also states `c`,
`knowledge` (the simulator knowledge it uses, copied into every result's provenance) and whether it
is `analytic`; an analytic one exposes the wavenumber hook `wavenumbers(freqs, theta)` that the
chains' cached geometry loops use, a non analytic one is evaluated through `greens`.

| Propagator | Module | Knowledge |
|------------|--------|-----------|
| `FreeFieldPropagator(c)` | `water/propagation.py` (processing side) | none: the default of `ProcessingConfig.propagator` and `BistaticConfig.propagator` (`None` means `FreeFieldPropagator(config.c)`) |
| `SimulatorAwarePropagator.from_truth(truth)` | `water/diagnostics.py` | `"dispersion"`: the FDTD scheme's numerical, direction dependent wavenumber; built from the `Truth` because it needs the solver config, so a script that uses it has to load the truth file on the inference path |
| `HelmholtzPropagator` (later stage) | — | none: a replica library from `solver/helmholtz.py` and the public environment; it implements `greens` and `path` only |

The free field default uses the config's sound speed (1495 m/s), not the chart's nominal speed
(`FreeFieldPropagator.from_environment(env)` gives that one), so that no result changes; the two
differ by about 1 m/s on the synthetic harbor.

### Provenance

Every result object the processing side returns carries a `contract.Provenance`
(`DetectionRun`, `AISAwareRun`, `BistaticRun`, and every air `WindowDetection`), and every
`results.json` written by the examples has it:

```
"provenance": {
  "truth_inputs": ["waveform", "track", "direct_removal"],   # [] when everything is estimated
  "simulator_knowledge": ["dispersion"],                      # [] when none
  "data": "observation" | "forward_model",
  "held_out": true | false
}
```

The chains set the flags they can see (`reference="true_waveform"`: `"waveform"`; `ship_positions(truth=...)`
or `Diagnostics(exact_track=...)`: `"track"`; truth components through the ECA: `"components"`; the
propagator's `knowledge`; an empirical localiser: `"empirical_replicas"`). The caller passes what only
it knows, e.g. `provenance=Provenance(truth_inputs=["direct_removal"])` when it built the traces
without the direct path, and whether the data are recorded and held out. In air:
`"range_calibration"`, `"range_method"`, `"altitude"` (the set-up steps of
`air/diagnostics.py` given from the truth) and `"hit_feedback"` (the run stops after `max_hits` hits scored against the
truth). `contract.annotate_figure` prints the flags on a figure (red when not everything was estimated); the air
pipeline plots and the bistatic, M3 and M4 figures use it.

Scoring takes the `Truth` as an explicit argument: `water/evaluate.py` (`score_detections`,
`truth_labels` for classifier labels), `water/scoreboard.py` (the [scoreboard](#scoreboard)), and
`evaluate_results` in the air pipeline. That is the one
sanctioned place truth enters after inference. The classifiers (`HarborClassifier`, the air
classifiers) infer from the observation; labels come from the truth only inside the scoring call.

**Run block.**  Every `summary.json`, `results.json` and `picture.jsonl` header of the stage 2 entry points carries a
`run` block from `contract.run_info()`: git commit and whether the tree had uncommitted changes, the processing config
file and its sha256 with the `--set` overrides, the observation file and its sha256, Python, NumPy and CuPy versions, the
GPU name or `"cpu"`, the seed, the wall time and the `Provenance` above (`held_out` true on the test recordings).
`tests/harbor/test_ci_end_to_end.py` checks that every entry point writes it.

**Documentation rule.** A table in `docs/` that contains a row given truth inputs or simulator
knowledge must contain the row with everything estimated from the traces for the same case beside
it, and the rows say which is which.

### Scoreboard

Every stage of the harbor picture (detect and locate every object in the harbor, radiating or only
reflecting, as a map over time) is scored by `water/scoreboard.py` with these definitions, fixed in
stage 1.  A chain produces maps `B` `(n_cells, n_frames)` on scan cells at frame times; the
scoreboard reads them with the `Truth` (`ChainMaps`, `set_threshold`, `score`).

* **Threshold `T`.** For each chain configuration, the identical chain runs on traces from which
  one object's contribution has been removed (its direct and scattered truth components subtracted,
  noise and everything else kept: an *object free run*).  The statistic of a frame is its largest
  local maximum outside the exclusion zones around the reporting ships and more than 10 m from
  every object still in the scene; `T` is the smallest value that at most 5 % of the frames of all
  object free runs exceed.  `T` is set on the tuning set only and then frozen.
* **Detection** of object *i* in frame *k*: a local maximum (no larger cell in its 3 x 3
  neighbourhood) above `T` within 5 m of the object's true position in that frame (static objects:
  their position).  Each peak is assigned to its nearest true object, so one peak detects at most
  one object (pilings 5 m apart).
* **Pd** per object: the fraction of the frames in which the object is active that detect it,
  pooled over the runs of a data split.
* **False alarms**: local maxima above `T` more than 10 m from every true object and outside the
  ship exclusion zones, as a rate per frame.
* **Localisation error**: median and 90th percentile distance from the detecting peak to the true
  position, over detected frames only.
* **Full record map**: the sum of a run's frame maps.  Its threshold is the largest local maximum of
  the object free runs' full record maps away from the objects still in the scene (no false peak on
  the tuning set).  Static objects are also scored on it (the fraction of runs that detect them).
* **Separation**: for the row of pilings, the number of pilings detected as distinct peaks on the
  full record map (nearest object assignment, 5 m).

**Data split.** The noise seed 0 mix of a dataset is the tuning set; seeds 1 to 5, mixed from the
cached FDTD gathers with `assemble_dataset(seed=...)`, are the test set.  Every parameter (threshold,
window length, grid spacing, exclusion radius, track smoothing) is chosen on the tuning set and the
test set is run once with those values; `Provenance.held_out` says which (every figure prints
"tuning set" or "test set").  A combined body gather (`render_vessel_gathers(combined_bodies=True)`)
cannot remove one body: the object free run of the echo chain then removes every body at once.

**Stage 2 scoring (the headline numbers from stage 2 on).**  Fixed before the test recordings existed, replacing the
definitions above where they differ (`scoreboard.score_picture`, read from the picture records):

* a detection within 5 m of an object's true position is a hit;
* a confirmed track (confirmed or static at least once) is matched to the object nearest over its fixes (median distance)
  if within 10 m (20 m on the 5 m harbor), otherwise it is a false track;
* the false alarm rate is the number of confirmed false tracks per minute of recording;
* Pd per object (and per kind of track: radiator, echo, either) is the fraction of frames, from the first frame the object
  could have been confirmed, in which a confirmed track matched to it is within 5 m.  The first frame it could have been
  confirmed: the first frame in which the chain feeding that tracker (passive for radiator tracks, echo for echo tracks)
  has an output and the object is active, plus `confirm - 1` frames of that chain.

The object free threshold above stays as a diagnostic: on the full scene it does not hold its 5 %
([history](harbor_history.md#findings-before-stage-2)).  **Data split from stage 2 on:** the noise seed 0 recordings are
the tuning set (every parameter of the processing configs was set looking at them), and the test set is noise seeds 1 to
3 of each (mixed from the cached gathers) and `multistatic_b`, a geometry no parameter has seen.

### Real data entry point

`Observation.from_arrays(traces, fs, geometry, environment, reports=None, t0=0.0, meta=...)` builds an
observation from plain arrays; saved with `Observation.save`, the whole system runs on it with
`python -m acoustic_sim.water.stream --observation <file> --processing <config> --out <dir>`; no scenario is involved (`tests/test_contract.py` runs
`run_bistatic` and `run_ais_aware` to completion on one with the free field propagator). A field
recording is converted into:

| Item | Form |
|------|------|
| `traces` | float array `(n_receivers, n_samples)`, rows in geometry order: every cable's channels in order from its first waypoint, then the hydrophones; dead channels stay in place as zeros (channel QC gives silent channels zero weight). Hydrophones in Pa. DAS channels as the chain models them: the second difference of pressure over the gauge length, `p(s + g/2) - 2 p(s) + p(s - g/2)`, which to leading order is `(g²/4) ∂²p/∂s²` and so, for axial strain `ε` (momentum: `ρ ∂²ε/∂t² = -∂²p/∂s²`), `-(ρ g²/4) ∂²ε/∂t²` (for strain rate `ε̇`, `-(ρ g²/4) ∂ε̇/∂t`), `ρ` the water density; a pressure mode cable is in Pa. |
| `fs` | sample rate [Hz] of every channel (resample before, one rate for all) |
| `t0` | time [s] of the first sample in the AIS time base |
| `geometry` | `water.sensors.SensorLayout`: per cable the waypoints `(x, y)` [m] in the chart's local east / north frame, `channel_spacing` [m], `das_mode` (`"strain"` or `"pressure"`) and `gauge_length` [m]; hydrophone positions `(x, y)` [m]. In air, `contract.MicArray` positions `(n, 2 or 3)` [m]. |
| `environment` | `contract.ChartEnvironment`: `water` boolean `(ny, nx)` on the regular coordinates `x` `(nx,)`, `y` `(ny,)` [m], `depth` [m], `c_nominal` [m/s]; in air `contract.AirEnvironment` (`c_nominal`, `wind` `(vx, vy, vz)` [m/s], `terrain`, `extent`) |
| `reports` | `water.ais.AISReports`: `vessel` (id per report, e.g. the MMSI as a string), `t` [s] in the time base of `t0`, `xy` [m] in the chart frame, `sog` [m/s] (knots × 0.5144), `cog` [rad] in the maths convention `atan2(vy, vx)` (from a nautical course in degrees: `radians(90 - cog)`) |
| `meta` | free text: `"instrument"`, `"units"` (the processing band is a choice of the processing, `BistaticConfig` / `ProcessingConfig` `f_lo`, `f_hi`, not part of the recording) |

No format specific reader (HDF5, SEG-Y, NMEA) exists yet.

## Module Inventory

`contract.py` comes first: `Observation`, `Truth`, `Track`, `Provenance`, `ChartEnvironment`,
`AirEnvironment`, `MicArray`, the `Geometry` and `Propagator` protocols, `annotate_figure`, and the
`split` command line.

### Forward side

#### Shared simulation core

| Module | Responsibility |
|--------|---------------|
| `model.py` | `VelocityModel` (2-D) and `VelocityModel3D` dataclasses.  Factory functions: `create_uniform_model`, `create_layered_model`, `create_gradient_model`, `create_checkerboard_model`, `create_valley_model`, `create_uniform_model_3d`, `create_layered_z_model_3d`, `model_3d_from_array`.  Anomaly injection: `add_circle_anomaly`, `add_rectangle_anomaly`. |
| `solver/domain_meta.py` | Shared `DomainMeta` / `DomainMeta3D` passed into the FDTD solvers. |
| `sources.py` | Signal generators: `make_wavelet_ricker`, `make_source_tone`, `make_source_noise`, `make_source_propeller`, `make_drone_harmonics`, `make_source_from_file`, `make_stationary_tonal`.  Trajectory classes (2-D): `StaticSource`, `MovingSource`, `CircularOrbitSource`, `FigureEightSource`, `LoiterApproachSource`, `EvasiveSource`, `CustomTrajectorySource`.  3-D variants add altitude: `StaticSource3D`, `MovingSource3D`, `CircularOrbitSource3D`, `ErraticQuadcopterSource3D` (Ornstein–Uhlenbeck mean-reverting random walk, bounded), etc.  All expose `position_at(step, dt)` and carry a signal array. |
| `receivers.py` | Array geometry factories.  2-D: `create_receiver_circle`, `create_receiver_line`, `create_receiver_concentric`, `create_receiver_nested_circular`, `create_receiver_l_shaped`, `create_receiver_log_spiral`, `create_receiver_random_disk`, `create_receiver_random`, `create_receiver_custom`.  3-D (adding a z-coordinate): `create_receiver_circle_3d`, `create_receiver_line_3d`, `create_receiver_nested_circular_3d`, `create_receiver_l_shaped_3d`, `create_receiver_random_disk_3d`, `create_receiver_custom_3d`.  `print_array_diagnostics` reports aperture, element spacing, and spatial Nyquist. |
| `solver/fdtd.py` | Shared `FDTDConfig`, `FDTDSolver` (2-D) and `FDTD3DSolver` (3-D); stencil helpers `fd2_coefficients`, `fd2_cfl_factor`, `fd1_coefficients`. `FDTDConfig` selects `precision` (`float64`/`float32`), `kernel` (`reference`/`fused`/`auto`), `boundary` (`sponge`/`cpml`) and `time_scheme` (`leapfrog`, prototypes `lw4`/`kspace`). Both air and water forward models use these classes; water adds environment and scatterer behavior in `water/fdtd_runner.py`. |
| `solver/fdtd_cuda.py` | Fused CUDA step kernel (one `cupy.RawKernel` launch per step) used by `kernel="fused"`/`"auto"` on a single GPU process. |
| `solver/snapshots.py` | `save_snapshot` / `save_snapshot_3d`: dB SPL wavefield PNGs written during `run(snapshot_dir=...)`; re-exported by `air/plotting.py`. |
| `solver/helmholtz.py` | `solve_helmholtz(model, source_xy, frequency_hz, damping_width)` — sparse direct solve of the 2-D Helmholtz equation with PML. |
| `backend.py` | `get_backend(use_cuda)` → returns `(xp, is_cuda)` where `xp` is numpy or cupy (shared by both sides). |
| `sampling.py` | `check_spatial_sampling`, `check_cfl`, `suggest_dx`. |
| `io.py` | `load_json`, `save_model`, `load_model`, `model_from_json`. |

#### Air forward model (`air/`)

| Module | Responsibility |
|--------|---------------|
| `air/domains.py` | Air domain builders (2-D: isotropic, wind, hills/vegetation, echo canyon, urban echo; 3-D: isotropic, wind, ground layer, hills/vegetation). |
| `air/forward.py` | Analytical 3-D point-source forward model (`simulate_3d_traces`, `simulate_scenario_3d`) and FDTD-backed variants (`simulate_3d_traces_fdtd`, `simulate_scenario_3d_fdtd`). |
| `air/noise.py` | Post-hoc air-domain and sensor noise injection. |
| `air/setup.py` | High-level air-domain, receiver, source, and timestep builders. |

#### Water forward model (`water/`)

A plan-view 2-D harbor on the shared FDTD solver: water/land geometry, shoreline absorption,
moving vessel sources and scatterers, synthetic AIS and measurement noise. It does not model a 3-D
ocean water column.

| Module | Responsibility |
|--------|----------------|
| `water/environment.py`, `water/scenario.py` | Water/land mask, informational depth, effective sound-speed map (`HarborEnvironment.public()` is its `ChartEnvironment`), crop support, JSON scenario configuration and builders |
| `water/vessels.py`, `water/signatures.py` | Source/target tracks (`build_track`; `Track` is a contract type), acoustic signatures and injection calibration |
| `water/fdtd_runner.py`, `water/fdtd_cuda_bodies.py`, `water/scatterer.py` | Harbor-specific FDTD behavior over the shared solver, moving body masks |
| `water/dispersion.py` | `DispersionModel`: the FDTD scheme's numerical wavenumber (used by `SimulatorAwarePropagator`) |
| `water/noise.py`, `water/dataset.py` | Ambient/shipping/instrument noise, channel gains/faults, source mixture; `HarborDataset` and its observation and truth files |
| `water/ais_sim.py`, `water/sensor_sampling.py` | Synthetic AIS reports from the tracks (`AISConfig`, `generate_ais`); the check that every FDTD sample point is in water and the layout's FDTD cache key |

#### ML data generation and the FNO surrogate (`ml/`)

| Module | Purpose |
|--------|---------|
| `data_generation.py` | Synthetic data factories for all three classifiers.  Physics-based source signals (rotor blade modulation, harmonic engine, wing-beat pulses). |
| `fno.py` | `AcousticFNO(nn.Module)` — Fourier Neural Operator surrogate for FDTD.  Input: 4-channel field (normalised velocity, source x/y Gaussian blobs, frequency encoding).  `SpectralConv2d` retains lowest Fourier modes; `FNOBlock2d` = spectral + pointwise conv + residual + GELU; `TraceDecoder` MLP maps latent features at receiver locations → time-domain traces.  Output: `(batch, n_recv, n_time_steps)`. |
| `fno_data_gen.py` | FDTD training data pipeline.  Randomises domain type, source params, receiver layout; runs `FDTDSolver` / `FDTD3DSolver`; saves `.npz` per sample.  CLI: `python -m acoustic_sim.ml.fno_data_gen --n-samples 500 --output-dir data/fno_train`. |
| `fno_training.py` | `FNODataset` (lazy `.npz` loader with receiver/trace padding), `train_fno` (relative L2 loss, masked receivers, cosine LR schedule, gradient clipping).  CLI: `python -m acoustic_sim.ml.fno_training --data-dir data/fno_train --epochs 200`. |
| `fno_inference.py` | `FNOForwardModel` — drop-in replacement for FDTD.  Loads a trained checkpoint and provides `predict(velocity_field, grid, receivers, source) → traces`. |

### Processing side

#### Water processing (`water/`)

| Module | Responsibility |
|--------|----------------|
| `water/sensors.py`, `water/ais.py` | Polyline cable geometry, pressure or gauge-length strain DAS (the sample points and `collapse`), hydrophones and channel splitting (`SensorLayout` is the water `Geometry`); AIS reports and the predictor |
| `water/propagation.py` | `green2d` and `FreeFieldPropagator`, the default replica model (nothing beyond the chart) |
| `water/detection.py` | Spectra, channel QC, common mode, noise floor and whitening, the matched field scanner and its stationary interference projector, peak extraction (used by the system); CFAR and the batch `run_detection` (history) |
| `water/stream.py` | **The system**: `HarborPicture` runs the chain below on traces as they arrive and makes one picture record per frame (`picture.jsonl`); `load_processing` reads a processing config (`examples/configs/processing/<recording>.json`, every parameter, `--set` overrides recorded); `run` on an observation; `python -m acoustic_sim.water.stream --observation --processing --out` |
| `water/streaming.py` | The steps of the system, causal: `StreamingPassive` (passive chain; it accepts AIS reports, the system never passes them), `RadiatorTracker` (tracks from peaks; one instance on the passive peaks, a second on the echo peaks; confirmation, static and lost status), `StreamingEcho` (direct paths of the strong radiator tracks removed, echoes mapped with each as illuminator), `ais_crosscheck` |
| `water/characteristics.py` | What each tracked object is, measured: position uncertainty (half maximum widths of the map peak, fit residual), velocity and heading with standard errors, AIS identity, received and source level and tonal lines of a radiator, echo strength, size indicator and extent of a body (no classifier) |
| `water/bistatic.py` | Used by the system: the data reference beam (`make_reference`), the ECA (`eca_remove`), `BistaticConfig`, the grid.  History: the batch `run_bistatic` with an AIS illuminator, the patch, adaptive, MVDR and MUSIC variants |
| `water/evaluate.py`, `water/scoreboard.py` | Scoring against the `Truth`: the [scoreboard](#scoreboard) (`score_picture`, `match_tracks`, `peak_hits`; the stage 1 threshold method as a diagnostic), `score_detections` |
| `water/diagnostics.py` | Either side: `SimulatorAwarePropagator`, `VesselPathLocaliser` (replicas from the dataset's own FDTD gathers), `true_illuminator_waveform` |
| `water/plotting.py`, `water/device.py` | Visualizations; CPU/CuPy dispatch |

#### History (water processing)

Library modules that only the scripts of `examples/harbor/90_history/` use, kept because committed figures came from them
(nothing is deleted; see [Harbor, history](harbor_history.md)):

| Module | What it was |
|--------|-------------|
| `water/suppression.py` | AIS-aware suppression: ship refinement and projection, re-whitening, the batch `run_ais_aware` (`StreamingPassive` still calls it when given AIS reports; the system gives none) |
| `water/occupancy.py` | Occupancy grid track-before-detect |
| `water/localiser.py` | Empirical replica refinement from a survey grid |
| `water/classifier.py` | The harbor CNN / CVAE classifier wrapper (`[ml]` extra) |
| `water/motion.py` | Motion compensated integration of batch bistatic maps |

For the system, its numbers and the operator output see [Harbor](harbor.md).

#### Air processing (`air/`)

There are two detection paths: the production SRP-PHAT pipeline (`air/detection/`, driven by
`examples/air/02_pipeline/run_pipeline.py`) and the older MFP/EKF library path (`air/processor.py`,
`air/tracker.py`, `air/detection_main.py`), which the parametric studies still use.
`air/detection_main.py` and `air/studies.py` orchestrate simulation and processing together and are
on neither side of the contract.

| Module | Responsibility |
|--------|---------------|
| `air/detection/__init__.py` | Re-exports all public symbols; quick-start docstring. |
| `air/detection/bearing.py` | **Bearing estimators.** `SRPBeamformer`, `MUSICEstimator`, `MVDRBeamformer`, the `BearingEstimator` ABC and estimator factory. |
| `air/detection/engine.py` | **`DetectionEngine`** — streaming window processor for bearing, range and tracking; carries the run's `Provenance` into every `WindowDetection`. |
| `air/detection/ranging.py` | **Range estimators.** `RMSRangeEstimator`, `TDOARangeEstimator`, `BearingRateRangeEstimator` and factory. |
| `air/detection/tracking.py` | **`CausalWLSTracker`** and `EMABearingSmoother` for streaming estimates. |
| `air/processor.py` | Matched-field processing in 2-D and 3-D (legacy MFP/EKF path). |
| `air/tracker.py` | EKF and multi-target tracking for the air-domain pipeline. |
| `air/fire_control.py` | Ballistic lead, engagement, and multi-target prioritization. |
| `air/diagnostics.py` | Either side: the truth trajectory for scoring and plotting, and the flagged set-up steps of the pipeline given from the truth (`true_cpa_distance`, `true_altitude`). |
| `air/detection_main.py` | Full MFP/EKF pipeline orchestration (simulation and processing). |
| `air/validate.py` | Amplitude, SNR, travel-time, localization and energy checks. |
| `air/config.py` | `DetectionConfig` for the MFP/EKF pipeline. |
| `air/studies.py` | Parametric studies (see [Studies](studies.md)). |
| `air/plotting.py` | Air-domain, gather, wavefield, beam, tracking and study visualizations. |
| `air/pipeline_plots.py` | Engagement pipeline figures (`plot_pipeline_summary`, `plot_radial_engagement`, `plot_beamformer_diagnostic`, all with the provenance flags); re-exported by `air/plotting.py`. |

#### ML classifiers (`ml/`, inference)

| Module | Purpose |
|--------|---------|
| `features.py` | `compute_mel_spectrogram` (hand-rolled STFT → mel → log), `compute_kinematic_features` (14-dim vector from positions + velocities).  Pure numpy — no torch required. |
| `acoustic_classifier.py` | `AcousticClassifier(nn.Module)` — 3-layer CNN, input `(B, 1, n_mels, n_time)`, output `(B, 6)`.  `get_embedding()` returns 64-dim vector. |
| `fusion_classifier.py` | `FusionClassifier` — acoustic CNN branch (64-dim) ‖ kinematic MLP branch (14→32→32).  Concatenated → FC(96→64→6).  `KinematicOnlyClassifier` for ablation. |
| `maneuver_classifier.py` | `ManeuverClassifier` — 2-layer 1-D CNN, input `(B, 6, N)` kinematic channels × timesteps, output `(B, 6)` maneuver classes. |
| `anomaly_detector.py` | CVAE model (convolutional variational autoencoder) for novel threat detection via reconstruction error. |
| `anomaly_integration.py` | `AnomalyDetector` wrapper: loads model + threshold, provides `process_frame()` and `process_mel_spectrogram()` returning `AnomalyResult` dataclass. |
| `training.py`, `anomaly_training.py` | Offline training (`train_classifier`, `train_fusion_classifier`, `evaluate_classifier`, the CVAE training loop and threshold calibration); `anomaly_training.py` generates its data with the forward model. |

### Compatibility import paths

Pre-split import paths keep working and return the very same module objects as the canonical
ones; new code should import from `acoustic_sim.air`, `acoustic_sim.water` and
`acoustic_sim.solver`.

| Legacy path | Canonical module |
|-------------|------------------|
| `acoustic_sim.harbor.<name>` | `acoustic_sim.water.<name>`, aliased on first import (so `acoustic_sim.harbor.classifier` loads PyTorch only when requested) |
| `acoustic_sim.{config,domains,fire_control,forward,noise,processor,plotting,setup,tracker,validate}` | `acoustic_sim.air.<same name>` |
| `acoustic_sim.detection`, `acoustic_sim.detection.{bearing,engine,ranging,tracking}` | `acoustic_sim.air.detection[...]` |
| `acoustic_sim.detection_main`, `acoustic_sim.studies` | `acoustic_sim.air.detection_main`, `acoustic_sim.air.studies` |
| `acoustic_sim.fdtd`, `acoustic_sim.fdtd_cuda` | `acoustic_sim.solver.fdtd`, `acoustic_sim.solver.fdtd_cuda` |

The root package also re-exports the main public functions and classes (`acoustic_sim.FDTDSolver`,
`acoustic_sim.run_detection`, ...) for existing scripts.

---

## Data Flow

### Forward Model (FDTD)

```
domain.example.json / CLI args
        │
        ▼
  build_domain()          → VelocityModel + DomainMeta
  build_receivers()       → mic_positions (n_mics, 2 or 3)
  build_source()          → Source object with signal + trajectory
  compute_dt()            → dt from CFL condition
        │
        ▼
  FDTDSolver.run()   or   FDTD3DSolver.run()
        │
        ▼
  output/                 → traces.npy  (n_mics × n_steps)
                            metadata.json
                            gather.png
                            snapshots/   (dB SPL PNGs every --snapshot-interval steps)
```

Metadata records source start/end coordinates, dt, sound speed,
receiver positions, and all simulation parameters.  The pipeline reads
only `traces.npy` and `metadata.json` — it never sees the velocity
model or domain internals.

### SRP-PHAT Engagement Pipeline (`run_pipeline.py`)

```
traces.npy + metadata.json
        │
        ▼
  load_simulation()       → Observation (traces, fs, mic positions, c, wind)
                            + Truth (source trajectory, solver parameters)
  air.diagnostics         → set-up given from the Truth, flagged in the
                            provenance: RMS calibration distance, auto
                            range choice, source altitude
        │
        ▼
  DetectionEngine(mic_positions, fs, win_len, ...)
        │   bearing_method: srp_phat | music | mvdr
        │   range_method:   rms | tdoa | bearing_rate (auto-selected)
        │   pre-compute steering matrix once
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  for each window:                                           │
  │    1. RMS gate (below min_signal_rms → skip)                │
  │    2. SRP-PHAT → bearing(s) (EMA-smoothed on unit circle)   │
  │    2a. [optional] Source classification:                     │
  │        beamformed audio → mel spectrogram → CNN             │
  │        → 6-class prediction → hostile/benign gate           │
  │        if P(non-drone) > threshold → CLASS_REJECT           │
  │    3. Range (method fixed before the loop; auto:          │
  │       true CPA ≤ auto_cpa_threshold → TDOA, else RMS)      │
  │    4. Cartesian (r, θ) → (x, y, z)                         │
  │    5. CausalWLSTracker.add_detection() → WLS fit            │
  │    5a. [optional] Maneuver detection:                       │
  │        kinematic buffer (20 steps) → 1D CNN                 │
  │        → maneuver class → covariance multiplier             │
  │        evasive 2.5x, diving 1.5x, turning 1.3x, hover 0.5x │
  │    5b. [optional] Anomaly detection:                        │
  │        mel spectrogram → CVAE → reconstruction error        │
  │        if novel + CNN uncertain → NOVEL_THREAT override     │
  │    6. Track blending:                                       │
  │       blend = min(range / 20, 1)                            │
  │       pos = blend × fitted + (1 − blend) × raw             │
  │    7. compute_bearing_rate() → dθ/dt                        │
  │    8. compute_lead_3d() → aim azimuth, elevation, TOF       │
  │    9. compute_engagement_3d() → can_fire?                   │
  │   10. if can_fire:                                          │
  │         projectile_path() → pellet trajectory               │
  │         find_cpa() → closest point of approach              │
  │         effective_threshold = max(pattern_r, hit_threshold)  │
  │         hit if miss_dist ≤ effective_threshold               │
  │         stop at max_hits                                    │
  └─────────────────────────────────────────────────────────────┘
        │
        ▼
  evaluate_results(truth)       → metrics (scoring against the Truth)
  plot_pipeline_summary()       → 6-panel PNG
  plot_radial_engagement()      → radial engagement PNG
  plot_beamformer_diagnostic()  → 4-panel beamformer PNG
  results_{2d,3d}.json          (with "provenance")
```

Key design decisions:

- **Causal window loop** — each window's bearing, range, track and fire
  decision use only data up to that window.  The set-up before the loop is
  not causal: it takes the peak RMS of the whole recording (RMS range
  calibration and the fire RMS gate) and the ground-truth distance at that
  peak (calibration and range-method choice).
- **Auto range method** — with `ranging.method = "auto"` the pipeline picks
  TDOA for the whole run if that ground-truth distance is within
  `auto_cpa_threshold_m` (3 m) and RMS otherwise.
- **Bearing-rate fire gating** — when bearing rate is large (target
  crossing), the pipeline computes a lead shot; when bearing rate is
  small (target radial), it fires a radial shot.
- **Track blending** — at close range, raw detections are trusted more
  than the WLS fit; the blend factor ramps linearly from 0 to 1 over
  0–20 m.
- **Effective hit threshold** — a hit is evaluated against
  `max(pattern_radius_at_cpa, hit_threshold)`, so the expanding shot
  pattern is always considered.

### Information Barrier

The pipeline's processing stages receive only the `Observation` ([Sensing contract](#sensing-contract)):

- `traces` — raw pressure time series at each microphone
- `fs` — sample rate
- `geometry` — microphone positions
- `environment` — nominal sound speed (and the wind and terrain type, unused so far)

They do **not** receive the velocity model, domain type, or any FDTD internal state.  The `Truth`
(source trajectory, solver parameters) is used for the evaluation metrics and hit scoring
(`find_cpa` against the true path) and, as flagged truth inputs, for the RMS range calibration,
the auto range-method choice and the source altitude before the window loop (see
[Algorithms §2.1](algorithms.md#21--rms-inverse-square-law)).  With `max_hits > 0` the loop stops
after that many scored hits, so the number of processed windows depends on the truth too
(`"hit_feedback"`).  `results_{2d,3d}.json` lists all of these under `provenance`.

---

## Execution Modes

### Single-process CPU

```bash
python examples/air/01_simulation/run_fdtd.py --domain isotropic ...
```

### Single-process GPU

```bash
python examples/air/01_simulation/run_fdtd.py --domain isotropic --use-cuda ...
```

### Multi-process MPI (CPU or GPU)

```bash
mpirun -np 4 python examples/air/01_simulation/run_fdtd.py --domain isotropic [--use-cuda] ...
```

MPI decomposes the grid along y (2-D) or z (3-D).  Ghost rows of width
`M = fd_order // 2` are exchanged every timestep.  CUDA + MPI: each
rank has its own GPU; halo exchange pulls ghost rows to host for
`MPI_Sendrecv`, then pushes back.

If `mpi4py` is not installed or the script is not launched with `mpirun`,
everything runs on rank 0 transparently.  Under MPI the solver always uses
the reference kernel (`kernel="auto"` falls back to it); `boundary="cpml"` and
the `lw4`/`kspace` time schemes raise `NotImplementedError` with more than one
process.

### Docker

```bash
./dev.sh shell                                                  # interactive shell (GPU auto-detected)
docker compose run --rm simulate --model-preset gradient --frequency 40   # Helmholtz CLI
```

The image is built on `ubuntu:24.04` with Python 3.12, OpenMPI, and CuPy (CUDA toolkit from
pip); the repository is bind-mounted.  See [Usage](usage.md#docker).

---

## 2-D vs 3-D Handling

All modules are unified.  The solver, source, receiver, tracker, and
fire-control code parameterises on dimensionality:

| Aspect | 2-D | 3-D |
|--------|-----|-----|
| Receiver shape | `(n_mics, 2)` | `(n_mics, 3)` |
| Source trajectory | `(x, y)` | `(x, y, z)` |
| FDTD solver | `FDTDSolver` (y-axis MPI) | `FDTD3DSolver` (z-slab MPI) |
| Velocity / field array | `(ny, nx)` | `(nz, ny, nx)` |
| EKF state (MFP/EKF path) | `[x, y, vx, vy]` | `[x, y, z, vx, vy, vz]` |
| Fire-control lead | bearing only | azimuth + elevation |

`run_pipeline.py` auto-detects dimensionality from the receiver position
array shape and adjusts accordingly.  2-D mic positions are promoted to
3-D by appending `z=0`.

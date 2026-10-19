# Architecture

## Package Organisation

All production code lives in `src/acoustic_sim/`. The shared simulation
core is in the package root and `solver/`; domain-specific application
code is grouped under `air/` (drone detection and engagement) and `water/`
(harbor sensing). Air and water FDTD runs use the same solver
implementation. `ml/` contains optional PyTorch models. `harbor/` and a few
root modules are compatibility import paths only (see
[Compatibility import paths](#compatibility-import-paths)).

### Module Inventory

#### Simulation core

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
| `backend.py` | `get_backend(use_cuda)` → returns `(xp, is_cuda)` where `xp` is numpy or cupy. |
| `sampling.py` | `check_spatial_sampling`, `check_cfl`, `suggest_dx`. |
| `io.py` | `load_json`, `save_model`, `load_model`, `model_from_json`. |

#### Air-domain application (`air/`)

Air-specific domain builders, detection/engagement pipelines, signal
processing, tracking, and studies live in `air/`. Shared source
trajectories, receiver geometry, velocity models, and the FDTD solver
remain in the common package or `solver/`. There are two detection paths:
the production SRP-PHAT pipeline (`air/detection/`, driven by
`examples/run_pipeline.py`) and the older MFP/EKF library path
(`air/processor.py`, `air/tracker.py`, `air/detection_main.py`), which the
parametric studies still use.

#### Detection sub-package (`air/detection/`)

The production SRP-PHAT pipeline uses modules in `air/detection/`, a
pluggable framework with abstract base classes and factory functions.

| Module | Responsibility |
|--------|---------------|
| `air/detection/__init__.py` | Re-exports all public symbols; quick-start docstring. |
| `air/detection/bearing.py` | **Bearing estimators.** `SRPBeamformer`, `MUSICEstimator`, `MVDRBeamformer`, the `BearingEstimator` ABC and estimator factory. |
| `air/detection/engine.py` | **`DetectionEngine`** — streaming window processor for bearing, range and tracking. |
| `air/detection/ranging.py` | **Range estimators.** `RMSRangeEstimator`, `TDOARangeEstimator`, `BearingRateRangeEstimator` and factory. |
| `air/detection/tracking.py` | **`CausalWLSTracker`** and `EMABearingSmoother` for streaming estimates. |

#### Air library modules

| Module | Responsibility |
|--------|---------------|
| `air/domains.py` | Air domain builders (2-D: isotropic, wind, hills/vegetation, echo canyon, urban echo; 3-D: isotropic, wind, ground layer, hills/vegetation). |
| `air/forward.py` | Analytical 3-D point-source forward model (`simulate_3d_traces`, `simulate_scenario_3d`) and FDTD-backed variants (`simulate_3d_traces_fdtd`, `simulate_scenario_3d_fdtd`). |
| `air/noise.py` | Post-hoc air-domain and sensor noise injection. |
| `air/processor.py` | Matched-field processing in 2-D and 3-D (legacy MFP/EKF path). |
| `air/tracker.py` | EKF and multi-target tracking for the air-domain pipeline. |
| `air/fire_control.py` | Ballistic lead, engagement, and multi-target prioritization. |
| `air/detection_main.py` | Full MFP/EKF pipeline orchestration. |
| `air/validate.py` | Amplitude, SNR, travel-time, localization and energy checks. |
| `air/setup.py` | High-level air-domain, receiver, source, and timestep builders. |
| `air/config.py` | `DetectionConfig` for the MFP/EKF pipeline. |
| `air/studies.py` | Parametric studies (see [Studies](studies.md)). |
| `air/plotting.py` | Air-domain, gather, wavefield, beam, tracking and study visualizations. |
| `air/pipeline_plots.py` | Engagement pipeline figures (`plot_pipeline_summary`, `plot_radial_engagement`, `plot_beamformer_diagnostic`); re-exported by `air/plotting.py`. |

#### ML classifiers (`ml/`)

| Module | Purpose |
|--------|---------|
| `features.py` | `compute_mel_spectrogram` (hand-rolled STFT → mel → log), `compute_kinematic_features` (14-dim vector from positions + velocities).  Pure numpy — no torch required. |
| `acoustic_classifier.py` | `AcousticClassifier(nn.Module)` — 3-layer CNN, input `(B, 1, n_mels, n_time)`, output `(B, 6)`.  `get_embedding()` returns 64-dim vector. |
| `fusion_classifier.py` | `FusionClassifier` — acoustic CNN branch (64-dim) ‖ kinematic MLP branch (14→32→32).  Concatenated → FC(96→64→6).  `KinematicOnlyClassifier` for ablation. |
| `maneuver_classifier.py` | `ManeuverClassifier` — 2-layer 1-D CNN, input `(B, 6, N)` kinematic channels × timesteps, output `(B, 6)` maneuver classes. |
| `data_generation.py` | Synthetic data factories for all three classifiers.  Physics-based source signals (rotor blade modulation, harmonic engine, wing-beat pulses). |
| `training.py` | `train_classifier`, `train_fusion_classifier`, `evaluate_classifier`, `evaluate_fusion_classifier`.  Adam optimiser, cross-entropy loss, confusion matrix evaluation. |
| `anomaly_detector.py` | CVAE model (convolutional variational autoencoder) for novel threat detection via reconstruction error. |
| `anomaly_integration.py` | `AnomalyDetector` wrapper: loads model + threshold, provides `process_frame()` and `process_mel_spectrogram()` returning `AnomalyResult` dataclass. |
| `anomaly_training.py` | Training script for the CVAE: synthetic data generation, training loop, threshold calibration. |

#### FNO surrogate (`ml/fno*.py`)

| Module | Purpose |
|--------|---------|
| `fno.py` | `AcousticFNO(nn.Module)` — Fourier Neural Operator surrogate for FDTD.  Input: 4-channel field (normalised velocity, source x/y Gaussian blobs, frequency encoding).  `SpectralConv2d` retains lowest Fourier modes; `FNOBlock2d` = spectral + pointwise conv + residual + GELU; `TraceDecoder` MLP maps latent features at receiver locations → time-domain traces.  Output: `(batch, n_recv, n_time_steps)`. |
| `fno_data_gen.py` | FDTD training data pipeline.  Randomises domain type, source params, receiver layout; runs `FDTDSolver` / `FDTD3DSolver`; saves `.npz` per sample.  CLI: `python -m acoustic_sim.ml.fno_data_gen --n-samples 500 --output-dir data/fno_train`. |
| `fno_training.py` | `FNODataset` (lazy `.npz` loader with receiver/trace padding), `train_fno` (relative L2 loss, masked receivers, cosine LR schedule, gradient clipping).  CLI: `python -m acoustic_sim.ml.fno_training --data-dir data/fno_train --epochs 200`. |
| `fno_inference.py` | `FNOForwardModel` — drop-in replacement for FDTD.  Loads a trained checkpoint and provides `predict(velocity_field, grid, receivers, source) → traces`. |

#### Water / harbor sensing application (`water/`)

This is a separate 2-D application built on the shared FDTD solver. It models a plan-view
harbor with water/land geometry, shoreline absorption, cable DAS and hydrophone receivers,
moving vessel sources and scatterers, synthetic AIS, measurement noise, and two distinct
processing pipelines. It is not part of `examples/run_pipeline.py` and does not model a 3-D
ocean water column.

| Module | Responsibility |
|--------|----------------|
| `water/environment.py`, `water/scenario.py` | Water/land mask, informational depth, effective sound-speed map, crop support, JSON scenario configuration and builders |
| `water/sensors.py` | Polyline cable geometry; pressure or gauge-length strain-like DAS output; hydrophones; layout validation and channel splitting |
| `water/vessels.py`, `water/signatures.py`, `water/ais.py` | Source/target tracks, acoustic signatures and injection calibration, noisy AIS reports and prediction |
| `water/fdtd_runner.py`, `water/scatterer.py` | Harbor-specific FDTD behavior over the shared solver |
| `water/noise.py`, `water/dataset.py` | Ambient/shipping/instrument noise, channel gains/faults, source mixture and labelled `HarborDataset` serialization |
| `water/detection.py`, `water/suppression.py`, `water/occupancy.py` | Passive matched-field detection, CFAR, AIS-guided position refinement and ship suppression, and track-before-detect |
| `water/bistatic.py`, `water/motion.py` | AIS-ship-illuminated echo processing, direct-path removal, and Bartlett/coherent/MVDR/MUSIC localization maps; motion-compensated integration (`motion_integrate`, re-exported by `bistatic.py`) |
| `water/dispersion.py`, `water/localiser.py` | FDTD numerical-dispersion-aware replicas and optional empirical propagation calibration |
| `water/evaluate.py`, `water/classifier.py`, `water/plotting.py`, `water/device.py` | Detection scoring, optional harbor CNN/CVAE wrapper, visualizations, and CPU/CuPy dispatch |

For implementation status, interpretation caveats, and complete scenario and example workflows,
see [Harbor](harbor.md).

#### Compatibility import paths

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
  load_simulation()       → traces, mic_positions, dt, is_3d
  compute_ground_truth()  → trajectory function (evaluation, hit scoring,
                            RMS calibration / auto range choice)
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
  evaluate_results()            → metrics
  plot_pipeline_summary()       → 6-panel PNG
  plot_radial_engagement()      → radial engagement PNG
  plot_beamformer_diagnostic()  → 4-panel beamformer PNG
  results_{2d,3d}.json
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

The pipeline knows only:

- `traces` — raw pressure time series at each microphone
- `mic_positions` — receiver coordinates
- `dt` — sample interval
- `sound_speed` — scalar from metadata

It does **not** receive the velocity model, domain type, or any FDTD
internal state.  The source trajectory (ground truth) is used for the
evaluation metrics and hit scoring (`find_cpa` against the true path),
and, as an oracle shortcut, for the RMS range calibration and the
auto range-method choice before the window loop (see
[Algorithms §2.1](algorithms.md#21--rms-inverse-square-law)).

---

## Execution Modes

### Single-process CPU

```bash
python examples/run_fdtd.py --domain isotropic ...
```

### Single-process GPU

```bash
python examples/run_fdtd.py --domain isotropic --use-cuda ...
```

### Multi-process MPI (CPU or GPU)

```bash
mpirun -np 4 python examples/run_fdtd.py --domain isotropic [--use-cuda] ...
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

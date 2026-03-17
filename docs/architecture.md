# Architecture

## Package Organisation

All production code lives in `src/acoustic_sim/`.  Each module handles
both 2-D and 3-D; there are no separate `_3d.py` files.  The only
exception is `ml/`, which contains optional PyTorch classifiers.

### Module Inventory

#### Simulation core

| Module | Responsibility |
|--------|---------------|
| `model.py` | `VelocityModel` (2-D) and `VelocityModel3D` dataclasses.  Factory functions: `create_uniform_model`, `create_layered_model`, `create_gradient_model`, `create_checkerboard_model`, `create_valley_model`, `create_uniform_model_3d`, `create_layered_z_model_3d`, `model_3d_from_array`.  Anomaly injection: `add_circle_anomaly`, `add_rectangle_anomaly`. |
| `domains.py` | `DomainMeta` / `DomainMeta3D` dataclasses (wind field, attenuation coefficients, terrain).  Builders: `create_isotropic_domain`, `create_wind_domain`, `create_hills_vegetation_domain`, `create_echo_canyon_domain`, `create_urban_echo_domain` (2-D); 3-D equivalents build `DomainMeta3D`. |
| `sources.py` | Signal generators: `make_wavelet_ricker`, `make_source_tone`, `make_source_noise`, `make_source_propeller`, `make_drone_harmonics`, `make_source_from_file`, `make_stationary_tonal`.  Trajectory classes (2-D): `StaticSource`, `MovingSource`, `CircularOrbitSource`, `FigureEightSource`, `LoiterApproachSource`, `EvasiveSource`, `CustomTrajectorySource`.  3-D variants add altitude: `StaticSource3D`, `MovingSource3D`, `CircularOrbitSource3D`, etc.  All expose `position_at(step, dt)` and carry a signal array. |
| `receivers.py` | Array geometry factories.  2-D: `create_receiver_circle`, `create_receiver_line`, `create_receiver_concentric`, `create_receiver_nested_circular`, `create_receiver_l_shaped`, `create_receiver_log_spiral`, `create_receiver_random_disk`, `create_receiver_random`, `create_receiver_custom`.  3-D: same names with `_3d` suffix, adding a z-coordinate.  `print_array_diagnostics` reports aperture, element spacing, and spatial Nyquist. |
| `fdtd.py` | `FDTDConfig` dataclass.  `FDTDSolver` (2-D) and `FDTD3DSolver` (3-D).  Leapfrog time stepping with configurable-order spatial stencils (2/4/6/8).  MPI domain decomposition: y-axis for 2-D, z-slab for 3-D.  Sponge-layer absorbing boundaries.  Bilinear (2-D) / trilinear (3-D) source injection and receiver sampling.  CuPy GPU path if `--use-cuda`.  `fd2_coefficients(order)` and `fd2_cfl_factor(order)` compute stencil weights and CFL limits. |
| `solver.py` | `solve_helmholtz(model, source_xy, frequency_hz, damping_width)` — sparse direct solve of the 2-D Helmholtz equation with PML. |
| `forward.py` | `simulate_3d_traces` — analytical 3-D point-source propagation with 1/r spreading, optional image-source ground reflection, and exponential air absorption.  `simulate_scenario_3d` wraps it with trajectory + receivers. |
| `backend.py` | `get_backend(use_cuda)` → returns `(xp, is_cuda)` where `xp` is numpy or cupy. |
| `sampling.py` | `check_spatial_sampling`, `check_cfl`, `suggest_dx`. |
| `io.py` | `load_json`, `save_model`, `load_model`, `model_from_json`. |

#### Detection / engagement (library)

| Module | Responsibility |
|--------|---------------|
| `noise.py` | Post-hoc noise injection: `generate_wind_noise` (spatially correlated, spectrally shaped), `generate_sensor_noise` (white Gaussian), `inject_sensor_faults` (elevated noise / dropout / spikes / DC offset), `inject_transient` (broadband impulse), `perturb_mic_positions`.  `add_all_noise` applies a noise recipe in one call. |
| `processor.py` | Matched field processing in 2-D and 3-D.  Polar grid construction, steering vector computation, CSDM estimation with harmonic selection, MVDR and conventional beamforming, broadband weighted sum, peak finding with parabolic interpolation.  Robustness: `compute_sensor_weights` (fault detection), `blank_transients`, `calibrate_positions` (TDOA self-calibration), `detect_stationary` (interference rejection).  Top-level: `matched_field_process(traces, mic_positions, dt, sound_speed, config)`. |
| `tracker.py` | `EKFTracker` (4-state: x, y, vx, vy), `EKFTracker3D` (6-state: x, y, z, vx, vy, vz).  Both use bearing + range + amplitude measurements.  `MultiTargetTracker` / `MultiTargetTracker3D` add nearest-neighbour data association and track initiation/deletion.  Top-level: `run_tracker`, `run_tracker_3d`, `run_multi_tracker`, `run_multi_tracker_3d`. |
| `fire_control.py` | Pellet ballistics: `time_of_flight`, `pellet_velocity_at_range`, `pattern_diameter`.  2-D: `compute_lead`, `compute_engagement`, `run_fire_control`.  3-D: `compute_lead_3d` (azimuth + elevation iterative lead), `compute_engagement_3d` (class-based rules, maneuver-adaptive thresholds, position uncertainty gating).  `prioritize_threats` ranks multiple targets. |
| `detection_main.py` | Full MFP/EKF pipeline orchestrator: `simulate_scenario`, `run_detection`, `run_detection_pipeline`, `evaluate_results`.  3-D: `run_detection_3d`, `run_detection_pipeline_3d`.  Contains ML integration scaffolding (`_classify_source`, `_detect_maneuvers`) that is not called by the SRP-PHAT pipeline. |
| `validate.py` | Sanity checks: `check_amplitude`, `check_snr`, `check_travel_times`, `check_localization`, `check_energy`, `run_all_checks`. |
| `setup.py` | High-level builders: `build_domain`, `build_receivers`, `build_source`, `compute_dt`. |
| `config.py` | `DetectionConfig` dataclass — all parameters for the MFP/EKF pipeline.  Not used by `run_pipeline.py`, which reads `pipeline.config.json` instead. |
| `studies.py` | Nine parametric studies (see [Studies](studies.md)). |
| `plotting.py` | 2-D + 3-D visualisation: domain, gather, wavefield, velocity model, beam power, vespagram, tracking, multi-track, snapshot, 3-D trajectory, altitude-vs-time, study comparison. |

#### ML classifiers (`ml/`)

| Module | Purpose |
|--------|---------|
| `features.py` | `compute_mel_spectrogram` (hand-rolled STFT → mel → log), `compute_kinematic_features` (14-dim vector from positions + velocities).  Pure numpy — no torch required. |
| `acoustic_classifier.py` | `AcousticClassifier(nn.Module)` — 3-layer CNN, input `(B, 1, n_mels, n_time)`, output `(B, 6)`.  `get_embedding()` returns 64-dim vector. |
| `fusion_classifier.py` | `FusionClassifier` — acoustic CNN branch (64-dim) ‖ kinematic MLP branch (14→32→32).  Concatenated → FC(96→64→6).  `KinematicOnlyClassifier` for ablation. |
| `maneuver_classifier.py` | `ManeuverClassifier` — 2-layer 1-D CNN, input `(B, 6, N)` kinematic channels × timesteps, output `(B, 6)` maneuver classes. |
| `data_generation.py` | Synthetic data factories for all three classifiers.  Physics-based source signals (rotor blade modulation, harmonic engine, wing-beat pulses). |
| `training.py` | `train_classifier`, `train_fusion_classifier`, `evaluate_classifier`, `evaluate_fusion_classifier`.  Adam optimiser, cross-entropy loss, confusion matrix evaluation. |

#### FNO surrogate (`ml/fno*.py`)

| Module | Purpose |
|--------|---------|
| `fno.py` | `AcousticFNO(nn.Module)` — Fourier Neural Operator surrogate for FDTD.  Input: 4-channel field (normalised velocity, source x/y Gaussian blobs, frequency encoding).  `SpectralConv2d` retains lowest Fourier modes; `FNOBlock2d` = spectral + pointwise conv + residual + GELU; `TraceDecoder` MLP maps latent features at receiver locations → time-domain traces.  Output: `(batch, n_recv, n_time_steps)`. |
| `fno_data_gen.py` | FDTD training data pipeline.  Randomises domain type, source params, receiver layout; runs `FDTDSolver` / `FDTD3DSolver`; saves `.npz` per sample.  CLI: `python -m acoustic_sim.ml.fno_data_gen --n-samples 500 --output-dir data/fno_train`. |
| `fno_training.py` | `FNODataset` (lazy `.npz` loader with receiver/trace padding), `train_fno` (relative L2 loss, masked receivers, cosine LR schedule, gradient clipping).  CLI: `python -m acoustic_sim.ml.fno_training --data-dir data/fno_train --epochs 200`. |
| `fno_inference.py` | `FNOForwardModel` — drop-in replacement for FDTD.  Loads a trained checkpoint and provides `predict(velocity_field, grid, receivers, source) → traces`. |

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
                            snapshots/   (optional field dumps)
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
  compute_ground_truth()  → trajectory function (for evaluation only)
        │
        ▼
  SRPBeamformer(mic_positions, fs, win_len, ...)
        │   pre-compute steering matrix once
        ▼
  ┌─────────────────────────────────────────────┐
  │  for each window:                           │
  │    1. RMS gate (below threshold → skip)     │
  │    2. SRP-PHAT → bearing (EMA-smoothed)     │
  │    3. RMS inverse-square → range estimate   │
  │    4. causal_ls_fit() → WLS track state     │
  │    5. compute_lead_3d() → aim angles + TOF  │
  │    6. compute_engagement_3d() → can_fire?   │
  │    7. if can_fire: score miss distance,     │
  │       count hits, stop at max_hits          │
  └─────────────────────────────────────────────┘
        │
        ▼
  evaluate_results()      → metrics
  plot_summary()          → 6-panel PNG
  plot_radial_engagement()→ radial engagement PNG
  plot_beamformer_diagnostic() → 4-panel beamformer PNG
  results_{2d,3d}.json
```

Key design decision: the pipeline is **strictly causal** — each window
sees only past data.  The WLS fit requires at least `min_detections`
points before producing a track.  Fire control runs on every window where
a valid track exists and the RMS exceeds the fire gate.

### Information Barrier

The pipeline knows only:

- `traces` — raw pressure time series at each microphone
- `mic_positions` — receiver coordinates
- `dt` — sample interval
- `sound_speed` — scalar from metadata

It does **not** receive the velocity model, domain type, source
trajectory, or any FDTD internal state.  Ground truth is used only for
evaluation metrics after the loop.

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
everything runs on rank 0 transparently.

### Docker

```bash
docker compose up dev        # interactive shell with GPU
docker compose run simulate  # batch run
```

The container is built from `nvidia/cuda:12.6.3-devel-ubuntu24.04` with
Python 3.12, OpenMPI, and CuPy pre-installed.

---

## 2-D vs 3-D Handling

All modules are unified.  The solver, source, receiver, tracker, and
fire-control code parameterises on dimensionality:

| Aspect | 2-D | 3-D |
|--------|-----|-----|
| Receiver shape | `(n_mics, 2)` | `(n_mics, 3)` |
| Source trajectory | `(x, y)` | `(x, y, z)` |
| FDTD solver | `FDTDSolver` (y-axis MPI) | `FDTD3DSolver` (z-slab MPI) |
| Grid | `(nx, ny)` | `(nx, ny, nz)` |
| EKF state | `[x, y, vx, vy]` | `[x, y, z, vx, vy, vz]` |
| Fire-control lead | bearing only | azimuth + elevation |

`run_pipeline.py` auto-detects dimensionality from the receiver position
array shape and adjusts accordingly.  2-D mic positions are promoted to
3-D by appending `z=0`.

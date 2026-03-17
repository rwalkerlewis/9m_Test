# acoustic-sim

Passive acoustic simulation and engagement system.  Two solvers produce
synthetic microphone traces; a real-time pipeline turns those traces into
fire-control solutions.

## What It Does

| Stage | Method |
|-------|--------|
| **Wave propagation** | 2-D / 3-D FDTD with MPI + CUDA, or frequency-domain Helmholtz |
| **Bearing estimation** | SRP-PHAT beamformer (pre-computed steering, 360 look directions) |
| **Range estimation** | RMS inverse-square-law calibrated at closest-point-of-approach |
| **Tracking** | Causal weighted-least-squares constant-velocity fit |
| **Fire control** | 3-D iterative ballistic lead with pellet drag and pattern spread |

An older MFP/EKF path (`detection_main.py`) still exists in the library
but is not used by the pipeline.

## Documentation

| Page | Contents |
|------|----------|
| [Architecture](architecture.md) | Package layout, module inventory, data flow |
| [API Reference](api.md) | Every public class, method, and function with signatures |
| [Physics](physics.md) | Wave equation, FD stencils, CFL, absorption, SPL |
| [Algorithms](algorithms.md) | SRP-PHAT, WLS tracker, ballistics, MFP/EKF |
| [Configuration](configuration.md) | Pipeline JSON config and DetectionConfig reference |
| [Usage](usage.md) | Installation, CLI commands, running examples |
| [Studies](studies.md) | Nine parametric robustness studies |

## Repository Layout

```
acoustic-sim/
├── src/acoustic_sim/           # Python package
│   ├── __init__.py             # public API re-exports
│   ├── __main__.py             # python -m acoustic_sim
│   ├── cli.py                  # Helmholtz CLI entry point
│   ├── config.py               # DetectionConfig dataclass
│   ├── model.py                # VelocityModel / VelocityModel3D
│   ├── sampling.py             # spatial sampling & CFL checks
│   ├── solver.py               # Helmholtz sparse solver
│   ├── backend.py              # NumPy / CuPy abstraction
│   ├── sources.py              # source signals & trajectory classes (2-D + 3-D)
│   ├── domains.py              # domain builders (2-D + 3-D)
│   ├── fdtd.py                 # FDTD solver (2-D + 3-D, MPI, CUDA)
│   ├── receivers.py            # array geometry factories (2-D + 3-D)
│   ├── forward.py              # analytical 3-D point-source model
│   ├── noise.py                # wind noise, sensor noise, faults, transients
│   ├── processor.py            # MFP beamformer (2-D + 3-D MVDR)
│   ├── tracker.py              # EKF tracker (2-D + 3-D, single + multi-target)
│   ├── fire_control.py         # ballistics & engagement (2-D + 3-D)
│   ├── detection_main.py       # MFP/EKF pipeline orchestrator
│   ├── plotting.py             # all visualisation (2-D + 3-D)
│   ├── validate.py             # automated sanity checks
│   ├── setup.py                # high-level simulation builders
│   ├── io.py                   # JSON / NPZ I/O
│   ├── studies.py              # parametric robustness studies
│   └── ml/                     # optional ML classifiers (see below)
│       ├── features.py         # mel spectrogram & kinematic features
│       ├── acoustic_classifier.py
│       ├── fusion_classifier.py
│       ├── maneuver_classifier.py
│       ├── training.py
│       └── data_generation.py
│
├── examples/
│   ├── run_fdtd.py             # 2-D FDTD forward model
│   ├── run_fdtd_3d.py          # 3-D FDTD forward model
│   ├── run_all_examples.py     # batch 18 FDTD combinations
│   ├── run_pipeline.py         # SRP-PHAT engagement pipeline
│   ├── pipeline.config.json    # pipeline configuration
│   ├── extract_array_traces.py # post-process field plane to traces
│   ├── run_valley.sh           # 2-D valley demo (FDTD → pipeline)
│   ├── run_valley_3d.sh        # 3-D valley demo
│   ├── run_wind_circular.sh    # wind + circular orbit demo
│   ├── domain.example.json     # gradient model config
│   ├── layered.example.json    # layered model config
│   └── valley.example.json     # valley model config
│
├── tests/                      # unit tests & diagnostics
├── output/                     # simulation outputs (traces, plots, JSON)
├── audio/                      # WAV source files
├── pyproject.toml
├── requirements.txt
├── Dockerfile                  # CUDA-enabled dev container
└── docker-compose.yml
```

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| numpy | arrays, linear algebra | yes |
| scipy | sparse solvers, signal processing | yes |
| matplotlib | plotting | yes |
| mpi4py | MPI parallelism for FDTD | yes (single-process fallback if absent) |
| cupy-cuda12x | GPU acceleration | optional |
| torch | ML classifiers | optional (not used by pipeline) |

## ML Module

Six files under `src/acoustic_sim/ml/` implement PyTorch-based
classifiers.  **None are wired into the pipeline.**  They require
`torch`, which is not a project dependency.

### What exists

| File | Purpose | Input → Output |
|------|---------|----------------|
| `acoustic_classifier.py` | 3-layer CNN source classifier | mel spectrogram → 6 source classes |
| `fusion_classifier.py` | two-branch CNN+MLP | mel spec + 14-dim kinematic vector → 6 source classes |
| `maneuver_classifier.py` | 1-D CNN maneuver detector | 6-channel kinematic window → 6 maneuver classes |
| `features.py` | feature extraction | traces → mel spectrogram; positions/velocities → 14-dim vector |
| `data_generation.py` | synthetic training data | generates labelled datasets for all three classifiers |
| `training.py` | train / evaluate loops | standard Adam + cross-entropy training |

Source classes: quadcopter, hexacopter, fixed\_wing, bird, ground\_vehicle, unknown.
Maneuver classes: steady, turning, accelerating, diving, evasive, hovering.

### Where they would add value

**Tracking** — The `ManeuverClassifier` detects maneuver state from
tracker output.  `detection_main.py` already has scaffold code
(`_detect_maneuvers()`) that maps maneuver labels to process-noise
multipliers (steady=1×, turning=5×, evasive=10×).  This would make the
tracker adaptive instead of constant-velocity, which is the single
largest potential tracking improvement.

**Engagement decisions** — The `FusionClassifier` combines acoustic
features with the 14-dim kinematic vector (computed from tracker
positions and velocities that already exist).  A bird or unknown label
would suppress fire; a drone label would authorise engagement.  The
confidence score gates the fire/no-fire decision.

**Standalone heuristics** — `compute_kinematic_features()` is pure
numpy (no torch dependency) and produces 14 features including hover
fraction, heading rate, curvature, and altitude statistics.  These could
be used as rule-based threat filters without any neural network.

### What would be needed

Training these classifiers requires installing `torch`, generating
synthetic datasets via `data_generation.py`, and running the training
loops in `training.py`.  No pre-trained weights exist.  The integration
scaffolding in `detection_main.py` (`_classify_source()`,
`_detect_maneuvers()`) already handles inference and fallback; wiring
them into `run_pipeline.py` would require adding a classification step
after detection and feeding the label into `compute_engagement_3d()`.

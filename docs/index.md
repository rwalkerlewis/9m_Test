# acoustic-sim

Passive acoustic simulation and engagement system.  Two forward solvers
produce synthetic microphone traces; a real-time detection pipeline turns
those traces into fire-control solutions against airborne targets.

## For the Non-Specialist

This software listens to sounds made by drones using an array of
microphones, figures out which direction (and how far away) the drone is,
tracks it as it moves, then aims a shotgun and decides when to fire.
Everything is simulated — the sound propagation through the atmosphere is
computed on a 3-D grid, and the entire detection-to-engagement chain
runs in real time on commodity hardware.

## What It Does

| Stage | Method | Module |
|-------|--------|--------|
| **Wave propagation** | 2-D / 3-D FDTD (MPI + CUDA) or Helmholtz | `fdtd`, `solver` |
| **Bearing estimation** | SRP-PHAT beamformer (primary), MUSIC, MVDR | `detection.bearing` |
| **Range estimation** | RMS inverse-square, GCC-PHAT TDOA, bearing-rate | `detection.ranging` |
| **Tracking** | Causal weighted-least-squares constant-velocity fit | `detection.tracking` |
| **Fire control** | 3-D iterative ballistic lead, CPA evaluation, pattern spread | `fire_control` |
| **Classification** | CNN / fusion / maneuver / anomaly classifiers (optional, PyTorch, pre-trained weights shipped) | `ml` |
| **FNO surrogate** | Fourier Neural Operator to replace FDTD (optional) | `ml.fno` |

An older MFP/EKF path (`detection_main.py`) remains in the library for
parametric studies but is not used by the production pipeline.

## Documentation

| Page | Contents |
|------|----------|
| [Architecture](architecture.md) | Package layout, module inventory, data flow diagrams |
| [Algorithms](algorithms.md) | SRP-PHAT, WLS tracker, ballistics, MFP/EKF, ML classifiers |
| [Physics](physics.md) | Wave equation, FD stencils, CFL, absorption, SPL, noise models |
| [Configuration](configuration.md) | Pipeline JSON config, `DetectionConfig`, CLI flags |
| [API Reference](api.md) | Every public class, method, and function with signatures |
| [Usage](usage.md) | Installation, CLI commands, running examples |
| [Studies](studies.md) | Nine parametric robustness studies |
| [Glossary](glossary.md) | Definitions of terms, acronyms, and concepts |

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
│   ├── solver.py               # Helmholtz sparse solver (2-D, PML)
│   ├── backend.py              # NumPy / CuPy abstraction
│   ├── sources.py              # source signals & trajectory classes
│   ├── domains.py              # domain builders (2-D + 3-D)
│   ├── fdtd.py                 # FDTD solver (2-D + 3-D, MPI, CUDA)
│   ├── receivers.py            # array geometry factories
│   ├── forward.py              # analytical 3-D point-source model
│   ├── noise.py                # wind noise, sensor noise, faults
│   ├── processor.py            # MFP beamformer (MVDR, polar grid)
│   ├── tracker.py              # EKF tracker (single + multi-target)
│   ├── fire_control.py         # ballistics, lead, CPA, engagement
│   ├── detection_main.py       # MFP/EKF pipeline orchestrator
│   ├── plotting.py             # all visualisation
│   ├── validate.py             # automated sanity checks
│   ├── setup.py                # high-level simulation builders
│   ├── io.py                   # JSON / NPZ I/O
│   ├── studies.py              # parametric robustness studies
│   ├── detection/              # pluggable detection framework
│   │   ├── __init__.py
│   │   ├── bearing.py          # SRPBeamformer, MUSICEstimator, MVDRBeamformer
│   │   ├── engine.py           # DetectionEngine (streaming processor)
│   │   ├── ranging.py          # RMS, TDOA, bearing-rate range estimators
│   │   └── tracking.py         # CausalWLSTracker, EMABearingSmoother
│   └── ml/                     # optional ML classifiers
│       ├── features.py         # mel spectrogram & kinematic features
│       ├── acoustic_classifier.py
│       ├── fusion_classifier.py
│       ├── maneuver_classifier.py
│       ├── anomaly_detector.py # CVAE model for novel threat detection
│       ├── anomaly_integration.py # AnomalyDetector wrapper
│       ├── anomaly_training.py # CVAE training & threshold calibration
│       ├── training.py
│       ├── data_generation.py
│       ├── fno.py              # Fourier Neural Operator surrogate
│       ├── fno_data_gen.py     # FNO training data pipeline
│       ├── fno_training.py     # FNO training loop
│       └── fno_inference.py    # FNO inference wrapper
│
├── examples/
│   ├── run_fdtd.py             # 2-D FDTD forward model
│   ├── run_fdtd_3d.py          # 3-D FDTD forward model
│   ├── run_fdtd_erratic.py     # erratic quadcopter FDTD scenario
│   ├── run_all_examples.py     # batch 18 FDTD combinations
│   ├── run_pipeline.py         # SRP-PHAT engagement pipeline (with optional ML)
│   ├── run_comparison.py       # baseline vs ML pipeline comparison
│   ├── run_ml_demo.py          # ML classification demo + confusion matrices
│   ├── train_all_ml.py         # train all ML classifiers
│   ├── generate_traces.py      # generate synthetic trace datasets
│   ├── run_fno.py              # FNO surrogate demo
│   ├── pipeline.config.json    # pipeline configuration (incl. optional ML section)
│   ├── extract_array_traces.py # post-process field plane to traces
│   ├── run_valley.sh           # 2-D valley demo (FDTD → pipeline)
│   ├── run_valley_3d.sh        # 3-D valley demo
│   └── run_wind_circular.sh    # wind + circular orbit demo
│
├── tests/                      # unit tests & diagnostics
├── output/                     # simulation outputs
│   └── models/                 # pre-trained ML checkpoints
│       ├── acoustic_classifier.pt
│       ├── maneuver_classifier.pt
│       └── fusion_classifier.pt
├── audio/                      # WAV source files
├── pyproject.toml
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| numpy | arrays, linear algebra | yes |
| scipy | sparse solvers, signal processing | yes |
| matplotlib | plotting | yes |
| mpi4py | MPI parallelism for FDTD | yes (single-process fallback) |
| cupy-cuda12x | GPU acceleration | optional |
| torch | ML classifiers, FNO surrogate | optional |

## ML Module

Nine classifier/anomaly files plus four FNO files under
`src/acoustic_sim/ml/`.  Requires `torch` (not a project dependency).
Pre-trained weights for the acoustic, maneuver, and fusion classifiers
are shipped in `output/models/`.

### Classifiers

| File | Input | Output |
|------|-------|--------|
| `acoustic_classifier.py` | mel spectrogram | 6 source classes |
| `fusion_classifier.py` | mel spectrogram + 14-dim kinematic vector | 6 source classes |
| `maneuver_classifier.py` | 6-channel kinematic window | 6 maneuver classes |
| `features.py` | traces / tracker state | mel spectrogram, 14-dim vector |
| `data_generation.py` | — | synthetic training sets |
| `training.py` | — | Adam + cross-entropy training loops |

Source classes: `quadcopter`, `hexacopter`, `fixed_wing`, `bird`,
`ground_vehicle`, `unknown`.

Maneuver classes: `steady`, `turning`, `accelerating`, `diving`,
`evasive`, `hovering`.

### FNO Surrogate

`AcousticFNO` is a Fourier Neural Operator that can replace the FDTD
solver for rapid forward propagation.  Data generation, training, and
inference modules are provided.  Not integrated into the production
pipeline.

### Integration Status

ML classifiers are integrated into the production SRP-PHAT pipeline
(`run_pipeline.py`) as optional components enabled via CLI flags
(`--enable-classification`, `--enable-maneuver`, `--enable-fusion`,
`--enable-anomaly`).  Pre-trained weights are shipped in
`output/models/`.  The MFP/EKF library path (`detection_main.py`) also
contains ML integration scaffolding via `_classify_source()` and
`_detect_maneuvers()`.

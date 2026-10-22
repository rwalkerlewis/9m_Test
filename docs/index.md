# acoustic-sim

Passive acoustic simulation and engagement system.  Two forward solvers
produce synthetic microphone traces; a real-time detection pipeline turns
those traces into fire-control solutions against airborne targets.

## For the Non-Specialist

The repository contains an airborne acoustic simulation and engagement
pipeline and a separate water-based harbor sensing research pipeline.
The airborne pipeline estimates direction and range, tracks simulated
sources, and evaluates a modeled engagement. The harbor pipeline uses a
2-D synthetic harbor, DAS cables, and hydrophones to study passive
detection and bistatic localization of hidden objects. Both are
simulations, not real-world sensing or weapon-control systems.

## What It Does

| Stage | Method | Module |
|-------|--------|--------|
| **Wave propagation** | 2-D / 3-D FDTD (MPI + CUDA) or Helmholtz | `solver.fdtd`, `solver.helmholtz` |
| **Bearing estimation** | SRP-PHAT beamformer (primary), MUSIC, MVDR | `air.detection.bearing` |
| **Range estimation** | RMS inverse-square, GCC-PHAT TDOA, bearing-rate | `air.detection.ranging` |
| **Tracking** | Causal weighted-least-squares constant-velocity fit | `air.detection.tracking` |
| **Fire control** | 3-D iterative ballistic lead, CPA evaluation, pattern spread | `air.fire_control` |
| **Classification** | CNN / fusion / maneuver / anomaly classifiers (optional, PyTorch; classifier weights shipped) | `ml` |
| **Harbor sensing** | 2-D harbor FDTD, DAS/hydrophone passive detection, bistatic localization | `water` |
| **FNO surrogate** | Fourier Neural Operator to replace FDTD (optional) | `ml.fno` |

An older MFP/EKF path (`air/detection_main.py`) remains in the library for
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
| [Harbor](harbor.md) | 2-D harbor DAS + hydrophone detection of hidden submerged objects (separate from the drone pipeline): scenario, processing chain, GPU, findings |
| [Noise](noise.md) | The harbor noise models: the default and `harbor_v2` (snapping shrimp, wind, quay machinery, fibre and interrogator noise), each component's source, and how far the numbers depend on it |
| [CI](ci.md) | What the two GitHub workflows run, and how to register a GPU runner |
| [Generated outputs](outputs.md) | What is tracked under `output/`, how to regenerate it, cleanup |
| [Glossary](glossary.md) | Definitions of terms, acronyms, and concepts |

## Repository Layout

```
acoustic-sim/
├── src/acoustic_sim/           # Python package
│   ├── __init__.py             # public API re-exports + legacy import aliases
│   ├── __main__.py             # python -m acoustic_sim
│   ├── cli.py                  # Helmholtz CLI entry point
│   ├── model.py                # VelocityModel / VelocityModel3D
│   ├── sampling.py             # spatial sampling & CFL checks
│   ├── backend.py              # NumPy / CuPy abstraction
│   ├── sources.py              # source signals & trajectory classes
│   ├── receivers.py            # array geometry factories
│   ├── io.py                   # JSON / NPZ I/O
│   ├── detection_main.py       # compatibility alias for air/detection_main.py
│   ├── studies.py              # compatibility alias for air/studies.py
│   ├── solver/                 # shared solvers
│   │   ├── fdtd.py             # FDTD solver (2-D + 3-D, MPI, CUDA)
│   │   ├── fdtd_cuda.py        # CUDA kernels for the FDTD solver
│   │   ├── snapshots.py        # wavefield snapshot PNGs (re-exported by air/plotting.py)
│   │   ├── helmholtz.py        # Helmholtz sparse solver (2-D, PML)
│   │   └── domain_meta.py      # DomainMeta / DomainMeta3D
│   ├── air/                    # airborne drone detection / engagement
│   │   ├── config.py           # DetectionConfig dataclass
│   │   ├── domains.py          # domain builders (2-D + 3-D)
│   │   ├── forward.py          # analytical 3-D point-source model
│   │   ├── noise.py            # wind noise, sensor noise, faults
│   │   ├── processor.py        # MFP beamformer (MVDR, polar grid)
│   │   ├── tracker.py          # EKF tracker (single + multi-target)
│   │   ├── fire_control.py     # ballistics, lead, CPA, engagement
│   │   ├── detection_main.py   # MFP/EKF pipeline orchestrator
│   │   ├── plotting.py         # visualisation (re-exports pipeline_plots, solver.snapshots)
│   │   ├── pipeline_plots.py   # engagement pipeline summary / radial / beamformer figures
│   │   ├── validate.py         # automated sanity checks
│   │   ├── setup.py            # high-level simulation builders
│   │   ├── studies.py          # parametric robustness studies
│   │   └── detection/          # pluggable detection framework
│   │       ├── bearing.py      # SRPBeamformer, MUSICEstimator, MVDRBeamformer
│   │       ├── engine.py       # DetectionEngine (streaming processor)
│   │       ├── ranging.py      # RMS, TDOA, bearing-rate range estimators
│   │       └── tracking.py     # CausalWLSTracker, EMABearingSmoother
│   ├── water/                  # harbor DAS + hydrophone pipeline (see harbor.md)
│   ├── harbor/                 # compatibility package aliasing water/
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
│   ├── train_all_ml.py         # train the source / maneuver / fusion classifiers
│   ├── generate_traces.py      # regenerate the drone datasets' traces (analytic model)
│   ├── run_fno.py              # FNO surrogate demo
│   ├── pipeline.config.json    # pipeline configuration (incl. optional ML section)
│   ├── extract_array_traces.py # post-process field plane to traces
│   ├── run_valley.sh           # 2-D valley demo (FDTD → pipeline)
│   ├── run_valley_3d.sh        # 3-D valley FDTD
│   ├── run_wind_circular.sh    # 2-D wind-domain FDTD (moving source, circular array)
│   ├── run_pipeline_*.sh       # pipeline on each pre-computed dataset
│   ├── harbor_*.py             # water-domain workflows (see harbor.md)
│   └── *.scenario.json, *.example.json  # harbor scenarios, velocity-model configs
│
├── scripts/                    # benchmarks, diagnostics, clean_outputs.py (see scripts/README.md)
├── tests/                      # pytest suite (tests/harbor: water domain)
├── docs/                       # this documentation
├── output/                     # simulation outputs (mostly gitignored; see outputs.md)
│   └── models/                 # pre-trained ML checkpoints (tracked)
│       ├── acoustic_classifier.pt
│       ├── maneuver_classifier.pt
│       └── fusion_classifier.pt
├── audio/                      # WAV source files
├── simulate_array.py           # thin wrapper around the Helmholtz CLI (acoustic-sim)
├── pyproject.toml
├── requirements.txt            # shim: installs the package from pyproject.toml
├── Dockerfile
├── docker-compose.yml          # dev container (no GPU needed)
├── docker-compose.gpu.yml      # adds NVIDIA GPU access
├── dev.sh                      # shell / test / run helper for the container
├── .devcontainer/              # VS Code Dev Containers (CPU default, gpu/ variant)
└── .github/workflows/tests.yml # CI: fast CPU-only test subset
```

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| numpy | arrays, linear algebra | yes |
| scipy | sparse solvers, signal processing | yes |
| matplotlib | plotting | yes |
| mpi4py | MPI parallelism for FDTD | optional, `[mpi]` (single-process without it) |
| cupy-cuda12x | GPU acceleration | optional, `[cuda]` |
| torch | ML classifiers, FNO surrogate | optional, `[ml]` |

Dependencies are declared in `pyproject.toml`; `requirements.txt` is a compatibility shim for `pip install -e .`.

## ML Module

Nine classifier/anomaly files plus four FNO files under
`src/acoustic_sim/ml/`.  Requires `torch` (the optional `[ml]` extra).
Pre-trained weights for the acoustic, maneuver, and fusion classifiers
are shipped in `output/air/models/`; the CVAE anomaly detector is trained
locally with `python src/acoustic_sim/ml/anomaly_training.py`.

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
`output/air/models/`.  The MFP/EKF library path (`air/detection_main.py`) also
contains ML integration scaffolding via `_classify_detections()` and
`_detect_maneuvers()`.

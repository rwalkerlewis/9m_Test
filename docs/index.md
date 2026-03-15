# acoustic-sim — Technical Documentation

## Project Overview

**acoustic-sim** is a comprehensive 2D/3D acoustic simulation and passive drone detection system. It provides physics-faithful wave propagation solvers in both two and three dimensions, a full signal-processing pipeline for localising, classifying, and tracking airborne acoustic sources, and a fire-control module for computing engagement solutions against detected targets.

The system is designed for three complementary use cases:

1. **General-purpose acoustic simulation** — Solve the acoustic wave equation on arbitrary heterogeneous velocity models using either a frequency-domain Helmholtz solver (2D) or time-domain Finite-Difference Time-Domain (FDTD) solvers in 2D and 3D with MPI parallelisation and optional CUDA acceleration. An analytical 3D forward model with spherical spreading and optional ground reflection is also provided.

2. **Passive acoustic drone detection, tracking, and engagement** — A three-stage pipeline that generates synthetic sensor data via FDTD (2D or 3D) or the analytical forward model, processes microphone traces using broadband Matched Field Processing (MFP) with MVDR beamforming, tracks detected targets with an Extended Kalman Filter (EKF), computes fire-control solutions for a shotgun engagement system, and evaluates performance against ground truth through a suite of nine robustness studies. The 3D pipeline extends the search grid with an elevation dimension and decomposes fire-control lead angles into azimuth and elevation components.

3. **ML-assisted source classification and maneuver detection** — An optional machine-learning subsystem provides acoustic source classification (CNN on mel spectrograms), kinematic fusion classification (two-branch CNN+MLP combining spectral and trajectory features), and maneuver detection (1D CNN on tracker state history). These classifiers feed into the 3D fire-control module to enable class-based engagement rules (e.g., suppress fire on non-threat targets such as birds) and maneuver-adaptive process noise.

### System Capabilities

| Capability | Description |
|---|---|
| **Helmholtz solver** | Frequency-domain solution of the 2D Helmholtz equation on user-defined velocity models with absorbing boundaries |
| **2D FDTD solver** | Time-domain leapfrog solver with configurable-order spatial stencils, sponge-layer ABCs, MPI domain decomposition, and CuPy GPU acceleration |
| **3D FDTD solver** | Full 3D extension of the FDTD solver with z-slab MPI decomposition, trilinear source injection and receiver sampling, and 6-face sponge-layer absorption |
| **Analytical 3D forward model** | Point-source spherical spreading (1/r) with optional ground reflection via the image-source method and exponential air absorption |
| **2D velocity models** | Uniform, layered, gradient, checkerboard, valley with randomised hill profiles; anomaly injection (circles, rectangles) |
| **3D velocity models** | Uniform, layered-z (horizontal layers defined by z-boundaries), ground-layer (air above / high-velocity material below) |
| **2D domain environments** | Isotropic, wind, hills+vegetation, echo canyon (parallel walls), urban echo (buildings) |
| **3D domain environments** | Isotropic, wind (3-component), ground-layer with impedance-contrast reflection |
| **Source models** | WAV files, synthetic propeller noise, drone harmonic series, pure tones, band-limited noise, Ricker wavelets |
| **Source classification signals** | Physics-based synthesisers for quadcopter, hexacopter, fixed-wing, bird, ground vehicle, and unknown source classes |
| **2D trajectory types** | Static, linear moving, circular orbit, figure-eight, loiter-approach, evasive random-walk, custom waypoints |
| **3D trajectory types** | All 2D types extended with altitude: static, linear, circular orbit at altitude, figure-eight with z-oscillation, loiter-approach with descent, evasive with altitude variation, custom 3D waypoints |
| **Receiver arrays** | Nested circular, circular, linear, concentric, L-shaped, log-spiral, random disk, random box, custom; all available in 2D and 3D (with configurable z-coordinate) |
| **Noise models** | Spatially correlated wind noise, sensor self-noise, stationary coherent interferers, impulsive transients |
| **2D Matched Field Processing** | Broadband MVDR beamforming on a polar grid with harmonic-selective CSDM, diagonal loading, stationary source rejection |
| **3D Matched Field Processing** | Extension of 2D MFP with a polar+z search grid, 5D steering vectors, and 3D peak finding |
| **Robustness features** | Sensor fault detection/weighting, transient blanking, cross-correlation TDOA position self-calibration |
| **2D EKF tracker** | Bearing-primary Extended Kalman Filter with amplitude-assisted range estimation; multi-target extension with nearest-neighbour data association |
| **3D EKF tracker** | 6-state [x, y, z, vx, vy, vz] Extended Kalman Filter with 3D range measurement and anisotropic initialisation; multi-target 3D tracker with 3D Euclidean gating |
| **2D fire control** | Iterative lead-angle solver, pellet ballistics with drag, engagement envelope (pattern vs. uncertainty), threat prioritisation |
| **3D fire control** | Azimuth + elevation lead angles, 3D intercept point prediction, class-based engagement rules (threat/non-threat), maneuver-adaptive engagement thresholds, 3D miss distance, 3D threat prioritisation |
| **Acoustic classifier** | Small CNN on log-mel spectrograms for 6-class source identification (quadcopter, hexacopter, fixed-wing, bird, ground vehicle, unknown) |
| **Fusion classifier** | Two-branch network combining acoustic CNN embeddings with a kinematic MLP for improved classification accuracy |
| **Maneuver classifier** | 1D CNN on tracker state history for 6-class maneuver detection (steady, turning, accelerating, diving, evasive, hovering) |
| **Training data generation** | Physics-based signal synthesisers for all source classes and maneuver types; 3D forward model for realistic propagation and noise injection |
| **Validation** | Five automated sanity checks (amplitude, SNR, travel time, localisation, energy conservation) |
| **Study framework** | Nine parametric studies covering array geometry, sensor count, faults, multi-drone, transients, haphazard placement, echo domains, position errors, and combined stress tests |

### Target Audience

This documentation is written for readers with postdoctoral-level training in physics, engineering, or a related quantitative discipline. Familiarity with partial differential equations, linear algebra, signal processing, and estimation theory is assumed. All physics and algorithms are nonetheless presented clearly and completely — no prior knowledge of this specific codebase is required.

---

## Documentation Contents

| Document | Description |
|---|---|
| **[Physics Background](physics.md)** | The acoustic and mathematical physics underlying every algorithm: wave equation (2D and 3D), Helmholtz equation, geometric spreading, SPL, wind effects, impedance contrast, ground reflection and the image-source method, drone signatures, noise models, array theory, beamforming (conventional and MVDR), matched field processing, Extended Kalman Filter theory (2D and 3D), shotgun ballistics, mel-spectrogram feature extraction, kinematic feature physics, and classification theory. All key equations included. |
| **[Algorithm Descriptions](algorithms.md)** | Step-by-step descriptions of how each physics-based algorithm is implemented: Helmholtz solver, 2D FDTD solver (including higher-order stencils, MPI decomposition, CUDA), 3D FDTD solver, analytical 3D forward model, 2D and 3D matched field processors, 2D and 3D EKF trackers, 2D and 3D fire control, noise generation, validation checks, acoustic classifier CNN, fusion classifier, maneuver classifier, and training data generation. |
| **[Code Architecture & Module Reference](architecture.md)** | Package structure, dependency graph, data flow through both 2D and 3D pipelines, and detailed per-module reference for every source file — every class, function, parameter, and return value documented with design rationale. Covers all 2D modules, 3D extension modules, and ML classification modules. |
| **[Usage Guide](usage.md)** | Installation (pip, MPI, CUDA, PyTorch, Docker), quick-start examples for both solvers, running the 2D and 3D detection pipelines, running FDTD examples, executing robustness studies, 3D pipeline usage, ML training and inference, and programmatic API usage with code samples. |
| **[Configuration Reference](configuration.md)** | Complete reference for every configurable parameter in `DetectionConfig`, `FDTDConfig`, JSON velocity-model configs, CLI arguments, 3D pipeline parameters, 3D forward model parameters, and ML training hyperparameters — with types, defaults, units, and physical meaning. |
| **[Study Methodology & Results](studies.md)** | Detailed documentation of all nine robustness studies: physical motivation, experimental design, configuration, results tables, physical interpretation of findings, and operational implications. Includes notes on extending studies to 3D. |

---

## Quick Start

### Install

```bash
pip install -e .
```

### Helmholtz solver (frequency-domain)

```bash
acoustic-sim --model-preset gradient --frequency 40
```

### FDTD solver (time-domain, MPI)

```bash
mpirun -np 4 python examples/run_fdtd.py \
    --domain isotropic --source-type static \
    --source-signal ricker --source-freq 25 \
    --array circular --total-time 0.3 \
    --output-dir output/quick_test
```

### Detection pipeline (2D)

```bash
python src/acoustic_sim/detection_main.py --output-dir output/demo
```

### Detection pipeline (3D)

```python
from acoustic_sim.sources_3d import CircularOrbitSource3D, make_drone_harmonics
from acoustic_sim.receivers_3d import create_receiver_circle_3d
from acoustic_sim.forward_3d import simulate_scenario_3d
from acoustic_sim.detection_main_3d import run_detection_3d, evaluate_results_3d

# Create source at 50 m altitude
signal = make_drone_harmonics(n_steps=40000, dt=0.00025)
source = CircularOrbitSource3D(cx=0, cy=0, radius=100, angular_velocity=0.1,
                                start_angle=0, signal=signal, altitude=50.0)
mics = create_receiver_circle_3d(cx=0, cy=0, radius=0.5, count=16, z=0.0)

scenario = simulate_scenario_3d([source], mics, dt=0.00025, n_steps=40000)
result = run_detection_3d(scenario["traces"], scenario["mic_positions"],
                          scenario["dt"], z_min=0, z_max=100, z_spacing=10)
```

### All robustness studies

```python
from acoustic_sim.studies import run_all_studies
run_all_studies()
```

For detailed instructions, see the [Usage Guide](usage.md).

---

## Repository Structure

```
acoustic-sim/
├── src/acoustic_sim/               # Python package
│   ├── __init__.py                 # Public API re-exports
│   ├── __main__.py                 # python -m acoustic_sim
│   ├── cli.py                      # Helmholtz CLI entry point
│   ├── config.py                   # DetectionConfig dataclass
│   │
│   │  ── 2D Simulation Core ──
│   ├── model.py                    # VelocityModel + creation helpers
│   ├── sampling.py                 # Spatial sampling & CFL checks
│   ├── solver.py                   # 2D Helmholtz solver
│   ├── backend.py                  # NumPy / CuPy backend abstraction
│   ├── sources.py                  # Source signals & trajectory classes
│   ├── domains.py                  # Domain builders (5 environment types)
│   ├── fdtd.py                     # 2D FDTD solver (MPI + CUDA)
│   ├── receivers.py                # Receiver array geometry factories
│   ├── io.py                       # JSON / NPZ I/O
│   ├── plotting.py                 # All 2D visualisation functions
│   │
│   │  ── 2D Detection Pipeline ──
│   ├── noise.py                    # Post-hoc noise generators
│   ├── processor.py                # 2D Matched field processor (MVDR)
│   ├── tracker.py                  # 2D EKF tracker & multi-target tracker
│   ├── fire_control.py             # 2D ballistics & engagement logic
│   ├── validate.py                 # Automated sanity checks
│   ├── setup.py                    # High-level builders
│   ├── detection_main.py           # 2D three-stage pipeline orchestrator
│   ├── studies.py                  # Parametric study framework
│   │
│   │  ── 3D Extension ──
│   ├── model_3d.py                 # VelocityModel3D + creation helpers
│   ├── domains_3d.py               # 3D domain builders (isotropic, wind, ground-layer)
│   ├── sources_3d.py               # 3D source trajectories (7 types with altitude)
│   ├── receivers_3d.py             # 3D receiver array wrappers
│   ├── forward_3d.py               # Analytical 3D forward model + FDTD bridge
│   ├── fdtd_3d.py                  # 3D FDTD solver (MPI z-slab decomposition)
│   ├── processor_3d.py             # 3D Matched field processor (polar+z grid)
│   ├── tracker_3d.py               # 3D EKF tracker (6-state) & multi-target
│   ├── fire_control_3d.py          # 3D fire control (az+el lead, class-based rules)
│   ├── plotting_3d.py              # 3D visualisation (trajectory, altitude, tracking)
│   ├── detection_main_3d.py        # 3D pipeline orchestrator with ML integration
│   │
│   │  ── ML Classification ──
│   └── ml/
│       ├── __init__.py             # ML package init
│       ├── features.py             # Mel spectrogram & kinematic feature extraction
│       ├── acoustic_classifier.py  # CNN for acoustic source classification
│       ├── fusion_classifier.py    # Two-branch acoustic+kinematic fusion classifier
│       ├── maneuver_classifier.py  # 1D CNN for maneuver detection
│       ├── training.py             # Training & evaluation loops
│       └── data_generation.py      # Physics-based training data synthesis
│
├── examples/                       # Example scripts & JSON configs
│   ├── run_fdtd.py                 # Single FDTD run with full CLI
│   ├── run_all_examples.py         # Orchestrate all 18 combinations
│   ├── run_full_pipeline.py        # End-to-end detection & targeting
│   ├── run_valley.sh               # Valley domain shell script
│   ├── run_wind_circular.sh        # Wind + circular orbit shell script
│   ├── domain.example.json         # Gradient model with anomalies
│   ├── layered.example.json        # Layered model with anomaly
│   └── valley.example.json         # Valley model configuration
├── audio/                          # WAV files for source signals
├── tests/                          # Test and debug scripts
│   ├── test_3d_extension.py        # 3D pipeline integration tests
│   ├── test_classification.py      # Acoustic classifier tests
│   ├── test_fusion.py              # Fusion classifier tests
│   ├── test_maneuver.py            # Maneuver classifier tests
│   ├── test_fdtd_3d.py             # 3D FDTD solver tests
│   ├── test_demo.py                # End-to-end demo test
│   ├── test_evaluation.py          # Evaluation metrics tests
│   ├── test_signal_threshold.py    # Signal threshold tests
│   ├── run_detection_eval.py       # Detection evaluation runner
│   ├── calibrate_rms_range.py      # RMS range calibration utility
│   ├── debug_accuracy.py           # Accuracy debugging script
│   ├── debug_fire_control.py       # Fire control debugging script
│   └── debug_tracker.py            # Tracker debugging script
├── docs/                           # This documentation
├── pyproject.toml                  # Package metadata & dependencies
├── requirements.txt                # Dependency list
├── Dockerfile                      # CUDA-enabled container
├── docker-compose.yml              # Dev & simulation services
├── simulate_array.py               # Legacy entry point
└── RESULTS.md                      # Study results summary
```

---

## Dependencies

| Package | Purpose | Required |
|---|---|---|
| `numpy` | Array operations, linear algebra | Yes |
| `scipy` | Sparse solvers, signal processing, I/O | Yes |
| `matplotlib` | All plotting and visualisation | Yes |
| `mpi4py` | MPI parallelisation for FDTD | Yes (but graceful single-process fallback) |
| `cupy-cuda12x` | GPU acceleration via CuPy | Optional |
| `torch` | ML classifiers (acoustic, fusion, maneuver) | Optional (required only for ML subsystem) |

---

## License & Citation

This software is provided as-is for research and educational purposes. If referencing this work, please cite the repository and accompanying results documentation.

# acoustic-sim — Technical Documentation

## Project Overview

**acoustic-sim** is a comprehensive 2D acoustic simulation and passive drone detection system. It provides physics-faithful wave propagation solvers, a full signal-processing pipeline for localising and tracking airborne acoustic sources, and a fire-control module for computing engagement solutions against detected targets.

The system is designed for two complementary use cases:

1. **General-purpose 2D acoustic simulation** — Solve the acoustic wave equation on arbitrary heterogeneous velocity models using either a frequency-domain Helmholtz solver or a time-domain Finite-Difference Time-Domain (FDTD) solver with MPI parallelisation and optional CUDA acceleration.

2. **Passive acoustic drone detection, tracking, and engagement** — A three-stage pipeline that generates synthetic sensor data via FDTD, processes the microphone traces using broadband Matched Field Processing (MFP) with MVDR beamforming, tracks detected targets with an Extended Kalman Filter (EKF), computes fire-control solutions for a shotgun engagement system, and evaluates performance against ground truth through a suite of nine robustness studies.

### System Capabilities

| Capability | Description |
|---|---|
| **Helmholtz solver** | Frequency-domain solution of the 2D Helmholtz equation on user-defined velocity models with absorbing boundaries |
| **FDTD solver** | Time-domain leapfrog solver with configurable-order spatial stencils, sponge-layer ABCs, MPI domain decomposition, and CuPy GPU acceleration |
| **Velocity models** | Uniform, layered, gradient, checkerboard, valley with randomised hill profiles; anomaly injection (circles, rectangles) |
| **Domain environments** | Isotropic, wind, hills+vegetation, echo canyon (parallel walls), urban echo (buildings) |
| **Source models** | WAV files, synthetic propeller noise, drone harmonic series, pure tones, band-limited noise, Ricker wavelets |
| **Trajectory types** | Static, linear moving, circular orbit, figure-eight, loiter-approach, evasive random-walk, custom waypoints |
| **Receiver arrays** | Nested circular, circular, linear, concentric, L-shaped, log-spiral, random disk, random box, custom |
| **Noise models** | Spatially correlated wind noise, sensor self-noise, stationary coherent interferers, impulsive transients |
| **Matched Field Processing** | Broadband MVDR beamforming on a polar grid with harmonic-selective CSDM, diagonal loading, stationary source rejection |
| **Robustness features** | Sensor fault detection/weighting, transient blanking, cross-correlation TDOA position self-calibration |
| **EKF tracker** | Bearing-primary Extended Kalman Filter with amplitude-assisted range estimation; multi-target extension with nearest-neighbour data association |
| **Fire control** | Iterative lead-angle solver, pellet ballistics with drag, engagement envelope (pattern vs. uncertainty), threat prioritisation |
| **Validation** | Five automated sanity checks (amplitude, SNR, travel time, localisation, energy conservation) |
| **Study framework** | Nine parametric studies covering array geometry, sensor count, faults, multi-drone, transients, haphazard placement, echo domains, position errors, and combined stress tests |

### Target Audience

This documentation is written for readers with postdoctoral-level training in physics, engineering, or a related quantitative discipline. Familiarity with partial differential equations, linear algebra, signal processing, and estimation theory is assumed. All physics and algorithms are nonetheless presented clearly and completely — no prior knowledge of this specific codebase is required.

---

## Documentation Contents

| Document | Description |
|---|---|
| **[Physics Background](physics.md)** | The acoustic and mathematical physics underlying every algorithm: wave equation, Helmholtz equation, geometric spreading, SPL, wind effects, impedance contrast, drone signatures, noise models, array theory, beamforming (conventional and MVDR), matched field processing, Extended Kalman Filter theory, and shotgun ballistics. All key equations included. |
| **[Algorithm Descriptions](algorithms.md)** | Step-by-step descriptions of how each physics-based algorithm is implemented: Helmholtz solver, FDTD solver (including higher-order stencils, MPI decomposition, CUDA), matched field processor, EKF tracker, fire control, noise generation, and validation checks. |
| **[Code Architecture & Module Reference](architecture.md)** | Package structure, dependency graph, data flow through the pipeline, and detailed per-module reference for every source file — every class, function, parameter, and return value documented with design rationale. |
| **[Usage Guide](usage.md)** | Installation (pip, MPI, CUDA, Docker), quick-start examples for both solvers, running the detection pipeline, running FDTD examples, executing robustness studies, and programmatic API usage with code samples. |
| **[Configuration Reference](configuration.md)** | Complete reference for every configurable parameter in `DetectionConfig`, `FDTDConfig`, JSON velocity-model configs, and CLI arguments — with types, defaults, units, and physical meaning. |
| **[Study Methodology & Results](studies.md)** | Detailed documentation of all nine robustness studies: physical motivation, experimental design, configuration, results tables, physical interpretation of findings, and operational implications. |

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

### Detection pipeline

```bash
python src/acoustic_sim/detection_main.py --output-dir output/demo
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
├── src/acoustic_sim/           # Python package (17 modules)
│   ├── __init__.py             # Public API re-exports
│   ├── __main__.py             # python -m acoustic_sim
│   ├── cli.py                  # Helmholtz CLI entry point
│   ├── config.py               # DetectionConfig dataclass
│   ├── model.py                # VelocityModel + creation helpers
│   ├── sampling.py             # Spatial sampling & CFL checks
│   ├── solver.py               # 2D Helmholtz solver
│   ├── backend.py              # NumPy / CuPy backend abstraction
│   ├── sources.py              # Source signals & trajectory classes
│   ├── domains.py              # Domain builders (5 environment types)
│   ├── fdtd.py                 # FDTD solver (MPI + CUDA)
│   ├── receivers.py            # Receiver array geometry factories
│   ├── io.py                   # JSON / NPZ I/O
│   ├── plotting.py             # All visualisation functions
│   ├── noise.py                # Post-hoc noise generators
│   ├── processor.py            # Matched field processor (MVDR)
│   ├── tracker.py              # EKF tracker & multi-target tracker
│   ├── fire_control.py         # Ballistics & engagement logic
│   ├── validate.py             # Automated sanity checks
│   ├── setup.py                # High-level builders
│   ├── detection_main.py       # Three-stage pipeline orchestrator
│   └── studies.py              # Parametric study framework
├── examples/                   # Example scripts & JSON configs
│   ├── run_fdtd.py             # Single FDTD run with full CLI
│   ├── run_all_examples.py     # Orchestrate all 18 combinations
│   ├── run_full_pipeline.py    # End-to-end detection & targeting
│   ├── domain.example.json     # Gradient model with anomalies
│   ├── layered.example.json    # Layered model with anomaly
│   └── valley.example.json     # Valley model configuration
├── audio/                      # WAV files for source signals
├── tests/                      # Test and debug scripts
├── docs/                       # This documentation
├── pyproject.toml              # Package metadata & dependencies
├── requirements.txt            # Dependency list
├── Dockerfile                  # CUDA-enabled container
├── docker-compose.yml          # Dev & simulation services
├── simulate_array.py           # Legacy entry point
└── RESULTS.md                  # Study results summary
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

---

## License & Citation

This software is provided as-is for research and educational purposes. If referencing this work, please cite the repository and accompanying results documentation.

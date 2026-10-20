# Acoustic Simulation, Detection, and Localization

Full technical documentation: [docs/](docs/index.md)

This repository contains two related but independent applications of acoustic modeling:

* **Airborne acoustic sensing:** 2-D/3-D wave simulation, microphone-array bearing/range
  estimation, tracking, optional machine learning, and a simulated engagement pipeline.
* **Water-based harbor sensing:** a 2-D plan-view harbor FDTD model with pressure or strain-mode
  distributed acoustic sensing (DAS), hydrophones, AIS-equipped vessels, hidden targets,
  moving-body scattering, passive detection, and bistatic localization using an AIS ship as an
  illuminator.

The harbor implementation and experimental status are documented in
[docs/harbor.md](docs/harbor.md). The passive matched-field chain runs in real time
(`water/streaming.py`, `examples/harbor_stream.py`) and finds radiating vessels without AIS, but it
does not detect the default weak UUV. The bistatic echo demo is idealized: it uses truth components and the true
illuminator waveform and is not an operational performance claim; its `demo_summary.json` and figures carry
the provenance flags. The processing side reads only an `Observation` (traces, sample rate,
geometry, public chart, AIS); everything else is `Truth` (see
[Architecture: Sensing contract](docs/architecture.md#sensing-contract)).

## Airborne Acoustic Pipeline

I chose all four directions and integrated them into a unified system.

The prompt offered signal enhancement, realistic simulation, advanced DOA, and array geometry as separate options. My first observation was that these are not independent problems. You cannot evaluate a DOA algorithm without a realistic simulation to test it against, and you cannot evaluate a simulation without a detection pipeline to consume its output. Rather than pick one piece, I built the end-to-end chain: a physics-based forward model that generates synthetic microphone traces, and a real-time detection and engagement pipeline that turns those traces into fire-control solutions.

The airborne system has two decoupled stages:

1. A 2D/3D FDTD acoustic wave propagation solver that produces synthetic pressure traces at configurable receiver locations. This replaces the clean simulation in the provided code with physics-based propagation through heterogeneous media including terrain, vegetation, and wind.

2. A real-time detection and engagement pipeline that takes the receiver traces and performs detection, bearing estimation, range estimation, tracking, and fire control.

The forward model and the pipeline are deliberately decoupled so that each can be developed, tested, and improved independently. The forward model can be made arbitrarily complex (more noise, more sources, more terrain) and the pipeline must handle whatever comes out of it.

## Approach and Design Decisions

**Forward model.** The forward model is a Finite Difference Time Domain solver with configurable stencil order (2nd, 4th, 6th, 8th). The solver supports both 2D and 3D domains with MPI domain decomposition and optional CUDA acceleration via CuPy. Sponge-layer absorbing boundaries (default) or, in the 2-D solver, a convolutional PML prevent reflections from the domain edges. A Helmholtz (frequency-domain) solver with PML boundaries is also available for steady-state analysis.

Source signals include propeller noise (blade-pass fundamental plus harmonics), drone harmonics, tones, Ricker wavelets, WAV files, and white noise. Sources are injected by bilinear (2D) or trilinear (3D) interpolation to support arbitrary trajectories. Source types include static, linear moving, circular orbit, loiter-approach, figure-eight, evasive, and erratic quadcopter.

Six domain configurations are available: isotropic (free space), wind (uniform flow with directional effects), hills and vegetation (valley between two reflective ridges with frequency-dependent absorption), echo canyon (parallel walls with strong multipath), urban echo (buildings with complex reflections), and ground layer (3-D air over a high-velocity ground half-space, producing ground reflections). The valley and echo domains introduce multipath and terrain effects that stress the detection pipeline. The FDTD scripts expose isotropic, wind and hills/vegetation (plus ground layer in 3-D); echo canyon and urban echo are library domains used by the robustness studies and the MFP/EKF pipeline.

**Detection pipeline.** The pipeline is a causal, real-time system. It processes receiver traces frame by frame in the order they would arrive.

| Stage | Algorithm |
|-------|-----------|
| Detection | RMS threshold gate |
| Classification | CNN on mel spectrograms (6-class source type, hostile/benign gate) [optional] |
| Bearing | SRP-PHAT with pre-computed steering vectors (360 azimuth bins) |
| Range | RMS inverse-square-law with CPA calibration |
| Tracking | Causal weighted-least-squares constant-velocity fit |
| Maneuver ID | 1D CNN on kinematic features (6-class maneuver state, covariance adaptation) [optional] |
| Anomaly | CVAE reconstruction error (novel threat detection) [optional] |
| Fire control | 3D iterative ballistic lead with pattern spread modeling |

All detections default to hostile. Engagement is scored by computed miss distance including projectile flight time. A target is considered neutralized after a configurable number of shots (default 3) fall within the acceptable miss distance. Computational latency is tracked because this is modeled as a real-time system.

**ML classifiers (integrated into the detection pipeline).** PyTorch classifiers for source type (CNN on mel spectrograms, 6 classes: quadcopter, hexacopter, fixed wing, bird, ground vehicle, unknown), maneuver state (1D CNN on kinematic features, 6 classes: steady, turning, accelerating, diving, evasive, hovering), a fusion classifier that combines acoustic and kinematic inputs, and a CVAE anomaly detector for novel threat flagging are wired into the production pipeline as optional components. Source classification gates engagement decisions by filtering non-drone detections before fire control. Maneuver classification adapts tracker covariance based on detected flight behavior (evasive targets get 2.5× covariance, hovering targets get 0.5×). Anomaly detection flags acoustically novel targets via reconstruction error when the primary classifier is uncertain. Trained weights for the source, maneuver and fusion classifiers are included in `output/models/`; the anomaly detector must be trained locally (see [ML training](#ml-training)). All ML components are disabled by default; with ML disabled, the pipeline produces identical results to a signal-processing-only baseline.

**FNO surrogate (included but not integrated).** A Fourier Neural Operator infrastructure to replace the FDTD solver for rapid forward propagation is included with data generation, training, and inference modules. The motivation is that the FDTD solver is too slow for real-time ensemble analysis or rapid parameter studies. An FNO trained on FDTD output could provide forward model evaluations at a fraction of the cost.

**Noise models.** The forward model supports five noise types: spatially correlated wind noise (exponential decay with spectral shaping), independent sensor self-noise (white Gaussian), sensor faults (elevated noise floor, dead channels, intermittent spikes), broadband transient events (explosions, gunshots), and microphone position errors (Gaussian survey perturbation).

**Robustness studies.** Nine parametric studies evaluate system behaviour under stress: array geometry variants, minimum sensor count, sensor faults, multi-drone scenarios, transient interference, haphazard array placement, echo-prone domains, sensor position errors, and mixed failure modes.

## Assumptions

- Single target. The production pipeline handles one source at a time. A multi-target tracker (EKF-based) exists in the library for parametric studies.
- Hostile by default. All detections default to hostile. When source classification is enabled (`--enable-classification`), non-drone detections (bird, ground vehicle) are filtered if classification confidence exceeds the threshold. Unknown sources default to hostile.
- Known array geometry. Receiver positions are assumed perfectly known (position-error noise can be injected for robustness testing).
- Synthetic sources only. The pipeline consumes FDTD output rather than raw recordings.
- Shotgun engagement model. Fire control assumes a shotgun-class effector with modeled pattern spread.

## Water-Based Harbor Pipeline

The separate water package (`src/acoustic_sim/water/`) synthesizes a harbor environment,
sensor layout, vessel tracks, noisy AIS reports, and sensor traces. Its FDTD model includes a
water mask, pressure-release land boundaries, shoreline damping, optional finite-size moving
scatterers, and pressure or strain-like DAS observations. It does **not** solve a 3-D water-column
or ocean-acoustics problem: depth is informational and only affects a small effective
sound-speed variation in the 2-D plan-view model.

Two processing paths are available:

1. **Passive target-emission detection:** whitened STFT, matched-field scan, stationary
   interference suppression, CFAR thresholding, optional AIS-aware ship cancellation, and an
   occupancy-grid tracker. See [docs/harbor.md](docs/harbor.md) and the
   [passive detection example](examples/harbor_m4_dlc.py).
2. **Bistatic echo localization:** an AIS ship illuminates a silent body; its scattered field is
   processed with Bartlett, PHAT-Bartlett, coherent matched filtering, MVDR, or MUSIC maps.
   See [bistatic echo localization](docs/harbor.md#bistatic-echo-localization) and the
   [bistatic demo](examples/harbor_bistatic_demo.py).

Quick examples:

```bash
# Build the default multi-vessel dataset; FDTD source gathers are cached.
python examples/harbor_m2_dataset.py

# Run the detect–locate–classify experiment.
python examples/harbor_m4_dlc.py

# Build a cropped, finer-grid bistatic dataset, then make localization figures.
python examples/harbor_bistatic_dataset.py --coarse
python examples/harbor_bistatic_demo.py
```

The bistatic commands are compute-intensive and use their own dataset/output paths. See
[Harbor usage and examples](docs/harbor.md#examples-and-workflows) for assumptions, outputs,
complete workflows, and other demos.

## What Worked

- The FDTD solver produces physically reasonable wavefields in both 2D and 3D, including multipath from terrain and absorption from vegetation.
- SRP-PHAT bearing estimation is robust and computationally efficient. The pre-computed steering vector approach scales well to 360 bins.
- The decoupled architecture (forward model separate from pipeline) proved valuable. Problems in one stage did not propagate to the other.
- Fire control with ballistic lead and miss distance scoring works for moving targets in 3D.
- The unified 2D/3D codebase avoids code duplication across dimensions.

## What Did Not Work (Or Needs More Work)

- Combining 2D and 3D detection algorithms into one unified framework was harder than expected. The 3D methods did not perform well on 2D data. The practical solution was to treat 2D examples as 3D with zero elevation.
- When scoring by computed miss distance (including projectile flight), simpler state estimation methods outperformed more complex ones. Adding noise and complex environments (valley, wind, erratic trajectories) has partially reversed this.
- The acoustic classifier is near-random on short pipeline windows (100 ms default) due to limited frequency resolution, so it defaults to not rejecting targets. This is the correct conservative behaviour for a C-UAS system, but classification accuracy would improve with longer observation windows or multi-window accumulation.
- Full 3D/4D wavefield storage (spatial dimensions plus time) for post-hoc receiver placement is impractical at the frequencies of interest. A field-plane extraction compromise is available (`extract_array_traces.py`) for saving a horizontal slice.

## How I Would Improve This For Real-World Deployment

**Immediate priorities:**

- Replace FDTD with trained FNO surrogates for rapid wavefield generation in ensemble and optimisation workflows.
- Improve ML classifier accuracy with longer observation windows or multi-window accumulation; current 100 ms windows limit frequency resolution.
- Multi-target support in the production pipeline (library-level multi-target EKF exists but is not wired into the SRP-PHAT path).
- Microphone gain and phase mismatch modelling.

**Exploiting FNO surrogates:**

- Ensemble forward models: cheap wavefield evaluation means no two runs need be identical. Enable stochastic inputs for noise, source path, and environmental conditions drawn from probability distributions.
- Dimensional analysis: vary all input factors to determine which parameters matter most and which can be simplified.
- Array geometry optimisation: combine FNO forward models with optimisation to find the best array configuration for a given operating environment, including minimising the number of sensors required.

**Detection and targeting pipeline:**

- Self-calibration for receiver position and gain errors.
- Integration of local ambient conditions at the sensors (detecting transient winds from large enough arrays).
- Transition from "all detections are hostile" to a classification-based rules of engagement system.

## Running the System

### Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[cuda]"   # GPU acceleration (cupy-cuda12x)
pip install -e ".[ml]"     # ML classifiers, FNO surrogate, anomaly detection (torch)
pip install -e ".[dev]"   # pytest
```

`pyproject.toml` is the single source of truth for dependencies; `requirements.txt` is only a
compatibility shim that installs the package (`-e .`) with its base dependencies.

MPI, only for multi-process FDTD (the solver runs single-process without it):

```bash
sudo apt-get install libopenmpi-dev openmpi-bin
pip install -e ".[mpi]"
```

Docker (works on any machine, GPU optional; the repository is bind-mounted, so edits need no rebuild):

```bash
./dev.sh shell        # build if needed, then open a shell in the dev container
./dev.sh test         # pytest tests/harbor inside the container
```

See [Usage](docs/usage.md#docker) for VS Code Dev Containers, GPU control and PyTorch.

For editing, building, committing, and pushing from VS Code, open the repository
root and run **Dev Containers: Reopen in Container** (or **Rebuild and Reopen in
Container** after configuration changes). The complete checkout, including
`.git`, is mounted read-write at `/workspace`, not just `src/`. See the
[development workflow](docs/usage.md#vs-code-dev-container) for build and Git setup.

### Run the FDTD forward model

```bash
# 2-D valley
bash examples/run_valley.sh

# 3-D valley
bash examples/run_valley_3d.sh

# 2-D wind domain
bash examples/run_wind_circular.sh
```

Or directly:

```bash
python examples/run_fdtd.py \
    --domain hills_vegetation \
    --source-type moving --source-signal propeller \
    --array circular --receiver-count 16 --receiver-radius 2.0 \
    --total-time 3.0 --dx 0.18 --output-dir output/valley_test \
    --use-cuda
```

### Run the engagement pipeline

Pipeline-only scripts for each pre-computed dataset:

```bash
bash examples/run_pipeline_valley.sh        # 2-D valley
bash examples/run_pipeline_valley_3d.sh     # 3-D valley
bash examples/run_pipeline_erratic.sh       # 3-D erratic quadcopter
bash examples/run_pipeline_isotropic.sh     # 2-D isotropic / wind
```

Or directly:

```bash
# Baseline (no ML)
python examples/run_pipeline.py output/valley_test

# With source classification gate
python examples/run_pipeline.py output/valley_test --enable-classification

# Full ML (classification + maneuver + anomaly)
python examples/run_pipeline.py output/valley_test \
    --enable-classification --enable-maneuver --enable-anomaly
```

ML flags need the `[ml]` extra. `--enable-anomaly` needs locally trained weights
(`output/models/anomaly_detector.pt` and `anomaly_threshold.json`); without them the pipeline
prints a warning and runs without anomaly detection.

ML flags can also be set via environment variables on the shell scripts:

```bash
ENABLE_CLASSIFICATION=true ENABLE_MANEUVER=true \
    bash examples/run_pipeline_erratic.sh
```

Auto-detects 2D vs 3D. Produces:

- `pipeline_summary_{2d,3d}.png` — 6-panel overview
- `radial_engagement_{2d,3d}.png` — engagement diagram
- `beamformer_diagnostic_{2d,3d}.png` — SRP-PHAT vs RMS comparison
- `results_{2d,3d}.json` — machine-readable metrics

### Helmholtz solver

```bash
acoustic-sim --model-preset gradient --frequency 40
```

### ML training

```bash
# Train the source, maneuver and fusion classifiers into output/models/
python examples/train_all_ml.py

# Train and calibrate the CVAE anomaly detector (writes anomaly_detector.pt + anomaly_threshold.json)
python src/acoustic_sim/ml/anomaly_training.py

# Baseline vs ML comparison across all datasets
python examples/run_comparison.py
```

### Batch FDTD

```bash
# Run 18 domain × source × array combinations
python examples/run_all_examples.py
```

### Tests

```bash
pip install -e ".[dev]"     # add ,ml to run the PyTorch tests (they skip without torch)
python -m pytest -q tests   # full suite: several minutes on a CPU (ML training, harbor FDTD)
./dev.sh test               # tests/harbor in the dev container (GPU cases run when a GPU is present)
```

GitHub Actions (`.github/workflows/tests.yml`) runs a fast CPU-only subset on Python 3.10 and
3.13: `tests/test_package_layout.py`, `tests/test_contract.py` (the sensing contract: import rules,
observation files, the plain-array entry point), `tests/harbor/test_dispersion.py` and two FDTD tests from
`tests/test_fdtd_3d.py`. GPU, MPI and PyTorch are not required there.

## Dependencies

Python ≥ 3.10.

| Package | Required |
|---------|----------|
| numpy | yes |
| scipy | yes |
| matplotlib | yes |
| mpi4py | optional — multi-process FDTD (`pip install -e ".[mpi]"`); single-process without it |
| cupy-cuda12x | optional — GPU acceleration (`pip install -e ".[cuda]"`) |
| torch | optional — ML classifiers, FNO surrogate, anomaly detection (`pip install -e ".[ml]"`) |

## Pre-computed Datasets

The repository tracks the *metadata and results* of the drone reference runs under `output/`
(`metadata.json`, `results_*.json`, a `README.md` per run), the trained ML weights in
`output/models/`, and selected harbor and performance results and figures.  Traces, datasets,
caches and other figures are not tracked (see `.gitignore`). Regenerate the drone traces with
`python examples/generate_traces.py` (fast analytic forward model, all four datasets) or re-run the
FDTD (`examples/run_valley.sh`, `run_valley_3d.sh`, `run_fdtd_erratic.py`).  [Generated outputs](docs/outputs.md) lists what is
tracked, how to regenerate each directory, and `scripts/clean_outputs.py` for reclaiming space.

| Directory | Dimension | Domain | Source | Duration |
|-----------|-----------|--------|--------|----------|
| `valley_test` | 2-D | hills & vegetation | moving propeller | 3.0 s |
| `valley_3d_test` | 3-D | hills & vegetation | moving propeller (altitude arc) | 1.0 s |
| `isotropic_2D` | 2-D | wind (5 m/s, 45°) | moving propeller | 3.0 s |
| `erratic_quadcopter` | 3-D | isotropic | erratic trajectory (8 m/s mean) | 3.0 s |

Once regenerated, each directory contains `traces.npy` next to the tracked `metadata.json` (FDTD runs also write `snapshots/` when `--snapshot-interval` is non-zero).

## Documentation

The `docs/` folder contains extensive documentation of the code, algorithms, and underlying physics.

| Page | Contents |
|------|----------|
| [Architecture](docs/architecture.md) | Package layout, module inventory, data flow |
| [Algorithms](docs/algorithms.md) | SRP-PHAT, WLS tracker, ballistics, MFP/EKF, ML classifiers |
| [Physics](docs/physics.md) | Wave equation, FD stencils, CFL, absorption, SPL, noise models |
| [Configuration](docs/configuration.md) | Pipeline JSON config, DetectionConfig, CLI flags |
| [API Reference](docs/api.md) | Every public class, method, and function |
| [Usage](docs/usage.md) | Installation, CLI commands, running examples |
| [Studies](docs/studies.md) | Nine parametric robustness studies |
| [Harbor](docs/harbor.md) | Water-based 2-D harbor, DAS/hydrophone sensing, passive and bistatic examples, assumptions and validation status |
| [Generated outputs](docs/outputs.md) | What is tracked under `output/`, how to regenerate it, cleanup |
| [Glossary](docs/glossary.md) | Definitions of terms, acronyms, and concepts |
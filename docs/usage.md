# Usage Guide

This document covers installation, running simulations, executing the detection pipeline, and programmatic API usage.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start: Helmholtz Solver](#2-quick-start-helmholtz-solver)
3. [Quick Start: FDTD Solver](#3-quick-start-fdtd-solver)
4. [Running the Detection Pipeline](#4-running-the-detection-pipeline)
5. [Running the Full Pipeline Example](#5-running-the-full-pipeline-example)
6. [Running All 18 FDTD Examples](#6-running-all-18-fdtd-examples)
7. [Running Robustness Studies](#7-running-robustness-studies)
8. [Programmatic API Usage](#8-programmatic-api-usage)
9. [Docker Usage](#9-docker-usage)
10. [Output Files and Plots](#10-output-files-and-plots)

---

## 1. Installation

### 1.1 From Source (pip)

```bash
# Clone the repository
git clone <repo-url>
cd acoustic-sim

# Install in editable mode
pip install -e .
```

This installs the `acoustic-sim` package and registers the `acoustic-sim` console script.

### 1.2 MPI Support

MPI is required for parallel FDTD execution. Install OpenMPI and the Python bindings:

```bash
# Ubuntu / Debian
sudo apt-get install libopenmpi-dev openmpi-bin
pip install mpi4py

# macOS (Homebrew)
brew install open-mpi
pip install mpi4py
```

The FDTD solver gracefully falls back to single-process mode if `mpi4py` is not installed or if the script is not launched with `mpirun`.

### 1.3 Optional CUDA Acceleration

GPU acceleration via CuPy requires an NVIDIA GPU with CUDA 12.x:

```bash
pip install -e ".[cuda]"
```

Or install CuPy separately:

```bash
pip install cupy-cuda12x
```

If CuPy is unavailable or no GPU is detected, the code falls back to NumPy with a warning.

### 1.4 Docker

A Docker image with all dependencies (including MPI and CUDA) is provided:

```bash
# Build and start interactive dev shell
docker compose up dev

# Or build manually
docker build -t acoustic-sim .
docker run -it acoustic-sim bash
```

### 1.5 Dev Container

The repository includes a `.devcontainer/devcontainer.json` for VS Code / Codespaces development:

```bash
# VS Code: "Reopen in Container" from the Command Palette
# GitHub Codespaces: create from the repository
```

### 1.6 Verify Installation

```bash
# Helmholtz solver
acoustic-sim --model-preset uniform --frequency 40

# Python import
python -c "import acoustic_sim; print('OK')"
```

---

## 2. Quick Start: Helmholtz Solver

The Helmholtz solver computes the steady-state pressure field for a monochromatic source.

### 2.1 Using Built-in Presets

```bash
# Uniform velocity model, 40 Hz source
acoustic-sim --model-preset uniform --frequency 40

# Gradient model, 80 Hz source, custom bounds
acoustic-sim --model-preset gradient --frequency 80 \
    --x-min -30 --x-max 30 --y-min -30 --y-max 30 --dx 0.3

# Valley model (hills + valley)
acoustic-sim --model-preset valley --frequency 25 --dx 0.5

# Checkerboard (resolution test)
acoustic-sim --model-preset checkerboard --frequency 40
```

Available presets: `uniform`, `layered`, `gradient`, `checkerboard`, `valley`.

### 2.2 Using JSON Configuration Files

```bash
acoustic-sim --model-file examples/domain.example.json --frequency 40
acoustic-sim --model-file examples/layered.example.json --frequency 60
acoustic-sim --model-file examples/valley.example.json --frequency 25
```

JSON files specify model type, bounds, grid spacing, and optional anomalies. See [Configuration Reference](configuration.md#3-json-config-format).

### 2.3 Using a Pre-Built Model

```bash
# Save a model
acoustic-sim --model-preset gradient --save-model-path my_model.npz --frequency 40

# Reload it
acoustic-sim --model-npz my_model.npz --frequency 80
```

### 2.4 Customising Receivers

```bash
# Circular array (default)
acoustic-sim --model-preset uniform --frequency 40 \
    --receiver-type circle --receiver-count 16 --receiver-radius 0.5

# Line array
acoustic-sim --model-preset uniform --frequency 40 \
    --receiver-type line --receiver-count 32 \
    --receiver-x0 -15 --receiver-y0 0 --receiver-x1 15 --receiver-y1 0
```

### 2.5 Output

The solver produces two PNG files:
- `velocity_model.png` — Velocity field with receiver and source markers
- `wavefield.png` — Pressure magnitude `|P(x,y)|`

Custom paths: `--velocity-plot path.png --field-plot path.png`

---

## 3. Quick Start: FDTD Solver

The FDTD solver computes time-domain pressure propagation with support for moving sources, multiple domain types, and various array geometries.

### 3.1 Basic Run

```bash
# Static source, isotropic domain, circular array, Ricker wavelet
python examples/run_fdtd.py \
    --domain isotropic --source-type static \
    --source-signal ricker --source-freq 25 \
    --array circular --receiver-radius 15 \
    --total-time 0.3 --output-dir output/basic_test
```

### 3.2 MPI Parallel Execution

```bash
# 4 MPI ranks
mpirun -np 4 python examples/run_fdtd.py \
    --domain isotropic --source-type static \
    --source-signal ricker --source-freq 25 \
    --array circular --total-time 0.3 \
    --output-dir output/mpi_test

# 8 MPI ranks (for larger grids)
mpirun -np 8 python examples/run_fdtd.py \
    --domain wind --dx 0.2 --total-time 0.5 \
    --output-dir output/large_grid
```

### 3.3 Moving Source with Propeller Noise

```bash
mpirun -np 4 python examples/run_fdtd.py \
    --domain wind --wind-speed 15 --wind-dir 45 \
    --source-type moving --source-x -30 --source-y 0 \
    --source-x1 30 --source-y1 0 --source-speed 50 \
    --source-signal propeller \
    --array concentric --output-dir output/moving_propeller
```

### 3.4 WAV File Source

```bash
mpirun -np 4 python examples/run_fdtd.py \
    --domain isotropic --source-type static \
    --source-signal file --source-wav audio/input.wav --max-seconds 0.3 \
    --array linear --output-dir output/wav_test
```

### 3.5 Higher-Order FD Stencils

```bash
# Order 4 (less numerical dispersion, wider stencil)
python examples/run_fdtd.py --fd-order 4 --output-dir output/order4

# Order 8 (very low dispersion, for reference solutions)
python examples/run_fdtd.py --fd-order 8 --output-dir output/order8
```

### 3.6 GPU Acceleration

```bash
python examples/run_fdtd.py --use-cuda --output-dir output/gpu_test
```

### 3.7 Source Signal Options

| `--source-signal` | Description | Key Parameters |
|---|---|---|
| `ricker` | Ricker (Mexican-hat) wavelet | `--source-freq` (peak frequency) |
| `tone` | Pure sine wave | `--source-freq` |
| `noise` | Band-limited coloured noise | (5–60 Hz default) |
| `propeller` | Synthetic rotor noise | `--blade-count`, `--rpm`, `--harmonics` |
| `file` | WAV file | `--source-wav`, `--max-seconds` |

### 3.8 Domain Options

| `--domain` | Description | Extra Parameters |
|---|---|---|
| `isotropic` | Uniform velocity, no wind | None |
| `wind` | Uniform + constant wind | `--wind-speed`, `--wind-dir` |
| `hills_vegetation` | Valley with ridges and vegetation | `--dirt-velocity`, `--seed` |

### 3.9 Array Options

| `--array` | Description | Key Parameters |
|---|---|---|
| `circular` | Single-ring | `--receiver-count`, `--receiver-radius` |
| `concentric` | Multiple rings | `--receiver-radii "10,20,30,40"` |
| `linear` | Straight line | `--receiver-x0/y0/x1/y1`, `--receiver-count` |

---

## 4. Running the Detection Pipeline

The detection pipeline simulates a drone, processes the microphone data, tracks the target, and computes fire-control solutions.

### 4.1 Command-Line Usage

```bash
# Default configuration (linear trajectory, isotropic domain)
python src/acoustic_sim/detection_main.py --output-dir output/demo

# Loiter-approach trajectory
python src/acoustic_sim/detection_main.py \
    --trajectory loiter_approach --total-time 2.0 \
    --output-dir output/loiter

# Wind domain with evasive trajectory
python src/acoustic_sim/detection_main.py \
    --trajectory evasive --domain wind \
    --total-time 1.0 --output-dir output/evasive_wind

# No noise (clean signal test)
python src/acoustic_sim/detection_main.py --no-noise --output-dir output/clean

# Custom array
python src/acoustic_sim/detection_main.py \
    --array-type circular --n-mics 24 --array-radius 1.0 \
    --output-dir output/large_array
```

### 4.2 CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--trajectory` | `linear` | Trajectory type: linear, circular, figure_eight, loiter_approach, evasive |
| `--domain` | `isotropic` | Domain: isotropic, wind, hills_vegetation, echo_canyon, urban_echo |
| `--total-time` | `0.5` | Simulation duration [s] |
| `--dx` | `0.05` | Grid spacing [m] |
| `--x-min/max` | `-15/15` | Domain bounds [m] |
| `--y-min/max` | `-15/15` | Domain bounds [m] |
| `--n-mics` | `16` | Number of microphones |
| `--array-type` | `circular` | Array geometry |
| `--array-radius` | `0.5` | Array radius [m] |
| `--drone-speed` | `15.0` | Drone speed [m/s] |
| `--source-level-dB` | `90.0` | Drone source level [dB SPL at 1 m] |
| `--fundamental-freq` | `150.0` | Drone fundamental frequency [Hz] |
| `--no-noise` | off | Disable all noise sources |
| `--no-stationary` | off | Disable stationary noise source |
| `--output-dir` | `output/detection` | Output directory |
| `--grid-spacing` | `1.0` | MFP grid spacing [m] |
| `--detection-threshold` | `0.15` | MFP detection threshold |
| `--snapshot-interval` | `0` | FDTD snapshot interval (0 = off) |

### 4.3 Programmatic Usage

```python
from acoustic_sim.config import DetectionConfig
from acoustic_sim.detection_main import run_detection_pipeline

# Default configuration
result = run_detection_pipeline()

# Custom configuration
config = DetectionConfig(
    trajectory_type="loiter_approach",
    domain_type="isotropic",
    total_time=2.0,
    n_mics=16,
    array_type="circular",
    array_radius=0.5,
    source_level_dB=90.0,
    output_dir="output/custom",
)
result = run_detection_pipeline(config)

# Access results
print(f"Detection rate: {result['detection_rate']*100:.0f}%")
print(f"Localisation error: {result['mean_loc_error']:.1f} m")
print(f"First shot miss: {result['first_shot_miss']:.2f} m")
```

### 4.4 Output Files

Each detection run produces in `output_dir/`:

| File | Description |
|---|---|
| `detection_domain.png` | Domain overview with mics, trajectory, weapon |
| `detection_gather.png` | Two-panel raw/filtered traces |
| `beam_power.png` | Multi-panel beam-power snapshots |
| `polar_beam_power.png` | Polar beam-power maps |
| `tracking.png` | Four-panel tracking display |
| `vespagram.png` | Beam power vs. slowness and time |

---

## 5. Running the Full Pipeline Example

The `examples/run_full_pipeline.py` script loads pre-computed FDTD data and runs the detection/targeting pipeline with visualisation.

### 5.1 Prerequisites

First run an FDTD simulation to generate traces:

```bash
mpirun -np 4 python examples/run_fdtd.py \
    --domain isotropic --source-type moving \
    --source-signal propeller --source-freq 180 \
    --source-x -40 --source-y 0 --source-x1 40 --source-y1 0 \
    --source-speed 50 --source-arc-height 15 \
    --array circular --receiver-count 16 \
    --total-time 2.0 --output-dir output/sim_data
```

### 5.2 Run the Pipeline

```bash
python examples/run_full_pipeline.py output/sim_data \
    --source-speed 50 --fundamental 180 --n-harmonics 4 \
    --output-dir output/pipeline_results
```

### 5.3 Output

- `pipeline_evaluation.png` — 6-panel evaluation (spatial overview, bearing/range vs time, error histograms, miss distances, summary)
- `radial_engagement.png` — Radial engagement view with projectile trajectories
- `pipeline_results.json` — Machine-readable metrics

---

## 6. Running All 18 FDTD Examples

```bash
# Default: 4 MPI ranks
python examples/run_all_examples.py

# Custom rank count
python examples/run_all_examples.py --np 2

# Custom output directory
python examples/run_all_examples.py --output-root my_output
```

This runs every combination of:

| Source Type | Domain | Array Geometry |
|---|---|---|
| static | isotropic | concentric |
| moving | wind | circular |
| | hills_vegetation | linear |

Total: 2 × 3 × 3 = 18 runs. Each produces `domain.png`, `gather.png`, `traces.npy`, `metadata.json`, and `snapshots/`.

---

## 7. Running Robustness Studies

### 7.1 All Studies (Batch)

```python
from acoustic_sim.studies import run_all_studies
results = run_all_studies()
```

This runs the four single-FDTD studies: sensor faults, transients, position errors, and mixed failures. Output in `output/studies/`.

### 7.2 Individual Studies

```python
from acoustic_sim.studies import (
    study_array_geometry,
    study_min_sensors,
    study_sensor_faults,
    study_multi_drone,
    study_transient_robustness,
    study_haphazard_array,
    study_echo_domains,
    study_position_errors,
    study_mixed_failures,
)

# Array geometry comparison
study_array_geometry(output_dir="output/studies/array_geometry")

# Sensor count sweep
study_min_sensors(sensor_counts=(4, 8, 16, 32), output_dir="output/studies/sensors")

# Sensor faults with custom fractions
study_sensor_faults(fault_fractions=(0.0, 0.1, 0.3, 0.5))

# Echo domain comparison
study_echo_domains()

# Position errors
study_position_errors(error_stds=(0.0, 0.5, 1.0, 2.0, 5.0))

# Mixed failures (progressive stress test)
study_mixed_failures()
```

### 7.3 Custom Base Configuration

Pass a custom `DetectionConfig` to any study:

```python
from acoustic_sim.config import DetectionConfig
from acoustic_sim.studies import study_sensor_faults

config = DetectionConfig(
    total_time=1.0,
    n_mics=24,
    array_radius=1.0,
    source_level_dB=95.0,
)
study_sensor_faults(base_config=config)
```

### 7.4 Study Output

Each study produces:
- Per-case subdirectories with detection plots
- `comparison.png` — Bar chart comparing all cases
- Console table with metrics

See [Study Methodology & Results](studies.md) for detailed interpretation.

---

## 8. Programmatic API Usage

### 8.1 Building a Custom Simulation

```python
import numpy as np
from acoustic_sim import (
    create_uniform_model,
    add_circle_anomaly,
    FDTDConfig,
    FDTDSolver,
    StaticSource,
    make_wavelet_ricker,
    create_receiver_circle,
    DomainMeta,
    plot_gather,
    check_spatial_sampling,
    check_cfl,
)

# Create a velocity model with an anomaly
model = create_uniform_model(-20, 20, -20, 20, dx=0.4, velocity=343.0)
model = add_circle_anomaly(model, cx=5, cy=-5, radius=3, velocity=290.0)

# Check grid resolution
sampling = check_spatial_sampling(model, frequency_hz=40.0)
print(sampling['message'])

# Create receivers
receivers = create_receiver_circle(cx=0, cy=0, radius=10, count=16)

# Create source
n_steps = 1000
dt = 0.0005
signal = make_wavelet_ricker(n_steps, dt, f0=25.0)
source = StaticSource(x=0.0, y=0.0, signal=signal)

# Configure and run FDTD
config = FDTDConfig(total_time=0.5, dt=dt, snapshot_interval=100)
solver = FDTDSolver(model, config, source, receivers, DomainMeta())
result = solver.run(snapshot_dir="output/custom/snapshots")

# Plot results
plot_gather(result["traces"], result["dt"], output_path="output/custom/gather.png")
```

### 8.2 Using the MFP Independently

```python
from acoustic_sim.processor import matched_field_process

# traces: (n_mics, n_samples) numpy array
# mic_positions: (n_mics, 2) numpy array
# dt: float (timestep)

result = matched_field_process(
    traces, mic_positions, dt,
    sound_speed=343.0,
    fundamental=150.0,
    n_harmonics=6,
    detection_threshold=0.25,
    enable_sensor_weights=True,
    enable_transient_blanking=True,
)

for det in result["detections"]:
    if det["detected"]:
        print(f"t={det['time']:.2f}s  bearing={det['bearing_deg']:.1f}°  "
              f"range={det['range']:.0f}m  coherence={det['coherence']:.3f}")
```

### 8.3 Using the Tracker Independently

```python
from acoustic_sim.tracker import run_tracker

track = run_tracker(
    result["detections"],
    process_noise_std=2.0,
    sigma_bearing_deg=3.0,
    sigma_range=100.0,
    initial_range_guess=200.0,
    array_center_x=500.0,
    array_center_y=500.0,
)

# Track output
print(f"Track length: {len(track['times'])} states")
print(f"Final position: {track['positions'][-1]}")
print(f"Final velocity: {track['velocities'][-1]}")
```

### 8.4 Custom Receiver Arrays

```python
from acoustic_sim import (
    create_receiver_nested_circular,
    create_receiver_custom,
    print_array_diagnostics,
)

# Nested circular (default for detection)
mics = create_receiver_nested_circular(
    cx=500, cy=500, inner_radius=0.15, outer_radius=0.5,
    n_inner=4, n_outer=8,
)

# Custom positions
custom_mics = create_receiver_custom([
    (500.0, 500.0),    # centre
    (500.5, 500.0),    # right
    (499.5, 500.0),    # left
    (500.0, 500.5),    # top
    (500.0, 499.5),    # bottom
])

# Diagnostics
print_array_diagnostics(mics, sound_speed=343.0)
```

### 8.5 Custom Trajectories

```python
import numpy as np
from acoustic_sim.sources import CustomTrajectorySource, make_drone_harmonics

# Create a custom trajectory
times = np.linspace(0, 60, 600)  # 60 seconds, 10 Hz
positions = np.column_stack([
    200 + 100 * np.cos(0.05 * times),  # circular path
    500 + 100 * np.sin(0.05 * times),
])

signal = make_drone_harmonics(n_steps=100000, dt=0.0001, fundamental=150.0)

source = CustomTrajectorySource(
    times=times,
    positions=positions,
    signal=signal,
)

# Query position at any time
x, y = source.position_at(step=5000, dt=0.0001)
```

### 8.6 Working with Noise

```python
from acoustic_sim.noise import (
    generate_wind_noise,
    generate_sensor_noise,
    inject_sensor_faults,
    inject_transient,
    perturb_mic_positions,
)

# Generate wind noise
wind = generate_wind_noise(
    mic_positions, n_samples=100000, dt=0.0001,
    level_dB=55.0, corner_freq=15.0, correlation_length=3.0,
)

# Inject faults on specific sensors
faulted, fault_list = inject_sensor_faults(
    traces, fault_type="elevated_noise",
    fault_sensors=[0, 3, 7], fault_level_dB=100.0,
)

# Inject a transient event
traces_with_transient = inject_transient(
    traces, dt=0.0001,
    event_time=15.0, event_pos=(550, 450),
    mic_positions=mic_positions, level_dB=130.0,
)

# Perturb positions (simulate placement errors)
reported = perturb_mic_positions(true_positions, error_std=2.0)
```

---

## 9. Docker Usage

### 9.1 Interactive Development Shell

```bash
docker compose run dev
```

This mounts the workspace, creates a virtualenv symlink, and drops you into a bash shell with all dependencies installed.

### 9.2 Running a Simulation

```bash
docker compose run simulate --model-preset gradient --frequency 40
```

### 9.3 MPI Inside Docker

```bash
docker compose run dev bash -c \
    "mpirun --allow-run-as-root -np 4 python examples/run_fdtd.py \
     --domain isotropic --output-dir output/docker_test"
```

### 9.4 GPU Access

The `docker-compose.yml` includes GPU device reservation. Ensure the NVIDIA Container Toolkit is installed:

```bash
docker compose run dev bash -c \
    "python examples/run_fdtd.py --use-cuda --output-dir output/gpu_test"
```

---

## 10. Output Files and Plots

### 10.1 FDTD Output (from `run_fdtd.py`)

| File | Format | Description |
|---|---|---|
| `domain.png` | PNG | Velocity model + receivers + source + wind |
| `gather.png` | PNG | Receiver traces (dB SPL) |
| `traces.npy` | NumPy | Raw traces, shape `(n_receivers, n_samples)` |
| `metadata.json` | JSON | Simulation parameters |
| `snapshots/` | PNGs | Numbered wavefield frames |

### 10.2 Detection Pipeline Output

| File | Format | Description |
|---|---|---|
| `detection_domain.png` | PNG | Domain with mics, trajectory, weapon |
| `detection_gather.png` | PNG | Two-panel raw/filtered traces |
| `beam_power.png` | PNG | Multi-panel beam-power snapshots |
| `polar_beam_power.png` | PNG | Polar beam-power maps |
| `tracking.png` | PNG | Four-panel tracking/fire-control display |
| `vespagram.png` | PNG | Slowness-time beam-power |

### 10.3 Study Output

| File | Format | Description |
|---|---|---|
| `comparison.png` | PNG | Bar-chart comparison across cases |
| `<case>/detection_*.png` | PNGs | Per-case detection plots |
| `<case>/beam_power.png` | PNG | Per-case beam power |
| `<case>/tracking.png` | PNG | Per-case tracking |
| `<case>/vespagram.png` | PNG | Per-case vespagram |

### 10.4 Interpreting the Gather Plot

The gather plot shows receiver traces as a 2D image with:
- **x-axis**: Receiver index (0 to N-1)
- **y-axis**: Time (top to bottom)
- **Colour**: Sound pressure level (dB SPL)

Look for:
- **Hyperbolic moveout**: Direct arrival from a point source — characteristic hyperbola across receivers
- **Linear moveout**: Moving source — arrival time shifts linearly with receiver position
- **Reverberations**: Late arrivals from reflections

### 10.5 Interpreting the Tracking Plot

Four panels:
1. **Bearing vs time**: Red = true, blue dots = estimated. Should track closely.
2. **Range vs time**: Red = true, blue = estimated. Range converges over time.
3. **Lead angle vs time**: Green = computed lead angle. Non-zero for crossing targets.
4. **Engagement envelope**: Green = FIRE, red = NO FIRE. Shows when engagement conditions are met.

### 10.6 Interpreting the Vespagram

The vespagram shows beam power as a function of:
- **x-axis**: Time [s]
- **y-axis**: Slowness [ms/m] (inverse of apparent velocity)

The cyan dashed line marks `1/c` (reference slowness). A bright track at this slowness indicates a coherent signal propagating at the speed of sound. A moving source will show time-varying slowness due to Doppler shift and changing geometry.

---

*Next: [Configuration Reference](configuration.md) — Complete parameter reference.*

# Usage

## Installation

### pip (editable)

```bash
pip install -e .
```

Core dependencies: `numpy`, `scipy`, `matplotlib`, `mpi4py`.

### MPI (required for multi-process FDTD)

```bash
# Ubuntu / Debian
sudo apt-get install libopenmpi-dev openmpi-bin
pip install mpi4py
```

### CUDA (optional GPU acceleration)

```bash
pip install -e ".[cuda]"   # installs cupy-cuda12x
```

### Docker

```bash
docker compose up dev       # interactive shell with GPU
```

The container is built on `nvidia/cuda:12.6.3-devel-ubuntu24.04` with
Python 3.12, OpenMPI, and CuPy.

---

## Helmholtz Solver (Frequency-Domain)

```bash
acoustic-sim --model-preset gradient --frequency 40
```

Full CLI:

```bash
acoustic-sim [--model-preset PRESET | --model-file JSON | --model-npz NPZ]
             [--frequency HZ] [--dx DX]
             [--x-min X --x-max X --y-min Y --y-max Y]
             [--source-x X --source-y Y]
             [--receiver-type {line|circle}]
             [--velocity-plot PATH] [--field-plot PATH]
             [--save-model-path NPZ]
```

Model presets: `uniform`, `layered`, `gradient`, `checkerboard`, `valley`.

---

## FDTD Forward Model (2-D)

### Single run

```bash
python examples/run_fdtd.py \
    --domain hills_vegetation \
    --source-type moving --source-signal propeller \
    --source-x -40 --source-y 0 --source-x1 40 --source-y1 0 \
    --source-speed 50 --blade-count 3 --rpm 3600 --harmonics 14 \
    --array circular --receiver-count 16 --receiver-radius 2.0 \
    --total-time 3.0 --dx 0.18 --fd-order 2 \
    --output-dir output/valley_test --use-cuda
```

### With MPI

```bash
mpirun -np 4 python examples/run_fdtd.py \
    --domain isotropic --source-type static \
    --source-signal ricker --source-freq 25 \
    --array circular --total-time 0.3 \
    --output-dir output/test
```

### All 18 combinations

```bash
python examples/run_all_examples.py [--np N]
```

Runs `{static, moving}` × `{isotropic, wind, hills_vegetation}` ×
`{concentric, circular, linear}`.

### Key FDTD CLI flags

| Flag | Description |
|------|-------------|
| `--domain` | `isotropic`, `wind`, `hills_vegetation` |
| `--dx` | grid spacing (m) |
| `--fd-order` | stencil accuracy: 2, 4, 6, 8 |
| `--total-time` | simulation duration (s) |
| `--source-type` | `static` or `moving` |
| `--source-signal` | `file`, `propeller`, `tone`, `noise`, `ricker` |
| `--source-x`, `--source-y` | start position |
| `--source-x1`, `--source-y1` | end position (moving) |
| `--source-speed` | source speed (m/s) |
| `--source-arc-height` | parabolic arc height (m) |
| `--blade-count`, `--rpm`, `--harmonics` | propeller params |
| `--source-freq` | tone / ricker frequency (Hz) |
| `--source-wav` | WAV file path |
| `--array` | `circular`, `concentric`, `linear`, etc. |
| `--receiver-count` | number of microphones |
| `--receiver-radius` | array radius (m) |
| `--receiver-cx`, `--receiver-cy` | array centre |
| `--snapshot-interval` | field dump every N steps (0 = off) |
| `--use-cuda` | enable GPU |
| `--output-dir` | output directory |

### Output

| File | Contents |
|------|----------|
| `traces.npy` | pressure time series `(n_mics, n_steps)` |
| `metadata.json` | all simulation parameters, receiver positions |
| `snapshots/` | optional pressure field dumps as `.npy` |
| `domain.png` | velocity model visualisation |
| `gather.png` | microphone trace gather |

---

## FDTD Forward Model (3-D)

Same interface as 2-D, with additional z-axis flags:

```bash
python examples/run_fdtd_3d.py \
    --domain hills_vegetation \
    --dx 0.25 --z-min -5.0 --z-max 50.0 \
    --source-type moving --source-signal propeller \
    --source-x -40 --source-y 0 --source-z 15 \
    --source-x1 40 --source-y1 0 --source-z1 15 \
    --source-speed 50 --source-arc-height 10 \
    --array circular --receiver-count 16 --receiver-radius 2.0 \
    --receiver-cz 5.5 \
    --total-time 3.0 --fd-order 8 \
    --output-dir output/valley_3d_test --use-cuda
```

---

## SRP-PHAT Engagement Pipeline

### Run on existing FDTD output

```bash
# 2-D
python examples/run_pipeline.py output/valley_test

# 3-D
python examples/run_pipeline.py output/valley_3d_test

# Custom config
python examples/run_pipeline.py output/valley_test \
    --config examples/pipeline.config.json

# CLI overrides
python examples/run_pipeline.py output/valley_test \
    --source-speed 60 --hit-threshold 3.0 --max-hits 5
```

#### ML-enabled modes

```bash
# With source classification gate
python examples/run_pipeline.py output/valley_test --enable-classification

# Full ML stack
python examples/run_pipeline.py output/valley_test \
    --enable-classification --enable-maneuver --enable-anomaly

# Custom classification threshold
python examples/run_pipeline.py output/valley_test \
    --enable-classification --classification-threshold 0.8
```

The pipeline auto-detects 2-D vs 3-D from the shape of `mic_positions`
in `metadata.json`.

### CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `sim_dir` | `output/valley_3d_test` | simulation output directory |
| `--config` | `examples/pipeline.config.json` | JSON config file |
| `--output-dir` | same as `sim_dir` | output directory for plots |
| `--source-speed` | from config (50.0) | override ground-truth source speed |
| `--hit-threshold` | from config (2.0) | override hit distance threshold |
| `--max-hits` | from config (3) | override max hits before stop |
| `--enable-classification` | off | enable source classification gate |
| `--enable-maneuver` | off | enable maneuver-adaptive tracking |
| `--enable-fusion` | off | enable fusion classification |
| `--enable-anomaly` | off | enable CVAE anomaly detection |
| `--classification-threshold` | from config (0.7) | override classification confidence threshold |
| `--maneuver-window` | from config (20) | override maneuver window size |

### Output

| File | Contents |
|------|----------|
| `pipeline_summary_{2d,3d}.png` | 6-panel summary: spatial, bearing, miss distance, track error, latency, text |
| `radial_engagement_{2d,3d}.png` | plan-view (+ elevation for 3-D) engagement diagram |
| `beamformer_diagnostic_{2d,3d}.png` | 4-panel SRP-PHAT vs RMS bearing comparison |
| `results_{2d,3d}.json` | machine-readable metrics (includes `ml` section with classification/anomaly stats when ML is enabled) |

---

## ML Training

Pre-trained weights are shipped in `output/models/`.  To retrain:

```bash
# Train all classifiers (generates synthetic data + trains)
python examples/train_all_ml.py

# Run ML demo (classification + confusion matrices)
python examples/run_ml_demo.py
```

---

## Shell Script Demos

Three end-to-end scripts run FDTD then the pipeline:

```bash
# 2-D valley: hills+vegetation, moving propeller, circular array
bash examples/run_valley.sh

# 3-D valley: same domain extruded to volume, source at altitude
bash examples/run_valley_3d.sh

# 2-D wind domain with circular orbit source
bash examples/run_wind_circular.sh
```

Each script sets all parameters as shell variables, runs `run_fdtd.py`
or `run_fdtd_3d.py`, then calls `run_pipeline.py` on the output.

---

## Extract Traces from Field Plane

If FDTD is run without explicitly placed receivers, traces can be
extracted post-hoc from saved field snapshots:

```bash
python examples/extract_array_traces.py \
    --sim-dir output/my_test \
    --array circular --count 16 --radius 2.0 \
    --cx 0 --cy 10
```

---

## MFP/EKF Pipeline (Library)

The older matched-field-processing pipeline is available in the library:

```bash
python src/acoustic_sim/detection_main.py \
    --trajectory loiter_approach \
    --domain isotropic \
    --total-time 60 \
    --output-dir output/mfp_demo
```

Or programmatically:

```python
from acoustic_sim.config import DetectionConfig
from acoustic_sim.detection_main import run_detection_pipeline

config = DetectionConfig(trajectory_type="linear", total_time=30)
results = run_detection_pipeline(config)
```

---

## Robustness Studies

```python
from acoustic_sim.studies import run_all_studies
results = run_all_studies()
```

See [Studies](studies.md) for details on each of the nine studies.

---

## Python API

### Build and run FDTD

```python
from acoustic_sim.setup import build_domain, build_receivers, build_source, compute_dt
from acoustic_sim.fdtd import FDTDConfig, FDTDSolver

domain, meta = build_domain("hills_vegetation", dx=0.18,
                             x_min=-50, x_max=50, y_min=-50, y_max=50)
receivers = build_receivers("circular", count=16, radius=2.0, cx=0, cy=10)
source = build_source("moving", "propeller", domain, dt=None,
                       source_x=-40, source_y=0, source_x1=40, source_y1=0,
                       speed=50, blade_count=3, rpm=3600, harmonics=14)
dt = compute_dt(domain, fd_order=2)
config = FDTDConfig(total_time=3.0, fd_order=2)
result = FDTDSolver.run(domain, meta, receivers, source, config)
# result["traces"].shape == (16, n_steps)
```

### Run the pipeline on traces

```python
from pathlib import Path
from examples.run_pipeline import load_config, run_pipeline

cfg = load_config(Path("examples/pipeline.config.json"))
results = run_pipeline(Path("output/valley_test"), Path("output/valley_test"), cfg)
```

### Use DetectionEngine directly

```python
from acoustic_sim.detection import DetectionEngine

engine = DetectionEngine(
    mic_positions=mics,       # (n_mics, 2 or 3)
    fs=fs,                    # sample rate
    window_samples=win,       # samples per window
    bearing_method="srp_phat",
    range_method="rms",       # or "tdoa", "bearing_rate"
    max_sources=1,
    tracker_max_history=20,
)
engine.calibrate_range(peak_rms, cpa_distance)

for seg in windows:
    det = engine.process_window(seg, t_center)
    if det.detected:
        print(det.bearing_deg, det.range_m, det.track)
```

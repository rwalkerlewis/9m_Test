# acoustic-sim

2-D / 3-D acoustic wave propagation and passive acoustic engagement.

FDTD solvers produce synthetic microphone traces.  A real-time SRP-PHAT
pipeline turns those traces into fire-control solutions.

Full documentation: [docs/](docs/index.md)

---

## Install

```bash
pip install -e .
```

MPI (for parallel FDTD):

```bash
sudo apt-get install libopenmpi-dev openmpi-bin
pip install mpi4py
```

GPU acceleration:

```bash
pip install -e ".[cuda]"
```

Docker:

```bash
docker compose up dev
```

---

## Quick Start

### 1. Run the FDTD forward model

```bash
# 2-D valley
bash examples/run_valley.sh

# 3-D valley
bash examples/run_valley_3d.sh
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

### 2. Run the engagement pipeline

```bash
python examples/run_pipeline.py output/valley_test
```

Auto-detects 2-D vs 3-D.  Produces:

- `pipeline_summary_{2d,3d}.png` — 6-panel overview
- `radial_engagement_{2d,3d}.png` — engagement diagram
- `beamformer_diagnostic_{2d,3d}.png` — SRP-PHAT vs RMS comparison
- `results_{2d,3d}.json` — machine-readable metrics

### 3. Helmholtz solver

```bash
acoustic-sim --model-preset gradient --frequency 40
```

---

## Pipeline

| Stage | Algorithm |
|-------|-----------|
| Detection | RMS threshold gate |
| Bearing | SRP-PHAT (pre-computed steering, 360 bins) |
| Range | RMS inverse-square-law, CPA-calibrated |
| Tracking | Causal weighted-least-squares fit |
| Fire control | 3-D iterative ballistic lead (azimuth + elevation) |

Configuration: [`examples/pipeline.config.json`](examples/pipeline.config.json)

---

## FDTD Solver

- 2-D and 3-D, unified codebase
- Configurable FD order (2, 4, 6, 8)
- MPI domain decomposition (y-axis 2-D, z-slab 3-D)
- Optional CUDA via CuPy (automatic fallback to NumPy)
- Sponge-layer absorbing boundaries
- Source signals: propeller, tone, noise, ricker, WAV file
- Domains: isotropic, wind, hills+vegetation

---

## ML Module (Optional)

Six files under `src/acoustic_sim/ml/` implement PyTorch classifiers for
source type (CNN on mel spectrograms), maneuver state (1-D CNN on
kinematics), and fusion classification (acoustic + kinematic).  **Not
wired into the pipeline.**  Requires `torch` (not a project dependency).

Potential value:
- **ManeuverClassifier** → adaptive tracker process noise
- **FusionClassifier** → class-based fire/no-fire decisions
- **compute_kinematic_features()** → pure-numpy threat heuristics (usable now)

See [docs/index.md](docs/index.md#ml-module) for details.

---

## Dependencies

| Package | Required |
|---------|----------|
| numpy | yes |
| scipy | yes |
| matplotlib | yes |
| mpi4py | yes (single-process fallback) |
| cupy-cuda12x | optional (GPU) |
| torch | optional (ML classifiers) |

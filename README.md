# acoustic-sim — 2D Acoustic Simulation

2D acoustic simulation built around a user-defined velocity model stored as a
NumPy array.  Includes a **Helmholtz (frequency-domain) solver** and a
**time-domain FDTD solver** with MPI parallelisation and optional CUDA
acceleration.

---

## Project Structure

```
├── src/acoustic_sim/       # Python package
│   ├── __init__.py         # Public API re-exports
│   ├── __main__.py         # python -m acoustic_sim
│   ├── cli.py              # Argument parsing & Helmholtz entry point
│   ├── model.py            # VelocityModel + creation helpers + anomalies
│   ├── sampling.py         # Spatial-sampling & CFL checks
│   ├── solver.py           # 2D Helmholtz solver
│   ├── backend.py          # NumPy / CuPy backend abstraction
│   ├── sources.py          # Audio source signals (WAV, propeller, tone, …)
│   ├── domains.py          # Domain builders (isotropic, wind, hills+veg)
│   ├── fdtd.py             # 2D FDTD solver with MPI + optional CUDA
│   ├── receivers.py        # Receiver geometry helpers
│   ├── io.py               # JSON/NPZ load & save
│   └── plotting.py         # Velocity model, wavefield, gather & snapshot plots
├── audio/                  # Sound / WAV files
├── examples/               # Example JSON configs & runner scripts
│   ├── run_fdtd.py         # Single FDTD run with full CLI control
│   └── run_all_examples.py # Orchestrate all 18 example combinations
├── simulate_array.py       # Legacy entry point (thin wrapper)
├── pyproject.toml          # Package metadata & dependencies
├── Dockerfile
├── docker-compose.yml
└── .devcontainer/
    └── devcontainer.json
```

---

## Installation

```bash
pip install -e .
```

MPI support requires an MPI library (e.g. OpenMPI):

```bash
# Ubuntu / Debian
sudo apt-get install libopenmpi-dev openmpi-bin
pip install mpi4py
```

Optional CUDA acceleration (requires an NVIDIA GPU):

```bash
pip install -e ".[cuda]"
```

Or use the Docker dev container:

```bash
docker compose up dev
```

---

## Quick Start — Helmholtz (frequency-domain)

```bash
acoustic-sim --model-preset gradient --frequency 40
```

---

## Quick Start — FDTD (time-domain, MPI)

### Single run

```bash
# Static source, isotropic domain, circular array, Ricker wavelet
mpirun -np 4 python examples/run_fdtd.py \
    --domain isotropic --source-type static \
    --source-signal ricker --source-freq 25 \
    --array circular --receiver-radius 15 \
    --total-time 0.3 --output-dir output/quick_test

# Static source with a WAV file as the source signal
mpirun -np 4 python examples/run_fdtd.py \
    --domain isotropic --source-type static \
    --source-signal file --source-wav audio/input.wav --max-seconds 0.3 \
    --array linear --output-dir output/wav_test

# Moving source with propeller model, wind domain
mpirun -np 4 python examples/run_fdtd.py \
    --domain wind --wind-speed 15 --wind-dir 45 \
    --source-type moving --source-x -30 --source-y 0 \
    --source-x1 30 --source-y1 0 --source-speed 50 \
    --source-signal propeller \
    --array concentric --output-dir output/wind_moving
```

### Run all 18 example combinations

```bash
python examples/run_all_examples.py --np 4
```

This runs every combination of:

| Source Type | Domain                | Array Geometry |
|-------------|-----------------------|----------------|
| static      | isotropic             | concentric     |
| moving      | isotropic + wind      | circular       |
|             | hills + vegetation    | linear         |

Each run produces in its output directory:

| File | Description |
|------|-------------|
| `domain.png` | Velocity model + receivers + source + wind overlay |
| `gather.png` | Seismic-style receiver gather (receiver × time) |
| `traces.npy` | Raw trace data, shape `(n_receivers, n_samples)` |
| `metadata.json` | Simulation parameters (dt, positions, etc.) |
| `snapshots/` | Numbered PNG frames of the wavefield for movie assembly |

---

## Source Signal Options

The FDTD solver accepts any 1-D time-series as the source signal.  Built-in
options:

| `--source-signal` | Description |
|-------------------|-------------|
| `file`            | Load a WAV file (`--source-wav`), LP-filter to grid resolution, resample |
| `propeller`       | Synthetic rotor noise (blade harmonics + broadband) |
| `tone`            | Pure sine wave at `--source-freq` Hz |
| `noise`           | Band-limited coloured noise |
| `ricker`          | Ricker (Mexican-hat) wavelet at `--source-freq` Hz |

Audio files are automatically low-pass filtered at the grid's maximum
resolvable frequency (`c_min / (10 × dx)`) and resampled to the simulation
timestep to ensure physical correctness.

---

## MPI & CUDA

**MPI**: The FDTD solver uses 2-D Cartesian domain decomposition.  Each rank
owns a rectangular sub-domain with one-cell ghost (halo) layers exchanged
every timestep.

```bash
mpirun -np 8 python examples/run_fdtd.py ...
```

**CUDA**: Pass `--use-cuda` to offload array operations to the GPU via CuPy.
Falls back to NumPy automatically if CuPy is not installed or no GPU is
detected.

---

## Spatial Sampling & CFL

The CLI automatically validates grid resolution:

- **Points-per-wavelength**: `λ_min / dx ≥ 10`
- **CFL**: `c_max · dt / dx ≤ 1/√2` (dt auto-computed with 0.9× safety margin)

---

## Docker

```bash
# Interactive dev shell
docker compose run dev

# Run simulation
docker compose run simulate --model-preset gradient --frequency 40
```

---

## Dependencies

- numpy
- scipy
- matplotlib
- mpi4py
- *Optional*: cupy-cuda12x (for GPU acceleration)

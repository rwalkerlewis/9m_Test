# acoustic-sim — 2D Acoustic Simulation

2D acoustic simulation built around a user-defined velocity model stored as a
NumPy array, with Helmholtz (frequency-domain) solving, heterogeneous ray
tracing, and spatial-sampling validation.

---

## Project Structure

```
├── src/acoustic_sim/       # Python package
│   ├── __init__.py         # Public API re-exports
│   ├── __main__.py         # python -m acoustic_sim
│   ├── cli.py              # Argument parsing & main entry point
│   ├── model.py            # VelocityModel + creation helpers + anomalies
│   ├── sampling.py         # Spatial-sampling & CFL checks
│   ├── solver.py           # 2D Helmholtz solver
│   ├── raytrace.py         # 2D heterogeneous ray tracing
│   ├── receivers.py        # Receiver geometry helpers
│   ├── io.py               # JSON/NPZ load & save
│   └── plotting.py         # Velocity model & wavefield plots
├── audio/                  # Sound / WAV files
├── examples/               # Example JSON velocity model configs
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

Or use the Docker dev container (runs as non-root `devuser`):

```bash
docker compose up dev
```

---

## Quick Start

### Built-in preset

```bash
acoustic-sim --model-preset gradient --frequency 40 --plot-rays
```

### From JSON config

```bash
acoustic-sim --model-file examples/domain.example.json --frequency 40 --plot-rays
```

### From a saved .npz model

```bash
acoustic-sim --model-npz my_model.npz --frequency 30
```

### Programmatic use

```python
import numpy as np
from acoustic_sim import model_from_array, check_spatial_sampling, solve_helmholtz

values = np.random.uniform(300, 400, (100, 100))
model = model_from_array(values, -20, 20, -20, 20)

sampling = check_spatial_sampling(model, frequency_hz=40.0)
print(sampling["message"])

source = np.array([0.0, 0.0])
field = solve_helmholtz(model, source, frequency_hz=40.0)
```

---

## Velocity Model JSON Format

`examples/domain.example.json`:

```json
{
  "bounds": { "x_min": -20, "x_max": 20, "y_min": -20, "y_max": 20 },
  "dx": 0.4,
  "type": "gradient",
  "background_velocity": 343.0,
  "v_bottom": 360.0,
  "v_top": 320.0,
  "anomalies": [
    { "type": "circle", "center": [8, -5], "radius": 3.5, "velocity": 290 },
    { "type": "rectangle", "x_min": -4, "x_max": 4, "y_min": 8, "y_max": 14, "velocity": 310 }
  ]
}
```

Supported model types: `uniform`, `layered`, `gradient`, `checkerboard`.

---

## Spatial Sampling Checks

The CLI automatically validates that grid spacing is sufficient:

- **Points-per-wavelength**: `λ_min / dx ≥ 10` (configurable via `--min-ppw`)
- **CFL** (for time-domain extensions): `c_max · dt / dx ≤ 1/√2`

```
PASS: 18.1 pts/wavelength >= 10 required. (lambda_min=7.250 m, dx=0.4000 m, ...)
```

---

## Docker

The container runs as a non-root user (`devuser`, UID 1000).

```bash
# Interactive dev shell
docker compose run dev

# Run simulation directly
docker compose run simulate --model-preset gradient --frequency 40 --plot-rays
```

---

## Dependencies

- numpy
- scipy
- matplotlib

# Audio Processing Home Task — Realistic Synthetic Source Extension

This repository now includes a significantly more realistic synthetic-array simulation pipeline built from the original example.

Implemented additions:

- **3D source positioning** with azimuth, distance, and **elevation**
- **Moving source trajectories** from input **legs** (path + speed)
- **Configurable propagation domain** with material contrasts (air/vegetation/topography/etc.)
- **Multiple source models**
  - File playback (`input.wav`)
  - Synthetic propeller model (blade count/RPM/harmonics/modulation/noise)
  - Tone/noise diagnostic models
- **Array realism effects**
  - Self-noise and wind-like low-frequency noise
  - Per-microphone gain/phase mismatch
  - Interferer sources
  - Simple multipath/scattering effect
- **Coincident Helmholtz propagation model** solved on the domain
- **Ray-tracing visualization overlay** on top of Helmholtz field plot

---

## Dependencies

The script uses:

- `numpy`
- `scipy`
- `matplotlib`

Install with:

```bash
python3 -m pip install numpy scipy matplotlib
```

---

## Quick Start

### 1) Legacy-like run (file source, static source geometry)

```bash
python3 simulate_array.py input.wav --output output_16ch.wav
```

### 2) Static 3D source (add elevation)

```bash
python3 simulate_array.py input.wav \
  --azimuth 70 \
  --elevation 18 \
  --distance 7 \
  --output output_16ch_static3d.wav \
  --field-plot field_static3d.png \
  --plot-rays
```

### 3) Moving source from legs + domain materials + Helmholtz + ray tracing

```bash
python3 simulate_array.py input.wav \
  --legs-file examples/legs.example.json \
  --domain-file examples/domain.example.json \
  --output output_16ch_moving.wav \
  --field-plot field_moving.png \
  --plot-rays \
  --ray-count 36 \
  --ray-bounces 2 \
  --max-seconds 15
```

### 4) Synthetic propeller source model (no input.wav needed)

```bash
python3 simulate_array.py \
  --source-model propeller \
  --source-model-file examples/source_model.example.json \
  --legs-file examples/legs.example.json \
  --domain-file examples/domain.example.json \
  --output output_propeller_16ch.wav \
  --field-plot field_propeller.png \
  --plot-rays
```

---

## Legs JSON format (`--legs-file`)

`examples/legs.example.json`:

```json
{
  "start": [4.0, 12.0, 2.0],
  "legs": [
    { "end": [8.0, 8.0, 2.4], "speed_m_s": 2.2 },
    { "end": [12.0, 2.0, 3.0], "speed_m_s": 3.0 }
  ]
}
```

Notes:

- Coordinates are `[x, y, z]` in meters.
- If a leg omits `start`, it begins at the previous leg's end.
- `speed_m_s` defines temporal progression over that segment.

---

## Domain JSON format (`--domain-file`)

`examples/domain.example.json` provides:

- Global bounds and grid spacing (`dx`)
- Material table (`wave_speed`, `attenuation`, `scattering`)
- Region list (`rectangle` and `circle` supported), each assigning a material

This domain influences:

1. Time-domain simulation (effective delay/attenuation/scattering along source-to-mic paths)
2. Helmholtz field solve (spatially varying wave speed + damping)

---

## Source Models

Use `--source-model`:

- `file` (default): mono WAV from positional `input` argument
- `propeller`: harmonic blade-pass model + modulation + broadband component
- `tone`: single-frequency synthetic tone
- `noise`: synthetic band-limited noise

Additional options include:

- `--blade-count`, `--rpm`, `--harmonics`, `--mod-depth`, `--broadband-level`
- `--duration`, `--sample-rate`
- `--source-model-file` for JSON-based parameter presets

---

## Helmholtz + Ray Overlay

The solver computes a 2D frequency-domain pressure field:

\[
\nabla^2 p + k(x,y)^2 p = -s
\]

with spatially varying medium properties derived from the domain/material map.

The saved plot (`--field-plot`) includes:

- Helmholtz magnitude field
- Domain region outlines
- Microphone positions
- Source path and start/end points
- Optional ray-tracing overlay (`--plot-rays`) with configurable count and bounce depth

---

## Key assumptions / limitations

- The Helmholtz field is 2D (`x-y`) for visualization and computational tractability.
- Time-domain moving-source simulation uses blockwise updates and fractional delay approximation.
- Ray tracing is a visualization-oriented geometric approximation (box-boundary reflections), not a full wave solver.

---

## Files

- `simulate_array.py` - main simulator + Helmholtz/ray plotting
- `examples/legs.example.json` - moving trajectory definition
- `examples/domain.example.json` - domain/material definition
- `examples/source_model.example.json` - propeller source model preset
- `input.wav` - mono recording for file source model

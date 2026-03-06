# Audio Processing Home Task

## Quick Start

```bash
uv run simulate_array.py input.wav
```

This simulates a 16-channel circular microphone array recording of a sound source at 45° azimuth, 5m distance. Output is saved to `output_16ch.wav`.

Options:
```bash
uv run simulate_array.py input.wav --azimuth 90 --distance 3
```

## Task Description

This is an open-ended project. Pick whatever interests you most, combine approaches, or propose something entirely different. The provided code and files are a reference and starting point - for some options you may only need `input.wav`.

Here are some directions to consider:

### Signal Enhancement
Implement noise reduction on mono sound (input.wav)

The input is a single-channel drone recording with background noise. Your goal is to produce `clean.wav` - an improved version of `input.wav` with reduced noise and clearer audio.

### Realistic Simulation
The current simulation is too clean. Make it more challenging:
- Microphone self-noise (thermal noise, quantization)
- Multiple sound sources / interferers
- Complex room/outdoor acoustics
- Microphone gain/phase mismatches
- Wind noise or environmental factors

Or fully simulate propeller sound from scratch based on blade count, RPM, size, etc.

### Advanced DOA
Implement a more advanced approach:
- Different algorithms (MUSIC, ESPRIT, neural network-based)
- Self-calibration and correction techniques
- Tracking moving sources over time

### Array Geometry
Is the current microphone array geometry good or bad for DOA estimation? Why?

## Deliverables

- Source code
- Any data you generated or used
- README explaining:
  - What you chose to work on
  - Assumptions made
  - What worked / what didn't
  - How you'd improve this for real-world deployment
- Plots or visualizations (if applicable)

## Files

- `simulate_array.py` - Main simulation and DOA estimation
- `input.wav` - [Download from Google Drive](https://drive.google.com/file/d/1qvOm2fkwWAoU8x3GbdhQVAt2R8kgjoRf/view?usp=sharing)

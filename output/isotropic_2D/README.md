# Wind Circular Test Output

Results from a 2-D FDTD acoustic simulation over a flat wind domain.

## How this was generated

1. **FDTD simulation** via `examples/run_wind_circular.sh` → `examples/run_fdtd.py`

No detection pipeline was run on this output.

## Scenario

| Parameter | Value |
|---|---|
| Domain | `wind` (flat, uniform with wind) |
| Dimensions | 2-D, 100 m × 100 m (−50 to +50 m each axis) |
| Grid spacing | dx = 0.18 m (resolves BPF = 180 Hz at 10 ppw) |
| FD order | 2 |
| Source | Moving propeller (3 blades, 3600 RPM, 14 harmonics) |
| Source path | (−45, 0) → (45, 0) m at 50 m/s |
| Array | 16-element circular, radius 2 m, centred at origin |
| Wind | 5 m/s at 45° (vx = 3.54, vy = 3.54 m/s) |
| Simulation time | 3.0 s (9 114 time steps) |

## Files

| File | Description |
|---|---|
| `metadata.json` | FDTD simulation parameters (grid, source, receivers, wind) |
| `traces.npy` | Receiver time-series array, shape (n_receivers, n_steps) |
| `path_slices.npy` | Pressure field slices along the source path |
| `domain.png` | Domain layout showing wind field, source path, and receivers |
| `gather.png` | Receiver gather plot (waveforms at each mic) |
| `snapshots/` | Per-timestep pressure field PNGs (not tracked in git) |

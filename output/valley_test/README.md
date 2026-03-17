# Valley 2-D Test Output

Results from a 2-D FDTD acoustic simulation and detection/engagement pipeline run.

## How this was generated

1. **FDTD simulation** via `examples/run_valley.sh` → `examples/run_fdtd.py`
2. **Detection pipeline** via `examples/run_pipeline.py output/valley_test`

## Scenario

| Parameter | Value |
|---|---|
| Domain | `hills_vegetation` (two ridges with vegetation) |
| Dimensions | 2-D, 100 m × 100 m (−50 to +50 m each axis) |
| Grid spacing | dx = 0.18 m (resolves BPF = 180 Hz at 10 ppw) |
| FD order | 2 |
| Source | Moving propeller (3 blades, 3600 RPM, 14 harmonics) |
| Source path | (−40, 0) → (40, 0) m at 50 m/s, parabolic arc peaking 15 m |
| Array | 16-element circular, radius 2 m, centred at (0, 10) m |
| Simulation time | 3.0 s (39 284 time steps) |

## Pipeline results

- **22 / 23** detection windows produced valid bearings
- **Mean bearing error**: 3.2°
- **Shots fired**: 3, **Hits**: 3 (threshold 2.0 m)
- **Mean miss distance**: 1.07 m (min 0.96, max 1.16 m)
- **Real-time margin**: 6.9× (processing well under real-time)

## Files

| File | Description |
|---|---|
| `metadata.json` | FDTD simulation parameters (grid, source, receivers) |
| `results_2d.json` | Pipeline detection/tracking/engagement results |
| `traces.npy` | Receiver time-series array, shape (n_receivers, n_steps) |
| `domain.png` | Domain layout showing terrain, source path, and receivers |
| `gather.png` | Receiver gather plot (waveforms at each mic) |
| `beamformer_diagnostic_2d.png` | SRP-PHAT beamformer output vs. true bearing |
| `pipeline_summary_2d.png` | Composite summary of detection and engagement |
| `radial_engagement_2d.png` | Radial view of engagement geometry and shots |
| `snapshots/` | Per-timestep pressure field PNGs (not tracked in git) |

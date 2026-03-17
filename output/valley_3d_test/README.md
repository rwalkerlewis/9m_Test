# Valley 3-D Test Output

Results from a 3-D FDTD acoustic simulation and detection/engagement pipeline run.

## How this was generated

1. **FDTD simulation** via `examples/run_valley_3d.sh` → `examples/run_fdtd_3d.py`
2. **Detection pipeline** via `examples/run_pipeline.py output/valley_3d_test`

## Scenario

| Parameter | Value |
|---|---|
| Domain | `hills_vegetation` (two ridges with vegetation, extruded to 3-D) |
| Dimensions | 3-D, 100 × 100 × 55 m (z: −5 to +50 m) |
| Grid spacing | dx = 0.25 m (f_max ≈ 274 Hz at 5 ppw) |
| Grid shape | 221 × 401 × 401 |
| FD order | 8 |
| Source | Moving propeller (3 blades, 3600 RPM, 14 harmonics) |
| Source path | (−40, 0, 15) → (40, 0, 15) m at 50 m/s, parabolic arc in z |
| Array | 16-element circular, radius 2 m, centred at (−5, 7) m, altitude 5.5 m |
| Simulation time | 1.0 s (14 722 time steps) |

## Pipeline results

- **18 / 21** detection windows produced valid bearings
- **Mean bearing error**: 1.6°
- **Shots fired**: 6, **Hits**: 3 (threshold 2.0 m)
- **Mean miss distance**: 2.32 m (min 1.46, max 4.53 m)
- **Real-time margin**: 8.9× (processing well under real-time)

## Files

| File | Description |
|---|---|
| `metadata.json` | FDTD simulation parameters (grid, source, receivers, 3-D bounds) |
| `results_3d.json` | Pipeline detection/tracking/engagement results |
| `traces.npy` | Receiver time-series array, shape (n_receivers, n_steps) |
| `domain_xy_ground.png` | XY domain slice at ground level |
| `domain_xy_altitude.png` | XY domain slice at receiver altitude |
| `domain_xz.png` | XZ cross-section through the valley |
| `domain_xz_south_ridge.png` | XZ cross-section through the south ridge |
| `gather.png` | Receiver gather plot (waveforms at each mic) |
| `beamformer_diagnostic_3d.png` | SRP-PHAT beamformer output vs. true bearing |
| `pipeline_summary_3d.png` | Composite summary of detection and engagement |
| `radial_engagement_3d.png` | Radial view of engagement geometry and shots |
| `snapshots/` | Per-timestep pressure field PNGs (not tracked in git) |

# Scripts

Standalone tools; pytest does not collect them.

## Drone pipeline diagnostics

Most expect a dataset directory written by `examples/run_fdtd.py`, e.g. `output/valley_test`.

| Script | Purpose |
|---|---|
| `analyse_3d_traces.py` | Inspect 3-D receiver traces |
| `calibrate_rms_range.py` | Calibrate the RMS range estimator against a known source |
| `debug_accuracy.py`, `debug_tracker.py`, `debug_fire_control.py` | Step through one pipeline stage with verbose output |
| `diagnose_mfp_bpm.py`, `diagnose_3d_mfp.py` | Matched-field diagnostics (2-D and 3-D) |
| `estimate_3d_grid.py` | Memory and run time estimate for a 3-D FDTD grid |
| `run_detection_eval.py` | Detection evaluation over a dataset directory |

## Harbor benchmarks

GPU timing and accuracy harness for the harbor forward model and processing (see the Performance
section of [docs/harbor.md](../docs/harbor.md#performance)).  Results go to `output/performance/`;
run them only on an otherwise idle machine.

| Script | Purpose |
|---|---|
| `bench_harbor.py` | Wall time and accuracy against saved reference outputs for every stage 0 case (`output/performance/<tag>.json`) |
| `bench_fdtd_sweeps.py` | Stages 2 and 3: time per step against body count, throughput against batch size (`fdtd_sweeps.json`) |
| `bench_cpml.py` | Stage 4: boundary reflection of the sponge and the CPML against angle and frequency (`cpml_reflection.json`) |
| `bench_timestep.py` | Stage 6: phase accuracy and wall reflection of the time stepping prototypes (`timestep_schemes.json`) |
| `plot_performance.py` | `timings.png` from `baseline.json` and `final.json` |

## Housekeeping

| Script | Purpose |
|---|---|
| `clean_outputs.py` | List (dry run) or `--delete` gitignored generated files under `output/`; see [docs/outputs.md](../docs/outputs.md) |

## Harbor diagnostics

| Script | Purpose |
|---|---|
| `diagnose_true_waveform.py` | Why the stage 1 echo run given the true waveform and the exact track maps the UUV worse than everything estimated: maps of the ECA outputs per truth component, the recorded direct path against the true waveform, the FDTD dispersion (reads the truth) |

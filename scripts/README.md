# Diagnostic scripts

Standalone diagnostics and calibration helpers for the drone detection pipeline.
They are not tests (pytest does not collect them) and most expect a dataset directory
written by `examples/run_fdtd.py`, e.g. `output/valley_test`.

| Script | Purpose |
|---|---|
| `analyse_3d_traces.py` | Inspect 3-D receiver traces |
| `calibrate_rms_range.py` | Calibrate the RMS range estimator against a known source |
| `debug_accuracy.py`, `debug_tracker.py`, `debug_fire_control.py` | Step through one pipeline stage with verbose output |
| `diagnose_mfp_bpm.py`, `diagnose_3d_mfp.py` | Matched-field diagnostics (2-D and 3-D) |
| `estimate_3d_grid.py` | Memory and run time estimate for a 3-D FDTD grid |
| `run_detection_eval.py` | Detection evaluation over a dataset directory |
| `clean_outputs.py` | List (dry run) or `--delete` gitignored generated files under `output/`; see [docs/outputs.md](../docs/outputs.md) |

# Generated Outputs

`output/` holds only what the current docs reference or `scripts/reproduce.py` regenerates, in the layout below
(`AGENTS.md`, The output folder). Figures, JSON, JSONL, CSV, markdown and logs are committed; `*.npz`, `*.npy`, `*.mp4`
and the FDTD and run caches are not (`.gitignore`). `python scripts/clean_outputs.py --check` fails on any file outside
the layout and on any committed file that is never committed; `tests/test_figures.py` runs it. Every figure is written
by `acoustic_sim.plotting.save_figure` (one panel, at least 1600 pixels wide, text of 11 pt or more, labelled axes, a
title naming the recording).

A script that stops being part of the system or a study moves its outputs out of `output/` in the same pull request;
git history keeps them, and [Harbor, history](harbor_history.md) names the commit.

## The layout

| Folder | What it holds | Written by | Read by |
|---|---|---|---|
| `recordings/<recording>/` (`simple`, `bistatic`, `multistatic`, `multistatic_b`, `harbor`) | One recording: `dataset.observation.npz` and `dataset.truth.npz` (not committed), `scenario_used.json`, `levels.json` (levels of every component on the sensors), `truth.json`, `build.json` (FDTD wall time and runs), `run.log` / `build.log`, `map.png` (the chart, sensors and tracks), `components_<component>.png` (the band limited gather of each component), `ferry_only_map.png` (the multi body recordings), `propagation_error.json` and its figures | `examples/harbor/01_datasets/harbor_multistatic_dataset.py` (`simple`, `multistatic`, `multistatic_b`), `harbor_bistatic_dataset.py`, `harbor_m2_dataset.py` (`harbor`), each with `--out output/recordings/<recording>` (`--figures-only` redraws the figures from the cache); `harbor_propagation_error.py` (`--plot-only` redraws its figures from the JSON) | the processing configs (`examples/configs/processing/*.json`, `observation`), `scripts/reproduce.py`, the walkthrough, the studies |
| `recordings/<recording>/seed<n>/` | The test recordings: noise seeds 1 to 3 mixed from the cached gathers (not committed) | `scripts/reproduce.py` | `scripts/reproduce.py` |
| `recordings/<recording>/cache/` | The FDTD gathers (not committed) | the dataset scripts | the dataset scripts, `scripts/reproduce.py`, the studies that remix a recording |
| `reproduce/` | `summary.json` (the committed numbers of `docs/harbor.md`), `latest.json` (the last run), `tables.md`, `run.log` | `scripts/reproduce.py` (`--update` makes `latest.json` the committed `summary.json`; every run fills the `reproduce` markers of `docs/harbor.md` from it) | `docs/harbor.md`, `scripts/reproduce.py --check` |
| `reproduce/runs/<recording>/` | One run of the system: `picture.jsonl`, `summary.json`, `characteristics.json`, `tracks.csv`, `passive.csv`, `echo.csv`, the check figures (`check_<chain>_tracks.png`, the frame geometry figures), `picture_frame.png`; `picture.mp4` not committed | `scripts/reproduce.py` through `examples/harbor/05_detection/harbor_stream.py` (`run_recording`); `harbor_stream.py` alone writes the same folder; `harbor_detect.py` writes a batch run in `batch/` | `scripts/reproduce.py`, `docs/harbor.md` |
| `walkthrough/<recording>/` | One image per plot of each step of the walkthrough (`01_recording_*.png` to `08_geometry_*.png`), `07_harbor_picture.json`; the run cache `cache/run.pkl` not committed | `examples/harbor/06_walkthrough/*.py` (`run_all.sh <recording>`; `scripts/reproduce.py` deletes the cache and runs it again) | `docs/harbor.md` |
| `studies/forward_fidelity/` | The leapfrog and k space summaries, the before and after table, the stage 1 echo run given the true waveform (`stage1_bistatic/`), the true waveform diagnostic (log and its numbers) | `scripts/reproduce.py` (summaries), `examples/harbor/01_datasets/harbor_numbers_before_after.py`, `examples/harbor/90_history/harbor_picture_stage1.py --out output/studies/forward_fidelity/stage1_bistatic`, `scripts/diagnose_true_waveform.py` | `docs/harbor.md` |
| `studies/level_calibration/` | `calibration.json`: the source level constant | `examples/harbor/05_detection/harbor_level_calibration.py` | the characteristics (`water/characteristics.py` through the processing configs), `scripts/reproduce.py`, `docs/harbor.md` |
| `studies/propagation_error/<recording>/` | The propagation error with uniform water, another grid or the leapfrog scheme | `examples/harbor/01_datasets/harbor_propagation_error.py` (`--uniform-speed`, `--dx`, `--time-scheme`) | `docs/harbor.md` |
| `studies/history/<script>/` | What a history script (`examples/harbor/90_history/`, `02_forward_model/`) writes when it is run again, and its caches (the stage 0a cancellation depth gathers in `harbor_cancellation_depth/cache/`) | that script | `docs/harbor_history.md` (the committed figures of the history are in git history at the commit it names) |
| `studies/performance/` | Benchmark timings and saved runs | `scripts/bench_harbor.py`, `bench_cpml.py`, `bench_fdtd_sweeps.py`, `bench_timestep.py`, `plot_performance.py` | `docs/harbor_history.md` (the performance work) |
| `studies/<study>/` | One folder per study: its `results.json`, figures and `run.log` | the study's script | the study's page in `docs/` |
| `air/<scenario>/` (`valley_test`, `valley_3d_test`, `isotropic_2D`, `erratic_quadcopter`) | Drone pipeline reference runs: `metadata.json`, `results_2d.json` / `results_3d.json`, `README.md` with the scenario; traces not committed | traces: `examples/air/01_simulation/run_valley.sh`, `run_valley_3d.sh`, `run_wind_circular.sh`, `run_fdtd_erratic.py` or `generate_traces.py`; results: `examples/air/02_pipeline/run_pipeline.py output/air/<scenario>` | `docs/usage.md`, `README.md` |
| `air/models/` | The drone pipeline's trained weights (`acoustic`, `fusion`, `maneuver` committed) | `examples/air/03_ml/train_all_ml.py` (the `[ml]` extra) | the air pipeline (`examples/configs/pipeline.config.json`), `src/acoustic_sim/ml/` |
| `air/comparison_results.json` | Baseline against ML pipeline | `examples/air/02_pipeline/run_comparison.py` | `docs/usage.md` |
| `reports/` | The reports of a sequence of tasks (`run_all.md`) | written by hand at the end of each task | the reader |

## Cleaning up generated files

`scripts/clean_outputs.py` lists the ignored, untracked files under `output/` with their size per directory. It is a
dry run unless `--delete` is given, it only considers files git reports as ignored (so committed results are never
candidates), and it refuses paths outside `output/`.

```bash
python scripts/clean_outputs.py --check                          # every file inside the layout?
python scripts/clean_outputs.py                                  # dry run: everything regenerable
python scripts/clean_outputs.py recordings/bistatic/cache        # dry run: one subtree
python scripts/clean_outputs.py recordings/bistatic/cache --delete
```

Delete run caches before datasets and datasets before FDTD caches: rebuilding a recording's gathers means running the
FDTD again, which takes from minutes to hours on a GPU.

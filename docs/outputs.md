# Generated Outputs

Everything under `output/` is written by the example and benchmark scripts.  A small set of
results is committed as deliverables; the bulk (datasets, FDTD caches, saved runs) is
gitignored and can be regenerated.  A full local tree is typically 10+ GB, almost all of it
ignored.

## What is tracked and what is generated

| Kind | Tracked? | Examples |
|------|----------|----------|
| Run metadata and results (`*.json`, per-run `README.md`) | yes | `valley_test/metadata.json`, `harbor_bistatic/*.json`, `performance/final.json` |
| Run logs (`run.log`, `*.log` next to results) | yes, except `performance/.*.log` | `harbor_multistatic/run.log` |
| Selected figures and animations (`*.png`, `*.gif`) | the committed ones only (force-added; new figures are ignored by `*.png`) | `harbor/m3_pd_vs_level.png`, `performance/timings.png` |
| Trained drone-pipeline ML weights | yes | `models/{acoustic,fusion,maneuver}_classifier.pt` |
| Traces, datasets, saved maps (`*.npy`, `*.npz`) | no | `harbor/dataset.npz`, `harbor_bistatic/eval/maps_*.npz` |
| FDTD caches and snapshots (`*/cache/`, `*/snapshots/`) | no | `harbor_bistatic/cache/`, `performance/runs/` |
| Other checkpoints and videos (`*.pt`, `*.mp4`) | no, unless listed above | `harbor/cnn_classifier.pt` |

The rules live in `.gitignore`.  `git ls-files output` lists the tracked deliverables;
`git status --ignored output` lists the generated files present locally.  Do not untrack committed
results or figures without agreeing it first: documentation and the performance write-up refer to
them.

## Regenerating key artifacts

Run commands from the repository root (prefix with `./dev.sh run` to use the dev container; harbor
FDTD runs want `--cuda`/a GPU for practical run times).  Each script's `--help` lists its options.

| Directory | Producer | Notes |
|-----------|----------|-------|
| `valley_test`, `valley_3d_test`, `isotropic_2D`, `erratic_quadcopter` | `examples/run_valley.sh`, `run_valley_3d.sh`, `run_pipeline_isotropic.sh`, `run_pipeline_erratic.sh` (FDTD), then `examples/run_pipeline.py output/<dir>` | Drone pipeline reference runs; each tracked directory has a `README.md` with its scenario |
| `models` | `examples/train_all_ml.py` | Needs the `[ml]` extra |
| `comparison_results.json`, `baseline_metrics.json` | `examples/run_comparison.py` | Baseline vs ML pipeline |
| `harbor` | `examples/harbor_m1_demo.py`, `harbor_m2_dataset.py` (writes `dataset.npz`), `harbor_m3_detect.py`, `harbor_m4_dlc.py` | `dataset.npz` is the input of most other harbor scripts |
| `harbor/cnn_classifier.pt`, `cvae_anomaly.pt` | `examples/harbor_build_classifier_dataset.py`, `harbor_train_classifier.py`, `harbor_train_anomaly.py` | Needs `[ml]`; not tracked |
| `harbor_demo` | `examples/harbor_mfp_demo.py` | Analytic forward model, CPU only |
| `harbor_domain`, `harbor_domain_video`, `harbor_presentation` | `examples/harbor_mfp_domain.py`, `harbor_domain_video.py`, `harbor_presentation.py` | Read `output/harbor/dataset.npz` |
| `harbor_m1_uuv` | `examples/harbor_m1_uuv.py`, `harbor_m1_uuv_video.py` | |
| `harbor_bistatic` | `examples/harbor_bistatic_budget.py`, `harbor_bistatic_dataset.py --coarse`, `harbor_bistatic_demo.py`, `harbor_bistatic_ablation.py` | FDTD cache in `harbor_bistatic/cache/` |
| `harbor_cancel` | `examples/harbor_cancellation_depth.py` | Cache in `harbor_cancel/cache/` |
| `harbor_multistatic` | `examples/harbor_multistatic_dataset.py` | |
| `harbor_shore0.5`, `harbor_shore1.0` | `examples/harbor_m2_dataset.py --config <variant> --out output/harbor_shore<x>` | Figures only; the shoreline-loss variant configs (`shore_loss_*` in `examples/harbor.scenario.json`) were not committed |
| `performance` | `scripts/bench_harbor.py` (timings JSON, saved runs), `scripts/plot_performance.py` (`timings.png`), `scripts/bench_cpml.py`, `bench_fdtd_sweeps.py`, `bench_timestep.py` | Benchmark JSON is a measured result: never edit it by hand |

## Cleaning up generated files

`scripts/clean_outputs.py` lists the ignored, untracked files under `output/` with their size per
directory.  It is a dry run unless `--delete` is given, it only considers files git reports as
ignored (so tracked deliverables are never candidates), and it refuses paths outside `output/`.

```bash
python scripts/clean_outputs.py                            # dry run: everything regenerable
python scripts/clean_outputs.py harbor_bistatic/cache      # dry run: one subtree
python scripts/clean_outputs.py harbor_bistatic/cache --delete
```

Delete caches and saved runs before datasets: regenerating `output/harbor/dataset.npz` or a
bistatic dataset means re-running the FDTD, which takes from minutes on a GPU to hours on a CPU.

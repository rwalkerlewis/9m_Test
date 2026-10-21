---
mode: agent
description: Branch figures_output from origin/main. You own src/acoustic_sim/plotting.py (new, shared), src/acoustic_sim/water/plott
---
Branch `figures_output` from `origin/main`. You own `src/acoustic_sim/plotting.py` (new, shared), `src/acoustic_sim/water/plotting.py`, `src/acoustic_sim/air/plotting.py`, `air/pipeline_plots.py`, every figure producing function in `examples/`, `scripts/reproduce.py` (its figure and output paths and the prose markers), `tests/test_figures.py`, the whole of `output/`, and the figure links and the prose of `docs/harbor.md`, `docs/harbor_history.md`, `docs/physics.md`, `docs/outputs.md`, `docs/usage.md`. You do not change any processing or forward model code, any number, or any processing config. This task runs no FDTD; it reruns the chain and the walkthrough only to regenerate figures.

Read `AGENTS.md` (Figures, The output folder, the reproduce marker rule), `docs/harbor.md`, `docs/outputs.md`, `scripts/reproduce.py`, `examples/harbor/06_walkthrough/_common.py`, `src/acoustic_sim/water/plotting.py`, and look at every figure under `output/` before changing anything.

## 1. The figure rule, enforced

Write `acoustic_sim/plotting.py` with `save_figure(fig, path, *, allow_two_panels=False)`: refuses (raises) a figure with more than one axes (two when the flag says the panels share an axis), a figure narrower than 1600 pixels at its dpi, any text below 11 pt, a missing axis label on either axis, a colour bar without a label, a title that does not name the recording; applies a consistent style (one font family, tight layout, 150 dpi); writes the PNG. Every `savefig` in `src`, `examples` and `scripts` goes through it. `tests/test_figures.py`: a static scan of those trees for `plt.subplots(` with `nrows * ncols > 2`, for `savefig(` outside `plotting.py`, and for figures saved without `save_figure`; plus unit tests of each refusal.

## 2. Split every figure

Every figure that is a grid of panels becomes one figure per panel, named `<figure>_<panel>.png` with a title that says which recording and which quantity. The list on `main` today, at least: `components.png` (seven panels: one per component), `map.png` with insets, `check_*_tracks.png` and `check_*_frame.png` (the frame row becomes its own figures: the map, the ray paths, the ellipse), `picture_frame.png`, `pd_summary.png`, `misses_*.png`, `scoreboard_*.png`, `propagation_error*.png` (one per variant, and the per hydrophone bars on their own), `before_after` figures, the `m1`, `m2`, `m3` figures of the 5 m harbor, `analytic_vs_fdtd.png`, `maps_by_config.png`, `budget.png`, the air pipeline summary figures. Overlapping labels (the `check_*_tracks.png` track names on top of each other) are offset or replaced by a numbered legend. Map figures get metres on both axes, north up, the sensors drawn, and a legend that names every marker type; time series get seconds and the unit of the trace.

Regenerate every figure the docs reference by rerunning the walkthrough and `scripts/reproduce.py` (processing unchanged, no `--update` of the numbers; if a number moves, stop and report).

## 3. Clean the output folder

Restructure to the layout in `AGENTS.md`:

- `output/recordings/<recording>/` for `harbor`, `harbor_simple`, `harbor_bistatic`, `harbor_multistatic`, `harbor_multistatic_b` (their `fine/` level is flattened unless two grids exist; the seed folders of the 5 m harbor move with it). Update every path in the processing configs, `scripts/reproduce.py`, the examples and the docs; the observation and truth files stay where the configs point, so point the configs at the new folders.
- `output/reproduce/` stays. `output/harbor_stream/` and `output/harbor_detect/` duplicate `output/reproduce/runs/` and are deleted.
- `output/studies/forward_fidelity/`, `output/studies/level_calibration/`, `output/studies/propagation_error/` for what the current docs cite.
- `output/air/<scenario>/` for `valley_test`, `valley_3d_test`, `isotropic_2D`, `erratic_quadcopter`, `models`, and the three root level files `comparison_results.json`, `iteration_log.md`, `baseline_metrics.json` (delete the last two if nothing references them).
- Deleted, with the commit named in `docs/harbor_history.md` beside each history figure link: `harbor_demo`, `harbor_domain`, `harbor_domain_video`, `harbor_shore0.5`, `harbor_shore1.0`, `harbor_cancel`, `harbor_m1_uuv`, `harbor_presentation`, `harbor_picture`, `performance`, the `ablation`, `demo` and `coarse` folders of the bistatic and multistatic recordings, and the `m1`, `m2`, `m3` figures of the 5 m harbor unless the history page shows them (if it does, keep only those and move them under `output/studies/history/`). The history scripts under `examples/harbor/90_history/` keep running and write under `output/studies/history/<script>/`.
- `docs/outputs.md` is rewritten as the map of the new layout: every folder, what writes it, what reads it.
- `scripts/clean_outputs.py` becomes the tool that checks the layout (fails on any file outside it) and is run by `tests/test_figures.py`.

## 4. The prose

`docs/harbor.md` quotes numbers in prose that the tables no longer support (the status paragraph and the "What they say" bullets still describe the leapfrog recordings: the tug at 18 % on `multistatic_b`, the UUVs not tracked on the multistatic recordings, false tracks of 2 to 12 per minute). Add the `<!-- reproduce:<key> -->` marker mechanism to `scripts/reproduce.py` (keys into `summary.json`, filled on every run, checked by `--check`), convert every number in prose to a marker, and rewrite the interpretive sentences against the current numbers; where a sentence cannot be made true from `summary.json`, delete it. The status paragraph at the top states the current numbers and the current criteria results.

## Rules

The repository rules in `AGENTS.md` apply. No number changes; `scripts/reproduce.py --check` passes at the end. Stop after section 4. Report the figure count before and after, every deleted output folder, and every sentence of prose that was deleted because the numbers no longer supported it.

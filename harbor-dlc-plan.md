> **Historical plan — not current instructions.** This plan was written against the former
> `src/acoustic_sim/harbor/` layout. That code now lives in `src/acoustic_sim/water/`
> (`acoustic_sim.harbor` remains only as a compatibility import alias); the paths below are
> kept as originally written. See `docs/harbor.md` for the current pipeline.

# Harbor DAS Detect–Locate–Classify Improvement Plan

## Overview

**Goal**: Extend the existing harbor simulation codebase so that FDTD-simulated DAS + hydrophone
traces from the 2D harbor scenario can be used to reliably *detect*, *locate*, and *classify*
unknown submerged objects — with "unknown" meaning anything whose acoustic signature or scattering
profile does not match the labelled AIS fleet (ferry, tug, moored).

**Approach**: Cascade pipeline in three stages:

1. **Detection** — improve passive Pd / false-alarm rate using AIS-aware suppression (already
   partially wired), tighter per-band CFAR, and line-enhancement tuning.
2. **Localisation** — two-stage refinement: fast analytic replica first-pass (existing MFP), then
   empirical (FDTD-computed) replica matching for sub-grid accuracy.
3. **Classification** — CNN for coarse known / unknown discrimination trained on M2 dataset labels,
   followed by a CVAE anomaly scorer for fine-grained "how different from all known ships" scoring.

**Scope**:
- Extend the M2 dataset generator with more unknown-class objects.
- Wire `suppression.run_ais_aware` into the M3 detection example end-to-end.
- Add a two-stage localisation module to the harbor package.
- Train and evaluate the CNN + CVAE classifier stack.
- Produce a new end-to-end example `harbor_m4_dlc.py` exercising the full pipeline.

**Non-goals**:
- 3D simulation, real-water deployment, real-time streaming.
- Changing the FDTD solver itself or any existing unit tests.
- GPU-only paths (all changes must fall back to CPU).

---

## Sub-Task 1 — Extend M2 Dataset with Unknown-Class Objects

**Status**: `[x] done`

### Intent
The current M2 dataset has four vessels: ferry (AIS), tug (AIS), moored (AIS), uuv1 (hidden).
The classifier needs examples of multiple distinct unknown-class acoustic profiles and scattering
bodies at training time. Add three more unknown objects with clearly different spectral identities:

- `uuv2`: a second UUV at a different depth slice (modelled in 2D as a different path), faster
  (3.5 knot), higher fundamental tonal (50–65 Hz range), level 0.05 Pa.
- `diver`: a nearly-silent swimmer profile — broadband bubble noise, very slow (0.8 knot), 0.01 Pa.
- `static_scatter`: a stationary **hard-body** scatterer (mine-like, rigid boundary) with
  `level_pa = 0` (passive reflector only); present in bistatic contributions but not in
  direct-path gathers. The `body_kind` field on `VesselConfig` defaults to `"hard"`.
  A companion example `harbor_m1_uuv_soft.py` demonstrates the same geometry with
  `body_kind = "soft"` so the two scattering regimes can be compared side-by-side.

### Expected Outcomes
- `scenario.py` default vessel list gains `uuv2`, `diver`, `static_scatter` entries,
  with `static_scatter.body_kind = "hard"`.
- `signatures.py` gains `diver_reference()` function (broadband, bubble-noise spectral slope).
- The `make_reference_signal()` dispatcher handles `kind == "diver"`.
- Running `harbor_m2_dataset.py` produces a `.npz` with the additional tracks / gathers.
- `m2_scenario.png` shows all seven objects.
- New example `examples/harbor_m1_scatterer_compare.py` runs hard-body and soft-body FDTD
  back-to-back for the same geometry and produces a side-by-side figure of the scattered fields.

### Todo List
1. Add `diver_reference(speed, fs, rng, f_max)` to `src/acoustic_sim/harbor/signatures.py`.
2. Add `"diver"` branch to `make_reference_signal()` in `signatures.py`.
3. Append `uuv2`, `diver`, `static_scatter` (hard body default) entries to `default_vessels()`
   in `scenario.py`; confirm `VesselConfig` already carries `body_kind` (check `scatterer.py`).
4. Write `examples/harbor_m1_scatterer_compare.py`:
   - Runs FDTD once with `body_kind = "hard"` and once with `body_kind = "soft"`.
   - Computes difference fields and plots: 3-panel figure (hard scattered / soft scattered / ratio dB).
5. Ensure `harbor_m2_dataset.py` plots all vessels in `m2_scenario.png` (no code change needed
   if the plotting loop already iterates `tracks`).
6. Run `harbor_m2_dataset.py` to verify the extended dataset is produced without errors.

### Relevant Context
- `src/acoustic_sim/harbor/signatures.py` — `ship_reference()`, `uuv_reference()`, `make_reference_signal()`.
- `src/acoustic_sim/harbor/scenario.py` — `default_vessels()`, `VesselConfig`.
- `src/acoustic_sim/harbor/scatterer.py` — body kind handling (hard vs. soft boundary).
- `examples/harbor_m1_uuv.py` — reference pattern for the scatterer compare example.
- `examples/harbor_m2_dataset.py` — dataset assembly driver.

---

## Sub-Task 2 — Wire AIS-Aware Suppression into M3 Detection and Tune It

**Status**: `[x] done`

### Intent
`suppression.run_ais_aware` is fully implemented but `harbor_m3_detect.py` uses only the bare
`run_detection` path (no AIS suppression). The known-ship adaptive projection should dramatically
reduce false alarms and unmask the quiet UUV. Additionally, the `line_halfwidth` enhancement in
`ProcessingConfig` (currently disabled, `= 0`) should be turned on to favour tonal UUV lines
over broadband ship leakage, and the CFAR tail-quantile should be validated for the new object count.

### Expected Outcomes
- M3 uses `run_ais_aware` for the fused and DAS-only configurations.
- `m3_results.json` shows improved Pd and lower FA/frame compared to the pre-suppression baseline.
- A new CLI flag `--suppress` (default on) controls the AIS-aware path so the bare path can be
  benchmarked for comparison.
- `m3_pd_vs_level.png` shows fused-suppressed as a separate curve alongside fused-baseline.

### Todo List
1. Import `run_ais_aware`, `SuppressionConfig` into `harbor_m3_detect.py`.
2. Add `--suppress / --no-suppress` argument (default `True`).
3. When suppression is on, call `run_ais_aware(traces, layout, fs, env, ds.ais, pc, sc)` instead
   of `run_detection`, using the same `grid`.
4. Set `line_halfwidth = 4` (4 bins each side) in `ProcessingConfig` for the UUV-detection sweep.
5. Add the suppressed-fused curve to `m3_pd_vs_level.png`.
6. Re-run the CFAR calibration with noise-only data (seeds 1000–1003) to re-derive the threshold
   with the extended object set; assert the new threshold is within 20% of the old value.
7. Validate: Pd at nominal UUV level (1.0×) in fused-suppressed mode must improve vs. fused-baseline.

### Relevant Context
- `examples/harbor_m3_detect.py` — lines 83–127 (main detection loop).
- `src/acoustic_sim/harbor/suppression.py` — `run_ais_aware()`, `SuppressionConfig`.
- `src/acoustic_sim/harbor/detection.py` — `ProcessingConfig.line_halfwidth`.
- `src/acoustic_sim/harbor/dataset.py` — `HarborDataset.ais` field (AISReports).

---

## Sub-Task 3 — Two-Stage Localisation Module

**Status**: `[x] done`

### Intent
The current scan grid uses 20 m cells with a 5 m refinement window. For multi-path environments
the analytic free-field replica mismatches the actual FDTD-propagated field, causing position bias
of 20–60 m. Implement a two-stage localiser:

**Stage 1 (fast)**: existing analytic MFP on the coarse grid → candidate set of cells.
**Stage 2 (empirical)**: for each candidate, look up the FDTD-computed Green's function at that
cell and cross-correlate with the observed data — provides sub-grid accuracy without re-running FDTD.

Two empirical replica sources are supported, selectable at runtime:

- **Vessel-path interpolation** (fast, default): the existing M2 per-vessel unit gathers are FDTD
  responses from known source *tracks*. Given a candidate grid cell, bilinearly interpolate the
  transfer function from the nearest track positions. Works wherever a cached vessel passed nearby.
- **Grid survey** (accurate, optional, expensive): a separate pre-computation step fires a Ricker
  pulse from every node of a uniform source grid (default 20 m spacing, same as the scan grid),
  records at all sensors, and stores the complex transfer functions. This gives an empirical replica
  at every candidate cell regardless of vessel paths. Triggered by `--grid-survey` in the M3
  example; the survey cache is stored alongside the vessel cache.

### Expected Outcomes
- New file `src/acoustic_sim/harbor/localiser.py` with `EmpiricalLocaliser` class.
- `EmpiricalLocaliser` supports both `mode="vessel_path"` and `mode="grid_survey"`.
- `EmpiricalLocaliser.refine(candidates_xy, wh, frame)` returns a refined `(x, y)` per candidate.
- `harbor_m3_detect.py` accepts `--empirical-refine [vessel_path|grid_survey]`.
- Grid-survey pre-computation exposed as a separate script step
  `examples/harbor_survey_grid.py` (so it can be run once and reused).
- Localisation RMSE at nominal UUV level in fused-suppressed mode decreases vs. the 5 m analytic
  refinement baseline (target: < 15 m RMSE for ranges up to 300 m with vessel-path mode;
  < 10 m with grid-survey mode).

### Todo List
1. Write `src/acoustic_sim/harbor/localiser.py` with `EmpiricalLocaliser`:
   - `__init__(cache_dir, layout, fs, pc, mode)` — loads the appropriate replica store.
   - `_load_vessel_path_replicas(cache_dir, fs, pc)`: reads cached unit gathers, computes STFT,
     stores as dict `{vessel_id: (T_positions_xy, N_receivers, F_bins) complex array}`.
   - `_load_grid_survey_replicas(cache_dir, fs, pc)`: reads survey cache `.npz` files produced
     by `harbor_survey_grid.py`, stores as `(G_survey, N, F)` complex array + grid coordinates.
   - `refine(candidates_xy, wh, frame_idx)`: interpolate replica to each candidate cell,
     score by normalised matched-field inner product with `wh.Z[:, :, frame_idx]`, return peak.
2. Write `examples/harbor_survey_grid.py`:
   - Accepts `--grid-dx` (default 20 m), `--out` (cache dir), `--time` (default 12 s).
   - Iterates over all water grid cells, fires one `StaticSource` Ricker pulse per cell,
     records all-receiver traces, saves STFT to `survey_cache/{ix}_{iy}.npz`.
   - Parallelises over cells with `concurrent.futures.ProcessPoolExecutor`.
3. Add `empirical_localiser` optional field to `ProcessingConfig` (type `EmpiricalLocaliser | None`,
   default `None`) and call `localiser.refine(...)` inside `extract_detections` after the
   existing analytic refinement step when the field is set.
4. Add `--empirical-refine [vessel_path|grid_survey]` flag to `harbor_m3_detect.py`;
   when `grid_survey` is chosen, assert the survey cache exists or error with a helpful message.
5. Evaluate RMSE for: analytic-only, vessel-path empirical, grid-survey empirical;
   write all three to `m3_results.json`.

### Relevant Context
- `src/acoustic_sim/harbor/detection.py` — `extract_detections()`, `ProcessingConfig`.
- `examples/harbor_m1_demo.py` — `StaticSource` + `HarborFDTDSolver.run_with_snapshots`
  pattern used for per-cell survey runs.
- `examples/harbor_m3_detect.py` — `render_vessel_gathers` call (line 72); cache at `out/cache/`.
- `src/acoustic_sim/harbor/sensors.py` — `SensorLayout`.

---

## Sub-Task 4 — Dataset Preparation for the Classifier

**Status**: `[x] done`

### Intent
The CNN and CVAE need training examples of isolated per-source DAS traces (not the mixed signal).
The M2 dataset already stores `clean` (oracle FDTD sum) and per-vessel gathers in the cache.
Add an extraction utility that slices the cached unit gathers into labelled spectral windows
suitable for the classifiers.

To build a corpus large enough for robust training, run M2 across multiple random seeds
(controlling noise draws, track wander, AIS jitter) so each vessel's signature is sampled under
diverse noise realisations. A `--n-seeds` CLI argument controls the number of seeds; default is 8
(≈ 37,000 training windows for the ship class after oversampling).

Labels:
- `"ship"` — ferry, tug, moored.
- `"uuv"` — uuv1, uuv2.
- `"diver"` — diver.
- `"unknown"` — static_scatter bistatic contributions.

Each example is a (n_mels=64, n_time_frames) mel-spectrogram extracted from one 4-second window
of one DAS channel's single-vessel trace.

### Expected Outcomes
- New script `examples/harbor_build_classifier_dataset.py` that:
  - Loads the M2 gather cache for seeds `0 … n_seeds-1`; re-runs M2 for any missing seeds.
  - Extracts per-vessel per-channel mel-spectrogram windows with sliding 50% overlap.
  - Saves `classifier_dataset.npz` with `X` (N, 1, 64, T), `y` (N,) int labels, `label_names`,
    `seed` (N,) per-example seed index.
- The dataset has balanced classes (oversample minority classes to match ship count).
- `examples/harbor_train_classifier.py` trains the CNN on this dataset, saves
  `output/harbor/cnn_classifier.pt`.

### Todo List
1. Write `examples/harbor_build_classifier_dataset.py`:
   - Accept `--n-seeds N` (default 8) and `--cache-dir` args.
   - For each seed, call `render_vessel_gathers` (which uses the cache, re-running only on miss).
   - Apply `compute_mel_spectrogram` from `src/acoustic_sim/ml/features.py` with a 4-second
     Hann window and 50% overlap.
   - Collect `(mel_spec, label, seed)` tuples for all vessels, all channels, all seeds.
2. Implement class-balanced oversampling (random repeat of minority-class windows).
3. Save `classifier_dataset.npz` with arrays `X`, `y`, `label_names`, `seed`.
4. Write `examples/harbor_train_classifier.py` using `AcousticClassifier(n_classes=4)` from
   `src/acoustic_sim/ml/acoustic_classifier.py`; train with cross-entropy, 80/20 seed-stratified
   split (whole seeds in test, not individual windows), save checkpoint and print per-class accuracy.

### Relevant Context
- `src/acoustic_sim/ml/features.py` — `compute_mel_spectrogram()`.
- `src/acoustic_sim/ml/acoustic_classifier.py` — `AcousticClassifier`.
- `src/acoustic_sim/harbor/dataset.py` — `HarborDataset`, `render_vessel_gathers` cache format.

---

## Sub-Task 5 — CVAE Anomaly Scorer

**Status**: `[x] done`

### Intent
The `AnomalyDetector` CVAE in `src/acoustic_sim/ml/anomaly_detector.py` exists but is trained and
evaluated only on generic data. Re-purpose it as a "normal-ship distribution" model: train it on
mel-spectrograms labelled `"ship"` only, so its reconstruction error is low for ferry/tug/moored
and high for anything else. The CVAE anomaly score becomes the second-stage classifier output.

### Expected Outcomes
- Script `examples/harbor_train_anomaly.py` trains the CVAE on ship-only windows from
  `classifier_dataset.npz`, saves `output/harbor/cvae_anomaly.pt`.
- `src/acoustic_sim/harbor/classifier.py` (new file) provides:
  - `HarborClassifier` wrapping both the CNN and CVAE.
  - `classify(mel_spec)` → `{"label": str, "cnn_logits": array, "anomaly_score": float}`.
- `harbor_m4_dlc.py` uses `HarborClassifier` after detections are localised.

### Todo List
1. Read `src/acoustic_sim/ml/anomaly_detector.py` fully to understand the CVAE interface.
2. Write `examples/harbor_train_anomaly.py`:
   - Load `classifier_dataset.npz`, filter to ship-only rows.
   - Train `AnomalyDetector` (or the existing `anomaly_training.py` entrypoint) on ship windows.
   - Save checkpoint.
3. Write `src/acoustic_sim/harbor/classifier.py` with `HarborClassifier`:
   - Loads CNN and CVAE checkpoints.
   - `classify(mel_spec)` runs CNN forward pass, then CVAE reconstruction error.
   - Returns structured result dict.
4. Unit-test `HarborClassifier` with a random input to confirm forward pass shapes are correct.

### Relevant Context
- `src/acoustic_sim/ml/anomaly_detector.py`, `anomaly_integration.py`, `anomaly_training.py`.
- `src/acoustic_sim/ml/acoustic_classifier.py`.
- Sub-Task 4 outputs (`classifier_dataset.npz`, `cnn_classifier.pt`).

---

## Sub-Task 6 — End-to-End Example: `harbor_m4_dlc.py`

**Status**: `[x] done`

### Intent
Produce a single runnable example that exercises the full detect–locate–classify pipeline on
the extended M2 dataset. This is the deliverable that proves the system works end-to-end.

**Pipeline**:
```
M2 dataset (.npz)
  ↓
AIS-aware suppression + STFT pre-processing
  ↓
Analytic MFP first-pass (coarse candidates, ≤ 20 m grid)
  ↓
Empirical replica refinement (FDTD cache)
  ↓
CFAR threshold → DetectionSet
  ↓
Per-detection mel-spectrogram extraction (nearest DAS channel)
  ↓
CNN coarse label + CVAE anomaly score
  ↓
Output: track map + classification overlay + ROC curve
```

### Expected Outcomes
- `harbor_m4_dlc.py` runs without errors from the CLI.
- `output/harbor/m4_track_map.png`: harbor map with colour-coded detection labels
  (green = known ship, red = unknown, size ∝ anomaly score).
- `output/harbor/m4_roc.png`: ROC curve (unknown vs. known) using CVAE anomaly score as the
  discriminant, evaluated on ground-truth labels.
- `output/harbor/m4_summary.json`: per-object Pd, RMSE, classifier accuracy.

### Todo List
1. Write `harbor_m4_dlc.py` importing from all prior sub-tasks.
2. Implement mel-spectrogram extraction at each detected position: find the nearest live DAS
   channel, slice the 4-second window centred on the detection time, compute mel-spectrogram.
3. Batch-classify all detections with `HarborClassifier`.
4. Generate `m4_track_map.png` using `plot_harbor` with detection scatter coloured by CNN label.
5. Compute ROC: CVAE anomaly score as the continuous discriminant; label = `uuv`/`diver`/
   `static_scatter` as positive, `ship` as negative; compute AUC and plot.
6. Write `m4_summary.json`.

### Relevant Context
- All prior sub-tasks.
- `src/acoustic_sim/harbor/plotting.py` — `plot_harbor()`.
- `examples/harbor_m3_detect.py` — reference for the detection loop pattern.

---

## Implementation Instructions (for Agent Mode)

Process each sub-task independently using `start_subtask`. Before starting each sub-task, the
agent must:
1. Read this plan file to load full context.
2. Read the files listed under **Relevant Context** for that sub-task.
3. Implement only the steps in that sub-task's **Todo List**.
4. After completion, update the sub-task **Status** to `[x] done` in this file.
5. Wait for user approval before proceeding to the next sub-task.

Sub-tasks must be executed in order (1 → 2 → 3 → 4 → 5 → 6) because each depends on outputs
of the previous one.

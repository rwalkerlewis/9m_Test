Work in the 9m_Test repo on a new branch `harbor_picture_2` from `main` (`git fetch origin && git checkout -b harbor_picture_2 origin/main`). `main` is the only line of development. Use the GPU for every FDTD run and every chain, with CPU fallback.

Read `docs/architecture.md`, `docs/harbor.md` (the walkthrough and "What the chains find today"), `src/acoustic_sim/water/streaming.py`, `src/acoustic_sim/water/scoreboard.py`, `src/acoustic_sim/water/detection.py`, `src/acoustic_sim/water/bistatic.py`, `src/acoustic_sim/water/evaluate.py`, `examples/harbor/05_detection/harbor_stream.py`, `examples/harbor/05_detection/harbor_detect.py`, `examples/harbor/06_walkthrough/_common.py`, `examples/README.md`, `tests/harbor/test_streaming.py`, `tests/harbor/test_scoreboard.py`, and `output/harbor_stream/*/summary.json` before changing anything.

## Where main stands

The real time chain in `water/streaming.py` (`StreamingPassive`, `RadiatorTracker`, `StreamingEcho`, AIS labels afterwards) is the water processing chain: it locates from the traces alone and it is the one that embodies the goal. Everything else on the water processing side is either an older version of a step (the batch passive chain with AIS in the loop, the batch echo chain, the occupancy filter, the empirical localiser, the M3 and M4 scripts, the CNN and CVAE classifiers) or an experiment that produced a number in the docs (link budget, cancellation depth, ablation, stage 1). A reader cannot tell which code is the system and which is history, the parameters of the system live in dictionaries inside three scripts, the numbers in the docs were measured on the same four recordings the parameters were set on, there is no script that reproduces them, and the chain reports positions but nothing about what the objects are.

This stage fixes those four things and adds the first operator facing output. It adds no new signal processing except the characteristics in section 6.

## 1. One chain, and history out of the way

1. The library's water processing side is `streaming.py` plus what it calls (`detection.py` spectra, whitening, scan and interference projector; `bistatic.py` reference beam, ECA, echo map; `propagation.py`; `sensors.py`, `ais.py`). Make `harbor_detect.py` run the same streaming objects over the whole record (a batch run is the stream with one chunk) or delete it; there must be one implementation of each step.
2. Move to `examples/harbor/90_history/` every script that is not the chain, the datasets, the forward model demos or the walkthrough: `harbor_m3_detect.py`, `harbor_m4_dlc.py`, `harbor_mfp_demo.py`, `harbor_mfp_domain.py`, `harbor_train_classifier.py`, `harbor_train_anomaly.py`, `harbor_build_classifier_dataset.py`, `harbor_survey_grid.py`, `harbor_bistatic_demo.py`, `harbor_bistatic_ablation.py`, `harbor_cancellation_depth.py`, `harbor_bistatic_budget.py`, `harbor_presentation.py`, `harbor_stack_movie.py`, `harbor_picture_stage1.py`. They stay runnable (fix their paths, keep their tests) because committed figures came from them, and `examples/README.md` lists them under "History" with one line each. Library modules used only by them (`occupancy.py`, `localiser.py`, `classifier.py`, `suppression.py` if the stream no longer calls it) are listed under "History" in `docs/architecture.md`; delete nothing in this stage.
3. `docs/harbor.md` becomes two files: `harbor.md` (what the system does, how to run it in three commands, the current numbers from section 3, the walkthrough, the operator output) and `harbor_history.md` (scenario details that only the old scripts use, stage 0, stage 1, the performance work, the findings tables). Every number that stays in `harbor.md` is one the reproduce script of section 3 regenerates.

## 2. Configuration is data, results carry their provenance

1. `examples/configs/processing/<recording>.json` for `simple`, `bistatic`, `multistatic`, `harbor`: every processing parameter in one file (band, `ProcessingConfig`, `StreamConfig`, `TrackerConfig`, `EchoStreamConfig`, `BistaticConfig`, the echo tracker of section 5, scan edge, chunk length). `harbor_stream.py`, `harbor_detect.py` and the walkthrough read it; the `DATASETS`, `TRACKER` and `ECHO` dictionaries go away. A `--set key=value` override exists for experiments and is recorded.
2. Every `summary.json`, `results.json` and `picture.jsonl` header carries a `run` block: git commit and whether the tree was dirty, the processing config file and its sha256, the observation file and its sha256, Python, NumPy and CuPy versions, GPU name or "cpu", the seed, wall time, and the `Provenance` block. Add `contract.run_info()` that builds it and a test that every entry point writes it.

## 3. Reproduce

1. `scripts/reproduce.py`: builds any missing recording (the four datasets plus the test recordings of section 4, with their FDTD caches), runs the chain on every recording (tuning and test), regenerates the walkthrough figures, and writes `output/reproduce/summary.json` with every number that `docs/harbor.md` quotes. Then it compares each number against the committed `output/reproduce/summary.json` and prints a table with pass or fail per number at stated tolerances: fraction of frames found within 3 percentage points, median error within 0.5 m, false tracks exactly, compute time and latency informational. `--check` only compares; `--update` rewrites the committed file (the PR says when it was run and on what GPU).
2. CI: a tiny scenario (50 by 50 m, 1 m grid, 2 s, one ferry, one wreck) through dataset build, stream and the characteristics of section 6 on the CPU in under three minutes, asserting the ferry is tracked within 3 m and the wreck appears in the echo map. This is the only end to end run CI does; the full reproduce is run by hand on the GPU.

## 4. Numbers on recordings the parameters never saw

Every constant in `TrackerConfig`, `StreamConfig` and `EchoStreamConfig` was set while looking at the four recordings. Freeze them now (they are the processing configs of section 2) and measure on recordings they have not seen:

1. Noise seeds 1 to 3 of each of the four datasets, mixed from the cached gathers (`assemble_dataset(seed=...)`; no FDTD).
2. One new geometry: `examples/configs/harbor_multistatic_b.scenario.json`, the multistatic harbor with every track changed (different headings, crossing points and speeds, the pilings and the hull moved, the boat starting on the other side), same sensors, same levels. Build it (six FDTD runs on the GPU) and run the chain with the `multistatic` processing config unchanged.
3. Report the real time table of `docs/harbor.md` twice, tuning recordings beside test recordings, with `held_out` set correctly in every provenance block. A drop from tuning to test is a finding; do not change a parameter after seeing the test numbers.

Scoring, fixed now and replacing the stage 1 definitions where they differ: a detection within 5 m of an object's true position is a hit; a confirmed track is matched to the object nearest over its fixes if within 10 m (20 m on the 5 m harbor), otherwise a false track; the false alarm rate is the number of confirmed false tracks per minute of recording; Pd per object is the fraction of frames, from the first frame the object could have been confirmed, in which a confirmed track matched to it is within 5 m. Keep the object free threshold method of `scoreboard.py` as a diagnostic, not as the headline number: `docs/harbor.md` already shows it does not hold its 5 % on the full scene.

Criteria, fixed now, on the test recordings: every ship (the ferry, the tug, the boat) tracked with Pd at least 0.8 on `multistatic_b`; false tracks at most 2 per minute on every test recording; the wreck of the simple harbor found by its echo with Pd at least 0.9 on seeds 1 to 3. The silent UUVs are reported, with no criterion: the direct path removal is still 3 to 11 dB and that is stage 3's problem.

## 5. Echo detections become tracks

The echo map's peaks are detections with no memory; the picture should be tracks only. Run a second `RadiatorTracker` instance (its own `TrackerConfig` in the processing config: wider gate for the weaker peaks, longer confirmation) on the echo peaks, with the exclusion zones of the radiator tracks applied so a residual direct path is not confirmed as a body. A confirmed echo track whose fitted speed stays below 0.2 m/s for 10 s is marked static. Score echo tracks the same way as radiator tracks. The walkthrough's step 6 and 7 draw echo tracks instead of raw echo peaks (keep the peaks faint underneath).

## 6. What the objects are: measured characteristics

Replace the idea of a classifier with measured quantities a harbor operator can read, attached to every track, each with the frames it was measured over. No neural network in this stage; the CNN and CVAE stay in history.

For every track:

- Position and its uncertainty: the half maximum width of the map peak that fed the latest fix (two axes, from the map), and the residual of the fixes about the fitted line over the last 4 s.
- Velocity and heading from the track fit, with the fit's standard error.
- How it is seen: own noise, echo, or both (a radiator track and an echo track within 5 m for 2 s are merged into one object seen both ways).
- AIS: the matched report's identity and the distance between the reported and the located position at the latest fix, or "no report".
- Status: tentative, confirmed, static, lost; frames detected over frames since confirmation.

For radiator tracks:

- Received level at the array (the beam toward the track, rms in the band, in the trace units) and an estimated source level at 100 m assuming cylindrical spreading from the located range. Calibrate the scale once on the simple harbor, where the ferry is 1.0 Pa and the boat 0.3 Pa at 100 m, and report the calibration constant; on the other recordings report the estimate and its error against the truth.
- Tonal lines: the strongest narrowband lines in the whitened spectrum of the beam toward the track (frequency, level over the local floor, how many frames they persisted). Report them as numbers; do not name a propeller type or a vessel class.

For echo tracks:

- Echo strength: the peak's map value relative to the illuminating track's direct level, in dB, per illuminator.
- A size indicator from the spectrum: the echo strength in the upper third of the band over the lower third, in dB. A body small against the wavelength scatters more at higher frequency, a body large against it about equally. In the 100 to 760 Hz band the wavelengths are 2 to 15 m, so this separates the pilings (1 m) and the diver from the moored hull (30 m) and says little about the UUVs (1 by 3 m). Report it as "smaller than about a wavelength" or "about a wavelength or larger", never in metres.
- Extent: the length of the echo peak at half maximum along and across the illuminator direction, for the frames the body is detected; a hull 30 m long should show it.

Validate on the multistatic recordings (tuning and `multistatic_b`): for every object detected in at least 10 frames, the table of truth size and type against the measured indicators. Criterion, fixed now: on frames where the moored hull, a UUV and a piling are all detected, the size indicator puts the hull below the piling (flatter spectrum) in at least 70 % of them. State in the docs that the forward model's bodies are two dimensional pressure release masks, so this is a check of the measurement, not of the physics of real hulls.

Add `water/characteristics.py` (processing side; reads the stream's spectra, maps and tracks, never the truth) and tests on synthetic inputs with known answers for each quantity.

## 7. The operator output

1. `picture.jsonl`: one record per frame, written as the stream runs, with the frame time, the latency, and the list of objects, each with every field of section 6 and the fields' measurement windows. Document the schema in `docs/harbor.md`; it is the interface a display or a logger would consume.
2. `HarborPictureView` in `water/plotting.py`: one figure updated per frame (chart, sensors, every object as a marker with its uncertainty ellipse, a short velocity vector, its label: identity if AIS matched, else "radiator" or "body", size indicator, source level; lost tracks fade), saved as a video per recording (`picture.mp4`) and shown live with `--show`. Truth is drawn only with `--truth`, dashed, and is off by default.
3. `python -m acoustic_sim.water.stream --observation <file> --processing <config> --out <dir>` runs the whole thing on an observation file alone, the entry point a field recording would use once converted with `Observation.from_arrays`.

## 8. One measurement of the forward model's own error

`docs/harbor.md` records that the simulated water propagates 1 to 3 ms faster than the chart's sound speed over the direct paths, direction dependent, which is the FDTD scheme's dispersion at 2 to 4 cells per wavelength at the top of the band. For each recording, measure it from the truth (a stationary impulse at one ship position, arrival time at each hydrophone against range over the chart speed, per direction) and write it into the truth file's `meta` as `propagation_error_ms` with the figure `output/<dataset>/propagation_error.png`. Do not change the forward model in this stage; the next forward side stage decides between a finer grid and the k space time stepping for the test recordings.

## Rules

1. Figures, JSON, JSONL and logs under `output/` are committed; `*.npz`, `*.npy`, `*.mp4` and FDTD caches stay gitignored (commit one frame of each video as PNG).
2. `pytest` passes, including the new tests and the CI end to end run.
3. Criteria never change after a result is seen; parameters never change after a test recording is seen.
4. Plain names throughout: say what was measured or given, as in the rest of the repository.
5. Commit in small steps and open a pull request against `main` with the tuning versus test table, the characteristics validation table, the reproduce check output and one frame of the operator view.
6. Stop after section 8. Report the test recording numbers, the pass or fail of each criterion, what the characteristics got right and wrong, and the measured propagation error per recording.
Work in the 9m_Test repo on a new branch `harbor_picture_1` from `main` (`git fetch origin && git checkout -b harbor_picture_1 origin/main`). `main` is the only line of development; ignore every other branch. Use the GPU (`--cuda`, `use_cuda=True`) for every FDTD run and every processing chain that supports it, with CPU fallback.

Read `docs/architecture.md` (Sensing contract), `docs/harbor.md`, `src/acoustic_sim/contract.py`, `src/acoustic_sim/water/bistatic.py`, `src/acoustic_sim/water/detection.py`, `src/acoustic_sim/water/suppression.py`, `src/acoustic_sim/water/occupancy.py`, `src/acoustic_sim/water/evaluate.py`, `src/acoustic_sim/water/diagnostics.py`, `src/acoustic_sim/water/ais.py`, `src/acoustic_sim/water/sensors.py`, `src/acoustic_sim/water/dataset.py`, `examples/harbor_bistatic_ablation.py`, `examples/harbor_m4_dlc.py`, `examples/harbor_multistatic_dataset.py`, `tests/test_contract.py` and the outputs under `output/harbor_bistatic/`, `output/harbor_multistatic/` and `output/harbor_cancel/` before changing anything.

## Goal of the program

The water side has one job: from the recorded traces, the sensor positions, a chart of the harbor with a constant sound speed and, when available, the AIS reports, detect and locate every object in the harbor, whether it radiates sound of its own or only reflects the sound of other objects. The output is a map over time of everything in the water.

This stage builds the scoreboard that every later stage is measured on and measures the chains that exist today, with everything estimated from the traces. It does not change any processing algorithm.

## Language

Rename the terms that hide what was done. In code, scripts, tests, result JSON, figure labels and docs:

- `reference="oracle"` becomes `reference="true_waveform"` (the ship's radiated waveform taken from the simulation).
- `Diagnostics.ship_xy` and the `oracle` flag on `ShipTrack` become `exact_track` (the ship's position taken from the simulation).
- The configuration names in `harbor_bistatic_ablation.py` (`oracle`, `eca_oracle`, `eca_data`, `blind`) become `direct_removed_exactly`, `true_waveform_exact_track`, `estimated_waveform_exact_track`, `all_estimated`.
- `oracle_illuminator` becomes `true_illuminator_waveform`; `Provenance.oracle_inputs` becomes `Provenance.truth_inputs`; `blind` as a property name becomes `all_estimated`.
- In prose, write what was done: "given the true waveform", "given the exact track", "direct path removed exactly", "everything estimated from the traces". The words "oracle" and "blind" do not appear anywhere in the repository after this stage (`grep -ri oracle src examples tests docs` returns nothing).

Keep the `results.json` files already committed under `output/` as they are; new results use the new names.

## Contract fixes

1. `examples/harbor_bistatic_ablation.py` builds its input from `truth.components` and `truth.gain` and defaults to the simulator aware propagator. The `all_estimated` configuration must read `obs.traces` and use `FreeFieldPropagator` with nothing else; the other configurations may assemble their input from the truth components because that is what they are for, and their `Provenance` must say so (`truth_inputs` includes `"components"`). The default propagator of every script is the free field one; the simulator aware one is `--propagator simulator`.
2. The processing band is a processing choice. Remove `band_hz` from the observation `meta` written by `dataset.py` (it comes from the grid spacing) and give `BistaticConfig` and `ProcessingConfig` explicit `f_lo` and `f_hi`. Every script sets them. For the simulation datasets the scripts may set `f_hi` from the truth file's grid and then must add `"band"` to `simulator_knowledge`; the default is a stated number (for the 0.5 m datasets, 100 to 760 Hz; for the 5 m dataset, 12 to 78 Hz) and is recorded in `results.json`.
3. Split `water/ais.py` into `water/ais.py` (processing side: `AISReports`, `AISTrack`, prediction) and `water/ais_sim.py` (forward side: `AISConfig`, `generate_ais`), and split `water/sensors.py` into the geometry and `collapse` (processing side) and the FDTD sample point generation (forward side, in `water/sensor_sampling.py` or inside `fdtd_runner.py`). Add both processing modules to `PROCESSING` and both forward modules to `FORWARD` in `tests/test_contract.py`. Update `contract._KINDS`.
4. `Provenance.held_out` is never set. Every script in this stage sets it correctly per result (see the data split below), and `annotate_figure` prints "tuning set" or "test set".

## The scoreboard

Write `src/acoustic_sim/water/scoreboard.py` (processing side may not import it; it imports `Truth`, so it lives with scoring in `evaluate.py`'s class of module and is listed with `evaluate.py` in the contract test).

Input: a sequence of maps over time (`(n_cells, n_frames)` with the cell coordinates and frame times, as every chain already produces), the same for the object free runs, and a `Truth`.

Definitions, fixed now:

- **Threshold.** For each chain configuration, run the identical chain on traces from which one object's contribution has been removed (the dataset's components make this exact: the object's direct and scattered components are left out of the mix, noise and everything else kept). Set the threshold T so that 5 percent of frames in these object free runs have any cell above T outside the exclusion zones around the reporting ships. T is set on the tuning set only and then frozen.
- **Detection of object i in frame k.** The map has a local maximum above T within 5 m of the object's true position in that frame (static objects: within 5 m of their position).
- **Pd per object.** Fraction of frames in which the object is active and detected.
- **False alarms.** Number of local maxima above T per frame more than 10 m from every true object and outside the ship exclusion zones, reported as a rate per frame.
- **Localisation error.** Median and 90th percentile distance from the detecting peak to the true position, over detected frames only.
- **Separation.** For the row of pilings, the number of pilings resolved as distinct peaks in the full record map.

Output: `Scoreboard` with `per_object` (id, kind, radiates, reports AIS, moving or static, Pd, median error, p90 error, frames), `false_alarm_rate`, `threshold`, `provenance`, `to_dict()`, and a plotting helper that draws one table figure (`scoreboard.png`) and one map figure with every object's Pd printed at its position. Unit tests on a synthetic map with known peaks: threshold at the 5 percent quantile, Pd and false alarm counts by construction, a static and a moving object.

## Data

Rebuild nothing that exists. Use:

1. `output/harbor_bistatic/fine/` (one ferry, one silent UUV, 0.5 m grid) and `output/harbor_multistatic/fine/` (ferry and tug on AIS, a radiating boat without AIS, two silent UUVs, a diver, six pilings, a moored hull). Split each into observation and truth files if only `dataset.npz` exists (`python -m acoustic_sim.contract split`).
2. `output/harbor/` (the 5 m, 600 Hz passive dataset with the radiating UUV); build it with `examples/harbor_m2_dataset.py` if its cache is gone.

Data split, fixed now: the noise seed 0 mix of each dataset is the tuning set; seeds 1 to 5, mixed from the cached gathers with `assemble_dataset(seed=...)`, are the test set. No FDTD run is needed for the test set. Every parameter (threshold, window length, grid spacing, exclusion radius, track smoothing) is chosen on the tuning set and the test set is run once with those values. If a test set result makes you want to change a parameter, do not; report it.

## Measurements

Script `examples/harbor_picture_stage1.py` with flags `--dataset`, `--chain`, `--propagator {free_field,simulator}`, `--cuda`, `--workers`.

For every object in every dataset, run the chain that applies and score it:

| Chain | Objects | Dataset | Configurations |
|---|---|---|---|
| Passive emission (`run_ais_aware`, matched field scan, occupancy filter) | radiating UUV, radiating boat, and the AIS ships themselves (a ship found without using its AIS report is a check of the locator) | `harbor`, `multistatic` | everything estimated from the traces; and the same with the simulator aware propagator, labelled |
| Echo from a ship on AIS (`run_bistatic`, coherent map, ECA with per sample delays, track smoothing as on `main`) | silent UUVs, diver, pilings, moored hull | `bistatic` (ferry only), `multistatic` (ferry, tug, each as the sound source in turn) | everything estimated; given the true waveform and exact track; direct path removed exactly. The last two are the upper limits and sit beside the first in every table |

For the echo chain the map is the unnormalised `coherent_norm="snr"` map so that one threshold applies across frames and datasets; add the option to return it unnormalised if `main` still divides by the frame maximum.

Report for every configuration, tuning set and test set side by side: the scoreboard table, the threshold, the false alarm rate, the per object Pd, and for the echo chain the direct path cancellation depth from the truth components (a diagnostic, labelled).

Criteria, fixed now, on the test set with everything estimated from the traces and the free field propagator:

1. Passive chain: the two AIS ships and the radiating boat in the multistatic dataset each have Pd at least 0.9 at the 5 percent false alarm rate (if the chain cannot find a ferry without its AIS report, nothing downstream matters).
2. Echo chain, bistatic dataset: Pd of the UUV at least 0.5.
3. Echo chain, multistatic dataset, ferry as source: at least three of the seven static bodies detected on the full record map.

The expectation written down before the run is that 1 passes and 2 and 3 fail, because the direct path of a moving ship is removed by 8 to 19 dB when its waveform is estimated from the traces and the echo sits 16 to 18 dB below it. Whatever happens, report the numbers.

## Where the misses go

For every frame of the echo chain in which the detecting peak is more than 20 m from the UUV, classify the peak: within 10 m of the ship exclusion edge, within 10 m of a first order quay image of the ship, on the direct path between ship and cable, or elsewhere. Plot the peak locations coloured by class and give the counts per configuration. This decides what stage 3 works on.

## Outputs

`output/harbor_picture/stage1/`: `scoreboard_<dataset>_<chain>_<config>.png` and `.json`, `pd_summary.png` (one figure, every object, every configuration, tuning and test side by side), `misses_<dataset>.png`, `results.json` (everything, with provenance per entry), `run.log`.

## Docs

Rewrite the status part of `docs/harbor.md`: a new section "What the chains find today" at the top of the status material, with the scoreboard summary table and the pass or fail of each criterion. Move the stage 0 material of the multistatic work into a "History" subsection; keep its numbers, replace its terms as above. State plainly what was not detected. Add the scoreboard definitions to `docs/architecture.md` under the sensing contract, since every later stage is scored by them.

## Rules

1. No algorithm change in this stage. Renames, contract fixes, the scoreboard, the script and the docs only. If the rename or a contract fix changes any existing test's numbers, stop and report.
2. Figures, JSON and logs under `output/` are committed; `*.npz`, `*.npy` and FDTD caches stay gitignored.
3. `pytest` passes (all of it), including the new scoreboard tests and the updated contract test.
4. Commit in small steps and open a pull request against `main` with the `pd_summary.png` figure and the criteria table in its description.
5. Stop after the docs. Report the per object Pd on the test set with everything estimated from the traces, the pass or fail of each criterion, the dominant miss class, and the cancellation depth achieved on the moving ships.

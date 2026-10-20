Work in the 9m_Test repo on a new branch `sensing_contract` created from `main` (`git fetch origin && git checkout -b sensing_contract origin/main`; main is at PR #22, the performance merge). Use the GPU where the existing code already does (`--cuda`), with CPU fallback. This is a restructuring stage: it changes interfaces and adds enforcement, and it must not change any numerical result. Every existing test and every existing example must produce the same numbers it produced before, within round off.

Read `docs/architecture.md`, `docs/harbor.md`, `src/acoustic_sim/water/dataset.py` (`HarborDataset`), `src/acoustic_sim/water/bistatic.py` (`BistaticConfig`, `Truth`, `run_bistatic`, `ReplicaModel`), `src/acoustic_sim/water/detection.py` (`ProcessingConfig`), `src/acoustic_sim/water/suppression.py`, `src/acoustic_sim/water/dispersion.py`, `src/acoustic_sim/water/evaluate.py`, `src/acoustic_sim/water/classifier.py`, `src/acoustic_sim/air/detection/engine.py`, `src/acoustic_sim/air/detection_main.py`, `examples/run_pipeline.py`, `examples/harbor_m4_dlc.py`, `examples/harbor_bistatic_ablation.py` and `tests/test_package_layout.py` before changing anything.

## The principle

The package has two sides and one contract between them.

The **forward side** generates synthetic data by solving the acoustic wave equation (FDTD, Helmholtz, or an analytic propagator) in air, in water, or both. It knows everything: the scenario, every track, every source waveform, the grid, the time step, the gains, the dead channels.

The **processing side** detects, locates, tracks and classifies unknown objects. It consumes either real sensor data or forward model data treated as if it were real, and it cannot tell the difference. It knows only what a deployed system would know: the recorded traces, the sample rate, the sensor geometry, publicly available environment information (land, bathymetry, a nominal sound speed), and in water the AIS reports.

Anything the processing side reads that is not in that list is a diagnostic. Diagnostics are allowed (they are how processing gain is measured), but every result they touch must say so in the data, not only in the prose.

Today the split is mostly honored at the function level (`run_ais_aware`, `run_bistatic`, the occupancy filter take traces, layout, fs, environment and AIS) and violated in three ways: the dispersion model (`dispersion.for_scenario(env, ds.cfg.fdtd)`) hands the detector the FDTD grid; oracle inputs (true waveform, exact track, truth components) enter `run_bistatic` through `Truth` and through `ship_positions(truth=...)`; and the dataset file carries observation and truth in one object, so nothing stops a script from reading `ds.tracks` on the inference path. This stage makes the split structural.

## Stage 0: the contract

### 1. Two data classes, two files

In a new module `src/acoustic_sim/contract.py` (domain neutral, imported by both `air/` and `water/`):

`Observation`: `traces` (n_receivers, n_samples), `fs`, `geometry` (receiver positions; for water the `SensorLayout` with cables and hydrophones, for air the microphone positions, through one small protocol with `positions()` and `n_receivers`), `environment` (an object exposing only public knowledge: land mask, depth, nominal sound speed, domain extent; for air, the wind and ground description a deployed system would be given), `reports` (AIS reports in water; `None` in air), `t0`, `meta` (free text about the recording: instrument, units, band; never scenario parameters). `Observation.save(path)` and `Observation.load(path)` write a single `.npz` with a JSON header. The header must not contain the FDTD configuration, the grid spacing, the time step, the scenario JSON, any track, any source level or any component.

`Truth`: `tracks`, `components` (direct and scattered per vessel, noise), `gain`, `dead`, `scenario` (the full scenario config including the FDTD config), `fdtd` (the solver config used), `meta`. Same save and load, a separate `.npz` written next to the observation (`<name>.observation.npz` and `<name>.truth.npz`).

`HarborDataset` stays as the forward side convenience object. Give it `.observation` and `.truth` properties that build the two objects, make `assemble_dataset` and `simulate_scenario` write both files, and keep `HarborDataset.load` reading the old single `dataset.npz` files so every existing cache and output stays usable. Add `python -m acoustic_sim.contract split <dataset.npz>` that writes the two files from an old one.

Do the same for the air pipeline: `examples/run_pipeline.py` currently loads traces plus a metadata dict that mixes receiver positions with the source trajectory parameters that `compute_ground_truth` uses. Split it into an `Observation` and a `Truth`; the pipeline's bearing, range and tracking stages receive only the observation, and `compute_ground_truth` is called only inside scoring and plotting.

### 2. Import rules, enforced by a test

Processing side modules: `water/detection.py`, `water/suppression.py`, `water/occupancy.py`, `water/bistatic.py`, `water/motion.py`, `water/localiser.py`, `water/classifier.py`, `water/evaluate.py` (scoring is allowed to import `Truth`, see below), `air/detection/*`, `air/processor.py`, `air/tracker.py`, `air/fire_control.py`, `ml/*` inference paths.

Forward side modules: `solver/*`, `model.py`, `sources.py`, `water/fdtd_runner.py`, `water/fdtd_cuda_bodies.py`, `water/scatterer.py`, `water/dispersion.py`, `water/signatures.py`, `water/vessels.py`, `water/noise.py`, `water/dataset.py`, `air/forward.py`, `air/domains.py`, `air/noise.py`, `air/setup.py`, `ml/data_generation.py`, `ml/fno*`.

Rule: no processing side module imports a forward side module, at module level or inside a function. Write `tests/test_contract.py` that walks both lists with `ast`, resolves relative imports, and fails on any violation with the file and line. Also assert that `Observation`'s header, after a round trip through `assemble_dataset` on the tiny test scenario, contains none of the keys listed in section 1. Add the test to the existing CI workflow.

Where a processing module currently imports a forward module, fix it one of three ways, in this order of preference: the shared type moves to `contract.py` (for example the `Track` dataclass, which AIS prediction and scoring both use, becomes a contract type); the forward dependency becomes an injected object (section 3); or the function is actually a diagnostic and moves to a new `water/diagnostics.py` or `air/diagnostics.py`, which may import both sides.

### 3. Propagation as an injected object

`ReplicaModel` in `bistatic.py` and the Green's function code in `detection.py` and `suppression.py` compute free field replicas and, with a `DispersionModel`, replicas corrected for the simulator's grid. Define a `Propagator` protocol in `contract.py` with `greens(sources_xy, receivers_xy, freqs)` (and the path factor used by the bistatic model), and two implementations:

`FreeFieldPropagator` in a processing side module (default everywhere; takes the nominal sound speed from the environment).

`SimulatorAwarePropagator` in `water/diagnostics.py`, wrapping the current dispersion corrected replica. It is constructed from a `Truth` (it needs the FDTD config), which is what makes its use visible: a script that wants it has to load the truth file on the inference path, and the import rule above forbids that in library code.

`ProcessingConfig.dispersion` and `BistaticConfig.dispersion` are removed and replaced by `propagator`. Update every script and test on main that passed `dispersion=`; there is no compatibility shim.

A Helmholtz replica library (a `HelmholtzPropagator` that needs only the public environment and `solver/helmholtz.py`) is a third implementation of the same protocol and a later stage; write the protocol so it fits, do not implement it here.

### 4. Oracle inputs and the results schema

`run_bistatic(..., truth=Truth(...))` stays, renamed to `diagnostics=`, and every oracle path (`reference="oracle"`, `ship_positions(truth=...)`, truth direct removal, truth components through the ECA) must set a flag. Add a `Provenance` dataclass to `contract.py`, attached to every result object the processing side returns (`BistaticRun`, the passive run object, the air `WindowDetection` list) and written into every `results.json`:

```
"provenance": {
  "oracle_inputs": ["waveform", "track", "direct_removal"],   # [] when blind
  "simulator_knowledge": ["dispersion"],                      # [] when none
  "data": "observation" | "forward_model",
  "held_out": true | false
}
```

The plotting helpers that write result tables and figure titles read the provenance and print the oracle and simulator flags on the figure. A table in `docs/` that contains an oracle or simulator aware row must contain the blind row for the same case beside it; this is a documentation rule, state it in `docs/architecture.md`.

Scoring (`water/evaluate.py`, `evaluate_results` in the air pipeline, the Pd and threshold code) takes `Truth` as an explicit argument. That is the one sanctioned place truth enters after inference. Classification follows the same rule: `HarborClassifier` and the air classifiers infer from the observation; labels come from `Truth` only inside the scoring call.

### 5. Real data entry point

Add `Observation.from_arrays(traces, fs, geometry, environment, reports=None, meta=...)` and document in `docs/architecture.md` the exact arrays a field recording must be converted into (units, orientation, sample rate, channel order, gauge length for strain DAS, AIS fields and time base). Do not write a format specific reader (HDF5, SEG-Y, NMEA) in this stage; write one test that builds an `Observation` from plain arrays with no scenario anywhere and runs `run_bistatic` and `run_ais_aware` on it to completion with the free field propagator. That test is the proof that the processing side works without the forward model.

### 6. Air and water as one chain

`docs/architecture.md` gets a new section, "Sensing contract", placed before the module inventory, that states the principle in section 0 of this prompt, the two data classes, the import rule, the propagator protocol, the provenance schema and the documentation rule. Rewrite the inventory so forward side and processing side are the top level grouping and air and water are subgroups within each, with `contract.py` listed first. The detect, locate, track, classify chain is one package with two forward models; the docs should read that way.

## Rules

1. No numerical change. Before touching anything, run `pytest` (all of it, not only `tests/harbor`) and the quick examples (`harbor_m1_demo.py`, `harbor_bistatic_demo.py` on the coarse dataset if `output/harbor_bistatic/coarse/dataset.npz` exists, `run_pipeline.py` on its default case) and save their `results.json` and summary numbers to a scratch directory outside the repo. After the refactor, run them again and diff. Report every difference, including none.
2. Existing `dataset.npz` files, FDTD caches and `output/` figures stay valid and are not regenerated. Caches keep their keys.
3. `tests/test_contract.py` passes, and the rest of `pytest` passes.
4. Figures, JSON results and logs under `output/` are committed. Datasets (`*.npz`, `*.npy`) and FDTD caches stay gitignored. The new observation and truth files follow the same rule.
5. Commit in small steps on `sensing_contract` and open a pull request against `main`. The PR description lists every processing side import that was a violation and how each was resolved, and the before and after diff from rule 1.
6. `main` is the only line of development. Ignore every other branch in the repository; do not read from them, merge them or mention them in the docs.
7. Stop after section 6. Report what the import test found, what moved where, the before and after diff, and anything in the processing side that could not be made to run without forward model knowledge.
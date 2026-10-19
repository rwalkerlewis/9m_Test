# Parametric Studies

Nine parametric studies in `src/acoustic_sim/air/studies.py` sweep
configuration parameters to evaluate detection pipeline robustness.
All operate on the MFP/EKF library pipeline (`air/detection_main.py`).
These nine studies do not include the separate harbor sensing experiments; for harbor passive
detection sweeps, AIS suppression, bistatic echo localization, and classifier workflows, see
[Harbor](harbor.md#examples-and-workflows).

Every study starts from the `DetectionConfig` defaults (0.5 m radius
array, dx = 0.05 m, 30 × 30 m domain) or a `base_config` you pass, with
the stationary interferer disabled unless the study turns it on.  Each
prints a table (detection rate, localisation error, first-shot miss,
pattern, hit) and writes `comparison.png` via `plot_study_comparison`
under `output/studies/<study>/` (the `output_dir` argument).

Studies 3, 5 and 8 run the FDTD **once** and inject faults, transients or
position errors post hoc before re-running detection; the others call the
full pipeline (`run_detection_pipeline`, one FDTD run) per case.

```python
from acoustic_sim.air.studies import run_all_studies
results = run_all_studies()
```

`run_all_studies` runs studies 3, 5, 8 and 9 (sensor faults, transient,
position errors, mixed failures).  Study 9 still needs one FDTD run per
scenario (seven in total), so the batch is not cheap.  Studies 1, 2, 4,
6 and 7 must be called individually.

---

## Study 1 — Array Geometry

**Function:** `study_array_geometry`

Runs the pipeline with each array type: circular, linear, L-shaped,
random and concentric (the `DetectionConfig.array_type` values), at the
configured microphone count and radius.

---

## Study 2 — Minimum Sensor Count

**Function:** `study_min_sensors(sensor_counts=(4, 6, 8, 12, 16, 24))`

Sweeps the number of microphones to find the minimum count that still
detects and engages reliably.

---

## Study 3 — Sensor Fault Robustness

**Function:** `study_sensor_faults(fault_fractions=(0.0, 0.1, 0.2, 0.3, 0.5))`

Single FDTD run.  Injects `elevated_noise` faults into the given fraction
of channels and runs detection with and without robust sensor weighting
(`enable_sensor_weights`).

---

## Study 4 — Multi-Drone Detection

**Function:** `study_multi_drone`

Compares one drone against two sources: the drone plus the stationary
source re-purposed as a second drone-like emitter (150 Hz, 90 dB at
(−8, −6) m), with `max_sources = 2`.  Reports the number of tracks.

---

## Study 5 — Transient Robustness

**Function:** `study_transient_robustness(transient_levels=(0, 110, 120, 130))`

Single FDTD run.  Injects a broadband impulse (explosion / gunshot) at
each level in dB and runs detection with and without transient blanking.

---

## Study 6 — Haphazard Array Placement

**Function:** `study_haphazard_array(n_trials=3)`

Compares the circular array with `n_trials` random placements (seeds
100, 101, …), simulating field-expedient deployment.

---

## Study 7 — Echo-Prone Domains

**Function:** `study_echo_domains`

Runs the pipeline in the isotropic, echo canyon and urban echo domains to
measure how multipath reflections degrade detection and localisation.

---

## Study 8 — Sensor Position Errors

**Function:** `study_position_errors(error_stds=(0.0, 1.0, 2.0, 5.0))`

Single FDTD run.  Perturbs the microphone positions the processor
assumes by Gaussian errors of each standard deviation [m] and runs
detection with and without TDOA self-calibration.

---

## Study 9 — Mixed Failure Modes

**Function:** `study_mixed_failures`

Seven progressively harder scenarios, each a full pipeline run:
clean; 20 % sensor faults; + 2 m position errors; + 120 dB transient;
+ echo canyon domain; + random array; and the last scenario again with
all mitigations (sensor weights, transient blanking, position
calibration).

---

## Metrics

All studies report:

| Metric | Description |
|--------|-------------|
| Detection rate | fraction of windows with a valid detection |
| Mean localisation error | mean Euclidean distance between detection and ground truth (m) |
| First-shot miss distance | miss distance of the first engagement (m) |
| First-shot pattern diameter | shot pattern size at the first-shot range (m) |
| First-shot hit | whether the first-shot miss is below the pattern radius (half the diameter) |
| Mean miss | average miss distance across all engagements (m) |

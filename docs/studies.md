# Parametric Studies

Nine parametric studies in `src/acoustic_sim/studies.py` sweep
configuration parameters to evaluate detection pipeline robustness.
All operate on the MFP/EKF library pipeline (`detection_main.py`).

Run all studies:

```python
from acoustic_sim.studies import run_all_studies
results = run_all_studies()
```

Each study generates console output (tabulated metrics) and a
comparison plot via `plot_study_comparison`.

---

## Study 1 — Array Geometry

**Function:** `study_array_geometry`

Compares detection performance across array types: nested circular,
circular, concentric, L-shaped, log spiral, and random disk.  All at
the same total aperture and microphone count.

**Metrics:** detection rate, mean localisation error, first-shot miss
distance, first-shot hit.

---

## Study 2 — Minimum Sensor Count

**Function:** `study_min_sensors`

Sweeps the number of microphones from 4 to the full array size.
Identifies the minimum count needed for reliable detection and
engagement.

---

## Study 3 — Sensor Fault Robustness

**Function:** `study_sensor_faults`

Injects faults (elevated noise, dropout, spikes, DC offset) into a
fraction of channels.  Compares baseline (no faults), faults without
mitigation, and faults with sensor-weight mitigation enabled
(`enable_sensor_weights=True`).

---

## Study 4 — Multi-Drone Detection

**Function:** `study_multi_drone`

Injects 1–3 simultaneous drone sources at different bearings and
ranges.  Evaluates multi-target detection rate, track accuracy, and
threat prioritisation.

---

## Study 5 — Transient Robustness

**Function:** `study_transient_robustness`

Injects a broadband impulse (simulating an explosion or gunshot) at a
specified time and location.  Compares baseline, transient without
mitigation, and transient with blanking enabled
(`enable_transient_blanking=True`).

---

## Study 6 — Haphazard Array Placement

**Function:** `study_haphazard_array`

Uses random microphone positions (no geometric structure) to simulate
field-expedient array placement.  Compares random arrays of various
sizes against the standard nested circular array.

---

## Study 7 — Echo-Prone Domains

**Function:** `study_echo_domains`

Runs the pipeline on echo canyon and urban echo domain types, where
multipath reflections create ghost detections.  Evaluates how the
stationary-source rejection filter handles reflected energy.

---

## Study 8 — Sensor Position Errors

**Function:** `study_position_errors`

Adds Gaussian perturbation to mic positions (simulating survey error).
Sweeps error standard deviation.  Compares uncalibrated vs. calibrated
(`enable_position_calibration=True`) with TDOA self-calibration.

---

## Study 9 — Mixed Failure Modes

**Function:** `study_mixed_failures`

Combined stress test: sensor faults + position errors + transient +
wind noise simultaneously.  Compares no-mitigation vs. all-mitigation
(sensor weights + transient blanking + position calibration).

---

## Metrics

All studies report:

| Metric | Description |
|--------|-------------|
| Detection rate | fraction of windows with a valid detection |
| Mean localisation error | mean Euclidean distance between detection and ground truth (m) |
| First-shot miss distance | miss distance of the first engagement (m) |
| First-shot pattern diameter | shot pattern size at the first-shot range (m) |
| First-shot hit | whether the first-shot miss is within the pattern |
| Mean miss | average miss distance across all engagements (m) |

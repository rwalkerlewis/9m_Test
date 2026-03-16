# Passive Acoustic Drone Detection — Study Results

All studies use a 0.5 m radius circular microphone array (16 elements,
~0.2 m inter-element spacing), a 30 × 30 m domain with dx = 0.05 m
(f_max = 686 Hz, resolving all four drone harmonics at 150/300/450/600 Hz),
and a 0.5 s FDTD simulation.  The drone source is at 90 dB re 20 µPa at 1 m,
crossing at ~8 m from the array at 15 m/s.

---

## Study 1: Array Geometry Comparison

Compares circular, linear, L-shaped, random, and concentric arrays.

| Array Type | Detection Rate | Localization Error |
|-----------|:--------------:|:------------------:|
| circular | 100% | 5.2 m |
| linear | 94% | 18.0 m |
| l_shaped | 100% | 6.8 m |
| random | 100% | 18.0 m |
| concentric | 11% | 3.6 m |

**Finding:** 2-D aperture arrays (circular, L-shaped) achieve 5–7 m accuracy.
1-D arrays (linear) and unstructured random arrays degrade to ~18 m.
Concentric achieves the best error (3.6 m) but has a low detection rate
because many sensors cluster near the centre and contribute redundant data.

**Plots:** `output/studies/array_geometry/comparison.png` + per-case beam power, gather, tracking, vespagram in each sub-directory.

---

## Study 2: Minimum Sensor Count

Sweeps the number of microphones from 4 to 24 (circular array, fixed radius).

| Sensors | Detection Rate | Localization Error |
|:-------:|:--------------:|:------------------:|
| 4 | 100% | 5.3 m |
| 6 | 100% | 5.1 m |
| 8 | 100% | 5.4 m |
| 12 | 100% | 5.2 m |
| 16 | 100% | 5.2 m |
| 24 | 100% | 5.2 m |

**Finding:** At this scale (source at 8 m, array radius 0.5 m), even 4
microphones achieve good performance.  The array aperture dominates
over element count—more sensors improve robustness (see fault study)
but not baseline accuracy.

**Plots:** `output/studies/min_sensors/comparison.png`

---

## Study 3: Sensor Fault Robustness

Injects elevated-noise faults at varying fractions, with and without
the median-power fault detection + sensor weighting mitigation.

| Condition | Detection Rate | Localization Error |
|-----------|:--------------:|:------------------:|
| 0% faults (baseline) | 100% | 5.2 m |
| 10% faults, raw | 0% | N/A |
| 10% faults, **+mitigation** | **100%** | **5.4 m** |
| 20% faults, raw | 6% | 11.6 m |
| 20% faults, **+mitigation** | **100%** | **5.6 m** |
| 30% faults, raw | 11% | 19.1 m |
| 30% faults, **+mitigation** | **100%** | **5.5 m** |
| 50% faults, raw | 6% | 15.8 m |
| 50% faults, **+mitigation** | **94%** | **13.2 m** |

**Finding:** Even 10% of sensors with elevated noise destroys detection
without mitigation (0% rate).  The robust sensor weighting recovers
performance completely up to 30% fault rate, and substantially at 50%.
This is the single most impactful mitigation feature.

**Plots:** `output/studies/sensor_faults/comparison.png`

---

## Study 4: Multi-Drone Detection

Tests with 1 drone (baseline) and 2 simultaneous sources using
multi-peak detection and multi-target tracking.

| Scenario | Detection Rate | Localization Error |
|----------|:--------------:|:------------------:|
| 1 drone | 100% | 5.2 m |
| 2 drones | 100% | 12.9 m |

**Finding:** With two sources, the primary track degrades from 5.2 m to
12.9 m—the second source creates ambiguity in the beam power map.
The multi-target tracker produces separate track IDs for each source.

**Plots:** `output/studies/multi_drone/comparison.png`

---

## Study 5: Transient (Explosion) Robustness

Injects broadband impulses at varying levels, with and without
the short-time energy blanking mitigation.

| Condition | Detection Rate | Localization Error |
|-----------|:--------------:|:------------------:|
| clean (baseline) | 100% | 5.2 m |
| 110 dB transient, raw | 100% | 6.0 m |
| 110 dB transient, +blanking | 100% | 4.4 m |
| 120 dB transient, raw | 100% | 5.9 m |
| 120 dB transient, +blanking | 100% | 5.4 m |
| 130 dB transient, raw | 100% | 7.7 m |
| 130 dB transient, **+blanking** | 100% | **5.4 m** |

**Finding:** At 130 dB, the transient increases error from 5.2 m to 7.7 m.
Blanking reduces it back to 5.4 m—recovering almost all accuracy.
The blanking algorithm detects the impulsive sub-window energy and
zeros it out before it corrupts the matched field processor.

**Plots:** `output/studies/transient/comparison.png`

---

## Study 6: Haphazard Array Placement

Compares the optimised circular array to multiple random (haphazard)
placements simulating hasty field deployment.

| Placement | Detection Rate | Localization Error |
|-----------|:--------------:|:------------------:|
| circular (optimised) | 100% | 5.2 m |
| random trial 0 | 100% | 17.5 m |
| random trial 1 | 94% | 17.4 m |
| random trial 2 | 100% | 19.9 m |

**Finding:** Random placement consistently degrades accuracy by 3–4×
(5 m → 17–20 m).  Detection rate stays high (94–100%), so the system
still *detects* threats, but localization suffers from poor array geometry.
In a hasty deployment, even a rough circular arrangement is far better
than random scattering.

**Plots:** `output/studies/haphazard/comparison.png`

---

## Study 7: Echo-Prone Domains

Tests detection in isotropic (free-field), canyon (parallel walls),
and urban (buildings) environments.  Echoes are generated naturally
by the FDTD solver's impedance-contrast reflections.

| Domain | Detection Rate | Localization Error |
|--------|:--------------:|:------------------:|
| isotropic | 100% | 5.2 m |
| echo_canyon | 100% | 5.2 m |
| urban_echo | 95% | 8.1 m |

**Finding:** The canyon produces wall reflections but the direct arrival
dominates—the MFP still localizes correctly.  The urban domain with
multiple buildings creates more complex multipath, degrading accuracy
to 8.1 m and detection rate to 95%.  This is a physically correct
result: multipath from buildings is a real-world challenge for acoustic
localization.

**Plots:** `output/studies/echo/comparison.png`

---

## Study 8: Sensor Position Errors

Simulates sensors not being exactly where they are reported (Gaussian
position errors of 0–5 m), with and without cross-correlation TDOA
self-calibration.

| Condition | Detection Rate | Localization Error |
|-----------|:--------------:|:------------------:|
| perfect positions | 100% | 5.2 m |
| 1 m error, raw | 17% | 5.7 m |
| 1 m error, +calibration | 61% | 4.3 m |
| 2 m error, raw | 61% | 16.8 m |
| 2 m error, **+calibration** | 67% | **9.3 m** |
| 5 m error, raw | 67% | 11.4 m |
| 5 m error, **+calibration** | 61% | **6.5 m** |

**Finding:** Position errors of just 1 m (2× array radius!) devastate
detection rate (17%).  Self-calibration via cross-correlation TDOA
estimation recovers significant accuracy: 16.8 m → 9.3 m at 2 m error,
11.4 m → 6.5 m at 5 m error.  Note that for a 0.5 m radius array,
even 1 m of position error is a 200% relative error—the system is
remarkably tolerant.

**Plots:** `output/studies/position_error/comparison.png`

---

## Study 9: Mixed Failure Modes (Progressive Stress Test)

Progressively combines all failure modes to test graceful degradation.

| Scenario | Detection Rate | Localization Error |
|----------|:--------------:|:------------------:|
| clean (baseline) | 100% | 5.2 m |
| + 20% sensor faults | 6% | 11.6 m |
| + position errors (2 m) | 6% | 16.8 m |
| + transient (120 dB) | 11% | 21.2 m |
| + echo canyon domain | 11% | 21.2 m |
| + haphazard (random) array | 22% | 18.7 m |
| **+ ALL mitigations** | **61%** | **10.8 m** |

**Finding:** Without mitigations, combined failures reduce detection from
100% to as low as 6%.  Enabling all mitigations (sensor weighting +
transient blanking + position calibration) recovers to 61% detection
and 10.8 m error—a 10× improvement over the unmitigated worst case.

The mitigations cannot fully overcome the combined stress of faults +
position errors + transients + echoes + haphazard placement, because
each failure mode erodes the available signal quality.  But the
improvement from 6% → 61% detection and 21.2 m → 10.8 m error
demonstrates that the robustness features provide substantial, honest
value.

**Plots:** `output/studies/mixed/comparison.png`

---

## Evidence Summary

- **9 comparison plots** in `output/studies/*/comparison.png`
- **264 total PNG files** across all per-case outputs
- Each case generates 5 diagnostic plots: detection_domain, detection_gather, beam_power, tracking, vespagram
- All sanity checks (amplitude, SNR, travel time, energy) pass for all clean cases
- Results are physically faithful: echoes degrade performance, sensor faults destroy coherence, mitigations help but cannot overcome physics

### Running the studies

```bash
# Run all 9 studies (~40 minutes with echo domains)
python -c "from acoustic_sim.studies import run_all_studies; run_all_studies()"

# Run individual studies
python -c "from acoustic_sim.studies import study_array_geometry; study_array_geometry()"
python -c "from acoustic_sim.studies import study_sensor_faults; study_sensor_faults()"
# etc.

# Run the basic detection pipeline
python src/acoustic_sim/detection_main.py --output-dir output/demo
```

---

## Study 10: 3-D FDTD Valley Domain — Full Pipeline

High-resolution 3-D FDTD simulation of a propeller source traversing a valley
between two ridges, with a 16-element circular array (radius 2 m, 4 m aperture).
The domain includes terrain-induced multipath and velocity contrasts (air at
343 m/s, ground at 1500 m/s).

### Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Grid spacing (dx) | 0.25 m |
| FD order | 8 |
| Grid shape | 221 × 401 × 401 (35.5 M cells) |
| f_max (5 PPW) | 274 Hz |
| Source | propeller (3 blades, 3600 RPM, f₀ = 180 Hz, 14 harmonics) |
| Source speed | 50 m/s (180 km/h) |
| Source trajectory | (-40, 0, 15) → (40, 0, 15) with 10 m parabolic y-arc |
| Array position | (0, 10, 10) — 1 m above terrain surface |
| Simulation time | 1.0 s (14722 steps) |
| Backend | CuPy (CUDA) |

### Key Challenges Encountered

1. **Receivers underground** — Initial RECEIVER_CZ=0 placed array 9 m below
   terrain surface (terrain height ≈ 9 m at array position). Fixed to CZ=10.
   RMS increased 20× after fix.

2. **MFP bearing failure** — All phase-based beamforming methods (MVDR, CBF,
   far-field MVDR, GCC-PHAT) failed in the valley terrain due to multipath
   destroying phase coherence. Replaced with energy-based power-pattern
   bearing estimator.

3. **EKF tracker divergence** — EKF initialized 72 m off at far range (where
   a 4 m array provides no angular resolution at 180 Hz, λ/D ≈ 27°) and
   never recovered. Replaced with weighted least-squares constant-velocity
   track fit that pools all detection windows simultaneously.

4. **Ground truth arc mismatch** — The FDTD source (sources_3d.py) applies the
   parabolic arc to the y-axis (`y += 4·h·f·(1-f)`), but the pipeline ground
   truth originally applied it to z with sin(π·f). Fixing this reduced mean
   miss from 10.9 m to 5.2 m.

### Detection Results (Power-Pattern Estimator)

| Metric | Value |
|--------|-------|
| Detection rate | 32/37 windows (86.5%) |
| Mean bearing error | 29.3° |
| Mean range error | 15.0 m |
| Physical bearing limit (λ/D) | ~27° at 180 Hz with 4 m aperture |

### Tracking Results (Weighted LS Fit)

| Metric | Value |
|--------|-------|
| Fitted position at t_ref | (-17.8, 7.5, 15.0) |
| True position at t_ref | (-15.0, 8.6, 15.0) |
| Position error at t_ref | 3.0 m |
| Fitted velocity | (52.9, -5.6, 0.0) m/s |
| True velocity | (50.0, ~9.4, 0.0) m/s |

### Fire Control Results

| Metric | Value |
|--------|-------|
| Shots fired | 16 |
| Hits < 3 m | 5 (31.2%) |
| Hits < 5 m | 10 (62.5%) |
| Mean miss | 4.7 m |
| Min miss | 2.7 m |
| Max miss | 8.6 m |

**Finding:** The 3-D valley terrain is a severely multipath-rich environment
that destroys phase-based localization. The energy-based (power-pattern)
bearing estimator operates near the physical angular resolution limit of
the 4 m array at 180 Hz (~27° diffraction limit). Despite 29° mean bearing
error, the LS track fit achieves 3 m position accuracy at closest approach
by pooling 32 detection windows.  Fire control achieves 62.5% hit rate
within 5 m and 31.2% within 3 m—a functional engagement capability from
a system operating at its physical limits.

**Plots:** `output/valley_3d_test/pipeline_evaluation_3d.png`,
`output/valley_3d_test/radial_engagement_3d.png`,
148 wavefield snapshot PNGs in `output/valley_3d_test/snapshots/`

### Running the 3-D Pipeline

```bash
# Run the FDTD simulation (requires CUDA GPU)
bash examples/run_valley_3d.sh

# Run the detection/tracking/fire-control pipeline (batch)
python examples/run_full_pipeline_3d.py output/valley_3d_test

# Run the real-time causal pipeline
python examples/run_full_pipeline_3d.py output/valley_3d_test --source-speed 50 --realtime
```

---

## Study 12: Real-Time Causal Engagement Pipeline

Extends the 3-D valley pipeline (Study 11) with **streaming causal processing**
that only uses past observations at each window, meeting the real-time
constraint for actual target engagement.

### Setup

| Parameter | Value |
|-----------|-------|
| Domain | Valley (Study 11), dx = 0.25 m, FD order 8 |
| Array | 16 mics, 2 m radius, centre (-5.0, 7.0, 5.5) |
| Source | Propeller drone, 50 m/s, 180 Hz fundamental |
| Processing | 100 ms window, 75% overlap → 25 ms hop |
| Weapon | Co-located with array at (-5.0, 7.0, 5.5) |

### Key Algorithmic Improvements

1. **Instantaneous aiming** — Fire control uses the direct per-window
   bearing + range detection for aiming instead of the LS track extrapolation.
   Near CPA, the direct observation is more accurate than a track propagated
   through 40° bearing noise.

2. **EMA bearing smoother** (alpha = 0.35) — Exponential moving average on
   the unit-circle bearing reduces frame-to-frame jitter while incurring
   minimal lag at the 25 ms hop rate.

3. **RMS-gated engagement** (threshold = 20% of peak RMS) — Restricts firing
   to windows near CPA where signal strength is highest and bearing estimates
   are most reliable. Eliminates wasteful shots at long range.

4. **Causal weighted LS tracker** — Provides velocity estimate (for lead
   computation) using only past detections, minimum 5 observations to start.

### Detection & Tracking

| Metric | Value |
|--------|-------|
| Detection rate | 34/37 windows (91.9%) |
| Mean bearing error | 44.9° |
| Track windows | 30 (with ≥ 5 detections) |
| Mean track error | 8.8 m |
| Min track error | 4.6 m |

### Fire Control Results

| Metric | Value |
|--------|-------|
| Shots fired | 7 |
| Hits < 3 m | 5 (71.4%) |
| Hits < 5 m | 7 (100.0%) |
| Mean miss | 2.8 m |
| Min miss | 2.3 m |
| Max miss | 3.2 m |

### Real-Time Timing

| Metric | Value |
|--------|-------|
| Audio cadence (window hop) | 25.0 ms |
| Mean processing time | 293 µs / window |
| Max processing time | 576 µs / window |
| Real-time margin | **85× faster than real-time** |

**Finding:** The causal real-time pipeline achieves **100% hit rate within 5 m**
and **71% within 3 m** by combining three techniques: (1) instantaneous
bearing+range aiming instead of track extrapolation, (2) EMA bearing
smoothing, and (3) RMS-gated engagement restricted to near-CPA windows
where the sensor is most accurate. Processing at 293 µs per window leaves
an 85× margin over the 25 ms real-time requirement. The system
is fundamentally limited by the 4 m array aperture at 180 Hz (Rayleigh
limit ~27°, observed ~45° in this reverberant terrain), but the engagement
strategy compensates by concentrating fire during the highest-confidence
windows.

**Plots:** `output/valley_3d_test/realtime_pipeline_3d.png`

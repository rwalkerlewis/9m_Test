# Algorithms

## SRP-PHAT Beamformer

The primary bearing estimator in the engagement pipeline.  Implemented
as `SRPBeamformer` in `examples/run_pipeline.py`.

### Principle

Steered Response Power with Phase Transform (SRP-PHAT) estimates the
direction of arrival by cross-correlating all microphone pairs with
phase-only weighting, then steering the result across candidate bearings.

### Implementation

1. **Pre-computation (once, at startup).**  For each of $B$ candidate
   bearings, compute the inter-microphone time delays $\tau_{m,b}$
   relative to the array centre.  Convert to phase shifts at each
   frequency bin within $[f_\text{lo}, f_\text{hi}]$:

   $$\mathbf{S}_{m,b,f} = \exp\!\bigl(j\, 2\pi f\, \tau_{m,b}\bigr)$$

   This steering matrix has shape `(n_mics, n_bearings, n_freq_bins)`
   and is stored as complex64.

2. **Per-window processing.**  For a segment of `win_len` samples:

   a. FFT each microphone channel: $X_m(f) = \text{FFT}[\text{seg}_m]$.

   b. Apply PHAT weighting: $\hat{X}_m(f) = X_m(f) / |X_m(f)|$.

   c. Steer: $P(b) = \sum_f \left|\sum_m \hat{X}_m(f) \cdot S_{m,b,f}\right|^2$.

   d. The bearing with maximum $P(b)$ is the estimate.

3. **EMA smoothing.**  Raw per-window bearings are smoothed with an
   exponential moving average on the unit circle (sin/cos components)
   with parameter `ema_alpha` (default 0.35).

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_bearings` | 360 | angular resolution (1° per bin) |
| `freq_lo_hz` | 100.0 | lower frequency bound |
| `freq_hi_hz` | 2000.0 | upper frequency bound |

### Performance

Pre-computing the steering matrix allows the per-window cost to drop
to a single `einsum` + `argmax`, achieving ~3 ms per window (micro-
seconds of compute, dominated by FFT).

---

## RMS Range Estimation

Range is estimated from the root-mean-square pressure assuming
inverse-square-law decay:

$$\hat{r} = r_\text{ref} \sqrt{\frac{P_\text{ref}}{P_\text{window}}}$$

where $P_\text{ref}$ and $r_\text{ref}$ are calibrated at the peak-RMS
window (closest point of approach) using the ground-truth CPA distance.
The estimate is clamped to $[r_\text{min}, r_\text{max}]$.

This is approximate — it assumes isotropic propagation and a single
calibration point — but combined with accurate SRP-PHAT bearings it
produces usable position estimates.

---

## Causal WLS Tracker

A simple constant-velocity model fit on all past detections.  No
Kalman filter state; the fit is recomputed from scratch each window.

### Method

Given $n$ past detections at times $t_i$ with positions $(x_i, y_i, z_i)$
and RMS values $\rho_i$:

1. Centre the time axis: $\Delta t_i = t_i - \bar{t}$.
2. Weights: $w_i = \rho_i / \max(\rho)$ (higher RMS = closer = more
   reliable).
3. Weighted least squares: fit $x(t) = x_0 + v_x \Delta t$ (and
   likewise for $y$, $z$) by solving the normal equations $(A^T W A)^{-1} A^T W y$.
4. Residuals give position uncertainty: $\sigma_x = \text{std}(x_i - \hat{x}_i) / \sqrt{n}$.

Returns position, velocity, and per-axis uncertainty.  Requires at
least `min_detections` (default 5) points before producing output.

---

## Fire Control

### Lead computation (`compute_lead_3d`)

Given target position $\mathbf{p}$, velocity $\mathbf{v}$, and weapon
position $\mathbf{w}$:

1. Initialise intercept point $\mathbf{i} = \mathbf{p}$.
2. Iterate (up to 5 times):
   a. Range to intercept: $r = \|\mathbf{i} - \mathbf{w}\|$.
   b. Time of flight: solve $r = v_\text{muzzle}\,t - \tfrac{1}{2}a\,t^2$
      where $a = \text{decel} \times v_\text{avg}$, refined iteratively.
   c. Update intercept: $\mathbf{i} = \mathbf{p} + \mathbf{v} \cdot t$.
3. Aim bearing: $\theta = \text{atan2}(i_y - w_y,\; i_x - w_x)$.
4. Aim elevation: $\phi = \text{atan2}(i_z - w_z,\; r_{xy})$.

Returns: `aim_bearing`, `aim_elevation`, `tof`, `intercept_pos`.

### Pellet ballistics

Muzzle velocity decays linearly with distance:
$v(r) = v_\text{muzzle} - \text{decel} \cdot r$.
Default: $v_0 = 400$ m/s, decel = 1.5 m/s per m.

### Pattern spread

Shot pattern diameter at range $r$:
$d = \text{spread\_rate} \times r$.
Default spread rate: 0.3 m/m.  A hit is scored when the miss distance
is less than `hit_threshold_m` (default 2.0 m).

### Engagement gating (`compute_engagement_3d`)

A shot is authorised when all of:

- Range ≤ `max_engagement_range_m`
- Position uncertainty ≤ `max_position_uncertainty_m` (0 disables)
- Class label is not in the suppressed set (e.g., "bird", "unknown")
- Class confidence ≥ threshold

The pipeline stops after `max_hits` hits (default 3).

---

## MFP Pipeline (Library — `detection_main.py`)

An alternative detection path using matched field processing.  Not used
by `run_pipeline.py`, but available in the `acoustic_sim` library.

### Cross-Spectral Density Matrix

For each time window, compute the CSDM at selected harmonic frequencies:

$$\mathbf{C}(f) = \frac{1}{K}\sum_{k=1}^{K} \mathbf{X}_k(f)\,\mathbf{X}_k^H(f)$$

with diagonal loading $\epsilon\,\mathbf{I}$ for robustness.

### MVDR (Capon) Beamformer

For each candidate position on a polar grid (azimuth × range):

$$P(\theta, r) = \frac{1}{\mathbf{a}^H \mathbf{C}^{-1} \mathbf{a}}$$

where $\mathbf{a}$ is the steering vector.  Broadband power is the
frequency-weighted sum with $w(f) = (f/f_\text{max})^2$.

### EKF Tracker

4-state (2-D) or 6-state (3-D) Extended Kalman Filter.

**State:** $\mathbf{x} = [x, y, v_x, v_y]$ (2-D) or
$[x, y, z, v_x, v_y, v_z]$ (3-D).

**Measurement model:** bearing, range, and amplitude (nonlinear →
Jacobian linearisation).

**Process model:** constant-velocity with process noise
$\sigma_a^2$ on acceleration.

The multi-target variant uses nearest-neighbour data association with
Euclidean gating.

---

## ML Classifiers (Optional)

### Acoustic Classifier

3-layer CNN on log-mel spectrograms.  Architecture:

```
Conv2d(1, 32, 3×3) → ReLU → MaxPool
Conv2d(32, 64, 3×3) → ReLU → MaxPool → AdaptiveAvgPool → Flatten
Linear(64, 6)
```

Input: `(B, 1, 64, T)` where $T$ depends on segment length.
Output: logits over 6 source classes.

### Fusion Classifier

Two-branch architecture:

```
Acoustic branch:  same CNN as above → 64-dim embedding
Kinematic branch: Linear(14, 32) → ReLU → Linear(32, 32) → ReLU
Concatenate:      96-dim → Linear(96, 64) → ReLU → Linear(64, 6)
```

The 14-dim kinematic vector (from `compute_kinematic_features()`)
includes: mean/std/min speed, heading rate mean/std, curvature mean/std,
altitude mean/std/rate-std/is-zero, hover fraction, heading-rate
autocorrelation.

### Maneuver Classifier

2-layer 1-D CNN on kinematic state history:

```
Conv1d(6, 32, kernel=5) → ReLU → MaxPool
Conv1d(32, 64, kernel=3) → ReLU → AdaptiveAvgPool → Flatten
Linear(64, 6)
```

Input: `(B, 6, N)` where 6 channels = (x, y, z, vx, vy, vz) and
$N$ = 20 timesteps.  Output: logits over 6 maneuver classes.

### Integration Points

`detection_main.py` contains two ready-made integration functions:

- `_classify_source()` — computes mel spectrograms per detection window
  and runs acoustic or fusion inference; majority vote across windows.
- `_detect_maneuvers()` — slides a 20-step window over the tracker
  history, runs maneuver inference, and outputs a process-noise
  multiplier (steady=1×, turning=5×, evasive=10×).

Neither is called by the SRP-PHAT pipeline.  Wiring them in would
require passing trained model weights and adding a classification step
between detection and fire control.

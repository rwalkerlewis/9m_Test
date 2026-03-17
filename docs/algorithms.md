# Algorithms

## For the Non-Specialist

This system listens to a drone with a ring of microphones, works out
which direction the sound is coming from and how far away it is, tracks
the drone's path, then computes a shotgun aim point that leads the
target.  All of this happens in real time, using only past data — the
system never peeks ahead.

The direction-finding works like "acoustic triangulation": every
microphone pair sees the sound arrive at slightly different times, and
by combining all those tiny time differences the software reconstructs
the bearing to the source.  Range comes from how loud the sound is
(farther = quieter).  Tracking smooths out the noisy single-window
estimates into a clean trajectory, and fire control solves the
geometry of a slow pellet intercepting a fast drone.

---

## 1  Bearing Estimation

Three bearing estimators are available.  All share the abstract base
class `BearingEstimator` and return a `BearingResult` containing one or
more `BearingDetection` (bearing + power).

### 1.1  SRP-PHAT (Primary)

Steered Response Power with Phase Transform.  This is the default
bearing estimator used by the engagement pipeline.

**Module:** `acoustic_sim.detection.bearing.SRPBeamformer`

#### Pre-computation (once)

For each candidate bearing $b \in \{0, 1, \ldots, B-1\}$, compute the
inter-microphone propagation delays relative to the array centre.  Let
$\mathbf{u}_b = (\cos\theta_b,\; \sin\theta_b)$ be the unit look
direction and $\mathbf{r}_m$ the position of microphone $m$.  The delay
for microphone $m$ at bearing $b$ is

$$
\tau_{m,b} = \frac{(\mathbf{r}_m - \bar{\mathbf{r}}) \cdot \mathbf{u}_b}{c}
$$

where $\bar{\mathbf{r}}$ is the array centroid and $c$ is the sound
speed.  Convert to a steering matrix indexed over frequency:

$$
S_{m,b,k} = \exp\!\bigl(j\, 2\pi f_k\, \tau_{m,b}\bigr)
$$

where $k$ runs over the FFT bins in $[f_\text{lo},\, f_\text{hi}]$.
The matrix has shape $(M \times B \times K)$ and is stored as complex64.

#### Per-window processing

For a segment of $N$ samples at each of $M$ microphones:

1. Compute the FFT of each channel:
   $X_m(f_k) = \mathrm{FFT}[\mathrm{seg}_m]$.

2. Apply PHAT (phase-only) weighting:

$$
\hat{X}_m(f_k) = \frac{X_m(f_k)}{|X_m(f_k)| + \epsilon}
$$

3. Compute steered power at each candidate bearing:

$$
P(b) = \sum_{k} \left| \sum_{m=1}^{M} \hat{X}_m(f_k) \cdot S_{m,b,k} \right|^2
$$

4. The bearing with maximum $P(b)$ is the primary estimate.  If
   multi-source mode is enabled (`max_sources > 1`), peaks separated
   by at least `min_peak_sep_deg` and exceeding
   `secondary_threshold × P_\text{max}` are returned.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_bearings` | 360 | Angular grid size ($1°$ per bin) |
| `freq_lo` | 100 Hz | Lower passband edge |
| `freq_hi` | 2000 Hz | Upper passband edge |
| `min_peak_sep_deg` | 15.0 | Minimum angular separation between peaks |
| `secondary_threshold` | 0.3 | Fraction of max power for secondary peaks |

### 1.2  MUSIC

Multiple Signal Classification — a subspace method that models the
received signal as a sum of $D$ narrowband plane waves in additive
noise.

**Module:** `acoustic_sim.detection.bearing.MUSICEstimator`

1. Estimate the cross-spectral density matrix (CSDM) $\hat{\mathbf{R}}$
   from the FFT of the segment, with diagonal loading
   $\delta\mathbf{I}$:

$$
\hat{\mathbf{R}} = \frac{1}{K}\sum_{k} \mathbf{X}(f_k)\,\mathbf{X}^H(f_k) + \delta\,\mathbf{I}
$$

2. Eigen-decompose $\hat{\mathbf{R}} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^H$.
   Estimate the number of sources $D$ using the Minimum Description
   Length (MDL) criterion:

$$
\mathrm{MDL}(k) = -N(M-k)\ln\!\left(\frac{\prod_{i=k+1}^{M}\lambda_i^{1/(M-k)}}{\frac{1}{M-k}\sum_{i=k+1}^{M}\lambda_i}\right) + \frac{1}{2}k(2M-k)\ln N
$$

   where $N$ is the number of snapshots and $\lambda_i$ are the
   eigenvalues sorted in decreasing order.

3. The noise subspace is $\mathbf{E}_n = [\mathbf{u}_{D+1}, \ldots, \mathbf{u}_M]$.
   Compute the pseudo-spectrum:

$$
P_\text{MUSIC}(\theta) = \frac{1}{\mathbf{a}^H(\theta)\,\mathbf{E}_n\,\mathbf{E}_n^H\,\mathbf{a}(\theta)}
$$

   where $\mathbf{a}(\theta)$ is the far-field steering vector at
   bearing $\theta$.

4. Peaks of $P_\text{MUSIC}(\theta)$ give the DOA estimates.

### 1.3  MVDR (Capon)

Minimum Variance Distortionless Response beamformer on a polar
(azimuth × range) grid.

**Module:** `acoustic_sim.detection.bearing.MVDRBeamformer`

The MVDR estimator inverts the CSDM at selected harmonic frequencies of
the expected source and computes:

$$
P_\text{MVDR}(\theta) = \frac{1}{\mathbf{a}^H(\theta)\,\hat{\mathbf{R}}^{-1}\,\mathbf{a}(\theta)}
$$

Broadband power is the frequency-weighted sum with
$w(f) = (f/f_\text{max})^2$.  This estimator also supports stationary
interference rejection and fault-weighted sensors.

### 1.4  EMA Bearing Smoother

Raw per-window bearings are smoothed with an exponential moving average
on the unit circle.  Let $\theta_k$ be the raw bearing at window $k$;
the smoother maintains running averages of $\sin$ and $\cos$:

$$
\bar{s}_k = \alpha \sin\theta_k + (1-\alpha)\,\bar{s}_{k-1}, \qquad
\bar{c}_k = \alpha \cos\theta_k + (1-\alpha)\,\bar{c}_{k-1}
$$

$$
\hat{\theta}_k = \mathrm{atan2}(\bar{s}_k,\; \bar{c}_k)
$$

Default $\alpha = 0.35$.  This avoids discontinuities at the
$\pm\pi$ wrap-around that plague naïve linear averaging.

---

## 2  Range Estimation

Three range estimators are available, sharing the abstract base class
`RangeEstimator`.  The pipeline can auto-select between them based on
distance (see §2.4).

### 2.1  RMS Inverse-Square Law

**Module:** `acoustic_sim.detection.ranging.RMSRangeEstimator`

Assumes free-field spherical spreading.  Given a calibration point
$(R_\text{ref},\, A_\text{ref})$ where $A$ is the RMS pressure:

$$
\hat{R} = R_\text{ref}\,\sqrt{\frac{A_\text{ref}}{A_\text{obs}}}
$$

The calibration is obtained from the peak-RMS window and its
corresponding ground-truth CPA distance.  The estimate is clamped to
$[R_\text{min},\, R_\text{max}]$.

**Strengths:** Fast, one scalar per window.  Works well at medium to
long range where propagation is approximately spherical.

**Weaknesses:** Assumes isotropic propagation, a single calibration
point, and no multipath.  Inaccurate in the nearfield where the
inverse-square law does not apply.

Uncertainty estimate: $\sigma_R = 0.3\,\hat{R}$.

### 2.2  TDOA Multilateration (GCC-PHAT)

**Module:** `acoustic_sim.detection.ranging.TDOARangeEstimator`

Uses Generalised Cross-Correlation with Phase Transform (GCC-PHAT) to
estimate time-delay-of-arrival for every microphone pair, then searches
along the DOA bearing ray for the range that minimises the TDOA
residuals.

#### GCC-PHAT TDOA extraction

For microphone pair $(i,j)$:

$$
G_{ij}(f) = \frac{X_i(f)\,X_j^*(f)}{|X_i(f)\,X_j^*(f)| + \epsilon}
$$

$$
R_{ij}(\tau) = \mathrm{IFFT}[G_{ij}(f)]
$$

The TDOA is the lag $\hat{\tau}_{ij}$ at the peak of $R_{ij}(\tau)$,
refined by parabolic interpolation.

#### Bearing-constrained range search

Given bearing $\theta$ from the DOA estimator, the candidate source
position at range $R$ is

$$
\mathbf{p}(R) = \bar{\mathbf{r}} + R\,(\cos\theta,\; \sin\theta)
$$

The predicted TDOA for pair $(i,j)$ is
$\tau_{ij}^\text{pred}(R) = (d_i - d_j)/c$ where
$d_m = \|\mathbf{p}(R) - \mathbf{r}_m\|$.  The cost function is

$$
J(R) = \sum_{(i,j)} \bigl(\tau_{ij}^\text{pred}(R) - \hat{\tau}_{ij}\bigr)^2
$$

Minimised over a log-spaced grid of $R$ values followed by golden-
section refinement on the best cell.

**Strengths:** Geometry-based; does not need amplitude calibration.
Accurate in the nearfield where microphone separations are large
relative to source distance.

**Weaknesses:** Requires $f_s \cdot d_\text{max}/c \gtrsim 50$ for
reliable GCC-PHAT peaks.

### 2.3  Bearing-Rate Range

**Module:** `acoustic_sim.detection.ranging.BearingRateRangeEstimator`

A kinematic estimator that infers range from the angular rate of change
of the bearing:

$$
\hat{R} \approx \frac{v_\perp}{|\dot{\theta}|}
$$

where $v_\perp$ is the component of source velocity perpendicular to
the line of sight (approximated as $v_\perp \approx v_\text{source}$
for a crossing target) and $\dot{\theta}$ is the EMA-smoothed bearing
rate.

The bearing rate is computed causally from
consecutive window bearings:

$$
\dot{\theta}_k = \alpha\,\frac{\Delta\theta}{\Delta t} + (1-\alpha)\,\dot{\theta}_{k-1}
$$

with wrap-around correction $|\Delta\theta| \leq \pi$.

When $|\dot{\theta}|$ falls below `min_rate_dps` (default 5°/s), range
is clamped to $R_\text{max}$ to avoid division by near-zero.

**Strengths:** Model-free, requires no cross-correlation, immune to
low-sample-rate TDOA resolution limits.

**Weaknesses:** Requires knowledge of approximate source speed;
over-estimates range for radial geometry (safe direction for
engagement).

### 2.4  Auto Range Selection

The pipeline (`run_pipeline.py`) supports automatic switching between
range methods based on the current range estimate.  When configured with
`auto_cpa_threshold_m`, the engine uses TDOA when the estimated range
falls below the threshold and RMS otherwise:

$$
\text{method} = \begin{cases}
\text{TDOA} & \text{if } \hat{R} \leq R_\text{threshold} \\
\text{RMS}  & \text{otherwise}
\end{cases}
$$

---

## 3  Tracking

### 3.1  Causal WLS Tracker

**Module:** `acoustic_sim.detection.tracking.CausalWLSTracker`

A constant-velocity model fit over a sliding window of past detections.
Unlike a Kalman filter, the entire fit is recomputed from scratch each
window — there is no recursive state update and therefore no risk of
filter divergence.

Given $n$ detections at times $t_i$ with Cartesian positions
$(x_i, y_i, z_i)$ and RMS amplitudes $\rho_i$:

1. Centre the time axis: $\Delta t_i = t_i - t_\text{ref}$ where
   $t_\text{ref}$ is the most recent detection time.

2. Compute weights: $w_i = \rho_i / \max_j(\rho_j)$.  Higher RMS
   implies closer range and therefore more reliable position.

3. Solve the weighted least-squares normal equations independently for
   each axis.  For axis $q \in \{x, y, z\}$:

$$
\begin{pmatrix} \hat{q}_0 \\ \hat{v}_q \end{pmatrix}
= (\mathbf{A}^\top \mathbf{W} \mathbf{A})^{-1}\,\mathbf{A}^\top \mathbf{W}\,\mathbf{q}
$$

   where

$$
\mathbf{A} = \begin{pmatrix} 1 & \Delta t_1 \\ \vdots & \vdots \\ 1 & \Delta t_n \end{pmatrix}, \quad
\mathbf{W} = \mathrm{diag}(w_1, \ldots, w_n)
$$

4. Residuals: $\sigma_q = \mathrm{std}(q_i - \hat{q}_i) / \sqrt{n}$.

5. The `TrackState` dataclass stores the fitted position
   $(\hat{x}_0,\, \hat{y}_0,\, \hat{z}_0)$, velocity
   $(\hat{v}_x,\, \hat{v}_y,\, \hat{v}_z)$, reference time $t_\text{ref}$,
   per-axis residuals, and detection count.

6. Extrapolation: `TrackState.position_at(t)` returns
   $\hat{\mathbf{p}}(t) = \hat{\mathbf{p}}_0 + \hat{\mathbf{v}} \cdot (t - t_\text{ref})$.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_detections` | 5 | Minimum points before producing a track |
| `max_history` | 20 | Sliding window size (oldest discarded) |

### 3.2  Track Blending

At close range the WLS fit can lag behind the true position because the
constant-velocity model cannot follow rapid angular changes.  The
pipeline blends raw and fitted positions:

$$
\hat{\mathbf{p}} = \beta\,\mathbf{p}_\text{fitted} + (1-\beta)\,\mathbf{p}_\text{raw}, \qquad
\beta = \min\!\left(\frac{R}{20},\; 1\right)
$$

At ranges beyond 20 m the fit dominates; at close range, raw detections
are trusted.

---

## 4  Fire Control

### 4.1  Pellet Ballistics

**Module:** `acoustic_sim.fire_control`

#### Velocity decay

Pellet velocity decays linearly with range due to drag:

$$
v(R) = \max\!\bigl(v_0 - d \cdot R,\; 0\bigr)
$$

where $v_0$ is the muzzle velocity (default 400 m/s) and $d$ is the
deceleration rate (default 1.5 (m/s)/m).

#### Time of flight

The range–time relation under linear drag is

$$
R = v_0\,t - \tfrac{1}{2}\,a\,t^2, \qquad a = d \cdot v_\text{avg}
$$

where $v_\text{avg}$ is iteratively refined as
$v_\text{avg} = \tfrac{1}{2}(v_0 + v_0 - d \cdot R)$.  The time of
flight is then $t = R / v_\text{avg}$.

#### Pattern spread

The shot pattern diameter at range $R$ is

$$
d_\text{pattern} = s \cdot R
$$

where $s$ is the spread rate.  The function `pattern_diameter(range_m,
spread_rate)` returns this value.  The default `spread_rate` in the
function signature is 0.025; the pipeline configuration
(`pipeline.config.json`) overrides this to 0.3.

### 4.2  Lead Computation (`compute_lead_3d`)

Given target position $\mathbf{p}$, target velocity $\mathbf{v}$, and
weapon position $\mathbf{w}$:

1. Initialise intercept point: $\mathbf{i}_0 = \mathbf{p}$.

2. Iterate up to 5 times:

   a. Compute range to current intercept:
      $R_k = \|\mathbf{i}_k - \mathbf{w}\|$.

   b. Compute time of flight $t_k$ by solving the ballistic range
      equation (§4.1) at range $R_k$.

   c. Update intercept:
      $\mathbf{i}_{k+1} = \mathbf{p} + \mathbf{v}\,t_k$.

3. Final aim direction:

$$
\theta_\text{aim} = \mathrm{atan2}(i_y - w_y,\; i_x - w_x)
$$

$$
\phi_\text{aim} = \mathrm{atan2}\!\bigl(i_z - w_z,\; \sqrt{(i_x - w_x)^2 + (i_y - w_y)^2}\bigr)
$$

Returns: `aim_bearing`, `aim_elevation`, `tof`, `intercept_pos`,
`pellet_speed_at_target`.

### 4.3  Bearing-Rate Fire Gating

**Module:** `acoustic_sim.fire_control.compute_bearing_rate`

Before firing, the pipeline evaluates the bearing rate
$\dot{\theta}$ over a short history window (default 0.15 s):

- **High bearing rate** (target crossing): the lead computation from
  §4.2 is used directly.
- **Low bearing rate** (target radial / receding): the shot is aimed
  at the current bearing with no or minimal lead.

This prevents the lead algorithm from computing wild aim points when the
target is moving directly toward or away from the weapon.

### 4.4  Engagement Gating (`compute_engagement_3d`)

A shot is authorised when all conditions are met:

| Condition | Parameter | Default |
|-----------|-----------|---------|
| Range to target ≤ max | `max_engagement_range_m` | 0 (disabled) |
| Position uncertainty ≤ max | `max_position_uncertainty_m` | 0 (disabled) |
| Class label not suppressed | `class_label` / `confidence_threshold` | "unknown" / 0.7 |
| Maneuver state check | `maneuver_class` | "steady" |

The class-based gating suppresses fire on labels "unknown" and "bird"
when the classifier confidence exceeds the threshold.  Maneuver-adaptive
thresholds are applied when a maneuver classifier is present (evasive
targets get tighter uncertainty gates).

### 4.5  CPA Evaluation and Hit Scoring

After fire is authorised, the pipeline simulates the shot:

1. **Projectile path** (`projectile_path`): Generates the 3-D pellet
   trajectory from weapon position along the aim direction, sampling
   $n$ points with ballistic velocity decay.

2. **Closest point of approach** (`find_cpa`): Evaluates the distance
   between the projectile path and the target trajectory (from
   ground-truth `target_position_fn`) at $n$ time samples over the
   time-of-flight window.  Returns the minimum distance and the time
   at which it occurs.

3. **Pattern radius at CPA**: $r_\text{pattern} = \tfrac{1}{2}\,s \cdot R_\text{CPA}$.

4. **Effective hit threshold**:

$$
\tau_\text{eff} = \max\!\bigl(r_\text{pattern},\; \tau_\text{hit}\bigr)
$$

   where $\tau_\text{hit}$ is the raw `hit_threshold` parameter
   (default 2.0 m for 3-D, 1.0 m for 2-D).

5. **Hit decision**: $\text{hit} = (d_\text{CPA} \leq \tau_\text{eff})$.

The pipeline stops firing after `max_hits` (default 3) scored hits.

---

## 5  MFP / EKF Pipeline (Library)

An alternative detection path using matched field processing.
Implemented in `detection_main.py`; not used by the production
SRP-PHAT pipeline.

### 5.1  Cross-Spectral Density Matrix

For each time window, the CSDM is computed at selected harmonic
frequencies:

$$
\hat{\mathbf{C}}(f) = \frac{1}{K}\sum_{k=1}^{K} \mathbf{X}_k(f)\,\mathbf{X}_k^H(f) + \delta\,\mathbf{I}
$$

where $K$ is the number of sub-windows and $\delta$ is the diagonal
loading level.

### 5.2  MVDR Beamformer (Polar Grid)

For each candidate position on a polar grid (azimuth × range),
compute the steering vector $\mathbf{a}(\theta, R)$ from the
microphone-to-candidate distances.  The MVDR power is:

$$
P(\theta, R) = \frac{1}{\mathbf{a}^H\,\hat{\mathbf{C}}^{-1}\,\mathbf{a}}
$$

Broadband power is the frequency-weighted sum:
$P_\text{BB} = \sum_f w(f)\,P(\theta,R,f)$ with $w(f) = (f/f_\text{max})^2$.

Peak finding with parabolic interpolation yields bearing and range
estimates.

### 5.3  EKF Tracker

4-state (2-D) or 6-state (3-D) Extended Kalman Filter.

**State vector:** $\mathbf{x} = [x,\, y,\, v_x,\, v_y]^\top$ (2-D)
or $[x,\, y,\, z,\, v_x,\, v_y,\, v_z]^\top$ (3-D).

**Process model:** constant velocity plus Gaussian acceleration noise:

$$
\mathbf{x}_{k+1} = \mathbf{F}\,\mathbf{x}_k + \mathbf{w}_k, \qquad
\mathbf{F} = \begin{pmatrix}\mathbf{I} & \Delta t\,\mathbf{I} \\ \mathbf{0} & \mathbf{I}\end{pmatrix}
$$

**Measurement model:** bearing $\theta = \mathrm{atan2}(y - y_m,\, x - x_m)$,
range $r = \|\mathbf{p} - \mathbf{p}_m\|$, and amplitude $A \propto 1/r$.
The Jacobian is computed analytically.

The multi-target variant (`MultiTargetTracker`) uses nearest-neighbour
data association with Euclidean gating, track initiation on
unassociated detections, and track deletion after $N$ missed updates.

---

## 6  ML Classifiers (Optional)

All classifiers require PyTorch.  No pre-trained weights are shipped.

### 6.1  Acoustic Classifier

**Module:** `acoustic_sim.ml.acoustic_classifier.AcousticClassifier`

Three-layer 2-D CNN on log-mel spectrograms:

```
Conv2d(1, 16, 3×3, pad=1) → BatchNorm2d(16) → ReLU → MaxPool2d(2)
Conv2d(16, 32, 3×3, pad=1) → BatchNorm2d(32) → ReLU → MaxPool2d(2)
Conv2d(32, 64, 3×3, pad=1) → BatchNorm2d(64) → ReLU → AdaptiveAvgPool2d(1)
Flatten → Linear(64, 6)
```

Input: $(B, 1, n_\text{mels}, T)$.  Output: logits over 6 source
classes.  `get_embedding()` returns the 64-dim vector before the final
linear layer.

### 6.2  Fusion Classifier

**Module:** `acoustic_sim.ml.fusion_classifier.FusionClassifier`

Two-branch architecture:

- **Acoustic branch:** identical CNN to §6.1 → 64-dim embedding.
- **Kinematic branch:** `Linear(14, 32) → ReLU → Linear(32, 32) → ReLU`
  → 32-dim embedding.
- **Fusion head:** concatenate (96-dim) → `Linear(96, 64) → ReLU → Linear(64, 6)`.

The 14-dim kinematic vector (from `compute_kinematic_features()`)
includes: mean/std/min speed, heading rate mean/std, curvature
mean/std, altitude mean/std/rate-std/is-zero, hover fraction, heading
rate autocorrelation.

### 6.3  Maneuver Classifier

**Module:** `acoustic_sim.ml.maneuver_classifier.ManeuverClassifier`

Two-layer 1-D CNN on kinematic state history:

```
Conv1d(6, 32, kernel=5, pad=2) → ReLU → AdaptiveAvgPool1d
Conv1d(32, 64, kernel=5, pad=2) → ReLU → AdaptiveAvgPool1d(1)
Flatten → Linear(64, 6)
```

Input: $(B, 6, N)$ where 6 channels = $(x, y, z, v_x, v_y, v_z)$ and
$N = 20$ timesteps.  Output: logits over 6 maneuver classes.

### 6.4  Integration Points

`detection_main.py` contains two ready-made integration functions:

- `_classify_source()` — runs acoustic or fusion inference per window;
  majority vote across windows.
- `_detect_maneuvers()` — slides a 20-step window over the tracker
  history and returns a process-noise multiplier
  (steady = $1\times$, turning = $5\times$, evasive = $10\times$).

Neither is called by the production SRP-PHAT pipeline.

---

## 7  FNO Surrogate (Optional)

**Module:** `acoustic_sim.ml.fno`

The Fourier Neural Operator is a deep-learning surrogate that can
replace the FDTD solver for rapid forward propagation.

### Architecture

- **Input:** 4-channel field — normalised velocity model, source $x/y$
  Gaussian blobs, frequency encoding.
- **`SpectralConv2d`:** retains the lowest Fourier modes via truncation
  in frequency space; equivalent to a global convolution kernel.
- **`FNOBlock2d`:** spectral convolution + pointwise
  $1\times1$ convolution + residual skip + GELU activation.
- **`TraceDecoder`:** MLP that maps latent features at receiver
  locations to time-domain traces.
- **Output:** $(B,\, n_\text{recv},\, n_\text{time})$.

Training uses relative $L^2$ loss, masked receiver padding, cosine
learning rate schedule, and gradient clipping.

`FNOForwardModel` is a drop-in replacement for FDTD:
`predict(velocity_field, grid, receivers, source) → traces`.

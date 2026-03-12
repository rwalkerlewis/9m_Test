# Algorithm Descriptions

This document describes every algorithm in `acoustic-sim` at a level of detail sufficient to reimplement them. Each section includes the mathematical formulation, pseudocode-level description, implementation notes, and cross-references to the source code.

For the underlying physics, see [Physics Background](physics.md).

---

## Table of Contents

1. [Helmholtz Solver](#1-helmholtz-solver)
2. [FDTD Solver](#2-fdtd-solver)
3. [Matched Field Processor](#3-matched-field-processor)
4. [Extended Kalman Filter Tracker](#4-extended-kalman-filter-tracker)
5. [Fire Control](#5-fire-control)
6. [Noise Generation](#6-noise-generation)
7. [Validation Checks](#7-validation-checks)

---

## 1. Helmholtz Solver

**Source:** `src/acoustic_sim/solver.py` → `solve_helmholtz()`

### 1.1 Problem Statement

Solve the 2D Helmholtz equation on a heterogeneous velocity model:

```
∇²P + k²(x,y)·P = -S(x,y)
```

where `k(x,y) = ω/c(x,y)` is the spatially varying wavenumber and `S` is a point source.

### 1.2 Discretisation

The Laplacian is discretised with the standard 5-point stencil on a uniform grid with spacing `dx`:

```
∇²P ≈ [P(i+1,j) + P(i-1,j) + P(i,j+1) + P(i,j-1) - 4·P(i,j)] / dx²
```

Each grid point `(i,j)` contributes one row to a linear system `A·p = b`:

```
A[row, row]           = -4/dx² + k²(i,j)·(1 + i·σ(i,j))
A[row, row ± 1]       = 1/dx²   (x-neighbours)
A[row, row ± nx]      = 1/dx²   (y-neighbours)
```

where `σ(i,j)` is the absorbing-layer damping coefficient and the factor `(1 + i·σ)` makes the wavenumber complex, providing absorption.

### 1.3 Absorbing Boundary Layer

The damping coefficient ramps quadratically from 0 at the inner edge of the sponge to 0.7 at the domain boundary:

```
σ(i,j) = 0.7 × ((w - d_edge) / w)²
```

where `d_edge = min(i, nx-1-i, j, ny-1-j)` is the distance from the nearest boundary and `w` is the sponge width (auto-chosen as `max(3, int(0.08 × min(nx, ny)))` if not specified).

### 1.4 Source Injection

The source is injected as a point source at the nearest grid point to `(source_x, source_y)`:

```
b[flat(sx, sy)] = -1/dx²
```

where `flat(ix, iy) = iy × nx + ix` converts 2D indices to a flat index.

### 1.5 Solution

The sparse system `A·p = b` is assembled as a `scipy.sparse.lil_matrix` (for efficient row-wise construction), converted to CSR format, and solved using `scipy.sparse.linalg.spsolve` (direct sparse LU factorisation).

The output is `|P(x,y)|` — the pressure magnitude at each grid point.

### 1.6 Complexity

For a grid of `N = nx × ny` points, the system has `N` unknowns with `~5N` nonzeros. Sparse LU factorisation has complexity `O(N^{3/2})` for a 2D problem with nested dissection ordering (scipy uses SuperLU). For the default grid (100 × 100 = 10,000 points), this is fast (< 1 second). For very large grids (> 10⁶ points), the Helmholtz solver becomes memory-intensive; the FDTD solver is preferred.

---

## 2. FDTD Solver

**Source:** `src/acoustic_sim/fdtd.py` → `FDTDSolver`

### 2.1 Time-Stepping Scheme

The 2D scalar wave equation is discretised using the **leapfrog** (Störmer–Verlet) scheme:

```
p^{n+1}[i,j] = 2·p^n[i,j] - p^{n-1}[i,j] + C²[i,j]·∇²_h p^n[i,j] - σ[i,j]·(p^n[i,j] - p^{n-1}[i,j])
```

where:
- `p^n` is the pressure field at timestep `n`
- `C²[i,j] = (c[i,j] × dt / dx)²` is the squared Courant number
- `∇²_h` is the discrete Laplacian (configurable order)
- `σ[i,j]` is the damping coefficient

This is a three-level explicit scheme: given `p^{n-1}` and `p^n`, the next field `p^{n+1}` is computed without solving a linear system.

### 2.2 Higher-Order Spatial Stencils

The spatial Laplacian uses central finite-difference stencils of configurable order (2, 4, 6, 8, ...). The stencil half-width is `M = fd_order // 2`.

**FD coefficients** for the second derivative are derived by solving a Vandermonde system that enforces polynomial exactness up to degree `2M`:

```
For j = 1, ..., M:
    Σ_{k=1}^{M} c_k · k^{2j} = δ_{j,1}

c_0 = -2 · Σ_{k=1}^{M} c_k
```

The resulting coefficients for common orders:

| Order | M | Coefficients [c₀, c₁, c₂, ...] | Stencil |
|---|---|---|---|
| 2 | 1 | [-2, 1] | `[-1, 2, -1] / dx²` |
| 4 | 2 | [-5/2, 4/3, -1/12] | 9-point |
| 8 | 4 | [-205/72, 8/5, -1/5, 8/315, -1/560] | 17-point |

The discrete Laplacian for a 2D field is:

```
∇²_h p[i,j] = c₀ · 2 · p[i,j]    (centre, ×2 for x + y)
             + Σ_{k=1}^{M} c_k · (p[i+k,j] + p[i-k,j] + p[i,j+k] + p[i,j-k])
```

### 2.3 CFL Condition for General Stencils

The CFL limit depends on the spectral radius of the stencil at Nyquist frequency. For a stencil with coefficients `[c₀, c₁, ..., c_M]`, the 1D spectral radius is:

```
ρ = |c₀ + 2·Σ_{k=1}^{M} c_k·(-1)^k|
```

The 2D CFL condition is:

```
dt ≤ 2·dx / ((c_max + |v_wind|) · √(2·ρ))
```

For order 2: `ρ = |-2 + 2·1·(-1)¹| = 4`, so `dt ≤ 2·dx / (c_eff · √8) = dx / (c_eff · √2)`, recovering the classical formula.

### 2.4 Sponge-Layer Absorbing Boundary

The damping array `σ[i,j]` has two components:

1. **Background absorption** (`air_absorption`, default 0.005): Applied everywhere.
2. **Sponge ramp**: At cells within `damping_width` of any boundary:

```
σ[i,j] = air_absorption + (damping_max - air_absorption) × ((w - d) / w)²
```

where `d = min(i, nx-1-i, j, ny-1-j)` and `w = damping_width`.

The damping term `σ·(p^n - p^{n-1})` in the update equation acts as a first-order absorbing mechanism: it subtracts a fraction of the velocity (proportional to the time difference of pressure fields) at each timestep, removing energy from outgoing waves.

### 2.5 Bilinear Source Injection

The source position `(sx, sy)` generally does not coincide with a grid node. The source amplitude is distributed to the four surrounding nodes using bilinear interpolation:

```
Given continuous indices (gx, gy):
    ix0 = floor(gx),  iy0 = floor(gy)
    fx = gx - ix0,     fy = gy - iy0

    p[iy0,   ix0  ] += A × (1-fx) × (1-fy)
    p[iy0,   ix0+1] += A ×   fx   × (1-fy)
    p[iy0+1, ix0  ] += A × (1-fx) ×   fy
    p[iy0+1, ix0+1] += A ×   fx   ×   fy
```

where `A = source_amplitude × signal[n]`. This smoothly distributes the point source across the grid, reducing numerical artefacts from a single-cell source.

For moving sources, the injection position changes at each timestep via `source.position_at(step, dt)`.

### 2.6 Bilinear Receiver Sampling

Receiver positions are also generally off-grid. The pressure at each receiver is sampled using the same bilinear weights, pre-computed once during initialisation:

```
p_recv = p[iy, ix]·(1-wx)·(1-wy) + p[iy, ix+1]·wx·(1-wy)
       + p[iy+1, ix]·(1-wx)·wy   + p[iy+1, ix+1]·wx·wy
```

This produces smooth, alias-free traces.

### 2.7 MPI Domain Decomposition

The global grid of `ny` rows is partitioned along the **y-axis** into contiguous strips, one per MPI rank. Each rank owns `local_ny = ny // size` rows (with remainder distributed to lower ranks) plus `M` ghost rows on each internal boundary for the stencil.

**Decomposition algorithm:**

```
For rank r of size P:
    base = ny // P
    remainder = ny % P
    row_start = Σ_{i=0}^{r-1} (base + (1 if i < remainder else 0))
    row_end = row_start + base + (1 if r < remainder else 0)
    ghost_top = M  if r > 0      else 0
    ghost_bot = M  if r < P-1    else 0
    pad_ny = local_ny + ghost_top + ghost_bot
```

**Halo exchange** (every timestep, for both `p_now` and `p_prev`):

```
For each pressure field:
    If rank > 0:
        Send top M owned rows to rank-1
        Receive M rows from rank-1 into top ghost zone
    If rank < P-1:
        Send bottom M owned rows to rank+1
        Receive M rows from rank+1 into bottom ghost zone
```

This uses `MPI.Sendrecv` for deadlock-free bidirectional exchange.

**Receiver sampling** across ranks: Each rank identifies which receivers fall within its local strip. After sampling, the global receiver traces are gathered to rank 0 via `MPI.gather`.

### 2.8 CUDA Acceleration

When `use_cuda=True`, the `get_backend()` function returns CuPy instead of NumPy. All large arrays (pressure fields, velocity model, damping) reside in GPU memory. The stencil computation and field update use CuPy array operations, which are automatically GPU-accelerated.

For MPI + CUDA, ghost rows must be transferred between host (CPU) and device (GPU) memory:

```
Send: GPU array → cupy.asnumpy() → MPI.Send → host buffer
Recv: host buffer → MPI.Recv → cupy.asarray() → GPU array
```

If CuPy is unavailable or no GPU is detected, the code falls back to NumPy with a warning.

### 2.9 Output

The solver produces:

- `traces`: `(n_receivers, n_steps)` array of pressure time-series at each receiver
- `dt`: actual timestep used
- `n_steps`: total number of timesteps
- Snapshots (optional): full wavefield images at regular intervals

---

## 3. Matched Field Processor

**Source:** `src/acoustic_sim/processor.py` → `matched_field_process()`

### 3.1 Overview

The MFP takes microphone traces and produces bearing/range estimates of acoustic sources. It operates on a polar grid and uses broadband MVDR beamforming with harmonic selection.

### 3.2 Processing Pipeline

For each time window (sliding with overlap):

```
INPUT: traces (n_mics × n_samples), mic_positions, dt
OUTPUT: list of detections with bearing, range, coherence, Cartesian position

1. Pre-processing:
   a. Position calibration (if enabled) — see §3.6
   b. Transient blanking (if enabled) — see §3.5
   c. Sensor fault detection & weighting (if enabled) — see §3.4
   d. Zero out traces from faulty sensors

2. Grid setup (computed once):
   a. Build polar grid: azimuths (0°–360°, 1° steps), ranges (20–500 m, 5 m steps)
   b. Convert to Cartesian: gx, gy = polar_to_cartesian(azimuths, ranges, cx, cy)
   c. Compute travel times: τ[i,j,m] = dist(grid[i,j], mic[m]) / c
   d. Select harmonic frequency bins from sub-window FFT resolution
   e. Compute steering vectors: a[f,i,j,m] = exp(-j·2π·f·τ[i,j,m])

3. Per time window:
   a. Check window RMS — skip if below min_signal_rms
   b. Compute CSDM:
      - Divide window into n_subwindows sub-windows
      - For each sub-window: Hann taper → FFT → select harmonic bins
      - Average outer products: C[f] = (1/K) Σ d_k·d_k^H
   c. MVDR beam power:
      - For each frequency bin f:
        - Diagonal loading: C += ε·trace(C)/M · I
        - If condition number < 10⁶: P[f,i,j] = 1/(a^H·C⁻¹·a)
        - Else (fallback): P[f,i,j] = a^H·C·a
   d. Frequency weighting: w[f] = (f/f_max)²
   e. Broadband sum: BPM[i,j] = Σ_f w[f]·P[f,i,j]
   f. Normalise: BPM /= max(BPM)
   g. Stationary source rejection:
      - Maintain history of last N beam-power maps
      - Compute coefficient of variation at each grid point
      - Mask points where CV < threshold (stationary sources)
   h. Peak finding with sub-grid interpolation:
      - Find global maximum
      - Parabolic interpolation in azimuth (circular) and range
      - Record peak, apply exclusion zone
      - Repeat for max_sources
   i. Convert peaks to Cartesian: x = cx + r·cos(θ), y = cy + r·sin(θ)
```

### 3.3 CSDM Computation Details

The CSDM is computed per harmonic frequency bin by averaging over sub-windows (Welch's method):

```python
for each sub-window k (of length sub_len):
    segment = traces[:, s0:s1]          # (n_mics, sub_len)
    windowed = segment × hann(sub_len)  # Hann taper
    spectra = FFT(windowed, axis=1)     # (n_mics, n_freq)
    for each harmonic bin f:
        d = spectra[:, f]               # (n_mics,) complex
        C[f] += outer(d, d.conj())      # rank-1 update
C /= n_subwindows
```

The sub-window length is `window_length / n_subwindows`. This averaging reduces the variance of the CSDM estimate at the cost of frequency resolution. With 4 sub-windows (default), the frequency resolution is `fs / sub_len ≈ 20 Hz`, which is adequate given the 10 Hz harmonic bandwidth.

### 3.4 Sensor Fault Detection

**Source:** `processor.py` → `compute_sensor_weights()`

The total power of each sensor's trace is compared to the median:

```
power[m] = Σ_t traces[m, t]²
ratio[m] = power[m] / median(power)

weights[m] = 0  if ratio[m] > fault_threshold (default 10)
           = 0  if ratio[m] < 1/fault_threshold
           = 1  otherwise
```

Sensors with power more than 10× the median (overloaded, noisy) or less than 1/10 the median (dead) are given zero weight and their traces are zeroed before CSDM computation.

### 3.5 Transient Blanking

**Source:** `processor.py` → `blank_transients()`

The trace is divided into short sub-windows (default 5 ms). The total energy in each sub-window is compared to the median:

```
For each sub-window w of length sub_len:
    energy[w] = Σ_{m,t} traces[m, w_start:w_end]²

bad[w] = True if energy[w] > threshold_factor × median(energy)

For bad sub-windows: set traces[:, w_start:w_end] = 0
```

This zeros out impulsive events (explosions, slammed doors) that would otherwise corrupt the CSDM. The default threshold factor of 10 catches events that are 10 dB above the median energy level.

### 3.6 Position Self-Calibration

**Source:** `processor.py` → `calibrate_positions()`

When microphone positions have errors, the steering vectors are computed from incorrect positions, degrading localisation. The calibration algorithm uses cross-correlation TDOA estimates between all microphone pairs to refine positions:

```
1. For each mic pair (i, j):
   a. Cross-correlate traces[i] and traces[j]
   b. Find the peak lag → observed TDOA: τ_obs
   c. Predict TDOA from reported positions: τ_pred = (d_i - d_j) / c
   d. Residual: Δτ = τ_obs - τ_pred

2. Build least-squares system A·δ = b:
   - A: Jacobian of TDOA w.r.t. position perturbations
   - b: vector of TDOA residuals
   - Constraint rows: mean perturbation = 0 (preserve centre)

3. Solve: δ = (A^T A)^{-1} A^T b
4. Corrected positions = reported + δ
```

This is a linearised TDOA self-calibration that works even without a known calibration source — it uses the ambient sound field (drone + noise) as the calibration signal. The accuracy depends on having sufficient signal coherence between microphone pairs.

---

## 4. Extended Kalman Filter Tracker

**Source:** `src/acoustic_sim/tracker.py` → `EKFTracker`, `run_tracker()`

### 4.1 Algorithm Outline

```
INPUT: list of detections (time, bearing, range, coherence, detected flag)
OUTPUT: smoothed track (times, positions, velocities, covariances)

1. Wait for first detection with valid bearing
2. Initialise state from bearing + range guess (anisotropic covariance)
3. For each subsequent detection:
   a. Predict: x⁻ = F·x⁺, P⁻ = F·P⁺·F^T + Q
   b. If detection available:
      - Compute innovation: ỹ = z - h(x⁻), wrap bearing to (-π,π]
      - Compute Jacobian H at x⁻
      - Update: K = P⁻·H^T·(H·P⁻·H^T + R)⁻¹
                x⁺ = x⁻ + K·ỹ
                P⁺ = (I - K·H)·P⁻
   c. If no detection: use predicted state as output (covariance grows)
4. Record state at each timestep
```

### 4.2 Initialisation Details

On the first valid bearing measurement `θ₀` with range estimate `r₀`:

```
x = [cx + r₀·cos(θ₀), cy + r₀·sin(θ₀), 0, 0]

P_pos = R_rot · diag(σ_r², (r₀·σ_θ)²) · R_rot^T

where R_rot = [cos(θ₀)  -sin(θ₀)]
              [sin(θ₀)   cos(θ₀)]

P = [P_pos    0    ]
    [  0    σ_v²·I₂]
```

This places the target at the detected bearing with large radial uncertainty (`σ_r = 100 m`) and small cross-range uncertainty (`r₀·σ_θ ≈ 200 × 0.05 ≈ 10 m`). Velocity is initialised at zero with `σ_v = 20 m/s`.

### 4.3 Bearing Wrapping

The bearing innovation `ỹ[0] = z_θ - h_θ(x⁻)` must be wrapped to `(-π, π]`:

```
ỹ[0] = ((ỹ[0] + π) mod 2π) - π
```

This prevents large innovations when the bearing crosses the 0°/360° boundary.

### 4.4 Multi-Target Tracker

**Source:** `tracker.py` → `MultiTargetTracker`, `run_multi_tracker()`

The multi-target tracker manages multiple EKF instances with nearest-neighbour association:

```
For each time step:
    1. Predict all existing tracks
    2. Build cost matrix: cost[t, d] = ||x_track[t] - x_detection[d]||
    3. Greedy assignment (min cost first):
       - If cost < gate_threshold: assign detection to track, update
       - Mark both as used
    4. Unassigned tracks: increment missed counter
    5. Unassigned detections: create new track
    6. Delete tracks with missed > max_missed
    7. Confirm tracks with ≥ 2 updates
```

---

## 5. Fire Control

**Source:** `src/acoustic_sim/fire_control.py`

### 5.1 Time of Flight

**Function:** `time_of_flight(range_m, muzzle_velocity, decel)`

```
v_at_range = v₀ - α·r
if v_at_range ≤ 0: return ∞  (out of range)
v_avg = (v₀ + v_at_range) / 2
return r / v_avg
```

### 5.2 Iterative Lead Angle

**Function:** `compute_lead(target_pos, target_vel, weapon_pos, ...)`

```
tof = time_of_flight(||target - weapon||)
for i in 1..max_iterations:
    intercept = target_pos + target_vel × tof
    new_range = ||intercept - weapon||
    new_tof = time_of_flight(new_range)
    if |new_tof - tof| < 1e-6: break
    tof = new_tof

aim_bearing = atan2(intercept_y - weapon_y, intercept_x - weapon_x)
lead_angle = aim_bearing - direct_bearing
return {aim_bearing, lead_angle, intercept, tof, converged}
```

### 5.3 Engagement Envelope

**Function:** `compute_engagement(target_pos, target_vel, target_cov, weapon_pos, ...)`

```
1. Compute position uncertainty: σ_pos = 2 × √(max eigenvalue of P_pos)
2. Compute lead solution → intercept range
3. Compute pattern diameter: d = β × r_intercept
4. Compute pellet velocity at intercept
5. Compute crossing speed: v_⊥ = |v_target - (v_target · n̂_LOS) · n̂_LOS|

Decision:
    if v_pellet ≤ 0: NO FIRE (OUT_OF_RANGE)
    if r > max_range: NO FIRE (TOO_FAR)
    if σ_pos > max_uncertainty: NO FIRE (UNCERTAINTY_TOO_HIGH)
    if d_pattern < σ_pos: NO FIRE (UNCERTAINTY_TOO_HIGH)
    if r_intercept > v₀/α: NO FIRE (MAX_RANGE_EXCEEDED)
    else: FIRE
```

### 5.4 Miss Distance Computation

**Function:** `compute_miss_distance(fire_control, true_positions, true_times, ...)`

For each fire-control timestep where a shot is taken:

```
t_impact = t_fire + tof
x_true_impact = interp(t_impact, true_times, true_positions)
miss = ||intercept_predicted - x_true_impact||
hit = (miss < pattern_diameter / 2)
```

### 5.5 Threat Prioritisation

**Function:** `prioritize_threats(tracks, weapon_pos, ...)`

```
For each track:
    r = ||position - weapon||
    v_closing = velocity · (weapon - position) / |weapon - position|
    σ_pos = 2 × √(max eigenvalue of P_pos)
    score = w_r/r + w_c·max(v_closing, 0) + w_q/σ_pos

Sort tracks by score (descending) → engage in order
```

---

## 6. Noise Generation

**Source:** `src/acoustic_sim/noise.py`

### 6.1 Spatially Correlated Wind Noise

**Function:** `generate_wind_noise(mic_positions, n_samples, dt, ...)`

```
1. Build spatial correlation matrix:
   C[i,j] = exp(-||x_i - x_j|| / L) + δ_{ij}·10⁻⁸
   
2. Cholesky decomposition: C = L·L^T

3. Generate white noise: w ~ N(0, I)  shape (n_mics, n_samples)

4. Apply spatial correlation: n = L · w

5. Spectral shaping (vectorised FFT):
   spectra = FFT(n, axis=1)
   For each frequency f:
       if f ≤ f_corner: shape = 1/√(f/f_c)
       if f > f_corner: shape = (f_c/f)²
       shape[0] = 0  (remove DC)
   spectra *= shape
   n = IFFT(spectra, axis=1)

6. Scale to target RMS level:
   n *= p_target / rms(n)
```

### 6.2 Sensor Self-Noise

**Function:** `generate_sensor_noise(n_mics, n_samples, dt, ...)`

```
n = σ · randn(n_mics, n_samples)
where σ = p_ref × 10^(level_dB/20)
```

### 6.3 Fault Injection

**Function:** `inject_sensor_faults(traces, ...)`

Four fault modes:

| Mode | Effect |
|---|---|
| `elevated_noise` | Add high-level white noise: `traces[m] += σ_fault · randn(n)` |
| `dropout` | Zero the sensor: `traces[m] = 0` |
| `spikes` | Random impulses: `traces[m, t] += ±p_fault` with probability `spike_rate` |
| `dc_offset` | Add constant: `traces[m] += p_fault` |

If `fault_sensors` is not specified, `fault_fraction × n_mics` sensors are randomly selected.

### 6.4 Transient Injection

**Function:** `inject_transient(traces, dt, event_time, event_pos, mic_positions, ...)`

```
1. Generate pulse: Gaussian-windowed white noise
   raw = randn(dur_samples)
   window = exp(-0.5·((t - dur/2) / (dur/4))²)
   pulse = raw × window / max(|raw × window|)

2. For each microphone m:
   distance = ||event_pos - mic_positions[m]||
   amplitude = p_source / max(distance, 1.0)
   arrival_sample = round((event_time + distance/c) / dt)
   traces[m, arrival:arrival+dur] += amplitude × pulse
```

### 6.5 Position Perturbation

**Function:** `perturb_mic_positions(true_positions, error_std, ...)`

```
reported = true_positions + randn(n_mics, 2) × error_std
```

This simulates field placement errors where microphones are not exactly at their reported positions.

---

## 7. Validation Checks

**Source:** `src/acoustic_sim/validate.py`

### 7.1 Amplitude Check

```
peak = max(|traces|)
PASS if peak ≤ 200 Pa (≈140 dB SPL)
```

Physical reasoning: Pressures exceeding 200 Pa indicate numerical instability or an unrealistic source level.

### 7.2 SNR Check

```
signal_power = mean(filtered_traces[closest_mic]²)
noise_power = mean(filtered_traces[farthest_mic]²)
SNR_dB = 10·log₁₀(signal_power / noise_power)
PASS if SNR_dB > 0
```

The farthest microphone receives the weakest signal, so its power approximates the noise floor. The closest microphone should have positive SNR.

### 7.3 Travel-Time Check

```
For a small test grid around the array centre:
    computed_tt = compute_travel_times(grid, mics, c)
    expected_tt = distance(grid, mics) / c
    error = |computed - expected| / dt

PASS if max error < 1 sample
```

Verifies that the travel-time computation is consistent with straight-line geometry.

### 7.4 Localisation Check

```
Run MFP on traces with known source position
Compare detected bearing to true bearing
PASS if bearing error < 10°
```

This is a self-consistency check: if the system can't localise a known source in ideal conditions, something is fundamentally wrong.

### 7.5 Energy Conservation Check

```
total_received = Σ_{m,t} traces[m,t]² × dt
expected = Σ_m (p_source / r_m)² × duration
ratio = total_received / expected
PASS if 0.01 < ratio < 100
```

The received energy should be within two orders of magnitude of the geometric spreading prediction. Exact agreement is not expected (2D vs 3D spreading, absorption, windowing effects), but wild disagreement indicates a bug.

---

*Next: [Code Architecture & Module Reference](architecture.md) — How these algorithms are organised into code modules.*

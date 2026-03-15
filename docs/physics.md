# Physics Background

This document presents the physical principles underlying every algorithm in `acoustic-sim`. Equations are given in standard mathematical notation. Where implementations make approximations or modelling choices, these are explicitly noted and justified.

---

## Table of Contents

1. [The Scalar Wave Equation in Two Dimensions](#1-the-scalar-wave-equation-in-two-dimensions)
2. [The Helmholtz Equation](#2-the-helmholtz-equation)
3. [Geometric Spreading and Attenuation](#3-geometric-spreading-and-attenuation)
4. [Sound Pressure Level](#4-sound-pressure-level)
5. [Wind Effects on Acoustic Propagation](#5-wind-effects-on-acoustic-propagation)
6. [Impedance Contrast and Reflections](#6-impedance-contrast-and-reflections)
7. [Drone Acoustic Signatures](#7-drone-acoustic-signatures)
8. [Noise in Acoustic Sensor Systems](#8-noise-in-acoustic-sensor-systems)
9. [Array Acoustics and Spatial Sampling](#9-array-acoustics-and-spatial-sampling)
10. [Beamforming Theory](#10-beamforming-theory)
11. [Matched Field Processing on a Polar Grid](#11-matched-field-processing-on-a-polar-grid)
12. [Extended Kalman Filter Theory](#12-extended-kalman-filter-theory)
13. [Shotgun Ballistics and Fire Control](#13-shotgun-ballistics-and-fire-control)
14. [3D Wave Equation and CFL Condition](#14-3d-wave-equation-and-cfl-condition)
15. [Ground Reflection and the Image-Source Method](#15-ground-reflection-and-the-image-source-method)
16. [Acoustic Feature Extraction — Mel Spectrograms](#16-acoustic-feature-extraction--mel-spectrograms)
17. [Kinematic Feature Physics](#17-kinematic-feature-physics)
18. [Classification Theory](#18-classification-theory)

---

## 1. The Scalar Wave Equation in Two Dimensions

### 1.1 Derivation Context

The acoustic wave equation governs the propagation of small-amplitude pressure perturbations through a fluid medium. It arises from the linearised conservation equations of fluid mechanics:

- **Conservation of mass** (continuity equation): relates density perturbations to velocity divergence.
- **Conservation of momentum** (Euler equation): relates pressure gradients to acceleration.
- **Equation of state**: relates pressure to density via the squared sound speed, `c²`.

Combining these and eliminating velocity yields the scalar wave equation for the pressure perturbation `p(x, y, t)`:

```
∂²p/∂t² = c²(x, y) · (∂²p/∂x² + ∂²p/∂y²) + S(x, y, t)
```

where:

- `p(x, y, t)` is the acoustic pressure perturbation [Pa]
- `c(x, y)` is the local sound speed [m/s], which defines the **velocity model**
- `S(x, y, t)` is a source term (injected pressure)

### 1.2 Key Assumptions

The derivation requires several assumptions, all of which are standard for linear acoustics:

1. **Small perturbations**: Pressure fluctuations are small compared to the ambient pressure (~101 kPa). For a 140 dB source (200 Pa peak), this ratio is ~0.2%, well within the linear regime.
2. **Inviscid fluid**: Viscous dissipation is neglected. This is valid for air at the frequencies of interest (< 1 kHz) over the propagation distances considered (< 1 km).
3. **Irrotational flow**: The acoustic velocity field is curl-free, allowing a scalar potential description.
4. **Quiescent background**: The ambient medium is at rest (wind effects are treated separately — see [Section 5](#5-wind-effects-on-acoustic-propagation)).
5. **Two-dimensional domain**: The simulation operates in 2D, representing either a horizontal slice through a 3D environment or a 2D proxy. Source injection and amplitude calibration use 3D spreading conventions (1/r) for physical realism.

### 1.3 The Velocity Model

The velocity model `c(x, y)` encodes the material properties of the propagation medium. In `acoustic-sim`, the velocity model is stored as a 2D array on a regular Cartesian grid with spacing `dx` and `dy` (typically equal). The `VelocityModel` dataclass holds:

- `x`, `y`: 1D arrays of cell-centre coordinates
- `values`: 2D array of shape `[ny, nx]` containing `c(x, y)` in m/s
- `dx`, `dy`: grid spacing

Several preset velocity models are provided:

| Model | Sound Speed Structure | Physical Analogue |
|---|---|---|
| **Uniform** | Constant `c` everywhere (default 343 m/s) | Open air, isotropic propagation |
| **Layered** | Horizontal layers, each with a distinct `c` | Temperature stratification |
| **Gradient** | Linear vertical variation from `v_bottom` to `v_top` | Continuous temperature gradient |
| **Checkerboard** | Alternating high/low cells | Resolution test pattern |
| **Valley** | Air (343 m/s) between two dirt ridges (1500 m/s) | Terrain with strong impedance contrasts |

Anomalies (circular or rectangular regions with altered velocity) can be added to any model to create scattering targets.

### 1.4 Boundary Conditions

Physical boundaries in open-air acoustics require absorbing conditions to prevent artificial reflections from domain edges. The code implements two mechanisms:

1. **Sponge layer (FDTD)**: A damping zone of width `damping_width` cells at each edge applies a damping coefficient `σ(x, y)` that ramps quadratically from zero at the inner edge to `damping_max` at the domain boundary. This dissipates outgoing waves. See [Algorithms — FDTD Solver](algorithms.md#2-fdtd-solver).

2. **Complex wavenumber (Helmholtz)**: The absorbing layer adds an imaginary component to the wavenumber: `k² → k²(1 + iσ)`, which causes exponential decay of the field within the sponge zone.

Impedance-contrast boundaries (e.g., air–rock interfaces in the valley model) produce reflections naturally through the velocity model — no special treatment is needed. The FDTD solver resolves these reflections automatically.

---

## 2. The Helmholtz Equation

### 2.1 Frequency-Domain Reduction

For a time-harmonic source at angular frequency `ω = 2πf`, we assume a steady-state solution of the form `p(x, y, t) = Re[P(x, y) · e^{-iωt}]`. Substituting into the wave equation and cancelling the time factor yields the **Helmholtz equation**:

```
∇²P(x, y) + k²(x, y) · P(x, y) = -S(x, y)
```

where:

- `P(x, y)` is the complex pressure amplitude
- `k(x, y) = ω / c(x, y)` is the spatially varying wavenumber [rad/m]
- `S(x, y)` is the source amplitude

The Helmholtz equation is an elliptic PDE — it describes the spatial pattern of a monochromatic wavefield without reference to time. It must be solved as a boundary-value problem (all grid points are coupled).

### 2.2 Absorbing Boundary Implementation

To suppress reflections from domain edges, the wavenumber within the sponge layer is made complex:

```
k²_eff = k² · (1 + i·σ(x, y))
```

where `σ(x, y) ≥ 0` is the damping coefficient, which ramps quadratically from zero at the sponge inner edge to a maximum of 0.7 at the domain boundary. This introduces exponential decay for waves propagating into the sponge.

### 2.3 Physical Interpretation

The Helmholtz solution gives the steady-state pressure amplitude at every grid point for a single frequency. It is useful for:

- Visualising wavefield patterns (diffraction, interference) at a specific frequency
- Verifying grid resolution (spatial aliasing manifests as visible artefacts)
- Quick single-frequency analysis without time-stepping

The output is `|P(x, y)|` — the pressure magnitude — displayed as a 2D image.

---

## 3. Geometric Spreading and Attenuation

### 3.1 Spreading Laws

As an acoustic wave propagates outward from a point source, its amplitude decreases due to the expansion of the wavefront:

- **3D (spherical spreading)**: `p(r) = p₀ / r` — pressure decays as the inverse of distance. Energy (proportional to `p²`) decays as `1/r²`, consistent with conservation over a sphere of area `4πr²`.

- **2D (cylindrical spreading)**: `p(r) = p₀ / √r` — pressure decays as the inverse square root of distance. Energy decays as `1/r`, consistent with a cylinder of circumference `2πr`.

The FDTD solver operates in 2D, so waves naturally exhibit cylindrical spreading (`1/√r`). However, source amplitudes are calibrated to physical 3D levels (dB SPL at 1 m), and the source injection uses point-source amplitudes consistent with 3D spreading. This is a standard modelling choice that maintains physical pressure levels while leveraging the computational efficiency of 2D simulation.

### 3.2 Atmospheric Absorption

In addition to geometric spreading, real atmospheric propagation involves frequency-dependent molecular absorption. The code models this as a small uniform damping coefficient `air_absorption` (default 0.005) applied everywhere in the FDTD domain. This provides a gentle, continuous decay that prevents numerical energy buildup without significantly affecting the physics at the scales of interest (< 1 km).

### 3.3 Vegetation Damping

In the hills+vegetation domain, a thin layer above each ridge surface is assigned an elevated attenuation coefficient (`veg_attenuation`, default 0.15). This models the scattering and absorption of sound waves by foliage — an effect that is significant in forested terrain and can degrade detection performance.

---

## 4. Sound Pressure Level

### 4.1 The dB SPL Scale

Acoustic pressures span many orders of magnitude. The decibel Sound Pressure Level (SPL) scale compresses this range into a manageable form:

```
SPL = 20 · log₁₀(p / p_ref)    [dB re 20 µPa]
```

where `p_ref = 20 × 10⁻⁶ Pa` is the reference pressure, corresponding to the nominal threshold of human hearing at 1 kHz.

| SPL [dB] | Pressure [Pa] | Physical Example |
|---|---|---|
| 0 | 2 × 10⁻⁵ | Threshold of hearing |
| 40 | 2 × 10⁻³ | Quiet library |
| 60 | 2 × 10⁻² | Normal conversation |
| 90 | 0.632 | Lawn mower at 1 m; **drone at 1 m (default)** |
| 120 | 20 | Threshold of pain |
| 140 | 200 | Jet engine at 1 m; **amplitude check ceiling** |

### 4.2 Source Level and Received Level

A drone source at 90 dB SPL at 1 m has a reference pressure of:

```
p_source = p_ref × 10^(90/20) = 20 × 10⁻⁶ × 10^4.5 ≈ 0.632 Pa
```

At range `r` (metres), assuming 3D spherical spreading:

```
p(r) = p_source / r
SPL(r) = SPL_source - 20·log₁₀(r)
```

For the default drone at 100 m: `SPL(100) = 90 - 40 = 50 dB`. At 200 m: `SPL(200) = 90 - 46 ≈ 44 dB`.

### 4.3 Noise Floor and SNR

The signal-to-noise ratio (SNR) at a microphone determines detectability:

```
SNR = 10 · log₁₀(P_signal / P_noise)    [dB]
```

where `P_signal` and `P_noise` are mean squared pressures. For the default scenario:

- Drone signal at 100 m: ~50 dB SPL
- Wind noise: 55 dB SPL (broadband)
- Sensor self-noise: 40 dB SPL

The drone is below the broadband wind noise — detection relies on the **narrowband** harmonic structure of the drone signal, which concentrates energy at specific frequencies where SNR can be positive even when broadband SNR is negative. This is exactly the role of the matched field processor.

---

## 5. Wind Effects on Acoustic Propagation

### 5.1 Effective Sound Speed

Wind modifies the speed at which sound waves propagate. For a plane wave travelling in direction `n̂`, the effective propagation speed is:

```
c_eff = c + v⃗_wind · n̂
```

where `v⃗_wind = (v_x, v_y)` is the wind velocity vector. Waves propagating downwind travel faster; waves propagating upwind travel slower.

### 5.2 Implementation in FDTD

The wind domain in `acoustic-sim` stores wind velocity components `(wind_vx, wind_vy)` as metadata (`DomainMeta`). The FDTD solver accounts for wind through the CFL stability condition:

```
dt ≤ cfl_safety × 2·dx / ((c_max + |v⃗_wind|) × √(2·ρ_stencil))
```

where `|v⃗_wind|` is the wind speed magnitude and `ρ_stencil` is the spectral radius of the FD stencil. This ensures stability even when wind adds to the maximum propagation speed.

The wind direction is specified meteorologically (direction the wind is **coming from**, clockwise from +y), then converted to Cartesian components:

```
v_x = v_wind · sin(θ)
v_y = v_wind · cos(θ)
```

### 5.3 Physical Effects

Wind creates several observable effects in acoustic propagation:

- **Asymmetric spreading**: Sound travels farther downwind than upwind.
- **Refraction**: Vertical wind gradients bend sound rays (not modelled in this 2D code, but the effective-velocity approximation captures the first-order effect).
- **Turbulent decorrelation**: Wind generates turbulent pressure fluctuations that act as noise (modelled as spatially correlated wind noise in the post-hoc noise generator).

---

## 6. Impedance Contrast and Reflections

### 6.1 Acoustic Impedance

The acoustic impedance of a medium is `Z = ρ·c`, where `ρ` is density and `c` is sound speed. At an interface between two media with impedances `Z₁` and `Z₂`, the pressure reflection coefficient is:

```
R = (Z₂ - Z₁) / (Z₂ + Z₁)
```

For air (`c ≈ 343 m/s`, `ρ ≈ 1.2 kg/m³`, `Z ≈ 412 Pa·s/m`) against rock (`c ≈ 1500 m/s`, `ρ ≈ 2500 kg/m³`, `Z ≈ 3.75 × 10⁶ Pa·s/m`):

```
R ≈ (3.75×10⁶ - 412) / (3.75×10⁶ + 412) ≈ 0.9998
```

This is nearly total reflection — almost all incident energy bounces back. In the `acoustic-sim` velocity model, this extreme impedance contrast is represented by assigning different sound speeds to different materials:

| Material | Velocity [m/s] | Domain Type |
|---|---|---|
| Air | 343 | All domains |
| Dirt/rock | 1500 | Valley, hills+vegetation |
| Canyon walls | 2000 | Echo canyon |
| Buildings | 2500 | Urban echo |

### 6.2 Multipath and Echoes

In the FDTD solver, reflections arise naturally from impedance contrasts in the velocity model — no special boundary condition code is needed. When a wave encounters a velocity discontinuity, part of the energy reflects and part transmits, governed by the local impedance ratio.

This produces multipath propagation: the receiver sees both the **direct arrival** (shortest path) and one or more **reflected arrivals** (longer paths via walls, buildings, or terrain). Multipath is a key challenge for acoustic localisation:

- **Echo canyon domain**: Two parallel high-impedance walls create systematic reflections. The direct arrival dominates, so MFP localisation remains accurate.
- **Urban echo domain**: Multiple buildings at different positions create complex multipath. This degrades localisation accuracy from 5.2 m (isotropic) to 8.1 m (urban), as demonstrated in Study 7.

---

## 7. Drone Acoustic Signatures

### 7.1 Blade Pass Frequency and Harmonics

The dominant acoustic signature of a multi-rotor drone is a harmonic series generated by the rotating blades. The fundamental frequency is the **Blade Pass Frequency** (BPF):

```
BPF = n_blades × RPM / 60    [Hz]
```

For the default configuration: `BPF = n_blades × RPM / 60`. With a fundamental of 150 Hz (the default `fundamental_freq`), the harmonic series consists of:

```
f_k = k × f_fundamental,    k = 1, 2, 3, ..., n_harmonics
```

With `n_harmonics = 6` and `f_fundamental = 150 Hz`:
- `f₁ = 150 Hz` (fundamental)
- `f₂ = 300 Hz`
- `f₃ = 450 Hz`
- `f₄ = 600 Hz`
- `f₅ = 750 Hz`
- `f₆ = 900 Hz`

Each harmonic has a decreasing relative amplitude, controlled by `harmonic_amplitudes` (default: `[1.0, 0.6, 0.35, 0.2, 0.12, 0.08]`). The combined signal is:

```
s(t) = Σ_k  a_k · sin(2π·f_k·t)
```

scaled so the peak amplitude equals the physical source pressure `p_source = p_ref × 10^(SPL/20)`.

### 7.2 Propeller Noise Model (Legacy)

The propeller signal generator (`make_source_propeller`) provides a more detailed rotor model:

- BPF harmonics with `1/√h` amplitude decay (where `h` is harmonic number)
- Amplitude modulation at the rotor frequency: `1 + m·sin(2π·f_rotor·t)` where `m` is modulation depth (default 0.25)
- Broadband turbulent noise: bandpass-filtered Gaussian noise (100 Hz – 8 kHz) at configurable level (default 12% of tonal)

### 7.3 Doppler Shift

When the source moves, the FDTD solver naturally produces Doppler shift because the source position changes at each timestep. The emitted wavefield is compressed ahead of the source and stretched behind it. For a source moving at speed `v_s`:

```
f_received = f_emitted × c / (c - v_s·cos(θ))
```

where `θ` is the angle between the source velocity and the source-to-receiver direction. This is an emergent effect — no explicit Doppler correction is applied in the code.

---

## 8. Noise in Acoustic Sensor Systems

The detection pipeline must distinguish the drone signal from several noise sources. Because the wave equation is linear, noise can be added to the FDTD traces post-hoc via superposition — the result is identical to having the noise present during propagation.

### 8.1 Wind Noise

Wind turbulence generates pressure fluctuations at each microphone. These fluctuations are:

- **Spatially correlated**: Nearby microphones experience similar turbulent eddies. The spatial correlation function is modelled as:

  ```
  C(r) = exp(-r / L)
  ```

  where `r` is the inter-microphone distance and `L` is the correlation length (default 3.0 m). The correlation matrix `C` is decomposed via Cholesky factorisation: `C = LL^T`. Multiplying white noise by `L` produces correlated noise.

- **Spectrally shaped**: Wind noise has a characteristic 1/f spectrum below a corner frequency (default 15 Hz), with steep rolloff above. This is implemented by FFT → multiply by spectral shaping function → IFFT.

The spectral shaping function is:

```
H(f) = 1/√(f/f_c)     for f ≤ f_c   (corner frequency)
H(f) = (f_c/f)²        for f > f_c   (4th-order rolloff)
H(0) = 0               (remove DC)
```

Wind noise is calibrated to a target RMS level in dB SPL (default 55 dB).

### 8.2 Sensor Self-Noise

Each microphone has an electronic noise floor modelled as uncorrelated white Gaussian noise at a specified RMS level (default 40 dB SPL). This is simply:

```
n_m(t) = σ · w_m(t)
```

where `w_m(t)` is standard Gaussian white noise and `σ = p_ref × 10^(SPL/20)`.

### 8.3 Stationary Coherent Interferers

A stationary noise source (e.g., a generator or HVAC unit) at a fixed position emits a tonal signal with harmonics, similar to the drone but with a different fundamental frequency (default 60 Hz) and position. This is modelled by running a second FDTD simulation with a `StaticSource` at the interferer position and adding the resulting traces to the drone traces.

The matched field processor handles this through **stationary source rejection**: grid points that exhibit low temporal variability (coefficient of variation below threshold) across multiple time windows are masked out.

### 8.4 Impulsive Transients

Short, high-energy events (explosions, gunshots) produce broadband transients that can overwhelm the beamformer. These are modelled as:

```
pulse(t) = w(t) · noise(t)
```

where `w(t)` is a Gaussian window of duration `τ` (default 10 ms) and `noise(t)` is white noise. The pulse propagates to each microphone with:

- Travel-time delay: `Δt_m = |x_event - x_m| / c`
- Amplitude decay: `p_m = p_source / max(r_m, 1 m)`

The MFP's **transient blanking** algorithm detects and zeros out affected sub-windows.

---

## 9. Array Acoustics and Spatial Sampling

### 9.1 Microphone Array Fundamentals

A microphone array consists of `M` sensors at known positions `{x_m}`. The array's ability to resolve the direction of arrival (DOA) of a wavefront depends on its geometry:

- **Aperture** (`D`): The maximum distance between any two sensors. Larger aperture → better angular resolution.
- **Element spacing** (`d_min`): The minimum distance between any two sensors. Smaller spacing → higher alias-free frequency.

### 9.2 Spatial Aliasing

Just as temporal sampling must satisfy the Nyquist criterion (`f_sample ≥ 2·f_max`), spatial sampling must satisfy a spatial Nyquist criterion. For a pair of sensors separated by distance `d`, the maximum frequency that can be unambiguously resolved is:

```
f_alias = c / (2·d)
```

Above this frequency, the phase difference between sensors exceeds ±π, causing directional ambiguity. For the default array (0.5 m radius, 16 elements, minimum baseline ~0.2 m):

```
f_alias = 343 / (2 × 0.2) = 858 Hz
```

The drone harmonics at 150–900 Hz are mostly below this limit, though the highest harmonic (900 Hz) is marginal.

### 9.3 Angular Resolution

The diffraction-limited angular resolution of an array with aperture `D` at wavelength `λ` is approximately:

```
Δθ ≈ λ / D    [radians]
```

For the default array (`D = 1.0 m` diameter) at 600 Hz (`λ = 0.57 m`):

```
Δθ ≈ 0.57 / 1.0 = 0.57 rad ≈ 33°
```

This is quite coarse, which is why the broadband MFP sums across all harmonics — higher harmonics (shorter wavelengths) provide better resolution, and the frequency weighting `w(f) = (f/f_max)²` emphasises them.

### 9.4 Numerical Grid Sampling

The FDTD grid must resolve the shortest wavelength present in the simulation. The **points-per-wavelength** (PPW) criterion requires:

```
PPW = λ_min / dx ≥ 10
```

where `λ_min = c_min / f_max`. With 10 points per wavelength, numerical dispersion (the tendency of discrete waves to travel at slightly wrong speeds) is kept below ~1%.

For the default detection scenario (`dx = 0.05 m`, `c_min = 343 m/s`):

```
f_max = c_min / (10 × dx) = 343 / (10 × 0.05) = 686 Hz
```

All four drone harmonics at 150/300/450/600 Hz are below this limit. The grid cannot resolve the 750 Hz and 900 Hz harmonics, so `make_drone_harmonics` automatically skips any harmonic above `f_max`.

### 9.5 The CFL Stability Condition

For the explicit leapfrog FDTD scheme on a 2D square grid, the Courant–Friedrichs–Lewy (CFL) condition ensures numerical stability:

```
C = c_max · dt / dx ≤ 1/√2 ≈ 0.707
```

where `C` is the Courant number. For higher-order FD stencils, the CFL limit depends on the stencil's spectral radius (see [Algorithms — FDTD Solver](algorithms.md#2-fdtd-solver) for the generalised formula).

The code auto-computes `dt` with a 0.9× safety margin:

```
dt = 0.9 × 2·dx / ((c_max + |v_wind|) × √(2·ρ_stencil))
```

---

## 10. Beamforming Theory

### 10.1 The Measurement Model

Consider `M` microphones at positions `{x_m}` observing a far-field plane wave from direction `θ` at frequency `f`. The signal at microphone `m` is:

```
p_m(f) = s(f) · exp(-j·2π·f·τ_m(θ)) + n_m(f)
```

where `s(f)` is the source spectrum, `τ_m(θ)` is the propagation delay from the source to microphone `m`, and `n_m(f)` is noise.

The **steering vector** for a candidate source position is:

```
a_m(f, θ) = exp(-j·2π·f·τ_m(θ))
```

and the full steering vector is `a(f, θ) = [a_1, a_2, ..., a_M]^T`.

### 10.2 Cross-Spectral Density Matrix

The CSDM at frequency `f` is the `M × M` matrix:

```
C(f) = E[d(f) · d^H(f)]
```

where `d(f) = [d_1(f), ..., d_M(f)]^T` are the Fourier-transformed microphone signals and `^H` denotes conjugate transpose. In practice, the CSDM is estimated by averaging over multiple sub-windows (Welch's method):

```
Ĉ(f) = (1/K) · Σ_{k=1}^{K}  d_k(f) · d_k^H(f)
```

where `K` is the number of sub-windows (default 4). Each sub-window is Hann-tapered before FFT to reduce spectral leakage.

### 10.3 Conventional (Delay-and-Sum) Beamforming

The conventional beamformer computes the beam power as:

```
P_conv(θ) = a^H(f, θ) · C(f) · a(f, θ)
```

This is equivalent to delaying each sensor's signal by `τ_m(θ)` and summing. When the steering direction matches the true source, the signals add coherently and `P` is maximised. Conventional beamforming is robust but has limited resolution — sidelobes from one source can mask another.

### 10.4 MVDR (Capon) Beamforming

The Minimum Variance Distortionless Response (MVDR) beamformer minimises total output power while maintaining unit gain in the look direction:

```
P_MVDR(θ) = 1 / [a^H(f, θ) · C⁻¹(f) · a(f, θ)]
```

where `C⁻¹` is the inverse of the CSDM. MVDR achieves **super-resolution** by placing nulls in the directions of interfering sources. It adapts to the noise environment through the CSDM inverse.

**Diagonal loading**: In practice, `C` can be ill-conditioned (especially with few sub-windows or correlated noise). A small fraction `ε` of the trace is added to the diagonal:

```
C_loaded = C + ε · trace(C)/M · I_M
```

The default `ε = 0.01` (1% of mean eigenvalue). If the condition number still exceeds 10⁶, the code falls back to conventional beamforming.

### 10.5 Broadband Processing

Drone signals have energy at multiple harmonics. To combine information across frequencies, the beam power at each frequency is weighted and summed:

```
P_broadband(θ) = Σ_f  w(f) · P(f, θ)
```

The weighting function is `w(f) = (f/f_max)²`, which emphasises higher harmonics. This is physically motivated: the array's angular resolution improves with frequency (`Δθ ∝ λ/D`), so higher frequencies provide more discriminating power. Squaring the weight accentuates this effect.

---

## 11. Matched Field Processing on a Polar Grid

### 11.1 Why a Polar Grid?

For a compact array (aperture `D ≈ 1 m`) observing a source at range `R > 20 m`, the array geometry provides:

- **Bearing**: Well-resolved. The angular resolution `Δθ ≈ λ/D` gives ~33° at 600 Hz, further sharpened by MVDR and broadband summation.
- **Range**: Poorly resolved. Range information comes from wavefront curvature, which is negligible when `R >> D`. A 1-metre array cannot distinguish a source at 100 m from one at 200 m from a single snapshot.

A polar grid `(azimuth × range)` is therefore natural: azimuth is the primary observable, and range acts as a nuisance parameter that is marginally constrained by amplitude, bearing rate, and trajectory geometry.

The polar grid is defined by:

- Azimuths: `0°` to `360°` in steps of `azimuth_spacing_deg` (default 1°)
- Ranges: `range_min` to `range_max` in steps of `range_spacing` (defaults: 20 m to 500 m in 5 m steps)

### 11.2 Steering Vectors on the Polar Grid

Each grid point `(θ_i, r_j)` is converted to Cartesian coordinates relative to the array centre `(cx, cy)`:

```
x_{ij} = cx + r_j · cos(θ_i)
y_{ij} = cy + r_j · sin(θ_i)
```

The travel time from grid point `(x_{ij}, y_{ij})` to microphone `m` at `(x_m, y_m)` is:

```
τ_m(i, j) = √[(x_{ij} - x_m)² + (y_{ij} - y_m)²] / c
```

The steering vector at frequency `f` is:

```
a_m(f, i, j) = exp(-j·2π·f·τ_m(i, j))
```

### 11.3 Harmonic Selection

Rather than processing the entire spectrum, only frequency bins near the expected drone harmonics are selected. For each harmonic `f_k = k × f_fundamental`:

```
selected_bins = { b : |f_b - f_k| ≤ Δf }
```

where `Δf` is the harmonic bandwidth (default 10 Hz half-width). This focuses computational effort on the signal-bearing frequencies and reduces noise contamination from frequency bands where only noise is present.

### 11.4 Sub-Grid Interpolation

After finding the peak grid cell in the beam-power map, parabolic interpolation refines the position. For a 1D peak at index `i` with neighbouring values `y_{i-1}`, `y_i`, `y_{i+1}`:

```
δ = (y_{i-1} - y_{i+1}) / [2 · (2·y_i - y_{i-1} - y_{i+1})]
```

The refined position is `θ_refined = θ_i + δ · Δθ`. This is applied independently in azimuth (with circular wrapping) and range.

---

## 12. Extended Kalman Filter Theory

### 12.1 State Space Formulation

The tracker models the drone as a point target with state:

```
x = [x, y, v_x, v_y]^T
```

where `(x, y)` is position in metres and `(v_x, v_y)` is velocity in m/s.

### 12.2 Motion Model

A constant-velocity motion model is used:

```
x_{k+1} = F · x_k + w_k
```

where the state transition matrix is:

```
F = | 1  0  Δt  0  |
    | 0  1  0   Δt |
    | 0  0  1   0  |
    | 0  0  0   1  |
```

and the process noise `w_k ~ N(0, Q)` accounts for unmeasured accelerations (manoeuvres). The process noise covariance for continuous-time white-noise acceleration with spectral density `q = σ_a²` is:

```
Q = q × | Δt⁴/4    0       Δt³/2    0     |
        |   0     Δt⁴/4     0      Δt³/2  |
        | Δt³/2    0        Δt²      0     |
        |   0     Δt³/2     0       Δt²   |
```

where `σ_a` is the acceleration noise standard deviation (default 2 m/s²).

### 12.3 Measurement Model

The MFP produces three observables at each time window:

1. **Bearing**: `θ = atan2(y - c_y, x - c_x)` — well-resolved
2. **Range**: `r = √[(x - c_x)² + (y - c_y)²]` — poorly resolved
3. **Amplitude**: `A = A_source / r` — noisy proxy for range via 1/r decay

The measurement function is:

```
h(x) = | atan2(y - c_y, x - c_x) |
       | √[(x-c_x)² + (y-c_y)²]  |
       | A_source / r              |
```

This is nonlinear (atan2, sqrt, 1/r), which is why an **Extended** Kalman Filter is used rather than a standard linear KF.

### 12.4 EKF Linearisation

The EKF linearises the measurement model via the Jacobian `H = ∂h/∂x`:

```
        | ∂θ/∂x    ∂θ/∂y    0  0 |       | -dy/r²    dx/r²    0  0 |
H   =   | ∂r/∂x    ∂r/∂y    0  0 |   =   |  dx/r     dy/r     0  0 |
        | ∂A/∂x    ∂A/∂y    0  0 |       | -A·dx/r³  -A·dy/r³ 0  0 |
```

where `dx = x - c_x`, `dy = y - c_y`, `r = √(dx² + dy²)`, and `A = A_source`.

### 12.5 Prediction Step

```
x̂⁻ = F · x̂⁺
P⁻  = F · P⁺ · F^T + Q
```

### 12.6 Update Step

```
ỹ = z - h(x̂⁻)           (innovation, with bearing wrapped to (-π, π])
S = H · P⁻ · H^T + R     (innovation covariance)
K = P⁻ · H^T · S⁻¹       (Kalman gain)
x̂⁺ = x̂⁻ + K · ỹ
P⁺ = (I - K·H) · P⁻
```

The measurement noise covariance is:

```
R = diag(σ_θ², σ_r², σ_A²)
```

where:
- `σ_θ` = 3° ≈ 0.0524 rad (bearing noise)
- `σ_r` = 100 m (range noise — intentionally large)
- `σ_A` = 50% of measured amplitude (adaptive)

### 12.7 Initialisation

On the first detection, the tracker places the initial state at the detected bearing and an estimated range (default 200 m). The initial covariance is **anisotropic**: tight in the cross-range direction (σ = r·σ_θ), loose in the radial direction (σ = σ_r). This reflects the measurement geometry — we know the direction well but not the distance.

The covariance is rotated from bearing/range coordinates into Cartesian x/y:

```
R_rot = | cos(θ)  -sin(θ) |
        | sin(θ)   cos(θ) |

P_pos = R_rot · diag(σ_r², σ_cross²) · R_rot^T
```

### 12.8 Range Observability

Range is the most challenging observable. It accrues information from:

1. **Bearing rate**: A source at range `r` moving at speed `v_⊥` (perpendicular to line-of-sight) produces a bearing rate `dθ/dt = v_⊥/r`. A nearby source sweeps bearing faster — the tracker uses this to constrain range.

2. **Amplitude decay**: `A ∝ 1/r` provides a noisy range proxy. The 50% noise assumption makes this a weak but useful constraint.

3. **Trajectory geometry**: As the source moves, successive bearing/amplitude measurements triangulate the position. Perpendicular crossings provide the most range information.

4. **Time integration**: The Kalman filter integrates all these cues over time, gradually collapsing the initially large range uncertainty.

### 12.9 Multi-Target Tracking

The `MultiTargetTracker` manages multiple independent EKF tracks with nearest-neighbour data association:

1. **Prediction**: All existing tracks are predicted to the current time.
2. **Association**: Each new detection is assigned to the nearest existing track within a gate distance (default 30 m). Unassigned detections create new tracks; unassigned tracks increment a missed-detection counter.
3. **Update**: Associated tracks receive a measurement update.
4. **Pruning**: Tracks with more than `max_missed` consecutive misses (default 5) are deleted.
5. **Confirmation**: A track is confirmed after receiving at least 2 updates.

---

## 13. Shotgun Ballistics and Fire Control

### 13.1 Pellet Kinematics

The fire-control module models a 12-gauge shotgun engagement. Pellet velocity decreases with range due to aerodynamic drag:

```
v(r) = v₀ - α · r
```

where:
- `v₀` = 400 m/s (muzzle velocity)
- `α` = 1.5 m/s per metre of travel (deceleration coefficient)
- Pellet reaches zero velocity at `r_max = v₀/α ≈ 267 m`

This linear deceleration model is a simplification of the full ballistic equations but captures the essential physics over engagement ranges (< 100 m).

### 13.2 Time of Flight

For a target at range `r`, the average pellet velocity is:

```
v_avg = (v₀ + v(r)) / 2 = (v₀ + v₀ - α·r) / 2 = v₀ - α·r/2
```

The time of flight is:

```
t_flight = r / v_avg = r / (v₀ - α·r/2)
```

If `v(r) ≤ 0` (target beyond maximum range), `t_flight = ∞` and engagement is not possible.

### 13.3 Shot Pattern Spread

The pellet pattern expands linearly with range:

```
d_pattern = β · r
```

where `β` is the pattern spread rate (default 0.025 m/m = 1 m diameter per 40 m range). At 50 m, the pattern is 1.25 m wide; at 100 m, it is 2.5 m wide.

### 13.4 Lead Angle Computation

For a moving target, the weapon must aim ahead of the current position to compensate for target motion during the pellet's time of flight. The **lead angle** is computed iteratively:

1. Estimate TOF to current target position: `t₁ = r_current / v_avg`
2. Predict target position at impact: `x_intercept = x_target + v_target · t₁`
3. Recompute TOF to the intercept point: `t₂ = |x_intercept - x_weapon| / v_avg`
4. Repeat steps 2–3 until convergence (typically 3–5 iterations, `|t_{n+1} - t_n| < 10⁻⁶ s`)

The converged intercept point defines the aim direction. The lead angle is the difference between the aim direction and the direct bearing to the current target position.

### 13.5 Engagement Envelope

The engagement envelope determines whether a shot should be taken. The decision requires the shot pattern at the intercept range to be larger than the position uncertainty:

```
d_pattern(r_intercept) ≥ 2σ_pos
```

where `σ_pos` is the maximum eigenvalue of the 2D position covariance from the tracker, scaled to 2-sigma (95% confidence). Additional constraints:

- Pellet must reach the target: `v(r_intercept) > 0`
- Maximum engagement range check (if configured)
- Maximum position uncertainty check (if configured)

If any condition fails, `can_fire = False` and a human-readable reason is provided.

### 13.6 Miss Distance

The true metric of fire-control effectiveness is the **miss distance**: the separation between the pellet pattern centre at the moment of impact and the target's actual position at that same moment.

```
miss = |x_intercept(t_fire) - x_true(t_fire + t_flight)|
```

A "hit" is defined as `miss < d_pattern(r_intercept) / 2`.

### 13.7 Threat Prioritisation

When multiple targets are tracked, the system assigns priority scores:

```
score = w_range/r + w_closing · max(v_closing, 0) + w_quality / σ_pos
```

where:
- `r` is range to the weapon
- `v_closing` is the closing speed (positive = approaching)
- `σ_pos` is position uncertainty
- Default weights: `w_range = 1.0`, `w_closing = 2.0`, `w_quality = 0.5`

Closing speed is weighted highest (×2) because an approaching target is the most imminent threat. Targets are engaged in priority order.

---

## 14. 3D Wave Equation and CFL Condition

### 14.1 Extension to Three Dimensions

The 3D scalar wave equation extends the 2D form ([Section 1](#1-the-scalar-wave-equation-in-two-dimensions)) with an additional spatial derivative:

```
∂²p/∂t² = c²(x, y, z) · (∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z²) + S(x, y, z, t)
```

All assumptions from the 2D case carry over: small perturbations, inviscid fluid, irrotational flow. The key physical difference is that the simulation now captures the full three-dimensional propagation geometry, including vertical wavefront curvature, elevation-dependent travel times, and correct spherical spreading.

### 14.2 Spherical Spreading in 3D

In 3D, a point source produces spherical wavefronts. Pressure decays as:

```
p(r) = p₀ / r
```

This is the physically correct spreading law for a point source in free space. The 2D simulation ([Section 3](#3-geometric-spreading-and-attenuation)) approximates this with cylindrical spreading (`1/√r`) and compensates by calibrating source amplitudes. The 3D simulation produces correct `1/r` decay naturally.

### 14.3 3D CFL Stability Condition

For the explicit leapfrog FDTD scheme on a 3D uniform cubic grid with spacing `dx`, the CFL condition becomes:

```
dt ≤ 2·dx / ((c_max + |v⃗_wind|) · √(3·ρ_stencil))
```

The factor `√3` (instead of `√2` for 2D) arises because a plane wave travelling along the body diagonal of a cube sees all three spatial derivatives simultaneously. For the standard second-order stencil (`ρ_stencil = 4`):

```
dt ≤ 2·dx / (c_eff · √12) = dx / (c_eff · √3) ≈ 0.577·dx / c_eff
```

Compare to the 2D limit of `dx / (c_eff · √2) ≈ 0.707·dx / c_eff`. The 3D stability constraint is tighter — a 3D simulation requires smaller timesteps (or coarser grids) than an equivalent 2D simulation.

### 14.4 3D Velocity Models

The 3D velocity model `c(x, y, z)` is stored as a 3D array `[nz, ny, nx]` on a uniform Cartesian grid. The `VelocityModel3D` dataclass mirrors the 2D `VelocityModel` with the addition of `z`, `dz`, and `nz`. Available model types:

| Model | Velocity Structure | Physical Analogue |
|---|---|---|
| **Uniform** | Constant `c` everywhere | Open air, isotropic 3D propagation |
| **Layered-z** | Horizontal layers defined by z-boundaries | Temperature or humidity stratification |
| **Ground-layer** | Air above `z_ground`, high-velocity material below | Air–ground impedance interface |

The ground-layer model assigns `c_air = 343 m/s` above the ground plane and `c_ground = 1500 m/s` below, creating a strong impedance contrast that naturally produces ground reflections in the FDTD simulation.

### 14.5 Memory Considerations

A 3D FDTD simulation requires two full pressure fields (current and previous), each of size `nz × ny × nx`. For a domain of 100 × 100 × 100 cells at 8 bytes per cell (float64), the memory is `2 × 10⁶ × 8 = 16 MB`. For a realistic domain (e.g., 200 × 200 × 125 cells with `dx = 1 m`), memory is `2 × 5 × 10⁶ × 8 = 80 MB` — manageable on modern hardware. MPI decomposition along the z-axis distributes this across ranks.

---

## 15. Ground Reflection and the Image-Source Method

### 15.1 Physical Motivation

In real outdoor environments, acoustic waves propagate not only directly from source to receiver but also via reflections off the ground surface. The ground reflection arrives later and (typically) with a phase flip, creating an interference pattern that can enhance or diminish the received signal depending on geometry and frequency.

### 15.2 The Image-Source Principle

The image-source method is a classical technique for computing ground reflections without explicitly modelling the ground as a boundary condition. The reflected wave from a source at `(x_s, y_s, z_s)` above a ground plane at `z = z_ground` is equivalent to the direct wave from an **image source** located at:

```
(x_img, y_img, z_img) = (x_s, y_s, 2·z_ground - z_s)
```

The image source is the mirror reflection of the true source across the ground plane. The total received pressure is the sum of the direct and reflected contributions:

```
p_total(t) = p_direct(t) + R · p_reflected(t)
```

where `R` is the ground reflection coefficient.

### 15.3 Reflection Coefficient

For a planar interface between air and ground, the reflection coefficient depends on the acoustic impedances of the two media:

```
R = (Z_ground - Z_air) / (Z_ground + Z_air)
```

For air (`Z ≈ 412 Pa·s/m`) and ground (`Z ≈ 3.75 × 10⁶ Pa·s/m`), `R ≈ +1.0` (nearly total reflection). However, real ground surfaces absorb some energy and produce a phase shift. The code uses a default coefficient of `R = -0.9`:

- **Magnitude** (0.9): 90% of incident pressure is reflected, 10% absorbed.
- **Sign** (negative): The reflection includes a 180° phase flip, consistent with a soft-boundary approximation for grassy/earthy surfaces at low frequencies.

### 15.4 Interference Pattern

The direct and reflected waves create a standing-wave pattern in the vertical direction. At a receiver height `z_r` above the ground, the path-length difference is:

```
Δr = √((x_s - x_r)² + (y_s - y_r)² + (z_s + z_r - 2·z_ground)²) 
   - √((x_s - x_r)² + (y_s - y_r)² + (z_s - z_r)²)
```

Constructive interference occurs when `Δr = n·λ` and destructive when `Δr = (n + ½)·λ`. With `R = -0.9` (phase flip), these conditions are swapped.

### 15.5 Implementation

The analytical 3D forward model (`forward_3d.py`) implements the image-source method by computing two contributions per source-microphone pair:

1. **Direct path**: Distance `r_direct`, amplitude `1/r_direct`, delay `r_direct/c`.
2. **Reflected path**: Distance `r_image`, amplitude `R/r_image`, delay `r_image/c`.

Both paths include exponential air absorption `exp(-α·r)`. The contribution of each path is applied via emission-time interpolation at the receiver's sample times.

For the 3D FDTD solver, ground reflection is produced naturally by the impedance contrast in the ground-layer velocity model — no explicit image-source computation is needed.

---

## 16. Acoustic Feature Extraction — Mel Spectrograms

### 16.1 Motivation

Source classification requires transforming raw pressure time-series into a compact representation that captures the spectro-temporal signature of the source. The **mel spectrogram** is the standard representation for audio classification tasks. It provides:

- **Time-frequency decomposition** via the Short-Time Fourier Transform (STFT)
- **Perceptually motivated frequency resolution** via the mel scale
- **Dynamic range compression** via logarithmic scaling

### 16.2 The Mel Scale

The mel scale maps physical frequency (Hz) to a perceptual pitch scale that is approximately linear below 1 kHz and logarithmic above:

```
mel(f) = 2595 · log₁₀(1 + f/700)
```

The inverse mapping is:

```
f(mel) = 700 · (10^(mel/2595) - 1)
```

This warping compresses the high-frequency axis, allocating more resolution to the low-frequency region where drone harmonics (100–900 Hz) reside. For the default parameters (`f_min = 20 Hz`, `f_max = Nyquist`), the mel filterbank provides approximately 3× finer resolution in the 100–500 Hz range compared to a linear frequency axis.

### 16.3 STFT Computation

The Short-Time Fourier Transform segments the signal into overlapping frames of length `n_fft` (default 512 samples), each windowed by a Hann function `w(n)`:

```
X(m, k) = Σ_{n=0}^{N-1} x(n + m·H) · w(n) · exp(-j·2π·k·n/N)
```

where `m` is the frame index, `k` is the frequency bin, `H` is the hop length (default 128 samples), and `N = n_fft`. The power spectrum is `|X(m, k)|²`.

### 16.4 Mel Filterbank

A bank of `n_mels` (default 64) triangular filters is applied to the power spectrum. The `m`-th filter is centred at mel frequency `f_m` and spans from `f_{m-1}` to `f_{m+1}` (the centres of the adjacent filters):

```
H_m(k) = 0                                 if f(k) < f_{m-1}
        = (f(k) - f_{m-1}) / (f_m - f_{m-1})  if f_{m-1} ≤ f(k) < f_m
        = (f_{m+1} - f(k)) / (f_{m+1} - f_m)  if f_m ≤ f(k) ≤ f_{m+1}
        = 0                                 if f(k) > f_{m+1}
```

Applying the filterbank to the power spectrum produces the mel spectrum:

```
S_mel(m, j) = Σ_k H_j(k) · |X(m, k)|²
```

### 16.5 Log Compression

The human auditory system responds approximately logarithmically to intensity. Log compression brings the dynamic range of the mel spectrum to a scale where both quiet and loud features are visible:

```
S_log(m, j) = log(max(S_mel(m, j), ε))
```

where `ε = 10⁻¹⁰` prevents `log(0)`.

The output is a 2D array of shape `(n_mels, n_time_frames)` — the **log-mel spectrogram**. This serves as the input to the acoustic classifier CNN.

### 16.6 Spectral Signatures by Source Class

Different source classes produce distinctive mel-spectrogram patterns:

| Source Class | Spectral Signature |
|---|---|
| **Quadcopter** | Strong harmonics at BPF × k with beat modulation from 4 rotors; broadband between harmonics |
| **Hexacopter** | Similar to quadcopter but richer beat pattern from 6 rotors; slightly different harmonic decay |
| **Fixed-wing** | Fewer, more widely spaced harmonics from a single propeller; steeper harmonic rolloff |
| **Bird** | Broadband wing-beat pulses at 3–12 Hz; occasional narrowband vocalisations at 1–8 kHz |
| **Ground vehicle** | Low-frequency engine harmonics (25–60 Hz fundamental); broadband tire noise at 200–1000 Hz |
| **Unknown** | Weak, unstructured broadband noise with occasional weak tones |

---

## 17. Kinematic Feature Physics

### 17.1 Motivation

Source classification based solely on acoustic features is limited by range, SNR, and propagation effects. Kinematic features — extracted from the tracker's estimated position and velocity history — provide a complementary information channel. Different source classes have physically distinct motion patterns:

- **Quadcopters and hexacopters** can hover, orbit, and maneuver agilely at low to moderate speeds.
- **Fixed-wing aircraft** must maintain a minimum airspeed and have limited turning rates.
- **Birds** fly with characteristic wingbeat patterns and have high agility but lower top speeds.
- **Ground vehicles** are constrained to `z ≈ 0` with moderate speeds and smooth trajectories.

### 17.2 Feature Vector

The kinematic feature extractor computes a 14-dimensional vector from tracker output:

| Index | Feature | Formula | Physical Meaning |
|---|---|---|---|
| 0 | Speed mean | `mean(||v||)` | Average translational speed |
| 1 | Speed std | `std(||v||)` | Speed variability (maneuverability) |
| 2 | Speed min | `min(||v||)` | Ability to hover (min ≈ 0 for rotorcraft) |
| 3 | Heading rate mean | `mean(|dθ/dt|)` | Average turn rate |
| 4 | Heading rate std | `std(dθ/dt)` | Turn rate variability |
| 5 | Curvature mean | `mean(|dθ/dt| / ||v||)` | Mean path curvature (tight turns → high) |
| 6 | Curvature std | `std(curvature)` | Curvature variability |
| 7 | Altitude mean | `mean(z)` | Typical operating height |
| 8 | Altitude std | `std(z)` | Altitude variability |
| 9 | Altitude rate std | `std(dz/dt)` | Vertical maneuver intensity |
| 10 | Zero-altitude fraction | `mean(|z| < 1 m)` | Fraction of time near ground (→ 1 for vehicles) |
| 11 | Hover fraction | `mean(||v|| < 1 m/s)` | Fraction of time near-stationary |
| 12 | Heading rate autocorrelation | `acf(dθ/dt, lag=1)` | Periodicity of turns (high for orbits) |
| 13 | Reserved | `0.0` | Placeholder for future features |

Heading rate is computed from unwrapped heading `θ = atan2(vy, vx)` via finite differences with timestep `dt`.

### 17.3 Discriminating Power

Key discriminating features by source class:

| Feature | Quadcopter | Fixed-wing | Bird | Ground vehicle |
|---|---|---|---|---|
| Speed min | ≈ 0 (hover) | > 10 m/s | > 5 m/s | > 2 m/s |
| Altitude mean | 20–100 m | 30–150 m | 5–200 m | ≈ 0 m |
| Zero-altitude fraction | 0.0 | 0.0 | 0.0 | 1.0 |
| Hover fraction | 0.0–0.5 | 0.0 | 0.0 | 0.0 |
| Curvature mean | variable | low | moderate | low |
| Heading rate autocorrelation | high (orbits) | low | variable | low |

The **zero-altitude fraction** is a strong discriminator for ground vehicles. **Hover fraction** uniquely identifies rotorcraft. **Speed minimum** separates fixed-wing (cannot hover) from rotorcraft (can hover). These are physics-based features that reflect fundamental flight-mechanics constraints.

---

## 18. Classification Theory

### 18.1 CNN for Spectro-Temporal Pattern Recognition

Convolutional Neural Networks (CNNs) are well-suited for mel-spectrogram classification because:

1. **Local patterns**: Drone harmonics appear as horizontal ridges in the spectrogram. Convolutional kernels detect these local spectral features regardless of position.
2. **Translation invariance**: The same harmonic pattern at different time offsets should produce the same classification. Pooling layers provide this invariance.
3. **Hierarchical features**: Lower layers detect spectral edges and harmonics; higher layers combine these into class-specific patterns.

The `AcousticClassifier` uses three convolutional layers with increasing channel depth (16 → 32 → 64), interspersed with batch normalisation and ReLU activations. Global average pooling collapses the spatial dimensions to a fixed-length 64-dimensional embedding, which is mapped to class logits by a single fully connected layer.

### 18.2 Two-Branch Fusion

The `FusionClassifier` combines acoustic and kinematic information:

- **Branch A (Acoustic)**: Identical architecture to `AcousticClassifier` up to the penultimate layer, producing a 64-dimensional acoustic embedding.
- **Branch B (Kinematic)**: A two-layer MLP (14 → 32 → 32) producing a 32-dimensional kinematic embedding.
- **Fusion**: The embeddings are concatenated (96 dimensions) and processed by two FC layers (96 → 64 → n_classes).

Fusion improves classification because the two modalities provide complementary information: acoustic features are strong when SNR is high (short range, low noise), while kinematic features are informative at any range (they come from the tracker, not the raw signal). When one modality is degraded, the other can compensate.

The acoustic branch can be initialised from a pre-trained `AcousticClassifier`, providing a warm start for fusion training.

### 18.3 Maneuver Detection as Temporal Classification

Maneuver detection classifies short segments of the tracker's state history into maneuver categories. The input is a `(6, N)` tensor: 6 state features (x, y, z, vx, vy, vz) over `N` time steps (default 20). Positions are mean-subtracted to remove absolute location dependence.

The `ManeuverClassifier` uses 1D convolutions along the time axis:

1. `Conv1d(6 → 32, kernel_size=5)`: Detects short-term velocity patterns.
2. `Conv1d(32 → 64, kernel_size=5)`: Detects higher-level motion patterns.
3. Global average pooling → FC(64 → n_classes).

Six maneuver classes capture the physically distinct motion regimes:

| Maneuver | Physical Signature |
|---|---|
| **Steady** | Constant speed, constant heading, constant altitude |
| **Turning** | Constant speed, changing heading (circular arc) |
| **Accelerating** | Changing speed, constant heading |
| **Diving** | Significant negative altitude rate, reduced horizontal speed |
| **Evasive** | Rapidly changing heading and speed (random-walk-like) |
| **Hovering** | Near-zero velocity in all directions |

### 18.4 Integration with Fire Control

The classification results feed into the 3D fire-control module:

1. **Class-based engagement rules**: The engagement decision (`can_fire`) considers the target classification. Threat classes (`quadcopter`, `hexacopter`, `fixed_wing`) are eligible for engagement; non-threat classes (`bird`, `ground_vehicle`, `unknown`) are not. A confidence threshold (default 0.7) must be exceeded for the classification to override the default engagement behaviour.

2. **Maneuver-adaptive process noise**: When the maneuver classifier detects an `evasive` maneuver, the EKF's process noise is multiplied by a large factor (default 10×), widening the predicted uncertainty. This causes the engagement envelope to contract (higher uncertainty → pattern diameter insufficient), preventing ill-advised shots during unpredictable motion. Conversely, `hovering` reduces the multiplier (0.5×), tightening the estimate and enabling more confident engagement.

---

## Summary of Physical Constants

| Symbol | Value | Description |
|---|---|---|
| `p_ref` | 20 × 10⁻⁶ Pa | Reference pressure (threshold of hearing) |
| `c_air` | 343 m/s | Speed of sound in air at 20°C |
| `c_dirt` | 1500 m/s | Speed of sound in rock/soil |
| `c_wall` | 2000 m/s | Canyon wall material velocity |
| `c_building` | 2500 m/s | Urban building material velocity |
| `ρ_air` | ~1.2 kg/m³ | Air density (not explicitly modelled) |
| `f_fundamental` | 150 Hz | Default drone fundamental frequency |
| `SPL_drone` | 90 dB | Drone source level at 1 m |
| `SPL_wind` | 55 dB | Wind noise broadband RMS |
| `SPL_sensor` | 40 dB | Sensor self-noise floor |
| `v_muzzle` | 400 m/s | Shotgun pellet muzzle velocity |
| `α` | 1.5 m/s/m | Pellet deceleration |
| `β` | 0.025 m/m | Pattern spread rate |
| `R_ground` | −0.9 | Default ground reflection coefficient |
| `f_mel_ref` | 700 Hz | Mel scale reference frequency |
| `n_mels` | 64 | Default mel filterbank bands |
| `n_fft` | 512 | Default FFT size for mel spectrogram |

---

*Next: [Algorithm Descriptions](algorithms.md) — How these physics are translated into computational procedures, including the 3D solvers and ML classifiers.*

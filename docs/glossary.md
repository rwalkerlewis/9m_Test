# Glossary

Quick-reference definitions for the terms, acronyms, and concepts used
throughout this project.  Entries are grouped by topic and sorted
alphabetically within each group.

---

## Acoustic Signal Processing

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Beamforming** | — | Spatial filtering that steers gain and phase across an array of microphones to enhance the signal arriving from a particular direction. |
| **Cross-Spectral Density Matrix** | CSDM | Frequency-domain covariance of sensor signals: $\hat{\mathbf{R}} = \frac{1}{K}\sum_k \mathbf{X}(f_k)\mathbf{X}^H(f_k) + \delta\mathbf{I}$.  Diagonal loading $\delta$ regularises against noise. |
| **Exponential Moving Average** | EMA | Recursive low-pass filter applied to per-window bearings on the unit circle; smoothing factor $\alpha = 0.35$ by default. |
| **Generalized Cross-Correlation — Phase Transform** | GCC-PHAT | TDOA extraction method: $G_{ij}(f)=X_i(f)X_j^*(f)/|X_i(f)X_j^*(f)|$; the inverse FFT peak gives the inter-sensor time lag. |
| **Minimum Variance Distortionless Response** | MVDR (Capon) | Adaptive beamformer that minimises output power subject to unity gain in the look direction; used on a polar (azimuth × range) grid. |
| **Multiple Signal Classification** | MUSIC | Subspace DOA method; eigendecomposes the CSDM and searches for steering vectors orthogonal to the noise subspace. |
| **Root-Mean-Square** | RMS | Square root of the mean squared pressure over a time window; proxy for signal power. |
| **Signal-to-Noise Ratio** | SNR | Ratio of signal RMS power to noise floor, usually expressed in dB. |
| **Sound Pressure Level** | SPL | $L_p = 20\log_{10}(p_\text{rms}/p_\text{ref})$ dB, with $p_\text{ref}=20\;\mu\text{Pa}$.  Default drone source is 90 dB SPL at 1 m. |
| **Steered Response Power — Phase Transform** | SRP-PHAT | Primary bearing estimator in the production pipeline.  Delay-and-sum beamforming with phase-only (PHAT) weighting over 100–2000 Hz; 360 candidate bearings at 1° spacing. |
| **Time-Delay of Arrival** | TDOA | Difference in arrival time of a wavefront at two sensors; extracted via GCC-PHAT; basis of near-field range estimation. |

## Water-Based Harbor Sensing

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Automatic Identification System** | AIS | Synthetic ship position, speed, and course reports used as side information for known-vessel prediction and suppression. |
| **Distributed Acoustic Sensing** | DAS | Fiber-optic sensing represented here as a dense line of pressure or strain-like channels along a cable. This model does not simulate the optical interrogator. |
| **Bistatic Localization** | — | Inferring a silent body's position from a signal transmitted by a separate illuminator (in this project, an AIS-reporting ship) and scattered to the receiver array. |
| **Gauge Length** | — | Distance along a DAS cable between the outer field samples used to form one simplified strain channel. |
| **Harbor Occupancy Grid** | — | Track-before-detect state that accumulates frame-by-frame matched-field likelihood across candidate water positions. |
| **Pressure-Release Boundary** | — | Boundary condition with pressure set to zero; used for land cells and soft-body approximations in the harbor FDTD model. |
| **Strain-Mode DAS** | — | Harbor cable mode that records the unnormalised second difference of pressure sampled at three points along the fiber gauge length. |
| **Synthetic Harbor** | — | Two-dimensional plan-view water/land grid with shoreline damping. Depth is informational, not a vertical water-column coordinate. |
| **Water Scatterer** | — | Finite-size moving mask that modifies the pressure field; supported models are soft/pressure-release and absorbing, not a rigid Neumann hull. |

## Bearing & Range Estimation

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Auto Range Selection** | — | With `ranging.method = "auto"`, `run_pipeline.py` picks one method for the whole run before the window loop: TDOA if the ground-truth distance at the peak-RMS window is ≤ `auto_cpa_threshold_m` (3 m), RMS otherwise. |
| **Bearing Estimation** | DOA | Determining the azimuth (and optionally elevation) from the array to the acoustic source.  Production pipeline uses SRP-PHAT. |
| **Bearing-Rate Range** | — | Kinematic range estimate: $\hat{R}\approx v_\perp / |\dot{\theta}|$; useful when sample rate limits TDOA accuracy. |
| **Closest Point of Approach** | CPA | Minimum distance between the projectile path and the target trajectory, evaluated over a time-of-flight window.  A hit is scored when CPA ≤ effective threshold. |
| **RMS Inverse-Square Law** | — | Range from a calibration pair: $\hat{R}=R_\text{ref}\sqrt{A_\text{ref}/A_\text{obs}}$; the pipeline calibrates once from the whole-recording peak RMS and the ground-truth distance at that peak; clamped to [5 m, 100 m] by the pipeline defaults. |

## Tracking & Kinematic State

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Causal WLS Tracker** | WLS | Weighted-least-squares constant-velocity fit over a sliding window of past detections (minimum 5).  Recomputed every window — no recursive state. |
| **Maneuver Classification** | — | CNN that classifies a kinematic window (6 channels × 20 steps) into one of six classes: *steady, turning, accelerating, diving, evasive, hovering*.  With maneuver-aware tracking it scales the tracker covariance cap (e.g. evasive → 2.5×) and the effective hit threshold. |
| **Track Blending** | — | Blending factor $\beta=\min(R/20,\;1)$: raw detections trusted at close range, WLS fit trusted beyond 20 m. |
| **Track State** | — | Per-target state vector: position $(x,y,z)$, velocity $(v_x,v_y,v_z)$, reference time, per-axis residuals, and detection count. |

## Fire Control & Ballistics

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Bearing-Rate Fire Gating** | — | Evaluates $\dot{\theta}$ over a 0.15 s window; ≥ 15°/s (crossing) → lead shot gated by the engagement envelope; < 15°/s (radial/receding) → radial shot at the current bearing. |
| **Effective Threshold** | — | $\tau_\text{eff}=m\,\max(r_\text{pattern},\;\tau_\text{hit})$, with $m=1$ unless a maneuver-aware multiplier applies; a shot scores a hit if the CPA miss $<\tau_\text{eff}$. |
| **Engagement Envelope** | — | `compute_engagement_3d` authorization gate for lead shots: threat class and confidence, evasive-maneuver uncertainty, pellet reach, max engagement range, and position uncertainty vs. limit and pattern diameter. |
| **Lead Computation** | — | Iterative ballistic intercept (up to 5 iterations) predicting where the target will be when pellets arrive; returns aim bearing, aim elevation, TOF, and intercept position. |
| **Pattern Diameter** | — | Shotgun pellet spread: $d=s\cdot R$; `pattern_diameter` defaults to $s=0.025$, `run_pipeline.py` uses 0.2 and `examples/configs/pipeline.config.json` sets 0.3. |
| **Pellet Velocity Decay** | — | $v(R)=\max(v_0 - d\cdot R,\;0)$; muzzle velocity $v_0=400$ m/s, deceleration rate $d=1.5$ (m/s)/m. |
| **Time of Flight** | TOF | Time for pellets to travel range $R$ under linear velocity decay: $t = R / v_\text{avg}$ with $v_\text{avg}=\tfrac12(v_0 + v_0 - dR)$; infinite if the pellet stops first. |

## FDTD Solver & Wave Physics

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Air Absorption** | — | Analytical model: $p(r)=p_0 e^{-\alpha_\text{air}\cdot r}$.  FDTD: uniform background damping `FDTDConfig.air_absorption` (default 0.005 per step). |
| **Courant–Friedrichs–Lewy Condition** | CFL | Stability limit for explicit time-stepping; 2-D: $\Delta t\leq 2\Delta x/\bigl((c_\text{max}+v_\text{wind})\sqrt{2\rho}\bigr)$; safety factor 0.9. |
| **Finite-Difference Time-Domain** | FDTD | Explicit solver for the scalar wave equation on a regular Cartesian grid (`solver/fdtd.py`); even-order spatial stencils (2, 4, 6, 8, …), leapfrog time integration (prototype `lw4`/`kspace` schemes), sponge or CPML boundaries, `float64`/`float32`, optional CuPy GPU acceleration with a fused CUDA kernel, and MPI domain decomposition. |
| **Geometric Spreading** | — | Amplitude attenuation from wavefront expansion: $1/r$ (3-D point source) or $1/\sqrt{r}$ (2-D line source). |
| **Ground Reflection** | — | Optional in the analytical 3-D model: image source at $(x_s, y_s, 2z_g - z_s)$ with reflection coefficient $-0.9$ by default.  In FDTD, reflections come from the velocity model (e.g. the 3-D `ground_layer` domain). |
| **Helmholtz Equation** | — | Frequency-domain form of the wave equation: $\nabla^2 P + k^2 P = 0$; solved with PML absorbing boundaries. |
| **Leapfrog Scheme** | — | Second-order time integration: $p^{n+1}=2p^n - p^{n-1} + c^2 \Delta t^2\,L[p^n] - \sigma(p^n-p^{n-1})$, where $L$ is the discrete Laplacian and $\sigma$ the damping. |
| **Perfectly Matched Layer** | PML / CPML | Absorbing boundary via complex coordinate stretching, theoretically reflectionless.  Used by the Helmholtz solver; the FDTD offers the convolutional form (`boundary="cpml"`, single process). |
| **Points Per Wavelength** | PPW | Grid resolution criterion: $\lambda/\Delta x$.  `check_spatial_sampling` requires ≥ 10 by default; < 1 % dispersion needs about 20 at 2nd order and 8 at 4th order (see [Physics](physics.md#numerical-dispersion)). |
| **Sponge Layer** | — | Default FDTD absorbing boundary: damping $\sigma$ ramped quadratically to `damping_max` over `damping_width` cells (default 40); simpler but less effective than CPML. |
| **Wave Equation** | — | Governing PDE: $\partial^2 p/\partial t^2 = c^2(\mathbf{x})\,\nabla^2 p$.  Heterogeneous sound speed $c(\mathbf{x})$ supports scattering and refraction. |
| **Wind Effects** | — | Effective sound speed: $c_\text{eff}=c+\hat{\mathbf{n}}\cdot\mathbf{w}$; wind velocity added to $c_\text{max}$ for CFL stability. |

## Velocity Models & Domains

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Echo Canyon** | — | Two parallel high-velocity walls (2000 m/s) producing strong multipath reflections. |
| **Hills & Vegetation** | — | Two ridges separated by a valley; dirt substrate ($c\approx 1500$ m/s) forms the hills with a frequency-dependent vegetation attenuation strip above the surface. |
| **Isotropic Domain** | — | Uniform velocity, no wind, no attenuation — the simplest test case. |
| **Urban Echo** | — | Four randomly placed rectangular buildings ($c\approx 2500$ m/s) creating complex multipath; seed-controlled layout. |
| **Velocity Model** | — | Regular grid of sound-speed values (m/s) defining the propagation medium: 2-D $(n_x\times n_y)$ or 3-D $(n_z\times n_y\times n_x)$. |
| **Wind Domain** | — | Uniform velocity plus a constant wind field specified by speed (m/s) and direction (degrees). |

## Noise & Perturbations

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Position Error** | — | Gaussian perturbation of reported microphone coordinates; simulates survey or placement inaccuracy. |
| **Sensor Faults** | — | Injected per-channel defects (`inject_sensor_faults`): *elevated_noise* (high white noise floor), *dropout* (zeros), *spikes* (random impulses), *dc_offset*. |
| **Sensor Self-Noise** | — | Independent white Gaussian noise per channel; models microphone thermal and electronic noise. |
| **Transient Event** | — | Broadband impulse (explosion, gunshot) properly delay-shifted to each microphone by distance; specified in dB SPL. |
| **Wind Noise** | — | Spatially correlated noise (exponential decay over a correlation length), spectrally shaped (low-pass); simulates atmospheric turbulence. |

## Source Signals

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Drone Harmonics** | — | Multi-harmonic source: $s(t)=\sum_{k=1}^{4}a_k\sin(2\pi k f_0 t)$; $f_0=150$ Hz; amplitudes $[1.0, 0.6, 0.3, 0.15]$; 90 dB SPL at 1 m. |
| **Propeller Noise** | BPF | Blade-pass fundamental ($f_\text{BPF}=\text{RPM}\times\text{blades}/60$) plus harmonics with amplitude modulation and broadband turbulent noise; simulates a rotor. |
| **Ricker Wavelet** | — | Second derivative of a Gaussian: $(1-2\pi^2 f_0^2 t^2)\,e^{-\pi^2 f_0^2 t^2}$; broadband solver stress test. |
| **Tonal Source** | — | Single sinusoid at a specified frequency. |

## Receiver Arrays

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Aperture** | — | Maximum element separation; spatial Nyquist: $f_\text{alias}=c/(2d_\text{element})$. |
| **Circular Array** | — | Single ring of evenly spaced elements. |
| **Concentric Array** | — | Multiple rings at different radii with a configurable number of elements per ring. |
| **L-Shaped Array** | — | Two perpendicular legs; good 2-D bearing resolution. |
| **Log-Spiral Array** | — | Golden-angle spacing for maximum baseline diversity; 13 elements by default. |
| **Nested Circular Array** | — | Centre microphone plus inner ring (4 at 0.15 m) and outer ring (8 at 0.5 m radius). |

## ML Classifiers

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Acoustic Classifier** | — | 3-layer CNN mapping a mel-spectrogram to 6 source classes: *quadcopter, hexacopter, fixed_wing, bird, ground_vehicle, unknown*. |
| **Convolutional Variational Autoencoder** | CVAE | Learns to reconstruct normal acoustic patterns; novel threats flagged via high reconstruction error exceeding a calibrated threshold. |
| **Fourier Neural Operator** | FNO | Spectral convolution network trained as a drop-in FDTD surrogate; input: velocity + source + frequency channels; output: receiver traces. |
| **Fusion Classifier** | — | Acoustic CNN branch (64-dim) concatenated with a kinematic MLP (14→32→32), feeding a 96→64→6 head; combines spectral and trajectory features. |
| **Kinematic Features** | — | 14-D vector from `compute_kinematic_features`: speed mean/std/min, heading-rate mean/std, curvature mean/std, altitude mean/std/rate-std/near-zero fraction, hover fraction, heading-rate lag-1 autocorrelation, and one reserved zero. |
| **Maneuver Classifier** | — | 1-D CNN over kinematic channels × timesteps; 6 classes: *steady, turning, accelerating, diving, evasive, hovering*. |
| **Mel-Spectrogram** | — | STFT → mel filterbank → log compression; logarithmic frequency scale approximating human auditory perception. |

## Pipeline Concepts

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Classification Confidence Threshold** | — | Fire-control gate: suppresses engagement when $P(\text{non-drone})$ exceeds this value (default 0.7). |
| **Detection Engine** | — | Streaming processor (`DetectionEngine`) that orchestrates bearing estimation, range estimation, EMA smoothing, and WLS tracking one window at a time. |
| **Engagement Pipeline** | — | End-to-end chain: FDTD traces → windowed detection → tracking → fire control → CPA evaluation. |
| **RMS Fire Gate Fraction** | — | Window RMS must exceed this fraction of peak RMS (default 0.20) to authorise firing; suppresses shots during silence. |
| **Window Detection** | — | `WindowDetection`: result for a single analysis window: time, detected flag, window RMS, bearing(s) (rad/deg), range, position, track state.  Fire decisions carry a reason code (`RADIAL_FIRE`, `BEHIND`, `NO_TRACK`, `CLASS_REJECT`, `TARGET_ENGAGED`, …) and ML results may relabel the class `NOVEL_THREAT`. |

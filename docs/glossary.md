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

## Bearing & Range Estimation

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Auto Range Selection** | — | Pipeline switches between TDOA (when CPA ≤ threshold, default 3 m) and RMS inverse-square (otherwise). |
| **Bearing Estimation** | DOA | Determining the azimuth (and optionally elevation) from the array to the acoustic source.  Production pipeline uses SRP-PHAT. |
| **Bearing-Rate Range** | — | Kinematic range estimate: $\hat{R}\approx v_\perp / |\dot{\theta}|$; useful when sample rate limits TDOA accuracy. |
| **Closest Point of Approach** | CPA | Minimum distance between the projectile path and the target trajectory, evaluated over a time-of-flight window.  A hit is scored when CPA ≤ effective threshold. |
| **RMS Inverse-Square Law** | — | Range from a calibration pair: $\hat{R}=R_\text{ref}\sqrt{A_\text{ref}/A_\text{obs}}$; assumes free-field spherical spreading; clamped to [5 m, 100 m]. |

## Tracking & Kinematic State

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Causal WLS Tracker** | WLS | Weighted-least-squares constant-velocity fit over a sliding window of past detections (minimum 5).  Recomputed every window — no recursive state. |
| **Maneuver Classification** | — | CNN that classifies a kinematic window (6 channels × 20 steps) into one of six classes: *steady, turning, accelerating, diving, evasive, hovering*.  Adapts tracker covariance (e.g. evasive → 2.5×). |
| **Track Blending** | — | Blending factor $\beta=\min(R/20,\;1)$: raw detections trusted at close range, WLS fit trusted beyond 20 m. |
| **Track State** | — | Per-target state vector: position $(x,y,z)$, velocity $(v_x,v_y,v_z)$, reference time, per-axis residuals, and detection count. |

## Fire Control & Ballistics

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Bearing-Rate Fire Gating** | — | Evaluates $\dot{\theta}$ over a 0.15 s window; high bearing rate → full lead shot; low bearing rate (radial/receding) → radial aim. |
| **Effective Threshold** | — | $\tau_\text{eff}=\max(r_\text{pattern},\;\tau_\text{hit})$; a pellet cloud scores a hit if CPA $\leq\tau_\text{eff}$. |
| **Engagement Envelope** | — | Authorization gate: checks range ≤ max engagement range, position uncertainty, class label, and maneuver state before allowing a shot. |
| **Lead Computation** | — | Iterative ballistic intercept (up to 5 iterations) predicting where the target will be when pellets arrive; returns aim bearing, aim elevation, TOF, and intercept position. |
| **Pattern Diameter** | — | Shotgun pellet spread: $d=s\cdot R$; spread rate $s$ defaults to 0.3. |
| **Pellet Velocity Decay** | — | $v(R)=\max(v_0 - d\cdot R,\;0)$; muzzle velocity $v_0=400$ m/s, deceleration rate $d=1.5$ (m/s)/m. |
| **Time of Flight** | TOF | Time for pellets to travel range $R$ under linear drag: $R=v_0 t - \tfrac{1}{2}at^2$. |

## FDTD Solver & Wave Physics

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Air Absorption** | — | Exponential attenuation with distance: $p(r)=p_0 e^{-\alpha_\text{air}\cdot r}$; default coefficient 0.005. |
| **Courant–Friedrichs–Lewy Condition** | CFL | Stability limit for explicit time-stepping; 2-D: $\Delta t\leq 2\Delta x/\bigl((c_\text{max}+v_\text{wind})\sqrt{2\rho}\bigr)$; safety factor 0.9. |
| **Finite-Difference Time-Domain** | FDTD | Explicit solver for the scalar wave equation on a regular Cartesian grid; supports 2nd- through 8th-order spatial stencils, leapfrog time integration, optional CuPy GPU acceleration, and MPI domain decomposition. |
| **Geometric Spreading** | — | Amplitude attenuation from wavefront expansion: $1/r$ (3-D point source) or $1/\sqrt{r}$ (2-D line source). |
| **Ground Reflection** | — | Image-source method: virtual source at $(x_s, y_s, -z_s)$ with reflection coefficient $\approx -0.9$ for rigid ground. |
| **Helmholtz Equation** | — | Frequency-domain form of the wave equation: $\nabla^2 P + k^2 P = 0$; solved with PML absorbing boundaries. |
| **Leapfrog Scheme** | — | Second-order time integration: $p^{n+1}=2p^n - p^{n-1} + c^2 \Delta t^2\,L[p^n]$, where $L$ is the discrete Laplacian. |
| **Perfectly Matched Layer** | PML | Absorbing boundary for frequency-domain solvers via complex coordinate stretching; theoretically reflectionless at all angles. |
| **Points Per Wavelength** | PPW | Grid resolution criterion: $\lambda/\Delta x$.  Needs ≥ 10 PPW for < 1 % dispersion at 2nd order; ≥ 8 PPW at 4th order. |
| **Sponge Layer** | — | FDTD absorbing boundary: quadratic ramp $\alpha(r)=\alpha_\text{max}\,r^2$ over $W$ cells; simpler but less effective than PML. |
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
| **Sensor Faults** | — | Injected per-channel defects: *elevated_noise* (high white noise floor), *dead* (zeros), *intermittent* (random spikes). |
| **Sensor Self-Noise** | — | Independent white Gaussian noise per channel; models microphone thermal and electronic noise. |
| **Transient Event** | — | Broadband impulse (explosion, gunshot) properly delay-shifted to each microphone by distance; specified in dB SPL. |
| **Wind Noise** | — | Spatially correlated noise (exponential decay over a correlation length), spectrally shaped (low-pass); simulates atmospheric turbulence. |

## Source Signals

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Drone Harmonics** | — | Multi-harmonic source: $s(t)=\sum_{k=1}^{6}a_k\sin(2\pi k f_0 t)$; $f_0=150$ Hz; amplitudes $[1.0, 0.6, 0.35, 0.2, 0.12, 0.08]$. |
| **Propeller Noise** | BPF | Blade-pass fundamental ($f_\text{BPF}=\text{RPM}\times\text{blades}/60$) plus harmonics with beat modulation; simulates multi-rotor UAV. |
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
| **Nested Circular Array** | — | Centre microphone plus inner ring (4) and outer ring (8); default aperture 0.5 m. |

## ML Classifiers

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Acoustic Classifier** | — | 3-layer CNN mapping a mel-spectrogram to 6 source classes: *quadcopter, hexacopter, fixed_wing, bird, ground_vehicle, unknown*. |
| **Convolutional Variational Autoencoder** | CVAE | Learns to reconstruct normal acoustic patterns; novel threats flagged via high reconstruction error exceeding a calibrated threshold. |
| **Fourier Neural Operator** | FNO | Spectral convolution network trained as a drop-in FDTD surrogate; input: velocity + source + frequency channels; output: receiver traces. |
| **Fusion Classifier** | — | Acoustic CNN branch (64-dim) concatenated with a kinematic MLP (14→32→32), feeding a 96→64→6 head; combines spectral and trajectory features. |
| **Kinematic Features** | — | 14-D vector: position (3), velocity (3), speed, acceleration (3), bearing rate, range rate, altitude rate. |
| **Maneuver Classifier** | — | 1-D CNN over kinematic channels × timesteps; 6 classes: *steady, turning, accelerating, diving, evasive, hovering*. |
| **Mel-Spectrogram** | — | STFT → mel filterbank → log compression; logarithmic frequency scale approximating human auditory perception. |

## Pipeline Concepts

| Term | Abbreviation | Definition |
|------|:------------:|------------|
| **Classification Confidence Threshold** | — | Fire-control gate: suppresses engagement when $P(\text{non-drone})$ exceeds this value (default 0.7). |
| **Detection Engine** | — | Streaming processor (`DetectionEngine`) that orchestrates bearing estimation, range estimation, EMA smoothing, and WLS tracking one window at a time. |
| **Engagement Pipeline** | — | End-to-end chain: FDTD traces → windowed detection → tracking → fire control → CPA evaluation. |
| **RMS Fire Gate Fraction** | — | Window RMS must exceed this fraction of peak RMS (default 0.20) to authorise firing; suppresses shots during silence. |
| **Window Detection** | — | Result object for a single analysis window: bearing (rad/deg), range, position, track state, and flags (`RMS_GATE`, `CLASS_REJECT`, `NOVEL_THREAT`). |

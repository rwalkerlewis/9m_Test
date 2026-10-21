# Physics

## For the Non-Specialist

Sound propagates through air as a pressure wave — like ripples in a
pond, but in three dimensions.  The simulation divides the air into a
fine grid and marches forward in tiny time steps, computing how each
cell's pressure changes based on its neighbours.  The result is a set
of synthetic microphone recordings that capture echo, wind, and
absorption effects that a real sensor would observe.

---

## Acoustic Wave Equation

### Continuous form

The 2-D scalar wave equation for pressure $p(x,y,t)$ in a heterogeneous
medium with spatially varying sound speed $c(x,y)$:

$$\frac{\partial^2 p}{\partial t^2} = c^2(x,y)\left(\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2}\right)$$

The 3-D extension adds a $z$-derivative:

$$\frac{\partial^2 p}{\partial t^2} = c^2(x,y,z)\left(\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} + \frac{\partial^2 p}{\partial z^2}\right)$$

### Helmholtz equation

For a single frequency $\omega$, assuming $p(x,y,t) = P(x,y)e^{-i\omega t}$:

$$\nabla^2 P + \frac{\omega^2}{c^2(x,y)} P = -s(x,y)$$

where $s$ is the source term.  Solved as a sparse linear system with
complex-coordinate-stretch PML at the boundaries (see §Absorbing
Boundaries).

---

## Temperature-Dependent Sound Speed

The default sound speed is computed from air temperature:

$$c = 331.3\,\sqrt{1 + \frac{T}{273.15}} \quad [\text{m/s}]$$

where $T$ is the temperature in °C.  Implemented in
`air/config.py: sound_speed_from_temperature()`.

---

## Finite-Difference Discretisation

### Spatial stencils

The second derivative $\partial^2 p / \partial x^2$ is approximated with
central differences of order $N$.  The stencil half-width is $M = N/2$.

| Order | $M$ | Stencil (scaled by $1/\Delta x^2$) |
|-------|-----|-------------------------------------|
| 2 | 1 | $[1, -2, 1]$ |
| 4 | 2 | $[-1/12,\; 4/3,\; -5/2,\; 4/3,\; -1/12]$ |
| 6 | 3 | $[1/90,\; -3/20,\; 3/2,\; -49/18,\; 3/2,\; -3/20,\; 1/90]$ |
| 8 | 4 | 9-point stencil from Taylor matching |

Stencil weights are computed analytically in `fd2_coefficients(order)`.

### Time stepping

The solver default (`FDTDConfig.time_scheme="leapfrog"`) is second order leapfrog
with a damping term:

$$p^{n+1} = 2p^n - p^{n-1} + c^2 \Delta t^2 \, L[p^n] - \sigma\,(p^n - p^{n-1})$$

where $L$ is the spatial Laplacian operator and $\sigma$ is the per-cell
damping (background `air_absorption` plus the sponge ramp, see below).

Two other schemes run in a single process with the sponge boundary: `"lw4"`
(fourth order in time by the modified equation, stable up to $\sqrt{3}$
times the leapfrog limit) and `"kspace"`:

$$p^{n+1} = 2p^n - p^{n-1} + (c/c_0)^2\,\mathcal{F}^{-1}\!\left[-4\sin^2\!\left(\tfrac{1}{2} c_0 |k| \Delta t\right)\mathcal{F}p^n\right] - \sigma\,(p^n - p^{n-1})$$

with $c_0$ the largest speed of the grid.  The pseudospectral Laplacian and
the $\sin^2$ temporal correction make the step exact for a homogeneous
medium at $c_0$ (any $\Delta t$, any wavenumber the grid holds) and
unconditionally stable; where the water is slower than $c_0$ the factor
$(c/c_0)^2$ is exact only to second order in $\Delta t$.

#### The harbor recordings step with the k space scheme

Every harbor scenario (`examples/configs/harbor*.scenario.json`) sets
`time_scheme: "kspace"`.  The leapfrog with the fourth order stencil at
2 to 4 cells per wavelength at the top of the band made the simulated water
carry the direct paths 1.1 to 5.2 ms early over 135 to 344 m on the 0.5 m
and 1 m recordings and up to 12.7 ms over 1047 m on the 5 m harbor
([history](harbor_history.md#performance)); with uniform water at the
chart's speed, the 5 m harbor's leapfrog is still 10.6 ms early at
1047 m, so that error was the scheme's and not the shallow water's.
(Those numbers also include one step of a trace timing error, fixed
together with the switch: column $m$ of a gather is now the field at
$m\,\Delta t$.)

Propagation error with the k space scheme, measured by
`examples/harbor/01_datasets/harbor_propagation_error.py` (a stationary
pulse at the first ship's mid record position, the arrival at each
hydrophone minus range over the chart's speed; negative is early).
"Straight path" is the travel time through the scenario's own sound speed
map along the straight line minus range over the chart's speed, what the
water contributes; the difference is the scheme's.  Criterion, fixed
before the recordings were built: within 0.2 ms at every hydrophone on the
0.5 m and 1 m recordings, 0.1 ms on the simple harbor.

| Recording | Grid | Error per hydrophone [ms] | Straight path [ms] | Criterion |
|---|---|---|---|---|
| simple | 1 m, 3 kHz | +0.00, +0.00, +0.00, +0.02 | 0.00 (uniform water) | 0.1 ms: pass |
| bistatic, multistatic | 0.5 m, 6 kHz | -0.21, -0.03, +0.05, -0.10 | -0.23, +0.00, +0.03, -0.12 | 0.2 ms: **fail** (hydrophone 1) |
| multistatic_b | 0.5 m, 6 kHz | -0.22, -0.04, +0.03, -0.12 | -0.23, -0.02, +0.02, -0.12 | 0.2 ms: **fail** (hydrophone 1) |
| harbor | 5 m, 600 Hz | -0.42, +0.10, -0.42, +0.21 | -0.51, +0.07, -0.48, +0.37 | none (reported) |

The 0.5 m recordings fail the criterion at hydrophone 1 by 0.01 and
0.02 ms.  The straight path through their sound speed map predicts the
whole of it: the path runs through water faster than the chart's single
speed (the median of the water, 1493.7 m/s).  With the water set to one
speed (`--uniform-speed`) every hydrophone is within 0.07 ms.  The
scheme's own error is within 0.03 ms of the straight path on every
0.5 m and 1 m hydrophone.

On the 5 m harbor (12 to 78 Hz, 4 cells per wavelength at 78 Hz) the
k space step is still exact for the propagation: with uniform water the
error is 0.00 ms at three hydrophones and -0.21 ms (an eighth of a step)
at hydrophone 4, near the shore.  The rest is the water: shallow water
slower than the chart's median speed along some paths, deeper and faster
along others (straight path -0.51 to +0.37 ms).  Refining to 2.5 m
(`--dx 2.5`) moves the arrivals by at most 0.05 ms at hydrophones 1 to 3
and by 0.47 ms at hydrophone 4 (also with uniform water: the shore
reflection inside the long gate a 12 Hz pulse needs), 0.04 of a period
at 78 Hz; the 5 m grid is not refined.

What the k space scheme does not make exact: the factor $(c/c_0)^2$ in
water slower than $c_0$ (at most `shallow_c_drop` = 10 m/s, 0.7 %, slower in these scenarios); a point
source and a point receiver on a grid at 2 cells per wavelength (the 1 m
grid above 380 Hz, not used by any recording); the pressure release
walls, placed on node lines (0.31 rad at 100 to 760 Hz on the 0.5 m grid,
history); the sponge.

Cost (`output/<recording>/build.json`, Quadro RTX 4000, float64, all
runs of a recording batched):

| Recording | Grid | Steps | FDTD runs | FDTD wall time | Per run per simulated second |
|---|---|---|---|---|---|
| simple | 501 x 401 | 60 000 | 4 | 356 s | 4.4 s |
| bistatic | 801 x 601 | 120 000 | 2 | 691 s | 17.3 s |
| multistatic | 801 x 601 | 120 000 | 6 | 2030 s | 16.9 s |
| multistatic_b (60 s) | 801 x 601 | 360 000 | 6 | 6091 s | 16.9 s |
| harbor (120 s) | 481 x 361 | 72 000 | 12 | 468 s | 0.33 s |

The k space step is two float64 FFTs per field per step on a card with
float64 at 1/32 of the float32 rate: 10 to 18 times the fused leapfrog
step on the same batch (8.3 against 0.81 ms per step on the 0.5 m grid in
the [history](harbor_history.md#performance)).  Every recording together
takes 2.7 hours of FDTD.

Fields are `float64` by default; `precision="float32"` halves memory and is
much faster on consumer GPUs.  On a single GPU process the default
`kernel="auto"` runs each step as one fused CUDA kernel
(`solver/fdtd_cuda.py`); `kernel="reference"` keeps the array code.

### CFL stability condition

Stability requires that the timestep not exceed:

**2-D:**

$$\Delta t \leq \frac{2\,\Delta x}{(c_\text{max} + v_\text{wind})\,\sqrt{2\,\rho}}$$

**3-D:**

$$\Delta t \leq \frac{2\,\Delta x}{(c_\text{max} + v_\text{wind})\,\sqrt{3\,\rho}}$$

where $c_\text{max}$ is the maximum sound speed in the domain,
$v_\text{wind} = \|\mathbf{w}\|$ is the wind speed magnitude, and
$\rho$ is the 1-D spectral radius of the FD stencil at Nyquist:

$$\rho = \left|c_0 + 2\sum_{k=1}^{M} c_k\,(-1)^k\right|$$

with $c_k$ being the stencil coefficients.  `fd2_cfl_factor(coefficients)`
returns $\rho$ (pass it `fd2_coefficients(order)`).  The solver applies a
safety factor (`cfl_safety`, default 0.9):
$\Delta t_\text{sim} = 0.9\,\Delta t_\text{CFL}$.  A user-supplied `dt`
above the limit of the chosen time scheme raises `ValueError`.

The simple CFL check in `sampling.check_cfl()` uses the 2nd-order
formula $C = c_\text{max}\,\Delta t / \Delta x \leq 1/\sqrt{2}$ for
quick validation.

### Numerical dispersion

Higher-order stencils reduce numerical dispersion at a given
points-per-wavelength (PPW):

| Order | PPW for < 1% dispersion |
|-------|-------------------------|
| 2 | 20 |
| 4 | 8 |
| 8 | 5 |

`check_spatial_sampling()` verifies that the grid spacing satisfies the
PPW requirement for a given frequency.

---

## Absorbing Boundaries

### Sponge layer (FDTD default)

With `boundary="sponge"` (default) the FDTD solver uses a damping zone of
`damping_width` = $W$ cells at each boundary.  The damping enters the update
as the $-\sigma\,(p^n - p^{n-1})$ term above, with

$$\sigma(r) = \sigma_\text{air} + (\sigma_\text{max} - \sigma_\text{air})\,r^2$$

where $r \in [0, 1]$ is the fractional distance into the sponge
(0 = interior edge, 1 = domain boundary), $\sigma_\text{max}$ is
`damping_max` (default 0.15) and $\sigma_\text{air}$ is `air_absorption`.
The quadratic onset avoids the spurious reflections that a hard truncation
would produce.

This is **not** a PML: it is simple and robust but needs a wide layer
(default 40 cells) and is not theoretically reflectionless.

### CPML (FDTD option)

`boundary="cpml"` replaces the sponge with a convolutional perfectly matched
layer of `cpml_width` cells (default 15) on the domain edges, using auxiliary
memory variables for the second-order wave equation (Pasalic & McGarry 2010).
The profile is $\sigma(l) = \sigma_\text{max}\,l^{\,\text{cpml\_order}}$ with
$\sigma_\text{max} = -(\text{order}+1)\,c_\text{max}\ln(r_0) / (2W\Delta x)$
for a target normal-incidence reflection `cpml_r0`; `cpml_alpha_max` adds a
complex frequency shift for grazing incidence.  CPML is single-process only
(not under MPI) and not combined with the `lw4`/`kspace` schemes.

### PML (Helmholtz)

The frequency-domain Helmholtz solver (`solver/helmholtz.py`) uses a Perfectly
Matched Layer with complex coordinate stretching.  In the PML region
the spatial coordinate $x$ is replaced by

$$\tilde{x} = x + \frac{i}{\omega}\int_0^x \sigma(x')\,dx'$$

where $\sigma(x)$ ramps from zero at the interior edge to a maximum at
the boundary.  This is a distinct mechanism from the FDTD sponge layer
and provides theoretically exact absorption for outgoing waves at all
angles and frequencies.

---

## Sound Pressure Level

All pressures are in Pascals.  SPL is referenced to
$p_\text{ref} = 20\,\mu\text{Pa}$:

$$L_p = 20 \log_{10}\!\left(\frac{p_\text{rms}}{p_\text{ref}}\right) \quad [\text{dB SPL}]$$

Default drone source level: 90 dB SPL at 1 m = 0.632 Pa.

---

## Geometric Spreading

In the far field, pressure decays as $1/r$ (3-D spherical) or
$1/\sqrt{r}$ (2-D cylindrical).  The RMS range estimator in the
pipeline calibrates with $p_\text{rms} \propto 1/r$ at the closest point
of approach but inverts with $\hat{R} = R_\text{ref}\sqrt{A_\text{ref}/A}$
(i.e. $1/r^2$); see [Algorithms §2.1](algorithms.md#21--rms-inverse-square-law).

---

## Air Absorption

Exponential attenuation with propagation distance:

$$p(r) = p_0 \cdot e^{-\alpha_\text{air}\, r}$$

In the FDTD solver, `air_absorption` is the uniform background
$\sigma$ in the damping term of the update (per-step numerical damping,
not a frequency-dependent atmospheric absorption model).  The analytical forward model
(`simulate_3d_traces`) applies it explicitly per source–receiver path.

---

## Wind Effects

Wind shifts the effective sound speed directionally.  With wind
velocity $\mathbf{w} = (w_x, w_y)$ (2-D) or $(w_x, w_y, w_z)$ (3-D):

$$c_\text{eff} = c + \hat{\mathbf{n}} \cdot \mathbf{w}$$

where $\hat{\mathbf{n}}$ is the propagation direction unit vector.  In
the FDTD solver, the wind magnitude is added to $c_\text{max}$ for CFL
computation.

`DomainMeta3D` includes a `wind_vz` field for vertical wind
components.

---

## Ground Reflection

The analytical 3-D forward model (`simulate_3d_traces`) optionally adds a
ground reflection via the image-source method
(`enable_ground_reflection=False` by default; plane at `ground_z`, default 0).
An image source is placed at $(x_s, y_s, 2z_g - z_s)$ and the
ground-reflected path is:

$$p_\text{total} = \frac{p_0}{r_d}\,e^{-\alpha r_d} + R_g\,\frac{p_0}{r_r}\,e^{-\alpha r_r}$$

where $r_d$ is the direct path length, $r_r$ is the reflected path
length, and $R_g$ is `ground_reflection_coeff` (default $-0.9$, a
single frequency-independent value).

In FDTD, ground reflections come from the velocity model itself, e.g. the
3-D `ground_layer` domain (air over a high-velocity ground half-space).

---

## Drone Acoustic Signatures

`make_drone_harmonics()` generates a multi-harmonic signal
representative of multi-rotor UAVs:

$$s(t) = \sum_{k=1}^{N_h} a_k \sin(2\pi k f_0 t)$$

Default: $f_0 = 150$ Hz, $N_h = 4$, amplitudes
$[1.0,\; 0.6,\; 0.3,\; 0.15]$, scaled to 90 dB SPL at 1 m.

`make_source_propeller()` generates blade-pass-frequency harmonics
(default 3 blades at 3600 RPM, 14 harmonics) with amplitude modulation at
the rotor rate plus broadband turbulent noise.

---

## Water-Based Harbor Model (2-D)

The water (harbor) package (`acoustic_sim.water`) reuses the scalar pressure FDTD solver in a **2-D
plan-view** approximation. A boolean mask identifies land cells, which
are held at zero pressure (pressure-release boundaries); water cells
carry the effective sound-speed field. The default depth map is
informational, while its distance-to-shore estimate produces a modest
shallow-water sound-speed drop. It does not add a vertical coordinate,
free surface, seabed, stratified water column, bathymetry-dependent
boundary condition, or elastic sediment model.

The solver adds a quadratic damping ramp in water near shore to limit
long-lived harbor reverberation and retains the ordinary sponge at the
outer grid edge. `HarborFDTDConfig.shore_loss_max = 0` gives reflecting
quays; increasing it absorbs more energy near land. This parameter is
per-step numerical damping, not a calibrated physical bottom-loss
coefficient.

Harbor sensors are sampled from the same pressure field. A pressure-mode
cable records pressure at each channel. A strain-mode cable samples at
three points along its gauge length and records the unnormalised second
difference, which introduces axial, frequency- and direction-dependent
response. This is a simplified DAS observation model, not a full optical
interrogator or elastic-fiber simulation.

Finite-size scatterers are represented by a smoothly moving mask that
multiplies the pressure field. Soft/hard mask kinds are pressure-release
approximations; the absorbing kind removes a configurable fraction of
pressure. These are numerical body models, not a validated rigid-hull
boundary solver. The separate bistatic-budget script includes idealized
rigid-cylinder theory for comparison, but that is not the boundary
condition used by FDTD harbor runs.

The FDTD wave equation is linear, so source-only runs can be cached and
added to form a noisy multi-vessel dataset. Scattered components are
computed by subtracting the matching run without the body. This
superposition assumes one body at a time; the dataset builder does not
model multiple scattering between bodies. See [Harbor](harbor.md) for
the processing chains and validation limits.

---

## Noise Models

| Type | Function | Physics |
|------|----------|---------|
| Wind noise | `generate_wind_noise` | Spatially correlated (exponential decay with inter-mic distance), spectrally shaped (low-pass with corner frequency), scaled to specified dB SPL |
| Sensor self-noise | `generate_sensor_noise` | White Gaussian, independent per channel, scaled to dB SPL |
| Sensor faults | `inject_sensor_faults` | Elevated noise floor, dropout (zeros), spikes (random impulses), DC offset — applied to a fraction of channels |
| Transient | `inject_transient` | Broadband impulse at a specified location and time, propagated to each receiver with delay and $1/r$ attenuation |
| Position error | `perturb_mic_positions` | Gaussian perturbation to receiver coordinates, simulating survey/placement errors |

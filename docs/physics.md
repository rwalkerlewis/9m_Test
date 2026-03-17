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
`config.sound_speed_from_temperature()`.

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

Second-order leapfrog:

$$p^{n+1} = 2p^n - p^{n-1} + c^2 \Delta t^2 \, L[p^n]$$

where $L$ is the spatial Laplacian operator.

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

with $c_k$ being the stencil coefficients.  `fd2_cfl_factor(order)`
returns $\rho$.  The solver applies a safety factor (default 0.9):
$\Delta t_\text{sim} = 0.9\,\Delta t_\text{CFL}$.

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

### Sponge layer (FDTD)

The FDTD solver uses a damping zone (sponge layer) of width $W$ cells
at each boundary.  After each timestep, pressure in the sponge is
multiplied by a decay factor:

$$p \leftarrow p \cdot (1 - \alpha(r))$$

where $r \in [0, 1]$ is the fractional distance into the sponge
(0 = interior edge, 1 = domain boundary) and
$\alpha(r) = \alpha_\text{max}\,r^2$ is a quadratic ramp.  The smooth
onset avoids the spurious reflections that a hard truncation would
produce.

This is **not** a PML — it is a simpler mechanism that works well in
practice for the FDTD time-domain solver but does not provide the
theoretically reflectionless absorption of a true PML.

### PML (Helmholtz)

The frequency-domain Helmholtz solver (`solver.py`) uses a Perfectly
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
pipeline uses the 3-D inverse-square law:

$$p_\text{rms} \propto \frac{1}{r}$$

calibrated at the closest point of approach.

---

## Air Absorption

Exponential attenuation with propagation distance:

$$p(r) = p_0 \cdot e^{-\alpha_\text{air}\, r}$$

In the FDTD solver, `air_absorption` is applied per timestep as a
uniform damping coefficient.  The analytical forward model
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

The analytical 3-D forward model supports a ground plane at $z = 0$
via the image-source method.  An image source is placed at
$(x_s, y_s, -z_s)$ and the ground-reflected path is:

$$p_\text{total} = \frac{p_0}{r_d}\,e^{-\alpha r_d} + R_g\,\frac{p_0}{r_r}\,e^{-\alpha r_r}$$

where $r_d$ is the direct path length, $r_r$ is the reflected path
length, and $R_g$ is the ground reflection coefficient (typically
$R_g \approx -0.9$ for rigid ground, frequency-dependent for porous
surfaces).

---

## Drone Acoustic Signatures

`make_drone_harmonics()` generates a multi-harmonic signal
representative of multi-rotor UAVs:

$$s(t) = \sum_{k=1}^{N_h} a_k \sin(2\pi k f_0 t)$$

Default: $f_0 = 150$ Hz, $N_h = 6$, amplitudes
$[1.0,\; 0.6,\; 0.35,\; 0.2,\; 0.12,\; 0.08]$.

`make_source_propeller()` generates blade-pass noise with beat
modulation.

---

## Noise Models

| Type | Function | Physics |
|------|----------|---------|
| Wind noise | `generate_wind_noise` | Spatially correlated (exponential decay with inter-mic distance), spectrally shaped (low-pass with corner frequency), scaled to specified dB SPL |
| Sensor self-noise | `generate_sensor_noise` | White Gaussian, independent per channel, scaled to dB SPL |
| Sensor faults | `inject_sensor_faults` | Elevated noise floor, dropout (zeros), spikes (random impulses), DC offset — applied to a fraction of channels |
| Transient | `inject_transient` | Broadband impulse at a specified location and time, propagated to each receiver with delay and $1/r$ attenuation |
| Position error | `perturb_mic_positions` | Gaussian perturbation to receiver coordinates, simulating survey/placement errors |

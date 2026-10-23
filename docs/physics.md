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

Cost (`output/recordings/<recording>/build.json`, Quadro RTX 4000, float64, all
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

### Precision

What runs in which precision, and why. The FDTD fields, `C2`, the sponge and (with the k space scheme) the FFTs and the
k space kernel run in the precision the scenario's `fdtd.precision` gives (`float64` in every scenario file); the traces
are stored as float32 either way. The passive chain, the replicas and the maps run in float32 and complex64. The ECA of
the echo chain runs in double precision by default (`BistaticConfig.eca_precision = "float64"`): its normal equations
come from sums over blocks of thousands of samples whose columns (adjacent lags of a band limited reference) are nearly
collinear, and the Cholesky solve needed a relative loading of 1e-14 in double ([history](harbor_history.md#performance)),
so it cannot simply be cast to single. `eca_precision = "mixed"` does the per sample work (the reference resampled at
the track's delays, the delayed copies, the projection applied to every channel) in float32 and the fit of every channel
and block by a QR of its delayed copies in float64, with no normal equations and so no loading. On a card with float64
at 1/32 (the Quadro RTX 4000) or 1/64 (consumer cards) of the float32 rate, the double precision parts set the hardware
the code can use.

Measured on the Quadro RTX 4000 (CUDA 12.9, cuFFT 11.4, CuPy 14.2) and on the CPU. The CPU and the GPU take
different code paths (the fused kernel, the batched runner and the bodies on the device are GPU only).

**Section 1, the direct traces: pass.** The k space FDTD in float32 against float64, the propagation error pulse of
each geometry (`examples/harbor/02_forward_model/harbor_float32.py`, `output/studies/float32/fdtd.json` on the CPU,
`fdtd_gpu.json` on the GPU). Criterion, fixed before the run: the direct arrival moves by at most 0.02 ms at every
hydrophone, the traces agree within 1e-3 of each receiver's peak at the worst receiver and 1e-4 at the median.

| Recording | Arrival change, largest [ms] | Traces, worst / median receiver (CPU) | Traces, worst / median receiver (GPU) |
|---|---|---|---|
| simple | 0.0000 | 1.8e-05 / 6.6e-06 | 1.1e-04 / 5.7e-05 |
| bistatic | 0.0000 | 2.4e-04 / 1.2e-05 | 4.6e-04 / 4.7e-05 |
| multistatic_b | 0.0000 | 1.8e-04 / 1.1e-05 | 4.4e-04 / 4.4e-05 |
| harbor | 0.0000 | 1.7e-06 / 8.6e-07 | 4.4e-06 / 1.5e-06 |

**Time per step and memory per run on the Quadro** (`harbor_float32.py --bench`, `fdtd_gpu_bench.json`): one run of
the first illuminator over 2 s on each recording's grid, warmed up, `solver.run()` timed alone; the memory is the peak
of CuPy's pool (fields, FFT work area, masks, trace buffer; the desktop holds about 1.2 GB of the 8 GB card besides).
Float32 with the waits described below (float64 does not wait).

| Recording | Grid | float64 [ms per step] | float32 [ms per step] | float64 with the bodies | float32 with the bodies | Pool, float64 / float32 [MB] |
|---|---|---|---|---|---|---|
| simple | 401 x 501 | 1.42 | 0.32 | 1.43 | 0.35 | 44 / 22 |
| bistatic | 601 x 801 | 2.46 | 0.60 | 2.48 | 0.63 | 108 / 54 |
| multistatic_b | 601 x 801 | 2.46 | 0.60 | 2.51 | 0.66 | 108 / 54 (112 / 58 with the bodies) |
| harbor | 361 x 481 | 0.65 | 0.30 | 0.66 | 0.33 | 27 / 14 |

The whole build of the recordings in float32 took 51, 110, 297, 104 and 898 s of FDTD (`simple`, `bistatic`,
`multistatic`, `harbor`, `multistatic_b`) against 356, 691, 2026, 468 and 6079 s in float64
(`output/studies/float32/build32/`; the `multistatic_b` float32 build is the one whose boat runs diverged, below).
Built again with the float32 k space runs one at a time (`runs_alone`, below): 92, 155, 485, 290 and 1473 s
(`output/studies/float32/build32a/`), 2495 s for all five against 1460 s batched in float32 and 9620 s in float64.

**Runs that differ between processes.** On the 601 x 801 grids the float32 k space runs on the GPU did not always give
the same traces: in about one fresh process in three a run differed from the same run in another process, in one in
ten by a disturbance of up to 2e-2 of the trace's peak that starts at one point of the field (once 1.4 in a full
length run). Float64 was never affected; nothing in the solver's set up was the cause; more contention for the GPU made
it more frequent; making the GPU finish every operation before the next is queued (`CUDA_LAUNCH_BLOCKING=1`) removed
most of it. The fused k space step now waits for the device after every operation (`FusedFDTD.step`), which brings
the large differences to 1 run in 240 and costs about 40 % of the float32 step (bistatic 0.43 to 0.60 ms); float64,
never affected (0 in 180 runs), does not wait. The array path (`FDTDSolver._scheme_term`, used when the fused kernel is
switched off) has no waits. The cause was not found (`output/studies/float32/gpu_determinism.md`, every condition tried and its
counts).

**The scattered component: fail in the first build, pass with the runs alone.** Criterion, fixed before the run: the
scattered component within 1e-3 of its rms (the rms of the float32 less float64 difference over the rms of the float64
component, every channel and sample, as the performance work measured it), from the recordings built in both
precisions (`harbor_float32_scattered.py`, `scattered.md`). In the first float32 build every component of `simple`,
`bistatic`, `multistatic` and `harbor` and the ferry's and the tug's of `multistatic_b` were within 3.3e-4, but the
boat's two runs on `multistatic_b`, run together as one batch, diverged (the field grew without bound after 15 to
25 s) in the build, when built again and in every further batched run: **fail**.

The cause of that divergence is the batched float32 inverse FFT (`output/studies/float32/batched_divergence.md`): on
the Quadro, `irfft2` over a batch of fields sometimes returns a corrupted field (19 in 100 000 calls of the k space
pair on the 601 x 801 grid; none in float64 and none on a 2D field), in cuFFT 11.4 and in the cuFFT of CUDA 13 alike.
Each corrupted step changes the whole field at once; the scheme damps the jump, and repeated jumps carried the boat's
runs to 1e16. Why the batched transform fails on this card was not found. The scheme itself is as stable in float32
as in float64 with the harbor's speed map and land mask: a broadband burst decays at the float64 rate in float32. So
`simulate_sources` now runs float32 k space jobs one at a time (`runs_alone` in `water/fdtd_runner.py`; float64 and
the leapfrog still batch). Run alone, the boat's runs stayed within 3.0e-5 of float64 over the full 60 s, and the five
recordings built again that way **pass**: every scattered component within 2.9e-5 to 2.29e-4 (the boat's on
`multistatic_b` 1.22e-4), every direct component within 1.6e-5 to 5.8e-5, the `harbor` noise free sum within 4.0e-6.
The runs alone still carry the rarer fault of the runs that differ between processes (above): a burst left to decay
in float32 met it three and four times in 20 s, as jumps of up to 4e-5 of the burst, while the four 60 s boat runs
alone met it not once; a float32 recording is therefore still compared with float64. The criterion now passes on
every component, which is what a switch of the default needs; the switch was not made: `precision` stays float64 in
every scenario file, the recordings were left at their float64 build, `scripts/reproduce.py` was not run and no number
of `docs/harbor.md` moved. Whether to switch is the repository owner's decision.

**Section 2, the ECA in mixed precision: pass.** The ECA in mixed precision against double on the stage 0a cases
(`harbor_cancellation_depth.py --eca-precision`, the gathers of `output/studies/direct_path/cache/`, the processing
config's `joint` reference and the beam, `output/studies/float32/stage0a_<precision>/`, the table by
`harbor_eca_precision.py`). Criterion, fixed before the run: the cancellation depth within 0.3 dB of the double run:
**pass** on the CPU, the largest difference 7e-8 dB, and on the GPU, 1.7e-7 dB (`eca_stage0a_gpu.md`). The time is the
reference and the ECA per second of data; the QR in double is slower than the loaded Cholesky it replaces on the CPU
and **on the GPU as well**, 3.8 to 4.9 times on the Quadro, so the mixed ECA buys no speed on this card:

| Case | Reference | Depth, double [dB] | Depth, mixed [dB] | Difference [dB] | s per s of data, CPU, double / mixed | GPU, double / mixed |
|---|---|---|---|---|---|---|
| stationary 193 m | joint | 25.17 | 25.17 | -0.000 | 1.89 / 5.98 | 0.48 / 1.83 |
| stationary 193 m | beam | 20.14 | 20.14 | -0.000 | 0.96 / 3.07 | 0.23 / 0.88 |
| moving 193 m | joint | 26.50 | 26.50 | -0.000 | 1.68 / 5.96 | 0.37 / 1.75 |
| moving 193 m | beam | 24.06 | 24.06 | -0.000 | 0.86 / 3.10 | 0.23 / 0.88 |
| stationary 99 m | joint | 23.93 | 23.93 | -0.000 | 1.75 / 6.04 | 0.36 / 1.75 |
| stationary 99 m | beam | 22.81 | 22.81 | -0.000 | 0.89 / 3.12 | 0.23 / 0.88 |
| moving 99 m | joint | 22.57 | 22.57 | -0.000 | 1.72 / 6.25 | 0.36 / 1.75 |
| moving 99 m | beam | 24.30 | 24.30 | +0.000 | 0.85 / 3.09 | 0.23 / 0.88 |
| stationary 29 m | joint | 28.07 | 28.07 | +0.000 | 1.65 / 6.33 | 0.36 / 1.76 |
| stationary 29 m | beam | 11.33 | 11.33 | -0.000 | 0.85 / 3.15 | 0.23 / 0.88 |
| moving 29 m | joint | 9.94 | 9.94 | +0.000 | 1.68 / 6.45 | 0.37 / 1.76 |
| moving 29 m | beam | 7.45 | 7.45 | -0.000 | 0.87 / 3.32 | 0.23 / 0.88 |

On the recordings (the system on the `simple`, `bistatic` and `multistatic` tuning recordings with each precision, the
truth's direct and scattered components through the same projections, `harbor_direct_path_depth.py --precision`,
`output/studies/float32/eca_recordings.json`): the direct path removal within 2e-7 dB of the double run for every
illuminator, the echo loss within 6e-8 dB, the echo map's peaks identical in all 213 frames compared: **pass**
(criteria 0.3 dB, 0.1 dB, one cell; on the CPU; on the GPU within 2e-7 dB, the peaks identical in all 213
frames, `eca_gpu_<precision>.json`). The echo chain's time per second of data, double to mixed: on the CPU 26.8 to 27.7
(`simple`), 55.4 to 61.6 (`bistatic`), 60.9 to 73.9 (`multistatic`); on the GPU 0.69 to 1.83, 0.71 to 2.39 and 1.51 to
5.43.

| Recording | Component | Double [dB] | Mixed [dB] | Difference [dB] |
|---|---|---|---|---|
| simple | direct path of ferry | 16.680 | 16.680 | +0.0000 |
| simple | echo loss, ferry>bodies | -0.689 | -0.689 | +0.0000 |
| simple | direct path of boat | 9.409 | 9.409 | -0.0000 |
| simple | echo loss, boat>bodies | -0.703 | -0.703 | +0.0000 |
| bistatic | direct path of ferry | 26.005 | 26.005 | +0.0000 |
| bistatic | echo loss, ferry>uuv1 | 0.353 | 0.353 | -0.0000 |
| multistatic | direct path of ferry | 19.291 | 19.291 | +0.0000 |
| multistatic | echo loss, ferry>bodies | 0.648 | 0.648 | -0.0000 |
| multistatic | direct path of tug | 13.287 | 13.287 | +0.0000 |
| multistatic | echo loss, tug>bodies | 0.528 | 0.528 | +0.0000 |
| multistatic | direct path of boat | 13.992 | 13.992 | -0.0000 |
| multistatic | echo loss, boat>bodies | 1.367 | 1.367 | -0.0000 |

What stays: `precision` is float64 in every scenario file (the scattered criterion passes with the runs alone, but
the switch is left to the repository owner) and `eca_precision` is
`float64` in every processing config (the prompt leaves processing parameters unchanged; `"mixed"` is available).

The figures of the float32 study (`output/studies/float32/`) are not shown on this page and are no longer committed; they are in git history at commit `1164bb4`, and the study's JSON and write ups stay.

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

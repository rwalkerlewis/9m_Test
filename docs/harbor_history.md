# Harbor: history

What the harbor work found before stage 2, kept for the record: the scenario details that only the history scripts
use, the batch chains with AIS in the loop, stage 0 and stage 1 of the multistatic work, the performance work and the
findings tables.  The system as it stands, how to run it and its current numbers are in [Harbor](harbor.md).  The
scripts these sections cite are in `examples/harbor/90_history/` (still runnable; they produced the committed figures
quoted here) unless they are part of the system.  Every number below was measured by the script cited with it, at the
time; [scripts/reproduce.py](../scripts/reproduce.py) does not regenerate them.

The outputs of the history (the figures, `results.json` files and logs of the batch chains, stage 0 and stage 1, the
performance work: `output/harbor_demo/`, `harbor_domain/`, `harbor_domain_video/`, `harbor_shore0.5/`,
`harbor_shore1.0/`, `harbor_cancel/`, `harbor_m1_uuv/`, `harbor_presentation/`, `harbor_picture/`, `performance/`,
`harbor_detect/`, `harbor_stream/`, the `ablation`, `ablation_freefield`, `demo`, `eval`, `coarse` folders and the
budget figures of `harbor_bistatic/`, the `coarse` folder of `harbor_multistatic/`, and the `m1`, `m2`, `m3` figures
and `m3_results.json` of the 5 m harbor in `output/harbor/`) were removed from `output/` when it was restructured
(`docs/outputs.md`); every one of them is in git history at commit `fd12458` under the path this page gives
(`git show fd12458:output/harbor_cancel/results.json`, `git checkout fd12458 -- output/harbor_demo`). Run again, a
history script writes to `output/studies/history/<script>/`; every figure it writes now has one panel.

## The default scenarios

| Item | Default |
|------|---------|
| Domain | 2400 m x 1800 m, 5 m grid, c is approximately 1492-1500 m/s (effective speed varies with shore distance) |
| Land | Pressure-release walls (opaque, thin breakwaters block sound) plus a lossy layer in the water next to the shore (`shore_loss_width`, `shore_loss_max`; approximate amplitude reflection R ~ 0.5 in the default runner configuration) |
| Sensors | One cable (10 m channel spacing, multiple bends) and 4 hydrophones. The default cable is pressure mode; scenario files can select strain mode and a gauge length. |
| Vessels/bodies | AIS ferry, tug and moored ship; non-AIS `uuv1`, `uuv2`, and diver; stationary `static_scatter` body. See `examples/configs/harbor.scenario.json`. |
| Noise | Colored ambient, three distant-shipping plane waves, DAS channel noise, common-mode noise, 8.6 % gain scatter, 2 % dead channels |
| Band | 12-78 Hz at 600 Hz sampling |

Source levels are calibrated free-field rms pressure at 100 m [Pa]. The passive detector's
default band is 12-78 Hz; the actual source spectrum is limited by grid spacing and sampling.
UUV signatures are sparse tonal signals with little broadband energy; ship signatures are
broadband with speed-dependent level and tonal components.

The bistatic example uses a **different, cropped fine-grid scenario** in
`examples/configs/harbor_bistatic.scenario.json`: it sets a 0.5 m grid, 6 kHz sample rate, strain-mode
DAS with 2 m channel/gauge spacing, a ferry illuminator and a silent 1 m × 3 m UUV body. Its
`--coarse` option changes to 1 m spacing and scales the sample rate/band. Do not infer its
settings or resource cost from the default 5 m / 600 Hz harbor scenario.

## The batch passive chain with AIS in the loop

1. **Pre-process:** channel QC, per-cable common-mode removal, Hann STFT, per-channel noise floor
   (time median, then a running median across frequency) and whitening.
2. **AIS-aware suppression** (optional, `run_ais_aware`): predict each reporting ship, refine its position
   with a fine Bartlett search, project its replica patch (or the data-adaptive subspace) out,
   and **re-estimate the noise floor from the cleaned data and re-whiten**.  Skipping the
   re-whitening leaves the floor inflated by ship energy and hides the target (it cost a factor
   of ~200 in the matched output at the UUV's tone).
3. **Matched-field scan:** free-field 2-D Green's function replicas over a grid, with a
   per-bin stationary-interference projection and a sparse spectral sum.  An optional
   line-enhancement step subtracts a running median across bins.
4. **Extraction:** non-maximum suppression and successive cancellation (sidelobes of an
   accepted peak are projected out before the next candidate is scored).
5. **Occupancy filter:** diffuse, multiply by a likelihood from each frame's map, repeat.

Without step 2 (`harbor_m3_detect.py`, or `harbor_m4_dlc.py --no-suppress`) the loud ships hide
the UUV.

## The batch bistatic chain (bistatic echo localisation)

The bistatic path treats a reporting ship as an illuminator and searches for a body from its
scattered return at the DAS/hydrophone receivers. It is distinct from passive UUV emission
detection: the target can be silent, and the processor must separate the ship's direct path
from its echo.

`bistatic.py` performs a band-limited STFT, refines the ship location from AIS, obtains a
per-frame illuminator reference (`reference="data"` or, given from the truth, `"true_waveform"`), optionally
removes the direct ship signal, then forms a scan map. Supported map processors are `bartlett`,
`bartlett_phat`, `coherent`, `mvdr`, and `music`. Direct-path removal choices include `eca`,
`patch`, `adaptive`, `patch+adaptive`, and `none`. The method and reference options are not
interchangeable performance guarantees; they are algorithm variants to compare in simulation.

Replicas are evaluated at the layout's actual field sample points. `SensorLayout.collapse`
converts the three samples per strain channel into the second difference measured by that
channel, so the strain response is included in the bistatic steering model.

The runnable FDTD bistatic demo is intentionally given information from the truth to isolate the
echo localization question: it subtracts the known direct ferry component, uses the true ferry
waveform and exact track, and measures the scattered component. Its figures and output state
these assumptions. This is a controlled upper-bound/algorithm demonstration, **not** an end-to-end
detection result with everything estimated from the traces. For a less optimistic view, compare it with the empirical
`reference="data"` and direct-path-removal variants in `bistatic.py`, or use the analytic
prototype as the known-replica reference.

The `harbor_bistatic_budget.py` script is analytical only (no FDTD): it compares idealized
soft/rigid infinite-cylinder scattering with a simplified link budget before launching an
expensive simulation. The rigid-cylinder case belongs to that theoretical budget and is not
implemented as a rigid-body FDTD boundary. It writes `budget.json`, `budget.png` (required
direct-path cancellation along the track for each grid, ship spectrum and cylinder model; best
receiver solid, median dashed, unclipped so values below 0 dB show remaining headroom) and
`budget_geometry.png` (plan view of the mid-track paths and the direct, echo and noise band
levels along the DAS cable). `--n-fig` sets how many track positions the curves use.

## GPU (at the time of the performance work)

The FDTD runs on the GPU by default (`HarborFDTDConfig.use_cuda` and `FDTDConfig.use_cuda` are `True`,
the scenario files say so too, and the scripts that run it take `--cpu` to opt out; without a GPU it
falls back to the CPU with a warning).  `ProcessingConfig(use_cuda=True)`, `OccupancyConfig(use_cuda=True)`
and `BistaticConfig(use_cuda=True)` run the processing on the GPU (CuPy, `pip install cupy-cuda12x`; the
scripts' `--cuda`).  The passive chain runs in single precision on the GPU; the bistatic
chain and the FDTD run in double precision (`HarborFDTDConfig.precision = "float32"` is opt in).
CPU and GPU agree: the passive scan maps correlate at 1.000000, the FDTD traces of the fused kernel
are identical to the array code to the last bit, and the bistatic maps agree to 1e-12 (the ECA to
1e-8, see [Performance](#performance)).  Measured on a Quadro RTX 4000 (8 GB, Turing, float64 at
1/32 of the float32 rate) with the fused kernel (the default on the GPU since the performance work):
a 20 s fine grid vessel run (0.5 m, 6 kHz, 120 000 steps, 481 000 cells) takes 11 s alone and 17 s
with ten bodies, against 3 min and 8 min with the array code and about 8 min on the CPU; the
`all_estimated+doppler+smooth4` bistatic configuration on the fine dataset takes 54 s on the GPU in the
final benchmark run (`final.json`; 47 s in the separate `stage5_gpu.json` run) against 522 s for
the original CPU code.  These are measurements on that specific GPU and scenario, not
portable runtime guarantees; unavailable CUDA hardware falls back to the CPU with a warning.

## Performance

Branch `performance` (October 2026).  Goal: an order of magnitude less compute for the harbor
forward model and the processing chain with no change of results beyond stated tolerances.  Every
change is measured twice by `scripts/bench_harbor.py`: wall time against the original code and
accuracy against the outputs the original code produced (saved once, gitignored).  All numbers below are on the Quadro RTX 4000 (`output/performance/*.json` at commit `fd12458`; the baseline is
`baseline.json`, from commit `ca42d55`). The refreshed final run used Python 3.13.13 / NumPy 2.4.3;
the original baseline used Python 3.12.3 / NumPy 2.5.4, so the end-to-end ratio is indicative, not
a strictly controlled same-interpreter comparison. The stage 0 cases:
the fine bistatic scenario (0.5 m, 6 kHz, 801 x 601 cells), the moving ferry alone for 2 and 10 s
and with the 1 m x 3 m UUV body, the multistatic scene (ferry with all ten bodies) for 2 s; the
`direct_removed_exactly`, `true_waveform_exact_track+doppler` and `all_estimated+doppler+smooth4` configurations of
`harbor_bistatic_ablation.py` on the fine dataset (79 frames); `eca_stage` on the stage 0a 193 m
moving case with fixed lags and with per sample delays (4 s track fit).

### Timing, before and after

Forward model (one run; `ms/step` in brackets):

| Case | original (array code, GPU; `baseline.json`) | fused float64, host masks (`stage1_fused64.json`) | defaults now: fused float64, device masks (`final.json`) | fused float32, host masks (`stage1_fused32.json`) |
|---|---|---|---|---|
| ferry alone, 2 s | 20.7 s (1.72) | 1.26 s (0.105) | 1.81 s (0.151) | 0.64 s (0.053) |
| ferry alone, 10 s | 93.2 s (1.55) | 5.6 s (0.093) | 8.62 s (0.144) | 2.3 s (0.038) |
| ferry + UUV body, 2 s | 25.0 s (2.08) | 2.7 s (0.225) | 1.98 s (0.165) | 3.0 s (0.254) |
| ferry + UUV body, 10 s | 119.9 s (2.00) | 19.4 s (0.324) | 9.56 s (0.159) | 20.2 s (0.337) |
| multistatic, 10 bodies, 2 s | 49.1 s (4.09) | 32.2 s (2.68) | 2.60 s (0.217) | 31.8 s (2.65) |

The float32 column still uses the host masks for the bodies, which is why its body cases are no
faster.  The stage 2 run itself (`stage2_device.json`) measured the device masks faster than the
final run on four of five cases (for example 1.21 s and 1.68 s for the 2 s ferry and multistatic
cases); the table reports the final run.  Time per step against body count (`fdtd_sweeps.json`, 2 s runs): array code 1.82 / 1.54 /
4.57 ms for 0 / 1 / 10 bodies, fused kernel with host masks 0.10 / 0.25 / 2.82 ms, fused kernel with
device masks 0.10 / 0.10 (one outlier run at 0.60) / 0.14 ms.  The fused float64 step on the fine
grid moves about 23 MB through memory in 0.1 ms, close to the card's bandwidth; the remaining cost
of a 20 s run with bodies is the second launch per step for the masks.

Batched runs (stage 3, `fdtd_sweeps.json`, B copies of the 2 s ferry run in one
`HarborFDTDBatch`): 3150, 3080, 3220, 3310, 3180 and 2570 runs per hour for B = 1, 2, 4, 8, 16, 32
(the free memory allowed 32).  Every batched run equals its single run to the last bit.  On this
grid batching does not raise throughput, because the single run is already bandwidth bound rather
than launch bound; it removes the Python launch overhead for small grids and it is what
`render_vessel_gathers` and the stage 0a script now use (the two 20 s runs of the fine dataset
rebuild in 26 s together).

Processing (`run_bistatic` wall time, and `eca_stage` alone):

| Case | original (CPU) | vectorized, CPU | vectorized, GPU (`use_cuda=True`) |
|---|---|---|---|
| `direct_removed_exactly` | 364 s | 230 s | 15.5 s |
| `true_waveform_exact_track+doppler` | 370 s | 294 s | 102.0 s |
| `all_estimated+doppler+smooth4` (dispersion corrected replicas) | 522 s | 394 s | 54.1 s |
| `all_estimated+doppler+smooth4`, free field replicas (everything estimated) | — | — | 35.3 s (not a benchmark run: shared GPU, `ablation_freefield/smooth4/results.json`) |
| `eca_stage`, 193 m moving, fixed lags | 22.2 s | 22.4 s | 15.1 s |
| `eca_stage`, 193 m moving, per sample delays | 28.7 s | 25.0 s | 14.4 s |
| `ship_positions` refinement, 193 m case (39 frames) | 98 s | 70 s | 3.1 s |

On the CPU the vectorization removes redundant work (the scan grid geometry was recomputed for
every bin because the per frame path calls evicted the single entry cache; the replicas of all
bins of a refinement frame now come from one Green's function call) but not the arithmetic, which
is the Green's function evaluation and the matrix products.  On the GPU the ECA is bound by the
float64 matrix products of the normal equations (about 1.4 Tflop per 20 s run at 1/32 rate), which
is why `eca_stage` gains less than the maps; a Toeplitz form of the normal equations would remove
that cost and is left for later.

Total on the stage 0 benchmark (all cases, forward model and processing): 1615 s with the original
code against 226 s with the corrected defaults on the GPU (`final.json`), a measured ratio of 7.2.
This does not reach the order-of-magnitude goal. The true-waveform Doppler accuracy fallback and the
interpreter mismatch above both affect this total; establish a same-interpreter baseline before
using this ratio as a deployment estimate.

### Accuracy and criteria

| Stage | Criterion | Result | Pass |
|---|---|---|---|
| 1 | fused float64 traces within 1e-12 relative per channel of the array code | identical (0.0) on four cases, 2.1e-13 max on the ten body case | yes |
| 1 | fused float32 direct traces within 1e-4 (median) and 1e-3 (max) of float64 | median 3.2e-5 to 3.7e-5; max 1.03e-3 (ferry 2 s), 6.6e-4, 6.2e-4, 6.4e-4, 7.0e-4 | **no** (max, one case, by 3 %) |
| 1 | fused float32 scattered field within 1e-3 of its rms | 6.7e-4, 5.8e-4, 6.6e-4 | yes |
| 1 | fused float32 stage 0a ECA depths within 0.5 dB | 0.00 dB (median), 0.0008 dB max per channel | yes |
| 1 | fused float32 `all_estimated+doppler+smooth4` (dispersion corrected replicas) estimates within one map cell in 95 % of frames | 100 % identical | yes |
| 2 | device masks within 1e-12 of `body_factor`; scattered traces within 1e-10 of the stage 1 run | masks 1.1e-16 (tests); traces 3.7e-13 max | yes |
| 3 | batched runs within 1e-12 of single runs | identical (0.0) for B up to 32 | yes |
| 4 | a CPML thickness at least 10 dB below the sponge in every band and angle | 15 cells, cubic profile: 20.5 dB (peak), 16.5 dB (worst band) margin | yes |
| 5 | vectorized outputs within 1e-10 of the original code, float64 | CPU: maps identical (3e-18 to 1e-15), ECA identical (0.0); GPU: maps 5e-14, 4.7e-14 and 4e-13, ECA 2e-14 (fixed lags) and 2e-12 (per sample delays) | **yes** |
| 6 | a 1 m scheme within 0.1 rad at 400 to 600 m over 100 to 760 Hz, wall error no worse than the 0.5 m scheme | k-space: 0.50 rad (axis), 1.05 rad (diagonal) over the full band; exact (0.001 rad) over 100 to 380 Hz | **no** |

Float32 fails the direct trace criterion on one of five cases by 3 % of the limit, so float64 stays
the default (`precision="float32"` is available and is 2 to 2.5 times faster per step without
bodies; its ECA depths and position estimates on the stage 0 cases are unchanged).

The GPU ECA's batched normal equations can be ill conditioned for the true waveform with per-sample
Doppler delays: with loading of 1e-14, cuSOLVER and LAPACK can produce materially different
projections. That specific configuration now uses the vectorized NumPy/SciPy solve while the rest
of the chain remains on the GPU; its map agrees to 4.7e-14. The fallback raises that benchmark
configuration from 41.2 s to about 102 s in the refreshed full run (102.0 s in `final.json`; 108.8 s
in the separate `stage5_gpu.json` rerun), still below its 370 s baseline. Other GPU ECA cases remain
batched on device: with the data reference (the configurations with everything estimated) cleaned traces agree to
2e-14 (fixed lags) and 2e-12 (per-sample delays), with no change of ECA depth or estimate. On the
CPU, the vectorized ECA keeps SciPy's factorisation and reference call sequence per channel and is
identical to the last bit on every stage 0 case.

CPML (`cpml_reflection.json`, open water, source and receivers 40 m from the edge, 0.5 m grid,
peak of the reflected field re the incident, dB; incidence 0, 27, 45, 63, 76, 82 degrees):

| Boundary | 0 | 27 | 45 | 63 | 76 | 82 | worst band, worst angle | ms/step (3.9 M cells) |
|---|---|---|---|---|---|---|---|---|
| sponge, 60 cells (current) | -76 | -51 | -35 | -19 | -8 | -4.5 | -3.0 | 0.74 |
| CPML 10 cells, quadratic | -70 | -47 | -46 | -45 | -45 | -26 | -24 | 1.06 |
| CPML 10 cells, cubic | -87 | -64 | -61 | -57 | -47 | -28 | -26 | 1.04 |
| CPML 15 cells, cubic | -101 | -81 | -82 | -76 | -43 | -25 | -23 | 1.04 |
| CPML 20 cells, cubic | -111 | -91 | -92 | -75 | -39 | -23 | -21 | 1.04 |
| CPML 30 cells, cubic | -124 | -105 | -104 | -66 | -35 | -21 | -19 | 1.08 |

The current sponge is nearly transparent at grazing incidence (its reflection at 82 degrees is
-4.5 dB), which the stage 0a results never saw because the sources sat in the middle of the crop.
The CPML costs about 40 % more per step (the memory variables of the layer cells) and thins the
boundary from 60 to 15 cells: on the fine harbor grid that is 601 x 801 against 511 x 711 cells, a
quarter of the cells, if the scenario were laid out for it.  The complex frequency shift
(`cpml_alpha_max`) made no difference here.  The sponge stays the default: switching changes the
forward model's output (the spurious reflections it removes are in every existing dataset and
cache), so the change belongs to a scenario revision, with `HarborFDTDConfig(boundary="cpml",
cpml_width=15, cpml_order=3)` recommended for new scenarios (the defaults when `boundary="cpml"` is
set alone are `cpml_width=15`, `cpml_order=2`).

Time stepping (`timestep_schemes.json`, `scripts/bench_timestep.py`; stationary source in open
water, receivers on grid nodes at 100 to 600 m on the axis and the diagonal, time gated, phase of
the transfer function over the analytic `H0^(2)(k r)` referred to the 100 m receiver; wall: a
pressure release half plane on a node line, reflected over direct against the image solution):

| Scheme | grid | ms/step | phase error 400 to 600 m, 100 to 760 Hz (axis / diagonal) | same, 100 to 380 Hz | wall, 100 to 760 Hz | wall, 100 to 380 Hz | cost per simulated second |
|---|---|---|---|---|---|---|---|
| leapfrog, free field k | 0.5 m | 0.81 | > pi / > pi | > pi / > pi | > pi | 0.68 rad | 4.8 s |
| leapfrog, dispersion corrected (current processing) | 0.5 m | 0.81 | 0.001 / 0.048 rad | 0.000 / 0.002 rad | (same run) | (same run) | 4.8 s |
| lw4 | 0.5 m | 1.37 | > pi / > pi | 1.7 / 0.25 rad | > pi | 0.18 rad | 8.2 s |
| k-space | 0.5 m | 8.3 | 0.000 / 0.000 rad | 0.000 / 0.000 rad | 0.31 rad | 0.15 rad | 50 s |
| leapfrog, corrected | 1 m | 0.20 | > pi | 0.018 / 0.036 rad | > pi | 2.6 rad | 0.6 s |
| lw4 | 1 m | 0.35 | > pi | > pi | > pi | 3.1 rad | 1.1 s |
| k-space | 1 m | 1.78 | 0.50 / 1.05 rad | 0.000 / 0.001 rad | 3.0 rad | 0.31 rad | 5.3 s |

The k-space update is exact where the grid resolves the wave: on the 1 m grid it is exact up to
380 Hz (four cells per wavelength) and its wall error there, 0.31 rad at every normal receiver,
is an effective wall position a tenth of a cell beyond the land node line, half the 0.5 m
leapfrog's error in that band.  Up to 760 Hz the 1 m grid has two cells per wavelength and no
scheme meets the criterion: the k-space propagation is still exact, but a point source and a point
receiver on such a grid are not the continuum's (the diagonal error is a constant 1.04 rad at every
range), and the leapfrog's wall numbers over the full band are its own dispersion over the extra
path.  The fourth order time stepping does not help at these resolutions: the spatial stencil
error dominates, and it costs 1.7 times the leapfrog step.  The k-space step costs ten times the
fused leapfrog step on the same grid (two float64 FFTs on a card with 1/32 rate float64), so the
1 m k-space run costs 1.1 times the current 0.5 m run per simulated second.  Both schemes stay opt
in (`time_scheme`); the leapfrog remains the default.

### Options and defaults

| Option | Default now | Changed | Note |
|---|---|---|---|
| `FDTDConfig.kernel`, `HarborFDTDConfig.kernel` | `"auto"` (fused on the GPU, array code on the CPU and under MPI) | yes, from `"reference"` | identical results; not in the cache key |
| `HarborFDTDConfig.body_masks` | `"auto"` (device masks with the fused kernel) | new | 1e-13; not in the cache key |
| `FDTDConfig.precision` | `"float64"` | no | `"float32"` opt in (criterion missed by 3 % on one case); in the cache key |
| `FDTDConfig.boundary` | `"sponge"` | no | `"cpml"` opt in, criterion passed; in the cache key when set |
| `FDTDConfig.time_scheme` | `"leapfrog"` | no | `"lw4"`, `"kspace"` prototypes, criterion failed |
| `BistaticConfig.impl` | `"vectorized"` | new | `"reference"` keeps the original loops |
| `BistaticConfig.use_cuda` | `False` | new | GPU chain, double precision; ECA within 2e-12 with the data reference, differs with the true waveform |
| `simulate_sources`, `HarborFDTDBatch` | used by `render_vessel_gathers` and the stage 0a script when the fused kernel is available | new | identical results, same cache keys |

The cache key of a gather ignores `kernel`, `body_masks` and `use_cuda` and includes `precision`,
`boundary` and the CPML and time scheme fields only when they differ from their defaults, so every
cache built before this work stays valid.

## Findings before stage 2

### What the chains found before stage 2

At the time `harbor_detect.py` ran the batch chains (`run_ais_aware`, `run_bistatic` with an AIS ship as illuminator) and
`harbor_stream.py` read its parameters from dictionaries in the script; both now run the one chain of stage 2
(`water/stream.py`) with the processing configs.  Tracks were scored only if confirmed at the end of the record (stage 2
scores every track that was ever confirmed), and the "found" columns are fractions of frames with a map peak within
5 m, not track Pd.

Detect once, check every detection against the true positions (`examples/harbor/05_detection/harbor_detect.py`,
everything estimated from the traces, free field replicas, the 3 strongest map peaks per frame as
detections, a hit within 5 m of an object; `output/harbor_detect/<dataset>/` at commit `fd12458`). The seed 0 mix of
each dataset.

| Dataset | Chain | Found (frames with a detection within 5 m, median error) | Not found | Detections > 10 m from every object | Time |
|---|---|---|---|---|---|
| bistatic (0.5 m, 20 s) | echo, ferry illuminates | silent UUV 75 %, 2.5 m | — | 1.7 per frame | 27 s |
| multistatic (0.5 m, 20 s) | passive | boat (no AIS) 57 %, 1.7 m; ferry 16 %, 4.1 m | tug | 1.8 per frame | 13 s |
| multistatic | echo, ferry illuminates | — (tug 23 %: its direct path dominates the map) | both UUVs, diver, six pilings, moored hull | 2.0 per frame | 27 s |
| harbor (5 m, 120 s) | passive | ferry 2 %, tug 3 % | moored ship, both UUVs, diver | 2.9 per frame | 9 s |

Not detected by these batch runs (one illuminator, `harbor_detect.py`): the second UUV, the diver, the
pilings, the moored hull, the radiating UUVs of the 5 m dataset, and every silent body of the multistatic
recording, where the direct paths of the ships that do not illuminate are left in the traces and swamp the echo
map. The real-time chain below removes every tracked ship's direct path and, on the same multistatic
recording, finds uuv2 in 18 % and uuv1 in 3 % of the frames.

**Real time, located from the traces alone** (`water/streaming.py`, `examples/harbor/05_detection/harbor_stream.py`). AIS
does not enter the processing: it is used afterwards, to label what was located (`ais_crosscheck`).
The traces arrive in 0.25 s chunks:

1. `StreamingPassive` (no AIS) locates the radiating objects, ships included: the 5 strongest peaks of
   every frame. It is the passive chain made causal: channel QC, noise floors and the stationary
   interference projector over the last seconds only.
2. `RadiatorTracker` turns the peaks into causal tracks (gate, confirmation after 6 fixes in 2 s, a track
   within 30 m of a stronger one is a sidelobe and is not confirmed).
3. `StreamingEcho` takes the tracks scoring at least a quarter of the strongest (`RadiatorTracker.strong`:
   the passive chain also locates the echo of a loud ship off a body, about 10 dB below the ship, and that
   echo must not be removed as a radiator; set on the bistatic and multistatic recordings), removes the
   direct path of each of them in turn by ECA with per sample delays (each with its own reference beam,
   its track fitted over the past 4 s and extrapolated), and maps the echoes with every one of them as an
   illuminator (the coherent `"snr"` maps summed). No quay image lags (`eca_images=False`, measured on the
   bistatic recording: 77 % of frames within 5 m of the UUV without them, 73 % with them, at half the ECA
   cost).
4. The AIS reports label the tracks afterwards.

On the Quadro RTX 4000, checked against the true positions (a detection within 5 m of an object):

| Recording | Compute per s of data (passive + echo) | Passive (no AIS) | Tracks (AIS label afterwards) | False tracks | Echo |
|---|---|---|---|---|---|
| simple (1 m, 20 s) | 0.62 s | ferry 100 %, boat 78 % | ferry 1.4 m (AIS: ferry), boat 0.9 m (no AIS) | 1: r19 at (205, 227), 18 m from the wreck | wreck 100 %, UUV 18 % |
| bistatic (0.5 m, 20 s) | 0.78 s | ferry 100 %, the UUV's echo 24 % | ferry 1.0 m (AIS: ferry) | 1: r18 at (163, 228), 41 m from the ferry | silent UUV 73 % |
| multistatic (0.5 m, 20 s) | 0.95 s | ferry 97 %, tug 69 %, boat 78 % | boat 2.6 m (no AIS), ferry 2.2 m (AIS: ferry), tug 2.3 m (AIS: tug) | 0 | uuv2 18 %, uuv1 3 %, the other bodies under 2 % |
| harbor (5 m, 120 s) | 0.50 s (passive only) | ferry 50 %, tug 2 % | ferry 13 m (AIS: ferry), tug 8 m (AIS: tug) | 0 | — |

Every confirmed track is scored: matched to the object nearest over its fixes (median distance) when within
10 m (20 m on the 5 m harbor), otherwise a false track (`summary.json`, `false_tracks`; drawn dashed and listed
in step 7 of the walkthrough). Both false tracks are sidelobes of the ferry that move with it (r18 stays 41 m
from it; r19 is described with the simple harbor above). Compute times vary by a few hundredths between runs.

Latency: about 1.7 s on the simple harbor and 1.9 s on the strain recordings (2 s steps), 1 s on the 5 m one.
What made it fit: free field replicas from fused GPU kernels (phase recurrence, `green2d_bins`, within 1e-14;
the passive strain replicas from the same recurrence at the three sample points of each channel, combined in
the kernel; the echo chain's strain channel replicas from the range table, within 7e-6), the interference eigendecompositions batched and spread over
blocks, the ECA on the newest blocks only (`eca_remove(blocks=...)`), its normal equations from prefix sums
(`eca_gram="prefix"`, equal to 1e-15) for all channels at once (`eca_chunk=200`), every illuminator's map in
one pass. The batch coherent map uses the same replicas: 4.9 s instead of 9.2 s on the fine bistatic dataset.

The passive replicas (scan, streaming chain, AIS ship refinement and projection) are the Green's function at
the layout's sample points collapsed to channels, as the echo chain's: on a strain cable the data are the
second difference over the gauge length. Until this review they were the Green's function at the channel
centres, a mismatch on the bistatic and multistatic recordings. With the matched replica the map at every
radiator doubles (multistatic seed 0, median over frames within 3 m of the truth: ferry 1472 to 3286, tug 91
to 230, boat 1574 to 3090), and so do the frame maxima. The strongest ship's sidelobes then take more of the
peaks of a frame, so the weaker objects are found less often: tug 85 to 69 %, boat 86 to 78 %, the UUV's echo
on the bistatic recording 58 to 24 % (passive), and in the batch passive chain (`harbor_detect.py`, where the
AIS ship refinement uses the same replicas) the boat 77 to 57 %. The passive compute went from
0.34 to 0.45 s per second of data.

Removing the other ships' direct paths is what lets the echoes through in the multi ship scene: with the
tug's and the boat's direct paths removed exactly (from the truth, a diagnostic) the ferry lit map finds
uuv1 in 39 % and uuv2 in 33 % of the frames; estimated, the removal reaches 7 dB (tug) and 3 dB (boat)
against 20 dB given both the true waveform and the exact track, which is where the remaining gap is.

**Scoreboard (stage 1, partial).** `water/scoreboard.py` implements the definitions of
[Architecture, Scoreboard](architecture.md#scoreboard) and `examples/harbor/90_history/harbor_picture_stage1.py`
runs it (tuning set seed 0, test set seeds 1 to 5). The run was stopped after the bistatic dataset
and the multistatic passive chain (`output/harbor_picture/stage1/results.json` at commit `fd12458`); with the threshold
at 5 % of object free frames:

| Dataset / chain | Configuration | Pd tuning | Pd test | Frames with a false alarm, tuning / test (target 5 %) |
|---|---|---|---|---|
| bistatic / echo | `all_estimated` | uuv1 0.27 | uuv1 0.30 | 4 % / 6 % |
| bistatic / echo | `true_waveform_exact_track` (given the truth) | uuv1 0.03 | uuv1 0.02 | 5 % / 3 % |
| bistatic / echo | `direct_removed_exactly` (given the truth) | uuv1 0.37 | uuv1 0.51 | 15 % / 18 % |
| multistatic / passive (scan map) | `all_estimated` | ferry 0.01, tug 0.00, boat 0.16 | ferry 0.01, tug 0.00, boat 0.08 | 16 % / 7 % |
| multistatic / passive (occupancy map) | `all_estimated` | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 1 % / 3 % |

**Which scoreboards are calibrated.** The threshold is set on runs with one object removed from the mix, at
5 % of their frames; the last column is the rate it actually gives on the full scene (`false_alarm_frame_rate`
in the JSON: frames with at least one false alarm). Only the two bistatic rows with the ECA hold the 5 %
(`all_estimated`, `true_waveform_exact_track`): the UUV is the only object and adds nothing beyond the false
alarm radius. The passive scan map does not: each object free run lacks that ship's sidelobes, which appear
beyond the 10 m radius in the full scene (16 % on the tuning mix). The occupancy map does not either, in the
other direction (1 to 3 %): it is renormalised to sum 1 every frame, so in the full scene the objects take
mass from the background, and a weak object's Pd depends on how loud the others are. Criterion 1 is scored on
the occupancy map, so its Pd is at a false alarm rate below 5 %, not at 5 %. `direct_removed_exactly` (18 %)
is not comparable with the other echo rows. On a map whose background does not depend on the scene the
threshold does hold (`tests/harbor/test_scoreboard.py`: 4.4 % on an independent noise run, against 1.2 % for
the same map normalised every frame).

Criteria fixed before the run: 1 (passive, ferry, tug and boat Pd >= 0.9) **fails**; 2 (echo,
bistatic UUV Pd >= 0.5) **fails** (0.30); 3 (multistatic static bodies) was not run. The UUV is the
strongest cell of 89 % of the bistatic frames (median 2.6 m), but its peak exceeds the object free
threshold in only 30 %: the threshold is set by the residual direct path at the ends of the record.
The passive threshold is set by the frames before the first AIS report, where the ships are neither
projected out nor excluded. The direct path of the ferry is removed by 10.8 dB (estimated waveform)
and 21.3 dB (true waveform and exact track), as in stage 0.

**Why given the true waveform and the exact track the UUV is mapped worse than with everything estimated**
(Pd 0.02 against 0.30, map peak within 5 m of the UUV in 26 % of frames against 77 %, although the direct
path is removed 10 dB deeper). Neither configuration given the truth is an upper limit. Measured on the
tuning mix (`scripts/diagnose_true_waveform.py`, seed 0, given the true waveform, the exact track and the truth
components):

* It is not the removal: the ECA with the true waveform takes 0.6 dB of the UUV echo (0.3 dB with the data
  reference). It is not the scale (the `"snr"` map is independent of the reference scale), the frames (both
  references are cut on the same frames) or a conjugation (per bin over the frames, the data reference and
  the true waveform are coherent at 0.98 with the exact track, 0.79 with the refined one).
* It is the timing. The data reference leads the true waveform by 2.6 to 2.8 ms. The direct path recorded at
  the hydrophones arrives 1.5 to 2.3 ms earlier than the true waveform propagated at the chart's sound
  speed, and the FDTD scheme's dispersion model (simulator knowledge, `SimulatorAwarePropagator`) predicts
  0.5 to 2.9 ms early for the same paths: same sign, same order, direction dependent (the hydrophones do not
  agree one by one; a band limited cross correlation is also ambiguous by about 0.6 ms). The simulated water
  propagates faster than the chart's single sound speed, by an amount that depends on direction.
* The data reference is estimated from the traces, so it carries the same propagation as the echo and the
  free field matched filter stays aligned. The true waveform is the emitted signal: against free field
  replicas the echo is 1 to 3 ms off, the coherent sum over 100 to 760 Hz loses focus, and the UUV echo alone
  maps 4.7 dB lower (map peak 813 against 2400) while the residual direct path and the noise do not drop
  (about 700 and 550). The UUV peak is then 4.2 m off (median), at the edge of the 5 m radius, and loses to
  the residuals.
* With the dispersion corrected replicas (simulator knowledge), the true waveform run finds the UUV in 49 % of
  the frames (from 23 %), 2.7 m off; the gap to the 73 % of everything estimated is not explained.
* `direct_removed_exactly` uses the same true waveform and still scores 0.51, because its threshold is set
  without any residual direct path, so the weaker echo peak still clears it.

A fair reference configuration would pair the true waveform with the simulator aware replicas (not run
here).

## Earlier history

### The passive emission experiment

In the passive-emission experiment summarized here, at the tested UUV level (-30.5 dB re 1 Pa
at 100 m, about 30 dB below the ferry) the UUV is
**not** detected: it is found only when it is about as loud as the ferry (a sweep of the UUV level
with the model-projection chain found it, with a 64 m occupancy error, only at -0.9 dB).
This is a result for that passive-source-emission experiment, not the bistatic echo demo or a
general guarantee about every later scenario configuration. The source level is a first guess,
not a tuned or measured value. Every item below was a single
bounded experiment with a pass/fail criterion fixed beforehand.

| Hypothesis for the ship residual | Measurement | Result |
|---|---|---|
| Ship projection removes the UUV too | Fraction of the UUV replica inside the ship subspace | 5-33 %: not the cause |
| Whitening floor inflated by ships | Matched output at the UUV tone, UUV-only vs all vessels | **Real bug**, fixed by re-whitening (about 200x) |
| DAS gain scatter limits cancellation | Divide out the true gains | No effect (B at the UUV 3.6 -> 3.5) |
| Motion, patch size, assumed sound speed | Ferry-only cancellation depth, real FDTD vs exact free-field control | Real: 3-6 dB in every setting; control: 16-45 dB |
| Shoreline reverberation | Same depth, shore reflection 0.45 / 0.07 / 0.005 | 5.2 / 6.6 / 6.5 dB: not the cause |
| Emission-time (retarded) positions | Retarded per-channel replica positions | Shift 1.2 m, no change (5.2 dB) |
| Harbor geometry and shadowing | Same ferry in an open-water FDTD | 4-8 dB: not the cause |
| **FDTD numerical dispersion** | Stationary source vs analytic Green's function | **Real**: phase error up to 0.8 rad (axis) / 2.5 rad (diagonal) at 800 m, 40-60 Hz, anisotropic |
| Brute-force fix by FD order / sample rate | Order 4/8 at 600/1200 Hz | None reaches 0.1 rad (leapfrog time-stepping error remains) |
| Dispersion-corrected replica (`dispersion.py`) | Same stationary map | **Fixed**: <= 0.102 rad at 400-800 m, both directions |
| ... and the moving ferry | Ferry-only cancellation depth with the corrected replica | Only 5.5 dB (1 s frames), 11.6 dB (2 s, open water): **the criterion (20 dB) was not met** |

### Stage 0 of the multistatic work (branch `multistatic`)

Goal of the long term work: locate every object in the harbor (radiating vessels, silent moving
bodies, static structures) by passive sensing with ship noise as the illuminator. Stage 0 asks
whether the bistatic chain works with everything estimated from the traces and builds the multi illuminator,
multi body scenario for the later stages. Every result states its truth inputs; every
experiment had its pass/fail criterion in the script docstring and `results.json` before it ran.

**Stage 0a, direct path cancellation against range** (`examples/harbor/90_history/harbor_cancellation_depth.py`,
fine 0.5 m grid, 10 s, the ferry signature alone at nearest channel ranges of 193, 99 and 29 m,
stationary and at 4 m/s parallel to the first cable leg; ECA depth over the live cable channels
in 100 to 760 Hz; `output/harbor_cancel/` at commit `fd12458`). References: `data` (delay and sum beam at the AIS
refined position, estimated from the traces), `true_waveform` (true waveform, AIS refined position), `true_waveform_exact_track` (true
waveform, exact position). Criterion: median depth with the `data` reference at least 20 dB.
The AIS refinement of every column used the FDTD dispersion corrected replicas (simulator
knowledge, so even the `data` column was not everything estimated); the rows marked *free field replicas* are the
same runs with the free field replicas a deployed system would use
(`--propagator free` at the time, `output/harbor_cancel/freefield/` at commit `fd12458`), and only their `data` column has everything estimated
(the `true_waveform` columns still use the true waveform).

| Source | Range | `data` | `true_waveform` | `true_waveform_exact_track` | `data`, Doppler + 4 s fit | `true_waveform`, Doppler + 4 s fit | `true_waveform_exact_track`, Doppler |
|---|---|---|---|---|---|---|---|
| stationary | 193 m | 8.8 dB | 34.4 dB | 32.9 dB | 8.7 dB | 33.2 dB | 31.5 dB |
| moving 4 m/s | 193 m | 2.3 dB | 5.5 dB | 5.5 dB | 9.2 dB | 21.4 dB | 24.4 dB |
| stationary | 99 m | 18.7 dB | 47.3 dB | 45.1 dB | 19.6 dB | 44.2 dB | 42.2 dB |
| moving 4 m/s | 99 m | 3.4 dB | 3.6 dB | 3.6 dB | 18.9 dB | 22.9 dB | 24.6 dB |
| stationary | 29 m | 7.0 dB | 37.4 dB | 51.0 dB | 11.9 dB | 29.4 dB | 50.6 dB |
| moving 4 m/s | 29 m | 2.7 dB | 2.2 dB | 2.2 dB | 7.2 dB | 14.9 dB | 21.5 dB |
| stationary, free field replicas | 193 m | 5.5 dB | 3.5 dB | 32.9 dB | 10.1 dB | 3.3 dB | 31.5 dB |
| moving 4 m/s, free field replicas | 193 m | 3.5 dB | 5.4 dB | 5.5 dB | 9.2 dB | 10.4 dB | 24.4 dB |
| stationary, free field replicas | 99 m | 19.0 dB | 44.5 dB | 45.1 dB | 19.9 dB | 41.5 dB | 42.2 dB |
| moving 4 m/s, free field replicas | 99 m | 3.4 dB | 3.6 dB | 3.6 dB | 18.8 dB | 19.0 dB | 24.6 dB |
| stationary, free field replicas | 29 m | 7.5 dB | 38.0 dB | 51.0 dB | 11.9 dB | 28.5 dB | 50.6 dB |
| moving 4 m/s, free field replicas | 29 m | 2.7 dB | 2.2 dB | 2.2 dB | 7.3 dB | 15.0 dB | 21.5 dB |

Median ECA depth over the live cable channels (`results.json`, `results_doppler.json`,
`results_doppler_smooth4.json`; the free field rows from the `_freefield` files).
**Criterion failed at every range** (the `data` column never reaches 20 dB, with either replica).
The free field replicas change mainly the 193 m stationary case: the refinement misses the ship by
18 m instead of about 1 m (the numerical dispersion of the 0.5 m grid is several wavelengths of
phase at that range), so the AIS refined `true_waveform` column falls from 34 to 4 dB there. Reading it with the interpretation fixed in advance:

* The stationary source is low at every range with the `data` reference while the `true_waveform`
  reference at the same AIS refined positions reaches 34 to 47 dB, so the removal method is not
  the limit; the reference estimated from the traces is. On the coarse debug case the same beam built from the
  noise free traces gave 19 dB against 8 dB from the noisy ones, and its coherence with the true
  waveform falls from 0.94 at 300 Hz to 0.16 at 100 Hz: the strain response goes as frequency
  squared, so the low end of the band is noise dominated in the beam, and a least squares
  projection on a noisy reference leaves a residual of about one over the reference's signal
  to noise ratio.
* The moving source fails at every range for every reference with the fixed lag ECA (2 to
  6 dB): the direct path delay drifts by `v T / c` inside a 0.5 s block, a full cycle at the
  band top. Per sample delays that follow the ship (`BistaticConfig.eca_doppler`, 8x FFT
  upsampling for the resampling) bring the exact track case to 21 to 25 dB. With the AIS
  refined track the per frame jitter of about 1 m becomes velocity noise (7 to 9 dB) until the
  track is fitted by a local straight line over 4 s (`track_smooth_s`; 15 to 23 dB).
* Near field channels are not the cause: the exact position at 29 m is cancelled by 51 dB.
  What fails at 29 m is the AIS refinement itself (5.6 m error, jumping between frames), so no
  `gate_radius` rerun was made.
* The per channel plots (`depth_per_channel.png`) show the data reference depth rising from 3 dB
  at the channels broadside to the source to 17 dB at the oblique ones (193 m), the true waveform
  20 to 55 dB everywhere.
* The ECA projection is a Cholesky solve of the normal equations with one refinement step (five
  times faster than the QR it replaced); on the stationary 193 m true waveform case it reads 1 dB
  (median) and 3 dB (10th percentile) below the exact QR projection. The criterion column is
  unaffected (8.8 dB either way).

**Stage 0b, the bistatic demo without its truth inputs** (`examples/harbor/90_history/harbor_bistatic_ablation.py`,
coarse and fine datasets, `output/harbor_bistatic/ablation/` at commit `fd12458`). Configurations: 1 `direct_removed_exactly`
(direct path removed with the truth component, true waveform, exact track: the demo), 2 `true_waveform_exact_track` (direct path in
the traces, removed by ECA; true waveform; exact track), 3 `estimated_waveform_exact_track` (data reference, exact
track), 4 `all_estimated` (data reference, AIS track with refinement). Criterion for `all_estimated` on the
fine dataset: median echo contrast at least 10 dB and at least half the frames within 5 m.
Every configuration of the original run used the FDTD dispersion corrected replicas, so the rows
named `all_estimated` used simulator knowledge. The rows marked *free field (everything estimated)*
are the same configurations with free field replicas and no truth input at all
(`--propagator free --cuda` at the time, `output/harbor_bistatic/ablation_freefield/` at commit `fd12458`); they are the rows of this table with everything estimated from the traces. The committed `results.json` files keep the configuration names of the time.

| Configuration | coarse: median error | within 5 m | 1.5 s within 5 m | contrast | ECA depth | fine: median error | within 5 m | 1.5 s within 5 m | contrast | ECA depth |
|---|---|---|---|---|---|---|---|---|---|---|
| `direct_removed_exactly` | 1.7 m | 90 % | 99 % | 17.8 dB | truth | 2.3 m | 92 % | 100 % | 20.2 dB | truth |
| `true_waveform_exact_track` | 71.2 m | 15 % | 3 % | 16.2 dB | 8.5 dB | 72.5 m | 0 % | 0 % | 18.9 dB | 3.5 dB |
| `estimated_waveform_exact_track` | 84.3 m | 5 % | 1 % | 17.9 dB | 5.6 dB | 89.8 m | 1 % | 0 % | 18.8 dB | 2.2 dB |
| `all_estimated` | 80.8 m | 5 % | 1 % | 16.8 dB | 5.3 dB | 89.8 m | 5 % | 0 % | 18.2 dB | 2.8 dB |
| `true_waveform_exact_track+doppler` | 3.8 m | 54 % | 86 % | 14.8 dB | 25.1 dB | 6.0 m | 48 % | 53 % | 14.9 dB | 21.3 dB |
| `estimated_waveform_exact_track+doppler` | 6.1 m | 42 % | 53 % | 15.3 dB | 14.8 dB | 2.7 m | 68 % | 72 % | 16.0 dB | 12.2 dB |
| `all_estimated+doppler` | 9.2 m | 27 % | 35 % | 15.4 dB | 10.3 dB | 6.8 m | 39 % | 43 % | 16.2 dB | 8.3 dB |
| `true_waveform_exact_track+doppler+smooth4` | 3.8 m | 54 % | 86 % | 14.8 dB | 25.1 dB | 6.0 m | 48 % | 53 % | 14.9 dB | 21.3 dB |
| `estimated_waveform_exact_track+doppler+smooth4` | 6.1 m | 42 % | 53 % | 15.3 dB | 14.8 dB | 2.7 m | 68 % | 72 % | 16.0 dB | 12.2 dB |
| `all_estimated+doppler+smooth4` | 8.5 m | 32 % | 37 % | 15.2 dB | 11.0 dB | 3.8 m | 58 % | 54 % | 16.1 dB | 11.0 dB |
| `all_estimated`, free field (everything estimated) | 78.5 m | 13 % | 1 % | 18.5 dB | 5.0 dB | 86.4 m | 11 % | 0 % | 19.7 dB | 2.8 dB |
| `all_estimated+doppler`, free field (everything estimated) | 67.3 m | 34 % | 43 % | 16.1 dB | 8.7 dB | 62.6 m | 38 % | 33 % | 18.5 dB | 7.3 dB |
| `all_estimated+doppler+smooth4`, free field (everything estimated) | 3.0 m | 62 % | 75 % | 15.5 dB | 11.0 dB | 3.2 m | 75 % | 77 % | 17.3 dB | 10.8 dB |

Per 0.5 s frame (`results.json`, rows above the free field ones: dispersion corrected replicas; the
`+doppler` rows are the same configurations with
`eca_doppler=True`, reported as variants outside the criterion). The coarse `direct_removed_exactly` row
reproduces `demo/demo_summary.json` exactly. **Criterion failed**: `all_estimated` on the fine dataset
has a median contrast of 18.2 dB but only 5 % of the frames within 5 m. The contrast (map peak
over map median) does not discriminate here, because with the direct path cancelled by only 2
to 4 dB the peak sits on the residual direct path, which is just as sharp as an echo. The step
that loses the localisation is **1 to 2**, the fixed lag ECA replacing the truth removal: 92 %
of frames within 5 m become 0 % and the ECA removes 3.5 dB of the moving ferry (the stage 0a
moving source result), with the echo lost by under 1 dB. Steps 2 to 3 and 3 to 4 change
nothing at that point. With per sample delays (`eca_doppler`) the ECA cancels 21 dB (true waveform) and 8 to 12 dB (data reference) and the chain works part of the time: the variant with everything estimated locates the UUV in 39 % of the frames (43 % after 1.5 s integration, median
error 6.8 m), the data reference with the exact track in 68 %. With the refined track also
fitted over 4 s (`track_smooth_s`, the `+smooth4` rows from `ablation/smooth4/`) the variant with everything estimated reaches 58 % of frames within 5 m, median error 3.8 m, contrast 16.1 dB, ECA depth
11 dB: the numbers the criterion asked for, but from a chain changed after the criterion was
fixed, so it is reported as a variant and not as a pass. With free field replicas (the chain with everything estimated, run after the sensing contract made the dispersion replicas visible as simulator knowledge)
the same variant does better, not worse: 75 % of the fine frames within 5 m, median 3.2 m; the
plain `all_estimated` configuration fails the criterion with either replica (11 % within 5 m). The data reference beats the true waveform once the ECA follows the ship, presumably because the beam carries the quay
multipath that the free field image lags only approximate. The per frame estimate fails on the
remaining frames by jumping to a sidelobe (p90 error about 200 m), not by a small miss.

What limits the `all_estimated` configuration, in order: the direct path residual (8 dB of ECA depth
against the 16 to 18 dB the echo sits below the direct path on the median cable channel,
`levels.json`), set by the estimated reference's signal to noise ratio (stage 0a); then the AIS
refined track, whose jitter the per sample delays read as velocity noise (track smoothing
recovers most of it); the echo loss in the ECA is small throughout (0.3 to 1.9 dB).

**Stage 0c, multistatic scenario and dataset** (`examples/configs/harbor_multistatic.scenario.json`,
`examples/harbor/01_datasets/harbor_multistatic_dataset.py`, now `output/recordings/multistatic/`). Same 0.5 m crop and
sensors as the bistatic scenario with three illuminators (the AIS ferry; an AIS tug crossing
the ferry's geometry from the north east; a small radiating boat without AIS), three silent
moving bodies (`uuv1` and `uuv2`, 1 m x 3 m soft hulls on different headings at 2 and 1.5 m/s;
a diver, 0.5 m x 1 m at 0.5 m/s with 2 m of wander) and seven static soft bodies (a row of six
1 m pilings 5 m apart at y = 250 m, `kind: "static"`; a moored 6 m x 30 m hull by the north
quay). All tracks are at least 15 m from land and outside the 30 m sponge. Each illuminator is
rendered with all ten bodies present and with none (`combined_bodies=True`), so the dataset
carries `direct:<id>` and `scattered:<id>>bodies` per illuminator plus the noise, and
`truth.json` gives every object's type, body, AIS status, position and velocity per frame.

Fine dataset levels on the median live cable channel, 200 to 760 Hz (`levels.json`): echo to
direct ferry -8.1 dB, tug -14.5 dB, boat -11.1 dB; echo to noise
ferry +5.7 dB, tug -0.3 dB, boat -4.7 dB (best channel +13.7, +5.4, +0.5 dB).
The FDTD runs took 38 min on the GPU (fine) and 19 min (coarse, where the sponge is kept at
30 m in metres, unlike the bistatic coarse run). `ferry_only_map.png` runs the bistatic chain given the truth (direct path removed with the truth component, true ferry waveform, exact ferry track)
on the ferry's scattered component alone: in the map summed over the 79 frames every body is
above the map median within 4 m of its position, uuv1 +6.4 dB, uuv2 +8.0 dB, diver +7.0 dB, piling1 +5.5 dB, piling2 +7.3 dB, piling3 +7.3 dB, piling4 +7.6 dB, piling5 +7.6 dB, piling6 +6.4 dB, moored +15.1 dB.
The echoes are there; the per frame maps are dominated by the sidelobe streaks of the strongest
echo, which is the job of the later CLEAN and fusion stages, not of this dataset check.

### Open (at the time)

The dispersion fix makes a stationary source's replica accurate to <= 0.1 rad, yet the moving
ferry is cancelled by only 5-12 dB.  The ship has to be cancelled by about 31 dB to match the UUV.
What limits the moving ferry is not established.  Unverified candidates: the 10-50 m near-field
channels (the ferry runs parallel to the cable about 30 m away for much of its track, and the
stationary map shows 10-25 % amplitude errors at those ranges), and the representation of a moving
source in the FDTD.  The suggested next experiment is the cancellation depth of a *stationary*
source at 400, 150 and 30 m: low at all ranges points at the moving-source representation, low only
near the cable points at the near-field channels (which could then be gated).

An analytic forward model (free-field channel with a known replica, as in a simple matched-field
demo) would make the technique work by construction; this package currently uses the FDTD only,
by choice.

After stage 0 the moving ferry question above is answered for the bistatic chain: the fixed
lag ECA could not follow the delay drift inside a block, the per sample delays can (21 to 25 dB
with the true waveform and track). What remains open is the reference estimated from the traces: a delay and sum
beam of the strain cable has too little signal to noise ratio at the low end of the band to
cancel the direct path by more than 8 to 19 dB, and the AIS refinement is unreliable within
about 30 m of the cable. Candidates for stage 1: a frequency weighted or iterative reference
(cancel, re-beam on the cleaned data for the next pass), the hydrophones as a pressure
reference, a constant velocity track fit across the whole run instead of per frame refinement,
and gating the near field channels. Multistatic fusion, CLEAN extraction and tracking were
deliberately not started.

## Scripts and workflows of the history

| Script | What it does | Time |
|---|---|---|
| `examples/harbor/02_forward_model/harbor_m1_demo.py` | Map, one FDTD source run, DAS gather, wavefield snapshots | ~15 s |
| `examples/harbor/01_datasets/harbor_m2_dataset.py` | Builds the default multi-vessel labelled dataset and overview figures (cached FDTD) | expensive on first build; reuses cache |
| `examples/harbor/02_forward_model/harbor_m1_scatterer_compare.py` | Three FDTD runs (no body, "hard" and "soft" body) to isolate the scattered field of a static reflector | three FDTD runs |
| `examples/harbor/90_history/harbor_m3_detect.py` | Baseline CFAR detection sweeps **without** AIS suppression (the loud ships hide the UUV); `--empirical-refine` uses the `localiser.py` calibration | processing sweep; reuses cache |
| `examples/harbor/90_history/harbor_survey_grid.py` | Pre-computes per-cell FDTD transfer functions for `harbor_m3_detect.py --empirical-refine grid_survey` (resume-safe cache) | one FDTD run per water cell of the scan grid |
| `examples/harbor/90_history/harbor_m4_dlc.py` | Detect–locate–classify experiment, noise-only CFAR calibration, optional AIS suppression, CNN/CVAE classification and evaluation | varies; FDTD gathers are cached |
| `examples/harbor/90_history/harbor_mfp_demo.py` | Analytic (free-field, exact replica) reference for the whole chain: AIS refinement, ferry and pier projection, passive 400 Hz MFP (motor on), bistatic echo MFP (motor off), occupancy track, Bartlett vs MVDR under replica mismatch, Pd vs UUV level. Figures and `results.json` in `output/harbor_demo/` at commit `fd12458` (now written to `output/studies/history/harbor_mfp_demo/`) | ~26 min on 16 CPU cores (`--trials 0` skips the Pd sweep: ~4 min) |
| `examples/harbor/90_history/harbor_mfp_domain.py` | The same analytic chain in the domain of `output/recordings/harbor/dataset.npz` (land, cable + hydrophones, 120 s tracks, AIS, signatures, band, noise and DAS gains from the dataset; analytic propagation with opaque land). Adds AIS-ship gain calibration, re-whitening on the cleaned residual, DAS common-mode and distant-shipping plane-wave projection. Figures and `results.json` in `output/harbor_domain/` at commit `fd12458` (now written to `output/studies/history/harbor_mfp_domain/`) | ~1 h on 16 CPU cores (`--pd-windows 0`: ~20 min) |
| `examples/harbor/02_forward_model/harbor_m1_uuv.py` | Single Ricker pulse with and without a stationary finite-size UUV; plots the difference as the scattered field | one FDTD pair |
| `examples/harbor/02_forward_model/harbor_m1_uuv_video.py` | Animate the pulse and predicted UUV echo on the cable | one FDTD pair |
| `examples/harbor/90_history/harbor_bistatic_budget.py` | Analytic 2-D bistatic link budget with idealized soft/rigid cylinder models; no FDTD | quick |
| `examples/harbor/01_datasets/harbor_bistatic_dataset.py` | Fine-grid FDTD scenario; caches direct and body-scattered components separately | expensive; `--coarse` is faster but under-resolves the hull |
| `examples/harbor/90_history/harbor_bistatic_demo.py` | Coherent bistatic localization figures given the truth (direct path removed exactly, true waveform, exact track) and error summary from a saved dataset | processing run |
| `examples/harbor/90_history/harbor_cancellation_depth.py` | Stage 0a: ECA cancellation depth of the ferry signature alone against range, stationary and moving, three references; `--doppler`, `--smooth`, `--gate` and `--propagator` (free field by default, `_freefield` suffix; `simulator` for the dispersion corrected replicas) variants write suffixed outputs | 6 cached FDTD runs (12 min on the GPU) + about 40 min of processing |
| `examples/harbor/90_history/harbor_bistatic_ablation.py` | Stage 0b: `run_bistatic` on the coarse and fine datasets with the truth inputs removed one at a time (`--doppler` adds the per sample delay ECA variants, `--smooth` the track fit, `--propagator simulator` the dispersion corrected replicas, `--cuda` the GPU chain); every result records its `provenance` | about 1.5 h of processing for both datasets |
| `examples/harbor/01_datasets/harbor_multistatic_dataset.py` | Stage 0c: three illuminators and ten bodies with combined scattering, truth per frame, levels, map, and a ferry-only check map given the truth | 6 FDTD runs (fine: about 1 h on the GPU) |
| `examples/harbor/06_walkthrough/` | [Walkthrough](#walkthrough-from-the-traces-to-the-harbor-picture): eight scripts, one figure per step of the chain (recording, passive map, tracks, AIS labels, direct path removal, echo map, harbor picture, geometry); `run_all.sh DATASET --cuda` | about 1 min per recording |
| `examples/harbor/05_detection/harbor_stream.py` | The real-time system (`water/streaming.py`): passive chain without AIS, tracker, echo chain on the tracks, AIS labels afterwards; feeds a recording in 0.25 s chunks, reports compute per second of data and latency, checks every output against the true positions; `--dataset multistatic|bistatic|harbor [--echo] --cuda`; writes `output/reproduce/runs/<dataset>/` | 15 to 60 s per run |
| `examples/harbor/90_history/harbor_picture_stage1.py` | Harbor picture stage 1: the [scoreboard](architecture.md#scoreboard) on the bistatic, multistatic and 5 m harbor datasets, tuning set (noise seed 0) and test set (seeds 1 to 5, remixed from the cached gathers), passive and echo chains; `--dataset`, `--chain`, `--propagator {free_field,simulator}`, `--cuda`, `--workers`; writes `output/studies/history/harbor_picture_stage1/` (its earlier outputs: `output/harbor_picture/stage1/` at commit `fd12458`) | about 1.2 h on the GPU (runs are cached in `cache/`) |
| `examples/harbor/02_forward_model/harbor_domain_video.py`, `examples/harbor/90_history/harbor_stack_movie.py`, `examples/harbor/90_history/harbor_presentation.py` | Visualize passive/bistatic scans, echo stacks, or the scenario pipeline | varies |
| `examples/harbor/90_history/harbor_train_classifier.py`, `examples/harbor/90_history/harbor_train_anomaly.py`, `examples/harbor/90_history/harbor_build_classifier_dataset.py` | Build harbor classification data and train CNN/CVAE models | training workload |

Run times are from the original experiments and are hardware and configuration specific; the first FDTD cache build is the main cost.  [Generated outputs](outputs.md) lists what each script writes and what is committed.

`examples/configs/harbor.scenario.json` is the default scenario (set `"fdtd": {"use_cuda": true}` to
simulate on the GPU). Outputs are written under `output/`; FDTD cache files can be reused between
dataset assembly and experiments.

### End-to-end examples

Run the passive scenario and detection/classification workflow:

```bash
python examples/harbor/01_datasets/harbor_m2_dataset.py
python examples/harbor/90_history/harbor_m4_dlc.py --config examples/configs/harbor.scenario.json --out output/studies/history/harbor_m4_dlc
```

The M4 script can skip AIS suppression with `--no-suppress`, enable GPU processing with
`--cuda`, and accepts `--pfa`, `--cal-seeds`, `--cnn`, and `--cvae` options. Missing classifier
checkpoints are reported as unclassified output rather than preventing the rest of the workflow.
Check each script's `--help` for current flags.

For the cropped strain-DAS bistatic setup:

```bash
python examples/harbor/90_history/harbor_bistatic_budget.py --out output/studies/history/harbor_bistatic_budget
python examples/harbor/01_datasets/harbor_bistatic_dataset.py --coarse
python examples/harbor/90_history/harbor_bistatic_demo.py \
    --dataset output/recordings/bistatic/coarse/dataset.npz \
    --out output/studies/history/harbor_bistatic_demo
```

`--coarse` switches from the default 0.5 m grid to 1 m and proportionally scales the sample
rate/band. It is faster but a 1 m × 3 m body is poorly resolved at 1 m cells. A fine-grid run
uses more compute and storage; cache and output directories are separate from the default
`output/harbor` dataset.

```python
from acoustic_sim.contract import Observation
from acoustic_sim.water.detection import ProcessingConfig, build_grid
from acoustic_sim.water.suppression import SuppressionConfig, run_ais_aware
from acoustic_sim.water.occupancy import OccupancyConfig, OccupancyGrid

obs = Observation.load("output/recordings/harbor/dataset.observation.npz")   # or HarborDataset.load(...).observation
pc = ProcessingConfig(win_s=2.0, hop_s=1.0, use_cuda=True)       # free field replicas: nothing beyond the chart
run = run_ais_aware(obs.traces, obs.geometry, obs.fs, obs.environment, obs.reports, pc, SuppressionConfig(),
                    grid=build_grid(obs.environment, pc))
track = OccupancyGrid(run.scanner.grid, OccupancyConfig(use_cuda=True)).run(run.scan.B, run.wh.times)
print(run.provenance.to_dict())
```

Replicas that follow the simulator's grid dispersion are a diagnostic: they need the truth file
(`propagator=SimulatorAwarePropagator.from_truth(Truth.load("output/recordings/harbor/dataset.truth.npz"))`
from `acoustic_sim.water.diagnostics`), they model the simulator's grid, not the ocean, and the
run's provenance then lists `"dispersion"`.

For bistatic processing, the corresponding API is `BistaticConfig` and `run_bistatic`:

```python
from acoustic_sim.water.bistatic import BistaticConfig, run_bistatic

bc = BistaticConfig(
    f_lo=100.0, f_hi=760.0, # the processing band: a choice of the processing (here the 0.5 m datasets')
    reference="data",       # "true_waveform" only for controlled experiments; flagged
    removal="eca",
    method="coherent",
)
echo_run = run_bistatic(obs.traces, obs.geometry, obs.fs, obs.environment, obs.reports, bc)
```

`BistaticConfig(propagator=SimulatorAwarePropagator.from_truth(truth))` compensates for the
numerical dispersion of that specific grid/time step; it is simulator knowledge (flagged in
`echo_run.provenance`) and not a model of real-water propagation.

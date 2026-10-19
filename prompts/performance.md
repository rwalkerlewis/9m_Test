Work in the 9m_Test repo on branch `performance` (`git fetch origin && git checkout performance`; it starts from `main` after PR #21). Use the GPU for every benchmark and report its model. Two other branches are open and touch `harbor/bistatic.py`: `blind_bistatic` (detection and tracking) and `inversion` (full domain inversion). Keep every public function signature and default result unchanged so those branches merge cleanly; add new code paths behind options and keep the current code as a selectable reference path.

Read `src/acoustic_sim/fdtd.py` (`FDTDConfig`, `FDTDSolver`, `_step`, `_build_damping_global`, `_sample_receivers`, `run`), `src/acoustic_sim/harbor/fdtd_runner.py` (`HarborFDTDSolver`, `_inject`, `_apply_scatterers`, `simulate_source`, the cache key), `src/acoustic_sim/harbor/scatterer.py`, `src/acoustic_sim/harbor/dispersion.py`, `src/acoustic_sim/harbor/bistatic.py` (`eca_remove`, `eca_stage`, `bistatic_map`, `ReplicaModel`, `ship_positions`, `motion_integrate`), `tests/harbor/` and the "GPU" and "Status and findings" sections of `docs/harbor.md` before changing anything.

## Goal

Reduce the compute cost of the harbor forward model and processing chain by an order of magnitude or more without changing results beyond stated tolerances. Every change is measured twice: wall time against the current code, and accuracy against the current code (or, for scheme changes, against an analytic or converged reference).

## Stage 0: benchmark and accuracy harness (do first)

Write `scripts/bench_harbor.py` that, for a fixed set of cases, records wall time and saves the outputs for later comparison:

1. FDTD: the fine bistatic scenario (0.5 m, 6 kHz), one moving source alone for 2 s and for 10 s; the same with the 1 m x 3 m UUV body; the multistatic scene with all 10 bodies for 2 s.
2. Processing: `run_bistatic` on the fine bistatic dataset for the `oracle`, `eca_oracle+doppler` and `blind+doppler+smooth4` configurations of `examples/harbor_bistatic_ablation.py`; `eca_stage` alone on the stage 0a 193 m moving case.

Save the reference outputs (traces, maps, estimates) from the current code to a local directory (gitignored) so every later stage compares against them. Commit the timing table as `output/performance/baseline.json`.

## Stage 1: fused FDTD time step

1. Implement the 2D update of `FDTDSolver._step` (Laplacian of order `fd_order`, leapfrog update, `C2`, damping `sigma`, Dirichlet edges) as a single `cupy.RawKernel`, with two field buffers swapped each step instead of `zeros_like` allocation. Fold the harbor land mask multiply into the same kernel for `HarborFDTDSolver`. Keep source injection and receiver sampling on the device and outside the kernel if simpler.
2. Support float64 and float32 (`FDTDConfig.precision`, default float64 for now).
3. Keep the current array code as `kernel="reference"`. Use it on the CPU and under MPI unless the fused kernel supports the halo exchange.
4. Keep the cache key independent of `kernel` but dependent on `precision`.

Criteria, fixed now:
- **Fused float64 against the reference:** traces agree to a relative L2 error of 1e-12 per channel.
- **Fused float32 against float64, on the stage 0 cases:**
  - direct traces within a relative L2 error of 1e-4 per channel (median) and 1e-3 (max);
  - the scattered field (run with body minus run without) within 1e-3 of its own rms;
  - stage 0a ECA depths within 0.5 dB;
  - stage 0b `blind+doppler+smooth4` per frame estimates identical to within one map cell in at least 95 % of frames.
- If float32 fails, report by how much and keep float64 as the default. Do not loosen the criterion.

## Stage 2: scatterer masks on the GPU

`_apply_scatterers` computes each body's factor with `body_factor` on the CPU every step and copies it to the device. Write a device kernel that evaluates the same smooth signed distance mask (rectangle and ellipse, soft and absorbing) from each body's centre and heading, for all bodies in one launch. Precompute the pose table for every step on the host once per run.

Criterion: masks agree with `body_factor` to 1e-12 (float64), and scattered traces agree with the stage 1 fused run to 1e-10 relative. Report the time per step against body count (0, 1, 10 bodies).

## Stage 3: batched runs

Add a batch dimension to the fused solver so B independent runs (different source signals, positions or trajectories, and different body sets) advance in the same launches. Use it in `render_vessel_gathers` (all jobs of one dataset build that share a grid) and in the stage 0a script. Choose B from free GPU memory.

Criterion: every batched run matches its single run to 1e-12 relative (float64). Report throughput in runs per hour against B.

## Stage 4: CPML absorbing boundary

Replace the damping sponge with a convolutional PML (auxiliary memory variables for the second order scalar wave equation) as `FDTDConfig.boundary = "cpml"`, keeping `"sponge"` as the default until this stage passes. Make the CPML thickness and profile configurable (start at 10 to 20 cells).

Measure the boundary reflection with a point source in open water against a reference run on a domain large enough that no reflection reaches the receivers in the window. Do this for the sponge at its current setting and for the CPML at several thicknesses, at normal and grazing incidence, across the band.

Criterion: a CPML thickness exists whose reflection is at least 10 dB below the current sponge across the band and at all tested angles. Report the cell count saved for the harbor scenarios.

## Stage 5: processing chain

Vectorize and move to the GPU (CuPy, with NumPy fallback), with no change in the mathematics:

1. `eca_remove`: batch all channels of a block. Build the delayed reference matrices with strided views, then use batched Cholesky solves of the normal equations with the existing loading and refinement step. Do the same for the Doppler variant.
2. `bistatic_map`: compute replica matrices once per frequency and reuse them across frames (extend the existing geometry cache). Vectorize the coherent method over frames (no Python loop over `k`), and the MVDR and MUSIC methods over frames with batched solves and eigendecompositions.
3. `ship_positions` refinement and `motion_integrate`: vectorize over frames and velocities.

Criterion: every output matches the current code to 1e-10 relative in float64 on the stage 0 benchmark cases. Report wall time per configuration before and after.

## Stage 6: higher order time stepping (prototype, opt in)

The docs found that leapfrog time error, not spatial order, limits phase accuracy (the reason `dispersion.py` exists). Prototype two alternatives as opt in schemes, without changing defaults:

1. A fourth order in time modified equation (Lax Wendroff type) update, with the stability limit derived and tested.
2. A k space corrected pseudospectral update (exact phase in a homogeneous medium), with the land handled by the same pressure release masking and a test of how well that works.

For each, at 0.5 m and 1 m grids, measure the phase error of a stationary source against the analytic 2D Green's function at 400 to 800 m (or the largest ranges the domain allows), axis and diagonal, across the band. Also measure the reflection from a straight pressure release wall against the image source solution.

Criterion, fixed now: a scheme on the 1 m grid matches or beats the current scheme on the 0.5 m grid with dispersion correction (at most 0.1 rad phase error), with a wall reflection error no worse than the current scheme's. If one passes, report its total cost against the current 0.5 m run. Do not make it the default in this task.

## Out of scope

Neural operator surrogates and reciprocal Green's function tables. They are deliberately left out.

## Rules

1. Tests in `tests/harbor/` (and `tests/` for the generic solver) for every new path:
   - fused kernel against the reference, in both precisions;
   - GPU masks against `body_factor`;
   - batched runs against single runs;
   - CPML reflection below a threshold on a small grid;
   - the vectorized ECA and maps against the reference implementations.
   GPU tests skip without CuPy, as the existing ones do. Keep `pytest` passing.
2. Benchmark results go in `output/performance/` (JSON and figures, committed); reference outputs and datasets stay gitignored.
3. Add a "Performance" section to `docs/harbor.md` with:
   - the before and after timing table per stage,
   - the accuracy results and pass/fail of each criterion,
   - which options are now defaults and which stay opt in.
   Update the GPU section's numbers.
4. Change a default (precision, boundary, kernel) only after its criterion passes, and say so in the docs and the PR.
5. Commit in small steps on `performance` and open a pull request against `main` with the timing and accuracy summary.
6. Stop after stage 6 and report the numbers, pass/fail per stage, the total speedup on the stage 0 benchmark, and anything the `blind_bistatic` and `inversion` branches need to change to pick up the new paths.

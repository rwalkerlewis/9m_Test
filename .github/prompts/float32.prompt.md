---
mode: agent
description: Branch float32 from origin/main. You own src/acoustic_sim/solver/ (the precision paths), water/fdtd_runner.py, water/fdt
---
Branch `float32` from `origin/main`. You own `src/acoustic_sim/solver/` (the precision paths), `water/fdtd_runner.py`, `water/fdtd_cuda_bodies.py`, the precision handling inside `bistatic.py`'s ECA (`_gram_prefix`, the Gram and solve; the algorithms themselves belong to `direct_path`, so coordinate: change dtypes and the solve method, not the estimators), `docs/physics.md` (precision section), `output/studies/float32/`, and the FDTD caches. You do not change processing parameters or the recordings' scenario files (the recordings are rebuilt in float32 only in section 3, as a separate commit, after the criteria pass). This task runs FDTD; coordinate before starting runs.

Read `docs/harbor_history.md` (the performance work: the float32 criteria table, the ECA conditioning note, the Toeplitz remark), `src/acoustic_sim/solver/fdtd.py`, `solver/fdtd_cuda.py`, `water/fdtd_runner.py`, `water/bistatic.py` (`_gram_prefix`, `eca_remove`), `scripts/bench_harbor.py`, `tests/harbor/test_performance.py` and `output/recordings/*/build.json` before changing anything.

## Why

The FDTD, the k space FFTs and the ECA run in float64; the passive chain, the replicas and the maps already run in float32 and complex64. On the Quadro RTX 4000 float64 runs at 1/32 of the float32 rate, and on every consumer card at 1/64, so the double precision parts set the hardware this code can use. The performance work measured float32 FDTD traces within 3.2e-5 (median) and 1.03e-3 (worst sample of one case) of float64, scattered fields within 7e-4 of their rms, ECA depths within 0.001 dB and every position estimate identical; the one miss was 3 percent over a round number. The ECA's Gram matrix comes from prefix sums over up to 120 000 samples and its solve needed 1e-14 loading in double: that part cannot be cast to single as it is.

## 1. FDTD and k space in float32

Make `precision="float32"` work for the k space scheme (single precision FFTs, the phase factors in single, the bodies on the device, the batched runner) and measure against float64 on every recording's scenario with the propagation error script: criterion, fixed now, the direct arrival error changes by at most 0.02 ms at every hydrophone, the direct traces agree within 1e-3 relative at the worst sample and 1e-4 at the median, the scattered component within 1e-3 of its rms. Report the time per step and the memory per run, both precisions, on the Quadro.

## 2. The ECA in mixed precision

The per sample work (resampling the reference with the track's delays, applying the projection to every channel) goes to float32. The accumulations stay double: the prefix sums are computed blockwise in float64 (or replaced by a direct float64 Gram of the block, whichever is faster at the block sizes in use), and the solve becomes a QR, or the Toeplitz solve the performance notes left for later, in float64 without loading. Criterion, fixed now: cancellation depth within 0.3 dB of the current double run on the six stage 0a cases and on every illuminator of every recording (from the truth components, a diagnostic), echo loss within 0.1 dB, map peaks within one cell. Report the ECA time per second of data before and after.

## 3. Switch, rebuild, reproduce

If both criteria pass: `precision: "float32"` in every harbor scenario file, rebuild the recordings (FDTD caches rebuilt, old ones deleted), `scripts/reproduce.py --update`, and the before and after table of every number in `docs/harbor.md`, with the reason for any that moved beyond tolerance. If a criterion fails: keep float64 as the default, report the failure and the numbers, and stop.

## 4. Docs

`docs/physics.md`: a precision section with the measurements of sections 1 and 2 and the statement of what runs in which precision and why (the accumulations). `docs/harbor.md`: the compute numbers.

## Rules

The repository rules in `AGENTS.md` apply, the figure rules included. Tests: float32 against float64 on the tiny CI scenario (traces and propagation error), the mixed precision ECA against the double one on the analytic test case, the Toeplitz or QR solve against the loaded normal equations on a well conditioned case. Stop after section 4. Report the criteria results, the time per step and the ECA time per second of data in both precisions, and the memory per run.

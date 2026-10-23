# The float32 build of `multistatic_b`: the boat's batched runs diverge

**Summary (follow up, 2026-10-22 evening).** The divergence comes from the batched float32 inverse FFT: on the Quadro
RTX 4000, `irfft2` over a batch of fields sometimes returns a corrupted field, in cuFFT 11.4 (CuPy, the container) and
in the cuFFT of CUDA 13 (torch, the host) alike; float64 and a single 2D field never did in these tests. Each
corrupted step puts a jump into the whole field, which the damped scheme then rings down, and repeated jumps carried
the batched boat runs to 1e16. Run one at a time (now the default for float32 k space runs, `runs_alone` in
`water/fdtd_runner.py`) the same runs stay within 3e-5 of float64 over the full 60 s, and the float32 build of the five
recordings built that way **passes** the scattered component criterion (section "A float32 build with the runs
alone" below). Why cuFFT's batched transform fails on this card was not found. Float64 stays the default in every
scenario file; switching is the repository owner's decision.

## The first build (task 6)

The float32 build of the five recordings (2026-10-22, the k space waits of `gpu_determinism.md` in) gave every
component of `simple`, `bistatic`, `multistatic` and `harbor`, and the ferry's and the tug's of `multistatic_b`, within
3.3e-4 of the float64 build (`scattered.md` as of commit 3387194; it now holds the build with the runs alone). The boat's two runs on `multistatic_b` (without and with the ten bodies,
run together as one batch of the fused kernel, `HarborFDTDBatch`) diverged: the field grew without bound within a
second, from about 1e-3 above 1500 Hz to 7e3, and the level table of the build gave 2e11 for the boat's direct
component against 0.11 in float64 (`build32/multistatic_b/build_stdout.log`). So the scattered component criterion
**fails** on `multistatic_b` and float64 stays the default.

Rule fixed before the build (`harbor_float32_scattered.py`): a component over the limit is put down to the GPU fault of
`gpu_determinism.md` only if its gathers, built again, give different traces. The second build's traces differed
from the first's in detail (NaN instead of 1e16), but diverged as well, and every further run did: the failure
repeats, so it is not read as that fault. The runs of the boat (a static source along the boat's track, 6000 Hz,
the `multistatic_b` grid 601 x 801), each the time at which the largest trace value first passes 1e3:

| Run | Batch | Bodies | Precision | Diverged |
|---|---|---|---|---|
| the build | 2 (without, with) | 10 in the second | float32 | yes, 19.88 s and 23.02 s |
| the build, the two gathers again | 2 | 10 in the second | float32 | yes, 18.94 s in both |
| `batchtest.py`, 60 s | 2 | 10 in the second | float32 | yes, 25.37 s and 25.39 s |
| `batchtest.py`, 30 s, `CUDA_LAUNCH_BLOCKING=1` | 2 | 10 in the second | float32 | yes, 14.78 s in both |
| `batchtest.py`, 30 s | 2 (the same run twice) | none | float32 | yes, 15.71 s in both |
| `boat.py`, 9.7 s | 1 | none | float32 | no (largest value 31) |
| `boat.py`, 30 s | 1 | none | float32 | no (largest value 36) |
| the committed float64 build | batched | as above | float64 | no |

What it says: the batched float32 k space run of this source diverges every time, the single run of the same source
does not (to 30 s), float64 never did, and neither the bodies nor the ordering of the GPU's work (launch blocking) is
the cause. The time varies from run to run (14.8 to 25.4 s), and two members of one batch given identical inputs end
at different values, so the batched computation is not even the same for identical members. No shared table of the
kernel had changed after a diverged run (the speed ratio, the sponge, the land mask, the k space kernel, the source and
receiver tables: compared with freshly made ones). The boat is in open water at those times, 70 to 85 m from land and
over 100 m from the grid's edge. A plain CuPy k space loop with two identical members on the same grid, uniform speed,
no land, 100 000 steps, stayed stable, but its members also differed by about 1e-5 from the first steps on: cuFFT's
batched single precision transform on these sizes does not give identical members identical results. **The cause of
the divergence was not found.** Not tried: a float32 build with batches of one (the single runs were tested to 30 s
only, not the 60 s of the recording), and the scheme's stability in float32 with the harbor's speed map and land mask. (Both were done in the follow up
below, which found the mechanism.)

The float32 build's logs and build files are in `build32/<recording>/` (the build times: `build.json`,
`fdtd_wall_s`; for `multistatic_b`, `build_first.json` is the whole first build, 6 runs in 898 s, and `build.json` the
rebuild of the boat's two gathers, 335 s); the recordings themselves were put back to the committed float64 build.

## Follow up: the runs alone over 60 s

The boat's source on `multistatic_b` in float32, one run at a time, the full 60 s of the recording (`scratch/f32/single60.py`,
`batched_divergence.json`, `single_runs_60s`), compared with the committed float64 gathers of the same runs:

| Run | Bodies | Largest trace value | Diverged | rms difference from float64 over its rms | FDTD time |
|---|---|---|---|---|---|
| batch of one (`HarborFDTDBatch`, one job) | none | 25.7 | no | 3.0e-5 | 218 s |
| batch of one | 10 | 25.7 | no | 3.0e-5 | 247 s |
| plain solver (`HarborFDTDSolver`, 2D fields) | none | 25.7 | no | 3.0e-5 | 220 s |
| plain solver | 10 | 25.7 | no | 3.0e-5 | 241 s |

The batch of one and the plain solver gave the same traces to the last bit. The boat's scattered component from
these runs (with the bodies less without) is within 1.2e-4 of float64's. The difference from float64 grows from 2.7e-5
of the rms in the first second to 5.1e-5 at 60 s and levels off; it does not grow without bound.

## Follow up: what the batched runs do

`divergence_growth_multistatic_b.png`: look at the two batched float32 curves leaving the others at 19.9 s and coming
back down by 22 s before the next jump; the runs alone lie on the float64 curve throughout.

The diverged gathers do not grow steadily. The largest value stays on the float64 curve for 19 s, jumps (by 1100
times in the run with the bodies, to 3.6e3 from 3.2), decays again, and jumps again at 23.0 s; after that the jumps
come every few seconds and the field sits between 1e8 and 1e16. The first jumps of the run with the bodies reach 43 to
197 of the 200 channels in the step they start and most of the rest within 40 steps (10 m of travel, on a cable
390 m long), so a jump is a change of the whole field at once, not a disturbance travelling from a point; and the
jumps at steps 138123 and 177976 are in both members of the batch at the same step. A scheme that rings a jump down is
stable; what has to be explained is the jumps.

The k space operator alone, `irfft2(rfft2(p) * K)`, applied over and over to one fixed float32 field with a wait after
every operation, exactly as `FusedFDTD.step` does it (`scratch/f32/fft_events.py`, the solver's own grid and kernel;
`scratch/f32/fft_iso.py`, the same with nothing from this repository; `batched_divergence.json`, `fft_in_solver_shape`
and `fft_isolated`). Every result is compared with the first; "corrupted" means a difference of 1e-4 of the result's
largest value or more.

| Transform | Layout | Grid | Precision | Library | Calls | Corrupted | Largest difference over the result's largest value |
|---|---|---|---|---|---|---|---|
| the pair | batch of two | 601 x 801 | float32 | CuPy, cuFFT 11.4 | 100 000 | 19 | 2.1e8 |
| the pair | batch of one (3D) | 601 x 801 | float32 | CuPy | 100 000 | 1 | 5.3e6 |
| the pair | 2D | 601 x 801 | float32 | CuPy | 100 000 | 0 | 0 |
| `rfft2` alone | batch of two | 601 x 801 | float32 | CuPy | 30 000 | 0 | 0 |
| `irfft2` alone | batch of two | 601 x 801 | float32 | CuPy | 30 000 | 33 | 2.8e8 |
| the pair | batch of two | 601 x 801 | float64 | CuPy | 30 000 | 0 | 0 |
| the pair | batch of four | 601 x 801 | float32 | CuPy | 30 000 | 8 | 1.8e8 |
| the pair | batch of one (3D) | 601 x 801 | float32 | CuPy | 30 000 | 0 | 0 |
| the pair | 2D | 601 x 801 | float32 | CuPy | 30 000 | 0 | 0 |
| the pair | batch of two | 600 x 800 | float32 | CuPy | 30 000 | 137 | 2.6e16 |
| the pair | batch of two | 601 x 800 | float32 | CuPy | 30 000 | 0 | 0 |
| the pair | batch of two | 600 x 801 | float32 | CuPy | 30 000 | 120 | 4.1e15 |
| the pair | batch of two | 361 x 481 (`harbor`) | float32 | CuPy | 30 000 | 0 | 0 |
| the pair | batch of two | 401 x 501 (`simple`) | float32 | CuPy | 30 000 | 0 | 0 |
| `irfft2` alone | batch of two | 601 x 801 | float32 | torch, cuFFT of CUDA 13 | 30 000 | 7 | 4.4e7 |
| the pair | batch of two | 601 x 801 | float32 | torch | 30 000 | 4 | 2.1e8 |
| the pair | batch of two | 600 x 800 | float32 | torch | 30 000 | 95 | 2.4e16 |
| the pair | 2D | 601 x 801 | float32 | torch | 30 000 | 0 | 0 |
| the pair, inverse in two stages (`ifft` along y, `irfft` along x) | batch of two | 601 x 801 | float32 | CuPy | 30 000 | 0 | 0 |
| the same | batch of two | 600 x 800 | float32 | CuPy | 30 000 | 3 | 3.2e-2 |

What it says: the batched float32 inverse transform sometimes returns a corrupted field, every member at once, while
the forward transform, float64 and the 2D transform did not in these tests. Two cuFFT builds (CUDA 12.9 in the
container, CUDA 13 in torch on the host) fail the same way, with the same size of error (2.1e8 in both), so it is not
CuPy and not one cuFFT release; whether it is cuFFT's batched kernels for these sizes or this card was not separated
(there is no second GPU to try). The grid size matters (600 x 800 is worse, 601 x 800 and the two smaller grids were
clean), so the harbor and simple recordings, which batch on smaller grids, are not on the sizes that failed. The
inverse done in two stages was clean on 601 x 801 but not on 600 x 800, so it is not a dependable way round.

Two more properties of the batched float32 transform: two identical members come out different by 6e-7 of the
largest value on every call (by 0 on 600 x 800, which fails all the same, so these small differences do not seed the
corruption), and a member with no source at all, batched with the boat's run, held about 1.2e-5 throughout, 1.5e-6
of the other member's level (`scratch/f32/pert.py`): the members are coupled at the rounding level. Two 30 s batches
of two (the second member silent, or the boat's source scaled by 1 + 1e-3) did not reach 1e3; since the batched runs
diverged at 14.8 to 25.4 s and the ferry's and tug's batched runs of the first build never did in 60 s, two clean 30 s
runs say nothing about the perturbation's size.

Not explained: the isolated batched pair failed about once in 5 000 calls, but in the solver the first jump came after
90 000 to 150 000 steps, and the ferry's and tug's batches on the same grid had none in 360 000 steps. The mechanism is
the same (the jump changes the whole field in one step, in both members), the rate in the solver is not predicted by
the isolated test.

## Follow up: the scheme's own stability in float32

A broadband burst (0.05 s of white noise at the boat's start) left to decay for 20 s on the `multistatic_b` grid, with
its speed map, land mask and sponge, run alone in each precision (`scratch/f32/decay.py`, `free_decay` in the json).
`free_decay_multistatic_b.png`: look at the float32 curve lying on the float64 curve down to 4.6e-8 of the burst at
7.1 s, and at its three jumps afterwards, each followed by decay at the float64 rate.

So the k space scheme with the harbor's speed map and land mask is as stable in float32 as in float64: nothing grows.
The three jumps (peaks at 7.4, 11.6 and 15.1 s) are a different fault from the batch's: each starts at a few channels and
reaches the rest 27 to 63 steps later, a disturbance travelling out from a point, the signature of the fault of
`gpu_determinism.md`. They were small in absolute terms (peaks of 4.9e-6 to 4.1e-5 of the burst's rms). The four 60 s boat runs
alone (1.44 million steps) had none: their two paths gave identical traces, which a jump in one of them would have
broken. Why the decay run met the fault and the boat runs did not is open; the protection against it is the
comparison with float64 and the rebuild rule of `harbor_float32_scattered.py`, not the run path.
The same float32 decay run made again (after the rebuild below) was not identical to the first: its jumps peaked at
9.1, 11.9, 14.1 and 17.6 s (`free_decay_multistatic_b.png`, the third curve), so the point fault falls at random steps
and is not fixed by the inputs. In both runs the jumps peaked at 1e-6 to 4e-5 of the burst whatever the field's level
just before, a size that does not scale with the field; what sets it was not found.

## A float32 build with the runs alone

What changed in the code: `simulate_sources` runs float32 k space jobs one at a time through `simulate_source` (2D
fields) unless a batch size is given (`runs_alone`); the cache key does not change. The five recordings were built
again in float32 from the same scenario files with only `fdtd.precision` changed, every gather run anew through this
path (fresh caches, `scratch/f32/build32a/`, build files in `build32a/`), not only the boat's two gathers, so that the
table below comes from one build path and its time is comparable with the earlier builds. Compared with the
committed float64 recordings by `harbor_float32_scattered.py`, the criterion and the rebuild rule unchanged
(`scattered.md`, `scattered.json`):

| Recording | Component | rms difference over rms, runs alone | the same, first float32 build (batched) |
|---|---|---|---|
| simple | scattered:boat>bodies | 2.87e-05 | 3.84e-05 |
| simple | scattered:ferry>bodies | 2.91e-05 | 3.85e-05 |
| bistatic | scattered:ferry>uuv1 | 2.29e-04 | 3.28e-04 |
| multistatic | scattered:boat>bodies | 1.24e-04 | 1.56e-04 |
| multistatic | scattered:ferry>bodies | 9.57e-05 | 1.26e-04 |
| multistatic | scattered:tug>bodies | 2.21e-04 | 2.41e-04 |
| multistatic_b | scattered:boat>bodies | 1.22e-04 | diverged |
| multistatic_b | scattered:ferry>bodies | 1.22e-04 | 1.54e-04 |
| multistatic_b | scattered:tug>bodies | 1.49e-04 | 1.70e-04 |
| multistatic_b | direct:boat | 3.01e-05 | diverged |
| harbor | the noise free sum (no components stored) | 4.00e-06 | 4.08e-06 |

The scattered component criterion (every scattered component within 1e-3 of its rms, fixed before the first float32
build): **pass**, the largest 2.29e-4 (`bistatic`). Every direct component is within 1.6e-5 to 5.8e-5 (`scattered.md`
has every row and the worst channel of each). No component was over the limit, so the rebuild rule was not needed and
no gather was built twice. The criterion passing on every component is what the prompt set for a switch of the
default; the switch was not made: every scenario file keeps `precision: "float64"`, the recordings stay the float64
build, `reproduce.py` was not run and no number of `docs/harbor.md` moved. The repository owner decides.

Time of the FDTD for `multistatic_b` (6 runs of 60 s, `build.json`, `fdtd_wall_s`): 1473 s with the runs alone in
float32, 898 s batched in float32 (the build whose boat runs diverged), 6079 s batched in float64; per run per
simulated second 4.1, 2.5 and 16.9 s. Running alone costs 1.6 times the batched float32 build and is still 4.1 times
faster than float64.
The whole build, FDTD only (`build32a/<recording>/build.json` against the first float32 build and the float64 build):

| Recording | Runs | float32, runs alone [s] | float32, batched [s] | float64, batched [s] |
|---|---|---|---|---|
| simple | 4 | 92 | 51 | 356 |
| bistatic | 2 | 155 | 110 | 691 |
| multistatic | 6 | 485 | 297 | 2026 |
| harbor | 12 | 290 | 104 | 468 |
| multistatic_b | 6 | 1473 | 898 | 6079 |
| all five | 30 | 2495 | 1460 | 9620 |

The 5 m `harbor` gains least (12 short runs on a small grid, where batching paid most). The builds after
`multistatic_b` record `git_dirty: true` because this file was being edited during them; the code was commit `a1ded29`
throughout.

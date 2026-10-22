# The float32 build of `multistatic_b`: the boat's batched runs diverge

The float32 build of the five recordings (2026-10-22, the k space waits of `gpu_determinism.md` in) gave every
component of `simple`, `bistatic`, `multistatic` and `harbor`, and the ferry's and the tug's of `multistatic_b`, within
3.3e-4 of the float64 build (`scattered.md`). The boat's two runs on `multistatic_b` (without and with the ten bodies,
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
only, not the 60 s of the recording), and the scheme's stability in float32 with the harbor's speed map and land mask.

The float32 build's logs and build files are in `build32/<recording>/` (the build times: `build.json`,
`fdtd_wall_s`); the recordings themselves were put back to the committed float64 build.

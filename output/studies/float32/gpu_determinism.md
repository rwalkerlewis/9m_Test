# The float32 k space FDTD on the Quadro: runs that differ between processes

Found while measuring section 1 on the GPU (2026-10-22). The float32 k space runs on the 601 x 801 grids (`bistatic`,
`multistatic`, `multistatic_b`) did not always give the same traces: a run in a fresh process sometimes differed from
the same run in another process, at a step that changes from run to run, while repeated runs inside one process were
identical. Two kinds of difference:

* small: of order 1e-7 to 1e-6 of the trace's peak, every channel at once (an error in the k space operator, which
  is global);
* large: 1e-4 to 2e-2 of the peak (once 1.39 in a full length propagation error run), a disturbance that starts at one
  point of the field (a different point each time) within the first steps and travels out at the speed of sound.

Repro (`scratch/f32/fast.py`, not committed): the `bistatic` grid, k space, a Ricker at the hydrophones' centroid,
300 steps, one fresh process per run, the traces compared bit for bit with a reference run. The "keeper" is a second
process running a small matrix product every 5 ms on the same GPU (the desktop also shares this card all the time).
Counts per 60 runs (two rows of 120 where marked):

| Condition | identical | small (below 1e-4 of peak) | large (1e-4 or more) |
|---|---|---|---|
| as committed before (no waits) | 44 | 6 | 10 (7 of them 1.71e-2 from step 13) |
| as committed before, again, traces saved | 36 | 21 | 3 |
| as committed before, keeper running | 33 | 21 | 6 |
| float64, as committed before | 60 | 0 | 0 |
| float64, keeper running | 60 | 0 | 0 |
| leapfrog scheme, float32 | 60 | 0 | 0 |
| the 5 m `harbor` grid (361 x 481), k space float32 | 60 | 0 | 0 |
| a wait before the first step only | 45 | 14 | 1 (6.9e-5) |
| the same with `CUDA_MODULE_LOADING=EAGER` | 46 | 13 | 1 (1.3e-5) |
| CuPy's pinned memory allocator off | 43 | 8 | 9 |
| the FFT plans made before the solver | 46 | 13 | 1 (3.3e-4) |
| the GPU kept busy for 3 s before the run | 59 | 1 | 0 |
| a wait after the inverse FFT, every step (180 runs) | 176 | 4 | 0 |
| waits after both FFTs, every step (120 runs) | 117 | 1 | 2 (8.4e-3) |
| a wait after the step kernel only | 47 | 11 | 2 (8.4e-3) |
| a wait after every operation, keeper running | 59 | 1 | 0 |
| `CUDA_LAUNCH_BLOCKING=1` | 60 | 0 | 0 |
| `CUDA_LAUNCH_BLOCKING=1`, keeper running | 59 | 1 | 0 |
| **committed now** (`FusedFDTD.step` waits after every k space operation; 120 runs) | 119 | 1 | 0 |
| **committed now**, keeper running (120 runs) | 119 | 0 | 1 (1.24e-3 from step 6) |
| the same step in plain CuPy, nothing from this repository, keeper running | 60 | 0 | 0 |
| a plain float32 stencil kernel, cold | 60 | 0 | 0 |
| the k space FFT pair alone in CuPy, cold | 60 | 0 | 0 |

What it says: only float32 on the large grid is affected (float64 none in 120, the smaller grid none in 60); more
contention for the GPU makes it more frequent, a warm GPU less; nothing in the solver's set up (pinned copies, module
loading, FFT plans) is the cause; making the GPU finish every operation before the next one is queued removes most of
it, and waits after every operation of the k space step (committed) bring the large differences from about 1 run in
10 to 1 in 240, not to zero. The same arithmetic written in plain CuPy did not reproduce it. **The cause was not
found.** It is consistent with a fault in how the card or its driver switches between processes (the card's firmware
timed out twice this month, Xid 119 and 154), but that was not shown.

What it costs: the waits make the float32 k space step slower (bistatic, `CUDA_LAUNCH_BLOCKING=1` as the upper bound:
0.43 to 0.69 ms per step; the committed waits: `fdtd_gpu_bench.json`).

What it means for the recordings: a float32 gather can still, rarely, carry such a disturbance. The recordings built in
float32 are therefore compared component by component with the float64 build (`scattered.json`, which lists the direct
components too): a component whose difference is far above the others is read as a run hit by this fault, its gather
is deleted from the cache and rebuilt, and the count is reported. The float64 recordings were not affected in any test.

Two side notes. The identical worst difference 1.26e-5 on `bistatic` and `multistatic_b` in the earlier repro is not a
coincidence: the two scenarios share the water, the hydrophones and the sample points, and that repro ran no bodies.
One leapfrog float32 run (first in its process, 1000 steps) differed from the next two at step 774; a second attempt
did not reproduce it, so it is one observation.

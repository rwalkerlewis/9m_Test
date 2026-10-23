# Checking the GPU on another card

The float32 k space FDTD misbehaved on the development machine's Quadro RTX 4000 in three ways
([Physics, Precision](physics.md#precision)): float32 runs that differ between processes, batched float32 inverse FFTs
that return a corrupted field, and an error state of the card itself during the layout study. Float64 was never
affected. Each finding fits a faulty card or driver as well as a fault in the code, CuPy or cuFFT. One run of
`scripts/gpu_determinism.py` on a different card settles which: it needs the repository and a CUDA card, nothing
else (no recording, no FDTD cache), and writes `results.json` and `run.log`.

## What the script measures

* **(a) Fresh processes.** N processes (20 by default) each run the same k space job in float32 (and 5 in float64): the
  601 by 801 grid of `examples/configs/harbor_bistatic.scenario.json`, a 400 Hz Ricker at the hydrophones' centroid, 5 s
  of simulated time, through the repository's solver. Every process is compared with the median of all of them; a
  process that differs by 1e-4 of the trace's peak or more is a *large* difference. With `--no-waits` a third set runs
  the float32 step without the device waits that `FusedFDTD.step` now has (the condition in which the fault was first
  found).
* **(b) Batched inverse FFTs.** M calls (20 000 by default) of CuPy's `irfft2` over two identical spectra of the same
  grid, in float32 and in float64, each compared with the unbatched result; a member off by 1e-4 of the peak or more
  makes the call *corrupted*.
* **(c) Versions.** The device, its compute capability, the driver, the CUDA runtime, cuFFT and CuPy, in the header of
  `results.json`.

The card is *deterministic* (`results.json`, `deterministic`) when no process of (a) with the committed solver has a
large difference and no call of (b) is corrupted, in either precision.

## Renting a card

Any card with a recent NVIDIA driver will do; an A100 (a datacenter card, full float64 rate) and an RTX 4090 (a desktop
card like the Quadro) are the two worth trying, since they differ from the Quadro in different ways.

* **Lambda Cloud** gives a virtual machine with Docker and the NVIDIA container toolkit installed. Clone the repository
  and use the repository's Docker image exactly as on the development machine: `./dev.sh build`, then the commands
  below prefixed with `./dev.sh run`.
* **RunPod** runs each pod as a container, so Docker cannot run inside it. Either build the image on a machine with
  Docker (`docker build -t <registry>/acoustic-sim:dev .`), push it and start the pod from that image, or start a pod
  from an Ubuntu 24.04 image with CUDA, clone the repository and `pip install -e ".[cuda,dev]"` (the same dependencies
  the Dockerfile installs). Run the commands below from the repository's root.

## The two commands

```bash
# The default: 14 minutes on the Quadro (float64 runs four times slower there), less on a faster card
python scripts/gpu_determinism.py --out output/studies/gpu_determinism/<card>

# The conditions of the original findings: the float32 step without the waits, and 100 000 batched transforms
python scripts/gpu_determinism.py --no-waits --m 100000 --out output/studies/gpu_determinism/<card>_full
```

Bring back the two `results.json` files and their `run.log`.

## Reading the result

On the Quadro (`output/studies/gpu_determinism/results.json`, the default command) 17 of 20 float32 processes differ
from their median by 1e-4 of the peak or more even with the waits, 21 of 20 000 batched float32 transforms are
corrupted, and float64 is deterministic (5 of 5 processes identical, no corrupted call).

* **The other card is deterministic in both commands** (no large difference in (a), with and without the waits, and no
  corrupted call in (b), on the same code and the same CuPy): the faults belong to the Quadro or its driver. Then the
  float32 switch can be decided on the other card's numbers, and the waits in `FusedFDTD.step` and `runs_alone` in
  `water/fdtd_runner.py` can be measured again there.
* **The other card shows the same faults:** they belong to the code, CuPy or cuFFT, and the next step is a plain CuPy
  reproduction outside the repository to report upstream.
* **Float64 shows any large difference or corrupted call on either card:** that is new and contradicts every run so
  far; stop and report it before anything else.

Compare the versions in the two headers first: a different cuFFT or driver is a second difference besides the card.

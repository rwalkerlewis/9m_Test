---
mode: agent
description: Branch forward_fidelity from origin/main. Forward side only: you own src/acoustic_sim/solver/, src/acoustic_sim/water/fd
---
Branch `forward_fidelity` from `origin/main`. Forward side only: you own `src/acoustic_sim/solver/`, `src/acoustic_sim/water/fdtd_runner.py`, `water/dataset.py`, `water/scenario.py`, the scenario files under `examples/configs/`, `examples/harbor/01_datasets/`, `scripts/reproduce.py`'s recording list, `output/reproduce/`, `docs/physics.md` and the "Recordings" and "propagation error" sections of `docs/harbor.md`. You do not edit `water/streaming.py`, `characteristics.py`, `stream.py`, the processing configs or any processing module. This task builds datasets: do not start FDTD runs while another session is building; coordinate through the user.

Read `docs/harbor.md` ("The forward model's own propagation error", "Recordings"), `docs/harbor_history.md` (the time stepping table of the performance work: leapfrog, lw4 and k space phase errors), `src/acoustic_sim/solver/fdtd.py` (`time_scheme`), `src/acoustic_sim/water/fdtd_runner.py`, `examples/harbor/01_datasets/harbor_multistatic_dataset.py`, `harbor_propagation_error.py` (wherever stage 2 put it), `scripts/reproduce.py` and `output/reproduce/summary.json` before changing anything.

## The problem

Stage 2 measured that the simulated water carries the direct paths 1.1 to 5.2 ms early over 135 to 344 m on the 0.5 m and 1 m recordings, direction dependent, and up to 12.7 ms on the 5 m harbor. A period at 760 Hz is 1.3 ms, so at the top of the band the recordings are wrong by more than two cycles over a few hundred metres. The processing is matched to the chart's single sound speed, which is what a deployed system would be; the numbers in `docs/harbor.md` therefore include an error that is the simulator's, not the method's, and the finding that the true waveform located the UUV worse than the estimated one was this error showing up. The performance work measured that the k space time stepping is exact where the grid resolves the wave (0.000 rad on the 0.5 m grid over 100 to 760 Hz; exact to 380 Hz on the 1 m grid) at about ten times the cost per step of the leapfrog.

## 1. Regenerate every recording with the k space scheme

1. Set `time_scheme: "kspace"` in every harbor scenario file (`harbor_simple`, `harbor_bistatic`, `harbor_multistatic`, `harbor_multistatic_b`, `harbor`). Check first that the k space step runs with the sponge boundary, moving bodies on the device, the batched runner and the fused kernel's cache key, and that `tests/harbor/test_performance.py`'s k space test still passes; fix what does not, with a test. The 5 m harbor at 12 to 78 Hz has 4 cells per wavelength at 78 Hz: state what the k space scheme gives there and whether the 5 m grid should also be refined (measure, do not assume).
2. `multistatic_b` becomes 60 s (`total_time`) so that false tracks per minute are measured on more than 20 s; every other recording keeps its length. Record the FDTD wall time per recording in `output/<recording>/build.json`.
3. Run `harbor_propagation_error.py` on every regenerated recording. Criterion, fixed now: the direct path arrival error at every hydrophone is within 0.2 ms on the 0.5 m and 1 m recordings (on the simple harbor, whose water has exactly the chart's speed, within 0.1 ms). Report the 5 m harbor's number and say what it is made of (shallow water slower than the chart's median speed is the physics of that scenario, not a scheme error; separate the two by one run with a uniform sound speed).
4. Rebuild the FDTD caches; the old caches are deleted (say so in the PR), the old recordings are not kept.

## 2. Rerun the system unchanged and report what moved

`scripts/reproduce.py --update` on all recordings with the processing configs exactly as they are. In the PR and in `docs/harbor.md`, the table of every number before and after, with the propagation error beside it. No processing parameter changes. State which movements are explained by the propagation error (the bistatic UUV, the true waveform rows in the history if they are rerun) and which are not.

Then one diagnostic the history asked for and never ran: the bistatic recording given the true waveform and the exact track, now that the propagation is right, beside the everything estimated row; if the true waveform still locates the UUV worse, say so and give the measured reason.

## 3. Docs

`docs/physics.md`: the k space scheme as the default for the harbor recordings, its cost, where it is exact and where it is not (the 5 m harbor), and the measured propagation error per recording. `docs/harbor.md`: the "Recordings" section says every recording is k space and 60 s for `multistatic_b`; the propagation error table is replaced by the new one.

## Rules

The repository rules in `AGENTS.md` apply. Stop after section 3. Report the propagation error per recording against the criterion, the FDTD cost per recording, and every number of `docs/harbor.md` that moved by more than its reproduce tolerance.

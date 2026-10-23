Branch `compute_budget` from `origin/main`. You own `src/acoustic_sim/water/streaming.py`, `stream.py`, `characteristics.py`, `bistatic.py`, `detection.py`, `propagation.py`, the processing configs, `scripts/bench_harbor.py`, `output/studies/compute/` (new), `output/reproduce/` and the compute sections of `docs/harbor.md`. This task changes how fast the chain runs and nothing about what it computes: every Pd, median error and false track count stays within the reproduce tolerance, and any step whose output changes beyond round off is reported and reverted. No FDTD runs.

Read `docs/harbor.md` (The system, Numbers: compute per second of data and latency), `output/reports/run_all.md` (the compute notes of tasks 2, 3 and 6), `src/acoustic_sim/water/streaming.py`, `stream.py`, `characteristics.py`, `bistatic.py` (the ECA, `estimate_reference`), `scripts/bench_harbor.py` and `examples/configs/processing/multistatic.json` before changing anything.

## Where it stands

The chain ran at 0.95 s of compute per second of data on the multistatic recording before stage 2 and runs at 2.1 to 2.3 now (simple 1.1, bistatic 1.5, latency 3.7 s). Since then it gained the characteristics (stage 2), the joint reference (direct path), the predicted map per confirmed track per frame and the Kalman filters (tracker), and the joint removal of every illuminator (the task before this one). None of them was added with a budget and nobody has measured where the time goes. An operator system that falls behind is not a system; the target is 1.0 s per second of data with 2 s latency on the 0.5 m recordings on the Quadro RTX 4000, with the same numbers.

## 1. Profile first

Instrument `HarborPicture` so every stage reports its time per frame: spectra and whitening, the interference projector, the passive map, the passive tracker (gate, predicted maps, filter updates), the reference estimation per illuminator, the ECA, the echo map, the echo tracker, each characteristic, AIS labelling, output writing. Run the multistatic and bistatic tuning recordings and write `output/studies/compute/profile.json` and one bar figure per recording. Report the table before changing anything, and name the three largest items. Also report the GPU memory high water mark and how much of the time is host to device transfer.

## 2. Reduce, in order of the profile

Candidates, each measured alone against the profile and kept only if the numbers do not move: compute the predicted sidelobe maps only when a track's predicted position has moved by more than a quarter cell or every N frames; evaluate the characteristics every fourth frame and hold the values (they are medians over the last 8 frames already); reuse the reference estimate's Gram across the illuminators of one block; share the STFT between the passive and echo chains instead of computing it twice; chunk the ECA so the GPU is never idle waiting on the host; move any remaining host side loop over channels or frames to the device; the mixed precision ECA only where the float32 study showed it faster (it was slower on the Quadro; measure, do not assume). State for each what it saved and whether any number moved.

## 3. Criteria, fixed now

On every 0.5 m and 1 m recording: compute per second of data at most 1.0 and median latency at most 2.0 s on the Quadro RTX 4000; `scripts/reproduce.py --check` passes with no number outside its tolerance (compute and latency are informational to the check, so state them separately); on the 5 m harbor compute does not rise. If 1.0 is not reached, report what remains and its cost per frame, and which of the remaining items would need an algorithm change rather than an implementation change.

## Rules

The repository rules in `AGENTS.md` apply, the figure rules included. Tests: a timing test that fails if any stage exceeds a per frame budget written into the processing config (so a later change that doubles the cost fails CI on the tiny scenario), and parity tests for every reduction (the held characteristic equals the per frame value within tolerance, the shared STFT equals the two separate ones). Update the compute numbers of `docs/harbor.md` through the markers. Stop after section 3. Report the profile before and after, each reduction's saving, and the pass or fail of the criteria.

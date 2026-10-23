Branch `false_tracks` from `origin/main`. You own the tracker classes and `StreamingPassive` in `src/acoustic_sim/water/streaming.py`, `stream.py`, `characteristics.py`, the `tracker` and `echo.tracker` blocks of `examples/configs/processing/*.json`, `output/studies/false_tracks/` (new), `output/reproduce/` and the tracker and numbers sections of `docs/harbor.md`. `bistatic.py` and `StreamingEcho` belong to `joint_removal`, which runs before this task; build on what it merged.

Read `docs/harbor.md` (The tracker, Numbers, the false track bullet), `output/reproduce/summary.json` (`false_track_list` of every recording), `output/reports/run_all.md` (task 3), `src/acoustic_sim/water/streaming.py`, `characteristics.py` and `examples/configs/processing/multistatic.json` before changing anything.

## What is known

False tracks are 3 to 9 per minute on the bistatic and multistatic test recordings and 7 on `multistatic_b` against a criterion of 2. Of the false tracks on the noise seeds, 18 are radiator tracks and 7 echo tracks. Two mechanisms are named in the docs: sidelobes of the loudest ship that its predicted map does not explain, and the passive chain locating a ship's echo off a silent body and confirming it as a radiator (the bistatic UUV is still reported as radiating). The test built for the second never fired: it required coherence of at least 0.7 between the passive peak's source spectrum and the illuminator's reference, and the measured coherence never exceeded 0.38. The threshold was set before anyone measured what coherence an echo shows. The boat on `multistatic_b` is tracked in 70 % of frames against 0.8; it stops moving at about 33 s of the 60 s record and no one has said whether the track is lost then.

## 1. Measure, then set the thresholds

On the tuning recordings, with the truth used only to label (a diagnostic), classify every peak that started a radiator track, confirmed or not, and every confirmed false track: within 5 m of a radiating object; within 5 m of a silent body (an echo located passively); explained by a confirmed track's predicted map at a fraction between 0.3 and 0.5 (near miss of the sidelobe test); on the direct path between a ship and the cable; elsewhere. Measure for each class the coherence with the nearest illuminator's reference, the ratio of the peak to the predicted map of the nearest confirmed track, the peak's half maximum widths, and its persistence. Plot each quantity's distribution per class (one figure per quantity). From these distributions set, on the tuning recordings only, the coherence threshold (or replace coherence with whichever quantity separates echoes from radiators best, stating the measured separation) and the sidelobe fraction. Record the chosen values and the separation they give before running the test recordings.

## 2. The boat on `multistatic_b`

Plot the boat's true track, its radiator track's fixes and the frames in which it is detected against time. State whether the losses are before, during or after it stops at 33 s, and the cause (the Kalman filter's velocity holding the prediction moving past a stopped ship; the sidelobe test explaining a stationary ship by another's predicted map; the ferry, also stationary from 45 s, sharing the map). Fix it only if the fix is a tracker rule that applies to every recording (for example process noise that admits a stop, or the static decision feeding the gate), tuned on the tuning recordings, and report before and after.

## 3. Run

`scripts/reproduce.py --update`. Criteria, fixed now, on the test recordings: false tracks at most 2 per minute on `multistatic_b` and at most 3 on every 20 s recording (one track; the record length makes 2 unreachable there and the standing criterion is kept for the 60 s record); the bistatic UUV reported as seen by echo only, not both ways, in at least 90 % of its frames; the boat on `multistatic_b` at least 0.8; no Pd falls by more than 3 points; compute per second of data not above its current value.

## Rules

The repository rules in `AGENTS.md` apply, the figure rules included. Tests: the peak classification on a synthetic map with a known radiator, a known echo and a known sidelobe; the chosen test on synthetic spectra with known coherence; the stop behaviour on a synthetic track that halts. Update the tracker section and the `picture.jsonl` schema if a field changes. Stop after section 3. Report the measured distributions (the separation each quantity gives), the thresholds chosen and why, the boat's cause, and the pass or fail of each criterion.

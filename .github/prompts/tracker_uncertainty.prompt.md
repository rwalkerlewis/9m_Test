---
mode: agent
description: Branch tracker_uncertainty from origin/main. Processing side only: you own src/acoustic_sim/water/streaming.py, stream.p
---
Branch `tracker_uncertainty` from `origin/main`. Processing side only: you own `src/acoustic_sim/water/streaming.py`, `stream.py`, `characteristics.py` (the uncertainty fields), `scoreboard.py` and `evaluate.py` (the scoring change in section 3), `examples/configs/processing/*.json`, `output/reproduce/` and the system, numbers, tracker and operator output sections of `docs/harbor.md`. You do not edit the solver, the datasets, the scenario files or the noise model. If `forward_fidelity` merges while you work, rebase and rerun the measurements on the regenerated recordings before opening the PR; whichever of the two branches merges second reruns `scripts/reproduce.py --update`.

Read `docs/harbor.md` (The system, Numbers on recordings the parameters never saw, Criteria, What the objects are), `src/acoustic_sim/water/streaming.py` (`RadiatorTracker`, `TrackerConfig`, `RadiatorTrack`, the exclusion discs in `StreamingPassive` and `StreamingEcho`), `src/acoustic_sim/water/stream.py`, `characteristics.py`, `examples/configs/processing/*.json`, `scripts/reproduce.py`, `tests/harbor/test_streaming.py` and `output/reproduce/summary.json` before changing anything.

## What stage 2 measured and what it points at

1. The tug on `multistatic_b` is tracked in 18 % of frames although its map peaks are within 5 m in 42 %: 6 fixes in 2 s inside a fixed 6 m gate are not met while its peaks come and go behind the louder ferry and boat, and no track confirms until 15 s. The tracker has no covariance, no process noise and no measurement noise; its gate, confirmation and drop rules are hand constants.
2. False tracks are 2 to 12 per minute against a criterion of 2. Three kinds: sidelobes of the loudest ship at 41 to 45 m, beyond the 30 m exclusion disc that stands in for the beam pattern; the passive chain locating a loud ship's echo off a silent body (the simple harbor's wreck, the bistatic UUV) and the radiator tracker confirming it, which also makes the characteristics report a silent body as "seen both ways"; and an echo track on the end of the 30 m moored hull counted false because the scoring measures to the hull's centre.
3. The operator output's position uncertainty is the map peak's half maximum width (14 to 47 m along the line of sight on the strain cable) while the fixes are within 1 to 3 m of the truth: the width is the map's resolution, not the track's error, and nothing in the output says how far the track can be trusted.

## 1. Constant velocity Kalman filter per track

Replace the straight line fit in `RadiatorTrack` by a constant velocity Kalman filter: state position and velocity; process noise a white acceleration with a value per tracker in the processing config (0.05 m/s² default for radiators, 0.2 for echo tracks); measurement noise per fix from the half maximum width of the map peak that produced it (`characteristics.py` measures it), with a floor of half a grid cell. The gate is the Mahalanobis distance at a stated probability (chi squared, 2 degrees of freedom, 0.99). The prediction and its covariance feed the echo chain's illuminator position, the "seen both ways" merge and the operator view's ellipse; a track is static when its velocity is within its own covariance of zero for 10 s. Confirmation and drop stay as counts and times in the config and now apply to gated fixes.

## 2. Sidelobes from the beam pattern, not from a radius

For every confirmed track, every frame, compute the map the array would produce for that track alone (the layout's response to a point source at the predicted position through the same propagator and the same whitening weights as the data map; the GPU replica kernels already do this). A data map peak is explained by a track if the predicted map at that cell, scaled to the track's own peak, exceeds a stated fraction of the data map there (0.5 default, in the config). Explained peaks start no tracks and are not fixes for other tracks. Remove `exclusion` and `strong_fraction` from the radiator `TrackerConfig`: an illuminator is any confirmed radiator track whose peak is above the noise floor by a stated margin. In the echo chain, replace the 25 m disc around each illuminator by the same test against the illuminator's predicted direct path map. Measure the compute per second of data before and after; the chain was at 1.2 to 1.5 s per second of data after stage 2, so report what this adds and what in the characteristics could be computed every fourth frame instead of every frame to get under 1.0.

## 3. The passive chain locating an echo

A passive peak that coincides, within the gate, with an echo map peak of a confirmed illuminator, and whose least squares source spectrum is coherent with that illuminator's data reference above a stated level (0.7 over the band, in the config), is an echo, not a radiator: it is handed to the echo tracker and does not start a radiator track. Report how many of the stage 2 false radiator tracks this removes on each recording.

Scoring change, stated in the docs: a track is matched to a static body by the distance to the body's outline (the mask the truth file carries), not to its centre, so a track on the end of the 30 m hull is a hit. Moving bodies keep the centre.

## 4. Measure

`scripts/reproduce.py` on every tuning and test recording. Report before and after: Pd per object, median error, false tracks per minute, compute per second of data, and the calibration of the uncertainty: the fraction of fixes whose true position lies inside the track's 95 % ellipse (it should be near 0.95; this is the test of whether the covariance means anything). Confirmation count and drop time may be changed once, before the test recordings are run, on the tuning recordings only; record the change.

Criteria, fixed now, on the test recordings: the tug on `multistatic_b` tracked with Pd at least 0.6; false tracks per minute at most half of stage 2's value on every recording and at most 2 on `multistatic_b`; no object's Pd falls by more than 3 percentage points; the 95 % ellipse contains the truth in 85 to 99 % of fixes.

## Rules

The repository rules in `AGENTS.md` apply. Tests: the filter on a synthetic straight track with known noise (state error against covariance, gate acceptance rate), a turning track, the sidelobe test on a synthetic two source map (the weak true source survives, the strong source's sidelobe does not), the echo coherence test, the static decision, the outline scoring. Update the `picture.jsonl` schema (covariance fields) and `output/reproduce/summary.json` with `--update`. Stop after section 4. Report the numbers, the pass or fail of each criterion, the ellipse coverage and the compute per second of data.

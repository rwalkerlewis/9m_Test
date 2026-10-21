---
mode: agent
description: Branch tracker_uncertainty from origin/main. Processing side: you own the tracker classes and StreamingPassive in src/ac
---
Branch `tracker_uncertainty` from `origin/main`. Processing side: you own the tracker classes and `StreamingPassive` in `src/acoustic_sim/water/streaming.py`, `stream.py`, the uncertainty fields of `characteristics.py`, `scoreboard.py` and `evaluate.py` (the outline scoring), the `tracker` and `echo.tracker` blocks of `examples/configs/processing/*.json`, `output/reproduce/` and the system, numbers and tracker sections of `docs/harbor.md`. Another session (`direct_path`) owns `bistatic.py` and `StreamingEcho`; do not edit them. Whichever of the two merges second rebases and reruns `scripts/reproduce.py --update`.

Read `docs/harbor.md` (The system, Numbers, Criteria, Before and after the k space recordings), `src/acoustic_sim/water/streaming.py`, `stream.py`, `characteristics.py`, `examples/configs/processing/*.json`, `scripts/reproduce.py`, `tests/harbor/test_streaming.py` and `output/reproduce/summary.json` before changing anything.

## What the k space recordings now show

With the forward model's timing error gone the ships are tracked (ferry 100 %, tug 86 to 100 %, boat 80 to 100 %) and uuv2 is tracked by its echo in 83 to 89 % of the multistatic frames. What remains on the tracker's side:

1. False tracks, 2 to 6 per minute on every test recording against a criterion of 2: sidelobes of the loudest ship at 41 to 45 m, outside the fixed 30 m exclusion disc that stands in for the beam pattern; the passive chain locating a loud ship's echo off a silent body (the simple harbor's wreck, the bistatic UUV) and the radiator tracker confirming it, which also makes the characteristics report the body as "seen both ways"; and an echo track on the end of the 30 m moored hull counted false because the scoring measures to its centre.
2. The simple harbor's wreck fell from 100 % to 67 to 98 % after the recordings were regenerated although its echo peak is within 5 m in 99 % of frames: its echo track loses continuity while a confirmed radiator track of the ferry's echo sits 6 m from it and echo peaks inside that track's 25 m exclusion are dropped. The docs call this a likely mechanism, not a measured cause; measure it.
3. The track has no covariance: gate, confirmation and drop are hand constants, and the operator output's uncertainty is the map's half maximum width (14 to 47 m along the line of sight on the strain cable), not the track's error (1 to 3 m).

## 1. Constant velocity Kalman filter per track

Replace the straight line fit in `RadiatorTrack` by a constant velocity Kalman filter: state position and velocity; process noise a white acceleration per tracker in the config (0.05 m/s² radiators, 0.2 echo tracks); measurement noise per fix from the half maximum widths of the map peak that produced it, with a floor of half a grid cell. Gate: Mahalanobis distance at a stated probability (chi squared, 2 degrees of freedom, 0.99). The prediction and covariance feed the echo chain's illuminator position (through the existing interface; do not change `StreamingEcho`), the "seen both ways" merge and the operator view's ellipse; static when the velocity is within its covariance of zero for 10 s. Confirmation and drop stay as counts and times, now on gated fixes.

## 2. Sidelobes from the beam pattern, not from a radius

For every confirmed radiator track, every frame, compute the map the array would produce for that track alone (the layout's response to a point source at the predicted position through the same propagator and whitening weights as the data map). A data map peak is explained by a track if the predicted map at that cell, scaled to the track's own peak, exceeds a stated fraction of the data map there (0.5 default, in the config). Explained peaks start no tracks and are not fixes for other tracks. Remove `exclusion` and `strong_fraction` from the radiator `TrackerConfig`; an illuminator is any confirmed radiator track whose peak is above the noise floor by a stated margin. Expose the predicted map per track so the echo chain can replace its 25 m disc with it later (that change belongs to `direct_path`). Measure the compute per second of data before and after.

## 3. The passive chain locating an echo

A passive peak that coincides, within the gate, with an echo map peak of a confirmed illuminator, and whose least squares source spectrum is coherent with that illuminator's reference above a stated level (0.7 over the band, in the config), is an echo: it goes to the echo tracker and starts no radiator track. Report how many of the current false radiator tracks this removes per recording, and whether the wreck's continuity (item 2 above) is restored; if it is not, find and report the cause.

Scoring: a track is matched to a static body by the distance to the body's outline from the truth file's mask, not its centre; moving bodies keep the centre. State it in the docs.

## 4. Measure

`scripts/reproduce.py` on every recording. Report before and after: Pd per object, median error, false tracks per minute, compute per second of data, and the fraction of fixes whose true position lies inside the track's 95 % ellipse (near 0.95 if the covariance means anything). Confirmation count and drop time may change once, before the test recordings are run, on the tuning recordings only; record it.

Criteria, fixed now, on the test recordings: false tracks per minute at most half of the current value on every recording and at most 2 on `multistatic_b`; the simple harbor's wreck at least 0.9 on every test seed; no object's Pd falls by more than 3 percentage points; the 95 % ellipse contains the truth in 85 to 99 % of fixes.

## Rules

The repository rules in `AGENTS.md` apply, the figure rules included: every new figure is one panel through `save_figure`. Tests: the filter on a synthetic straight track with known noise (state error against covariance, gate acceptance), a turning track, the sidelobe test on a synthetic two source map, the echo coherence test, the static decision, the outline scoring. Update the `picture.jsonl` schema (covariance fields). Stop after section 4. Report the numbers, the pass or fail of each criterion, the ellipse coverage and the compute per second of data.

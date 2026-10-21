Work on `main` of 9m_Test. A review of the latest merge (PR #24, harbor picture stage 1) found the problems below. Fix them in the order given, one commit per numbered item, with a commit message that says what was wrong and what changed. Run `pytest tests/harbor tests/test_contract.py` before each commit and the full suite at the end. Do not change any result in the docs by hand: where a fix changes a number, rerun the script that produces it, commit the new figures and JSON, and update the docs from the new output.

Ground rules that still apply: the processing side may use only the sensor traces, the geometry of the area and the constant sound speed. Anything taken from the truth file must be recorded in the run's provenance and said in plain words on the figure and in the docs ("given the true ship track", not a label). Never skip, weaken or delete a test to get green.

## 1. Passive replicas ignore strain mode

`MatchedFieldScanner` in `src/acoustic_sim/water/detection.py` builds replicas as `green2d(k r)` at the channel centres (`layout.receivers`) and never applies `SensorLayout.collapse`. `StreamingPassive` in `streaming.py` (lines 96 and 242) uses the same geometry. On a strain cable (the bistatic and multistatic scenarios) the data carry the second difference over the gauge length, so every passive map from those recordings uses the wrong replica. The echo chain (`_channel_greens`, around line 547) already does this correctly; follow it.

- Make the scanner and `StreamingPassive` evaluate the Green's function at `layout.sample_points` and collapse to channels, on CPU and GPU, including the fast phase recurrence in `_greens` and the CUDA `_greens_kernel`. Pressure layouts must give exactly the same replicas as before.
- Add tests: on a strain layout the scanner replica equals `collapse(green2d)` at the sample points; a passive map of a synthetic point source on a strain cable peaks at the source and its peak is higher than with the old replica; the GPU passive kernel matches the CPU (skips without a GPU, like the existing GPU tests).
- Rerun every passive result made on a strain recording (`harbor_stream`, `harbor_detect`, stage 1, the walkthrough for bistatic and multistatic) and update the docs tables from the new output.

## 2. The run given the true waveform and exact track maps the UUV worse than the run with everything estimated

`examples/harbor_picture_stage1.py` lines 24 to 26 and `prompts/harbor_picture_stage1.md` line 65 call the `true_waveform_exact_track` configuration an upper limit. The committed results contradict that: UUV Pd 0.02 against 0.30 for `all_estimated`, map peak within 5 m of the UUV in 26 % of frames against 77 %, although the direct path is removed deeper (21 dB against 11 dB). Find out why before changing any wording. Likely places to look: the scale or delay of the true waveform against the reference the chain builds from the traces, the frame alignment of the true waveform, the sign or conjugation of the reference in the coherent map, and whether the deeper removal also removes part of the echo. Report what you found. If it is a bug, fix it, add a test that would have caught it, and rerun stage 1. If it is not a bug, explain the mechanism in the docs and drop the words "upper limit".

## 3. The occupancy scoreboard does not hold its false alarm rate

`occupancy.py` lines 85 to 87 renormalise the occupancy map to sum 1 every frame. `harbor_picture_stage1.py` lines 368 to 370 multiply it by the number of cells and score it with a threshold from `set_threshold` (`scoreboard.py:161`), which is set on runs with one object removed. In those runs the freed mass moves onto noise peaks, so the false alarm rate in the full scene is not the rate the threshold was set for, and a weak object's Pd depends on how loud the others are. Criterion 1 (ferry, tug and boat Pd at least 0.9) is scored on this map. Either score occupancy on an unnormalised quantity (for example the per frame likelihood product before normalisation, or the log odds), or measure the false alarm rate the occupancy threshold actually gives on the full scene and report it beside Pd. State in the docs which scoreboard is calibrated. Add a test that sets the threshold on a scene and checks the measured false alarm rate on an independent noise run of the same scene is within a stated tolerance of the target.

## 4. False tracks are not counted or shown

`harbor_stream` produces tracks that match no object: `r19` in the simple harbor (18.4 m from every object) and `r27` in the bistatic recording (39 m). In `output/walkthrough/simple/07_harbor_picture.png` `r19` sits on the wreck and is labelled "(no AIS)", so it reads as an object radiating without AIS; most likely it is the wreck's echo of a ship tracked as a radiator. The step 7 table scores only true objects.

- Score every track: each one is matched to the nearest object within the association radius or counted as a false track. Report the false track count and their positions in the step 7 table, the JSON and the docs.
- Draw false tracks differently on the step 7 figure and label them as unmatched.
- Find out whether `r19` is the wreck's echo. If it is, say so in the docs. If a simple test can tell an echo from a radiator (for example coherence with a known ship's reference at the bistatic delay), propose it in your report but do not build it in this pass.

## 5. Docs that contradict the results

- `docs/harbor.md` line 417 says the second UUV and every silent body in the multi illuminator scene are "not detected anywhere". The real time table below it gives uuv2 14 % and uuv1 6 % on the same recording. Make the sentence say which run it describes.
- `examples/walkthrough/01_recording.py` docstring and docs step 1 say "a strain DAS cable"; the simple scenario's cable is pressure mode. Say which cable each recording uses.
- `docs/harbor.md` line 21 lists `run_all.sh <bistatic|multistatic|harbor>` and leaves out `simple`, and the escaped `&lt;` `&gt;` show literally. Fix both.

## 6. Comments in `streaming.py` that do not match the code

- Line 713 says direct paths are removed strongest track first. The loop follows insertion order of `self.radiators`. Either sort by the track's estimated level before removal (and check the result does not get worse on the four recordings) or change the comment to the actual order. Say which you did and why.
- The module docstring says the chain is causal frame by frame. Channel QC (line 143), common mode removal and the noise floor use the later frames of the same block. Say "causal per block of N frames" with the actual block length, or make those steps use only past frames.
- Lines 155 and 181: warmup frames go into `Xc` without the channel weights and later frames with them, and recomputed weights are never applied to older frames. Store unweighted spectra and apply the current weights when they are used, and add a test that a channel zeroed by QC contributes nothing to the re whitening floor from the first block on.

## 7. Small fixes

- `examples/walkthrough/07_harbor_picture.py`: `others` (line 39) and the `cols` argument of `score()` (line 31) are unused, so a detection that belongs to a neighbouring object counts as a hit. Assign each detection to its nearest object first, then score. This matters for the pilings 2 to 5 m apart. Rerun step 7.
- `scoreboard.py:97`: every cell of a flat region counts as a local maximum, so one flat false peak counts as several false alarms. Count each connected flat maximum once, and guard the threshold when every frame maximum is minus infinity or zero. Add a test for both.
- `bistatic.py` lines 324 to 341: for a free field propagator the replica wavenumber uses `bc.c` and ignores `propagator.c`. Use one sound speed: take it from the propagator, or raise if the two differ. Add a test.
- `bistatic.py` lines 437 and 479 mark a ship present 60 s before its first AIS report. Leave the batch behaviour but make the streaming chain use only reports already received, and note in the docstring that the batch version looks ahead.

## 8. Tests that cannot fail or do not test the path that runs

- The four tests in `tests/test_3d_extension.py` return a bool instead of asserting, so they pass whatever happens. Change them to assert, and fix anything that then fails.
- `test_streaming.py` feeds the true ship track to the echo chain (line 80). Add an end to end test of the path `harbor_stream.py` actually runs: `StreamingPassive` to `RadiatorTracker` to `StreamingEcho`, on a small synthetic scene with one ship and one silent scatterer, checking the scatterer is located within a stated distance with no truth passed in.
- Add tests for the paths no passive test reaches: `_adaptive`, `project_out_ships`, and exclusion zones built from AIS reports.

When done, report for each item: what was wrong, what you changed, which numbers moved and by how much (old against new), and anything you found but left alone. Push to a branch named `review_fixes` and open a PR against `main`.
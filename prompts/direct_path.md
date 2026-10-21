Branch `direct_path` from `origin/main`. Processing side: you own `src/acoustic_sim/water/bistatic.py` (`make_reference`, `eca_stage`, `eca_remove`, the reference options of `BistaticConfig`), `StreamingEcho` in `streaming.py`, the `echo` and `bistatic` blocks of `examples/configs/processing/*.json`, a new study folder `output/studies/direct_path/`, and the direct path sections of `docs/harbor.md`. Another session (`tracker_uncertainty`) owns the tracker classes and `StreamingPassive`; do not edit them. Whichever merges second rebases and reruns `scripts/reproduce.py --update`. No FDTD runs.

Read `docs/harbor.md` (The system, Numbers, Before and after the k space recordings, the true waveform diagnostic), `docs/harbor_history.md` (stage 0a cancellation depth against range, stage 0b ablation), `src/acoustic_sim/water/bistatic.py`, `StreamingEcho` in `streaming.py`, `examples/harbor/90_history/harbor_cancellation_depth.py`, `examples/configs/processing/multistatic.json`, `output/reproduce/summary.json` and `output/recordings/*/levels.json` (or `output/harbor_*/fine/levels.json` if the output folder has not been restructured yet) before changing anything.

## The limit

A silent body is seen only by the ship noise it reflects, 8 to 18 dB below the ship's direct sound on the median cable channel (`levels.json`). The direct path is removed by least squares fitting a reference waveform to every channel. With the reference estimated from the traces (a delay and sum beam of the cable toward the tracked ship) the removal reaches about 11 dB on the ferry and 3 to 7 dB on the tug and the boat; given the true waveform and the exact track it reaches 22 dB. The gap is the reference, and it is why uuv1, the diver, the pilings and the moored hull are not tracked on the multistatic recordings while uuv2, which sits where the ferry's residual is weakest, is. Stage 0a measured on the stationary ferry: 8.8 dB at 193 m with the beam against 34 dB with the true waveform, and the beam's coherence with the true waveform fell from 0.94 at 300 Hz to 0.16 at 100 Hz because a strain channel's response goes as frequency squared, so the low end of the band is noise dominated in a plain average.

The true waveform diagnostic also showed that a reference weighted by the received spectrum maps the echo better than the emitted waveform does: the goal is the ship's sound as the array receives it, estimated with the least noise, not the emitted signal.

## 1. Reference estimators, measured on the cancellation depth framework

Add `BistaticConfig.ref_mode` with these estimators of the reference waveform from the traces, and score every one on the stage 0a cases (the ferry signature alone at 193, 99 and 29 m, stationary and moving at 4 m/s; the cached gathers exist, this is processing only; `harbor_cancellation_depth.py` moves back from history into `examples/harbor/05_detection/` as the study script and writes `output/studies/direct_path/`):

1. `beam`: the current delay and sum (the baseline).
2. `weighted`: each channel and frequency weighted by its modelled response to the ship (the second difference of the Green's function along the cable for strain channels, the Green's function for pressure channels and hydrophones) over its noise power estimated from a first pass residual; the maximum ratio combination. The response comes from the chain's propagator at the tracked position; no truth.
3. `hydrophones`: the four pressure sensors alone, and `weighted` including them.
4. `iterated`: `weighted`, then ECA, then the modelled echo of each confirmed echo track subtracted from the traces, then the beam again; one iteration.
5. `joint`: the reference and the per channel direct path solved together as one least squares problem over a block (the direct path of every channel is a delayed, scaled copy of one unknown waveform; solve for the waveform and the per channel scales jointly, with the delays from the track), which is the formulation a joint source and reflectivity inversion would start from. State its cost.

Report, per mode, per case: median and 10th percentile ECA depth over live cable channels, coherence of the reference with the true waveform against frequency (a diagnostic, from the truth, labelled), and compute time per second of data. Criterion, fixed now: at least one mode with no truth input reaches a median depth of 20 dB on the stationary source at 193 m and 99 m, and 15 dB on the moving source at 193 m with the tracked position. The best mode by the mean over the six cases is the default of every processing config.

## 2. The chain with the new reference

Run `scripts/reproduce.py` with the best mode on every recording. Report before and after: Pd per object (the silent bodies above all: uuv1, diver, pilings, moored hull), echo track false tracks per minute, the direct path removal depth per illuminator per recording (from the truth components, a diagnostic, labelled), compute per second of data. The exclusion disc around each illuminator stays at 25 m in this task; `tracker_uncertainty` exposes the predicted map that replaces it later.

Criteria, fixed now, on the test recordings: uuv1 tracked with Pd at least 0.5 on the multistatic seeds; at least three of the seven static bodies (pilings, moored hull) tracked on `multistatic_b`; no object's Pd falls by more than 3 percentage points; compute per second of data at most 1.5 on the 0.5 m recordings.

## 3. Docs

`docs/harbor.md`: the direct path step describes the chosen estimator in plain words; the cancellation depth table per mode in a "Direct path removal" subsection; the walkthrough's step 5 figures regenerated (one panel each). The size criterion of stage 2 (hull below piling on the size indicator) is evaluated if the hull and a piling are now tracked; report it either way.

## Rules

The repository rules in `AGENTS.md` apply, the figure rules included. Tests: each reference mode on an analytic free field case with a known waveform (coherence above a stated value, depth above a stated value), the joint solve recovering a known waveform and scales from noise free synthetic blocks, and the chain end to end with the new default. Stop after section 3. Report the depth table, the pass or fail of each criterion, and which silent bodies are now tracked.

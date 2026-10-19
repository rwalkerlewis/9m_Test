Work in the 9m_Test repo on branch `blind_bistatic` (`git fetch origin && git checkout blind_bistatic`; it starts from `main` after PR #21). Use the GPU (`--cuda`) for every FDTD run, with CPU fallback.

Read `docs/harbor.md` (especially "Stage 0 of the multistatic work"), `src/acoustic_sim/harbor/bistatic.py`, `examples/harbor_cancellation_depth.py`, `examples/harbor_bistatic_ablation.py`, `examples/harbor_bistatic_dataset.py`, `prompts/multistatic_stage0.md` and the stage 0 outputs in `output/harbor_cancel/` and `output/harbor_bistatic/ablation/` before changing anything.

## Goal

Make the single illuminator, single target bistatic chain work blind (data reference, AIS track, ECA) and measure it honestly. Stage 1 does not start multistatic fusion, static scene imaging or multi target extraction.

Stage 0 found:

1. With a stationary source, the blind (`data`) reference limits ECA: 9 dB at 193 m against 34 dB with the oracle waveform. The beam is noise dominated at the low end of the band, because strain response goes as frequency squared.
2. Fixed lag ECA cannot cancel a moving ship. `eca_doppler` with the exact track reaches 21 to 25 dB. With the AIS refined track it reaches 15 to 23 dB, and only after a 4 s straight line fit (`track_smooth_s`).
3. `blind+doppler+smooth4` on the fine dataset reached 58 % of frames within 5 m. The 4 s window was chosen after seeing that one dataset (one ferry track, one noise seed), so it is a tuned result, not a validated one.
4. The echo contrast metric (map peak over map median) reads 16 to 19 dB whether the location is 2 m or 90 m off. It carries no information and is dropped as a criterion.
5. p90 error is about 205 m in every Doppler configuration: frames are either close or far off. Where the far frames go was not examined.
6. Without smoothing, `eca_doppler` turns the per frame jitter of the refined position into velocity, and that hurts even a stationary source: the 29 m stationary oracle case drops from 37 dB to 6 dB.

Repo convention as before: every experiment has its pass/fail criterion in the script docstring and `results.json` before it runs, and criteria are never changed after a result is seen. Any parameter tuned in this stage is tuned only on the development data defined in stage 1c and reported on the held out data.

## Stage 1a: better blind reference

Extend `make_reference` / `BistaticConfig` with reference options and score them in the stage 0a framework (`examples/harbor_cancellation_depth.py`, add a `--reference-mode` flag or a loop over modes; the FDTD gathers are cached, so this is processing only):

1. `ref_channels="hydrophones"` (already supported): pressure sensors with a flat response.
2. Strain equalisation: in the frequency domain, divide each strain channel by its modelled response to a plane wave from the ship direction (the second difference of the Green's function along the cable, from `ReplicaModel`), with a floor so nulls are not amplified, before the delay and sum.
3. Per frequency SNR weighted beam: weight each channel and frequency by its estimated signal to noise ratio (signal from the beam, noise from the residual after a first ECA pass), as in a maximum ratio combiner.
4. Iterated reference: build the reference, run ECA, form the map, subtract the modelled echo of the strongest map peak from the traces, rebuild the reference. One iteration only.

Report the ECA depth (median and 10th percentile over live cable channels) for every mode, every range, stationary and moving (moving with `eca_doppler` and the stage 1b tracker once it exists; stationary without Doppler).

Criterion, fixed now: at least one mode reaches a median depth of 20 dB with no oracle input for the stationary source at both 193 m and 99 m. Report the best mode and use it in the rest of stage 1.

Outputs in `output/harbor_reference/`: `depth_by_mode.png`, `coherence_by_mode.png` (coherence of each reference with the true waveform against frequency), `results.json`, `run.log`.

## Stage 1b: track for the Doppler ECA

Replace the straight line fit as the default with a constant velocity Kalman filter (and smoother over the record, since this is post processing) on the AIS reports (position, speed and course, with the `AISConfig` error statistics as measurement noise) plus the acoustically refined positions per frame (measurement noise estimated from the refinement peak width or fixed at the stage 0 observed jitter of about 1 m). Fix the process noise from a stated ship acceleration level before running anything. Keep `track_smooth_s` as an option.

Add a low speed fallback: when the smoothed speed is below a threshold (for example 0.3 m/s), the ECA uses fixed lags at the smoothed position instead of per sample delays.

Criterion, fixed now: on the stage 0a cases, the AIS refined track with the Kalman smoother and the fallback reaches within 3 dB of the `oracle_exact` depth for every moving case and within 3 dB of the non Doppler `oracle` depth for every stationary case (all with the oracle waveform, so the track is isolated from the reference).

Outputs in `output/harbor_track/`: track error against time per case, depth table, `results.json`, `run.log`.

## Stage 1c: detection measured honestly

### Data

1. **Development:** the existing fine bistatic dataset (ferry track 1, noise seed 0). All tuning happens here.
2. **Held out noise:** the same cached gathers mixed with noise seeds 1 to 5 (`assemble_dataset(seed=...)`, no new FDTD).
3. **Held out geometry:** a second ferry track (different heading, crossing at a different bistatic angle to the UUV, still inside the open water region) with the same UUV, in `examples/harbor_bistatic_track2.scenario.json`. This needs two new fine FDTD runs (direct, and with the body), mixed with seeds 0 to 2.

### Metric

The current `bistatic_map` divides the whole map by its maximum, which makes any threshold relative to that dataset. Add an option to return the unnormalised map (the `coherent_norm="snr"` map is the natural choice, since its floor is flat over the grid) and use it for detection.

1. **Body free runs:** the same traces with the scattered component removed (`scatter_gain={"uuv1": 0}`, or the components without the echo), processed by the identical chain.
2. **Threshold:** set the threshold T per configuration so that in the body free runs a fraction 0.05 of frames have any cell outside the ship exclusion above T (frame false alarm rate 0.05). Use the development data only.
3. **Pd:** the fraction of frames in which the map peak exceeds T and lies within 5 m of the UUV truth.
4. Also report median error, within 5 m, and the 1.5 s integrated versions as before. Do not report peak over median contrast as a criterion.

### Where the misses go

For every frame with an error over 20 m, classify the peak location: within 10 m of the ship exclusion edge, within 10 m of a first order quay image of the ship (`clutter_images`), along the direct path between ship and cable, or elsewhere. Plot the miss locations on the map coloured by class, and report the counts. If one class dominates, state the likely cause; do not fix it in this stage unless the fix is a one line exclusion and the result is reported both with and without it.

### Configurations

Run, on development and both held out sets:

1. `oracle` (stage 0 baseline, unchanged),
2. `eca_oracle+doppler` with the exact track,
3. blind: best stage 1a reference, stage 1b tracker with fallback, `eca_doppler`,
4. blind with the stage 0 settings (`data` reference, `track_smooth_s=4`), for comparison.

Criterion, fixed now: configuration 3 reaches Pd at least 0.5 at the 0.05 frame false alarm rate, averaged over the held out noise seeds, and at least 0.5 averaged over the held out geometry runs. Report development and held out numbers side by side; a large drop from development to held out is itself a finding.

Script: `examples/harbor_bistatic_detect_eval.py`. Outputs in `output/harbor_bistatic/eval/`: `pd_by_config.png` (development, held out noise, held out geometry), `misses_map.png`, `error_vs_time.png`, `results.json`, `run.log`, plus the held out geometry dataset figures (`map.png`, `components.png`, `levels.json`) under `output/harbor_bistatic/fine_track2/`.

## Rules

1. Figures, JSON results and logs under `output/` are committed. Datasets (`*.npz`, `*.npy`) and FDTD caches stay gitignored.
2. Add tests under `tests/harbor/` for every new function (reference modes on a small analytic case where the answer is known, the Kalman smoother on a synthetic straight track with known noise, the low speed fallback, the unnormalised map option, the threshold and Pd computation). Keep `pytest tests/harbor` passing.
3. Update `docs/harbor.md` with a "Stage 1" section: measured numbers, pass/fail of each criterion, what did not work, and a row for each new script in the examples table.
4. Commit in small steps on `blind_bistatic` and open a pull request against `main` with a summary table of the stage 1a, 1b and 1c results.
5. Stop after stage 1c. Report the numbers, the pass/fail of each criterion, the dominant miss class, and whether the blind chain is ready to be repeated per illuminator for multistatic fusion.

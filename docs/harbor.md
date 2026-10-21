# Harbor: locating every object in a harbor from DAS and hydrophone traces

A plan view harbor in which a fibre optic cable (DAS) and hydrophones record ships and silent bodies. The system
locates every radiating object from the traces alone, tracks it, removes the ships' direct sound and tracks the silent
bodies by the ship noise they reflect, measures what each object is (where, how fast, how loud, how big against the
wavelength), and writes one picture of the harbor per quarter second. AIS reports only label what was found. The
recordings are synthetic (the package's FDTD forward model, `src/acoustic_sim/water/`); the processing cannot tell them
from a field recording ([Sensing contract](architecture.md#sensing-contract)).

**Status (stage 2, on the k space recordings): the ships are located and tracked from the traces alone and some of the
silent bodies by their echo; <!-- reproduce:ntrue:criteria[*]/passed -->0<!-- /reproduce --> of the <!-- reproduce:count:criteria[*]/passed -->4<!-- /reproduce --> criteria fixed for this stage are met on the test
recordings.** The ferry is tracked in <!-- reproduce:range:recordings/[bms][iu]*/objects/ferry/pd/radiator|pct -->100 %<!-- /reproduce -->
of the frames of every 0.5 m and 1 m recording, the tug in <!-- reproduce:range:recordings/multistatic_seed*/objects/tug/pd/radiator|pct -->100 %<!-- /reproduce --> and the boat in
<!-- reproduce:range:recordings/multistatic_seed*/objects/boat/pd/radiator|pct -->100 %<!-- /reproduce --> of the frames of the multistatic test seeds, and on the new geometry `multistatic_b`
the tug in <!-- reproduce:recordings/multistatic_b/objects/tug/pd/radiator|pct -->86 %<!-- /reproduce --> and the boat in <!-- reproduce:recordings/multistatic_b/objects/boat/pd/radiator|pct -->80 %<!-- /reproduce -->. By their echo, the simple harbor's wreck is
tracked in <!-- reproduce:range:recordings/simple_seed*/objects/wreck/pd/echo|pct -->67 to 98 %<!-- /reproduce --> of the frames of the test seeds, the bistatic UUV in
<!-- reproduce:range:recordings/bistatic_seed*/objects/uuv1/pd/echo|pct -->59 to 69 %<!-- /reproduce -->, and of the silent bodies of the multi ship scene the second UUV in
<!-- reproduce:range:recordings/multistatic_seed*/objects/uuv2/pd/echo|pct -->83 to 89 %<!-- /reproduce --> (<!-- reproduce:recordings/multistatic_b/objects/uuv2/pd/echo|pct -->45 %<!-- /reproduce --> on `multistatic_b`) and the first in at most
<!-- reproduce:max:recordings/multistatic*/objects/uuv1/pd/echo|pct -->30 %<!-- /reproduce -->; the diver, the pilings and the moored hull are tracked in at most <!-- reproduce:max:recordings/multistatic*/objects/[dpm]*/pd/echo|pct -->0 %<!-- /reproduce --> of the frames of
any multistatic recording. The criteria: the ships on `multistatic_b` (tug <!-- reproduce:criteria[0]/values/tug|frac2 -->0.86<!-- /reproduce -->, boat
<!-- reproduce:criteria[0]/values/boat|f3 -->0.796<!-- /reproduce --> against 0.8), false tracks (<!-- reproduce:range:criteria[1]/values/*|f1 -->2.0 to 6.0<!-- /reproduce --> per minute on the test recordings against
2), the wreck (<!-- reproduce:range:criteria[2]/values/*|frac2 -->0.67 to 0.98<!-- /reproduce --> against 0.9), the size order (not testable: no frame had the hull, a UUV and a piling
all measured). The chain takes <!-- reproduce:range:recordings/[bm]*/compute_per_second_of_data/total|f2 -->0.97 to 1.58<!-- /reproduce --> s of compute per second of data on the 0.5 m recordings
on a Quadro RTX 4000 (real time is 1 s) (for the compute before stage 2 see the
[history](harbor_history.md#findings-before-stage-2)). Numbers, with tuning beside test recordings, in
[Numbers on recordings the parameters never saw](#numbers-on-recordings-the-parameters-never-saw).

Earlier work (the batch chains with AIS in the loop, stage 0 and stage 1, the performance work) is in
[Harbor, history](harbor_history.md).

## Running it

Three commands (in the dev container prefix each with `./dev.sh run`; the GPU is used when there is one, `--cpu`
otherwise):

```bash
# The system on one recording: picture.jsonl, summary.json, picture.mp4 (the operator view)
python -m acoustic_sim.water.stream --observation output/recordings/simple/dataset.observation.npz \
    --processing examples/configs/processing/simple.json --out output/picture_simple

# The same, checked against where the objects really are (Pd, false tracks, characteristics against the truth)
python examples/harbor/05_detection/harbor_stream.py --dataset simple

# Every number of this page, on every tuning and test recording, compared with the committed numbers
python scripts/reproduce.py
```

`scripts/reproduce.py` builds any missing recording (the FDTD runs are cached), runs the system on all <!-- reproduce:count:recordings/*/family -->17<!-- /reproduce --> recordings,
regenerates the walkthrough figures and writes `output/reproduce/latest.json` and `tables.md`; it compares every number
with the committed `output/reproduce/summary.json` (fractions of frames within 3 percentage points, median errors within
0.5 m, false tracks exactly; compute time and latency informational) and exits 1 if one differs. `--check` only compares,
`--update` makes the run the committed one. CI runs one tiny scenario end to end on the CPU
(`tests/harbor/test_ci_end_to_end.py`: 50 x 50 m, 2 s, a ferry and a wreck, about a minute); the full reproduce is run by
hand on the GPU.

## The system

One implementation of every step, `acoustic_sim.water.stream` (`HarborPicture`) over the objects of
`water/streaming.py`, fed the traces in 0.25 s chunks as they would arrive (a batch run, `harbor_detect.py`, is the same
stream with one chunk):

1. `StreamingPassive` locates the radiating objects, ships included, without AIS: channel QC, common mode, noise floors
   and the stationary interference projector over the last seconds, the matched field map of every frame, its 5
   strongest peaks.
2. `RadiatorTracker` turns the peaks into tracks (a peak joins the track it is predicted to be within 6 m of; a track is
   confirmed after 6 fixes in 2 s; one within 30 m of a stronger track is a sidelobe and is not confirmed).
3. `StreamingEcho` takes the confirmed tracks scoring at least a quarter of the strongest, removes the direct path of
   each from the traces (ECA with per sample delays, each with its own data reference beam, the track fitted over the
   past 4 s), and maps the echoes with every one of them as an illuminator.
4. A second `RadiatorTracker` tracks the echo peaks (gate 6 m, 8 fixes in 3 s, kept 8 s without a fix), the peaks within
   25 m of a confirmed radiator track left out so a residual direct path is not confirmed as a body. A confirmed track
   whose fitted speed stays below 0.2 m/s for 10 s is static.
5. `water/characteristics.py` measures every track (below); a radiator track and an echo track within 5 m of each other
   for 2 s are one object seen both ways.
6. The AIS reports received by each frame label the radiator tracks.

The 5 m harbor runs steps 1, 2, 5 and 6 (2 s frames, its own tracker settings). Every parameter of a recording is in
one file, `examples/configs/processing/<recording>.json` (band, scan edge, chunk, `ProcessingConfig`, `StreamConfig`,
both `TrackerConfig`s, `EchoStreamConfig`, `BistaticConfig`, the characteristics, and the scoring radii, which only the
scoring reads). `--set key=value` overrides one for an experiment (`--set echo.tracker.gate=8`) and is recorded. Every
output carries a `run` block (git commit and whether the tree was dirty, the config and observation files with their
sha256, versions, the GPU, the seed, the wall time, the provenance; [Architecture](architecture.md#provenance)).

## Numbers on recordings the parameters never saw

The constants of the processing configs were set while looking at the four recordings (noise seed 0 of `simple`,
`bistatic`, `multistatic`, `harbor`: the tuning set). They were frozen (commit tagged `stage2-freeze`) before the test
recordings existed: noise seeds 1 to 3 of each, mixed from the same cached FDTD gathers, and `multistatic_b`, the
multistatic harbor with every track changed (other headings, crossing points and speeds, the pilings and the hull moved,
the boat starting on the other side; same sensors and levels), run with the `multistatic` config unchanged. No
parameter was changed after the test numbers were seen. Scoring ([Architecture](architecture.md#scoreboard)): a track
is matched to the object nearest over its fixes if within 10 m (20 m on the 5 m harbor), otherwise it is a false track;
Pd is the fraction of frames, from the first frame the object could have been confirmed, with a confirmed track matched
to it within 5 m; the false alarm rate is false tracks per minute. Every track that was ever confirmed is scored (before
stage 2 only the tracks confirmed at the end of the record were, so these false track counts are higher than the
history's). Every number of this page is in `output/reproduce/summary.json` (or the file the page names) and is written
into the page by `scripts/reproduce.py`; how they moved when the recordings were rebuilt with the k space scheme is in
[Before and after the k space recordings](#before-and-after-the-k-space-recordings).

### Real time, tuning recordings beside test recordings

Pd: fraction of frames, from the first frame the object could have been confirmed, with a confirmed track matched to it within 5 m (radiator tracks for what radiates, echo tracks for silent bodies); test: the three noise seeds (each value) and multistatic_b. False tracks: confirmed tracks matched to no object, per minute.

<!-- reproduce-table:realtime -->
| Recording | Split | Compute per s of data | Latency (median) | Pd per object | False tracks (per minute) |
|---|---|---|---|---|---|
| simple | tuning | 0.71 | 1.9 s | ferry 100 %; boat 100 %; uuv1 28 %; wreck 86 % | 2 (6.0) |
| simple | test | 0.70 / 0.70 / 0.70 | 1.9 s / 1.9 s / 1.9 s | ferry 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 23 % / 30 % / 25 %; wreck 67 % / 86 % / 98 % | 1 (3.0) / 1 (3.0) / 2 (6.0) |
| bistatic | tuning | 0.97 | 2.0 s | ferry 100 %; uuv1 67 % | 3 (9.0) |
| bistatic | test | 0.98 / 0.99 / 1.07 | 2.0 s / 2.0 s / 2.0 s | ferry 100 % / 100 % / 100 %; uuv1 69 % / 59 % / 67 % | 2 (6.0) / 2 (6.0) / 2 (6.0) |
| multistatic | tuning | 1.39 | 2.6 s | ferry 100 %; tug 100 %; boat 100 %; uuv2 83 %; not tracked: uuv1, diver, piling1, piling2, piling3, piling4, piling5, piling6, moored | 2 (6.0) |
| multistatic | test | 1.58 / 1.40 / 1.33 | 2.6 s / 2.6 s / 2.6 s | ferry 100 % / 100 % / 100 %; tug 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 0 % / 30 % / 0 %; uuv2 83 % / 89 % / 86 %; not tracked: diver, piling1, piling2, piling3, piling4, piling5, piling6, moored | 2 (6.0) / 1 (3.0) / 2 (6.0) |
| harbor | tuning | 0.52 | 1.1 s | ferry 72 %; tug 2 %; moored 0 %; uuv1 0 %; uuv2 0 %; diver 0 % | 5 (2.5) |
| harbor | test | 0.52 / 0.53 / 0.53 | 1.1 s / 1.1 s / 1.1 s | ferry 72 % / 78 % / 61 %; tug 6 % / 9 % / 6 %; moored 0 % / 0 % / 0 %; uuv1 0 % / 0 % / 0 %; uuv2 0 % / 0 % / 0 %; diver 0 % / 0 % / 0 % | 4 (2.0) / 4 (2.0) / 4 (2.0) |
| multistatic_b | test | 1.41 | 2.5 s | ferry 100 %; tug 86 %; boat 80 %; uuv1 27 %; uuv2 45 %; not tracked: diver, piling1, piling2, piling3, piling4, piling5, piling6, moored | 6 (6.0) |
<!-- /reproduce-table -->

Per recording, `output/reproduce/runs/<recording>/`. What they say:

* **Tuning to test.** The noise seeds do not move the ships: on every seed of the simple, bistatic and multistatic
  recordings the ferry, the tug and the boat are tracked in <!-- reproduce:range:recordings/[bms][iu]*_seed*/objects/[ftb]*/pd/radiator|pct -->100 %<!-- /reproduce --> of the frames. They move the
  silent bodies: the simple harbor's wreck <!-- reproduce:range:recordings/simple_seed*/objects/wreck/pd/echo|pct -->67 to 98 %<!-- /reproduce --> on the test seeds against <!-- reproduce:recordings/simple/objects/wreck/pd/echo|pct -->86 %<!-- /reproduce --> on the tuning
  recording, its UUV <!-- reproduce:range:recordings/simple_seed*/objects/uuv1/pd/echo|pct -->23 to 30 %<!-- /reproduce --> against <!-- reproduce:recordings/simple/objects/uuv1/pd/echo|pct -->28 %<!-- /reproduce -->, the bistatic UUV <!-- reproduce:range:recordings/bistatic_seed*/objects/uuv1/pd/echo|pct -->59 to 69 %<!-- /reproduce --> against
  <!-- reproduce:recordings/bistatic/objects/uuv1/pd/echo|pct -->67 %<!-- /reproduce -->, the multistatic second UUV <!-- reproduce:range:recordings/multistatic_seed*/objects/uuv2/pd/echo|pct -->83 to 89 %<!-- /reproduce --> against <!-- reproduce:recordings/multistatic/objects/uuv2/pd/echo|pct -->83 %<!-- /reproduce --> and the first
  <!-- reproduce:range:recordings/multistatic_seed*/objects/uuv1/pd/echo|pct -->0 to 30 %<!-- /reproduce --> against <!-- reproduce:recordings/multistatic/objects/uuv1/pd/echo|pct -->0 %<!-- /reproduce -->.
* **The new geometry.** On `multistatic_b` the ferry, the tug and the boat are tracked in <!-- reproduce:recordings/multistatic_b/objects/ferry/pd/radiator|pct -->100 %<!-- /reproduce -->,
  <!-- reproduce:recordings/multistatic_b/objects/tug/pd/radiator|pct -->86 %<!-- /reproduce --> and <!-- reproduce:recordings/multistatic_b/objects/boat/pd/radiator|pct -->80 %<!-- /reproduce --> of the frames, the two UUVs by their echo in <!-- reproduce:recordings/multistatic_b/objects/uuv1/pd/echo|pct -->27 %<!-- /reproduce --> and <!-- reproduce:recordings/multistatic_b/objects/uuv2/pd/echo|pct -->45 %<!-- /reproduce -->.
* **False tracks.** Every track ever confirmed is scored, so on a <!-- reproduce:recordings/simple/record_s|f0 -->20<!-- /reproduce --> s recording a single false track
  already exceeds the criterion of 2 per minute. On the test recordings <!-- reproduce:sum:recordings/*_seed*/false_tracks/radiator -->21<!-- /reproduce --> of the false tracks
  of the noise seeds are radiator tracks and <!-- reproduce:sum:recordings/*_seed*/false_tracks/echo -->6<!-- /reproduce --> echo tracks (<!-- reproduce:recordings/multistatic_b/false_tracks/radiator -->3<!-- /reproduce --> and <!-- reproduce:recordings/multistatic_b/false_tracks/echo -->3<!-- /reproduce --> on
  `multistatic_b`). The false radiator tracks of the 0.5 m and 1 m recordings lie near the loudest ship or near a
  silent body: sidelobes of a ship outside the radiator tracker's 30 m exclusion, and the passive chain locating a
  ship's echo off a silent body (on the simple harbor, off the wreck) as if it radiated. Echo tracks near the moored
  hull are counted false when they are more than 10 m from its centre, which is what the scoring measures to, although
  the hull is 30 m long. The positions of every false track are in `false_track_list` of `summary.json`.
* **Silent bodies of the multi ship scene.** Of the two UUVs, the diver, the six pilings and the moored hull, only the
  UUVs are tracked by their echo on any multistatic recording (the others in at most <!-- reproduce:max:recordings/multistatic*/objects/[dpm]*/pd/echo|pct -->0 %<!-- /reproduce --> of the
  frames); the removal of the ships' direct sound is the limit ([Before and after the k space
  recordings](#before-and-after-the-k-space-recordings), the true waveform diagnostic).
* **The 5 m harbor** tracks the ferry in <!-- reproduce:range:recordings/harbor*/objects/ferry/pd/radiator|pct -->61 to 78 %<!-- /reproduce --> of its frames and the tug in
  <!-- reproduce:range:recordings/harbor*/objects/tug/pd/radiator|pct -->2 to 9 %<!-- /reproduce -->, with <!-- reproduce:range:recordings/harbor_seed*/false_tracks/per_minute|f1 -->2.0<!-- /reproduce --> false tracks per minute on the test seeds.

### Criteria

Fixed before the test recordings were built (`scripts/reproduce.py`, `CRITERIA`), scored on the test recordings:

<!-- reproduce-table:criteria -->
| Criterion | Values | Result |
|---|---|---|
| every ship (ferry, tug, boat) tracked with Pd >= 0.8 on multistatic_b (radiator tracks) | {'ferry': 1.0, 'tug': 0.86, 'boat': 0.8} | **fail** |
| false tracks at most 2 per minute on every test recording | {'simple_seed1': 3.0, 'simple_seed2': 3.0, 'simple_seed3': 6.0, 'bistatic_seed1': 6.0, 'bistatic_seed2': 6.0, 'bistatic_seed3': 6.0, 'multistatic_seed1': 6.0, 'multistatic_seed2': 3.0, 'multistatic_seed3': 6.0, 'harbor_seed1': 2.0, 'harbor_seed2': 2.0, 'harbor_seed3': 2.0, 'multistatic_b': 6.0} | **fail** |
| the wreck of the simple harbor found by its echo with Pd >= 0.9 on seeds 1 to 3 (echo tracks) | {'simple_seed1': 0.67, 'simple_seed2': 0.86, 'simple_seed3': 0.98} | **fail** |
| on frames where the moored hull, a UUV and a piling are all measured, the size indicator puts the hull below the piling in at least 70 % of them (multistatic recordings, tuning and multistatic_b) | no frame had the hull, a UUV and a piling all measured: not testable | **not met: not testable** |
<!-- /reproduce-table -->

The size criterion could not be evaluated: no echo track of the hull, a UUV and a piling was confirmed together on
either multistatic recording, so there was no frame with all three measured.

## Walkthrough: from the traces to the harbor picture

`examples/harbor/06_walkthrough/` takes one recording through the system, one image per plot
(`./examples/harbor/06_walkthrough/run_all.sh DATASET`, DATASET one of `simple`, `bistatic`, `multistatic`, `harbor`; each
script also runs on its own). The chain runs once (`water/stream.py`, cached in `output/walkthrough/<dataset>/cache/`;
`scripts/reproduce.py` deletes the cache and runs it again) and the images go to `output/walkthrough/<dataset>/`. Nothing in
the processing reads the truth or the AIS reports: the truth is drawn (dashed lines, circles) only to show where the
objects really are, and AIS enters in step 4 only. Below: the multistatic recording (three ships, two UUVs, a diver, six
pilings and a moored hull) and, for the echo steps, the [simple harbor](#the-simple-harbor). All four recordings have the
full set (the 5 m harbor without steps 5 and 6).

**1. The recording** (`01_recording.py`). What the system gets: the chart, the sensor positions (a DAS cable and four
hydrophones) and the traces (`01_recording_gather.png`, `_spectrogram.png`, `_trace.png`). The cable of the bistatic and
multistatic recordings is in strain mode (the second difference of the field over the gauge length); the simple harbor
and the 5 m harbor have a pressure mode cable.

![step 1](../output/walkthrough/multistatic/01_recording_chart.png)

Look at where the cable and the four hydrophones lie against the water: the cable runs along two sides of the scene only.

**2. Passive location** (`02_passive_map.py`). Every 0.25 s the matched field processor steers the array at every cell of
the scan grid; a radiating object lights up. The 5 strongest peaks of each frame are the passive detections; ships are
located like everything else, without their AIS reports (three frames, `02_passive_map_1.png` to `_3.png`).

![step 2](../output/walkthrough/multistatic/02_passive_map_2.png)

Look at the peaks on the true ship positions and at the long thin shape of each peak, elongated toward the cable.

**3. Tracks** (`03_tracks.py`). Peaks that follow on in space and time become tracks; a track is confirmed after 6 fixes in
2 s, and one that sits within 30 m of a stronger track (a sidelobe) is not. Every peak is in `03_tracks_peaks.png`, the
median distance of every track to the nearest true object in `03_tracks_errors.png`.

![step 3](../output/walkthrough/multistatic/03_tracks_tracks.png)

Look at the confirmed tracks lying on the dashed true tracks of the three ships.

**4. AIS afterwards** (`04_ais_crosscheck.py`). Only now are the AIS reports read: a track within 25 m of a reporting ship
takes its name; a track that matches no report is a radiating object without AIS (here the boat). The distance of every
track to every reporting ship over time is `04_ais_crosscheck_distance.png`.

![step 4](../output/walkthrough/multistatic/04_ais_crosscheck_tracks.png)

Look at the label of each track: the two AIS ships by name, the boat as a radiator without a report.

**5. Removing the ships' own sound** (`05_direct_path_removal.py`). A silent body is seen only by the ship noise it
reflects, far below the ship's direct sound. For every strong track the chain forms a reference beam toward it
and removes the delayed copies of that reference from every channel (ECA, delays following the track). On one channel
(`_trace.png`, `_spectrum.png`) and over every channel:

![step 5](../output/walkthrough/simple/05_direct_path_removal_channels.png)

Look at how far the level of every channel falls after the removal, and at the channels where it falls least.

**6. The echo map and the echo tracks** (`06_echo_map.py`). On the cleaned traces every cell is tested as a reflector of
each tracked ship's sound. The peaks of every frame feed the echo tracker, which keeps a body only when its echo comes
back for 8 frames in 3 s; the picture shows the echo tracks, the raw peaks faint underneath. Three frames
(`06_echo_map_1.png` to `_3.png`) and the record summed (each frame over its median, so the scale is dB over the noise
level), with every confirmed echo track: on the simple harbor the wreck is tracked in <!-- reproduce:recordings/simple/objects/wreck/pd/echo|pct -->86 %<!-- /reproduce --> of the frames, the UUV in
<!-- reproduce:recordings/simple/objects/uuv1/pd/echo|pct -->28 %<!-- /reproduce -->.

![step 6](../output/walkthrough/simple/06_echo_map_full_record.png)

Look at the bright spot on the wreck and the echo tracks over it and over the UUV's path.

**7. The harbor picture** (`07_harbor_picture.py`). Radiator tracks (with their AIS label) and echo tracks over the true
positions, scored with the stage 2 definitions (Pd per object: `07_harbor_picture_pd.png`, `07_harbor_picture.json`). On
the multistatic recording the ferry, the tug and the boat are tracked in <!-- reproduce:recordings/multistatic/objects/ferry/pd/radiator|pct -->100 %<!-- /reproduce -->, <!-- reproduce:recordings/multistatic/objects/tug/pd/radiator|pct -->100 %<!-- /reproduce --> and <!-- reproduce:recordings/multistatic/objects/boat/pd/radiator|pct -->100 %<!-- /reproduce --> of the
frames and the second UUV by its echo in <!-- reproduce:recordings/multistatic/objects/uuv2/pd/echo|pct -->83 %<!-- /reproduce -->; there are <!-- reproduce:recordings/multistatic/false_tracks/radiator -->0<!-- /reproduce --> false radiator tracks and <!-- reproduce:recordings/multistatic/false_tracks/echo -->2<!-- /reproduce --> false echo tracks.

![step 7](../output/walkthrough/multistatic/07_harbor_picture_map.png)

Look at which true positions have a track on them and which do not: the pilings, the diver and the moored hull have none.

**8. The geometry** (`08_geometry.py`). Where the chain projects each track ahead (the straight line fit the tracker gates
with and the echo chain places the illuminator by) against where the object really was (`08_geometry_projection.png`),
how far ahead that holds (`08_geometry_lookahead.png`), and one frame's geometry in three figures: its echo map
(`08_geometry_frame_map.png`), the ray paths (the illuminators where the echo chain placed them, their direct paths to a
few sensors, the echo path through the strongest peak) and the ellipse of equal delay through that peak
(`08_geometry_ellipse.png`: an echo can come from anywhere on that ellipse; the array's aperture picks the point).

![step 8](../output/walkthrough/multistatic/08_geometry_ray_paths.png)

Look at the solid red path from the illuminator to the strongest echo peak and at the dashed paths from that peak to the sensors.

The same frame geometry is in `check_<chain>_frame_map.png`, `check_echo_frame_rays.png` and `check_echo_frame_ellipse.png` of every run of `harbor_stream.py` and `scripts/reproduce.py`.

## What the objects are: measured characteristics

No classifier: measured quantities a harbor operator can read, attached to every track with the frames they were
measured over (`water/characteristics.py`; it reads the stream's spectra, maps and tracks, never the truth).

| For | Quantity | How it is measured |
|---|---|---|
| every track | position and its uncertainty | the fitted position; the full widths at half maximum of the map peak that fed the latest fix along its two principal axes and the major axis direction; the rms residual of the fixes about the straight line fitted over the last 4 s |
| | velocity, heading | from that fit, with the standard errors of the fit |
| | how it is seen | own noise, echo, or both (a radiator and an echo track within 5 m for 2 s) |
| | AIS | the reporting ship whose dead reckoned reports lie within 25 m of the track (median over its last 8 fixes), and the distance between reported and located position at the latest fix; else "no report" |
| | status | tentative, confirmed, static (fitted speed below 0.2 m/s for 10 s), lost (no fix for the drop time; kept 4 s, faded); frames detected over frames since confirmation |
| radiator | received level | the least squares source spectrum toward the track from every live channel with the chain's own replicas (so strain and pressure channels share one unit), propagated to the median range from the track to the channels, rms in the band, in trace units |
| | source level at 100 m | the received level with cylindrical spreading from that range, times one constant calibrated on the simple harbor (ferry 1.0 Pa, boat 0.3 Pa at 100 m: <!-- reproduce:@studies/level_calibration/calibration.json:level_calibration|f4 -->5.1152<!-- /reproduce -->; `harbor_level_calibration.py`, `output/studies/level_calibration/calibration.json`) |
| | tonal lines | the 3 strongest local maxima of that spectrum at least 6 dB over its running median (21 bins): frequency, level over the floor, frames in which a line within one bin was found |
| echo | echo strength, per illuminator | the least squares source spectrum at the echo's cell over the illuminator's direct field there (its data reference divided by the beam's gain, propagated to the cell), coherent over the band, dB |
| | size indicator | that ratio in the upper third of the band over the lower third, dB: above 0 dB "smaller than about a wavelength", else "about a wavelength or larger" (never metres) |
| | extent | the length of the half maximum region of the echo peak along and across the direction from the strongest illuminator |

Reported values are medians over the last 8 frames the track was measured in. A body small against the wavelength
scatters more at high frequency, a body large against it about equally; in the 100 to 760 Hz band the wavelengths are 2
to 15 m, so the indicator can separate a 1 m piling or a diver from the 30 m moored hull and says little about a 1 x 3 m
UUV. **The forward model's bodies are two dimensional pressure release masks**, so the comparison with the true sizes
is a check of the measurement, not of the physics of real hulls.

### Validation against the truth

<!-- reproduce-table:characteristics -->
| Recording | Object | True type, size | Seen | Detected frames | Size ratio [dB] | Size indicator | Echo extent along / across [m] | Source level [Pa] (true) | Half max width [m] | Speed [m/s] (true) |
|---|---|---|---|---|---|---|---|---|---|---|
| multistatic | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.26 (1) | 9.2 x 2.3 | 4.0 (4.0) |
| multistatic | tug | ship, - | own noise 66 | 66 | - | - | - | 0.83 (0.7) | 15.5 x 3.5 | 3.0 (3.0) |
| multistatic | boat | ship, - | own noise 66 | 66 | - | - | - | 0.59 (0.3) | 46.9 x 6.2 | 5.0 (5.0) |
| multistatic | uuv2 | uuv, 3 x 1 m | echo 53 | 53 | -8.1 | about a wavelength or larger | 4.6 / 4.1 | - | 6.9 x 2.3 | 1.5 (1.5) |
| multistatic_b | ferry | ship, - | own noise 226 | 226 | - | - | - | 1.15 (1) | 11.6 x 3.7 | 3.4 (3.5) |
| multistatic_b | tug | ship, - | own noise 194 | 194 | - | - | - | 0.73 (0.7) | 32.8 x 4.5 | 2.5 (2.5) |
| multistatic_b | boat | ship, - | own noise 197 | 180 | - | - | - | 0.45 (0.3) | 21.2 x 4.4 | 4.3 (3.7) |
| multistatic_b | uuv1 | uuv, 3 x 1 m | echo 143 | 60 | -5.9 | about a wavelength or larger | 6.5 / 7.2 | - | 9.6 x 3.6 | 1.7 (1.7) |
| multistatic_b | uuv2 | uuv, 3 x 1 m | echo 190 | 101 | -2.1 | about a wavelength or larger | 3.8 / 6.0 | - | 6.9 x 4.2 | 1.3 (1.2) |
| simple | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.09 (1) | 5.2 x 3.5 | 4.0 (4.0) |
| simple | boat | ship, - | own noise 66 | 66 | - | - | - | 0.28 (0.3) | 4.6 x 2.3 | 2.5 (2.5) |
| simple | uuv1 | uuv, 6 x 2 m | echo 18 | 18 | -4.7 | about a wavelength or larger | 4.1 / 3.5 | - | 4.9 x 2.3 | 1.4 (1.5) |
| simple | wreck | static, 4 x 4 m | echo 64, own noise 32 | 55 | -4.8 | about a wavelength or larger | 3.5 / 4.8 | 0.18 (0) | 4.6 x 4.6 | 0.2 (0.0) |
| simple_seed1 | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.07 (1) | 5.2 x 3.5 | 4.0 (4.0) |
| simple_seed1 | boat | ship, - | own noise 66 | 66 | - | - | - | 0.28 (0.3) | 4.6 x 2.3 | 2.5 (2.5) |
| simple_seed1 | uuv1 | uuv, 6 x 2 m | echo 16 | 15 | -11.5 | about a wavelength or larger | 3.4 / 3.5 | - | 4.6 x 2.3 | 1.4 (1.5) |
| simple_seed1 | wreck | static, 4 x 4 m | echo 64, own noise 34 | 43 | -5.0 | about a wavelength or larger | 3.5 / 4.8 | 0.18 (0) | 5.2 x 3.5 | 0.2 (0.0) |
| bistatic | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.32 (1) | 6.9 x 2.3 | 4.0 (4.0) |
| bistatic | uuv1 | uuv, 3 x 1 m | own noise 29, echo 43 | 55 | -1.2 | about a wavelength or larger | 3.9 / 3.1 | 0.17 (0) | 8.8 x 2.9 | 2.0 (2.0) |
| bistatic_seed1 | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.36 (1) | 6.9 x 2.3 | 4.0 (4.0) |
| bistatic_seed1 | uuv1 | uuv, 3 x 1 m | own noise 28, echo 44 | 44 | -0.5 | about a wavelength or larger | 3.9 / 3.0 | 0.18 (0) | 7.7 x 2.9 | 2.1 (2.0) |
<!-- /reproduce-table -->

What the characteristics get right and wrong:

* **Speed** of the tracked ships against the truth (table above): the multistatic boat <!-- reproduce:characteristics/multistatic[object=boat]/median/speed|f1 -->5.0<!-- /reproduce --> m/s against
  <!-- reproduce:characteristics/multistatic[object=boat]/true_speed|f1 -->5.0<!-- /reproduce -->, the `multistatic_b` ferry <!-- reproduce:characteristics/multistatic_b[object=ferry]/median/speed|f1 -->3.4<!-- /reproduce --> against <!-- reproduce:characteristics/multistatic_b[object=ferry]/true_speed|f1 -->3.5<!-- /reproduce --> and boat <!-- reproduce:characteristics/multistatic_b[object=boat]/median/speed|f1 -->4.3<!-- /reproduce --> against <!-- reproduce:characteristics/multistatic_b[object=boat]/true_speed|f1 -->3.7<!-- /reproduce -->; the UUVs
  tracked by their echo <!-- reproduce:range:characteristics/*[object=uuv*]/median/speed|f1 -->1.3 to 2.1<!-- /reproduce --> m/s against <!-- reproduce:range:characteristics/*[object=uuv*]/true_speed|f1 -->1.2 to 2.0<!-- /reproduce -->.
* **Source level** at 100 m against the truth (table below): <!-- reproduce:range:source_levels[recording=simple*]/error_db|+f1 -->-0.9 to +0.8<!-- /reproduce --> dB on the simple harbor, where the constant
  was calibrated; <!-- reproduce:range:source_levels[recording=bistatic*]/error_db|+f1 -->+2.4 to +2.7<!-- /reproduce --> dB for the bistatic ferry; <!-- reproduce:range:source_levels[recording=multistatic*][object=ferry,tug]/error_db|+f1 -->+0.3 to +2.5<!-- /reproduce --> dB for the multistatic ferry and
  tug and <!-- reproduce:range:source_levels[recording=multistatic*][object=boat]/error_db|+f1 -->+3.5 to +6.2<!-- /reproduce --> dB for the multistatic boat, the quietest ship, whose map peak shares energy with the
  louder ships' sidelobes. The constant from the simple harbor does not carry to the 5 m harbor
  (<!-- reproduce:range:source_levels[recording=harbor*]/error_db|+f1 -->+13.6 to +14.0<!-- /reproduce --> dB: another grid, band and signature spectrum).
* **Size indicator**: every body tracked by its echo in the table is called "about a wavelength or larger": the
  simple harbor's 4 m wreck (<!-- reproduce:characteristics/simple[object=wreck]/median/size_ratio_db|+f1 -->-4.8<!-- /reproduce --> and <!-- reproduce:characteristics/simple_seed1[object=wreck]/median/size_ratio_db|+f1 -->-5.0<!-- /reproduce --> dB; wavelengths 4 to 15 m there), but also its 6 x 2 m
  UUV (<!-- reproduce:characteristics/simple[object=uuv1]/median/size_ratio_db|+f1 -->-4.7<!-- /reproduce --> and <!-- reproduce:characteristics/simple_seed1[object=uuv1]/median/size_ratio_db|+f1 -->-11.5<!-- /reproduce --> dB) and the 1 x 3 m UUVs of the bistatic recording (<!-- reproduce:characteristics/bistatic[object=uuv1]/median/size_ratio_db|+f1 -->-1.2<!-- /reproduce --> and
  <!-- reproduce:characteristics/bistatic_seed1[object=uuv1]/median/size_ratio_db|+f1 -->-0.5<!-- /reproduce --> dB) and of the multistatic recordings (<!-- reproduce:range:characteristics/multistatic*[object=uuv*]/median/size_ratio_db|+f1 -->-8.1 to -2.1<!-- /reproduce --> dB). On these recordings it does not separate a
  body smaller than the wavelength from one larger. The criterion of the stage (hull below piling) is untestable
  because neither is tracked.
* **How it is seen**: the bistatic UUV is reported "seen both ways" (by its own noise in
  <!-- reproduce:characteristics/bistatic[object=uuv1]/seen/own noise -->29<!-- /reproduce --> frames, by its echo in <!-- reproduce:characteristics/bistatic[object=uuv1]/seen/echo -->43<!-- /reproduce -->): the passive chain locates the ferry's echo off it and the radiator
  tracker tracks that, so a silent body is labelled as radiating. The same mechanism makes the simple harbor's wreck
  "seen both ways" (own noise in <!-- reproduce:characteristics/simple[object=wreck]/seen/own noise -->32<!-- /reproduce --> frames, echo in <!-- reproduce:characteristics/simple[object=wreck]/seen/echo -->64<!-- /reproduce -->).
* **Half maximum widths** are <!-- reproduce:range:characteristics/simple*[*]/median/width_major|f1 -->4.6 to 5.2<!-- /reproduce --> m along their major axis on the simple harbor's loop and
  <!-- reproduce:range:characteristics/multistatic*[object=ferry,tug,boat]/median/width_major|f0 -->9 to 47<!-- /reproduce --> m along by <!-- reproduce:range:characteristics/multistatic*[object=ferry,tug,boat]/median/width_minor|f0 -->2 to 6<!-- /reproduce --> m across for the ships on the strain cable of the
  multistatic recordings: the cable sees the ships from one side, and the map peak is elongated toward it. They are
  the map's resolution, not the location error: the median error of the ships' tracks on the 0.5 m recordings is
  <!-- reproduce:range:recordings/[bm]*/objects/[ftb]*/median_error_m/radiator|f1 -->0.5 to 1.0<!-- /reproduce --> m.
* **Tonal lines** are reported with their persistence (`picture.jsonl`); a ship's lines persist only in a minority of
  the frames its track was measured in (2 Hz bins of the 0.5 s frames), so they are numbers for an operator, not yet a
  signature.

### Source level estimates against the truth (radiator tracks; calibration constant <!-- reproduce:level_calibration|f4 -->5.1152<!-- /reproduce --> from the simple harbor's tuning recording)

<!-- reproduce-table:source_levels -->
| Recording | Object | True [Pa] | Estimate [Pa] | Error [dB] |
|---|---|---|---|---|
| simple | ferry | 1 | 1.091 | +0.8 |
| simple | boat | 0.3 | 0.282 | -0.5 |
| bistatic | ferry | 1 | 1.322 | +2.4 |
| multistatic | ferry | 1 | 1.264 | +2.0 |
| multistatic | tug | 0.7 | 0.828 | +1.5 |
| multistatic | boat | 0.3 | 0.594 | +5.9 |
| harbor | ferry | 1 | 4.986 | +14.0 |
| simple_seed1 | ferry | 1 | 1.071 | +0.6 |
| simple_seed1 | boat | 0.3 | 0.276 | -0.7 |
| simple_seed2 | ferry | 1 | 1.067 | +0.6 |
| simple_seed2 | boat | 0.3 | 0.279 | -0.6 |
| simple_seed3 | ferry | 1 | 1.060 | +0.5 |
| simple_seed3 | boat | 0.3 | 0.272 | -0.9 |
| bistatic_seed1 | ferry | 1 | 1.356 | +2.6 |
| bistatic_seed2 | ferry | 1 | 1.366 | +2.7 |
| bistatic_seed3 | ferry | 1 | 1.340 | +2.5 |
| multistatic_seed1 | ferry | 1 | 1.329 | +2.5 |
| multistatic_seed1 | tug | 0.7 | 0.844 | +1.6 |
| multistatic_seed1 | boat | 0.3 | 0.598 | +6.0 |
| multistatic_seed2 | ferry | 1 | 1.303 | +2.3 |
| multistatic_seed2 | tug | 0.7 | 0.854 | +1.7 |
| multistatic_seed2 | boat | 0.3 | 0.613 | +6.2 |
| multistatic_seed3 | ferry | 1 | 1.275 | +2.1 |
| multistatic_seed3 | tug | 0.7 | 0.843 | +1.6 |
| multistatic_seed3 | boat | 0.3 | 0.595 | +5.9 |
| harbor_seed1 | ferry | 1 | 4.952 | +13.9 |
| harbor_seed2 | ferry | 1 | 4.793 | +13.6 |
| harbor_seed3 | ferry | 1 | 4.851 | +13.7 |
| multistatic_b | ferry | 1 | 1.147 | +1.2 |
| multistatic_b | tug | 0.7 | 0.726 | +0.3 |
| multistatic_b | boat | 0.3 | 0.451 | +3.5 |
<!-- /reproduce-table -->

## The operator output

### `picture.jsonl`

One JSON record per frame, written as the stream runs (`--out <dir>/picture.jsonl`); the first line is the header. It is
the interface a display or a logger would consume.

```
{"header": {"schema": "harbor_picture/1", "recording": ..., "run": {...run block...}, "fields": ...}}
{"t": 12.5,                       # frame centre [s] in the time base of the recording
 "latency_s": 1.9,                # last sample of the frame recorded -> record written
 "chains": {"passive": true, "echo": true},     # which chains produced this frame
 "objects": [
   {"id": "r1+e3",                # the track ids, joined when one object is seen both ways
    "seen": "own noise" | "echo" | "both",
    "tracks": {"radiator": "r1", "echo": "e3"},
    "track_state": {"r1": {"status", "xy", "fix_t", "fix_xy", "detected"}, ...},   # each track's own state
    "status": "tentative" | "confirmed" | "static" | "lost",
    "detected_frames": 37, "frames_since_confirmation": 51,
    "position": {"xy": [x, y],    # fitted position at t [m]
                 "fix_t", "fix_xy", "detected",          # latest fix, and whether this frame gave one
                 "halfmax_width_m": [major, minor], "halfmax_major_deg", "width_measured_t",
                 "residual_m", "window_s": [t0, t1], "n_fixes"},
    "velocity": {"vxy", "speed", "speed_se", "heading_deg", "heading_se_deg", "window_s", "n_fixes"},
    "ais": {"id": "ferry", "distance_m": 9.6, "fix_t": 14.0} | {"id": null, "note": "no report"},
    "lost_t": ...,                # lost tracks only
    "radiator": {"received_level", "range_m", "source_level_100m_pa", "calibration",
                 "tonal_lines": [{"f_hz", "level_db", "frames"}], "lines_frames_measured", "window_s", "n_frames"},
    "echo": {"strength_db": {"r1": -8.2, ...}, "size_ratio_db", "size", "extent_m": {"along", "across"},
             "illuminator", "window_s", "n_frames", "xy"}}]}
```

Angles are degrees clockwise from north; positions in the chart's local east / north frame [m]; a tentative object
carries only its position. `acoustic_sim.water.stream.read_picture` reads a file back.

### The operator view

`HarborPictureView` (`water/plotting.py`) draws one figure per record: the chart and the sensors, every object as a
marker (circle: own noise, square: echo, diamond: both) with its half maximum ellipse, a velocity vector 5 s ahead and
its label (AIS identity, else "radiator" or "body"; the size indicator; the source level), tentative tracks as dots,
lost tracks faded. It is saved as `picture.mp4` per recording and shown live with `--show`. The truth is drawn only with
`--truth` (dashed) and is off by default.

![operator view, multistatic_b](../output/reproduce/runs/multistatic_b/picture_frame.png)

One frame of `multistatic_b` (a test recording) at 13.5 s: the ferry with its AIS identity, the boat (no AIS), a false
track (r12, a sidelobe of the boat 41 m from it), and the tug only as a trail of tentative fixes on the left: it is not
confirmed until 15 s. Every recording's frame is in `output/reproduce/runs/<recording>/picture_frame.png`; the videos
(`picture.mp4`) are not committed.

### On a field recording

`python -m acoustic_sim.water.stream --observation <file> --processing <config> --out <dir>` needs nothing but the
observation file: convert a recording with `Observation.from_arrays` ([Architecture, real data entry
point](architecture.md#real-data-entry-point)), save it, and start from the processing config of the recording closest
to it (band, grid, chunk).

## Recordings

| Recording | Grid, band | Sensors | Objects |
|---|---|---|---|
| `simple` | 500 x 400 m basin, 1 m, 3 kHz, 20 s; 100-380 Hz | a pressure DAS loop (299 channels) and 4 hydrophones | AIS ferry (1.0 Pa), boat without AIS (0.3 Pa), silent 6 x 2 m UUV, static 4 m wreck |
| `bistatic` | 400 x 300 m crop, 0.5 m, 6 kHz, 20 s; 100-760 Hz | a strain DAS cable (2 m gauge) and 4 hydrophones | AIS ferry, silent 1 x 3 m UUV |
| `multistatic` | as `bistatic` | as `bistatic` | AIS ferry (1.0 Pa) and tug (0.7 Pa), boat without AIS (0.3 Pa), two silent UUVs, a diver, six 1 m pilings 5 m apart, a moored 30 x 6 m hull |
| `multistatic_b` (test only) | as `multistatic`, but 60 s | as `multistatic` | the same objects, every track, the pilings and the hull moved |
| `harbor` | 2400 x 1800 m, 5 m, 600 Hz, 120 s; 12-78 Hz | a pressure DAS cable and 4 hydrophones | AIS ferry and tug, moored ship, radiating UUVs and diver, a silent scatterer |

Every recording is simulated with the k space time stepping (`time_scheme: "kspace"` in each scenario file; the
scheme, where it is exact and its cost are in [Physics](physics.md#time-stepping)), so its direct paths arrive within
<!-- reproduce:absmax:propagation_error_ms/*/hydrophones[*]/error_ms|absf2 -->0.42<!-- /reproduce --> ms of range over the chart's speed (below). `multistatic_b` is 60 s long so that its false tracks per minute are
measured on more than 20 s; on that record the boat and then the ferry reach their last waypoints and stay there,
still radiating. Each recording is in `output/recordings/<recording>/` (the scenario used, the levels on the sensors,
the map, the propagation error, `build.json` with the FDTD wall time): <!-- reproduce:@recordings/simple/build.json:fdtd_wall_s|f0 -->356<!-- /reproduce --> s for `simple`
(<!-- reproduce:@recordings/simple/build.json:fdtd_runs -->4<!-- /reproduce --> runs), <!-- reproduce:@recordings/bistatic/build.json:fdtd_wall_s|f0 -->691<!-- /reproduce --> s for `bistatic` (<!-- reproduce:@recordings/bistatic/build.json:fdtd_runs -->2<!-- /reproduce -->), <!-- reproduce:@recordings/multistatic/build.json:fdtd_wall_s|f0 -->2026<!-- /reproduce --> s for `multistatic` (<!-- reproduce:@recordings/multistatic/build.json:fdtd_runs -->6<!-- /reproduce -->),
<!-- reproduce:@recordings/multistatic_b/build.json:fdtd_wall_s|f0 -->6079<!-- /reproduce --> s for `multistatic_b` (<!-- reproduce:@recordings/multistatic_b/build.json:fdtd_runs -->6<!-- /reproduce -->), <!-- reproduce:@recordings/harbor/build.json:fdtd_wall_s|f0 -->468<!-- /reproduce --> s for `harbor` (<!-- reproduce:@recordings/harbor/build.json:fdtd_runs -->12<!-- /reproduce -->), on a Quadro RTX 4000.

The multistatic recordings each take six FDTD runs (every illuminator with all bodies and with none); every recording's
gathers are cached, so noise seeds and level changes need no FDTD run. Building: `harbor_multistatic_dataset.py`
(`--config` for `simple` and `multistatic_b`), `harbor_bistatic_dataset.py`, `harbor_m2_dataset.py`, each with
`--out output/recordings/<recording>`; `scripts/reproduce.py` builds whatever is missing. `--figures-only` redraws a
recording's figures from its cached gathers without an FDTD run. The default 5 m scenario and the bistatic scenario
are described in [the history](harbor_history.md#the-default-scenarios).

### The simple harbor

`examples/configs/harbor_simple.scenario.json` (`environment.kind = "simple"`) is built so that its wavefields are easy to
resolve: one straight 20 m quay along the north edge (a single reflector), open water elsewhere absorbed by the 40 m
sponge, 1500 m/s everywhere (the chart's single sound speed is the true one), a pressure mode DAS loop around the scene
(80 to 420 m by 80 to 330 m, 4 m channels: every angle covered, no strain directivity) and four corner hydrophones, the
objects inside the loop and 50 m or more apart. The levels of direct sound, echo and noise on the cable are in
`output/recordings/simple/levels.json`.

### The forward model's own propagation error

`harbor_propagation_error.py` measures it on each recording from the truth side: one FDTD run of a stationary Ricker
pulse at the first ship's mid record position, the arrival at each hydrophone (the peak of the envelope of the band
limited cross correlation with the free field prediction at the chart's speed) minus range over that speed; negative is
early. It is written into the truth file's `meta` as `propagation_error_ms` and drawn in
`output/recordings/<recording>/propagation_error.png` (`--plot-only` redraws the figure from the JSON without the FDTD
run); the runs with uniform water, another grid or the leapfrog scheme are in
`output/studies/propagation_error/<recording>/`. Criterion, fixed before the k space recordings were built: within
0.2 ms at every hydrophone on the 0.5 m and 1 m recordings, within 0.1 ms on the simple harbor (whose water has exactly
the chart's speed); none for the 5 m harbor.

<!-- reproduce-table:propagation_error -->
| Recording | Band [Hz] | Chart speed [m/s] | Per hydrophone: error [ms] (direction, range) |
|---|---|---|---|
| simple | 100-380 | 1500.0 | +0.00 (207 deg, 224 m); +0.00 (126 deg, 344 m); +0.00 (312 deg, 135 m); +0.02 (72 deg, 294 m) |
| bistatic | 100-760 | 1493.7 | -0.21 (221 deg, 227 m); -0.03 (140 deg, 234 m); +0.05 (288 deg, 158 m); -0.10 (174 deg, 181 m) |
| multistatic | 100-760 | 1493.7 | -0.21 (221 deg, 227 m); -0.03 (140 deg, 234 m); +0.05 (288 deg, 158 m); -0.10 (174 deg, 181 m) |
| harbor | 12-78 | 1496.0 | -0.42 (297 deg, 1047 m); +0.10 (302 deg, 153 m); -0.42 (44 deg, 672 m); +0.21 (153 deg, 268 m) |
| multistatic_b | 100-760 | 1493.7 | -0.22 (222 deg, 205 m); -0.04 (135 deg, 231 m); +0.03 (296 deg, 152 m); -0.12 (168 deg, 167 m) |
<!-- /reproduce-table -->

* **simple**: within <!-- reproduce:absmax:propagation_error_ms/simple/hydrophones[*]/error_ms|absf2 -->0.02<!-- /reproduce --> ms at every hydrophone: **pass**.
* **bistatic, multistatic** (the same crop, sensors and ferry track, so the same pulse): <!-- reproduce:propagation_error_ms/bistatic/hydrophones[0]/error_ms|+f2 -->-0.21<!-- /reproduce --> ms at hydrophone 1:
  **fail**. **multistatic_b**: <!-- reproduce:propagation_error_ms/multistatic_b/hydrophones[0]/error_ms|+f2 -->-0.22<!-- /reproduce --> ms at hydrophone 1: **fail**. Their hydrophone ranges
  differ from the other 0.5 m recordings because the 60 s record puts the ferry's mid record position elsewhere. In
  both, the travel time along the straight line through the scenario's own sound speed map predicts the error at
  hydrophone 1 (<!-- reproduce:propagation_error_ms/bistatic/hydrophones[0]/straight_path_ms|+f2 -->-0.23<!-- /reproduce --> and <!-- reproduce:propagation_error_ms/multistatic_b/hydrophones[0]/straight_path_ms|+f2 -->-0.23<!-- /reproduce --> ms, `straight_path_ms` in `propagation_error.json`; at every hydrophone
  <!-- reproduce:range:propagation_error_ms/bistatic/hydrophones[*]/straight_path_ms|+f2 -->-0.23 to +0.03<!-- /reproduce --> ms against the measured <!-- reproduce:range:propagation_error_ms/bistatic/hydrophones[*]/error_ms|+f2 -->-0.21 to +0.05<!-- /reproduce --> ms on `bistatic`): the path runs through water
  faster than the chart's single speed (the median of the water). With the water at one speed (`--uniform-speed`)
  every hydrophone is within <!-- reproduce:absmax:@studies/propagation_error/bistatic/propagation_error_uniform.json:hydrophones[*]/error_ms|absf2 -->0.07<!-- /reproduce --> ms on `bistatic` and <!-- reproduce:absmax:@studies/propagation_error/multistatic_b/propagation_error_uniform.json:hydrophones[*]/error_ms|absf2 -->0.05<!-- /reproduce --> ms on `multistatic_b`. What fails the
  criterion is the water, not the scheme.
* **harbor** (5 m, 4 cells per wavelength at 78 Hz): <!-- reproduce:range:propagation_error_ms/harbor/hydrophones[*]/error_ms|+f2 -->-0.42 to +0.21<!-- /reproduce --> ms over <!-- reproduce:range:propagation_error_ms/harbor/hydrophones[*]/range_m|f0 -->153 to 1047<!-- /reproduce --> m. The straight path
  through the sound speed map gives <!-- reproduce:range:propagation_error_ms/harbor/hydrophones[*]/straight_path_ms|+f2 -->-0.51 to +0.37<!-- /reproduce --> ms (shallow water slower than the chart's median speed on
  some paths, deeper and faster water on others); with uniform water the scheme's own error is
  <!-- reproduce:@studies/propagation_error/harbor/propagation_error_uniform.json:hydrophones[0]/error_ms|+f2 -->+0.00<!-- /reproduce -->, <!-- reproduce:@studies/propagation_error/harbor/propagation_error_uniform.json:hydrophones[1]/error_ms|+f2 -->+0.00<!-- /reproduce -->, <!-- reproduce:@studies/propagation_error/harbor/propagation_error_uniform.json:hydrophones[2]/error_ms|+f2 -->+0.00<!-- /reproduce --> and <!-- reproduce:@studies/propagation_error/harbor/propagation_error_uniform.json:hydrophones[3]/error_ms|+f2 -->-0.21<!-- /reproduce --> ms at hydrophones 1 to 4 (the fourth near the
  shore). The same uniform water with the leapfrog gives <!-- reproduce:range:@studies/propagation_error/harbor/propagation_error_leapfrog_uniform.json:hydrophones[*]/error_ms|+f1 -->-10.6 to -1.8<!-- /reproduce --> ms: the
  <!-- reproduce:absmax:@studies/forward_fidelity/summary_leapfrog.json:propagation_error_ms/harbor/hydrophones[*]/error_ms|absf1 -->12.7<!-- /reproduce --> ms of the earlier leapfrog recording was the scheme's dispersion, not the shallow water. A 2.5 m grid gives
  <!-- reproduce:@studies/propagation_error/harbor/propagation_error_dx2.5.json:hydrophones[0]/error_ms|+f2 -->-0.47<!-- /reproduce -->, <!-- reproduce:@studies/propagation_error/harbor/propagation_error_dx2.5.json:hydrophones[1]/error_ms|+f2 -->+0.05<!-- /reproduce -->, <!-- reproduce:@studies/propagation_error/harbor/propagation_error_dx2.5.json:hydrophones[2]/error_ms|+f2 -->-0.42<!-- /reproduce --> and <!-- reproduce:@studies/propagation_error/harbor/propagation_error_dx2.5.json:hydrophones[3]/error_ms|+f2 -->+0.68<!-- /reproduce --> ms at hydrophones 1 to 4 against the 5 m grid's
  <!-- reproduce:propagation_error_ms/harbor/hydrophones[0]/error_ms|+f2 -->-0.42<!-- /reproduce -->, <!-- reproduce:propagation_error_ms/harbor/hydrophones[1]/error_ms|+f2 -->+0.10<!-- /reproduce -->, <!-- reproduce:propagation_error_ms/harbor/hydrophones[2]/error_ms|+f2 -->-0.42<!-- /reproduce --> and <!-- reproduce:propagation_error_ms/harbor/hydrophones[3]/error_ms|+f2 -->+0.21<!-- /reproduce --> ms, and with uniform water <!-- reproduce:@studies/propagation_error/harbor/propagation_error_uniform_dx2.5.json:hydrophones[3]/error_ms|+f2 -->+0.26<!-- /reproduce --> ms at hydrophone 4 against
  <!-- reproduce:@studies/propagation_error/harbor/propagation_error_uniform.json:hydrophones[3]/error_ms|+f2 -->-0.21<!-- /reproduce -->: the change at hydrophone 4 is the same with uniform water, so it is the shore reflection inside the
  long gate a 12 Hz pulse needs; the 5 m grid is not refined.

The leapfrog recordings before this change arrived <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:propagation_error_ms/[bms][iu]*/hydrophones[*]/error_ms|+f1 -->-5.2 to -1.1<!-- /reproduce --> ms off over
<!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:propagation_error_ms/[bms][iu]*/hydrophones[*]/range_m|f0 -->135 to 344<!-- /reproduce --> m on the 0.5 m and 1 m grids and <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:propagation_error_ms/harbor/hydrophones[*]/error_ms|+f1 -->-12.7 to -3.3<!-- /reproduce --> ms on the 5 m harbor. Those numbers
include one sample of a trace timing error (the solver stored the field at (m + 1) dt in column m), fixed with the
switch.

### Before and after the k space recordings

`scripts/reproduce.py --update` on every recording with the processing configs unchanged (no parameter was changed;
the source level calibration constant was not refitted). Before: `output/reproduce/summary.json` at commit `8cfaabb`
(`output/studies/forward_fidelity/summary_leapfrog.json`), the leapfrog recordings; after: the k space recordings
(`output/studies/forward_fidelity/summary_kspace.json`, the summary as that change committed it). "Moved": by more
than the reproduce tolerance (Pd 3 percentage points, median error 0.5 m, false tracks exactly). Pd and median error
are of the object's own track kind (radiator tracks for what radiates, echo tracks for silent bodies);
`examples/harbor/01_datasets/harbor_numbers_before_after.py` writes the table
(`output/studies/forward_fidelity/before_after.md`).

<!-- reproduce-table:file:studies/forward_fidelity/before_after.md -->
| Recording | Propagation error, worst hydrophone [ms] before / after | Object (track kind) | Pd before / after | Median error [m] before / after | Moved |
|---|---|---|---|---|---|
| simple | -5.23 / +0.02 | ferry (radiator) | 100 % / 100 % | 1.1 / 0.5 | error |
| simple | -5.23 / +0.02 | boat (radiator) | 100 % / 100 % | 0.6 / 0.3 |  |
| simple | -5.23 / +0.02 | uuv1 (echo) | 25 % / 28 % | 2.5 / 2.7 | Pd |
| simple | -5.23 / +0.02 | wreck (echo) | 100 % / 86 % | 2.0 / 2.8 | Pd, error |
| simple | -5.23 / +0.02 | false tracks (per minute) | 3 (9.0) / 2 (6.0) | | count |
| bistatic | -3.17 / -0.21 | ferry (radiator) | 100 % / 100 % | 0.9 / 0.5 |  |
| bistatic | -3.17 / -0.21 | uuv1 (echo) | 59 % / 67 % | 3.1 / 1.9 | Pd, error |
| bistatic | -3.17 / -0.21 | false tracks (per minute) | 2 (6.0) / 3 (9.0) | | count |
| multistatic | -3.17 / -0.21 | ferry (radiator) | 100 % / 100 % | 1.4 / 0.5 | error |
| multistatic | -3.17 / -0.21 | tug (radiator) | 64 % / 100 % | 2.2 / 0.7 | Pd, error |
| multistatic | -3.17 / -0.21 | boat (radiator) | 86 % / 100 % | 2.1 / 1.0 | Pd, error |
| multistatic | -3.17 / -0.21 | uuv2 (echo) | 0 % / 83 % | - / 1.7 | Pd |
| multistatic | -3.17 / -0.21 | false tracks (per minute) | 3 (9.0) / 2 (6.0) | | count |
| harbor | -12.71 / -0.42 | ferry (radiator) | 56 % / 72 % | 2.1 / 1.2 | Pd, error |
| harbor | -12.71 / -0.42 | tug (radiator) | 0 % / 2 % | - / 2.0 |  |
| harbor | -12.71 / -0.42 | false tracks (per minute) | 6 (3.0) / 5 (2.5) | | count |
| simple_seed1 | -5.23 / +0.02 | ferry (radiator) | 100 % / 100 % | 1.1 / 0.5 | error |
| simple_seed1 | -5.23 / +0.02 | boat (radiator) | 100 % / 100 % | 0.6 / 0.2 |  |
| simple_seed1 | -5.23 / +0.02 | uuv1 (echo) | 25 % / 23 % | 2.7 / 2.8 |  |
| simple_seed1 | -5.23 / +0.02 | wreck (echo) | 100 % / 67 % | 2.0 / 2.9 | Pd, error |
| simple_seed1 | -5.23 / +0.02 | false tracks (per minute) | 3 (9.0) / 1 (3.0) | | count |
| simple_seed2 | -5.23 / +0.02 | ferry (radiator) | 100 % / 100 % | 1.1 / 0.5 | error |
| simple_seed2 | -5.23 / +0.02 | boat (radiator) | 100 % / 100 % | 0.5 / 0.2 |  |
| simple_seed2 | -5.23 / +0.02 | uuv1 (echo) | 25 % / 30 % | 3.5 / 3.0 | Pd, error |
| simple_seed2 | -5.23 / +0.02 | wreck (echo) | 89 % / 86 % | 2.3 / 2.8 | Pd, error |
| simple_seed2 | -5.23 / +0.02 | false tracks (per minute) | 2 (6.0) / 1 (3.0) | | count |
| simple_seed3 | -5.23 / +0.02 | ferry (radiator) | 100 % / 100 % | 1.1 / 0.5 | error |
| simple_seed3 | -5.23 / +0.02 | boat (radiator) | 100 % / 100 % | 0.6 / 0.2 |  |
| simple_seed3 | -5.23 / +0.02 | uuv1 (echo) | 25 % / 25 % | 3.2 / 2.6 | error |
| simple_seed3 | -5.23 / +0.02 | wreck (echo) | 83 % / 98 % | 1.7 / 2.5 | Pd, error |
| simple_seed3 | -5.23 / +0.02 | false tracks (per minute) | 2 (6.0) / 2 (6.0) | |  |
| bistatic_seed1 | -3.17 / -0.21 | ferry (radiator) | 100 % / 100 % | 0.9 / 0.5 |  |
| bistatic_seed1 | -3.17 / -0.21 | uuv1 (echo) | 55 % / 69 % | 3.1 / 1.9 | Pd, error |
| bistatic_seed1 | -3.17 / -0.21 | false tracks (per minute) | 3 (9.0) / 2 (6.0) | | count |
| bistatic_seed2 | -3.17 / -0.21 | ferry (radiator) | 100 % / 100 % | 0.9 / 0.5 |  |
| bistatic_seed2 | -3.17 / -0.21 | uuv1 (echo) | 56 % / 59 % | 3.4 / 2.1 | Pd, error |
| bistatic_seed2 | -3.17 / -0.21 | false tracks (per minute) | 3 (9.0) / 2 (6.0) | | count |
| bistatic_seed3 | -3.17 / -0.21 | ferry (radiator) | 100 % / 100 % | 0.9 / 0.5 |  |
| bistatic_seed3 | -3.17 / -0.21 | uuv1 (echo) | 55 % / 67 % | 3.0 / 2.1 | Pd, error |
| bistatic_seed3 | -3.17 / -0.21 | false tracks (per minute) | 3 (9.0) / 2 (6.0) | | count |
| multistatic_seed1 | -3.17 / -0.21 | ferry (radiator) | 100 % / 100 % | 1.2 / 0.5 | error |
| multistatic_seed1 | -3.17 / -0.21 | tug (radiator) | 65 % / 100 % | 2.5 / 0.7 | Pd, error |
| multistatic_seed1 | -3.17 / -0.21 | boat (radiator) | 85 % / 100 % | 3.0 / 0.9 | Pd, error |
| multistatic_seed1 | -3.17 / -0.21 | uuv2 (echo) | 0 % / 83 % | - / 1.3 | Pd |
| multistatic_seed1 | -3.17 / -0.21 | false tracks (per minute) | 4 (12.0) / 2 (6.0) | | count |
| multistatic_seed2 | -3.17 / -0.21 | ferry (radiator) | 100 % / 100 % | 1.2 / 0.5 | error |
| multistatic_seed2 | -3.17 / -0.21 | tug (radiator) | 64 % / 100 % | 2.3 / 0.7 | Pd, error |
| multistatic_seed2 | -3.17 / -0.21 | boat (radiator) | 65 % / 100 % | 2.4 / 0.9 | Pd, error |
| multistatic_seed2 | -3.17 / -0.21 | uuv1 (echo) | 0 % / 30 % | - / 2.3 | Pd |
| multistatic_seed2 | -3.17 / -0.21 | uuv2 (echo) | 0 % / 89 % | - / 1.5 | Pd |
| multistatic_seed2 | -3.17 / -0.21 | false tracks (per minute) | 3 (9.0) / 1 (3.0) | | count |
| multistatic_seed3 | -3.17 / -0.21 | ferry (radiator) | 100 % / 100 % | 1.3 / 0.5 | error |
| multistatic_seed3 | -3.17 / -0.21 | tug (radiator) | 73 % / 100 % | 2.5 / 0.7 | Pd, error |
| multistatic_seed3 | -3.17 / -0.21 | boat (radiator) | 97 % / 100 % | 2.1 / 0.9 | Pd, error |
| multistatic_seed3 | -3.17 / -0.21 | uuv2 (echo) | 0 % / 86 % | - / 1.7 | Pd |
| multistatic_seed3 | -3.17 / -0.21 | false tracks (per minute) | 3 (9.0) / 2 (6.0) | | count |
| harbor_seed1 | -12.71 / -0.42 | ferry (radiator) | 56 % / 72 % | 2.1 / 1.0 | Pd, error |
| harbor_seed1 | -12.71 / -0.42 | tug (radiator) | 0 % / 6 % | - / 2.0 | Pd |
| harbor_seed1 | -12.71 / -0.42 | false tracks (per minute) | 5 (2.5) / 4 (2.0) | | count |
| harbor_seed2 | -12.71 / -0.42 | ferry (radiator) | 54 % / 78 % | 2.1 / 1.2 | Pd, error |
| harbor_seed2 | -12.71 / -0.42 | tug (radiator) | 0 % / 9 % | - / 2.0 | Pd |
| harbor_seed2 | -12.71 / -0.42 | false tracks (per minute) | 7 (3.5) / 4 (2.0) | | count |
| harbor_seed3 | -12.71 / -0.42 | ferry (radiator) | 52 % / 61 % | 2.1 / 1.2 | Pd, error |
| harbor_seed3 | -12.71 / -0.42 | tug (radiator) | 0 % / 6 % | - / 2.0 | Pd |
| harbor_seed3 | -12.71 / -0.42 | false tracks (per minute) | 4 (2.0) / 4 (2.0) | |  |
| multistatic_b | -3.55 / -0.22 | ferry (radiator) | 100 % / 100 % | 2.1 / 0.5 | error |
| multistatic_b | -3.55 / -0.22 | tug (radiator) | 18 % / 86 % | 1.9 / 0.7 | Pd, error |
| multistatic_b | -3.55 / -0.22 | boat (radiator) | 100 % / 80 % | 0.9 / 0.5 | Pd |
| multistatic_b | -3.55 / -0.22 | uuv1 (echo) | 0 % / 27 % | - / 2.1 | Pd |
| multistatic_b | -3.55 / -0.22 | uuv2 (echo) | 0 % / 45 % | - / 1.7 | Pd |
| multistatic_b | -3.55 / -0.22 | false tracks (per minute) | 4 (12.0) / 6 (6.0) | | count |
<!-- /reproduce-table -->

Criteria (fixed before the test recordings existed): <!-- reproduce:ntrue:@studies/forward_fidelity/summary_leapfrog.json:criteria[*]/passed -->0<!-- /reproduce --> of <!-- reproduce:count:@studies/forward_fidelity/summary_leapfrog.json:criteria[*]/passed -->4<!-- /reproduce --> were met before and
<!-- reproduce:ntrue:@studies/forward_fidelity/summary_kspace.json:criteria[*]/passed -->0<!-- /reproduce --> of <!-- reproduce:count:@studies/forward_fidelity/summary_kspace.json:criteria[*]/passed -->4<!-- /reproduce --> after. Every ship on `multistatic_b` with Pd at least 0.8: the tug
<!-- reproduce:@studies/forward_fidelity/summary_leapfrog.json:criteria[0]/values/tug|f2 -->0.18<!-- /reproduce --> before and <!-- reproduce:@studies/forward_fidelity/summary_kspace.json:criteria[0]/values/tug|f2 -->0.86<!-- /reproduce --> after, the boat <!-- reproduce:@studies/forward_fidelity/summary_leapfrog.json:criteria[0]/values/boat|f2 -->1.00<!-- /reproduce --> and <!-- reproduce:@studies/forward_fidelity/summary_kspace.json:criteria[0]/values/boat|f3 -->0.796<!-- /reproduce --> (printed
<!-- reproduce:@studies/forward_fidelity/summary_kspace.json:criteria[0]/values/boat|pct -->80 %<!-- /reproduce --> in the table, below 0.8). False tracks at most 2 per minute on every test recording:
<!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:criteria[1]/values/*|f1 -->2.0 to 12.0<!-- /reproduce --> before, <!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:criteria[1]/values/*|f1 -->2.0 to 6.0<!-- /reproduce --> after. The wreck by its echo with Pd at least 0.9 on seeds 1 to
3: <!-- reproduce:@studies/forward_fidelity/summary_leapfrog.json:criteria[2]/values/simple_seed1|f2 -->1.00<!-- /reproduce -->, <!-- reproduce:@studies/forward_fidelity/summary_leapfrog.json:criteria[2]/values/simple_seed2|f2 -->0.89<!-- /reproduce -->, <!-- reproduce:@studies/forward_fidelity/summary_leapfrog.json:criteria[2]/values/simple_seed3|f2 -->0.83<!-- /reproduce --> before, <!-- reproduce:@studies/forward_fidelity/summary_kspace.json:criteria[2]/values/simple_seed1|f2 -->0.67<!-- /reproduce -->, <!-- reproduce:@studies/forward_fidelity/summary_kspace.json:criteria[2]/values/simple_seed2|f2 -->0.86<!-- /reproduce -->, <!-- reproduce:@studies/forward_fidelity/summary_kspace.json:criteria[2]/values/simple_seed3|f2 -->0.98<!-- /reproduce --> after. The size order: not
testable before or after.

**What the propagation error explains.** The direct paths now arrive when the chart's speed says, so every free field
replica fits the recorded wave:

* The ships are located closer: the median error of every radiator track on the 0.5 m and 1 m recordings is
  <!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:recordings/[bms][iu]*/objects/[ftb]*/median_error_m/radiator|f1 -->0.2 to 1.0<!-- /reproduce --> m, from <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:recordings/[bms][iu]*/objects/[ftb]*/median_error_m/radiator|f1 -->0.5 to 3.0<!-- /reproduce --> m. The 5 m harbor's ferry is tracked in
  <!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:recordings/harbor*/objects/ferry/pd/radiator|pct -->61 to 78 %<!-- /reproduce --> of the frames, from <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:recordings/harbor*/objects/ferry/pd/radiator|pct -->52 to 56 %<!-- /reproduce -->, <!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:recordings/harbor*/objects/ferry/median_error_m/radiator|f1 -->1.0 to 1.2<!-- /reproduce --> m off, from
  <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:recordings/harbor*/objects/ferry/median_error_m/radiator|f1 -->2.1<!-- /reproduce --> m.
* The multistatic tug and boat are tracked in <!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:recordings/multistatic_seed*/objects/[tb]*/pd/radiator|pct -->100 %<!-- /reproduce --> of the frames of every multistatic seed,
  from <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:recordings/multistatic_seed*/objects/[tb]*/pd/radiator|pct -->64 to 97 %<!-- /reproduce -->.
* The echoes focus: the bistatic UUV is tracked in <!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:recordings/bistatic*/objects/uuv1/pd/echo|pct -->59 to 69 %<!-- /reproduce --> of the frames,
  <!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:recordings/bistatic*/objects/uuv1/median_error_m/echo|f1 -->1.9 to 2.1<!-- /reproduce --> m off, from <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:recordings/bistatic*/objects/uuv1/pd/echo|pct -->55 to 59 %<!-- /reproduce --> and
  <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:recordings/bistatic*/objects/uuv1/median_error_m/echo|f1 -->3.0 to 3.4<!-- /reproduce --> m; on the multistatic recordings the second UUV is tracked by its echo in
  <!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:recordings/multistatic_seed*/objects/uuv2/pd/echo|pct -->83 to 89 %<!-- /reproduce --> of the frames of the seeds, from <!-- reproduce:max:@studies/forward_fidelity/summary_leapfrog.json:recordings/multistatic_seed*/objects/uuv2/pd/echo|pct -->0 %<!-- /reproduce -->. The bistatic UUV is the movement the
  earlier history attributed to this error.
* False tracks fall on most test recordings (<!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:recordings/[bms][iu]*_seed*/false_tracks/per_minute|f1 -->3.0 to 6.0<!-- /reproduce --> per minute on the 0.5 m and 1 m seeds, from
  <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:recordings/[bms][iu]*_seed*/false_tracks/per_minute|f1 -->6.0 to 12.0<!-- /reproduce -->): on the simple harbor the passive chain's track of the ferry's echo off the wreck now sits close
  enough to the wreck to be matched to it, and is counted as the wreck seen by its own noise, not as a false track.

**What it does not explain.**

* **The simple harbor's wreck** is tracked by its echo in <!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:recordings/simple_seed*/objects/wreck/pd/echo|pct -->67 to 98 %<!-- /reproduce --> of the frames of the test seeds
  (<!-- reproduce:@studies/forward_fidelity/summary_kspace.json:recordings/simple/objects/wreck/pd/echo|pct -->86 %<!-- /reproduce --> on the tuning recording), from <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:recordings/simple_seed*/objects/wreck/pd/echo|pct -->83 to 100 %<!-- /reproduce -->, and <!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:recordings/simple*/objects/wreck/median_error_m/echo|f1 -->2.5 to 2.9<!-- /reproduce --> m off, from
  <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:recordings/simple*/objects/wreck/median_error_m/echo|f1 -->1.7 to 2.3<!-- /reproduce --> m, although that water had the largest error before and has none now. The echo map is not weaker: a
  peak of it is within 5 m of the wreck in <!-- reproduce:range:@studies/forward_fidelity/summary_kspace.json:recordings/simple*/peaks/echo/wreck|pct -->99 %<!-- /reproduce --> of the frames, against <!-- reproduce:range:@studies/forward_fidelity/summary_leapfrog.json:recordings/simple*/peaks/echo/wreck|pct -->99 to 100 %<!-- /reproduce -->
  before. What changed is the echo track's continuity: it is lost and confirmed again while the passive chain holds a
  confirmed track of the ferry's echo off the wreck, and echo peaks within the 25 m exclusion radius of a confirmed
  radiator track are left out. That is a likely mechanism, not a measured cause.
* **`multistatic_b`** is a different record (60 s, the boat and the ferry stationary at the end), so its movements (the
  tug <!-- reproduce:@studies/forward_fidelity/summary_leapfrog.json:recordings/multistatic_b/objects/tug/pd/radiator|pct -->18 %<!-- /reproduce --> to <!-- reproduce:@studies/forward_fidelity/summary_kspace.json:recordings/multistatic_b/objects/tug/pd/radiator|pct -->86 %<!-- /reproduce -->, the boat <!-- reproduce:@studies/forward_fidelity/summary_leapfrog.json:recordings/multistatic_b/objects/boat/pd/radiator|pct -->100 %<!-- /reproduce --> to <!-- reproduce:@studies/forward_fidelity/summary_kspace.json:recordings/multistatic_b/objects/boat/pd/radiator|pct -->80 %<!-- /reproduce -->, the UUVs <!-- reproduce:@studies/forward_fidelity/summary_kspace.json:recordings/multistatic_b/objects/uuv1/pd/echo|pct -->27 %<!-- /reproduce --> and <!-- reproduce:@studies/forward_fidelity/summary_kspace.json:recordings/multistatic_b/objects/uuv2/pd/echo|pct -->45 %<!-- /reproduce -->, false tracks
  <!-- reproduce:@studies/forward_fidelity/summary_leapfrog.json:recordings/multistatic_b/false_tracks/per_minute|f1 -->12.0<!-- /reproduce --> to <!-- reproduce:@studies/forward_fidelity/summary_kspace.json:recordings/multistatic_b/false_tracks/per_minute|f1 -->6.0<!-- /reproduce --> per minute) are not the propagation error's alone.
* The bistatic tuning recording has <!-- reproduce:@studies/forward_fidelity/summary_kspace.json:recordings/bistatic/false_tracks/count -->3<!-- /reproduce --> false tracks, from <!-- reproduce:@studies/forward_fidelity/summary_leapfrog.json:recordings/bistatic/false_tracks/count -->2<!-- /reproduce -->.

**The bistatic recording given the true waveform and the exact track.** The stage 1 echo chain
(`harbor_picture_stage1.py --dataset bistatic --chain echo --split tuning`, output in
`output/studies/forward_fidelity/stage1_bistatic/`; given the true waveform and the exact track, beside everything
estimated from the traces) on the k space tuning recording: the UUV is found in
<!-- reproduce:@studies/forward_fidelity/stage1_bistatic/results.json:entries/bistatic?echo?ferry?true_waveform_exact_track/tuning/per_object[0]/pd|pct -->15 %<!-- /reproduce --> of the frames given the true waveform and the exact track, against
<!-- reproduce:@studies/forward_fidelity/stage1_bistatic/results.json:entries/bistatic?echo?ferry?all_estimated/tuning/per_object[0]/pd|pct -->56 %<!-- /reproduce --> with everything estimated (before, on the leapfrog recording:
<!-- reproduce:@studies/forward_fidelity/stage1_leapfrog_results.json:entries/bistatic?echo?ferry?true_waveform_exact_track/tuning/per_object[0]/pd|pct -->3 %<!-- /reproduce --> against <!-- reproduce:@studies/forward_fidelity/stage1_leapfrog_results.json:entries/bistatic?echo?ferry?all_estimated/tuning/per_object[0]/pd|pct -->27 %<!-- /reproduce -->, `stage1_leapfrog_results.json`). The true waveform still locates the UUV worse.
The measured reason (`scripts/diagnose_true_waveform.py components lags`, its log
`output/studies/forward_fidelity/diagnose_true_waveform.log` and the numbers of the log in `diagnose_true_waveform.json`):

* It is no longer the timing: the data reference is offset from the true waveform by
  <!-- reproduce:range:@studies/forward_fidelity/diagnose_true_waveform.json:components/[ad][la]*/delay_ms|+f2 -->-0.07 to -0.05<!-- /reproduce --> ms.
* It is not the removal: given the true waveform and the exact track the direct path is removed by
  <!-- reproduce:@studies/forward_fidelity/diagnose_true_waveform.json:components/true_waveform_exact_track/depth_db|f1 -->21.9<!-- /reproduce --> dB (<!-- reproduce:@studies/forward_fidelity/diagnose_true_waveform.json:components/true_waveform_refined_track/depth_db|f1 -->11.0<!-- /reproduce --> dB with the refined track, <!-- reproduce:@studies/forward_fidelity/diagnose_true_waveform.json:components/all_estimated/depth_db|f1 -->11.3<!-- /reproduce --> dB with everything estimated), and the UUV echo
  loses <!-- reproduce:@studies/forward_fidelity/diagnose_true_waveform.json:components/true_waveform_exact_track/echo_loss_db|f1 -->0.5<!-- /reproduce --> dB.
* It is the map of each component: with the true waveform the UUV echo alone maps lower (map value at the UUV
  <!-- reproduce:@studies/forward_fidelity/diagnose_true_waveform.json:components/true_waveform_exact_track/maps/echo/at_uuv|f0 -->1070<!-- /reproduce --> against <!-- reproduce:@studies/forward_fidelity/diagnose_true_waveform.json:components/all_estimated/maps/echo/at_uuv|f0 -->2970<!-- /reproduce --> with the data reference and the refined track) while the rest of the traces (noise and
  reverberation) map higher (largest value <!-- reproduce:@studies/forward_fidelity/diagnose_true_waveform.json:components/true_waveform_exact_track/maps/rest/max|f0 -->575<!-- /reproduce --> against <!-- reproduce:@studies/forward_fidelity/diagnose_true_waveform.json:components/all_estimated/maps/rest/max|f0 -->257<!-- /reproduce -->) and so does the residual direct path
  (<!-- reproduce:@studies/forward_fidelity/diagnose_true_waveform.json:components/true_waveform_exact_track/maps/direct_residual/max|f0 -->747<!-- /reproduce --> against <!-- reproduce:@studies/forward_fidelity/diagnose_true_waveform.json:components/all_estimated/maps/direct_residual/max|f0 -->534<!-- /reproduce -->). The data reference is the ship's sound as the array receives it, so the
  matched filter it gives is weighted by the received spectrum; the true waveform is the emitted signal, weighted
  otherwise across the band. That the weighting is the cause is an inference from these three numbers, not a
  separate experiment.
* Not explained: the recorded direct component at the hydrophones of the moving ferry lags the true waveform propagated
  at the chart's speed by <!-- reproduce:range:@studies/forward_fidelity/diagnose_true_waveform.json:lags/*/median_ms|f2 -->0.17 to 0.88<!-- /reproduce --> ms (median per hydrophone over 1 s blocks, correlation
  <!-- reproduce:range:@studies/forward_fidelity/diagnose_true_waveform.json:lags/*/corr|f2 -->0.69 to 0.71<!-- /reproduce -->), more than the stationary pulse's error at the same hydrophones.

## Physical and interpretation limits

This is a **2-D plan-view scalar pressure model**, not a 3-D shallow-water solver. The depth map
is informational; distance from shore is used to derive a small effective sound-speed drop.
Land cells are pressure-release boundaries. Shore loss is a numerical damping ramp that limits
ringing, not a measured seabed absorption law. The pressure-only solver has no rigid Neumann hull
boundary; `Scatterer` bodies use smooth pressure masks (soft/pressure-release or absorbing
approximations). Body geometry should span multiple grid cells for meaningful shape resolution.

The dataset builder caches each radiating vessel's unit-level FDTD gather and then mixes levels
and noise linearly. Scattering is represented by a run with a body minus the matching run
without it; by default each body is run on its own, so multiple scattering between distinct
bodies is not included. `render_vessel_gathers(combined_bodies=True)` (the multistatic dataset)
instead runs every radiating vessel once with all the bodies present and once with none, which
keeps the multiple scattering but loses the per body split.

## Modules (`src/acoustic_sim/water/`)

| Module | Role |
|--------|------|
| `stream.py` | The system: `HarborPicture`, `run`, `load_processing`, the `picture.jsonl` records, the command line |
| `streaming.py` | `StreamingPassive`, `RadiatorTracker` (radiator and echo tracks), `StreamingEcho`, `ais_crosscheck` |
| `characteristics.py` | The measured characteristics of every track |
| `detection.py` | Spectra, channel QC, common mode, noise floor, the matched field scanner and its interference projector |
| `bistatic.py` | The data reference beam, the ECA, `BistaticConfig` (the batch chain `run_bistatic` is history) |
| `propagation.py` | `FreeFieldPropagator` and `green2d`: the replicas from the chart alone |
| `sensors.py`, `ais.py` | `SensorLayout` (pressure and strain DAS, hydrophones, `collapse`); AIS reports and the predictor |
| `scoreboard.py`, `evaluate.py` | Scoring against the `Truth` (`score_picture`, `match_tracks`, `peak_hits`; the stage 1 threshold method as a diagnostic) |
| `plotting.py` | The chart and geometry overlays, `HarborPictureView` |
| `scenario.py`, `environment.py`, `vessels.py`, `signatures.py`, `noise.py`, `ais_sim.py`, `fdtd_runner.py`, `fdtd_cuda_bodies.py`, `scatterer.py`, `dispersion.py`, `dataset.py`, `sensor_sampling.py` | The forward model: scenario, land and sound speed, tracks and signatures, noise, synthetic AIS, the FDTD runner and its bodies, the mixer and the observation / truth files |
| `diagnostics.py` | Forward model knowledge on the inference path, flagged in the provenance (`SimulatorAwarePropagator`, ...) |
| `device.py` | CPU / GPU selection |
| `suppression.py`, `occupancy.py`, `localiser.py`, `classifier.py`, `motion.py` | History ([Architecture](architecture.md#history-water-processing)) |

`acoustic_sim.harbor` is a compatibility alias of `acoustic_sim.water`.

## Tests

Run the test files of what changed (`./dev.sh run python -m pytest tests/harbor/<file>`); the GPU variants skip
without CuPy or a GPU. The stage 2 additions:

* `test_characteristics.py`: every characteristic on synthetic inputs with known answers (the fit of a straight track and
  its standard error, the half maximum widths and direction of an elliptical Gaussian peak, Parseval's band rms, the
  least squares source spectrum and the levels on pressure and strain cables, tonal lines over a flat floor, the echo
  strength and the size ratio of an echo of known strength, the AIS identity and distance with only past reports);
* `test_ci_end_to_end.py`: the tiny scenario through the dataset build, the stream and the characteristics on the CPU
  (ferry tracked within 3 m, the wreck the peak of the echo map), and the run block of every entry point;
* `test_streaming.py`: the streaming chain, its causality, the tracker, the echo chain end to end.

The rest cover the environment and sensors, the FDTD runner, signatures, AIS, noise, the dataset mixer, the bistatic ECA
and the vectorized paths, the performance paths and the stage 1 scoreboard ([history](harbor_history.md)).

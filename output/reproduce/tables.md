### Real time, tuning recordings beside test recordings

Pd: fraction of frames, from the first frame the object could have been confirmed, with a confirmed track matched to it within 5 m (radiator tracks for what radiates, echo tracks for silent bodies); test: the three noise seeds (each value) and multistatic_b. False tracks: confirmed tracks matched to no object, per minute.

| Recording | Split | Compute per s of data | Latency (median) | Pd per object | False tracks (per minute) |
|---|---|---|---|---|---|
| simple | tuning | 0.72 | 1.9 s | ferry 100 %; boat 100 %; uuv1 28 %; wreck 86 % | 2 (6.0) |
| simple | test | 0.70 / 0.70 / 0.70 | 1.9 s / 1.9 s / 1.9 s | ferry 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 23 % / 30 % / 25 %; wreck 67 % / 86 % / 98 % | 1 (3.0) / 1 (3.0) / 2 (6.0) |
| bistatic | tuning | 1.07 | 2.0 s | ferry 100 %; uuv1 67 % | 3 (9.0) |
| bistatic | test | 1.24 / 1.08 / 1.16 | 2.0 s / 2.0 s / 2.0 s | ferry 100 % / 100 % / 100 %; uuv1 69 % / 59 % / 67 % | 2 (6.0) / 2 (6.0) / 2 (6.0) |
| multistatic | tuning | 1.49 | 2.6 s | ferry 100 %; tug 100 %; boat 100 %; uuv2 83 %; not tracked: uuv1, diver, piling1, piling2, piling3, piling4, piling5, piling6, moored | 2 (6.0) |
| multistatic | test | 1.35 / 1.50 / 1.72 | 2.6 s / 2.6 s / 2.6 s | ferry 100 % / 100 % / 100 %; tug 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 0 % / 30 % / 0 %; uuv2 83 % / 89 % / 86 %; not tracked: diver, piling1, piling2, piling3, piling4, piling5, piling6, moored | 2 (6.0) / 1 (3.0) / 2 (6.0) |
| harbor | tuning | 0.53 | 1.1 s | ferry 72 %; tug 2 %; moored 0 %; uuv1 0 %; uuv2 0 %; diver 0 % | 5 (2.5) |
| harbor | test | 0.52 / 0.52 / 0.53 | 1.1 s / 1.1 s / 1.1 s | ferry 72 % / 78 % / 61 %; tug 6 % / 9 % / 6 %; moored 0 % / 0 % / 0 %; uuv1 0 % / 0 % / 0 %; uuv2 0 % / 0 % / 0 %; diver 0 % / 0 % / 0 % | 4 (2.0) / 4 (2.0) / 4 (2.0) |
| multistatic_b | test | 1.50 | 2.5 s | ferry 100 %; tug 86 %; boat 80 %; uuv1 27 %; uuv2 45 %; not tracked: diver, piling1, piling2, piling3, piling4, piling5, piling6, moored | 6 (6.0) |

### Criteria

| Criterion | Values | Result |
|---|---|---|
| every ship (ferry, tug, boat) tracked with Pd >= 0.8 on multistatic_b (radiator tracks) | {'ferry': 1.0, 'tug': 0.86, 'boat': 0.8} | **fail** |
| false tracks at most 2 per minute on every test recording | {'simple_seed1': 3.0, 'simple_seed2': 3.0, 'simple_seed3': 6.0, 'bistatic_seed1': 6.0, 'bistatic_seed2': 6.0, 'bistatic_seed3': 6.0, 'multistatic_seed1': 6.0, 'multistatic_seed2': 3.0, 'multistatic_seed3': 6.0, 'harbor_seed1': 2.0, 'harbor_seed2': 2.0, 'harbor_seed3': 2.0, 'multistatic_b': 6.0} | **fail** |
| the wreck of the simple harbor found by its echo with Pd >= 0.9 on seeds 1 to 3 (echo tracks) | {'simple_seed1': 0.67, 'simple_seed2': 0.86, 'simple_seed3': 0.98} | **fail** |
| on frames where the moored hull, a UUV and a piling are all measured, the size indicator puts the hull below the piling in at least 70 % of them (multistatic recordings, tuning and multistatic_b) | no frame had the hull, a UUV and a piling all measured: not testable | **not met: not testable** |

### Characteristics against the truth (objects detected in at least 10 frames; the multistatic recordings, and the tuning and first test recording of the simple and bistatic harbors, where bodies are tracked by their echo)

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

### Source level estimates against the truth (radiator tracks)

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

### Propagation error of the forward model (stationary pulse, arrival minus range over the chart speed)

| Recording | Band [Hz] | Chart speed [m/s] | Per hydrophone: error [ms] (direction, range) |
|---|---|---|---|
| simple | 100-380 | 1500.0 | +0.00 (207 deg, 224 m); +0.00 (126 deg, 344 m); +0.00 (312 deg, 135 m); +0.02 (72 deg, 294 m) |
| bistatic | 100-760 | 1493.7 | -0.21 (221 deg, 227 m); -0.03 (140 deg, 234 m); +0.05 (288 deg, 158 m); -0.10 (174 deg, 181 m) |
| multistatic | 100-760 | 1493.7 | -0.21 (221 deg, 227 m); -0.03 (140 deg, 234 m); +0.05 (288 deg, 158 m); -0.10 (174 deg, 181 m) |
| harbor | 12-78 | 1496.0 | -0.42 (297 deg, 1047 m); +0.10 (302 deg, 153 m); -0.42 (44 deg, 672 m); +0.21 (153 deg, 268 m) |
| multistatic_b | 100-760 | 1493.7 | -0.22 (222 deg, 205 m); -0.04 (135 deg, 231 m); +0.03 (296 deg, 152 m); -0.12 (168 deg, 167 m) |

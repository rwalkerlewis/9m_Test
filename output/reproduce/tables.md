### Real time, tuning recordings beside test recordings

Pd: fraction of frames, from the first frame the object could have been confirmed, with a confirmed track matched to it within 5 m (radiator tracks for what radiates, echo tracks for silent bodies); test: the three noise seeds (each value) and multistatic_b. False tracks: confirmed tracks matched to no object, per minute.

| Recording | Split | Compute per s of data | Latency (median) | Pd per object | False tracks (per minute) |
|---|---|---|---|---|---|
| simple | tuning | 0.71 | 1.9 s | ferry 100 %; boat 100 %; uuv1 25 %; wreck 100 % | 3 (9.0) |
| simple | test | 0.93 / 0.71 / 0.70 | 1.9 s / 1.9 s / 1.9 s | ferry 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 25 % / 25 % / 25 %; wreck 100 % / 89 % / 83 % | 3 (9.0) / 2 (6.0) / 2 (6.0) |
| bistatic | tuning | 1.21 | 2.0 s | ferry 100 %; uuv1 59 % | 2 (6.0) |
| bistatic | test | 0.97 / 1.20 / 1.21 | 2.0 s / 2.0 s / 2.0 s | ferry 100 % / 100 % / 100 %; uuv1 55 % / 56 % / 55 % | 3 (9.0) / 3 (9.0) / 3 (9.0) |
| multistatic | tuning | 1.38 | 2.6 s | ferry 100 %; tug 64 %; boat 86 %; not tracked: uuv1, uuv2, diver, piling1, piling2, piling3, piling4, piling5, piling6, moored | 3 (9.0) |
| multistatic | test | 1.52 / 1.51 / 1.47 | 2.8 s / 2.5 s / 2.6 s | ferry 100 % / 100 % / 100 %; tug 65 % / 64 % / 73 %; boat 85 % / 65 % / 97 %; not tracked: uuv1, uuv2, diver, piling1, piling2, piling3, piling4, piling5, piling6, moored | 4 (12.0) / 3 (9.0) / 3 (9.0) |
| harbor | tuning | 0.53 | 1.1 s | ferry 56 %; tug 0 %; moored 0 %; uuv1 0 %; uuv2 0 %; diver 0 % | 6 (3.0) |
| harbor | test | 0.52 / 0.52 / 0.52 | 1.1 s / 1.1 s / 1.1 s | ferry 56 % / 54 % / 52 %; tug 0 % / 0 % / 0 %; moored 0 % / 0 % / 0 %; uuv1 0 % / 0 % / 0 %; uuv2 0 % / 0 % / 0 %; diver 0 % / 0 % / 0 % | 5 (2.5) / 7 (3.5) / 4 (2.0) |
| multistatic_b | test | 1.35 | 2.8 s | ferry 100 %; tug 18 %; boat 100 %; not tracked: uuv1, uuv2, diver, piling1, piling2, piling3, piling4, piling5, piling6, moored | 4 (12.0) |

### Criteria

| Criterion | Values | Result |
|---|---|---|
| every ship (ferry, tug, boat) tracked with Pd >= 0.8 on multistatic_b (radiator tracks) | {'ferry': 1.0, 'tug': 0.18, 'boat': 1.0} | **fail** |
| false tracks at most 2 per minute on every test recording | {'simple_seed1': 9.0, 'simple_seed2': 6.0, 'simple_seed3': 6.0, 'bistatic_seed1': 9.0, 'bistatic_seed2': 9.0, 'bistatic_seed3': 9.0, 'multistatic_seed1': 12.0, 'multistatic_seed2': 9.0, 'multistatic_seed3': 9.0, 'harbor_seed1': 2.5, 'harbor_seed2': 3.5, 'harbor_seed3': 2.0, 'multistatic_b': 12.0} | **fail** |
| the wreck of the simple harbor found by its echo with Pd >= 0.9 on seeds 1 to 3 (echo tracks) | {'simple_seed1': 1.0, 'simple_seed2': 0.89, 'simple_seed3': 0.83} | **fail** |
| on frames where the moored hull, a UUV and a piling are all measured, the size indicator puts the hull below the piling in at least 70 % of them (multistatic recordings, tuning and multistatic_b) | no frame had the hull, a UUV and a piling all measured: not testable | **not met: not testable** |

### Characteristics against the truth (objects detected in at least 10 frames; the multistatic recordings, and the tuning and first test recording of the simple and bistatic harbors, where bodies are tracked by their echo)

| Recording | Object | True type, size | Seen | Detected frames | Size ratio [dB] | Size indicator | Echo extent along / across [m] | Source level [Pa] (true) | Half max width [m] | Speed [m/s] (true) |
|---|---|---|---|---|---|---|---|---|---|---|
| multistatic | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.15 (1) | 13.9 x 3.8 | 4.0 (4.0) |
| multistatic | tug | ship, - | own noise 67 | 42 | - | - | - | 0.79 (0.7) | 25.1 x 7.5 | 2.8 (3.0) |
| multistatic | boat | ship, - | own noise 66 | 57 | - | - | - | 0.54 (0.3) | 47.1 x 6.9 | 4.9 (5.0) |
| multistatic_b | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.15 (1) | 14.2 x 6.1 | 3.6 (3.5) |
| multistatic_b | tug | ship, - | own noise 12 | 12 | - | - | - | 0.68 (0.7) | 40.4 x 5.4 | 2.9 (2.5) |
| multistatic_b | boat | ship, - | own noise 66 | 66 | - | - | - | 0.45 (0.3) | 28.5 x 5.4 | 4.6 (4.5) |
| simple | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.10 (1) | 4.6 x 3.5 | 4.0 (4.0) |
| simple | boat | ship, - | own noise 66 | 66 | - | - | - | 0.27 (0.3) | 4.6 x 2.3 | 2.5 (2.5) |
| simple | uuv1 | uuv, 6 x 2 m | echo 16 | 16 | -4.9 | about a wavelength or larger | 3.5 / 3.3 | - | 4.6 x 2.3 | 1.0 (1.5) |
| simple | wreck | static, 4 x 4 m | echo 64 | 64 | -4.7 | about a wavelength or larger | 4.8 / 4.8 | - | 4.6 x 4.6 | 0.2 (0.0) |
| simple_seed1 | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.08 (1) | 4.6 x 3.5 | 4.0 (4.0) |
| simple_seed1 | boat | ship, - | own noise 66 | 66 | - | - | - | 0.26 (0.3) | 4.6 x 2.3 | 2.5 (2.5) |
| simple_seed1 | uuv1 | uuv, 6 x 2 m | echo 16 | 16 | -0.3 | about a wavelength or larger | 4.8 / 3.5 | - | 5.2 x 3.5 | 1.1 (1.5) |
| simple_seed1 | wreck | static, 4 x 4 m | echo 64, own noise 28 | 64 | -4.4 | about a wavelength or larger | 4.8 / 4.8 | 0.15 (0) | 4.6 x 4.6 | 0.2 (0.0) |
| bistatic | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.22 (1) | 11.6 x 4.1 | 4.0 (4.0) |
| bistatic | uuv1 | uuv, 3 x 1 m | own noise 55, echo 38 | 47 | +0.6 | smaller than about a wavelength | 5.8 / 3.7 | 0.18 (0) | 44.4 x 5.5 | 3.5 (2.0) |
| bistatic_seed1 | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.25 (1) | 10.4 x 4.4 | 4.0 (4.0) |
| bistatic_seed1 | uuv1 | uuv, 3 x 1 m | own noise 40, echo 36 | 58 | +1.1 | smaller than about a wavelength | 5.9 / 4.6 | 0.18 (0) | 36.0 x 4.2 | 2.2 (2.0) |

### Source level estimates against the truth (radiator tracks; calibration constant 5.1152 from the simple harbor's tuning recording)

| Recording | Object | True [Pa] | Estimate [Pa] | Error [dB] |
|---|---|---|---|---|
| simple | ferry | 1 | 1.100 | +0.8 |
| simple | boat | 0.3 | 0.270 | -0.9 |
| bistatic | ferry | 1 | 1.217 | +1.7 |
| multistatic | ferry | 1 | 1.151 | +1.2 |
| multistatic | tug | 0.7 | 0.787 | +1.0 |
| multistatic | boat | 0.3 | 0.536 | +5.0 |
| harbor | ferry | 1 | 4.837 | +13.7 |
| simple_seed1 | ferry | 1 | 1.079 | +0.7 |
| simple_seed1 | boat | 0.3 | 0.264 | -1.1 |
| simple_seed2 | ferry | 1 | 1.077 | +0.6 |
| simple_seed2 | boat | 0.3 | 0.266 | -1.0 |
| simple_seed3 | ferry | 1 | 1.065 | +0.5 |
| simple_seed3 | boat | 0.3 | 0.261 | -1.2 |
| bistatic_seed1 | ferry | 1 | 1.247 | +1.9 |
| bistatic_seed2 | ferry | 1 | 1.237 | +1.9 |
| bistatic_seed3 | ferry | 1 | 1.228 | +1.8 |
| multistatic_seed1 | ferry | 1 | 1.176 | +1.4 |
| multistatic_seed1 | tug | 0.7 | 0.799 | +1.2 |
| multistatic_seed1 | boat | 0.3 | 0.543 | +5.2 |
| multistatic_seed2 | ferry | 1 | 1.179 | +1.4 |
| multistatic_seed2 | tug | 0.7 | 0.790 | +1.0 |
| multistatic_seed2 | boat | 0.3 | 0.551 | +5.3 |
| multistatic_seed3 | ferry | 1 | 1.165 | +1.3 |
| multistatic_seed3 | tug | 0.7 | 0.794 | +1.1 |
| multistatic_seed3 | boat | 0.3 | 0.546 | +5.2 |
| harbor_seed1 | ferry | 1 | 4.791 | +13.6 |
| harbor_seed2 | ferry | 1 | 4.876 | +13.8 |
| harbor_seed3 | ferry | 1 | 4.755 | +13.5 |
| multistatic_b | ferry | 1 | 1.150 | +1.2 |
| multistatic_b | tug | 0.7 | 0.680 | -0.2 |
| multistatic_b | boat | 0.3 | 0.449 | +3.5 |

### Propagation error of the forward model (stationary pulse, arrival minus range over the chart speed)

| Recording | Band [Hz] | Chart speed [m/s] | Per hydrophone: error [ms] (direction, range) |
|---|---|---|---|
| simple | 100-380 | 1500.0 | -2.71 (207 deg, 224 m); -5.23 (126 deg, 344 m); -2.44 (312 deg, 135 m); -2.29 (72 deg, 294 m) |
| bistatic | 100-760 | 1493.7 | -3.17 (221 deg, 227 m); -3.07 (140 deg, 234 m); -1.21 (288 deg, 158 m); -1.10 (174 deg, 181 m) |
| multistatic | 100-760 | 1493.7 | -3.17 (221 deg, 227 m); -3.07 (140 deg, 234 m); -1.21 (288 deg, 158 m); -1.10 (174 deg, 181 m) |
| harbor | 12-78 | 1496.0 | -12.71 (297 deg, 1047 m); -3.33 (302 deg, 153 m); -11.56 (44 deg, 672 m); -4.17 (153 deg, 268 m) |
| multistatic_b | 100-760 | 1493.7 | -3.55 (227 deg, 271 m); -2.32 (152 deg, 219 m); -1.15 (280 deg, 202 m); -1.21 (188 deg, 196 m) |

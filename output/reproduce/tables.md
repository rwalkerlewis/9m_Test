### Real time, tuning recordings beside test recordings

Pd: fraction of frames, from the first frame the object could have been confirmed, with a confirmed track matched to it within 5 m (radiator tracks for what radiates, echo tracks for silent bodies); test: the three noise seeds (each value) and multistatic_b. False tracks: confirmed tracks matched to no object, per minute.

| Recording | Split | Compute per s of data | Latency (median) | Pd per object, old noise model | Pd per object, harbor_v2 | False tracks (per minute), old / harbor_v2 |
|---|---|---|---|---|---|---|
| simple | tuning | 1.21 | 2.8 s | ferry 100 %; boat 100 %; uuv1 58 %; wreck 100 % | ferry 100 %; boat 100 %; uuv1 20 %; wreck 98 % | 1 (3.0); harbor_v2 2 (6.0) |
| simple | test | 1.21 / 1.23 / 1.21 | 2.8 s / 2.8 s / 2.8 s | ferry 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 66 % / 64 % / 58 %; wreck 100 % / 100 % / 100 % | ferry 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 0 % / 0 % / 25 %; wreck 97 % / 98 % / 98 % | 0 (0.0) / 0 (0.0) / 1 (3.0); harbor_v2 1 (3.0) / 2 (6.0) / 1 (3.0) |
| bistatic | tuning | 1.30 | 2.6 s | ferry 100 %; uuv1 70 % | ferry 100 %; uuv1 53 % | 2 (6.0); harbor_v2 0 (0.0) |
| bistatic | test | 1.20 / 1.45 / 1.49 | 2.6 s / 2.6 s / 2.6 s | ferry 100 % / 100 % / 100 %; uuv1 70 % / 59 % / 69 % | ferry 100 % / 100 % / 100 %; uuv1 78 % / 80 % / 80 % | 1 (3.0) / 2 (6.0) / 4 (12.0); harbor_v2 0 (0.0) / 0 (0.0) / 1 (3.0) |
| multistatic | tuning | 2.85 | 5.6 s | ferry 100 %; tug 100 %; boat 100 %; uuv1 84 %; uuv2 95 %; moored 83 %; not tracked: diver, piling1, piling2, piling3, piling4, piling5, piling6 | ferry 100 %; tug 100 %; boat 100 %; uuv1 0 %; uuv2 88 %; moored 62 % | 2 (6.0); harbor_v2 2 (6.0) |
| multistatic | test | 2.93 / 2.90 / 2.90 | 5.7 s / 5.7 s / 5.7 s | ferry 100 % / 100 % / 100 %; tug 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 89 % / 80 % / 80 %; uuv2 94 % / 94 % / 91 %; moored 91 % / 83 % / 86 %; not tracked: diver, piling1, piling2, piling3, piling4, piling5, piling6 | ferry 100 % / 100 % / 100 %; tug 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 0 % / 14 % / 0 %; uuv2 91 % / 92 % / 83 %; moored 62 % / 72 % / 62 % | 3 (9.0) / 2 (6.0) / 3 (9.0); harbor_v2 1 (3.0) / 1 (3.0) / 1 (3.0) |
| harbor | tuning | 0.54 | 1.1 s | ferry 83 %; tug 4 %; moored 0 %; uuv1 0 %; uuv2 0 %; diver 0 % | ferry 93 %; tug 4 %; moored 0 %; uuv1 0 %; uuv2 0 %; diver 0 % | 3 (1.5); harbor_v2 2 (1.0) |
| harbor | test | 0.55 / 0.54 / 0.54 | 1.1 s / 1.1 s / 1.1 s | ferry 74 % / 83 % / 67 %; tug 6 % / 7 % / 6 %; moored 0 % / 0 % / 0 %; uuv1 0 % / 0 % / 0 %; uuv2 0 % / 0 % / 0 %; diver 0 % / 0 % / 0 % | ferry 69 % / 81 % / 81 %; tug 6 % / 6 % / 6 %; moored 0 % / 0 % / 0 %; uuv1 0 % / 0 % / 0 %; uuv2 0 % / 0 % / 0 %; diver 0 % / 0 % / 0 % | 4 (2.0) / 3 (1.5) / 4 (2.0); harbor_v2 3 (1.5) / 2 (1.0) / 0 (0.0) |
| multistatic_b | test | 2.96 | 5.5 s | ferry 99 %; tug 100 %; boat 70 %; uuv1 60 %; uuv2 96 %; diver 74 %; not tracked: piling1, piling2, piling3, piling4, piling5, piling6, moored | ferry 99 %; tug 100 %; boat 56 %; uuv1 46 %; uuv2 82 %; diver 0 % | 6 (6.0); harbor_v2 4 (4.0) |

### Silent bodies by their echo, old noise model beside harbor_v2 (every recording with an echo chain)

| Recording | Silent body | Pd, old noise model | Pd, harbor_v2 |
|---|---|---|---|
| simple | uuv1 | 58 % | 20 % |
| simple | wreck | 100 % | 98 % |
| bistatic | uuv1 | 70 % | 53 % |
| multistatic | uuv1 | 84 % | 0 % |
| multistatic | uuv2 | 95 % | 88 % |
| multistatic | moored | 83 % | 62 % |
| simple_seed1 | uuv1 | 66 % | 0 % |
| simple_seed1 | wreck | 100 % | 97 % |
| simple_seed2 | uuv1 | 64 % | 0 % |
| simple_seed2 | wreck | 100 % | 98 % |
| simple_seed3 | uuv1 | 58 % | 25 % |
| simple_seed3 | wreck | 100 % | 98 % |
| bistatic_seed1 | uuv1 | 70 % | 78 % |
| bistatic_seed2 | uuv1 | 59 % | 80 % |
| bistatic_seed3 | uuv1 | 69 % | 80 % |
| multistatic_seed1 | uuv1 | 89 % | 0 % |
| multistatic_seed1 | uuv2 | 94 % | 91 % |
| multistatic_seed1 | moored | 91 % | 62 % |
| multistatic_seed2 | uuv1 | 80 % | 14 % |
| multistatic_seed2 | uuv2 | 94 % | 92 % |
| multistatic_seed2 | moored | 83 % | 72 % |
| multistatic_seed3 | uuv1 | 80 % | 0 % |
| multistatic_seed3 | uuv2 | 91 % | 83 % |
| multistatic_seed3 | moored | 86 % | 62 % |
| multistatic_b | uuv1 | 60 % | 46 % |
| multistatic_b | uuv2 | 96 % | 82 % |
| multistatic_b | diver | 74 % | 0 % |

### The tuning seeds 4 to 9 (tuning recordings, both noise models)

| Recording | Seeds | Pd per object, old noise model | Pd per object, harbor_v2 | False tracks per recording, old / harbor_v2 |
|---|---|---|---|---|
| simple | 4, 5, 6, 7, 8, 9 | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 64 % / 34 % / 61 % / 39 % / 61 % / 58 %; wreck 98 % / 98 % / 100 % / 100 % / 100 % / 100 % | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 0 % / 0 % / 25 % / 0 % / 25 % / 25 %; wreck 98 % / 98 % / 100 % / 100 % / 98 % / 98 % | 1 / 1 / 2 / 1 / 0 / 1; harbor_v2 1 / 1 / 2 / 1 / 1 / 1 |
| bistatic | 4, 5, 6, 7, 8, 9 | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 69 % / 69 % / 61 % / 22 % / 61 % / 58 % | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 86 % / 84 % / 75 % / 50 % / 62 % / 84 % | 1 / 1 / 0 / 2 / 1 / 3; harbor_v2 0 / 0 / 0 / 1 / 1 / 0 |
| multistatic | 4, 5, 6, 7, 8, 9 | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; tug 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 91 % / 70 % / 58 % / 84 % / 77 % / 88 %; uuv2 97 % / 97 % / 89 % / 95 % / 95 % / 95 %; moored 75 % / 89 % / 83 % / 83 % / 84 % / 86 % | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; tug 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 80 % / 22 % / 0 % / 0 % / 0 % / 62 %; uuv2 91 % / 88 % / 83 % / 86 % / 91 % / 92 %; moored 69 % / 80 % / 64 % / 47 % / 59 % / 73 % | 2 / 3 / 2 / 2 / 3 / 2; harbor_v2 1 / 2 / 2 / 1 / 1 / 2 |
| multistatic_b | 4, 5, 6, 7, 8, 9 | ferry 95 % / 99 % / 99 % / 99 % / 99 % / 99 %; tug 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 69 % / 73 % / 69 % / 80 % / 74 % / 69 %; uuv1 62 % / 64 % / 60 % / 65 % / 61 % / 60 %; uuv2 88 % / 84 % / 88 % / 90 % / 96 % / 96 % | ferry 99 % / 99 % / 99 % / 99 % / 99 % / 99 %; tug 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 57 % / 57 % / 57 % / 57 % / 57 % / 57 %; uuv1 53 % / 57 % / 52 % / 41 % / 49 % / 58 %; uuv2 75 % / 82 % / 88 % / 83 % / 83 % / 84 % | 6 / 6 / 6 / 5 / 6 / 4; harbor_v2 4 / 4 / 4 / 4 / 4 / 4 |

### Criteria

| Criterion | Values | Result |
|---|---|---|
| every ship (ferry, tug, boat) tracked with Pd >= 0.8 on multistatic_b (radiator tracks) | {'ferry': 0.99, 'tug': 1.0, 'boat': 0.7} | **fail** |
| false tracks at most 2 per minute on every test recording | {'simple_seed1': 0.0, 'simple_seed2': 0.0, 'simple_seed3': 3.0, 'bistatic_seed1': 3.0, 'bistatic_seed2': 6.0, 'bistatic_seed3': 12.0, 'multistatic_seed1': 9.0, 'multistatic_seed2': 6.0, 'multistatic_seed3': 9.0, 'harbor_seed1': 2.0, 'harbor_seed2': 1.5, 'harbor_seed3': 2.0, 'multistatic_b': 6.0} | **fail** |
| the wreck of the simple harbor found by its echo with Pd >= 0.9 on seeds 1 to 3 (echo tracks) | {'simple_seed1': 1.0, 'simple_seed2': 1.0, 'simple_seed3': 1.0} | pass |
| on frames where the moored hull, a UUV and a piling are all measured, the size indicator puts the hull below the piling in at least 70 % of them (multistatic recordings, tuning and multistatic_b) | no frame had the hull, a UUV and a piling all measured: not testable | **not met: not testable** |

### Characteristics against the truth (objects detected in at least 10 frames; the multistatic recordings, and the tuning and first test recording of the simple and bistatic harbors, where bodies are tracked by their echo)

| Recording | Object | True type, size | Seen | Detected frames | Size ratio [dB] | Size indicator | Echo extent along / across [m] | Source level [Pa] (true) | Half max width [m] | Speed [m/s] (true) |
|---|---|---|---|---|---|---|---|---|---|---|
| multistatic | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.26 (1) | 9.2 x 2.3 | 4.0 (4.0) |
| multistatic | tug | ship, - | own noise 66 | 66 | - | - | - | 0.83 (0.7) | 15.5 x 3.5 | 3.0 (3.0) |
| multistatic | boat | ship, - | own noise 66 | 66 | - | - | - | 0.59 (0.3) | 46.9 x 6.2 | 4.9 (5.0) |
| multistatic | uuv1 | uuv, 3 x 1 m | echo 54 | 54 | -4.7 | about a wavelength or larger | 3.5 / 4.7 | - | 6.9 x 2.8 | 2.0 (2.0) |
| multistatic | uuv2 | uuv, 3 x 1 m | echo 61 | 61 | -5.4 | about a wavelength or larger | 5.2 / 3.7 | - | 6.9 x 2.3 | 1.5 (1.5) |
| multistatic | moored | ship, 30 x 6 m | echo 53 | 53 | -3.0 | about a wavelength or larger | 7.0 / 6.4 | - | 11.6 x 2.3 | 0.3 (0.0) |
| multistatic_b | ferry | ship, - | own noise 243, both 4 | 224 | -6.9 | about a wavelength or larger | 21.9 / 13.2 | 1.14 (1) | 14.1 x 3.7 | 3.5 (3.5) |
| multistatic_b | tug | ship, - | own noise 220, both 6 | 226 | -16.6 | about a wavelength or larger | 5.9 / 6.7 | 0.73 (0.7) | 29.9 x 4.5 | 2.5 (2.5) |
| multistatic_b | boat | ship, - | own noise 175, both 17 | 158 | +10.8 | smaller than about a wavelength | 4.3 / 8.3 | 0.45 (0.3) | 27.3 x 4.5 | 4.5 (3.7) |
| multistatic_b | uuv1 | uuv, 3 x 1 m | echo 202, both 6 | 134 | -8.3 | about a wavelength or larger | 5.9 / 6.7 | 0.71 (0) | 9.0 x 4.0 | 1.7 (1.7) |
| multistatic_b | uuv2 | uuv, 3 x 1 m | echo 215 | 215 | -2.6 | about a wavelength or larger | 3.3 / 4.2 | - | 4.6 x 2.3 | 1.1 (1.2) |
| multistatic_b | diver | diver, 1 x 0.5 m | echo 195 | 166 | -11.6 | about a wavelength or larger | 4.0 / 7.0 | - | 9.2 x 4.4 | 0.5 (0.6) |
| simple | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.09 (1) | 5.2 x 3.5 | 4.0 (4.0) |
| simple | boat | ship, - | own noise 66 | 66 | - | - | - | 0.28 (0.3) | 4.6 x 2.3 | 2.5 (2.5) |
| simple | uuv1 | uuv, 6 x 2 m | echo 37 | 37 | -4.3 | about a wavelength or larger | 4.1 / 4.1 | - | 5.2 x 3.5 | 1.6 (1.5) |
| simple | wreck | static, 4 x 4 m | echo 64 | 64 | -6.2 | about a wavelength or larger | 4.5 / 4.2 | - | 4.6 x 3.5 | 0.1 (0.0) |
| simple_seed1 | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.07 (1) | 5.2 x 3.5 | 4.0 (4.0) |
| simple_seed1 | boat | ship, - | own noise 66 | 66 | - | - | - | 0.28 (0.3) | 4.6 x 2.3 | 2.5 (2.5) |
| simple_seed1 | uuv1 | uuv, 6 x 2 m | echo 42 | 42 | -3.5 | about a wavelength or larger | 4.0 / 4.2 | - | 5.2 x 3.5 | 1.5 (1.5) |
| simple_seed1 | wreck | static, 4 x 4 m | echo 64 | 64 | -5.7 | about a wavelength or larger | 4.5 / 4.5 | - | 4.6 x 3.5 | 0.1 (0.0) |
| bistatic | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.32 (1) | 6.9 x 2.3 | 4.0 (4.0) |
| bistatic | uuv1 | uuv, 3 x 1 m | own noise 31, echo 45 | 59 | +1.8 | smaller than about a wavelength | 3.9 / 2.9 | 0.18 (0) | 8.7 x 2.9 | 1.8 (2.0) |
| bistatic_seed1 | ferry | ship, - | own noise 66 | 66 | - | - | - | 1.36 (1) | 6.9 x 2.3 | 4.0 (4.0) |
| bistatic_seed1 | uuv1 | uuv, 3 x 1 m | own noise 31, echo 45 | 59 | +3.1 | smaller than about a wavelength | 3.9 / 2.9 | 0.18 (0) | 9.8 x 2.9 | 2.0 (2.0) |

### Source level estimates against the truth (radiator tracks)

| Recording | Object | True [Pa] | Estimate [Pa] | Error [dB] |
|---|---|---|---|---|
| simple | ferry | 1 | 1.091 | +0.8 |
| simple | boat | 0.3 | 0.282 | -0.5 |
| bistatic | ferry | 1 | 1.322 | +2.4 |
| multistatic | ferry | 1 | 1.264 | +2.0 |
| multistatic | tug | 0.7 | 0.828 | +1.5 |
| multistatic | boat | 0.3 | 0.594 | +5.9 |
| harbor | ferry | 1 | 5.040 | +14.0 |
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
| harbor_seed1 | ferry | 1 | 5.002 | +14.0 |
| harbor_seed2 | ferry | 1 | 5.019 | +14.0 |
| harbor_seed3 | ferry | 1 | 4.923 | +13.8 |
| multistatic_b | ferry | 1 | 1.137 | +1.1 |
| multistatic_b | tug | 0.7 | 0.728 | +0.3 |
| multistatic_b | boat | 0.3 | 0.452 | +3.6 |

### Propagation error of the forward model (stationary pulse, arrival minus range over the chart speed)

| Recording | Band [Hz] | Chart speed [m/s] | Per hydrophone: error [ms] (direction, range) |
|---|---|---|---|
| simple | 100-380 | 1500.0 | +0.00 (207 deg, 224 m); +0.00 (126 deg, 344 m); +0.00 (312 deg, 135 m); +0.02 (72 deg, 294 m) |
| bistatic | 100-760 | 1493.7 | -0.21 (221 deg, 227 m); -0.03 (140 deg, 234 m); +0.05 (288 deg, 158 m); -0.10 (174 deg, 181 m) |
| multistatic | 100-760 | 1493.7 | -0.21 (221 deg, 227 m); -0.03 (140 deg, 234 m); +0.05 (288 deg, 158 m); -0.10 (174 deg, 181 m) |
| harbor | 12-78 | 1496.0 | -0.42 (297 deg, 1047 m); +0.10 (302 deg, 153 m); -0.42 (44 deg, 672 m); +0.21 (153 deg, 268 m) |
| multistatic_b | 100-760 | 1493.7 | -0.22 (222 deg, 205 m); -0.04 (135 deg, 231 m); +0.03 (296 deg, 152 m); -0.12 (168 deg, 167 m) |

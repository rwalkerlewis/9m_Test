### Real time, tuning recordings beside test recordings

Pd: fraction of frames, from the first frame the object could have been confirmed, with a confirmed track matched to it within 5 m (radiator tracks for what radiates, echo tracks for silent bodies); test: the three noise seeds (each value) and multistatic_b. False tracks: confirmed tracks matched to no object, per minute.

| Recording | Split | Compute per s of data | Latency (median) | Pd per object, old noise model | Pd per object, harbor_v2 | False tracks (per minute), old / harbor_v2 |
|---|---|---|---|---|---|---|
| simple | tuning | 1.00 | 2.3 s | ferry 100 %; boat 100 %; uuv1 64 %; wreck 100 % | ferry 100 %; boat 100 %; uuv1 27 %; wreck 100 % | 0 (0.0); harbor_v2 3 (9.0) |
| simple | test | 0.97 / 0.97 / 1.01 | 2.3 s / 2.3 s / 2.3 s | ferry 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 67 % / 67 % / 64 %; wreck 100 % / 100 % / 100 % | ferry 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 30 % / 30 % / 30 %; wreck 100 % / 100 % / 100 % | 2 (6.0) / 0 (0.0) / 1 (3.0); harbor_v2 2 (6.0) / 2 (6.0) / 2 (6.0) |
| bistatic | tuning | 1.04 | 2.2 s | ferry 100 %; uuv1 70 % | ferry 100 %; uuv1 55 % | 4 (12.0); harbor_v2 0 (0.0) |
| bistatic | test | 1.05 / 1.11 / 1.13 | 2.3 s / 2.3 s / 2.3 s | ferry 100 % / 100 % / 100 %; uuv1 70 % / 61 % / 70 % | ferry 100 % / 100 % / 100 %; uuv1 64 % / 82 % / 79 % | 5 (15.0) / 3 (9.0) / 5 (15.0); harbor_v2 1 (3.0) / 0 (0.0) / 1 (3.0) |
| multistatic | tuning | 1.57 | 2.9 s | ferry 100 %; tug 100 %; boat 100 %; uuv1 91 %; uuv2 94 %; moored 82 %; not tracked: diver, piling1, piling2, piling3, piling4, piling5, piling6 | ferry 100 %; tug 100 %; boat 100 %; uuv1 0 %; uuv2 88 %; moored 64 % | 3 (9.0); harbor_v2 2 (6.0) |
| multistatic | test | 1.62 / 1.54 / 1.58 | 3.0 s / 2.9 s / 2.9 s | ferry 100 % / 100 % / 100 %; tug 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 97 % / 91 % / 88 %; uuv2 94 % / 91 % / 94 %; piling3 0 % / 0 % / 45 %; piling4 21 % / 0 % / 0 %; moored 94 % / 67 % / 85 %; not tracked: diver, piling1, piling2, piling5, piling6 | ferry 100 % / 100 % / 100 %; tug 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 %; uuv1 64 % / 52 % / 15 %; uuv2 91 % / 91 % / 85 %; piling3 0 % / 0 % / 0 %; piling4 0 % / 0 % / 0 %; moored 79 % / 73 % / 39 % | 3 (9.0) / 3 (9.0) / 3 (9.0); harbor_v2 2 (6.0) / 1 (3.0) / 2 (6.0) |
| harbor | tuning | 0.54 | 1.1 s | ferry 83 %; tug 4 %; moored 0 %; uuv1 0 %; uuv2 0 %; diver 0 % | ferry 93 %; tug 4 %; moored 0 %; uuv1 0 %; uuv2 0 %; diver 0 % | 3 (1.5); harbor_v2 2 (1.0) |
| harbor | test | 0.54 / 0.54 / 0.54 | 1.1 s / 1.1 s / 1.1 s | ferry 74 % / 83 % / 67 %; tug 6 % / 7 % / 6 %; moored 0 % / 0 % / 0 %; uuv1 0 % / 0 % / 0 %; uuv2 0 % / 0 % / 0 %; diver 0 % / 0 % / 0 % | ferry 69 % / 81 % / 81 %; tug 6 % / 6 % / 6 %; moored 0 % / 0 % / 0 %; uuv1 0 % / 0 % / 0 %; uuv2 0 % / 0 % / 0 %; diver 0 % / 0 % / 0 % | 4 (2.0) / 3 (1.5) / 4 (2.0); harbor_v2 3 (1.5) / 2 (1.0) / 0 (0.0) |
| multistatic_b | test | 1.81 | 3.0 s | ferry 99 %; tug 100 %; boat 69 %; uuv1 62 %; uuv2 96 %; diver 66 %; not tracked: piling1, piling2, piling3, piling4, piling5, piling6, moored | ferry 99 %; tug 100 %; boat 55 %; uuv1 47 %; uuv2 72 %; diver 0 % | 8 (8.0); harbor_v2 7 (7.0) |

### Silent bodies by their echo, old noise model beside harbor_v2 (every recording with an echo chain)

| Recording | Silent body | Pd, old noise model | Pd, harbor_v2 |
|---|---|---|---|
| simple | uuv1 | 64 % | 27 % |
| simple | wreck | 100 % | 100 % |
| bistatic | uuv1 | 70 % | 55 % |
| multistatic | uuv1 | 91 % | 0 % |
| multistatic | uuv2 | 94 % | 88 % |
| multistatic | moored | 82 % | 64 % |
| simple_seed1 | uuv1 | 67 % | 30 % |
| simple_seed1 | wreck | 100 % | 100 % |
| simple_seed2 | uuv1 | 67 % | 30 % |
| simple_seed2 | wreck | 100 % | 100 % |
| simple_seed3 | uuv1 | 64 % | 30 % |
| simple_seed3 | wreck | 100 % | 100 % |
| bistatic_seed1 | uuv1 | 70 % | 64 % |
| bistatic_seed2 | uuv1 | 61 % | 82 % |
| bistatic_seed3 | uuv1 | 70 % | 79 % |
| multistatic_seed1 | uuv1 | 97 % | 64 % |
| multistatic_seed1 | uuv2 | 94 % | 91 % |
| multistatic_seed1 | piling4 | 21 % | 0 % |
| multistatic_seed1 | moored | 94 % | 79 % |
| multistatic_seed2 | uuv1 | 91 % | 52 % |
| multistatic_seed2 | uuv2 | 91 % | 91 % |
| multistatic_seed2 | moored | 67 % | 73 % |
| multistatic_seed3 | uuv1 | 88 % | 15 % |
| multistatic_seed3 | uuv2 | 94 % | 85 % |
| multistatic_seed3 | piling3 | 45 % | 0 % |
| multistatic_seed3 | moored | 85 % | 39 % |
| multistatic_b | uuv1 | 62 % | 47 % |
| multistatic_b | uuv2 | 96 % | 72 % |
| multistatic_b | diver | 66 % | 0 % |

### The tuning seeds 4 to 9 (tuning recordings, both noise models)

| Recording | Seeds | Pd per object, old noise model | Pd per object, harbor_v2 | False tracks per recording, old / harbor_v2 |
|---|---|---|---|---|
| simple | 4, 5, 6, 7, 8, 9 | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 67 % / 67 % / 64 % / 67 % / 67 % / 64 %; wreck 100 % / 100 % / 100 % / 100 % / 100 % / 100 % | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 27 % / 30 % / 30 % / 27 % / 30 % / 30 %; wreck 100 % / 100 % / 100 % / 100 % / 100 % / 100 % | 2 / 1 / 1 / 2 / 1 / 2; harbor_v2 2 / 2 / 2 / 2 / 2 / 2 |
| bistatic | 4, 5, 6, 7, 8, 9 | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 70 % / 70 % / 61 % / 24 % / 61 % / 58 % | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 85 % / 88 % / 82 % / 82 % / 61 % / 88 % | 1 / 5 / 2 / 5 / 3 / 5; harbor_v2 0 / 0 / 0 / 1 / 1 / 1 |
| multistatic | 4, 5, 6, 7, 8, 9 | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; tug 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 91 % / 82 % / 88 % / 94 % / 88 % / 91 %; uuv2 97 % / 97 % / 91 % / 94 % / 94 % / 94 %; piling5 0 % / 0 % / 0 % / 52 % / 0 % / 0 %; moored 73 % / 91 % / 85 % / 73 % / 82 % / 82 % | ferry 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; tug 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; uuv1 85 % / 52 % / 0 % / 42 % / 0 % / 73 %; uuv2 94 % / 82 % / 85 % / 85 % / 85 % / 94 %; piling5 0 % / 0 % / 0 % / 0 % / 0 % / 0 %; moored 70 % / 82 % / 64 % / 52 % / 79 % / 73 % | 3 / 3 / 3 / 2 / 3 / 3; harbor_v2 2 / 3 / 2 / 2 / 1 / 3 |
| multistatic_b | 4, 5, 6, 7, 8, 9 | ferry 95 % / 99 % / 99 % / 99 % / 99 % / 98 %; tug 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 68 % / 72 % / 68 % / 79 % / 73 % / 68 %; uuv1 64 % / 65 % / 64 % / 62 % / 62 % / 61 %; uuv2 84 % / 84 % / 83 % / 87 % / 98 % / 97 %; diver 23 % / 17 % / 40 % / 0 % / 32 % / 0 % | ferry 99 % / 99 % / 99 % / 99 % / 99 % / 99 %; tug 100 % / 100 % / 100 % / 100 % / 100 % / 100 %; boat 56 % / 56 % / 56 % / 56 % / 56 % / 56 %; uuv1 55 % / 62 % / 58 % / 57 % / 58 % / 58 %; uuv2 75 % / 79 % / 83 % / 68 % / 79 % / 80 %; diver 0 % / 0 % / 0 % / 0 % / 0 % / 0 % | 8 / 8 / 6 / 7 / 9 / 5; harbor_v2 5 / 6 / 9 / 5 / 6 / 4 |

### Criteria

| Criterion | Values | Result |
|---|---|---|
| every ship (ferry, tug, boat) tracked with Pd >= 0.8 on multistatic_b (radiator tracks) | {'ferry': 0.99, 'tug': 1.0, 'boat': 0.69} | **fail** |
| false tracks at most 2 per minute on every test recording | {'simple_seed1': 6.0, 'simple_seed2': 0.0, 'simple_seed3': 3.0, 'bistatic_seed1': 15.0, 'bistatic_seed2': 9.0, 'bistatic_seed3': 15.0, 'multistatic_seed1': 9.0, 'multistatic_seed2': 9.0, 'multistatic_seed3': 9.0, 'harbor_seed1': 2.0, 'harbor_seed2': 1.5, 'harbor_seed3': 2.0, 'multistatic_b': 8.0} | **fail** |
| the wreck of the simple harbor found by its echo with Pd >= 0.9 on seeds 1 to 3 (echo tracks) | {'simple_seed1': 1.0, 'simple_seed2': 1.0, 'simple_seed3': 1.0} | pass |
| on frames where the moored hull, a UUV and a piling are all measured, the size indicator puts the hull below the piling in at least 70 % of them (multistatic recordings, tuning and multistatic_b) | no frame had the hull, a UUV and a piling all measured: not testable | **not met: not testable** |

### Characteristics against the truth (objects detected in at least 10 frames; the multistatic recordings, and the tuning and first test recording of the simple and bistatic harbors, where bodies are tracked by their echo)

| Recording | Object | True type, size | Seen | Detected frames | Size ratio [dB] | Size indicator | Echo extent along / across [m] | Source level [Pa] (true) | Half max width [m] | Speed [m/s] (true) |
|---|---|---|---|---|---|---|---|---|---|---|
| multistatic | ferry | ship, - | own noise 33 | 33 | - | - | - | 1.26 (1) | 9.2 x 2.3 | 4.0 (4.0) |
| multistatic | tug | ship, - | own noise 33 | 33 | - | - | - | 0.83 (0.7) | 14.3 x 3.3 | 3.0 (3.0) |
| multistatic | boat | ship, - | own noise 33 | 33 | - | - | - | 0.59 (0.3) | 47.1 x 6.4 | 4.9 (5.0) |
| multistatic | uuv1 | uuv, 3 x 1 m | echo 30 | 30 | -4.9 | about a wavelength or larger | 3.6 / 4.2 | - | 6.9 x 2.3 | 1.9 (2.0) |
| multistatic | uuv2 | uuv, 3 x 1 m | echo 31 | 31 | -8.7 | about a wavelength or larger | 4.7 / 4.6 | - | 6.9 x 2.3 | 1.5 (1.5) |
| multistatic | moored | ship, 30 x 6 m | echo 27 | 27 | -2.9 | about a wavelength or larger | 7.0 / 5.8 | - | 11.6 x 2.3 | 0.3 (0.0) |
| multistatic_b | ferry | ship, - | own noise 123 | 112 | - | - | - | 1.14 (1) | 14.1 x 3.7 | 3.5 (3.5) |
| multistatic_b | tug | ship, - | own noise 108, both 5 | 113 | -16.3 | about a wavelength or larger | 5.1 / 4.6 | 0.73 (0.7) | 28.4 x 4.5 | 2.5 (2.5) |
| multistatic_b | boat | ship, - | own noise 86, echo 20, both 10 | 88 | +6.7 | smaller than about a wavelength | 5.4 / 8.0 | 0.45 (0.3) | 24.6 x 4.5 | 4.4 (3.7) |
| multistatic_b | uuv1 | uuv, 3 x 1 m | echo 103, both 5 | 70 | -7.5 | about a wavelength or larger | 5.6 / 5.4 | 0.77 (0) | 9.9 x 3.9 | 1.7 (1.7) |
| multistatic_b | uuv2 | uuv, 3 x 1 m | echo 109 | 109 | -3.2 | about a wavelength or larger | 3.4 / 4.0 | - | 4.6 x 2.3 | 1.1 (1.2) |
| multistatic_b | diver | diver, 1 x 0.5 m | echo 93 | 75 | -10.3 | about a wavelength or larger | 4.7 / 8.1 | - | 6.9 x 2.3 | 0.8 (0.6) |
| simple | ferry | ship, - | own noise 33 | 33 | - | - | - | 1.09 (1) | 5.2 x 3.5 | 4.0 (4.0) |
| simple | boat | ship, - | own noise 33 | 33 | - | - | - | 0.28 (0.3) | 4.6 x 2.3 | 2.5 (2.5) |
| simple | uuv1 | uuv, 6 x 2 m | echo 21 | 21 | -5.5 | about a wavelength or larger | 4.4 / 4.1 | - | 5.2 x 3.5 | 1.6 (1.5) |
| simple | wreck | static, 4 x 4 m | echo 33 | 33 | -5.8 | about a wavelength or larger | 4.5 / 3.9 | - | 4.6 x 3.5 | 0.2 (0.0) |
| simple_seed1 | ferry | ship, - | own noise 33 | 33 | - | - | - | 1.07 (1) | 5.2 x 3.5 | 4.0 (4.0) |
| simple_seed1 | boat | ship, - | own noise 33 | 33 | - | - | - | 0.28 (0.3) | 4.6 x 2.3 | 2.5 (2.5) |
| simple_seed1 | uuv1 | uuv, 6 x 2 m | echo 22 | 22 | -5.4 | about a wavelength or larger | 4.0 / 4.3 | - | 5.9 x 3.5 | 1.5 (1.5) |
| simple_seed1 | wreck | static, 4 x 4 m | echo 33 | 33 | -4.5 | about a wavelength or larger | 4.2 / 4.2 | - | 4.6 x 3.5 | 0.2 (0.0) |
| bistatic | ferry | ship, - | own noise 33 | 33 | - | - | - | 1.33 (1) | 6.9 x 2.3 | 4.0 (4.0) |
| bistatic | uuv1 | uuv, 3 x 1 m | own noise 15, echo 23 | 30 | +2.9 | smaller than about a wavelength | 3.8 / 2.9 | 0.18 (0) | 7.3 x 2.8 | 1.7 (2.0) |
| bistatic_seed1 | ferry | ship, - | own noise 33 | 33 | - | - | - | 1.36 (1) | 6.9 x 2.3 | 4.0 (4.0) |
| bistatic_seed1 | uuv1 | uuv, 3 x 1 m | own noise 15, echo 23 | 30 | +4.3 | smaller than about a wavelength | 3.9 / 3.1 | 0.18 (0) | 8.8 x 2.9 | 1.8 (2.0) |

### Source level estimates against the truth (radiator tracks)

| Recording | Object | True [Pa] | Estimate [Pa] | Error [dB] |
|---|---|---|---|---|
| simple | ferry | 1 | 1.089 | +0.7 |
| simple | boat | 0.3 | 0.282 | -0.5 |
| bistatic | ferry | 1 | 1.326 | +2.5 |
| multistatic | ferry | 1 | 1.264 | +2.0 |
| multistatic | tug | 0.7 | 0.828 | +1.5 |
| multistatic | boat | 0.3 | 0.594 | +5.9 |
| harbor | ferry | 1 | 5.040 | +14.0 |
| simple_seed1 | ferry | 1 | 1.071 | +0.6 |
| simple_seed1 | boat | 0.3 | 0.276 | -0.7 |
| simple_seed2 | ferry | 1 | 1.065 | +0.5 |
| simple_seed2 | boat | 0.3 | 0.279 | -0.6 |
| simple_seed3 | ferry | 1 | 1.059 | +0.5 |
| simple_seed3 | boat | 0.3 | 0.271 | -0.9 |
| bistatic_seed1 | ferry | 1 | 1.357 | +2.7 |
| bistatic_seed2 | ferry | 1 | 1.366 | +2.7 |
| bistatic_seed3 | ferry | 1 | 1.345 | +2.6 |
| multistatic_seed1 | ferry | 1 | 1.332 | +2.5 |
| multistatic_seed1 | tug | 0.7 | 0.844 | +1.6 |
| multistatic_seed1 | boat | 0.3 | 0.598 | +6.0 |
| multistatic_seed2 | ferry | 1 | 1.303 | +2.3 |
| multistatic_seed2 | tug | 0.7 | 0.854 | +1.7 |
| multistatic_seed2 | boat | 0.3 | 0.620 | +6.3 |
| multistatic_seed3 | ferry | 1 | 1.275 | +2.1 |
| multistatic_seed3 | tug | 0.7 | 0.846 | +1.6 |
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

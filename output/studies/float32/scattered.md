| Recording | Component | rms difference over rms | worst channel |
|---|---|---|---|
| simple | direct:boat | 1.62e-05 | 3.10e-05 |
| simple | direct:ferry | 1.64e-05 | 3.17e-05 |
| simple | scattered:boat>bodies | 3.84e-05 | 5.19e-05 |
| simple | scattered:ferry>bodies | 3.85e-05 | 5.41e-05 |
| bistatic | direct:ferry | 6.15e-05 | 2.30e-04 |
| bistatic | scattered:ferry>uuv1 | 3.28e-04 | 9.46e-03 |
| multistatic | direct:boat | 5.48e-05 | 6.86e-04 |
| multistatic | direct:ferry | 6.17e-05 | 2.33e-04 |
| multistatic | direct:tug | 5.77e-05 | 1.93e-03 |
| multistatic | scattered:boat>bodies | 1.56e-04 | 5.53e-04 |
| multistatic | scattered:ferry>bodies | 1.26e-04 | 3.48e-04 |
| multistatic | scattered:tug>bodies | 2.41e-04 | 1.02e-03 |
| multistatic_b | direct:boat | nan | nan |
| multistatic_b | direct:ferry | 5.96e-05 | 1.04e-04 |
| multistatic_b | direct:tug | 6.11e-05 | 4.09e-04 |
| multistatic_b | scattered:boat>bodies | nan | nan |
| multistatic_b | scattered:ferry>bodies | 1.54e-04 | 1.14e-03 |
| multistatic_b | scattered:tug>bodies | 1.70e-04 | 1.25e-03 |
| harbor | clean | 4.08e-06 | 6.69e-06 |

Criterion: every scattered component within 1e-03 of its rms: FAIL (largest inf).

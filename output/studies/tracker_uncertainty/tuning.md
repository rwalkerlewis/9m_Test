# Tracker tuning (tuning recordings only: noise seed 0 of simple, bistatic, multistatic, harbor)

Every run below was made before any test recording was run with the new tracker (`scratch` runs of
`harbor_stream.run_recording` with `--set` overrides). Values per run: false tracks per minute; ellipse coverage (fraction
of fixes of matched confirmed tracks whose true position lies inside the 95 % ellipse); Pd of the objects that moved.
Confirmation count and drop time were not changed.

| Run | Radiator `meas_scale` | Echo `meas_scale` | simple | bistatic | multistatic | harbor |
|---|---|---|---|---|---|---|
| s1 | 1.0 | 1.0 (echo maps not yet given to the echo tracker: floor only) | 3.0; 0.82 | 9.0; 0.62 | 9.0; 0.84 | 2.0; 0.81; ferry 0.20 |
| s2 | 2.0 | 2.0 (same) | 3.0; 0.84 | 6.0; 0.69 | 12.0; 0.98 | 2.0; 0.69; ferry 0.11 |
| m0.5 | 0.5 | 0.5 | 0.0; 0.91 | 6.0; 0.67 | 9.0; 0.93 | 3.0; 0.75; ferry 0.41 |
| m0.25 | 0.25 | 0.25 | 0.0; 0.84 | 6.0; 0.63 | 6.0; 0.93 | 3.0; 0.79; ferry 0.44 |
| e1 | 0.25 | 1.0 | 0.0; 0.91 | 6.0; 0.72 | 6.0; 0.99 | as m0.25 |
| e2 | 0.25 | 2.0 | 0.0; 0.99 | 9.0; 1.00 | 6.0; 1.00 | as m0.25 |
| h01 | 0.1 | 1.0 | 0.0; 0.91 | 6.0; 0.72 | 6.0; 0.81 | 4.5; 0.75; ferry 0.81 |
| e15 | 0.25 (harbor 0.1) | 1.5 | 0.0; 0.92 | 9.0; 0.89 | 6.0; 1.00 | 4.5; 0.75; ferry 0.81 |
| g1 (chosen) | 0.25 (harbor 0.1) | 1.5 | 0.0; 0.92 | 9.0; 0.89 | 6.0; 1.00 | 1.5; 0.80; ferry 0.83 |

g1 differs from e15 by a correction, not a parameter: the sidelobe test looked for the block's cells within 3 m, which
missed on the 5 m harbor's 5 m grid; it now looks within one and a half cells.

Why the measurement noise is smaller than the half maximum widths: the widths are the map's resolution, and a strong
peak is located far more finely than its width (on the 0.5 m recordings the ships' fixes are within 1 m while the
widths are 10 to 47 m along the line of sight); with the widths as they are, the filter trusts its prediction too much
and lags the 5 m harbor's ferry (Pd 0.20). The echo tracks keep a larger scale: an echo is offset from the body's
centre by a metre or two (the illuminated side), which the covariance has to hold.

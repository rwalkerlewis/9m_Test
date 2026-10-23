# Compute budget: acceptance rules, written before the reductions were measured

* A reduction is kept only if, on the four tuning recordings, every Pd, median error and false track count stays within
  the reproduce tolerance (fractions 3 percentage points, medians 0.5 m, false tracks exact) of the run without it.
* A held characteristic (`characteristics.every_frames`) is kept only if, in addition, every object's median source
  level and size ratio stay within 1 dB, and its echo extents within 1 m, of the per frame values on the tuning
  recordings (from `characteristics.json`).
* Criteria of the task (prompt `compute_budget`, section 3): compute per second of data at most 1.0 and median latency
  at most 2.0 s on every 0.5 m and 1 m recording on the Quadro RTX 4000; `scripts/reproduce.py --check` passes with no
  number outside its tolerance; on the 5 m harbor compute does not rise.

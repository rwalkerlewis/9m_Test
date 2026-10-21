Branch `layout_study` from `origin/main`. You own a new module `src/acoustic_sim/water/coverage.py`, `tests/harbor/test_coverage.py`, a new folder `examples/harbor/07_layout_study/`, new scenario files `examples/configs/layout/`, `output/studies/layout/` and `docs/layout_study.md`. You do not edit the chain, the processing configs, the existing scenario files, the existing recordings, `scripts/reproduce.py` or `docs/harbor.md`. This task builds nine small datasets (four FDTD runs each, about 24 s each on the GPU with the leapfrog scheme; if `forward_fidelity` has merged, use the k space scheme as its scenario files do and expect about ten times that); do not start them while another session is building.

Read `docs/harbor.md` (The system, Running it, The simple harbor, Numbers), `src/acoustic_sim/water/sensors.py`, `src/acoustic_sim/water/scenario.py` (`build_layout`), `src/acoustic_sim/water/stream.py` (`HarborPicture`, the public entry point), `src/acoustic_sim/water/propagation.py`, `examples/harbor/05_detection/harbor_stream.py` (how it scores; use the same scoring functions, do not copy them), `examples/configs/processing/simple.json`, `examples/configs/harbor_simple.scenario.json` and `examples/harbor/01_datasets/harbor_multistatic_dataset.py` before changing anything.

## The question

A strain sensing cable responds to a wave as the square of the cosine of the angle between the wave and the fibre, so a straight leg is deaf broadside; the hydrophones are omnidirectional but are four channels among about two hundred in the passive scan. Stage 2's half maximum widths show it: 4.6 by 2.3 m on the simple harbor's loop against 14 to 47 m along the line of sight on the strain cable of the 0.5 m recordings, which sees the ships from one side. Where the cable is laid, how many bends it has and how many hydrophones fill its blind directions decide what the system can see before any processing runs. Nobody has measured this.

## 1. Coverage without running anything

`water/coverage.py` (processing side: it needs only the chart, the sensor layout and a propagator). For every cell of the scan grid:

- **Strain coverage**: the fraction of cable channels whose response to a wave from that cell (the square of the cosine of the angle between the cell to channel direction and the fibre tangent) exceeds 0.5, and the same weighted by cylindrical spreading from the cell.
- **Aperture**: the angular spread, in degrees, of the sensors that see the cell (response above 0.5; hydrophones always).
- **Shadowing**: the fraction of sensors whose straight path from the cell does not cross land.
- **Predicted resolution**: the half maximum widths, along and across the dominant direction, of the layout's free field matched field response to a source at the cell at the band centre (the replica against itself).

Plot the four as maps over the chart for the layout of every scenario on `main`, with the objects' tracks drawn (the truth is allowed here; this is a design tool).

## 2. Layouts to compare

In the simple harbor (500 by 400 m basin, 1 m grid, 100 to 380 Hz; a ferry, a boat, a silent UUV, a wreck), keep the objects, the noise and the levels and vary only the sensors. Strain mode for every cable variant (the simple harbor's cable on `main` is pressure mode, which hides the directivity); same total cable length within 10 percent, 4 m channels, 2 m gauge.

| Variant | Cable | Hydrophones |
|---|---|---|
| A | one straight leg along the north quay | 0 |
| B | the same leg | 4 at the basin corners |
| C | an L: north quay and east side | 0 |
| D | a U: north, east, south | 0 |
| E | the current loop | 0 |
| F | the current loop | 4 at the corners (the `main` layout in strain mode) |
| G | the current loop | 8: corners and mid sides |
| H | the same length as a zigzag of 45 degree segments across the north half | 0 |
| P | the current loop in pressure mode | 4 (the `main` layout as it is: the reference of an omnidirectional cable) |

One scenario file per variant under `examples/configs/layout/`, differing from `harbor_simple.scenario.json` only in its sensor block; one dataset per variant under `output/studies/layout/<variant>/`, seeds 0 to 2 mixed from the cached gathers.

## 3. Run the system unchanged and score

`examples/harbor/07_layout_study/run_layout_study.py --cuda`: for every variant and seed, `HarborPicture` with `examples/configs/processing/simple.json` unchanged, scored as `main` scores (Pd per object from confirmed tracks within 5 m, median error, false tracks per minute, the stage 2 definitions). Record per object the four coverage quantities at its median position. If a variant fails because a setting assumes the loop (a scan edge, a gate), report it as a finding; no parameter changes per variant.

## 4. Report

`output/studies/layout/summary.json` and `docs/layout_study.md`: the table (variant, cable length, channels, hydrophones, per object Pd and median error with the spread over seeds, false tracks per minute, compute per second of data); the coverage maps, one figure per quantity per variant; the rank correlation between each coverage quantity at the objects' positions and Pd over all variants and objects, stating which quantity predicts detection and how well, including if none does; and at most five sentences a buyer should take from it: what a straight cable misses, what bends buy, what hydrophones buy, whether the current loop is a sensible default, and what a 2D pressure release simulation with one sound speed cannot answer.

## Rules

The repository rules in `AGENTS.md` apply, the figure rules included (the coverage maps are one figure per quantity per variant, not one grid). Tests for `coverage.py` on layouts with known answers (a straight cable along x: coverage 1 on axis and 0 broadside; a wall shadowing half the sensors; the resolution of a short straight array against the textbook beam width). Stop after the report.

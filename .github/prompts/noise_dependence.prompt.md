---
mode: agent
description: Branch noise_dependence from origin/main. You own scripts/reproduce.py (a second noise model run), src/acoustic_sim/wate
---
Branch `noise_dependence` from `origin/main`. You own `scripts/reproduce.py` (a second noise model run), `src/acoustic_sim/water/noise.py` (parameter sweeps only, no new components), `output/studies/noise_model/`, `output/reproduce/`, `docs/noise.md`, `docs/field_recording.md` (new) and the noise paragraphs of `docs/harbor.md`. No processing code changes. No FDTD runs (every mix comes from the cached gathers).

Read `docs/noise.md`, `output/studies/noise_model/`, `output/reports/run_all.md` (task 4), `src/acoustic_sim/water/noise.py` and `docs/harbor.md` (status, Numbers) before changing anything.

## What the noise study found

With the `harbor_v2` components at their assumed levels, the ships did not move, but the silent bodies did: uuv1 on the multistatic seeds fell from 62 to 83 % to 0 to 6 %, the moored hull from about 80 % to 34 to 72 %, the bistatic UUV moved both ways. Several of those levels are marked assumed because no measured spectrum was found: the shrimp rate per metre and band energy, the machinery, the fibre drift, the interrogator floor and shape. The silent body numbers at the top of `docs/harbor.md` are therefore numbers about one noise model, and the page does not say so.

## 1. Both noise models in the headline numbers

`scripts/reproduce.py` runs every recording twice, with the old noise model (the recordings as they are) and with `harbor_v2` at its documented levels (mixed from the same gathers, same seeds, `output/recordings/<recording>/noise2/`), and `summary.json` carries both. The real time table in `docs/harbor.md` gets a second Pd column per object, old noise and `harbor_v2` side by side, and every silent body number in the status paragraph is given as its range over the two noise models. The criteria are evaluated on the old model (as fixed) and reported on both.

## 2. Which assumed number matters

A sweep on the multistatic tuning recording and the bistatic tuning recording, one assumed parameter at a time over a stated range (shrimp band energy and rate: 10 dB below and above the assumed value; machinery level: off, assumed, 10 dB above; fibre drift and interrogator floor: 10 dB below and above; fading fraction: half and double), everything else at the documented value, the chain unchanged. For each parameter: the Pd of every silent body and the false tracks per minute against the parameter, one figure per object. Report which parameters move a silent body by more than 10 points over their range and which move nothing; those that move the numbers are the ones a field recording has to pin.

## 3. What one field recording must contain

`docs/field_recording.md`: the shortest recording that would replace the assumed numbers with measured ones, written so it can be handed to whoever can make it: interrogator and fibre type with the gauge length and channel spacing; a co located hydrophone at a stated position on the cable; duration and times of day (a quiet night hour, a working hour with quay machinery, a period with a known vessel movement with its AIS log); sample rate and band; the metadata needed to build an `Observation` (`docs/architecture.md`, Real data entry point); the water temperature and depth at the site (for the shrimp question); and the analysis that would be done on it in the order it answers the parameters of section 2. State plainly which of the model's numbers that recording cannot settle.

## Rules

The repository rules in `AGENTS.md` apply, the figure rules included. Tests: the second noise run is reproducible from the same seed; the sweep script writes its criterion free `results.json` with every parameter value used. Stop after section 3. Report the two column table for the silent bodies, the parameters that matter, and the field recording's one paragraph summary.

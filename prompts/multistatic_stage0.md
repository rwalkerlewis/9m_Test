Work in the 9m_Test repo on branch `multistatic` (`git fetch origin && git checkout multistatic`). Use the GPU: pass `--cuda` to every FDTD script, with CPU fallback.

Read `docs/harbor.md`, `src/acoustic_sim/harbor/bistatic.py`, `src/acoustic_sim/harbor/scatterer.py`, `src/acoustic_sim/harbor/dataset.py`, `examples/harbor_bistatic_dataset.py`, `examples/harbor_bistatic_demo.py`, `examples/harbor_bistatic.scenario.json` and `examples/harbor_cancellation_depth.py` before changing anything.

## Already on the branch (not yet run)

1. `bistatic.py`: the ECA block of `run_bistatic` is factored out as `eca_stage(traces, layout, fs, env, sp, ship, bc, oracle_signal, components)`, which returns the cleaned arrays and the reference waveform. `run_bistatic` calls it. Nothing else changed. There is no test covering `run_bistatic`; add one (a small analytic or coarse case that checks `eca_stage` gives the same output as the old inline code and that a truth component passed through it is attenuated).
2. `examples/harbor_cancellation_depth.py`: stage 0a below. Written but never executed. Debug it first with `--coarse --time 2 --workers 1 --out /tmp/cancel_test --cache /tmp/cancel_cache` before the full run.
3. `tests/harbor/test_dataset.py::test_scenario_config_roundtrip` updated to the current default vessel list.

## Goal

The long term goal is to locate every object in the harbor (radiating vessels, silent moving bodies, static structures) using only passive sensing, with ship noise as the illuminator. This task covers the first stage only: find out whether the bistatic chain works without oracle inputs, and build a multi illuminator, multi body scenario and dataset for later stages. Do not start multistatic fusion, CLEAN extraction or tracking.

The current bistatic demo (`output/harbor_bistatic/demo/demo_summary.json`, median error 1.7 m) was run on the coarse dataset and uses three oracle inputs: the direct path removed with the truth component, the true ferry waveform, and the exact ferry track. Every result in this task must state which oracle inputs it uses.

Repo convention: every experiment has a pass/fail criterion written into the script docstring and `results.json` before it runs. Never change a criterion after seeing the result.

## Stage 0a: cancellation depth against range

Run `examples/harbor_cancellation_depth.py --cuda` (fine grid, 10 s simulated). It places the ferry signature alone at nearest channel ranges of about 200, 100 and 30 m (the 400 m by 300 m crop with its 30 m sponge allows no more than about 200 m), stationary and then moving at 4 m/s parallel to the first cable leg. It reports ECA depth per channel in the 100 Hz to 0.95 f_max band for three references: `data` (blind), `oracle` (true waveform, AIS refined position), `oracle_exact` (true waveform and exact position).

Criterion: median depth over live cable channels with the `data` reference at least 20 dB. Interpretation fixed in advance: low at every range for the stationary source points at the reference or removal method; low only at 30 m points at near field channels; stationary passes but moving fails points at the moving source representation or the ECA block length.

Check that the picked positions in `positions.png` make sense (open water, ranges as intended) before trusting the numbers. If near field channels are the cause, add a `gate_radius` option to `BistaticConfig` that zeroes channels closer than that distance to the refined illuminator per ECA block in both the reference beam and the map, and rerun once with `--gate 50`.

Outputs in `output/harbor_cancel/`: `depth_vs_range.png`, `depth_per_channel.png`, `positions.png`, `results.json`, `run.log`.

## Stage 0b: bistatic demo without oracles

Build the fine bistatic dataset (`python examples/harbor_bistatic_dataset.py --cuda`) and also the coarse one (`--coarse`) so the original demo numbers can be reproduced. Write `examples/harbor_bistatic_ablation.py` that runs `run_bistatic` in four configurations, changing one assumption at a time:

1. Current oracle setup (truth direct removal, oracle waveform, exact track). On the coarse dataset this must reproduce `demo_summary.json`.
2. Oracle waveform and exact track, direct path in the traces and removed by ECA.
3. Data reference waveform, exact track, ECA.
4. Data reference, AIS track with refinement, ECA (fully blind).

For each, on both datasets, report median and p90 error, fraction within 5 m, median echo contrast in dB, the 1.5 s integrated result, and the ECA depth of the direct component and loss of the echo component (truth components go in `Truth.components` only for these diagnostics, never into the processing).

Criterion for configuration 4 on the fine dataset, fixed now: median echo contrast at least 10 dB and at least half the frames within 5 m. If it fails, identify which step between configurations 1 and 4 loses the most contrast.

Outputs in `output/harbor_bistatic/ablation/`: a bar figure per metric across configurations, `maps_by_config.png`, `results.json`, `run.log`.

## Stage 0c: multistatic scenario and dataset

Create `examples/harbor_multistatic.scenario.json` on the same 0.5 m crop and sensor layout as the bistatic scenario:

1. Illuminators: the ferry (AIS), a tug (AIS, a track crossing the ferry's geometry), and a small radiating boat with no AIS (`kind: "ship"`, `ais: false`).
2. Silent moving bodies (`level_pa: 0`): two UUVs (1 m x 3 m, different headings, 1 to 2 m/s) and a diver (about 0.5 m x 1 m, slow, with `wander_sigma`).
3. Static bodies (single waypoint, `speed: 0`): a row of six pilings (`body_shape: "ellipse"`, 1 m diameter so each spans two cells; 0.5 m would be one cell) a few metres apart, and a moored hull of about 6 m x 30 m near a quay.

Keep every track at least 15 m from land (`build_track` checks this) and outside the sponge.

The current `render_vessel_gathers` runs each body on its own, so multiple scattering is lost. Add an option (for example `combined_bodies=True`) that runs each radiating vessel once with all bodies present and once with none, storing `direct:<id>` and `scattered:<id>>bodies`. Add a test for it on a tiny grid. Then write `examples/harbor_multistatic_dataset.py` (based on `harbor_bistatic_dataset.py`) that builds the dataset with `keep_components=True`, saves truth for every object per frame (position, velocity, body kind, radiates, reports AIS), and supports `--coarse` and `--cuda`.

Outputs in `output/harbor_multistatic/`: `map.png` (layout, every track and static body labelled by type), `components.png` (per illuminator direct and scattered levels on the cable), `levels.json`, `scenario_used.json`, `run.log`. Then run `run_bistatic` with the ferry only, oracle setup, and save `ferry_only_map.png` with truth markers for every body to confirm their echoes appear.

## Rules

1. Figures, JSON results and logs under `output/` are committed. Datasets (`*.npz`, `*.npy`) and FDTD caches stay gitignored.
2. Keep `pytest tests/harbor` passing.
3. Update `docs/harbor.md`: the stage 0a and 0b measured numbers with pass/fail, the new scenario, and a row for each new script in the examples table. State plainly what did not work.
4. Commit in small steps on `multistatic` and open a pull request against `main` with a summary table of the 0a and 0b results.
5. Stop after stage 0c. Report the numbers, the pass/fail of each criterion, and what you think limits configuration 4.

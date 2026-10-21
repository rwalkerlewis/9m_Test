# Examples

Scripts that run the package end to end, in the order you would use them. Every script is run from the repository root
(`python examples/<folder>/<script>.py --help`; in the dev container `./dev.sh run python ...`) and writes to `output/`.
Configurations live in `configs/`. The harbor folders are numbered in workflow order: build the datasets (1), then any of
2 to 5; the walkthrough (6) needs the datasets of 1 and the real-time system of 5. Details and results:
[docs/harbor.md](../docs/harbor.md) (harbor), [docs/usage.md](../docs/usage.md) (air).

## Configurations (`configs/`)

Scenario, domain and pipeline JSON files read by the scripts below (`--config`).

| File | Contents |
|---|---|
| `domain.example.json` | Air velocity model: gradient with a circle and a rectangle anomaly (`acoustic-sim --model-file`) |
| `harbor.scenario.json` | The default 5 m harbor: 2400 x 1800 m, ferry, tug, moored ship, UUVs, diver (`harbor_m2_dataset.py` and the M1 to M4 scripts) |
| `harbor_bistatic.scenario.json` | Cropped 0.5 m harbor: a ferry lights a silent UUV, strain DAS (`harbor_bistatic_dataset.py`) |
| `harbor_multistatic.scenario.json` | 0.5 m harbor with three ships, two UUVs, a diver, six pilings and a moored hull (`harbor_multistatic_dataset.py`) |
| `harbor_simple.scenario.json` | The simple harbor: easy wavefields, a ferry and a boat lighting a UUV and a wreck (`harbor_multistatic_dataset.py --config`) |
| `layered.example.json` | Air velocity model: layered, with an anomaly |
| `pipeline.config.json` | Defaults of the air detection and engagement pipeline (`air/02_pipeline/run_pipeline.py --config`) |
| `valley.example.json` | Air velocity model: valley with randomised hills |

## Harbor 1: build the datasets (`harbor/01_datasets/`)

FDTD runs of the harbor scenarios, mixed into the recordings every later step reads (`output/harbor*/`). Run these first; their gathers are cached.

| File | What it does |
|---|---|
| `harbor_bistatic_dataset.py` | Bistatic harbor dataset: an AIS ship illuminates a silent UUV body in a cropped, fine grid harbor. |
| `harbor_build_classifier_dataset.py` | Harbor Sub-Task 4: build labelled mel-spectrogram dataset for the CNN classifier. |
| `harbor_m2_dataset.py` | Harbor M2: build the labelled dataset for the default scenario and plot it. |
| `harbor_multistatic_dataset.py` | Stage 0c: multi illuminator, multi body harbor dataset for the later multistatic stages. |
| `harbor_survey_grid.py` | Pre-compute per-cell FDTD transfer functions for the harbor grid survey. |

## Harbor 2: the forward model (`harbor/02_forward_model/`)

What the simulation does: one source, scattering bodies, wavefield and gather videos.

| File | What it does |
|---|---|
| `harbor_domain_video.py` | Videos of the wavefield and of the DAS gather in the harbor domain of ``output/harbor/dataset.npz``. |
| `harbor_m1_demo.py` | Harbor M0/M1 demo: environment map, one FDTD source run, DAS gather and wavefield snapshots. |
| `harbor_m1_scatterer_compare.py` | Harbor M1 variant: compare hard-body vs. soft-body scattering from a static reflector. |
| `harbor_m1_uuv.py` | Harbor M1 variant: one static source, a single pulse, and a static UUV in the water. |
| `harbor_m1_uuv_video.py` | Video of the single-pulse gather of ``examples/harbor/02_forward_model/harbor_m1_uuv.py`` arriving on the cable, with the reflections off the stationary UUV highlighted as they appear. |

## Harbor 3: passive detection of radiating objects (`harbor/03_passive/`)

Matched field processing (analytic, then on the FDTD data), the M3 detector, the M4 detect-locate-classify pipeline and its classifiers.

| File | What it does |
|---|---|
| `harbor_m3_detect.py` | Harbor M3: detection and localisation of the hidden UUV. |
| `harbor_m4_dlc.py` | Harbor M4: end-to-end detect–locate–classify pipeline. |
| `harbor_mfp_demo.py` | Simplified harbor matched field processing demo (analytic forward model). |
| `harbor_mfp_domain.py` | Analytic matched field test in the harbor domain of a saved FDTD dataset. |
| `harbor_train_anomaly.py` | Harbor Sub-Task 5: train the CVAE anomaly detector on ship-only mel-spectrograms. |
| `harbor_train_classifier.py` | Harbor Sub-Task 4: train the CNN acoustic classifier on the classifier dataset. |

## Harbor 4: silent bodies by their echo (`harbor/04_bistatic/`)

A ship as the illuminator: link budget, the bistatic demo and its ablation, direct path cancellation depth, presentation figures.

| File | What it does |
|---|---|
| `harbor_bistatic_ablation.py` | Stage 0b: the bistatic demo with its truth inputs removed one at a time. |
| `harbor_bistatic_budget.py` | Harbor bistatic link budget: a silent UUV lit by an AIS ship. |
| `harbor_bistatic_demo.py` | Bistatic demo: a silent UUV located from the ferry's echo on the DAS cable (pitch figures). |
| `harbor_cancellation_depth.py` | Stage 0a: direct path cancellation depth of ECA against range, stationary and moving source. |
| `harbor_presentation.py` | Presentation demo: one ship, one silent stopped UUV, a DAS cable, in the harbor of ``output/harbor/dataset.npz``. |
| `harbor_stack_movie.py` | Ten second movie: the echoes of a silent UUV are stacked pulse by pulse until it stands out unambiguously. |

## Harbor 5: detection and scoring (`harbor/05_detection/`)

The chains on the recordings: batch detection with a check against the truth, the real-time system, and the stage 1 scoreboard.

| File | What it does |
|---|---|
| `harbor_detect.py` | Detect objects in a harbor recording and check every detection against where the objects really are. |
| `harbor_picture_stage1.py` | Harbor picture, stage 1: the scoreboard, and the chains of ``main`` measured on it. |
| `harbor_stream.py` | The real-time harbor picture: locate every radiator from the traces, track it, map the echoes, check. |

## Harbor 6: walkthrough (`harbor/06_walkthrough/`)

One recording through the real-time chain, one figure per step (`run_all.sh DATASET --cuda`); start here to see what the system does.

| File | What it does |
|---|---|
| `01_recording.py` | Step 1, the recording: what the system gets. |
| `02_passive_map.py` | Step 2, passive location: where does sound come from? |
| `03_tracks.py` | Step 3, tracks: which detections belong together? |
| `04_ais_crosscheck.py` | Step 4, AIS afterwards: which tracks are reporting ships? |
| `05_direct_path_removal.py` | Step 5, removing the ships' own sound: making the echoes visible. |
| `06_echo_map.py` | Step 6, the echo map: where is ship noise being reflected? |
| `07_harbor_picture.py` | Step 7, the harbor picture: everything located, and how well. |
| `08_geometry.py` | Step 8, the geometry behind the picture: where the chain thinks things are going, and which paths the sound takes. |
| `run_all.sh` | All walkthrough figures for one recording: ./examples/harbor/06_walkthrough/run_all.sh multistatic --cuda |

## Air 1: simulation (`air/01_simulation/`)

FDTD runs of the airborne (drone) scenarios and trace generation.

| File | What it does |
|---|---|
| `extract_array_traces.py` | Extract receiver traces from a saved FDTD field plane. |
| `generate_traces.py` | Regenerate synthetic microphone traces for all simulation scenarios. |
| `run_all_examples.py` | Run all 18 FDTD example combinations. |
| `run_fdtd.py` | Run a single FDTD acoustic simulation. |
| `run_fdtd_3d.py` | Run a single 3-D FDTD acoustic simulation. |
| `run_fdtd_erratic.py` | Run a 3-D FDTD simulation with an erratic quadcopter source. |
| `run_valley.sh` | Valley domain — moving propeller source, circular receiver array |
| `run_valley_3d.sh` | 3-D Valley domain — moving propeller source, circular receiver array |
| `run_wind_circular.sh` | Wind domain — moving propeller source, circular receiver array |

## Air 2: detection and engagement pipeline (`air/02_pipeline/`)

The unified pipeline on the simulated recordings, and the baseline against ML comparison.

| File | What it does |
|---|---|
| `run_comparison.py` | Automated comparison of baseline vs ML-augmented pipelines. |
| `run_pipeline.py` | Unified acoustic detection and engagement pipeline. |
| `run_pipeline_erratic.sh` | Pipeline run — erratic_quadcopter (3-D isotropic domain, erratic trajectory) |
| `run_pipeline_isotropic.sh` | Pipeline run — isotropic_2D (2-D wind domain) |
| `run_pipeline_valley.sh` | Pipeline run — valley_test (2-D hills_vegetation domain) |
| `run_pipeline_valley_3d.sh` | Pipeline run — valley_3d_test (3-D hills_vegetation domain) |

## Air 3: machine learning (`air/03_ml/`)

Classifiers and the Fourier neural operator (need PyTorch).

| File | What it does |
|---|---|
| `run_fno.py` | End-to-end FNO surrogate workflow: generate data, train, infer. |
| `run_ml_demo.py` | Train, evaluate, and visualise the acoustic source classifier. |
| `train_all_ml.py` | Train all ML classifiers and save checkpoints. |

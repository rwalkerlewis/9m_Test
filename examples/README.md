# Examples

Scripts that run the package end to end, in the order you would use them. Every script is run from the repository root
(`python examples/<folder>/<script>.py --help`; in the dev container `./dev.sh run python ...`) and writes to `output/`.
Configurations live in `configs/`. The harbor folders are numbered in workflow order: build the recordings (1), look at
the forward model (2), run the system (5), walk through it (6). The harbor system itself is `acoustic_sim.water.stream`;
`90_history/` keeps the scripts of earlier stages. Details and results: [docs/harbor.md](../docs/harbor.md) (harbor),
[docs/harbor_history.md](../docs/harbor_history.md) (earlier harbor work), [docs/usage.md](../docs/usage.md) (air).

## Configurations (`configs/`)

Scenario, domain, pipeline and processing JSON files read by the scripts below.

| File | Contents |
|---|---|
| `processing/<recording>.json` | Every processing parameter of the harbor system for one recording (`simple`, `bistatic`, `multistatic`, `harbor`): band, scan edge, chunk, `ProcessingConfig`, `StreamConfig`, the radiator `TrackerConfig`, the echo chain (`EchoStreamConfig`, `BistaticConfig`, the echo `TrackerConfig`), the characteristics, and the scoring radii (read only by the scoring) |
| `domain.example.json` | Air velocity model: gradient with a circle and a rectangle anomaly (`acoustic-sim --model-file`) |
| `harbor.scenario.json` | The default 5 m harbor: 2400 x 1800 m, ferry, tug, moored ship, UUVs, diver (`harbor_m2_dataset.py`) |
| `harbor_bistatic.scenario.json` | Cropped 0.5 m harbor: a ferry lights a silent UUV, strain DAS (`harbor_bistatic_dataset.py`) |
| `harbor_multistatic.scenario.json` | 0.5 m harbor with three ships, two UUVs, a diver, six pilings and a moored hull (`harbor_multistatic_dataset.py`) |
| `harbor_multistatic_b.scenario.json` | The multistatic harbor with every track, the pilings and the hull moved, same sensors and levels: a test geometry no parameter has seen |
| `harbor_simple.scenario.json` | The simple harbor: easy wavefields, a ferry and a boat lighting a UUV and a wreck (`harbor_multistatic_dataset.py --config`) |
| `layered.example.json` | Air velocity model: layered, with an anomaly |
| `pipeline.config.json` | Defaults of the air detection and engagement pipeline (`air/02_pipeline/run_pipeline.py --config`) |
| `valley.example.json` | Air velocity model: valley with randomised hills |

## Harbor 1: build the recordings (`harbor/01_datasets/`)

FDTD runs of the harbor scenarios, mixed into the recordings every later step reads (`output/recordings/<recording>/`). Run these first;
their gathers are cached. `scripts/reproduce.py` builds every recording it needs, the test recordings included.

| File | What it does |
|---|---|
| `harbor_bistatic_dataset.py` | Bistatic harbor dataset: an AIS ship illuminates a silent UUV body in a cropped, fine grid harbor. |
| `harbor_m2_dataset.py` | Harbor M2: build the labelled dataset for the default scenario and plot it. |
| `harbor_multistatic_dataset.py` | Multi illuminator, multi body harbor dataset (the multistatic, simple and multistatic_b recordings). |
| `harbor_propagation_error.py` | The forward model's own propagation error on a recording (arrival against range over the chart speed per hydrophone), written into its truth file. |

## Harbor 2: the forward model (`harbor/02_forward_model/`)

What the simulation does: one source, scattering bodies, wavefield and gather videos.

| File | What it does |
|---|---|
| `harbor_domain_video.py` | Videos of the wavefield and of the DAS gather in the harbor domain of ``output/recordings/harbor/dataset.npz``. |
| `harbor_m1_demo.py` | Harbor M0/M1 demo: environment map, one FDTD source run, DAS gather and wavefield snapshots. |
| `harbor_m1_scatterer_compare.py` | Harbor M1 variant: compare hard-body vs. soft-body scattering from a static reflector. |
| `harbor_m1_uuv.py` | Harbor M1 variant: one static source, a single pulse, and a static UUV in the water. |
| `harbor_m1_uuv_video.py` | Video of the single-pulse gather of ``examples/harbor/02_forward_model/harbor_m1_uuv.py`` arriving on the cable, with the reflections off the stationary UUV highlighted as they appear. |

## Harbor 5: the system on the recordings (`harbor/05_detection/`)

| File | What it does |
|---|---|
| `harbor_stream.py` | The real-time harbor picture (`acoustic_sim.water.stream`) on one recording, scored against the truth: picture.jsonl, summary, characteristics, check figures, the operator view video. |
| `harbor_detect.py` | The same chain over the whole record at once (a batch run is the stream with one chunk). |
| `harbor_level_calibration.py` | The one calibration of the source level estimate, on the simple harbor's tuning recording. |
| `harbor_cancellation_depth.py` | The direct path removal depth of every reference estimator (`ref_mode`) on the stage 0a cases (a ship alone at three ranges, stationary and moving). |
| `harbor_direct_path_depth.py` | The removal depth of every ship's direct path on the recordings, per reference estimator (a diagnostic from the truth components). |
| `harbor_direct_path_criteria.py` | The criteria of the direct path change on the chain, from the summaries before and after. |

## Harbor 6: walkthrough (`harbor/06_walkthrough/`)

One recording through the real-time chain, one figure per plot (`run_all.sh DATASET`); start here to see what the system does.

| File | What it does |
|---|---|
| `01_recording.py` | Step 1, the recording: what the system gets. |
| `02_passive_map.py` | Step 2, passive location: where does sound come from? |
| `03_tracks.py` | Step 3, tracks: which detections belong together? |
| `04_ais_crosscheck.py` | Step 4, AIS afterwards: which tracks are reporting ships? |
| `05_direct_path_removal.py` | Step 5, removing the ships' own sound: making the echoes visible. |
| `06_echo_map.py` | Step 6, the echo map: where is ship noise being reflected? The echo tracks over it. |
| `07_harbor_picture.py` | Step 7, the harbor picture: everything located as tracks, and how well. |
| `08_geometry.py` | Step 8, the geometry behind the picture: where the chain thinks things are going, and which paths the sound takes. |
| `run_all.sh` | All walkthrough figures for one recording: ./examples/harbor/06_walkthrough/run_all.sh multistatic |

## History (`harbor/90_history/`)

Scripts of earlier stages, not part of the system. They stay runnable because committed figures came from them;
[docs/harbor_history.md](../docs/harbor_history.md) has their results.

| File | What it was |
|---|---|
| `harbor_m3_detect.py` | M3: CFAR detection of the UUV without AIS suppression (the loud ships hide it). |
| `harbor_m4_dlc.py` | M4: detect, locate, classify with AIS suppression and the CNN / CVAE classifiers. |
| `harbor_mfp_demo.py` | The whole chain on an analytic forward model (exact replicas), with a Pd sweep. |
| `harbor_mfp_domain.py` | The analytic chain in the domain of the 5 m harbor dataset. |
| `harbor_train_classifier.py` | Training of the harbor CNN classifier. |
| `harbor_train_anomaly.py` | Training of the harbor CVAE anomaly detector. |
| `harbor_build_classifier_dataset.py` | The labelled mel spectrogram dataset of the CNN classifier. |
| `harbor_survey_grid.py` | Per cell FDTD transfer functions for the empirical localiser. |
| `harbor_bistatic_demo.py` | The bistatic demo given the truth (direct path removed exactly, true waveform, exact track). |
| `harbor_bistatic_ablation.py` | Stage 0b: the bistatic demo with its truth inputs removed one at a time. |
| `harbor_bistatic_budget.py` | The analytic bistatic link budget with idealised cylinders. |
| `harbor_presentation.py` | Presentation demo: one ship, one silent stopped UUV, in the 5 m harbor. |
| `harbor_stack_movie.py` | Movie of echoes stacked pulse by pulse. |
| `harbor_picture_stage1.py` | Stage 1: the threshold scoreboard on the batch and streaming chains. |

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

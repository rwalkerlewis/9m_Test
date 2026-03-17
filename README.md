# Passive Acoustic Drone Detection and Engagement System

Response to the Audio Processing Home Task.

Full technical documentation: [docs/](docs/index.md)

## What I Chose To Work On

I chose all four directions and integrated them into a unified system.

The prompt offered signal enhancement, realistic simulation, advanced DOA, and array geometry as separate options. My first observation was that these are not independent problems. You cannot evaluate a DOA algorithm without a realistic simulation to test it against, and you cannot evaluate a simulation without a detection pipeline to consume its output. Rather than pick one piece, I built the end-to-end chain: a physics-based forward model that generates synthetic microphone traces, and a real-time detection and engagement pipeline that turns those traces into fire-control solutions.

The system has two decoupled stages:

1. A 2D/3D FDTD acoustic wave propagation solver that produces synthetic pressure traces at configurable receiver locations. This replaces the clean simulation in the provided code with physics-based propagation through heterogeneous media including terrain, vegetation, and wind.

2. A real-time detection and engagement pipeline that takes the receiver traces and performs detection, bearing estimation, range estimation, tracking, and fire control.

The forward model and the pipeline are deliberately decoupled so that each can be developed, tested, and improved independently. The forward model can be made arbitrarily complex (more noise, more sources, more terrain) and the pipeline must handle whatever comes out of it.

## Approach and Design Decisions

**Forward model.** I started with a Helmholtz (frequency domain) representation of the acoustic wave equation over a 2D plane, but realized that moving sources require time-domain propagation. I implemented a Finite Difference Time Domain solver with configurable stencil order (2nd, 4th, 6th, 8th). The solver supports both 2D and 3D domains with MPI domain decomposition and optional CUDA acceleration via CuPy. Sponge-layer absorbing boundaries prevent reflections from the domain edges.

The source signal is purely synthetic, modeled after the method in the provided script, though the system also accepts WAV files, tones, Ricker wavelets, and noise as source inputs. Sources are injected by bilinear (2D) or trilinear (3D) interpolation to support arbitrary trajectories.

I included three domain configurations: isotropic (free space), wind (uniform flow), and a valley between two reflective hills with vegetation absorption. The valley examples are the most physically interesting because they introduce multipath and terrain effects that stress the detection pipeline.

**Detection pipeline.** The pipeline is a causal, real-time system. It processes receiver traces frame by frame in the order they would arrive.

| Stage | Algorithm |
|-------|-----------|
| Detection | RMS threshold gate |
| Bearing | SRP-PHAT with pre-computed steering vectors (360 azimuth bins) |
| Range | RMS inverse-square-law with CPA calibration |
| Tracking | Causal weighted-least-squares constant-velocity fit |
| Fire control | 3D iterative ballistic lead with pattern spread modeling |

All detections are currently treated as hostile. Engagement is scored by computed miss distance including projectile flight time. A target is considered neutralized after a configurable number of shots (default 3) fall within the acceptable miss distance. Computational latency is tracked because this is modeled as a real-time system.

**ML classifiers (included but not integrated into the production pipeline).** I built PyTorch classifiers for source type (CNN on mel spectrograms, 6 classes: quadcopter, hexacopter, fixed wing, bird, ground vehicle, unknown), maneuver state (1D CNN on kinematic features, 6 classes: steady, turning, accelerating, diving, evasive, hovering), and a fusion classifier that combines acoustic and kinematic inputs. These are not wired into the SRP-PHAT pipeline. No pre-trained weights are shipped. Their value would be in classifying whether a detection is hostile or benign, and in adapting tracker process noise based on detected maneuver state.

**FNO surrogate (included but not integrated).** A Fourier Neural Operator infrastructure to replace the FDTD solver for rapid forward propagation is included with data generation, training, and inference modules. The motivation is that the FDTD solver is too slow for real-time ensemble analysis or rapid parameter studies. An FNO trained on FDTD output could provide forward model evaluations at a fraction of the cost.

## Assumptions

- Single target. The system currently handles one source at a time.
- Hostile by default. No friend-or-foe classification in the production pipeline.
- Known array geometry. Receiver positions are assumed perfectly known. Real deployments would require self-calibration.
- Synthetic sources only. The provided input.wav was used as a reference but the pipeline consumes FDTD output rather than raw recordings.
- Shotgun engagement model. Fire control assumes a shotgun-class effector with modeled pattern spread.

## What Worked

- The FDTD solver produces physically reasonable wavefields in both 2D and 3D, including multipath from terrain and absorption from vegetation.
- SRP-PHAT bearing estimation is robust and computationally efficient. The pre-computed steering vector approach scales well to 360 bins.
- The decoupled architecture (forward model separate from pipeline) proved valuable. Problems in one stage did not propagate to the other.
- Fire control with ballistic lead and miss distance scoring works for moving targets in 3D.
- The unified 2D/3D codebase avoids code duplication across dimensions.

## What Did Not Work (Or Needs More Work)

- Combining 2D and 3D detection algorithms into one unified framework was harder than expected. The 3D methods did not perform well on 2D data. The practical solution was to treat 2D examples as 3D with zero elevation.
- When scoring by computed miss distance (including projectile flight), simpler state estimation methods outperformed more complex ones. I expect this would reverse as the simulation adds more noise, but it was a surprise.
- Fire control for the isotropic (free space) case with a stationary target was unexpectedly difficult to debug, because the system expects a moving target and a stationary one is a degenerate case.
- The ML classifiers are not integrated. I chose to return a working system rather than a partially integrated one.
- I originally wanted to generate complete 3D/4D wavefields (spatial dimensions plus time) so receivers could be placed anywhere in post-processing. The file sizes at the frequencies of interest made this impractical. Receiver locations must be specified before the forward model runs.

## How I Would Improve This For Real-World Deployment

**Immediate priorities:**

- Iron down the forward model, likely replacing FDTD with trained FNO surrogates for rapid wavefield generation.
- Add realistic noise: sensor self-noise, wind noise (including transient gusts), multiple interfering sources, microphone gain and phase mismatches.
- Add complex environments: variable weather, echo-prone terrain, urban canyons.
- Integrate the ML classifiers into the production pipeline for hostile/benign discrimination and maneuver-adaptive tracking.
- Multi-target support.

**Exploiting FNO surrogates:**

- Ensemble forward models: cheap wavefield evaluation means no two runs need be identical. Enable stochastic inputs for noise, source path, and environmental conditions drawn from probability distributions.
- Dimensional analysis: vary all input factors to determine which parameters matter most and which can be simplified.
- Array geometry optimization: combine FNO forward models with optimization to find the best array configuration for a given operating environment, including minimizing the number of sensors required.
- Robustness testing: evaluate performance when receiver positions are uncertain or incorrect, simulating hasty field deployment.
- Echo-prone environments: test and improve detection algorithms in multipath-heavy settings.

**Detection and targeting pipeline:**

- More realistic engagement scenarios: multiple simultaneous targets, varied topography, realistic drone flight profiles (evasive maneuvers, terrain masking).
- Self-calibration for receiver position and gain errors.
- Integration of local ambient conditions at the sensors (detecting transient winds from large enough arrays).
- Transition from "all detections are hostile" to a classification-based rules of engagement system.

## Running the System

### Install

```bash
pip install -e .
```

MPI (for parallel FDTD):

```bash
sudo apt-get install libopenmpi-dev openmpi-bin
pip install mpi4py
```

GPU acceleration:

```bash
pip install -e ".[cuda]"
```

Docker:

```bash
docker compose up dev
```

### Run the FDTD forward model

```bash
# 2-D valley
bash examples/run_valley.sh

# 3-D valley
bash examples/run_valley_3d.sh
```

Or directly:

```bash
python examples/run_fdtd.py \
    --domain hills_vegetation \
    --source-type moving --source-signal propeller \
    --array circular --receiver-count 16 --receiver-radius 2.0 \
    --total-time 3.0 --dx 0.18 --output-dir output/valley_test \
    --use-cuda
```

### Run the engagement pipeline

```bash
python examples/run_pipeline.py output/valley_test
```

Auto-detects 2D vs 3D. Produces:

- `pipeline_summary_{2d,3d}.png` - 6-panel overview
- `radial_engagement_{2d,3d}.png` - engagement diagram
- `beamformer_diagnostic_{2d,3d}.png` - SRP-PHAT vs RMS comparison
- `results_{2d,3d}.json` - machine-readable metrics

### Helmholtz solver

```bash
acoustic-sim --model-preset gradient --frequency 40
```

## Dependencies

| Package | Required |
|---------|----------|
| numpy | yes |
| scipy | yes |
| matplotlib | yes |
| mpi4py | yes (single-process fallback) |
| cupy-cuda12x | optional (GPU) |
| torch | optional (ML classifiers and FNO) |

## Documentation

The docs/ folder contains extensive documentation of the code, algorithms, and underlying physics. Each directory under output/ has a README.md detailing the resultant plots.

| Page | Contents |
|------|----------|
| [Architecture](docs/architecture.md) | Package layout, module inventory, data flow |
| [Algorithms](docs/algorithms.md) | SRP-PHAT, WLS tracker, ballistics, MFP/EKF, ML classifiers |
| [Physics](docs/physics.md) | Wave equation, FD stencils, CFL, absorption, SPL, noise models |
| [Configuration](docs/configuration.md) | Pipeline JSON config, DetectionConfig, CLI flags |
| [API Reference](docs/api.md) | Every public class, method, and function |
| [Usage](docs/usage.md) | Installation, CLI commands, running examples |
| [Studies](docs/studies.md) | Nine parametric robustness studies |
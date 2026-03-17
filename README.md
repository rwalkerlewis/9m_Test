# acoustic-sim

2-D / 3-D acoustic wave propagation and passive acoustic engagement.

FDTD solvers produce synthetic microphone traces.  A real-time SRP-PHAT
pipeline turns those traces into fire-control solutions.

Full documentation: [docs/](docs/index.md)

---

## Install

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

---

## Quick Start

### 1. Run the FDTD forward model

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

### 2. Run the engagement pipeline

```bash
python examples/run_pipeline.py output/valley_test
```

Auto-detects 2-D vs 3-D.  Produces:

- `pipeline_summary_{2d,3d}.png` — 6-panel overview
- `radial_engagement_{2d,3d}.png` — engagement diagram
- `beamformer_diagnostic_{2d,3d}.png` — SRP-PHAT vs RMS comparison
- `results_{2d,3d}.json` — machine-readable metrics

### 3. Helmholtz solver

```bash
acoustic-sim --model-preset gradient --frequency 40
```

---

## Pipeline

| Stage | Algorithm |
|-------|-----------|
| Detection | RMS threshold gate |
| Bearing | SRP-PHAT (pre-computed steering, 360 bins) |
| Range | RMS inverse-square-law, CPA-calibrated |
| Tracking | Causal weighted-least-squares fit |
| Fire control | 3-D iterative ballistic lead (azimuth + elevation) |

Configuration: [`examples/pipeline.config.json`](examples/pipeline.config.json)

---

## FDTD Solver

- 2-D and 3-D, unified codebase
- Configurable FD order (2, 4, 6, 8)
- MPI domain decomposition (y-axis 2-D, z-slab 3-D)
- Optional CUDA via CuPy (automatic fallback to NumPy)
- Sponge-layer absorbing boundaries
- Source signals: propeller, tone, noise, ricker, WAV file
- Domains: isotropic, wind, hills+vegetation

---

## ML Module (Optional)

Six files under `src/acoustic_sim/ml/` implement PyTorch classifiers for
source type (CNN on mel spectrograms), maneuver state (1-D CNN on
kinematics), and fusion classification (acoustic + kinematic).  **Not
wired into the pipeline.**  Requires `torch` (not a project dependency).

Potential value:
- **ManeuverClassifier** → adaptive tracker process noise
- **FusionClassifier** → class-based fire/no-fire decisions
- **compute_kinematic_features()** → pure-numpy threat heuristics (usable now)

See [docs/index.md](docs/index.md#ml-module) for details.

---

## Dependencies

| Package | Required |
|---------|----------|
| numpy | yes |
| scipy | yes |
| matplotlib | yes |
| mpi4py | yes (single-process fallback) |
| cupy-cuda12x | optional (GPU) |
| torch | optional (ML classifiers) |


## Discussion

When given this problem my first thought was that a modeling enviornment was missing,
and I began with that before expanding much further to the suggestions listed in the prompting
document. I began by playing with using the Helmholz representation of the acoustic wave equation over a 2D plane, but realized that if I wanted to add moving sources I needed to model the wave equation in time domain.
Thus after a bit of effort I implemented a Finite Difference representation of the acoustic wave equation. Following some difficulties with constructive noise I implemented a variable higher order implementation for the finite differences (I am not sure if this was ultimately necessary as I added at the same time that I tore out the FDTD model and rewrote it). 

I ultimately settled on a process where the forward model is decoupled from the detection and targeting algorithm. With this forward model source behavior, added noise, transient explosions, and sound wave interactions can be modelled. The idea was to generate complete 3D or 4D wavefields (ndim + time) so that the end user could place receivers wherever they wanted. Unfortunately this would result in unacceptably large files for the frequencies of investigation necessary. Instead, the receiver locations are specified by the forward model and a first effort toward a Fourier Neural Operator training infrastructure to replace the FDTD is included. 

The injected source is purely synthetic and generation is modeled after the method given in your script, although other options, including taking sources from wav files, are available. The source is injected in the domain by either bilinear or trilinear interpolation depending on whether it is a 2D or 3D domain.

Aside from the isotropic 2D domain, I included two rudimentary examples with the acoustic array in a valley between two reflective hills. For all examples the array is circular as my focus was getting the system functional above array manipulation. It is set such that differing array geometry can be tested and I am interested in doing so, however I opted to provide this packet in a functional form first.

After the forward model is generated it is passed to a detection and targeting pipeline. Here only the transient pressure at a given sensor location is used to detect and engage targets. Future work would likely include local ambient conditions at the sensors as well (e.g. detecting transient winds by big enough arrays). At the moment all detections are considered hostile and are also considered neutralzed following a user defined (set at 3) number of shots that fall within the acceptable miss distance. Computational time is considered here as well as it is modeled as a real time system.

The biggest surpise I encountered was from attempting to combine the detection and targeting algorithms for 3D and 2D into one unified framework. It turned out that the 3D methods did not work well for 2D data and it was easier to just consider 2D examples in 3D space with zero elevation in the model. Also when I started enforcing scoring results by the computed miss distance (considering projectile flight as well) the more complicated means of state estimation did not hold up as well as simpler methods. I imagine that would change as more noise is added in the model and I have left more advanced algorithms in. Also ensuring that the fire control worked for the isotropic case seemed to be fiendishly difficult to debug for a system expecting a moving target.

I also included but did not implement more advanced ML classifiers. Ultimately it was more important to return something. The real benefit these classifers could bring would be to interpret source behavior to classify the type of source, and whether it is likely to be hostile or benign (e.g. a bird).

Were I to continue to work with you my first priority would be to iron down the forward model generation, likely using FNOs to build wavefields. After that, I would include all sorts of noise, transient signals, variable weather, wild and complex domains, simulated vehicles, and whatever else we could think of. I would use these cases to test our detection and targeting algorithms to failure, improving from what we learn from model runs in complement from that learned through live testing.

A quick collectiof of other ways I would exploit FNO

- Ensemble forward models: cheap wavefield modeling means that no two runs need be identical. Then we could start to play with stochastic inputs for noise, source path, etc, as well as drawing from probability distributions.
- Dimensional analysis of forward model input factors: if we can vary everything we can figure out what parameters matter less
- Testing different model geometry: This combined with optimization is an essential step. Also optimize for the fewest sensors
- Testing stations with uncertain or wrong positioning information: can this work if the receivers are hastily deployed?
- Test environments prone to echos


The detection and targeting pipeline requires more time and effort than I was able to include to arrive at a reasonable solution for conditions better representing reality. I paid lip service to this by including the erratic_quadcopter example. Much more complexity needs to be considered, including multiple targets, more realistic noise, and more varied topography to state a few.

The docs folder contains extensive documentation of the code, algorithms, and underlying physics. Each directory under output has a README.md detailing the resultant plots.

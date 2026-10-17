# Harbor: DAS + hydrophone detection of hidden submerged objects

A 2-D synthetic harbor in which fiber-optic cable (modelled as a dense line of
pressure receivers) and a few hydrophones listen for an unknown, hidden,
slow submerged object (a small UUV) among ships that broadcast AIS.  The
forward model is the FDTD solver of this package; everything else is in
`src/acoustic_sim/harbor/`.

**Status: the simulation, sensing and processing chain work and are tested, but the UUV is
not yet detected at the intended source level.**  See [Status and findings](#status-and-findings)
for what was measured and what is still unexplained.

## Scenario

| Item | Default |
|------|---------|
| Domain | 2400 m x 1800 m, 5 m grid, c = 1490-1500 m/s (depth-dependent) |
| Land | Pressure-release walls (opaque, thin breakwaters block sound) plus a lossy layer in the water next to the shore (`shore_loss_width`, `shore_loss_max`; R ~ 0.45 by default) |
| Sensors | One 125-channel DAS cable (10 m spacing, two bends) and 4 hydrophones.  DAS is plain pressure at the fiber nodes (no strain directivity) |
| Vessels | Ferry, tug and a moored ship (all AIS) and one hidden UUV `uuv1` (no AIS) |
| Noise | Colored ambient, three distant-shipping plane waves, DAS channel noise, common-mode noise, 8.6 % gain scatter, 2 % dead channels |
| Band | 12-78 Hz at 600 Hz sampling |

Source levels are free-field rms pressure at 100 m [Pa].  The UUV is a sparse tonal
comb (30-40 Hz plus a second harmonic) with little broadband; ships are broadband with a
speed-dependent level and a harmonic series.

## Modules (`src/acoustic_sim/harbor/`)

| Module | Role |
|--------|------|
| `environment.py` | Land mask, depth, sound speed; `build_synthetic_harbor` |
| `sensors.py` | `Cable`, `SensorLayout` |
| `fdtd_runner.py` | `HarborFDTDSolver` (land + shore loss on top of `FDTDSolver`), per-source runs with an on-disk cache, GPU sampling |
| `signatures.py`, `vessels.py`, `ais.py`, `noise.py` | Signatures and level calibration, trajectories, AIS reports and a predictor, noise and sensor imperfections |
| `dataset.py` | One cached FDTD job per vessel, mixed linearly with noise and AIS into a labelled `HarborDataset` |
| `detection.py` | Whitened STFT, matched-field scan, stationary-interference projection, sidelobe-cancelling peak extraction, CFAR threshold |
| `suppression.py` | AIS-aware suppression: per-frame position refinement, ship projection (model or data-adaptive), re-whitening |
| `occupancy.py` | Occupancy-grid track-before-detect |
| `dispersion.py` | Replica wavenumbers that follow the FDTD scheme's numerical dispersion |
| `evaluate.py` | Association with truth, Pd / error / false alarms |
| `device.py` | CPU / GPU selection |

Because the wave equation is linear, each vessel is simulated once at unit level and cached;
level sweeps and removing a vessel need no re-simulation (`assemble_dataset(levels=...)`).

## Processing chain

1. **Pre-process:** channel QC, per-cable common-mode removal, Hann STFT, per-channel noise floor
   (time median, then a running median across frequency) and whitening.
2. **AIS-aware suppression** (`run_ais_aware`): predict each reporting ship, refine its position
   with a fine Bartlett search, project its replica patch (or the data-adaptive subspace) out,
   and **re-estimate the noise floor from the cleaned data and re-whiten**.  Skipping the
   re-whitening leaves the floor inflated by ship energy and hides the target (it cost a factor
   of ~200 in the matched output at the UUV's tone).
3. **Matched-field scan:** free-field 2-D Green's function replicas over a grid, with a
   per-bin stationary-interference projection and a sparse spectral sum.  An optional
   line-enhancement step subtracts a running median across bins.
4. **Extraction:** non-maximum suppression and successive cancellation (sidelobes of an
   accepted peak are projected out before the next candidate is scored).
5. **Occupancy filter:** diffuse, multiply by a likelihood from each frame's map, repeat.

## GPU

`ProcessingConfig(use_cuda=True)`, `OccupancyConfig(use_cuda=True)` and
`HarborFDTDConfig(use_cuda=True)` run the processing and the FDTD on the GPU (CuPy,
`pip install cupy-cuda12x`; single precision for the processing).  CPU and GPU agree: the scan
maps correlate at 1.000000 and the FDTD traces agree to 4e-17.  Measured on a Quadro RTX 4000:
the chain is about 3x faster than the CPU once warm, and an FDTD vessel run takes about 1.5 min
instead of about 8 min.

## Status and findings

At the default UUV level (-30.5 dB re 1 Pa at 100 m, about 30 dB below the ferry) the UUV is
**not** detected: it is found only when it is about as loud as the ferry (a sweep of the UUV level
with the model-projection chain found it, with a 64 m occupancy error, only at -0.9 dB).
The source level is a first guess, not a tuned or measured value.  Every item below was a single
bounded experiment with a pass/fail criterion fixed beforehand.

| Hypothesis for the ship residual | Measurement | Result |
|---|---|---|
| Ship projection removes the UUV too | Fraction of the UUV replica inside the ship subspace | 5-33 %: not the cause |
| Whitening floor inflated by ships | Matched output at the UUV tone, UUV-only vs all vessels | **Real bug**, fixed by re-whitening (about 200x) |
| DAS gain scatter limits cancellation | Divide out the true gains | No effect (B at the UUV 3.6 -> 3.5) |
| Motion, patch size, assumed sound speed | Ferry-only cancellation depth, real FDTD vs exact free-field control | Real: 3-6 dB in every setting; control: 16-45 dB |
| Shoreline reverberation | Same depth, shore reflection 0.45 / 0.07 / 0.005 | 5.2 / 6.6 / 6.5 dB: not the cause |
| Emission-time (retarded) positions | Retarded per-channel replica positions | Shift 1.2 m, no change (5.2 dB) |
| Harbor geometry and shadowing | Same ferry in an open-water FDTD | 4-8 dB: not the cause |
| **FDTD numerical dispersion** | Stationary source vs analytic Green's function | **Real**: phase error up to 0.8 rad (axis) / 2.5 rad (diagonal) at 800 m, 40-60 Hz, anisotropic |
| Brute-force fix by FD order / sample rate | Order 4/8 at 600/1200 Hz | None reaches 0.1 rad (leapfrog time-stepping error remains) |
| Dispersion-corrected replica (`dispersion.py`) | Same stationary map | **Fixed**: <= 0.102 rad at 400-800 m, both directions |
| ... and the moving ferry | Ferry-only cancellation depth with the corrected replica | Only 5.5 dB (1 s frames), 11.6 dB (2 s, open water): **the criterion (20 dB) was not met** |

### Open

The dispersion fix makes a stationary source's replica accurate to <= 0.1 rad, yet the moving
ferry is cancelled by only 5-12 dB.  The ship has to be cancelled by about 31 dB to match the UUV.
What limits the moving ferry is not established.  Unverified candidates: the 10-50 m near-field
channels (the ferry runs parallel to the cable about 30 m away for much of its track, and the
stationary map shows 10-25 % amplitude errors at those ranges), and the representation of a moving
source in the FDTD.  The suggested next experiment is the cancellation depth of a *stationary*
source at 400, 150 and 30 m: low at all ranges points at the moving-source representation, low only
near the cable points at the near-field channels (which could then be gated).

An analytic forward model (free-field channel with a known replica, as in a simple matched-field
demo) would make the technique work by construction; this package currently uses the FDTD only,
by choice.

## Examples

| Script | What it does | Time |
|---|---|---|
| `examples/harbor_m1_demo.py` | Map, one FDTD source run, DAS gather, wavefield snapshots | ~15 s |
| `examples/harbor_m2_dataset.py` | Builds the labelled 4-vessel dataset and overview figures (cached FDTD) | ~6 min on a GPU config |
| `examples/harbor_m3_detect.py` | Baseline CFAR detection sweeps **without** AIS suppression (the loud ships hide the UUV) | ~15 min |

`examples/harbor.scenario.json` is the default scenario (set `"fdtd": {"use_cuda": true}` to
simulate on the GPU).

```python
from acoustic_sim.harbor import HarborDataset, build_environment
from acoustic_sim.harbor.detection import ProcessingConfig, build_grid
from acoustic_sim.harbor.suppression import SuppressionConfig, run_ais_aware
from acoustic_sim.harbor.occupancy import OccupancyConfig, OccupancyGrid
from acoustic_sim.harbor.dispersion import for_scenario

ds = HarborDataset.load("output/harbor/dataset.npz")
env = build_environment(ds.cfg)
pc = ProcessingConfig(win_s=2.0, hop_s=1.0, use_cuda=True,
                      dispersion=for_scenario(env, ds.cfg.fdtd))   # drop this for real data
run = run_ais_aware(ds.traces, ds.layout, ds.fs, env, ds.ais, pc, SuppressionConfig(),
                    grid=build_grid(env, pc))
track = OccupancyGrid(run.scanner.grid, OccupancyConfig(use_cuda=True)).run(run.scan.B, run.wh.times)
```

Use `dispersion=` only for FDTD-generated data: it models the simulator's grid, not the ocean.

## Tests

`pytest tests/harbor` (90+ tests; the GPU variants skip automatically without CuPy / a GPU).
They cover the environment and sensors, the FDTD runner (amplitude against the analytic
Green's function, moveout, linearity, cache, shoreline), signatures, AIS, noise, the dataset
mixer, detection and scoring, ship refinement and projection (including a synthetic scene in
which only the data-adaptive projection removes a multipath ship), the occupancy filter,
dispersion, and GPU / CPU parity.

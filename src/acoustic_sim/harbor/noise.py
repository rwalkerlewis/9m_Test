"""Noise and sensor imperfections added to the noise-free FDTD gathers.

Components (all random, reproducible from an ``np.random.Generator``):

* **ambient** - colored, spatially incoherent noise in the water at every sensor;
* **distant shipping** - a few coherent plane waves from the open-sea sector,
  seen as correlated noise across the cable;
* **instrument** - hydrophone self noise; DAS per-channel white noise plus a
  cable-wide common-mode (laser/phase) noise;
* **coupling** - per-channel DAS gain scatter and dead channels.

Ambient and shipping noise pass through the coupling gain (they are acoustic);
instrument noise does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_sim.harbor.sensors import SensorLayout
from acoustic_sim.harbor.signatures import shaped_noise


@dataclass
class NoiseConfig:
    """Noise levels are rms pressure [Pa] over the full band at each sensor.

    ``shipping_azimuth_deg`` is the sector (maths convention, degrees) of
    directions *toward* the distant sources; (200, 340) is the southern
    open-sea side of the default harbor.
    """

    ambient_rms: float = 0.02
    shipping_rms: float = 0.02
    n_shipping: int = 3
    shipping_azimuth_deg: tuple[float, float] = (200.0, 340.0)
    hydrophone_self_rms: float = 0.005
    das_noise_rms: float = 0.05
    das_common_mode_rms: float = 0.03
    das_gain_sigma: float = 0.1
    das_dead_fraction: float = 0.02
    c: float = 1500.0

    def __post_init__(self) -> None:
        self.shipping_azimuth_deg = tuple(self.shipping_azimuth_deg)  # JSON round-trips tuples as lists


@dataclass
class NoiseRealisation:
    """One random draw of noise and sensor effects."""

    ambient: np.ndarray      # (n_rec, n) acoustic noise incl. distant shipping, before coupling
    instrument: np.ndarray   # (n_rec, n) electronic noise, after coupling
    gain: np.ndarray         # (n_rec,) coupling gain
    dead: np.ndarray         # (n_rec,) bool
    shipping_azimuth_deg: np.ndarray

    def apply(self, clean: np.ndarray) -> np.ndarray:
        """Noisy measurement of *clean* acoustic pressure."""
        out = self.gain[:, None] * (clean + self.ambient) + self.instrument
        out[self.dead] = 0.0
        return out.astype(np.float32)

    @property
    def noise_only(self) -> np.ndarray:
        return self.apply(np.zeros_like(self.ambient))


def generate_noise(
    layout: SensorLayout,
    fs: float,
    n_steps: int,
    cfg: NoiseConfig,
    rng: np.random.Generator,
) -> NoiseRealisation:
    n_rec = layout.n_receivers
    n_das = layout.n_das
    pos = layout.receivers

    ambient = np.empty((n_rec, n_steps), dtype=np.float32)
    for i in range(n_rec):
        ambient[i] = cfg.ambient_rms * shaped_noise(n_steps, fs, rng, 5.0, 120.0, slope=0.5)

    az = rng.uniform(*np.radians(cfg.shipping_azimuth_deg), size=cfg.n_shipping)
    if cfg.n_shipping > 0 and cfg.shipping_rms > 0:
        f = np.fft.rfftfreq(n_steps, 1.0 / fs)
        centre = pos.mean(axis=0)
        for a in az:
            spec = np.fft.rfft(cfg.shipping_rms * shaped_noise(n_steps, fs, rng, 8.0, 70.0, slope=1.0))
            u = np.array([np.cos(a), np.sin(a)])
            advance = (pos - centre) @ u / cfg.c  # nearer to the source -> earlier arrival
            ambient += np.fft.irfft(spec[None, :] * np.exp(2j * np.pi * f[None, :] * advance[:, None]), n_steps).astype(np.float32)

    instrument = np.empty((n_rec, n_steps), dtype=np.float32)
    instrument[:n_das] = cfg.das_noise_rms * rng.standard_normal((n_das, n_steps))
    instrument[n_das:] = cfg.hydrophone_self_rms * rng.standard_normal((n_rec - n_das, n_steps))
    for name, sl in layout.slices.items():
        if name == "hydrophones":
            continue
        instrument[sl] += (cfg.das_common_mode_rms * shaped_noise(n_steps, fs, rng, 1.0, 100.0, slope=1.0)).astype(np.float32)

    gain = np.ones(n_rec)
    gain[:n_das] = np.exp(cfg.das_gain_sigma * rng.standard_normal(n_das))
    dead = np.zeros(n_rec, dtype=bool)
    n_dead = int(round(cfg.das_dead_fraction * n_das))
    if n_dead:
        dead[rng.choice(n_das, size=n_dead, replace=False)] = True

    return NoiseRealisation(ambient=ambient, instrument=instrument, gain=gain, dead=dead, shipping_azimuth_deg=np.degrees(az))

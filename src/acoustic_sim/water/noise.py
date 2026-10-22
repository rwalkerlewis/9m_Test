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

``NoiseConfig.model = "harbor_v2"`` adds the separately switchable components of :class:`HarborV2Config` (snapping
shrimp, surface agitation, quay machinery, fibre strain noise, interrogator noise); every component's spectral shape,
level and whether it is measured, derived or assumed are in ``docs/noise.md``.  The old model (``"default"``) stays the
default of every scenario file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from acoustic_sim.water.sensors import SensorLayout
from acoustic_sim.water.signatures import shaped_noise


@dataclass
class NoiseConfig:
    """Noise levels are rms pressure [Pa] over the full band at each sensor.

    ``shipping_azimuth_deg`` is the sector (maths convention, degrees) of
    directions *toward* the distant sources; (200, 340) is the southern
    open-sea side of the default harbor.

    ``band_scale`` multiplies the fixed band edges of the coloured components
    (ambient 5 to 120 Hz, distant shipping 8 to 70 Hz, common mode 1 to 100 Hz),
    for scenarios on a finer grid whose band is wider; the rms values are then
    spread over the scaled band.  Instrument noise is white at the sample rate.
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
    band_scale: float = 1.0
    model: str = "default"
    harbor_v2: "HarborV2Config" = field(default_factory=lambda: HarborV2Config())

    def __post_init__(self) -> None:
        self.shipping_azimuth_deg = tuple(self.shipping_azimuth_deg)  # JSON round-trips tuples as lists
        if self.model not in ("default", "harbor_v2"):
            raise ValueError(f"noise model must be 'default' or 'harbor_v2', got {self.model!r}")
        if isinstance(self.harbor_v2, dict):
            self.harbor_v2 = HarborV2Config.from_dict(self.harbor_v2)


@dataclass
class NoiseRealisation:
    """One random draw of noise and sensor effects."""

    ambient: np.ndarray      # (n_rec, n) acoustic noise incl. distant shipping, before coupling
    instrument: np.ndarray   # (n_rec, n) electronic noise, after coupling
    gain: np.ndarray         # (n_rec,) coupling gain
    dead: np.ndarray         # (n_rec,) bool
    shipping_azimuth_deg: np.ndarray
    das_white: np.ndarray | None = None   # (n_das, n) the flat DAS channel noise inside ``instrument``
    parts: dict = field(default_factory=dict)   # harbor_v2: the components added (``add_harbor_v2``)

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
    env=None,
    machinery: np.ndarray | None = None,
) -> NoiseRealisation:
    """One draw of the noise (the components of the module docstring; with ``cfg.model == "harbor_v2"`` also those of
    :class:`HarborV2Config`, *env* and *machinery* as :func:`add_harbor_v2` needs them).  The old model's draws come
    first, so a harbor_v2 draw has the same old components as the default model's draw of the same seed."""
    n_rec = layout.n_receivers
    n_das = layout.n_das
    pos = layout.receivers

    bs = cfg.band_scale
    ambient = np.empty((n_rec, n_steps), dtype=np.float32)
    for i in range(n_rec):
        ambient[i] = cfg.ambient_rms * shaped_noise(n_steps, fs, rng, 5.0 * bs, 120.0 * bs, slope=0.5)

    az = rng.uniform(*np.radians(cfg.shipping_azimuth_deg), size=cfg.n_shipping)
    if cfg.n_shipping > 0 and cfg.shipping_rms > 0:
        f = np.fft.rfftfreq(n_steps, 1.0 / fs)
        centre = pos.mean(axis=0)
        for a in az:
            spec = np.fft.rfft(cfg.shipping_rms * shaped_noise(n_steps, fs, rng, 8.0 * bs, 70.0 * bs, slope=1.0))
            u = np.array([np.cos(a), np.sin(a)])
            advance = (pos - centre) @ u / cfg.c  # nearer to the source -> earlier arrival
            ambient += np.fft.irfft(spec[None, :] * np.exp(2j * np.pi * f[None, :] * advance[:, None]), n_steps).astype(np.float32)

    instrument = np.empty((n_rec, n_steps), dtype=np.float32)
    das_white = (cfg.das_noise_rms * rng.standard_normal((n_das, n_steps))).astype(np.float32)
    instrument[:n_das] = das_white
    instrument[n_das:] = cfg.hydrophone_self_rms * rng.standard_normal((n_rec - n_das, n_steps))
    for name, sl in layout.slices.items():
        if name == "hydrophones":
            continue
        instrument[sl] += (cfg.das_common_mode_rms * shaped_noise(n_steps, fs, rng, 1.0 * bs, 100.0 * bs, slope=1.0)).astype(np.float32)

    gain = np.ones(n_rec)
    gain[:n_das] = np.exp(cfg.das_gain_sigma * rng.standard_normal(n_das))
    dead = np.zeros(n_rec, dtype=bool)
    n_dead = int(round(cfg.das_dead_fraction * n_das))
    if n_dead:
        dead[rng.choice(n_das, size=n_dead, replace=False)] = True

    real = NoiseRealisation(ambient=ambient, instrument=instrument, gain=gain, dead=dead, shipping_azimuth_deg=np.degrees(az),
                            das_white=das_white)
    if cfg.model == "harbor_v2":
        real = add_harbor_v2(real, layout, fs, n_steps, cfg, rng, env=env, machinery=machinery)
    return real


# ----------------------------------------------------------------------
# harbor_v2: components with sources behind them (docs/noise.md)
# ----------------------------------------------------------------------

@dataclass
class ShrimpConfig:
    """Snapping shrimp: impulsive clicks from colonies along the land edges (quays, structures).

    Clicks are a Poisson process, ``rate_per_m`` per second per metre of land edge; each click comes from a point on the
    edge (the edge sampled every ``edge_spacing_m``), lasts ``click_samples`` samples (one or two at these sample rates:
    the click is far shorter than a sample) and has a peak to peak source level drawn from a normal distribution in dB
    (``sl_pp_db``, ``sl_pp_db_sd``, re 1 uPa at 1 m), scaled to the simulated band by ``band_fraction_db`` (the part of
    a click's energy below the grid's top frequency).  It reaches every sensor by the free field sum: delayed by range
    over ``c``, spread spherically to ``transition_m`` and cylindrically beyond.  Snapping shrimp live within about
    ``max_abs_latitude_deg`` of the equator: the component is off when the scenario's ``latitude_deg`` is unknown or
    outside that (``off_reason``).
    """

    enabled: bool = False
    rate_per_m: float = 0.3
    sl_pp_db: float = 186.0
    sl_pp_db_sd: float = 1.5
    band_fraction_db: float = -20.0
    edge_spacing_m: float = 5.0
    click_samples: int = 1
    transition_m: float = 5.0
    latitude_deg: float | None = None
    max_abs_latitude_deg: float = 40.0

    def off_reason(self) -> str | None:
        if not self.enabled:
            return "disabled"
        if self.latitude_deg is None:
            return "the scenario gives no latitude (snapping shrimp live within about 40 degrees of the equator)"
        if abs(self.latitude_deg) > self.max_abs_latitude_deg:
            return f"latitude {self.latitude_deg:g} is outside the {self.max_abs_latitude_deg:g} degrees snapping shrimp live in"
        return None


@dataclass
class SurfaceConfig:
    """Wind driven surface agitation: spatially incoherent noise at every sample point whose power spectrum is
    ``level_1khz_db`` (dB re 1 uPa^2/Hz at 1 kHz, by sea state) times ``(f / 1 kHz)^-slope``, flattened below
    ``knee_hz``; its level wanders over minutes (a log normal random walk of ``variation_db`` standard deviation and
    ``variation_s`` correlation time)."""

    enabled: bool = False
    sea_state: int = 2
    levels_1khz_db: tuple = (44.5, 53.5, 58.8, 62.6, 65.5, 67.9, 69.9)    # sea state 0..6: 44.5 + 30 log10(n + 1)
    slope: float = 1.7
    knee_hz: float = 300.0
    variation_db: float = 2.0
    variation_s: float = 120.0

    def __post_init__(self) -> None:
        self.levels_1khz_db = tuple(self.levels_1khz_db)


@dataclass
class MachineryConfig:
    """Quay machinery (a pump or a generator): stationary sources at ``positions`` on the quay edge, each a tonal series
    on ``line_hz`` (the mains frequency and its harmonics) plus broadband noise, propagated through the forward model
    (one FDTD run per scenario, cached; ``water.dataset.machinery_gather``) and mixed at ``level_pa`` (rms at 100 m, as
    the vessels)."""

    enabled: bool = False
    positions: tuple = ()
    level_pa: float = 0.1
    line_hz: float = 50.0
    n_lines: int = 8
    broadband_db: float = -10.0       # broadband power relative to the lines

    def __post_init__(self) -> None:
        self.positions = tuple(tuple(p) for p in self.positions)


@dataclass
class FibreConfig:
    """Strain noise of the fibre itself (DAS channels only): a slow drift whose power rises as ``f^-drift_slope`` below
    ``drift_corner_hz`` (``drift_rms``), and traffic on the quay the cable runs along: vehicles pass along each cable at
    ``vehicle_speed`` [m/s], ``vehicles_per_min`` of them, and each shakes the ground in ``traffic_band_hz``; that
    disturbance travels along the cable at the ground wave speed ``traffic_speed`` [m/s] (so neighbouring channels see
    it with a delay of their distance over that speed) and is felt over ``corr_length_m`` around the vehicle; rms
    ``traffic_rms`` (trace units) while a vehicle passes."""

    enabled: bool = False
    drift_rms: float = 0.02
    drift_corner_hz: float = 0.1
    drift_slope: float = 2.0
    traffic_rms: float = 0.02
    traffic_speed: float = 300.0
    vehicle_speed: float = 15.0
    vehicles_per_min: float = 2.0
    corr_length_m: float = 150.0
    traffic_band_hz: tuple = (3.0, 30.0)

    def __post_init__(self) -> None:
        self.traffic_band_hz = tuple(self.traffic_band_hz)


@dataclass
class InterrogatorConfig:
    """Interrogator noise of every DAS channel, replacing the flat ``das_noise_rms``: a floor of ``floor_rms`` (rms over
    the sample band, trace units) whose power rises as ``(corner_hz / f)^slope`` below ``corner_hz`` (the laser's phase
    noise), and fading: at any time a fraction ``fade_fraction`` of the channels is in a fade of mean duration
    ``fade_s`` with a floor ``fade_factor`` times higher."""

    enabled: bool = False
    floor_rms: float = 0.05
    corner_hz: float = 50.0
    slope: float = 1.0
    fade_fraction: float = 0.10
    fade_s: float = 2.0
    fade_factor: float = 3.16


@dataclass
class HarborV2Config:
    """The components of ``NoiseConfig.model = "harbor_v2"``, each switched on its own; the old components stay."""

    shrimp: ShrimpConfig = field(default_factory=ShrimpConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    machinery: MachineryConfig = field(default_factory=MachineryConfig)
    fibre: FibreConfig = field(default_factory=FibreConfig)
    interrogator: InterrogatorConfig = field(default_factory=InterrogatorConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "HarborV2Config":
        kinds = {"shrimp": ShrimpConfig, "surface": SurfaceConfig, "machinery": MachineryConfig, "fibre": FibreConfig,
                 "interrogator": InterrogatorConfig}
        return cls(**{k: (v if not isinstance(v, dict) else kinds[k](**v)) for k, v in d.items()})

    def to_dict(self) -> dict:
        return asdict(self)


UPA = 1e-6     # 1 micropascal [Pa]


def _psd_noise(n: int, fs: float, rng: np.random.Generator, psd, rows: int = 1) -> np.ndarray:
    """Gaussian noise (rows, n) whose one sided power spectral density is ``psd(f)`` [unit^2 / Hz]."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    p = np.zeros_like(f)
    p[1:] = psd(f[1:])
    amp = np.sqrt(p * fs * n / 2.0)                        # E|X_k|^2 = psd * fs * n / 2 for a one sided density
    X = (rng.standard_normal((rows, len(f))) + 1j * rng.standard_normal((rows, len(f)))) / np.sqrt(2.0) * amp[None, :]
    X[:, 0] = 0.0
    if n % 2 == 0:
        X[:, -1] = X[:, -1].real * np.sqrt(2.0)
    return np.fft.irfft(X, n, axis=1)


def land_edge_points(env, spacing: float) -> np.ndarray:
    """Points (M, 2) on the water side of every land edge, about *spacing* apart (water cells next to land, thinned on
    a lattice of that spacing)."""
    from scipy.ndimage import binary_dilation

    land = ~env.water
    edge = env.water & binary_dilation(land)
    iy, ix = np.nonzero(edge)
    x0, _, y0, _ = env.extent
    xy = np.column_stack([x0 + (ix + 0.5) * env.dx, y0 + (iy + 0.5) * env.dx])
    if not len(xy):
        return xy
    key = np.floor(xy / spacing).astype(np.int64)
    _, keep = np.unique(key[:, 0] * 1_000_003 + key[:, 1], return_index=True)
    return xy[np.sort(keep)]


def shrimp_noise(layout: SensorLayout, fs: float, n: int, cfg: ShrimpConfig, rng: np.random.Generator, env,
                 c: float = 1500.0) -> tuple[np.ndarray, dict]:
    """Snapping shrimp clicks at every receiver (N, n) [Pa], and the click list's statistics."""
    pts = land_edge_points(env, cfg.edge_spacing_m)
    S = layout.sample_points
    out = np.zeros((len(S), n))
    if not len(pts):
        return layout.collapse(out), {"clicks": 0, "edge_points": 0}
    edge_len = len(pts) * cfg.edge_spacing_m
    T = n / fs
    k = rng.poisson(cfg.rate_per_m * edge_len * T)
    t = rng.uniform(0.0, T, size=k)
    src = pts[rng.integers(0, len(pts), size=k)]
    sl = cfg.sl_pp_db + cfg.sl_pp_db_sd * rng.standard_normal(k) + cfg.band_fraction_db
    amp_1m = 10.0 ** (sl / 20.0) * UPA / 2.0                   # peak amplitude at 1 m [Pa] (half the peak to peak)
    sign = rng.choice([-1.0, 1.0], size=k)
    for j in range(0, k, 2048):                                  # vectorized over clicks, in chunks
        r = np.hypot(src[j:j + 2048, None, 0] - S[None, :, 0], src[j:j + 2048, None, 1] - S[None, :, 1])
        r = np.maximum(r, 1.0)
        tr = cfg.transition_m
        spread = np.where(r <= tr, 1.0 / r, 1.0 / (tr * np.sqrt(r / tr)))
        a = (sign[j:j + 2048] * amp_1m[j:j + 2048])[:, None] * spread               # (K, S)
        idx = np.rint((t[j:j + 2048, None] + r / c) * fs).astype(np.int64)
        for q in range(cfg.click_samples):
            ii = idx + q
            ok = ii < n
            cols = np.broadcast_to(np.arange(len(S))[None, :], ii.shape)
            np.add.at(out, (cols[ok], ii[ok]), a[ok] * (1.0 if q == 0 else -1.0))     # two samples: a doublet
    return layout.collapse(out), {"clicks": int(k), "edge_points": int(len(pts)), "edge_length_m": float(edge_len)}


def surface_noise(layout: SensorLayout, fs: float, n: int, cfg: SurfaceConfig, rng: np.random.Generator) -> np.ndarray:
    """Wind noise at every receiver (N, n) [Pa]: incoherent between sample points, the Knudsen shape and level of the
    sea state, its level wandering over minutes."""
    L = cfg.levels_1khz_db[int(np.clip(cfg.sea_state, 0, len(cfg.levels_1khz_db) - 1))]
    p0 = 10.0 ** (L / 10.0) * UPA ** 2                           # Pa^2/Hz at 1 kHz

    def psd(f):
        return p0 * (np.maximum(f, cfg.knee_hz) / 1000.0) ** (-cfg.slope)

    S = layout.sample_points
    x = _psd_noise(n, fs, rng, psd, rows=len(S))
    # Slow level variation: an Ornstein Uhlenbeck process in dB.
    dt = 1.0 / fs
    m = max(int(np.ceil(n * dt / 1.0)), 2)                      # one value per second, interpolated
    a = np.exp(-1.0 / cfg.variation_s)
    w = np.empty(m)
    w[0] = cfg.variation_db * rng.standard_normal()
    for i in range(1, m):
        w[i] = a * w[i - 1] + cfg.variation_db * np.sqrt(1 - a * a) * rng.standard_normal()
    gain = 10.0 ** (np.interp(np.arange(n) * dt, np.arange(m), w) / 20.0)
    return layout.collapse(x * gain[None, :])


def machinery_signal(n: int, fs: float, cfg: MachineryConfig, rng: np.random.Generator, f_max: float) -> np.ndarray:
    """The source waveform of one machine (n,), rms 1: lines on ``line_hz`` and its harmonics below *f_max* (random
    phases, falling 3 dB per harmonic) plus broadband noise ``broadband_db`` below the lines' power."""
    t = np.arange(n) / fs
    x = np.zeros(n)
    for h in range(1, cfg.n_lines + 1):
        f = h * cfg.line_hz
        if f >= f_max:
            break
        x += 10.0 ** (-3.0 * (h - 1) / 20.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    x /= max(np.sqrt(np.mean(x ** 2)), 1e-30)
    from acoustic_sim.water.signatures import shaped_noise

    b = shaped_noise(n, fs, rng, 20.0, f_max, slope=0.5)
    x = x + 10.0 ** (cfg.broadband_db / 20.0) * b
    return x / np.sqrt(np.mean(x ** 2))


def fibre_noise(layout: SensorLayout, fs: float, n: int, cfg: FibreConfig, rng: np.random.Generator) -> np.ndarray:
    """Fibre strain noise of the DAS channels (N_das, n): slow drift (independent per channel) and quay traffic (the
    vehicles' ground motion travelling along each cable at ``traffic_speed``, under an envelope of ``corr_length_m``
    that moves with each vehicle at ``vehicle_speed``)."""
    out = np.zeros((layout.n_das, n))
    k = 0
    lo, hi = cfg.traffic_band_hz
    t = np.arange(n) / fs
    for cable in layout.cables:
        nc = cable.n_channels
        s = np.arange(nc) * cable.channel_spacing                   # position along the cable [m]
        if cfg.drift_rms > 0:
            drift = _psd_noise(n, fs, rng, lambda f: (np.maximum(f, 1e-3) / cfg.drift_corner_hz) ** (-cfg.drift_slope)
                               * (f <= 20 * cfg.drift_corner_hz), rows=nc)
            out[k:k + nc] += cfg.drift_rms * drift / max(np.sqrt(np.mean(drift ** 2)), 1e-30)
        if cfg.traffic_rms > 0:
            q = _psd_noise(n + int(np.ceil(s[-1] / cfg.traffic_speed * fs)) + 2, fs, rng,
                           lambda f: ((f >= lo) & (f <= hi)).astype(float))[0]
            q /= max(np.sqrt(np.mean(q ** 2)), 1e-30)
            d = s / cfg.traffic_speed * fs                            # the delay of each channel [samples]
            j = np.arange(n)[None, :] + (d[-1] - d)[:, None]          # channel at s hears q delayed by s / speed
            wave = np.interp(j.ravel(), np.arange(len(q)), q).reshape(nc, n)
            env = np.zeros((nc, n))
            n_veh = rng.poisson(cfg.vehicles_per_min * n / fs / 60.0)
            length = s[-1] + 4 * cfg.corr_length_m
            for t0 in rng.uniform(-length / cfg.vehicle_speed, n / fs, size=n_veh):
                pos = -2 * cfg.corr_length_m + cfg.vehicle_speed * (t - t0)            # the vehicle along the cable
                env += np.exp(-0.5 * ((s[:, None] - pos[None, :]) / cfg.corr_length_m) ** 2)
            out[k:k + nc] += cfg.traffic_rms * wave * env
        k += nc
    return out


def interrogator_noise(layout: SensorLayout, fs: float, n: int, cfg: InterrogatorConfig,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Interrogator noise of the DAS channels (N_das, n) and the fade mask (N_das, n) bool."""
    def shape(f):
        return 1.0 + (cfg.corner_hz / np.maximum(f, 1e-3)) ** cfg.slope

    x = _psd_noise(n, fs, rng, shape, rows=layout.n_das)
    flat = _psd_noise(n, fs, rng, lambda f: np.ones_like(f), rows=1)
    x *= cfg.floor_rms / np.sqrt(np.mean(flat ** 2) * np.mean(shape(np.fft.rfftfreq(n, 1.0 / fs)[1:])))
    # Fading: a two state Markov chain per channel (mean fade ``fade_s``, stationary fraction ``fade_fraction``).
    fade = np.zeros((layout.n_das, n), dtype=bool)
    if cfg.fade_fraction > 0:
        dt = 1.0 / fs
        p_out = dt / cfg.fade_s                                   # leave a fade
        p_in = p_out * cfg.fade_fraction / (1.0 - cfg.fade_fraction)
        step = max(int(round(0.05 * fs)), 1)                      # the chain is stepped every 50 ms
        m = int(np.ceil(n / step))
        state = rng.random(layout.n_das) < cfg.fade_fraction
        u = rng.random((layout.n_das, m))
        for j in range(m):
            state = np.where(state, u[:, j] >= p_out * step, u[:, j] < p_in * step)
            fade[:, j * step:(j + 1) * step] = state[:, None]
    x = np.where(fade, cfg.fade_factor * x, x)
    return x, fade


def add_harbor_v2(real: NoiseRealisation, layout: SensorLayout, fs: float, n: int, cfg: NoiseConfig,
                  rng: np.random.Generator, env=None, machinery: np.ndarray | None = None) -> NoiseRealisation:
    """The harbor_v2 components added to an old model realisation (acoustic ones before the coupling gain, the fibre's
    and the interrogator's after it).  *env* (the forward environment) is needed by the shrimp; *machinery* (N, n), the
    machinery gather at unit level (``dataset.machinery_gather``), by the machinery.  ``real.parts`` records what was
    added and why a component was left out."""
    v2 = cfg.harbor_v2
    parts = {}
    nd = layout.n_das
    if v2.shrimp.enabled:
        why = v2.shrimp.off_reason() or (None if env is not None else "no environment given")
        if why is None:
            x, info = shrimp_noise(layout, fs, n, v2.shrimp, rng, env, cfg.c)
            real.ambient += x.astype(np.float32)
            parts["shrimp"] = info
        else:
            parts["shrimp"] = {"off": why}
    if v2.surface.enabled:
        real.ambient += surface_noise(layout, fs, n, v2.surface, rng).astype(np.float32)
        parts["surface"] = {"sea_state": v2.surface.sea_state}
    if v2.machinery.enabled:
        if machinery is None:
            parts["machinery"] = {"off": "no machinery gather for this scenario"}
        else:
            real.ambient += (v2.machinery.level_pa * machinery).astype(np.float32)
            parts["machinery"] = {"level_pa": v2.machinery.level_pa, "positions": [list(p) for p in v2.machinery.positions]}
    if v2.fibre.enabled and nd:
        real.instrument[:nd] += fibre_noise(layout, fs, n, v2.fibre, rng).astype(np.float32)
        parts["fibre"] = {}
    if v2.interrogator.enabled and nd:
        x, fade = interrogator_noise(layout, fs, n, v2.interrogator, rng)
        real.instrument[:nd] += (x - 0.0).astype(np.float32)
        real.instrument[:nd] -= real.das_white.astype(np.float32) if real.das_white is not None else 0.0
        parts["interrogator"] = {"fading_fraction": float(fade.mean())}
    real.parts = parts
    return real

"""Passive bistatic detection of a silent body using an AIS ship as the illuminator.

The analytic prototype is ``examples/harbor_mfp_demo.py``; this module is its port to the
FDTD harbor data (strain mode DAS, a real scattering body, quay reflections).

Chain for one dataset
---------------------
1. :func:`stft` of all channels, channel QC and per cable common mode removal (from
   ``detection.py``), band ``f_lo`` to ``f_hi``.
2. :func:`ship_positions`: AIS prediction per frame and a fine Bartlett refinement with
   per channel PHAT normalisation (the AIS error of 8 m is several wavelengths here).
3. :func:`reference_spectrum`: the illuminator spectrum per frame and bin, either from the
   data (beam steered at the refined ship position) or the oracle (true waveform).
4. :func:`remove_direct`: project out the ship's replica patch and/or the data adaptive
   dominant subspace, per bin and frame.
5. :func:`bistatic_map`: for every frame a map over the scan grid, with one of several
   processors (see :class:`BistaticConfig.method`), in the ``ScanResult`` convention so
   the occupancy filter and the peak extraction of ``detection.py`` apply unchanged.

Replicas always go through :func:`replicas`, which evaluates the 2-D Green's function at the
layout's **sample points** and collapses them with :meth:`SensorLayout.collapse`, so strain
channels get the second difference of the Green's function and pressure channels the Green's
function itself.  Convention: STFT ``exp(-i w t)``, outgoing ``H0^(2)(kr)`` (as ``detection.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import map_coordinates
from scipy.signal import get_window, resample

from acoustic_sim.harbor.ais import AISReports
from acoustic_sim.harbor.detection import DetectionSet, ScanResult, _nms, channel_weights, green2d, remove_common_mode
from acoustic_sim.harbor.detection import ProcessingConfig
from acoustic_sim.harbor.device import CPU, Device, get_device
from acoustic_sim.harbor.dispersion import DispersionModel, replica_wavenumbers
from acoustic_sim.harbor.environment import HarborEnvironment
from acoustic_sim.harbor.sensors import SensorLayout
from acoustic_sim.harbor.track import TrackConfig, kalman_track

METHODS = ("bartlett", "bartlett_phat", "coherent", "mvdr", "music")
REMOVALS = ("none", "eca", "patch", "adaptive", "patch+adaptive")
REFERENCES = ("data", "oracle")
REF_MODES = ("beam", "equalised", "snr", "mrc", "iterated")
TRACK_FILTERS = ("none", "line", "kalman")
MAP_NORMS = ("max", "none")


@dataclass
class BistaticConfig:
    """Parameters of the bistatic chain.

    Attributes
    ----------
    win_s, hop_s : float
        STFT window and hop [s].
    f_lo, f_hi : float
        Band [Hz] (``f_hi`` defaults to 0.95 of the grid's usable band).
    bin_stride : int
        Use every n-th bin of the band.
    c : float
        Replica sound speed [m/s].
    grid_dx, grid_margin : float
        Scan grid spacing and distance from land [m].
    edge : float
        Keep the scan grid this far [m] from the domain boundary (the sponge).
    reference : str
        ``"data"`` (beam at the refined ship position) or ``"oracle"`` (true ship waveform).
    refine, refine_half, refine_step : bool, float, float
        Refine the AIS position with a Bartlett search of this half width and step [m].
    removal : str
        ``"eca"`` (default): time domain least squares projection of every channel onto the
        reference delayed by the lags around the predicted direct and first order image
        arrivals, per block of ``eca_block_s`` seconds, lag windows ``eca_halfwidth_s`` wide
        (:func:`eca_remove`).  ``"patch"`` (free field replica patch projection at the ship),
        ``"adaptive"`` (dominant data subspace over a window of frames), ``"patch+adaptive"``
        or ``"none"`` (frequency domain, kept for comparison).
    eca_images : bool
        Include the quay and pier image arrivals (:func:`clutter_images`) in the ECA lags.
    eca_doppler : bool
        Follow the moving ship within every ECA block: the data reference beam advances each
        channel by its range to the ship **per sample**, and every ECA regressor is the
        reference resampled at the per sample delay of its path (direct or image) plus a fixed
        lag offset.  With the default per block fixed lags the direct path delay drifts by up
        to ``v T_block / c`` (1.3 ms for 4 m/s and 0.5 s, half a cycle at 380 Hz) inside a
        block, which caps the depth near 10 dB for a moving ship (stage 0a).
    track_smooth_s : float
        With ``eca_doppler``, fit the refined ship track by a local straight line over this
        window [s] before interpolating it per sample (stage 0a: the per frame refinement
        jitters by about 1 m, which per sample delays read as velocity noise; 9 dB with the
        jittery track against 24 dB with the exact one for the ship at 4 m/s).  0: no smoothing.
    ref_channels : str
        Channels for the data reference beam: ``"all"`` or a cable name or ``"hydrophones"``.
    ref_mode : str
        How the data reference is built from the channels (stage 1a).  ``"beam"``: plain delay
        and sum (stage 0).  ``"equalised"``: every channel is first divided, in the frequency
        domain, by its modelled response to the ship (the collapsed Green's function of
        :class:`ReplicaModel`: the second difference along the cable for strain channels),
        floored at ``ref_floor`` times the largest response over the channels at that frequency
        so the strain nulls are not amplified.  ``"snr"``: a maximum ratio combiner, every
        channel and frequency weighted by ``sqrt(S) / N`` with ``S`` the power of the direct path
        fitted by a first ECA pass with the plain beam and ``N`` the power of its residual.
        ``"mrc"``: the model based maximum ratio combiner, ``sum_i conj(a_i) X_i / N_i`` normalised,
        with ``a_i`` the modelled response and ``N_i`` the residual power of a first ECA pass.
        ``"iterated"``: plain beam, ECA, coherent map, the modelled echo of the strongest map
        peak is fitted (direct path and image lags excluded) and subtracted from the traces, and
        the beam is rebuilt (one iteration).
    ref_floor : float
        Floor of the equaliser relative to the largest modelled response at each frequency.
    ref_taper_hz : float
        The equalised and SNR weighted references are limited to the band, with a cosine taper
        of this width [Hz] outside ``f_lo`` and ``f_hi``.
    track_filter : str
        Track used for the ECA lags, the per sample delays and the map (stage 1b).
        ``"kalman"`` (default): constant velocity Kalman filter and smoother over the AIS reports
        and the refined positions (:func:`acoustic_sim.harbor.track.kalman_track`, process noise
        ``track_accel``, refinement error ``track_meas_sigma``, gate ``track_gate_sigma``, AIS
        errors ``ais_pos_sigma``, ``ais_sog_sigma``, ``ais_cog_sigma``).  ``"line"``: the local
        straight line fit over ``track_smooth_s`` (stage 0).  ``"none"``: the refined positions.
    fallback_speed : float
        With ``eca_doppler``, blocks in which the smoothed speed is below this [m/s] use fixed
        lags at the smoothed position instead of per sample delays (a stationary ship's
        residual jitter must not be read as velocity).
    map_norm : str
        ``"max"``: every map is divided by its largest value (stage 0; a threshold is then
        relative to that dataset).  ``"none"``: the map is left in its own units (with
        ``coherent_norm="snr"`` the floor is the noise power, flat over the grid), which a
        threshold set on body free runs needs.
    patch_half, svd_rtol : float
        Patch half width [m] (3 x 3 points) and singular value cut of the patch replicas.
    adaptive_rank, adaptive_window : int
        Rank removed and half window in frames for the adaptive projection.
    exclusion_radius : float
        Map cells within this distance [m] of the ship are zeroed.
    method : str
        Map processor: ``"bartlett"`` (incoherent over frequency), ``"bartlett_phat"``
        (per channel magnitude normalised, resolution set by bandwidth not by the ship
        spectrum), ``"coherent"`` (matched filter against the illuminator waveform,
        coherent over frequency; the demo's ``bistatic()``), ``"mvdr"`` or ``"music"``
        (adaptive, from the cross spectral matrix over ``adaptive_window`` frames).
    coherent_norm : str
        ``"none"``: the coherent map is ``|Z|^2`` (cells near the ship are weighted by the ship to
        cell path gain, so the noise floor of the map peaks in a ring around the ship).  ``"snr"``:
        ``|Z|^2`` divided by the expected noise power of the cell, ``sum_f |G(x, s)|^2 |r_f|^2 / norm``,
        the normalised matched filter whose floor is flat over the grid.
    mvdr_loading : float
        Diagonal loading relative to the trace for MVDR and MUSIC.
    music_rank : int
        Signal subspace rank for MUSIC.
    min_separation, max_peaks : float, int
        Peak extraction per frame.
    dispersion : DispersionModel | None
        Numerical dispersion of the FDTD scheme that produced the data (``dispersion.for_scenario``);
        ``None`` for real data.
    use_cuda : bool
        Evaluate the replicas (AIS refinement, map) on the GPU (CuPy, single precision; falls
        back to the CPU with a warning).  The ECA stays on the CPU.
    """

    win_s: float = 0.5
    hop_s: float = 0.25
    f_lo: float = 100.0
    f_hi: float | None = None
    bin_stride: int = 2
    c: float = 1495.0
    grid_dx: float = 2.0
    grid_margin: float = 5.0
    edge: float = 40.0
    reference: str = "data"
    refine: bool = True
    refine_half: float = 10.0
    refine_step: float = 1.0
    removal: str = "eca"
    eca_block_s: float = 0.5
    eca_halfwidth_s: float = 0.004
    eca_images: bool = True
    eca_doppler: bool = False
    track_smooth_s: float = 0.0
    ref_channels: str = "all"
    ref_mode: str = "beam"
    ref_floor: float = 0.1
    ref_taper_hz: float = 20.0
    track_filter: str = "kalman"
    track_accel: float = 0.05
    track_meas_sigma: float = 1.0
    track_gate_sigma: float = 5.0
    ais_pos_sigma: float = 8.0
    ais_sog_sigma: float = 0.2
    ais_cog_sigma: float = 0.035
    fallback_speed: float = 0.3
    map_norm: str = "max"
    patch_half: float = 1.0
    svd_rtol: float = 0.05
    adaptive_rank: int = 2
    adaptive_window: int = 3
    exclusion_radius: float = 25.0
    method: str = "coherent"
    coherent_norm: str = "none"
    mvdr_loading: float = 0.05
    music_rank: int = 3
    min_separation: float = 20.0
    max_peaks: int = 3
    dispersion: DispersionModel | None = None
    use_cuda: bool = False

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.removal not in REMOVALS:
            raise ValueError(f"removal must be one of {REMOVALS}, got {self.removal!r}")
        if self.reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, got {self.reference!r}")
        if self.ref_mode not in REF_MODES:
            raise ValueError(f"ref_mode must be one of {REF_MODES}, got {self.ref_mode!r}")
        if self.track_filter not in TRACK_FILTERS:
            raise ValueError(f"track_filter must be one of {TRACK_FILTERS}, got {self.track_filter!r}")
        if self.map_norm not in MAP_NORMS:
            raise ValueError(f"map_norm must be one of {MAP_NORMS}, got {self.map_norm!r}")

    @property
    def track_config(self) -> TrackConfig:
        return TrackConfig(accel=self.track_accel, meas_sigma=self.track_meas_sigma, gate_sigma=self.track_gate_sigma,
                           ais_pos_sigma=self.ais_pos_sigma, ais_sog_sigma=self.ais_sog_sigma, ais_cog_sigma=self.ais_cog_sigma)


# ----------------------------------------------------------------------
# STFT and replicas
# ----------------------------------------------------------------------

@dataclass
class Spectra:
    """STFT of all channels: ``X`` (N, F, K) complex, ``freqs`` (F,), ``times`` (K,), ``weights`` (N,) 0/1."""

    X: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    weights: np.ndarray
    fs: float
    win_s: float

    @property
    def wavenumbers(self) -> np.ndarray:
        return 2.0 * np.pi * self.freqs

    def copy(self) -> "Spectra":
        return Spectra(self.X.copy(), self.freqs, self.times, self.weights, self.fs, self.win_s)


def stft(traces: np.ndarray, layout: SensorLayout, fs: float, bc: BistaticConfig, f_max: float) -> Spectra:
    """Hann STFT of the band, after channel QC and per cable common mode removal."""
    pc = ProcessingConfig(win_s=bc.win_s, hop_s=bc.hop_s)
    w = channel_weights(traces, layout, pc)
    x = remove_common_mode(np.asarray(traces, dtype=np.float64), layout, w)
    nper, hop = int(round(bc.win_s * fs)), int(round(bc.hop_s * fs))
    starts = np.arange(0, x.shape[1] - nper + 1, hop)
    window = get_window("hann", nper)
    f = np.fft.rfftfreq(nper, 1.0 / fs)
    f_hi = bc.f_hi if bc.f_hi is not None else 0.95 * f_max
    band = np.flatnonzero((f >= bc.f_lo) & (f <= f_hi))[:: bc.bin_stride]
    X = np.empty((x.shape[0], len(band), len(starts)), dtype=np.complex128)
    for k, s0 in enumerate(starts):
        X[:, :, k] = np.fft.rfft(x[:, s0:s0 + nper] * window, axis=1)[:, band]
    X *= w[:, None, None]
    return Spectra(X, f[band], (starts + nper / 2.0) / fs, w, fs, bc.win_s)


def oracle_spectra(signal: np.ndarray, fs: float, sp: Spectra) -> np.ndarray:
    """STFT ``(F, K)`` of a known waveform on the frames and bins of *sp*."""
    nper, hop = int(round(sp.win_s * fs)), int(round((sp.times[1] - sp.times[0]) * fs)) if len(sp.times) > 1 else 1
    starts = np.rint(sp.times * fs - nper / 2.0).astype(int)
    window = get_window("hann", nper)
    f = np.fft.rfftfreq(nper, 1.0 / fs)
    band = np.array([int(np.argmin(np.abs(f - fk))) for fk in sp.freqs])
    out = np.empty((len(band), len(starts)), dtype=np.complex128)
    for k, s0 in enumerate(starts):
        out[:, k] = np.fft.rfft(signal[s0:s0 + nper] * window)[band]
    return out


class ReplicaModel:
    """Replicas ``a(x, f)`` (P, N) for points *x*: Green's function at the sample points, collapsed.

    With *dispersion* (a :class:`DispersionModel` of the FDTD scheme that produced the data) the
    wavenumber of every point to sample pair follows the scheme's direction dependent numerical
    value instead of ``w / c``; on a coarse grid (4 cells per wavelength) the numerical phase
    speed is 1 to 2 percent off and direction dependent, which shifts a bistatic fix by metres.
    Leave it ``None`` for real data.  *k* passed to the calls is the free field ``w / c``.

    With *device* (:func:`acoustic_sim.harbor.device.get_device`) on the GPU, the replicas are
    evaluated and returned as CuPy arrays in the device's precision (complex64); the callers
    work with ``model.xp`` and bring results back with :meth:`asnumpy`.  Measured on the fine
    bistatic dataset: the Green's function evaluation is 75 percent of the chain on the CPU.
    """

    def __init__(self, layout: SensorLayout, c: float, min_range: float = 1.0,
                 dispersion: DispersionModel | None = None, device: Device | None = None) -> None:
        self.layout, self.c, self.min_range = layout, c, min_range
        self.samples = layout.sample_points
        self.weights = None
        self.dispersion = dispersion
        self.device = device or CPU
        self.xp = self.device.xp
        self._samples_dev = self.xp.asarray(self.samples, dtype=self.device.fdtype)
        self._cache: tuple | None = None

    def asnumpy(self, a):
        return self.device.asnumpy(a)

    def _geometry(self, pts: np.ndarray, targets: np.ndarray) -> tuple:
        """Ranges (P, M) and, with dispersion, propagation angles for *pts* to *targets* (device
        arrays); the last geometry is cached because the replicas of every bin share it (the AIS
        refinement and the scan map call this once per bin for the same points)."""
        key = (pts.shape, targets.shape, pts.tobytes(), targets.tobytes())      # exact; a 17000 point grid is 270 kB, cheap to compare
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        xp = self.xp
        p = xp.asarray(pts, dtype=self.device.fdtype)
        t = self._samples_dev if targets is self.samples else xp.asarray(targets, dtype=self.device.fdtype)
        dx = p[:, None, 0] - t[None, :, 0]
        dy = p[:, None, 1] - t[None, :, 1]
        r = xp.maximum(xp.hypot(dx, dy), self.min_range)
        theta = xp.arctan2(xp.abs(dy), xp.abs(dx)) if self.dispersion is not None else None
        self._cache = (key, (r, theta))
        return r, theta

    def _kr(self, pts: np.ndarray, targets: np.ndarray, k: float):
        r, theta = self._geometry(pts, targets)
        if self.dispersion is None:
            return k * r
        f = k * self.c / (2.0 * np.pi)
        kk = replica_wavenumbers(self.dispersion, [f], self.c, theta, xp=self.xp, dtype=self.device.fdtype)[0]
        return kk * r

    def __call__(self, pts: np.ndarray, k: float):
        """Replicas (P, N) for *pts* (P, 2) at wavenumber *k* (a device array)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        a = self.layout.collapse(green2d(self._kr(pts, self.samples, k), xp=self.xp), xp=self.xp)
        if self.weights is not None:
            a = a * self.xp.asarray(self.weights, dtype=self.device.fdtype)[None, :]
        return a

    def path(self, pts: np.ndarray, src: np.ndarray, k: float):
        """Free field path factor ``H0^(2)(k |x - s|)`` from *src* to *pts*: (P,) for one source,
        (P, M) for *src* (M, 2) (a device array)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        src = np.asarray(src, dtype=float)
        out = green2d(self._kr(pts, np.atleast_2d(src), k), xp=self.xp)
        return out[:, 0] if src.ndim == 1 else out


replicas = ReplicaModel  # alias used in the docs


def build_grid(env: HarborEnvironment, bc: BistaticConfig) -> np.ndarray:
    x0, x1, y0, y1 = env.extent
    xs = np.arange(x0 + bc.edge, x1 - bc.edge + 1e-6, bc.grid_dx)
    ys = np.arange(y0 + bc.edge, y1 - bc.edge + 1e-6, bc.grid_dx)
    gx, gy = np.meshgrid(xs, ys)
    g = np.column_stack([gx.ravel(), gy.ravel()])
    return g[env.is_water(g, margin=bc.grid_margin)]


# ----------------------------------------------------------------------
# Ship position and reference
# ----------------------------------------------------------------------

@dataclass
class ShipTrack:
    id: str
    predicted: np.ndarray     # (K, 2) AIS
    refined: np.ndarray       # (K, 2)
    present: np.ndarray       # (K,) bool
    score: np.ndarray         # (K,) refinement peak / mean
    smoothed: np.ndarray | None = None    # (K, 2) track used by the ECA and the map (fit_track)
    velocity: np.ndarray | None = None    # (K, 2) its velocity [m/s]
    exact: bool = False                   # refined is the truth (ablations): never filtered

    @property
    def xy(self) -> np.ndarray:
        """Positions used by the processing: the smoothed track when fitted, else the refined one."""
        return self.refined if self.smoothed is None else self.smoothed


def fit_track(ship: ShipTrack, times: np.ndarray, ais: AISReports | None, bc: BistaticConfig) -> ShipTrack:
    """Fill ``ship.smoothed`` and ``ship.velocity`` per ``bc.track_filter`` (stage 1b).

    An exact track (``ship.exact``) is kept as it is.  ``"kalman"`` fuses the AIS reports of the
    ship (when *ais* is given) with the refined positions; ``"line"`` is the stage 0 local straight
    line fit over ``bc.track_smooth_s``; ``"none"`` keeps the refined positions.  The velocity of
    the last two is the time derivative of the fitted track.
    """
    times = np.asarray(times, dtype=float)
    if ship.exact or bc.track_filter == "none":
        xy = np.array(ship.refined, dtype=float, copy=True)
    elif bc.track_filter == "line":
        xy = smooth_track(times, ship.refined, ship.present, bc.track_smooth_s)
    else:
        present = ship.present if ship.present.any() else np.ones(len(times), bool)
        fit = kalman_track(times, ship.refined, present, ais, ship.id, bc.track_config)
        ship.smoothed, ship.velocity = fit.xy, fit.v
        return ship
    ship.smoothed = xy
    ship.velocity = np.gradient(xy, times, axis=0) if len(times) > 1 else np.zeros_like(xy)
    return ship


def _phat(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.abs(X), 1e-30)


def ship_positions(sp: Spectra, ais: AISReports, model: ReplicaModel, bc: BistaticConfig, vessel: str | None = None,
                   truth: np.ndarray | None = None) -> ShipTrack:
    """AIS prediction per frame, refined by a PHAT Bartlett search in a window around it.

    *truth* (K, 2) replaces the AIS prediction (for ablations: exact position, no refinement).
    """
    vid = vessel or ais.vessels[0]
    tr = ais.for_vessel(vid)
    pred = tr.predict(sp.times) if truth is None else np.asarray(truth, dtype=float)
    present = (sp.times >= tr.t[0] - 60.0) & (tr.age(sp.times) <= 60.0) if truth is None else np.ones(len(sp.times), bool)
    refined = pred.copy()
    score = np.ones(len(sp.times))
    if not bc.refine or truth is not None:
        return ShipTrack(vid, pred, refined, present, score, exact=truth is not None)
    h = np.arange(-bc.refine_half, bc.refine_half + 1e-6, bc.refine_step)
    off = np.array([(a, b) for a in h for b in h])
    bins = np.arange(0, len(sp.freqs), 3)
    xp = model.xp
    Yw = xp.asarray(_phat(sp.X[:, bins, :]) * sp.weights[:, None, None], dtype=model.device.cdtype)   # (N, n_bins, K)
    for k in range(len(sp.times)):
        if not present[k]:
            continue
        pts = pred[k] + off
        power = xp.zeros(len(pts), dtype=model.device.fdtype)
        for j, i in enumerate(bins):
            a = model(pts, sp.wavenumbers[i] / bc.c)
            power += xp.abs(a.conj() @ Yw[:, j, k]) ** 2 / xp.maximum(xp.sum(xp.abs(a) ** 2, axis=1), 1e-30)
        power = model.asnumpy(power)
        best = int(np.argmax(power))
        refined[k] = pts[best]
        score[k] = float(power[best] / max(power.mean(), 1e-30))
    return ShipTrack(vid, pred, refined, present, score)


def clutter_images(env: HarborEnvironment, s: np.ndarray, max_range: float = 400.0) -> np.ndarray:
    """Image positions of the source *s* in the nearest land along the four grid directions.

    The quays are pressure release walls, so the first order reflections of the ship are
    images of it in the shoreline and the pier face.  Returns ``(M, 2)`` (``M`` <= 4).
    """
    out = []
    for d in (np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0])):
        steps = np.arange(env.dx, max_range, env.dx)
        pts = s[None, :] + steps[:, None] * d[None, :]
        inside = (pts[:, 0] >= env.extent[0]) & (pts[:, 0] <= env.extent[1]) & (pts[:, 1] >= env.extent[2]) & (pts[:, 1] <= env.extent[3])
        pts = pts[inside]
        if len(pts) == 0:
            continue
        land = ~env.is_water(pts)
        if land.any():
            wall = pts[np.flatnonzero(land)[0]]
            out.append(2.0 * wall - s)
    return np.array(out).reshape(-1, 2)


def _fractional_delay(x: np.ndarray, delays: np.ndarray, fs: float) -> np.ndarray:
    """Delay each row of *x* (N, n) by ``delays`` [s] (negative advances), circularly via the FFT."""
    n = x.shape[1]
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return np.fft.irfft(np.fft.rfft(x, axis=1) * np.exp(-2j * np.pi * f[None, :] * delays[:, None]), n, axis=1)


UPSAMPLE = 8


def _resample_at(x: np.ndarray, positions: np.ndarray, factor: int = UPSAMPLE) -> np.ndarray:
    """*x* (n,) evaluated at fractional sample *positions*: FFT upsampling by *factor* then
    linear interpolation (plain linear interpolation attenuates the band top by several
    percent, modulated at the rate at which a moving ship's delay crosses sample boundaries,
    which caps the cancellation near 20 dB).  Zero outside ``[0, n)``."""
    n = len(x)
    up = resample(np.asarray(x, dtype=np.float64), n * factor)
    return np.interp(positions * factor, np.arange(n * factor, dtype=float), up, left=0.0, right=0.0)


def _apply_weights(seg: np.ndarray, freqs_w: np.ndarray, W: np.ndarray, fs: float) -> np.ndarray:
    """Filter each row of *seg* (N, n) by the complex frequency weights ``W`` (N, F_w) given on
    *freqs_w* (linear interpolation onto the segment's bins; zero outside the table)."""
    n = seg.shape[1]
    f = np.fft.rfftfreq(n, 1.0 / fs)
    Wi = np.empty((W.shape[0], len(f)), dtype=np.complex128)
    for i in range(W.shape[0]):
        Wi[i] = np.interp(f, freqs_w, W[i].real, left=0.0, right=0.0) + 1j * np.interp(f, freqs_w, W[i].imag, left=0.0, right=0.0)
    return np.fft.irfft(np.fft.rfft(seg, axis=1) * Wi, n, axis=1)


def _filtered_traces(x: np.ndarray, equaliser: tuple[np.ndarray, np.ndarray], nb: int, fs: float) -> np.ndarray:
    """*x* (N, n) filtered block by block (blocks of *nb* samples, weights per block, overlap of
    0.5 s on each side against the circular wrap) by the equaliser table ``(freqs_w, W (B, N, F_w))``."""
    freqs_w, W = equaliser
    n = x.shape[1]
    out = np.zeros_like(x)
    pad = int(0.5 * fs)
    for b in range(int(np.ceil(n / nb))):
        lo, hi = b * nb, min((b + 1) * nb, n)
        a, z = max(lo - pad, 0), min(hi + pad, n)
        out[:, lo:hi] = _apply_weights(x[:, a:z], freqs_w, W[min(b, W.shape[0] - 1)], fs)[:, lo - a:hi - a]
    return out


def make_reference(traces: np.ndarray, layout: SensorLayout, fs: float, ship_xy: np.ndarray, block_s: float,
                   bc: BistaticConfig, channels: np.ndarray | None = None, oracle_signal: np.ndarray | None = None,
                   ship_t: np.ndarray | None = None, equaliser: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Illuminator waveform aligned to the emission time, shape ``(n,)``.

    ``bc.reference == "oracle"``: the true ship waveform.  ``"data"``: delay and sum of the
    *channels* (default: every live channel) toward the ship position of each block of
    *block_s* seconds (*ship_xy* (K_b, 2), one row per block), each channel advanced by its
    range so the sum is at the emission time.  With *ship_t* (n, 2), the ship position at
    every sample, each channel is instead advanced by its per sample range (a moving ship;
    ``bc.eca_doppler``).  The data reference also contains the echo and the multipath, as in
    every passive radar; that is accepted.

    *equaliser* ``(freqs_w, W)`` with ``W`` (B, N_sel, F_w) complex applies a per block, per
    channel frequency weight (interpolated onto the FFT bins, zero outside the table) to each
    selected channel before the alignment; the channels are then **summed**, so the weights
    carry the normalisation (:func:`equaliser_weights`, :func:`snr_weights`).  Without it the
    live channels are averaged (stage 0).
    """
    n = traces.shape[1]
    if bc.reference == "oracle":
        if oracle_signal is None:
            raise ValueError("reference='oracle' needs oracle_signal")
        return np.asarray(oracle_signal, dtype=np.float64)[:n]
    idx = np.arange(layout.n_receivers) if channels is None else np.asarray(channels)
    x = np.asarray(traces, dtype=np.float64)[idx]
    rec = layout.receivers[idx]
    nb = int(round(block_s * fs))
    if ship_t is not None:
        if equaliser is not None:
            x = _filtered_traces(x, equaliser, nb, fs)
        j = np.arange(n, dtype=float)
        live = np.sum(np.abs(x) ** 2, axis=1) > 0
        out = np.zeros(n)
        for i in np.flatnonzero(live):
            adv = np.hypot(ship_t[:, 0] - rec[i, 0], ship_t[:, 1] - rec[i, 1]) / bc.c * fs
            out += _resample_at(x[i], j + adv)
        return out if equaliser is not None else out / max(live.sum(), 1)
    out = np.zeros(n)
    for b in range(len(ship_xy)):
        lo, hi = b * nb, min((b + 1) * nb, n)
        if hi <= lo:
            break
        # Advance each channel by its range to the ship (a margin around the block avoids wrap).
        pad = int(0.5 * fs)
        a, z = max(lo - pad, 0), min(hi + pad, n)
        delays = -np.hypot(*(rec - ship_xy[b]).T) / bc.c
        seg = x[:, a:z]
        if equaliser is not None:
            seg = _apply_weights(seg, equaliser[0], equaliser[1][min(b, equaliser[1].shape[0] - 1)], fs)
        seg = _fractional_delay(seg, delays, fs)
        if equaliser is not None:
            out[lo:hi] = seg[:, lo - a:hi - a].sum(axis=0)
        else:
            w = np.sum(np.abs(seg) ** 2, axis=1) > 0
            out[lo:hi] = seg[w, lo - a:hi - a].mean(axis=0)
    return out


def _band_taper(freqs: np.ndarray, f_lo: float, f_hi: float, taper: float) -> np.ndarray:
    """1 inside ``[f_lo, f_hi]``, cosine to 0 over *taper* Hz outside, 0 beyond."""
    t = np.ones(len(freqs))
    lo = (freqs < f_lo) & (freqs >= f_lo - taper)
    hi = (freqs > f_hi) & (freqs <= f_hi + taper)
    t[lo] = 0.5 * (1.0 + np.cos(np.pi * (f_lo - freqs[lo]) / taper))
    t[hi] = 0.5 * (1.0 + np.cos(np.pi * (freqs[hi] - f_hi) / taper))
    t[(freqs < f_lo - taper) | (freqs > f_hi + taper)] = 0.0
    return t


def equaliser_weights(model: ReplicaModel, layout: SensorLayout, channels: np.ndarray, ship_xy: np.ndarray,
                      bc: BistaticConfig, f_hi: float, df: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Strain equalisation table ``(freqs, W (B, N_sel, F))`` for :func:`make_reference` (``ref_mode="equalised"``).

    For every block position *ship_xy* (B, 2), channel ``i`` and frequency the weight is
    ``exp(-i k r_i) / a_i(f)``: the inverse of the modelled response ``a_i`` (the collapsed
    Green's function of *model*, i.e. the second difference along the cable for a strain
    channel, with the dispersion correction when the model has one) with the plain range delay
    ``exp(-i k r_i)`` factored out, because :func:`make_reference` applies that delay itself.
    ``|a_i|`` is floored at ``bc.ref_floor`` times the largest ``|a_j|`` over the selected
    channels at that frequency, the table is tapered to the band and divided by the number of
    channels (so the plain beam is the special case ``a_i = exp(-i k r_i)``).
    """
    channels = np.asarray(channels)
    freqs = np.arange(max(bc.f_lo - bc.ref_taper_hz, df), f_hi + bc.ref_taper_hz + 1e-9, df)
    taper = _band_taper(freqs, bc.f_lo, f_hi, bc.ref_taper_hz)
    rec = layout.receivers[channels]
    ship_xy = np.atleast_2d(ship_xy)
    r = np.hypot(rec[None, :, 0] - ship_xy[:, None, 0], rec[None, :, 1] - ship_xy[:, None, 1])    # (B, N_sel)
    saved = model.weights
    model.weights = None
    W = np.zeros((len(ship_xy), len(channels), len(freqs)), dtype=np.complex128)
    try:
        for j, f in enumerate(freqs):
            k = 2.0 * np.pi * f / bc.c
            a = model.asnumpy(model(ship_xy, k))[:, channels].astype(np.complex128)             # (B, N_sel)
            mag = np.abs(a)
            floor = bc.ref_floor * mag.max(axis=1, keepdims=True)
            a_eff = np.where(mag < floor, a * (floor / np.maximum(mag, 1e-300)), a)
            a_eff = np.where(mag < 1e-300, floor, a_eff)
            W[:, :, j] = taper[j] * np.exp(-1j * k * r) / a_eff
    finally:
        model.weights = saved
    return freqs, W / len(channels)


def mrc_weights(model: ReplicaModel, layout: SensorLayout, channels: np.ndarray, ship_xy: np.ndarray, sp_res: Spectra,
                t_blocks: np.ndarray, bc: BistaticConfig, window_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Model based maximum ratio combiner table ``(freqs, W (B, N_sel, F))`` (``ref_mode="mrc"``).

    With the channel model ``X_i = a_i S + n_i`` (``a_i`` the modelled response of
    :func:`equaliser_weights`, ``N_i`` the noise power of channel ``i`` and bin from the STFT
    *sp_res* of a first ECA pass residual, averaged over the frames within *window_s* of each
    block centre) the minimum variance unbiased estimate of ``S`` is
    ``sum_i conj(a_i) X_i / N_i / sum_i |a_i|^2 / N_i``.  As in :func:`equaliser_weights` the
    plain range delay is factored out for :func:`make_reference`; unlike the equaliser, a weak
    (strain null) channel gets a small weight instead of a floored large one.
    """
    channels = np.asarray(channels)
    freqs = np.concatenate([[max(bc.f_lo - bc.ref_taper_hz, 1.0)], sp_res.freqs, [sp_res.freqs[-1] + bc.ref_taper_hz]])
    rec = layout.receivers[channels]
    ship_xy = np.atleast_2d(ship_xy)
    r = np.hypot(rec[None, :, 0] - ship_xy[:, None, 0], rec[None, :, 1] - ship_xy[:, None, 1])    # (B, N_sel)
    Pr = np.abs(sp_res.X[channels]) ** 2
    saved = model.weights
    model.weights = None
    W = np.zeros((len(ship_xy), len(channels), len(freqs)), dtype=np.complex128)
    try:
        for j, f in enumerate(sp_res.freqs):
            k = 2.0 * np.pi * f / bc.c
            a = model.asnumpy(model(ship_xy, k))[:, channels].astype(np.complex128)             # (B, N_sel)
            for b, tb in enumerate(t_blocks):
                m = np.abs(sp_res.times - tb) <= window_s
                if not m.any():
                    m = np.ones(len(sp_res.times), bool)
                N = Pr[:, j, m].mean(axis=1)
                live = N > 0
                w = np.zeros(len(channels), dtype=np.complex128)
                w[live] = np.conj(a[b, live]) / N[live]
                norm = max(float(np.sum(np.abs(a[b, live]) ** 2 / N[live])), 1e-30)
                W[b, :, j + 1] = w * np.exp(-1j * k * r[b]) / norm
    finally:
        model.weights = saved
    return freqs, W


def snr_weights(sp_x: Spectra, sp_res: Spectra, channels: np.ndarray, t_blocks: np.ndarray, bc: BistaticConfig,
                window_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Maximum ratio combiner table ``(freqs, W (B, N_sel, F))`` (``ref_mode="snr"``).

    *sp_x* is the STFT of the direct path fitted by a first ECA pass (the traces minus the
    residual) and *sp_res* that of the residual; per block (frames within ``window_s`` of the
    block centre), channel and bin the signal power ``S`` and noise power ``N`` are their mean
    powers and the weight is ``sqrt(S) / N`` divided by ``sum_i S_i / N_i`` (unit gain on the
    signal), tapered to the band.  A dead channel has ``S = 0`` and weight 0.
    """
    channels = np.asarray(channels)
    freqs = np.concatenate([[max(bc.f_lo - bc.ref_taper_hz, 1.0)], sp_x.freqs, [sp_x.freqs[-1] + bc.ref_taper_hz]])
    W = np.zeros((len(t_blocks), len(channels), len(freqs)))
    Px = np.abs(sp_x.X[channels]) ** 2
    Pr = np.abs(sp_res.X[channels]) ** 2
    for b, tb in enumerate(t_blocks):
        m = np.abs(sp_x.times - tb) <= window_s
        if not m.any():
            m = np.ones(len(sp_x.times), bool)
        S = Px[:, :, m].mean(axis=2)
        N = np.maximum(Pr[:, :, m].mean(axis=2), 1e-30)
        w = np.sqrt(S) / N
        norm = np.maximum((S / N).sum(axis=0, keepdims=True), 1e-30)
        W[b, :, 1:-1] = w / norm
    return freqs, W.astype(np.complex128)


@dataclass
class ECAReport:
    """Per channel direct path cancellation measured against truth components."""

    removed_db: np.ndarray          # (N,) energy of the direct component removed [dB]
    echo_lost_db: np.ndarray        # (N,) energy of the scattered component removed [dB]


def _project_out(S: np.ndarray, segs: list[np.ndarray], loading: float = 1e-14) -> list[np.ndarray]:
    """Each segment minus its least squares projection on the columns of *S* (block, L).

    Normal equations with a Cholesky solve and one step of iterative refinement: ``S^T S`` is
    (L, L) with L of order 100 and costs one BLAS3 product, against a QR of the (block, L)
    matrix (85 ms per channel and block on the fine grid, 4000 of them per pass).  Adjacent lags
    of a band limited reference are nearly collinear and the normal equations square that
    condition number, so a relative diagonal *loading* keeps the factorisation stable and the
    refinement step (residual in full precision, corrected through the same factorisation)
    recovers the precision lost in the solve.  Measured on the fine stage 0a case against the
    QR: 2 dB short without the refinement, 0.1 dB with it, five times faster."""
    G = S.T @ S
    G[np.diag_indices_from(G)] += loading * max(np.trace(G) / G.shape[0], 1e-300)
    try:
        c, low = cho_factor(G, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        Q, _ = np.linalg.qr(S)
        return [seg - Q @ (Q.T @ seg) for seg in segs]
    out = []
    for seg in segs:
        x = cho_solve((c, low), S.T @ seg, check_finite=False)
        r = seg - S @ x
        x += cho_solve((c, low), S.T @ r, check_finite=False)
        out.append(seg - S @ x)
    return out


def eca_remove(
    traces: list[np.ndarray],
    reference: np.ndarray,
    layout: SensorLayout,
    fs: float,
    ship_xy: np.ndarray,
    images: list[np.ndarray],
    block_s: float,
    bc: BistaticConfig,
    ship_t: np.ndarray | None = None,
    scatterers: list[np.ndarray] | tuple = (),
    include_direct: bool = True,
) -> list[np.ndarray]:
    """Extensive cancellation (least squares projection on delayed copies of the reference).

    For every channel and block of *block_s* seconds, the channel is projected onto the
    reference delayed by the lags around the predicted direct arrival (``|s - r| / c``) and
    around each first order image arrival, each window ``bc.eca_halfwidth_s`` wide.  Only
    those lags are used, so an echo at another bistatic delay survives.  The same projection
    is applied to every array in *traces* (the measurement and, for diagnostics, its truth
    components), which is what makes the cancellation depth measurable.

    With *ship_t* (n, 2), the ship position at every sample, each regressor is the reference
    resampled at the **per sample** delay of its path (direct, or an image whose wall is fixed
    per block) plus a fixed lag offset, so the delay drift of a moving ship inside a block is
    followed instead of fitted by a time invariant filter.

    *scatterers* (per block, (M, 2) points) add the bistatic paths ship -> point -> channel
    (``ref_mode="iterated"`` fits the echo of the strongest map peak this way); with
    ``include_direct=False`` only the image and scatterer lags are used.
    """
    n = traces[0].shape[1]
    nb = int(round(block_s * fs))
    half = int(round(bc.eca_halfwidth_s * fs))
    outs = [np.array(x, dtype=np.float64, copy=True) for x in traces]
    rec = layout.receivers
    ref = np.asarray(reference, dtype=np.float64)
    j_all = np.arange(n, dtype=float)
    ref_up = resample(ref, n * UPSAMPLE) if ship_t is not None else None
    j_up = np.arange(n * UPSAMPLE, dtype=float)
    for b in range(len(ship_xy)):
        lo, hi = b * nb, min((b + 1) * nb, n)
        if hi <= lo:
            break
        s_b = np.asarray(ship_xy[b], dtype=float)
        kinds: list[tuple[str, np.ndarray]] = [("direct", s_b)] if include_direct else []
        if len(images) and len(images[b]):
            kinds += [("image", np.asarray(im, dtype=float)) for im in np.atleast_2d(images[b])]
        if len(scatterers) and len(scatterers[b]):
            kinds += [("scatter", np.asarray(p, dtype=float)) for p in np.atleast_2d(scatterers[b])]
        if not kinds:
            continue
        if ship_t is not None:
            a, z = max(lo - half, 0), min(hi + half, n)
            s_blk = ship_t[a:z]                                            # per sample ship position over the block plus margin

        def nominal_delay(kind: str, p: np.ndarray, r: np.ndarray) -> float:
            if kind == "scatter":
                return (np.hypot(*(s_b - p)) + np.hypot(*(p - r))) / bc.c * fs
            return np.hypot(*(p - r)) / bc.c * fs

        def sample_delay(kind: str, p: np.ndarray, r: np.ndarray) -> np.ndarray:
            if kind == "direct":
                return np.hypot(s_blk[:, 0] - r[0], s_blk[:, 1] - r[1]) / bc.c * fs
            if kind == "image":
                img = 2.0 * (0.5 * (p + s_b))[None, :] - s_blk                 # image = 2 wall - ship, the wall fixed per block
                return np.hypot(img[:, 0] - r[0], img[:, 1] - r[1]) / bc.c * fs
            return (np.hypot(s_blk[:, 0] - p[0], s_blk[:, 1] - p[1]) + np.hypot(*(p - r))) / bc.c * fs

        for i in range(rec.shape[0]):
            nominal = np.array([int(round(nominal_delay(kd, p, rec[i]))) for kd, p in kinds])
            lags = np.unique(np.concatenate([np.arange(L0 - half, L0 + half + 1) for L0 in nominal]))
            lags = lags[(lags >= 0) & (lags < hi)]
            if not include_direct:
                # An echo model must not contain the direct path (or an image): drop its lag window.
                excl = [int(round(nominal_delay("direct", s_b, rec[i])))]
                excl += [int(round(nominal_delay("image", np.asarray(im, float), rec[i]))) for im in (np.atleast_2d(images[b]) if len(images) and len(images[b]) else [])]
                lags = lags[np.all(np.abs(lags[:, None] - np.array(excl)[None, :]) > half, axis=1)]
            if len(lags) == 0:
                continue
            if ship_t is None:
                S = np.stack([np.concatenate([np.zeros(max(L - lo, 0)), ref[max(lo - L, 0):hi - L]]) if lo - L < 0 else ref[lo - L:hi - L]
                              for L in lags], axis=1)                    # (block, lags)
            else:
                # Each lag belongs to the path whose nominal delay is nearest; its column is the
                # reference at (t - d_path(t)) shifted by the fixed offset (L - nominal).
                owner = np.argmin(np.abs(lags[:, None] - nominal[None, :]), axis=1)
                bases = {}
                cols = []
                for L, p in zip(lags, owner):
                    if p not in bases:
                        d = sample_delay(kinds[p][0], kinds[p][1], rec[i])
                        bases[p] = np.interp((j_all[a:z] - d) * UPSAMPLE, j_up, ref_up, left=0.0, right=0.0)   # (z - a,)
                    off = int(L - nominal[p])
                    # column[j] = base[j - off] for j in [lo, hi); base index 0 is sample a.
                    src_lo, src_hi = lo - off - a, hi - off - a
                    col = np.zeros(hi - lo)
                    s0, s1 = max(src_lo, 0), min(src_hi, z - a)
                    if s1 > s0:
                        col[s0 - src_lo:s1 - src_lo] = bases[p][s0:s1]
                    cols.append(col)
                S = np.stack(cols, axis=1)
            cleaned = _project_out(S, [out[i, lo:hi] for out in outs])
            for out, seg in zip(outs, cleaned):
                out[i, lo:hi] = seg
    return outs


def reference_spectrum(sp: Spectra, ship: ShipTrack, model: ReplicaModel, bc: BistaticConfig,
                       oracle: np.ndarray | None = None, waveform: np.ndarray | None = None, fs: float | None = None) -> np.ndarray:
    """Illuminator spectrum ``r`` (F, K).

    From *waveform* (the time domain reference of :func:`make_reference`, aligned to the
    emission time) when given; else the oracle STFT; else a beam at the refined ship position
    per frame and bin (the analytic demo's estimate).
    """
    if waveform is not None:
        return oracle_spectra(waveform, fs, sp)
    if bc.reference == "oracle":
        if oracle is None:
            raise ValueError("reference='oracle' needs the oracle spectra")
        return oracle
    r = np.zeros((len(sp.freqs), len(sp.times)), dtype=np.complex128)
    for k in range(len(sp.times)):
        if not ship.present[k]:
            continue
        for i, kk in enumerate(sp.wavenumbers / bc.c):
            a = model.asnumpy(model(ship.xy[k][None], kk)[0])
            r[i, k] = np.vdot(a, sp.X[:, i, k]) / max(np.vdot(a, a).real, 1e-30)
    return r


# ----------------------------------------------------------------------
# Frequency domain direct path removal (kept for comparison with ECA)
# ----------------------------------------------------------------------

def remove_direct(sp: Spectra, ship: ShipTrack, model: ReplicaModel, bc: BistaticConfig) -> Spectra:
    """Project the ship out of every frame and bin (``"patch"`` and/or ``"adaptive"``).

    Measured on the coarse dataset (``examples/harbor_bistatic_detect.py --experiment removal``):
    the free field replica patch explains only about half of the direct field energy on the
    strain cable (the broadside leg mostly records quay reflections because of the cos^2
    directivity), so it cancels 5 to 9 dB; the data adaptive projection removes the echo as
    much as the direct path because in every snapshot both are the same waveform.  ECA
    (:func:`eca_remove`) is the default.
    """
    out = sp.copy()
    if bc.removal in ("none", "eca"):
        return out
    K, F = len(sp.times), len(sp.freqs)
    if "patch" in bc.removal:
        h = np.array([-bc.patch_half, 0.0, bc.patch_half])
        patch = np.array([(a, b) for a in h for b in h])
        for k in range(K):
            if not ship.present[k]:
                continue
            pts = ship.xy[k] + patch
            for i, kk in enumerate(sp.wavenumbers / bc.c):
                A = model.asnumpy(model(pts, kk)).T                    # (N, 9)
                u, s, _ = np.linalg.svd(A, full_matrices=False)
                u = u[:, s > bc.svd_rtol * s[0]]
                z = out.X[:, i, k]
                out.X[:, i, k] = z - u @ (u.conj().T @ z)
    if "adaptive" in bc.removal:
        src = out.X.copy()
        for k in range(K):
            if not ship.present[k]:
                continue
            lo, hi = max(0, k - bc.adaptive_window), min(K, k + bc.adaptive_window + 1)
            for i in range(F):
                D = src[:, i, lo:hi]
                u, _, _ = np.linalg.svd(D, full_matrices=False)
                u = u[:, : min(bc.adaptive_rank, u.shape[1])]
                z = out.X[:, i, k]
                out.X[:, i, k] = z - u @ (u.conj().T @ z)
    return out


# ----------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------

@dataclass
class BistaticRun:
    """Products of one pass."""

    sp: Spectra
    clean: Spectra
    ship: ShipTrack
    ref: np.ndarray                 # (F, K)
    grid: np.ndarray
    scan: ScanResult
    Z: np.ndarray | None = None     # (G, K) coherent complex map (method == "coherent")
    dets: DetectionSet | None = None
    eca: "ECAReport | None" = None  # cancellation depth per channel (needs truth components)
    reference: np.ndarray | None = None   # time domain reference waveform used


def _csm(X: np.ndarray, k: int, window: int, loading: float) -> np.ndarray:
    lo, hi = max(0, k - window), min(X.shape[1], k + window + 1)
    D = X[:, lo:hi]
    R = D @ D.conj().T / D.shape[1]
    return R + loading * np.trace(R).real / R.shape[0] * np.eye(R.shape[0])


def bistatic_map(clean: Spectra, ship: ShipTrack, ref: np.ndarray, grid: np.ndarray, model: ReplicaModel,
                 bc: BistaticConfig, method: str | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Per frame map ``B`` (G, K) for *method*; also the complex coherent map ``Z`` for ``"coherent"``.

    Every processor normalises each cell by its replica energy, so a map is a fraction of the
    data energy explained by that cell; ``bartlett_phat`` first divides each channel by its
    magnitude.  The coherent processor correlates ``a_f(x)^H y_f`` against the illuminator
    spectrum times the ship to cell path, ``r_f G(k |x - s|)``, and sums over frequency before
    taking the magnitude: the matched filter against the illuminator waveform.
    """
    method = method or bc.method
    G, K, F = len(grid), len(clean.times), len(clean.freqs)
    xp, dev = model.xp, model.device
    B = xp.zeros((G, K), dtype=dev.fdtype)
    Z = xp.zeros((G, K), dtype=dev.cdtype) if method == "coherent" else None
    W = xp.zeros((G, K), dtype=dev.fdtype) if method == "coherent" else None
    present = xp.asarray(ship.present.astype(dev.fdtype))
    ship_xy = ship.xy
    for i, kk in enumerate(clean.wavenumbers / bc.c):
        A = model(grid, kk)                                            # (G, N)
        norm = xp.maximum(xp.sum(xp.abs(A) ** 2, axis=1), 1e-30)
        Xi = clean.X[:, i, :]                                          # (N, K)
        if method in ("bartlett", "bartlett_phat"):
            Y = _phat(Xi) * clean.weights[:, None] if method == "bartlett_phat" else Xi
            B += xp.abs(A.conj() @ xp.asarray(Y, dtype=dev.cdtype)) ** 2 / norm[:, None]
        elif method == "coherent":
            Y = A.conj() @ xp.asarray(Xi, dtype=dev.cdtype)            # (G, K)
            path = model.path(grid, ship_xy, kk)                       # (G, K): ship of every frame to every cell
            r = xp.asarray(ref[i], dtype=dev.cdtype) * present         # (K,)
            Z += Y * xp.conj(path) * xp.conj(r)[None, :] / norm[:, None]
            W += xp.abs(path) ** 2 * (xp.abs(r) ** 2)[None, :] / norm[:, None]
        elif method in ("mvdr", "music"):
            A_h = model.asnumpy(A)
            norm_h = model.asnumpy(norm)
            Bk = np.zeros((G, K))
            for k in range(K):
                R = _csm(Xi, k, bc.adaptive_window, bc.mvdr_loading)
                if method == "mvdr":
                    Ri = np.linalg.inv(R)
                    q = np.einsum("gn,nm,gm->g", A_h.conj(), Ri, A_h).real
                    Bk[:, k] += norm_h / np.maximum(q, 1e-30)
                else:
                    lam, vec = np.linalg.eigh(R)
                    En = vec[:, : max(R.shape[0] - bc.music_rank, 1)]
                    q = np.sum(np.abs(A_h.conj() @ En) ** 2, axis=1)
                    Bk[:, k] += norm_h / np.maximum(q, 1e-30)
            B += xp.asarray(Bk, dtype=dev.fdtype)
    if method == "coherent":
        B = xp.abs(Z) ** 2
        if bc.coherent_norm == "snr":
            B /= xp.maximum(W, 1e-30)
    B = model.asnumpy(B).astype(np.float64)
    Z = model.asnumpy(Z).astype(np.complex128) if Z is not None else None
    if bc.map_norm == "max":
        B /= max(B.max(), 1e-30)
    return B, Z


def exclusion(grid: np.ndarray, ship: ShipTrack, radius: float) -> np.ndarray:
    mask = np.zeros((len(grid), len(ship.present)), dtype=bool)
    for k in np.flatnonzero(ship.present):
        mask[:, k] = np.hypot(*(grid - ship.xy[k]).T) < radius
    return mask


def extract_peaks(B: np.ndarray, grid: np.ndarray, times: np.ndarray, bc: BistaticConfig, floor: float = 0.0) -> DetectionSet:
    """Greedy non-maximum suppression per frame; scores are the map values."""
    frame, xy, score, rank = [], [], [], []
    for k in range(B.shape[1]):
        for r, g in enumerate(_nms(B[:, k], grid, bc.min_separation, bc.max_peaks, floor)):
            frame.append(k)
            xy.append(grid[g])
            score.append(B[g, k])
            rank.append(r)
    if not frame:
        e = np.empty((0, 2))
        return DetectionSet(np.empty(0, int), np.empty(0), e, e, np.empty(0), np.empty(0, int), [])
    kk = np.array(frame)
    xy = np.array(xy)
    return DetectionSet(kk, times[kk], xy, xy.copy(), np.array(score), np.array(rank), [])


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def smooth_track(times: np.ndarray, xy: np.ndarray, present: np.ndarray, window_s: float) -> np.ndarray:
    """Local straight line fit of *xy* (K, 2) over ``|t - t_k| <= window_s / 2`` per frame (frames
    with *present* False are left out of the fits and returned unchanged).  ``window_s <= 0``
    returns *xy*."""
    if window_s <= 0:
        return xy
    out = np.array(xy, dtype=float, copy=True)
    for k in np.flatnonzero(present):
        m = present & (np.abs(times - times[k]) <= 0.5 * window_s)
        if m.sum() < 3:
            continue
        A = np.column_stack([np.ones(m.sum()), times[m] - times[k]])
        coef, *_ = np.linalg.lstsq(A, xy[m], rcond=None)
        out[k] = coef[0]
    return out


def eca_stage(
    traces: np.ndarray,
    layout: SensorLayout,
    fs: float,
    env: HarborEnvironment,
    sp: Spectra,
    ship: ShipTrack,
    bc: BistaticConfig,
    oracle_signal: np.ndarray | None = None,
    components: list[np.ndarray] | tuple = (),
    ais: AISReports | None = None,
    grid: np.ndarray | None = None,
    model: ReplicaModel | None = None,
    f_max: float | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Time domain direct path removal of :func:`run_bistatic`: track, reference, image lags, ECA.

    The track of the ship (``ship.smoothed``, fitted here by :func:`fit_track` from *ais* and
    the refined positions if the caller has not) gives the ship position of every ECA block
    and, with ``bc.eca_doppler``, of every sample; blocks whose smoothed speed is below
    ``bc.fallback_speed`` keep a fixed position (fixed lags).  The reference follows
    ``bc.reference`` and ``bc.ref_mode`` (the ``"iterated"`` mode forms a coherent map and needs
    *grid*, *model* and *f_max*; they are built when missing).  The same projection is applied
    to *traces* and to each array in *components*.  Returns the cleaned arrays
    (``[traces] + components``) and the reference waveform.
    """
    if ship.smoothed is None:
        ship = fit_track(ship, sp.times, ais, bc)
    xy = ship.xy
    n = traces.shape[1]
    nb = int(round(bc.eca_block_s * fs))
    n_blocks = int(np.ceil(n / nb))
    t_blocks = (np.arange(n_blocks) + 0.5) * bc.eca_block_s
    ship_b = np.column_stack([np.interp(t_blocks, sp.times, xy[:, 0]), np.interp(t_blocks, sp.times, xy[:, 1])])
    ship_t = None
    if bc.eca_doppler:
        t_s = np.arange(n) / fs
        ship_t = np.column_stack([np.interp(t_s, sp.times, xy[:, 0]), np.interp(t_s, sp.times, xy[:, 1])])
        speed_b = np.interp(t_blocks, sp.times, np.hypot(ship.velocity[:, 0], ship.velocity[:, 1]))
        slow = speed_b < bc.fallback_speed
        if slow.all():
            ship_t = None                                        # fixed lags everywhere
        else:
            for b in np.flatnonzero(slow):                       # fixed lags in the slow blocks
                ship_t[b * nb:min((b + 1) * nb, n)] = ship_b[b]
    if bc.ref_channels == "all":
        chans = np.flatnonzero(sp.weights > 0)
    else:
        sl = layout.slices[bc.ref_channels]
        chans = np.arange(sl.start, sl.stop)[sp.weights[sl] > 0]
    images = [clutter_images(env, sxy) for sxy in ship_b] if bc.eca_images else []
    f_hi = bc.f_hi if bc.f_hi is not None else (0.95 * f_max if f_max is not None else float(sp.freqs[-1]))

    def beam(x: np.ndarray, equaliser=None) -> np.ndarray:
        return make_reference(x, layout, fs, ship_b, bc.eca_block_s, bc, chans, oracle_signal, ship_t=ship_t, equaliser=equaliser)

    def eca(arrays: list[np.ndarray], waveform: np.ndarray) -> list[np.ndarray]:
        return eca_remove(arrays, waveform, layout, fs, ship_b, images, bc.eca_block_s, bc, ship_t=ship_t)

    if bc.reference == "oracle" or bc.ref_mode == "beam":
        waveform = beam(traces)
    elif bc.ref_mode == "equalised":
        model = model or ReplicaModel(layout, bc.c, dispersion=bc.dispersion, device=get_device(bc.use_cuda))
        waveform = beam(traces, equaliser_weights(model, layout, chans, ship_b, bc, f_hi))
    elif bc.ref_mode == "snr":
        residual = eca([traces], beam(traces))[0]
        sp_fit = stft(np.asarray(traces, dtype=np.float64) - residual, layout, fs, bc, f_hi / 0.95)
        sp_res = stft(residual, layout, fs, bc, f_hi / 0.95)
        waveform = beam(traces, snr_weights(sp_fit, sp_res, chans, t_blocks, bc))
    elif bc.ref_mode == "mrc":
        model = model or ReplicaModel(layout, bc.c, dispersion=bc.dispersion, device=get_device(bc.use_cuda))
        residual = eca([traces], beam(traces))[0]
        sp_res = stft(residual, layout, fs, bc, f_hi / 0.95)
        waveform = beam(traces, mrc_weights(model, layout, chans, ship_b, sp_res, t_blocks, bc))
    else:                                                        # "iterated"
        model = model or ReplicaModel(layout, bc.c, dispersion=bc.dispersion, device=get_device(bc.use_cuda))
        model.weights = sp.weights
        grid = grid if grid is not None else build_grid(env, bc)
        first = beam(traces)
        clean = stft(eca([traces], first)[0], layout, fs, bc, f_hi / 0.95)
        clean.weights = sp.weights
        ref = reference_spectrum(sp, ship, model, bc, waveform=first, fs=fs)
        B, _ = bistatic_map(clean, ship, ref, grid, model, bc, method="coherent")
        B[exclusion(grid, ship, bc.exclusion_radius)] = 0.0
        k_b = np.clip(np.searchsorted(sp.times, t_blocks), 0, len(sp.times) - 1)
        peaks = [grid[np.argmax(B[:, k])][None, :] for k in k_b]       # strongest cell of the frame nearest each block
        echo = np.asarray(traces, dtype=np.float64) - eca_remove([traces], first, layout, fs, ship_b, [], bc.eca_block_s, bc,
                                                                 ship_t=ship_t, scatterers=peaks, include_direct=False)[0]
        waveform = beam(np.asarray(traces, dtype=np.float64) - echo)
    outs = eca([traces] + list(components), waveform)
    return outs, waveform


@dataclass
class Truth:
    """Optional truth for ablations and diagnostics."""

    ship_xy: np.ndarray | None = None          # exact ship positions at the frame times (K, 2)
    components: dict[str, np.ndarray] | None = None   # truth components to pass through the same cancellation


def run_bistatic(
    traces: np.ndarray,
    layout: SensorLayout,
    fs: float,
    env: HarborEnvironment,
    ais: AISReports,
    bc: BistaticConfig,
    f_max: float,
    grid: np.ndarray | None = None,
    oracle_signal: np.ndarray | None = None,
    truth: Truth | None = None,
    floor: float = 0.0,
) -> BistaticRun:
    """The whole chain on one set of traces.

    *oracle_signal* is the true illuminator waveform at the emission time (for
    ``reference="oracle"``); ``truth.ship_xy`` bypasses AIS and refinement; ``truth.components``
    go through the same ECA projection and their residual energies are reported in
    ``run.eca`` (cancellation depth and echo loss per channel).  Peaks above *floor* are
    returned (threshold later).
    """
    truth = truth or Truth()
    sp = stft(traces, layout, fs, bc, f_max)
    model = ReplicaModel(layout, bc.c, dispersion=bc.dispersion, device=get_device(bc.use_cuda))
    model.weights = sp.weights
    ship = ship_positions(sp, ais, model, bc, truth=truth.ship_xy)
    ship = fit_track(ship, sp.times, ais, bc)
    grid = grid if grid is not None else build_grid(env, bc)

    report = None
    waveform = None
    if bc.removal == "eca":
        comps = list((truth.components or {}).values())
        outs, waveform = eca_stage(traces, layout, fs, env, sp, ship, bc, oracle_signal, comps, ais=ais, grid=grid, model=model, f_max=f_max)
        clean_traces = outs[0]
        if comps:
            names = list(truth.components)
            e0 = {k: np.sum(np.asarray(v, dtype=np.float64) ** 2, axis=1) for k, v in truth.components.items()}
            e1 = {k: np.sum(o ** 2, axis=1) for k, o in zip(names, outs[1:])}
            db = {k: 10 * np.log10(np.maximum(e0[k], 1e-30) / np.maximum(e1[k], 1e-30)) for k in names}
            direct = next((k for k in names if k.startswith("direct")), names[0])
            echo = next((k for k in names if k.startswith("scattered")), None)
            report = ECAReport(db[direct], db[echo] if echo else np.full(layout.n_receivers, np.nan))
        clean = stft(clean_traces, layout, fs, bc, f_max)
        clean.weights = sp.weights
        ref = reference_spectrum(sp, ship, model, bc, waveform=waveform, fs=fs)
    else:
        oracle = oracle_spectra(oracle_signal, fs, sp) if oracle_signal is not None else None
        ref = reference_spectrum(sp, ship, model, bc, oracle=oracle)
        clean = remove_direct(sp, ship, model, bc)

    B, Z = bistatic_map(clean, ship, ref, grid, model, bc)
    B[exclusion(grid, ship, bc.exclusion_radius)] = 0.0
    scan = ScanResult(grid, sp.times, B, {}, np.zeros(len(sp.freqs), int))
    dets = extract_peaks(B, grid, sp.times, bc, floor)
    return BistaticRun(sp, clean, ship, ref, grid, scan, Z, dets, report, waveform)


# ----------------------------------------------------------------------
# Motion compensated (synthetic aperture) integration
# ----------------------------------------------------------------------

def motion_integrate(Z: np.ndarray, grid: np.ndarray, times: np.ndarray, speeds: np.ndarray,
                     coherent: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the per frame maps along constant velocity hypotheses.

    For every velocity ``v`` in *speeds* (V, 2) the frame maps are shifted to the central
    time, ``Z_k(x + v (t_k - t_mid))``, and summed (coherently if *coherent*, else their
    magnitudes squared).  Returns the best map ``(G,)`` over velocities and the per velocity
    peak values ``(V,)``.  The ship moves during the window, so the bistatic ellipses sweep
    across a target; integrating along the true velocity is a synthetic aperture.
    """
    x, y = grid[:, 0], grid[:, 1]
    xs, ys = np.unique(x), np.unique(y)
    dx = xs[1] - xs[0]
    ix = np.rint((x - xs[0]) / dx).astype(int)
    iy = np.rint((y - ys[0]) / dx).astype(int)
    shape = (len(ys), len(xs))
    t_mid = times[len(times) // 2]
    best, peaks = None, np.zeros(len(speeds))
    for v_i, v in enumerate(speeds):
        acc = np.zeros(len(grid), dtype=np.complex128 if coherent else float)
        for k, t in enumerate(times):
            img = np.zeros(shape, dtype=Z.dtype)
            img[iy, ix] = Z[:, k]
            shift = v * (t - t_mid)
            cx = (x + shift[0] - xs[0]) / dx
            cy = (y + shift[1] - ys[0]) / dx
            re = map_coordinates(img.real, [cy, cx], order=1, mode="constant")
            im = map_coordinates(img.imag, [cy, cx], order=1, mode="constant") if np.iscomplexobj(Z) else 0.0
            val = re + 1j * im if np.iscomplexobj(Z) else re
            acc += val if coherent else np.abs(val) ** 2
        m = np.abs(acc) ** 2 if coherent else acc
        peaks[v_i] = m.max()
        if best is None or peaks[v_i] > best.max():
            best = m
    return best, peaks

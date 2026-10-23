"""Passive bistatic detection of a silent body using an AIS ship as the illuminator.

The analytic prototype is ``examples/harbor/90_history/harbor_mfp_demo.py``; this module is its port to the
FDTD harbor data (strain mode DAS, a real scattering body, quay reflections).

Chain for one dataset
---------------------
1. :func:`stft` of all channels, channel QC and per cable common mode removal (from
   ``detection.py``), band ``f_lo`` to ``f_hi``.
2. :func:`ship_positions`: AIS prediction per frame and a fine Bartlett refinement with
   per channel PHAT normalisation (the AIS error of 8 m is several wavelengths here).
3. :func:`reference_spectrum`: the illuminator spectrum per frame and bin, either from the
   data (beam steered at the refined ship position) or given from the truth (``"true_waveform"``).
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

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import get_window, resample

from acoustic_sim.contract import ChartEnvironment, Propagator, Provenance
from acoustic_sim.water.ais import AISReports
from acoustic_sim.water.profile import PROFILE
from acoustic_sim.water.detection import DetectionSet, ScanResult, _nms, channel_weights, remove_common_mode
from acoustic_sim.water.detection import ProcessingConfig
from acoustic_sim.water.device import get_device
from acoustic_sim.water.motion import (  # noqa: F401  (moved; re-exported for the old path)
    _motion_integrate_reference,
    _motion_integrate_vectorized,
    motion_integrate,
)
from acoustic_sim.water.propagation import FreeFieldPropagator, green2d, green2d_bins, resolve
from acoustic_sim.water.sensors import SensorLayout

METHODS = ("bartlett", "bartlett_phat", "coherent", "mvdr", "music")
REMOVALS = ("none", "eca", "patch", "adaptive", "patch+adaptive")
REFERENCES = ("data", "true_waveform")
REF_MODES = ("beam", "weighted", "weighted_cable", "hydrophones", "iterated", "joint")
IMPLS = ("reference", "vectorized")
MAP_SCALES = ("max", "none")
ECA_GRAMS = ("matmul", "prefix")
REMOVAL_ORDERS = ("sequential", "joint")


@dataclass
class BistaticConfig:
    """Parameters of the bistatic chain.

    Attributes
    ----------
    win_s, hop_s : float
        STFT window and hop [s].
    f_lo, f_hi : float
        Processing band [Hz], a choice of the processing (the defaults are the band of the 0.5 m
        harbor datasets, 100 to 760 Hz); nothing reads it from the data.
    bin_stride : int
        Use every n-th bin of the band.
    c : float
        Replica sound speed [m/s].
    grid_dx, grid_margin : float
        Scan grid spacing and distance from land [m].
    edge : float
        Keep the scan grid this far [m] from the domain boundary (the sponge).
    reference : str
        ``"data"`` (beam at the refined ship position) or ``"true_waveform"`` (the ship's radiated
        waveform taken from the simulation).
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
        How the data reference is estimated from the traces (:func:`estimate_reference`; ``reference == "data"``
        with per sample delays only).  ``"beam"``: the delay and sum of the channels toward the tracked ship.
        ``"weighted"``: each channel and frequency weighted by its modelled response to the ship (the chain's
        propagator at the tracked position: the Green's function for a pressure channel, its second difference
        along the cable for a strain channel) over its noise power, estimated from the residual of a first
        pass (the beam and the ECA): the maximum ratio combination.  ``"weighted_cable"``: the same on the cable
        channels only.  ``"hydrophones"``: the same on the hydrophones only.  ``"iterated"``: ``"weighted"``,
        the ECA, the modelled echo of every body given (the echo tracks) subtracted from the traces, then the
        beam again.  ``"joint"``: the waveform and one scale per channel solved together by alternating least
        squares over every block (the direct path of every channel a scaled copy of one waveform at the
        track's delays), from the beam, ``ref_iterations`` times.
    ref_iterations : int
        Alternations of ``"joint"``.
    ref_psd_s : float
        Segment length [s] of the noise power estimate of the weighted modes (Welch).
    removal_order : str
        How the direct paths of several illuminators are removed (:func:`remove_direct_paths`).
        ``"sequential"``: one after the other, each reference estimated on the traces already cleaned of the
        previous ones and each ECA on its own.  ``"joint"``: every reference estimated on the traces minus the
        other illuminators' current fits, ``joint_passes`` times over all of them (the first pass is the sequential
        one), then one least squares solve per block over the delayed copies of every reference at once.
    joint_passes : int
        Passes of ``"joint"`` over the references (0: one solve on the beams of the raw traces).
    joint_gram : str
        How the one least squares solve of ``"joint"`` forms its normal equations: ``"matmul"`` (default) the matrix
        product of the delayed copies; ``"prefix"`` from prefix sums of the sliding products of the references'
        resampled copies, within one reference and between every two (each column a shift of one of them; equal to
        rounding).
    ref_noise : str
        Where the weighted and joint reference modes take each channel's noise power from: ``"first_eca"`` (default)
        the residual of a first pass (the beam and its ECA) on the same traces; ``"previous"`` the residual the
        streaming chain left in its previous step (every illuminator removed), when that illuminator was removed then
        (an illuminator new in this step still has its first pass).
    removal_band : list | None
        ``[f1, f2]`` [Hz]: the streaming chain removes the direct paths on the traces band limited to it and decimated
        by the largest integer that divides the ECA block with the decimated Nyquist at least 1.25 ``f2``, then brings
        the cleaned traces and the references back to the recording's rate (what lies outside the band is not kept:
        the maps read ``f_lo`` to ``f_hi`` only).  ``None`` (default): the full band at the recording's rate.
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
    eca_gram : str
        How the ECA forms its normal equations ``S^T S`` with per sample delays (``eca_doppler``) and no image
        lags: ``"matmul"`` (default) the matrix product; ``"prefix"`` the sliding correlations of the one
        resampled reference every column is a shift of, from prefix sums (about 20 times fewer operations,
        equal to rounding; the real-time chain uses it).
    eca_chunk : int
        Channels whose normal equations the vectorized ECA forms and solves together (memory against
        kernel launches).
    eca_precision : str
        ``"float64"`` (default): the vectorized ECA in double precision with the loaded normal equations.
        ``"mixed"``: the per sample work (the reference resampled at the track's delays, the delayed copies, the
        projection applied to every channel) in single precision, and the least squares fit of every channel and block
        by a QR of its delayed copies in double precision, without loading (:func:`_project_out_qr`).
    map_scale : str
        ``"max"`` (default): the maps of a run are divided by their largest value (over all frames), so a
        map is relative to the run's peak.  ``"none"``: the maps are returned as computed (with
        ``coherent_norm="snr"`` in units of the noise power), so one threshold applies across frames,
        runs and datasets (the scoreboard, ``water/scoreboard.py``).
    mvdr_loading : float
        Diagonal loading relative to the trace for MVDR and MUSIC.
    music_rank : int
        Signal subspace rank for MUSIC.
    min_separation, max_peaks : float, int
        Peak extraction per frame.
    propagator : Propagator | None
        Replica model (:class:`acoustic_sim.contract.Propagator`).  ``None`` is the free
        field ``FreeFieldPropagator(c)`` (nothing beyond the public chart); ``water.diagnostics.SimulatorAwarePropagator.from_truth`` follows the
        numerical dispersion of the FDTD scheme that produced the data (simulator knowledge, flagged in
        the run's provenance).
    impl : str
        ``"vectorized"`` (default): the replica tables are built once per bin and reused across
        frames, the coherent, MVDR and MUSIC maps run over all frames at once, the ECA builds the
        delayed reference matrices of all channels of a block from strided views and solves the
        normal equations in a batch, and the refinement and the motion integration run over frames
        and velocities at once.  ``"reference"``: the original per channel / per frame loops.  Same
        mathematics; on the CPU the two agree to the last bit (ECA, refinement, coherent map).
    use_cuda : bool
        Run the vectorized chain on the GPU (CuPy, double precision; falls back to NumPy without a GPU).
    """

    win_s: float = 0.5
    hop_s: float = 0.25
    f_lo: float = 100.0
    f_hi: float = 760.0
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
    ref_iterations: int = 3
    ref_psd_s: float = 0.5
    removal_order: str = "sequential"
    joint_passes: int = 2
    joint_gram: str = "matmul"
    ref_noise: str = "first_eca"
    removal_band: list | None = None
    patch_half: float = 1.0
    svd_rtol: float = 0.05
    adaptive_rank: int = 2
    adaptive_window: int = 3
    exclusion_radius: float = 25.0
    method: str = "coherent"
    coherent_norm: str = "none"
    map_scale: str = "max"
    eca_gram: str = "matmul"
    eca_chunk: int = 32
    eca_precision: str = "float64"
    mvdr_loading: float = 0.05
    music_rank: int = 3
    min_separation: float = 20.0
    max_peaks: int = 3
    propagator: Propagator | None = field(default=None, repr=False)
    impl: str = "vectorized"
    use_cuda: bool = False

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.removal not in REMOVALS:
            raise ValueError(f"removal must be one of {REMOVALS}, got {self.removal!r}")
        if self.reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, got {self.reference!r}")
        if self.impl not in IMPLS:
            raise ValueError(f"impl must be one of {IMPLS}, got {self.impl!r}")
        if self.eca_precision not in ("float64", "mixed"):
            raise ValueError(f"eca_precision must be 'float64' or 'mixed', got {self.eca_precision!r}")
        if self.removal_order not in REMOVAL_ORDERS:
            raise ValueError(f"removal_order must be one of {REMOVAL_ORDERS}, got {self.removal_order!r}")
        if self.ref_mode not in REF_MODES:
            raise ValueError(f"ref_mode must be one of {REF_MODES}, got {self.ref_mode!r}")
        if self.eca_gram not in ECA_GRAMS:
            raise ValueError(f"eca_gram must be one of {ECA_GRAMS}, got {self.eca_gram!r}")
        if self.map_scale not in MAP_SCALES:
            raise ValueError(f"map_scale must be one of {MAP_SCALES}, got {self.map_scale!r}")


def _xp(bc: BistaticConfig):
    """Array module of the vectorized chain for *bc* (CuPy with ``use_cuda`` and a GPU, else NumPy)."""
    return get_device(bc.use_cuda and bc.impl == "vectorized").xp


def _asnumpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _module_of(a):
    """NumPy or CuPy, whichever *a* belongs to."""
    if isinstance(a, np.ndarray):
        return np
    import cupy

    return cupy


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


def stft(traces: np.ndarray, layout: SensorLayout, fs: float, bc: BistaticConfig) -> Spectra:
    """Hann STFT of the band, after channel QC and per cable common mode removal."""
    pc = ProcessingConfig(win_s=bc.win_s, hop_s=bc.hop_s)
    w = channel_weights(traces, layout, pc)
    x = remove_common_mode(np.asarray(traces, dtype=np.float64), layout, w)
    nper, hop = int(round(bc.win_s * fs)), int(round(bc.hop_s * fs))
    starts = np.arange(0, x.shape[1] - nper + 1, hop)
    window = get_window("hann", nper)
    f = np.fft.rfftfreq(nper, 1.0 / fs)
    band = np.flatnonzero((f >= bc.f_lo) & (f <= bc.f_hi))[:: bc.bin_stride]
    X = np.empty((x.shape[0], len(band), len(starts)), dtype=np.complex128)
    for k, s0 in enumerate(starts):
        X[:, :, k] = np.fft.rfft(x[:, s0:s0 + nper] * window, axis=1)[:, band]
    X *= w[:, None, None]
    return Spectra(X, f[band], (starts + nper / 2.0) / fs, w, fs, bc.win_s)


def waveform_spectra(signal: np.ndarray, fs: float, sp: Spectra) -> np.ndarray:
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

    The Green's function comes from *propagator* (:class:`acoustic_sim.contract.Propagator`; ``None``
    is the free field at *c*).  An analytic propagator that ``needs_angles`` (the simulator
    aware one of ``water/diagnostics.py``) gives every point to sample pair the FDTD scheme's
    direction dependent numerical wavenumber instead of ``w / c``; on a coarse grid (4 cells per
    wavelength) the numerical phase speed is 1 to 2 percent off, which shifts a bistatic fix by
    metres.  A non analytic propagator is evaluated through its ``greens``.  *k* passed to the calls
    is the free field ``w / c``.

    *xp* is the array module (NumPy, or CuPy for the GPU chain): the replicas come back as arrays
    of that module; points and sources are always given as host arrays.  :meth:`many` evaluates
    several wavenumbers on one geometry and :meth:`paths` several sources at one wavenumber, the
    building blocks of the vectorized chain.
    """

    def __init__(self, layout: SensorLayout, c: float, min_range: float = 1.0,
                 propagator: Propagator | None = None, xp=np) -> None:
        self.layout, self.c, self.min_range = layout, c, min_range
        self.samples = layout.sample_points
        self.weights = None
        self.propagator = resolve(propagator, c)
        self.xp = xp
        self._cache: dict[tuple, tuple] = {}          # geometry key -> (r, theta); a few recent entries
        self._weights_cache: tuple | None = None

    def _geometry(self, pts: np.ndarray, targets: np.ndarray) -> tuple:
        """Ranges (P, M) and, with dispersion, propagation angles for *pts* to *targets* (arrays of
        ``self.xp``); the last geometry is cached because the replicas of every bin share it (the
        AIS refinement and the scan map call this once per bin for the same points)."""
        key = (pts.shape, targets.shape, pts.tobytes(), targets.tobytes())      # exact; a 17000 point grid is 270 kB, cheap to compare
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        xp = self.xp
        pts_d, tgt_d = xp.asarray(pts), xp.asarray(targets)
        dx = pts_d[:, None, 0] - tgt_d[None, :, 0]
        dy = pts_d[:, None, 1] - tgt_d[None, :, 1]
        r = xp.maximum(xp.hypot(dx, dy), self.min_range)
        theta = xp.arctan2(xp.abs(dy), xp.abs(dx)) if self.propagator.needs_angles else None
        if len(self._cache) >= 4:                       # the scan grid geometry must survive the per frame path calls
            del self._cache[next(iter(self._cache))]
        self._cache[key] = (r, theta)
        return r, theta

    def _kr_many(self, pts: np.ndarray, targets: np.ndarray, ks: np.ndarray):
        """``k r`` (B, P, M) for the wavenumbers *ks* (B,) on one geometry."""
        r, theta = self._geometry(pts, targets)
        ks = np.atleast_1d(np.asarray(ks, dtype=float))
        if not self.propagator.needs_angles:
            return self.xp.asarray(ks)[:, None, None] * r
        f = ks * self.c / (2.0 * np.pi)
        kk = self.propagator.wavenumbers(f, theta, self.xp)
        return kk * r

    def _kr(self, pts: np.ndarray, targets: np.ndarray, k: float):
        r, theta = self._geometry(pts, targets)
        if not self.propagator.needs_angles:
            return k * r
        f = k * self.c / (2.0 * np.pi)
        kk = self.propagator.wavenumbers([f], theta, self.xp)[0]
        return kk * r

    def _green(self, pts: np.ndarray, targets: np.ndarray, ks):
        """Green's functions (B, P, M) for the free field wavenumbers *ks* (B,): the analytic loop, or
        the propagator's ``greens`` for a non analytic one."""
        if self.propagator.analytic:
            return green2d(self._kr_many(pts, targets, ks), self.xp)
        f = np.atleast_1d(np.asarray(ks, dtype=float)) * self.c / (2.0 * np.pi)
        return self.propagator.greens(pts, targets, f, self.xp)

    def _weights_d(self):
        if self.weights is None:
            return None
        if self._weights_cache is None or self._weights_cache[0] is not self.weights:
            self._weights_cache = (self.weights, self.xp.asarray(self.weights))
        return self._weights_cache[1]

    def __call__(self, pts: np.ndarray, k: float):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        g = green2d(self._kr(pts, self.samples, k), self.xp) if self.propagator.analytic else self._green(pts, self.samples, [k])[0]
        a = self.layout.collapse(g, self.xp)
        w = self._weights_d()
        if w is not None:
            a = a * w[None, :]
        return a

    def many(self, pts: np.ndarray, ks: np.ndarray):
        """Replicas (B, P, N) of *pts* for the wavenumbers *ks* (B,), one geometry evaluation."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        a = self.layout.collapse(self._green(pts, self.samples, ks), self.xp)
        w = self._weights_d()
        if w is not None:
            a = a * w[None, None, :]
        return a

    def path(self, pts: np.ndarray, src: np.ndarray, k: float):
        """Free field path factor ``H0^(2)(k |x - s|)`` (P,) from *src* to *pts*."""
        return self.paths(pts, src, k)[:, 0]

    def paths(self, pts: np.ndarray, srcs: np.ndarray, k: float):
        """Path factors (P, M) from every source in *srcs* (M, 2) to *pts*."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        srcs = np.atleast_2d(np.asarray(srcs, dtype=float))
        if not self.propagator.analytic:
            return self._green(pts, srcs, [k])[0]
        return green2d(self._kr(pts, srcs, k), self.xp)


replicas = ReplicaModel  # alias used in the docs


def build_grid(env: ChartEnvironment, bc: BistaticConfig) -> np.ndarray:
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
    exact_track: bool = False  # positions are the exact track (``ship_positions(truth=...)``), not AIS


def _phat(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.abs(X), 1e-30)


def ship_positions(sp: Spectra, ais: AISReports, model: ReplicaModel, bc: BistaticConfig, vessel: str | None = None,
                   truth: np.ndarray | None = None) -> ShipTrack:
    """AIS prediction per frame, refined by a PHAT Bartlett search in a window around it.

    Batch only, and it looks ahead: a ship is marked present from 60 s **before** its first AIS report (and the
    prediction interpolates between reports), which a real-time chain cannot do.  The streaming chain does not
    use this function: ``StreamingPassive`` dead reckons from the reports received by each frame
    (``suppression.predict_ships`` with ``causal=True``) and ``StreamingEcho`` reads no AIS.

    *truth* (K, 2) replaces the AIS prediction (for ablations: exact position, no refinement); the
    track is then marked ``exact_track`` and every run built on it says so in its provenance.
    """
    if bc.impl == "reference":
        ship = _ship_positions_reference(sp, ais, model, bc, vessel, truth)
    else:
        ship = _ship_positions_vectorized(sp, ais, model, bc, vessel, truth)
    ship.exact_track = truth is not None
    return ship


def _ship_track_prelude(sp: Spectra, ais: AISReports, bc: BistaticConfig, vessel: str | None, truth: np.ndarray | None):
    # Batch look-ahead: present from 60 s before the first report (see ship_positions).
    vid = vessel or ais.vessels[0]
    tr = ais.for_vessel(vid)
    pred = tr.predict(sp.times) if truth is None else np.asarray(truth, dtype=float)
    present = (sp.times >= tr.t[0] - 60.0) & (tr.age(sp.times) <= 60.0) if truth is None else np.ones(len(sp.times), bool)
    return vid, pred, present


def _ship_positions_vectorized(sp: Spectra, ais: AISReports, model: ReplicaModel, bc: BistaticConfig, vessel: str | None,
                               truth: np.ndarray | None) -> ShipTrack:
    """The reference search with the replicas of all bins of a frame built in one call (one geometry,
    one Green's function evaluation) and the data of all frames moved to the device once; the power
    is still accumulated bin by bin in the reference order, so the CPU result is identical."""
    vid, pred, present = _ship_track_prelude(sp, ais, bc, vessel, truth)
    refined = pred.copy()
    score = np.ones(len(sp.times))
    if not bc.refine or truth is not None:
        return ShipTrack(vid, pred, refined, present, score)
    xp = model.xp
    h = np.arange(-bc.refine_half, bc.refine_half + 1e-6, bc.refine_step)
    off = np.array([(a, b) for a in h for b in h])
    bins = np.arange(0, len(sp.freqs), 3)
    ks = sp.wavenumbers[bins] / bc.c
    Y = _phat(xp.asarray(sp.X[:, bins, :])) * xp.asarray(sp.weights)[:, None, None]       # (N, Bins, K)
    for k in range(len(sp.times)):
        if not present[k]:
            continue
        pts = pred[k] + off
        A = model.many(pts, ks)                                                              # (Bins, P, N)
        norm = xp.maximum(xp.sum(xp.abs(A) ** 2, axis=2), 1e-30)                            # (Bins, P)
        power = xp.zeros(len(pts))
        for i in range(len(bins)):
            power += xp.abs(A[i].conj() @ Y[:, i, k]) ** 2 / norm[i]
        power = _asnumpy(power)
        best = int(np.argmax(power))
        refined[k] = pts[best]
        score[k] = power[best] / max(power.mean(), 1e-30)
    return ShipTrack(vid, pred, refined, present, score)


def _ship_positions_reference(sp: Spectra, ais: AISReports, model: ReplicaModel, bc: BistaticConfig, vessel: str | None = None,
                              truth: np.ndarray | None = None) -> ShipTrack:
    """The original per frame, per bin loop (``impl = "reference"``; NumPy replica model)."""
    vid = vessel or ais.vessels[0]
    tr = ais.for_vessel(vid)
    pred = tr.predict(sp.times) if truth is None else np.asarray(truth, dtype=float)
    present = (sp.times >= tr.t[0] - 60.0) & (tr.age(sp.times) <= 60.0) if truth is None else np.ones(len(sp.times), bool)
    refined = pred.copy()
    score = np.ones(len(sp.times))
    if not bc.refine or truth is not None:
        return ShipTrack(vid, pred, refined, present, score)
    h = np.arange(-bc.refine_half, bc.refine_half + 1e-6, bc.refine_step)
    off = np.array([(a, b) for a in h for b in h])
    bins = np.arange(0, len(sp.freqs), 3)
    for k in range(len(sp.times)):
        if not present[k]:
            continue
        pts = pred[k] + off
        power = np.zeros(len(pts))
        for i in bins:
            a = model(pts, sp.wavenumbers[i] / bc.c)
            y = _phat(sp.X[:, i, k]) * sp.weights
            power += np.abs(a.conj() @ y) ** 2 / np.maximum(np.sum(np.abs(a) ** 2, axis=1), 1e-30)
        best = int(np.argmax(power))
        refined[k] = pts[best]
        score[k] = power[best] / max(power.mean(), 1e-30)
    return ShipTrack(vid, pred, refined, present, score)


def clutter_images(env: ChartEnvironment, s: np.ndarray, max_range: float = 400.0) -> np.ndarray:
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


def _reference_doppler_vectorized(x: np.ndarray, rec: np.ndarray, ship_t: np.ndarray, j: np.ndarray, fs: float,
                                  bc: BistaticConfig, chunk: int = 16) -> np.ndarray:
    """Sum over channels of ``_resample_at(x[i], j + adv_i)``: the FFT upsampling of *chunk* channels at a
    time along one axis (the same transform per row) and one interpolation call per chunk."""
    xp = _xp(bc)
    n = x.shape[1]
    up_n = n * UPSAMPLE
    grid = xp.arange(up_n, dtype=float)
    ship_d = xp.asarray(ship_t)
    rec_d = xp.asarray(rec)
    j_d = xp.asarray(j)
    out = xp.zeros(n)
    for c0 in range(0, x.shape[0], chunk):
        xc = x[c0:c0 + chunk]
        if xp is np:
            up = resample(np.asarray(xc, dtype=np.float64), up_n, axis=1)
        else:
            from cupyx.scipy.signal import resample as cp_resample

            up = cp_resample(xp.asarray(xc, dtype=np.float64), up_n, axis=1)
        adv = xp.hypot(ship_d[None, :, 0] - rec_d[c0:c0 + chunk, None, 0], ship_d[None, :, 1] - rec_d[c0:c0 + chunk, None, 1]) / bc.c * fs
        pos = (j_d[None, :] + adv) * UPSAMPLE                                               # (chunk, n)
        for i in range(up.shape[0]):
            out += xp.interp(pos[i], grid, up[i], left=0.0, right=0.0)
    return _asnumpy(out)


def make_reference(traces: np.ndarray, layout: SensorLayout, fs: float, ship_xy: np.ndarray, block_s: float,
                   bc: BistaticConfig, channels: np.ndarray | None = None, true_waveform: np.ndarray | None = None,
                   ship_t: np.ndarray | None = None) -> np.ndarray:
    """Illuminator waveform aligned to the emission time, shape ``(n,)``.

    ``bc.reference == "true_waveform"``: the true ship waveform.  ``"data"``: delay and sum of the
    *channels* (default: every live channel) toward the ship position of each block of
    *block_s* seconds (*ship_xy* (K_b, 2), one row per block), each channel advanced by its
    range so the sum is at the emission time.  With *ship_t* (n, 2), the ship position at
    every sample, each channel is instead advanced by its per sample range (a moving ship;
    ``bc.eca_doppler``).  The data reference also contains the echo and the multipath, as in
    every passive radar; that is accepted.
    """
    n = traces.shape[1]
    if bc.reference == "true_waveform":
        if true_waveform is None:
            raise ValueError("reference='true_waveform' needs true_waveform")
        return np.asarray(true_waveform, dtype=np.float64)[:n]
    idx = np.arange(layout.n_receivers) if channels is None else np.asarray(channels)
    x = np.asarray(traces, dtype=np.float64)[idx]
    rec = layout.receivers[idx]
    if ship_t is not None:
        j = np.arange(n, dtype=float)
        live = np.sum(np.abs(x) ** 2, axis=1) > 0
        if bc.impl == "vectorized":
            return _reference_doppler_vectorized(x[live], rec[live], ship_t, j, fs, bc) / max(live.sum(), 1)
        out = np.zeros(n)
        for i in np.flatnonzero(live):
            adv = np.hypot(ship_t[:, 0] - rec[i, 0], ship_t[:, 1] - rec[i, 1]) / bc.c * fs
            out += _resample_at(x[i], j + adv)
        return out / max(live.sum(), 1)
    nb = int(round(block_s * fs))
    out = np.zeros(n)
    for b in range(len(ship_xy)):
        lo, hi = b * nb, min((b + 1) * nb, n)
        if hi <= lo:
            break
        # Advance each channel by its range to the ship (a margin around the block avoids wrap).
        pad = int(0.5 * fs)
        a, z = max(lo - pad, 0), min(hi + pad, n)
        delays = -np.hypot(*(rec - ship_xy[b]).T) / bc.c
        seg = _fractional_delay(x[:, a:z], delays, fs)
        w = np.sum(np.abs(seg) ** 2, axis=1) > 0
        out[lo:hi] = seg[w, lo - a:hi - a].mean(axis=0)
    return out


# ----------------------------------------------------------------------
# Reference estimators (BistaticConfig.ref_mode)
# ----------------------------------------------------------------------

def _aligned_rows(x, rec: np.ndarray, ship_t: np.ndarray, fs: float, bc: BistaticConfig, chunk: int = 16):
    """Every row of *x* (N, n) advanced by its per sample range to the ship (the terms of the delay and sum of
    :func:`_reference_doppler_vectorized`, kept apart): row ``i`` at ``t`` is ``x_i(t + |s(t) - r_i| / c)``.
    Returns an array of ``_xp(bc)``."""
    xp = _xp(bc)
    n = x.shape[1]
    up_n = n * UPSAMPLE
    grid = xp.arange(up_n, dtype=float)
    ship_d, rec_d = xp.asarray(ship_t), xp.asarray(rec)
    j = xp.arange(n, dtype=float)
    out = xp.zeros((x.shape[0], n))
    for c0 in range(0, x.shape[0], chunk):
        xc = x[c0:c0 + chunk]
        if xp is np:
            up = resample(np.asarray(xc, dtype=np.float64), up_n, axis=1)
        else:
            from cupyx.scipy.signal import resample as cp_resample

            up = cp_resample(xp.asarray(xc, dtype=np.float64), up_n, axis=1)
        adv = xp.hypot(ship_d[None, :, 0] - rec_d[c0:c0 + chunk, None, 0], ship_d[None, :, 1] - rec_d[c0:c0 + chunk, None, 1]) / bc.c * fs
        pos = (j[None, :] + adv) * UPSAMPLE
        for i in range(up.shape[0]):
            out[c0 + i] = xp.interp(pos[i], grid, up[i], left=0.0, right=0.0)
    return out


def channel_response(layout: SensorLayout, ship_xy: np.ndarray, freqs: np.ndarray, bc: BistaticConfig,
                     chans: np.ndarray) -> np.ndarray:
    """Modelled response (F, Nc) of the channels *chans* to a unit source at *ship_xy* at *freqs* [Hz], with the
    delay to the channel's position removed: ``collapse(G(k |p - s|)) exp(+i k |r_i - s|)`` through the chain's
    propagator (the Green's function for a pressure channel, its second difference along the cable for a strain
    channel).  What remains is the amplitude and the phase of the channel's response that a delay and sum
    ignores."""
    model = ReplicaModel(layout, bc.c, propagator=bc.propagator)
    ks = 2.0 * np.pi * np.asarray(freqs, dtype=float) / bc.c
    H = np.asarray(_asnumpy(model.many(np.atleast_2d(ship_xy), ks)))[:, 0, :][:, chans]          # (F, Nc)
    r = np.hypot(*(layout.receivers[chans] - np.asarray(ship_xy)[None, :]).T)
    return H * np.exp(1j * ks[:, None] * r[None, :])


def reference_psd_freqs(fs: float, bc: BistaticConfig) -> np.ndarray:
    """The frequencies [Hz] of the noise power of the weighted reference modes (1 / ``ref_psd_s`` apart, to 0.49 fs)."""
    df = 1.0 / bc.ref_psd_s
    return np.arange(df, 0.49 * fs, df)


def residual_psd(residual: np.ndarray, fs: float, freqs: np.ndarray, seg_s: float, xp=np) -> np.ndarray:
    """Noise power (Nc, F) of every row of *residual* at *freqs*: Welch with *seg_s* second Hann segments,
    interpolated; floored at a millionth of the median so a dead row does not divide by zero.  With *xp* CuPy the
    Welch estimate is made on the device (the same estimate to round off) and the interpolation is one weighted sum
    of two columns per frequency."""
    nper = min(int(round(seg_s * fs)), residual.shape[1])
    if xp is np:
        from scipy.signal import welch

        f, P = welch(np.asarray(residual, dtype=np.float64), fs=fs, nperseg=nper, axis=1)
        out = np.stack([np.interp(freqs, f, p) for p in P])
    else:
        from cupyx.scipy.signal import welch

        f, P = welch(xp.asarray(residual, dtype=np.float64), fs=fs, nperseg=nper, axis=1)
        f = _asnumpy(f)
        pos = np.interp(np.asarray(freqs, float), f, np.arange(len(f), dtype=float))      # fractional column of each frequency
        lo = np.minimum(np.floor(pos).astype(int), len(f) - 2)
        w = xp.asarray(pos - lo)
        out = _asnumpy(P[:, lo] * (1.0 - w)[None, :] + P[:, lo + 1] * w[None, :])
    return np.maximum(out, 1e-6 * max(float(np.median(out)), 1e-300))


def _band_filter_blocks(Y, W_blocks: list[np.ndarray], wf: np.ndarray, nb: int, fs: float, bc: BistaticConfig,
                        pad_s: float = 0.25):
    """``sum_i W_i * Y_i`` per block: block ``b`` of the output is the block of the rows of *Y* (Nc, n) filtered
    by ``W_blocks[b]`` (F, Nc), given at the frequencies *wf* and zero outside them, and summed over rows."""
    xp = _xp(bc)
    n = Y.shape[1]
    pad = int(round(pad_s * fs))
    out = xp.zeros(n)
    for b, Wb in enumerate(W_blocks):
        lo, hi = b * nb, min((b + 1) * nb, n)
        if hi <= lo:
            break
        a, z = max(lo - pad, 0), min(hi + pad, n)
        L = z - a
        f = np.fft.rfftfreq(L, 1.0 / fs)
        inb = (f >= wf[0]) & (f <= wf[-1])
        Wf = np.zeros((len(f), Wb.shape[1]), dtype=complex)
        Wf[inb] = np.stack([np.interp(f[inb], wf, Wb[:, i].real) + 1j * np.interp(f[inb], wf, Wb[:, i].imag)
                            for i in range(Wb.shape[1])], axis=1)
        spec = xp.fft.rfft(Y[:, a:z], axis=1)                                                     # (Nc, F)
        seg = xp.fft.irfft(xp.sum(spec * xp.asarray(Wf.T), axis=0), L)
        out[lo:hi] = seg[lo - a:hi - a]
    return out


def _joint_reference(Y, sig2: np.ndarray, init, nb: int, iterations: int, bc: BistaticConfig):
    """Alternating least squares per block: ``Y_i(t) = a_i s(t) + n_i(t)`` with ``var n_i = sig2_i``; the scales
    ``a_i = <Y_i, s> / <s, s>``, then the waveform ``s = sum_i a_i Y_i / sig2_i / sum_i a_i^2 / sig2_i``, from
    *init*, *iterations* times (the rank one fit of the aligned, noise whitened channels)."""
    xp = _xp(bc)
    n = Y.shape[1]
    w = xp.asarray(1.0 / np.maximum(sig2, 1e-300))
    out = xp.array(init, copy=True)
    for lo in range(0, n, nb):
        hi = min(lo + nb, n)
        Yb, s = Y[:, lo:hi], out[lo:hi]
        for _ in range(iterations):
            a = (Yb @ s) / max(float(s @ s), 1e-300)
            den = float(xp.sum(a * a * w))
            if den <= 0:
                break
            s = ((a * w) @ Yb) / den
        out[lo:hi] = s
    return out


def _mode_channels(layout: SensorLayout, chans: np.ndarray, mode: str) -> np.ndarray:
    chans = np.asarray(chans)
    if mode == "hydrophones":
        sl = layout.slices.get("hydrophones")
        return chans[(chans >= sl.start) & (chans < sl.stop)] if sl is not None else chans[:0]
    if mode == "weighted_cable":
        keep = np.zeros(layout.n_receivers, bool)
        for c in layout.cables:
            sl = layout.slices[c.name]
            keep[sl] = True
        return chans[keep[chans]]
    return chans


def estimate_reference(traces: np.ndarray, layout: SensorLayout, fs: float, ship_b: np.ndarray, bc: BistaticConfig,
                       chans: np.ndarray, ship_t: np.ndarray, blocks: set[int] | None = None,
                       bodies: list[np.ndarray] | None = None, info: dict | None = None,
                       noise_psd: np.ndarray | None = None) -> np.ndarray:
    """The data reference of ``bc.ref_mode`` (:class:`BistaticConfig`), aligned to the emission time, shape (n,).

    *ship_b* (K_b, 2): the tracked ship position per ECA block; *ship_t* (n, 2): per sample; *chans*: the live
    channels; *blocks*: the ECA blocks whose residual estimates the noise power (default: all); *bodies*: the
    positions (2,) of the echo tracks whose modelled echo ``"iterated"`` subtracts.  Nothing here reads the truth.
    *info*, when given, receives the intermediate products (``"psd"``, ``"freqs"``, ``"channels"``).  *noise_psd*
    (N, F) at the frequencies of :func:`reference_psd_freqs`, every channel (``bc.ref_noise = "previous"``): the noise
    power of the weighted modes, instead of the residual of a first pass."""
    mode = bc.ref_mode
    beam = replace(bc, reference="data", ref_mode="beam")
    if mode == "beam":
        return make_reference(traces, layout, fs, ship_b, bc.eca_block_s, beam, chans, ship_t=ship_t)
    xp = _xp(bc)
    nb = int(round(bc.eca_block_s * fs))
    x = np.asarray(traces, dtype=np.float64)
    # First pass: the beam and the ECA; the residual is the noise of every channel.
    PROFILE.begin("echo: reference estimation / beam")
    Y = None
    if mode == "joint":
        # The beam is the mean of the live channels advanced by their ranges: the rows the joint solve needs anyway
        # (``_aligned_rows``, the same interpolation as ``make_reference``), so they are made once.
        use0 = np.sort(np.asarray(chans))
        Y = _aligned_rows(x[use0], layout.receivers[use0], ship_t, fs, bc)
        live = xp.asarray(np.sum(x[use0] ** 2, axis=1) > 0)
        ref0 = _asnumpy(xp.sum(Y[live], axis=0)) / max(int(live.sum()), 1)
    else:
        ref0 = make_reference(x, layout, fs, ship_b, bc.eca_block_s, beam, chans, ship_t=ship_t)
    PROFILE.end("echo: reference estimation / beam")
    res = None
    if noise_psd is None:
        PROFILE.begin("echo: reference estimation / first ECA")
        res = eca_remove([x], ref0, layout, fs, ship_b, [], bc.eca_block_s, beam, ship_t=ship_t, blocks=blocks)[0]
        PROFILE.end("echo: reference estimation / first ECA")
    if blocks:
        cols = np.concatenate([np.arange(b * nb, min((b + 1) * nb, x.shape[1])) for b in sorted(blocks)])
    else:
        cols = np.arange(x.shape[1])
    chans = np.sort(np.asarray(chans))
    use = _mode_channels(layout, chans, "weighted" if mode in ("iterated", "joint") else mode)
    if len(use) == 0:
        raise ValueError(f"ref_mode {mode!r}: no live channel of that kind")
    # The weights cover the whole spectrum, not only the processing band: the direct path has energy outside the band
    # and a reference without it leaves that energy in every channel (measured on the analytic case: the true
    # waveform limited to 100 to 600 Hz removes 14 dB of a direct path in 80 to 700 Hz, unlimited 69 dB).
    wf = reference_psd_freqs(fs, bc)
    if noise_psd is not None:
        psd = np.asarray(noise_psd)[use]
    else:
        PROFILE.begin("echo: reference estimation / noise spectra")
        psd = residual_psd(res[use][:, cols], fs, wf, bc.ref_psd_s, xp)                                 # (Nc, F)
        PROFILE.end("echo: reference estimation / noise spectra")
    if info is not None:
        info.update(psd=psd, freqs=wf, channels=use)
    PROFILE.begin("echo: reference estimation / aligned channels")
    if Y is None or not np.array_equal(use, np.sort(np.asarray(chans))):
        Y = _aligned_rows(x[use], layout.receivers[use], ship_t, fs, bc)
    PROFILE.end("echo: reference estimation / aligned channels")
    if mode == "joint":
        sig2 = np.mean(psd[:, (wf >= bc.f_lo) & (wf <= bc.f_hi)], axis=1)
        with PROFILE.stage("echo: reference estimation / joint solve"):
            return _asnumpy(_joint_reference(Y, sig2, xp.mean(Y, axis=0), nb, bc.ref_iterations, bc))
    # Maximum ratio combination: the emitted waveform estimated as sum_i conj(A_i) X_i / N_i / sum_i |A_i|^2 / N_i,
    # then given the response the beam of every live channel has to it (the mean of A over *chans*), so that the
    # reference is the ship's sound as the array receives it, with the least noise.  Without the second factor the
    # reference's spectrum is sum |A|^2 / N (a strain cable: the fourth power of frequency), which the ECA's short
    # filters cannot undo (measured on the analytic case: 21 dB against 23 dB for the beam despite the higher
    # coherence; the hydrophones alone 6 dB).
    coarse = wf[::4] if len(wf) > 8 else wf                     # the response without its delay is smooth in frequency
    pos = np.searchsorted(chans, use)
    W = []
    for b in range(len(ship_b)):
        A = channel_response(layout, ship_b[b], coarse, bc, chans)
        A = np.stack([np.interp(wf, coarse, A[:, i].real) + 1j * np.interp(wf, coarse, A[:, i].imag) for i in range(A.shape[1])], axis=1)
        Au = A[:, pos]
        mrc = np.conj(Au) / psd.T
        den = np.sum(mrc * Au, axis=1).real
        W.append(mrc * (np.mean(A, axis=1) / np.maximum(den, 1e-300))[:, None])
    ref = _band_filter_blocks(Y, W, wf, nb, fs, bc)
    ref = _asnumpy(ref)
    if mode != "iterated":
        return ref
    # Iterated: the echo of every body modelled on the residual of the weighted reference and subtracted from the
    # traces, then the beam again.
    res1 = eca_remove([x], ref, layout, fs, ship_b, [], bc.eca_block_s, beam, ship_t=ship_t, blocks=blocks)[0]
    clean = x
    for body in bodies or []:
        body = np.asarray(body, dtype=float)
        extra = np.hypot(*(ship_t - body[None, :]).T) / bc.c * fs                                    # source to body [samples]
        j = np.arange(x.shape[1], dtype=float)
        shifted = _resample_at(ref, j - extra)
        body_t = np.repeat(body[None, :], x.shape[1], axis=0)
        body_b = np.repeat(body[None, :], len(ship_b), axis=0)
        res2 = eca_remove([res1], shifted, layout, fs, body_b, [], bc.eca_block_s, beam, ship_t=body_t, blocks=blocks)[0]
        echo = res1 - res2
        clean = clean - echo
        res1 = res2
    return make_reference(clean, layout, fs, ship_b, bc.eca_block_s, beam, chans, ship_t=ship_t)


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
    blocks: set[int] | None = None,
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

    ``bc.impl = "vectorized"`` builds the regressor matrices of all channels of a block from
    strided views of the (padded, resampled) reference and solves the normal equations of the
    channels with the same lag count together (:func:`_project_out_batched`); the reference loop
    is :func:`_eca_remove_reference`.  *blocks* (indices) restricts the cancellation to those blocks (the
    others are returned unchanged): the real-time chain cleans only the newest blocks of its buffer.
    """
    if bc.impl == "reference":
        return _eca_remove_reference(traces, reference, layout, fs, ship_xy, images, block_s, bc, ship_t, blocks)
    xp = _xp(bc)
    if xp is not np and bc.reference == "true_waveform" and ship_t is not None:
        # True-waveform-driven per-sample-delay normal equations can be ill-conditioned enough that
        # cuSOLVER and LAPACK produce materially different projections. Keep this rare path
        # on the CPU so the GPU implementation meets the reference accuracy contract.
        cpu_bc = replace(bc, use_cuda=False)
        return _eca_remove_vectorized(traces, reference, layout, fs, ship_xy, images, block_s, cpu_bc, ship_t, chunk=bc.eca_chunk,
                                      blocks=blocks)
    return _eca_remove_vectorized(traces, reference, layout, fs, ship_xy, images, block_s, bc, ship_t, chunk=bc.eca_chunk, blocks=blocks)


def _gram_prefix(basep, start, block: int):
    """``S^T S`` (Ng, L, L) for ``S[g, t, l] = basep[g, start[g, l] + t]`` (one base per channel) from prefix
    sums of the lagged products: ``G[l, m] = sum_t b[s_l + t] b[s_m + t] = P_d[s + block] - P_d[s]`` with
    ``s = min(s_l, s_m)``, ``d = |s_l - s_m|`` and ``P_d`` the cumulative sum of ``b[w] b[w + d]``."""
    xp = _module_of(basep)
    Ng, L = start.shape
    st = xp.asarray(start)
    d_all = xp.abs(st[:, :, None] - st[:, None, :])                                          # (Ng, L, L)
    lo_all = xp.minimum(st[:, :, None], st[:, None, :])
    dmax = int(d_all.max())
    n = basep.shape[1]
    padded = xp.concatenate([basep, xp.zeros((Ng, dmax))], axis=1)
    lagged = xp.lib.stride_tricks.sliding_window_view(padded, dmax + 1, axis=1)[:, :n]       # (Ng, n, dmax + 1): b[w + d]
    P = xp.zeros((Ng, n + 1, dmax + 1))
    xp.cumsum(basep[:, :, None] * lagged, axis=1, out=P[:, 1:])
    g = xp.arange(Ng)[:, None, None]
    return P[g, lo_all + block, d_all] - P[g, lo_all, d_all]


def _gram_cross_prefix(bp, sp, bq, sq, block: int):
    """``S_p^T S_q`` (Ng, Lp, Lq) for ``S_p[g, t, l] = bp[g, sp[g, l] + t]`` and ``S_q[g, t, m] = bq[g, sq[g, m] + t]``
    (two bases on the same time axis, one each per channel) from prefix sums of their sliding products:
    ``G[l, m] = sum_t bp[s + t] bq[s + e + t] = Q_e[s + block] - Q_e[s]`` with ``s = sp[l]``, ``e = sq[m] - sp[l]``
    and ``Q_e`` the cumulative sum of ``bp[w] bq[w + e]`` (``bq`` zero outside its length)."""
    xp = _module_of(bp)
    Ng, n = bp.shape
    spd, sqd = xp.asarray(sp), xp.asarray(sq)
    e = sqd[:, None, :] - spd[:, :, None]                                                     # (Ng, Lp, Lq)
    e0 = e.min(axis=(1, 2))                                                                   # (Ng,)
    J = int((e.max(axis=(1, 2)) - e0).max()) + 1
    pad = int(xp.abs(e).max()) + J + 1
    bqp = xp.concatenate([xp.zeros((Ng, pad)), bq, xp.zeros((Ng, pad + J))], axis=1)
    win = xp.lib.stride_tricks.sliding_window_view(bqp, J, axis=1)                           # win[g, i, j] = bqp[g, i + j]
    rows = xp.arange(n)[None, :] + (e0 + pad)[:, None]                                        # bq[w + e0 + j] = bqp[w + e0 + pad + j]
    shifted = win[xp.arange(Ng)[:, None], rows]                                               # (Ng, n, J)
    Q = xp.zeros((Ng, n + 1, J))
    xp.cumsum(bp[:, :, None] * shifted, axis=1, out=Q[:, 1:])
    g = xp.arange(Ng)[:, None, None]
    s0 = xp.broadcast_to(spd[:, :, None], e.shape)
    j = e - e0[:, None, None]
    return Q[g, s0 + block, j] - Q[g, s0, j]


def _project_out_batched(S, segs, loading: float = 1e-14, G=None):
    """Batched :func:`_project_out`: *S* (Ng, block, L), *segs* (Ng, block, R) -> cleaned (Ng, block, R).

    The normal equations of all channels come from one stacked product (identical to the per
    channel products).  On the CPU the factorisation and solves are SciPy's ``cho_factor`` /
    ``cho_solve`` per channel, exactly the reference calls, so the result is identical to the
    last bit; on the GPU they are CuPy's batched Cholesky and triangular solves.
    """
    xp = _module_of(S)
    Ng, block, L = S.shape
    St = S.transpose(0, 2, 1)
    G = xp.matmul(St, S) if G is None else G                                                # (Ng, L, L): identical to the per channel S.T @ S
    if xp is np:
        # Per channel and per array, the exact call sequence of _project_out (loading from the channel's
        # own trace, matrix-vector products): a batched matrix-matrix product rounds differently, and
        # the condition number of the normal equations amplifies that.
        out = np.empty_like(segs)
        for g in range(Ng):
            Sg = S[g]
            Gg = G[g]
            Gg[np.diag_indices_from(Gg)] += loading * max(np.trace(Gg) / L, 1e-300)
            try:
                c, low = cho_factor(Gg, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                Q, _ = np.linalg.qr(Sg)
                for r_i in range(segs.shape[2]):
                    out[g, :, r_i] = segs[g, :, r_i] - Q @ (Q.T @ segs[g, :, r_i])
                continue
            for r_i in range(segs.shape[2]):
                seg = np.ascontiguousarray(segs[g, :, r_i])
                x = cho_solve((c, low), Sg.T @ seg, check_finite=False)
                r = seg - Sg @ x
                x += cho_solve((c, low), Sg.T @ r, check_finite=False)
                out[g, :, r_i] = seg - Sg @ x
        return out
    tr = xp.trace(G, axis1=1, axis2=2) / L
    diag = xp.arange(L)
    G[:, diag, diag] += (loading * xp.maximum(tr, 1e-300))[:, None]
    rhs = xp.matmul(St, segs)                                                               # (Ng, L, R)
    from cupyx.scipy.linalg import solve_triangular

    try:
        Lc = xp.linalg.cholesky(G)
    except Exception:
        host = _project_out_batched(_asnumpy(S), _asnumpy(segs), loading)
        return xp.asarray(host)
    if not bool(xp.isfinite(Lc).all()):
        host = _project_out_batched(_asnumpy(S), _asnumpy(segs), loading)
        return xp.asarray(host)
    Lt = Lc.transpose(0, 2, 1)

    def solve(b):
        return solve_triangular(Lt, solve_triangular(Lc, b, lower=True), lower=False)

    x = solve(rhs)
    r = segs - xp.matmul(S, x)
    x = x + solve(xp.matmul(St, r))
    return segs - xp.matmul(S, x)


def _project_out_qr(S, segs):
    """Each segment minus its least squares fit on the columns of *S*: *S* (Ng, block, L) and *segs* (Ng, block, R) in
    single precision; the fit by a QR of every channel's ``S`` in double precision (no normal equations, so no loading),
    the projection applied in single precision.  Returns (Ng, block, R) in the dtype of *segs*."""
    xp = _module_of(S)
    S64 = S.astype(np.float64)
    try:
        Q, R = xp.linalg.qr(S64)                                                            # batched: (Ng, block, L), (Ng, L, L)
    except Exception:
        QR = [xp.linalg.qr(S64[g]) for g in range(S64.shape[0])]
        Q, R = xp.stack([q for q, _ in QR]), xp.stack([r for _, r in QR])
    rhs = xp.matmul(Q.transpose(0, 2, 1), segs.astype(np.float64))                          # (Ng, L, R) accumulated in double
    if xp is np:
        from scipy.linalg import solve_triangular as st

        x = np.stack([st(R[g], rhs[g], lower=False, check_finite=False) for g in range(R.shape[0])])
    else:
        from cupyx.scipy.linalg import solve_triangular as st

        x = st(R, rhs, lower=False)
    return segs - xp.matmul(S, x.astype(S.dtype))


def _eca_lags(nominal: np.ndarray, half: int, hi: int) -> list[np.ndarray]:
    """Per channel lag sets of the reference: the union of the windows around the nominal delays
    (``nominal`` (N, P)), restricted to ``[0, hi)``."""
    out = []
    for nom in nominal:
        lags = np.unique(np.concatenate([np.arange(L0 - half, L0 + half + 1) for L0 in nom]))
        out.append(lags[(lags >= 0) & (lags < hi)])
    return out


def _eca_remove_vectorized(traces, reference, layout, fs, ship_xy, images, block_s, bc, ship_t, chunk: int = 32,
                           blocks: set[int] | None = None) -> list[np.ndarray]:
    xp = _xp(bc)
    from_window = xp.lib.stride_tricks.sliding_window_view
    n = traces[0].shape[1]
    nb = int(round(block_s * fs))
    half = int(round(bc.eca_halfwidth_s * fs))
    outs = [np.array(x, dtype=np.float64, copy=True) for x in traces]
    rec = layout.receivers
    ref = np.asarray(reference, dtype=np.float64)
    if ship_t is None:
        refp = xp.asarray(np.concatenate([np.zeros(n), ref]))                                # lag L of block [lo, hi): refp[n + lo - L : n + hi - L]
    else:
        PROFILE.begin("ECA / reference upsampled")
        ref_up = xp.asarray(resample(ref, n * UPSAMPLE))
        PROFILE.end("ECA / reference upsampled")
        if bc.eca_precision == "mixed":
            ref_up = ref_up.astype(np.float32)
        j_up = xp.arange(n * UPSAMPLE, dtype=float)
        j_all = xp.arange(n, dtype=float)
        rec_d = xp.asarray(rec)
    for b in range(len(ship_xy)):
        if blocks is not None and b not in blocks:
            continue
        lo, hi = b * nb, min((b + 1) * nb, n)
        if hi <= lo:
            break
        block = hi - lo
        srcs = [ship_xy[b]] + ([images[b]] if len(images) and len(images[b]) else [])
        pts = np.vstack([np.atleast_2d(p) for p in srcs])                                   # (P, 2)
        nominal = np.rint(np.hypot(pts[None, :, 0] - rec[:, None, 0], pts[None, :, 1] - rec[:, None, 1]) / bc.c * fs).astype(int)
        lag_sets = _eca_lags(nominal, half, hi)
        if ship_t is not None:
            a, z = max(lo - half, 0), min(hi + half, n)
            s_blk = xp.asarray(ship_t[a:z])                                                 # (z - a, 2)
            walls = 0.5 * (pts[1:] + ship_xy[b][None, :])                                   # image = 2 wall - ship
            paths = xp.stack([s_blk] + [2.0 * xp.asarray(w)[None, :] - s_blk for w in walls])   # (P, z - a, 2)
            pad = half
        groups: dict[int, list[int]] = {}
        for i, lags in enumerate(lag_sets):
            if len(lags):
                groups.setdefault(len(lags), []).append(i)
        for L, chans in groups.items():
            for c0 in range(0, len(chans), chunk):
                idx = np.array(chans[c0:c0 + chunk])
                lags = np.stack([lag_sets[i] for i in idx])                                    # (Ng, L)
                PROFILE.begin("ECA / delayed copies")
                if ship_t is None:
                    W = from_window(refp, block)                                              # (n_win, block): row w = refp[w : w + block]
                    S = W[xp.asarray(n + lo - lags)]                                           # (Ng, L, block)
                else:
                    d = xp.hypot(paths[None, :, :, 0] - rec_d[idx][:, None, None, 0],
                                 paths[None, :, :, 1] - rec_d[idx][:, None, None, 1]) / bc.c * fs    # (Ng, P, z - a)
                    pos = (j_all[a:z][None, None, :] - d) * UPSAMPLE
                    bases = xp.interp(pos.ravel(), j_up, ref_up, left=0.0, right=0.0).reshape(pos.shape)
                    if bc.eca_precision == "mixed":
                        bases = bases.astype(np.float32)
                    basep = xp.concatenate([xp.zeros(bases.shape[:2] + (pad,)), bases, xp.zeros(bases.shape[:2] + (pad,))], axis=-1)
                    owner = np.argmin(np.abs(lags[:, :, None] - nominal[idx][:, None, :]), axis=2)      # (Ng, L)
                    off = lags - np.take_along_axis(nominal[idx], owner, axis=1)
                    start = pad + (lo - a) - off                                                      # window start in basep
                    W = from_window(basep, block, axis=-1)                                           # (Ng, P, n_win, block)
                    S = W[xp.asarray(np.arange(len(idx)))[:, None], xp.asarray(owner), xp.asarray(start)]   # (Ng, L, block)
                PROFILE.end("ECA / delayed copies")
                PROFILE.begin("ECA / normal equations")
                G = None
                if ship_t is not None and bc.eca_gram == "prefix" and basep.shape[1] == 1 and bc.eca_precision != "mixed":
                    G = _gram_prefix(basep[:, 0, :], start, block)
                S = xp.ascontiguousarray(S.transpose(0, 2, 1))                                        # (Ng, block, L)
                segs = xp.asarray(np.stack([out[idx, lo:hi] for out in outs], axis=-1))               # (Ng, block, R)
                PROFILE.end("ECA / normal equations")
                PROFILE.begin("ECA / solve")
                if bc.eca_precision == "mixed":
                    cleaned = _asnumpy(_project_out_qr(S.astype(np.float32), segs.astype(np.float32)))
                else:
                    cleaned = _asnumpy(_project_out_batched(S, segs, G=G))
                PROFILE.end("ECA / solve")
                for r_i, out in enumerate(outs):
                    out[idx, lo:hi] = cleaned[:, :, r_i]
    return outs


def _eca_remove_reference(
    traces: list[np.ndarray],
    reference: np.ndarray,
    layout: SensorLayout,
    fs: float,
    ship_xy: np.ndarray,
    images: list[np.ndarray],
    block_s: float,
    bc: BistaticConfig,
    ship_t: np.ndarray | None = None,
    blocks: set[int] | None = None,
) -> list[np.ndarray]:
    """The original per channel loop of :func:`eca_remove` (``impl = "reference"``).

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
        if blocks is not None and b not in blocks:
            continue
        lo, hi = b * nb, min((b + 1) * nb, n)
        if hi <= lo:
            break
        srcs = [ship_xy[b]] + ([images[b]] if len(images) and len(images[b]) else [])
        pts = np.vstack([np.atleast_2d(p) for p in srcs])
        if ship_t is not None:
            # Per sample positions of every path over the block plus the lag margin.
            a, z = max(lo - half, 0), min(hi + half, n)
            s_blk = ship_t[a:z]
            walls = 0.5 * (pts[1:] + ship_xy[b][None, :])                 # image = 2 wall - ship
            paths = [s_blk] + [2.0 * w[None, :] - s_blk for w in walls]
        for i in range(rec.shape[0]):
            nominal = np.array([int(round(np.hypot(*(p - rec[i])) / bc.c * fs)) for p in pts])
            lags = np.unique(np.concatenate([np.arange(L0 - half, L0 + half + 1) for L0 in nominal]))
            lags = lags[(lags >= 0) & (lags < hi)]
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
                        d = np.hypot(paths[p][:, 0] - rec[i, 0], paths[p][:, 1] - rec[i, 1]) / bc.c * fs
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


def _eca_remove_joint(traces: list[np.ndarray], references: list[np.ndarray], layout: SensorLayout, fs: float,
                      ships_b: list[np.ndarray], ships_t: list[np.ndarray], images: list[list[np.ndarray]],
                      block_s: float, bc: BistaticConfig, blocks: set[int] | None = None,
                      chunk: int = 32) -> list[np.ndarray]:
    """ECA over several illuminators at once: for every channel and block, one least squares fit on the delayed
    copies of every illuminator's reference (each at the per sample delays of its own track, ``ships_t[k]`` (n, 2),
    with the lag windows of :func:`eca_remove` around its direct arrival and, with ``images[k]``, its image
    arrivals), the fit removed from every array of *traces*.

    Returns the cleaned arrays.  The normal equations are formed by the matrix product (the prefix sum Gram of
    ``eca_gram = "prefix"`` needs every column to be a shift of one base, which several references are not); with one
    illuminator use :func:`eca_remove`."""
    xp = _xp(bc)
    from_window = xp.lib.stride_tricks.sliding_window_view
    n = traces[0].shape[1]
    K = len(references)
    nb = int(round(block_s * fs))
    half = int(round(bc.eca_halfwidth_s * fs))
    outs = [np.array(x, dtype=np.float64, copy=True) for x in traces]
    rec = layout.receivers
    PROFILE.begin("joint ECA / references upsampled")
    refs_up = [xp.asarray(resample(np.asarray(r, dtype=np.float64), n * UPSAMPLE)) for r in references]
    PROFILE.end("joint ECA / references upsampled")
    j_up = xp.arange(n * UPSAMPLE, dtype=float)
    j_all = xp.arange(n, dtype=float)
    rec_d = xp.asarray(rec)
    n_blocks = min(len(sb) for sb in ships_b)
    for b in range(n_blocks):
        if blocks is not None and b not in blocks:
            continue
        lo, hi = b * nb, min((b + 1) * nb, n)
        if hi <= lo:
            break
        block = hi - lo
        a, z = max(lo - half, 0), min(hi + half, n)
        per = []                                                             # per illuminator: nominal, lag sets, paths
        for k in range(K):
            imgs = images[k][b] if images and len(images[k]) and len(images[k][b]) else []
            pts = np.vstack([np.atleast_2d(ships_b[k][b])] + ([np.atleast_2d(imgs)] if len(imgs) else []))
            nominal = np.rint(np.hypot(pts[None, :, 0] - rec[:, None, 0], pts[None, :, 1] - rec[:, None, 1]) / bc.c * fs).astype(int)
            s_blk = xp.asarray(ships_t[k][a:z])
            walls = 0.5 * (pts[1:] + ships_b[k][b][None, :])
            paths = xp.stack([s_blk] + [2.0 * xp.asarray(w)[None, :] - s_blk for w in walls])
            per.append((nominal, _eca_lags(nominal, half, hi), paths))
        groups: dict[tuple, list[int]] = {}
        for i in range(len(rec)):
            key = tuple(len(p[1][i]) for p in per)
            if sum(key):
                groups.setdefault(key, []).append(i)
        for key, chans in groups.items():
            for c0 in range(0, len(chans), chunk):
                idx = np.array(chans[c0:c0 + chunk])
                PROFILE.begin("joint ECA / delayed copies")
                Ss = []
                pref = []                                                                            # (base, starts) per reference
                for k, (nominal, lag_sets, paths) in enumerate(per):
                    if key[k] == 0:
                        continue
                    lags = np.stack([lag_sets[i] for i in idx])                                      # (Ng, L_k)
                    d = xp.hypot(paths[None, :, :, 0] - rec_d[idx][:, None, None, 0],
                                 paths[None, :, :, 1] - rec_d[idx][:, None, None, 1]) / bc.c * fs   # (Ng, P, z - a)
                    pos = (j_all[a:z][None, None, :] - d) * UPSAMPLE
                    bases = xp.interp(pos.ravel(), j_up, refs_up[k], left=0.0, right=0.0).reshape(pos.shape)
                    basep = xp.concatenate([xp.zeros(bases.shape[:2] + (half,)), bases, xp.zeros(bases.shape[:2] + (half,))], axis=-1)
                    owner = np.argmin(np.abs(lags[:, :, None] - nominal[idx][:, None, :]), axis=2)
                    off = lags - np.take_along_axis(nominal[idx], owner, axis=1)
                    start = half + (lo - a) - off
                    W = from_window(basep, block, axis=-1)                                           # (Ng, P, n_win, block)
                    Ss.append(W[xp.asarray(np.arange(len(idx)))[:, None], xp.asarray(owner), xp.asarray(start)])
                    pref.append((basep[:, 0, :], start) if basep.shape[1] == 1 else None)
                S = xp.ascontiguousarray(xp.concatenate(Ss, axis=1).transpose(0, 2, 1))                   # (Ng, block, L)
                G = None
                if bc.joint_gram == "prefix" and all(x is not None for x in pref):
                    with PROFILE.stage("joint ECA / prefix Gram"):
                        G = xp.concatenate([xp.concatenate([_gram_prefix(bp_, sp_, block) if i == j_ else
                                                            _gram_cross_prefix(bp_, sp_, bq_, sq_, block)
                                                            for j_, (bq_, sq_) in enumerate(pref)], axis=2)
                                            for i, (bp_, sp_) in enumerate(pref)], axis=1)
                segs = xp.asarray(np.stack([out[idx, lo:hi] for out in outs], axis=-1))                    # (Ng, block, R)
                PROFILE.end("joint ECA / delayed copies")
                with PROFILE.stage("joint ECA / normal equations and solve"):
                    cleaned = _asnumpy(_project_out_batched(S, segs, G=G))
                for r_i, out in enumerate(outs):
                    out[idx, lo:hi] = cleaned[:, :, r_i]
    return outs


def remove_direct_paths(traces: np.ndarray, components: list[np.ndarray], layout: SensorLayout, fs: float,
                        ships: list[tuple[np.ndarray, np.ndarray]], chans: np.ndarray, bc: BistaticConfig,
                        blocks: set[int] | None = None, bodies: list[np.ndarray] | None = None,
                        images: list[list[np.ndarray]] | None = None,
                        noise_psd: list[np.ndarray | None] | None = None) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Remove the direct path of every illuminator in *ships* (``(ship_b, ship_t)`` each: the position per ECA block
    and per sample, in removal order) from *traces* (N, n), the same projections applied to every array of
    *components* (a diagnostic).  ``bc.removal_order`` (:class:`BistaticConfig`): ``"sequential"``, each reference
    (``bc.ref_mode``, :func:`estimate_reference`) on the traces cleaned of the previous illuminators and each ECA on
    its own; ``"joint"``, ``bc.joint_passes`` passes over the illuminators in which each reference is estimated on the
    traces minus the other illuminators' current fits (each fit its own ECA on those traces), then one ECA over all
    references per block (:func:`_eca_remove_joint`).  Returns the cleaned traces, the cleaned components and the
    references (one per illuminator, in order).  *noise_psd*: per illuminator the channels' noise power its reference
    takes instead of a first pass (``bc.ref_noise = "previous"``, :func:`estimate_reference`), or ``None``.
    Nothing here reads the truth."""
    x = np.asarray(traces, dtype=np.float64)
    comps = [np.asarray(c, dtype=np.float64) for c in components]
    imgs = images if images is not None else [[] for _ in ships]
    npsd = noise_psd if noise_psd is not None else [None for _ in ships]
    if bc.removal_order == "sequential" or len(ships) <= 1:
        refs = []
        for k, (ship_b, ship_t) in enumerate(ships):
            with PROFILE.stage("echo: reference estimation"):
                ref = estimate_reference(x, layout, fs, ship_b, bc, chans, ship_t, blocks=blocks, bodies=bodies,
                                         noise_psd=npsd[k])
            with PROFILE.stage("echo: ECA"):
                cleaned = eca_remove([x] + comps, ref, layout, fs, ship_b, imgs[k], bc.eca_block_s, bc, ship_t=ship_t,
                                     blocks=blocks)
            x, comps = cleaned[0], cleaned[1:]
            refs.append(ref)
        return x, comps, refs
    if not bc.eca_doppler:
        raise ValueError("removal_order 'joint' needs eca_doppler (per sample delays)")
    sb_all, st_all = [sb for sb, _ in ships], [st for _, st in ships]
    chunk = max(1, bc.eca_chunk // len(ships))                 # channels per solve: the lag count grows with the ships
    if bc.joint_passes == 0:                                   # one solve on the beams of the raw traces
        beam = replace(bc, reference="data", ref_mode="beam")
        refs = [make_reference(x, layout, fs, sb, bc.eca_block_s, beam, chans, ship_t=st) for sb, st in ships]
    else:
        # Block coordinate passes: each reference (bc.ref_mode, with its own first pass for the noise power) on the
        # traces minus the other illuminators' current fits, and its fit (its own ECA on those traces) updated at
        # once.  The first pass is the sequential removal; each further pass re-estimates every reference with the
        # others' direct sound taken out.
        fits = [np.zeros_like(x) for _ in ships]
        refs = [None] * len(ships)
        for _ in range(bc.joint_passes):
            for k, (sb, st) in enumerate(ships):
                y = x - np.sum([fits[j] for j in range(len(ships)) if j != k], axis=0)
                with PROFILE.stage("echo: reference estimation"):
                    refs[k] = estimate_reference(y, layout, fs, sb, bc, chans, st, blocks=blocks, bodies=bodies,
                                                 noise_psd=npsd[k])
                with PROFILE.stage("echo: ECA"):
                    fits[k] = y - eca_remove([y], refs[k], layout, fs, sb, imgs[k], bc.eca_block_s, bc, ship_t=st,
                                             blocks=blocks)[0]
    with PROFILE.stage("echo: ECA, joint solve"):
        cleaned = _eca_remove_joint([x] + comps, refs, layout, fs, sb_all, st_all, imgs, bc.eca_block_s, bc, blocks,
                                    chunk=chunk)
    return cleaned[0], cleaned[1:], refs


def reference_spectrum(sp: Spectra, ship: ShipTrack, model: ReplicaModel, bc: BistaticConfig,
                       given: np.ndarray | None = None, waveform: np.ndarray | None = None, fs: float | None = None) -> np.ndarray:
    """Illuminator spectrum ``r`` (F, K).

    From *waveform* (the time domain reference of :func:`make_reference`, aligned to the
    emission time) when given; else the STFT *given* of the true waveform; else a beam at the refined ship position
    per frame and bin (the analytic demo's estimate).
    """
    if waveform is not None:
        return waveform_spectra(waveform, fs, sp)
    if bc.reference == "true_waveform":
        if given is None:
            raise ValueError("reference='true_waveform' needs the spectra of the true waveform")
        return given
    r = np.zeros((len(sp.freqs), len(sp.times)), dtype=np.complex128)
    if bc.impl == "vectorized":
        xp = model.xp
        X_d = xp.asarray(sp.X)
        for k in range(len(sp.times)):
            if not ship.present[k]:
                continue
            a = model.many(ship.refined[k][None], sp.wavenumbers / bc.c)[:, 0, :]          # (F, N)
            num = xp.sum(xp.conj(a) * X_d[:, :, k].T, axis=1)
            den = xp.maximum(xp.sum(xp.conj(a) * a, axis=1).real, 1e-30)
            r[:, k] = _asnumpy(num / den)
        return r
    for k in range(len(sp.times)):
        if not ship.present[k]:
            continue
        for i, kk in enumerate(sp.wavenumbers / bc.c):
            a = model(ship.refined[k][None], kk)[0]
            r[i, k] = np.vdot(a, sp.X[:, i, k]) / max(np.vdot(a, a).real, 1e-30)
    return r


# ----------------------------------------------------------------------
# Frequency domain direct path removal (kept for comparison with ECA)
# ----------------------------------------------------------------------

def remove_direct(sp: Spectra, ship: ShipTrack, model: ReplicaModel, bc: BistaticConfig) -> Spectra:
    """Project the ship out of every frame and bin (``"patch"`` and/or ``"adaptive"``).

    Measured on the coarse dataset (the removal experiment of ``harbor_bistatic_detect.py``, a script no longer in the repository):
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
            pts = ship.refined[k] + patch
            for i, kk in enumerate(sp.wavenumbers / bc.c):
                A = _asnumpy(model(pts, kk)).T                         # (N, 9)
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
    provenance: Provenance = field(default_factory=Provenance)


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

    ``bc.impl = "vectorized"``: the replica of every bin is built once and applied to all frames,
    the ship to cell paths of all frames come from one Green's function call, and the MVDR and
    MUSIC processors use batched inverses and eigendecompositions over frames
    (:func:`_bistatic_map_vectorized`); the per frame loops are :func:`_bistatic_map_reference`.
    """
    if bc.impl == "reference":
        return _bistatic_map_reference(clean, ship, ref, grid, model, bc, method)
    return _bistatic_map_vectorized(clean, ship, ref, grid, model, bc, method)


def _csm_batch(Xi, window: int, loading: float):
    """Loaded cross spectral matrices of every frame, (K, N, N) (the :func:`_csm` of each frame)."""
    xp = _module_of(Xi)
    N, K = Xi.shape
    R = xp.empty((K, N, N), dtype=Xi.dtype)
    eye = xp.eye(N)
    for k in range(K):
        lo, hi = max(0, k - window), min(K, k + window + 1)
        D = Xi[:, lo:hi]
        Rk = D @ D.conj().T / D.shape[1]
        R[k] = Rk + loading * xp.trace(Rk).real / N * eye
    return R


def _bistatic_map_vectorized(clean, ship, ref, grid, model, bc, method, frame_chunk: int = 8):
    method = method or bc.method
    xp = model.xp
    G, K, F = len(grid), len(clean.times), len(clean.freqs)
    B = xp.zeros((G, K))
    Z = xp.zeros((G, K), dtype=np.complex128) if method == "coherent" else None
    W = xp.zeros((G, K)) if method == "coherent" else None
    present = np.flatnonzero(ship.present)
    present_d = xp.asarray(present)
    ship_p = ship.refined[present]
    X_d = xp.asarray(clean.X)
    w_d = xp.asarray(clean.weights)
    ref_d = xp.asarray(ref) if ref is not None else None
    ks = clean.wavenumbers / bc.c
    # Free field on the GPU: the replicas of successive bins by phase recurrence (green2d_bins, ~1e-14).
    fast = xp is not np and isinstance(model.propagator, FreeFieldPropagator)
    if fast:
        gen_a = green2d_bins(model._geometry(grid, model.samples)[0], ks, xp)
        gen_p = green2d_bins(model._geometry(grid, ship_p)[0], ks, xp) if method == "coherent" and len(present) else None
        w_rep = model._weights_d()
    for i, kk in enumerate(ks):
        if fast:
            A = model.layout.collapse(next(gen_a), xp)
            if w_rep is not None:
                A = A * w_rep[None, :]
        else:
            A = model(grid, kk)                                                             # (G, N)
        norm = xp.maximum(xp.sum(xp.abs(A) ** 2, axis=1), 1e-30)
        Xi = X_d[:, i, :]                                                                   # (N, K)
        if method in ("bartlett", "bartlett_phat"):
            Y = _phat(Xi) * w_d[:, None] if method == "bartlett_phat" else Xi
            B += xp.abs(A.conj() @ Y) ** 2 / norm[:, None]
        elif method == "coherent":
            if len(present) == 0:
                continue
            Y = A.conj() @ Xi                                                               # (G, K)
            path = next(gen_p) if fast else model.paths(grid, ship_p, kk)                   # (G, Kp)
            r_i = ref_d[i, present_d][None, :]
            Z[:, present_d] += Y[:, present_d] * xp.conj(path) * xp.conj(r_i) / norm[:, None]
            W[:, present_d] += xp.abs(path) ** 2 * xp.abs(r_i) ** 2 / norm[:, None]
        elif method in ("mvdr", "music"):
            R = _csm_batch(Xi, bc.adaptive_window, bc.mvdr_loading)                          # (K, N, N)
            Ac = A.conj()
            if method == "mvdr":
                Ri = xp.linalg.inv(R)
                for k0 in range(0, K, frame_chunk):
                    AR = xp.matmul(Ac[None], Ri[k0:k0 + frame_chunk])                        # (kc, G, N)
                    q = xp.sum(AR * A[None], axis=2).real                                   # (kc, G)
                    B[:, k0:k0 + frame_chunk] += (norm[None, :] / xp.maximum(q, 1e-30)).T
            else:
                lam, vec = xp.linalg.eigh(R)
                En = vec[:, :, : max(R.shape[1] - bc.music_rank, 1)]                           # (K, N, N - r)
                for k0 in range(0, K, frame_chunk):
                    q = xp.sum(xp.abs(xp.matmul(Ac[None], En[k0:k0 + frame_chunk])) ** 2, axis=2)   # (kc, G)
                    B[:, k0:k0 + frame_chunk] += (norm[None, :] / xp.maximum(q, 1e-30)).T
    if method == "coherent":
        B = xp.abs(Z) ** 2
        if bc.coherent_norm == "snr":
            B /= xp.maximum(W, 1e-30)
    if bc.map_scale == "max":
        B /= max(float(B.max()), 1e-30)
    return _asnumpy(B), None if Z is None else _asnumpy(Z)


def _bistatic_map_reference(clean: Spectra, ship: ShipTrack, ref: np.ndarray, grid: np.ndarray, model: ReplicaModel,
                            bc: BistaticConfig, method: str | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """The original per bin, per frame loops of :func:`bistatic_map` (``impl = "reference"``)."""
    method = method or bc.method
    G, K, F = len(grid), len(clean.times), len(clean.freqs)
    B = np.zeros((G, K))
    Z = np.zeros((G, K), dtype=np.complex128) if method == "coherent" else None
    W = np.zeros((G, K)) if method == "coherent" else None
    for i, kk in enumerate(clean.wavenumbers / bc.c):
        A = model(grid, kk)                                            # (G, N)
        norm = np.maximum(np.sum(np.abs(A) ** 2, axis=1), 1e-30)
        Xi = clean.X[:, i, :]                                          # (N, K)
        if method in ("bartlett", "bartlett_phat"):
            Y = _phat(Xi) * clean.weights[:, None] if method == "bartlett_phat" else Xi
            B += np.abs(A.conj() @ Y) ** 2 / norm[:, None]
        elif method == "coherent":
            Y = A.conj() @ Xi                                          # (G, K)
            for k in range(K):
                if not ship.present[k]:
                    continue
                path = model.path(grid, ship.refined[k], kk)
                Z[:, k] += Y[:, k] * np.conj(path) * np.conj(ref[i, k]) / norm
                W[:, k] += np.abs(path) ** 2 * np.abs(ref[i, k]) ** 2 / norm
        elif method in ("mvdr", "music"):
            for k in range(K):
                R = _csm(Xi, k, bc.adaptive_window, bc.mvdr_loading)
                if method == "mvdr":
                    Ri = np.linalg.inv(R)
                    q = np.einsum("gn,nm,gm->g", A.conj(), Ri, A).real
                    B[:, k] += norm / np.maximum(q, 1e-30)
                else:
                    lam, vec = np.linalg.eigh(R)
                    En = vec[:, : max(R.shape[0] - bc.music_rank, 1)]
                    q = np.sum(np.abs(A.conj() @ En) ** 2, axis=1)
                    B[:, k] += norm / np.maximum(q, 1e-30)
    if method == "coherent":
        B = np.abs(Z) ** 2
        if bc.coherent_norm == "snr":
            B /= np.maximum(W, 1e-30)
    if bc.map_scale == "max":
        B /= max(B.max(), 1e-30)
    return B, Z


def exclusion(grid: np.ndarray, ship: ShipTrack, radius: float) -> np.ndarray:
    mask = np.zeros((len(grid), len(ship.present)), dtype=bool)
    for k in np.flatnonzero(ship.present):
        mask[:, k] = np.hypot(*(grid - ship.refined[k]).T) < radius
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
    env: ChartEnvironment,
    sp: Spectra,
    ship: ShipTrack,
    bc: BistaticConfig,
    true_waveform: np.ndarray | None = None,
    components: list[np.ndarray] | tuple = (),
    bodies: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Time domain direct path removal of :func:`run_bistatic`: reference (``bc.ref_mode``, :func:`estimate_reference`;
    *bodies* for ``"iterated"``), image lags, ECA.

    The ship position of every ECA block is interpolated from ``ship.refined`` (and, with
    ``bc.eca_doppler``, the position of every sample).  The same projection is applied to
    *traces* and to each array in *components*.  Returns the cleaned arrays
    (``[traces] + components``) and the reference waveform.
    """
    nb = int(round(bc.eca_block_s * fs))
    n_blocks = int(np.ceil(traces.shape[1] / nb))
    t_blocks = (np.arange(n_blocks) + 0.5) * bc.eca_block_s
    ship_b = np.column_stack([np.interp(t_blocks, sp.times, ship.refined[:, 0]), np.interp(t_blocks, sp.times, ship.refined[:, 1])])
    ship_t = None
    if bc.eca_doppler:
        t_s = np.arange(traces.shape[1]) / fs
        refined = smooth_track(sp.times, ship.refined, ship.present, bc.track_smooth_s)
        ship_t = np.column_stack([np.interp(t_s, sp.times, refined[:, 0]), np.interp(t_s, sp.times, refined[:, 1])])
    if bc.ref_channels == "all":
        chans = np.flatnonzero(sp.weights > 0)
    else:
        sl = layout.slices[bc.ref_channels]
        chans = np.arange(sl.start, sl.stop)[sp.weights[sl] > 0]
    if bc.reference == "data" and bc.ref_mode != "beam":
        if ship_t is None:
            raise ValueError(f"ref_mode {bc.ref_mode!r} needs eca_doppler (per sample delays)")
        waveform = estimate_reference(traces, layout, fs, ship_b, bc, chans, ship_t, bodies=bodies)
    else:
        waveform = make_reference(traces, layout, fs, ship_b, bc.eca_block_s, bc, chans, true_waveform, ship_t=ship_t)
    images = [clutter_images(env, sxy) for sxy in ship_b] if bc.eca_images else []
    outs = eca_remove([traces] + list(components), waveform, layout, fs, ship_b, images, bc.eca_block_s, bc, ship_t=ship_t)
    return outs, waveform


@dataclass
class Diagnostics:
    """Truth inputs for ablations and diagnostics (every one is flagged in the run's provenance)."""

    exact_track: np.ndarray | None = None      # exact ship positions at the frame times (K, 2)
    components: dict[str, np.ndarray] | None = None   # truth components to pass through the same cancellation


def bistatic_provenance(bc: BistaticConfig, ship: ShipTrack, diagnostics: Diagnostics,
                        provenance: Provenance | None) -> Provenance:
    """The caller's *provenance* plus the truth inputs and simulator knowledge of this run."""
    given = []
    if bc.reference == "true_waveform":
        given.append("waveform")
    if ship.exact_track:
        given.append("track")
    if diagnostics.components and bc.removal == "eca":
        given.append("components")
    return (provenance or Provenance()).with_flags(given, resolve(bc.propagator, bc.c).knowledge)


def run_bistatic(
    traces: np.ndarray,
    layout: SensorLayout,
    fs: float,
    env: ChartEnvironment,
    ais: AISReports,
    bc: BistaticConfig,
    grid: np.ndarray | None = None,
    true_waveform: np.ndarray | None = None,
    diagnostics: Diagnostics | None = None,
    floor: float = 0.0,
    provenance: Provenance | None = None,
) -> BistaticRun:
    """The whole chain on one set of traces.

    *true_waveform* is the true illuminator waveform at the emission time (for
    ``reference="true_waveform"``); ``diagnostics.exact_track`` bypasses AIS and refinement;
    ``diagnostics.components`` go through the same ECA projection and their residual energies are
    reported in ``run.eca`` (cancellation depth and echo loss per channel).  Peaks above *floor*
    are returned (threshold later).  *provenance* is what the caller knows about *traces* (synthetic
    or recorded, held out, truth inputs used to build them, e.g. ``"direct_removal"``); the run adds
    its own truth inputs and the propagator's simulator knowledge (``run.provenance``).
    """
    truth = diagnostics or Diagnostics()
    sp = stft(traces, layout, fs, bc)
    model = ReplicaModel(layout, bc.c, propagator=bc.propagator, xp=_xp(bc))
    model.weights = sp.weights
    ship = ship_positions(sp, ais, model, bc, truth=truth.exact_track)
    grid = grid if grid is not None else build_grid(env, bc)

    report = None
    waveform = None
    if bc.removal == "eca":
        comps = list((truth.components or {}).values())
        outs, waveform = eca_stage(traces, layout, fs, env, sp, ship, bc, true_waveform, comps)
        clean_traces = outs[0]
        if comps:
            names = list(truth.components)
            e0 = {k: np.sum(np.asarray(v, dtype=np.float64) ** 2, axis=1) for k, v in truth.components.items()}
            e1 = {k: np.sum(o ** 2, axis=1) for k, o in zip(names, outs[1:])}
            db = {k: 10 * np.log10(np.maximum(e0[k], 1e-30) / np.maximum(e1[k], 1e-30)) for k in names}
            direct = next((k for k in names if k.startswith("direct")), names[0])
            echo = next((k for k in names if k.startswith("scattered")), None)
            report = ECAReport(db[direct], db[echo] if echo else np.full(layout.n_receivers, np.nan))
        clean = stft(clean_traces, layout, fs, bc)
        clean.weights = sp.weights
        ref = reference_spectrum(sp, ship, model, bc, waveform=waveform, fs=fs)
    else:
        given = waveform_spectra(true_waveform, fs, sp) if true_waveform is not None else None
        ref = reference_spectrum(sp, ship, model, bc, given=given)
        clean = remove_direct(sp, ship, model, bc)

    B, Z = bistatic_map(clean, ship, ref, grid, model, bc)
    B[exclusion(grid, ship, bc.exclusion_radius)] = 0.0
    scan = ScanResult(grid, sp.times, B, {}, np.zeros(len(sp.freqs), int))
    dets = extract_peaks(B, grid, sp.times, bc, floor)
    return BistaticRun(sp, clean, ship, ref, grid, scan, Z, dets, report, waveform,
                       bistatic_provenance(bc, ship, truth, provenance))

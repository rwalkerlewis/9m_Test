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
from scipy.ndimage import map_coordinates
from scipy.signal import get_window

from acoustic_sim.harbor.ais import AISReports
from acoustic_sim.harbor.detection import DetectionSet, ScanResult, _nms, channel_weights, green2d, remove_common_mode
from acoustic_sim.harbor.detection import ProcessingConfig
from acoustic_sim.harbor.dispersion import DispersionModel, replica_wavenumbers
from acoustic_sim.harbor.environment import HarborEnvironment
from acoustic_sim.harbor.sensors import SensorLayout

METHODS = ("bartlett", "bartlett_phat", "coherent", "mvdr", "music")
REMOVALS = ("none", "eca", "patch", "adaptive", "patch+adaptive")
REFERENCES = ("data", "oracle")


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
    ref_channels : str
        Channels for the data reference beam: ``"all"`` or a cable name or ``"hydrophones"``.
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
    ref_channels: str = "all"
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

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.removal not in REMOVALS:
            raise ValueError(f"removal must be one of {REMOVALS}, got {self.removal!r}")
        if self.reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, got {self.reference!r}")


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
    """

    def __init__(self, layout: SensorLayout, c: float, min_range: float = 1.0,
                 dispersion: DispersionModel | None = None) -> None:
        self.layout, self.c, self.min_range = layout, c, min_range
        self.samples = layout.sample_points
        self.weights = None
        self.dispersion = dispersion

    def _kr(self, pts: np.ndarray, targets: np.ndarray, k: float) -> np.ndarray:
        dx = pts[:, None, 0] - targets[None, :, 0]
        dy = pts[:, None, 1] - targets[None, :, 1]
        r = np.maximum(np.hypot(dx, dy), self.min_range)
        if self.dispersion is None:
            return k * r
        f = k * self.c / (2.0 * np.pi)
        kk = replica_wavenumbers(self.dispersion, [f], self.c, np.arctan2(np.abs(dy), np.abs(dx)))[0]
        return kk * r

    def __call__(self, pts: np.ndarray, k: float) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        a = self.layout.collapse(green2d(self._kr(pts, self.samples, k)))
        if self.weights is not None:
            a = a * self.weights[None, :]
        return a

    def path(self, pts: np.ndarray, src: np.ndarray, k: float) -> np.ndarray:
        """Free field path factor ``H0^(2)(k |x - s|)`` (P,) from *src* to *pts*."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        return green2d(self._kr(pts, np.atleast_2d(np.asarray(src, dtype=float)), k))[:, 0]


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


def make_reference(traces: np.ndarray, layout: SensorLayout, fs: float, ship_xy: np.ndarray, block_s: float,
                   bc: BistaticConfig, channels: np.ndarray | None = None, oracle_signal: np.ndarray | None = None) -> np.ndarray:
    """Illuminator waveform aligned to the emission time, shape ``(n,)``.

    ``bc.reference == "oracle"``: the true ship waveform.  ``"data"``: delay and sum of the
    *channels* (default: every live channel) toward the ship position of each block of
    *block_s* seconds (*ship_xy* (K_b, 2), one row per block), each channel advanced by its
    range so the sum is at the emission time.  The data reference also contains the echo and
    the multipath, as in every passive radar; that is accepted.
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


@dataclass
class ECAReport:
    """Per channel direct path cancellation measured against truth components."""

    removed_db: np.ndarray          # (N,) energy of the direct component removed [dB]
    echo_lost_db: np.ndarray        # (N,) energy of the scattered component removed [dB]


def eca_remove(
    traces: list[np.ndarray],
    reference: np.ndarray,
    layout: SensorLayout,
    fs: float,
    ship_xy: np.ndarray,
    images: list[np.ndarray],
    block_s: float,
    bc: BistaticConfig,
) -> list[np.ndarray]:
    """Extensive cancellation (least squares projection on delayed copies of the reference).

    For every channel and block of *block_s* seconds, the channel is projected onto the
    reference delayed by the lags around the predicted direct arrival (``|s - r| / c``) and
    around each first order image arrival, each window ``bc.eca_halfwidth_s`` wide.  Only
    those lags are used, so an echo at another bistatic delay survives.  The same projection
    is applied to every array in *traces* (the measurement and, for diagnostics, its truth
    components), which is what makes the cancellation depth measurable.
    """
    n = traces[0].shape[1]
    nb = int(round(block_s * fs))
    half = int(round(bc.eca_halfwidth_s * fs))
    outs = [np.array(x, dtype=np.float64, copy=True) for x in traces]
    rec = layout.receivers
    ref = np.asarray(reference, dtype=np.float64)
    for b in range(len(ship_xy)):
        lo, hi = b * nb, min((b + 1) * nb, n)
        if hi <= lo:
            break
        srcs = [ship_xy[b]] + ([images[b]] if len(images) and len(images[b]) else [])
        pts = np.vstack([np.atleast_2d(p) for p in srcs])
        for i in range(rec.shape[0]):
            lags = np.unique(np.concatenate([
                np.arange(int(round(np.hypot(*(p - rec[i])) / bc.c * fs)) - half, int(round(np.hypot(*(p - rec[i])) / bc.c * fs)) + half + 1)
                for p in pts]))
            lags = lags[(lags >= 0) & (lags < hi)]
            if len(lags) == 0:
                continue
            S = np.stack([np.concatenate([np.zeros(max(L - lo, 0)), ref[max(lo - L, 0):hi - L]]) if lo - L < 0 else ref[lo - L:hi - L]
                          for L in lags], axis=1)                        # (block, lags)
            Q, _ = np.linalg.qr(S)
            for out in outs:
                seg = out[i, lo:hi]
                out[i, lo:hi] = seg - Q @ (Q.T @ seg)
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
            a = model(ship.refined[k][None], kk)[0]
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
            pts = ship.refined[k] + patch
            for i, kk in enumerate(sp.wavenumbers / bc.c):
                A = model(pts, kk).T                                   # (N, 9)
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
    model = ReplicaModel(layout, bc.c, dispersion=bc.dispersion)
    model.weights = sp.weights
    ship = ship_positions(sp, ais, model, bc, truth=truth.ship_xy)
    grid = grid if grid is not None else build_grid(env, bc)

    report = None
    waveform = None
    if bc.removal == "eca":
        nb = int(round(bc.eca_block_s * fs))
        n_blocks = int(np.ceil(traces.shape[1] / nb))
        t_blocks = (np.arange(n_blocks) + 0.5) * bc.eca_block_s
        ship_b = np.column_stack([np.interp(t_blocks, sp.times, ship.refined[:, 0]), np.interp(t_blocks, sp.times, ship.refined[:, 1])])
        if bc.ref_channels == "all":
            chans = np.flatnonzero(sp.weights > 0)
        else:
            sl = layout.slices[bc.ref_channels]
            chans = np.arange(sl.start, sl.stop)[sp.weights[sl] > 0]
        waveform = make_reference(traces, layout, fs, ship_b, bc.eca_block_s, bc, chans, oracle_signal)
        images = [clutter_images(env, sxy) for sxy in ship_b] if bc.eca_images else []
        comps = list((truth.components or {}).values())
        outs = eca_remove([traces] + comps, waveform, layout, fs, ship_b, images, bc.eca_block_s, bc)
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

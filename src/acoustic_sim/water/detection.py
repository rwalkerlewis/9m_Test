"""Detection and localisation: whitened STFT -> matched-field scan -> peaks.

Pipeline
--------
1. **Pre-processing** (:func:`preprocess`): channel QC (dead / noisy DAS channels
   get zero weight), per-cable common-mode removal, Hann STFT, per-channel noise
   floor (time median, then a running median across frequency so narrow lines
   stay visible) and whitening.
2. **Matched-field scan** (:class:`MatchedFieldScanner`): for every analysis bin
   and grid cell the whitened data are correlated with the free-field 2-D
   Green's function replica (``H0^(2)`` for the STFT's ``exp(-i w t)`` convention),
   giving ``M ~ Exp(1)`` under noise.  Stationary coherent interference (distant
   shipping plane waves) is removed per bin by projecting out the dominant
   eigenvectors of the median cross-spectral matrix, which favours components
   that persist over time and so spares the moving target.
3. **Combination**: the per-frame map is the sparse spectral sum
   ``B(x) = sum_f max(M_f(x) - tau, 0)``, so a source's lines add up while
   unremarkable bins contribute nothing.
4. **Extraction** (:func:`extract_detections`): greedy non-maximum suppression per
   frame, local refinement on a fine grid and the matched-field line spectrum
   at the refined position.
5. **CFAR** (:func:`threshold_from_maxima`): the detection threshold is set from
   the distribution of the per-frame map maximum on noise-only data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import get_window

from acoustic_sim.contract import ChartEnvironment, Propagator, Provenance
from acoustic_sim.water.device import CPU, Device, get_device
from acoustic_sim.water.propagation import green2d, resolve  # noqa: F401  (green2d re-exported for the old path)
from acoustic_sim.water.sensors import SensorLayout


@dataclass
class ProcessingConfig:
    """Parameters of the detection chain.

    Attributes
    ----------
    win_s, hop_s : float
        STFT window and hop [s]; the bin spacing is ``1 / win_s``.
    f_lo, f_hi, bin_stride : float, float, int
        Scanned band [Hz] and bin decimation (2 = every other bin).
    common_mode : bool
        Subtract the per-cable median across live DAS channels.
    sensor_groups : tuple[str] | None
        Use only these groups (cable names and/or ``"hydrophones"``); ``None`` uses all.
        Used for ablations.
    qc_low, qc_high : float
        DAS channels with rms below ``qc_low`` or above ``qc_high`` times the
        median are given zero weight.
    floor_smooth_bins : int
        Running-median width across frequency for the noise floor.
    c : float
        Sound speed assumed by the replicas [m/s].
    min_range : float
        Replica ranges are clamped to at least this [m].
    grid_dx, grid_margin : float
        Scan grid spacing and minimum distance to land [m].
    grid_bounds : (x0, x1, y0, y1) | None
        Scan region; ``None`` uses the domain without the absorbing edge layer.
    clip_tau : float
        Per-bin clip level of the sparse spectral sum (in units of the
        ``Exp(1)`` noise distribution).
    interference_rank_max, interference_eig_factor : int, float
        Cap and eigenvalue threshold (relative to the Marchenko-Pastur edge) for
        the per-bin stationary-interference projection.  ``rank_max = 0``
        disables it.
    min_separation, max_peaks, n_candidates : float, int, int
        Non-maximum suppression distance [m], peaks accepted per frame, and map peaks
        examined per frame (candidates beyond ``max_peaks`` leave room for sidelobes
        that are rejected by successive cancellation).
    refine_dx, refine_half : float
        Fine-grid spacing and half-width [m] for position refinement.
    n_lines : int
        Number of strongest matched-field lines stored per detection.
    use_cuda : bool
        Run the whole chain on the GPU (CuPy, single precision); falls back to the CPU
        with a warning when CuPy or a GPU is unavailable.
    propagator : Propagator | None
        Replica model (:class:`acoustic_sim.contract.Propagator`).  ``None`` is the blind
        ``FreeFieldPropagator(c)`` (``water/propagation.py``).  The FDTD scheme's numerical,
        direction dependent wavenumber is ``water.diagnostics.SimulatorAwarePropagator``, built from
        the :class:`~acoustic_sim.contract.Truth`; it is simulator knowledge and the results say so.
    empirical_localiser : EmpiricalLocaliser | None
        Optional refinement against FDTD rendered replicas (``water/localiser.py``; simulator knowledge).
    line_halfwidth : int
        Line enhancement.  If > 0, the matched-field output of every cell is compared
        with its running median over ``2 * line_halfwidth + 1`` neighbouring scanned
        bins before clipping, so smooth broadband leakage (a loud ship's sidelobes) cancels
        and only narrowband lines (a UUV's tonal) survive.  0 disables it.
    """

    win_s: float = 4.0
    hop_s: float = 2.0
    f_lo: float = 12.0
    f_hi: float = 78.0
    bin_stride: int = 2
    common_mode: bool = True
    sensor_groups: tuple[str, ...] | None = None
    qc_low: float = 0.1
    qc_high: float = 5.0
    floor_smooth_bins: int = 13
    c: float = 1495.0
    min_range: float = 5.0
    grid_dx: float = 20.0
    grid_margin: float = 10.0
    grid_bounds: tuple[float, float, float, float] | None = None
    clip_tau: float = 6.0
    interference_rank_max: int = 12
    interference_eig_factor: float = 1.2
    min_separation: float = 80.0
    max_peaks: int = 3
    n_candidates: int = 8
    refine_dx: float = 5.0
    refine_half: float = 30.0
    n_lines: int = 8
    use_cuda: bool = False
    line_halfwidth: int = 0
    propagator: Propagator | None = field(default=None, repr=False)
    empirical_localiser: object | None = field(default=None, repr=False)


# ----------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------

@dataclass
class Whitened:
    """Whitened in-band STFT of all receivers.

    ``Z`` and ``sigma`` live on ``dev`` (GPU or CPU); the small index arrays are host NumPy.
    """

    freqs: np.ndarray          # (F,) all in-band bin centres
    times: np.ndarray          # (K,) frame centres [s]
    Z: object                  # (N, F, K) complex64, X / sigma * weight   (device array)
    sigma: object              # (N, F) noise amplitude floor              (device array)
    weights: np.ndarray        # (N,) 0/1 channel QC (host)
    scan_bins: np.ndarray      # indices into freqs that are scanned
    dev: Device = CPU

    @property
    def weights_d(self):
        return self.dev.xp.asarray(self.weights, dtype=self.dev.fdtype)


def channel_weights(traces: np.ndarray, layout: SensorLayout, cfg: ProcessingConfig) -> np.ndarray:
    """0/1 weight per receiver: zero for dead (silent) or abnormally noisy channels."""
    rms = np.sqrt(np.mean(np.asarray(traces, dtype=np.float64) ** 2, axis=1))
    w = np.ones(layout.n_receivers)
    for name, sl in layout.slices.items():
        r = rms[sl]
        if len(r) == 0:
            continue
        if cfg.sensor_groups is not None and name not in cfg.sensor_groups:
            w[sl] = 0.0
        elif name == "hydrophones":
            w[sl] = r > 0
        else:
            ref = np.median(r[r > 0]) if np.any(r > 0) else 1.0
            w[sl] = (r > cfg.qc_low * ref) & (r < cfg.qc_high * ref)
    return w


def remove_common_mode(traces, layout: SensorLayout, weights: np.ndarray, xp=np):
    """Subtract, per cable, the median across live channels at every sample."""
    out = xp.array(traces, copy=True)
    for name, sl in layout.slices.items():
        if name == "hydrophones":
            continue
        live = weights[sl] > 0
        if live.any():
            idx = xp.asarray(np.flatnonzero(live))
            block = out[sl]
            out[sl] = block - xp.median(block[idx], axis=0)
    return out


def noise_floor(power, cfg: ProcessingConfig, dev: Device):
    """Per-channel noise amplitude floor ``(N, F)`` from STFT power ``(N, F, K)``.

    Median over time (the median of an ``Exp(1)`` variable is ``ln 2``), then a running
    median across frequency so that narrow lines stay above the floor.
    """
    xp = dev.xp
    floor = xp.median(power, axis=2) / np.log(2.0)
    floor = dev.median_filter(floor, size=(1, cfg.floor_smooth_bins), mode="nearest")
    sigma = xp.sqrt(xp.maximum(floor, 1e-30))
    # Dead channels (and channels emptied by a projection) must not get a vanishing floor.
    live = sigma > 1e-12
    ref = xp.median(sigma[live]) if bool(live.any()) else 1.0
    return xp.maximum(sigma, 1e-3 * ref)


def rewhiten(wh: "Whitened", cfg: ProcessingConfig) -> "Whitened":
    """Re-estimate the noise floor from (already cleaned) data and whiten again.

    After known sources are projected out, the original floor is still inflated by their
    energy; dividing by it would bury the weak sources that remain.  Channels with zero
    weight stay zero.
    """
    dev = wh.dev
    X = wh.Z.astype(dev.cdtype) * wh.sigma[:, :, None]
    sigma = noise_floor(dev.xp.abs(X) ** 2, cfg, dev)
    Z = (X / sigma[:, :, None]).astype(np.complex64)
    return Whitened(wh.freqs, wh.times, Z, sigma, wh.weights, wh.scan_bins, dev)


def preprocess(traces: np.ndarray, layout: SensorLayout, fs: float, cfg: ProcessingConfig) -> Whitened:
    """Channel QC, common-mode removal, Hann STFT, per-channel noise floor and whitening."""
    dev = get_device(cfg.use_cuda)
    xp = dev.xp
    w = channel_weights(traces, layout, cfg)
    x = xp.asarray(traces, dtype=dev.fdtype)
    if cfg.common_mode:
        x = remove_common_mode(x, layout, w, xp)

    nper = int(round(cfg.win_s * fs))
    hop = int(round(cfg.hop_s * fs))
    starts = np.arange(0, x.shape[1] - nper + 1, hop)
    idx = xp.asarray(starts[:, None] + np.arange(nper)[None, :])
    window = xp.asarray(get_window("hann", nper), dtype=dev.fdtype)
    spec = xp.fft.rfft(x[:, idx] * window, axis=-1)                    # (N, K, nper/2 + 1)
    f = np.fft.rfftfreq(nper, 1.0 / fs)
    band = (f >= cfg.f_lo) & (f <= cfg.f_hi)
    X = xp.swapaxes(spec[:, :, xp.asarray(np.flatnonzero(band))], 1, 2)  # (N, F, K)

    sigma = noise_floor(xp.abs(X) ** 2, cfg, dev)
    Z = (X / sigma[:, :, None]) * xp.asarray(w, dtype=dev.fdtype)[:, None, None]
    return Whitened(
        freqs=f[band], times=(nper / 2.0 + starts) / fs, Z=Z.astype(np.complex64), sigma=sigma,
        weights=w, scan_bins=np.arange(int(band.sum()))[:: cfg.bin_stride], dev=dev,
    )


def knowledge(cfg: ProcessingConfig) -> list[str]:
    """Simulator knowledge *cfg* brings into a result (its propagator, an empirical localiser)."""
    out = list(resolve(cfg.propagator, cfg.c).knowledge)
    if cfg.empirical_localiser is not None:
        out.append("empirical_replicas")
    return out


def run_provenance(cfg: ProcessingConfig, provenance: Provenance | None) -> Provenance:
    """The caller's *provenance* (default: synthetic data, not held out) plus what *cfg* knows."""
    return (provenance or Provenance()).with_flags(simulator_knowledge=knowledge(cfg))


# ----------------------------------------------------------------------
# Scan
# ----------------------------------------------------------------------

def build_grid(env: ChartEnvironment, cfg: ProcessingConfig, edge: float = 150.0) -> np.ndarray:
    """Scan grid ``(G, 2)`` of water cells at least ``grid_margin`` from land."""
    x0, x1, y0, y1 = cfg.grid_bounds or (env.extent[0] + edge, env.extent[1] - edge,
                                         env.extent[2] + edge, env.extent[3] - edge)
    xs = np.arange(x0, x1 + 1e-6, cfg.grid_dx)
    ys = np.arange(y0, y1 + 1e-6, cfg.grid_dx)
    gx, gy = np.meshgrid(xs, ys)
    g = np.column_stack([gx.ravel(), gy.ravel()])
    return g[env.is_water(g, margin=cfg.grid_margin)]


@dataclass
class ScanResult:
    grid: np.ndarray                      # (G, 2)
    times: np.ndarray                     # (K,)
    B: np.ndarray                         # (G, K) sparse spectral statistic
    projectors: dict[int, np.ndarray]     # bin index -> (N, r) interference eigenvectors
    ranks: np.ndarray                     # (n_scan_bins,)

    @property
    def frame_max(self) -> np.ndarray:
        return self.B.max(axis=0)


class MatchedFieldScanner:
    """Free-field matched-field scanner over a fixed grid for a given sensor layout."""

    def __init__(self, layout: SensorLayout, grid: np.ndarray, cfg: ProcessingConfig) -> None:
        self.layout, self.grid, self.cfg = layout, np.asarray(grid), cfg
        self.dev = get_device(cfg.use_cuda)
        self.rec = self.dev.xp.asarray(layout.receivers, dtype=self.dev.fdtype)
        self.propagator = resolve(cfg.propagator, cfg.c)

    def _geometry(self, xy):
        """``(r, theta)`` of source points *xy* (host or device) to all receivers, both ``(P, N)`` on the device.

        ``theta = atan2(|dy|, |dx|)`` is the propagation angle from the grid axes (for the dispersion model).
        """
        xp = self.dev.xp
        xy = xp.asarray(xy, dtype=self.dev.fdtype)
        dx = xy[:, None, 0] - self.rec[None, :, 0]
        dy = xy[:, None, 1] - self.rec[None, :, 1]
        r = xp.maximum(xp.hypot(dx, dy), self.cfg.min_range)  # the Green's function is singular at r = 0
        return r, xp.arctan2(xp.abs(dy), xp.abs(dx)), xy

    def _replicas(self, geom, wh: Whitened, i: int):
        """Whitened replicas ``(G, N)`` for bin *i*."""
        r, theta, xy = geom
        prop = self.propagator
        if prop.analytic:
            k = prop.wavenumbers([wh.freqs[i]], theta, self.dev.xp, self.dev.fdtype)[0]
            g = green2d(k * r, self.dev.xp)
        else:
            g = prop.greens(self.dev.asnumpy(xy), self.layout.receivers, [wh.freqs[i]], self.dev.xp, self.dev.fdtype)[0]
        a = g / wh.sigma[None, :, i] * wh.weights_d[None, :]
        return a.astype(self.dev.cdtype)

    def _interference(self, Zi, weights: np.ndarray):
        """Dominant eigenvectors of the median cross-spectral matrix (stationary interference)."""
        cfg, xp = self.cfg, self.dev.xp
        if cfg.interference_rank_max <= 0:
            return None
        outer = Zi[:, None, :] * xp.conj(Zi[None, :, :])
        rm = xp.median(outer.real, axis=2) + 1j * xp.median(outer.imag, axis=2)
        lam, vec = xp.linalg.eigh((rm + rm.conj().T) / 2.0)
        n_live, n_frames = float(weights.sum()), Zi.shape[1]
        thr = cfg.interference_eig_factor * (1.0 + np.sqrt(n_live / n_frames)) ** 2 * np.log(2.0)
        rank = min(int(np.sum(self.dev.asnumpy(lam) > thr)), cfg.interference_rank_max)
        return vec[:, -rank:] if rank > 0 else None

    @staticmethod
    def _project(a, U):
        """Apply ``P = I - U U^H`` to row vectors in *a* (G, N)."""
        return a if U is None else a - (a @ U.conj()) @ U.T

    def scan(self, wh: Whitened) -> ScanResult:
        xp, cdt = self.dev.xp, self.dev.cdtype
        tau = self.cfg.clip_tau
        geom = self._geometry(self.grid)
        lines = self.cfg.line_halfwidth > 0
        B = xp.zeros((len(self.grid), len(wh.times)), dtype=np.float32)
        M_all = xp.empty((len(wh.scan_bins), len(self.grid), len(wh.times)), dtype=np.float32) if lines else None
        projectors: dict[int, object] = {}
        ranks = []
        for j, i in enumerate(wh.scan_bins):
            Zi = wh.Z[:, int(i), :].astype(cdt)
            U = self._interference(Zi, wh.weights)
            ranks.append(0 if U is None else U.shape[1])
            if U is not None:
                projectors[int(i)] = U
                Zi = Zi - U @ (U.conj().T @ Zi)
            A = self._project(self._replicas(geom, wh, i), U)
            norm = xp.maximum(xp.sum(xp.abs(A) ** 2, axis=1), 1e-12)
            M = xp.abs(A.conj() @ Zi) ** 2 / norm[:, None]
            if lines:
                M_all[j] = M
            else:
                B += xp.maximum(M - tau, 0.0).astype(np.float32)
        if lines:
            width = 2 * self.cfg.line_halfwidth + 1
            background = self.dev.median_filter(M_all, size=(width, 1, 1), mode="nearest")
            B = xp.maximum(M_all - background - tau, 0.0).sum(axis=0)
        return ScanResult(self.grid, wh.times, self.dev.asnumpy(B), projectors, np.array(ranks))

    def match_frame(
        self,
        wh: Whitened,
        scan: ScanResult,
        xy: np.ndarray,
        frame: int,
        bins: np.ndarray,
        deflate_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        """Matched-field output ``M[b, d]`` (host array) at positions *xy* (D, 2) in one frame for *bins*.

        ``deflate_xy`` (m, 2) are positions of already accepted sources: their replicas
        are projected out of the data and of every candidate replica (successive
        cancellation), so sidelobes of a strong source score ~0 while independent
        sources survive.
        """
        xp, cdt = self.dev.xp, self.dev.cdtype
        geom = self._geometry(xy)
        g_def = self._geometry(deflate_xy) if deflate_xy is not None and len(deflate_xy) else None
        rows = []
        for i in bins:
            U = scan.projectors.get(int(i))
            z = wh.Z[:, int(i), frame].astype(cdt)
            if U is not None:
                z = z - U @ (U.conj().T @ z)
            A = self._project(self._replicas(geom, wh, i), U)
            if g_def is not None:
                q, _ = xp.linalg.qr(self._project(self._replicas(g_def, wh, i), U).T)
                z = z - q @ (q.conj().T @ z)
                A = A - (A @ q.conj()) @ q.T
            norm = xp.maximum(xp.sum(xp.abs(A) ** 2, axis=1), 1e-12)
            rows.append(xp.abs(A.conj() @ z) ** 2 / norm)
        return self.dev.asnumpy(xp.stack(rows)).astype(np.float64)


# ----------------------------------------------------------------------
# Detections
# ----------------------------------------------------------------------

@dataclass
class DetectionSet:
    """Peaks of the per-frame maps (all above a low floor; apply ``score >= threshold``)."""

    frame: np.ndarray                    # (D,) frame index
    t: np.ndarray                        # (D,) frame centre time [s]
    xy: np.ndarray                       # (D, 2) refined position [m]
    xy_coarse: np.ndarray                # (D, 2) grid position [m]
    score: np.ndarray                    # (D,) sparse spectral statistic at the coarse peak
    rank: np.ndarray                     # (D,) 0 = strongest peak of its frame
    lines: list[np.ndarray] = field(default_factory=list)   # per detection (n, 2): [freq, M] strongest first

    def __len__(self) -> int:
        return len(self.t)

    def above(self, threshold: float) -> "DetectionSet":
        m = self.score >= threshold
        return DetectionSet(self.frame[m], self.t[m], self.xy[m], self.xy_coarse[m], self.score[m],
                            self.rank[m], [ln for ln, keep in zip(self.lines, m) if keep])


def _nms(score: np.ndarray, grid: np.ndarray, min_sep: float, max_peaks: int, floor: float) -> list[int]:
    order = np.argsort(score)[::-1]
    keep: list[int] = []
    for g in order:
        if score[g] <= floor or len(keep) >= max_peaks:
            break
        if all(np.hypot(*(grid[g] - grid[k])) >= min_sep for k in keep):
            keep.append(int(g))
    return keep


def extract_detections(
    scanner: MatchedFieldScanner,
    wh: Whitened,
    scan: ScanResult,
    floor: float,
) -> DetectionSet:
    """Peaks of every frame's map with ``score > floor``, refined and annotated with line spectra.

    Peaks are accepted strongest first.  Each further candidate is re-scored after the
    replicas of the already accepted peaks are projected out (successive cancellation);
    candidates that do not survive with a residual score above *floor* are sidelobes and
    are dropped.  Rank-0 scores are the scan statistic; later ranks carry the residual score.
    """
    cfg = scanner.cfg
    tau = cfg.clip_tau
    bins = wh.scan_bins
    h = np.arange(-cfg.refine_half, cfg.refine_half + 1e-6, cfg.refine_dx)
    offsets = np.array([(a, b) for a in h for b in h])

    frame_idx, rank_idx, scores, xys, xy_coarse, lines = [], [], [], [], [], []
    for k in range(scan.B.shape[1]):
        accepted: list[np.ndarray] = []
        cands = _nms(scan.B[:, k], scan.grid, cfg.min_separation, cfg.n_candidates, floor)
        for g in cands:
            if len(accepted) >= cfg.max_peaks:
                break
            xy0 = scan.grid[g]
            deflate = np.array(accepted) if accepted else None
            m0 = scanner.match_frame(wh, scan, xy0[None], k, bins, deflate)[:, 0]
            score = float(scan.B[g, k]) if not accepted else float(np.sum(np.maximum(m0 - tau, 0.0)))
            strong = np.flatnonzero(m0 > tau)
            if len(strong) == 0 or score <= floor:
                continue
            cand = xy0 + offsets
            mf = scanner.match_frame(wh, scan, cand, k, bins[strong], deflate)
            best = int(np.argmax(np.sum(np.maximum(mf - tau, 0.0), axis=0)))
            order = np.argsort(mf[:, best])[::-1][: cfg.n_lines]
            best_xy = cand[best]
            if cfg.empirical_localiser is not None:
                refined = cfg.empirical_localiser.refine(best_xy[None], wh, k)
                best_xy = refined[0]
            accepted.append(best_xy)
            frame_idx.append(k)
            rank_idx.append(len(accepted) - 1)
            scores.append(score)
            xys.append(best_xy)
            xy_coarse.append(xy0)
            lines.append(np.column_stack([wh.freqs[bins[strong][order]], mf[order, best]]))
    if not frame_idx:
        empty = np.empty((0, 2))
        return DetectionSet(np.empty(0, int), np.empty(0), empty, empty, np.empty(0), np.empty(0, int), [])
    kk = np.array(frame_idx)
    return DetectionSet(kk, wh.times[kk], np.array(xys), np.array(xy_coarse), np.array(scores),
                        np.array(rank_idx), lines)


# ----------------------------------------------------------------------
# CFAR
# ----------------------------------------------------------------------

def threshold_from_maxima(maxima: np.ndarray, pfa: float, tail_quantile: float = 0.7) -> float:
    """Detection threshold for a per-frame false-alarm probability *pfa*.

    The upper tail of the noise-only per-frame map maxima is modelled as exponential
    above the *tail_quantile* point, which extrapolates below the sample size.
    """
    maxima = np.asarray(maxima, dtype=float)
    u = float(np.quantile(maxima, tail_quantile))
    excess = maxima[maxima > u] - u
    beta = float(np.mean(excess)) if len(excess) else 1.0
    return u + beta * np.log((1.0 - tail_quantile) / pfa)


# ----------------------------------------------------------------------
# Convenience runner
# ----------------------------------------------------------------------

@dataclass
class DetectionRun:
    """Intermediate products of one pass of the chain."""

    wh: Whitened
    scan: ScanResult
    scanner: MatchedFieldScanner
    dets: DetectionSet
    provenance: Provenance = field(default_factory=Provenance)


def run_detection(
    traces: np.ndarray,
    layout: SensorLayout,
    fs: float,
    env: ChartEnvironment,
    cfg: ProcessingConfig,
    floor: float = 0.0,
    grid: np.ndarray | None = None,
    provenance: Provenance | None = None,
) -> DetectionRun:
    """Pre-process, scan and extract peaks above *floor* (use a low floor, threshold later).

    *provenance* is what the caller knows about the data (synthetic or recorded, held out, oracle
    inputs it used to build *traces*); the run adds the simulator knowledge of *cfg*.
    """
    wh = preprocess(traces, layout, fs, cfg)
    scanner = MatchedFieldScanner(layout, grid if grid is not None else build_grid(env, cfg), cfg)
    scan = scanner.scan(wh)
    return DetectionRun(wh, scan, scanner, extract_detections(scanner, wh, scan, floor), run_provenance(cfg, provenance))

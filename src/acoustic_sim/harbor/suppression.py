"""AIS-aware suppression of known vessels.

For every analysis frame and every AIS vessel that is currently reporting:

1. **Predict** the position from the AIS reports (interpolated, or dead-reckoned
   when causal / after the last report).
2. **Refine** it: a fine Bartlett (matched-field) search in a small window around
   the AIS fix, using the whitened data of that frame.  The AIS error (metres) is
   a large fraction of a wavelength here, so a cancelling replica built from the
   raw AIS position would miss.
3. **Project out**: the replicas of a small patch of points around the refined
   position are stacked, truncated by SVD and removed jointly (one subspace
   projection per frequency bin and frame, so removals of several ships do not
   interfere).  The patch makes the cancellation tolerant to residual position error.
4. **Exclude** cells within ``exclusion_radius`` of each refined position from the
   subsequent map, where ship leakage is largest.

The result is a whitened data set in which the known ships are (mostly) gone and
whatever remains is matched-field processed for unknown sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from acoustic_sim.harbor.ais import AISReports
from acoustic_sim.harbor.detection import (
    DetectionSet,
    MatchedFieldScanner,
    ProcessingConfig,
    ScanResult,
    Whitened,
    build_grid,
    extract_detections,
    green2d,
    preprocess,
    rewhiten,
)
from acoustic_sim.harbor.dispersion import replica_wavenumbers
from acoustic_sim.harbor.environment import HarborEnvironment
from acoustic_sim.harbor.sensors import SensorLayout


@dataclass
class SuppressionConfig:
    """Parameters of the AIS-aware suppression.

    Attributes
    ----------
    refine : bool
        Use the refined position for significant ships (otherwise the AIS prediction).
    refine_half, refine_step : float
        Half-width and spacing [m] of the refinement search window.
    refine_bin_stride : int
        Use every n-th scanned bin for the refinement.
    patch_half : float
        Half-width [m] of the (3 x 3) replica patch removed around each ship.  Keep it
        small against the wavelength (here 20-100 m): every extra dimension removed from
        the data also removes part of any unknown source.
    svd_rtol : float
        Patch-replica singular values below ``svd_rtol * s_max`` are dropped.
    min_significance : float
        A ship is projected out in a frame only if its refined matched-field score
        exceeds the noise expectation by this many standard deviations (``sum(M)``
        over the ``B`` refinement bins has mean ``B`` and std ``sqrt(B)`` in noise).
        Weaker ships are not projected (they are below the noise) but are still
        excluded from the map.
    max_age : float
        Vessels whose latest report is older than this [s] are treated as gone.
    causal : bool
        Dead-reckon from the latest earlier report (real time) instead of interpolating.
    exclusion_radius : float
        Map cells this close [m] to a refined ship position are zeroed.
    mode : str
        ``"model"``: project out free-field replica patches of the refined ship positions.
        ``"adaptive"``: project out the dominant singular vectors of the data in a short
        sliding window around each frame (captures the ship's true multipath channel).
        ``"both"``: model projection followed by the adaptive one.
    adaptive_window : int
        Half-width (in frames) of the sliding window.
    ranks_per_ship : int
        Adaptive projection removes at most ``ranks_per_ship`` dimensions for each
        significant ship in the frame.  The cap is what keeps a (much weaker) unknown
        source, which is also coherent over the window, from being removed with them.
    """

    refine: bool = True
    refine_half: float = 16.0
    refine_step: float = 2.0
    refine_bin_stride: int = 4
    patch_half: float = 1.5
    svd_rtol: float = 0.05
    min_significance: float = 8.0
    max_age: float = 60.0
    causal: bool = False
    exclusion_radius: float = 60.0
    mode: str = "adaptive"
    adaptive_window: int = 3
    ranks_per_ship: int = 2


@dataclass
class ShipEstimates:
    """Per-frame ship positions used for the suppression."""

    ids: list[str]
    times: np.ndarray            # (K,)
    predicted: np.ndarray        # (K, V, 2) AIS prediction
    refined: np.ndarray          # (K, V, 2) after matched-field refinement (= predicted if insignificant)
    present: np.ndarray          # (K, V) bool, AIS vessel is reporting
    significant: np.ndarray | None = None   # (K, V) bool, ship is detectable in the data (project it out)
    significance: np.ndarray | None = None  # (K, V) refinement z-score

    def exclusion_mask(self, grid: np.ndarray, radius: float) -> np.ndarray:
        """Boolean ``(G, K)``: True for grid cells within *radius* of any present ship."""
        mask = np.zeros((len(grid), len(self.times)), dtype=bool)
        for v in range(len(self.ids)):
            for k in np.flatnonzero(self.present[:, v]):
                mask[:, k] |= np.hypot(*(grid - self.refined[k, v]).T) < radius
        return mask


def _replicas(rec, pts: np.ndarray, wh: Whitened, bins, pc: ProcessingConfig):
    """Whitened replicas ``(B, P, N)`` (device) of points *pts* (P, 2) for the bins *bins*.

    *rec* are the receiver positions on the device of *wh*.
    """
    dev = wh.dev
    xp = dev.xp
    p = xp.asarray(pts, dtype=dev.fdtype)
    dx = p[:, None, 0] - rec[None, :, 0]
    dy = p[:, None, 1] - rec[None, :, 1]
    r = xp.maximum(xp.hypot(dx, dy), pc.min_range)
    k = replica_wavenumbers(pc.dispersion, wh.freqs[bins], pc.c, xp.arctan2(xp.abs(dy), xp.abs(dx)), xp, dev.fdtype)
    a = green2d(k * r[None], xp)
    sig = wh.sigma[:, _bin_index(bins, xp)].T                                # (B, N)
    return (a / sig[:, None, :] * wh.weights_d[None, None, :]).astype(dev.cdtype)


def _bin_index(bins: np.ndarray, xp):
    """Slice for regularly spaced *bins* (a cheap view), otherwise an index array."""
    if len(bins) > 1:
        step = int(bins[1] - bins[0])
        if np.all(np.diff(bins) == step):
            return slice(int(bins[0]), int(bins[-1]) + 1, step)
    return xp.asarray(bins)


def _grid_offsets(half: float, step: float) -> np.ndarray:
    h = np.arange(-half, half + 1e-6, step)
    return np.array([(a, b) for a in h for b in h])


def predict_ships(wh: Whitened, ais: AISReports, sc: SuppressionConfig) -> ShipEstimates:
    ids = ais.vessels
    K, V = len(wh.times), len(ids)
    pred = np.zeros((K, V, 2))
    present = np.zeros((K, V), dtype=bool)
    for v, vid in enumerate(ids):
        tr = ais.for_vessel(vid)
        pred[:, v] = tr.predict(wh.times, causal=sc.causal)
        present[:, v] = (wh.times >= tr.t[0]) & (tr.age(wh.times) <= sc.max_age)
    return ShipEstimates(ids, wh.times, pred, pred.copy(), present,
                         significant=present.copy(), significance=np.full((K, V), np.nan))


def refine_ships(
    wh: Whitened, layout: SensorLayout, est: ShipEstimates, pc: ProcessingConfig, sc: SuppressionConfig
) -> ShipEstimates:
    """Fine Bartlett search around each predicted position; also scores each ship's significance."""
    dev = wh.dev
    xp = dev.xp
    rec = xp.asarray(layout.receivers, dtype=dev.fdtype)
    bins = wh.scan_bins[:: sc.refine_bin_stride]
    sel = _bin_index(bins, xp)
    off = _grid_offsets(sc.refine_half, sc.refine_step)
    refined = est.predicted.copy()
    zscore = np.full(est.present.shape, np.nan)
    significant = np.zeros(est.present.shape, dtype=bool)
    n_bins = float(len(bins))
    for k in range(len(wh.times)):
        z = wh.Z[:, sel, k].T.astype(dev.cdtype)                            # (B, N)
        for v in np.flatnonzero(est.present[k]):
            pts = est.predicted[k, v] + off
            A = _replicas(rec, pts, wh, bins, pc)                           # (B, M, N)
            y = xp.einsum("bmn,bn->bm", A.conj(), z)
            norm = xp.maximum(xp.sum(xp.abs(A) ** 2, axis=2), 1e-12)
            power = dev.asnumpy(xp.sum(xp.abs(y) ** 2 / norm, axis=0)).astype(np.float64)
            best = int(np.argmax(power))
            zscore[k, v] = (power[best] - n_bins) / np.sqrt(n_bins)
            significant[k, v] = zscore[k, v] >= sc.min_significance
            if significant[k, v] and sc.refine:
                refined[k, v] = pts[best]
    return ShipEstimates(est.ids, est.times, est.predicted, refined, est.present, significant, zscore)


def project_out_ships(
    wh: Whitened, layout: SensorLayout, est: ShipEstimates, pc: ProcessingConfig, sc: SuppressionConfig
) -> Whitened:
    """Remove the joint replica subspace of all significant ships from every frame and scanned bin.

    One batched SVD per frame covers all bins at once; singular directions below
    ``svd_rtol * s_max`` are masked out rather than sliced, which keeps the batch rectangular.
    """
    dev = wh.dev
    xp = dev.xp
    rec = xp.asarray(layout.receivers, dtype=dev.fdtype)
    bins = wh.scan_bins
    sel = _bin_index(bins, xp)
    patch = _grid_offsets(sc.patch_half, sc.patch_half)                     # 3 x 3 points
    Z = wh.Z.copy()
    for k in range(len(wh.times)):
        vs = np.flatnonzero(est.significant[k])
        if len(vs) == 0:
            continue
        pts = (est.refined[k, vs][:, None, :] + patch[None]).reshape(-1, 2)
        A = _replicas(rec, pts, wh, bins, pc)                               # (B, P, N)
        u, s, _ = xp.linalg.svd(xp.swapaxes(A, 1, 2), full_matrices=False)  # u: (B, N, P)
        uk = u * (s > sc.svd_rtol * s[:, :1])[:, None, :]
        z = Z[:, sel, k].T.astype(dev.cdtype)                               # (B, N)
        coef = xp.einsum("bnp,bn->bp", uk.conj(), z)
        Z[:, sel, k] = (z - xp.einsum("bnp,bp->bn", uk, coef)).T
    return Whitened(wh.freqs, wh.times, Z, wh.sigma, wh.weights, wh.scan_bins, dev)


def project_out_adaptive(wh: Whitened, est: ShipEstimates, sc: SuppressionConfig) -> Whitened:
    """Remove the dominant data-driven subspace of the ships from every frame and scanned bin.

    For each frame, the whitened snapshots of the surrounding ``2 * adaptive_window + 1``
    frames form an ``(N, window)`` matrix per bin.  Its leading left singular vectors, at
    most ``ranks_per_ship`` per significant ship, span the ships' field at that moment
    (whatever its multipath) and are projected out of the centre frame.  One batched SVD
    per frame covers all scanned bins.
    """
    dev = wh.dev
    xp = dev.xp
    sel = _bin_index(wh.scan_bins, xp)
    K = len(wh.times)
    Z = wh.Z.copy()
    for k in range(K):
        r = sc.ranks_per_ship * int(est.significant[k].sum())
        if r == 0:
            continue
        lo, hi = max(0, k - sc.adaptive_window), min(K, k + sc.adaptive_window + 1)
        D = xp.swapaxes(wh.Z[:, sel, lo:hi].astype(dev.cdtype), 0, 1)       # (B, N, W)
        u, _, _ = xp.linalg.svd(D, full_matrices=False)
        uk = u[:, :, : min(r, u.shape[2])]
        z = wh.Z[:, sel, k].T.astype(dev.cdtype)                            # (B, N)
        coef = xp.einsum("bnr,bn->br", uk.conj(), z)
        Z[:, sel, k] = (z - xp.einsum("bnr,br->bn", uk, coef)).T
    return Whitened(wh.freqs, wh.times, Z, wh.sigma, wh.weights, wh.scan_bins, dev)


@dataclass
class AISAwareRun:
    """Products of one AIS-aware detection pass."""

    wh: Whitened                 # whitened data before suppression
    wh_clean: Whitened           # after ship projection and re-whitening with the cleaned noise floor
    ships: ShipEstimates
    scanner: MatchedFieldScanner
    scan: ScanResult             # map with the ship-exclusion zones zeroed
    dets: DetectionSet


def _check_mode(sc: SuppressionConfig) -> None:
    if sc.mode not in ("model", "adaptive", "both"):
        raise ValueError(f"mode must be 'model', 'adaptive' or 'both', got {sc.mode!r}")


def run_ais_aware(
    traces: np.ndarray,
    layout: SensorLayout,
    fs: float,
    env: HarborEnvironment,
    ais: AISReports,
    pc: ProcessingConfig,
    sc: SuppressionConfig | None = None,
    floor: float = 1e18,
    grid: np.ndarray | None = None,
) -> AISAwareRun:
    """Pre-process, suppress the AIS ships, scan and extract peaks above *floor*."""
    sc = sc or SuppressionConfig()
    _check_mode(sc)
    wh = preprocess(traces, layout, fs, pc)
    est = predict_ships(wh, ais, sc)
    est = refine_ships(wh, layout, est, pc, sc)
    wh_clean = wh
    if est.ids:
        if sc.mode in ("model", "both"):
            wh_clean = project_out_ships(wh_clean, layout, est, pc, sc)
        if sc.mode in ("adaptive", "both"):
            wh_clean = project_out_adaptive(wh_clean, est, sc)
        wh_clean = rewhiten(wh_clean, pc)
    scanner = MatchedFieldScanner(layout, grid if grid is not None else build_grid(env, pc), pc)
    scan = scanner.scan(wh_clean)
    if est.ids:
        scan.B[est.exclusion_mask(scanner.grid, sc.exclusion_radius)] = 0.0
    dets = extract_detections(scanner, wh_clean, scan, floor)
    return AISAwareRun(wh, wh_clean, est, scanner, scan, dets)

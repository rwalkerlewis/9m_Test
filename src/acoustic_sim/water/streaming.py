"""Real-time passive detection: the AIS-aware matched field chain, causal per block of frames.

:class:`StreamingPassive` receives the traces in blocks as they are recorded (:meth:`push`) and
returns, for every new frame, the strongest map peaks.  It is the chain of ``suppression.run_ais_aware``
with every step that looked at the whole record made causal at the granularity of a block of
``StreamConfig.block_frames`` frames (8 frames, 2 s, on the 0.5 m recordings of ``examples/harbor/05_detection/harbor_stream.py``;
1 frame on the 5 m harbor): a frame's result uses no sample recorded after the end of its block, but the
steps below use the whole block, so an early frame of a block sees the later frames of the same block:

* channel QC on the last ``history_s`` seconds of samples up to the end of the block, and the per cable
  common mode (median over the live channels) with those weights;
* the noise floor (median over time) over the last ``history_s`` seconds of frames up to the end of the
  block, before and after the ships are projected out (the past spectra are kept without channel weights;
  the current weights are applied when they are used);
* AIS positions dead reckoned from the latest report at or before the frame (``causal``), refined
  per frame as before;
* the adaptive ship projection over the ``2 * adaptive_window + 1`` frames **ending** at the frame;
* the stationary interference projector from the last ``history_s`` seconds, refreshed every
  ``interference_every`` frames.

Frames are processed ``block_frames`` at a time (the replicas of a bin are built once per block),
so the latency of a frame is the wait for the rest of its block plus the compute time of the
block; both are reported per frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.signal import get_window

from acoustic_sim.contract import ChartEnvironment
from acoustic_sim.water.ais import AISReports
from acoustic_sim.water.detection import (
    MatchedFieldScanner,
    ProcessingConfig,
    Whitened,
    _nms,
    build_grid,
    channel_weights,
    noise_floor,
    remove_common_mode,
)
from acoustic_sim.water.device import get_device
from acoustic_sim.water.propagation import FreeFieldPropagator, green2d, green2d_bins, resolve
from acoustic_sim.water.sensors import SensorLayout
from acoustic_sim.water.suppression import SuppressionConfig, predict_ships, project_out_ships, refine_ships


@dataclass
class StreamConfig:
    """Parameters of the streaming wrapper (the chain's own are in ``ProcessingConfig`` / ``SuppressionConfig``).

    ``block_frames``: frames processed together; ``history_s``: past data used by the QC, the noise
    floors and the interference projector; ``warmup_s``: data before the first output;
    ``interference_every``: frames between projector updates; ``max_peaks`` / ``min_separation``:
    peaks reported per frame; ``full_patterns``: with ``push(explain=...)``, the predicted map of every position over
    the whole scan grid (``FrameResult.patterns``; the tracker needs only the response between the block's peaks and
    tracks, ``pair_R``, which is always computed).
    """

    block_frames: int = 4
    history_s: float = 10.0
    warmup_s: float = 2.0
    interference_every: int = 16
    max_peaks: int = 3
    min_separation: float = 20.0
    full_patterns: bool = False


@dataclass
class FrameResult:
    t: float                          # frame centre [s]
    xy: np.ndarray                    # (P, 2) peaks, strongest first
    score: np.ndarray                 # (P,)
    ships: dict[str, np.ndarray]      # refined position of every reporting ship
    compute_s: float                  # compute time of the frame's block
    latency_s: float                  # last sample of the frame recorded -> result available
    B: np.ndarray | None = field(default=None, repr=False)
    spectrum: np.ndarray | None = field(default=None, repr=False)   # with keep_spectra: band spectrum (N, F) of the frame
    weights: np.ndarray | None = field(default=None, repr=False)    # ... and the channel weights it was formed with
    refs: dict | None = field(default=None, repr=False)             # echo chain: reference spectrum (F,) per illuminator
    patterns: dict | None = field(default=None, repr=False)         # passive chain: predicted map (G,) per track, 1 at it
    pair_cells: np.ndarray | None = field(default=None, repr=False) # passive chain: the block's peak and track cells (C, 2)
    pair_R: np.ndarray | None = field(default=None, repr=False)     # ... the response at each to a source at each (C, C)


class StreamingPassive:
    """Causal frame by frame passive detection on a fixed scan grid (see the module docstring)."""

    def __init__(self, layout: SensorLayout, fs: float, env: ChartEnvironment, pc: ProcessingConfig,
                 sc: SuppressionConfig | None = None, stc: StreamConfig | None = None, grid: np.ndarray | None = None,
                 keep_maps: bool = False, keep_spectra: bool = False) -> None:
        self.layout, self.fs, self.env = layout, float(fs), env
        self.pc, self.stc = pc, stc or StreamConfig()
        self.sc = replace(sc or SuppressionConfig(), causal=True)    # AIS dead reckoned from past reports only
        self.keep_maps, self.keep_spectra = keep_maps, keep_spectra
        self.dev = get_device(pc.use_cuda)
        xp = self.dev.xp
        self.nper, self.hop = int(round(pc.win_s * fs)), int(round(pc.hop_s * fs))
        self.window = xp.asarray(get_window("hann", self.nper), dtype=self.dev.fdtype)
        f = np.fft.rfftfreq(self.nper, 1.0 / fs)
        self.band = np.flatnonzero((f >= pc.f_lo) & (f <= pc.f_hi))
        self.freqs = f[self.band]
        self.scan_bins = np.arange(len(self.band))[:: pc.bin_stride]
        self.scanner = MatchedFieldScanner(layout, grid if grid is not None else build_grid(env, pc), pc)
        self.grid = self.scanner.grid
        self._geom = self.scanner._geometry(self.grid)
        self._fast = isinstance(resolve(pc.propagator, pc.c), FreeFieldPropagator)
        if self._fast:
            self._init_fast_greens()
        self.H = max(int(round(self.stc.history_s / pc.hop_s)), 2 * self.sc.adaptive_window + 1)
        self.n_hist = int(round(self.stc.history_s * fs))
        N = layout.n_receivers
        self.raw = np.zeros((N, 0), dtype=np.float64)            # recent samples (QC, frame overlap)
        self.t_samples = 0                                        # samples received
        self.next_start = 0                                       # sample index of the next frame start
        self.X = []                                               # spectra of past frames, (N, F) device arrays
        self.Xc = []                                              # same after the ship projection (no channel weights)
        self.times: list[float] = []
        self.U: dict[int, object] = {}
        self.frames_done = 0

    # ------------------------------------------------------------------

    def push(self, samples: np.ndarray, reports: AISReports | None = None,
             explain: dict[str, np.ndarray] | None = None) -> list[FrameResult]:
        """Append *samples* (N, n) and process every complete block.  *reports* (optional): AIS reports
        received so far, whose ships are then refined, projected out and excluded; without them every
        radiator, ships included, is located from the traces alone.  *explain* (track id -> predicted position):
        the map the array would produce for a point source at each position alone (the layout's response through
        the same replicas, whitening weights and interference projection as the data map, summed over the scanned
        bins), normalised to 1 at that position's cell, is returned with every frame of the block
        (``FrameResult.patterns``)."""
        if reports is None:
            reports = _NO_REPORTS
        self.explain = dict(explain or {})
        self.raw = np.concatenate([self.raw, np.asarray(samples, dtype=np.float64)], axis=1)
        self.t_samples += samples.shape[1]
        out = []
        need = self.nper + (self.stc.block_frames - 1) * self.hop
        while self.t_samples - self.next_start >= need:
            out += self._block(reports)
        keep = max(self.n_hist, self.t_samples - self.next_start)
        if self.raw.shape[1] > keep:
            self.raw = self.raw[:, -keep:]
        return out

    def _samples(self, start: int, stop: int) -> np.ndarray:
        off = self.t_samples - self.raw.shape[1]
        return self.raw[:, start - off:stop - off]

    def _block(self, reports: AISReports) -> list[FrameResult]:
        t0 = time.perf_counter()
        xp, dev, pc, sc, stc = self.dev.xp, self.dev, self.pc, self.sc, self.stc
        b = stc.block_frames
        starts = self.next_start + self.hop * np.arange(b)
        stop = starts[-1] + self.nper
        self.next_start += b * self.hop
        lo = max(stop - self.n_hist, self.t_samples - self.raw.shape[1])
        w = channel_weights(self._samples(lo, stop), self.layout, pc)              # QC on the recent past
        x = xp.asarray(self._samples(starts[0], stop), dtype=dev.fdtype)
        if pc.common_mode:
            x = remove_common_mode(x, self.layout, w, xp)
        idx = xp.asarray((starts - starts[0])[:, None] + np.arange(self.nper)[None, :])
        spec = xp.fft.rfft(x[:, idx] * self.window, axis=-1)[:, :, xp.asarray(self.band)]   # (N, b, F)
        times = (starts + self.nper / 2.0) / self.fs
        for k in range(b):
            self.X.append(spec[:, k, :].astype(dev.cdtype))
            self.times.append(float(times[k]))
        self.X, self.times = self.X[-self.H:], self.times[-self.H:]
        if times[-1] < stc.warmup_s:
            self.Xc += [self.X[-b + k] for k in range(b)]
            self.Xc = self.Xc[-self.H:]
            return []

        wd = xp.asarray(w, dtype=dev.fdtype)
        Xh = xp.stack(self.X, axis=2)                                               # (N, F, H)
        sigma = noise_floor(xp.abs(Xh) ** 2, pc, dev)
        Zh = (Xh / sigma[:, :, None]) * wd[:, None, None]
        Zb = Zh[:, :, -b:]
        wh = Whitened(self.freqs, times, Zb.astype(np.complex64), sigma, w, self.scan_bins, dev)

        # AIS: causal prediction from the reports received by each frame, refinement, projection.
        rep = _reports_until(reports, times[-1])
        if len(rep):
            est = refine_ships(wh, self.layout, predict_ships(wh, rep, sc), pc, sc)
            if sc.mode in ("model", "both"):
                wh = project_out_ships(wh, self.layout, est, pc, sc)
                Zh = xp.concatenate([Zh[:, :, :-b], wh.Z.astype(dev.cdtype)], axis=2)
            if sc.mode in ("adaptive", "both"):
                Zb = self._adaptive(Zh, est, b)
            else:
                Zb = Zh[:, :, -b:]
        else:
            est = None
            Zb = Zh[:, :, -b:]
        # Store the cleaned spectra without the channel weights: the cleaned values of the live channels and the
        # raw spectra of the channels this block's QC zeroed (a later block may keep them), so the weights of the
        # current block apply to every stored frame alike.
        live = (wd > 0)[:, None]
        for k in range(b):
            self.Xc.append(xp.where(live, Zb[:, :, k] * sigma, self.X[-b + k]).astype(dev.cdtype))
        self.Xc = self.Xc[-self.H:]
        Xch = xp.stack(self.Xc, axis=2) * wd[:, None, None]
        sigma2 = noise_floor(xp.abs(Xch) ** 2, pc, dev)                             # re-whitening on the cleaned past
        Zc = (Xch / sigma2[:, :, None]).astype(dev.cdtype)                          # (N, F, H)
        wc = Whitened(self.freqs, np.array(self.times), Zc, sigma2, w, self.scan_bins, dev)

        # Scan the block's frames.
        bins = [int(i) for i in self.scan_bins]
        if not self.U:
            self.U = self._interference_all(Zc, w, bins)                               # first block: every bin
        else:                                                                         # then a slice per block, all bins
            per = int(np.ceil(len(bins) * b / stc.interference_every))                # every interference_every frames
            c0 = (self.frames_done // b * per) % len(bins)
            sl = (bins + bins)[c0:c0 + per]
            new = self._interference_all(Zc, w, sl)
            for i in sl:
                if i in new:
                    self.U[i] = new[i]
                else:
                    self.U.pop(i, None)                                               # no interference above the floor now
        B = xp.zeros((len(self.grid), b), dtype=np.float32)
        pat_ids = list(getattr(self, "explain", {}) or {})
        pat_cells = [int(np.argmin(np.hypot(*(self.grid - np.asarray(self.explain[k], float)).T))) for k in pat_ids]
        Pat = xp.zeros((len(self.grid), len(pat_ids)), dtype=np.float32) if pat_ids and stc.full_patterns else None
        greens = None
        if self._fast:
            greens = self._greens()
            greens.send(None)                                                         # prime: each send(scale) returns a bin's replicas
        for i in self.scan_bins:
            i = int(i)
            Zi = Zc[:, i, :]
            U = self.U.get(i)
            Zbi = Zi[:, -b:]
            if U is not None:
                Zbi = Zbi - U @ (U.conj().T @ Zbi)
            if greens is not None:
                A = greens.send((wd / sigma2[:, i]).astype(dev.fdtype))
            else:
                A = self.scanner._replicas(self._geom, wc, i)
            A = self.scanner._project(A, U)
            norm = xp.maximum(xp.sum(xp.abs(A) ** 2, axis=1), 1e-12)
            M = xp.abs(A.conj() @ Zbi) ** 2 / norm[:, None]
            B += xp.maximum(M - pc.clip_tau, 0.0).astype(np.float32)
            if Pat is not None:                                                       # predicted maps: the beam pattern
                cells = xp.asarray(pat_cells)
                Ag = A[cells]                                                         # (J, N)
                Pat += (xp.abs(A.conj() @ Ag.T) ** 2 / (norm[:, None] * norm[cells][None, :])).astype(np.float32)
        B = dev.asnumpy(B).astype(np.float64)
        patterns = None
        if Pat is not None:
            Pn = dev.asnumpy(Pat).astype(np.float64)
            patterns = {k: Pn[:, j] / max(Pn[pat_cells[j], j], 1e-30) for j, k in enumerate(pat_ids)}
        ships: list[dict[str, np.ndarray]] = [{} for _ in range(b)]
        if est is not None:
            for v, vid in enumerate(est.ids):
                for k in np.flatnonzero(est.present[:, v]):
                    B[np.hypot(*(self.grid - est.refined[k, v]).T) < sc.exclusion_radius, k] = 0.0
                    ships[k][vid] = est.refined[k, v]
        peak_cells = [_nms(B[:, k], self.grid, stc.min_separation, stc.max_peaks, 0.0) for k in range(b)]
        pair_cells = np.unique(np.concatenate([np.asarray(g, int) for g in peak_cells] + [np.asarray(pat_cells, int)]))
        pair_R = self._pair_response(pair_cells, wc, wd, sigma2, bins) if len(pair_cells) else np.zeros((0, 0))
        self.frames_done += b
        compute = time.perf_counter() - t0
        spectra = dev.asnumpy(spec).astype(np.complex64) if self.keep_spectra else None
        out = []
        for k in range(b):
            g = _nms(B[:, k], self.grid, stc.min_separation, stc.max_peaks, 0.0)
            frame_end = (starts[k] + self.nper) / self.fs
            out.append(FrameResult(float(times[k]), self.grid[g], B[g, k], ships[k], compute,
                                   float(stop / self.fs - frame_end + compute), B[:, k].copy() if self.keep_maps else None,
                                   spectra[:, k, :] if spectra is not None else None, w if spectra is not None else None,
                                   patterns=patterns, pair_cells=self.grid[pair_cells], pair_R=pair_R))
        return out

    def _pair_response(self, cells: np.ndarray, wc: Whitened, wd, sigma2, bins: list[int]) -> np.ndarray:
        """The array's response at each of the grid *cells* to a point source at each other, through the same
        replicas, whitening weights and interference projection as the data map: ``R[a, b]`` the mean over the scanned
        bins of ``|a_a^H a_b|^2 / (|a_a|^2 |a_b|^2)`` (1 on the diagonal), the map a source at ``b`` alone produces at
        ``a`` relative to its own peak."""
        xp, dev = self.dev.xp, self.dev
        geom = self.scanner._geometry(self.grid[cells])
        R = xp.zeros((len(cells), len(cells)), dtype=np.float64)
        for i in bins:
            A = self.scanner._replicas(geom, wc, i)
            A = self.scanner._project(A, self.U.get(i))
            n = xp.maximum(xp.sum(xp.abs(A) ** 2, axis=1), 1e-12)
            R += (xp.abs(A.conj() @ A.T) ** 2 / (n[:, None] * n[None, :])).astype(np.float64)
        return dev.asnumpy(R / max(len(bins), 1))

    def _init_fast_greens(self) -> None:
        """Tables for :meth:`_greens`: the free field replicas of the scanned bins in order, from the ranges
        of every grid cell to every sample point of the layout (three per strain channel)."""
        xp, dev = self.dev.xp, self.dev
        r = self._geom[0]                                                            # (G, S) device, >= min_range
        c = resolve(self.pc.propagator, self.pc.c).c
        self._k = 2.0 * np.pi * self.freqs[self.scan_bins] / c
        self._r = r
        self._inv_sqrt_r = (1.0 / xp.sqrt(r)).astype(dev.fdtype)
        self._inv_r = (1.0 / r).astype(dev.fdtype)
        dk = float(self._k[1] - self._k[0]) if len(self._k) > 1 else 0.0
        self._step = xp.exp(-1j * dk * r).astype(dev.cdtype)
        near = r < 8.0 / float(self._k[0])                                            # where the asymptotic series is not used
        self._near = xp.nonzero(near)
        self._strain = self.layout.has_strain
        self._ones = xp.ones(r.shape[1], dtype=dev.fdtype)
        if self._strain and xp is not np:
            sidx, coef, owner, ocoef = _collapse_tables(self.layout)
            self._sidx, self._coef = xp.asarray(sidx.ravel()), xp.asarray(coef.ravel())
            g_n, s_n = self._near
            self._near_own = (g_n, xp.asarray(owner)[s_n], xp.asarray(ocoef)[s_n])

    def _greens(self, exact_every: int = 16):
        """Coroutine over the scanned bins, in order: ``send(s)`` with the per channel scale *s* (N,)
        (weight over noise floor) returns that bin's replicas ``collapse(green2d(k r)) * s`` (G, N).  The phase
        advances by one complex multiply per bin (re-anchored to the exact phase every *exact_every* bins),
        amplitude and asymptotic series come from cached ``1 / sqrt(r)`` and ``1 / r`` (one fused GPU
        kernel), and the exact Hankel function replaces them where ``k r < 8``.  Without strain cables the
        sample points are the channels and the scale is applied inside the kernel; with them the kernel
        evaluates every sample point unscaled, and the second difference (``SensorLayout.collapse``) and the
        scale follow."""
        xp, dev = self.dev.xp, self.dev
        gpu = xp is not np
        kern = (_strain_greens_kernel() if self._strain else _greens_kernel()) if gpu else None
        G, N = self._r.shape[0], self.layout.n_receivers
        phase = None
        out = None
        scale = yield None
        for j, k in enumerate(self._k):
            if j % exact_every == 0:
                phase = xp.exp(-1j * (k * self._r - np.pi / 4.0)).astype(dev.cdtype)
            sc = scale.astype(dev.fdtype)[None, :]
            amp = np.float32(np.sqrt(2.0 / (np.pi * k)))
            rn = self._r[self._near]
            if gpu and self._strain:
                # One thread per (cell, channel): its sample points' replicas combined as SensorLayout.collapse
                # does, times the scale, and their phases advanced; then the exact Hankel function where k r < 8.
                out = xp.empty((G, N), dtype=dev.cdtype)
                kern(self._step, self._inv_sqrt_r, self._inv_r, self._sidx, self._coef, sc[0], np.float32(k), amp,
                     np.int32(phase.shape[1]), np.int32(N), out, phase)
                ir = 1.0 / rn
                asym = (amp / xp.sqrt(rn) * (1.0 - 9.0 * ir ** 2 / (128.0 * k * k) + 1j * ir / (8.0 * k))
                        * xp.exp(-1j * (k * rn - np.pi / 4.0)))
                g_n, own, oc = self._near_own
                delta = xp.where(k * rn < 8.0, green2d(k * rn, xp) - asym, 0.0) * (oc * sc[0][own])
                ov = out.view(dev.fdtype)                                              # (G, 2N): real, imag interleaved
                xp.add.at(ov, (g_n, 2 * own), delta.real.astype(dev.fdtype))
                xp.add.at(ov, (g_n, 2 * own + 1), delta.imag.astype(dev.fdtype))
                scale = yield out
                continue
            s_pts = self._ones[None, :] if self._strain else sc
            if kern is not None:
                out = xp.empty_like(phase)
                kern(phase, self._step, self._inv_sqrt_r, self._inv_r, s_pts, np.float32(k), amp, out, phase)
            else:
                series = 1.0 + 1j * self._inv_r / (8.0 * k) - 9.0 * self._inv_r ** 2 / (128.0 * k * k)
                out = (amp * self._inv_sqrt_r * series * phase * s_pts).astype(dev.cdtype)
                phase = (phase * self._step).astype(dev.cdtype)
            exact = green2d(k * rn, xp) * s_pts[0][self._near[1]]
            out[self._near] = xp.where(k * rn < 8.0, exact, out[self._near]).astype(dev.cdtype)
            if self._strain:
                out = (self.layout.collapse(out, xp) * sc).astype(dev.cdtype)
            scale = yield out

    def _interference_all(self, Zc, w: np.ndarray, bins: list[int], chunk: int = 16) -> dict:
        """``MatchedFieldScanner._interference`` for every scanned bin, with the eigendecompositions batched
        over bins (one call per *chunk* bins instead of one per bin)."""
        pc, xp = self.pc, self.dev.xp
        out: dict[int, object] = {}
        if pc.interference_rank_max <= 0:
            return out
        n_live, n_frames = float(w.sum()), Zc.shape[2]
        thr = pc.interference_eig_factor * (1.0 + np.sqrt(n_live / n_frames)) ** 2 * np.log(2.0)
        rms = []
        for c0 in range(0, len(bins), chunk):                                          # the medians in chunks (memory)
            Zs = xp.moveaxis(Zc[:, xp.asarray(bins[c0:c0 + chunk]), :], 1, 0)          # (b, N, H)
            outer = Zs[:, :, None, :] * xp.conj(Zs[:, None, :, :])                     # (b, N, N, H)
            rms.append(xp.median(outer.real, axis=3) + 1j * xp.median(outer.imag, axis=3))
        rm = xp.concatenate(rms, axis=0)
        lam, vec = xp.linalg.eigh((rm + xp.conj(xp.swapaxes(rm, 1, 2))) / 2.0)          # one batched call
        ranks = np.minimum(np.sum(self.dev.asnumpy(lam) > thr, axis=1), pc.interference_rank_max)
        for j, i in enumerate(bins):
            if ranks[j] > 0:
                out[i] = vec[j][:, -int(ranks[j]):]
        return out

    def _adaptive(self, Zh, est, b: int):
        """``project_out_adaptive`` with the window of each frame ending at that frame (causal)."""
        xp, dev, sc = self.dev.xp, self.dev, self.sc
        sel = xp.asarray(self.scan_bins)
        H = Zh.shape[2]
        out = Zh[:, :, -b:].copy()
        width = 2 * sc.adaptive_window + 1
        for k in range(b):
            r = sc.ranks_per_ship * int(est.significant[k].sum())
            if r == 0:
                continue
            c = H - b + k
            D = xp.swapaxes(Zh[:, sel, max(0, c - width + 1):c + 1].astype(dev.cdtype), 0, 1)   # (B, N, W)
            # Leading left singular vectors of D from the (W, W) Gram matrix: u = D v / s.
            lam, v = xp.linalg.eigh(xp.einsum("bnw,bnx->bwx", D.conj(), D))
            r = min(r, D.shape[2])
            lam, v = lam[:, ::-1][:, :r], v[:, :, ::-1][:, :, :r]
            uk = xp.einsum("bnw,bwr->bnr", D, v) / xp.sqrt(xp.maximum(lam, 1e-30))[:, None, :]
            z = Zh[:, sel, c].T.astype(dev.cdtype)
            coef = xp.einsum("bnr,bn->br", uk.conj(), z)
            out[:, sel, k] = (z - xp.einsum("bnr,br->bn", uk, coef)).T
        return out


_NO_REPORTS = AISReports(np.array([], dtype=str), np.empty(0), np.empty((0, 2)), np.empty(0), np.empty(0))


def _reports_until(rep: AISReports, t: float) -> AISReports:
    m = rep.t <= t
    return AISReports(rep.vessel[m], rep.t[m], rep.xy[m], rep.sog[m], rep.cog[m])


_KERNEL = None


def _greens_kernel():
    """Fused CuPy kernel: replicas of one bin at every sample point from the running phase (times the per point
    scale), then the phase advanced in place."""
    global _KERNEL
    if _KERNEL is None:
        import cupy

        _KERNEL = cupy.ElementwiseKernel(
            "complex64 ph, complex64 step, float32 isr, float32 ir, float32 sc, float32 k, float32 amp",
            "complex64 g, complex64 ph_out",
            """
            complex<float> series(1.0f - 9.0f * ir * ir / (128.0f * k * k), ir / (8.0f * k));
            g = ph * series * (amp * isr * sc);
            ph_out = ph * step;
            """,
            "stream_greens")
    return _KERNEL


def _collapse_tables(layout: SensorLayout):
    """Per channel its (up to three) sample points and the weights of ``SensorLayout.collapse`` (1, -2, 1 for a
    strain channel, 1 for a pressure channel or hydrophone), and per sample point its channel and weight."""
    sidx, coef, start = [], [], 0
    for c in layout.cables:
        for ch in range(c.n_channels):
            if c.is_strain:
                sidx.append([start + 3 * ch, start + 3 * ch + 1, start + 3 * ch + 2])
                coef.append([1.0, -2.0, 1.0])
            else:
                sidx.append([start + ch] * 3)
                coef.append([1.0, 0.0, 0.0])
        start += c.n_samples
    for h in range(layout.n_hydrophones):
        sidx.append([start + h] * 3)
        coef.append([1.0, 0.0, 0.0])
    sidx, coef = np.array(sidx, dtype=np.int32), np.array(coef, dtype=np.float32)
    owner = np.empty(layout.n_samples, dtype=np.int64)
    ocoef = np.zeros(layout.n_samples, dtype=np.float32)
    for chn in range(len(sidx)):
        for j in range(3):
            if coef[chn, j] != 0.0:
                owner[sidx[chn, j]], ocoef[sidx[chn, j]] = chn, coef[chn, j]
    return sidx, coef, owner, ocoef


_STRAIN_KERNEL = None


def _strain_greens_kernel():
    """Fused CuPy kernel of :meth:`StreamingPassive._greens` on a layout with strain cables: out[g, c] = sc[c]
    sum_j coef[c, j] G(g, s_cj) from the running phase of the sample points, whose phases it then advances
    (every sample point belongs to one channel, so each phase is advanced once)."""
    global _STRAIN_KERNEL
    if _STRAIN_KERNEL is None:
        import cupy

        _STRAIN_KERNEL = cupy.ElementwiseKernel(
            "raw complex64 step, raw float32 isr, raw float32 ir, raw int32 sidx, raw float32 coef, raw float32 sc, "
            "float32 k, float32 amp, int32 S, int32 N",
            "complex64 out, raw complex64 ph",
            """
            int g = i / N, c = i % N;
            complex<float> acc(0.0f, 0.0f);
            for (int j = 0; j < 3; j++) {
                float cf = coef[3 * c + j];
                if (cf == 0.0f) continue;
                long long q = (long long)g * S + sidx[3 * c + j];
                float u = ir[q];
                complex<float> series(1.0f - 9.0f * u * u / (128.0f * k * k), u / (8.0f * k));
                complex<float> p = ph[q];
                acc += p * series * (cf * amp * isr[q]);
                ph[q] = p * step[q];
            }
            out = acc * sc[c];
            """, "stream_strain_greens")
    return _STRAIN_KERNEL

# ----------------------------------------------------------------------
# Tracks from the passive locations
# ----------------------------------------------------------------------

@dataclass
class TrackerConfig:
    """Parameters of a :class:`RadiatorTracker` (radiator tracks and echo tracks each have their own).

    Every track is a constant velocity Kalman filter (state: position and velocity).  ``accel``: the process noise, a
    white acceleration of this standard deviation [m/s^2]; ``meas_scale``, ``meas_floor``: the measurement noise of a
    fix, ``meas_scale`` times the standard deviation of a Gaussian with the half maximum widths of the map peak that
    produced it (along its two axes; width / 2.355) with a floor of ``meas_floor`` [m] per axis; ``init_speed``: the
    standard deviation [m/s] of the velocity of a new track; ``gate_probability``: a peak joins a track when its
    Mahalanobis distance to the track's prediction is within the chi squared quantile of this probability (2 degrees
    of freedom); ``confirm``: gated fixes within ``confirm_window_s`` that confirm a track; ``drop_s``: a track without
    a gated fix for this long is dropped; ``lost_s``: a dropped confirmed track is kept this long as lost (for the
    picture); ``static_s``: a confirmed track whose velocity is within its covariance of zero (Mahalanobis distance
    within the ``gate_probability`` quantile) for this long is static; ``shape_radius``: the region [m] the half
    maximum widths are measured in.

    ``explain_fraction`` (radiator tracks, with the predicted maps of :class:`StreamingPassive`): a peak is explained by
    a confirmed track (a sidelobe of it) when the map the array would produce for that track alone, scaled to the
    track's own peak, exceeds this fraction of the data map at the peak; an explained peak starts no track and is no fix
    for another track.  ``illuminator_margin_db``: a confirmed radiator track whose peak is this far above the map's
    noise floor (the median of the frame's map over the cells it is above zero) is an illuminator of the echo chain.
    ``echo_coherence``: a radiator track about to confirm whose least squares source spectrum is coherent with an
    illuminator's above this level (over its frames, mean over the band) and which sits, within the gate, on an echo map
    peak of that illuminator is an echo and is never confirmed (``HarborPicture``).  ``exclusion`` [m] (echo tracks): a
    track does not confirm within this distance of a stronger confirmed track (0: off).
    """

    accel: float = 0.05
    meas_scale: float = 1.0
    meas_floor: float = 1.0
    init_speed: float = 3.0
    gate_probability: float = 0.99
    confirm: int = 6
    confirm_window_s: float = 2.0
    drop_s: float = 2.0
    static_s: float = 10.0
    lost_s: float = 4.0
    shape_radius: float = 20.0
    explain_fraction: float = 0.5
    illuminator_margin_db: float = 10.0
    echo_coherence: float = 0.7
    exclusion: float = 0.0

    @property
    def gate_chi2(self) -> float:
        """The chi squared quantile (2 degrees of freedom) of ``gate_probability``: ``-2 ln(1 - p)``."""
        return float(-2.0 * np.log(1.0 - self.gate_probability))


def fix_covariance(widths_m, major_deg: float | None, cfg: TrackerConfig) -> np.ndarray:
    """Measurement covariance (2, 2) of a fix from the half maximum widths ``(major, minor)`` [m] of its map peak and
    the direction of the major axis (degrees clockwise from north); without widths the floor alone."""
    floor = cfg.meas_floor ** 2 * np.eye(2)
    if widths_m is None or major_deg is None or not np.all(np.isfinite(widths_m)):
        return floor
    s = cfg.meas_scale * np.asarray(widths_m, dtype=float) / 2.355
    a = np.radians(major_deg)
    u = np.array([np.sin(a), np.cos(a)])                     # east, north
    v = np.array([-u[1], u[0]])
    return s[0] ** 2 * np.outer(u, u) + s[1] ** 2 * np.outer(v, v) + floor


@dataclass
class RadiatorTrack:
    """One track: its fixes ``(t, xy, score)`` and a constant velocity Kalman filter (``x``: east, north, their
    velocities; ``P`` its covariance; ``t_state`` the time of the state)."""

    id: str
    fixes: list = field(default_factory=list)       # (t, xy, score)
    confirmed: bool = False
    confirmed_t: float | None = None                # first time the track was confirmed
    slow_since: float | None = None                 # velocity within its covariance of zero since this time
    static: bool = False
    lost_t: float | None = None                     # dropped at this time (kept as lost for ``lost_s``)
    x: np.ndarray | None = None
    P: np.ndarray | None = None
    t_state: float = 0.0
    accel: float = 0.05
    held: bool = False                              # about to confirm but held (an echo, ``HarborPicture``)
    states: list = field(default_factory=list)      # (t, x, P) after every update

    def start(self, t: float, z: np.ndarray, R: np.ndarray, init_speed: float) -> None:
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.zeros((4, 4))
        self.P[:2, :2] = R
        self.P[2:, 2:] = init_speed ** 2 * np.eye(2)
        self.t_state = t

    def predicted(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """State and covariance at *t* (constant velocity, white acceleration process noise)."""
        dt = t - self.t_state
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.accel ** 2
        Q = np.zeros((4, 4))
        Q[:2, :2] = q * dt ** 4 / 4.0 * np.eye(2)
        Q[:2, 2:] = Q[2:, :2] = q * dt ** 3 / 2.0 * np.eye(2)
        Q[2:, 2:] = q * dt ** 2 * np.eye(2)
        return F @ self.x, F @ self.P @ F.T + Q

    def innovation(self, t: float, z: np.ndarray, R: np.ndarray) -> float:
        """Mahalanobis distance squared of the fix *z* (covariance *R*) to the prediction at *t*."""
        x, P = self.predicted(t)
        S = P[:2, :2] + R
        d = np.asarray(z, float) - x[:2]
        return float(d @ np.linalg.solve(S, d))

    def update(self, t: float, z: np.ndarray, R: np.ndarray) -> None:
        x, P = self.predicted(t)
        S = P[:2, :2] + R
        K = P[:, :2] @ np.linalg.inv(S)
        self.x = x + K @ (np.asarray(z, float) - x[:2])
        I_KH = np.eye(4)
        I_KH[:, :2] -= K
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T          # Joseph form
        self.t_state = t
        self.states.append((t, self.x.copy(), self.P.copy()))

    def velocity(self, fit_s: float | None = None) -> np.ndarray | None:
        """The filter's velocity [m/s] (``None`` before the second fix)."""
        return None if self.x is None or len(self.fixes) < 2 else self.x[2:].copy()

    def predict(self, t: float, fit_s: float | None = None) -> np.ndarray:
        """The filter's position at *t*."""
        return self.predicted(t)[0][:2]

    @property
    def score(self) -> float:
        return float(np.median([f[2] for f in self.fixes[-8:]]))


class RadiatorTracker:
    """Causal tracks from the per frame peaks of a map (the passive chain's radiators, or the echo chain's bodies):
    every track a constant velocity Kalman filter (:class:`TrackerConfig`).  Per frame, strongest peak first: the
    peaks a confirmed track explains (its predicted map, ``FrameResult.patterns``) are dropped unless inside that
    track's own gate; each other peak joins the untaken track of the smallest Mahalanobis distance within the gate,
    or starts a new track; a track is confirmed after ``confirm`` gated fixes within ``confirm_window_s`` (unless
    *hold* says it is an echo).  *grid*: the map's cells (for the half maximum widths of each peak; without it every
    fix gets the floor covariance)."""

    def __init__(self, cfg: TrackerConfig | None = None, prefix: str = "r", grid: np.ndarray | None = None,
                 hold=None) -> None:
        self.cfg = cfg or TrackerConfig()
        self.tracks: dict[str, RadiatorTrack] = {}
        self.lost: dict[str, RadiatorTrack] = {}       # dropped tracks that had been confirmed, for ``lost_s``
        self.prefix = prefix
        self.grid = grid
        self.hold = hold                                # hold(track, t) -> True: do not confirm it (yet)
        xs = np.unique(grid[:, 0]) if grid is not None else np.empty(0)
        self.cell = float(np.min(np.diff(xs))) if len(xs) > 1 else 1.0   # the map's cell size [m]
        self.frames: list[float] = []                  # times of the frames the tracker was updated with
        self.explained: list[tuple[float, np.ndarray, str]] = []   # (t, peak, track) peaks dropped as sidelobes
        self._n = 0

    def _fix_cov(self, frame: FrameResult, xy: np.ndarray) -> tuple[np.ndarray, dict | None]:
        if self.grid is None or frame.B is None:
            return fix_covariance(None, None, self.cfg), None
        from acoustic_sim.water.characteristics import peak_shape

        sh = peak_shape(frame.B, self.grid, xy, self.cfg.shape_radius)
        return fix_covariance(sh.get("widths_m"), sh.get("major_deg"), self.cfg), sh

    def _response(self, frame: FrameResult, src: np.ndarray, at: np.ndarray, key: str | None = None) -> float | None:
        """The map a source at *src* alone produces at *at*, relative to its own peak (``FrameResult.pair_R`` between
        the nearest of the block's cells, else the full predicted map of track *key*); ``None`` without either."""
        if frame.pair_R is not None and len(frame.pair_R):
            c = frame.pair_cells
            i, j = int(np.argmin(np.hypot(*(c - at).T))), int(np.argmin(np.hypot(*(c - src).T)))
            tol = 1.5 * self.cell
            if np.hypot(*(c[i] - at)) <= tol and np.hypot(*(c[j] - src)) <= tol:
                return float(frame.pair_R[i, j])
        if frame.patterns and key in frame.patterns and self.grid is not None:
            return float(frame.patterns[key][int(np.argmin(np.hypot(*(self.grid - at).T)))])
        return None

    def _own_peak(self, frame: FrameResult, xy: np.ndarray) -> float:
        near = np.hypot(*(self.grid - xy).T) <= 1.5 * self.cell
        return float(np.max(frame.B[near])) if near.any() else 0.0

    def _explains(self, frame: FrameResult, tr: RadiatorTrack, xy: np.ndarray, value: float, R: np.ndarray, t: float,
                  src: np.ndarray | None = None) -> bool:
        """Whether the confirmed track *tr* (its peak at *src*, default its prediction) explains a peak of *value* at
        *xy*: outside its gate, and its predicted map there, scaled to its own peak, at least ``explain_fraction`` of
        the data map."""
        if self.grid is None or frame.B is None or tr.innovation(t, xy, R) <= self.cfg.gate_chi2:
            return False
        src = tr.predict(t) if src is None else np.asarray(src, float)
        r = self._response(frame, src, xy, tr.id)
        return r is not None and r * self._own_peak(frame, src) >= self.cfg.explain_fraction * max(value, 1e-30)

    def _explaining(self, frame: FrameResult, xy: np.ndarray, value: float, R: np.ndarray, t: float) -> str | None:
        """The confirmed track whose predicted map explains the peak at *xy* (not one whose own gate it is in)."""
        for k, tr in self.tracks.items():
            if tr.confirmed and self._explains(frame, tr, xy, value, R, t):
                return k
        return None

    def update(self, frame: FrameResult) -> None:
        cfg, t = self.cfg, frame.t
        self.frames.append(t)
        for tid in [k for k, tr in self.tracks.items() if t - tr.fixes[-1][0] > cfg.drop_s]:
            tr = self.tracks.pop(tid)
            if tr.confirmed_t is not None:
                tr.lost_t = t
                self.lost[tid] = tr
        for tid in [k for k, tr in self.lost.items() if t - tr.lost_t > cfg.lost_s]:
            self.lost.pop(tid)
        taken: set[str] = set()
        for xy, sc in zip(frame.xy, frame.score):
            xy = np.asarray(xy, float)
            R, _ = self._fix_cov(frame, xy)
            by = self._explaining(frame, xy, float(sc), R, t)
            if by is not None:
                self.explained.append((t, xy, by))
                continue
            best, dbest = None, cfg.gate_chi2
            for firm in (True, False):                  # a confirmed track first: a tentative one does not steal its fixes
                for k, tr in self.tracks.items():
                    if k in taken or tr.confirmed is not firm:
                        continue
                    d2 = tr.innovation(t, xy, R)
                    if d2 <= dbest:
                        best, dbest = k, d2
                if best is not None:
                    break
            if best is None and any(tr.confirmed and tr.innovation(t, xy, R) <= cfg.gate_chi2 for tr in self.tracks.values()):
                continue                                # in the gate of a confirmed track that took another peak: no new track
            if best is None:
                self._n += 1
                best = f"{self.prefix}{self._n}"
                tr = RadiatorTrack(best, accel=cfg.accel)
                tr.start(t, xy, R, cfg.init_speed)
                tr.states.append((t, tr.x.copy(), tr.P.copy()))
                self.tracks[best] = tr
            else:
                self.tracks[best].update(t, xy, R)
            self.tracks[best].fixes.append((t, xy, float(sc)))
            taken.add(best)
        kept: list[RadiatorTrack] = []
        for tr in sorted(self.tracks.values(), key=lambda x: -x.score):                   # strongest first
            recent = sum(1 for f in tr.fixes if f[0] >= t - cfg.confirm_window_s)
            near = cfg.exclusion > 0 and any(np.hypot(*(o.predict(t) - tr.predict(t))) < cfg.exclusion for o in kept)
            if not tr.confirmed and tr.fixes[-1][0] == t and frame.B is not None and self.grid is not None:
                # A track confirming in the same frame as a stronger one: a sidelobe of it is not confirmed.
                fx = tr.fixes[-1]
                Rf, _ = self._fix_cov(frame, fx[1])
                near = near or any(o.fixes[-1][0] == t and self._explains(frame, o, fx[1], fx[2], Rf, t, src=o.fixes[-1][1])
                                   for o in kept)
            if not tr.confirmed and recent >= cfg.confirm and not near:
                tr.held = bool(self.hold is not None and self.hold(tr, t))
                tr.confirmed = not tr.held
            elif tr.confirmed and near:
                tr.confirmed = False
            if tr.confirmed:
                kept.append(tr)
                if tr.confirmed_t is None:
                    tr.confirmed_t = t
            if tr.fixes[-1][0] == t and tr.x is not None:                                # static: velocity ~ 0
                v, Pv = tr.x[2:], tr.P[2:, 2:]
                slow = len(tr.fixes) >= 3 and float(v @ np.linalg.solve(Pv, v)) <= cfg.gate_chi2
                tr.slow_since = (tr.slow_since if tr.slow_since is not None else t) if slow else None
            tr.static = tr.confirmed and tr.slow_since is not None and t - tr.slow_since >= cfg.static_s

    def confirmed(self) -> dict[str, RadiatorTrack]:
        return {k: tr for k, tr in self.tracks.items() if tr.confirmed}

    def illuminators(self, frame: FrameResult | None = None) -> dict[str, RadiatorTrack]:
        """Confirmed tracks whose peak is ``illuminator_margin_db`` above the map's noise floor (the median of the
        frame's map over the cells above zero; without a map every confirmed track), strongest first: the radiators
        whose direct paths the echo chain removes and whose echoes it maps."""
        conf = self.confirmed()
        if frame is None or frame.B is None:
            return dict(sorted(conf.items(), key=lambda kv: -kv[1].score))
        pos = frame.B[frame.B > 0]
        floor = float(np.median(pos)) if len(pos) else 0.0
        lim = floor * 10 ** (self.cfg.illuminator_margin_db / 10.0)
        return dict(sorted(((k, tr) for k, tr in conf.items() if tr.score >= lim), key=lambda kv: -kv[1].score))


def ais_crosscheck(tracks: dict[str, RadiatorTrack], reports: AISReports, gate: float = 25.0, last: int = 8) -> dict[str, str | None]:
    """AIS used after the location: label each track with the reporting ship whose (dead reckoned) report
    positions are within *gate* m of the track's last *last* fixes (median), else ``None``."""
    out: dict[str, str | None] = {}
    for k, tr in tracks.items():
        best, dbest = None, gate
        for v in reports.vessels:
            at = reports.for_vessel(v)
            fx = [f for f in tr.fixes[-last:] if f[0] >= at.t[0]]
            if not fx:
                continue
            d = float(np.median([np.hypot(*(at.predict(f[0], causal=True)[0] - f[1])) for f in fx]))
            if d < dbest:
                best, dbest = v, d
        out[k] = best
    return out


# ----------------------------------------------------------------------
# Echo chain
# ----------------------------------------------------------------------

@dataclass
class EchoStreamConfig:
    """Parameters of the streaming echo chain (the chain's own are in ``BistaticConfig``).

    ``step_blocks``: ECA blocks (``eca_block_s``) cleaned per step; ``combine``: how the maps of several
    illuminators are combined (``"sum"`` or ``"max"``); ``history_s``: samples kept before
    the blocks being cleaned (the reference delays and the QC); ``lookahead_s``: samples needed after
    them (the reference beam advances every channel by its range to the ship); ``max_peaks`` /
    ``min_separation``: peaks reported per frame.
    """

    step_blocks: int = 2
    combine: str = "sum"
    history_s: float = 1.0
    lookahead_s: float = 0.4
    max_peaks: int = 3
    min_separation: float = 20.0


class StreamingEcho:
    """Causal echo chain: ``bistatic.run_bistatic`` with the data reference, ECA with per sample delays and the
    coherent ``"snr"`` map, step by step, with the illuminator and every other radiator located passively.

    The caller passes the position fixes of the radiating objects (the passive chain's tracks,
    :class:`RadiatorTracker`) and which of them illuminates.  Every step (``step_blocks`` ECA blocks): each
    track is fitted by a straight line over the past ``track_smooth_s`` only and extrapolated ahead; the
    direct path of the illuminator and of every other radiator is removed from the newest blocks by ECA
    (``bc.removal_order``: one after the other, or all in one solve, ``bistatic.remove_direct_paths``);
    the frames inside the blocks are mapped in single precision with the illuminator's reference.  AIS is
    not used.
    """

    def __init__(self, layout: SensorLayout, fs: float, env: ChartEnvironment, bc,
                 stc: EchoStreamConfig | None = None, grid: np.ndarray | None = None, remove_others: bool = True,
                 keep_maps: bool = False, keep_clean: bool = False, keep_spectra: bool = False) -> None:
        from acoustic_sim.water.bistatic import build_grid

        if bc.reference != "data" or bc.removal != "eca" or not bc.eca_doppler or bc.method != "coherent":
            raise ValueError("StreamingEcho runs the data reference, ECA with per sample delays and the coherent map")
        if not isinstance(resolve(bc.propagator, bc.c), FreeFieldPropagator):
            raise ValueError("StreamingEcho needs the free field propagator")
        self.layout, self.fs, self.env, self.bc = layout, float(fs), env, bc
        self.keep_maps, self.keep_clean, self.keep_spectra = keep_maps, keep_clean, keep_spectra
        self.kept_clean: dict[int, np.ndarray] = {}     # with keep_clean: every cleaned block (for figures)
        self.source: str | None = None
        self.remove_others = remove_others
        self.stc = stc or EchoStreamConfig()
        self.dev = get_device(bc.use_cuda)
        xp = self.dev.xp
        self.nper, self.hop = int(round(bc.win_s * fs)), int(round(bc.hop_s * fs))
        self.nb = int(round(bc.eca_block_s * fs))
        f = np.fft.rfftfreq(self.nper, 1.0 / fs)
        self.band = np.flatnonzero((f >= bc.f_lo) & (f <= bc.f_hi))[:: bc.bin_stride]
        self.freqs = f[self.band]
        self.ks = 2.0 * np.pi * self.freqs / bc.c
        self.window = get_window("hann", self.nper)
        self.grid = grid if grid is not None else build_grid(env, bc)
        gs, ss = xp.asarray(self.grid, dtype=np.float32), xp.asarray(layout.sample_points, dtype=np.float32)
        self.r_samples = xp.maximum(xp.hypot(gs[:, None, 0] - ss[None, :, 0], gs[:, None, 1] - ss[None, :, 1]), 1.0)
        if xp is not np:
            self._init_channel_greens()
        self.raw = np.zeros((layout.n_receivers, 0))
        self.raw0 = 0                                   # global index of raw[:, 0]
        self.n_in = 0                                   # samples received
        self.clean: dict[int, np.ndarray] = {}          # ECA block index -> cleaned samples (N, nb)
        self.refd: dict[int, np.ndarray] = {}           # ECA block index -> reference samples (nb,)
        self.radiators: dict[str, dict] = {}            # radiator -> fixes "t", "xy", their causal smoothing "sm", newest "line"
        self.bodies: list[np.ndarray] = []              # echo track positions (ref_mode "iterated")
        # Diagnostic only, set by a caller that has the truth (``stream.run(components=...)``): full record arrays
        # (N, n) passed through the same projections; their band energy per channel before and after each step.
        self.components: dict[str, np.ndarray] | None = None
        self.component_energy: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self.next_block = 0                             # first ECA block not cleaned
        self.next_frame = 0                             # first frame not mapped

    def _init_channel_greens(self) -> None:
        """Tables of :meth:`_channel_greens`: per channel its (up to three) sample points and the weights of
        ``SensorLayout.collapse`` (1, -2, 1 for a strain channel, 1 for a pressure one), and the sample point
        pairs that need the exact Hankel function at the lowest bin, sorted by range."""
        xp = self.dev.xp
        sidx, coef, owner, ocoef = _collapse_tables(self.layout)
        self._sidx, self._coef = xp.asarray(sidx.ravel()), xp.asarray(coef.ravel())
        r = self.r_samples
        g_n, s_n = xp.nonzero(r < 8.0 / float(self.ks[0]))
        rn = r[g_n, s_n]
        order = xp.argsort(rn)
        g_n, s_n, rn = g_n[order], s_n[order], rn[order]
        self._near = (g_n, s_n, rn, xp.asarray(owner)[s_n], xp.asarray(ocoef)[s_n])
        self._near_r = self.dev.asnumpy(rn)

    def _channel_greens(self, w):
        """Replicas ``collapse(green2d(k r)) * w`` (G, N) of every scanned bin, in order: one fused kernel per
        bin evaluates the three sample points of each channel from the range table (asymptotic series of
        ``propagation.green2d``), combines them as ``SensorLayout.collapse`` does and applies the weight;
        where ``k r < 8`` the difference to the exact Hankel function is then added."""
        xp = self.dev.xp
        if xp is np:
            for k in self.ks:
                yield (self.layout.collapse(green2d(k * self.r_samples, np), np) * w[None, :]).astype(np.complex64)
            return
        G, S = self.r_samples.shape
        N = self.layout.n_receivers
        kern = _channel_kernel()
        g_n, s_n, rn, own, oc = self._near
        wf = w.astype(np.float32)
        for k in self.ks:
            amp = np.float32(np.sqrt(2.0 / (np.pi * k)))
            out = xp.empty((G, N), dtype=np.complex64)
            kern(self.r_samples, self._sidx, self._coef, wf, np.float32(k), amp, np.int32(S), np.int32(N), out)
            m = int(np.searchsorted(self._near_r, 8.0 / k))
            if m:
                r_m = rn[:m]
                ir = 1.0 / r_m
                asym = (xp.exp(-1j * (k * r_m - np.pi / 4.0)) * (1.0 - 9.0 * ir ** 2 / (128.0 * k * k) + 1j * ir / (8.0 * k))
                        * amp / xp.sqrt(r_m))
                delta = ((green2d(k * r_m, xp) - asym) * (wf[own[:m]] * oc[:m])).astype(np.complex64)
                ov = out.view(np.float32)                                              # (G, 2N): real, imag interleaved
                xp.add.at(ov, (g_n[:m], 2 * own[:m]), delta.real)
                xp.add.at(ov, (g_n[:m], 2 * own[:m] + 1), delta.imag)
            yield out

    def push(self, samples: np.ndarray, radiators: dict[str, list[tuple[float, np.ndarray]]] | None = None,
             illuminator: str | None = None, bodies: list[np.ndarray] | None = None) -> list[FrameResult]:
        """Append *samples* (N, n).  *radiators*: new position fixes ``(t, xy)`` of the radiating objects (the
        passive chain's tracks, by id); their direct paths are removed in the order the radiators first appeared;
        *illuminator*: the id of the one whose echoes are mapped (``None``: keep the previous one); *bodies*: the
        positions of the confirmed echo tracks (``None``: keep the previous ones), whose modelled echoes the
        ``"iterated"`` reference subtracts."""
        self.raw = np.concatenate([self.raw, np.asarray(samples, dtype=np.float64)], axis=1)
        self.n_in += samples.shape[1]
        for rid, fixes in (radiators or {}).items():
            for t, xy in sorted(fixes, key=lambda f: f[0]):
                self._add_fix(rid, float(t), np.asarray(xy, float))
        if illuminator is not None:
            self.source = illuminator
        if bodies is not None:
            self.bodies = [np.asarray(b, dtype=float) for b in bodies]
        out = []
        look = int(round(self.stc.lookahead_s * self.fs))
        while (self.next_block + self.stc.step_blocks) * self.nb + look <= self.n_in:
            out += self._step()
        keep_from = max(0, self.next_block * self.nb - int(round(self.stc.history_s * self.fs)) - self.nper)
        if keep_from > self.raw0:
            self.raw = self.raw[:, keep_from - self.raw0:]
            self.raw0 = keep_from
        return out

    # ------------------------------------------------------------------

    def _raw(self, a: int, z: int) -> np.ndarray:
        return self.raw[:, a - self.raw0:z - self.raw0]

    def _spectrum(self, seg: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Band spectrum (N, F) of one frame after common mode removal (as ``bistatic.stft``)."""
        x = remove_common_mode(seg, self.layout, weights)
        return np.fft.rfft(x * self.window, axis=1)[:, self.band] * weights[:, None]

    def _add_fix(self, name: str, t: float, xy: np.ndarray) -> None:
        """Append a fix and its causal smoothing: the straight line fitted to the fixes of the past
        ``track_smooth_s`` (refitted without those more than 10 m from the first fit) evaluated at the fix."""
        r = self.radiators.setdefault(name, {"t": [], "xy": [], "sm": [], "line": None})
        r["t"].append(t)
        r["xy"].append(xy)
        tt, pts = np.array(r["t"]), np.array(r["xy"])
        m = tt >= t - self.bc.track_smooth_s
        if m.sum() >= 3:
            A = np.column_stack([np.ones(m.sum()), tt[m] - t])
            coef, *_ = np.linalg.lstsq(A, pts[m], rcond=None)
            ok = np.hypot(*(pts[m] - A @ coef).T) <= 10.0
            if 3 <= ok.sum() < m.sum():
                coef, *_ = np.linalg.lstsq(A[ok], pts[m][ok], rcond=None)
            r["sm"].append(coef[0])
            r["line"] = (t, coef[0], coef[1])
        else:
            r["sm"].append(xy)
            r["line"] = (t, xy, np.zeros(2))

    def _fixes(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        r = self.radiators.get(name)
        if r is None:
            return np.empty(0), np.empty((0, 2))
        return np.array(r["t"]), np.array(r["xy"]).reshape(-1, 2)

    def _live(self, name: str | None, t: float) -> bool:
        """The radiator has at least 3 fixes, the newest within 2 s before *t*."""
        r = self.radiators.get(name) if name is not None else None
        return r is not None and len(r["t"]) >= 3 and r["t"][0] <= t <= r["t"][-1] + 2.0

    def _smoothed(self, t: np.ndarray, name: str | None = None) -> np.ndarray:
        """Position of *name* (default: the source) at times *t*: its smoothed fixes interpolated, and beyond the
        newest fix the newest line extrapolated."""
        r = self.radiators[name or self.source]
        tt, sm = np.array(r["t"]), np.array(r["sm"])
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.column_stack([np.interp(t, tt, sm[:, 0]), np.interp(t, tt, sm[:, 1])])
        t1, p1, v1 = r["line"]
        late = t > t1
        out[late] = p1[None, :] + np.outer(t[late] - t1, v1)
        return out

    def _step(self) -> list[FrameResult]:
        from acoustic_sim.water.bistatic import remove_direct_paths

        t0 = time.perf_counter()
        bc, stc = self.bc, self.stc
        b0, b1 = self.next_block, self.next_block + stc.step_blocks
        self.next_block = b1
        h_blocks = int(np.ceil(stc.history_s / bc.eca_block_s))
        a = max(b0 - h_blocks, 0) * self.nb                                                # buffer start, on a block boundary
        a = max(a, int(np.ceil(self.raw0 / self.nb)) * self.nb)
        z = min(b1 * self.nb + int(round(stc.lookahead_s * self.fs)), self.n_in)
        buf = self._raw(a, z)
        weights = channel_weights(buf, self.layout, ProcessingConfig(win_s=bc.win_s, hop_s=bc.hop_s))
        t_now = b1 * self.nb / self.fs
        live = [r for r in self.radiators if self._live(r, t_now)]
        if self.source is not None:                         # one illuminator: it first, the others only removed
            live = ([self.source] if self.source in live else []) + [r for r in live if r != self.source] if self.remove_others else \
                [r for r in live if r == self.source]
        if not live or (self.source is not None and live[0] != self.source):
            for b in range(b0, b1):
                self.clean[b] = self._raw(b * self.nb, (b + 1) * self.nb)
                self.refd[b] = {}
            return self._frames(t0, weights)
        t_s = (a + np.arange(z - a)) / self.fs
        nblk = int(np.ceil((z - a) / self.nb))
        mid = np.minimum(np.arange(nblk) * self.nb + self.nb // 2, z - a - 1)
        rel = set(range(b0 - a // self.nb, b1 - a // self.nb))
        chans = np.flatnonzero(weights > 0)
        # The direct path of every live radiator (with an illuminator, it first), by bc.removal_order: one after the
        # other in the order the radiators first appeared in a push, each reference on the traces already cleaned of
        # the previous ones ("sequential"; strongest first was measured worse on the multistatic recording, uuv2 18 ->
        # 13 % of frames), or all in one least squares solve per block ("joint"); each reference is kept to map that
        # radiator's echoes.
        x = buf                                            # as run_bistatic: reference and ECA on the traces,
        refs = {}                                          # the common mode removed in the STFT
        comps = {k: np.asarray(v[:, a:z], dtype=np.float64) for k, v in self.components.items()} if self.components else {}
        before = {k: self._block_energy(v, b0 - a // self.nb, b1 - a // self.nb) for k, v in comps.items()}
        ships = []
        for name in live:
            ship_t = self._smoothed(t_s, name)
            ships.append((ship_t[mid], ship_t))
        x, cleaned, ref_list = remove_direct_paths(x, list(comps.values()), self.layout, self.fs, ships, chans, bc,
                                                   blocks=rel, bodies=self.bodies)
        refs = dict(zip(live, ref_list))
        comps = dict(zip(comps, cleaned))
        for k, v in comps.items():                         # diagnostic, from the truth: energy before and after per channel
            self.component_energy.setdefault(k, []).append((before[k], self._block_energy(v, b0 - a // self.nb, b1 - a // self.nb)))
        for b in range(b0, b1):
            lo = b * self.nb - a
            self.clean[b] = x[:, lo:lo + self.nb]
            self.refd[b] = {name: r[lo:lo + self.nb] for name, r in refs.items()}
            if self.keep_clean:
                self.kept_clean[b] = self.clean[b]
        return self._frames(t0, weights)

    def _block_energy(self, x: np.ndarray, r0: int, r1: int) -> np.ndarray:
        """Band energy per channel (N,) of the blocks ``r0`` to ``r1`` (relative to the buffer) of *x*."""
        seg = x[:, r0 * self.nb:r1 * self.nb]
        f = np.fft.rfftfreq(seg.shape[1], 1.0 / self.fs)
        m = (f >= self.bc.f_lo) & (f <= self.bc.f_hi)
        return np.sum(np.abs(np.fft.rfft(seg, axis=1)[:, m]) ** 2, axis=1)

    def _frames(self, t0: float, weights: np.ndarray) -> list[FrameResult]:
        """Map every frame whose samples are all cleaned: the coherent ``"snr"`` map of every illuminator whose
        reference covers the frame (each in units of the noise power), summed, with the cells near every
        radiator excluded."""
        xp, dev, bc, stc = self.dev.xp, self.dev, self.bc, self.stc
        done = self.next_block * self.nb
        k1 = (done - self.nper) // self.hop
        ks = [k for k in range(self.next_frame, k1 + 1)]
        self.next_frame = max(self.next_frame, k1 + 1)
        first = (ks[0] if ks else self.next_frame) * self.hop // self.nb          # oldest block still needed
        for b in [b for b in self.clean if b < first]:
            self.clean.pop(b)
            self.refd.pop(b)
        if not ks:
            return []
        times = (np.array(ks) * self.hop + self.nper / 2.0) / self.fs
        blocks = [list(range((k * self.hop) // self.nb, (k * self.hop + self.nper - 1) // self.nb + 1)) for k in ks]
        illum = [[n for n in self.refd[bl[0]] if all(n in self.refd[b] for b in bl)] for bl in blocks]
        if self.source is not None:
            illum = [[n for n in il if n == self.source] for il in illum]
        names = sorted({n for il in illum for n in il})
        pos = {n: self._smoothed(times, n) for n in names}
        B = np.zeros((len(self.grid), len(ks)))
        live = [j for j in range(len(ks)) if illum[j]]
        X = R = None
        if live:
            X = np.empty((self.layout.n_receivers, len(self.band), len(live)), dtype=np.complex64)
            R = {n: np.zeros((len(self.band), len(live)), dtype=np.complex64) for n in names}
            for q, j in enumerate(live):
                bl = blocks[j]
                o = ks[j] * self.hop - bl[0] * self.nb
                seg = np.concatenate([self.clean[b] for b in bl], axis=1)
                X[:, :, q] = self._spectrum(seg[:, o:o + self.nper], weights)
                for n in illum[j]:
                    ref = np.concatenate([self.refd[b][n] for b in bl])
                    R[n][:, q] = np.fft.rfft(ref[o:o + self.nper] * self.window)[self.band]
            Xd = xp.asarray(X)
            nI, nK = len(names), len(live)
            Rd = xp.asarray(np.stack([R[n] for n in names], axis=1))                     # (F, I, K)
            wd = xp.asarray(weights, dtype=np.float32)
            gs = xp.asarray(self.grid, dtype=np.float32)
            sp_ = xp.asarray(np.concatenate([pos[n][live] for n in names]), dtype=np.float32)      # (I K, 2)
            r_path = xp.maximum(xp.hypot(gs[:, None, 0] - sp_[None, :, 0], gs[:, None, 1] - sp_[None, :, 1]), 1.0)
            gen_p = green2d_bins(r_path, self.ks, xp)                                  # every illuminator at once
            gen_a = self._channel_greens(wd)
            Z = xp.zeros((len(self.grid), nI, nK), dtype=np.complex64)
            W = xp.zeros((len(self.grid), nI, nK), dtype=np.float32)
            for i in range(len(self.ks)):
                A = next(gen_a)
                norm = xp.maximum(xp.sum(xp.abs(A) ** 2, axis=1), 1e-30)[:, None, None]
                Y = (A.conj() @ Xd[:, i, :])[:, None, :] / norm                         # shared by every illuminator
                path = next(gen_p).reshape(len(self.grid), nI, nK)
                r_i = Rd[i][None, :, :]
                Z += Y * xp.conj(path) * xp.conj(r_i)
                W += (xp.abs(path) ** 2 * xp.abs(r_i) ** 2 / norm).astype(np.float32)
            maps = xp.abs(Z) ** 2 / xp.maximum(W, 1e-30)                                 # (G, I, K), zero where an illuminator has no reference
            Bl = maps.sum(axis=1) if stc.combine == "sum" else maps.max(axis=1)
            B[:, live] = dev.asnumpy(Bl).astype(np.float64)
            for q, j in enumerate(live):
                for n in self.radiators:
                    if self._live(n, times[j]):
                        p = self._smoothed(times[j:j + 1], n)[0]
                        B[np.hypot(*(self.grid - p).T) < bc.exclusion_radius, j] = 0.0
        compute = time.perf_counter() - t0
        q_of = {j: q for q, j in enumerate(live)}
        out = []
        for j, k in enumerate(ks):
            g = _nms(B[:, j], self.grid, stc.min_separation, stc.max_peaks, 0.0) if illum[j] else []
            frame_end = (k * self.hop + self.nper) / self.fs
            keep = self.keep_spectra and j in q_of
            out.append(FrameResult(float(times[j]), self.grid[g], B[g, j], {n: pos[n][j] for n in illum[j]},
                                   compute, float(self.n_in / self.fs - frame_end + compute),
                                   B[:, j].copy() if self.keep_maps else None,
                                   X[:, :, q_of[j]].copy() if keep else None, weights if keep else None,
                                   {n: R[n][:, q_of[j]].copy() for n in illum[j]} if keep else None))
        return out


_CHANNEL_KERNEL = None


def _channel_kernel():
    """Fused kernel of :meth:`StreamingEcho._channel_greens`: out[g, c] = w[c] sum_j coef[c, j] G_k(r[g, s_cj])
    with the asymptotic ``H0^(2)`` of ``propagation.green2d``."""
    global _CHANNEL_KERNEL
    if _CHANNEL_KERNEL is None:
        import cupy

        _CHANNEL_KERNEL = cupy.ElementwiseKernel(
            "raw float32 r, raw int32 sidx, raw float32 coef, raw float32 w, float32 k, float32 amp, int32 S, int32 N",
            "complex64 out",
            """
            int g = i / N, c = i % N;
            float re = 0.0f, im = 0.0f;
            for (int j = 0; j < 3; j++) {
                float cf = coef[3 * c + j];
                if (cf == 0.0f) continue;
                float rr = r[(long long)g * S + sidx[3 * c + j]];
                float u = 1.0f / rr;
                float sr = 1.0f - 9.0f * u * u / (128.0f * k * k), si = u / (8.0f * k);
                float s, co;
                sincosf(k * rr - 0.78539816339744831f, &s, &co);           // exp(-i (k r - pi/4)) = co - i s
                float a = cf * amp * rsqrtf(rr);
                re += a * (co * sr + s * si);
                im += a * (co * si - s * sr);
            }
            out = complex<float>(re * w[c], im * w[c]);
            """, "stream_channel_greens")
    return _CHANNEL_KERNEL

"""Real-time passive detection: the AIS-aware matched field chain, causal and frame by frame.

:class:`StreamingPassive` receives the traces in blocks as they are recorded (:meth:`push`) and
returns, for every new frame, the strongest map peaks.  It is the chain of ``suppression.run_ais_aware``
with every step that looked at the whole record made causal:

* channel QC on the last ``history_s`` seconds of samples;
* the noise floor (median over time) over the last ``history_s`` seconds of frames, before and
  after the ships are projected out;
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
    peaks reported per frame.
    """

    block_frames: int = 4
    history_s: float = 10.0
    warmup_s: float = 2.0
    interference_every: int = 16
    max_peaks: int = 3
    min_separation: float = 20.0


@dataclass
class FrameResult:
    t: float                          # frame centre [s]
    xy: np.ndarray                    # (P, 2) peaks, strongest first
    score: np.ndarray                 # (P,)
    ships: dict[str, np.ndarray]      # refined position of every reporting ship
    compute_s: float                  # compute time of the frame's block
    latency_s: float                  # last sample of the frame recorded -> result available
    B: np.ndarray | None = field(default=None, repr=False)


class StreamingPassive:
    """Causal frame by frame passive detection on a fixed scan grid (see the module docstring)."""

    def __init__(self, layout: SensorLayout, fs: float, env: ChartEnvironment, pc: ProcessingConfig,
                 sc: SuppressionConfig | None = None, stc: StreamConfig | None = None, grid: np.ndarray | None = None,
                 keep_maps: bool = False) -> None:
        self.layout, self.fs, self.env = layout, float(fs), env
        self.pc, self.stc = pc, stc or StreamConfig()
        self.sc = replace(sc or SuppressionConfig(), causal=True)    # AIS dead reckoned from past reports only
        self.keep_maps = keep_maps
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
        self.Xc = []                                              # same after the ship projection
        self.times: list[float] = []
        self.U: dict[int, object] = {}
        self.frames_done = 0

    # ------------------------------------------------------------------

    def push(self, samples: np.ndarray, reports: AISReports) -> list[FrameResult]:
        """Append *samples* (N, n) and process every complete block; *reports* are the AIS reports
        received so far (only those at or before each frame are used)."""
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
        for k in range(b):
            self.Xc.append((Zb[:, :, k] * sigma).astype(dev.cdtype))
        self.Xc = self.Xc[-self.H:]
        Xch = xp.stack(self.Xc, axis=2)
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
        B = dev.asnumpy(B).astype(np.float64)
        ships: list[dict[str, np.ndarray]] = [{} for _ in range(b)]
        if est is not None:
            for v, vid in enumerate(est.ids):
                for k in np.flatnonzero(est.present[:, v]):
                    B[np.hypot(*(self.grid - est.refined[k, v]).T) < sc.exclusion_radius, k] = 0.0
                    ships[k][vid] = est.refined[k, v]
        self.frames_done += b
        compute = time.perf_counter() - t0
        out = []
        for k in range(b):
            g = _nms(B[:, k], self.grid, stc.min_separation, stc.max_peaks, 0.0)
            frame_end = (starts[k] + self.nper) / self.fs
            out.append(FrameResult(float(times[k]), self.grid[g], B[g, k], ships[k], compute,
                                   float(stop / self.fs - frame_end + compute), B[:, k].copy() if self.keep_maps else None))
        return out

    def _init_fast_greens(self) -> None:
        """Tables for :meth:`_greens`: the free field replicas of the scanned bins in order."""
        xp, dev = self.dev.xp, self.dev
        r = self._geom[0]                                                            # (G, N) device, >= min_range
        c = resolve(self.pc.propagator, self.pc.c).c
        self._k = 2.0 * np.pi * self.freqs[self.scan_bins] / c
        self._r = r
        self._inv_sqrt_r = (1.0 / xp.sqrt(r)).astype(dev.fdtype)
        self._inv_r = (1.0 / r).astype(dev.fdtype)
        dk = float(self._k[1] - self._k[0]) if len(self._k) > 1 else 0.0
        self._step = xp.exp(-1j * dk * r).astype(dev.cdtype)
        near = r < 8.0 / float(self._k[0])                                            # where the asymptotic series is not used
        self._near = xp.nonzero(near)

    def _greens(self, exact_every: int = 16):
        """Coroutine over the scanned bins, in order: ``send(s)`` with the per channel scale *s* (N,)
        (weight over noise floor) returns that bin's replicas ``green2d(k r) * s`` (G, N).  The phase
        advances by one complex multiply per bin (re-anchored to the exact phase every *exact_every* bins),
        amplitude and asymptotic series come from cached ``1 / sqrt(r)`` and ``1 / r`` (one fused GPU
        kernel), and the exact Hankel function replaces them where ``k r < 8``."""
        xp, dev = self.dev.xp, self.dev
        kern = _greens_kernel() if xp is not np else None
        phase = None
        out = None
        scale = yield None
        for j, k in enumerate(self._k):
            if j % exact_every == 0:
                phase = xp.exp(-1j * (k * self._r - np.pi / 4.0)).astype(dev.cdtype)
            sc = scale.astype(dev.fdtype)[None, :]
            if kern is not None:
                out = xp.empty_like(phase)
                kern(phase, self._step, self._inv_sqrt_r, self._inv_r, sc, np.float32(k),
                     np.float32(np.sqrt(2.0 / (np.pi * k))), out, phase)
            else:
                series = 1.0 + 1j * self._inv_r / (8.0 * k) - 9.0 * self._inv_r ** 2 / (128.0 * k * k)
                out = (np.sqrt(2.0 / (np.pi * k)) * self._inv_sqrt_r * series * phase * sc).astype(dev.cdtype)
                phase = (phase * self._step).astype(dev.cdtype)
            rn = self._r[self._near]
            exact = green2d(k * rn, xp) * sc[0][self._near[1]]
            out[self._near] = xp.where(k * rn < 8.0, exact, out[self._near]).astype(dev.cdtype)
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


def _reports_until(rep: AISReports, t: float) -> AISReports:
    m = rep.t <= t
    return AISReports(rep.vessel[m], rep.t[m], rep.xy[m], rep.sog[m], rep.cog[m])


_KERNEL = None


def _greens_kernel():
    """Fused CuPy kernel: replicas of one bin from the running phase, then the phase advanced in place."""
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


# ----------------------------------------------------------------------
# Echo chain
# ----------------------------------------------------------------------

@dataclass
class EchoStreamConfig:
    """Parameters of the streaming echo chain (the chain's own are in ``BistaticConfig``).

    ``step_blocks``: ECA blocks (``eca_block_s``) cleaned per step; ``history_s``: samples kept before
    the blocks being cleaned (the reference delays and the QC); ``lookahead_s``: samples needed after
    them (the reference beam advances every channel by its range to the ship); ``max_peaks`` /
    ``min_separation``: peaks reported per frame.
    """

    step_blocks: int = 2
    history_s: float = 1.0
    lookahead_s: float = 0.4
    max_peaks: int = 3
    min_separation: float = 20.0


class StreamingEcho:
    """Causal echo chain with one AIS ship as the illuminator: ``bistatic.run_bistatic`` with the data
    reference, ECA with per sample delays and the coherent ``"snr"`` map, step by step.

    Every step (``step_blocks`` ECA blocks): the ship is refined in every frame whose samples have
    arrived (AIS predicted from the reports received), the track is fitted by a straight line over
    the past ``track_smooth_s`` only and extrapolated ahead, the reference beam is formed on a short
    buffer, the ECA cleans the newest blocks, and the frames inside them are mapped in single
    precision with the free field replicas of ``propagation.green2d_bins``.
    """

    def __init__(self, layout: SensorLayout, fs: float, env: ChartEnvironment, bc, source: str,
                 stc: EchoStreamConfig | None = None, grid: np.ndarray | None = None) -> None:
        from acoustic_sim.water.bistatic import ReplicaModel, build_grid

        if bc.reference != "data" or bc.removal != "eca" or not bc.eca_doppler or bc.method != "coherent":
            raise ValueError("StreamingEcho runs the data reference, ECA with per sample delays and the coherent map")
        if not isinstance(resolve(bc.propagator, bc.c), FreeFieldPropagator):
            raise ValueError("StreamingEcho needs the free field propagator")
        self.layout, self.fs, self.env, self.bc, self.source = layout, float(fs), env, bc, source
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
        self.model = ReplicaModel(layout, bc.c, propagator=bc.propagator, xp=xp)
        gs, ss = xp.asarray(self.grid, dtype=np.float32), xp.asarray(layout.sample_points, dtype=np.float32)
        self.r_samples = xp.maximum(xp.hypot(gs[:, None, 0] - ss[None, :, 0], gs[:, None, 1] - ss[None, :, 1]), 1.0)
        if xp is not np:
            self._init_channel_greens()
        self.raw = np.zeros((layout.n_receivers, 0))
        self.raw0 = 0                                   # global index of raw[:, 0]
        self.n_in = 0                                   # samples received
        self.clean: dict[int, np.ndarray] = {}          # ECA block index -> cleaned samples (N, nb)
        self.refd: dict[int, np.ndarray] = {}           # ECA block index -> reference samples (nb,)
        self.track: dict[int, tuple[float, np.ndarray, bool]] = {}   # frame -> (t, refined xy, present)
        self.next_block = 0                             # first ECA block not cleaned
        self.next_frame = 0                             # first frame not mapped

    def _init_channel_greens(self) -> None:
        """Tables of :meth:`_channel_greens`: per channel its (up to three) sample points and the weights of
        ``SensorLayout.collapse`` (1, -2, 1 for a strain channel, 1 for a pressure one), and the sample point
        pairs that need the exact Hankel function at the lowest bin, sorted by range."""
        xp = self.dev.xp
        sidx, coef, start = [], [], 0
        for c in self.layout.cables:
            for ch in range(c.n_channels):
                if c.is_strain:
                    sidx.append([start + 3 * ch, start + 3 * ch + 1, start + 3 * ch + 2])
                    coef.append([1.0, -2.0, 1.0])
                else:
                    sidx.append([start + ch] * 3)
                    coef.append([1.0, 0.0, 0.0])
            start += c.n_samples
        for h in range(self.layout.n_hydrophones):
            sidx.append([start + h] * 3)
            coef.append([1.0, 0.0, 0.0])
        sidx, coef = np.array(sidx, dtype=np.int32), np.array(coef, dtype=np.float32)
        owner = np.empty(self.layout.n_samples, dtype=np.int64)
        ocoef = np.zeros(self.layout.n_samples, dtype=np.float32)
        for chn in range(len(sidx)):
            for j in range(3):
                if coef[chn, j] != 0.0:
                    owner[sidx[chn, j]], ocoef[sidx[chn, j]] = chn, coef[chn, j]
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

    def push(self, samples: np.ndarray, reports: AISReports) -> list[FrameResult]:
        """Append *samples* (N, n); *reports* are the AIS reports received so far (only the source's are used)."""
        self.raw = np.concatenate([self.raw, np.asarray(samples, dtype=np.float64)], axis=1)
        self.n_in += samples.shape[1]
        rep = _reports_until(reports, self.n_in / self.fs)
        m = rep.vessel == self.source
        rep = AISReports(rep.vessel[m], rep.t[m], rep.xy[m], rep.sog[m], rep.cog[m])
        out = []
        look = int(round(self.stc.lookahead_s * self.fs))
        while (self.next_block + self.stc.step_blocks) * self.nb + look <= self.n_in:
            out += self._step(rep)
        keep_from = max(0, self.next_block * self.nb - int(round(self.stc.history_s * self.fs)) - self.nper)
        if keep_from > self.raw0:
            self.raw = self.raw[:, keep_from - self.raw0:]
            self.raw0 = keep_from
        return out

    # ------------------------------------------------------------------

    def _raw(self, a: int, z: int) -> np.ndarray:
        return self.raw[:, a - self.raw0:z - self.raw0]

    def _refine(self, rep: AISReports, weights: np.ndarray) -> None:
        """Refine the ship in every frame whose samples have arrived and that is not refined yet."""
        from acoustic_sim.water.bistatic import Spectra, ship_positions

        if len(rep) == 0:
            return
        k0 = max(self.track) + 1 if self.track else 0
        k1 = (self.n_in - self.nper) // self.hop
        ks = [k for k in range(k0, k1 + 1) if k * self.hop >= self.raw0]
        if not ks:
            return
        X = np.stack([self._spectrum(self._raw(k * self.hop, k * self.hop + self.nper), weights) for k in ks], axis=2)
        sp = Spectra(X, self.freqs, (np.array(ks) * self.hop + self.nper / 2.0) / self.fs, weights, self.fs, self.bc.win_s)
        self.model.weights = weights
        ship = ship_positions(sp, rep, self.model, self.bc, vessel=self.source)
        for j, k in enumerate(ks):
            self.track[k] = (float(sp.times[j]), ship.refined[j].copy(), bool(ship.present[j]))

    def _spectrum(self, seg: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Band spectrum (N, F) of one frame after common mode removal (as ``bistatic.stft``)."""
        x = remove_common_mode(seg, self.layout, weights)
        return np.fft.rfft(x * self.window, axis=1)[:, self.band] * weights[:, None]

    def _smoothed(self, t: np.ndarray) -> np.ndarray:
        """Ship position at times *t* from the causal track: at each frame the straight line fitted to the
        refined positions of the past ``track_smooth_s`` evaluated at that frame, interpolated between
        frames and extrapolated from the newest line beyond the newest frame."""
        ks = sorted(k for k, v in self.track.items() if v[2])
        tt = np.array([self.track[k][0] for k in ks])
        xy = np.array([self.track[k][1] for k in ks])
        T = self.bc.track_smooth_s
        sm = xy.copy()
        line = (xy[-1], np.zeros(2))
        for j in range(len(ks)):
            m = (tt <= tt[j]) & (tt >= tt[j] - T)
            if m.sum() >= 3:
                A = np.column_stack([np.ones(m.sum()), tt[m] - tt[j]])
                coef, *_ = np.linalg.lstsq(A, xy[m], rcond=None)
                sm[j] = coef[0]
                if j == len(ks) - 1:
                    line = (coef[0], coef[1])
        out = np.column_stack([np.interp(t, tt, sm[:, 0]), np.interp(t, tt, sm[:, 1])])
        late = t > tt[-1]
        out[late] = line[0][None, :] + np.outer(t[late] - tt[-1], line[1])
        return out

    def _step(self, rep: AISReports) -> list[FrameResult]:
        from acoustic_sim.water.bistatic import eca_remove, make_reference

        t0 = time.perf_counter()
        xp, dev, bc, stc = self.dev.xp, self.dev, self.bc, self.stc
        b0, b1 = self.next_block, self.next_block + stc.step_blocks
        self.next_block = b1
        h_blocks = int(np.ceil(stc.history_s / bc.eca_block_s))
        a = max(b0 - h_blocks, 0) * self.nb                                                # buffer start, on a block boundary
        a = max(a, int(np.ceil(self.raw0 / self.nb)) * self.nb)
        z = min(b1 * self.nb + int(round(stc.lookahead_s * self.fs)), self.n_in)
        buf = self._raw(a, z)
        weights = channel_weights(buf, self.layout, ProcessingConfig(win_s=bc.win_s, hop_s=bc.hop_s))
        self._refine(rep, weights)
        if not any(v[2] for v in self.track.values()):
            for b in range(b0, b1):
                self.clean[b] = self._raw(b * self.nb, (b + 1) * self.nb)
                self.refd[b] = np.zeros(self.nb)
            return self._frames(t0, weights, rep_live=False)
        t_s = (a + np.arange(z - a)) / self.fs
        ship_t = self._smoothed(t_s)
        nblk = int(np.ceil((z - a) / self.nb))
        ship_b = ship_t[np.minimum(np.arange(nblk) * self.nb + self.nb // 2, z - a - 1)]
        x = buf                                            # as run_bistatic: reference and ECA on the traces,
        chans = np.flatnonzero(weights > 0)                # the common mode removed in the STFT
        waveform = make_reference(x, self.layout, self.fs, ship_b, bc.eca_block_s, bc, chans, ship_t=ship_t)
        rel = set(range(b0 - a // self.nb, b1 - a // self.nb))
        cleaned = eca_remove([x], waveform, self.layout, self.fs, ship_b, [], bc.eca_block_s, bc, ship_t=ship_t, blocks=rel)[0]
        for b in range(b0, b1):
            lo = b * self.nb - a
            self.clean[b] = cleaned[:, lo:lo + self.nb]
            self.refd[b] = waveform[lo:lo + self.nb]
        return self._frames(t0, weights, rep_live=True)

    def _frames(self, t0: float, weights: np.ndarray, rep_live: bool) -> list[FrameResult]:
        """Map every frame whose samples are all cleaned."""
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
        present = np.array([k in self.track and self.track[k][2] for k in ks]) & rep_live
        ship = np.array([self.track[k][1] if k in self.track else [np.nan, np.nan] for k in ks])
        B = np.zeros((len(self.grid), len(ks)))
        live = np.flatnonzero(present)
        if len(live):
            X = np.empty((self.layout.n_receivers, len(self.band), len(live)), dtype=np.complex64)
            R = np.empty((len(self.band), len(live)), dtype=np.complex64)
            for j, i in enumerate(live):
                s0 = ks[i] * self.hop
                seg = np.concatenate([self.clean[b] for b in range(s0 // self.nb, (s0 + self.nper - 1) // self.nb + 1)], axis=1)
                ref = np.concatenate([self.refd[b] for b in range(s0 // self.nb, (s0 + self.nper - 1) // self.nb + 1)])
                o = s0 - (s0 // self.nb) * self.nb
                X[:, :, j] = self._spectrum(seg[:, o:o + self.nper], weights)
                R[:, j] = np.fft.rfft(ref[o:o + self.nper] * self.window)[self.band]
            Xd, Rd = xp.asarray(X), xp.asarray(R)
            wd = xp.asarray(weights, dtype=np.float32)
            sp_ = xp.asarray(ship[live], dtype=np.float32)
            gs = xp.asarray(self.grid, dtype=np.float32)
            r_path = xp.maximum(xp.hypot(gs[:, None, 0] - sp_[None, :, 0], gs[:, None, 1] - sp_[None, :, 1]), 1.0)
            gen_a = self._channel_greens(wd)
            gen_p = green2d_bins(r_path, self.ks, xp)
            Z = xp.zeros((len(self.grid), len(live)), dtype=np.complex64)
            W = xp.zeros((len(self.grid), len(live)), dtype=np.float32)
            for i in range(len(self.ks)):
                A = next(gen_a)
                path = next(gen_p)
                norm = xp.maximum(xp.sum(xp.abs(A) ** 2, axis=1), 1e-30)
                Y = A.conj() @ Xd[:, i, :]
                r_i = Rd[i][None, :]
                Z += Y * xp.conj(path) * xp.conj(r_i) / norm[:, None]
                W += (xp.abs(path) ** 2 * xp.abs(r_i) ** 2 / norm[:, None]).astype(np.float32)
            Bl = xp.abs(Z) ** 2 / xp.maximum(W, 1e-30)
            B[:, live] = dev.asnumpy(Bl).astype(np.float64)
            for j, i in enumerate(live):
                B[np.hypot(*(self.grid - ship[i]).T) < bc.exclusion_radius, i] = 0.0
        compute = time.perf_counter() - t0
        out = []
        for j, k in enumerate(ks):
            g = _nms(B[:, j], self.grid, stc.min_separation, stc.max_peaks, 0.0) if present[j] else []
            frame_end = (k * self.hop + self.nper) / self.fs
            out.append(FrameResult(float(times[j]), self.grid[g], B[g, j], {self.source: ship[j]} if present[j] else {},
                                   compute, float(self.n_in / self.fs - frame_end + compute)))
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

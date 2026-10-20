"""FDTD runner for the harbor: per-source simulations with on-disk caching.

The wave equation is linear, so a scenario is simulated by running one FDTD
job per source and summing the resulting sensor gathers (plus noise added
afterwards).  Each job is cached on disk, keyed by everything that affects
the result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from acoustic_sim.solver.fdtd import BOUNDARIES, FDTDConfig, FDTDSolver, fd2_coefficients, fd2_cfl_factor
from acoustic_sim.water.environment import HarborEnvironment
from acoustic_sim.water.scatterer import Scatterer, body_factor
from acoustic_sim.water.sensor_sampling import layout_fingerprint
from acoustic_sim.water.sensors import SensorLayout
from acoustic_sim.sources import CustomTrajectorySource, StaticSource


BODY_MASKS = ("auto", "host", "device")

# Fields that do not change the result (excluded from the cache key) and fields added after the
# caches of the existing datasets were built (excluded while at their default, so those keys stay valid).
_KEY_EXCLUDE = ("use_cuda", "kernel", "body_masks")
_KEY_DEFAULTS = {"precision": "float64", "boundary": "sponge", "cpml_width": 15, "cpml_order": 2, "cpml_r0": 1e-6, "cpml_alpha_max": 0.0,
                 "time_scheme": "leapfrog"}


@dataclass
class HarborFDTDConfig:
    """FDTD settings for harbor runs.

    Attributes
    ----------
    fs : float
        Output sample rate [Hz]; the solver steps at ``dt = 1 / fs``.
    total_time : float
        Simulated duration [s].
    fd_order : int
        Spatial stencil order.
    damping_width, damping_max : int, float
        Sponge layer at the grid edges (open-sea boundary).
    background_absorption : float
        Small per-step damping everywhere (water absorption).
    shore_loss_width, shore_loss_max : float
        Land is a pressure-release wall (p = 0 on land cells, so even thin
        breakwaters are opaque).  To stop the harbor ringing forever, water
        within ``shore_loss_width`` metres of land gets extra damping ramping
        quadratically up to ``shore_loss_max`` (per step) at the wall, which
        mimics sloping beaches and rip-rap.  The amplitude reflection
        coefficient of a wall is roughly ``exp(-mean_sigma * width * fs / c)``
        with ``mean_sigma = shore_loss_max / 3``; the defaults give R ≈ 0.5
        (measured 0.51).
        Set ``shore_loss_max = 0`` for perfectly reflecting quays.
    precision : str
        ``"float64"`` (default) or ``"float32"`` fields (:class:`~acoustic_sim.solver.fdtd.FDTDConfig`).
        Part of the cache key.
    kernel : str
        ``"auto"`` (default): one fused CUDA kernel per step on the GPU, the array code elsewhere;
        ``"reference"`` forces the array code, ``"fused"`` requests the kernel.  Not part of the
        cache key: both kernels compute the same traces (identical to the last bit in float64 on
        the stage 0 benchmark).
    boundary, cpml_width, cpml_order, cpml_r0, cpml_alpha_max
        Edge boundary: ``"sponge"`` (default, the ramp above) or ``"cpml"`` (a convolutional PML of
        ``cpml_width`` cells; see :class:`~acoustic_sim.solver.fdtd.FDTDConfig`).  Part of the cache key
        when not at the defaults.
    time_scheme : str
        ``"leapfrog"`` (default), ``"lw4"`` or ``"kspace"`` (:class:`~acoustic_sim.solver.fdtd.FDTDConfig`;
        prototypes).  Part of the cache key when not the default.
    body_masks : str
        How moving bodies are imposed with the fused kernel: ``"host"`` evaluates
        :func:`~acoustic_sim.water.scatterer.body_factor` on the CPU every step (the reference),
        ``"device"`` evaluates the same mask in a CUDA kernel from a pose table built once per run,
        ``"auto"`` picks ``"device"`` with the fused kernel.  Not part of the cache key.
    """

    fs: float = 600.0
    total_time: float = 120.0
    fd_order: int = 4
    damping_width: int = 30
    damping_max: float = 0.15
    background_absorption: float = 1e-4
    shore_loss_width: float = 40.0
    shore_loss_max: float = 0.15
    use_cuda: bool = False
    precision: str = "float64"
    kernel: str = "auto"
    body_masks: str = "auto"
    boundary: str = "sponge"
    cpml_width: int = 15
    cpml_order: int = 2
    cpml_r0: float = 1e-6
    cpml_alpha_max: float = 0.0
    time_scheme: str = "leapfrog"

    def __post_init__(self) -> None:
        if self.body_masks not in BODY_MASKS:
            raise ValueError(f"body_masks must be one of {BODY_MASKS}, got {self.body_masks!r}")
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"boundary must be one of {BOUNDARIES}, got {self.boundary!r}")

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def n_steps(self) -> int:
        return int(round(self.total_time * self.fs))


def cfl_dt(env: HarborEnvironment, fd_order: int = 4) -> float:
    """Largest stable timestep [s] for *env* (the solver's own CFL limit)."""
    spec = fd2_cfl_factor(fd2_coefficients(fd_order))
    return 2.0 * env.dx / (float(np.max(env.model.values)) * np.sqrt(2.0 * spec))


@dataclass
class Gather:
    """Sensor traces from one FDTD run, shape ``(n_receivers, n_steps)``."""

    traces: np.ndarray
    fs: float

    @property
    def n_steps(self) -> int:
        return self.traces.shape[1]

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n_steps) / self.fs

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, traces=self.traces, fs=self.fs)

    @classmethod
    def load(cls, path: str | Path) -> "Gather":
        with np.load(path) as d:
            return cls(traces=d["traces"], fs=float(d["fs"]))


class HarborFDTDSolver(FDTDSolver):
    """:class:`FDTDSolver` with a shoreline (land mask) on top of the sponge edges, and
    optional moving scattering bodies (:class:`~acoustic_sim.water.scatterer.Scatterer`)."""

    def __init__(
        self,
        env: HarborEnvironment,
        hcfg: HarborFDTDConfig,
        source: StaticSource | CustomTrajectorySource,
        receivers: np.ndarray,
        scatterers: list[Scatterer] | tuple[Scatterer, ...] = (),
    ) -> None:
        self._env = env
        self._hcfg = hcfg
        self._scatterers = list(scatterers)
        fcfg = FDTDConfig(
            # Half a step of slack so the solver's ceil() yields exactly n_steps.
            total_time=(hcfg.n_steps - 0.5) * hcfg.dt,
            dt=hcfg.dt,
            damping_width=hcfg.damping_width,
            damping_max=hcfg.damping_max,
            air_absorption=hcfg.background_absorption,
            snapshot_interval=0,
            use_cuda=hcfg.use_cuda,
            fd_order=hcfg.fd_order,
            precision=hcfg.precision,
            kernel=hcfg.kernel,
            boundary=hcfg.boundary,
            cpml_width=hcfg.cpml_width,
            cpml_order=hcfg.cpml_order,
            cpml_r0=hcfg.cpml_r0,
            cpml_alpha_max=hcfg.cpml_alpha_max,
            time_scheme=hcfg.time_scheme,
        )
        super().__init__(env.model, fcfg, source, receivers)
        assert self.n_steps == hcfg.n_steps

        g_top = self.row_start - self._ghost_top
        g_bot = self.row_end + self._ghost_bot
        self._water_local = self.xp.asarray(env.water[g_top:g_bot, :].astype(self._dtype))
        self.body_masks = "host"
        if self.kernel == "fused" and hcfg.body_masks in ("auto", "device"):
            self.body_masks = "device"

    # -- fused kernel hooks -------------------------------------------------

    def _fused_mask(self):
        return self._water_local

    def _fused_hooks(self) -> list:
        if not self._scatterers:
            return []
        if self.body_masks == "device":
            from acoustic_sim.water.fdtd_cuda_bodies import DeviceBodyMasks

            self._device_bodies = DeviceBodyMasks([self._scatterers], self.model.x, self.model.y, self._env.dx,
                                                  self.n_steps, self.dt, self._dtype)
            return [self._device_bodies.apply]
        return [self._apply_scatterers]

    def _build_damping_global(self, ny: int, nx: int) -> np.ndarray:
        sigma = super()._build_damping_global(ny, nx)
        w, smax = self._hcfg.shore_loss_width, self._hcfg.shore_loss_max
        if w > 0 and smax > 0:
            dist = distance_transform_edt(self._env.water) * self._env.dx  # metres from land, 0 on land
            ramp = np.clip((w - dist) / w, 0.0, 1.0) ** 2
            sigma = np.maximum(sigma, smax * ramp)
        return sigma

    def _inject(self, p, n: int) -> None:
        super()._inject(p, n)
        p *= self._water_local  # pressure-release land
        if self._scatterers:
            self._apply_scatterers(p, n)

    def _apply_scatterers(self, p, n: int) -> None:
        """Multiply the pressure inside every body by its (smooth) factor at this step."""
        t = n * self.dt
        x, y, dx = self.model.x, self.model.y, self._env.dx
        for sc in self._scatterers:
            if not sc.track.active_at(t)[0]:
                continue
            centre, heading = sc.pose_at(t)
            res = body_factor(sc, centre, heading, x, y, dx)
            if res is None:
                continue
            rows, cols, factor = res
            # Global rows -> this rank's padded local rows.
            r0 = rows.start - self.row_start + self._ghost_top
            r1 = rows.stop - self.row_start + self._ghost_top
            lo, hi = max(r0, 0), min(r1, p.shape[0])
            if hi <= lo:
                continue
            f = factor[lo - r0: hi - r0]
            p[lo:hi, cols] *= self.xp.asarray(f) if self.is_cuda else f

    def _sample_receivers(self, n: int) -> None:
        """Sample on the device (single process); the base class copies the whole field to the host every step."""
        if not self.is_cuda or self.use_mpi:
            return super()._sample_receivers(n)
        xp = self.xp
        if getattr(self, "_trace_buf", None) is None:
            self._trace_buf = xp.zeros((self.receivers.shape[0], self.n_steps), dtype=xp.float64)
            self._rx = [xp.asarray(a) for a in (self._recv_iy, self._recv_ix, self._recv_wx, self._recv_wy)]
        iy, ix, wx, wy = self._rx
        p = self.p_now
        self._trace_buf[:, n] = (
            p[iy, ix] * (1 - wx) * (1 - wy) + p[iy, ix + 1] * wx * (1 - wy)
            + p[iy + 1, ix] * (1 - wx) * wy + p[iy + 1, ix + 1] * wx * wy
        )

    def _finish_traces(self) -> None:
        super()._finish_traces()
        if getattr(self, "_trace_buf", None) is not None:
            self.traces[:] = self.xp.asnumpy(self._trace_buf)

    def run(self, *args, **kwargs):
        out = super().run(*args, **kwargs)
        self._finish_traces()
        return out

    def run_with_snapshots(self, steps: list[int], verbose: bool = False) -> tuple[dict, dict[int, np.ndarray]]:
        """Like :meth:`run` but also returns the full pressure field at the given steps (rank 0)."""
        wanted = set(steps)
        fields: dict[int, np.ndarray] = {}
        for n in range(self.n_steps):
            self._step(n)
            if n in wanted:
                field = self._gather_field()
                if self.rank == 0:
                    fields[n] = field
            if verbose and self.rank == 0 and n % 1000 == 0:
                print(f"  step {n:>6d} / {self.n_steps}")
        self._finish_traces()
        out = {"traces": self.traces, "dt": self.dt, "n_steps": self.n_steps}
        return out, fields


class HarborFDTDBatch(HarborFDTDSolver):
    """``B`` independent harbor runs on the same grid, advanced in the same CUDA launches.

    *jobs* is a list of ``(source, scatterers)``; the runs share the environment, the FDTD
    settings and the receivers and differ in their source signal and trajectory and in the set
    of bodies.  Needs the fused kernel on the GPU (``kernel = "fused"`` or ``"auto"``,
    ``use_cuda = True``); bodies use the device masks.  ``run()`` returns traces of shape
    ``(B, n_receivers, n_steps)``, each identical to the single run of that job (to 1e-12).
    """

    def __init__(self, env: HarborEnvironment, hcfg: HarborFDTDConfig, jobs: list[tuple], receivers: np.ndarray) -> None:
        if not jobs:
            raise ValueError("a batch needs at least one job")
        source0, scatterers0 = jobs[0]
        super().__init__(env, hcfg, source0, receivers, scatterers=scatterers0)
        if self.kernel != "fused":
            raise ValueError("batched runs need the fused kernel on the GPU (use_cuda=True, kernel='fused' or 'auto')")
        self._jobs = [(src, list(sc)) for src, sc in jobs]
        self.B = len(jobs)
        xp = self.xp
        self.p_now = xp.zeros((self.B, self._pad_ny, self.global_nx), dtype=self._dtype)
        self.p_prev = xp.zeros((self.B, self._pad_ny, self.global_nx), dtype=self._dtype)
        self.traces = np.zeros((self.B, receivers.shape[0], self.n_steps))
        self.body_masks = "device" if any(sc for _, sc in self._jobs) else "host"

    def _make_fused(self):
        from acoustic_sim.solver.fdtd_cuda import FusedFDTD

        fk = FusedFDTD(
            self._fd_coeffs, (self.p_now, self.p_prev), self.C2, self.sigma, self._fused_mask(),
            [src for src, _ in self._jobs], (self._recv_iy, self._recv_ix, self._recv_wx, self._recv_wy), self.traces,
            self.n_steps, self.dt, float(self.model.x[0]), float(self.model.y[0]), float(self.model.dx),
            self.cfg.source_amplitude, self._dtype,
            cpml=self._cpml, scheme=self._scheme,
        )
        fk.post_hooks = list(self._fused_hooks())
        return fk

    def _fused_hooks(self) -> list:
        if not any(sc for _, sc in self._jobs):
            return []
        from acoustic_sim.water.fdtd_cuda_bodies import DeviceBodyMasks

        self._device_bodies = DeviceBodyMasks([sc for _, sc in self._jobs], self.model.x, self.model.y, self._env.dx,
                                              self.n_steps, self.dt, self._dtype)
        return [self._device_bodies.apply]

    def _gather_field(self):
        return self.xp.asnumpy(self.p_now[:, self._ghost_top:self._ghost_top + self.local_ny, :])


def batch_size(n_jobs: int, env: HarborEnvironment, hcfg: HarborFDTDConfig, n_receivers: int, max_batch: int = 64,
               device_fraction: float = 0.6, host_bytes: int | None = None) -> int:
    """How many jobs to run in one :class:`HarborFDTDBatch`: as many as fit in *device_fraction* of the
    free device memory (two fields per run plus the trace buffer) and in *host_bytes* of host memory
    for the float64 traces (default: a quarter of the available host memory), at most *max_batch*."""
    if n_jobs <= 1:
        return max(n_jobs, 1)
    itemsize = np.dtype(hcfg.precision).itemsize
    per_run_device = 2 * env.ny * env.nx * itemsize + n_receivers * 4096 * itemsize + 64 * 2 ** 20 // max(n_jobs, 1)
    per_run_host = n_receivers * hcfg.n_steps * 8
    try:
        import cupy as cp

        free_dev, _ = cp.cuda.runtime.memGetInfo()
    except Exception:
        free_dev = 2 ** 30
    if host_bytes is None:
        try:
            import os

            host_bytes = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 4
        except (ValueError, OSError, AttributeError):
            host_bytes = 8 * 2 ** 30
    b_dev = int(device_fraction * free_dev // per_run_device)
    b_host = int(host_bytes // per_run_host)
    return int(max(1, min(n_jobs, max_batch, b_dev, b_host)))


def _fused_available(hcfg: HarborFDTDConfig) -> bool:
    """Would :class:`HarborFDTDSolver` use the fused kernel for *hcfg* (GPU present, single process)?"""
    if not hcfg.use_cuda or hcfg.kernel not in ("fused", "auto"):
        return False
    from acoustic_sim.backend import get_backend
    from acoustic_sim.solver.fdtd import _get_comm

    _, is_cuda = get_backend(True)
    return is_cuda and _get_comm()[2] == 1


# ----------------------------------------------------------------------
# Cache + top-level runner
# ----------------------------------------------------------------------

def _key_fields(cfg: HarborFDTDConfig) -> dict:
    """The config fields that enter the cache key (see ``_KEY_EXCLUDE`` and ``_KEY_DEFAULTS``)."""
    return {k: v for k, v in asdict(cfg).items()
            if k not in _KEY_EXCLUDE and not (k in _KEY_DEFAULTS and v == _KEY_DEFAULTS[k])}


def _cache_key(
    env: HarborEnvironment,
    layout: SensorLayout,
    cfg: HarborFDTDConfig,
    signal: np.ndarray,
    position: np.ndarray | None,
    trajectory: tuple[np.ndarray, np.ndarray] | None,
    scatterers: list[Scatterer] | tuple[Scatterer, ...] = (),
) -> str:
    h = hashlib.sha1()
    h.update(env.fingerprint().encode())
    h.update(layout_fingerprint(layout).encode())
    for sc in scatterers:
        h.update(sc.fingerprint().encode())
    h.update(json.dumps(_key_fields(cfg), sort_keys=True).encode())
    h.update(np.ascontiguousarray(signal, dtype=np.float64).tobytes())
    if position is not None:
        h.update(b"static")
        h.update(np.asarray(position, dtype=np.float64).tobytes())
    else:
        t, xy = trajectory  # type: ignore[misc]
        h.update(b"moving")
        h.update(np.asarray(t, dtype=np.float64).tobytes())
        h.update(np.asarray(xy, dtype=np.float64).tobytes())
    return h.hexdigest()


def simulate_source(
    env: HarborEnvironment,
    layout: SensorLayout,
    cfg: HarborFDTDConfig,
    signal: np.ndarray,
    *,
    position: tuple[float, float] | None = None,
    trajectory: tuple[np.ndarray, np.ndarray] | None = None,
    cache_dir: str | Path | None = None,
    verbose: bool = False,
    scatterers: list[Scatterer] | tuple[Scatterer, ...] = (),
) -> Gather:
    """Run one source through the harbor and record all sensors.

    Parameters
    ----------
    signal : ndarray, shape ``(cfg.n_steps,)``
        Source waveform sampled at ``cfg.fs``.
    position : (x, y), optional
        Fixed source location.
    trajectory : (times, xy), optional
        Moving source: ``times`` shape ``(M,)`` [s] and ``xy`` shape ``(M, 2)``,
        linearly interpolated.  Exactly one of *position* / *trajectory* is required.
    cache_dir : path, optional
        If given, results are cached there and reused on identical inputs.
    scatterers : list of Scatterer, optional
        Bodies present in the water during this run (they scatter the source's field).
        Strain cables are sampled at their three points per channel and collapsed, so the
        returned gather always has ``layout.n_receivers`` rows.
    """
    if (position is None) == (trajectory is None):
        raise ValueError("provide exactly one of position or trajectory")
    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape != (cfg.n_steps,):
        raise ValueError(f"signal must have shape ({cfg.n_steps},), got {signal.shape}")

    start_xy = np.asarray(position if position is not None else trajectory[1], dtype=float)
    if not np.all(env.is_water(start_xy.reshape(-1, 2))):
        raise ValueError("source position(s) must lie in water")

    path = None
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        key = _cache_key(env, layout, cfg, signal, None if position is None else np.asarray(position), trajectory,
                         scatterers)
        path = Path(cache_dir) / f"gather_{key}.npz"
        if path.exists():
            return Gather.load(path)

    if position is not None:
        source = StaticSource(x=float(position[0]), y=float(position[1]), signal=signal)
    else:
        times, xy = trajectory
        source = CustomTrajectorySource(times=np.asarray(times, float), positions=np.asarray(xy, float), signal=signal)

    solver = HarborFDTDSolver(env, cfg, source, layout.sample_points, scatterers=scatterers)
    out = solver.run(verbose=verbose)
    gather = _to_gather(out["traces"], layout, cfg)
    if path is not None:
        gather.save(path)
    return gather


def _to_gather(traces: np.ndarray, layout: SensorLayout, cfg: HarborFDTDConfig) -> Gather:
    traces = layout.collapse(traces.T).T if layout.has_strain else traces
    return Gather(traces=np.ascontiguousarray(traces, dtype=np.float32), fs=cfg.fs)


def _make_source(signal: np.ndarray, position, trajectory):
    if position is not None:
        return StaticSource(x=float(position[0]), y=float(position[1]), signal=signal)
    times, xy = trajectory
    return CustomTrajectorySource(times=np.asarray(times, float), positions=np.asarray(xy, float), signal=signal)


def simulate_sources(
    env: HarborEnvironment,
    layout: SensorLayout,
    cfg: HarborFDTDConfig,
    jobs: list[dict],
    *,
    cache_dir: str | Path | None = None,
    verbose: bool = False,
    batch: int | None = None,
) -> list[Gather]:
    """Several sources through the same harbor, batched on the GPU when the fused kernel is available.

    Each job is a dict with the keyword arguments of :func:`simulate_source` (``signal`` and one of
    ``position`` / ``trajectory``; optional ``scatterers``).  Cached jobs are loaded; the others
    run together in batches of *batch* (default: :func:`batch_size`, from the free device memory)
    through :class:`HarborFDTDBatch`, and each result is cached as if :func:`simulate_source` had
    produced it (same key, same float32 traces to rounding).  Without the fused kernel (CPU,
    ``kernel = "reference"``, MPI) the jobs run one after the other.
    """
    results: list[Gather | None] = [None] * len(jobs)
    pending: list[int] = []
    paths: list[Path | None] = [None] * len(jobs)
    for i, job in enumerate(jobs):
        signal = np.asarray(job["signal"], dtype=np.float64)
        position, trajectory = job.get("position"), job.get("trajectory")
        scatterers = tuple(job.get("scatterers", ()))
        if (position is None) == (trajectory is None):
            raise ValueError("provide exactly one of position or trajectory")
        if signal.shape != (cfg.n_steps,):
            raise ValueError(f"signal must have shape ({cfg.n_steps},), got {signal.shape}")
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            key = _cache_key(env, layout, cfg, signal, None if position is None else np.asarray(position), trajectory, scatterers)
            paths[i] = Path(cache_dir) / f"gather_{key}.npz"
            if paths[i].exists():
                results[i] = Gather.load(paths[i])
                continue
        pending.append(i)
    if not pending:
        return results  # type: ignore[return-value]
    if not _fused_available(cfg):
        for i in pending:
            job = jobs[i]
            results[i] = simulate_source(env, layout, cfg, job["signal"], position=job.get("position"), trajectory=job.get("trajectory"),
                                         cache_dir=cache_dir, verbose=verbose, scatterers=tuple(job.get("scatterers", ())))
        return results  # type: ignore[return-value]
    for i in pending:
        start_xy = np.asarray(jobs[i]["position"] if jobs[i].get("position") is not None else jobs[i]["trajectory"][1], dtype=float)
        if not np.all(env.is_water(start_xy.reshape(-1, 2))):
            raise ValueError("source position(s) must lie in water")
    B = batch if batch is not None else batch_size(len(pending), env, cfg, layout.n_samples)
    for k in range(0, len(pending), B):
        chunk = pending[k:k + B]
        solver = HarborFDTDBatch(env, cfg, [(_make_source(np.asarray(jobs[i]["signal"], dtype=np.float64), jobs[i].get("position"),
                                                          jobs[i].get("trajectory")), tuple(jobs[i].get("scatterers", ()))) for i in chunk],
                                 layout.sample_points)
        if verbose:
            print(f"  batch of {len(chunk)} runs, {cfg.n_steps} steps")
        out = solver.run(verbose=verbose)
        for b, i in enumerate(chunk):
            results[i] = _to_gather(out["traces"][b], layout, cfg)
            if paths[i] is not None:
                results[i].save(paths[i])
    return results  # type: ignore[return-value]

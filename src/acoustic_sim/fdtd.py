"""2-D FDTD acoustic wave-equation solver with MPI + CUDA support.

Supports three execution modes, all from the same code path:

1. **Single-process CPU** — ``python run_fdtd.py``
2. **Single-process GPU** — ``python run_fdtd.py --use-cuda``
3. **Multi-process (MPI) CPU/GPU** — ``mpirun -n 4 python run_fdtd.py [--use-cuda]``

Physics
=======
2-D scalar wave equation on a uniform Cartesian grid::

    d²p/dt² = c² (d²p/dx² + d²p/dy²)

discretised with central differences of user-defined order in space
and second-order leapfrog in time.

Spatial FD order
----------------
The ``fd_order`` parameter (2, 4, 6, 8, …) controls the accuracy of
the spatial Laplacian.  The stencil half-width is ``M = fd_order // 2``.

* Order 2 (M=1): classic 5-point stencil ``[-1, 2, -1] / dx²``
* Order 4 (M=2): 9-point stencil ``[1/12, -4/3, 5/2, -4/3, 1/12] / dx²``
* Order 8 (M=4): 17-point stencil, etc.

The CFL limit is adjusted for the spectral radius of the chosen stencil.

MPI decomposition
=================
The global grid of *ny* rows is split across ranks along the **y-axis**.
Each rank owns a contiguous strip of rows plus **M ghost rows** on each
internal boundary for the stencil.

CUDA / CuPy
============
When ``use_cuda=True``, all large arrays live on device memory via CuPy.
Halo exchange pulls ghost rows to host for MPI, then pushes back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from acoustic_sim.backend import get_backend
from acoustic_sim.domains import DomainMeta
from acoustic_sim.model import VelocityModel
from acoustic_sim.sources import MovingSource, StaticSource


# Try to import mpi4py; if unavailable, we run single-process.
try:
    from mpi4py import MPI as _MPI

    _COMM = _MPI.COMM_WORLD
    _HAS_MPI = _COMM.Get_size() > 1  # only activate if launched with mpirun
except ImportError:
    _MPI = None  # type: ignore[assignment]
    _COMM = None  # type: ignore[assignment]
    _HAS_MPI = False


def _get_comm() -> tuple[Any, int, int]:
    """Return (comm, rank, size).  Falls back to (None, 0, 1)."""
    if _HAS_MPI:
        return _COMM, _COMM.Get_rank(), _COMM.Get_size()
    return None, 0, 1


# -----------------------------------------------------------------------
# Finite-difference coefficients
# -----------------------------------------------------------------------

def fd2_coefficients(order: int) -> np.ndarray:
    """Central FD coefficients for d²f/dx² at accuracy *order* (2, 4, 6, …).

    Returns an array of length ``M + 1`` where ``M = order // 2``.
    ``coeffs[0]`` is the centre weight; ``coeffs[k]`` (k ≥ 1) is the
    weight for offsets ±k.  Multiplying by ``1/dx²`` gives d²f/dx².

    Derived by solving the Vandermonde system that enforces exactness
    for polynomials up to degree ``2M``.
    """
    M = order // 2
    if M < 1:
        raise ValueError(f"fd_order must be >= 2, got {order}")

    # System: sum_{k=1}^M c_k * k^(2j) = delta_{j,1}  for j = 1..M
    A = np.zeros((M, M))
    b = np.zeros(M)
    for j in range(M):
        for i in range(M):
            A[j, i] = (i + 1) ** (2 * (j + 1))
    b[0] = 1.0

    c = np.linalg.solve(A, b)
    c0 = -2.0 * np.sum(c)
    return np.concatenate([[c0], c])


def fd2_cfl_factor(coeffs: np.ndarray) -> float:
    """Return the 1-D spectral radius of the stencil at Nyquist.

    CFL for the 2-D wave equation::

        dt <= 2 * dx / (c_eff * sqrt(2 * spec_radius))

    where ``spec_radius = |c0 + 2 * sum(c_k * (-1)^k)|``.
    """
    val = coeffs[0]
    for k in range(1, len(coeffs)):
        val += 2.0 * coeffs[k] * ((-1) ** k)
    return abs(val)


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

@dataclass
class FDTDConfig:
    """User-tunable simulation parameters.

    Attributes
    ----------
    total_time : float
        Simulation duration [s].
    dt : float | None
        Timestep [s].  *None* → auto-compute from CFL.
    cfl_safety : float
        Fraction of the CFL limit to use for dt (0 < cfl_safety < 1).
    damping_width : int
        Sponge-layer thickness in grid cells.
    damping_max : float
        Peak damping coefficient at the domain edge.
    air_absorption : float
        Small background damping applied everywhere.
    snapshot_interval : int
        Save a wavefield snapshot every N steps.  0 = disabled.
    source_amplitude : float
        Peak source pressure [Pa].
    use_cuda : bool
        Use CuPy for GPU acceleration.
    fd_order : int
        Spatial finite-difference order (2, 4, 6, 8, …).  Higher orders
        have wider stencils and tighter CFL limits but less numerical
        dispersion.
    """

    total_time: float = 0.3
    dt: float | None = None
    cfl_safety: float = 0.9
    damping_width: int = 40
    damping_max: float = 0.15
    air_absorption: float = 0.005
    snapshot_interval: int = 50
    source_amplitude: float = 1.0
    use_cuda: bool = False
    fd_order: int = 2


# -----------------------------------------------------------------------
# MPI domain decomposition helpers
# -----------------------------------------------------------------------

def _split_rows(ny: int, size: int) -> list[tuple[int, int]]:
    """Return [(row_start, row_end), ...] for each rank (exclusive end)."""
    base = ny // size
    remainder = ny % size
    splits: list[tuple[int, int]] = []
    start = 0
    for r in range(size):
        count = base + (1 if r < remainder else 0)
        splits.append((start, start + count))
        start += count
    return splits


# -----------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------

class FDTDSolver:
    """2-D FDTD acoustic solver with MPI domain decomposition and
    optional CUDA acceleration.

    Parameters
    ----------
    model : VelocityModel
        *Global* sound-speed field (ny, nx).
    config : FDTDConfig
        Simulation parameters (including ``use_cuda``).
    source : StaticSource | MovingSource
        Source position + signal.
    receivers : ndarray (n_recv, 2)
        Receiver positions [[x0,y0], ...].
    domain_meta : DomainMeta | None
        Wind / attenuation metadata.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: VelocityModel,
        config: FDTDConfig,
        source: StaticSource | MovingSource,
        receivers: np.ndarray,
        domain_meta: DomainMeta | None = None,
        **_kw: Any,
    ) -> None:
        self.model = model
        self.cfg = config
        self.source = source
        self.receivers = receivers
        self.meta = domain_meta or DomainMeta()

        # -- MPI setup ------------------------------------------------------
        self.comm, self.rank, self.size = _get_comm()
        self.use_mpi = self.size > 1

        # -- Array backend (numpy or cupy) ----------------------------------
        self.xp, self.is_cuda = get_backend(config.use_cuda)
        xp = self.xp

        ny, nx = model.ny, model.nx
        dx = model.dx
        c_global = model.values  # (ny, nx) NumPy

        # -- FD stencil coefficients ----------------------------------------
        self._fd_coeffs = fd2_coefficients(config.fd_order)
        self._M = config.fd_order // 2  # stencil half-width
        spec_radius = fd2_cfl_factor(self._fd_coeffs)

        # -- Timestep via CFL (computed on full domain, all ranks agree) ----
        c_max = float(np.max(c_global))
        v_wind = math.sqrt(self.meta.wind_vx ** 2 + self.meta.wind_vy ** 2)
        cfl_limit = 2.0 * dx / ((c_max + v_wind) * math.sqrt(2.0 * spec_radius))

        if config.dt is not None:
            self.dt = config.dt
            if self.dt > cfl_limit:
                raise ValueError(
                    f"dt={self.dt:.2e} exceeds CFL limit {cfl_limit:.2e}"
                )
        else:
            self.dt = config.cfl_safety * cfl_limit

        self.n_steps = int(math.ceil(config.total_time / self.dt))

        # -- Domain decomposition -------------------------------------------
        self.global_ny = ny
        self.global_nx = nx
        self.splits = _split_rows(ny, self.size)
        self.row_start, self.row_end = self.splits[self.rank]
        self.local_ny = self.row_end - self.row_start
        M = self._M
        self._ghost_top = M if self.rank > 0 else 0
        self._ghost_bot = M if self.rank < self.size - 1 else 0
        self._pad_ny = self.local_ny + self._ghost_top + self._ghost_bot

        # -- Local slice of global arrays -----------------------------------
        g_top = self.row_start - self._ghost_top
        g_bot = self.row_end + self._ghost_bot
        c_local = c_global[g_top:g_bot, :]

        # -- Squared Courant number -----------------------------------------
        self.C2 = xp.asarray((c_local * self.dt / dx) ** 2)

        # -- Damping (built on full grid, then sliced) ----------------------
        sigma_global = self._build_damping_global(ny, nx)
        self.sigma = xp.asarray(sigma_global[g_top:g_bot, :])

        # -- Pressure fields (local, on device if CUDA) --------------------
        self.p_now = xp.zeros((self._pad_ny, nx), dtype=np.float64)
        self.p_prev = xp.zeros((self._pad_ny, nx), dtype=np.float64)

        # -- Receiver pre-computation (global indices → local) -------------
        self._precompute_receivers(model, dx)

        # -- Trace storage (rank 0 stores all) ------------------------------
        if self.rank == 0:
            self.traces = np.zeros((receivers.shape[0], self.n_steps))
        else:
            self.traces = None

    # ------------------------------------------------------------------
    # Damping
    # ------------------------------------------------------------------

    def _build_damping_global(self, ny: int, nx: int) -> np.ndarray:
        """Build the full (ny, nx) damping array on host."""
        dx_edge = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))
        dy_edge = np.minimum(np.arange(ny), ny - 1 - np.arange(ny))
        dist = np.minimum(dy_edge[:, np.newaxis], dx_edge[np.newaxis, :])

        sigma = np.full((ny, nx), self.cfg.air_absorption)
        w = self.cfg.damping_width
        mask = dist < w
        ramp = ((w - dist[mask]) / w) ** 2
        sigma[mask] = self.cfg.air_absorption + (
            self.cfg.damping_max - self.cfg.air_absorption
        ) * ramp
        return sigma

    # ------------------------------------------------------------------
    # Receiver pre-computation
    # ------------------------------------------------------------------

    def _precompute_receivers(self, model: VelocityModel, dx: float) -> None:
        """Compute bilinear indices/weights; figure out which receivers
        fall into this rank's local row strip."""
        rx_frac = (self.receivers[:, 0] - model.x[0]) / dx
        ry_frac = (self.receivers[:, 1] - model.y[0]) / dx

        gix = np.clip(np.floor(rx_frac).astype(int), 0, self.global_nx - 2)
        giy = np.clip(np.floor(ry_frac).astype(int), 0, self.global_ny - 2)
        wx = np.clip(rx_frac - gix, 0.0, 1.0)
        wy = np.clip(ry_frac - giy, 0.0, 1.0)

        # Receiver at giy needs rows giy and giy+1 — both must be owned.
        in_local = (giy >= self.row_start) & (giy + 1 < self.row_end)

        self._recv_global_idx = np.where(in_local)[0]
        local_iy = giy[in_local] - self.row_start + self._ghost_top
        self._recv_iy = local_iy
        self._recv_ix = gix[in_local]
        self._recv_wx = wx[in_local]
        self._recv_wy = wy[in_local]

    # ------------------------------------------------------------------
    # Source injection
    # ------------------------------------------------------------------

    def _inject(self, p: Any, n: int) -> None:
        """Inject source into local pressure field (in-place).

        Only the rank owning the source cell performs the injection.
        """
        sx, sy = self.source.position_at(n, self.dt)

        fx = (sx - self.model.x[0]) / self.model.dx
        fy = (sy - self.model.y[0]) / self.model.dx

        gix = int(math.floor(fx))
        giy = int(math.floor(fy))

        if gix < 0 or gix + 1 >= self.global_nx:
            return
        if giy < self.row_start or giy + 1 >= self.row_end:
            return

        liy = giy - self.row_start + self._ghost_top
        wx = fx - gix
        wy = fy - giy

        sig_val = self.source.signal[min(n, len(self.source.signal) - 1)]
        amp = self.cfg.source_amplitude * sig_val

        p[liy,     gix    ] += (1 - wx) * (1 - wy) * amp
        p[liy,     gix + 1] +=      wx  * (1 - wy) * amp
        p[liy + 1, gix    ] += (1 - wx) *      wy  * amp
        p[liy + 1, gix + 1] +=      wx  *      wy  * amp

    # ------------------------------------------------------------------
    # Receiver sampling
    # ------------------------------------------------------------------

    def _sample_receivers(self, n: int) -> None:
        """Sample pressure at local receivers, then gather to rank 0."""
        xp = self.xp
        n_local = len(self._recv_global_idx)
        local_vals = np.zeros(n_local)

        if n_local > 0:
            p_host = xp.asnumpy(self.p_now) if self.is_cuda else self.p_now
            iy = self._recv_iy
            ix = self._recv_ix
            wx = self._recv_wx
            wy = self._recv_wy
            local_vals = (
                p_host[iy,     ix    ] * (1 - wx) * (1 - wy)
              + p_host[iy,     ix + 1] *      wx  * (1 - wy)
              + p_host[iy + 1, ix    ] * (1 - wx) *      wy
              + p_host[iy + 1, ix + 1] *      wx  *      wy
            )

        if not self.use_mpi:
            self.traces[:, n] = local_vals
            return

        # MPI gather (global_idx, value) pairs to rank 0
        local_data = np.column_stack([
            self._recv_global_idx.astype(np.float64),
            local_vals,
        ]) if n_local > 0 else np.empty((0, 2))

        gathered = self.comm.gather(local_data, root=0)

        if self.rank == 0:
            for chunk in gathered:
                if chunk.size == 0:
                    continue
                idxs = chunk[:, 0].astype(int)
                vals = chunk[:, 1]
                self.traces[idxs, n] = vals

    # ------------------------------------------------------------------
    # Halo exchange
    # ------------------------------------------------------------------

    def _halo_exchange(self) -> None:
        """Exchange M ghost rows with MPI neighbours for each field."""
        if not self.use_mpi:
            return

        xp = self.xp
        comm = self.comm
        rank = self.rank
        nx = self.global_nx
        M = self._M

        def _rows_to_host(arr, start, count):
            block = arr[start:start + count, :]
            return xp.asnumpy(block) if self.is_cuda else np.array(block)

        def _rows_from_host(arr, start, buf):
            if self.is_cuda:
                arr[start:start + buf.shape[0], :] = xp.asarray(buf)
            else:
                arr[start:start + buf.shape[0], :] = buf

        gt = self._ghost_top

        for field_idx, field in enumerate([self.p_now, self.p_prev]):
            tag_base = field_idx * 2
            TAG_DOWN = tag_base
            TAG_UP = tag_base + 1

            # Send top M owned rows UP, receive into top ghost
            if rank > 0:
                send = _rows_to_host(field, gt, M)
                recv = np.empty((M, nx))
                comm.Sendrecv(
                    sendbuf=send, dest=rank - 1, sendtag=TAG_UP,
                    recvbuf=recv, source=rank - 1, recvtag=TAG_DOWN,
                )
                _rows_from_host(field, 0, recv)

            # Send bottom M owned rows DOWN, receive into bottom ghost
            if rank < self.size - 1:
                bot_start = gt + self.local_ny - M
                send = _rows_to_host(field, bot_start, M)
                recv = np.empty((M, nx))
                comm.Sendrecv(
                    sendbuf=send, dest=rank + 1, sendtag=TAG_DOWN,
                    recvbuf=recv, source=rank + 1, recvtag=TAG_UP,
                )
                _rows_from_host(field, self._pad_ny - M, recv)

    # ------------------------------------------------------------------
    # Time step
    # ------------------------------------------------------------------

    def _step(self, n: int) -> None:
        """Advance one timestep on the local subdomain."""
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs

        self._halo_exchange()

        p = self.p_now
        pp = self.p_prev
        ny_loc = self._pad_ny
        nx_loc = self.global_nx

        # Interior region: M rows/cols from each edge
        rs, re = M, ny_loc - M
        cs, ce = M, nx_loc - M
        s = (slice(rs, re), slice(cs, ce))

        # Laplacian: centre weight (×2 for x + y contributions)
        lap = coeffs[0] * 2.0 * p[s]

        # Off-centre weights (±k in both dimensions)
        for k in range(1, M + 1):
            ck = coeffs[k]
            lap += ck * (
                p[rs + k : re + k, cs : ce]    # down
              + p[rs - k : re - k, cs : ce]    # up
              + p[rs : re, cs + k : ce + k]    # right
              + p[rs : re, cs - k : ce - k]    # left
            )

        p_next = (
            2.0 * p[s] - pp[s]
            + self.C2[s] * lap
            - self.sigma[s] * (p[s] - pp[s])
        )

        p_new = xp.zeros_like(p)
        p_new[s] = p_next

        # Dirichlet on global boundaries this rank owns
        if self.rank == 0:
            gt = self._ghost_top
            p_new[gt:gt + M, :] = 0.0
        if self.rank == self.size - 1:
            gt = self._ghost_top
            p_new[gt + self.local_ny - M:gt + self.local_ny, :] = 0.0
        p_new[:, :M] = 0.0
        p_new[:, -M:] = 0.0

        self._inject(p_new, n)

        self.p_prev = p
        self.p_now = p_new

        self._sample_receivers(n)

    # ------------------------------------------------------------------
    # Gather full field (for snapshots)
    # ------------------------------------------------------------------

    def _gather_field(self) -> np.ndarray | None:
        """Gather the full pressure field to rank 0."""
        xp = self.xp
        gt = self._ghost_top

        owned = self.p_now[gt: gt + self.local_ny, :]
        owned_host = xp.asnumpy(owned) if self.is_cuda else np.array(owned)

        if not self.use_mpi:
            return owned_host

        gathered = self.comm.gather(owned_host, root=0)
        if self.rank == 0:
            return np.concatenate(gathered, axis=0)
        return None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        snapshot_dir: str | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run the full simulation.

        Returns (on rank 0) a dict with traces, dt, n_steps, path_slices.
        Other ranks return a dict with empty traces.
        """
        from acoustic_sim.plotting import save_snapshot

        is_root = self.rank == 0

        if snapshot_dir is not None and is_root:
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)

        for n in range(self.n_steps):
            self._step(n)

            if (
                snapshot_dir is not None
                and self.cfg.snapshot_interval > 0
                and n % self.cfg.snapshot_interval == 0
            ):
                full_field = self._gather_field()
                if is_root:
                    sx, sy = self.source.position_at(n, self.dt)
                    save_snapshot(
                        self.model, full_field, n, snapshot_dir,
                        receivers=self.receivers,
                        source_xy=np.array([sx, sy]),
                    )

            if verbose and is_root and n % 500 == 0:
                print(f"  step {n:>6d} / {self.n_steps}")

        if verbose and is_root:
            print(f"  step {self.n_steps:>6d} / {self.n_steps}  (done)")

        return {
            "traces": self.traces if is_root else np.empty((0, 0)),
            "dt": self.dt,
            "n_steps": self.n_steps,
            "path_slices": [],
        }

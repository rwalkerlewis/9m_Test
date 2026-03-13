"""3-D FDTD acoustic wave-equation solver with MPI + CUDA support.

Mirrors the 2-D solver in ``fdtd.py``, extended to three spatial
dimensions.

Physics
=======
3-D scalar wave equation on a uniform Cartesian grid::

    d²p/dt² = c² (d²p/dx² + d²p/dy² + d²p/dz²)

discretised with central differences of user-defined order in space
and second-order leapfrog in time.

CFL condition
=============
For 3-D the CFL stability limit is::

    dt <= 2 * dx / (c_eff * sqrt(3 * spec_radius))

where the factor √3 (instead of √2 for 2-D) accounts for the
additional spatial dimension.

MPI decomposition
=================
The global grid of *nz* slabs is split across ranks along the
**z-axis**.  Each rank owns a contiguous set of z-slabs plus **M ghost
slabs** on each internal boundary for the stencil.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from acoustic_sim.backend import get_backend
from acoustic_sim.domains_3d import DomainMeta3D
from acoustic_sim.fdtd import FDTDConfig, fd2_coefficients, fd2_cfl_factor
from acoustic_sim.model_3d import VelocityModel3D


# MPI setup — same pattern as 2-D solver.
try:
    from mpi4py import MPI as _MPI

    _COMM = _MPI.COMM_WORLD
    _HAS_MPI = _COMM.Get_size() > 1
except ImportError:
    _MPI = None  # type: ignore[assignment]
    _COMM = None  # type: ignore[assignment]
    _HAS_MPI = False


def _get_comm() -> tuple[Any, int, int]:
    if _HAS_MPI:
        return _COMM, _COMM.Get_rank(), _COMM.Get_size()
    return None, 0, 1


def _split_slabs(nz: int, size: int) -> list[tuple[int, int]]:
    """Return [(slab_start, slab_end), ...] for each rank (exclusive end)."""
    base = nz // size
    remainder = nz % size
    splits: list[tuple[int, int]] = []
    start = 0
    for r in range(size):
        count = base + (1 if r < remainder else 0)
        splits.append((start, start + count))
        start += count
    return splits


# Re-export config so users can import from this module.
FDTD3DConfig = FDTDConfig


# -----------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------

class FDTD3DSolver:
    """3-D FDTD acoustic solver with MPI domain decomposition and
    optional CUDA acceleration.

    Parameters
    ----------
    model : VelocityModel3D
        *Global* 3-D sound-speed field (nz, ny, nx).
    config : FDTDConfig
        Simulation parameters.
    source : any 3D source object
        Must have ``position_at(step, dt) -> (x, y, z)`` and ``signal``.
    receivers : ndarray (n_recv, 3)
        Receiver positions ``[[x, y, z], ...]``.
    domain_meta : DomainMeta3D | None
        Wind / attenuation metadata.
    """

    def __init__(
        self,
        model: VelocityModel3D,
        config: FDTDConfig,
        source: Any,
        receivers: np.ndarray,
        domain_meta: DomainMeta3D | None = None,
        **_kw: Any,
    ) -> None:
        self.model = model
        self.cfg = config
        self.source = source
        self.receivers = np.asarray(receivers, dtype=np.float64)
        if self.receivers.ndim == 1:
            self.receivers = self.receivers.reshape(1, -1)
        if self.receivers.shape[1] == 2:
            self.receivers = np.column_stack([
                self.receivers, np.zeros(self.receivers.shape[0])
            ])
        self.meta = domain_meta or DomainMeta3D()

        # -- MPI setup --
        self.comm, self.rank, self.size = _get_comm()
        self.use_mpi = self.size > 1

        # -- Backend --
        self.xp, self.is_cuda = get_backend(config.use_cuda)
        xp = self.xp

        nz, ny, nx = model.nz, model.ny, model.nx
        dx = model.dx
        c_global = model.values  # (nz, ny, nx)

        # -- FD stencil --
        self._fd_coeffs = fd2_coefficients(config.fd_order)
        self._M = config.fd_order // 2
        spec_radius = fd2_cfl_factor(self._fd_coeffs)

        # -- CFL: 3-D uses sqrt(3 * spec_radius) --
        c_max = float(np.max(c_global))
        v_wind = math.sqrt(
            self.meta.wind_vx ** 2
            + self.meta.wind_vy ** 2
            + self.meta.wind_vz ** 2
        )
        cfl_limit = 2.0 * dx / (
            (c_max + v_wind) * math.sqrt(3.0 * spec_radius)
        )

        if config.dt is not None:
            self.dt = config.dt
            if self.dt > cfl_limit:
                raise ValueError(
                    f"dt={self.dt:.2e} exceeds 3-D CFL limit {cfl_limit:.2e}"
                )
        else:
            self.dt = config.cfl_safety * cfl_limit

        self.n_steps = int(math.ceil(config.total_time / self.dt))

        # -- Domain decomposition along z --
        self.global_nz = nz
        self.global_ny = ny
        self.global_nx = nx
        M = self._M
        self.splits = _split_slabs(nz, self.size)
        self.slab_start, self.slab_end = self.splits[self.rank]
        self.local_nz = self.slab_end - self.slab_start
        self._ghost_lo = M if self.rank > 0 else 0
        self._ghost_hi = M if self.rank < self.size - 1 else 0
        self._pad_nz = self.local_nz + self._ghost_lo + self._ghost_hi

        # -- Local slice of global arrays --
        g_lo = self.slab_start - self._ghost_lo
        g_hi = self.slab_end + self._ghost_hi
        c_local = c_global[g_lo:g_hi, :, :]

        # -- Squared Courant number --
        self.C2 = xp.asarray((c_local * self.dt / dx) ** 2)

        # -- Damping --
        sigma_global = self._build_damping_global(nz, ny, nx)
        self.sigma = xp.asarray(sigma_global[g_lo:g_hi, :, :])

        # -- Pressure fields (local, 3-D) --
        self.p_now = xp.zeros((self._pad_nz, ny, nx), dtype=np.float64)
        self.p_prev = xp.zeros((self._pad_nz, ny, nx), dtype=np.float64)

        # -- Receivers --
        self._precompute_receivers(model)

        # -- Trace storage --
        if self.rank == 0:
            self.traces = np.zeros((self.receivers.shape[0], self.n_steps))
        else:
            self.traces = None

    # ------------------------------------------------------------------
    # Damping (3-D sponge on all 6 faces)
    # ------------------------------------------------------------------

    def _build_damping_global(self, nz: int, ny: int, nx: int) -> np.ndarray:
        dx_edge = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))
        dy_edge = np.minimum(np.arange(ny), ny - 1 - np.arange(ny))
        dz_edge = np.minimum(np.arange(nz), nz - 1 - np.arange(nz))
        # 3-D minimum distance to any face.
        dist = np.minimum(
            np.minimum(
                dy_edge[:, np.newaxis],  # (ny, 1)
                dx_edge[np.newaxis, :],  # (1, nx)
            )[np.newaxis, :, :],         # (1, ny, nx)
            dz_edge[:, np.newaxis, np.newaxis],  # (nz, 1, 1)
        )  # (nz, ny, nx)

        sigma = np.full((nz, ny, nx), self.cfg.air_absorption)
        w = self.cfg.damping_width
        mask = dist < w
        ramp = ((w - dist[mask]) / w) ** 2
        sigma[mask] = self.cfg.air_absorption + (
            self.cfg.damping_max - self.cfg.air_absorption
        ) * ramp
        return sigma

    # ------------------------------------------------------------------
    # Receiver pre-computation (trilinear)
    # ------------------------------------------------------------------

    def _precompute_receivers(self, model: VelocityModel3D) -> None:
        rx_frac = (self.receivers[:, 0] - model.x[0]) / model.dx
        ry_frac = (self.receivers[:, 1] - model.y[0]) / model.dy
        rz_frac = (self.receivers[:, 2] - model.z[0]) / model.dz

        gix = np.clip(np.floor(rx_frac).astype(int), 0, self.global_nx - 2)
        giy = np.clip(np.floor(ry_frac).astype(int), 0, self.global_ny - 2)
        giz = np.clip(np.floor(rz_frac).astype(int), 0, self.global_nz - 2)
        wx = np.clip(rx_frac - gix, 0.0, 1.0)
        wy = np.clip(ry_frac - giy, 0.0, 1.0)
        wz = np.clip(rz_frac - giz, 0.0, 1.0)

        # Receiver at giz needs slabs giz and giz+1 — both must be owned.
        in_local = (giz >= self.slab_start) & (giz + 1 < self.slab_end)

        self._recv_global_idx = np.where(in_local)[0]
        self._recv_iz = giz[in_local] - self.slab_start + self._ghost_lo
        self._recv_iy = giy[in_local]
        self._recv_ix = gix[in_local]
        self._recv_wx = wx[in_local]
        self._recv_wy = wy[in_local]
        self._recv_wz = wz[in_local]

    # ------------------------------------------------------------------
    # Source injection (trilinear, 8 cells)
    # ------------------------------------------------------------------

    def _inject(self, p: Any, n: int) -> None:
        sx, sy, sz = self.source.position_at(n, self.dt)

        fx = (sx - self.model.x[0]) / self.model.dx
        fy = (sy - self.model.y[0]) / self.model.dy
        fz = (sz - self.model.z[0]) / self.model.dz

        gix = int(math.floor(fx))
        giy = int(math.floor(fy))
        giz = int(math.floor(fz))

        if gix < 0 or gix + 1 >= self.global_nx:
            return
        if giy < 0 or giy + 1 >= self.global_ny:
            return
        if giz < self.slab_start or giz + 1 >= self.slab_end:
            return

        liz = giz - self.slab_start + self._ghost_lo
        wx = fx - gix
        wy = fy - giy
        wz = fz - giz

        sig_val = self.source.signal[min(n, len(self.source.signal) - 1)]
        amp = self.cfg.source_amplitude * sig_val

        # Trilinear injection into 8 surrounding cells.
        for diz, wwz in ((0, 1.0 - wz), (1, wz)):
            for diy, wwy in ((0, 1.0 - wy), (1, wy)):
                for dix, wwx in ((0, 1.0 - wx), (1, wx)):
                    p[liz + diz, giy + diy, gix + dix] += amp * wwx * wwy * wwz

    # ------------------------------------------------------------------
    # Receiver sampling (trilinear)
    # ------------------------------------------------------------------

    def _sample_receivers(self, n: int) -> None:
        xp = self.xp
        n_local = len(self._recv_global_idx)
        local_vals = np.zeros(n_local)

        if n_local > 0:
            p_host = xp.asnumpy(self.p_now) if self.is_cuda else self.p_now
            iz = self._recv_iz
            iy = self._recv_iy
            ix = self._recv_ix
            wx = self._recv_wx
            wy = self._recv_wy
            wz = self._recv_wz
            # Trilinear interpolation over 8 corners.
            local_vals = (
                p_host[iz,     iy,     ix    ] * (1 - wz) * (1 - wy) * (1 - wx)
              + p_host[iz,     iy,     ix + 1] * (1 - wz) * (1 - wy) *      wx
              + p_host[iz,     iy + 1, ix    ] * (1 - wz) *      wy  * (1 - wx)
              + p_host[iz,     iy + 1, ix + 1] * (1 - wz) *      wy  *      wx
              + p_host[iz + 1, iy,     ix    ] *      wz  * (1 - wy) * (1 - wx)
              + p_host[iz + 1, iy,     ix + 1] *      wz  * (1 - wy) *      wx
              + p_host[iz + 1, iy + 1, ix    ] *      wz  *      wy  * (1 - wx)
              + p_host[iz + 1, iy + 1, ix + 1] *      wz  *      wy  *      wx
            )

        if not self.use_mpi:
            self.traces[:, n] = local_vals
            return

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
    # Halo exchange (z-slabs)
    # ------------------------------------------------------------------

    def _halo_exchange(self) -> None:
        if not self.use_mpi:
            return

        xp = self.xp
        comm = self.comm
        rank = self.rank
        ny, nx = self.global_ny, self.global_nx
        M = self._M

        def _slabs_to_host(arr, start, count):
            block = arr[start:start + count, :, :]
            return xp.asnumpy(block) if self.is_cuda else np.array(block)

        def _slabs_from_host(arr, start, buf):
            if self.is_cuda:
                arr[start:start + buf.shape[0], :, :] = xp.asarray(buf)
            else:
                arr[start:start + buf.shape[0], :, :] = buf

        gl = self._ghost_lo

        for field_idx, field in enumerate([self.p_now, self.p_prev]):
            tag_base = field_idx * 2
            TAG_DOWN = tag_base
            TAG_UP = tag_base + 1

            if rank > 0:
                send = _slabs_to_host(field, gl, M)
                recv = np.empty((M, ny, nx))
                comm.Sendrecv(
                    sendbuf=send, dest=rank - 1, sendtag=TAG_UP,
                    recvbuf=recv, source=rank - 1, recvtag=TAG_DOWN,
                )
                _slabs_from_host(field, 0, recv)

            if rank < self.size - 1:
                bot_start = gl + self.local_nz - M
                send = _slabs_to_host(field, bot_start, M)
                recv = np.empty((M, ny, nx))
                comm.Sendrecv(
                    sendbuf=send, dest=rank + 1, sendtag=TAG_DOWN,
                    recvbuf=recv, source=rank + 1, recvtag=TAG_UP,
                )
                _slabs_from_host(field, self._pad_nz - M, recv)

    # ------------------------------------------------------------------
    # Time step
    # ------------------------------------------------------------------

    def _step(self, n: int) -> None:
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs

        self._halo_exchange()

        p = self.p_now
        pp = self.p_prev
        nz_loc = self._pad_nz
        ny_loc = self.global_ny
        nx_loc = self.global_nx

        # Interior region: M cells from each face.
        zs, ze = M, nz_loc - M
        ys, ye = M, ny_loc - M
        xs, xe = M, nx_loc - M
        s = (slice(zs, ze), slice(ys, ye), slice(xs, xe))

        # 3-D Laplacian: centre weight ×3 for three dimensions.
        lap = coeffs[0] * 3.0 * p[s]

        # Off-centre weights in all three directions.
        for k in range(1, M + 1):
            ck = coeffs[k]
            lap += ck * (
                # z-direction
                p[zs + k : ze + k, ys : ye, xs : xe]
              + p[zs - k : ze - k, ys : ye, xs : xe]
                # y-direction
              + p[zs : ze, ys + k : ye + k, xs : xe]
              + p[zs : ze, ys - k : ye - k, xs : xe]
                # x-direction
              + p[zs : ze, ys : ye, xs + k : xe + k]
              + p[zs : ze, ys : ye, xs - k : xe - k]
            )

        p_next = (
            2.0 * p[s] - pp[s]
            + self.C2[s] * lap
            - self.sigma[s] * (p[s] - pp[s])
        )

        p_new = xp.zeros_like(p)
        p_new[s] = p_next

        # Dirichlet BCs on global z-boundaries this rank owns.
        if self.rank == 0:
            gl = self._ghost_lo
            p_new[gl:gl + M, :, :] = 0.0
        if self.rank == self.size - 1:
            gl = self._ghost_lo
            p_new[gl + self.local_nz - M:gl + self.local_nz, :, :] = 0.0
        # y and x boundaries.
        p_new[:, :M, :] = 0.0
        p_new[:, -M:, :] = 0.0
        p_new[:, :, :M] = 0.0
        p_new[:, :, -M:] = 0.0

        self._inject(p_new, n)

        self.p_prev = p
        self.p_now = p_new

        self._sample_receivers(n)

    # ------------------------------------------------------------------
    # Gather full field
    # ------------------------------------------------------------------

    def _gather_field(self) -> np.ndarray | None:
        xp = self.xp
        gl = self._ghost_lo
        owned = self.p_now[gl: gl + self.local_nz, :, :]
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
        snapshot_z_index: int | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run the full 3-D simulation.

        Parameters
        ----------
        snapshot_dir : str or None
            Directory for snapshot images.  None = disabled.
        snapshot_z_index : int or None
            Which z-slab to save as snapshot.  None = middle slab.
        verbose : bool

        Returns
        -------
        dict with ``traces``, ``dt``, ``n_steps``.
        """
        is_root = self.rank == 0

        if snapshot_dir is not None and is_root:
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)

        if snapshot_z_index is None:
            snapshot_z_index = self.global_nz // 2

        for n in range(self.n_steps):
            self._step(n)

            if (
                snapshot_dir is not None
                and self.cfg.snapshot_interval > 0
                and n % self.cfg.snapshot_interval == 0
            ):
                full_field = self._gather_field()
                if is_root and full_field is not None:
                    iz = min(snapshot_z_index, full_field.shape[0] - 1)
                    slice_2d = full_field[iz, :, :]
                    # Save as simple numpy (plotting is user's responsibility).
                    fpath = Path(snapshot_dir) / f"snapshot3d_{n:06d}.npy"
                    np.save(str(fpath), slice_2d)

            if verbose and is_root and n % 500 == 0:
                print(f"  step {n:>6d} / {self.n_steps}")

        if verbose and is_root:
            print(f"  step {self.n_steps:>6d} / {self.n_steps}  (done)")

        return {
            "traces": self.traces if is_root else np.empty((0, 0)),
            "dt": self.dt,
            "n_steps": self.n_steps,
        }

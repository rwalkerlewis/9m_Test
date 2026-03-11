"""2-D FDTD acoustic wave-equation solver with MPI domain decomposition.

Supports:
* optional CUDA acceleration (via CuPy, see :mod:`acoustic_sim.backend`)
* wind advection (linearised convected wave equation)
* per-cell attenuation (vegetation / absorbing zones)
* sponge absorbing boundaries
* moving and static sources with audio-based signals
* receiver trace recording with bilinear interpolation
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from acoustic_sim.backend import get_backend
from acoustic_sim.domains import DomainMeta
from acoustic_sim.model import VelocityModel
from acoustic_sim.sources import StaticSource, MovingSource, inject_source


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FDTDConfig:
    """Parameters that control the FDTD simulation."""

    total_time: float = 0.3       # [s]
    dt: float | None = None       # auto from CFL when None
    cfl_safety: float = 0.9
    damping_width: int = 20       # sponge layer thickness [cells]
    damping_max: float = 0.05     # peak sponge damping coefficient
    snapshot_interval: int = 50   # save snapshot every N steps (0 = off)
    use_cuda: bool = False


# ---------------------------------------------------------------------------
# MPI domain decomposition
# ---------------------------------------------------------------------------

class MPIDomain:
    """2-D Cartesian MPI domain decomposition with ghost (halo) cells."""

    def __init__(
        self,
        model: VelocityModel,
        comm: Any,
    ) -> None:
        self.comm = comm
        self.rank: int = comm.Get_rank()
        self.size: int = comm.Get_size()

        # Full global grid sizes.
        self.gny, self.gnx = model.ny, model.nx

        # 2-D process topology.
        from mpi4py import MPI

        dims = MPI.Compute_dims(self.size, 2)
        self.cart = comm.Create_cart(dims, periods=[False, False], reorder=True)
        self.cart_rank = self.cart.Get_rank()
        self.coords = self.cart.Get_coords(self.cart_rank)
        self.dims = dims  # (rows, cols) of process grid

        # Neighbour ranks (MPI_PROC_NULL if at edge).
        self.nbr_up, self.nbr_down = self.cart.Shift(0, 1)
        self.nbr_left, self.nbr_right = self.cart.Shift(1, 1)

        # Compute local slice of the global grid (interior, no ghosts).
        def _split(n: int, nprocs: int, rank: int) -> tuple[int, int]:
            base = n // nprocs
            rem = n % nprocs
            start = rank * base + min(rank, rem)
            size = base + (1 if rank < rem else 0)
            return start, size

        self.iy_start, self.local_ny = _split(self.gny, dims[0], self.coords[0])
        self.ix_start, self.local_nx = _split(self.gnx, dims[1], self.coords[1])

        # Ghost-padded dimensions (1 ghost cell on each side).
        self.ghost = 1
        self.padded_ny = self.local_ny + 2 * self.ghost
        self.padded_nx = self.local_nx + 2 * self.ghost

    # -- halo exchange -------------------------------------------------------

    def exchange_halos(self, field: np.ndarray) -> None:
        """Exchange ghost layers with four Cartesian neighbours."""
        from mpi4py import MPI

        g = self.ghost
        tag = 0
        reqs: list[Any] = []

        # Up / down (along y / axis-0 of the array).
        send_up = np.ascontiguousarray(field[g, :])
        send_dn = np.ascontiguousarray(field[-g - 1, :])
        recv_up = np.empty_like(send_up)
        recv_dn = np.empty_like(send_dn)
        reqs.append(self.cart.Isend(send_up, dest=self.nbr_up, tag=tag))
        reqs.append(self.cart.Irecv(recv_dn, source=self.nbr_down, tag=tag))
        reqs.append(self.cart.Isend(send_dn, dest=self.nbr_down, tag=tag + 1))
        reqs.append(self.cart.Irecv(recv_up, source=self.nbr_up, tag=tag + 1))

        # Left / right (along x / axis-1).
        send_left = np.ascontiguousarray(field[:, g])
        send_right = np.ascontiguousarray(field[:, -g - 1])
        recv_left = np.empty_like(send_left)
        recv_right = np.empty_like(send_right)
        reqs.append(self.cart.Isend(send_left, dest=self.nbr_left, tag=tag + 2))
        reqs.append(self.cart.Irecv(recv_right, source=self.nbr_right, tag=tag + 2))
        reqs.append(self.cart.Isend(send_right, dest=self.nbr_right, tag=tag + 3))
        reqs.append(self.cart.Irecv(recv_left, source=self.nbr_left, tag=tag + 3))

        MPI.Request.Waitall(reqs)

        if self.nbr_down != MPI.PROC_NULL:
            field[-1, :] = recv_dn
        if self.nbr_up != MPI.PROC_NULL:
            field[0, :] = recv_up
        if self.nbr_right != MPI.PROC_NULL:
            field[:, -1] = recv_right
        if self.nbr_left != MPI.PROC_NULL:
            field[:, 0] = recv_left

    # -- gather full field to rank 0 -----------------------------------------

    def gather_field(self, local_interior: np.ndarray) -> np.ndarray | None:
        """Gather local interior arrays to a full global field on rank 0."""
        from mpi4py import MPI

        # Send local_interior to rank 0.
        local_flat = np.ascontiguousarray(local_interior).ravel()
        sizes = self.comm.gather(local_flat.size, root=0)
        if self.rank == 0:
            recv_buf = np.empty(sum(sizes), dtype=local_flat.dtype)
            displs = [0] + list(np.cumsum(sizes[:-1]))
        else:
            recv_buf = None
            displs = None

        # Use Gatherv.
        self.comm.Gatherv(local_flat, (recv_buf, sizes, displs, MPI.DOUBLE), root=0)

        if self.rank == 0:
            global_field = np.zeros((self.gny, self.gnx), dtype=np.float64)
            offset = 0
            for r in range(self.size):
                coords_r = self.cart.Get_coords(r)
                def _split(n, nprocs, rk):
                    base = n // nprocs
                    rem = n % nprocs
                    start = rk * base + min(rk, rem)
                    size = base + (1 if rk < rem else 0)
                    return start, size
                iy_s, lny = _split(self.gny, self.dims[0], coords_r[0])
                ix_s, lnx = _split(self.gnx, self.dims[1], coords_r[1])
                chunk = recv_buf[offset : offset + lny * lnx].reshape(lny, lnx)
                global_field[iy_s : iy_s + lny, ix_s : ix_s + lnx] = chunk
                offset += lny * lnx
            return global_field
        return None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class FDTDSolver:
    """2-D FDTD acoustic solver with MPI + optional CUDA."""

    def __init__(
        self,
        model: VelocityModel,
        config: FDTDConfig,
        source: StaticSource | MovingSource,
        receivers: np.ndarray,
        comm: Any,
        domain_meta: DomainMeta | None = None,
    ) -> None:
        from mpi4py import MPI

        self.model = model
        self.config = config
        self.source = source
        self.receivers = receivers  # (n_recv, 2)
        self.comm = comm
        self.meta = domain_meta or DomainMeta()

        self.xp, self.is_cuda = get_backend(config.use_cuda)
        xp = self.xp

        # MPI domain.
        self.mpi = MPIDomain(model, comm)
        m = self.mpi
        g = m.ghost

        # Local velocity slice (with ghost padding).
        full_c = model.values  # (gny, gnx)
        local_c = full_c[
            max(m.iy_start - g, 0) : m.iy_start + m.local_ny + g,
            max(m.ix_start - g, 0) : m.ix_start + m.local_nx + g,
        ]
        # Pad if at edge (ghost outside domain → replicate boundary).
        pad_top = g - (m.iy_start - max(m.iy_start - g, 0))
        pad_bot = (m.padded_ny) - local_c.shape[0] - pad_top
        pad_left = g - (m.ix_start - max(m.ix_start - g, 0))
        pad_right = (m.padded_nx) - local_c.shape[1] - pad_left
        local_c = np.pad(local_c, ((pad_top, pad_bot), (pad_left, pad_right)), mode="edge")
        self.local_c = xp.asarray(local_c.astype(np.float64))

        # Timestep from CFL.
        c_max = float(np.max(model.values))
        dx = model.dx
        cfl_dt = dx / (c_max * math.sqrt(2.0))
        self.dt = config.dt if config.dt is not None else config.cfl_safety * cfl_dt
        if self.dt > cfl_dt:
            raise ValueError(
                f"dt={self.dt:.2e} exceeds CFL limit {cfl_dt:.2e}. "
                f"Reduce dt or increase dx."
            )

        self.n_steps = int(math.ceil(config.total_time / self.dt))

        # Coefficient array: (c * dt / dx)².
        self.coeff = (self.local_c * self.dt / dx) ** 2

        # Sponge absorbing boundary.
        self._build_sponge()

        # Merge vegetation / external attenuation.
        if self.meta.attenuation is not None:
            ext_a = self.meta.attenuation[
                max(m.iy_start - g, 0) : m.iy_start + m.local_ny + g,
                max(m.ix_start - g, 0) : m.ix_start + m.local_nx + g,
            ]
            ext_a = np.pad(ext_a, ((pad_top, pad_bot), (pad_left, pad_right)), mode="constant")
            self.sponge = self.sponge + xp.asarray(ext_a)

        # Wind.
        self.wind_vx = self.meta.wind_vx
        self.wind_vy = self.meta.wind_vy
        self.has_wind = abs(self.wind_vx) > 1e-8 or abs(self.wind_vy) > 1e-8
        if self.has_wind:
            wind_mag = math.hypot(self.wind_vx, self.wind_vy)
            if wind_mag >= c_max:
                raise ValueError(
                    f"Wind speed ({wind_mag:.1f} m/s) must be subsonic "
                    f"(< c_max = {c_max:.1f} m/s)."
                )
            self.w_cx = self.wind_vx * self.dt / dx
            self.w_cy = self.wind_vy * self.dt / dx

        # Pressure fields (ghost-padded).
        self.p_now = xp.zeros((m.padded_ny, m.padded_nx), dtype=np.float64)
        self.p_prev = xp.zeros_like(self.p_now)

        # Receiver bookkeeping — which receivers are in this rank's subdomain?
        self.n_recv = receivers.shape[0]
        self._setup_receivers()

    # -- internal helpers ----------------------------------------------------

    def _build_sponge(self) -> None:
        """Quadratic sponge damping array on the *global* boundary cells."""
        m = self.mpi
        g = m.ghost
        cfg = self.config
        xp = self.xp

        sponge = np.zeros((m.padded_ny, m.padded_nx), dtype=np.float64)
        w = cfg.damping_width
        a_max = cfg.damping_max

        for lj in range(m.padded_ny):
            gj = m.iy_start + lj - g
            for li in range(m.padded_nx):
                gi = m.ix_start + li - g
                d_edge = min(gi, m.gnx - 1 - gi, gj, m.gny - 1 - gj)
                if d_edge < w:
                    sponge[lj, li] = a_max * ((w - d_edge) / w) ** 2

        self.sponge = xp.asarray(sponge)

    def _setup_receivers(self) -> None:
        """Determine which receivers are in this rank and their local indices."""
        m = self.mpi
        g = m.ghost
        model = self.model
        self.local_recv_ids: list[int] = []
        self.local_recv_gx: list[float] = []
        self.local_recv_gy: list[float] = []
        for i, (rx, ry) in enumerate(self.receivers):
            gx = (rx - float(model.x[0])) / model.dx
            gy = (ry - float(model.y[0])) / model.dy
            ix = int(round(gx))
            iy = int(round(gy))
            if (
                m.ix_start <= ix < m.ix_start + m.local_nx
                and m.iy_start <= iy < m.iy_start + m.local_ny
            ):
                self.local_recv_ids.append(i)
                self.local_recv_gx.append(gx)
                self.local_recv_gy.append(gy)

        # Traces array: full size, this rank fills its own columns.
        self.traces = np.zeros((self.n_recv, self.n_steps), dtype=np.float64)

    def _record_traces(self, step: int) -> None:
        """Sample pressure at each local receiver via bilinear interpolation."""
        m = self.mpi
        g = m.ghost
        xp = self.xp
        p = self.p_now
        if self.is_cuda:
            p = xp.asnumpy(p)  # type: ignore[union-attr]

        for rid, gx, gy in zip(
            self.local_recv_ids, self.local_recv_gx, self.local_recv_gy
        ):
            ix0 = int(math.floor(gx))
            iy0 = int(math.floor(gy))
            fx = gx - ix0
            fy = gy - iy0
            # Local padded indices.
            li = ix0 - m.ix_start + g
            lj = iy0 - m.iy_start + g
            val = 0.0
            for dj, wy in ((0, 1.0 - fy), (1, fy)):
                for di, wx in ((0, 1.0 - fx), (1, fx)):
                    jj = lj + dj
                    ii = li + di
                    if 0 <= jj < p.shape[0] and 0 <= ii < p.shape[1]:
                        val += float(p[jj, ii]) * wx * wy
            self.traces[rid, step] = val

    # -- main loop -----------------------------------------------------------

    def run(
        self,
        snapshot_dir: str | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run the full FDTD simulation.

        Returns
        -------
        dict
            ``{"traces": np.ndarray, "dt": float, "n_steps": int}``
            *traces* has shape ``(n_receivers, n_steps)`` and is only
            valid on rank 0.
        """
        from mpi4py import MPI

        xp = self.xp
        m = self.mpi
        g = m.ghost
        dt = self.dt
        dx = self.model.dx
        cfg = self.config

        if snapshot_dir is not None and self.comm.Get_rank() == 0:
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)

        # Estimate vmin/vmax from first few steps for consistent colorscale.
        snap_vmax = 0.0

        for n in range(self.n_steps):
            # 1. Halo exchange (on numpy arrays — transfer from GPU if needed).
            if self.is_cuda:
                p_np = xp.asnumpy(self.p_now)  # type: ignore[union-attr]
                m.exchange_halos(p_np)
                self.p_now = xp.asarray(p_np)
            else:
                m.exchange_halos(self.p_now)

            # 2. Laplacian (interior of padded array).
            p = self.p_now
            lap = (
                p[2:, 1:-1]
                + p[:-2, 1:-1]
                + p[1:-1, 2:]
                + p[1:-1, :-2]
                - 4.0 * p[1:-1, 1:-1]
            )

            # 3. Update.
            interior = slice(g, -g), slice(g, -g)
            p_next = (
                2.0 * p[interior]
                - self.p_prev[interior]
                + self.coeff[interior] * lap
                - self.sponge[interior] * (p[interior] - self.p_prev[interior])
            )

            # 4. Wind advection cross-term.
            if self.has_wind:
                dp_dt = (p[interior] - self.p_prev[interior]) / dt
                # Spatial gradient of dp_dt (central differences on dp_dt).
                # dp_dt lives on the interior; we need its spatial neighbours,
                # so we use the slightly larger stencil from the padded p arrays.
                dp_dt_full = (p - self.p_prev) / dt
                adv_x = self.w_cx * (
                    dp_dt_full[1:-1, 2:] - dp_dt_full[1:-1, :-2]
                ) / 2.0
                adv_y = self.w_cy * (
                    dp_dt_full[2:, 1:-1] - dp_dt_full[:-2, 1:-1]
                ) / 2.0
                p_next -= 2.0 * dt * (adv_x + adv_y)

            # 5. Write back into a new padded array.
            p_new = xp.zeros_like(self.p_now)
            p_new[interior] = p_next

            # 6. Inject source.
            src_x, src_y = self.source.position_at(n, dt)
            amp = float(self.source.signal[min(n, len(self.source.signal) - 1)])
            if self.is_cuda:
                p_new_np = xp.asnumpy(p_new)  # type: ignore[union-attr]
            else:
                p_new_np = p_new
            inject_source(
                p_new_np,
                src_x,
                src_y,
                amp * dt * dt,
                self.model.x,
                self.model.y,
                dx,
                self.model.dy,
                ix_offset=m.ix_start - g,
                iy_offset=m.iy_start - g,
            )
            if self.is_cuda:
                p_new = xp.asarray(p_new_np)

            # 7. Rotate fields.
            self.p_prev = self.p_now
            self.p_now = p_new

            # 8. Record traces.
            self._record_traces(n)

            # 9. Snapshots.
            if (
                snapshot_dir is not None
                and cfg.snapshot_interval > 0
                and n % cfg.snapshot_interval == 0
            ):
                if self.is_cuda:
                    loc_int = xp.asnumpy(self.p_now[g:-g, g:-g])  # type: ignore[union-attr]
                else:
                    loc_int = self.p_now[g:-g, g:-g]
                global_field = m.gather_field(np.ascontiguousarray(loc_int))
                if self.comm.Get_rank() == 0 and global_field is not None:
                    vmax_now = float(np.max(np.abs(global_field)))
                    if vmax_now > snap_vmax:
                        snap_vmax = vmax_now
                    from acoustic_sim.plotting import save_snapshot

                    save_snapshot(
                        self.model,
                        global_field,
                        n,
                        snapshot_dir,
                        receivers=self.receivers,
                        source_xy=np.array([src_x, src_y]),
                        vmin=-max(snap_vmax, 1e-30),
                        vmax=max(snap_vmax, 1e-30),
                    )

            if verbose and self.comm.Get_rank() == 0 and n % max(self.n_steps // 10, 1) == 0:
                pct = 100.0 * n / self.n_steps
                print(f"  FDTD step {n}/{self.n_steps}  ({pct:.0f}%)")

        # 10. Reduce traces to rank 0.
        all_traces = np.zeros_like(self.traces)
        self.comm.Reduce(self.traces, all_traces, op=MPI.SUM, root=0)

        return {"traces": all_traces, "dt": self.dt, "n_steps": self.n_steps}

"""Unified elastic velocity-stress FDTD solver (2-D and 3-D).

Solves the isotropic elastic wave equation using the Virieux (1986)
staggered-grid finite-difference scheme.  By setting Vs = 0 (μ = 0) in
air cells, the elastic equations reduce exactly to the acoustic wave
equation, so a single solver handles both the air and ground domains
without explicit interface coupling.

Features
--------
* Variable spatial FD order (2, 4, 6, 8) — runtime selectable.
* GPU acceleration via CuPy (array operations only, no custom kernels).
* MPI domain decomposition along the z-axis.
* Viscoelastic attenuation via Standard Linear Solid (SLS) memory
  variables (one mechanism).
* Sponge-layer absorbing boundaries on all faces.
* Microphone receivers (pressure in air) and geophone receivers
  (vertical velocity in ground).

This module is completely standalone.  It does **not** import from or
share any code with the existing acoustic ``fdtd.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from acoustic_sim.backend import get_backend
from acoustic_sim.elastic_model import ElasticModel2D, ElasticModel3D
from acoustic_sim.receivers import ReceiverSpec

# Try MPI; if unavailable, run single-process.
try:
    from mpi4py import MPI as _MPI

    _COMM = _MPI.COMM_WORLD
    _HAS_MPI = _COMM.Get_size() > 1
except ImportError:
    _MPI = None  # type: ignore[assignment]
    _COMM = None  # type: ignore[assignment]
    _HAS_MPI = False


def _get_comm() -> tuple[Any, int, int]:
    """Return ``(comm, rank, size)``.  Falls back to ``(None, 0, 1)``."""
    if _HAS_MPI:
        return _COMM, _COMM.Get_rank(), _COMM.Get_size()
    return None, 0, 1


# ═══════════════════════════════════════════════════════════════════════
# Stencil coefficients — first derivative, staggered grid
# ═══════════════════════════════════════════════════════════════════════

_STAGGERED_COEFFS: dict[int, np.ndarray] = {
    2: np.array([1.0]),
    4: np.array([9.0 / 8.0, -1.0 / 24.0]),
    6: np.array([75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0]),
    8: np.array([
        1225.0 / 1024.0,
        -245.0 / 3072.0,
        49.0 / 5120.0,
        -5.0 / 7168.0,
    ]),
}


def fd1_staggered_coefficients(order: int) -> np.ndarray:
    r"""Staggered-grid first-derivative FD coefficients.

    For a first derivative at a half-grid point:

    .. math::

        \left.\frac{\partial f}{\partial x}\right|_{i+\frac12}
        \approx \frac{1}{\Delta x} \sum_{k=1}^{M} c_k
        \bigl(f_{i+k} - f_{i-k+1}\bigr)

    where *M = order / 2* and ``c_k = coefficients[k-1]``.

    Supported orders: 2, 4, 6, 8.

    Parameters
    ----------
    order : int
        Spatial FD order (2, 4, 6, or 8).

    Returns
    -------
    np.ndarray
        Coefficients ``[c1, c2, …, cM]``.
    """
    if order not in _STAGGERED_COEFFS:
        raise ValueError(
            f"Unsupported FD order {order}; choose from {sorted(_STAGGERED_COEFFS)}"
        )
    return _STAGGERED_COEFFS[order].copy()


def elastic_cfl_factor(coeffs: np.ndarray) -> float:
    r"""CFL stability factor for the velocity-stress FDTD.

    The 1-D CFL limit is

    .. math::

        \Delta t \le \frac{\Delta x}{V_{\max} \cdot S}

    where *S* is the sum of the absolute stencil coefficients.
    For *N*-D multiply *S* by :math:`\sqrt{N}`.

    Returns *S* (the 1-D factor).
    """
    return float(np.sum(np.abs(coeffs)))


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ElasticFDTDConfig:
    """Configuration for the elastic FDTD solver.

    Attributes
    ----------
    total_time : float
        Simulation duration [s].
    dt : float or None
        Timestep [s]; ``None`` → auto-compute from CFL.
    cfl_safety : float
        Fraction of the CFL limit to use for dt.
    damping_width : int
        Sponge-layer thickness in grid cells.
    damping_max : float
        Peak sponge damping coefficient at domain edge.
    snapshot_interval : int
        Save a wavefield snapshot every N steps (0 = disabled).
    source_amplitude : float
        Peak source pressure [Pa].
    use_cuda : bool
        Use CuPy for GPU acceleration.
    fd_order : int
        Spatial FD order (2, 4, 6, or 8).
    enable_attenuation : bool
        Enable SLS viscoelastic attenuation.
    attenuation_f0 : float
        Reference frequency for SLS relaxation [Hz].
    """

    total_time: float = 0.3
    dt: float | None = None
    cfl_safety: float = 0.8
    damping_width: int = 40
    damping_max: float = 0.05
    snapshot_interval: int = 50
    source_amplitude: float = 1.0
    use_cuda: bool = False
    fd_order: int = 4
    enable_attenuation: bool = True
    attenuation_f0: float = 50.0


# ═══════════════════════════════════════════════════════════════════════
# MPI domain decomposition helper
# ═══════════════════════════════════════════════════════════════════════

def _split_slabs(n: int, size: int) -> list[tuple[int, int]]:
    """Split *n* items across *size* ranks.  Returns ``[(start, end), …]``."""
    base = n // size
    remainder = n % size
    splits: list[tuple[int, int]] = []
    start = 0
    for r in range(size):
        count = base + (1 if r < remainder else 0)
        splits.append((start, start + count))
        start += count
    return splits


# ═══════════════════════════════════════════════════════════════════════
# 2-D Elastic FDTD Solver
# ═══════════════════════════════════════════════════════════════════════

class ElasticFDTD2DSolver:
    """2-D elastic velocity-stress FDTD solver.

    Solves the Virieux (1986) staggered-grid elastic wave equation on an
    (x, z) grid with variable-order spatial FD stencils.

    The source is injected as an isotropic pressure perturbation
    (``σxx -= amp``, ``σzz -= amp``).  Microphones record pressure
    ``p = −(σxx + σzz) / 2``, geophones record ``vz``.

    Parameters
    ----------
    model : ElasticModel2D
        Material properties on the global grid.
    config : ElasticFDTDConfig
        Simulation parameters.
    source : any source with ``position_at(step, dt)`` and ``signal``
        The source object (StaticSource, MovingSource, etc.).
    receiver_spec : ReceiverSpec
        Combined microphone + geophone specification.
    """

    def __init__(
        self,
        model: ElasticModel2D,
        config: ElasticFDTDConfig,
        source: Any,
        receiver_spec: ReceiverSpec,
        **_kw: Any,
    ) -> None:
        self.model = model
        self.cfg = config
        self.source = source
        self.receiver_spec = receiver_spec

        # -- MPI setup --
        self.comm, self.rank, self.size = _get_comm()
        self.use_mpi = self.size > 1

        # -- Backend --
        self.xp, self.is_cuda = get_backend(config.use_cuda)
        xp = self.xp

        nz, nx = model.nz, model.nx
        dx = model.dx

        # -- FD stencil --
        self._fd_coeffs = fd1_staggered_coefficients(config.fd_order)
        self._M = config.fd_order // 2
        S = elastic_cfl_factor(self._fd_coeffs)

        # -- CFL: dt <= dx / (Vmax * S * sqrt(ndim)) --
        vp_max = model.vp_max
        cfl_limit = dx / (vp_max * S * math.sqrt(2.0))

        if config.dt is not None:
            self.dt = config.dt
            if self.dt > cfl_limit:
                raise ValueError(
                    f"dt={self.dt:.2e} exceeds CFL limit {cfl_limit:.2e}"
                )
        else:
            self.dt = config.cfl_safety * cfl_limit

        self.n_steps = int(math.ceil(config.total_time / self.dt))

        # -- Domain decomposition along z --
        self.global_nz = nz
        self.global_nx = nx
        M = self._M
        self.splits = _split_slabs(nz, self.size)
        self.slab_start, self.slab_end = self.splits[self.rank]
        self.local_nz = self.slab_end - self.slab_start
        self._ghost_lo = M if self.rank > 0 else 0
        self._ghost_hi = M if self.rank < self.size - 1 else 0
        self._pad_nz = self.local_nz + self._ghost_lo + self._ghost_hi

        # -- Local slice of global material arrays --
        g_lo = self.slab_start - self._ghost_lo
        g_hi = self.slab_end + self._ghost_hi

        lam_global = model.lambda_
        mu_global = model.mu
        rho_global = model.rho
        vs_global = model.vs

        lam_local = lam_global[g_lo:g_hi, :]
        mu_local = mu_global[g_lo:g_hi, :]
        rho_local = rho_global[g_lo:g_hi, :]
        vs_local = vs_global[g_lo:g_hi, :]

        # Precomputed material arrays on device
        # Normal stress moduli at cell centres (integer grid)
        self.lambda_ = xp.asarray(lam_local)
        self.mu = xp.asarray(mu_local)
        self.lambda_2mu = self.lambda_ + 2.0 * self.mu

        # Averaged buoyancy for velocity update (Graves 1996):
        #   vx at (i+1/2, j): rho_avg = (rho[i,j] + rho[i+1,j])/2
        #   vz at (i, j+1/2): rho_avg = (rho[i,j] + rho[i,j+1])/2
        pad_nz = self._pad_nz
        buoy_x = np.zeros((pad_nz, nx), dtype=np.float64)
        buoy_z = np.zeros((pad_nz, nx), dtype=np.float64)
        buoy_x[:, :-1] = 2.0 / (rho_local[:, :-1] + rho_local[:, 1:])
        buoy_x[:, -1] = 1.0 / rho_local[:, -1]
        buoy_z[:-1, :] = 2.0 / (rho_local[:-1, :] + rho_local[1:, :])
        buoy_z[-1, :] = 1.0 / rho_local[-1, :]
        self.buoyancy_vx = xp.asarray(buoy_x)
        self.buoyancy_vz = xp.asarray(buoy_z)

        # Averaged shear modulus at (i+1/2, j+1/2) for sxz update.
        # Use harmonic mean: zero if any contributor is zero (fluid).
        mu_xz = np.zeros((pad_nz, nx), dtype=np.float64)
        m00 = mu_local[:-1, :-1]
        m10 = mu_local[:-1, 1:]
        m01 = mu_local[1:, :-1]
        m11 = mu_local[1:, 1:]
        nonzero = (m00 > 0) & (m10 > 0) & (m01 > 0) & (m11 > 0)
        mu_xz_inner = np.zeros_like(m00)
        mu_xz_inner[nonzero] = 4.0 / (
            1.0 / m00[nonzero] + 1.0 / m10[nonzero]
            + 1.0 / m01[nonzero] + 1.0 / m11[nonzero]
        )
        mu_xz[:-1, :-1] = mu_xz_inner
        self.mu_xz = xp.asarray(mu_xz)

        self.mu_mask = xp.asarray((vs_local > 0).astype(np.float64))

        # -- Field arrays --
        shape = (self._pad_nz, nx)
        self.vx = xp.zeros(shape, dtype=np.float64)
        self.vz = xp.zeros(shape, dtype=np.float64)
        self.sxx = xp.zeros(shape, dtype=np.float64)
        self.szz = xp.zeros(shape, dtype=np.float64)
        self.sxz = xp.zeros(shape, dtype=np.float64)

        # -- Sponge layer damping --
        self._damping = xp.asarray(
            self._build_damping(nz, nx, g_lo, g_hi)
        )

        # -- SLS attenuation --
        self._setup_attenuation(model, g_lo, g_hi)

        # -- Receivers --
        self._precompute_receivers()

        # -- Trace storage --
        n_recv = receiver_spec.n_receivers
        if self.rank == 0:
            self.traces = np.zeros((n_recv, self.n_steps))
        else:
            self.traces = None

    # ------------------------------------------------------------------
    # Sponge layer
    # ------------------------------------------------------------------

    def _build_damping(
        self, nz: int, nx: int, g_lo: int, g_hi: int,
    ) -> np.ndarray:
        """Build sponge damping array for the local subdomain."""
        w = self.cfg.damping_width
        dmax = self.cfg.damping_max
        if w <= 0:
            return np.zeros((g_hi - g_lo, nx), dtype=np.float64)

        # Global distance to nearest boundary
        dx_edge = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))
        dz_edge = np.minimum(np.arange(nz), nz - 1 - np.arange(nz))
        dist = np.minimum(
            dz_edge[:, np.newaxis],
            dx_edge[np.newaxis, :],
        )

        sigma = np.zeros((nz, nx), dtype=np.float64)
        mask = dist < w
        sigma[mask] = dmax * ((w - dist[mask]) / w) ** 2
        return sigma[g_lo:g_hi, :]

    # ------------------------------------------------------------------
    # SLS attenuation setup
    # ------------------------------------------------------------------

    def _setup_attenuation(
        self, model: ElasticModel2D, g_lo: int, g_hi: int,
    ) -> None:
        """Precompute SLS memory variable coefficients."""
        xp = self.xp
        self._has_attenuation = self.cfg.enable_attenuation

        if not self._has_attenuation:
            self.Rxx = None
            self.Rzz = None
            self.Rxz = None
            return

        f0 = self.cfg.attenuation_f0
        dt = self.dt

        qp_local = model.qp[g_lo:g_hi, :]
        qs_local = model.qs[g_lo:g_hi, :]

        # SLS relaxation times
        tau_sigma = 1.0 / (2.0 * np.pi * f0)

        # tau_epsilon = tau_sigma * (2Q+1)/(2Q-1)
        # For very large Q → tau_epsilon ≈ tau_sigma (no attenuation)
        tau_eps_p = tau_sigma * (2.0 * qp_local + 1.0) / np.maximum(2.0 * qp_local - 1.0, 0.01)
        tau_eps_s = tau_sigma * (2.0 * qs_local + 1.0) / np.maximum(2.0 * qs_local - 1.0, 0.01)

        # Precomputed coefficients:
        #   decay = exp(-dt / tau_sigma)
        #   coeff = (tau_eps / tau_sigma - 1) * (1 - decay)
        decay = math.exp(-dt / tau_sigma)
        self._sls_decay = decay

        coeff_p = (tau_eps_p / tau_sigma - 1.0) * (1.0 - decay)
        coeff_s = (tau_eps_s / tau_sigma - 1.0) * (1.0 - decay)

        self._sls_coeff_p = xp.asarray(coeff_p)
        self._sls_coeff_s = xp.asarray(coeff_s)

        # Memory variables
        shape = (self._pad_nz, self.global_nx)
        self.Rxx = xp.zeros(shape, dtype=np.float64)
        self.Rzz = xp.zeros(shape, dtype=np.float64)
        self.Rxz = xp.zeros(shape, dtype=np.float64)

    # ------------------------------------------------------------------
    # Staggered-grid spatial derivatives
    # ------------------------------------------------------------------

    def _diff_x_forward(self, f: Any) -> Any:
        r"""∂f/∂x at (i+½, j): forward staggered difference in x.

        Uses interior region [M : -M] in both dimensions.
        """
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs
        nz, nx = f.shape
        out = xp.zeros_like(f)
        for k in range(1, M + 1):
            c = coeffs[k - 1]
            # f[i+k] - f[i-k+1]  (half-grid offset)
            out[:, M:-M] += c * (
                f[:, M + k: nx - M + k]
                - f[:, M - k + 1: nx - M - k + 1]
            )
        out /= self.model.dx
        return out

    def _diff_x_backward(self, f: Any) -> Any:
        r"""∂f/∂x at (i, j): backward staggered difference in x.

        Adjoint of forward: f[i+k-1] - f[i-k]
        """
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs
        nz, nx = f.shape
        out = xp.zeros_like(f)
        for k in range(1, M + 1):
            c = coeffs[k - 1]
            out[:, M:-M] += c * (
                f[:, M + k - 1: nx - M + k - 1]
                - f[:, M - k: nx - M - k]
            )
        out /= self.model.dx
        return out

    def _diff_z_forward(self, f: Any) -> Any:
        r"""∂f/∂z at (i, j+½): forward staggered difference in z."""
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs
        nz, nx = f.shape
        out = xp.zeros_like(f)
        for k in range(1, M + 1):
            c = coeffs[k - 1]
            out[M:-M, :] += c * (
                f[M + k: nz - M + k, :]
                - f[M - k + 1: nz - M - k + 1, :]
            )
        out /= self.model.dz
        return out

    def _diff_z_backward(self, f: Any) -> Any:
        r"""∂f/∂z at (i, j): backward staggered difference in z."""
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs
        nz, nx = f.shape
        out = xp.zeros_like(f)
        for k in range(1, M + 1):
            c = coeffs[k - 1]
            out[M:-M, :] += c * (
                f[M + k - 1: nz - M + k - 1, :]
                - f[M - k: nz - M - k, :]
            )
        out /= self.model.dz
        return out

    # ------------------------------------------------------------------
    # Source injection
    # ------------------------------------------------------------------

    def _inject(self, n: int) -> None:
        """Inject pressure source into σxx and σzz."""
        xp = self.xp
        # position_at returns (x, y) for 2D sources — we interpret y as z
        pos = self.source.position_at(n, self.dt)
        sx, sz = pos[0], pos[1]

        fx = (sx - self.model.x[0]) / self.model.dx
        fz = (sz - self.model.z[0]) / self.model.dz

        gix = int(math.floor(fx))
        giz = int(math.floor(fz))

        if gix < 0 or gix + 1 >= self.global_nx:
            return
        if giz < self.slab_start or giz + 1 >= self.slab_end:
            return

        liz = giz - self.slab_start + self._ghost_lo
        wx = fx - gix
        wz = fz - giz

        sig_val = self.source.signal[min(n, len(self.source.signal) - 1)]
        amp = self.cfg.source_amplitude * sig_val

        # Inject as isotropic pressure: sxx -= amp, szz -= amp
        # Using bilinear interpolation over 4 surrounding cells
        corners = [
            (liz,     gix,     (1 - wz) * (1 - wx)),
            (liz,     gix + 1, (1 - wz) * wx),
            (liz + 1, gix,     wz * (1 - wx)),
            (liz + 1, gix + 1, wz * wx),
        ]
        for iz, ix, w in corners:
            self.sxx[iz, ix] -= amp * w
            self.szz[iz, ix] -= amp * w

    # ------------------------------------------------------------------
    # Receiver sampling
    # ------------------------------------------------------------------

    def _precompute_receivers(self) -> None:
        """Compute interpolation indices/weights for all receivers."""
        spec = self.receiver_spec
        pos = spec.positions  # (n_recv, 2) with (x, z)

        rx_frac = (pos[:, 0] - self.model.x[0]) / self.model.dx
        rz_frac = (pos[:, 1] - self.model.z[0]) / self.model.dz

        gix = np.clip(np.floor(rx_frac).astype(int), 0, self.global_nx - 2)
        giz = np.clip(np.floor(rz_frac).astype(int), 0, self.global_nz - 2)
        wx = np.clip(rx_frac - gix, 0.0, 1.0)
        wz = np.clip(rz_frac - giz, 0.0, 1.0)

        # Which receivers are in this rank's slab?
        in_local = (giz >= self.slab_start) & (giz + 1 < self.slab_end)

        self._recv_global_idx = np.where(in_local)[0]
        local_iz = giz[in_local] - self.slab_start + self._ghost_lo
        self._recv_iz = local_iz
        self._recv_ix = gix[in_local]
        self._recv_wx = wx[in_local]
        self._recv_wz = wz[in_local]

        # Field component for each local receiver
        self._recv_fields = [
            spec.field_components[i] for i in self._recv_global_idx
        ]

    def _sample_receivers(self, n: int) -> None:
        """Sample field values at receiver positions."""
        xp = self.xp
        n_local = len(self._recv_global_idx)
        local_vals = np.zeros(n_local)

        if n_local > 0:
            # Transfer fields to host if needed
            if self.is_cuda:
                sxx_h = xp.asnumpy(self.sxx)
                szz_h = xp.asnumpy(self.szz)
                vz_h = xp.asnumpy(self.vz)
            else:
                sxx_h = self.sxx
                szz_h = self.szz
                vz_h = self.vz

            iz = self._recv_iz
            ix = self._recv_ix
            wx = self._recv_wx
            wz = self._recv_wz

            for li in range(n_local):
                field_type = self._recv_fields[li]
                _iz = iz[li]
                _ix = ix[li]
                _wx = wx[li]
                _wz = wz[li]

                if field_type == "pressure":
                    # pressure = -(sxx + szz) / 2
                    p00 = -(sxx_h[_iz, _ix] + szz_h[_iz, _ix]) / 2.0
                    p01 = -(sxx_h[_iz, _ix + 1] + szz_h[_iz, _ix + 1]) / 2.0
                    p10 = -(sxx_h[_iz + 1, _ix] + szz_h[_iz + 1, _ix]) / 2.0
                    p11 = -(sxx_h[_iz + 1, _ix + 1] + szz_h[_iz + 1, _ix + 1]) / 2.0
                    local_vals[li] = (
                        p00 * (1 - _wz) * (1 - _wx)
                        + p01 * (1 - _wz) * _wx
                        + p10 * _wz * (1 - _wx)
                        + p11 * _wz * _wx
                    )
                elif field_type == "vz":
                    local_vals[li] = (
                        vz_h[_iz, _ix] * (1 - _wz) * (1 - _wx)
                        + vz_h[_iz, _ix + 1] * (1 - _wz) * _wx
                        + vz_h[_iz + 1, _ix] * _wz * (1 - _wx)
                        + vz_h[_iz + 1, _ix + 1] * _wz * _wx
                    )

        if not self.use_mpi:
            if self.traces is not None:
                self.traces[self._recv_global_idx, n] = local_vals
            return

        # MPI gather to rank 0
        local_data = (
            np.column_stack([
                self._recv_global_idx.astype(np.float64),
                local_vals,
            ])
            if n_local > 0
            else np.empty((0, 2))
        )
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
        """Exchange ghost zones for all field arrays."""
        if not self.use_mpi:
            return

        xp = self.xp
        comm = self.comm
        rank = self.rank
        nx = self.global_nx
        M = self._M
        gl = self._ghost_lo

        def _rows_to_host(arr, start, count):
            block = arr[start:start + count, :]
            return xp.asnumpy(block) if self.is_cuda else np.array(block)

        def _rows_from_host(arr, start, buf):
            if self.is_cuda:
                arr[start:start + buf.shape[0], :] = xp.asarray(buf)
            else:
                arr[start:start + buf.shape[0], :] = buf

        fields = [self.vx, self.vz, self.sxx, self.szz, self.sxz]
        for fi, field in enumerate(fields):
            tag_base = fi * 2
            TAG_DOWN = tag_base
            TAG_UP = tag_base + 1

            if rank > 0:
                send = _rows_to_host(field, gl, M)
                recv = np.empty((M, nx))
                comm.Sendrecv(
                    sendbuf=send, dest=rank - 1, sendtag=TAG_UP,
                    recvbuf=recv, source=rank - 1, recvtag=TAG_DOWN,
                )
                _rows_from_host(field, 0, recv)

            if rank < self.size - 1:
                bot_start = gl + self.local_nz - M
                send = _rows_to_host(field, bot_start, M)
                recv = np.empty((M, nx))
                comm.Sendrecv(
                    sendbuf=send, dest=rank + 1, sendtag=TAG_DOWN,
                    recvbuf=recv, source=rank + 1, recvtag=TAG_UP,
                )
                _rows_from_host(field, self._pad_nz - M, recv)

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def _step(self, n: int) -> None:
        """Advance one timestep."""
        xp = self.xp
        dt = self.dt

        self._halo_exchange()

        # 1. Update velocities from stress gradients
        # vx at (i+1/2, j): buoyancy_vx * (Dx+(sxx) + Dz-(sxz))
        # vz at (i, j+1/2): buoyancy_vz * (Dx-(sxz) + Dz+(szz))
        dvx_dt = self.buoyancy_vx * (
            self._diff_x_forward(self.sxx)
            + self._diff_z_backward(self.sxz)
        )
        dvz_dt = self.buoyancy_vz * (
            self._diff_x_backward(self.sxz)
            + self._diff_z_forward(self.szz)
        )

        self.vx += dt * dvx_dt
        self.vz += dt * dvz_dt

        # Apply sponge damping to velocities
        self.vx *= (1.0 - self._damping)
        self.vz *= (1.0 - self._damping)

        # 2. Compute velocity gradients for stress update
        # sxx at (i, j): Dx-(vx), Dz-(vz)
        # sxz at (i+1/2, j+1/2): Dz+(vx), Dx+(vz)
        dvx_dx = self._diff_x_backward(self.vx)
        dvz_dz = self._diff_z_backward(self.vz)
        dvx_dz = self._diff_z_forward(self.vx)
        dvz_dx = self._diff_x_forward(self.vz)

        # 3. Update stresses
        dsxx = self.lambda_2mu * dvx_dx + self.lambda_ * dvz_dz
        dszz = self.lambda_ * dvx_dx + self.lambda_2mu * dvz_dz
        dsxz = self.mu_xz * (dvx_dz + dvz_dx)

        # SLS attenuation memory variable update
        if self._has_attenuation and self.Rxx is not None:
            decay = self._sls_decay

            # Update memory variables
            self.Rxx = decay * self.Rxx + self._sls_coeff_p * dsxx
            self.Rzz = decay * self.Rzz + self._sls_coeff_p * dszz
            self.Rxz = decay * self.Rxz + self._sls_coeff_s * dsxz

            # Add memory variable contribution to stress update
            self.sxx += dt * dsxx - self.Rxx
            self.szz += dt * dszz - self.Rzz
            self.sxz += dt * dsxz - self.Rxz
        else:
            self.sxx += dt * dsxx
            self.szz += dt * dszz
            self.sxz += dt * dsxz

        # Apply sponge damping to stresses
        self.sxx *= (1.0 - self._damping)
        self.szz *= (1.0 - self._damping)
        self.sxz *= (1.0 - self._damping)

        # 4. Zero shear stress in air cells (where Vs=0)
        self.sxz *= self.mu_mask

        # 5. Inject source
        self._inject(n)

        # 6. Sample receivers
        self._sample_receivers(n)

    # ------------------------------------------------------------------
    # Gather full field (for snapshots)
    # ------------------------------------------------------------------

    def _gather_pressure_field(self) -> np.ndarray | None:
        """Gather the full pressure field to rank 0."""
        xp = self.xp
        gl = self._ghost_lo
        pressure = -(self.sxx + self.szz) / 2.0
        owned = pressure[gl: gl + self.local_nz, :]
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
        """Run the full 2-D elastic simulation.

        Returns (on rank 0) a dict with ``traces``, ``dt``, ``n_steps``,
        ``receiver_spec``.
        """
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
                field = self._gather_pressure_field()
                if is_root and field is not None:
                    self._save_snapshot(field, n, snapshot_dir)

            if verbose and is_root and n % 500 == 0:
                print(f"  step {n:>6d} / {self.n_steps}")

        if verbose and is_root:
            print(f"  step {self.n_steps:>6d} / {self.n_steps}  (done)")

        return {
            "traces": self.traces if is_root else np.empty((0, 0)),
            "dt": self.dt,
            "n_steps": self.n_steps,
            "receiver_spec": self.receiver_spec,
        }

    def _save_snapshot(
        self, field: np.ndarray, step: int, output_dir: str,
    ) -> None:
        """Save a wavefield snapshot as an image."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        ext = self.model.extent
        vmax = max(np.max(np.abs(field)) * 0.5, 1e-20)
        ax.imshow(
            field, origin="lower",
            extent=[ext[0], ext[1], ext[2], ext[3]],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            aspect="auto", interpolation="bilinear",
        )
        ax.axhline(0.0, color="green", linestyle="--", linewidth=0.8,
                    label="Air-ground interface")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"Pressure field — step {step}")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/snapshot_{step:06d}.png", dpi=120)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 3-D Elastic FDTD Solver
# ═══════════════════════════════════════════════════════════════════════

class ElasticFDTD3DSolver:
    """3-D elastic velocity-stress FDTD solver.

    Extends the 2-D solver to the full 3-D system with 3 velocity
    components (vx, vy, vz) and 6 stress components
    (σxx, σyy, σzz, σxy, σxz, σyz).

    Parameters
    ----------
    model : ElasticModel3D
        Material properties on the global 3-D grid.
    config : ElasticFDTDConfig
        Simulation parameters.
    source : any 3-D source with ``position_at(step, dt)`` and ``signal``
    receiver_spec : ReceiverSpec
        Combined microphone + geophone specification.
    """

    def __init__(
        self,
        model: ElasticModel3D,
        config: ElasticFDTDConfig,
        source: Any,
        receiver_spec: ReceiverSpec,
        **_kw: Any,
    ) -> None:
        self.model = model
        self.cfg = config
        self.source = source
        self.receiver_spec = receiver_spec

        # -- MPI setup --
        self.comm, self.rank, self.size = _get_comm()
        self.use_mpi = self.size > 1

        # -- Backend --
        self.xp, self.is_cuda = get_backend(config.use_cuda)
        xp = self.xp

        nz, ny, nx = model.nz, model.ny, model.nx
        dx = model.dx

        # -- FD stencil --
        self._fd_coeffs = fd1_staggered_coefficients(config.fd_order)
        self._M = config.fd_order // 2
        S = elastic_cfl_factor(self._fd_coeffs)

        # -- CFL: 3-D → sqrt(3) --
        vp_max = model.vp_max
        cfl_limit = dx / (vp_max * S * math.sqrt(3.0))

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

        # -- Local material arrays --
        g_lo = self.slab_start - self._ghost_lo
        g_hi = self.slab_end + self._ghost_hi

        lam_local = model.lambda_[g_lo:g_hi, :, :]
        mu_local = model.mu[g_lo:g_hi, :, :]
        rho_local = model.rho[g_lo:g_hi, :, :]
        vs_local = model.vs[g_lo:g_hi, :, :]

        # Normal stress moduli at cell centres
        self.lambda_ = xp.asarray(lam_local)
        self.mu = xp.asarray(mu_local)
        self.lambda_2mu = self.lambda_ + 2.0 * self.mu

        # Averaged buoyancy for velocity update (3-D)
        pad_nz = self._pad_nz
        buoy_x = np.zeros((pad_nz, ny, nx), dtype=np.float64)
        buoy_y = np.zeros((pad_nz, ny, nx), dtype=np.float64)
        buoy_z = np.zeros((pad_nz, ny, nx), dtype=np.float64)
        buoy_x[:, :, :-1] = 2.0 / (rho_local[:, :, :-1] + rho_local[:, :, 1:])
        buoy_x[:, :, -1] = 1.0 / rho_local[:, :, -1]
        buoy_y[:, :-1, :] = 2.0 / (rho_local[:, :-1, :] + rho_local[:, 1:, :])
        buoy_y[:, -1, :] = 1.0 / rho_local[:, -1, :]
        buoy_z[:-1, :, :] = 2.0 / (rho_local[:-1, :, :] + rho_local[1:, :, :])
        buoy_z[-1, :, :] = 1.0 / rho_local[-1, :, :]
        self.buoyancy_vx = xp.asarray(buoy_x)
        self.buoyancy_vy = xp.asarray(buoy_y)
        self.buoyancy_vz = xp.asarray(buoy_z)

        # Averaged shear moduli at half-grid positions for off-diagonal stresses.
        # Use harmonic mean (zero if any contributor is zero → fluid).
        def _harmonic_avg_2(a, b):
            """Harmonic average of two arrays, zero where either is zero."""
            out = np.zeros_like(a)
            nz_mask = (a > 0) & (b > 0)
            out[nz_mask] = 2.0 / (1.0 / a[nz_mask] + 1.0 / b[nz_mask])
            return out

        # mu for sxy at (i+1/2, j+1/2, k): avg over x and y
        mu_xy = np.zeros((pad_nz, ny, nx), dtype=np.float64)
        mu_xy[:, :-1, :-1] = _harmonic_avg_2(
            _harmonic_avg_2(mu_local[:, :-1, :-1], mu_local[:, :-1, 1:]),
            _harmonic_avg_2(mu_local[:, 1:, :-1], mu_local[:, 1:, 1:]),
        )
        self.mu_xy = xp.asarray(mu_xy)

        # mu for sxz at (i+1/2, j, k+1/2): avg over x and z
        mu_xz = np.zeros((pad_nz, ny, nx), dtype=np.float64)
        mu_xz[:-1, :, :-1] = _harmonic_avg_2(
            _harmonic_avg_2(mu_local[:-1, :, :-1], mu_local[:-1, :, 1:]),
            _harmonic_avg_2(mu_local[1:, :, :-1], mu_local[1:, :, 1:]),
        )
        self.mu_xz = xp.asarray(mu_xz)

        # mu for syz at (i, j+1/2, k+1/2): avg over y and z
        mu_yz = np.zeros((pad_nz, ny, nx), dtype=np.float64)
        mu_yz[:-1, :-1, :] = _harmonic_avg_2(
            _harmonic_avg_2(mu_local[:-1, :-1, :], mu_local[:-1, 1:, :]),
            _harmonic_avg_2(mu_local[1:, :-1, :], mu_local[1:, 1:, :]),
        )
        self.mu_yz = xp.asarray(mu_yz)

        self.mu_mask = xp.asarray((vs_local > 0).astype(np.float64))

        # -- Field arrays --
        shape = (self._pad_nz, ny, nx)
        self.vx = xp.zeros(shape, dtype=np.float64)
        self.vy = xp.zeros(shape, dtype=np.float64)
        self.vz = xp.zeros(shape, dtype=np.float64)
        self.sxx = xp.zeros(shape, dtype=np.float64)
        self.syy = xp.zeros(shape, dtype=np.float64)
        self.szz = xp.zeros(shape, dtype=np.float64)
        self.sxy = xp.zeros(shape, dtype=np.float64)
        self.sxz = xp.zeros(shape, dtype=np.float64)
        self.syz = xp.zeros(shape, dtype=np.float64)

        # -- Sponge layer --
        self._damping = xp.asarray(
            self._build_damping_3d(nz, ny, nx, g_lo, g_hi)
        )

        # -- SLS attenuation --
        self._setup_attenuation_3d(model, g_lo, g_hi)

        # -- Receivers --
        self._precompute_receivers_3d()

        # -- Traces --
        n_recv = receiver_spec.n_receivers
        if self.rank == 0:
            self.traces = np.zeros((n_recv, self.n_steps))
        else:
            self.traces = None

    # ------------------------------------------------------------------
    # Sponge layer (3-D)
    # ------------------------------------------------------------------

    def _build_damping_3d(
        self, nz: int, ny: int, nx: int, g_lo: int, g_hi: int,
    ) -> np.ndarray:
        w = self.cfg.damping_width
        dmax = self.cfg.damping_max
        if w <= 0:
            return np.zeros((g_hi - g_lo, ny, nx), dtype=np.float64)

        dx_edge = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))
        dy_edge = np.minimum(np.arange(ny), ny - 1 - np.arange(ny))
        dz_edge = np.minimum(np.arange(nz), nz - 1 - np.arange(nz))
        dist = np.minimum(
            np.minimum(
                dy_edge[:, np.newaxis],
                dx_edge[np.newaxis, :],
            )[np.newaxis, :, :],
            dz_edge[:, np.newaxis, np.newaxis],
        )

        sigma = np.zeros((nz, ny, nx), dtype=np.float64)
        mask = dist < w
        sigma[mask] = dmax * ((w - dist[mask]) / w) ** 2
        return sigma[g_lo:g_hi, :, :]

    # ------------------------------------------------------------------
    # SLS attenuation (3-D)
    # ------------------------------------------------------------------

    def _setup_attenuation_3d(
        self, model: ElasticModel3D, g_lo: int, g_hi: int,
    ) -> None:
        xp = self.xp
        self._has_attenuation = self.cfg.enable_attenuation

        if not self._has_attenuation:
            self.Rxx = self.Ryy = self.Rzz = None
            self.Rxy = self.Rxz = self.Ryz = None
            return

        f0 = self.cfg.attenuation_f0
        dt = self.dt
        tau_sigma = 1.0 / (2.0 * np.pi * f0)

        qp_local = model.qp[g_lo:g_hi, :, :]
        qs_local = model.qs[g_lo:g_hi, :, :]

        tau_eps_p = tau_sigma * (2.0 * qp_local + 1.0) / np.maximum(2.0 * qp_local - 1.0, 0.01)
        tau_eps_s = tau_sigma * (2.0 * qs_local + 1.0) / np.maximum(2.0 * qs_local - 1.0, 0.01)

        decay = math.exp(-dt / tau_sigma)
        self._sls_decay = decay

        self._sls_coeff_p = xp.asarray((tau_eps_p / tau_sigma - 1.0) * (1.0 - decay))
        self._sls_coeff_s = xp.asarray((tau_eps_s / tau_sigma - 1.0) * (1.0 - decay))

        shape = (self._pad_nz, self.global_ny, self.global_nx)
        self.Rxx = xp.zeros(shape, dtype=np.float64)
        self.Ryy = xp.zeros(shape, dtype=np.float64)
        self.Rzz = xp.zeros(shape, dtype=np.float64)
        self.Rxy = xp.zeros(shape, dtype=np.float64)
        self.Rxz = xp.zeros(shape, dtype=np.float64)
        self.Ryz = xp.zeros(shape, dtype=np.float64)

    # ------------------------------------------------------------------
    # 3-D staggered-grid spatial derivatives
    # ------------------------------------------------------------------

    def _diff3_x_fwd(self, f: Any) -> Any:
        """∂f/∂x at (i+½, j, k)."""
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs
        nz, ny, nx = f.shape
        out = xp.zeros_like(f)
        for k in range(1, M + 1):
            c = coeffs[k - 1]
            out[:, :, M:-M] += c * (
                f[:, :, M + k: nx - M + k]
                - f[:, :, M - k + 1: nx - M - k + 1]
            )
        out /= self.model.dx
        return out

    def _diff3_x_bwd(self, f: Any) -> Any:
        """∂f/∂x at (i, j, k)."""
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs
        nz, ny, nx = f.shape
        out = xp.zeros_like(f)
        for k in range(1, M + 1):
            c = coeffs[k - 1]
            out[:, :, M:-M] += c * (
                f[:, :, M + k - 1: nx - M + k - 1]
                - f[:, :, M - k: nx - M - k]
            )
        out /= self.model.dx
        return out

    def _diff3_y_fwd(self, f: Any) -> Any:
        """∂f/∂y at (i, j+½, k)."""
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs
        nz, ny, nx = f.shape
        out = xp.zeros_like(f)
        for k in range(1, M + 1):
            c = coeffs[k - 1]
            out[:, M:-M, :] += c * (
                f[:, M + k: ny - M + k, :]
                - f[:, M - k + 1: ny - M - k + 1, :]
            )
        out /= self.model.dy
        return out

    def _diff3_y_bwd(self, f: Any) -> Any:
        """∂f/∂y at (i, j, k)."""
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs
        nz, ny, nx = f.shape
        out = xp.zeros_like(f)
        for k in range(1, M + 1):
            c = coeffs[k - 1]
            out[:, M:-M, :] += c * (
                f[:, M + k - 1: ny - M + k - 1, :]
                - f[:, M - k: ny - M - k, :]
            )
        out /= self.model.dy
        return out

    def _diff3_z_fwd(self, f: Any) -> Any:
        """∂f/∂z at (i, j, k+½)."""
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs
        nz, ny, nx = f.shape
        out = xp.zeros_like(f)
        for k in range(1, M + 1):
            c = coeffs[k - 1]
            out[M:-M, :, :] += c * (
                f[M + k: nz - M + k, :, :]
                - f[M - k + 1: nz - M - k + 1, :, :]
            )
        out /= self.model.dz
        return out

    def _diff3_z_bwd(self, f: Any) -> Any:
        """∂f/∂z at (i, j, k)."""
        xp = self.xp
        M = self._M
        coeffs = self._fd_coeffs
        nz, ny, nx = f.shape
        out = xp.zeros_like(f)
        for k in range(1, M + 1):
            c = coeffs[k - 1]
            out[M:-M, :, :] += c * (
                f[M + k - 1: nz - M + k - 1, :, :]
                - f[M - k: nz - M - k, :, :]
            )
        out /= self.model.dz
        return out

    # ------------------------------------------------------------------
    # Source injection (3-D)
    # ------------------------------------------------------------------

    def _inject_3d(self, n: int) -> None:
        """Inject pressure source into σxx, σyy, σzz."""
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

        # Trilinear injection into 8 corners
        xp = self.xp
        for diz, wz_ in ((0, 1 - wz), (1, wz)):
            for diy, wy_ in ((0, 1 - wy), (1, wy)):
                for dix, wx_ in ((0, 1 - wx), (1, wx)):
                    w = wz_ * wy_ * wx_
                    iz = liz + diz
                    iy = giy + diy
                    ix = gix + dix
                    self.sxx[iz, iy, ix] -= amp * w
                    self.syy[iz, iy, ix] -= amp * w
                    self.szz[iz, iy, ix] -= amp * w

    # ------------------------------------------------------------------
    # Receiver sampling (3-D)
    # ------------------------------------------------------------------

    def _precompute_receivers_3d(self) -> None:
        spec = self.receiver_spec
        pos = spec.positions  # (n_recv, 3) with (x, y, z)

        rx_frac = (pos[:, 0] - self.model.x[0]) / self.model.dx
        ry_frac = (pos[:, 1] - self.model.y[0]) / self.model.dy
        rz_frac = (pos[:, 2] - self.model.z[0]) / self.model.dz

        gix = np.clip(np.floor(rx_frac).astype(int), 0, self.global_nx - 2)
        giy = np.clip(np.floor(ry_frac).astype(int), 0, self.global_ny - 2)
        giz = np.clip(np.floor(rz_frac).astype(int), 0, self.global_nz - 2)
        wx = np.clip(rx_frac - gix, 0.0, 1.0)
        wy = np.clip(ry_frac - giy, 0.0, 1.0)
        wz = np.clip(rz_frac - giz, 0.0, 1.0)

        in_local = (giz >= self.slab_start) & (giz + 1 < self.slab_end)

        self._recv_global_idx = np.where(in_local)[0]
        local_iz = giz[in_local] - self.slab_start + self._ghost_lo
        self._recv_iz = local_iz
        self._recv_iy = giy[in_local]
        self._recv_ix = gix[in_local]
        self._recv_wx = wx[in_local]
        self._recv_wy = wy[in_local]
        self._recv_wz = wz[in_local]
        self._recv_fields = [
            spec.field_components[i] for i in self._recv_global_idx
        ]

    def _sample_receivers_3d(self, n: int) -> None:
        xp = self.xp
        n_local = len(self._recv_global_idx)
        local_vals = np.zeros(n_local)

        if n_local > 0:
            if self.is_cuda:
                sxx_h = xp.asnumpy(self.sxx)
                syy_h = xp.asnumpy(self.syy)
                szz_h = xp.asnumpy(self.szz)
                vz_h = xp.asnumpy(self.vz)
            else:
                sxx_h = self.sxx
                syy_h = self.syy
                szz_h = self.szz
                vz_h = self.vz

            for li in range(n_local):
                ftype = self._recv_fields[li]
                iz = self._recv_iz[li]
                iy = self._recv_iy[li]
                ix = self._recv_ix[li]
                _wx = self._recv_wx[li]
                _wy = self._recv_wy[li]
                _wz = self._recv_wz[li]

                val = 0.0
                for diz, wz_ in ((0, 1 - _wz), (1, _wz)):
                    for diy, wy_ in ((0, 1 - _wy), (1, _wy)):
                        for dix, wx_ in ((0, 1 - _wx), (1, _wx)):
                            w = wz_ * wy_ * wx_
                            jz = iz + diz
                            jy = iy + diy
                            jx = ix + dix
                            if ftype == "pressure":
                                val += w * (
                                    -(sxx_h[jz, jy, jx]
                                      + syy_h[jz, jy, jx]
                                      + szz_h[jz, jy, jx]) / 3.0
                                )
                            elif ftype == "vz":
                                val += w * vz_h[jz, jy, jx]
                local_vals[li] = val

        if not self.use_mpi:
            if self.traces is not None:
                self.traces[self._recv_global_idx, n] = local_vals
            return

        local_data = (
            np.column_stack([
                self._recv_global_idx.astype(np.float64),
                local_vals,
            ])
            if n_local > 0
            else np.empty((0, 2))
        )
        gathered = self.comm.gather(local_data, root=0)
        if self.rank == 0:
            for chunk in gathered:
                if chunk.size == 0:
                    continue
                idxs = chunk[:, 0].astype(int)
                vals = chunk[:, 1]
                self.traces[idxs, n] = vals

    # ------------------------------------------------------------------
    # Halo exchange (3-D)
    # ------------------------------------------------------------------

    def _halo_exchange_3d(self) -> None:
        if not self.use_mpi:
            return

        xp = self.xp
        comm = self.comm
        rank = self.rank
        ny, nx = self.global_ny, self.global_nx
        M = self._M
        gl = self._ghost_lo

        def _slabs_to_host(arr, start, count):
            block = arr[start:start + count, :, :]
            return xp.asnumpy(block) if self.is_cuda else np.array(block)

        def _slabs_from_host(arr, start, buf):
            if self.is_cuda:
                arr[start:start + buf.shape[0], :, :] = xp.asarray(buf)
            else:
                arr[start:start + buf.shape[0], :, :] = buf

        fields = [
            self.vx, self.vy, self.vz,
            self.sxx, self.syy, self.szz,
            self.sxy, self.sxz, self.syz,
        ]
        for fi, field in enumerate(fields):
            tag_base = fi * 2
            TAG_DOWN = tag_base
            TAG_UP = tag_base + 1

            if rank > 0:
                send = _slabs_to_host(field, gl, M)
                recv_buf = np.empty((M, ny, nx))
                comm.Sendrecv(
                    sendbuf=send, dest=rank - 1, sendtag=TAG_UP,
                    recvbuf=recv_buf, source=rank - 1, recvtag=TAG_DOWN,
                )
                _slabs_from_host(field, 0, recv_buf)

            if rank < self.size - 1:
                bot_start = gl + self.local_nz - M
                send = _slabs_to_host(field, bot_start, M)
                recv_buf = np.empty((M, ny, nx))
                comm.Sendrecv(
                    sendbuf=send, dest=rank + 1, sendtag=TAG_DOWN,
                    recvbuf=recv_buf, source=rank + 1, recvtag=TAG_UP,
                )
                _slabs_from_host(field, self._pad_nz - M, recv_buf)

    # ------------------------------------------------------------------
    # Time stepping (3-D)
    # ------------------------------------------------------------------

    def _step_3d(self, n: int) -> None:
        xp = self.xp
        dt = self.dt

        self._halo_exchange_3d()

        # 1. Update velocities (Virieux staggered grid)
        # vx at (i+1/2, j, k): buoyancy_vx * (Dx+(sxx) + Dy-(sxy) + Dz-(sxz))
        dvx_dt = self.buoyancy_vx * (
            self._diff3_x_fwd(self.sxx)
            + self._diff3_y_bwd(self.sxy)
            + self._diff3_z_bwd(self.sxz)
        )
        # vy at (i, j+1/2, k): buoyancy_vy * (Dx-(sxy) + Dy+(syy) + Dz-(syz))
        dvy_dt = self.buoyancy_vy * (
            self._diff3_x_bwd(self.sxy)
            + self._diff3_y_fwd(self.syy)
            + self._diff3_z_bwd(self.syz)
        )
        # vz at (i, j, k+1/2): buoyancy_vz * (Dx-(sxz) + Dy-(syz) + Dz+(szz))
        dvz_dt = self.buoyancy_vz * (
            self._diff3_x_bwd(self.sxz)
            + self._diff3_y_bwd(self.syz)
            + self._diff3_z_fwd(self.szz)
        )

        self.vx += dt * dvx_dt
        self.vy += dt * dvy_dt
        self.vz += dt * dvz_dt

        # Sponge damping on velocities
        self.vx *= (1.0 - self._damping)
        self.vy *= (1.0 - self._damping)
        self.vz *= (1.0 - self._damping)

        # 2. Velocity gradients for stress update (Virieux staggering)
        # Normal stresses at (i, j, k): Dx-(vx), Dy-(vy), Dz-(vz)
        dvx_dx = self._diff3_x_bwd(self.vx)
        dvy_dy = self._diff3_y_bwd(self.vy)
        dvz_dz = self._diff3_z_bwd(self.vz)

        # sxy at (i+1/2, j+1/2, k): Dy+(vx), Dx+(vy)
        dvx_dy = self._diff3_y_fwd(self.vx)
        dvy_dx = self._diff3_x_fwd(self.vy)

        # sxz at (i+1/2, j, k+1/2): Dz+(vx), Dx+(vz)
        dvx_dz = self._diff3_z_fwd(self.vx)
        dvz_dx = self._diff3_x_fwd(self.vz)

        # syz at (i, j+1/2, k+1/2): Dz+(vy), Dy+(vz)
        dvy_dz = self._diff3_z_fwd(self.vy)
        dvz_dy = self._diff3_y_fwd(self.vz)

        # 3. Stress updates
        dsxx = self.lambda_2mu * dvx_dx + self.lambda_ * (dvy_dy + dvz_dz)
        dsyy = self.lambda_ * dvx_dx + self.lambda_2mu * dvy_dy + self.lambda_ * dvz_dz
        dszz = self.lambda_ * (dvx_dx + dvy_dy) + self.lambda_2mu * dvz_dz
        dsxy = self.mu_xy * (dvx_dy + dvy_dx)
        dsxz = self.mu_xz * (dvx_dz + dvz_dx)
        dsyz = self.mu_yz * (dvy_dz + dvz_dy)

        # SLS attenuation
        if self._has_attenuation and self.Rxx is not None:
            decay = self._sls_decay
            self.Rxx = decay * self.Rxx + self._sls_coeff_p * dsxx
            self.Ryy = decay * self.Ryy + self._sls_coeff_p * dsyy
            self.Rzz = decay * self.Rzz + self._sls_coeff_p * dszz
            self.Rxy = decay * self.Rxy + self._sls_coeff_s * dsxy
            self.Rxz = decay * self.Rxz + self._sls_coeff_s * dsxz
            self.Ryz = decay * self.Ryz + self._sls_coeff_s * dsyz

            self.sxx += dt * dsxx - self.Rxx
            self.syy += dt * dsyy - self.Ryy
            self.szz += dt * dszz - self.Rzz
            self.sxy += dt * dsxy - self.Rxy
            self.sxz += dt * dsxz - self.Rxz
            self.syz += dt * dsyz - self.Ryz
        else:
            self.sxx += dt * dsxx
            self.syy += dt * dsyy
            self.szz += dt * dszz
            self.sxy += dt * dsxy
            self.sxz += dt * dsxz
            self.syz += dt * dsyz

        # Sponge damping on stresses
        self.sxx *= (1.0 - self._damping)
        self.syy *= (1.0 - self._damping)
        self.szz *= (1.0 - self._damping)
        self.sxy *= (1.0 - self._damping)
        self.sxz *= (1.0 - self._damping)
        self.syz *= (1.0 - self._damping)

        # 4. Zero shear stress in air
        self.sxy *= self.mu_mask
        self.sxz *= self.mu_mask
        self.syz *= self.mu_mask

        # 5. Inject source
        self._inject_3d(n)

        # 6. Sample receivers
        self._sample_receivers_3d(n)

    # ------------------------------------------------------------------
    # Gather pressure field (3-D)
    # ------------------------------------------------------------------

    def _gather_pressure_field_3d(self) -> np.ndarray | None:
        xp = self.xp
        gl = self._ghost_lo
        pressure = -(self.sxx + self.syy + self.szz) / 3.0
        owned = pressure[gl: gl + self.local_nz, :, :]
        owned_host = xp.asnumpy(owned) if self.is_cuda else np.array(owned)

        if not self.use_mpi:
            return owned_host

        gathered = self.comm.gather(owned_host, root=0)
        if self.rank == 0:
            return np.concatenate(gathered, axis=0)
        return None

    # ------------------------------------------------------------------
    # Run (3-D)
    # ------------------------------------------------------------------

    def run(
        self,
        snapshot_dir: str | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run the full 3-D elastic simulation."""
        is_root = self.rank == 0

        if snapshot_dir is not None and is_root:
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)

        for n in range(self.n_steps):
            self._step_3d(n)

            if (
                snapshot_dir is not None
                and self.cfg.snapshot_interval > 0
                and n % self.cfg.snapshot_interval == 0
            ):
                field = self._gather_pressure_field_3d()
                if is_root and field is not None:
                    self._save_snapshot_3d(field, n, snapshot_dir)

            if verbose and is_root and n % 500 == 0:
                print(f"  step {n:>6d} / {self.n_steps}")

        if verbose and is_root:
            print(f"  step {self.n_steps:>6d} / {self.n_steps}  (done)")

        return {
            "traces": self.traces if is_root else np.empty((0, 0)),
            "dt": self.dt,
            "n_steps": self.n_steps,
            "receiver_spec": self.receiver_spec,
        }

    def _save_snapshot_3d(
        self, field: np.ndarray, step: int, output_dir: str,
    ) -> None:
        """Save x-z cross-section snapshot at y=ny//2."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        y_mid = field.shape[1] // 2
        slc = field[:, y_mid, :]

        fig, ax = plt.subplots(figsize=(12, 5))
        ext = self.model.extent_xz
        vmax = max(np.max(np.abs(slc)) * 0.5, 1e-20)
        ax.imshow(
            slc, origin="lower",
            extent=[ext[0], ext[1], ext[2], ext[3]],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            aspect="auto", interpolation="bilinear",
        )
        ax.axhline(0.0, color="green", linestyle="--", linewidth=0.8)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"Pressure (x-z, y=mid) — step {step}")
        fig.tight_layout()
        fig.savefig(f"{output_dir}/snapshot_{step:06d}.png", dpi=120)
        plt.close(fig)

"""Dispersion-corrected wavenumbers for the harbor FDTD scheme.

The solver (central finite differences of order ``order`` in space, leapfrog in time) propagates a plane
wave of frequency ``f`` and direction ``theta`` (from the grid x axis) with a wavenumber that is not
``2 pi f / c``.  Its dispersion relation is

    (2 sin(w dt / 2) / dt)^2 = c^2 [ L(k cos(theta)) + L(k sin(theta)) ],
    L(kappa) = -(c_0 + 2 sum_m c_m cos(m kappa dx)) / dx^2,

with ``c_m`` the stencil weights of ``acoustic_sim.solver.fdtd.fd2_coefficients``.  The wavenumber depends on
the direction, so the error is anisotropic.  A matched-field replica built with the numerical wavenumber
instead of ``w / c`` follows the simulated field's phase to a small fraction of a radian over hundreds of
metres, which a replica with a single sound speed cannot (grid dispersion grows as range x f^4).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from acoustic_sim.solver.fdtd import fd2_coefficients

N_THETA = 46          # table points over theta in [0, pi/2] (the scheme is symmetric under the grid's symmetries)


@dataclass(frozen=True)
class DispersionModel:
    """Parameters of the FDTD scheme whose dispersion the replica should follow."""

    dx: float
    dt: float
    c: float
    order: int = 4

    def wavenumber(self, freq, theta, iters: int = 40) -> np.ndarray:
        """Numerical wavenumber ``|k|`` [rad/m], shape ``(len(freq), len(theta))``.

        Solved by Newton iteration from the continuous value ``2 pi f / c``.
        """
        coef = fd2_coefficients(self.order)
        m = np.arange(1, len(coef))
        f = np.atleast_1d(np.asarray(freq, dtype=float))[:, None]
        th = np.atleast_1d(np.asarray(theta, dtype=float))[None, :]
        w2 = (2.0 / self.dt * np.sin(np.pi * f * self.dt)) ** 2
        cx, sx = np.cos(th), np.sin(th)

        def lam(kappa):                      # L(kappa) >= 0
            return -(coef[0] + 2.0 * np.sum(coef[1:, None, None] * np.cos(m[:, None, None] * kappa * self.dx), axis=0)) / self.dx ** 2

        def dlam(kappa):                     # dL/dkappa
            return 2.0 * np.sum(m[:, None, None] * coef[1:, None, None] * np.sin(m[:, None, None] * kappa * self.dx), axis=0) / self.dx

        k = np.broadcast_to(2.0 * np.pi * f / self.c, (f.shape[0], th.shape[1])).copy()
        for _ in range(iters):
            g = self.c ** 2 * (lam(k * cx) + lam(k * sx)) - w2
            dg = self.c ** 2 * (cx * dlam(k * cx) + sx * dlam(k * sx))
            k = k - g / dg
        return k

    def table(self, freq) -> np.ndarray:
        """``(len(freq), N_THETA)`` wavenumbers on a regular grid of ``theta`` in ``[0, pi/2]``."""
        return self.wavenumber(freq, np.linspace(0.0, np.pi / 2.0, N_THETA))


@lru_cache(maxsize=64)
def _cached_table(model: DispersionModel, freqs: tuple[float, ...]) -> np.ndarray:
    return model.table(np.asarray(freqs))


def wavenumber_pairs(table, theta, xp=np):
    """Wavenumber for every (point, receiver) pair.

    *table* is ``(B, N_THETA)`` (device array), *theta* is ``(P, N)`` with
    ``theta = atan2(|dy|, |dx|)`` of the source-to-receiver offset.
    Returns ``(B, P, N)``, linearly interpolated in ``theta``.
    """
    pos = theta * ((N_THETA - 1) / (np.pi / 2.0))
    i0 = xp.clip(xp.floor(pos).astype(int), 0, N_THETA - 2)
    w = (pos - i0).astype(table.dtype)
    return table[:, i0] * (1.0 - w) + table[:, i0 + 1] * w


def replica_wavenumbers(model: DispersionModel | None, freqs, c: float, theta, xp=np, dtype=np.float64):
    """``(B, P, N)`` wavenumbers for the replica of bins *freqs* (host floats).

    With *model* the numerical wavenumber of the FDTD scheme, otherwise the isotropic ``2 pi f / c``.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if model is None:
        k = xp.asarray(2.0 * np.pi * freqs / c, dtype=dtype)
        return xp.broadcast_to(k[:, None, None], (len(freqs),) + tuple(theta.shape))
    table = xp.asarray(_cached_table(model, tuple(float(f) for f in freqs)), dtype=dtype)
    return wavenumber_pairs(table, theta, xp)


def for_scenario(env, fdtd_cfg) -> DispersionModel:
    """Dispersion model of the scheme that produced the data: grid spacing, time step, order, and the
    median water sound speed of *env*."""
    c = float(np.median(env.model.values[env.water]))
    return DispersionModel(dx=float(env.dx), dt=float(fdtd_cfg.dt), c=c, order=int(fdtd_cfg.fd_order))

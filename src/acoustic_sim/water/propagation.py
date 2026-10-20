"""Replica model of the processing chains built from the public chart alone: the free field 2-D Green's function.

:class:`FreeFieldPropagator` implements the :class:`acoustic_sim.contract.Propagator` protocol from a
nominal sound speed alone, which is all a deployed system knows.  It is the default of
``ProcessingConfig.propagator`` and ``BistaticConfig.propagator``.  The simulator aware alternative
(the FDTD scheme's numerical wavenumber) is ``water/diagnostics.py``'s ``SimulatorAwarePropagator``.

Convention: STFT ``exp(-i w t)``, outgoing ``H0^(2)(kr)``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel2


def green2d(kr, xp=np):
    """Conjugate 2-D Green's function ``H0^(2)(kr)`` (exact for small kr, asymptotic otherwise).

    *xp* is the array module of *kr* (NumPy or CuPy).  The CuPy branch evaluates both
    regimes everywhere and selects with ``where``, which is faster than masking on a GPU.
    """
    if xp is np:
        kr = np.asarray(kr)
        out = np.empty(kr.shape, dtype=np.complex128 if kr.dtype == np.float64 else np.complex64)
        near = kr < 8.0
        out[near] = hankel2(0, kr[near])
        far = kr[~near]
        series = 1.0 + 1j / (8.0 * far) - 9.0 / (128.0 * far ** 2)   # Hankel asymptotic series (H^(2))
        out[~near] = np.sqrt(2.0 / (np.pi * far)) * np.exp(-1j * (far - np.pi / 4.0)) * series
        return out
    from cupyx.scipy.special import j0, y0

    near = kr < 8.0
    kn = xp.where(near, kr, 1.0).astype(kr.dtype)
    kf = xp.where(near, 8.0, kr).astype(kr.dtype)
    h_near = j0(kn) - 1j * y0(kn)
    series = 1.0 + 1j / (8.0 * kf) - 9.0 / (128.0 * kf ** 2)
    h_far = xp.sqrt(2.0 / (np.pi * kf)) * xp.exp(-1j * (kf - np.pi / 4.0)) * series
    return xp.where(near, h_near, h_far)


def pair_geometry(sources_xy, receivers_xy, xp=np, dtype=np.float64, min_range: float = 1.0):
    """Ranges ``r`` (P, N), clamped to *min_range*, and angles ``theta = atan2(|dy|, |dx|)`` from the grid axes."""
    s = xp.asarray(np.atleast_2d(np.asarray(sources_xy, dtype=float)), dtype=dtype)
    q = xp.asarray(np.atleast_2d(np.asarray(receivers_xy, dtype=float)), dtype=dtype)
    dx = s[:, None, 0] - q[None, :, 0]
    dy = s[:, None, 1] - q[None, :, 1]
    return xp.maximum(xp.hypot(dx, dy), min_range), xp.arctan2(xp.abs(dy), xp.abs(dx))


class AnalyticPropagator:
    """``greens`` and ``path`` of an analytic propagator from its ``wavenumbers`` hook."""

    analytic = True
    needs_angles = False
    knowledge: tuple[str, ...] = ()
    min_range = 1.0

    def wavenumbers(self, freqs, theta, xp=np, dtype=np.float64):  # pragma: no cover - interface
        raise NotImplementedError

    def greens(self, sources_xy, receivers_xy, freqs, xp=np, dtype=np.float64):
        """``H0^(2)(k r)`` (F, P, N) from every source to every receiver."""
        r, theta = pair_geometry(sources_xy, receivers_xy, xp, dtype, self.min_range)
        return green2d(self.wavenumbers(freqs, theta, xp, dtype) * r[None], xp)

    def path(self, points_xy, source_xy, freqs, xp=np, dtype=np.float64):
        """Propagation factor (F, P) from *source_xy* (2,) to every point."""
        return self.greens(points_xy, np.atleast_2d(source_xy), freqs, xp, dtype)[:, :, 0]


class FreeFieldPropagator(AnalyticPropagator):
    """Isotropic free field: ``k = 2 pi f / c`` for every direction (no simulator knowledge)."""

    def __init__(self, c: float) -> None:
        self.c = float(c)

    @classmethod
    def from_environment(cls, env) -> "FreeFieldPropagator":
        """The environment's nominal sound speed (``ChartEnvironment.c_nominal``)."""
        return cls(env.c_nominal)

    def wavenumbers(self, freqs, theta, xp=np, dtype=np.float64):
        """``(F, P, N)`` wavenumbers of the bins *freqs* (host floats) for angles *theta* (only its shape is used)."""
        freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        k = xp.asarray(2.0 * np.pi * freqs / self.c, dtype=dtype)
        return xp.broadcast_to(k[:, None, None], (len(freqs),) + tuple(theta.shape))

    def __repr__(self) -> str:
        return f"FreeFieldPropagator(c={self.c})"


def resolve(propagator, c: float):
    """The propagator of a config: *propagator*, or the free field one at the config's *c*.

    A free field propagator must agree with the config's ``c``: the chains compute their wavenumbers
    from the config speed, so a different speed would silently be ignored.
    """
    if propagator is None:
        return FreeFieldPropagator(c)
    if isinstance(propagator, FreeFieldPropagator) and propagator.c != float(c):
        raise ValueError(f"FreeFieldPropagator(c={propagator.c}) disagrees with the config's c={c}; set the config's c")
    return propagator

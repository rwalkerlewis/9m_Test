"""Lightweight causal track fitting for real-time pipelines.

Provides the weighted least-squares (WLS) constant-velocity fitter used
by the streaming pipeline, alongside an EMA bearing smoother.

For full extended Kalman filter tracking, see :mod:`acoustic_sim.tracker`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# -----------------------------------------------------------------------
# Track state container
# -----------------------------------------------------------------------

@dataclass
class TrackState:
    """Causal track estimate at a given instant."""

    x0: float
    y0: float
    z0: float
    vx: float
    vy: float
    vz: float
    t_ref: float
    res_x: float
    res_y: float
    res_z: float
    n_det: int

    def position_at(self, t: float) -> np.ndarray:
        """Extrapolate position to time *t*."""
        dt = t - self.t_ref
        return np.array([
            self.x0 + self.vx * dt,
            self.y0 + self.vy * dt,
            self.z0 + self.vz * dt,
        ])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])

    def covariance_6x6(
        self,
        floor: float = 0.5,
        cap: float = 1.0,
    ) -> np.ndarray:
        """Build a diagonal 6×6 covariance for fire-control input."""
        cov = np.zeros((6, 6))
        cov[0, 0] = min(max(self.res_x, floor), cap) ** 2
        cov[1, 1] = min(max(self.res_y, floor), cap) ** 2
        cov[2, 2] = min(max(self.res_z, floor), cap) ** 2
        cov[3, 3] = cov[4, 4] = cov[5, 5] = 1.0
        return cov


# -----------------------------------------------------------------------
# Causal WLS track fitter
# -----------------------------------------------------------------------

class CausalWLSTracker:
    """Weighted least-squares constant-velocity track fitter.

    Accumulates detections and fits a constant-velocity model
    ``pos(t) = pos_0 + vel · (t − t_ref)`` using RMS-weighting so
    high-SNR (close-range) detections dominate the fit.

    Parameters
    ----------
    min_detections : int
        Minimum number of accumulated detections before a track is
        produced.
    max_history : int
        Maximum number of recent detections kept for fitting.  Older
        observations are discarded so that the velocity estimate
        tracks the *current* target state rather than being biased by
        noisy far-range measurements accumulated early on.  ``0`` means
        unlimited (keep all detections).
    """

    def __init__(self, min_detections: int = 5, max_history: int = 20) -> None:
        self.min_detections = min_detections
        self.max_history = max_history
        self._times: list[float] = []
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._zs: list[float] = []
        self._rms: list[float] = []

    def reset(self) -> None:
        """Clear all accumulated detections."""
        self._times.clear()
        self._xs.clear()
        self._ys.clear()
        self._zs.clear()
        self._rms.clear()

    @property
    def n_detections(self) -> int:
        return len(self._times)

    def add_detection(
        self,
        t: float,
        x: float,
        y: float,
        z: float,
        rms: float,
    ) -> None:
        """Record a detection."""
        self._times.append(t)
        self._xs.append(x)
        self._ys.append(y)
        self._zs.append(z)
        self._rms.append(rms)
        # Trim to sliding window
        if self.max_history > 0 and len(self._times) > self.max_history:
            excess = len(self._times) - self.max_history
            del self._times[:excess]
            del self._xs[:excess]
            del self._ys[:excess]
            del self._zs[:excess]
            del self._rms[:excess]

    def fit(self) -> TrackState | None:
        """Fit constant-velocity model to accumulated detections.

        Returns ``None`` if fewer than ``min_detections`` are available.
        """
        n = len(self._times)
        if n < self.min_detections:
            return None

        t_arr = np.asarray(self._times)
        x_arr = np.asarray(self._xs)
        y_arr = np.asarray(self._ys)
        z_arr = np.asarray(self._zs)
        rms_arr = np.asarray(self._rms)

        t_ref = 0.5 * (t_arr[0] + t_arr[-1])
        dt = t_arr - t_ref
        weights = rms_arr / max(float(rms_arr.max()), 1e-12)
        A = np.column_stack([np.ones(n), dt])
        W = np.diag(weights)

        def _wls(y: np.ndarray) -> np.ndarray:
            AtW = A.T @ W
            return np.linalg.lstsq(AtW @ A, AtW @ y, rcond=None)[0]

        cx = _wls(x_arr)
        cy = _wls(y_arr)
        cz = _wls(z_arr)
        sq_n = max(math.sqrt(n), 1.0)

        return TrackState(
            x0=float(cx[0]), y0=float(cy[0]), z0=float(cz[0]),
            vx=float(cx[1]), vy=float(cy[1]), vz=float(cz[1]),
            t_ref=t_ref,
            res_x=float(np.std(x_arr - A @ cx)) / sq_n,
            res_y=float(np.std(y_arr - A @ cy)) / sq_n,
            res_z=float(np.std(z_arr - A @ cz)) / sq_n,
            n_det=n,
        )


# -----------------------------------------------------------------------
# EMA bearing smoother
# -----------------------------------------------------------------------

class EMABearingSmoother:
    """Exponential moving average on the unit circle.

    Avoids wrap-around artefacts by tracking sin/cos components
    separately.

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1].  Higher = more responsive.
    """

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self._sin = 0.0
        self._cos = 0.0
        self._initialised = False

    def reset(self) -> None:
        self._sin = 0.0
        self._cos = 0.0
        self._initialised = False

    def update(self, bearing_rad: float) -> float:
        """Feed a raw bearing and return the smoothed value."""
        s = math.sin(bearing_rad)
        c = math.cos(bearing_rad)
        if not self._initialised:
            self._sin = s
            self._cos = c
            self._initialised = True
        else:
            a = self.alpha
            self._sin = a * s + (1 - a) * self._sin
            self._cos = a * c + (1 - a) * self._cos

        brg = math.atan2(self._sin, self._cos)
        if brg < 0:
            brg += 2 * math.pi
        return brg

"""AIS reports and a position predictor for AIS-equipped vessels (processing side).

Courses use the maths convention: ``cog = atan2(vy, vx)`` in radians.  The synthetic report
generator (``AISConfig``, ``generate_ais``) is on the forward side, in ``water/ais_sim.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AISReports:
    """All AIS messages of a scenario (sorted by time within each vessel)."""

    vessel: np.ndarray   # (N,) str ids
    t: np.ndarray        # (N,) report time [s]
    xy: np.ndarray       # (N, 2) reported position [m]
    sog: np.ndarray      # (N,) reported speed [m/s]
    cog: np.ndarray      # (N,) reported course [rad]

    def __len__(self) -> int:
        return len(self.t)

    def for_vessel(self, vessel_id: str) -> "AISTrack":
        m = self.vessel == vessel_id
        order = np.argsort(self.t[m])
        return AISTrack(vessel_id, self.t[m][order], self.xy[m][order], self.sog[m][order], self.cog[m][order])

    @property
    def vessels(self) -> list[str]:
        return sorted(set(self.vessel.tolist()))


class AISTrack:
    """Reports of one vessel with interpolation / dead-reckoning prediction."""

    def __init__(self, vessel_id: str, t: np.ndarray, xy: np.ndarray, sog: np.ndarray, cog: np.ndarray) -> None:
        self.id = vessel_id
        self.t, self.xy, self.sog, self.cog = t, xy, sog, cog

    def __len__(self) -> int:
        return len(self.t)

    def predict(self, t: np.ndarray | float, causal: bool = False) -> np.ndarray:
        """Predicted position(s) at time(s) *t*, shape ``(len(t), 2)``.

        ``causal=False`` linearly interpolates between reports (post-processing).
        Times after the last report are dead-reckoned from it.  ``causal=True``
        dead-reckons every query from the latest report at or before it (real-time use);
        queries before the first report use the first report.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if len(self) == 0:
            raise ValueError("no AIS reports for this vessel")
        if not causal:
            out = np.column_stack([np.interp(t, self.t, self.xy[:, 0]), np.interp(t, self.t, self.xy[:, 1])])
            late = t > self.t[-1]
            if np.any(late):
                out[late] = self._dead_reckon(len(self) - 1, t[late])
            return out
        idx = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self) - 1)
        out = np.empty((len(t), 2))
        for i in np.unique(idx):
            m = idx == i
            out[m] = self._dead_reckon(i, t[m])
        return out

    def _dead_reckon(self, i: int, t: np.ndarray) -> np.ndarray:
        v = self.sog[i] * np.array([np.cos(self.cog[i]), np.sin(self.cog[i])])
        return self.xy[i] + np.outer(t - self.t[i], v)

    def age(self, t: np.ndarray | float) -> np.ndarray:
        """Seconds since the latest report at or before *t* (inf before the first)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.searchsorted(self.t, t, side="right") - 1
        return np.where(idx >= 0, t - self.t[np.maximum(idx, 0)], np.inf)

"""Sensor layout: DAS cables (dense receiver lines) and discrete hydrophones."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from acoustic_sim.harbor.environment import HarborEnvironment


@dataclass
class Cable:
    """A fiber cable modelled as a dense line of pressure receivers.

    Channels are placed at equal arc-length spacing along the polyline
    ``waypoints``, starting at the first waypoint.
    """

    name: str
    waypoints: np.ndarray  # (N, 2) metres
    channel_spacing: float = 10.0

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 2 or len(self.waypoints) < 2:
            raise ValueError("waypoints must have shape (N>=2, 2)")
        if self.channel_spacing <= 0:
            raise ValueError("channel_spacing must be positive")

    @property
    def length(self) -> float:
        return float(np.sum(np.hypot(*np.diff(self.waypoints, axis=0).T)))

    @property
    def n_channels(self) -> int:
        return int(np.floor(self.length / self.channel_spacing)) + 1

    @property
    def arclength(self) -> np.ndarray:
        """Along-cable distance [m] of each channel."""
        return np.arange(self.n_channels) * self.channel_spacing

    @property
    def positions(self) -> np.ndarray:
        """Channel positions, shape ``(n_channels, 2)``."""
        seg = np.hypot(*np.diff(self.waypoints, axis=0).T)
        s_node = np.concatenate([[0.0], np.cumsum(seg)])
        s = self.arclength
        return np.column_stack([
            np.interp(s, s_node, self.waypoints[:, 0]),
            np.interp(s, s_node, self.waypoints[:, 1]),
        ])

    @property
    def tangents(self) -> np.ndarray:
        """Unit tangent at each channel, shape ``(n_channels, 2)``."""
        seg_vec = np.diff(self.waypoints, axis=0)
        seg_len = np.hypot(*seg_vec.T)
        s_node = np.concatenate([[0.0], np.cumsum(seg_len)])
        idx = np.clip(np.searchsorted(s_node, self.arclength, side="right") - 1, 0, len(seg_vec) - 1)
        return seg_vec[idx] / seg_len[idx, None]


@dataclass
class SensorLayout:
    """All sensors in one scenario.

    Receiver ordering is: every cable's channels in order, then hydrophones.
    """

    cables: list[Cable] = field(default_factory=list)
    hydrophones: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    def __post_init__(self) -> None:
        self.hydrophones = np.asarray(self.hydrophones, dtype=float).reshape(-1, 2)

    @property
    def n_das(self) -> int:
        return sum(c.n_channels for c in self.cables)

    @property
    def n_hydrophones(self) -> int:
        return len(self.hydrophones)

    @property
    def n_receivers(self) -> int:
        return self.n_das + self.n_hydrophones

    @property
    def das_positions(self) -> np.ndarray:
        if not self.cables:
            return np.empty((0, 2))
        return np.vstack([c.positions for c in self.cables])

    @property
    def receivers(self) -> np.ndarray:
        """Stacked receiver positions ``(n_receivers, 2)`` in solver order."""
        return np.vstack([self.das_positions, self.hydrophones])

    @property
    def slices(self) -> dict[str, slice]:
        """Row slices into a gather: one per cable plus ``"hydrophones"``."""
        out: dict[str, slice] = {}
        start = 0
        for c in self.cables:
            out[c.name] = slice(start, start + c.n_channels)
            start += c.n_channels
        out["hydrophones"] = slice(start, start + self.n_hydrophones)
        return out

    def split(self, traces: np.ndarray) -> dict[str, np.ndarray]:
        """Split a ``(n_receivers, n_t)`` gather into per-cable and hydrophone arrays."""
        if traces.shape[0] != self.n_receivers:
            raise ValueError(f"expected {self.n_receivers} rows, got {traces.shape[0]}")
        return {name: traces[sl] for name, sl in self.slices.items()}

    def validate(self, env: HarborEnvironment, margin: float = 10.0) -> None:
        """Raise if any receiver is on land or closer than *margin* m to it."""
        ok = env.is_water(self.receivers, margin=margin)
        if not np.all(ok):
            bad = np.flatnonzero(~ok)
            raise ValueError(
                f"{bad.size} receiver(s) on land or within {margin} m of shore, "
                f"first at index {bad[0]} {self.receivers[bad[0]]}"
            )

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(self.receivers, dtype=np.float64).tobytes())
        return h.hexdigest()

"""Sensor layout: DAS cables (dense receiver lines) and discrete hydrophones (processing side).

The layout, the strain gauge sample points of the instrument model and :meth:`SensorLayout.collapse`
are what the replicas need.  The solver side check of the sample points and the FDTD cache key are in
``water/sensor_sampling.py`` (forward side).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DAS_MODES = ("pressure", "strain")


@dataclass
class Cable:
    """A fiber cable modelled as a dense line of receivers.

    Channels are placed at equal arc-length spacing along the polyline
    ``waypoints``, starting at the first waypoint.

    ``das_mode = "pressure"`` samples the pressure at the channel (the original toy
    model).  ``das_mode = "strain"`` mimics what a DAS interrogator measures, the
    axial strain over a gauge length ``gauge_length``: the field is sampled at
    ``s - g/2``, ``s`` and ``s + g/2`` along the fiber and the second difference
    ``p(s + g/2) - 2 p(s) + p(s - g/2)`` is the channel output.  For a plane wave this
    equals ``-4 sin^2(k g cos(theta) / 4) p``, i.e. ``-(k g cos(theta))^2 p / 4`` to leading
    order: the cos^2 directivity of DAS, and it is right for every arrival (scattered, multipath) without knowing its
    direction.  The output is unnormalised (no ``1/(rho omega^2)``); replicas built
    with :meth:`SensorLayout.collapse` of the same Green's function match it exactly.
    """

    name: str
    waypoints: np.ndarray  # (N, 2) metres
    channel_spacing: float = 10.0
    das_mode: str = "pressure"
    gauge_length: float = 10.0

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 2 or len(self.waypoints) < 2:
            raise ValueError("waypoints must have shape (N>=2, 2)")
        if self.channel_spacing <= 0:
            raise ValueError("channel_spacing must be positive")
        if self.das_mode not in DAS_MODES:
            raise ValueError(f"das_mode must be one of {DAS_MODES}, got {self.das_mode!r}")
        if self.gauge_length <= 0:
            raise ValueError("gauge_length must be positive")

    @property
    def is_strain(self) -> bool:
        return self.das_mode == "strain"

    @property
    def sample_points(self) -> np.ndarray:
        """Points at which the field is sampled: ``(n_channels, 2)`` for pressure,
        ``(3 n_channels, 2)`` for strain (``s - g/2, s, s + g/2`` per channel, in that order)."""
        pos = self.positions
        if not self.is_strain:
            return pos
        half = 0.5 * self.gauge_length * self.tangents
        return np.stack([pos - half, pos, pos + half], axis=1).reshape(-1, 2)

    @property
    def n_samples(self) -> int:
        return self.n_channels * (3 if self.is_strain else 1)

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
    ``receivers`` are the channel positions (what the processing uses);
    ``sample_points`` are where the FDTD samples the field (three per strain
    channel), and :meth:`collapse` turns sampled traces into channel traces.
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
        """Stacked channel positions ``(n_receivers, 2)`` in solver order."""
        return np.vstack([self.das_positions, self.hydrophones])

    @property
    def has_strain(self) -> bool:
        return any(c.is_strain for c in self.cables)

    @property
    def n_samples(self) -> int:
        return sum(c.n_samples for c in self.cables) + self.n_hydrophones

    @property
    def sample_points(self) -> np.ndarray:
        """Field sampling points ``(n_samples, 2)``: every cable's sample points, then hydrophones."""
        parts = [c.sample_points for c in self.cables] + [self.hydrophones]
        return np.vstack([p for p in parts if len(p)])

    def collapse(self, sampled, xp=np):
        """Channel values from sampled values: second difference for strain cables, identity otherwise.

        *sampled* has the sample points on its **last** axis (``(..., n_samples)``); the result
        has the receivers on its last axis (``(..., n_receivers)``).  Works for traces
        ``(n_samples, n_t).T`` and for replica tables alike.
        """
        if not self.has_strain:
            return sampled
        out, start = [], 0
        for c in self.cables:
            block = sampled[..., start:start + c.n_samples]
            start += c.n_samples
            if c.is_strain:
                b = block.reshape(block.shape[:-1] + (c.n_channels, 3))
                out.append(b[..., 2] - 2.0 * b[..., 1] + b[..., 0])
            else:
                out.append(block)
        out.append(sampled[..., start:])
        return xp.concatenate(out, axis=-1)

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

    def positions(self) -> np.ndarray:
        """Channel positions in recording order (the :class:`acoustic_sim.contract.Geometry` protocol)."""
        return self.receivers

    def to_header(self) -> dict:
        """JSON description, with the DAS mode and gauge length (public instrument parameters)."""
        return {"kind": "sensor_layout",
                "cables": [{"name": c.name, "waypoints": c.waypoints.tolist(), "channel_spacing": c.channel_spacing,
                            "das_mode": c.das_mode, "gauge_length": c.gauge_length} for c in self.cables],
                "hydrophones": self.hydrophones.tolist()}

    @classmethod
    def from_header(cls, h: dict, arrays: dict | None = None) -> "SensorLayout":
        return cls(cables=[Cable(c["name"], np.asarray(c["waypoints"], float), c["channel_spacing"],
                                 c.get("das_mode", "pressure"), c.get("gauge_length", 10.0)) for c in h["cables"]],
                   hydrophones=np.asarray(h["hydrophones"], float))

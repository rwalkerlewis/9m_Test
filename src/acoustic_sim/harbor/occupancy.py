"""Occupancy-grid track-before-detect.

A probability map over the scan grid is propagated frame by frame: *predict* by
diffusing it (the unknown target can move at up to ``speed_mps``), then *update* by
multiplying with a likelihood derived from that frame's matched-field map.  A weak
target that is only marginally visible in any single frame accumulates evidence
along its path, while noise peaks, which do not persist at a plausible
speed, do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_sim.harbor.device import get_device


@dataclass
class OccupancyConfig:
    """Parameters of the occupancy filter.

    Attributes
    ----------
    speed_mps : float
        Diffusion speed of the prediction step [m/s].
    gamma, cap : float
        Log-likelihood per frame is ``gamma * min(B, cap)`` for the map statistic ``B``.
    leak : float
        Fraction of the probability redistributed uniformly each frame, so a new
        target can still be picked up after a mistaken lock.
    use_cuda : bool
        Run the filter on the GPU.
    """

    speed_mps: float = 2.0
    gamma: float = 0.1
    cap: float = 60.0
    leak: float = 0.01
    use_cuda: bool = False


@dataclass
class OccupancyHistory:
    times: np.ndarray            # (K,)
    xy: np.ndarray               # (K, 2) argmax position after each update
    contrast: np.ndarray         # (K,) peak / mean occupancy (1 = flat)
    snapshots: dict[int, np.ndarray]   # frame -> (G,) occupancy (normalised to sum 1)


class OccupancyGrid:
    """Occupancy filter on a regular (possibly land-masked) grid of cell centres *grid* (G, 2)."""

    def __init__(self, grid: np.ndarray, cfg: OccupancyConfig | None = None) -> None:
        self.grid = np.asarray(grid, dtype=float)
        self.cfg = cfg or OccupancyConfig()
        self.dev = get_device(self.cfg.use_cuda)
        xp = self.dev.xp
        x, y = self.grid[:, 0], self.grid[:, 1]
        steps = np.diff(np.unique(x))
        self.dx = float(steps.min()) if len(steps) else 1.0
        self.ix = xp.asarray(np.rint((x - x.min()) / self.dx).astype(int))
        self.iy = xp.asarray(np.rint((y - y.min()) / self.dx).astype(int))
        self.shape = (int(self.iy.max()) + 1, int(self.ix.max()) + 1)
        self.mask2d = xp.zeros(self.shape, dtype=self.dev.fdtype)
        self.mask2d[self.iy, self.ix] = 1.0
        self.reset()

    def reset(self) -> None:
        self.p = self.dev.xp.full(len(self.grid), 1.0 / len(self.grid), dtype=self.dev.fdtype)

    def _diffuse(self, p, sigma_cells: float):
        img = self.dev.xp.zeros(self.shape, dtype=self.dev.fdtype)
        img[self.iy, self.ix] = p
        num = self.dev.gaussian_filter(img, sigma_cells, mode="constant")
        den = self.dev.gaussian_filter(self.mask2d, sigma_cells, mode="constant")
        return (num / self.dev.xp.maximum(den, 1e-12))[self.iy, self.ix]

    def update(self, B, dt: float) -> None:
        """One predict + update step with the map statistic *B* (G,) (host or device)."""
        xp, cfg = self.dev.xp, self.cfg
        sigma = max(cfg.speed_mps * dt / self.dx, 0.3)
        p = self._diffuse(self.p, sigma)
        p = (1.0 - cfg.leak) * p / p.sum() + cfg.leak / len(p)
        p = p * xp.exp(cfg.gamma * xp.minimum(xp.asarray(B, dtype=self.dev.fdtype), cfg.cap))
        self.p = p / p.sum()

    def run(self, B: np.ndarray, times: np.ndarray, snapshot_frames: tuple[int, ...] = ()) -> OccupancyHistory:
        """Filter all frames of ``B`` (G, K); returns the track of the occupancy peak."""
        xp = self.dev.xp
        self.reset()
        B_d = xp.asarray(B, dtype=self.dev.fdtype)
        dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        xy = np.empty((len(times), 2))
        contrast = np.empty(len(times))
        snaps: dict[int, np.ndarray] = {}
        for k in range(len(times)):
            self.update(B_d[:, k], dt)
            g = int(xp.argmax(self.p))
            xy[k] = self.grid[g]
            contrast[k] = float(self.p[g]) * len(self.p)
            if k in snapshot_frames:
                snaps[k] = self.dev.asnumpy(self.p).copy()
        return OccupancyHistory(times, xy, contrast, snaps)

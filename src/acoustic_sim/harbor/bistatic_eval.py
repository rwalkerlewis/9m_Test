"""Detection statistics of the bistatic maps measured against body free runs (stage 1c).

The stage 0 metrics (median error, fraction of frames within 5 m) say where the map peak is but
not whether it would have been declared a detection.  Here a threshold is set per configuration
on the maps of the **same traces without the body** (``scatter_gain={"uuv1": 0}``) so that a
fixed fraction of frames has any cell above it (the frame false alarm rate), and a frame counts
as a detection when its peak exceeds the threshold **and** lies within a radius of the truth.
The maps must be unnormalised (``BistaticConfig.map_norm="none"``) for the threshold to mean
the same thing in every run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_sim.harbor.bistatic import clutter_images
from acoustic_sim.harbor.environment import HarborEnvironment
from acoustic_sim.harbor.sensors import SensorLayout

MISS_CLASSES = ("ship_edge", "quay_image", "direct_path", "elsewhere")


def frame_threshold(frame_max: np.ndarray, pfa: float) -> float:
    """Smallest threshold such that at most a fraction *pfa* of the frames has its maximum above it.

    With ``n`` frames the ``floor(pfa n)`` largest maxima may exceed the threshold, which is the
    next largest one (so exactly ``floor(pfa n)`` frames of the body free run are false alarms).
    """
    m = np.sort(np.asarray(frame_max, dtype=float))[::-1]
    if len(m) == 0:
        raise ValueError("no frames")
    k = int(np.floor(pfa * len(m)))
    return float(m[min(k, len(m) - 1)])


def integrate_frames(B: np.ndarray, n: int = 3) -> np.ndarray:
    """Sum of every frame with its ``n - 1`` neighbours (``n = 3``: 1.5 s at a 0.5 s hop... 0.25 s hop
    frames of 0.5 s, the stage 0 "1.5 s integration")."""
    out = np.zeros_like(B)
    h = n // 2
    for k in range(B.shape[1]):
        out[:, k] = B[:, max(0, k - h):k + h + 1].sum(axis=1)
    return out


@dataclass
class DetectionStats:
    """Per frame peaks and the summary of one configuration on one dataset."""

    est: np.ndarray           # (K, 2) map peak per frame
    peak: np.ndarray          # (K,) its value
    err: np.ndarray           # (K,) distance to the truth [m]
    detected: np.ndarray      # (K,) peak above the threshold and within the radius
    threshold: float
    radius: float

    @property
    def pd(self) -> float:
        return float(np.mean(self.detected))

    @property
    def declared(self) -> float:
        """Fraction of frames with any cell above the threshold (the realised frame alarm rate)."""
        return float(np.mean(self.peak > self.threshold))

    def summary(self) -> dict:
        return {
            "pd": self.pd, "declared": self.declared, "threshold": self.threshold, "radius_m": self.radius,
            "within_radius": float(np.mean(self.err <= self.radius)), "median_error_m": float(np.median(self.err)),
            "p90_error_m": float(np.percentile(self.err, 90)), "frames": int(len(self.err)),
        }


def detection_stats(B: np.ndarray, grid: np.ndarray, truth_xy: np.ndarray, threshold: float, radius: float = 5.0) -> DetectionStats:
    """Peak of every frame of *B* (G, K), its error to *truth_xy* (K, 2) and the detections at *threshold*."""
    i = np.argmax(B, axis=0)
    est = grid[i]
    peak = B[i, np.arange(B.shape[1])]
    err = np.hypot(*(est - np.asarray(truth_xy, dtype=float)).T)
    return DetectionStats(est, peak, err, (peak > threshold) & (err <= radius), float(threshold), float(radius))


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of point *p* (2,) to the segments *a* -> *b* ((M, 2) each)."""
    ab = b - a
    t = np.clip(np.sum((p[None, :] - a) * ab, axis=1) / np.maximum(np.sum(ab ** 2, axis=1), 1e-30), 0.0, 1.0)
    q = a + t[:, None] * ab
    return np.hypot(*(p[None, :] - q).T)


def classify_misses(est: np.ndarray, err: np.ndarray, ship_xy: np.ndarray, env: HarborEnvironment, layout: SensorLayout,
                    live: np.ndarray, exclusion_radius: float, err_min: float = 20.0, near: float = 10.0) -> np.ndarray:
    """Class of every frame whose error exceeds *err_min* (``""`` for the others), in this order:

    ``"ship_edge"``: within *near* of the edge of the ship exclusion disc; ``"quay_image"``: within
    *near* of a first order image of the ship in the quays (:func:`clutter_images`);
    ``"direct_path"``: within *near* of a segment from the ship to a live cable channel (the fan
    on which a direct path residual of that channel is matched by the coherent map);
    ``"elsewhere"``.
    """
    est = np.asarray(est, dtype=float)
    ship_xy = np.asarray(ship_xy, dtype=float)
    rec = layout.receivers[: layout.n_das][np.asarray(live)[: layout.n_das]]
    out = np.array([""] * len(est), dtype=object)
    for k in np.flatnonzero(np.asarray(err) > err_min):
        s = ship_xy[k]
        if abs(np.hypot(*(est[k] - s)) - exclusion_radius) <= near:
            out[k] = "ship_edge"
            continue
        images = clutter_images(env, s)
        if len(images) and np.min(np.hypot(*(est[k][None, :] - images).T)) <= near:
            out[k] = "quay_image"
            continue
        if len(rec) and np.min(_segment_distance(est[k], np.repeat(s[None, :], len(rec), axis=0), rec)) <= near:
            out[k] = "direct_path"
            continue
        out[k] = "elsewhere"
    return out


def miss_counts(classes: np.ndarray) -> dict[str, int]:
    return {c: int(np.sum(classes == c)) for c in MISS_CLASSES}

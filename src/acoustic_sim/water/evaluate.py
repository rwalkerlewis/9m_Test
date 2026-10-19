"""Scoring of detections against ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from acoustic_sim.water.detection import DetectionSet
from acoustic_sim.water.sensors import SensorLayout
from acoustic_sim.water.vessels import Track


def associate(dets: DetectionSet, tracks: dict[str, Track], gate: float) -> tuple[list[str | None], np.ndarray]:
    """Label each detection with the nearest *active* vessel within *gate* metres (else ``None``).

    Returns the labels and the distance to the labelled vessel (``nan`` when unlabelled).
    """
    labels: list[str | None] = []
    dist = np.full(len(dets), np.nan)
    for d in range(len(dets)):
        best, best_d = None, np.inf
        for vid, tr in tracks.items():
            if not tr.active_at(dets.t[d])[0]:
                continue
            dd = float(np.hypot(*(tr.position_at(dets.t[d])[0] - dets.xy[d])))
            if dd < best_d:
                best, best_d = vid, dd
        if best is not None and best_d <= gate:
            labels.append(best)
            dist[d] = best_d
        else:
            labels.append(None)
    return labels, dist


@dataclass
class Score:
    """Outcome of scoring one run at one threshold.

    ``rows`` has one entry per (frame, target) with the target active:
    ``id, frame, t, range_m, detected, err_m`` where ``range_m`` is the distance to the
    nearest sensor and ``err_m`` the position error of the closest associated detection.
    """

    threshold: float
    n_frames: int
    rows: list[dict] = field(default_factory=list)
    n_false: int = 0
    n_other: int = 0          # detections associated with non-target vessels (e.g. AIS ships)

    def pd(self, target: str | None = None, rng: tuple[float, float] | None = None) -> float:
        sel = [r for r in self.rows if (target is None or r["id"] == target)
               and (rng is None or rng[0] <= r["range_m"] < rng[1])]
        return float(np.mean([r["detected"] for r in sel])) if sel else float("nan")

    def errors(self, target: str | None = None) -> np.ndarray:
        return np.array([r["err_m"] for r in self.rows
                         if r["detected"] and (target is None or r["id"] == target)])

    def rmse(self, target: str | None = None) -> float:
        e = self.errors(target)
        return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")

    @property
    def false_alarms_per_frame(self) -> float:
        return self.n_false / max(self.n_frames, 1)


def score_detections(
    dets: DetectionSet,
    tracks: dict[str, Track],
    layout: SensorLayout,
    frame_times: np.ndarray,
    threshold: float,
    targets: list[str] | None = None,
    gate: float = 100.0,
    present: list[str] | None = None,
) -> Score:
    """Score detections with ``score >= threshold`` against the truth tracks.

    Vessels with ``level_pa <= 0`` are absent from the mix and are ignored, unless they are
    listed in *present* (silent scattering bodies, see ``HarborDataset.meta["bodies"]``).
    """
    present = set(present or [])
    tracks = {k: t for k, t in tracks.items() if t.level_pa > 0 or k in present}
    targets = targets if targets is not None else [k for k, t in tracks.items() if not t.ais]
    targets = [k for k in targets if k in tracks]
    kept = dets.above(threshold)
    labels, dist = associate(kept, tracks, gate)
    out = Score(threshold=threshold, n_frames=len(frame_times))
    out.n_false = sum(1 for lb in labels if lb is None)
    out.n_other = sum(1 for lb in labels if lb is not None and lb not in targets)
    sensors = layout.receivers
    for vid in targets:
        tr = tracks[vid]
        for k, t in enumerate(frame_times):
            if not tr.active_at(t)[0]:
                continue
            pos = tr.position_at(t)[0]
            hit = [dist[d] for d, lb in enumerate(labels) if lb == vid and kept.frame[d] == k]
            out.rows.append({
                "id": vid, "frame": k, "t": float(t),
                "range_m": float(np.min(np.hypot(*(sensors - pos).T))),
                "detected": bool(hit),
                "err_m": float(min(hit)) if hit else float("nan"),
            })
    return out

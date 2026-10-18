"""Synthetic AIS reports and a position predictor for AIS-equipped vessels.

Courses use the maths convention: ``cog = atan2(vy, vx)`` in radians.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_sim.harbor.vessels import Track


@dataclass
class AISConfig:
    """AIS reporting behaviour.

    Attributes
    ----------
    interval_moving, interval_stationary : float
        Report period [s] while moving / stationary.
    stationary_speed : float
        Below this speed [m/s] a vessel reports at the slow rate.
    pos_sigma : float
        Std of the reported-position error per axis [m].
    sog_sigma, cog_sigma : float
        Std of reported speed [m/s] and course [rad].
    time_jitter : float
        Std of report-time jitter [s].
    dropout_prob : float
        Independent probability a report is lost.
    outage_prob, outage_reports : float, tuple
        Probability per report that a burst outage starts, and its length range in reports.
    """

    interval_moving: float = 5.0
    interval_stationary: float = 20.0
    stationary_speed: float = 0.5
    pos_sigma: float = 8.0
    sog_sigma: float = 0.2
    cog_sigma: float = 0.035
    time_jitter: float = 0.3
    dropout_prob: float = 0.05
    outage_prob: float = 0.01
    outage_reports: tuple[int, int] = (3, 6)

    def __post_init__(self) -> None:
        self.outage_reports = tuple(self.outage_reports)  # JSON round-trips tuples as lists


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


def generate_ais(tracks: dict[str, Track], cfg: AISConfig, rng: np.random.Generator) -> AISReports:
    """Noisy AIS reports for every ``track.ais`` vessel while it is active."""
    vessel, ts, xys, sogs, cogs = [], [], [], [], []
    for track in tracks.values():
        if not track.ais:
            continue
        t_act = track.t[track.active]
        if len(t_act) == 0:
            continue
        t_rep = float(t_act[0]) + rng.uniform(0.0, cfg.interval_moving)
        outage_left = 0
        while t_rep <= t_act[-1]:
            speed = float(track.speed_at(t_rep)[0])
            interval = cfg.interval_moving if speed >= cfg.stationary_speed else cfg.interval_stationary
            if outage_left == 0 and rng.random() < cfg.outage_prob:
                outage_left = int(rng.integers(cfg.outage_reports[0], cfg.outage_reports[1] + 1))
            lost = outage_left > 0 or rng.random() < cfg.dropout_prob
            outage_left = max(outage_left - 1, 0)
            if not lost:
                t_meas = t_rep + cfg.time_jitter * rng.standard_normal()
                t_meas = float(np.clip(t_meas, t_act[0], t_act[-1]))
                pos = track.position_at(t_meas)[0] + cfg.pos_sigma * rng.standard_normal(2)
                # Course from the finite-difference velocity of the true track.
                v = (track.position_at(t_meas + 0.5)[0] - track.position_at(t_meas - 0.5)[0])
                sog = float(np.hypot(*v)) + cfg.sog_sigma * rng.standard_normal()
                cog = float(np.arctan2(v[1], v[0])) + cfg.cog_sigma * rng.standard_normal()
                vessel.append(track.id)
                ts.append(t_meas)
                xys.append(pos)
                sogs.append(max(sog, 0.0))
                cogs.append(cog)
            t_rep += interval
    return AISReports(
        vessel=np.array(vessel, dtype=str),
        t=np.array(ts, dtype=float),
        xy=np.array(xys, dtype=float).reshape(-1, 2),
        sog=np.array(sogs, dtype=float),
        cog=np.array(cogs, dtype=float),
    )

"""Synthetic AIS reports from ground-truth tracks (forward side).

The report container and the predictor the processing uses are in ``water/ais.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_sim.contract import Track
from acoustic_sim.water.ais import AISReports


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

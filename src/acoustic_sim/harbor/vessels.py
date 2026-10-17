"""Vessel configuration and ground-truth trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from acoustic_sim.harbor.environment import HarborEnvironment

KINDS = ("ship", "uuv")


@dataclass
class VesselConfig:
    """One simulated vessel.

    Attributes
    ----------
    id : str
        Unique name, e.g. ``"ferry"``.
    kind : str
        ``"ship"`` (surface vessel, broadcasts AIS) or ``"uuv"`` (hidden, no AIS).
    waypoints : list of [x, y]
        Path followed at constant *speed*; a single waypoint means a stationary vessel.
    speed : float
        Speed along the path [m/s].
    t_start, t_end : float
        Time window in which the vessel moves and radiates; ``t_end=None`` means
        the end of the scenario.  Outside it the vessel is silent.
    level_pa : float
        Free-field rms pressure at 100 m [Pa] at nominal speed.
    ais : bool | None
        Broadcast AIS?  ``None`` means ``kind == "ship"``.
    wander_sigma, wander_tau : float
        Smooth random lateral wander: std [m] and length scale [s] (UUV-like meander).
    seed : int
        Seeds the signature and wander so a vessel is reproducible (and cacheable).
    """

    id: str
    kind: str
    waypoints: list[list[float]]
    speed: float = 0.0
    t_start: float = 0.0
    t_end: float | None = None
    level_pa: float = 1.0
    ais: bool | None = None
    wander_sigma: float = 0.0
    wander_tau: float = 20.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if len(self.waypoints) > 1 and self.speed <= 0:
            raise ValueError(f"vessel {self.id!r} has a path but speed <= 0")

    @property
    def has_ais(self) -> bool:
        return self.kind == "ship" if self.ais is None else self.ais


@dataclass
class Track:
    """Ground-truth trajectory sampled on a regular time grid."""

    id: str
    kind: str
    ais: bool
    t: np.ndarray                  # (M,) seconds
    xy: np.ndarray                 # (M, 2) metres
    speed: np.ndarray              # (M,) m/s
    active: np.ndarray             # (M,) bool, vessel moving and radiating
    level_pa: float = 1.0
    meta: dict = field(default_factory=dict)

    def position_at(self, t: np.ndarray | float) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.column_stack([np.interp(t, self.t, self.xy[:, 0]), np.interp(t, self.t, self.xy[:, 1])])

    def speed_at(self, t: np.ndarray | float) -> np.ndarray:
        return np.interp(np.atleast_1d(np.asarray(t, dtype=float)), self.t, self.speed)

    def active_at(self, t: np.ndarray | float) -> np.ndarray:
        return np.interp(np.atleast_1d(np.asarray(t, dtype=float)), self.t, self.active.astype(float)) > 0.5


def build_track(
    vcfg: VesselConfig,
    total_time: float,
    env: HarborEnvironment | None = None,
    dt: float = 0.5,
    margin: float = 15.0,
    turn_smoothing_s: float = 3.0,
) -> Track:
    """Ground-truth track: constant-speed path with rounded corners and optional wander.

    If *env* is given, every point while the vessel is active must be in water
    at least *margin* m from land.
    """
    t = np.arange(0.0, total_time + 0.5 * dt, dt)
    wp = np.asarray(vcfg.waypoints, dtype=float)
    t_end = total_time if vcfg.t_end is None else vcfg.t_end
    active = (t >= vcfg.t_start) & (t <= t_end)

    if len(wp) == 1:
        xy = np.repeat(wp, len(t), axis=0)
    else:
        # Smooth on a time axis padded on both sides.  If the vessel is already
        # under way at t = 0 the first segment is extended backwards, so smoothing
        # does not drag the start forward.
        sigma = turn_smoothing_s / dt
        pad = int(np.ceil(4.0 * sigma))
        t_full = (np.arange(-pad, len(t) + pad)) * dt
        d0 = (wp[1] - wp[0]) / np.hypot(*(wp[1] - wp[0]))
        back = vcfg.speed * pad * dt if vcfg.t_start <= 0 else 0.0
        wp_ext = np.vstack([wp[0] - d0 * back, wp]) if back > 0 else wp
        seg = np.hypot(*np.diff(wp_ext, axis=0).T)
        s_node = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.clip(vcfg.speed * (t_full - vcfg.t_start) + back, 0.0, s_node[-1])
        xy = np.column_stack([np.interp(s, s_node, wp_ext[:, 0]), np.interp(s, s_node, wp_ext[:, 1])])
        if sigma > 0:
            xy = np.column_stack([gaussian_filter1d(xy[:, i], sigma, mode="nearest") for i in range(2)])
        xy = xy[pad: pad + len(t)]

    if vcfg.wander_sigma > 0 and len(wp) > 1:
        # Smooth random lateral offset: Gaussian-filtered white noise with std wander_sigma.
        rng = np.random.default_rng(vcfg.seed + 7919)
        s = vcfg.wander_tau / dt
        ou = gaussian_filter1d(rng.standard_normal(len(t)), s, mode="wrap") * vcfg.wander_sigma * np.sqrt(2.0 * np.sqrt(np.pi) * s)
        d = np.gradient(xy, dt, axis=0)
        heading = d / np.maximum(np.hypot(*d.T), 1e-9)[:, None]
        normal = np.column_stack([-heading[:, 1], heading[:, 0]])
        moving = (np.hypot(*d.T) > 0.05).astype(float)
        xy = xy + normal * (ou * gaussian_filter1d(moving, 3.0 / dt))[:, None]

    speed = np.hypot(*np.gradient(xy, dt, axis=0).T)
    # Not moving outside the activity window.
    speed = np.where(active, speed, 0.0)

    if env is not None:
        ok = env.is_water(xy[active], margin=margin)
        if not np.all(ok):
            raise ValueError(
                f"vessel {vcfg.id!r} leaves the water (or comes within {margin} m of land) "
                f"first at t = {t[active][np.flatnonzero(~ok)[0]]:.1f} s"
            )

    return Track(
        id=vcfg.id, kind=vcfg.kind, ais=vcfg.has_ais, t=t, xy=xy, speed=speed,
        active=active, level_pa=vcfg.level_pa, meta={"seed": vcfg.seed},
    )

"""Constant velocity Kalman smoother for an illuminator track (stage 1b of the blind bistatic work).

The Doppler ECA of :mod:`acoustic_sim.harbor.bistatic` needs the ship position at every sample.
Stage 0 found that the per frame AIS refinement jitters by about 1 m, which per sample delays
read as velocity noise, and that a local straight line fit over 4 s recovers most of the depth
but was tuned on one dataset.  This module replaces the line fit by a constant velocity Kalman
filter with a Rauch Tung Striebel smoother over the whole record (post processing) fed by

* the AIS reports: position (``pos_sigma`` per axis) and velocity from speed and course
  (``sog_sigma``, ``cog_sigma`` linearised to a velocity covariance);
* the acoustically refined ship position of every frame (``meas_sigma``, default 1 m, the
  stage 0 observed jitter), gated at ``gate_sigma`` standard deviations of the predicted
  innovation so a refinement that jumped to a sidelobe does not drag the track.

The process noise is a continuous white acceleration of density ``accel`` [m/s^2] fixed before
any run: ``Q = accel^2 [[dt^3 / 3, dt^2 / 2], [dt^2 / 2, dt]]`` per axis.  State ``(x, y, vx, vy)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_sim.harbor.ais import AISReports


@dataclass
class TrackConfig:
    """Parameters of :func:`kalman_track`.

    Attributes
    ----------
    accel : float
        Process noise: white acceleration density [m/s^2].  0.05 m/s^2 is a ferry changing
        speed by 0.5 m/s over 10 s; the stage 0a and 0b tracks are straight at constant speed.
    meas_sigma : float
        Standard deviation [m] of an acoustically refined position (stage 0: about 1 m).
    gate_sigma : float
        Refined positions whose innovation exceeds this many standard deviations are skipped.
    ais_pos_sigma, ais_sog_sigma, ais_cog_sigma : float
        AIS measurement errors (:class:`acoustic_sim.harbor.ais.AISConfig`).
    init_pos_sigma, init_vel_sigma : float
        Prior standard deviations of the first state.
    """

    accel: float = 0.05
    meas_sigma: float = 1.0
    gate_sigma: float = 5.0
    ais_pos_sigma: float = 8.0
    ais_sog_sigma: float = 0.2
    ais_cog_sigma: float = 0.035
    init_pos_sigma: float = 20.0
    init_vel_sigma: float = 2.0


@dataclass
class TrackFit:
    """Smoothed track at the requested times."""

    t: np.ndarray          # (K,)
    xy: np.ndarray         # (K, 2)
    v: np.ndarray          # (K, 2)
    sigma_xy: np.ndarray   # (K, 2) posterior standard deviation of the position
    used: np.ndarray       # (M,) bool, which of the acoustic measurements passed the gate

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.v[:, 0], self.v[:, 1])


def _transition(dt: float, accel: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = accel ** 2 * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
    Q = np.zeros((4, 4))
    for i in range(2):
        Q[np.ix_([i, i + 2], [i, i + 2])] = q
    return F, Q


def ais_measurements(ais: AISReports | None, vessel: str, cfg: TrackConfig) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray, str]]:
    """AIS reports as ``(t, z, H, R, kind)`` measurements: a position and a velocity per report.

    The velocity covariance is the speed / course covariance mapped through the Jacobian of
    ``v = sog (cos cog, sin cog)``, with a small floor so a zero speed does not give a singular
    covariance.
    """
    if ais is None:
        return []
    tr = ais.for_vessel(vessel)
    out = []
    Hp = np.zeros((2, 4))
    Hp[0, 0] = Hp[1, 1] = 1.0
    Hv = np.zeros((2, 4))
    Hv[0, 2] = Hv[1, 3] = 1.0
    for t, xy, sog, cog in zip(tr.t, tr.xy, tr.sog, tr.cog):
        out.append((float(t), np.asarray(xy, float), Hp, cfg.ais_pos_sigma ** 2 * np.eye(2), "ais_pos"))
        c, s = np.cos(cog), np.sin(cog)
        J = np.array([[c, -sog * s], [s, sog * c]])
        R = J @ np.diag([cfg.ais_sog_sigma ** 2, cfg.ais_cog_sigma ** 2]) @ J.T + (0.05 ** 2) * np.eye(2)
        out.append((float(t), sog * np.array([c, s]), Hv, R, "ais_vel"))
    return out


def kalman_track(
    times: np.ndarray,
    refined: np.ndarray | None,
    present: np.ndarray | None,
    ais: AISReports | None,
    vessel: str,
    cfg: TrackConfig | None = None,
    out_times: np.ndarray | None = None,
) -> TrackFit:
    """Constant velocity Kalman filter and RTS smoother over the AIS reports and the refined positions.

    *times* (K,) and *refined* (K, 2) are the frame times and the acoustically refined ship
    positions (frames with *present* False carry no measurement; ``refined=None`` uses the AIS
    alone).  The smoothed state is returned at *out_times* (default *times*).
    """
    cfg = cfg or TrackConfig()
    times = np.asarray(times, dtype=float)
    out_times = times if out_times is None else np.asarray(out_times, dtype=float)
    meas = ais_measurements(ais, vessel, cfg)
    Hp = np.zeros((2, 4))
    Hp[0, 0] = Hp[1, 1] = 1.0
    acoustic_idx = []
    if refined is not None:
        present = np.ones(len(times), bool) if present is None else np.asarray(present, bool)
        for k in np.flatnonzero(present):
            acoustic_idx.append(len(meas))
            meas.append((float(times[k]), np.asarray(refined[k], float), Hp, cfg.meas_sigma ** 2 * np.eye(2), "acoustic"))
    if not meas:
        raise ValueError("kalman_track needs AIS reports or refined positions")
    # Time grid: every measurement time and every output time (a step with no measurement is a prediction).
    grid = np.unique(np.concatenate([[m[0] for m in meas], out_times]))
    by_step: dict[int, list[int]] = {}
    for j, m in enumerate(meas):
        by_step.setdefault(int(np.searchsorted(grid, m[0])), []).append(j)
    order = np.argsort([m[0] for m in meas], kind="stable")
    first = meas[order[0]]
    x = np.zeros(4)
    P = np.diag([cfg.init_pos_sigma ** 2] * 2 + [cfg.init_vel_sigma ** 2] * 2)
    if first[4] == "ais_pos" or first[4] == "acoustic":
        x[:2] = first[1]
    vel0 = next((m for m in (meas[j] for j in order) if m[4] == "ais_vel"), None)
    if vel0 is not None:
        x[2:] = vel0[1]
    used = np.ones(len(meas), bool)

    n = len(grid)
    xf, Pf = np.zeros((n, 4)), np.zeros((n, 4, 4))        # filtered
    xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4))        # predicted
    t_prev = grid[0]
    for i, t in enumerate(grid):
        if i > 0:
            F, Q = _transition(float(t - t_prev), cfg.accel)
            x, P = F @ x, F @ P @ F.T + Q
        xp[i], Pp[i] = x, P
        for j in by_step.get(i, []):
            _, z, H, R, kind = meas[j]
            nu = z - H @ x
            S = H @ P @ H.T + R
            if kind == "acoustic":
                d2 = float(nu @ np.linalg.solve(S, nu))
                if d2 > cfg.gate_sigma ** 2 * len(nu):
                    used[j] = False
                    continue
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ nu
            P = (np.eye(4) - K @ H) @ P @ (np.eye(4) - K @ H).T + K @ R @ K.T       # Joseph form
        xf[i], Pf[i] = x, P
        t_prev = t
    # RTS smoother.
    xs, Ps = xf.copy(), Pf.copy()
    for i in range(n - 2, -1, -1):
        F, _ = _transition(float(grid[i + 1] - grid[i]), cfg.accel)
        C = Pf[i] @ F.T @ np.linalg.inv(Pp[i + 1])
        xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Pf[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T
    idx = np.searchsorted(grid, out_times)
    sig = np.sqrt(np.maximum(np.stack([Ps[:, 0, 0], Ps[:, 1, 1]], axis=1), 0.0))
    return TrackFit(out_times, xs[idx, :2], xs[idx, 2:], sig[idx], used[np.array(acoustic_idx, int)] if acoustic_idx else used)

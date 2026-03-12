"""Kalman-filter tracker for matched-field-processor detections.

Implements a constant-velocity motion model with state vector
``[x, y, vx, vy]``.  The filter smooths the noisy MFP position
estimates, fills in missed detections via prediction-only steps,
and provides velocity estimates needed by the fire-control module.
"""

from __future__ import annotations

import math

import numpy as np


class KalmanTracker:
    """Constant-velocity Kalman filter.

    State vector: ``[x, y, vx, vy]``

    Parameters
    ----------
    process_noise_std : float
        Standard deviation of acceleration noise [m/s²].
        Represents expected drone manoeuvre capability.
    measurement_noise_std : float
        Standard deviation of position measurement [m].
        Approximately equal to the MFP grid spacing.
    """

    def __init__(
        self,
        process_noise_std: float = 3.0,
        measurement_noise_std: float = 5.0,
    ) -> None:
        self.q_std = process_noise_std
        self.r_std = measurement_noise_std

        # State estimate.
        self.x = np.zeros(4)
        # State covariance — start with large uncertainty.
        self.P = np.diag([100.0, 100.0, 50.0, 50.0])
        self._initialised = False

        # Observation matrix: we measure [x, y].
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Measurement noise covariance.
        self.R = np.diag([measurement_noise_std ** 2,
                          measurement_noise_std ** 2])

    # ------------------------------------------------------------------

    def initialise(self, x: float, y: float) -> None:
        """Set the initial state from the first detection."""
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([self.r_std ** 2, self.r_std ** 2, 50.0, 50.0])
        self._initialised = True

    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Time-update (prediction) step."""
        # State-transition matrix.
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # Process noise covariance (continuous white-noise jerk model).
        q = self.q_std ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        Q = q * np.array([
            [dt4 / 4, 0,       dt3 / 2, 0      ],
            [0,       dt4 / 4, 0,       dt3 / 2],
            [dt3 / 2, 0,       dt2,     0      ],
            [0,       dt3 / 2, 0,       dt2    ],
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------

    def update(self, z: np.ndarray) -> None:
        """Measurement-update step.

        Parameters
        ----------
        z : array_like, shape (2,)
            Measured position ``[x, y]``.
        """
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x                # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        I4 = np.eye(4)
        self.P = (I4 - K @ self.H) @ self.P

    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Return ``[x, y, vx, vy]``."""
        return self.x.copy()

    def get_position(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    def get_velocity(self) -> tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))

    def get_covariance(self) -> np.ndarray:
        """Return the 4×4 state covariance matrix."""
        return self.P.copy()


# -----------------------------------------------------------------------
# High-level tracker runner
# -----------------------------------------------------------------------

def run_tracker(
    detections: list[dict],
    *,
    process_noise_std: float = 3.0,
    measurement_noise_std: float = 5.0,
) -> dict:
    """Run the Kalman tracker on a sequence of MFP detections.

    Parameters
    ----------
    detections : list of dicts
        Each dict must contain at least ``time`` (float),
        ``x`` (float or nan), ``y`` (float or nan),
        ``detected`` (bool).
    process_noise_std, measurement_noise_std : float
        Kalman filter tuning.

    Returns
    -------
    dict with:
        times        (N,)    — detection window centre times
        positions    (N, 2)  — smoothed (x, y)
        velocities   (N, 2)  — estimated (vx, vy)
        covariances  (N, 4, 4) — state covariance at each step
        raw_detections (N, 2) — raw MFP positions (may contain NaN)
        speeds       (N,)    — |v|
        headings     (N,)    — atan2(vy, vx)
    """
    kf = KalmanTracker(process_noise_std, measurement_noise_std)

    times = []
    positions = []
    velocities = []
    covariances = []
    raw_dets = []
    speeds = []
    headings = []

    prev_t = None
    # For velocity initialization from first two detections.
    _first_det: dict | None = None

    for det in detections:
        t = det["time"]
        is_det = det["detected"]
        mx = det["x"]
        my = det["y"]

        raw_dets.append([mx, my])

        if not kf._initialised:
            if is_det:
                if _first_det is None:
                    # Store first detection, wait for second.
                    _first_det = {"x": mx, "y": my, "t": t}
                    times.append(t)
                    positions.append([mx, my])
                    velocities.append([0.0, 0.0])
                    covariances.append(kf.get_covariance())
                    speeds.append(0.0)
                    headings.append(0.0)
                    prev_t = t
                    continue
                else:
                    # Second detection: estimate velocity and initialise.
                    dt_init = t - _first_det["t"]
                    if dt_init > 1e-12:
                        vx0 = (mx - _first_det["x"]) / dt_init
                        vy0 = (my - _first_det["y"]) / dt_init
                    else:
                        vx0, vy0 = 0.0, 0.0
                    kf.initialise(mx, my)
                    kf.x[2] = vx0
                    kf.x[3] = vy0
                    # Tighter velocity uncertainty since we have an estimate.
                    kf.P[2, 2] = 10.0
                    kf.P[3, 3] = 10.0
            else:
                times.append(t)
                positions.append([float("nan"), float("nan")])
                velocities.append([0.0, 0.0])
                covariances.append(kf.get_covariance())
                speeds.append(0.0)
                headings.append(0.0)
                prev_t = t
                continue

        # Predict.
        if prev_t is not None:
            dt = t - prev_t
            if dt > 0:
                kf.predict(dt)

        # Update if we have a detection.
        if is_det:
            kf.update(np.array([mx, my]))

        state = kf.get_state()
        times.append(t)
        positions.append([state[0], state[1]])
        velocities.append([state[2], state[3]])
        covariances.append(kf.get_covariance())
        spd = math.sqrt(state[2] ** 2 + state[3] ** 2)
        speeds.append(spd)
        headings.append(math.atan2(state[3], state[2]))

        prev_t = t

    return {
        "times": np.array(times),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "covariances": np.array(covariances),
        "raw_detections": np.array(raw_dets),
        "speeds": np.array(speeds),
        "headings": np.array(headings),
    }


# -----------------------------------------------------------------------
# Multi-target tracker
# -----------------------------------------------------------------------

class _Track:
    """Internal bookkeeping for one target track."""

    _next_id = 0

    def __init__(self, kf: KalmanTracker, t0: float, x: float, y: float):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.kf = kf
        self.kf.initialise(x, y)
        self.times: list[float] = [t0]
        self.positions: list[list[float]] = [[x, y]]
        self.velocities: list[list[float]] = [[0.0, 0.0]]
        self.covariances: list[np.ndarray] = [kf.get_covariance()]
        self.missed = 0
        self.confirmed = False  # becomes True after init_count updates

    def predict(self, dt: float) -> None:
        self.kf.predict(dt)

    def update(self, z: np.ndarray, t: float) -> None:
        self.kf.update(z)
        s = self.kf.get_state()
        self.times.append(t)
        self.positions.append([s[0], s[1]])
        self.velocities.append([s[2], s[3]])
        self.covariances.append(self.kf.get_covariance())
        self.missed = 0

    def mark_missed(self, t: float) -> None:
        s = self.kf.get_state()
        self.times.append(t)
        self.positions.append([s[0], s[1]])
        self.velocities.append([s[2], s[3]])
        self.covariances.append(self.kf.get_covariance())
        self.missed += 1

    def to_dict(self) -> dict:
        return {
            "track_id": self.id,
            "times": np.array(self.times),
            "positions": np.array(self.positions),
            "velocities": np.array(self.velocities),
            "covariances": np.array(self.covariances),
            "speeds": np.array([math.hypot(v[0], v[1])
                                for v in self.velocities]),
            "headings": np.array([math.atan2(v[1], v[0])
                                  for v in self.velocities]),
        }


class MultiTargetTracker:
    """Multi-target tracker with nearest-neighbour data association.

    Parameters
    ----------
    process_noise_std, measurement_noise_std : float
        Kalman filter tuning (applied to every track).
    gate_threshold : float
        Euclidean distance gate [m] for associating a detection to an
        existing track.
    max_missed : int
        Drop a track after this many consecutive missed updates.
    init_count : int
        Number of associated updates before a track is confirmed.
    """

    def __init__(
        self,
        process_noise_std: float = 3.0,
        measurement_noise_std: float = 5.0,
        gate_threshold: float = 30.0,
        max_missed: int = 5,
        init_count: int = 2,
    ) -> None:
        self.q = process_noise_std
        self.r = measurement_noise_std
        self.gate = gate_threshold
        self.max_missed = max_missed
        self.init_count = init_count
        self.tracks: list[_Track] = []
        self._prev_t: float | None = None

    def update(self, detections: list[dict], t: float) -> None:
        """Process one window's detections.

        Parameters
        ----------
        detections : list of {x, y, coherence, ...} dicts.
        t : float — window centre time.
        """
        dt = 0.0
        if self._prev_t is not None:
            dt = t - self._prev_t
        self._prev_t = t

        # Predict all existing tracks.
        if dt > 0:
            for tr in self.tracks:
                tr.predict(dt)

        # Build cost matrix (Euclidean distance).
        used_dets = set()
        used_tracks = set()

        if self.tracks and detections:
            n_t = len(self.tracks)
            n_d = len(detections)
            cost = np.full((n_t, n_d), np.inf)
            for ti, tr in enumerate(self.tracks):
                px, py = tr.kf.get_position()
                for di, det in enumerate(detections):
                    dx = det["x"] - px
                    dy = det["y"] - py
                    d = math.hypot(dx, dy)
                    if d < self.gate:
                        cost[ti, di] = d

            # Greedy nearest-neighbour assignment.
            for _ in range(min(n_t, n_d)):
                if np.all(np.isinf(cost)):
                    break
                ti, di = np.unravel_index(np.argmin(cost), cost.shape)
                if np.isinf(cost[ti, di]):
                    break
                det = detections[di]
                self.tracks[ti].update(np.array([det["x"], det["y"]]), t)
                used_dets.add(di)
                used_tracks.add(ti)
                cost[ti, :] = np.inf
                cost[:, di] = np.inf

        # Mark unassociated tracks as missed.
        for ti, tr in enumerate(self.tracks):
            if ti not in used_tracks:
                tr.mark_missed(t)

        # Initiate new tracks from unassociated detections.
        for di, det in enumerate(detections):
            if di not in used_dets:
                kf = KalmanTracker(self.q, self.r)
                tr = _Track(kf, t, det["x"], det["y"])
                self.tracks.append(tr)

        # Prune dead tracks.
        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]

        # Confirm tracks.
        for tr in self.tracks:
            if len(tr.times) >= self.init_count:
                tr.confirmed = True

    def get_tracks(self) -> list[dict]:
        """Return confirmed tracks as dicts."""
        return [tr.to_dict() for tr in self.tracks if tr.confirmed]

    def get_all_tracks(self) -> list[dict]:
        """Return all tracks (including unconfirmed)."""
        return [tr.to_dict() for tr in self.tracks]


def run_multi_tracker(
    multi_detections: list[list[dict]],
    times: np.ndarray,
    *,
    process_noise_std: float = 3.0,
    measurement_noise_std: float = 5.0,
    gate_threshold: float = 30.0,
    max_missed: int = 5,
) -> list[dict]:
    """Run multi-target tracker on multi-peak detection output.

    Parameters
    ----------
    multi_detections : list of lists
        One inner list per time window, each containing 0..N detection dicts.
    times : (n_windows,) array of window centre times.

    Returns
    -------
    list of track dicts, each with times/positions/velocities/covariances.
    """
    mtt = MultiTargetTracker(
        process_noise_std, measurement_noise_std,
        gate_threshold=gate_threshold, max_missed=max_missed,
    )
    for i, dets in enumerate(multi_detections):
        t = float(times[i]) if i < len(times) else float(i)
        mtt.update(dets, t)

    return mtt.get_tracks()

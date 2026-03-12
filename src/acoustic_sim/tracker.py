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

    for det in detections:
        t = det["time"]
        is_det = det["detected"]
        mx = det["x"]
        my = det["y"]

        raw_dets.append([mx, my])

        if not kf._initialised:
            if is_det:
                kf.initialise(mx, my)
            else:
                # Nothing to track yet.
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

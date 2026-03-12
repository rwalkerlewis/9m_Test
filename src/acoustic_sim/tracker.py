"""Extended Kalman Filter tracker for bearing-primary measurements.

The tracker receives measurements from the MFP:
- **bearing** (azimuth): high confidence (~3° for compact array)
- **range**: low confidence (~100m single-window, improves over time)
- **amplitude**: noisy proxy for range via 1/r decay

State vector: ``[x, y, vx, vy]`` in absolute Cartesian coordinates.

The measurement model is nonlinear (atan2, sqrt), so an Extended
Kalman Filter (EKF) is used with the analytical Jacobian.

Range observability
===================
Range is poorly determined from a single bearing measurement on a compact
array.  Range information accrues from:
1. **Bearing rate** — a nearby source sweeps bearing faster.
2. **Amplitude** — 1/r decay gives a noisy range proxy.
3. **Trajectory geometry** — perpendicular motion improves range.

The tracker initialises with large range uncertainty (~150 m) which
collapses as the geometry provides information.
"""

from __future__ import annotations

import math

import numpy as np


# =====================================================================
#  EKF Tracker
# =====================================================================

class EKFTracker:
    """Extended Kalman Filter with bearing / range / amplitude measurements.

    State: ``[x, y, vx, vy]`` in metres and m/s.

    Parameters
    ----------
    process_noise_std : float
        Acceleration noise std [m/s²].
    sigma_bearing : float
        Bearing measurement noise std [radians].
    sigma_range : float
        Range measurement noise std [metres].
    initial_range_guess : float
        Default range for first-detection initialization [m].
    source_level_estimate : float
        Assumed source level [Pa at 1 m] for amplitude → range mapping.
    """

    def __init__(
        self,
        process_noise_std: float = 2.0,
        sigma_bearing: float = 0.0524,   # ~3 degrees in radians
        sigma_range: float = 100.0,
        initial_range_guess: float = 200.0,
        source_level_estimate: float = 0.632,  # 90 dB re 20µPa
    ) -> None:
        self.q_std = process_noise_std
        self.sigma_bearing = sigma_bearing
        self.sigma_range = sigma_range
        self.initial_range_guess = initial_range_guess
        self.source_level = source_level_estimate

        self.x = np.zeros(4)
        self.P = np.eye(4) * 1e4
        self._initialised = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise_from_bearing(
        self,
        bearing: float,
        range_est: float | None = None,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> None:
        """Place the initial state at *bearing* from the array centre."""
        r = range_est if range_est is not None else self.initial_range_guess
        self.x = np.array([
            center_x + r * math.cos(bearing),
            center_y + r * math.sin(bearing),
            0.0,
            0.0,
        ])
        # Covariance: tight along bearing, loose along range.
        # Rotate the covariance into the bearing/range frame.
        sigma_cross = r * self.sigma_bearing  # cross-range uncertainty
        sigma_radial = self.sigma_range       # range uncertainty
        sigma_vel = 20.0                      # no velocity info yet

        # Rotation matrix from bearing/range to x/y.
        cb, sb = math.cos(bearing), math.sin(bearing)
        R = np.array([[cb, -sb], [sb, cb]])
        P_pos_br = np.diag([sigma_radial ** 2, sigma_cross ** 2])
        P_pos_xy = R @ P_pos_br @ R.T

        self.P = np.zeros((4, 4))
        self.P[:2, :2] = P_pos_xy
        self.P[2, 2] = sigma_vel ** 2
        self.P[3, 3] = sigma_vel ** 2
        self._initialised = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Constant-velocity prediction."""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])
        q = self.q_std ** 2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        Q = q * np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    # Measurement model
    # ------------------------------------------------------------------

    def _h(self, state: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Predicted measurement [bearing, range, amplitude]."""
        dx = state[0] - cx
        dy = state[1] - cy
        rng = math.sqrt(dx**2 + dy**2)
        bearing = math.atan2(dy, dx)
        amp = self.source_level / max(rng, 1.0)
        return np.array([bearing, rng, amp])

    def _H(self, state: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Jacobian of the measurement model (3 × 4)."""
        dx = state[0] - cx
        dy = state[1] - cy
        r2 = dx**2 + dy**2
        r = math.sqrt(r2)
        r = max(r, 1e-6)
        r2 = max(r2, 1e-12)

        H = np.zeros((3, 4))
        # d(bearing)/d(x,y)
        H[0, 0] = -dy / r2
        H[0, 1] =  dx / r2
        # d(range)/d(x,y)
        H[1, 0] = dx / r
        H[1, 1] = dy / r
        # d(amplitude)/d(x,y)  (amplitude ∝ 1/r)
        H[2, 0] = -self.source_level * dx / (r2 * r)
        H[2, 1] = -self.source_level * dy / (r2 * r)
        return H

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        bearing: float,
        range_est: float,
        amplitude: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> None:
        """EKF measurement update with bearing / range / amplitude."""
        z = np.array([bearing, range_est, amplitude])
        z_pred = self._h(self.x, center_x, center_y)
        H = self._H(self.x, center_x, center_y)

        # Innovation with bearing wrapping.
        y = z - z_pred
        y[0] = (y[0] + math.pi) % (2 * math.pi) - math.pi  # wrap to (-π, π]

        # Adaptive amplitude noise based on range uncertainty.
        sigma_amp = max(amplitude * 0.5, 1e-6)  # 50% amplitude noise
        R = np.diag([self.sigma_bearing**2, self.sigma_range**2, sigma_amp**2])

        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # skip update if singular

        self.x = self.x + K @ y
        I4 = np.eye(4)
        self.P = (I4 - K @ H) @ self.P

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_position(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    def get_velocity(self) -> tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    def get_range_uncertainty(self, cx: float = 0.0, cy: float = 0.0) -> float:
        """Range uncertainty (1-σ) from the covariance in the radial direction."""
        dx = self.x[0] - cx
        dy = self.x[1] - cy
        r = max(math.sqrt(dx**2 + dy**2), 1e-6)
        r_hat = np.array([dx / r, dy / r])
        pos_cov = self.P[:2, :2]
        return float(math.sqrt(r_hat @ pos_cov @ r_hat))


# =====================================================================
#  High-level tracker runner
# =====================================================================

def run_tracker(
    detections: list[dict],
    *,
    process_noise_std: float = 2.0,
    sigma_bearing_deg: float = 3.0,
    sigma_range: float = 100.0,
    initial_range_guess: float = 200.0,
    source_level_dB: float = 90.0,
    array_center_x: float = 0.0,
    array_center_y: float = 0.0,
    # Legacy params (ignored but kept for API compat)
    measurement_noise_std: float = 5.0,
) -> dict:
    """Run the EKF tracker on MFP detections.

    Parameters
    ----------
    detections : list of dicts from ``matched_field_process``.
        Each must have ``time``, ``detected``, and if detected:
        ``bearing`` (rad), ``range`` (m), ``coherence`` (amplitude proxy).

    Returns
    -------
    dict with times, positions, velocities, covariances, bearings, ranges,
    range_uncertainties, speeds, headings.
    """
    p_ref = 20e-6
    source_level_pa = p_ref * 10.0 ** (source_level_dB / 20.0)
    sigma_bearing_rad = math.radians(sigma_bearing_deg)

    kf = EKFTracker(
        process_noise_std=process_noise_std,
        sigma_bearing=sigma_bearing_rad,
        sigma_range=sigma_range,
        initial_range_guess=initial_range_guess,
        source_level_estimate=source_level_pa,
    )
    cx, cy = array_center_x, array_center_y

    times, positions, velocities, covariances = [], [], [], []
    raw_bearings, raw_ranges = [], []
    range_uncertainties, speeds, headings = [], [], []
    prev_t = None

    for det in detections:
        t = det["time"]
        is_det = det["detected"]

        bearing = det.get("bearing", float("nan"))
        range_est = det.get("range", float("nan"))
        amplitude = det.get("coherence", 0.0)

        raw_bearings.append(bearing)
        raw_ranges.append(range_est)

        if not kf._initialised:
            if is_det and not math.isnan(bearing):
                kf.initialise_from_bearing(bearing, range_est, cx, cy)
            else:
                times.append(t)
                positions.append([float("nan"), float("nan")])
                velocities.append([0.0, 0.0])
                covariances.append(kf.get_covariance())
                range_uncertainties.append(float("nan"))
                speeds.append(0.0)
                headings.append(0.0)
                prev_t = t
                continue

        # Predict.
        if prev_t is not None:
            dt = t - prev_t
            if dt > 0:
                kf.predict(dt)

        # Update.
        if is_det and not math.isnan(bearing):
            # Use amplitude as proxy (coherence ≈ beam power ∝ received SNR).
            kf.update(bearing, range_est, amplitude, cx, cy)

        state = kf.get_state()
        times.append(t)
        positions.append([state[0], state[1]])
        velocities.append([state[2], state[3]])
        covariances.append(kf.get_covariance())
        range_uncertainties.append(kf.get_range_uncertainty(cx, cy))
        spd = math.sqrt(state[2]**2 + state[3]**2)
        speeds.append(spd)
        headings.append(math.atan2(state[3], state[2]))

        prev_t = t

    return {
        "times": np.array(times),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "covariances": np.array(covariances),
        "raw_bearings": np.array(raw_bearings),
        "raw_ranges": np.array(raw_ranges),
        "range_uncertainties": np.array(range_uncertainties),
        "speeds": np.array(speeds),
        "headings": np.array(headings),
        # Legacy compatibility.
        "raw_detections": np.column_stack([
            np.array(raw_bearings),
            np.array(raw_ranges),
        ]) if raw_bearings else np.empty((0, 2)),
    }


# =====================================================================
#  Multi-target tracker (updated to use EKF)
# =====================================================================

class _Track:
    """Internal bookkeeping for one EKF track."""
    _next_id = 0

    def __init__(self, kf: EKFTracker, t0: float, bearing: float,
                 range_est: float, cx: float, cy: float):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.kf = kf
        self.kf.initialise_from_bearing(bearing, range_est, cx, cy)
        s = kf.get_state()
        self.times = [t0]
        self.positions = [[s[0], s[1]]]
        self.velocities = [[s[2], s[3]]]
        self.covariances = [kf.get_covariance()]
        self.missed = 0
        self.confirmed = False

    def predict(self, dt: float) -> None:
        self.kf.predict(dt)

    def update(self, det: dict, t: float, cx: float, cy: float) -> None:
        self.kf.update(det["bearing"], det["range"], det.get("coherence", 0.0), cx, cy)
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
            "speeds": np.array([math.hypot(v[0], v[1]) for v in self.velocities]),
            "headings": np.array([math.atan2(v[1], v[0]) for v in self.velocities]),
        }


class MultiTargetTracker:
    """Multi-target EKF tracker with nearest-neighbour data association."""

    def __init__(
        self,
        process_noise_std: float = 2.0,
        sigma_bearing_deg: float = 3.0,
        sigma_range: float = 100.0,
        initial_range_guess: float = 200.0,
        gate_threshold: float = 30.0,
        max_missed: int = 5,
        source_level_dB: float = 90.0,
    ) -> None:
        self.q = process_noise_std
        self.sig_b = math.radians(sigma_bearing_deg)
        self.sig_r = sigma_range
        self.r_guess = initial_range_guess
        self.gate = gate_threshold
        self.max_missed = max_missed
        p_ref = 20e-6
        self.src_pa = p_ref * 10.0 ** (source_level_dB / 20.0)
        self.tracks: list[_Track] = []
        self._prev_t: float | None = None
        self._cx = 0.0
        self._cy = 0.0

    def set_array_center(self, cx: float, cy: float) -> None:
        self._cx = cx
        self._cy = cy

    def update(self, detections: list[dict], t: float) -> None:
        dt = 0.0
        if self._prev_t is not None:
            dt = t - self._prev_t
        self._prev_t = t

        if dt > 0:
            for tr in self.tracks:
                tr.predict(dt)

        used_dets: set[int] = set()
        used_tracks: set[int] = set()

        if self.tracks and detections:
            n_t, n_d = len(self.tracks), len(detections)
            cost = np.full((n_t, n_d), np.inf)
            for ti, tr in enumerate(self.tracks):
                px, py = tr.kf.get_position()
                for di, det in enumerate(detections):
                    dx = det.get("x", float("nan")) - px
                    dy = det.get("y", float("nan")) - py
                    d = math.hypot(dx, dy)
                    if d < self.gate:
                        cost[ti, di] = d
            for _ in range(min(n_t, n_d)):
                if np.all(np.isinf(cost)):
                    break
                ti, di = np.unravel_index(int(np.argmin(cost)), cost.shape)
                if np.isinf(cost[ti, di]):
                    break
                self.tracks[ti].update(detections[di], t, self._cx, self._cy)
                used_dets.add(di)
                used_tracks.add(ti)
                cost[ti, :] = np.inf
                cost[:, di] = np.inf

        for ti, tr in enumerate(self.tracks):
            if ti not in used_tracks:
                tr.mark_missed(t)

        for di, det in enumerate(detections):
            if di not in used_dets:
                kf = EKFTracker(self.q, self.sig_b, self.sig_r,
                                self.r_guess, self.src_pa)
                bearing = det.get("bearing", 0.0)
                rng = det.get("range", self.r_guess)
                tr = _Track(kf, t, bearing, rng, self._cx, self._cy)
                self.tracks.append(tr)

        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]
        for tr in self.tracks:
            if len(tr.times) >= 2:
                tr.confirmed = True

    def get_tracks(self) -> list[dict]:
        return [tr.to_dict() for tr in self.tracks if tr.confirmed]

    def get_all_tracks(self) -> list[dict]:
        return [tr.to_dict() for tr in self.tracks]


def run_multi_tracker(
    multi_detections: list[list[dict]],
    times: np.ndarray,
    *,
    process_noise_std: float = 2.0,
    sigma_bearing_deg: float = 3.0,
    sigma_range: float = 100.0,
    initial_range_guess: float = 200.0,
    gate_threshold: float = 30.0,
    max_missed: int = 5,
    source_level_dB: float = 90.0,
    array_center_x: float = 0.0,
    array_center_y: float = 0.0,
    # Legacy compat
    measurement_noise_std: float = 5.0,
) -> list[dict]:
    """Run multi-target EKF tracker on multi-peak detections."""
    mtt = MultiTargetTracker(
        process_noise_std, sigma_bearing_deg, sigma_range,
        initial_range_guess, gate_threshold, max_missed, source_level_dB,
    )
    mtt.set_array_center(array_center_x, array_center_y)
    for i, dets in enumerate(multi_detections):
        t = float(times[i]) if i < len(times) else float(i)
        mtt.update(dets, t)
    return mtt.get_tracks()

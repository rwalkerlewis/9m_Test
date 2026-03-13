"""3D Extended Kalman Filter tracker.

State vector: ``[x, y, z, vx, vy, vz]`` in absolute Cartesian coordinates.

Extends the 2D EKF tracker with a z-dimension.  When all z-coordinates
are zero, the (x, y, vx, vy) components are identical to the 2D tracker
and z, vz remain at zero.
"""

from __future__ import annotations

import math

import numpy as np


# =====================================================================
#  3D EKF Tracker
# =====================================================================

class EKFTracker3D:
    """Extended Kalman Filter for 3D tracking.

    State: ``[x, y, z, vx, vy, vz]`` in metres and m/s.
    """

    def __init__(
        self,
        process_noise_std: float = 2.0,
        sigma_bearing: float = 0.0524,
        sigma_range: float = 100.0,
        sigma_elevation: float = 0.1,
        initial_range_guess: float = 200.0,
        source_level_estimate: float = 0.632,
    ) -> None:
        self.q_std = process_noise_std
        self.sigma_bearing = sigma_bearing
        self.sigma_range = sigma_range
        self.sigma_elevation = sigma_elevation
        self.initial_range_guess = initial_range_guess
        self.source_level = source_level_estimate

        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e4
        self._initialised = False

        # Adaptive process noise multiplier (for maneuver detection).
        self._q_multiplier = 1.0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise_from_detection(
        self,
        bearing: float,
        range_est: float | None = None,
        z_est: float = 0.0,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> None:
        """Place the initial state from a detection."""
        r = range_est if range_est is not None else self.initial_range_guess
        self.x = np.array([
            center_x + r * math.cos(bearing),
            center_y + r * math.sin(bearing),
            z_est,
            0.0,
            0.0,
            0.0,
        ])
        sigma_cross = r * self.sigma_bearing
        sigma_radial = self.sigma_range
        sigma_vel = 20.0
        sigma_z = max(abs(z_est) * 0.5, 10.0)

        cb, sb = math.cos(bearing), math.sin(bearing)
        R = np.array([[cb, -sb], [sb, cb]])
        P_pos_br = np.diag([sigma_radial ** 2, sigma_cross ** 2])
        P_pos_xy = R @ P_pos_br @ R.T

        self.P = np.zeros((6, 6))
        self.P[:2, :2] = P_pos_xy
        self.P[2, 2] = sigma_z ** 2
        self.P[3, 3] = sigma_vel ** 2
        self.P[4, 4] = sigma_vel ** 2
        self.P[5, 5] = sigma_vel ** 2
        self._initialised = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Constant-velocity prediction in 3D."""
        F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], dtype=np.float64)

        q = (self.q_std ** 2) * self._q_multiplier
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        Q = q * np.array([
            [dt4 / 4, 0, 0, dt3 / 2, 0, 0],
            [0, dt4 / 4, 0, 0, dt3 / 2, 0],
            [0, 0, dt4 / 4, 0, 0, dt3 / 2],
            [dt3 / 2, 0, 0, dt2, 0, 0],
            [0, dt3 / 2, 0, 0, dt2, 0],
            [0, 0, dt3 / 2, 0, 0, dt2],
        ], dtype=np.float64)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    # Measurement model
    # ------------------------------------------------------------------

    def _h(self, state: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Predicted measurement [bearing, range_3d, amplitude]."""
        dx = state[0] - cx
        dy = state[1] - cy
        dz = state[2]
        rng = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        bearing = math.atan2(dy, dx)
        amp = self.source_level / max(rng, 1.0)
        return np.array([bearing, rng, amp])

    def _H(self, state: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Jacobian of the measurement model (3 × 6)."""
        dx = state[0] - cx
        dy = state[1] - cy
        dz = state[2]
        r2_horiz = dx ** 2 + dy ** 2
        r2 = r2_horiz + dz ** 2
        r = max(math.sqrt(r2), 1e-6)
        r2 = max(r2, 1e-12)
        r2_horiz = max(r2_horiz, 1e-12)

        H = np.zeros((3, 6))
        # d(bearing)/d(x,y,z,vx,vy,vz)
        H[0, 0] = -dy / r2_horiz
        H[0, 1] = dx / r2_horiz
        # H[0, 2] = 0  (bearing doesn't depend on z)
        # d(range_3d)/d(x,y,z)
        H[1, 0] = dx / r
        H[1, 1] = dy / r
        H[1, 2] = dz / r
        # d(amplitude)/d(x,y,z) — amplitude ∝ 1/r
        H[2, 0] = -self.source_level * dx / (r2 * r)
        H[2, 1] = -self.source_level * dy / (r2 * r)
        H[2, 2] = -self.source_level * dz / (r2 * r)
        return H

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        bearing: float,
        range_est: float,
        amplitude: float,
        z_est: float = 0.0,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> None:
        """EKF measurement update with bearing / range / amplitude."""
        z_meas = np.array([bearing, range_est, amplitude])
        z_pred = self._h(self.x, center_x, center_y)
        H = self._H(self.x, center_x, center_y)

        y = z_meas - z_pred
        y[0] = (y[0] + math.pi) % (2 * math.pi) - math.pi

        sigma_amp = max(amplitude * 0.5, 1e-6)
        R = np.diag([self.sigma_bearing ** 2, self.sigma_range ** 2, sigma_amp ** 2])

        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.x = self.x + K @ y
        I6 = np.eye(6)
        self.P = (I6 - K @ H) @ self.P

    # ------------------------------------------------------------------
    # Adaptive process noise
    # ------------------------------------------------------------------

    def set_process_noise_multiplier(self, multiplier: float) -> None:
        """Set the process noise multiplier (from maneuver detector)."""
        self._q_multiplier = max(multiplier, 0.1)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_position(self) -> tuple[float, float, float]:
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    def get_velocity(self) -> tuple[float, float, float]:
        return (float(self.x[3]), float(self.x[4]), float(self.x[5]))

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    def get_range_uncertainty(self, cx: float = 0.0, cy: float = 0.0) -> float:
        dx = self.x[0] - cx
        dy = self.x[1] - cy
        dz = self.x[2]
        r = max(math.sqrt(dx ** 2 + dy ** 2 + dz ** 2), 1e-6)
        r_hat = np.array([dx / r, dy / r, dz / r])
        pos_cov = self.P[:3, :3]
        return float(math.sqrt(r_hat @ pos_cov @ r_hat))


# =====================================================================
#  High-level 3D tracker runner
# =====================================================================

def run_tracker_3d(
    detections: list[dict],
    *,
    process_noise_std: float = 2.0,
    sigma_bearing_deg: float = 3.0,
    sigma_range: float = 100.0,
    initial_range_guess: float = 200.0,
    source_level_dB: float = 90.0,
    array_center_x: float = 0.0,
    array_center_y: float = 0.0,
) -> dict:
    """Run the 3D EKF tracker on MFP detections.

    Parameters
    ----------
    detections : list of dicts from ``matched_field_process_3d``.
        Each must have ``time``, ``detected``, and if detected:
        ``bearing`` (rad), ``range`` (m), ``z`` (m), ``coherence``.

    Returns
    -------
    dict with times, positions (N,3), velocities (N,3), covariances (N,6,6), etc.
    """
    p_ref = 20e-6
    source_level_pa = p_ref * 10.0 ** (source_level_dB / 20.0)
    sigma_bearing_rad = math.radians(sigma_bearing_deg)

    kf = EKFTracker3D(
        process_noise_std=process_noise_std,
        sigma_bearing=sigma_bearing_rad,
        sigma_range=sigma_range,
        initial_range_guess=initial_range_guess,
        source_level_estimate=source_level_pa,
    )
    cx, cy = array_center_x, array_center_y

    times, positions, velocities, covariances = [], [], [], []
    raw_bearings, raw_ranges, raw_zs = [], [], []
    range_uncertainties, speeds, headings = [], [], []
    prev_t = None

    for det in detections:
        t = det["time"]
        is_det = det["detected"]

        bearing = det.get("bearing", float("nan"))
        range_est = det.get("range", float("nan"))
        z_est = det.get("z", 0.0)
        if z_est is None or (isinstance(z_est, float) and math.isnan(z_est)):
            z_est = 0.0
        amplitude = det.get("coherence", 0.0)

        raw_bearings.append(bearing)
        raw_ranges.append(range_est)
        raw_zs.append(z_est)

        if not kf._initialised:
            if is_det and not math.isnan(bearing):
                kf.initialise_from_detection(bearing, range_est, z_est, cx, cy)
            else:
                times.append(t)
                positions.append([float("nan"), float("nan"), float("nan")])
                velocities.append([0.0, 0.0, 0.0])
                covariances.append(kf.get_covariance())
                range_uncertainties.append(float("nan"))
                speeds.append(0.0)
                headings.append(0.0)
                prev_t = t
                continue

        if prev_t is not None:
            dt_step = t - prev_t
            if dt_step > 0:
                kf.predict(dt_step)

        if is_det and not math.isnan(bearing):
            kf.update(bearing, range_est, amplitude, z_est, cx, cy)

        state = kf.get_state()
        times.append(t)
        positions.append([state[0], state[1], state[2]])
        velocities.append([state[3], state[4], state[5]])
        covariances.append(kf.get_covariance())
        range_uncertainties.append(kf.get_range_uncertainty(cx, cy))
        spd = math.sqrt(state[3] ** 2 + state[4] ** 2 + state[5] ** 2)
        speeds.append(spd)
        headings.append(math.atan2(state[4], state[3]))

        prev_t = t

    return {
        "times": np.array(times),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "covariances": np.array(covariances),
        "raw_bearings": np.array(raw_bearings),
        "raw_ranges": np.array(raw_ranges),
        "raw_zs": np.array(raw_zs),
        "range_uncertainties": np.array(range_uncertainties),
        "speeds": np.array(speeds),
        "headings": np.array(headings),
    }


# =====================================================================
#  Multi-target 3D tracker
# =====================================================================

class _Track3D:
    """Internal bookkeeping for one 3D EKF track."""
    _next_id = 0

    def __init__(self, kf: EKFTracker3D, t0: float, bearing: float,
                 range_est: float, z_est: float, cx: float, cy: float):
        self.id = _Track3D._next_id
        _Track3D._next_id += 1
        self.kf = kf
        self.kf.initialise_from_detection(bearing, range_est, z_est, cx, cy)
        s = kf.get_state()
        self.times = [t0]
        self.positions = [[s[0], s[1], s[2]]]
        self.velocities = [[s[3], s[4], s[5]]]
        self.covariances = [kf.get_covariance()]
        self.missed = 0
        self.confirmed = False
        self.class_label = "unknown"
        self.class_confidence = 0.0
        self.maneuver_class = "steady"

    def predict(self, dt: float) -> None:
        self.kf.predict(dt)

    def update(self, det: dict, t: float, cx: float, cy: float) -> None:
        z_est = det.get("z", 0.0)
        if z_est is None or (isinstance(z_est, float) and math.isnan(z_est)):
            z_est = 0.0
        self.kf.update(det["bearing"], det["range"], det.get("coherence", 0.0),
                       z_est, cx, cy)
        s = self.kf.get_state()
        self.times.append(t)
        self.positions.append([s[0], s[1], s[2]])
        self.velocities.append([s[3], s[4], s[5]])
        self.covariances.append(self.kf.get_covariance())
        self.missed = 0

    def mark_missed(self, t: float) -> None:
        s = self.kf.get_state()
        self.times.append(t)
        self.positions.append([s[0], s[1], s[2]])
        self.velocities.append([s[3], s[4], s[5]])
        self.covariances.append(self.kf.get_covariance())
        self.missed += 1

    def to_dict(self) -> dict:
        return {
            "track_id": self.id,
            "times": np.array(self.times),
            "positions": np.array(self.positions),
            "velocities": np.array(self.velocities),
            "covariances": np.array(self.covariances),
            "speeds": np.array([math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                                for v in self.velocities]),
            "headings": np.array([math.atan2(v[1], v[0]) for v in self.velocities]),
            "class_label": self.class_label,
            "class_confidence": self.class_confidence,
            "maneuver_class": self.maneuver_class,
        }


class MultiTargetTracker3D:
    """Multi-target 3D EKF tracker with nearest-neighbour data association."""

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
        self.tracks: list[_Track3D] = []
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
                px, py, pz = tr.kf.get_position()
                for di, det in enumerate(detections):
                    dx = det.get("x", float("nan")) - px
                    dy = det.get("y", float("nan")) - py
                    dz = (det.get("z", 0.0) or 0.0) - pz
                    d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
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
                kf = EKFTracker3D(self.q, self.sig_b, self.sig_r,
                                  initial_range_guess=self.r_guess,
                                  source_level_estimate=self.src_pa)
                bearing = det.get("bearing", 0.0)
                rng = det.get("range", self.r_guess)
                z_est = det.get("z", 0.0) or 0.0
                tr = _Track3D(kf, t, bearing, rng, z_est, self._cx, self._cy)
                self.tracks.append(tr)

        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]
        for tr in self.tracks:
            if len(tr.times) >= 2:
                tr.confirmed = True

    def get_tracks(self) -> list[dict]:
        return [tr.to_dict() for tr in self.tracks if tr.confirmed]

    def get_all_tracks(self) -> list[dict]:
        return [tr.to_dict() for tr in self.tracks]


def run_multi_tracker_3d(
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
) -> list[dict]:
    """Run multi-target 3D EKF tracker on multi-peak detections."""
    mtt = MultiTargetTracker3D(
        process_noise_std, sigma_bearing_deg, sigma_range,
        initial_range_guess, gate_threshold, max_missed, source_level_dB,
    )
    mtt.set_array_center(array_center_x, array_center_y)
    for i, dets in enumerate(multi_detections):
        t = float(times[i]) if i < len(times) else float(i)
        mtt.update(dets, t)
    return mtt.get_tracks()

"""Fire-control solution for a 12-gauge shotgun engagement.

Computes pellet time-of-flight (with deceleration), iterative lead
angle, and engagement envelope (pattern size vs. position uncertainty).

Physics notes
=============
* Pellet deceleration is modelled as a constant drag per metre of
  travel: ``v(r) = v_muzzle - decel * r``.
* Time of flight is the positive root of the kinematic equation:
  ``range = v_muzzle * t - 0.5 * a * t²`` where ``a = decel * v_avg``
  (approximated iteratively).
* Lead angle is computed iteratively: predict intercept → recompute
  TOF to intercept range → repeat until convergence.
"""

from __future__ import annotations

import math

import numpy as np


# -----------------------------------------------------------------------
# Pellet ballistics
# -----------------------------------------------------------------------

def time_of_flight(
    range_m: float,
    muzzle_velocity: float = 400.0,
    decel: float = 1.5,
) -> float:
    """Solve ``range = v₀·t − ½·a·t²`` for *t* (positive root).

    The effective deceleration *a* in m/s² is approximated as
    ``decel × v_avg``, where ``v_avg`` is iteratively refined.

    Parameters
    ----------
    range_m : float
        Distance to target [m].
    muzzle_velocity : float
        Pellet muzzle velocity [m/s].
    decel : float
        Velocity loss per metre of travel [m/s per m].

    Returns
    -------
    float
        Time of flight [s], or ``inf`` if the pellet cannot reach.
    """
    if range_m <= 0:
        return 0.0

    v_at_range = muzzle_velocity - decel * range_m
    if v_at_range <= 0:
        return float("inf")

    # Average velocity over the trajectory.
    v_avg = 0.5 * (muzzle_velocity + v_at_range)
    if v_avg <= 0:
        return float("inf")

    tof = range_m / v_avg
    return tof


def pellet_velocity_at_range(
    range_m: float,
    muzzle_velocity: float = 400.0,
    decel: float = 1.5,
) -> float:
    """Return pellet speed at *range_m* [m/s]."""
    v = muzzle_velocity - decel * range_m
    return max(v, 0.0)


def pattern_diameter(
    range_m: float,
    spread_rate: float = 0.025,
) -> float:
    """Shotgun pattern diameter at *range_m* [m].

    Default spread rate: 1 m diameter per 40 m ≈ 0.025 m/m.
    """
    return spread_rate * range_m


# -----------------------------------------------------------------------
# Lead angle (iterative)
# -----------------------------------------------------------------------

def compute_lead(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    weapon_pos: np.ndarray,
    muzzle_velocity: float = 400.0,
    decel: float = 1.5,
    max_iter: int = 5,
) -> dict:
    """Compute iterative lead-angle solution.

    Parameters
    ----------
    target_pos : (2,) — [x, y] current estimated target position.
    target_vel : (2,) — [vx, vy] current estimated target velocity.
    weapon_pos : (2,) — [x, y] weapon (shooter) position.
    muzzle_velocity : float
    decel : float
    max_iter : int

    Returns
    -------
    dict with:
        aim_bearing   — radians, direction to aim
        lead_angle    — radians, difference from direct bearing
        intercept_pos — (2,) predicted impact point
        tof           — time of flight to intercept [s]
        converged     — bool
    """
    tp = np.asarray(target_pos, dtype=np.float64)
    tv = np.asarray(target_vel, dtype=np.float64)
    wp = np.asarray(weapon_pos, dtype=np.float64)

    # Direct bearing to current position.
    direct = tp - wp
    direct_bearing = math.atan2(direct[1], direct[0])
    current_range = float(np.linalg.norm(direct))

    # Iterative refinement.
    tof = time_of_flight(current_range, muzzle_velocity, decel)
    intercept = tp.copy()
    converged = False

    for _ in range(max_iter):
        intercept = tp + tv * tof
        new_range = float(np.linalg.norm(intercept - wp))
        new_tof = time_of_flight(new_range, muzzle_velocity, decel)
        if abs(new_tof - tof) < 1e-6:
            converged = True
            tof = new_tof
            break
        tof = new_tof

    aim_bearing = math.atan2(intercept[1] - wp[1], intercept[0] - wp[0])
    lead_angle = aim_bearing - direct_bearing
    # Normalise to (-π, π].
    lead_angle = (lead_angle + math.pi) % (2 * math.pi) - math.pi

    return {
        "aim_bearing": aim_bearing,
        "lead_angle": lead_angle,
        "intercept_pos": intercept,
        "tof": tof,
        "converged": converged,
    }


# -----------------------------------------------------------------------
# Engagement envelope
# -----------------------------------------------------------------------

def compute_engagement(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_cov: np.ndarray,
    weapon_pos: np.ndarray,
    muzzle_velocity: float = 400.0,
    decel: float = 1.5,
    spread_rate: float = 0.025,
    max_iter: int = 5,
    max_position_uncertainty: float = 0.0,
    max_engagement_range: float = 0.0,
) -> dict:
    """Determine whether engagement is feasible.

    The engagement envelope is the set of conditions where the shotgun
    pattern at the predicted intercept point is larger than the 2-σ
    position uncertainty from the tracker.

    Parameters
    ----------
    max_position_uncertainty : float
        Maximum allowed 2-sigma position uncertainty to engage (m).
        If 0, only requires pattern > uncertainty.
        If > 0, also requires pos_unc < max_position_uncertainty.
    max_engagement_range : float
        Maximum range (m) at which to engage. 0 = no limit.

    Returns
    -------
    dict with:
        can_fire              — bool
        range                 — current range [m]
        pattern_diam          — pattern diameter at intercept [m]
        position_uncertainty  — 2-σ positional uncertainty [m]
        crossing_speed        — target speed perpendicular to LOS [m/s]
        reason                — human-readable status string
    """
    tp = np.asarray(target_pos, dtype=np.float64)
    tv = np.asarray(target_vel, dtype=np.float64)
    wp = np.asarray(weapon_pos, dtype=np.float64)

    current_range = float(np.linalg.norm(tp - wp))

    # Position uncertainty (2-sigma of the position block of covariance).
    pos_cov = np.asarray(target_cov)[:2, :2]
    eigvals = np.linalg.eigvalsh(pos_cov)
    sigma_max = math.sqrt(max(eigvals.max(), 0.0))
    pos_unc = 2.0 * sigma_max  # 2-σ

    # Lead solution.
    lead = compute_lead(tp, tv, wp, muzzle_velocity, decel, max_iter)
    tof = lead["tof"]
    intercept = lead["intercept_pos"]
    intercept_range = float(np.linalg.norm(intercept - wp))

    # Pattern at intercept.
    pat_diam = pattern_diameter(intercept_range, spread_rate)

    # Pellet velocity at intercept.
    v_pellet = pellet_velocity_at_range(intercept_range, muzzle_velocity, decel)

    # Crossing speed (component of target velocity perpendicular to LOS).
    los = tp - wp
    los_norm = los / max(np.linalg.norm(los), 1e-6)
    v_along = float(np.dot(tv, los_norm))
    crossing = float(np.linalg.norm(tv - v_along * los_norm))

    # Decision logic.
    can_fire = True
    reason = "FIRE"

    if tof == float("inf") or v_pellet <= 0:
        can_fire = False
        reason = "OUT_OF_RANGE"
    elif max_engagement_range > 0 and current_range > max_engagement_range:
        can_fire = False
        reason = "TOO_FAR"
    elif max_position_uncertainty > 0 and pos_unc > max_position_uncertainty:
        can_fire = False
        reason = "UNCERTAINTY_TOO_HIGH"
    elif pat_diam < pos_unc:
        can_fire = False
        reason = "UNCERTAINTY_TOO_HIGH"
    elif intercept_range > muzzle_velocity / decel:
        can_fire = False
        reason = "MAX_RANGE_EXCEEDED"

    return {
        "can_fire": can_fire,
        "range": current_range,
        "pattern_diam": pat_diam,
        "position_uncertainty": pos_unc,
        "crossing_speed": crossing,
        "reason": reason,
    }


# -----------------------------------------------------------------------
# Full fire-control runner
# -----------------------------------------------------------------------

def run_fire_control(
    track: dict,
    *,
    weapon_position: tuple[float, float] = (0.0, 0.0),
    muzzle_velocity: float = 400.0,
    pellet_decel: float = 1.5,
    pattern_spread_rate: float = 0.025,
    max_iterations: int = 5,
    max_hits: int = 0,
    hit_threshold: float = 3.0,
    ground_truth_fn: callable | None = None,
    max_position_uncertainty: float = 0.0,
    max_engagement_range: float = 0.0,
) -> dict:
    """Compute fire-control solution at every tracked time step.

    Parameters
    ----------
    track : dict
        Output of :func:`acoustic_sim.tracker.run_tracker`.
    weapon_position : (x, y)
    muzzle_velocity, pellet_decel, pattern_spread_rate : float
    max_iterations : int
    max_hits : int
        Stop firing after this many confirmed hits. 0 = unlimited.
        Requires *ground_truth_fn* to evaluate hits.
    hit_threshold : float
        Miss distance (m) below which a shot counts as a hit.
    ground_truth_fn : callable or None
        ``ground_truth_fn(t) -> (x, y)`` returns true target position.
        Required when *max_hits* > 0.
    max_position_uncertainty : float
        Maximum allowed 2-sigma position uncertainty to engage (m).
        If 0, only requires pattern > uncertainty.
    max_engagement_range : float
        Maximum range (m) at which to engage. 0 = no limit.

    Returns
    -------
    dict with arrays:
        times, aim_bearings, lead_angles, tofs, can_fire (bool),
        ranges, intercept_positions (N, 2), reasons (list[str])
    """
    wp = np.array(weapon_position, dtype=np.float64)
    N = len(track["times"])

    aim_bearings = np.zeros(N)
    lead_angles = np.zeros(N)
    tofs = np.zeros(N)
    can_fire = np.zeros(N, dtype=bool)
    ranges = np.zeros(N)
    intercepts = np.zeros((N, 2))
    reasons: list[str] = []
    
    hits = 0  # Track confirmed hits for max_hits

    for i in range(N):
        pos = track["positions"][i]
        vel = track["velocities"][i]
        cov = track["covariances"][i]

        if np.any(np.isnan(pos)):
            aim_bearings[i] = float("nan")
            lead_angles[i] = float("nan")
            tofs[i] = float("nan")
            can_fire[i] = False
            ranges[i] = float("nan")
            intercepts[i] = [float("nan"), float("nan")]
            reasons.append("NO_TRACK")
            continue

        lead = compute_lead(pos, vel, wp, muzzle_velocity, pellet_decel,
                            max_iterations)
        eng = compute_engagement(pos, vel, cov, wp, muzzle_velocity,
                                 pellet_decel, pattern_spread_rate,
                                 max_iterations, max_position_uncertainty,
                                 max_engagement_range)

        aim_bearings[i] = lead["aim_bearing"]
        lead_angles[i] = lead["lead_angle"]
        tofs[i] = lead["tof"]
        ranges[i] = eng["range"]
        intercepts[i] = lead["intercept_pos"]
        
        # Check if we've reached max hits for this target
        if max_hits > 0 and hits >= max_hits:
            can_fire[i] = False
            reasons.append("TARGET_ENGAGED")
        else:
            can_fire[i] = eng["can_fire"]
            reasons.append(eng["reason"])
            if eng["can_fire"] and ground_truth_fn is not None:
                t = track["times"][i]
                gt_x, gt_y = ground_truth_fn(t)
                ix, iy = lead["intercept_pos"]
                miss = math.sqrt((ix - gt_x)**2 + (iy - gt_y)**2)
                if miss < hit_threshold:
                    hits += 1

    return {
        "times": track["times"].copy(),
        "aim_bearings": aim_bearings,
        "lead_angles": lead_angles,
        "tofs": tofs,
        "can_fire": can_fire,
        "ranges": ranges,
        "intercept_positions": intercepts,
        "reasons": reasons,
    }


# -----------------------------------------------------------------------
# Miss distance (the real success metric)
# -----------------------------------------------------------------------

def compute_miss_distance(
    fire_control: dict,
    true_positions: np.ndarray,
    true_times: np.ndarray,
    weapon_position: tuple[float, float] | np.ndarray = (0.0, 0.0),
    pattern_spread_rate: float = 0.025,
) -> dict:
    """Compute how far each shot would miss the actual target.

    For each fire-control time step, this determines where the pellet
    pattern centre would arrive (the predicted intercept from the
    tracker's estimate) versus where the drone *actually* is at the
    time of impact (true position at ``t_fire + tof``).

    Parameters
    ----------
    fire_control : dict
        Output of :func:`run_fire_control`.
    true_positions : (n_true_steps, 2)
        True drone positions at each FDTD step.
    true_times : (n_true_steps,)
        Times corresponding to *true_positions*.

    Returns
    -------
    dict with:
        miss_distances   (N,)  — miss distance at each FC time step [m]
        pattern_diameters (N,) — pattern diameter at each intercept [m]
        would_hit        (N,)  — bool, miss < pattern_radius
        first_shot_idx   int   — index of the first eligible shot
        first_shot_time  float — time of first eligible shot [s]
        first_shot_miss  float — miss distance of first shot [m]
        first_shot_pattern float — pattern diameter of first shot [m]
        first_shot_hit   bool  — whether first shot would hit
    """
    fc_times = fire_control["times"]
    tofs = fire_control["tofs"]
    intercepts = fire_control["intercept_positions"]
    can_fire = fire_control["can_fire"]
    ranges = fire_control["ranges"]
    N = len(fc_times)

    miss_distances = np.full(N, np.nan)
    pat_diams = np.full(N, np.nan)
    would_hit = np.zeros(N, dtype=bool)

    for i in range(N):
        if np.isnan(tofs[i]) or np.any(np.isnan(intercepts[i])):
            continue

        # Time of impact.
        t_impact = fc_times[i] + tofs[i]

        # True position at impact time (interpolate).
        if t_impact < true_times[0] or t_impact > true_times[-1]:
            continue
        true_x = float(np.interp(t_impact, true_times, true_positions[:, 0]))
        true_y = float(np.interp(t_impact, true_times, true_positions[:, 1]))

        # Miss distance = ||predicted_intercept - true_position_at_impact||.
        miss = math.hypot(intercepts[i, 0] - true_x,
                          intercepts[i, 1] - true_y)
        miss_distances[i] = miss

        # Pattern diameter at intercept range.
        wp = np.asarray(weapon_position, dtype=np.float64)
        int_range = float(np.linalg.norm(intercepts[i] - wp))
        pd = pattern_diameter(int_range, pattern_spread_rate)
        pat_diams[i] = pd

        # Would the shot hit? (miss < pattern radius)
        would_hit[i] = miss < pd / 2.0

    # First eligible shot: earliest timestep where can_fire is True
    # (or if can_fire is never True, earliest timestep with a valid track).
    first_idx = -1
    if np.any(can_fire):
        first_idx = int(np.argmax(can_fire))
    else:
        # Fall back: first timestep with a valid miss distance.
        valid = ~np.isnan(miss_distances)
        if np.any(valid):
            first_idx = int(np.argmax(valid))

    first_time = float(fc_times[first_idx]) if first_idx >= 0 else float("nan")
    first_miss = float(miss_distances[first_idx]) if first_idx >= 0 else float("nan")
    first_pat = float(pat_diams[first_idx]) if first_idx >= 0 else float("nan")
    first_hit = bool(would_hit[first_idx]) if first_idx >= 0 else False

    return {
        "miss_distances": miss_distances,
        "pattern_diameters": pat_diams,
        "would_hit": would_hit,
        "first_shot_idx": first_idx,
        "first_shot_time": first_time,
        "first_shot_miss": first_miss,
        "first_shot_pattern": first_pat,
        "first_shot_hit": first_hit,
    }


# -----------------------------------------------------------------------
# Threat prioritisation (multi-target)
# -----------------------------------------------------------------------

def prioritize_threats(
    tracks: list[dict],
    weapon_pos: np.ndarray | tuple[float, float],
    time_idx: int = -1,
    w_range: float = 1.0,
    w_closing: float = 2.0,
    w_quality: float = 0.5,
) -> list[dict]:
    """Score and rank tracked targets by threat priority.

    ``priority = w_range/range + w_closing * max(closing_speed, 0)
                 + w_quality / position_uncertainty``

    Parameters
    ----------
    tracks : list of track dicts (from MultiTargetTracker).
    weapon_pos : (2,)
    time_idx : int
        Which time step to evaluate (-1 = latest).
    w_range, w_closing, w_quality : float
        Weighting coefficients.

    Returns
    -------
    list of track dicts augmented with ``priority_score``, ``range``,
    ``closing_speed``, ``position_uncertainty``, sorted by priority
    (highest first).
    """
    wp = np.asarray(weapon_pos, dtype=np.float64)
    scored = []

    for tr in tracks:
        pos = tr["positions"][time_idx]
        vel = tr["velocities"][time_idx]
        cov = tr["covariances"][time_idx]

        if np.any(np.isnan(pos)):
            continue

        rng = float(np.linalg.norm(pos - wp))
        if rng < 1.0:
            rng = 1.0

        # Closing speed (positive = approaching).
        los = wp - pos
        los_norm = los / max(np.linalg.norm(los), 1e-6)
        closing = float(np.dot(vel, los_norm))

        # Position uncertainty.
        pos_cov = cov[:2, :2]
        eigvals = np.linalg.eigvalsh(pos_cov)
        sigma = math.sqrt(max(eigvals.max(), 0.0))
        pos_unc = max(2.0 * sigma, 0.1)

        score = (w_range / rng
                 + w_closing * max(closing, 0.0)
                 + w_quality / pos_unc)

        entry = dict(tr)  # shallow copy
        entry["priority_score"] = score
        entry["range"] = rng
        entry["closing_speed"] = closing
        entry["position_uncertainty"] = pos_unc
        scored.append(entry)

    scored.sort(key=lambda t: t["priority_score"], reverse=True)
    return scored


def run_multi_fire_control(
    tracks: list[dict],
    *,
    weapon_position: tuple[float, float] = (0.0, 0.0),
    muzzle_velocity: float = 400.0,
    pellet_decel: float = 1.5,
    pattern_spread_rate: float = 0.025,
    max_iterations: int = 5,
    w_range: float = 1.0,
    w_closing: float = 2.0,
    w_quality: float = 0.5,
) -> list[dict]:
    """Fire-control solution for multiple targets, sorted by threat priority.

    Returns list of dicts, one per track, each containing the track dict
    augmented with ``fire_control`` and priority fields.
    """
    wp = np.asarray(weapon_position, dtype=np.float64)

    prioritized = prioritize_threats(
        tracks, wp, time_idx=-1,
        w_range=w_range, w_closing=w_closing, w_quality=w_quality,
    )

    results = []
    for tr in prioritized:
        fc = run_fire_control(
            tr,
            weapon_position=weapon_position,
            muzzle_velocity=muzzle_velocity,
            pellet_decel=pellet_decel,
            pattern_spread_rate=pattern_spread_rate,
            max_iterations=max_iterations,
        )
        entry = dict(tr)
        entry["fire_control"] = fc
        results.append(entry)

    return results


# ═════════════════════════════════════════════════════════════════════
# Merged from fire_control_3d
# ═════════════════════════════════════════════════════════════════════



# -----------------------------------------------------------------------
# 3D Lead angle (iterative)
# -----------------------------------------------------------------------

def compute_lead_3d(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    weapon_pos: np.ndarray,
    muzzle_velocity: float = 400.0,
    decel: float = 1.5,
    max_iter: int = 5,
) -> dict:
    """Compute iterative lead-angle solution in 3D.

    Parameters
    ----------
    target_pos : (3,) — [x, y, z]
    target_vel : (3,) — [vx, vy, vz]
    weapon_pos : (3,) — [x, y, z]

    Returns
    -------
    dict with aim_bearing, aim_elevation, lead_angle_az, lead_angle_el,
    intercept_pos (3,), tof, converged.
    """
    tp = np.asarray(target_pos, dtype=np.float64)[:3]
    tv = np.asarray(target_vel, dtype=np.float64)[:3]
    wp = np.asarray(weapon_pos, dtype=np.float64)[:3]

    direct = tp - wp
    direct_range = float(np.linalg.norm(direct))
    direct_bearing = math.atan2(direct[1], direct[0])
    direct_elevation = math.atan2(direct[2], math.sqrt(direct[0] ** 2 + direct[1] ** 2))

    tof = time_of_flight(direct_range, muzzle_velocity, decel)
    intercept = tp.copy()
    converged = False

    for _ in range(max_iter):
        if tof == float("inf") or np.isnan(tof):
            break
        intercept = tp + tv * tof
        new_range = float(np.linalg.norm(intercept - wp))
        new_tof = time_of_flight(new_range, muzzle_velocity, decel)
        if abs(new_tof - tof) < 1e-6:
            converged = True
            tof = new_tof
            break
        tof = new_tof

    aim_dir = intercept - wp
    aim_bearing = math.atan2(aim_dir[1], aim_dir[0])
    aim_horiz = math.sqrt(aim_dir[0] ** 2 + aim_dir[1] ** 2)
    aim_elevation = math.atan2(aim_dir[2], aim_horiz)

    lead_angle_az = aim_bearing - direct_bearing
    lead_angle_az = (lead_angle_az + math.pi) % (2 * math.pi) - math.pi
    lead_angle_el = aim_elevation - direct_elevation

    return {
        "aim_bearing": aim_bearing,
        "aim_elevation": aim_elevation,
        "lead_angle": lead_angle_az,  # backward compat
        "lead_angle_az": lead_angle_az,
        "lead_angle_el": lead_angle_el,
        "intercept_pos": intercept,
        "tof": tof,
        "converged": converged,
    }


# -----------------------------------------------------------------------
# 3D Engagement envelope
# -----------------------------------------------------------------------

def compute_engagement_3d(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_cov: np.ndarray,
    weapon_pos: np.ndarray,
    muzzle_velocity: float = 400.0,
    decel: float = 1.5,
    spread_rate: float = 0.025,
    max_iter: int = 5,
    max_position_uncertainty: float = 0.0,
    max_engagement_range: float = 0.0,
    class_label: str = "unknown",
    class_confidence: float = 0.0,
    confidence_threshold: float = 0.7,
    maneuver_class: str = "steady",
) -> dict:
    """Determine whether engagement is feasible in 3D.

    Parameters
    ----------
    target_cov : (6, 6) or larger — 3D covariance.
    class_label : str — source classification.
    class_confidence : float — classification confidence.
    maneuver_class : str — current maneuver state.
    """
    tp = np.asarray(target_pos, dtype=np.float64)[:3]
    tv = np.asarray(target_vel, dtype=np.float64)[:3]
    wp = np.asarray(weapon_pos, dtype=np.float64)[:3]

    current_range = float(np.linalg.norm(tp - wp))

    # Position uncertainty (3D: 3×3 position block).
    pos_cov = np.asarray(target_cov)[:3, :3]
    eigvals = np.linalg.eigvalsh(pos_cov)
    sigma_max = math.sqrt(max(eigvals.max(), 0.0))
    pos_unc = 2.0 * sigma_max

    lead = compute_lead_3d(tp, tv, wp, muzzle_velocity, decel, max_iter)
    tof = lead["tof"]
    intercept = lead["intercept_pos"]
    intercept_range = float(np.linalg.norm(intercept - wp))
    pat_diam = pattern_diameter(intercept_range, spread_rate)
    v_pellet = pellet_velocity_at_range(intercept_range, muzzle_velocity, decel)

    # Crossing speed (3D perpendicular component).
    los = tp - wp
    los_norm = los / max(np.linalg.norm(los), 1e-6)
    v_along = float(np.dot(tv, los_norm))
    crossing = float(np.linalg.norm(tv - v_along * los_norm))

    # Decision logic.
    can_fire = True
    reason = "FIRE"

    # Class-based engagement rules.
    threat_classes = {"quadcopter", "hexacopter", "fixed_wing"}
    non_threat_classes = {"bird", "ground_vehicle", "unknown"}

    if class_label in non_threat_classes:
        can_fire = False
        reason = "NON_THREAT"
    elif class_confidence < confidence_threshold and class_label != "unknown":
        can_fire = False
        reason = "LOW_CONFIDENCE"

    # Maneuver-based rules.
    if can_fire and maneuver_class == "evasive":
        # Increase uncertainty threshold during evasive maneuvers.
        if pos_unc > pat_diam * 3.0:
            can_fire = False
            reason = "EVASIVE_UNCERTAINTY"

    if can_fire:
        if tof == float("inf") or v_pellet <= 0:
            can_fire = False
            reason = "OUT_OF_RANGE"
        elif max_engagement_range > 0 and current_range > max_engagement_range:
            can_fire = False
            reason = "TOO_FAR"
        elif max_position_uncertainty > 0 and pos_unc > max_position_uncertainty:
            can_fire = False
            reason = "UNCERTAINTY_TOO_HIGH"
        elif pat_diam < pos_unc:
            can_fire = False
            reason = "UNCERTAINTY_TOO_HIGH"
        elif intercept_range > muzzle_velocity / decel:
            can_fire = False
            reason = "MAX_RANGE_EXCEEDED"

    return {
        "can_fire": can_fire,
        "range": current_range,
        "pattern_diam": pat_diam,
        "position_uncertainty": pos_unc,
        "crossing_speed": crossing,
        "reason": reason,
    }


# -----------------------------------------------------------------------
# Full 3D fire-control runner
# -----------------------------------------------------------------------

def run_fire_control_3d(
    track: dict,
    *,
    weapon_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    muzzle_velocity: float = 400.0,
    pellet_decel: float = 1.5,
    pattern_spread_rate: float = 0.025,
    max_iterations: int = 5,
    max_hits: int = 0,
    hit_threshold: float = 3.0,
    ground_truth_fn: callable | None = None,
    max_position_uncertainty: float = 0.0,
    max_engagement_range: float = 0.0,
    class_label: str = "unknown",
    class_confidence: float = 0.0,
    confidence_threshold: float = 0.7,
    maneuver_class: str = "steady",
) -> dict:
    """Compute 3D fire-control solution at every tracked time step."""
    wp = np.array(weapon_position[:3], dtype=np.float64)
    if len(wp) == 2:
        wp = np.array([wp[0], wp[1], 0.0])
    N = len(track["times"])

    aim_bearings = np.zeros(N)
    aim_elevations = np.zeros(N)
    lead_angles = np.zeros(N)
    lead_angles_el = np.zeros(N)
    tofs = np.zeros(N)
    can_fire = np.zeros(N, dtype=bool)
    ranges = np.zeros(N)
    intercepts = np.zeros((N, 3))
    reasons: list[str] = []

    hits = 0

    for i in range(N):
        pos = track["positions"][i]
        vel = track["velocities"][i]
        cov = track["covariances"][i]

        if np.any(np.isnan(pos)):
            aim_bearings[i] = float("nan")
            aim_elevations[i] = float("nan")
            lead_angles[i] = float("nan")
            lead_angles_el[i] = float("nan")
            tofs[i] = float("nan")
            can_fire[i] = False
            ranges[i] = float("nan")
            intercepts[i] = [float("nan")] * 3
            reasons.append("NO_TRACK")
            continue

        lead = compute_lead_3d(pos, vel, wp, muzzle_velocity, pellet_decel,
                               max_iterations)
        eng = compute_engagement_3d(pos, vel, cov, wp, muzzle_velocity,
                                     pellet_decel, pattern_spread_rate,
                                     max_iterations, max_position_uncertainty,
                                     max_engagement_range,
                                     class_label, class_confidence,
                                     confidence_threshold, maneuver_class)

        aim_bearings[i] = lead["aim_bearing"]
        aim_elevations[i] = lead["aim_elevation"]
        lead_angles[i] = lead["lead_angle"]
        lead_angles_el[i] = lead["lead_angle_el"]
        tofs[i] = lead["tof"]
        ranges[i] = eng["range"]
        intercepts[i] = lead["intercept_pos"]

        if max_hits > 0 and hits >= max_hits:
            can_fire[i] = False
            reasons.append("TARGET_ENGAGED")
        else:
            can_fire[i] = eng["can_fire"]
            reasons.append(eng["reason"])
            if eng["can_fire"] and ground_truth_fn is not None:
                t = track["times"][i]
                gt = ground_truth_fn(t)
                gt = np.asarray(gt, dtype=np.float64)
                ix = lead["intercept_pos"]
                miss = float(np.linalg.norm(ix[:len(gt)] - gt[:len(ix)]))
                if miss < hit_threshold:
                    hits += 1

    return {
        "times": track["times"].copy(),
        "aim_bearings": aim_bearings,
        "aim_elevations": aim_elevations,
        "lead_angles": lead_angles,
        "lead_angles_el": lead_angles_el,
        "tofs": tofs,
        "can_fire": can_fire,
        "ranges": ranges,
        "intercept_positions": intercepts,
        "reasons": reasons,
    }


# -----------------------------------------------------------------------
# 3D Miss distance
# -----------------------------------------------------------------------

def compute_miss_distance_3d(
    fire_control: dict,
    true_positions: np.ndarray,
    true_times: np.ndarray,
    weapon_position: tuple | np.ndarray = (0.0, 0.0, 0.0),
    pattern_spread_rate: float = 0.025,
) -> dict:
    """Compute 3D miss distances."""
    fc_times = fire_control["times"]
    tofs = fire_control["tofs"]
    intercepts = fire_control["intercept_positions"]
    can_fire = fire_control["can_fire"]
    N = len(fc_times)

    true_pos = np.asarray(true_positions)
    if true_pos.shape[1] == 2:
        true_pos = np.column_stack([true_pos, np.zeros(true_pos.shape[0])])

    miss_distances = np.full(N, np.nan)
    pat_diams = np.full(N, np.nan)
    would_hit = np.zeros(N, dtype=bool)

    wp = np.asarray(weapon_position, dtype=np.float64)
    if len(wp) == 2:
        wp = np.array([wp[0], wp[1], 0.0])

    for i in range(N):
        if np.isnan(tofs[i]) or np.any(np.isnan(intercepts[i])):
            continue

        t_impact = fc_times[i] + tofs[i]
        if t_impact < true_times[0] or t_impact > true_times[-1]:
            continue

        true_x = float(np.interp(t_impact, true_times, true_pos[:, 0]))
        true_y = float(np.interp(t_impact, true_times, true_pos[:, 1]))
        true_z = float(np.interp(t_impact, true_times, true_pos[:, 2]))

        miss = math.sqrt(
            (intercepts[i, 0] - true_x) ** 2
            + (intercepts[i, 1] - true_y) ** 2
            + (intercepts[i, 2] - true_z) ** 2
        )
        miss_distances[i] = miss

        int_range = float(np.linalg.norm(intercepts[i] - wp))
        pd = pattern_diameter(int_range, pattern_spread_rate)
        pat_diams[i] = pd
        would_hit[i] = miss < pd / 2.0

    first_idx = -1
    if np.any(can_fire):
        first_idx = int(np.argmax(can_fire))
    else:
        valid = ~np.isnan(miss_distances)
        if np.any(valid):
            first_idx = int(np.argmax(valid))

    first_time = float(fc_times[first_idx]) if first_idx >= 0 else float("nan")
    first_miss = float(miss_distances[first_idx]) if first_idx >= 0 else float("nan")
    first_pat = float(pat_diams[first_idx]) if first_idx >= 0 else float("nan")
    first_hit = bool(would_hit[first_idx]) if first_idx >= 0 else False

    return {
        "miss_distances": miss_distances,
        "pattern_diameters": pat_diams,
        "would_hit": would_hit,
        "first_shot_idx": first_idx,
        "first_shot_time": first_time,
        "first_shot_miss": first_miss,
        "first_shot_pattern": first_pat,
        "first_shot_hit": first_hit,
    }


# -----------------------------------------------------------------------
# 3D Threat prioritisation
# -----------------------------------------------------------------------

def prioritize_threats_3d(
    tracks: list[dict],
    weapon_pos: np.ndarray | tuple,
    time_idx: int = -1,
    w_range: float = 1.0,
    w_closing: float = 2.0,
    w_quality: float = 0.5,
) -> list[dict]:
    """Score and rank tracked targets by threat priority (3D)."""
    wp = np.asarray(weapon_pos, dtype=np.float64)
    if len(wp) == 2:
        wp = np.array([wp[0], wp[1], 0.0])
    scored = []

    for tr in tracks:
        pos = tr["positions"][time_idx]
        vel = tr["velocities"][time_idx]
        cov = tr["covariances"][time_idx]

        if np.any(np.isnan(pos)):
            continue

        rng = float(np.linalg.norm(pos[:3] - wp[:3]))
        if rng < 1.0:
            rng = 1.0

        los = wp[:3] - pos[:3]
        los_norm = los / max(np.linalg.norm(los), 1e-6)
        closing = float(np.dot(vel[:3], los_norm))

        pos_cov = cov[:3, :3]
        eigvals = np.linalg.eigvalsh(pos_cov)
        sigma = math.sqrt(max(eigvals.max(), 0.0))
        pos_unc = max(2.0 * sigma, 0.1)

        score = (w_range / rng
                 + w_closing * max(closing, 0.0)
                 + w_quality / pos_unc)

        entry = dict(tr)
        entry["priority_score"] = score
        entry["range"] = rng
        entry["closing_speed"] = closing
        entry["position_uncertainty"] = pos_unc
        scored.append(entry)

    scored.sort(key=lambda t: t["priority_score"], reverse=True)
    return scored

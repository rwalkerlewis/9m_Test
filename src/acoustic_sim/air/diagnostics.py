"""Air diagnostics: processing steps that read the truth, and say so.

The SRP-PHAT engagement pipeline (``examples/run_pipeline.py``) receives an
:class:`~acoustic_sim.contract.Observation`.  Three of its set-up steps have no blind counterpart yet
and read the :class:`~acoustic_sim.contract.Truth` through this module; each adds a flag to the
pipeline's :class:`~acoustic_sim.contract.Provenance`:

* ``"range_calibration"``: the RMS range estimator is calibrated with the true source distance at
  the time of the recording's peak RMS (:func:`oracle_cpa_distance`);
* ``"range_method"``: ``ranging.method = "auto"`` picks TDOA or RMS from that same true distance;
* ``"altitude"``: with ``source.altitude_estimate_m = null`` the source altitude is the scenario's.

The truth trajectory itself (:func:`truth_trajectory`) is otherwise used only for scoring
(``evaluate_results``, hit scoring) and plotting.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from acoustic_sim.contract import Truth


def truth_trajectory(truth: Truth, source_speed: float):
    """``(trajectory, duration)``: the true source position ``trajectory(t) -> (x, y, z)``.

    From ``truth.scenario`` (the simulation metadata): a trajectory file (``source_trajectory_file``,
    relative to ``truth.meta["sim_dir"]``) when there is one, else the straight or arced path from
    ``source_x, source_y, source_z`` to ``source_x1, ...`` at *source_speed*.
    """
    metadata = truth.scenario
    sim_dir = truth.meta.get("sim_dir")
    sim_dir = Path(sim_dir) if sim_dir is not None else None
    # -- Erratic trajectory from file --
    traj_file = metadata.get("source_trajectory_file")
    if traj_file and sim_dir is not None:
        traj_path = sim_dir / traj_file
        if traj_path.exists():
            traj = np.load(str(traj_path))
            dt_sim = metadata["dt"]
            duration = (traj.shape[0] - 1) * dt_sim

            def trajectory(t: float) -> tuple[float, float, float]:
                frac = min(max(t / duration, 0.0), 1.0) if duration > 0 else 0.0
                idx_f = frac * (traj.shape[0] - 1)
                i0 = int(idx_f)
                i1 = min(i0 + 1, traj.shape[0] - 1)
                w = idx_f - i0
                pos = traj[i0] * (1.0 - w) + traj[i1] * w
                return float(pos[0]), float(pos[1]), float(pos[2])

            return trajectory, duration

    start_x = metadata.get("source_x", -40.0)
    start_y = metadata.get("source_y", 0.0)
    start_z = metadata.get("source_z", 0.0)
    end_x = metadata.get("source_x1", -start_x)
    end_y = metadata.get("source_y1", start_y)
    end_z = metadata.get("source_z1", start_z)

    has_z = "source_z" in metadata
    arc_height = metadata.get("source_arc_height", 0.0)
    horiz_dist = math.hypot(end_x - start_x, end_y - start_y)
    duration = horiz_dist / source_speed if source_speed > 0 else 3.0

    def trajectory(t: float) -> tuple[float, float, float]:
        frac = min(max(t / duration, 0.0), 1.0)
        x = start_x + (end_x - start_x) * frac
        if has_z:
            y = (start_y + (end_y - start_y) * frac
                 + arc_height * 4.0 * frac * (1.0 - frac))
        else:
            y_base = start_y + (end_y - start_y) * frac
            y = y_base + arc_height * math.sin(math.pi * frac)
        z = start_z + (end_z - start_z) * frac
        return x, y, z

    return trajectory, duration


def oracle_cpa_distance(truth: Truth, source_speed: float, t: float, array_center: np.ndarray) -> float:
    """True source distance [m] from the array centre at time *t* (at least 1 m): an oracle input
    (``"range_calibration"``, and ``"range_method"`` when it chooses the range estimator)."""
    trajectory, _ = truth_trajectory(truth, source_speed)
    return max(float(np.linalg.norm(np.array(trajectory(t)) - array_center)), 1.0)


def oracle_altitude(truth: Truth) -> float | None:
    """The scenario's source altitude, or ``None`` for a 2-D scenario: an oracle input (``"altitude"``)."""
    z = truth.scenario.get("source_z")
    return None if z is None else float(z)

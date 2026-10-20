"""Where the FDTD samples the field for a sensor layout (forward side).

The layout itself (cables, hydrophones, the strain gauge sample points of the instrument model and
:meth:`~acoustic_sim.water.sensors.SensorLayout.collapse`) is processing side, in ``water/sensors.py``:
replicas need the same three points per strain channel the solver samples.  What only the forward
model needs is here: the check that every sample point is in the solver's water (:func:`validate_layout`)
and the hash of the sample points that keys the FDTD gather cache (:func:`layout_fingerprint`).
"""

from __future__ import annotations

import hashlib

import numpy as np

from acoustic_sim.water.sensors import SensorLayout


def fdtd_sample_points(layout: SensorLayout) -> np.ndarray:
    """Points at which the FDTD records the field, ``(n_samples, 2)``: every cable's sample points
    (three per strain channel), then the hydrophones; the solver's output rows are in this order."""
    return layout.sample_points


def validate_layout(layout: SensorLayout, env, margin: float = 10.0) -> None:
    """Raise if any sampling point is on land or closer than *margin* m to it."""
    pts = fdtd_sample_points(layout)
    ok = env.is_water(pts, margin=margin)
    if not np.all(ok):
        bad = np.flatnonzero(~ok)
        raise ValueError(
            f"{bad.size} receiver sample point(s) on land or within {margin} m of shore, "
            f"first at index {bad[0]} {pts[bad[0]]}"
        )


def layout_fingerprint(layout: SensorLayout) -> str:
    """Hash of everything that affects the sampled traces (unchanged for pressure layouts; the FDTD
    cache key)."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(layout.receivers, dtype=np.float64).tobytes())
    if layout.has_strain:
        h.update(b"strain")
        h.update(np.ascontiguousarray(fdtd_sample_points(layout), dtype=np.float64).tobytes())
    return h.hexdigest()

"""Receiver (microphone array) geometry helpers.

Provides factory functions for common array layouts used in acoustic
beamforming and matched field processing:

* **Nested circular** — ``create_receiver_nested_circular`` (default)
* **Linear** — ``create_receiver_line``
* **Circular** — ``create_receiver_circle``
* **Concentric** — ``create_receiver_concentric``
* **L-shaped** — ``create_receiver_l_shaped``
* **Logarithmic spiral** — ``create_receiver_log_spiral``
* **Random within disk** — ``create_receiver_random_disk``
* **Random within box** — ``create_receiver_random``
* **Custom** — ``create_receiver_custom``

All geometry functions return an ``(N, 2)`` numpy array of (x, y)
positions in metres, measured from the array centre.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# -----------------------------------------------------------------------
# Core geometries
# -----------------------------------------------------------------------

def create_receiver_nested_circular(
    cx: float = 0.0,
    cy: float = 0.0,
    inner_radius: float = 0.15,
    outer_radius: float = 0.50,
    n_inner: int = 4,
    n_outer: int = 8,
) -> np.ndarray:
    """Nested circular array: 1 centre + inner ring + outer ring.

    Default: 13 elements (1 + 4 + 8) optimised for broadband MFP on a
    compact array.  The two radii create non-redundant cross-baselines
    that suppress spatial aliasing in the broadband sum.

    Returns shape ``(1 + n_inner + n_outer, 2)``.
    """
    pts = [[cx, cy]]  # centre element
    for i in range(n_inner):
        angle = 2 * np.pi * i / n_inner
        pts.append([cx + inner_radius * np.cos(angle),
                     cy + inner_radius * np.sin(angle)])
    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer
        pts.append([cx + outer_radius * np.cos(angle),
                     cy + outer_radius * np.sin(angle)])
    return np.array(pts, dtype=np.float64)


def create_receiver_line(
    x_start: float, y_start: float,
    x_end: float, y_end: float,
    count: int,
) -> np.ndarray:
    """Line of receivers.  Returns shape ``(count, 2)``."""
    return np.column_stack([
        np.linspace(x_start, x_end, count),
        np.linspace(y_start, y_end, count),
    ])


def create_receiver_circle(
    cx: float, cy: float,
    radius: float, count: int,
) -> np.ndarray:
    """Single-ring circular array.  Returns shape ``(count, 2)``."""
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
    ])


def create_receiver_concentric(
    cx: float, cy: float,
    radii: Sequence[float],
    counts_per_ring: int | Sequence[int] = 12,
) -> np.ndarray:
    """Concentric rings.  Returns ``(N_total, 2)``."""
    if isinstance(counts_per_ring, int):
        counts_per_ring = [counts_per_ring] * len(radii)
    rings = [create_receiver_circle(cx, cy, r, n)
             for r, n in zip(radii, counts_per_ring)]
    return np.vstack(rings)


def create_receiver_l_shaped(
    n1: int, n2: int, spacing: float,
    origin_x: float = 0.0, origin_y: float = 0.0,
) -> np.ndarray:
    """L-shaped array.  Returns ``(n1 + n2 - 1, 2)``."""
    x_arm = np.column_stack([
        origin_x + np.arange(n1) * spacing,
        np.full(n1, origin_y),
    ])
    y_arm = np.column_stack([
        np.full(n2 - 1, origin_x),
        origin_y + np.arange(1, n2) * spacing,
    ])
    return np.vstack([x_arm, y_arm])


def create_receiver_log_spiral(
    count: int = 13,
    radius: float = 0.50,
    cx: float = 0.0,
    cy: float = 0.0,
) -> np.ndarray:
    """Logarithmic (golden-angle) spiral for maximum baseline diversity.

    Element i sits at radius ``r_max * (i / N)`` and angle
    ``golden_angle * i``.  The centre element is at (cx, cy).

    Returns shape ``(count, 2)``.
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~137.5°
    pts = [[cx, cy]]
    for i in range(1, count):
        r = radius * (i / count)
        theta = golden_angle * i
        pts.append([cx + r * np.cos(theta), cy + r * np.sin(theta)])
    return np.array(pts, dtype=np.float64)


def create_receiver_random_disk(
    count: int = 13,
    radius: float = 0.50,
    cx: float = 0.0,
    cy: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Random positions within a disk.  Returns ``(count, 2)``."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < count:
        x = rng.uniform(-radius, radius)
        y = rng.uniform(-radius, radius)
        if x * x + y * y <= radius * radius:
            pts.append([cx + x, cy + y])
    return np.array(pts, dtype=np.float64)


def create_receiver_random(
    count: int, x_min: float, x_max: float,
    y_min: float, y_max: float, seed: int = 42,
) -> np.ndarray:
    """Random positions within a bounding box.  Returns ``(count, 2)``."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(x_min, x_max, count),
        rng.uniform(y_min, y_max, count),
    ])


def create_receiver_custom(
    positions: list[tuple[float, float]] | np.ndarray,
) -> np.ndarray:
    """Wrap a list of ``(x, y)`` tuples as an ``(N, 2)`` array."""
    return np.asarray(positions, dtype=np.float64).reshape(-1, 2)


# -----------------------------------------------------------------------
# Array diagnostics
# -----------------------------------------------------------------------

def print_array_diagnostics(
    positions: np.ndarray,
    sound_speed: float = 343.0,
) -> dict:
    """Print and return a diagnostic summary of array geometry.

    Computes all unique baseline lengths, spatial aliasing frequencies,
    and diffraction-limited resolution at key frequencies.

    Parameters
    ----------
    positions : (N, 2)
    sound_speed : float

    Returns
    -------
    dict with diagnostic values.
    """
    n = positions.shape[0]
    n_baselines = n * (n - 1) // 2

    # All unique baselines.
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            dists.append(d)
    dists = np.array(dists)

    d_min = float(np.min(dists))
    d_max = float(np.max(dists))
    f_alias_min = sound_speed / (2 * d_min) if d_min > 0 else np.inf
    f_alias_max = sound_speed / (2 * d_max) if d_max > 0 else np.inf

    diag_freqs = [300, 600, 1000, 1500]
    resolutions = {}
    for f in diag_freqs:
        lam = sound_speed / f
        res_deg = np.degrees(lam / d_max) if d_max > 0 else 360.0
        resolutions[f] = res_deg

    print(f"\n  Array Diagnostics")
    print(f"  {'='*50}")
    print(f"  Elements:               {n}")
    print(f"  Unique baselines:       {n_baselines}")
    print(f"  Min baseline:           {d_min:.4f} m")
    print(f"  Max baseline:           {d_max:.4f} m")
    print(f"  Alias freq (min BL):    {f_alias_min:.0f} Hz")
    print(f"  Alias freq (max BL):    {f_alias_max:.0f} Hz")
    for f, res in resolutions.items():
        print(f"  Resolution at {f:>5d} Hz: {res:>6.1f} deg")
    print()

    return {
        "n_elements": n,
        "n_baselines": n_baselines,
        "min_baseline": d_min,
        "max_baseline": d_max,
        "alias_freq_min_bl": f_alias_min,
        "alias_freq_max_bl": f_alias_max,
        "resolutions": resolutions,
    }


# ═════════════════════════════════════════════════════════════════════
# Merged from receivers_3d
# ═════════════════════════════════════════════════════════════════════



def _to_3d(positions_2d: np.ndarray, z: float | np.ndarray = 0.0) -> np.ndarray:
    """Convert (N, 2) → (N, 3) with given z-coordinate(s)."""
    n = positions_2d.shape[0]
    if isinstance(z, (int, float)):
        z_arr = np.full(n, z, dtype=np.float64)
    else:
        z_arr = np.asarray(z, dtype=np.float64)
        if len(z_arr) != n:
            raise ValueError(f"z array length {len(z_arr)} != n_receivers {n}")
    return np.column_stack([positions_2d, z_arr])


def create_receiver_l_shaped_3d(
    n1: int, n2: int, spacing: float,
    origin_x: float = 0.0, origin_y: float = 0.0,
    z: float | np.ndarray = 0.0,
) -> np.ndarray:
    """L-shaped array in 3D.  Returns ``(n1 + n2 - 1, 3)``."""
    pos_2d = create_receiver_l_shaped(n1, n2, spacing, origin_x, origin_y)
    return _to_3d(pos_2d, z)


def create_receiver_circle_3d(
    cx: float, cy: float, radius: float, count: int,
    z: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Circular array in 3D.  Returns ``(count, 3)``."""
    pos_2d = create_receiver_circle(cx, cy, radius, count)
    return _to_3d(pos_2d, z)


def create_receiver_nested_circular_3d(
    cx: float = 0.0, cy: float = 0.0,
    inner_radius: float = 0.15, outer_radius: float = 0.50,
    n_inner: int = 4, n_outer: int = 8,
    z: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Nested circular array in 3D."""
    pos_2d = create_receiver_nested_circular(cx, cy, inner_radius, outer_radius,
                                              n_inner, n_outer)
    return _to_3d(pos_2d, z)


def create_receiver_line_3d(
    x_start: float, y_start: float, x_end: float, y_end: float,
    count: int, z: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Line of receivers in 3D."""
    pos_2d = create_receiver_line(x_start, y_start, x_end, y_end, count)
    return _to_3d(pos_2d, z)


def create_receiver_random_disk_3d(
    count: int = 13, radius: float = 0.50,
    cx: float = 0.0, cy: float = 0.0,
    z: float | np.ndarray = 0.0, seed: int = 42,
) -> np.ndarray:
    """Random positions within a disk in 3D."""
    pos_2d = create_receiver_random_disk(count, radius, cx, cy, seed)
    return _to_3d(pos_2d, z)


def create_receiver_custom_3d(
    positions: list[tuple[float, float, float]] | np.ndarray,
) -> np.ndarray:
    """Wrap a list of ``(x, y, z)`` tuples as an ``(N, 3)`` array."""
    return np.asarray(positions, dtype=np.float64).reshape(-1, 3)


# ═════════════════════════════════════════════════════════════════════
# Elastic / seismic receiver support (microphones + geophones)
# ═════════════════════════════════════════════════════════════════════

from dataclasses import dataclass


@dataclass
class ReceiverSpec:
    """Combined receiver specification with sensor type metadata.

    Attributes
    ----------
    positions : np.ndarray
        Receiver positions, shape ``(N, 2)`` (2-D) or ``(N, 3)`` (3-D).
    sensor_types : list[str]
        ``"microphone"`` or ``"geophone"`` per receiver.
    field_components : list[str]
        ``"pressure"`` for microphones, ``"vz"`` for geophones.
    """

    positions: np.ndarray
    sensor_types: list[str]
    field_components: list[str]

    @property
    def n_receivers(self) -> int:
        return self.positions.shape[0]

    @property
    def n_microphones(self) -> int:
        return sum(1 for s in self.sensor_types if s == "microphone")

    @property
    def n_geophones(self) -> int:
        return sum(1 for s in self.sensor_types if s == "geophone")

    @property
    def mic_indices(self) -> np.ndarray:
        """Indices of microphones in the positions array."""
        return np.array(
            [i for i, s in enumerate(self.sensor_types) if s == "microphone"],
            dtype=int,
        )

    @property
    def geo_indices(self) -> np.ndarray:
        """Indices of geophones in the positions array."""
        return np.array(
            [i for i, s in enumerate(self.sensor_types) if s == "geophone"],
            dtype=int,
        )


def create_colocated_array_2d(
    cx: float = 0.0,
    radius: float = 2.0,
    count: int = 16,
    mic_z: float = 1.5,
    geo_z: float = -0.05,
) -> ReceiverSpec:
    """Create a colocated microphone + geophone circular array in 2-D.

    Microphones are placed at ``z = mic_z`` and geophones at
    ``z = geo_z`` (just below the surface), sharing the same x positions.

    In 2-D the position array has shape ``(2*count, 2)`` with columns
    ``(x, z)``.

    Parameters
    ----------
    cx : float
        Array centre x-coordinate [m].
    radius : float
        Array radius [m].
    count : int
        Number of elements per ring (same for mics and geophones).
    mic_z : float
        Microphone altitude [m] (positive = above ground).
    geo_z : float
        Geophone depth [m] (negative = below ground).

    Returns
    -------
    ReceiverSpec
    """
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    x_pos = cx + radius * np.cos(angles)

    mic_positions = np.column_stack([x_pos, np.full(count, mic_z)])
    geo_positions = np.column_stack([x_pos, np.full(count, geo_z)])

    positions = np.vstack([mic_positions, geo_positions])
    sensor_types = ["microphone"] * count + ["geophone"] * count
    field_components = ["pressure"] * count + ["vz"] * count

    return ReceiverSpec(
        positions=positions,
        sensor_types=sensor_types,
        field_components=field_components,
    )


def create_colocated_array_3d(
    cx: float = 50.0,
    cy: float = 50.0,
    radius: float = 2.0,
    count: int = 16,
    mic_z: float = 1.5,
    geo_z: float = -0.05,
) -> ReceiverSpec:
    """Create a colocated microphone + geophone circular array in 3-D.

    Microphones at ``z = mic_z``, geophones at ``z = geo_z``, sharing
    the same ``(x, y)`` positions on a circle of given radius.

    Returns
    -------
    ReceiverSpec
        Positions have shape ``(2*count, 3)``.
    """
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    x_pos = cx + radius * np.cos(angles)
    y_pos = cy + radius * np.sin(angles)

    mic_positions = np.column_stack([
        x_pos, y_pos, np.full(count, mic_z),
    ])
    geo_positions = np.column_stack([
        x_pos, y_pos, np.full(count, geo_z),
    ])

    positions = np.vstack([mic_positions, geo_positions])
    sensor_types = ["microphone"] * count + ["geophone"] * count
    field_components = ["pressure"] * count + ["vz"] * count

    return ReceiverSpec(
        positions=positions,
        sensor_types=sensor_types,
        field_components=field_components,
    )

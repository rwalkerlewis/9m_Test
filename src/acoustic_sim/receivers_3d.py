"""3D receiver (microphone array) geometry helpers.

Extends the 2D receiver factories with an optional z-coordinate.
All functions return ``(N, 3)`` arrays.  When z is omitted, default z=0.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from acoustic_sim.receivers import (
    create_receiver_circle,
    create_receiver_concentric,
    create_receiver_custom,
    create_receiver_l_shaped,
    create_receiver_line,
    create_receiver_log_spiral,
    create_receiver_nested_circular,
    create_receiver_random,
    create_receiver_random_disk,
)


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

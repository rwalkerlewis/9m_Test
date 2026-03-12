"""Receiver (microphone array) geometry helpers.

Provides factory functions for common array layouts used in acoustic
beamforming and matched field processing:

* **Linear** — ``create_receiver_line``
* **Circular** — ``create_receiver_circle``
* **Concentric** — ``create_receiver_concentric``
* **L-shaped** — ``create_receiver_l_shaped``
* **Random** — ``create_receiver_random``
* **Custom** — ``create_receiver_custom``

All functions return an ``(N, 2)`` numpy array of (x, y) positions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def create_receiver_line(
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    count: int,
) -> np.ndarray:
    """Line of receivers.  Returns shape ``(count, 2)``."""
    return np.column_stack(
        [
            np.linspace(x_start, x_end, count),
            np.linspace(y_start, y_end, count),
        ]
    )


def create_receiver_circle(
    cx: float,
    cy: float,
    radius: float,
    count: int,
) -> np.ndarray:
    """Circular receiver array.  Returns shape ``(count, 2)``."""
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.column_stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles)]
    )


def create_receiver_concentric(
    cx: float,
    cy: float,
    radii: Sequence[float],
    counts_per_ring: int | Sequence[int] = 12,
) -> np.ndarray:
    """Concentric circular arrays at varying radii.

    Parameters
    ----------
    cx, cy : float
        Centre of the concentric rings.
    radii : sequence of float
        Radius of each ring, e.g. ``[5, 10, 15, 20]``.
    counts_per_ring : int or sequence of int
        Number of receivers per ring.  A single int is used for every ring.

    Returns
    -------
    np.ndarray, shape ``(N_total, 2)``
    """
    if isinstance(counts_per_ring, int):
        counts_per_ring = [counts_per_ring] * len(radii)
    rings = [
        create_receiver_circle(cx, cy, r, n)
        for r, n in zip(radii, counts_per_ring)
    ]
    return np.vstack(rings)


def create_receiver_l_shaped(
    n1: int,
    n2: int,
    spacing: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> np.ndarray:
    """L-shaped array: *n1* elements along +x, *n2* along +y.

    The corner element at ``(origin_x, origin_y)`` is shared (not
    duplicated), so the total count is ``n1 + n2 - 1``.

    Returns shape ``(n1 + n2 - 1, 2)``.
    """
    x_arm = np.column_stack([
        origin_x + np.arange(n1) * spacing,
        np.full(n1, origin_y),
    ])
    y_arm = np.column_stack([
        np.full(n2 - 1, origin_x),
        origin_y + np.arange(1, n2) * spacing,
    ])
    return np.vstack([x_arm, y_arm])


def create_receiver_random(
    count: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    seed: int = 42,
) -> np.ndarray:
    """Random receiver positions within a bounding box.

    Returns shape ``(count, 2)``.
    """
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

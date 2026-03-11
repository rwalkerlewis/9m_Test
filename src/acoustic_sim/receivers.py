"""Receiver geometry helpers."""

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

"""Receiver geometry helpers."""

from __future__ import annotations

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

"""2-D ray tracing through a heterogeneous velocity field."""

from __future__ import annotations

import numpy as np

from acoustic_sim.model import VelocityModel


def trace_rays(
    source_xy: np.ndarray,
    model: VelocityModel,
    ray_count: int = 24,
    max_steps: int = 2000,
    step_size: float | None = None,
) -> list[np.ndarray]:
    """Trace rays through a heterogeneous velocity field.

    Rays bend toward regions of lower velocity via a simple gradient-based
    ray-marching scheme (approximation to the eikonal equation).
    """
    x0, x1 = model.x[0], model.x[-1]
    y0, y1 = model.y[0], model.y[-1]
    if step_size is None:
        step_size = model.dx * 0.5

    eps = model.dx * 0.5
    rays: list[np.ndarray] = []

    for angle in np.linspace(0, 2 * np.pi, ray_count, endpoint=False):
        pos = np.array(source_xy[:2], dtype=np.float64)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        points = [pos.copy()]

        for _ in range(max_steps):
            c = model.velocity_at(pos[0], pos[1])
            dc_dx = (
                model.velocity_at(pos[0] + eps, pos[1])
                - model.velocity_at(pos[0] - eps, pos[1])
            ) / (2 * eps)
            dc_dy = (
                model.velocity_at(pos[0], pos[1] + eps)
                - model.velocity_at(pos[0], pos[1] - eps)
            ) / (2 * eps)

            grad = np.array([dc_dx, dc_dy])
            direction -= (step_size / max(c, 1.0)) * grad
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                break
            direction /= norm

            pos = pos + step_size * direction
            if pos[0] < x0 or pos[0] > x1 or pos[1] < y0 or pos[1] > y1:
                pos[0] = np.clip(pos[0], x0, x1)
                pos[1] = np.clip(pos[1], y0, y1)
                points.append(pos.copy())
                break
            points.append(pos.copy())

        rays.append(np.array(points))
    return rays

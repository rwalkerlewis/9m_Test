"""I/O helpers — JSON config loading, velocity-model persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from acoustic_sim.model import (
    VelocityModel,
    add_circle_anomaly,
    add_rectangle_anomaly,
    create_checkerboard_model,
    create_gradient_model,
    create_layered_model,
    create_uniform_model,
)


def load_json(path: str | None) -> dict[str, Any] | None:
    """Load and return a JSON file, or *None* if *path* is falsy."""
    if not path:
        return None
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON config not found: {path}")
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_model(model: VelocityModel, path: str) -> None:
    """Persist a velocity model to a ``.npz`` file."""
    np.savez(
        path, x=model.x, y=model.y, values=model.values, dx=model.dx, dy=model.dy
    )
    print(f"Saved velocity model to {path}")


def load_model(path: str) -> VelocityModel:
    """Load a velocity model from a ``.npz`` file."""
    data = np.load(path)
    return VelocityModel(
        x=data["x"],
        y=data["y"],
        values=data["values"],
        dx=float(data["dx"]),
        dy=float(data["dy"]),
    )


def model_from_json(cfg: dict[str, Any]) -> VelocityModel:
    """Build a velocity model from a JSON configuration dict."""
    bounds = cfg.get("bounds", {})
    x_min = float(bounds.get("x_min", -20.0))
    x_max = float(bounds.get("x_max", 20.0))
    y_min = float(bounds.get("y_min", -20.0))
    y_max = float(bounds.get("y_max", 20.0))
    dx = float(cfg.get("dx", 0.4))
    background = float(cfg.get("background_velocity", 343.0))
    model_type = cfg.get("type", "uniform")

    if model_type == "layered":
        layers = [
            (float(entry["y"]), float(entry["velocity"]))
            for entry in cfg.get("layers", [])
        ]
        model = create_layered_model(x_min, x_max, y_min, y_max, dx, layers, background)
    elif model_type == "gradient":
        model = create_gradient_model(
            x_min, x_max, y_min, y_max, dx,
            v_bottom=float(cfg.get("v_bottom", 360.0)),
            v_top=float(cfg.get("v_top", 330.0)),
        )
    elif model_type == "checkerboard":
        model = create_checkerboard_model(
            x_min, x_max, y_min, y_max, dx,
            cell_size=float(cfg.get("cell_size", 4.0)),
            v_base=background,
            perturbation=float(cfg.get("perturbation", 20.0)),
        )
    else:
        model = create_uniform_model(x_min, x_max, y_min, y_max, dx, background)

    for anomaly in cfg.get("anomalies", []):
        atype = anomaly.get("type", "circle")
        vel = float(anomaly["velocity"])
        if atype == "circle":
            model = add_circle_anomaly(
                model,
                cx=float(anomaly["center"][0]),
                cy=float(anomaly["center"][1]),
                radius=float(anomaly["radius"]),
                velocity=vel,
            )
        elif atype == "rectangle":
            model = add_rectangle_anomaly(
                model,
                x0=float(anomaly["x_min"]),
                x1=float(anomaly["x_max"]),
                y0=float(anomaly["y_min"]),
                y1=float(anomaly["y_max"]),
                velocity=vel,
            )
    return model

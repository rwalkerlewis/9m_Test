"""acoustic_sim — 2D acoustic simulation on user-defined velocity models."""

from acoustic_sim.model import (
    VelocityModel,
    add_circle_anomaly,
    add_rectangle_anomaly,
    create_checkerboard_model,
    create_gradient_model,
    create_layered_model,
    create_uniform_model,
    create_valley_model,
    model_from_array,
)
from acoustic_sim.sampling import check_cfl, check_spatial_sampling, suggest_dx
from acoustic_sim.solver import solve_helmholtz
from acoustic_sim.receivers import create_receiver_circle, create_receiver_line
from acoustic_sim.io import load_json, load_model, model_from_json, save_model
from acoustic_sim.plotting import plot_velocity_model, plot_wavefield

__all__ = [
    "VelocityModel",
    "add_circle_anomaly",
    "add_rectangle_anomaly",
    "check_cfl",
    "check_spatial_sampling",
    "create_checkerboard_model",
    "create_gradient_model",
    "create_layered_model",
    "create_receiver_circle",
    "create_receiver_line",
    "create_uniform_model",
    "create_valley_model",
    "load_json",
    "load_model",
    "model_from_array",
    "model_from_json",
    "plot_velocity_model",
    "plot_wavefield",
    "save_model",
    "solve_helmholtz",
    "suggest_dx",
]

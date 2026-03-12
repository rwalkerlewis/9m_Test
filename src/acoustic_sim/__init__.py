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
from acoustic_sim.receivers import (
    create_receiver_circle,
    create_receiver_concentric,
    create_receiver_line,
)
from acoustic_sim.io import load_json, load_model, model_from_json, save_model
from acoustic_sim.plotting import (
    plot_domain,
    plot_gather,
    plot_velocity_model,
    plot_wavefield,
    save_snapshot,
)
from acoustic_sim.backend import get_backend
from acoustic_sim.sources import (
    MovingSource,
    StaticSource,
    inject_source,
    load_wav_mono,
    make_source_from_file,
    make_source_noise,
    make_source_propeller,
    make_source_tone,
    make_wavelet_ricker,
    prepare_source_signal,
)
from acoustic_sim.domains import (
    DomainMeta,
    create_hills_vegetation_domain,
    create_isotropic_domain,
    create_wind_domain,
)
from acoustic_sim.fdtd import FDTDConfig, FDTDSolver
from acoustic_sim.setup import build_domain, build_receivers, build_source, compute_dt

__all__ = [
    "DomainMeta",
    "FDTDConfig",
    "FDTDSolver",
    "MovingSource",
    "StaticSource",
    "VelocityModel",
    "add_circle_anomaly",
    "add_rectangle_anomaly",
    "build_domain",
    "build_receivers",
    "build_source",
    "check_cfl",
    "check_spatial_sampling",
    "compute_dt",
    "create_checkerboard_model",
    "create_gradient_model",
    "create_hills_vegetation_domain",
    "create_isotropic_domain",
    "create_layered_model",
    "create_receiver_circle",
    "create_receiver_concentric",
    "create_receiver_line",
    "create_uniform_model",
    "create_valley_model",
    "create_wind_domain",
    "get_backend",
    "inject_source",
    "load_json",
    "load_model",
    "load_wav_mono",
    "make_source_from_file",
    "make_source_noise",
    "make_source_propeller",
    "make_source_tone",
    "make_wavelet_ricker",
    "model_from_array",
    "model_from_json",
    "plot_domain",
    "plot_gather",
    "plot_velocity_model",
    "plot_wavefield",
    "prepare_source_signal",
    "save_model",
    "save_snapshot",
    "solve_helmholtz",
    "suggest_dx",
]

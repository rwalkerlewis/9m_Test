"""Command-line interface for acoustic_sim."""

from __future__ import annotations

import argparse

import numpy as np

from acoustic_sim.io import load_json, load_model, model_from_json, save_model
from acoustic_sim.model import (
    create_checkerboard_model,
    create_gradient_model,
    create_layered_model,
    create_uniform_model,
    create_valley_model,
)
from acoustic_sim.plotting import plot_velocity_model, plot_wavefield
from acoustic_sim.receivers import create_receiver_circle, create_receiver_line
from acoustic_sim.sampling import check_spatial_sampling
from acoustic_sim.solver import solve_helmholtz


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D acoustic simulation on a user-defined velocity model.",
    )

    # --- velocity model source ---
    mg = parser.add_mutually_exclusive_group()
    mg.add_argument(
        "--model-file", type=str, default=None,
        help="JSON config describing the velocity model",
    )
    mg.add_argument(
        "--model-npz", type=str, default=None,
        help="Pre-built velocity model in .npz format (saved with save_model)",
    )
    mg.add_argument(
        "--model-preset",
        choices=["uniform", "layered", "gradient", "checkerboard", "valley"],
        default=None,
        help="Use a built-in velocity model preset",
    )

    # --- domain geometry (used by presets or when no model is given) ---
    parser.add_argument("--x-min", type=float, default=-20.0)
    parser.add_argument("--x-max", type=float, default=20.0)
    parser.add_argument("--y-min", type=float, default=-20.0)
    parser.add_argument("--y-max", type=float, default=20.0)
    parser.add_argument("--dx", type=float, default=0.4, help="Grid spacing [m]")
    parser.add_argument(
        "--bg-velocity", type=float, default=343.0,
        help="Background wave speed [m/s]",
    )

    # --- source ---
    parser.add_argument("--source-x", type=float, default=0.0,
                        help="Source x position [m]")
    parser.add_argument("--source-y", type=float, default=0.0,
                        help="Source y position [m]")

    # --- receivers ---
    parser.add_argument(
        "--receiver-type", choices=["line", "circle"], default="circle",
        help="Receiver geometry",
    )
    parser.add_argument("--receiver-count", type=int, default=16)
    parser.add_argument(
        "--receiver-radius", type=float, default=0.2,
        help="Radius for circular receiver array [m]",
    )
    parser.add_argument("--receiver-x0", type=float, default=-15.0,
                        help="Line receiver x start [m]")
    parser.add_argument("--receiver-y0", type=float, default=0.0,
                        help="Line receiver y start [m]")
    parser.add_argument("--receiver-x1", type=float, default=15.0,
                        help="Line receiver x end [m]")
    parser.add_argument("--receiver-y1", type=float, default=0.0,
                        help="Line receiver y end [m]")

    # --- Helmholtz ---
    parser.add_argument("--frequency", type=float, default=480.0,
                        help="Helmholtz source frequency [Hz]")
    parser.add_argument("--min-ppw", type=float, default=10.0,
                        help="Minimum required points per wavelength")

    # --- valley geometry ---
    parser.add_argument("--hill-south-y", type=float, default=-20.0,
                        help="y centre of southern ridge [m]")
    parser.add_argument("--hill-north-y", type=float, default=20.0,
                        help="y centre of northern ridge [m]")
    parser.add_argument("--hill-peak-height", type=float, default=18.0,
                        help="Maximum ridge height [m]")
    parser.add_argument("--saddle-width", type=float, default=12.0,
                        help="Width of the saddle notch [m]")
    parser.add_argument("--saddle-depth", type=float, default=0.55,
                        help="Saddle depth fraction (0=none, 1=floor)")
    parser.add_argument("--dirt-velocity", type=float, default=1500.0,
                        help="Wave speed inside hills [m/s]")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for hill shapes")

    # --- outputs ---
    parser.add_argument("--velocity-plot", type=str, default="velocity_model.png")
    parser.add_argument("--field-plot", type=str, default="wavefield.png")
    parser.add_argument("--save-model-path", type=str, default=None,
                        help="Save the velocity model to .npz")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ---- build / load velocity model ----
    if args.model_npz:
        model = load_model(args.model_npz)
        print(
            f"Loaded velocity model from {args.model_npz}  "
            f"shape={model.shape}, c=[{model.c_min:.1f}, {model.c_max:.1f}] m/s"
        )
    elif args.model_file:
        cfg = load_json(args.model_file)
        if cfg is None:
            raise FileNotFoundError(args.model_file)
        model = model_from_json(cfg)
        print(
            f"Built velocity model from {args.model_file}  "
            f"shape={model.shape}, c=[{model.c_min:.1f}, {model.c_max:.1f}] m/s"
        )
    else:
        preset = args.model_preset or "uniform"
        if preset == "layered":
            model = create_layered_model(
                args.x_min, args.x_max, args.y_min, args.y_max, args.dx,
                layers=[(-10.0, 360.0), (0.0, 343.0), (10.0, 320.0)],
                background=args.bg_velocity,
            )
        elif preset == "gradient":
            model = create_gradient_model(
                args.x_min, args.x_max, args.y_min, args.y_max, args.dx,
                v_bottom=360.0, v_top=320.0,
            )
        elif preset == "checkerboard":
            model = create_checkerboard_model(
                args.x_min, args.x_max, args.y_min, args.y_max, args.dx,
            )
        elif preset == "valley":
            model = create_valley_model(
                args.x_min, args.x_max, args.y_min, args.y_max, args.dx,
                air_velocity=args.bg_velocity,
                dirt_velocity=args.dirt_velocity,
                seed=args.seed,
                hill_south_y=args.hill_south_y,
                hill_north_y=args.hill_north_y,
                hill_peak_height=args.hill_peak_height,
                saddle_width=args.saddle_width,
                saddle_depth_frac=args.saddle_depth,
            )
        else:
            model = create_uniform_model(
                args.x_min, args.x_max, args.y_min, args.y_max, args.dx,
                velocity=args.bg_velocity,
            )
        print(
            f"Using '{preset}' preset  shape={model.shape}, "
            f"c=[{model.c_min:.1f}, {model.c_max:.1f}] m/s"
        )

    # ---- spatial-sampling check ----
    sampling = check_spatial_sampling(model, args.frequency, args.min_ppw)
    print(f"Spatial sampling: {sampling['message']}")
    if not sampling["valid"]:
        print("WARNING: insufficient spatial sampling — results may be inaccurate.")

    # ---- receivers ----
    source_xy = np.array([args.source_x, args.source_y], dtype=np.float64)
    if args.receiver_type == "circle":
        receivers = create_receiver_circle(
            args.source_x, args.source_y,
            args.receiver_radius, args.receiver_count,
        )
    else:
        receivers = create_receiver_line(
            args.receiver_x0, args.receiver_y0,
            args.receiver_x1, args.receiver_y1,
            args.receiver_count,
        )
    print(f"Receivers: {receivers.shape[0]} in '{args.receiver_type}' layout")

    # ---- velocity model plot ----
    plot_velocity_model(
        model, output_path=args.velocity_plot,
        receivers=receivers, source_xy=source_xy,
    )

    # ---- Helmholtz solve ----
    print(f"Solving Helmholtz at {args.frequency:.1f} Hz …")
    field = solve_helmholtz(model, source_xy, args.frequency)

    # ---- wavefield plot ----
    plot_wavefield(
        model, field, output_path=args.field_plot,
        receivers=receivers, source_xy=source_xy,
    )

    # ---- optional model save ----
    if args.save_model_path:
        save_model(model, args.save_model_path)

    print("Done.")

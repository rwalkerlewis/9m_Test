#!/usr/bin/env python3
"""Run a single 3-D FDTD acoustic simulation.

Usage::

    python examples/run_fdtd_3d.py --domain hills_vegetation \\
        --source-type moving --source-signal propeller \\
        --array circular --output-dir output/valley_3d

This is the 3-D counterpart of ``run_fdtd.py``.  It uses
:class:`~acoustic_sim.fdtd_3d.FDTD3DSolver` and all 3-D domain /
source / receiver builders.

Domain choices:  isotropic, wind, hills_vegetation, ground_layer
Source types:    static, moving
Signal types:    ricker, tone, noise, propeller, file, drone_harmonics
Array layouts:   circular, concentric, linear, l_shaped, random
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the package is importable when running from the repo root.
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from acoustic_sim.domains_3d import (
    DomainMeta3D,
    create_ground_layer_domain_3d,
    create_hills_vegetation_domain_3d,
    create_isotropic_domain_3d,
    create_wind_domain_3d,
)
from acoustic_sim.fdtd import fd2_coefficients, fd2_cfl_factor
from acoustic_sim.fdtd_3d import FDTD3DConfig, FDTD3DSolver
from acoustic_sim.model_3d import VelocityModel3D
from acoustic_sim.plotting import plot_gather
from acoustic_sim.receivers_3d import (
    create_receiver_circle_3d,
    create_receiver_line_3d,
)
from acoustic_sim.receivers import (
    create_receiver_concentric,
    create_receiver_l_shaped,
    create_receiver_random,
)
from acoustic_sim.sources import (
    make_drone_harmonics,
    make_source_from_file,
    make_source_noise,
    make_source_propeller,
    make_source_tone,
    make_wavelet_ricker,
)
from acoustic_sim.sources_3d import MovingSource3D, StaticSource3D


# ---------------------------------------------------------------------------
# Domain builder
# ---------------------------------------------------------------------------

def build_domain_3d(
    domain: str = "isotropic",
    *,
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
    z_min: float = -5.0,
    z_max: float = 50.0,
    dx: float = 1.0,
    velocity: float = 343.0,
    wind_speed: float = 0.0,
    wind_direction_deg: float = 0.0,
    dirt_velocity: float = 1500.0,
    seed: int = 42,
) -> tuple[VelocityModel3D, DomainMeta3D]:
    """Dispatch to the appropriate 3-D domain builder."""
    grid_kw = dict(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_min, z_max=z_max,
        dx=dx,
    )
    if domain == "isotropic":
        return create_isotropic_domain_3d(velocity=velocity, **grid_kw)
    if domain == "wind":
        return create_wind_domain_3d(
            velocity=velocity,
            wind_speed=wind_speed,
            wind_direction_deg=wind_direction_deg,
            **grid_kw,
        )
    if domain == "hills_vegetation":
        return create_hills_vegetation_domain_3d(
            air_velocity=velocity,
            dirt_velocity=dirt_velocity,
            seed=seed,
            **grid_kw,
        )
    if domain == "ground_layer":
        return create_ground_layer_domain_3d(
            air_velocity=velocity,
            ground_velocity=dirt_velocity,
            ground_z=0.0,
            **grid_kw,
        )
    raise ValueError(f"Unknown 3D domain: {domain!r}")


# ---------------------------------------------------------------------------
# Receiver builder
# ---------------------------------------------------------------------------

def build_receivers_3d(
    array: str = "circular",
    *,
    count: int = 16,
    radius: float = 2.0,
    radii: list[float] | None = None,
    x0: float = -40.0,
    y0: float = 0.0,
    x1: float = 40.0,
    y1: float = 0.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
    spacing: float = 3.0,
    n1: int = 8,
    n2: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Build a 3-D receiver array.  Returns shape ``(n_recv, 3)``."""
    if radii is None:
        radii = [10.0, 20.0, 30.0, 40.0]

    if array == "circular":
        return create_receiver_circle_3d(
            center_x, center_y, radius, count, z=center_z,
        )
    if array == "linear":
        return create_receiver_line_3d(
            x0, y0, x1, y1, count, z=center_z,
        )
    if array == "concentric":
        pos_2d = create_receiver_concentric(
            center_x, center_y, radii, count,
        )
        z_arr = np.full(pos_2d.shape[0], center_z)
        return np.column_stack([pos_2d, z_arr])
    if array == "l_shaped":
        pos_2d = create_receiver_l_shaped(n1, n2, spacing, center_x, center_y)
        z_arr = np.full(pos_2d.shape[0], center_z)
        return np.column_stack([pos_2d, z_arr])
    if array == "random":
        pos_2d = create_receiver_random(count, x0, x1, y0, y1, seed=seed)
        z_arr = np.full(pos_2d.shape[0], center_z)
        return np.column_stack([pos_2d, z_arr])
    raise ValueError(f"Unknown 3D array type: {array!r}")


# ---------------------------------------------------------------------------
# CFL / dt for 3-D
# ---------------------------------------------------------------------------

def compute_dt_3d(
    model: VelocityModel3D,
    meta: DomainMeta3D,
    cfl_safety: float = 0.9,
    fd_order: int = 2,
) -> tuple[float, float]:
    """Return ``(dt, f_max)`` from 3-D CFL and spatial sampling.

    The 3-D CFL limit uses a √3 factor (vs √2 in 2-D) to account
    for the extra spatial dimension.
    """
    coeffs = fd2_coefficients(fd_order)
    spec_radius = fd2_cfl_factor(coeffs)

    c_max = float(np.max(model.values))
    v_wind = math.sqrt(
        meta.wind_vx ** 2 + meta.wind_vy ** 2 + meta.wind_vz ** 2
    )
    dt = (
        cfl_safety
        * 2.0
        * model.dx
        / ((c_max + v_wind) * math.sqrt(3.0 * spec_radius))
    )
    f_max = float(np.min(model.values)) / (10.0 * model.dx)
    return dt, f_max


# ---------------------------------------------------------------------------
# Source builder
# ---------------------------------------------------------------------------

def build_source_3d(
    source_type: str = "static",
    signal_type: str = "ricker",
    *,
    n_steps: int,
    dt: float,
    f_max: float,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    x1: float = 30.0,
    y1: float = 0.0,
    z1: float = 0.0,
    speed: float = 50.0,
    arc_height: float = 0.0,
    freq: float = 25.0,
    blade_count: int = 3,
    rpm: float = 3600.0,
    harmonics: int = 14,
    seed: int = 42,
    wav_path: str = "audio/input.wav",
    max_seconds: float | None = None,
    source_level_dB: float = 90.0,
    n_harmonics: int = 4,
    fundamental_freq: float = 150.0,
) -> StaticSource3D | MovingSource3D:
    """Build a 3-D source object with its signal."""
    sig = _build_signal(
        signal_type,
        n_steps=n_steps, dt=dt, f_max=f_max,
        freq=freq, blade_count=blade_count, rpm=rpm,
        harmonics=harmonics, seed=seed,
        wav_path=wav_path, max_seconds=max_seconds,
        source_level_dB=source_level_dB,
        n_harmonics=n_harmonics,
        fundamental_freq=fundamental_freq,
    )

    if source_type == "static":
        return StaticSource3D(x=x, y=y, z=z, signal=sig)
    if source_type == "moving":
        return MovingSource3D(
            x0=x, y0=y, z0=z,
            x1=x1, y1=y1, z1=z1,
            speed=speed, signal=sig,
            arc_height=arc_height,
        )
    raise ValueError(f"Unknown 3D source type: {source_type!r}")


def _build_signal(
    kind: str,
    *,
    n_steps: int,
    dt: float,
    f_max: float,
    freq: float,
    blade_count: int,
    rpm: float,
    harmonics: int,
    seed: int,
    wav_path: str,
    max_seconds: float | None,
    source_level_dB: float = 90.0,
    n_harmonics: int = 4,
    fundamental_freq: float = 150.0,
) -> np.ndarray:
    if kind == "ricker":
        return make_wavelet_ricker(n_steps, dt, freq)
    if kind == "tone":
        return make_source_tone(n_steps, dt, freq)
    if kind == "noise":
        return make_source_noise(n_steps, dt, f_low=5.0, f_high=f_max, seed=seed)
    if kind == "propeller":
        return make_source_propeller(
            n_steps, dt, f_max=f_max,
            blade_count=blade_count, rpm=rpm,
            harmonics=harmonics, seed=seed,
        )
    if kind == "file":
        return make_source_from_file(
            wav_path, n_steps, dt, f_max, max_seconds=max_seconds,
        )
    if kind == "drone_harmonics":
        return make_drone_harmonics(
            n_steps, dt,
            fundamental=fundamental_freq,
            n_harmonics=n_harmonics,
            source_level_dB=source_level_dB,
            f_max=f_max,
        )
    raise ValueError(f"Unknown signal type: {kind!r}")


# ---------------------------------------------------------------------------
# Domain plotting helpers (x-y and x-z slices)
# ---------------------------------------------------------------------------

def _plot_domain_xy(
    model: VelocityModel3D,
    output_path: str,
    z_index: int | None = None,
    receivers: np.ndarray | None = None,
    source_xy: np.ndarray | None = None,
    source_path: np.ndarray | None = None,
    attenuation: np.ndarray | None = None,
    title: str = "Domain (x-y slice)",
) -> None:
    """Plot an x-y slice of the 3-D velocity model at a given z-index."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if z_index is None:
        z_index = model.nz // 2
    z_index = max(0, min(z_index, model.nz - 1))
    z_val = float(model.z[z_index])
    vel_slice = model.values[z_index, :, :]  # (ny, nx)

    fig, ax = plt.subplots(figsize=(9, 7))
    ext = model.extent_xy
    im = ax.imshow(
        vel_slice, origin="lower",
        extent=[ext[0], ext[1], ext[2], ext[3]],
        cmap="terrain", aspect="equal", interpolation="bicubic",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wave speed [m/s]")

    # Vegetation overlay.
    if attenuation is not None:
        atten_slice = attenuation[z_index, :, :]
        veg_mask = np.ma.masked_where(atten_slice < 1e-6, atten_slice)
        ax.imshow(
            veg_mask, origin="lower",
            extent=[ext[0], ext[1], ext[2], ext[3]],
            cmap="Greens", alpha=0.45, aspect="equal",
        )

    if receivers is not None:
        ax.scatter(
            receivers[:, 0], receivers[:, 1],
            s=20, c="cyan", edgecolors="black", linewidths=0.4,
            zorder=5, label="Receivers",
        )

    if source_path is not None and len(source_path) >= 2:
        ax.plot(
            source_path[:, 0], source_path[:, 1],
            ls="--", lw=1.4, color="yellow", alpha=0.8, zorder=5,
        )
        n_arrows = min(5, len(source_path) - 1)
        idxs = np.linspace(0, len(source_path) - 2, n_arrows, dtype=int)
        for i in idxs:
            ax.annotate(
                "",
                xy=(source_path[i + 1, 0], source_path[i + 1, 1]),
                xytext=(source_path[i, 0], source_path[i, 1]),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.6),
            )
        ax.scatter(source_path[0, 0], source_path[0, 1],
                   s=80, c="yellow", marker="*", edgecolors="black",
                   zorder=6, label="Source start")
        ax.scatter(source_path[-1, 0], source_path[-1, 1],
                   s=60, c="orange", marker="s", edgecolors="black",
                   zorder=6, label="Source end")
    elif source_xy is not None:
        ax.scatter(source_xy[0], source_xy[1],
                   s=80, c="yellow", marker="*", edgecolors="black",
                   zorder=6, label="Source")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{title}  (z = {z_val:.1f} m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote x-y domain plot to {output_path}")


def _plot_domain_xz(
    model: VelocityModel3D,
    output_path: str,
    y_index: int | None = None,
    attenuation: np.ndarray | None = None,
    title: str = "Domain (x-z slice)",
) -> None:
    """Plot an x-z slice of the 3-D velocity model at a given y-index."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if y_index is None:
        y_index = model.ny // 2
    y_index = max(0, min(y_index, model.ny - 1))
    y_val = float(model.y[y_index])
    vel_slice = model.values[:, y_index, :]  # (nz, nx)

    fig, ax = plt.subplots(figsize=(10, 5))
    ext = model.extent_xz
    im = ax.imshow(
        vel_slice, origin="lower",
        extent=[ext[0], ext[1], ext[2], ext[3]],
        cmap="terrain", aspect="auto", interpolation="bicubic",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wave speed [m/s]")

    if attenuation is not None:
        atten_slice = attenuation[:, y_index, :]
        veg_mask = np.ma.masked_where(atten_slice < 1e-6, atten_slice)
        ax.imshow(
            veg_mask, origin="lower",
            extent=[ext[0], ext[1], ext[2], ext[3]],
            cmap="Greens", alpha=0.45, aspect="auto",
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"{title}  (y = {y_val:.1f} m)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote x-z domain plot to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-D FDTD acoustic simulation runner",
    )

    # Domain
    g = p.add_argument_group("Domain")
    g.add_argument("--domain",
                   choices=["isotropic", "wind", "hills_vegetation",
                            "ground_layer"],
                   default="isotropic")
    g.add_argument("--velocity", type=float, default=343.0)
    g.add_argument("--dx", type=float, default=1.0)
    g.add_argument("--x-min", type=float, default=-50.0)
    g.add_argument("--x-max", type=float, default=50.0)
    g.add_argument("--y-min", type=float, default=-50.0)
    g.add_argument("--y-max", type=float, default=50.0)
    g.add_argument("--z-min", type=float, default=-5.0)
    g.add_argument("--z-max", type=float, default=50.0)
    g.add_argument("--wind-speed", type=float, default=0.0)
    g.add_argument("--wind-dir", type=float, default=0.0)
    g.add_argument("--dirt-velocity", type=float, default=1500.0)
    g.add_argument("--seed", type=int, default=42)

    # Source position
    g = p.add_argument_group("Source")
    g.add_argument("--source-type", choices=["static", "moving"],
                   default="static")
    g.add_argument("--source-x", type=float, default=0.0)
    g.add_argument("--source-y", type=float, default=0.0)
    g.add_argument("--source-z", type=float, default=15.0,
                   help="Source z (altitude) at start [m]")
    g.add_argument("--source-x1", type=float, default=30.0)
    g.add_argument("--source-y1", type=float, default=0.0)
    g.add_argument("--source-z1", type=float, default=15.0,
                   help="Source z (altitude) at end [m]")
    g.add_argument("--source-speed", type=float, default=50.0)
    g.add_argument("--source-arc-height", type=float, default=0.0,
                   help="Parabolic arc height [m] added to z (vertical arc)")

    # Source signal
    g = p.add_argument_group("Source signal")
    g.add_argument("--source-signal",
                   choices=["file", "propeller", "tone", "noise", "ricker",
                            "drone_harmonics"],
                   default="ricker")
    g.add_argument("--source-wav", default="audio/input.wav")
    g.add_argument("--max-seconds", type=float, default=None)
    g.add_argument("--source-freq", type=float, default=25.0)
    g.add_argument("--blade-count", type=int, default=3)
    g.add_argument("--rpm", type=float, default=3600.0)
    g.add_argument("--harmonics", type=int, default=14)

    # Receivers
    g = p.add_argument_group("Receivers")
    g.add_argument("--array",
                   choices=["circular", "concentric", "linear",
                            "l_shaped", "random"],
                   default="circular")
    g.add_argument("--receiver-count", type=int, default=16)
    g.add_argument("--receiver-radius", type=float, default=2.0)
    g.add_argument("--receiver-radii", default="10,20,30,40")
    g.add_argument("--receiver-cx", type=float, default=0.0,
                   help="Receiver array centre X")
    g.add_argument("--receiver-cy", type=float, default=0.0,
                   help="Receiver array centre Y")
    g.add_argument("--receiver-cz", type=float, default=0.0,
                   help="Receiver array centre Z (altitude)")
    g.add_argument("--receiver-x0", type=float, default=-40.0)
    g.add_argument("--receiver-y0", type=float, default=0.0)
    g.add_argument("--receiver-x1", type=float, default=40.0)
    g.add_argument("--receiver-y1", type=float, default=0.0)

    # Simulation
    g = p.add_argument_group("Simulation")
    g.add_argument("--total-time", type=float, default=0.3)
    g.add_argument("--snapshot-interval", type=int, default=50)
    g.add_argument("--damping-width", type=int, default=10)
    g.add_argument("--damping-max", type=float, default=0.15)
    g.add_argument("--source-amplitude", type=float, default=1.0)
    g.add_argument("--air-absorption", type=float, default=0.005)
    g.add_argument("--use-cuda", action="store_true",
                   help="Use CuPy for GPU acceleration")
    g.add_argument("--fd-order", type=int, default=2,
                   help="FD spatial order of accuracy (2, 4, 6, …)")

    # Field plane (for decoupled array placement)
    g = p.add_argument_group("Field plane")
    g.add_argument("--field-plane-z", type=float, default=None,
                   help="Save horizontal pressure slice at this altitude "
                        "[m] at every timestep.  Enables post-hoc "
                        "receiver placement without re-running FDTD.")
    g.add_argument("--field-plane-subsample", type=int, default=4,
                   help="Spatial subsampling factor for field plane "
                        "(default 4; with dx=0.25 → 1.0 m spacing)")

    # Output
    p.add_argument("--output-dir", default="output/test_3d")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Determine MPI rank (rank 0 does I/O).
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0
    is_root = rank == 0

    out = Path(args.output_dir)
    if is_root:
        out.mkdir(parents=True, exist_ok=True)

    # -- Build 3-D domain --------------------------------------------------
    model, meta = build_domain_3d(
        args.domain,
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        z_min=args.z_min, z_max=args.z_max,
        dx=args.dx, velocity=args.velocity,
        wind_speed=args.wind_speed,
        wind_direction_deg=args.wind_dir,
        dirt_velocity=args.dirt_velocity,
        seed=args.seed,
    )
    if is_root:
        print(f"Domain: {meta.description}  grid={model.shape}  "
              f"c=[{model.c_min:.1f}, {model.c_max:.1f}] m/s")
        ncells = model.nx * model.ny * model.nz
        mem_mb = ncells * 8 * 2 / 1e6
        print(f"  Grid cells: {ncells:,}  Estimated memory: {mem_mb:.1f} MB")

    # -- Build 3-D receivers -----------------------------------------------
    radii = [float(r) for r in args.receiver_radii.split(",")]
    receivers = build_receivers_3d(
        args.array,
        count=args.receiver_count,
        radius=args.receiver_radius,
        radii=radii,
        x0=args.receiver_x0, y0=args.receiver_y0,
        x1=args.receiver_x1, y1=args.receiver_y1,
        center_x=args.receiver_cx, center_y=args.receiver_cy,
        center_z=args.receiver_cz,
    )
    if is_root:
        print(f"Receivers: {receivers.shape[0]} in '{args.array}' layout  "
              f"z={args.receiver_cz:.1f} m")

    # -- Compute dt (3-D CFL) and build config -----------------------------
    dt, f_max = compute_dt_3d(model, meta, cfl_safety=0.9,
                               fd_order=args.fd_order)
    n_steps = int(math.ceil(args.total_time / dt))
    if is_root:
        print(f"dt={dt:.2e} s,  n_steps={n_steps},  f_max={f_max:.1f} Hz")

    cfg = FDTD3DConfig(
        total_time=args.total_time,
        dt=dt,
        snapshot_interval=args.snapshot_interval,
        damping_width=args.damping_width,
        damping_max=args.damping_max,
        source_amplitude=args.source_amplitude,
        air_absorption=args.air_absorption,
        use_cuda=args.use_cuda,
        fd_order=args.fd_order,
    )

    # -- Build 3-D source --------------------------------------------------
    source = build_source_3d(
        args.source_type, args.source_signal,
        n_steps=n_steps, dt=dt, f_max=f_max,
        x=args.source_x, y=args.source_y, z=args.source_z,
        x1=args.source_x1, y1=args.source_y1, z1=args.source_z1,
        speed=args.source_speed,
        arc_height=args.source_arc_height,
        freq=args.source_freq,
        blade_count=args.blade_count,
        rpm=args.rpm, harmonics=args.harmonics,
        seed=args.seed,
        wav_path=args.source_wav,
        max_seconds=args.max_seconds,
    )

    # -- Domain plots (rank 0 only) ----------------------------------------
    if is_root:
        # Build source path for visualization.
        src_path = None
        if args.source_type == "moving":
            frac = np.linspace(0.0, 1.0, 60)
            src_path = np.column_stack([
                args.source_x + frac * (args.source_x1 - args.source_x),
                args.source_y + frac * (args.source_y1 - args.source_y),
            ])

        # Find z-index closest to source altitude for the x-y slice.
        src_z_mid = 0.5 * (args.source_z + args.source_z1)
        z_idx_src = int(np.argmin(np.abs(model.z - src_z_mid)))
        # Also find a ground-level z-index.
        z_idx_ground = int(np.argmin(np.abs(model.z - 0.0)))

        _plot_domain_xy(
            model,
            output_path=str(out / "domain_xy_ground.png"),
            z_index=z_idx_ground,
            receivers=receivers,
            source_xy=np.array([args.source_x, args.source_y]),
            source_path=src_path,
            attenuation=meta.attenuation,
            title=f"Domain (x-y, ground level): {args.domain}",
        )
        _plot_domain_xy(
            model,
            output_path=str(out / "domain_xy_altitude.png"),
            z_index=z_idx_src,
            receivers=receivers,
            source_xy=np.array([args.source_x, args.source_y]),
            source_path=src_path,
            attenuation=meta.attenuation,
            title=f"Domain (x-y, source alt): {args.domain}",
        )

        # x-z slice through the valley (y=0 or nearest index).
        y_idx_mid = int(np.argmin(np.abs(model.y - 0.0)))
        _plot_domain_xz(
            model,
            output_path=str(out / "domain_xz.png"),
            y_index=y_idx_mid,
            attenuation=meta.attenuation,
            title=f"Domain (x-z): {args.domain}",
        )

        # x-z slice through the southern ridge centre.
        y_idx_south = int(np.argmin(np.abs(model.y - (-20.0))))
        _plot_domain_xz(
            model,
            output_path=str(out / "domain_xz_south_ridge.png"),
            y_index=y_idx_south,
            attenuation=meta.attenuation,
            title=f"Domain (x-z, south ridge): {args.domain}",
        )

    # -- Run 3-D FDTD ------------------------------------------------------
    snap_dir = str(out / "snapshots")
    solver = FDTD3DSolver(
        model=model, config=cfg, source=source,
        receivers=receivers, domain_meta=meta,
    )
    result = solver.run(
        snapshot_dir=snap_dir, verbose=True,
        field_plane_z=args.field_plane_z,
        field_plane_subsample=args.field_plane_subsample,
    )

    # -- Save outputs (rank 0 only) ----------------------------------------
    if is_root:
        traces = result["traces"]
        np.save(str(out / "traces.npy"), traces)
        print(f"Saved traces {traces.shape} to {out / 'traces.npy'}")

        # -- Save field plane (if recorded) ----------------------------
        if "field_plane" in result:
            fp = result["field_plane"]
            np.save(str(out / "field_plane.npy"), fp)
            print(f"Saved field_plane {fp.shape} "
                  f"({fp.nbytes / 1e6:.0f} MB) to "
                  f"{out / 'field_plane.npy'}")

        metadata = {
            "domain": args.domain,
            "source_type": args.source_type,
            "source_signal": args.source_signal,
            "array": args.array,
            "dx": args.dx,
            "dt": result["dt"],
            "n_steps": result["n_steps"],
            "total_time": args.total_time,
            "n_receivers": int(receivers.shape[0]),
            "receiver_positions": receivers.tolist(),
            "source_x": args.source_x,
            "source_y": args.source_y,
            "source_z": args.source_z,
            "source_x1": args.source_x1,
            "source_y1": args.source_y1,
            "source_z1": args.source_z1,
            "x_min": args.x_min,
            "x_max": args.x_max,
            "y_min": args.y_min,
            "y_max": args.y_max,
            "z_min": args.z_min,
            "z_max": args.z_max,
            "velocity": args.velocity,
            "wind_vx": meta.wind_vx,
            "wind_vy": meta.wind_vy,
            "wind_vz": meta.wind_vz,
            "use_cuda": args.use_cuda,
            "fd_order": args.fd_order,
            "grid_shape": list(model.shape),
        }

        if "field_plane" in result:
            metadata["field_plane_z"] = result["field_plane_z"]
            metadata["field_plane_x"] = result["field_plane_x"].tolist()
            metadata["field_plane_y"] = result["field_plane_y"].tolist()
            metadata["field_plane_subsample"] = result["field_plane_subsample"]

        with open(str(out / "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        plot_gather(
            traces, result["dt"],
            output_path=str(out / "gather.png"),
            title=(
                f"3D Gather: {args.domain} / {args.source_type} / "
                f"{args.array}"
            ),
        )
        print("Done.")


if __name__ == "__main__":
    main()

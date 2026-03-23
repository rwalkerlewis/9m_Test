#!/usr/bin/env python3
"""Example 4: 3-D coupled air–ground simulation with a moving source.

A propeller source flies an arc path through the domain.  Microphones
and geophones are placed on separate circles with independent radii
and counts.

Usage::

    bash examples/run_coupled_moving_3d.sh
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from acoustic_sim.elastic_fdtd import (
    ElasticFDTDConfig,
    ElasticFDTD3DSolver,
    fd1_staggered_coefficients,
    elastic_cfl_factor,
)
from acoustic_sim.elastic_model import GroundConfig, create_coupled_air_ground_3d
from acoustic_sim.receivers import ReceiverSpec
from acoustic_sim.sources import MovingSource3D, make_source_propeller, make_wavelet_ricker
from acoustic_sim.plotting import plot_gather, save_snapshot_3d


# ---------------------------------------------------------------------------
# Domain visualisation helpers
# ---------------------------------------------------------------------------

def _plot_domain_xy(
    model,
    output_path: str,
    z_index: int | None = None,
    receivers: np.ndarray | None = None,
    source_path: np.ndarray | None = None,
    title: str = "Elastic domain (x-y)",
) -> None:
    """Plot an x-y slice of the Vp field with receiver and source overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if z_index is None:
        z_index = model.nz // 2
    z_index = max(0, min(z_index, model.nz - 1))
    z_val = float(model.z[z_index])
    vel_slice = model.vp[z_index, :, :]

    fig, ax = plt.subplots(figsize=(9, 7))
    ext = model.extent_xy
    im = ax.imshow(
        vel_slice, origin="lower",
        extent=[ext[0], ext[1], ext[2], ext[3]],
        cmap="terrain", aspect="equal", interpolation="bicubic",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Vp [m/s]")

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

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{title}  (z = {z_val:.1f} m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote x-y domain plot to {output_path}")


def _plot_domain_xz(
    model,
    output_path: str,
    y_index: int | None = None,
    title: str = "Elastic domain (x-z)",
) -> None:
    """Plot an x-z slice of the Vp field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if y_index is None:
        y_index = model.ny // 2
    y_index = max(0, min(y_index, model.ny - 1))
    y_val = float(model.y[y_index])
    vel_slice = model.vp[:, y_index, :]

    fig, ax = plt.subplots(figsize=(10, 5))
    ext = model.extent_xz
    im = ax.imshow(
        vel_slice, origin="lower",
        extent=[ext[0], ext[1], ext[2], ext[3]],
        cmap="terrain", aspect="auto", interpolation="bicubic",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Vp [m/s]")
    ax.axhline(0.0, color="green", linestyle="--", linewidth=0.8, label="z = 0")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"{title}  (y = {y_val:.1f} m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote x-z domain plot to {output_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="3-D coupled air–ground (moving source)")
    g = p.add_argument_group("Domain")
    g.add_argument("--x-min", type=float, default=0.0)
    g.add_argument("--x-max", type=float, default=100.0)
    g.add_argument("--y-min", type=float, default=0.0)
    g.add_argument("--y-max", type=float, default=100.0)
    g.add_argument("--z-min", type=float, default=-5.0)
    g.add_argument("--z-max", type=float, default=15.0)
    g.add_argument("--dx", type=float, default=None)
    g.add_argument("--ground-vp", type=float, default=500.0)
    g.add_argument("--ground-vs", type=float, default=250.0)
    g.add_argument("--ground-density", type=float, default=1800.0)
    g.add_argument("--ground-qp", type=float, default=20.0)
    g.add_argument("--ground-qs", type=float, default=10.0)

    g = p.add_argument_group("Source")
    g.add_argument("--source-x0", type=float, default=-40.0)
    g.add_argument("--source-y0", type=float, default=0.0)
    g.add_argument("--source-x1", type=float, default=40.0)
    g.add_argument("--source-y1", type=float, default=0.0)
    g.add_argument("--source-z0", type=float, default=15.0)
    g.add_argument("--source-z1", type=float, default=15.0)
    g.add_argument("--source-speed", type=float, default=50.0)
    g.add_argument("--source-arc-height", type=float, default=10.0)
    g.add_argument("--source-signal", choices=["propeller", "ricker"],
                   default="propeller")
    g.add_argument("--source-freq", type=float, default=25.0)
    g.add_argument("--blade-count", type=int, default=3)
    g.add_argument("--rpm", type=int, default=3600)
    g.add_argument("--harmonics", type=int, default=14)

    g = p.add_argument_group("Receivers")
    g.add_argument("--array-cx", type=float, default=-5.0)
    g.add_argument("--array-cy", type=float, default=7.0)
    g.add_argument("--mic-radius", type=float, default=2.0)
    g.add_argument("--mic-count", type=int, default=10)
    g.add_argument("--mic-z", type=float, default=5.5)
    g.add_argument("--geo-radius", type=float, default=3.0)
    g.add_argument("--geo-count", type=int, default=8)
    g.add_argument("--geo-z", type=float, default=-0.05)

    g = p.add_argument_group("Simulation")
    g.add_argument("--total-time", type=float, default=0.3)
    g.add_argument("--fd-order", type=int, default=4)
    g.add_argument("--cfl-safety", type=float, default=0.8)
    g.add_argument("--damping-width", type=int, default=10)
    g.add_argument("--damping-max", type=float, default=0.05)
    g.add_argument("--snapshot-interval", type=int, default=200)
    g.add_argument("--source-amplitude", type=float, default=1.0)
    g.add_argument("--use-cuda", action="store_true")
    g.add_argument("--enable-attenuation", action="store_true", default=True)
    g.add_argument("--no-attenuation", dest="enable_attenuation",
                   action="store_false")

    g = p.add_argument_group("Field plane")
    g.add_argument("--field-plane-z", type=float, default=None,
                   help="Altitude for horizontal field-plane extraction [m]")
    g.add_argument("--field-plane-subsample", type=int, default=4)

    p.add_argument("--output-dir", default="output/coupled_moving_3d")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0
    is_root = rank == 0

    out = Path(args.output_dir)
    if is_root:
        out.mkdir(parents=True, exist_ok=True)

    ground = GroundConfig(
        vp=args.ground_vp, vs=args.ground_vs,
        density=args.ground_density,
        qp=args.ground_qp, qs=args.ground_qs,
    )
    if args.dx is None:
        v_min = min(ground.vs, 343.0)
        ppw = {2: 20, 4: 10, 6: 7, 8: 6}.get(args.fd_order, 10)
        args.dx = max(v_min / (ppw * 200.0), 0.5)
        if is_root:
            print(f"Auto dx = {args.dx:.3f} m")

    model = create_coupled_air_ground_3d(
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        z_min=args.z_min, z_max=args.z_max,
        dx=args.dx, ground=ground,
    )
    if is_root:
        ncells = model.nx * model.ny * model.nz
        print(f"Domain: {model.shape}  cells={ncells:,}")

    cfg = ElasticFDTDConfig(
        total_time=args.total_time,
        fd_order=args.fd_order,
        cfl_safety=args.cfl_safety,
        damping_width=args.damping_width,
        damping_max=args.damping_max,
        snapshot_interval=args.snapshot_interval,
        source_amplitude=args.source_amplitude,
        use_cuda=args.use_cuda,
        enable_attenuation=args.enable_attenuation,
    )

    coeffs = fd1_staggered_coefficients(args.fd_order)
    S = elastic_cfl_factor(coeffs)
    dt_est = args.cfl_safety * args.dx / (model.vp_max * S * math.sqrt(3.0))
    n_steps = int(math.ceil(args.total_time / dt_est)) + 100

    # f_max for the *source signal* is limited by the medium it lives in
    # (air, Vp=343 m/s), not the ground shear velocity.  Using Vs gives
    # f_max ~100 Hz which truncates all propeller harmonics (BPF ≥ 180 Hz).
    # With 4th-order FD, 5 ppw gives <1% numerical dispersion in air.
    ppw_source = {2: 10, 4: 5, 6: 4, 8: 3}.get(args.fd_order, 5)
    f_max = 343.0 / (ppw_source * args.dx)
    if is_root:
        print(f"Source f_max = {f_max:.0f} Hz  (ppw={ppw_source})")

    if args.source_signal == "propeller":
        sig = make_source_propeller(n_steps, dt_est, f_max=f_max,
                                    blade_count=args.blade_count,
                                    rpm=args.rpm, harmonics=args.harmonics)
    else:
        sig = make_wavelet_ricker(n_steps, dt_est, args.source_freq)

    source = MovingSource3D(
        x0=args.source_x0, y0=args.source_y0, z0=args.source_z0,
        x1=args.source_x1, y1=args.source_y1, z1=args.source_z1,
        speed=args.source_speed, signal=sig,
        arc_height=args.source_arc_height,
    )

    # Microphones — evenly spaced on a circle
    mic_angles = np.linspace(0, 2 * np.pi, args.mic_count, endpoint=False)
    mic_positions = np.column_stack([
        args.array_cx + args.mic_radius * np.cos(mic_angles),
        args.array_cy + args.mic_radius * np.sin(mic_angles),
        np.full(args.mic_count, args.mic_z),
    ])

    # Geophones — cardinal / intercardinal directions
    geo_angles = np.linspace(0, 2 * np.pi, args.geo_count, endpoint=False)
    geo_positions = np.column_stack([
        args.array_cx + args.geo_radius * np.cos(geo_angles),
        args.array_cy + args.geo_radius * np.sin(geo_angles),
        np.full(args.geo_count, args.geo_z),
    ])

    positions = np.vstack([mic_positions, geo_positions])
    sensor_types = ["microphone"] * args.mic_count + ["geophone"] * args.geo_count
    field_components = ["pressure"] * args.mic_count + ["vz"] * args.geo_count
    receivers = ReceiverSpec(
        positions=positions,
        sensor_types=sensor_types,
        field_components=field_components,
    )
    if is_root:
        print(f"Receivers: {receivers.n_receivers}")

    # -- Source path for plots -------------------------------------------------
    src_dt_est = dt_est
    n_path = max(2, int(math.ceil(args.total_time / src_dt_est)))
    source_path = np.array(
        [source.position_at(i, src_dt_est) for i in range(n_path)]
    )

    # -- Domain plots ----------------------------------------------------------
    if is_root:
        z_ground_idx = int(np.argmin(np.abs(model.z - 0.0)))
        z_alt_idx = int(np.argmin(np.abs(model.z - args.source_z0)))

        _plot_domain_xy(
            model, str(out / "domain_xy_ground.png"),
            z_index=z_ground_idx,
            receivers=positions,
            source_path=source_path,
            title="Elastic Vp (ground level)",
        )
        _plot_domain_xy(
            model, str(out / "domain_xy_altitude.png"),
            z_index=z_alt_idx,
            receivers=positions,
            source_path=source_path,
            title="Elastic Vp (source altitude)",
        )
        _plot_domain_xz(
            model, str(out / "domain_xz.png"),
            title="Elastic Vp (x-z centre)",
        )

    # -- Run -------------------------------------------------------------------
    snap_dir = str(out / "snapshots")
    solver = ElasticFDTD3DSolver(model, cfg, source, receivers)
    if is_root:
        print(f"dt={solver.dt:.2e}, n_steps={solver.n_steps}")
    result = solver.run(
        snapshot_dir=snap_dir, verbose=is_root,
        field_plane_z=args.field_plane_z,
        field_plane_subsample=args.field_plane_subsample,
    )

    if is_root:
        traces = result["traces"]
        np.save(str(out / "traces.npy"), traces)
        mic_idx = receivers.mic_indices
        geo_idx = receivers.geo_indices
        np.save(str(out / "mic_traces.npy"), traces[mic_idx])
        np.save(str(out / "geo_traces.npy"), traces[geo_idx])

        # -- Save field plane (if recorded) ----------------------------
        if "field_plane" in result:
            fp = result["field_plane"]
            np.save(str(out / "field_plane.npy"), fp)
            print(f"Saved field_plane {fp.shape} "
                  f"({fp.nbytes / 1e6:.0f} MB) to "
                  f"{out / 'field_plane.npy'}")

        metadata = {
            "example": "coupled_moving_3d",
            "domain_shape": list(model.shape),
            "dx": args.dx,
            "dt": result["dt"],
            "n_steps": result["n_steps"],
            "total_time": args.total_time,
            "velocity": 343.0,
            "source_x": args.source_x0,
            "source_y": args.source_y0,
            "source_z": args.source_z0,
            "source_x1": args.source_x1,
            "source_y1": args.source_y1,
            "source_z1": args.source_z1,
            "source_start": [args.source_x0, args.source_y0, args.source_z0],
            "source_end": [args.source_x1, args.source_y1, args.source_z1],
            "source_speed": args.source_speed,
            "source_arc_height": args.source_arc_height,
            "ground": {"vp": ground.vp, "vs": ground.vs, "density": ground.density},
            "n_microphones": receivers.n_microphones,
            "n_geophones": receivers.n_geophones,
            "mic_radius": args.mic_radius,
            "geo_radius": args.geo_radius,
            "mic_positions": mic_positions.tolist(),
            "geo_positions": geo_positions.tolist(),
            "receiver_positions": positions.tolist(),
        }

        if "field_plane" in result:
            metadata["field_plane_z"] = result["field_plane_z"]
            metadata["field_plane_x"] = result["field_plane_x"].tolist()
            metadata["field_plane_y"] = result["field_plane_y"].tolist()
            metadata["field_plane_subsample"] = result["field_plane_subsample"]

        with open(str(out / "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        if traces.shape[1] > 0:
            plot_gather(traces[mic_idx], result["dt"],
                        output_path=str(out / "gather_mics.png"),
                        title="3D Microphone gather (moving)")
            plot_gather(traces[geo_idx], result["dt"],
                        output_path=str(out / "gather_geos.png"),
                        title="3D Geophone gather (moving)")
        print("Done.")


if __name__ == "__main__":
    main()

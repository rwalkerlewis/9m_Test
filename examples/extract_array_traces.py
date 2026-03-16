#!/usr/bin/env python3
"""Extract receiver traces from a saved FDTD field plane.

Given a ``field_plane.npy`` produced by ``run_fdtd_3d.py --field-plane-z``,
this script bilinearly interpolates the horizontal pressure slice at
arbitrary receiver positions to produce ``traces.npy`` — without
re-running the forward model.

Usage
-----
    python examples/extract_array_traces.py \
        --input-dir output/valley_cuda \
        --output-dir output/valley_array2 \
        --array circular --receiver-count 16 --receiver-radius 2.0 \
        --receiver-cx -5.0 --receiver-cy 7.0 --receiver-cz 5.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root or the examples/ directory.
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root))

from examples.run_fdtd_3d import build_receivers_3d  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract receiver traces from a saved field plane",
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory containing field_plane.npy and "
                        "metadata.json from a prior FDTD run")
    p.add_argument("--output-dir", required=True,
                   help="Directory for extracted traces.npy and "
                        "metadata.json")

    g = p.add_argument_group("Receiver array")
    g.add_argument("--array",
                   choices=["circular", "concentric", "linear",
                            "l_shaped", "random"],
                   default="circular")
    g.add_argument("--receiver-count", type=int, default=16)
    g.add_argument("--receiver-radius", type=float, default=2.0)
    g.add_argument("--receiver-radii", default="10,20,30,40")
    g.add_argument("--receiver-cx", type=float, default=0.0)
    g.add_argument("--receiver-cy", type=float, default=0.0)
    g.add_argument("--receiver-cz", type=float, default=0.0,
                   help="Receiver z (altitude); recorded in metadata "
                        "but not used for interpolation (the field "
                        "plane is already at a fixed z).")
    g.add_argument("--receiver-x0", type=float, default=-40.0)
    g.add_argument("--receiver-y0", type=float, default=0.0)
    g.add_argument("--receiver-x1", type=float, default=40.0)
    g.add_argument("--receiver-y1", type=float, default=0.0)

    return p.parse_args(argv)


def extract_traces(
    field_plane: np.ndarray,
    plane_x: np.ndarray,
    plane_y: np.ndarray,
    receivers: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate the field plane at receiver (x, y) positions.

    Parameters
    ----------
    field_plane : ndarray, shape (n_steps, ny, nx), float32
    plane_x : ndarray, shape (nx,)  — x coordinates of plane columns
    plane_y : ndarray, shape (ny,)  — y coordinates of plane rows
    receivers : ndarray, shape (n_recv, 3)  — [[x, y, z], ...]

    Returns
    -------
    traces : ndarray, shape (n_recv, n_steps), float64
    """
    n_steps, ny, nx = field_plane.shape
    n_recv = receivers.shape[0]

    dx = plane_x[1] - plane_x[0]
    dy = plane_y[1] - plane_y[0]

    traces = np.zeros((n_recv, n_steps), dtype=np.float64)

    for i in range(n_recv):
        rx, ry = receivers[i, 0], receivers[i, 1]

        # Fractional grid indices on the subsampled plane.
        fx = (rx - plane_x[0]) / dx
        fy = (ry - plane_y[0]) / dy

        ix = int(np.floor(fx))
        iy = int(np.floor(fy))

        ix = max(0, min(ix, nx - 2))
        iy = max(0, min(iy, ny - 2))

        wx = np.clip(fx - ix, 0.0, 1.0)
        wy = np.clip(fy - iy, 0.0, 1.0)

        # Bilinear interpolation across all timesteps at once.
        traces[i] = (
            field_plane[:, iy,     ix    ] * (1 - wy) * (1 - wx)
          + field_plane[:, iy,     ix + 1] * (1 - wy) *      wx
          + field_plane[:, iy + 1, ix    ] *      wy  * (1 - wx)
          + field_plane[:, iy + 1, ix + 1] *      wy  *      wx
        )

    return traces


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load field plane and metadata ----------------------------------
    fp_path = in_dir / "field_plane.npy"
    meta_path = in_dir / "metadata.json"

    if not fp_path.exists():
        print(f"ERROR: {fp_path} not found.  Re-run the FDTD with "
              f"--field-plane-z to generate it.", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    if "field_plane_x" not in meta:
        print("ERROR: metadata.json has no field_plane_x.  Was the FDTD "
              "run executed with --field-plane-z?", file=sys.stderr)
        sys.exit(1)

    print(f"Loading field plane from {fp_path} …")
    field_plane = np.load(str(fp_path))
    plane_x = np.asarray(meta["field_plane_x"])
    plane_y = np.asarray(meta["field_plane_y"])
    print(f"  shape={field_plane.shape}  "
          f"x=[{plane_x[0]:.1f}, {plane_x[-1]:.1f}]  "
          f"y=[{plane_y[0]:.1f}, {plane_y[-1]:.1f}]  "
          f"z={meta['field_plane_z']:.2f} m")

    # -- Build receiver array -------------------------------------------
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
    print(f"Receivers: {receivers.shape[0]} in '{args.array}' layout  "
          f"centre=({args.receiver_cx}, {args.receiver_cy}, "
          f"{args.receiver_cz})")

    # -- Warn if receivers are outside the field plane extent -----------
    for i in range(receivers.shape[0]):
        rx, ry = receivers[i, 0], receivers[i, 1]
        if (rx < plane_x[0] or rx > plane_x[-1]
                or ry < plane_y[0] or ry > plane_y[-1]):
            print(f"  WARNING: receiver {i} at ({rx:.2f}, {ry:.2f}) "
                  f"is outside the field plane extent — "
                  f"trace will be clamped.", file=sys.stderr)

    # -- Extract traces -------------------------------------------------
    traces = extract_traces(field_plane, plane_x, plane_y, receivers)
    print(f"Extracted traces: {traces.shape}")

    # -- Save outputs ---------------------------------------------------
    np.save(str(out_dir / "traces.npy"), traces)
    print(f"Saved {out_dir / 'traces.npy'}")

    # Build updated metadata: copy everything from the FDTD run, then
    # overwrite the receiver-specific fields.
    out_meta = dict(meta)
    out_meta["array"] = args.array
    out_meta["n_receivers"] = int(receivers.shape[0])
    out_meta["receiver_positions"] = receivers.tolist()
    out_meta["extracted_from"] = str(fp_path)

    with open(str(out_dir / "metadata.json"), "w") as f:
        json.dump(out_meta, f, indent=2)
    print(f"Saved {out_dir / 'metadata.json'}")
    print("Done.")


if __name__ == "__main__":
    main()

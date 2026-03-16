#!/usr/bin/env python3
"""Diagnose 3D MFP bearing by running a single window and plotting beam-power maps.

Prints per-window bearing, compares to ground truth, and saves beam-power
polar plot to the output directory.

Usage:
    python examples/diagnose_3d_mfp.py output/valley_3d_test
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acoustic_sim.processor_3d import matched_field_process_3d


def load_simulation(sim_dir):
    """Load simulation traces and metadata."""
    sim_dir = Path(sim_dir)
    import json as _json
    traces = np.load(sim_dir / "traces.npy")
    with open(sim_dir / "metadata.json") as f:
        metadata = _json.load(f)
    mic_positions = np.array(metadata["receiver_positions"])
    return {
        "traces": traces,
        "mic_positions": mic_positions,
        "metadata": metadata,
        "dt": metadata["dt"],
        "duration": traces.shape[1] * metadata["dt"],
    }


def compute_ground_truth(meta: dict, speed: float = 50.0):
    """Return (callable(t) -> (x,y,z), duration)."""
    x0 = meta["source_x"]
    y0 = meta["source_y"]
    z0 = meta.get("source_z", 0.0)
    x1 = meta.get("source_x1", x0)
    y1 = meta.get("source_y1", y0)
    z1 = meta.get("source_z1", z0)
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    dur = dist / speed if speed > 0 else 1.0

    def gt(t):
        frac = max(0.0, min(1.0, t / dur)) if dur > 0 else 0.0
        return (x0 + frac * dx, y0 + frac * dy, z0 + frac * dz)

    return gt, dur


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", help="Simulation output directory")
    parser.add_argument("--speed", type=float, default=50.0)
    parser.add_argument("--fundamental", type=float, default=180.0)
    parser.add_argument("--n-harmonics", type=int, default=4)
    args = parser.parse_args()

    data = load_simulation(args.sim_dir)
    traces = data["traces"]
    mic_positions = data["mic_positions"]
    dt = data["dt"]
    meta = data["metadata"]

    n_mics, n_samples = traces.shape
    fs = 1.0 / dt
    cx = float(np.mean(mic_positions[:, 0]))
    cy = float(np.mean(mic_positions[:, 1]))
    print(f"Array center: ({cx:.2f}, {cy:.2f})")
    print(f"Mics: {n_mics}, fs={fs:.0f} Hz, samples={n_samples}")

    gt_fn, dur = compute_ground_truth(meta, args.speed)

    # Run MFP
    result = matched_field_process_3d(
        traces, mic_positions, dt,
        fundamental=args.fundamental,
        n_harmonics=args.n_harmonics,
        detection_threshold=0.02,
        min_signal_rms=5e-5,
        window_length=0.2,
        window_overlap=0.5,
        n_subwindows=2,
        harmonic_bandwidth=20.0,
        range_min=5.0,
        range_max=100.0,
        z_min=15.0,
        z_max=15.0,
        z_spacing=1.0,
    )

    azimuths = result["azimuths"]
    ranges = result["ranges"]
    z_values = result["z_values"]
    detections = result["detections"]
    selected_freqs = result["selected_freqs"]

    print(f"\nSelected frequencies: {selected_freqs}")
    print(f"Grid: {len(azimuths)} az x {len(ranges)} range x {len(z_values)} z")
    print(f"Z values: {z_values}")
    print()

    # Per-detection analysis
    print(f"{'Window':>6} {'Time':>6} {'Det':>4} {'MFP_brg':>8} {'True_brg':>9} "
          f"{'Err':>6} {'MFP_rng':>8} {'True_rng':>9} {'MFP_z':>6} {'True_z':>7} {'RMS':>10}")
    print("-" * 95)

    for i, d in enumerate(detections):
        t = d["time"]
        gt_x, gt_y, gt_z = gt_fn(t)
        true_brg = math.degrees(math.atan2(gt_y - cy, gt_x - cx))
        if true_brg < 0:
            true_brg += 360.0
        true_rng = math.hypot(gt_x - cx, gt_y - cy)

        if d["detected"]:
            mfp_brg = d["bearing_deg"]
            err = mfp_brg - true_brg
            if err > 180:
                err -= 360
            if err < -180:
                err += 360
            print(f"{i:6d} {t:6.3f} {'Y':>4} {mfp_brg:8.1f} {true_brg:9.1f} "
                  f"{err:+6.1f} {d['range']:8.1f} {true_rng:9.1f} "
                  f"{d.get('z', 0):6.1f} {gt_z:7.1f} {d.get('window_rms', 0):10.6f}")
        else:
            print(f"{i:6d} {t:6.3f} {'N':>4} {'---':>8} {true_brg:9.1f} "
                  f"{'---':>6} {'---':>8} {true_rng:9.1f} "
                  f"{'---':>6} {gt_z:7.1f} {d.get('window_rms', 0):10.6f}")

    # Plot beam-power maps for first few detected windows
    det_indices = [i for i, d in enumerate(detections) if d["detected"]]
    n_plot = min(4, len(det_indices))
    if n_plot == 0:
        print("\nNo detections to plot.")
        sys.exit(0)

    fig, axes = plt.subplots(2, n_plot, figsize=(5 * n_plot, 10))
    if n_plot == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(det_indices[:n_plot]):
        d = detections[idx]
        t = d["time"]
        gt_x, gt_y, gt_z = gt_fn(t)
        true_brg_rad = math.atan2(gt_y - cy, gt_x - cx)
        true_rng = math.hypot(gt_x - cx, gt_y - cy)

        bpm_3d = d.get("beam_power_map_3d", None)
        bpm_2d = d.get("beam_power_map", None)
        if bpm_3d is not None and bpm_3d.ndim == 3:
            # Max over z for top row
            bpm_xy = np.max(bpm_3d, axis=2)
            # Slice at best z for bottom row
            iz_best = np.unravel_index(np.argmax(bpm_3d), bpm_3d.shape)[2]
        elif bpm_2d is not None:
            bpm_xy = bpm_2d
            iz_best = 0
        else:
            continue

        # Top: polar plot (azimuth vs range)
        ax = axes[0, col]
        az_mesh, rng_mesh = np.meshgrid(azimuths, ranges, indexing="ij")
        c = ax.pcolormesh(
            np.degrees(az_mesh), rng_mesh, bpm_xy,
            shading="auto", cmap="inferno",
        )
        ax.axvline(math.degrees(true_brg_rad) % 360, color="cyan", ls="--", lw=1.5, label="True bearing")
        ax.axhline(true_rng, color="cyan", ls=":", lw=1, label="True range")
        if d["detected"]:
            ax.plot(d["bearing_deg"], d["range"], "g*", ms=12, label="MFP peak")
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Range (m)")
        ax.set_title(f"Window {idx} t={t:.3f}s\nz-max projection")
        ax.legend(fontsize=7, loc="upper right")
        plt.colorbar(c, ax=ax)

        # Bottom: azimuth cut at best range bin
        ax2 = axes[1, col]
        ir_peak = np.argmax(np.max(bpm_xy, axis=0))
        az_cut = bpm_xy[:, ir_peak]
        ax2.plot(np.degrees(azimuths), az_cut, "k-", lw=1)
        ax2.axvline(math.degrees(true_brg_rad) % 360, color="cyan", ls="--", lw=1.5, label="True")
        if d["detected"]:
            ax2.axvline(d["bearing_deg"], color="lime", ls="--", lw=1.5, label="MFP peak")
        ax2.set_xlabel("Azimuth (deg)")
        ax2.set_ylabel("Norm. beam power")
        ax2.set_title(f"Az cut at range={ranges[ir_peak]:.0f}m")
        ax2.legend(fontsize=7)

    fig.tight_layout()
    out = Path(args.sim_dir) / "mfp_diagnostic_3d.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    sys.exit(0)


if __name__ == "__main__":
    main()

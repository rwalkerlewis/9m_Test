#!/usr/bin/env python3
"""Analyse frequency content and signal levels of 3-D FDTD traces.

Prints RMS per mic, spectral peaks, and energy-by-band breakdown
to guide MFP parameter tuning.

Usage::

    python examples/analyse_3d_traces.py output/valley_3d_test

Expected output: frequency peaks, RMS values, and energy distribution
that inform the choice of fundamental, harmonics, and detection thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.fft import rfft, rfftfreq


def analyse(sim_dir: Path) -> None:
    traces = np.load(sim_dir / "traces.npy")
    with open(sim_dir / "metadata.json") as f:
        meta = json.load(f)

    dt = meta["dt"]
    dx = meta["dx"]
    c = meta["velocity"]
    fs = 1.0 / dt

    print("=== GRID & SAMPLING ===")
    print(f"  dx = {dx} m,  c = {c} m/s")
    print(f"  f_max (2 PPW) = {c / (2 * dx):.1f} Hz")
    print(f"  f_max (6 PPW) = {c / (6 * dx):.1f} Hz")
    print(f"  Sampling rate = {fs:.1f} Hz,  Nyquist = {fs / 2:.1f} Hz")
    print(f"  Trace shape: {traces.shape}")
    print()

    # Per-mic RMS
    print("=== RMS PER MIC ===")
    rms_vals = []
    for i in range(traces.shape[0]):
        rms = float(np.sqrt(np.mean(traces[i] ** 2)))
        rms_vals.append(rms)
        print(f"  mic {i:2d}: rms = {rms:.6f}")
    mean_rms = np.mean(rms_vals)
    print(f"  MEAN RMS: {mean_rms:.6f}")
    print()

    # Average spectrum
    n_mics = traces.shape[0]
    specs = [np.abs(rfft(traces[i])) for i in range(n_mics)]
    avg_spec = np.mean(specs, axis=0)
    freqs = rfftfreq(traces.shape[1], dt)

    # Top peaks
    top_idx = np.argsort(avg_spec)[-20:][::-1]
    print("=== TOP 20 SPECTRAL PEAKS (avg across mics) ===")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {freqs[idx]:7.1f} Hz   amplitude = {avg_spec[idx]:.4f}")
    print()

    # Energy by band
    bands = [
        (0, 10), (10, 30), (30, 60), (60, 90), (90, 120),
        (120, 150), (150, 200), (200, 300), (300, 500), (500, 1000),
    ]
    total_energy = float(np.sum(avg_spec ** 2))
    print("=== ENERGY BY FREQUENCY BAND ===")
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_e = float(np.sum(avg_spec[mask] ** 2))
        pct = 100.0 * band_e / total_energy if total_energy > 0 else 0.0
        print(f"  {lo:5d} – {hi:5d} Hz : {pct:6.2f}%")
    print()

    # Windowed analysis — check per-window RMS
    win_samples = int(0.15 / dt)
    hop = win_samples // 2
    n_windows = 0
    above_001 = 0
    above_0005 = 0
    above_0001 = 0
    for start in range(0, traces.shape[1] - win_samples, hop):
        chunk = traces[:, start:start + win_samples]
        w_rms = float(np.sqrt(np.mean(chunk ** 2)))
        n_windows += 1
        if w_rms > 0.01:
            above_001 += 1
        if w_rms > 0.0005:
            above_0005 += 1
        if w_rms > 0.0001:
            above_0001 += 1

    print("=== WINDOWED RMS ANALYSIS (150 ms windows) ===")
    print(f"  Total windows: {n_windows}")
    print(f"  RMS > 0.01:   {above_001} ({100*above_001/n_windows:.0f}%)")
    print(f"  RMS > 0.0005: {above_0005} ({100*above_0005/n_windows:.0f}%)")
    print(f"  RMS > 0.0001: {above_0001} ({100*above_0001/n_windows:.0f}%)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sim_dir",
        type=Path,
        help="Simulation output directory containing traces.npy and metadata.json",
    )
    args = parser.parse_args()

    analyse(args.sim_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()

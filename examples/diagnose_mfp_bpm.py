#!/usr/bin/env python3
"""Deeper diagnostic: check if beam-power map has ANY signal near truth.

Prints beam power at MFP peak vs beam power at ground-truth bearing.
Also tests with wider bandwidth for Doppler accommodation.

Usage:
    python examples/diagnose_mfp_bpm.py output/valley_3d_test
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acoustic_sim.processor_3d import (
    build_3d_grid,
    broadband_weighted_sum_3d,
    compute_csdm,
    compute_steering_vectors_3d,
    compute_travel_times_3d,
    find_peaks_3d,
    mvdr_beam_power_3d,
    polar_to_cartesian,
    select_harmonic_bins,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir")
    parser.add_argument("--speed", type=float, default=50.0)
    parser.add_argument("--fundamental", type=float, default=180.0)
    parser.add_argument("--bandwidth", type=float, default=20.0)
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    traces = np.load(sim_dir / "traces.npy")
    with open(sim_dir / "metadata.json") as f:
        meta = json.load(f)
    mic_pos = np.array(meta["receiver_positions"])

    n_mics, n_samples = traces.shape
    dt = meta["dt"]
    fs = 1.0 / dt

    cx, cy = float(np.mean(mic_pos[:, 0])), float(np.mean(mic_pos[:, 1]))
    print(f"Array center: ({cx:.2f}, {cy:.2f}), mics={n_mics}, fs={fs:.0f}")

    # Source trajectory
    x0, y0, z0 = meta["source_x"], meta["source_y"], meta.get("source_z", 15.0)
    x1, y1, z1 = meta.get("source_x1", -x0), meta.get("source_y1", y0), meta.get("source_z1", z0)
    horiz_dist = math.hypot(x1 - x0, y1 - y0)
    duration = horiz_dist / args.speed

    def gt(t):
        frac = max(0.0, min(1.0, t / duration))
        return (x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac, z0 + (z1 - z0) * frac)

    # Build grid with fixed z
    azimuths, ranges, z_values = build_3d_grid(
        azimuth_spacing_deg=1.0,
        range_min=5.0, range_max=100.0, range_spacing=2.0,
        z_min=15.0, z_max=15.0, z_spacing=1.0,
    )
    gx, gy = polar_to_cartesian(azimuths, ranges, cx, cy)
    tt = compute_travel_times_3d(gx, gy, z_values, mic_pos)

    # Frequency setup
    win_sec = 0.2
    win_len = int(round(win_sec * fs))
    n_sub = 2
    sub_len = win_len // n_sub
    freq_bins = select_harmonic_bins(sub_len, dt, args.fundamental, 2, args.bandwidth)
    fft_freqs = np.fft.rfftfreq(sub_len, d=dt)
    sel_freqs = fft_freqs[freq_bins[freq_bins < len(fft_freqs)]]
    print(f"Bandwidth: ±{args.bandwidth} Hz → {len(sel_freqs)} freq bins")
    print(f"Selected freqs: {sel_freqs}")
    sv = compute_steering_vectors_3d(sel_freqs, tt)

    # Process each window
    hop = win_len // 2
    print(f"\n{'Win':>4} {'Time':>6} {'PeakBrg':>8} {'TrueBrg':>8} {'Err':>6} "
          f"{'PeakPow':>9} {'TruePow':>9} {'Ratio':>6}")
    print("-" * 70)

    pos = 0
    idx = 0
    while pos + win_len <= n_samples:
        t_center = (pos + win_len / 2.0) * dt
        window_rms = float(np.sqrt(np.mean(traces[:, pos:pos + win_len] ** 2)))

        if window_rms < 5e-5:
            pos += hop
            idx += 1
            continue

        csdm = compute_csdm(traces, dt, pos, win_len, freq_bins, n_sub)
        bp = mvdr_beam_power_3d(csdm, sv, epsilon=0.01)
        bpm = broadband_weighted_sum_3d(bp, sel_freqs)

        bpm_max = float(np.max(bpm))
        if bpm_max > 1e-30:
            bpm /= bpm_max

        # Peak location
        flat = np.argmax(bpm)
        ia, ir, iz = np.unravel_index(flat, bpm.shape)
        peak_brg = float(np.degrees(azimuths[ia]))
        peak_pow = bpm[ia, ir, iz]

        # Ground truth location in grid
        gt_x, gt_y, gt_z = gt(t_center)
        true_brg_rad = math.atan2(gt_y - cy, gt_x - cx)
        if true_brg_rad < 0:
            true_brg_rad += 2 * math.pi
        true_rng = math.hypot(gt_x - cx, gt_y - cy)
        true_brg = math.degrees(true_brg_rad)

        ia_true = int(np.argmin(np.abs(azimuths - true_brg_rad)))
        ir_true = int(np.argmin(np.abs(ranges - true_rng)))
        true_pow = bpm[ia_true, ir_true, 0]

        err = peak_brg - true_brg
        if err > 180:
            err -= 360
        if err < -180:
            err += 360

        ratio = peak_pow / max(true_pow, 1e-30)
        print(f"{idx:4d} {t_center:6.3f} {peak_brg:8.1f} {true_brg:8.1f} {err:+6.1f} "
              f"{peak_pow:9.4f} {true_pow:9.4f} {ratio:6.1f}")

        pos += hop
        idx += 1

    # Try wider bandwidth
    for bw in [40.0, 60.0]:
        freq_bins2 = select_harmonic_bins(sub_len, dt, args.fundamental, 2, bw)
        sel_freqs2 = fft_freqs[freq_bins2[freq_bins2 < len(fft_freqs)]]
        sv2 = compute_steering_vectors_3d(sel_freqs2, tt)

        # Process middle window
        mid_pos = n_samples // 2 - win_len // 2
        t_mid = (mid_pos + win_len / 2.0) * dt
        csdm = compute_csdm(traces, dt, mid_pos, win_len, freq_bins2, n_sub)
        bp = mvdr_beam_power_3d(csdm, sv2, epsilon=0.01)
        bpm = broadband_weighted_sum_3d(bp, sel_freqs2)
        bpm_max = float(np.max(bpm))
        if bpm_max > 1e-30:
            bpm /= bpm_max

        flat = np.argmax(bpm)
        ia, ir, iz = np.unravel_index(flat, bpm.shape)
        peak_brg = float(np.degrees(azimuths[ia]))

        gt_x, gt_y, gt_z = gt(t_mid)
        true_brg = math.degrees(math.atan2(gt_y - cy, gt_x - cx))
        if true_brg < 0:
            true_brg += 360.0
        err = peak_brg - true_brg
        if err > 180:
            err -= 360
        if err < -180:
            err += 360

        print(f"\nBW ±{bw:.0f}Hz ({len(sel_freqs2)} bins): mid-window peak_brg={peak_brg:.1f} "
              f"true_brg={true_brg:.1f} err={err:+.1f}°")

    # --- Try Conventional Beamforming (CBF) instead of MVDR ---
    print("\n\n=== Conventional Beamforming (CBF) ===")
    print(f"{'Win':>4} {'Time':>6} {'PeakBrg':>8} {'TrueBrg':>8} {'Err':>6}")
    print("-" * 45)

    pos = 0
    idx = 0
    while pos + win_len <= n_samples:
        t_center = (pos + win_len / 2.0) * dt
        window_rms = float(np.sqrt(np.mean(traces[:, pos:pos + win_len] ** 2)))
        if window_rms < 5e-5:
            pos += hop
            idx += 1
            continue

        csdm = compute_csdm(traces, dt, pos, win_len, freq_bins, n_sub)

        # CBF: power = a^H C a (no inversion)
        n_freq, n_az, n_range, n_gz, n_mic = sv.shape
        bp_cbf = np.zeros((n_freq, n_az, n_range, n_gz))
        for fi in range(n_freq):
            C = csdm[fi]
            for ia_i in range(n_az):
                for ir_i in range(n_range):
                    a = sv[fi, ia_i, ir_i, 0]
                    bp_cbf[fi, ia_i, ir_i, 0] = np.real(a.conj() @ C @ a)

        bpm_cbf = broadband_weighted_sum_3d(bp_cbf, sel_freqs)
        bpm_max = float(np.max(bpm_cbf))
        if bpm_max > 1e-30:
            bpm_cbf /= bpm_max

        flat = np.argmax(bpm_cbf)
        ia, ir, iz = np.unravel_index(flat, bpm_cbf.shape)
        peak_brg = float(np.degrees(azimuths[ia]))

        gt_x, gt_y, gt_z = gt(t_center)
        true_brg = math.degrees(math.atan2(gt_y - cy, gt_x - cx))
        if true_brg < 0:
            true_brg += 360.0

        err = peak_brg - true_brg
        if err > 180:
            err -= 360
        if err < -180:
            err += 360
        print(f"{idx:4d} {t_center:6.3f} {peak_brg:8.1f} {true_brg:8.1f} {err:+6.1f}")

        pos += hop
        idx += 1

    # --- Try far-field (plane-wave) beamforming ---
    print("\n\n=== Far-field (plane-wave) MVDR ===")
    azimuths_ff, ranges_ff, z_ff = build_3d_grid(
        azimuth_spacing_deg=1.0,
        range_min=500.0, range_max=500.0, range_spacing=1.0,
        z_min=15.0, z_max=15.0, z_spacing=1.0,
    )
    gx_ff, gy_ff = polar_to_cartesian(azimuths_ff, ranges_ff, cx, cy)
    tt_ff = compute_travel_times_3d(gx_ff, gy_ff, z_ff, mic_pos)
    sv_ff = compute_steering_vectors_3d(sel_freqs, tt_ff)

    print(f"{'Win':>4} {'Time':>6} {'PeakBrg':>8} {'TrueBrg':>8} {'Err':>6}")
    print("-" * 45)
    pos = 0
    idx = 0
    while pos + win_len <= n_samples:
        t_center = (pos + win_len / 2.0) * dt
        window_rms = float(np.sqrt(np.mean(traces[:, pos:pos + win_len] ** 2)))
        if window_rms < 5e-5:
            pos += hop
            idx += 1
            continue

        csdm = compute_csdm(traces, dt, pos, win_len, freq_bins, n_sub)
        bp = mvdr_beam_power_3d(csdm, sv_ff, epsilon=0.01)
        bpm = broadband_weighted_sum_3d(bp, sel_freqs)
        bpm_max = float(np.max(bpm))
        if bpm_max > 1e-30:
            bpm /= bpm_max

        flat = np.argmax(bpm)
        ia, ir, iz = np.unravel_index(flat, bpm.shape)
        peak_brg = float(np.degrees(azimuths_ff[ia]))

        gt_x, gt_y, gt_z = gt(t_center)
        true_brg = math.degrees(math.atan2(gt_y - cy, gt_x - cx))
        if true_brg < 0:
            true_brg += 360.0
        err = peak_brg - true_brg
        if err > 180:
            err -= 360
        if err < -180:
            err += 360
        print(f"{idx:4d} {t_center:6.3f} {peak_brg:8.1f} {true_brg:8.1f} {err:+6.1f}")

        pos += hop
        idx += 1

    # --- GCC-PHAT TDOA-based bearing for comparison ---
    print("\n\n=== GCC-PHAT TDOA (best mic pair per window) ===")
    print(f"{'Win':>4} {'Time':>6} {'GCC_brg':>8} {'TrueBrg':>8} {'Err':>6}")
    print("-" * 45)
    from scipy.signal import fftconvolve

    pos = 0
    idx = 0
    while pos + win_len <= n_samples:
        t_center = (pos + win_len / 2.0) * dt
        window_rms = float(np.sqrt(np.mean(traces[:, pos:pos + win_len] ** 2)))
        if window_rms < 5e-5:
            pos += hop
            idx += 1
            continue

        seg = traces[:, pos:pos + win_len]
        n_fft_gcc = 2 * seg.shape[1]

        # For each opposing mic pair, compute GCC-PHAT TDOA
        best_pairs = []
        for i in range(n_mics // 2):
            j = (i + n_mics // 2) % n_mics
            S1 = np.fft.rfft(seg[i], n=n_fft_gcc)
            S2 = np.fft.rfft(seg[j], n=n_fft_gcc)
            cross = S1 * S2.conj()
            denom = np.abs(cross)
            denom[denom < 1e-30] = 1e-30
            gcc = np.fft.irfft(cross / denom)
            max_lag = int(round(0.02 * fs))  # ±20ms max lag
            gcc_center = np.concatenate([gcc[-max_lag:], gcc[:max_lag + 1]])
            peak_idx = np.argmax(np.abs(gcc_center))
            tdoa = (peak_idx - max_lag) / fs
            # Bearing from TDOA: tdoa = baseline . k_hat / c
            b = mic_pos[j, :2] - mic_pos[i, :2]
            b_len = np.linalg.norm(b)
            if b_len < 1e-6:
                continue
            cos_angle = tdoa * 343.0 / b_len
            cos_angle = max(-1.0, min(1.0, cos_angle))
            # b_hat direction
            b_angle = math.atan2(b[1], b[0])
            # TDOA gives angle relative to baseline
            theta = math.acos(cos_angle)
            bearing1 = b_angle + theta
            bearing2 = b_angle - theta
            best_pairs.append((i, j, tdoa, b_angle, bearing1, bearing2, np.max(np.abs(gcc_center))))

        # Average bearings from all pairs using circular mean
        gt_x, gt_y, gt_z = gt(t_center)
        true_brg_rad = math.atan2(gt_y - cy, gt_x - cx)
        if true_brg_rad < 0:
            true_brg_rad += 2 * math.pi
        true_brg = math.degrees(true_brg_rad)

        # Pick bearing candidate closest to centroid for each pair, then average
        sin_sum, cos_sum = 0.0, 0.0
        for _, _, _, _, b1, b2, weight in best_pairs:
            # pick the candidate closest to running mean (start with b1)
            for b in [b1, b2]:
                sin_sum += weight * math.sin(b)
                cos_sum += weight * math.cos(b)
        if abs(sin_sum) + abs(cos_sum) > 1e-12:
            avg_brg = math.degrees(math.atan2(sin_sum, cos_sum))
            if avg_brg < 0:
                avg_brg += 360.0
        else:
            avg_brg = float("nan")

        err = avg_brg - true_brg
        if err > 180:
            err -= 360
        if err < -180:
            err += 360
        print(f"{idx:4d} {t_center:6.3f} {avg_brg:8.1f} {true_brg:8.1f} {err:+6.1f}")

        pos += hop
        idx += 1


if __name__ == "__main__":
    main()

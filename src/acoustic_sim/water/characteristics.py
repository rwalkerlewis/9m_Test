"""Measured characteristics of the tracked objects: what a harbor operator can read about each one.

No classifier: every quantity is a measurement on the stream's spectra, maps and tracks, with the frames it
was measured over.  The module reads only processing products (``streaming.FrameResult`` maps and spectra,
``RadiatorTrack`` fixes, the layout, the AIS reports), never the truth.

For every track:

* position and its uncertainty: the half maximum widths of the map peak that fed the latest fix (two
  principal axes, :func:`peak_shape`) and the rms residual of the fixes about the straight line fitted over
  the last ``fit_s`` seconds (:func:`kinematics`);
* velocity and heading from that fit, with their standard errors;
* the AIS report matched to it, if any, and the distance between the reported and the located position.

For radiator tracks (:func:`source_spectrum`, :func:`received_level`, :func:`tonal_lines`):

* the received level: the field at the array of the least squares source spectrum toward the track (every
  live channel, with the replica of each channel as the chain models it, so pressure and strain channels
  share one unit), rms in the band, at the median range from the track to the live channels;
* the source level at 100 m from it assuming cylindrical spreading, times one calibration constant;
* the strongest narrowband lines of that spectrum over its local floor.

For echo tracks (:func:`echo_strength`):

* the echo strength per illuminator: the least squares source spectrum at the echo's cell over the
  illuminator's direct field there (the echo chain's data reference, scaled to a source spectrum by the beam's
  gain, propagated to the cell), coherent over the band, in dB;
* a size indicator: that strength in the upper third of the band over the lower third, in dB;
* the extent of the echo peak at half maximum along and across the illuminator direction (:func:`peak_shape`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, median_filter

from acoustic_sim.water.ais import AISReports
from acoustic_sim.water.propagation import green2d
from acoustic_sim.water.sensors import SensorLayout

SMALL = "smaller than about a wavelength"
LARGE = "about a wavelength or larger"


@dataclass
class CharacteristicsConfig:
    """Parameters of the characteristics (processing side; part of the processing config).

    ``level_calibration``: the constant the source level estimate is multiplied by (calibrated once, on the
    simple harbor, against its known source levels); ``report_frames``: the reported level, lines and echo values
    are medians over the last this many frames a track was measured in; ``n_lines`` / ``line_min_db`` /
    ``line_floor_bins``: tonal lines reported, their minimum level over the local floor [dB] and the half width
    of the running median that is the floor [bins]; ``line_tolerance_bins``: a line persists while one is found
    within this many bins; ``size_threshold_db``: a size ratio above this is "smaller than about a wavelength";
    ``peak_radius_m``: the half maximum region of a map peak is searched within this distance of it;
    ``fit_s``: the window of the track fit [s]; ``merge_radius_m`` / ``merge_s``: a radiator track and an echo
    track within this distance for this long are one object seen both ways; ``ais_gate_m``: a track takes the
    identity of the reporting ship within this distance.
    """

    level_calibration: float = 1.0
    report_frames: int = 8
    n_lines: int = 3
    line_min_db: float = 6.0
    line_floor_bins: int = 10
    line_tolerance_bins: int = 1
    size_threshold_db: float = 0.0
    peak_radius_m: float = 20.0
    fit_s: float = 4.0
    merge_radius_m: float = 5.0
    merge_s: float = 2.0
    ais_gate_m: float = 25.0


# ----------------------------------------------------------------------
# Every track: kinematics and the shape of the map peak
# ----------------------------------------------------------------------

def kinematics(fixes, t: float, fit_s: float) -> dict:
    """Straight line fit to the fixes ``(t, xy, ...)`` of the last *fit_s* seconds before *t*.

    Returns the fitted position at *t*, the velocity, speed and heading (degrees clockwise from north) with
    their standard errors (from the scatter of the fixes about the line), the rms residual of the fixes about
    the line, and the window (first and last fix time, number of fixes).  With fewer than 3 fixes in the window
    the position is the last fix and the velocity is ``None``.
    """
    tt = np.array([f[0] for f in fixes], dtype=float)
    xy = np.array([f[1] for f in fixes], dtype=float).reshape(-1, 2)
    m = (tt >= tt[-1] - fit_s) & (tt <= t)
    n = int(m.sum())
    out = {"window_s": [float(tt[m][0]) if n else float(tt[-1]), float(tt[m][-1]) if n else float(tt[-1])], "n_fixes": n}
    if n < 3:
        out.update(xy=xy[-1].tolist(), velocity=None, speed=None, heading_deg=None, speed_se=None, heading_se_deg=None,
                   residual_m=None)
        return out
    A = np.column_stack([np.ones(n), tt[m] - t])
    coef, *_ = np.linalg.lstsq(A, xy[m], rcond=None)
    res = xy[m] - A @ coef
    dof = max(2 * n - 4, 1)
    s2 = float(np.sum(res ** 2)) / dof                          # per axis variance of a fix about the line
    se_v = float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1]))    # standard error of each velocity component
    v = coef[1]
    speed = float(np.hypot(*v))
    out.update(xy=coef[0].tolist(), velocity=v.tolist(), speed=speed,
               heading_deg=float(np.degrees(np.arctan2(v[0], v[1])) % 360.0),
               speed_se=se_v, heading_se_deg=float(np.degrees(se_v / speed)) if speed > 0 else None,
               residual_m=float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))))
    return out


def _grid_index(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    xs = np.unique(grid[:, 0])
    dx = float(np.min(np.diff(xs))) if len(xs) > 1 else 1.0
    return (np.rint((grid[:, 0] - grid[:, 0].min()) / dx).astype(int),
            np.rint((grid[:, 1] - grid[:, 1].min()) / dx).astype(int), dx)


def peak_shape(B: np.ndarray, grid: np.ndarray, xy, radius: float, direction=None) -> dict:
    """The half maximum region of the map peak at (or nearest to) *xy*: the cells within *radius* of it, at or
    above half its value and connected to it.

    Returns the full widths at half maximum along the region's two principal axes (``widths_m``, major first; for
    an elliptical region the full axis lengths, from its second moments, the cell size included), the direction of
    the major axis (degrees clockwise from north) and, with *direction* (a vector), the length of the region along
    and across it (``along_m``, ``across_m``: the span of the cell centres plus one cell).
    """
    grid = np.asarray(grid, dtype=float)
    B = np.asarray(B, dtype=float)
    ix, iy, dx = _grid_index(grid)
    d = np.hypot(*(grid - np.asarray(xy, float)).T)
    near = d <= radius
    g0 = int(np.flatnonzero(near)[np.argmax(B[near])]) if near.any() else int(np.argmin(d))
    peak = B[g0]
    sel = near & (B >= 0.5 * peak) if peak > 0 else np.zeros(len(B), bool)
    sel[g0] = True
    img = np.zeros((iy.max() + 1, ix.max() + 1), bool)
    img[iy[sel], ix[sel]] = True
    lab, _ = label(img, structure=np.ones((3, 3), bool))
    cells = np.flatnonzero(sel & (lab[iy, ix] == lab[iy[g0], ix[g0]]))
    pts = grid[cells]
    cov = np.cov(pts.T, bias=True) if len(pts) > 1 else np.zeros((2, 2))
    cov = cov + np.eye(2) * dx ** 2 / 12.0                     # each cell is a dx x dx square, not a point
    lam, vec = np.linalg.eigh(cov)
    widths = 4.0 * np.sqrt(np.maximum(lam[::-1], 0.0))          # an ellipse with semi-axis a has variance a^2 / 4
    major = vec[:, 1]
    out = {"xy": grid[g0].tolist(), "value": float(peak), "cells": int(len(cells)), "widths_m": widths.tolist(),
           "major_deg": float(np.degrees(np.arctan2(major[0], major[1])) % 180.0)}
    if direction is not None:
        u = np.asarray(direction, float)
        u = u / max(np.hypot(*u), 1e-12)
        w = np.array([-u[1], u[0]])
        rel = pts - grid[g0]
        out["along_m"] = float(np.ptp(rel @ u) + dx)
        out["across_m"] = float(np.ptp(rel @ w) + dx)
    return out


# ----------------------------------------------------------------------
# Radiator tracks: level and tonal lines
# ----------------------------------------------------------------------

def channel_replicas(layout: SensorLayout, xy, freqs: np.ndarray, c: float, min_range: float = 1.0) -> np.ndarray:
    """Replicas ``(F, N)`` of a unit source at *xy* at every channel as the chains model them: ``green2d`` at the
    layout's sample points, collapsed to channels (``SensorLayout.collapse``: the second difference on a strain
    cable)."""
    r = np.maximum(np.hypot(*(layout.sample_points - np.asarray(xy, float)).T), min_range)
    k = 2.0 * np.pi * np.asarray(freqs, float) / c
    return layout.collapse(green2d(k[:, None] * r[None, :]), np)


def source_spectrum(X: np.ndarray, weights: np.ndarray, layout: SensorLayout, xy, freqs: np.ndarray, c: float) -> np.ndarray:
    """Least squares source spectrum ``(F,)`` at *xy* from one frame's spectra *X* ``(N, F)`` (not weighted):
    ``sum_c w_c conj(a_c) X_c / sum_c w_c |a_c|^2`` over the channels with weight ``w_c > 0``."""
    a = channel_replicas(layout, xy, freqs, c)                  # (F, N)
    w = np.asarray(weights, float)[None, :]
    return np.sum(w * np.conj(a) * np.asarray(X).T, axis=1) / np.maximum(np.sum(w * np.abs(a) ** 2, axis=1), 1e-30)


def band_rms(spectrum: np.ndarray, nper: int, window: np.ndarray) -> float:
    """rms of the band limited signal whose one sided, windowed STFT values over the band are *spectrum*
    (Parseval for a frame of *nper* samples weighted by *window*: ``rms^2 = 2 sum |X_f|^2 / (nper sum w^2)``)."""
    return float(np.sqrt(2.0 * np.sum(np.abs(spectrum) ** 2) / (nper * np.sum(np.asarray(window) ** 2))))


def received_level(s: np.ndarray, freqs: np.ndarray, c: float, range_m: float, nper: int, window: np.ndarray) -> float:
    """rms in the band [trace units] of the field at *range_m* from a source of spectrum *s* (``green2d``)."""
    k = 2.0 * np.pi * np.asarray(freqs, float) / c
    return band_rms(s * green2d(k * max(range_m, 1.0)), nper, window)


def source_level_100m(level: float, range_m: float, calibration: float) -> float:
    """Source level at 100 m from a received level at *range_m* with cylindrical spreading (amplitude as
    ``1 / sqrt(r)``), times *calibration*."""
    return float(calibration * level * np.sqrt(max(range_m, 1.0) / 100.0))


def tonal_lines(power: np.ndarray, freqs: np.ndarray, n: int, min_db: float, floor_bins: int) -> list[dict]:
    """The *n* strongest narrowband lines of a power spectrum: local maxima at least *min_db* above the local
    floor (running median over ``2 floor_bins + 1`` bins), with their frequency, bin and level over the floor."""
    p = np.asarray(power, float)
    floor = np.maximum(median_filter(p, size=2 * floor_bins + 1, mode="nearest"), 1e-300)
    rel = 10.0 * np.log10(np.maximum(p, 1e-300) / floor)
    left, right = np.r_[-np.inf, p[:-1]], np.r_[p[1:], -np.inf]
    peaks = np.flatnonzero((p >= left) & (p > right) & (rel >= min_db))
    peaks = peaks[np.argsort(rel[peaks])[::-1]][:n]
    return [{"f_hz": float(freqs[i]), "bin": int(i), "level_db": float(rel[i])} for i in peaks]


def line_persistence(history: list[list[int]], line_bin: int, tolerance: int) -> int:
    """Frames of *history* (per frame, the bins of its lines) with a line within *tolerance* bins of *line_bin*."""
    return int(sum(any(abs(b - line_bin) <= tolerance for b in bins) for bins in history))


# ----------------------------------------------------------------------
# Echo tracks: echo strength and size indicator
# ----------------------------------------------------------------------

def beam_gain(layout: SensorLayout, weights: np.ndarray, xy, freqs: np.ndarray, c: float) -> np.ndarray:
    """Gain ``(F,)`` of the echo chain's data reference toward a source at *xy*: the mean over the live channels of
    each channel's replica advanced by its range (``bistatic.make_reference``), so reference = source spectrum x gain."""
    a = channel_replicas(layout, xy, freqs, c)                  # (F, N)
    live = np.asarray(weights) > 0
    k = 2.0 * np.pi * np.asarray(freqs, float) / c
    r = np.hypot(*(layout.receivers - np.asarray(xy, float)).T)
    return np.mean((a * np.exp(1j * k[:, None] * r[None, :]))[:, live], axis=1)


def echo_strength(X: np.ndarray, weights: np.ndarray, layout: SensorLayout, cell_xy, illuminator_xy, reference: np.ndarray,
                  freqs: np.ndarray, c: float) -> dict:
    """Echo strength of the cell *cell_xy* lit by the illuminator at *illuminator_xy*, from one frame of the echo
    chain: *X* ``(N, F)`` the cleaned spectra (weighted, as the echo chain forms them), *reference* ``(F,)`` that
    illuminator's data reference spectrum.

    ``Y_f``, the least squares source spectrum at the cell, against the illuminator's direct field there,
    ``D_f = green2d(k |cell - illuminator|) * reference_f / gain_f`` (:func:`beam_gain`): the coherent ratio
    ``|sum Y conj(D)| / sum |D|^2`` over the whole band, its lower and upper thirds, in dB, and the size ratio
    (upper over lower third, dB)."""
    freqs = np.asarray(freqs, float)
    a = channel_replicas(layout, cell_xy, freqs, c)             # (F, N)
    w = np.asarray(weights, float)[None, :]
    Y = np.sum(w * np.conj(a) * np.asarray(X).T, axis=1) / np.maximum(np.sum(w ** 2 * np.abs(a) ** 2, axis=1), 1e-30)
    k = 2.0 * np.pi * freqs / c
    path = green2d(k * max(float(np.hypot(*(np.asarray(cell_xy, float) - np.asarray(illuminator_xy, float)))), 1.0))
    D = path * np.asarray(reference) / beam_gain(layout, weights, illuminator_xy, freqs, c)
    thirds = np.array_split(np.arange(len(freqs)), 3)

    def ratio(sel) -> float:
        return float(20.0 * np.log10(max(abs(np.sum(Y[sel] * np.conj(D[sel]))) / max(np.sum(np.abs(D[sel]) ** 2), 1e-300), 1e-30)))

    full, lo, hi = ratio(slice(None)), ratio(thirds[0]), ratio(thirds[2])
    return {"strength_db": full, "lower_third_db": lo, "upper_third_db": hi, "size_ratio_db": hi - lo}


def size_class(ratio_db: float, threshold_db: float) -> str:
    """The size indicator in words: a body small against the wavelength scatters more at the top of the band."""
    return SMALL if ratio_db > threshold_db else LARGE


# ----------------------------------------------------------------------
# AIS
# ----------------------------------------------------------------------

def ais_match(fixes, reports: AISReports, t: float, gate: float, last: int = 8) -> dict:
    """The reporting ship whose reports (received by *t*, dead reckoned) lie within *gate* m of the track's last
    *last* fixes (median), and the distance between its reported and the located position at the latest fix;
    ``{"id": None}`` when no report matches ("no report")."""
    m = reports.t <= t
    rep = AISReports(reports.vessel[m], reports.t[m], reports.xy[m], reports.sog[m], reports.cog[m])
    best, dbest = None, gate
    for v in rep.vessels:
        at = rep.for_vessel(v)
        fx = [f for f in fixes[-last:] if f[0] >= at.t[0]]
        if not fx:
            continue
        d = float(np.median([np.hypot(*(at.predict(f[0], causal=True)[0] - f[1])) for f in fx]))
        if d < dbest:
            best, dbest = v, d
    if best is None:
        return {"id": None}
    at = rep.for_vessel(best)
    f = fixes[-1]
    return {"id": str(best), "distance_m": float(np.hypot(*(at.predict(f[0], causal=True)[0] - f[1]))), "t_fix": float(f[0])}

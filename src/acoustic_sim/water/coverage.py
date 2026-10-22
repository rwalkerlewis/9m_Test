"""What a sensor layout can see before any processing runs: coverage maps from the chart, the layout and a propagator.

Processing side (it reads the public chart, the layout and a free field propagator, never the truth).  For every cell
of a grid:

* :func:`strain_coverage`: the fraction of cable channels whose response to a wave from the cell exceeds 0.5 (a strain
  channel responds as the square of the cosine of the angle between the cell to channel direction and the fibre
  tangent, a pressure channel responds to every direction), and the same with every channel weighted by cylindrical
  spreading from the cell (``1 / sqrt(r)``);
* :func:`aperture`: the angular spread [degrees] of the sensors that see the cell (response above 0.5; hydrophones
  always), 360 minus the largest gap between their bearings from the cell;
* :func:`shadowing`: the fraction of sensors whose straight path from the cell does not cross land;
* :func:`predicted_resolution`: the full widths at half maximum, along and across the dominant direction, of the
  layout's free field matched field response to a point source at the cell at one frequency (the replica against
  itself), and the direction of the major axis.

``python -m acoustic_sim.water.coverage`` is not provided: ``examples/harbor/07_layout_study/`` draws the maps.
"""

from __future__ import annotations

import numpy as np

from acoustic_sim.water.propagation import green2d
from acoustic_sim.water.sensors import SensorLayout


def channel_responses(layout: SensorLayout, cells: np.ndarray) -> np.ndarray:
    """Response (G, N) of every receiver to a plane wave arriving from each cell: ``cos^2`` of the angle between the
    cell to channel direction and the fibre tangent for a strain channel, 1 for a pressure channel and a hydrophone."""
    cells = np.atleast_2d(np.asarray(cells, dtype=float))
    rec = layout.receivers
    out = np.ones((len(cells), len(rec)))
    k = 0
    for cable in layout.cables:
        n = cable.n_channels
        if cable.is_strain:
            d = rec[None, k:k + n, :] - cells[:, None, :]
            r = np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-9)
            cos = (d[..., 0] * cable.tangents[None, :, 0] + d[..., 1] * cable.tangents[None, :, 1]) / r
            out[:, k:k + n] = cos ** 2
        k += n
    return out


def _cable_mask(layout: SensorLayout) -> np.ndarray:
    m = np.zeros(layout.n_receivers, bool)
    m[:layout.n_das] = True
    return m


def strain_coverage(layout: SensorLayout, cells: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """``(fraction, weighted)`` (G,): the fraction of the cable channels responding above *threshold* to a wave from
    each cell, and the same with every channel weighted by ``1 / sqrt(r)`` from the cell.  Zero without a cable."""
    cells = np.atleast_2d(np.asarray(cells, dtype=float))
    cab = _cable_mask(layout)
    if not cab.any():
        z = np.zeros(len(cells))
        return z, z.copy()
    resp = channel_responses(layout, cells)[:, cab]
    sees = resp > threshold
    r = np.maximum(np.hypot(*(layout.receivers[cab][None, :, :] - cells[:, None, :]).transpose(2, 0, 1)), 1.0)
    w = 1.0 / np.sqrt(r)
    return sees.mean(axis=1), (sees * w).sum(axis=1) / w.sum(axis=1)


def aperture(layout: SensorLayout, cells: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Angular spread [degrees] (G,) of the sensors that see each cell: 360 minus the largest gap between the
    bearings, from the cell, of the channels responding above *threshold* and of every hydrophone (0 when fewer than two
    see it)."""
    cells = np.atleast_2d(np.asarray(cells, dtype=float))
    resp = channel_responses(layout, cells)
    sees = resp > threshold
    sees[:, ~_cable_mask(layout)] = True
    rec = layout.receivers
    out = np.zeros(len(cells))
    for g, c in enumerate(cells):
        idx = np.flatnonzero(sees[g])
        if len(idx) < 2:
            continue
        b = np.sort(np.degrees(np.arctan2(rec[idx, 1] - c[1], rec[idx, 0] - c[0])) % 360.0)
        gaps = np.diff(np.concatenate([b, [b[0] + 360.0]]))
        out[g] = 360.0 - gaps.max()
    return out


def shadowing(layout: SensorLayout, cells: np.ndarray, env, step: float | None = None) -> np.ndarray:
    """Fraction (G,) of the sensors whose straight path from each cell stays in water (*env*: the chart, its
    ``water`` mask, ``extent`` and ``dx``); the path is sampled every *step* metres (default half a chart cell)."""
    cells = np.atleast_2d(np.asarray(cells, dtype=float))
    x0, x1, y0, y1 = env.extent
    dx = float(env.dx)
    step = step or 0.5 * dx
    rec = layout.receivers
    ny, nx = env.water.shape
    out = np.zeros(len(cells))
    for g, c in enumerate(cells):
        d = rec - c
        L = np.hypot(d[:, 0], d[:, 1])
        m = int(np.ceil(L.max() / step)) + 1
        s = np.linspace(0.0, 1.0, m)
        px = c[0] + s[None, :] * d[:, 0:1]
        py = c[1] + s[None, :] * d[:, 1:2]
        ix = np.clip(((px - x0) / dx).astype(int), 0, nx - 1)
        iy = np.clip(((py - y0) / dx).astype(int), 0, ny - 1)
        out[g] = float(np.mean(env.water[iy, ix].all(axis=1)))
    return out


def response_map(layout: SensorLayout, src: np.ndarray, pts: np.ndarray, freq: float, c: float) -> np.ndarray:
    """``|a(x)^H a(src)|^2 / (|a(x)|^2 |a(src)|^2)`` (P,) at the points *pts*: the free field matched field response
    of the layout at *freq* to a point source at *src* (replicas: ``green2d`` at the sample points, collapsed)."""
    k = 2.0 * np.pi * freq / c
    S = layout.sample_points

    def rep(p):
        r = np.maximum(np.hypot(*(np.atleast_2d(p)[:, None, :] - S[None, :, :]).transpose(2, 0, 1)), 1.0)
        return layout.collapse(green2d(k * r, np), np)

    a0 = rep(src)[0]
    A = rep(pts)
    num = np.abs(A @ np.conj(a0)) ** 2
    return num / np.maximum(np.sum(np.abs(A) ** 2, axis=1) * np.sum(np.abs(a0) ** 2), 1e-300)


def half_max_widths(values: np.ndarray, pts: np.ndarray, centre: np.ndarray, dx: float) -> dict:
    """Full widths at half maximum along the principal axes of the region at or above half of the response at
    *centre*, connected to it (on a regular grid of spacing *dx*): ``widths_m`` (major, minor), ``major_deg``
    (clockwise from north)."""
    from scipy.ndimage import label

    xs, ys = np.unique(pts[:, 0]), np.unique(pts[:, 1])
    img = np.zeros((len(ys), len(xs)))
    ix = np.rint((pts[:, 0] - xs[0]) / dx).astype(int)
    iy = np.rint((pts[:, 1] - ys[0]) / dx).astype(int)
    img[iy, ix] = values
    c = int(np.argmin(np.hypot(*(pts - centre).T)))
    lab, _ = label(img >= 0.5 * values[c])
    sel = lab[iy[c], ix[c]]
    yy, xx = np.nonzero(lab == sel)
    if len(xx) < 2:
        return {"widths_m": [dx, dx], "major_deg": 0.0}
    X = np.column_stack([xs[0] + xx * dx, ys[0] + yy * dx])
    C = np.cov(X.T) + dx * dx / 12.0 * np.eye(2)                    # the cell size included
    w, V = np.linalg.eigh(C)
    # an elliptical region of uniform fill: full axis length = 4 sqrt(eigenvalue)
    major = V[:, 1]
    return {"widths_m": [float(4 * np.sqrt(w[1])), float(4 * np.sqrt(w[0]))],
            "major_deg": float(np.degrees(np.arctan2(major[0], major[1])) % 180.0)}


def predicted_resolution(layout: SensorLayout, cells: np.ndarray, freq: float, c: float, half: float = 40.0,
                         dx: float = 1.0) -> np.ndarray:
    """``(G, 3)``: the full widths at half maximum (major, minor) [m] of the layout's response to a point source at
    each cell (:func:`response_map` on a window of half width *half* and spacing *dx*) and the direction of the major
    axis (degrees clockwise from north)."""
    cells = np.atleast_2d(np.asarray(cells, dtype=float))
    g = np.arange(-half, half + 0.5 * dx, dx)
    X, Y = np.meshgrid(g, g)
    off = np.column_stack([X.ravel(), Y.ravel()])
    out = np.zeros((len(cells), 3))
    for i, c0 in enumerate(cells):
        pts = c0[None, :] + off
        v = response_map(layout, c0, pts, freq, c)
        hw = half_max_widths(v, pts, c0, dx)
        out[i] = [hw["widths_m"][0], hw["widths_m"][1], hw["major_deg"]]
    return out


def coverage_maps(layout: SensorLayout, cells: np.ndarray, env, freq: float, c: float, res_cells: np.ndarray | None = None
                  ) -> dict:
    """Every quantity of the module docstring on *cells* (the predicted resolution on *res_cells*, default *cells*)."""
    frac, weighted = strain_coverage(layout, cells)
    rc = cells if res_cells is None else res_cells
    res = predicted_resolution(layout, rc, freq, c)
    return {"cells": cells, "strain_fraction": frac, "strain_weighted": weighted, "aperture_deg": aperture(layout, cells),
            "shadowing": shadowing(layout, cells, env), "res_cells": rc, "resolution_major_m": res[:, 0],
            "resolution_minor_m": res[:, 1], "resolution_major_deg": res[:, 2]}

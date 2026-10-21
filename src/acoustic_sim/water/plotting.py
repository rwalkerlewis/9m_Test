"""Plots for the harbor model: the chart and sensors, and the geometry overlays of the result figures (ray paths,
equal delay ellipses, projected track positions, one frame's map with all of them)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from acoustic_sim.contract import ChartEnvironment
from acoustic_sim.water.sensors import SensorLayout


def plot_harbor(
    env: ChartEnvironment,
    layout: SensorLayout | None = None,
    ax: plt.Axes | None = None,
    show_depth: bool = True,
    sponge_cells: int | None = None,
):
    """Map of land (grey), water depth (blue), cables and hydrophones.

    *env* is the public chart (``ChartEnvironment``) or the forward ``HarborEnvironment``.

    *sponge_cells* optionally outlines the absorbing grid-edge layer.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    x0, x1, y0, y1 = env.extent

    if show_depth:
        depth = np.ma.masked_where(env.land, env.depth)
        im = ax.imshow(depth, origin="lower", extent=(x0, x1, y0, y1), cmap="Blues_r", vmin=0, vmax=25)
        plt.colorbar(im, ax=ax, label="depth [m]", shrink=0.7)
    ax.imshow(
        np.where(env.land, 1.0, np.nan), origin="lower", extent=(x0, x1, y0, y1),
        cmap=ListedColormap(["0.55"]), vmin=0, vmax=1,
    )

    if sponge_cells:
        w = sponge_cells * env.dx
        ax.plot([x0 + w, x1 - w, x1 - w, x0 + w, x0 + w], [y0 + w, y0 + w, y1 - w, y1 - w, y0 + w],
                "k:", lw=0.8, label="sponge edge")

    if layout is not None:
        for cable in layout.cables:
            p = cable.positions
            ax.plot(p[:, 0], p[:, 1], "-", color="orange", lw=2, label=f"{cable.name} ({cable.n_channels} ch)")
            ax.plot(p[::10, 0], p[::10, 1], ".", color="red", ms=3)
        if layout.n_hydrophones:
            h = layout.hydrophones
            ax.plot(h[:, 0], h[:, 1], "v", color="k", mfc="yellow", ms=9, label="hydrophone")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(env.name)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    return ax


# ----------------------------------------------------------------------
# Geometry overlays: ray paths, bistatic ellipses, projected positions
# ----------------------------------------------------------------------

def sensor_subset(layout: SensorLayout, n_das: int = 8) -> np.ndarray:
    """A few sensors to draw rays to: *n_das* channels spread along the DAS cables, and every hydrophone."""
    das = layout.das_positions
    pick = das[np.linspace(0, len(das) - 1, min(n_das, len(das))).round().astype(int)] if len(das) else np.empty((0, 2))
    return np.vstack([pick, layout.hydrophones]) if layout.n_hydrophones else pick


def draw_direct_rays(ax, source_xy, sensors: np.ndarray, color: str = "#1f77b4", label: str | None = "direct path",
                     **kw) -> None:
    """Straight rays from a radiator to *sensors* (the free field direct paths the replicas model)."""
    s = np.asarray(source_xy, dtype=float)
    for i, p in enumerate(np.atleast_2d(sensors)):
        ax.plot([s[0], p[0]], [s[1], p[1]], "-", color=color, lw=0.7, alpha=0.55, label=label if i == 0 else None, **kw)


def draw_echo_rays(ax, source_xy, body_xy, sensors: np.ndarray, color: str = "#ff7f0e",
                   label: str | None = "echo path (source - body - sensor)") -> None:
    """The bistatic path of an echo: source to body (solid), body to *sensors* (dashed)."""
    s, b = np.asarray(source_xy, dtype=float), np.asarray(body_xy, dtype=float)
    ax.plot([s[0], b[0]], [s[1], b[1]], "-", color=color, lw=1.4, alpha=0.9, label=label)
    for p in np.atleast_2d(sensors):
        ax.plot([b[0], p[0]], [b[1], p[1]], "--", color=color, lw=0.7, alpha=0.6)


def bistatic_ellipse(source_xy, receiver_xy, through_xy, n: int = 361) -> np.ndarray:
    """Points (n, 2) of the ellipse with foci *source_xy* and *receiver_xy* through *through_xy*: every point on it has
    the same bistatic range ``|x - s| + |x - r|``, so an echo from any of them arrives with the same delay."""
    s, r, b = (np.asarray(v, dtype=float) for v in (source_xy, receiver_xy, through_xy))
    L = np.hypot(*(b - s)) + np.hypot(*(b - r))
    c = 0.5 * (s + r)
    f = 0.5 * np.hypot(*(r - s))
    a = 0.5 * L
    bb = np.sqrt(max(a * a - f * f, 0.0))
    ang = np.arctan2(*(r - s)[::-1])
    u = np.linspace(0.0, 2.0 * np.pi, n)
    x, y = a * np.cos(u), bb * np.sin(u)
    return np.column_stack([c[0] + x * np.cos(ang) - y * np.sin(ang), c[1] + x * np.sin(ang) + y * np.cos(ang)])


def draw_bistatic_ellipse(ax, source_xy, receiver_xy, through_xy, color: str = "#ff7f0e",
                          label: str | None = "equal delay ellipse (source, array centre)") -> None:
    e = bistatic_ellipse(source_xy, receiver_xy, through_xy)
    ax.plot(e[:, 0], e[:, 1], ":", color=color, lw=1.1, alpha=0.9, label=label)


def projected_positions(fixes, t_now: float, horizons=(0.5, 1.0, 2.0), fit_s: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Positions a track projects ahead of *t_now*: the straight line fitted to its fixes ``(t, xy, ...)`` of the past
    *fit_s* seconds (as ``RadiatorTrack.predict``) evaluated at ``t_now + horizons``.  Returns (times, xy)."""
    tt = np.array([f[0] for f in fixes if f[0] <= t_now])
    xy = np.array([f[1] for f in fixes if f[0] <= t_now]).reshape(-1, 2)
    times = t_now + np.asarray(horizons, dtype=float)
    if len(tt) == 0:
        return times, np.full((len(times), 2), np.nan)
    m = tt >= tt[-1] - fit_s
    if m.sum() < 3:
        return times, np.repeat(xy[-1:], len(times), axis=0)
    A = np.column_stack([np.ones(m.sum()), tt[m] - tt[-1]])
    coef, *_ = np.linalg.lstsq(A, xy[m], rcond=None)
    return times, coef[0][None, :] + np.outer(times - tt[-1], coef[1])


def draw_projection(ax, fixes, t_now: float, horizons=(0.5, 1.0, 2.0), fit_s: float = 4.0, color: str = "k",
                    label: str | None = "projected (line fit, +0.5/1/2 s)") -> np.ndarray:
    """The track's fixes up to *t_now* (thin line), its position at *t_now* (dot) and the positions it projects ahead
    (open markers joined by an arrow).  Returns the projected positions."""
    past = np.array([f[1] for f in fixes if f[0] <= t_now]).reshape(-1, 2)
    if not len(past):
        return np.empty((0, 2))
    _, proj = projected_positions(fixes, t_now, horizons, fit_s)
    ax.plot(past[:, 0], past[:, 1], "-", color=color, lw=1.2, alpha=0.7)
    ax.plot(*past[-1], "o", color=color, ms=5)
    ax.plot(proj[:, 0], proj[:, 1], "o", mfc="none", mec=color, ms=6, label=label)
    ax.annotate("", proj[-1], past[-1], arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    return proj


def grid_image(grid: np.ndarray, values: np.ndarray):
    """Scattered values on a regular (possibly land masked) grid of cell centres as an image (NaN off the grid) and
    its extent, for ``imshow(origin="lower")``."""
    xs, ys = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    dx = float(np.min(np.diff(xs))) if len(xs) > 1 else 1.0
    img = np.full((int(round((ys[-1] - ys[0]) / dx)) + 1, int(round((xs[-1] - xs[0]) / dx)) + 1), np.nan)
    img[np.rint((grid[:, 1] - ys[0]) / dx).astype(int), np.rint((grid[:, 0] - xs[0]) / dx).astype(int)] = values
    return img, (xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dx / 2, ys[-1] + dx / 2)


def draw_frame_geometry(ax, grid: np.ndarray, B: np.ndarray, layout: SensorLayout, radiators: dict | None = None,
                        peaks: np.ndarray | None = None, echo: bool = True, threshold: float | None = None,
                        exclusion: np.ndarray | None = None):
    """One frame's map (dB re its peak) with the geometry that produced it: every radiator in *radiators* (id ->
    position the chain used) with its direct rays to a few sensors; for an echo map, the path radiator - strongest
    peak - sensors and the equal delay ellipse through that peak (foci: the radiator and the array centre);
    *peaks* (P, 2) marked, strongest in red; *threshold* drawn as a contour of the map; *exclusion* (G,) cells
    hatched.  Returns the image handle (for a colour bar)."""
    radiators = radiators or {}
    B = np.asarray(B, dtype=float)
    img, ext = grid_image(grid, 10 * np.log10(np.maximum(B / max(B.max(), 1e-30), 1e-3)))
    im = ax.imshow(img, origin="lower", extent=ext, cmap="viridis", vmin=-20, vmax=0, alpha=0.85)
    if threshold is not None and B.max() > threshold:
        lin, ext2 = grid_image(grid, B)
        xs = np.linspace(ext2[0], ext2[1], lin.shape[1] + 1)
        ys = np.linspace(ext2[2], ext2[3], lin.shape[0] + 1)
        ax.contour(0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1]), np.nan_to_num(lin, nan=0.0), levels=[threshold],
                   colors="w", linewidths=1.0)
        ax.plot([], [], "-", color="w", lw=1.0, label="threshold contour")
    if exclusion is not None and np.any(exclusion):
        ex = grid[np.asarray(exclusion, bool)]
        ax.plot(ex[:, 0], ex[:, 1], ",", color="0.85", alpha=0.6)
        ax.plot([], [], "s", color="0.85", ms=5, label="exclusion zone (map zeroed)")
    sensors = sensor_subset(layout)
    centre = np.mean(layout.receivers, axis=0)
    for j, (k, p) in enumerate(radiators.items()):
        draw_direct_rays(ax, p, sensors, color="#1f77b4", label="direct paths (radiator - sensors)" if j == 0 else None)
        ax.plot(*p, "*", color="#1f77b4", ms=14, mec="w", label="radiator where the chain placed it" if j == 0 else None)
        ax.annotate(str(k), p, xytext=(6, 6), textcoords="offset points", fontsize=8, color="#1f77b4")
    if peaks is not None and len(peaks):
        peaks = np.atleast_2d(peaks)
        if echo and radiators:
            src = next(iter(radiators.values()))
            draw_echo_rays(ax, src, peaks[0], sensors, color="#d62728", label="echo path (radiator - peak - sensors)")
            draw_bistatic_ellipse(ax, src, centre, peaks[0], color="#d62728", label="equal delay ellipse through the peak")
        ax.plot(peaks[1:, 0], peaks[1:, 1], "x", color="w", ms=8, mew=2, label="other peaks")
        ax.plot(*peaks[0], "x", color="#d62728", ms=12, mew=2.5, label="strongest peak")
    ax.plot(*centre, "+", color="k", ms=10, mew=2, label="array centre")
    return im

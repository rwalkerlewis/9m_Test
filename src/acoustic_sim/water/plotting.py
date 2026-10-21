"""Plots for the harbor model: the chart and sensors, and the geometry overlays of the result figures (ray paths,
equal delay ellipses, projected track positions, one frame's map with all of them)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from acoustic_sim.contract import ChartEnvironment
from acoustic_sim.plotting import new_figure, save_figure
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
        _, ax = new_figure()
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

    ax.set_xlabel("x (east) [m]")
    ax.set_ylabel("y (north) [m]")
    ax.set_title(env.name)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    return ax


def declutter_labels(ax, annotations, gap_px: float = 2.0, max_steps: int = 60) -> None:
    """Move text annotations (offsets in points) so that no two overlap: each label, in order of its height on the
    image, is shifted down until its box clears every label placed before it."""
    anns = [a for a in annotations if a is not None and a.get_text()]
    if len(anns) < 2:
        return
    fig = ax.figure
    try:
        fig.tight_layout()                              # the layout the figure is saved with
    except Exception:
        pass
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt_per_px = 72.0 / fig.dpi
    boxes = []
    for a in sorted(anns, key=lambda a: -a.get_window_extent(renderer).y1):
        for _ in range(max_steps):
            bb = a.get_window_extent(renderer)
            hit = next((b for b in boxes if bb.x0 < b.x1 and bb.x1 > b.x0 and bb.y0 < b.y1 + gap_px and bb.y1 > b.y0 - gap_px), None)
            if hit is None:
                break
            dx, dy = a.xyann
            a.xyann = (dx, dy - (bb.y1 - hit.y0 + gap_px) * pt_per_px)
        boxes.append(a.get_window_extent(renderer))


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


def draw_frame_map(ax, grid: np.ndarray, B: np.ndarray, peaks: np.ndarray | None = None, threshold: float | None = None,
                   exclusion: np.ndarray | None = None):
    """One frame's map (dB re its peak): *peaks* (P, 2) marked, strongest in red; *threshold* drawn as a contour of the
    map; *exclusion* (G,) cells marked.  Returns the image handle (for a colour bar)."""
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
    _draw_peaks(ax, peaks)
    return im


def _draw_peaks(ax, peaks) -> None:
    if peaks is not None and len(peaks):
        peaks = np.atleast_2d(peaks)
        if len(peaks) > 1:
            ax.plot(peaks[1:, 0], peaks[1:, 1], "x", color="0.15", mec="w", ms=9, mew=2, label="other peaks")
        ax.plot(*peaks[0], "x", color="#d62728", ms=13, mew=2.5, label="strongest peak")


def _draw_radiators(ax, radiators: dict) -> None:
    for j, (k, p) in enumerate(radiators.items()):
        ax.plot(*p, "*", color="#1f77b4", ms=16, mec="w", label="radiator where the chain placed it" if j == 0 else None)
        ax.annotate(str(k), p, xytext=(7, 7), textcoords="offset points", color="#1f77b4")


def draw_frame_rays(ax, layout: SensorLayout, radiators: dict | None = None, peaks: np.ndarray | None = None,
                    echo: bool = True) -> None:
    """The ray paths of one frame: every radiator in *radiators* (id -> position the chain used) with its direct rays to
    a few sensors; for an echo map, the path radiator - strongest peak - sensors."""
    radiators = radiators or {}
    sensors = sensor_subset(layout)
    for j, p in enumerate(radiators.values()):
        draw_direct_rays(ax, p, sensors, color="#1f77b4", label="direct paths (radiator - sensors)" if j == 0 else None)
    _draw_radiators(ax, radiators)
    if peaks is not None and len(peaks):
        peaks = np.atleast_2d(peaks)
        if echo and radiators:
            draw_echo_rays(ax, next(iter(radiators.values())), peaks[0], sensors, color="#d62728",
                           label="echo path (radiator - peak - sensors)")
        ax.plot(*peaks[0], "x", color="#d62728", ms=13, mew=2.5, label="strongest peak")
    ax.plot(sensors[:, 0], sensors[:, 1], "o", color="0.2", mfc="none", ms=7, label="sensors the rays are drawn to")


def draw_frame_ellipse(ax, layout: SensorLayout, radiators: dict | None = None, peaks: np.ndarray | None = None) -> None:
    """The equal delay ellipse of one echo frame: foci the first radiator and the array centre, through the strongest
    peak (an echo from anywhere on it arrives with the same delay)."""
    radiators = radiators or {}
    centre = np.mean(layout.receivers, axis=0)
    _draw_radiators(ax, radiators)
    if peaks is not None and len(peaks) and radiators:
        peaks = np.atleast_2d(peaks)
        draw_bistatic_ellipse(ax, next(iter(radiators.values())), centre, peaks[0], color="#d62728",
                              label="equal delay ellipse through the peak")
        ax.plot(*peaks[0], "x", color="#d62728", ms=13, mew=2.5, label="strongest peak")
    ax.plot(*centre, "+", color="k", ms=12, mew=2, label="array centre")


def draw_frame_geometry(ax, grid: np.ndarray, B: np.ndarray, layout: SensorLayout, radiators: dict | None = None,
                        peaks: np.ndarray | None = None, echo: bool = True, threshold: float | None = None,
                        exclusion: np.ndarray | None = None):
    """One frame's map with all of its geometry in one axes (:func:`draw_frame_map`, :func:`draw_frame_rays`, and for an
    echo map :func:`draw_frame_ellipse`).  The figures of the repository draw the three on their own; this stays for
    callers that want them together.  Returns the image handle (for a colour bar)."""
    im = draw_frame_map(ax, grid, B, None, threshold, exclusion)
    draw_frame_rays(ax, layout, radiators, None, echo)
    centre = np.mean(layout.receivers, axis=0)
    if echo and radiators and peaks is not None and len(peaks):
        draw_bistatic_ellipse(ax, next(iter(radiators.values())), centre, np.atleast_2d(peaks)[0], color="#d62728",
                              label="equal delay ellipse through the peak")
    ax.plot(*centre, "+", color="k", ms=12, mew=2, label="array centre")
    _draw_peaks(ax, peaks)
    return im


class HarborPictureView:
    """The operator view: one figure updated per picture record (``water/stream.py``), the chart and sensors with every
    object as a marker (circle: seen by its own noise, square: by its echo, diamond: both), its half maximum
    ellipse, a velocity vector (5 s ahead) and its label (AIS identity, else "radiator" or "body"; size indicator;
    source level); tentative tracks as small dots, lost tracks faded.  The truth (``Truth``) is drawn, dashed, only
    when given.  :meth:`update` takes a record (live with ``show``), :meth:`save` writes the video and one frame."""

    def __init__(self, env: ChartEnvironment, layout: SensorLayout, title: str = "", truth=None, show: bool = False,
                 lead_s: float = 5.0) -> None:
        self.env, self.layout, self.title, self.truth, self.show, self.lead_s = env, layout, title, truth, show, lead_s
        self.records: list[dict] = []
        self.fig = self.ax = None
        if show:
            plt.ion()
            self._setup()

    def _setup(self) -> None:
        self.fig, self.ax = new_figure((12.0, 10.0))

    def update(self, rec: dict) -> None:
        self.records.append(rec)
        if self.show:
            self.draw(rec)
            plt.pause(0.001)

    def draw(self, rec: dict) -> None:
        from matplotlib.patches import Ellipse

        if self.fig is None:
            self._setup()
        ax = self.ax
        ax.clear()
        env = self.env
        x0, x1, y0, y1 = env.extent
        ax.imshow(np.where(env.water, 1.0, 0.0), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys_r", vmin=-0.8, vmax=1.0)
        das = self.layout.das_positions
        ax.plot(das[:, 0], das[:, 1], "-", color="#555555", lw=1.8)
        if self.layout.n_hydrophones:
            ax.plot(self.layout.hydrophones[:, 0], self.layout.hydrophones[:, 1], "^", color="#555555", ms=6)
        t = rec["t"]
        anns = []
        if self.truth is not None:
            for oid, tr in self.truth.tracks.items():
                col = "#1f77b4" if tr.level_pa > 0 else "#ff7f0e"
                ax.plot(tr.xy[:, 0], tr.xy[:, 1], "--", color=col, lw=0.8, alpha=0.6)
                ax.plot(*tr.position_at(t)[0], "o", ms=12, mfc="none", mec=col, ls="--", alpha=0.7)
        for o in rec["objects"]:
            xy = np.asarray(o["position"]["xy"], float)
            if o["status"] == "tentative":
                ax.plot(*xy, ".", color="0.45", ms=3)
                continue
            alpha = 0.3 if o["status"] == "lost" else 1.0
            col = {"own noise": "#1f77b4", "echo": "#d62728", "both": "#9467bd"}[o["seen"]]
            mk = {"own noise": "o", "echo": "s", "both": "D"}[o["seen"]]
            ax.plot(*xy, mk, color=col, ms=7, mec="k", alpha=alpha)
            w = o["position"].get("halfmax_width_m")
            if w:
                ax.add_patch(Ellipse(xy, w[0], w[1], angle=90.0 - o["position"]["halfmax_major_deg"], fill=False, ec=col,
                                     lw=1.0, alpha=alpha))
            v = o["velocity"]["vxy"]
            if v is not None:
                ax.annotate("", xy + self.lead_s * np.asarray(v), xy, arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=alpha))
            name = o["ais"]["id"] or ("radiator" if o["seen"] != "echo" else "body")
            lines = [f"{name} ({o['id']})" + (", static" if o["status"] == "static" else "") + (", lost" if o["status"] == "lost" else "")]
            if "echo" in o:
                lines.append(f"{o['echo']['size']} ({o['echo']['size_ratio_db']:+.1f} dB)")
            if "radiator" in o and o["radiator"]["source_level_100m_pa"] is not None:
                lines.append(f"source level {o['radiator']['source_level_100m_pa']:.2f} Pa at 100 m")
            anns.append(ax.annotate("\n".join(lines), xy, xytext=(8, 6), textcoords="offset points", fontsize=11, color=col,
                                    alpha=alpha, bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none", alpha=0.6 * alpha)))
        r = self.layout.positions()                             # the area the sensors cover, with a margin
        m = 0.08 * max(np.ptp(r[:, 0]), np.ptp(r[:, 1]))
        ax.set_xlim(max(x0, r[:, 0].min() - m), min(x1, r[:, 0].max() + m))
        ax.set_ylim(max(y0, r[:, 1].min() - m), min(y1, r[:, 1].max() + m))
        ax.set_aspect("equal")
        ax.set_xlabel("x (east) [m]")
        ax.set_ylabel("y (north) [m]")
        ax.plot([], [], "o", color="#1f77b4", mec="k", label="seen by its own noise")
        ax.plot([], [], "s", color="#d62728", mec="k", label="seen by its echo")
        ax.plot([], [], "D", color="#9467bd", mec="k", label="seen both ways")
        ax.plot([], [], ".", color="0.45", label="tentative track")
        ax.plot([], [], "-", color="#555555", lw=1.8, label="DAS cable")
        if self.layout.n_hydrophones:
            ax.plot([], [], "^", color="#555555", ms=6, ls="none", label="hydrophones")
        if self.truth is not None:
            ax.plot([], [], "--", color="0.4", label="true positions (for comparison)")
        ax.legend(fontsize=11, loc="lower left")
        ax.set_title(f"{self.title}: operator picture at t = {t:.2f} s, latency {rec['latency_s']:.2f} s\nellipses: half maximum "
                     f"width of the map peak; arrows: {self.lead_s:g} s ahead", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        declutter_labels(ax, anns)

    def save(self, path, frame_png=None, fps: float = 8.0, frame_at: float = 0.75) -> None:
        """Write every record as a video frame to *path* (``.mp4``, ffmpeg) and the record at *frame_at* of the run as
        a PNG to *frame_png*."""
        from matplotlib.animation import FFMpegWriter

        if not self.records:
            return
        if self.fig is None:
            self._setup()
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        with writer.saving(self.fig, str(path), dpi=100):
            for rec in self.records:
                self.draw(rec)
                writer.grab_frame()
        if frame_png is not None:
            self.draw(self.records[min(int(frame_at * len(self.records)), len(self.records) - 1)])
            save_figure(self.fig, frame_png, recording=self.title or None, close=False)
        plt.close(self.fig)
        self.fig = None

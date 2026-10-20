"""Plots for the harbor model."""

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

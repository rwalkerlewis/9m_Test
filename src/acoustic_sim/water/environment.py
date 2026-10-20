"""Synthetic 2-D harbor environment (plan view).

The environment is a regular grid holding a sound-speed field, a land mask
and a (purely informational) depth map.  Grid node ``(iy, ix)`` sits at
``(x[ix], y[iy])`` with ``x = ix * dx`` and ``y = iy * dx``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt

from acoustic_sim.contract import ChartEnvironment
from acoustic_sim.model import VelocityModel


@dataclass
class HarborEnvironment:
    """Sound speed, land mask and depth on a regular grid.

    Attributes
    ----------
    model : VelocityModel
        Sound-speed field [m/s] (``model.values`` has shape ``(ny, nx)``).
    water : np.ndarray
        Boolean ``(ny, nx)`` mask, True where there is water.
    depth : np.ndarray
        Water depth [m] (0 on land).  Informational only in the 2-D model.
    polygons : list[np.ndarray]
        Land polygons (vertex arrays) used to build the mask, kept for plotting.
    """

    model: VelocityModel
    water: np.ndarray
    depth: np.ndarray
    polygons: list[np.ndarray]
    name: str = "harbor"

    @property
    def dx(self) -> float:
        return float(self.model.dx)

    @property
    def nx(self) -> int:
        return self.model.nx

    @property
    def ny(self) -> int:
        return self.model.ny

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return self.model.extent

    @property
    def land(self) -> np.ndarray:
        return ~self.water

    def _index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        ix = np.clip(np.rint((xy[:, 0] - self.model.x[0]) / self.dx).astype(int), 0, self.nx - 1)
        iy = np.clip(np.rint((xy[:, 1] - self.model.y[0]) / self.dx).astype(int), 0, self.ny - 1)
        return iy, ix

    def distance_to_land(self, xy: np.ndarray) -> np.ndarray:
        """Distance [m] from each point to the nearest land node (nearest-node lookup)."""
        dist = distance_transform_edt(self.water) * self.dx
        return dist[self._index(xy)]

    def is_water(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """True where *xy* is inside the grid, in water and at least *margin* m from land."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        x0, x1, y0, y1 = self.extent
        inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
        ok = inside & self.water[self._index(xy)]
        if margin > 0:
            ok &= self.distance_to_land(xy) >= margin
        return ok

    def public(self) -> ChartEnvironment:
        """What a deployed system knows of this environment: the land / water chart, the depth and the
        median water sound speed as the nominal speed (no sound speed map, no solver parameters)."""
        return ChartEnvironment(water=self.water.copy(), depth=self.depth.copy(), x=np.asarray(self.model.x, dtype=np.float64).copy(),
                                y=np.asarray(self.model.y, dtype=np.float64).copy(), c_nominal=float(np.median(self.model.values[self.water])),
                                polygons=[np.asarray(p, float) for p in self.polygons], name=self.name)

    def fingerprint(self) -> str:
        """Stable hash of everything that affects wave propagation."""
        h = hashlib.sha1()
        h.update(np.float64(self.dx).tobytes())
        h.update(np.ascontiguousarray(self.model.values, dtype=np.float64).tobytes())
        h.update(np.packbits(self.water).tobytes())
        return h.hexdigest()


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def segment_polygon(p0, p1, width: float) -> np.ndarray:
    """Rectangle of the given *width* centred on the segment p0→p1."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / np.hypot(*d) * (width / 2.0)
    return np.array([p0 + n, p1 + n, p1 - n, p0 - n])


def rasterise_polygons(polygons: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean ``(ny, nx)`` mask, True inside any polygon."""
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = np.zeros(X.size, dtype=bool)
    for poly in polygons:
        mask |= MplPath(poly).contains_points(pts)
    return mask.reshape(X.shape)


# ----------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------

def _finish_environment(
    polygons: list[np.ndarray],
    width: float,
    height: float,
    dx: float,
    c0: float,
    shallow_c_drop: float,
    name: str,
) -> HarborEnvironment:
    nx = int(round(width / dx)) + 1
    ny = int(round(height / dx)) + 1
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx

    land = rasterise_polygons(polygons, x, y) if polygons else np.zeros((ny, nx), dtype=bool)
    water = ~land

    # Depth grows with distance from shore and saturates at 20 m.
    dist = distance_transform_edt(water) * dx
    depth = np.where(water, np.clip(3.0 + 0.05 * dist, 3.0, 20.0), 0.0)

    # Shallow water gets a slightly lower effective speed (mild refraction).
    c = c0 - shallow_c_drop * (1.0 - np.clip(depth / 20.0, 0.0, 1.0))
    c = np.where(water, c, c0)

    model = VelocityModel(x=x, y=y, values=c.astype(np.float64), dx=dx, dy=dx)
    return HarborEnvironment(model=model, water=water, depth=depth, polygons=polygons, name=name)


def uniform_water_environment(
    width: float = 1000.0,
    height: float = 750.0,
    dx: float = 5.0,
    c0: float = 1500.0,
) -> HarborEnvironment:
    """Open water with no land, used for validation."""
    return _finish_environment([], width, height, dx, c0, 0.0, "open_water")


def build_simple_harbor(
    width: float = 500.0,
    height: float = 400.0,
    dx: float = 1.0,
    c0: float = 1500.0,
    quay_width: float = 20.0,
) -> HarborEnvironment:
    """A harbor whose wavefields are easy to resolve: a rectangular basin, one straight quay along the north
    edge (``quay_width`` m of land) and open water on the other three sides (absorbed by the solver's sponge),
    with a constant sound speed *c0* everywhere (no shallow water slowdown, so the chart's single sound speed is
    the true one).  The only reflector is the quay face, one clean image of every source.

    Layout (metres, +y is north)::

        land (quay)                    y = height - quay_width .. height
        open water                     everywhere else
    """
    quay = np.array([(-1.0, height - quay_width), (width + 1.0, height - quay_width), (width + 1.0, height + 1.0),
                     (-1.0, height + 1.0)], dtype=float)
    return _finish_environment([quay], width, height, dx, c0, 0.0, "simple_harbor")


def harbor_polygons() -> list[np.ndarray]:
    """Land polygons of the synthetic harbor, in harbor coordinates [m]."""
    return [
        # North + west land mass (shoreline with a jagged quay)
        np.array([
            (0, 1800), (2400, 1800), (2400, 1560), (2000, 1500), (1650, 1470),
            (1250, 1500), (900, 1430), (550, 1470), (330, 1380), (300, 1000),
            (330, 700), (250, 500), (0, 450),
        ], dtype=float),
        # East land mass
        np.array([
            (2400, 1560), (2000, 1500), (2050, 1200), (2150, 900),
            (2200, 650), (2400, 600),
        ], dtype=float),
        # Piers jutting south from the north shore
        np.array([(800, 1180), (830, 1180), (830, 1440), (800, 1440)], dtype=float),
        np.array([(1500, 1200), (1530, 1200), (1530, 1470), (1500, 1470)], dtype=float),
        # Breakwater, west and east of the entrance
        segment_polygon((300, 560), (1250, 520), 40.0),
        segment_polygon((1450, 520), (2200, 620), 40.0),
    ]


def build_synthetic_harbor(
    dx: float = 5.0,
    c0: float = 1500.0,
    shallow_c_drop: float = 10.0,
    crop: tuple[float, float, float, float] | None = None,
) -> HarborEnvironment:
    """Synthetic 2400 m x 1800 m harbor: enclosed basin, breakwater, 200 m entrance.

    Layout (metres, +y is north)::

        north shore with two piers          y ~ 1450-1500
        basin  x 330-2000, y 560-1450
        breakwater at y ~ 520-620, entrance gap x 1250-1450
        open sea south of the breakwater and east of x ~ 2200

    *crop* = ``(x0, x1, y0, y1)`` cuts a rectangle out of the harbor and returns it as its own
    environment with the origin moved to ``(x0, y0)``, so a fine grid can be run on a part of the
    harbor (the bistatic scenario) with the same shoreline and piers.  Cuts through land polygons
    are handled by rasterising the shifted polygons on the smaller grid.
    """
    polygons = harbor_polygons()
    if crop is None:
        return _finish_environment(polygons, 2400.0, 1800.0, dx, c0, shallow_c_drop, "synthetic_harbor")
    x0, x1, y0, y1 = crop
    shifted = [poly - np.array([x0, y0]) for poly in polygons]
    env = _finish_environment(shifted, x1 - x0, y1 - y0, dx, c0, shallow_c_drop, "synthetic_harbor_crop")
    return env

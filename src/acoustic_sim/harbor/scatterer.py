"""Moving scattering bodies for the harbor FDTD.

A :class:`Scatterer` is a rectangular body (length along its heading, width across it)
that follows a :class:`~acoustic_sim.harbor.vessels.Track` and scatters the field of the
other sources.  It is imposed on the pressure field the same way the land mask is: after
every time step the pressure inside the body is multiplied by a factor.

* ``kind = "soft"``: factor 0 inside, a pressure release (air filled) body.  This is the
  right model for a hull whose interior is air at these wavelengths.
* ``kind = "absorbing"``: factor ``1 - strength`` inside, a lossy body; with
  ``strength`` near 1 it behaves like a soft body, with small values it is a weak
  scatterer.  Useful as a smooth alternative when a hard mask radiates.

The mask is **smooth at the body boundary**: the factor ramps from 0 to 1 over one cell
of signed distance, and it is recomputed every step from the exact (subcell) body
position.  A hard cell mask that jumps by whole cells as the body moves would zero the
pressure in each newly covered cell and release zero pressure cells behind it, which
radiates an impulse at every hop; the smooth mask changes by a small amount each step
instead.  ``tests/harbor/test_scatterer.py`` measures how much a moving body radiates
with no source present.

A rigid (Neumann) body is not provided: the solver is a pressure only scheme with
constant density, and a stable staircase Neumann condition was out of scope.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from acoustic_sim.harbor.vessels import Track

KINDS = ("soft", "absorbing", "hard")
SHAPES = ("rect", "ellipse")


@dataclass
class Scatterer:
    """One moving body.

    Attributes
    ----------
    id : str
        Name, e.g. ``"uuv1"``.
    track : Track
        Ground truth trajectory (position and heading over time).
    length, width : float
        Body size [m] along and across the heading.  Each should span at least two
        grid cells, or the body is barely represented.
    kind : str
        ``"soft"`` or ``"absorbing"``.
    strength : float
        For ``"absorbing"``: fraction of the pressure removed per step inside the body.
    shape : str
        ``"rect"`` (a hull) or ``"ellipse"`` (a circle when ``length == width``; used to
        validate against cylinder theory).
    smooth : bool
        Ramp the factor over one cell at the boundary (default).  ``False`` gives the hard
        cell mask, kept only to measure what it radiates when the body moves.
    """

    id: str
    track: Track
    length: float
    width: float
    kind: str = "soft"
    strength: float = 0.5
    shape: str = "rect"
    smooth: bool = True

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, got {self.shape!r}")
        if self.length <= 0 or self.width <= 0:
            raise ValueError("length and width must be positive")
        if not 0.0 < self.strength <= 1.0:
            raise ValueError("strength must be in (0, 1]")

    @property
    def equivalent_radius(self) -> float:
        """Radius [m] of the circle with the same area (for comparing with cylinder theory)."""
        return float(np.sqrt(self.length * self.width / np.pi))

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(self.kind.encode())
        h.update(self.shape.encode())
        h.update(b"smooth" if self.smooth else b"hard")
        h.update(np.array([self.length, self.width, self.strength], dtype=np.float64).tobytes())
        h.update(np.asarray(self.track.t, dtype=np.float64).tobytes())
        h.update(np.asarray(self.track.xy, dtype=np.float64).tobytes())
        return h.hexdigest()

    def pose_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Centre ``(x, y)`` and unit heading at time *t* (heading of the last motion when stopped)."""
        tr = self.track
        xy = tr.position_at(t)[0]
        dt = 0.5
        d = tr.position_at(t + dt)[0] - tr.position_at(t - dt)[0]
        n = float(np.hypot(*d))
        if n < 1e-6:
            # Stationary: use the overall direction of the track, or +x.
            d = tr.xy[-1] - tr.xy[0]
            n = float(np.hypot(*d))
            if n < 1e-6:
                return xy, np.array([1.0, 0.0])
        return xy, d / n


def body_factor(
    scat: Scatterer,
    centre: np.ndarray,
    heading: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
) -> tuple[slice, slice, np.ndarray] | None:
    """Multiplicative pressure factor of the body on the grid cells around it.

    Returns ``(rows, cols, factor)`` with ``factor`` of shape ``(len(rows), len(cols))``
    (1 outside the body, ``1 - strength`` or 0 inside, ramping over one cell), or
    ``None`` if the body is outside the grid.  *x*, *y* are the global grid coordinates.
    """
    half = 0.5 * (scat.length + scat.width) + 2.0 * dx
    ix0 = int(np.floor((centre[0] - half - x[0]) / dx))
    ix1 = int(np.ceil((centre[0] + half - x[0]) / dx)) + 1
    iy0 = int(np.floor((centre[1] - half - y[0]) / dx))
    iy1 = int(np.ceil((centre[1] + half - y[0]) / dx)) + 1
    ix0, ix1 = max(ix0, 0), min(ix1, len(x))
    iy0, iy1 = max(iy0, 0), min(iy1, len(y))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    X, Y = np.meshgrid(x[ix0:ix1] - centre[0], y[iy0:iy1] - centre[1])
    along = X * heading[0] + Y * heading[1]
    across = -X * heading[1] + Y * heading[0]
    if scat.shape == "ellipse":
        ra, rb = 0.5 * scat.length, 0.5 * scat.width
        # Exact for a circle; for an ellipse the scaled radial distance is a close approximation.
        rho = np.hypot(along / ra, across / rb)
        sdf = (rho - 1.0) * np.sqrt(ra * rb)
    else:
        qa = np.abs(along) - 0.5 * scat.length
        qb = np.abs(across) - 0.5 * scat.width
        outside = np.hypot(np.maximum(qa, 0.0), np.maximum(qb, 0.0))
        inside = np.minimum(np.maximum(qa, qb), 0.0)
        sdf = outside + inside                               # signed distance to the rectangle
    if scat.smooth:
        ramp = np.clip(sdf / dx + 0.5, 0.0, 1.0)             # 0 inside, 1 outside, one cell wide ramp
    else:
        ramp = (sdf > 0.0).astype(float)                     # hard cell mask
    strength = 1.0 if scat.kind in ("soft", "hard") else scat.strength
    factor = 1.0 - strength * (1.0 - ramp)
    return slice(iy0, iy1), slice(ix0, ix1), factor

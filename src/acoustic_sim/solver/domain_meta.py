"""Shared metadata describing per-cell properties supplied to the FDTD solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DomainMeta:
    """Per-cell 2-D physics that supplements a velocity model."""

    wind_vx: float = 0.0
    wind_vy: float = 0.0
    attenuation: np.ndarray | None = None
    description: str = ""


@dataclass
class DomainMeta3D:
    """Per-cell 3-D physics that supplements a velocity model."""

    wind_vx: float = 0.0
    wind_vy: float = 0.0
    wind_vz: float = 0.0
    attenuation: np.ndarray | None = None
    description: str = ""

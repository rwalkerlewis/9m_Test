"""Elastic velocity model dataclasses and creation helpers.

Provides :class:`ElasticModel2D` and :class:`ElasticModel3D` for
isotropic elastic media, plus :class:`GroundConfig` for specifying
the shallow subsurface half-space.

These models are used by the elastic FDTD solver
(:mod:`acoustic_sim.elastic_fdtd`) and are independent of the existing
acoustic velocity model classes in :mod:`acoustic_sim.model`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Ground configuration
# ---------------------------------------------------------------------------

@dataclass
class GroundConfig:
    """Parameters for a homogeneous isotropic elastic half-space.

    Attributes
    ----------
    vp : float
        P-wave velocity [m/s].
    vs : float
        S-wave velocity [m/s].
    density : float
        Mass density [kg/m³].
    qp : float
        P-wave quality factor (attenuation).  Higher = less attenuation.
    qs : float
        S-wave quality factor.
    depth : float
        Depth of the ground domain below the interface [m].
    """

    vp: float = 500.0
    vs: float = 250.0
    density: float = 1800.0
    qp: float = 20.0
    qs: float = 10.0
    depth: float = 5.0


# ---------------------------------------------------------------------------
# 2-D Elastic Model
# ---------------------------------------------------------------------------

@dataclass
class ElasticModel2D:
    """2-D elastic model on a regular Cartesian grid.

    The grid axes are *x* (horizontal) and *z* (vertical, positive up).
    The air–ground interface sits at *z = 0*: air occupies *z > 0* and
    ground occupies *z ≤ 0*.

    Attributes
    ----------
    x : np.ndarray
        1-D array of x cell-centre coordinates [m].
    z : np.ndarray
        1-D array of z cell-centre coordinates [m] (ascending order).
    vp : np.ndarray
        P-wave velocity [m/s], shape ``(nz, nx)``.
    vs : np.ndarray
        S-wave velocity [m/s], shape ``(nz, nx)``.  Zero in air (fluid).
    rho : np.ndarray
        Density [kg/m³], shape ``(nz, nx)``.
    qp : np.ndarray
        P-wave quality factor, shape ``(nz, nx)``.
    qs : np.ndarray
        S-wave quality factor, shape ``(nz, nx)``.
    dx : float
        Grid spacing in x [m].
    dz : float
        Grid spacing in z [m].
    """

    x: np.ndarray
    z: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    qp: np.ndarray
    qs: np.ndarray
    dx: float
    dz: float

    @property
    def nx(self) -> int:
        return len(self.x)

    @property
    def nz(self) -> int:
        return len(self.z)

    @property
    def shape(self) -> tuple[int, int]:
        """``(nz, nx)``."""
        return (self.nz, self.nx)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, z_min, z_max)`` for *imshow* extent."""
        return (
            float(self.x[0]),
            float(self.x[-1]),
            float(self.z[0]),
            float(self.z[-1]),
        )

    @property
    def vp_max(self) -> float:
        return float(np.max(self.vp))

    @property
    def vs_min_nonzero(self) -> float:
        """Minimum non-zero S-wave velocity (smallest wavelength driver)."""
        mask = self.vs > 0
        if not np.any(mask):
            return float(np.min(self.vp))
        return float(np.min(self.vs[mask]))

    @property
    def lambda_(self) -> np.ndarray:
        """First Lamé parameter λ = ρ(Vp² − 2Vs²), shape ``(nz, nx)``."""
        return self.rho * (self.vp ** 2 - 2.0 * self.vs ** 2)

    @property
    def mu(self) -> np.ndarray:
        """Shear modulus μ = ρVs², shape ``(nz, nx)``."""
        return self.rho * self.vs ** 2


# ---------------------------------------------------------------------------
# 3-D Elastic Model
# ---------------------------------------------------------------------------

@dataclass
class ElasticModel3D:
    """3-D elastic model on a regular Cartesian grid.

    Axes: *x*, *y* (horizontal), *z* (vertical, positive up).
    Air–ground interface at *z = 0*.

    Attributes
    ----------
    x, y, z : np.ndarray
        1-D arrays of cell-centre coordinates [m].
    vp : np.ndarray
        P-wave velocity [m/s], shape ``(nz, ny, nx)``.
    vs : np.ndarray
        S-wave velocity [m/s], shape ``(nz, ny, nx)``.
    rho : np.ndarray
        Density [kg/m³], shape ``(nz, ny, nx)``.
    qp : np.ndarray
        P-wave quality factor, shape ``(nz, ny, nx)``.
    qs : np.ndarray
        S-wave quality factor, shape ``(nz, ny, nx)``.
    dx : float
    dy : float
    dz : float
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    qp: np.ndarray
    qs: np.ndarray
    dx: float
    dy: float
    dz: float

    @property
    def nx(self) -> int:
        return len(self.x)

    @property
    def ny(self) -> int:
        return len(self.y)

    @property
    def nz(self) -> int:
        return len(self.z)

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(nz, ny, nx)``."""
        return (self.nz, self.ny, self.nx)

    @property
    def extent_xy(self) -> tuple[float, float, float, float]:
        return (
            float(self.x[0]),
            float(self.x[-1]),
            float(self.y[0]),
            float(self.y[-1]),
        )

    @property
    def extent_xz(self) -> tuple[float, float, float, float]:
        return (
            float(self.x[0]),
            float(self.x[-1]),
            float(self.z[0]),
            float(self.z[-1]),
        )

    @property
    def vp_max(self) -> float:
        return float(np.max(self.vp))

    @property
    def vs_min_nonzero(self) -> float:
        mask = self.vs > 0
        if not np.any(mask):
            return float(np.min(self.vp))
        return float(np.min(self.vs[mask]))

    @property
    def lambda_(self) -> np.ndarray:
        return self.rho * (self.vp ** 2 - 2.0 * self.vs ** 2)

    @property
    def mu(self) -> np.ndarray:
        return self.rho * self.vs ** 2


# ---------------------------------------------------------------------------
# Creation helpers
# ---------------------------------------------------------------------------

def _default_ground() -> GroundConfig:
    return GroundConfig()


def create_coupled_air_ground_2d(
    x_min: float = -100.0,
    x_max: float = 100.0,
    z_min: float = -5.0,
    z_max: float = 15.0,
    dx: float = 0.5,
    air_vp: float = 343.0,
    air_density: float = 1.225,
    ground: GroundConfig | None = None,
) -> ElasticModel2D:
    """Create a 2-D coupled air–ground elastic model.

    Parameters
    ----------
    x_min, x_max : float
        Horizontal extent [m].
    z_min, z_max : float
        Vertical extent [m].  ``z_min`` should be negative (below interface)
        and ``z_max`` positive (above interface).
    dx : float
        Uniform grid spacing [m] (used for both x and z).
    air_vp : float
        P-wave velocity in air [m/s].
    air_density : float
        Air density [kg/m³].
    ground : GroundConfig or None
        Ground properties.  Defaults to soft soil.

    Returns
    -------
    ElasticModel2D
    """
    if ground is None:
        ground = _default_ground()

    x = np.arange(x_min, x_max + 0.5 * dx, dx)
    z = np.arange(z_min, z_max + 0.5 * dx, dx)
    nz, nx = len(z), len(x)

    vp = np.empty((nz, nx), dtype=np.float64)
    vs = np.empty((nz, nx), dtype=np.float64)
    rho = np.empty((nz, nx), dtype=np.float64)
    qp = np.empty((nz, nx), dtype=np.float64)
    qs = np.empty((nz, nx), dtype=np.float64)

    for iz, zval in enumerate(z):
        if zval > 0.0:
            # Air
            vp[iz, :] = air_vp
            vs[iz, :] = 0.0
            rho[iz, :] = air_density
            qp[iz, :] = 9999.0  # effectively no attenuation
            qs[iz, :] = 9999.0
        else:
            # Ground
            vp[iz, :] = ground.vp
            vs[iz, :] = ground.vs
            rho[iz, :] = ground.density
            qp[iz, :] = ground.qp
            qs[iz, :] = ground.qs

    return ElasticModel2D(
        x=x, z=z, vp=vp, vs=vs, rho=rho, qp=qp, qs=qs, dx=dx, dz=dx,
    )


def create_coupled_air_ground_3d(
    x_min: float = 0.0,
    x_max: float = 100.0,
    y_min: float = 0.0,
    y_max: float = 100.0,
    z_min: float = -5.0,
    z_max: float = 15.0,
    dx: float = 0.5,
    air_vp: float = 343.0,
    air_density: float = 1.225,
    ground: GroundConfig | None = None,
) -> ElasticModel3D:
    """Create a 3-D coupled air–ground elastic model.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        Horizontal extent [m].
    z_min, z_max : float
        Vertical extent [m].
    dx : float
        Uniform grid spacing [m] (all three axes).
    air_vp : float
        P-wave velocity in air [m/s].
    air_density : float
        Air density [kg/m³].
    ground : GroundConfig or None
        Ground properties.

    Returns
    -------
    ElasticModel3D
    """
    if ground is None:
        ground = _default_ground()

    x = np.arange(x_min, x_max + 0.5 * dx, dx)
    y = np.arange(y_min, y_max + 0.5 * dx, dx)
    z = np.arange(z_min, z_max + 0.5 * dx, dx)
    nz, ny, nx = len(z), len(y), len(x)

    vp = np.empty((nz, ny, nx), dtype=np.float64)
    vs = np.empty((nz, ny, nx), dtype=np.float64)
    rho = np.empty((nz, ny, nx), dtype=np.float64)
    qp_arr = np.empty((nz, ny, nx), dtype=np.float64)
    qs_arr = np.empty((nz, ny, nx), dtype=np.float64)

    for iz, zval in enumerate(z):
        if zval > 0.0:
            vp[iz, :, :] = air_vp
            vs[iz, :, :] = 0.0
            rho[iz, :, :] = air_density
            qp_arr[iz, :, :] = 9999.0
            qs_arr[iz, :, :] = 9999.0
        else:
            vp[iz, :, :] = ground.vp
            vs[iz, :, :] = ground.vs
            rho[iz, :, :] = ground.density
            qp_arr[iz, :, :] = ground.qp
            qs_arr[iz, :, :] = ground.qs

    return ElasticModel3D(
        x=x, y=y, z=z, vp=vp, vs=vs, rho=rho,
        qp=qp_arr, qs=qs_arr, dx=dx, dy=dx, dz=dx,
    )

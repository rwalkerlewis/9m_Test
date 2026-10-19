"""Shared FDTD and frequency-domain acoustic solvers."""

from acoustic_sim.solver.domain_meta import DomainMeta, DomainMeta3D
from acoustic_sim.solver.fdtd import (
    BOUNDARIES,
    FDTD3DConfig,
    FDTD3DSolver,
    FDTDConfig,
    FDTDSolver,
    fd1_coefficients,
    fd2_cfl_factor,
    fd2_coefficients,
)
from acoustic_sim.solver.helmholtz import solve_helmholtz

__all__ = [
    "BOUNDARIES",
    "DomainMeta",
    "DomainMeta3D",
    "FDTD3DConfig",
    "FDTD3DSolver",
    "FDTDConfig",
    "FDTDSolver",
    "fd1_coefficients",
    "fd2_cfl_factor",
    "fd2_coefficients",
    "solve_helmholtz",
]

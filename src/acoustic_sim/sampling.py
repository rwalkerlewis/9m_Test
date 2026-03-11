"""Spatial-sampling and CFL stability checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from acoustic_sim.model import VelocityModel


def check_spatial_sampling(
    model: VelocityModel,
    frequency_hz: float,
    min_ppw: float = 10.0,
) -> dict[str, Any]:
    """Verify that the grid resolves the shortest wavelength.

    Returns a dict with keys:
        valid, min_wavelength, ppw, required_ppw, max_dx, message
    """
    c_min = model.c_min
    if c_min <= 0:
        return {
            "valid": False,
            "min_wavelength": 0.0,
            "ppw": 0.0,
            "required_ppw": min_ppw,
            "max_dx": 0.0,
            "message": "FAIL: velocity model contains non-positive speeds.",
        }
    wavelength_min = c_min / frequency_hz
    dx_max = max(model.dx, model.dy)
    ppw = wavelength_min / dx_max
    max_allowed_dx = wavelength_min / min_ppw
    valid = ppw >= min_ppw

    if valid:
        msg = (
            f"PASS: {ppw:.1f} pts/wavelength >= {min_ppw:.0f} required. "
            f"(lambda_min={wavelength_min:.3f} m, dx={dx_max:.4f} m, "
            f"c_min={c_min:.1f} m/s, f={frequency_hz:.1f} Hz)"
        )
    else:
        msg = (
            f"FAIL: {ppw:.1f} pts/wavelength < {min_ppw:.0f} required. "
            f"Reduce dx to <= {max_allowed_dx:.4f} m or lower the frequency. "
            f"(lambda_min={wavelength_min:.3f} m, dx={dx_max:.4f} m, "
            f"c_min={c_min:.1f} m/s, f={frequency_hz:.1f} Hz)"
        )
    return {
        "valid": valid,
        "min_wavelength": wavelength_min,
        "ppw": ppw,
        "required_ppw": min_ppw,
        "max_dx": max_allowed_dx,
        "message": msg,
    }


def check_cfl(
    model: VelocityModel,
    dt: float,
) -> dict[str, Any]:
    """CFL stability check for an explicit time-domain scheme.

    For the 2-D wave equation on a square grid the CFL condition is
    ``c_max * dt / dx <= 1/sqrt(2)``.
    """
    c_max = model.c_max
    dx_min = min(model.dx, model.dy)
    courant = c_max * dt / dx_min
    limit = 1.0 / np.sqrt(2.0)
    valid = courant <= limit
    if valid:
        msg = (
            f"PASS: Courant number {courant:.4f} <= {limit:.4f}. "
            f"(c_max={c_max:.1f} m/s, dt={dt:.2e} s, dx={dx_min:.4f} m)"
        )
    else:
        max_dt = limit * dx_min / c_max
        msg = (
            f"FAIL: Courant number {courant:.4f} > {limit:.4f}. "
            f"Reduce dt to <= {max_dt:.2e} s. "
            f"(c_max={c_max:.1f} m/s, dt={dt:.2e} s, dx={dx_min:.4f} m)"
        )
    return {"valid": valid, "courant": courant, "limit": limit, "message": msg}


def suggest_dx(c_min: float, frequency_hz: float, min_ppw: float = 10.0) -> float:
    """Return the maximum allowable dx for a given frequency and c_min."""
    return c_min / (frequency_hz * min_ppw)

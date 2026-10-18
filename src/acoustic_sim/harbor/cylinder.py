"""Analytic 2D scattering by a circular cylinder (partial wave series).

Used to validate the FDTD scatterer and in the link budget.  Time convention
``exp(-i omega t)``: outgoing waves are ``H_n^(1)``.  A line source at ``xs`` radiates
``G(x) = H_0^(1)(k |x - xs|)``; the FDTD point source is this times a real constant, so
ratios of scattered to incident field are convention and calibration free.
"""

from __future__ import annotations

import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp


def partial_wave_coefficients(k: float, a: float, kind: str, n_max: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Orders ``n`` and coefficients ``c_n`` with ``p_s = sum_n c_n (...) H_n^(1)(k r) e^{i n phi}``.

    ``kind = "soft"``: ``c_n = -J_n(ka) / H_n(ka)`` (pressure release, Dirichlet).
    ``kind = "rigid"``: ``c_n = -J_n'(ka) / H_n'(ka)`` (Neumann).
    """
    ka = k * a
    if n_max is None:
        n_max = int(np.ceil(ka)) + 12
    n = np.arange(-n_max, n_max + 1)
    if kind == "soft":
        c = -jv(n, ka) / hankel1(n, ka)
    elif kind == "rigid":
        c = -jvp(n, ka) / h1vp(n, ka)
    else:
        raise ValueError("kind must be 'soft' or 'rigid'")
    return n, c


def plane_wave_far_field_amplitude(k: float, a: float, phi: np.ndarray, kind: str = "soft") -> np.ndarray:
    """``f(phi)`` with ``p_s ~ sqrt(2 / (pi k r)) exp(i (k r - pi/4)) f(phi)`` for a unit plane wave
    incident along +x; *phi* is the scattering angle from the incidence direction (0 = forward)."""
    n, c = partial_wave_coefficients(k, a, kind)
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    # Plane wave e^{ikx} = sum_n i^n J_n(kr) e^{i n phi}; scattered = sum_n i^n c_n H_n(kr) e^{i n phi},
    # far field H_n ~ sqrt(2/(pi k r)) e^{i(kr - n pi/2 - pi/4)}  ->  f = sum_n c_n e^{i n phi}.
    return np.sum(c[:, None] * np.exp(1j * n[:, None] * phi[None, :]), axis=0)


def line_source_scattered(
    k: float,
    a: float,
    centre: np.ndarray,
    source: np.ndarray,
    receivers: np.ndarray,
    kind: str = "soft",
) -> tuple[np.ndarray, np.ndarray]:
    """Incident and scattered field at *receivers* for a line source ``H_0^(1)(k |x - xs|)``
    and a cylinder of radius *a* at *centre*.  Returns ``(incident, scattered)``, each ``(R,)``.

    Graf's addition theorem gives, for receiver radius ``r > a`` and source radius ``rs > a``,
    ``p_s = sum_n c_n H_n(k rs) H_n(k r) e^{i n (phi - phi_s)}``.
    """
    centre = np.asarray(centre, dtype=float)
    ds = np.asarray(source, dtype=float) - centre
    rs, phis = np.hypot(*ds), np.arctan2(ds[1], ds[0])
    dr = np.atleast_2d(np.asarray(receivers, dtype=float)) - centre
    r, phi = np.hypot(dr[:, 0], dr[:, 1]), np.arctan2(dr[:, 1], dr[:, 0])
    if rs <= a or np.any(r <= a):
        raise ValueError("source and receivers must lie outside the cylinder")
    n, c = partial_wave_coefficients(k, a, kind)
    inc = hankel1(0, k * np.hypot(*(np.atleast_2d(receivers) - np.asarray(source)).T))
    terms = (c * hankel1(n, k * rs))[:, None] * hankel1(n[:, None], k * r[None, :]) * np.exp(1j * n[:, None] * (phi - phis)[None, :])
    return inc, np.sum(terms, axis=0)

"""Motion compensated (synthetic aperture) integration of per frame bistatic maps.

Split out of :mod:`acoustic_sim.water.bistatic`, which re-exports :func:`motion_integrate`.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from acoustic_sim.water.device import get_device


def _asnumpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def motion_integrate(Z: np.ndarray, grid: np.ndarray, times: np.ndarray, speeds: np.ndarray,
                     coherent: bool = True, impl: str = "vectorized", use_cuda: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the per frame maps along constant velocity hypotheses.

    For every velocity ``v`` in *speeds* (V, 2) the frame maps are shifted to the central
    time, ``Z_k(x + v (t_k - t_mid))``, and summed (coherently if *coherent*, else their
    magnitudes squared).  Returns the best map ``(G,)`` over velocities and the per velocity
    peak values ``(V,)``.  The ship moves during the window, so the bistatic ellipses sweep
    across a target; integrating along the true velocity is a synthetic aperture.

    *impl* ``"vectorized"`` (default) interpolates all frames and velocities in one call per chunk
    of velocities (the frame maps stacked, bilinear in the map and exact along the frame axis),
    optionally on the GPU; ``"reference"`` is the original loop.
    """
    if impl == "reference":
        return _motion_integrate_reference(Z, grid, times, speeds, coherent)
    return _motion_integrate_vectorized(Z, grid, times, speeds, coherent, get_device(use_cuda).xp)


def _motion_integrate_vectorized(Z, grid, times, speeds, coherent, xp, budget: int = 2 ** 27):
    x, y = grid[:, 0], grid[:, 1]
    xs, ys = np.unique(x), np.unique(y)
    dx = xs[1] - xs[0]
    ix = np.rint((x - xs[0]) / dx).astype(int)
    iy = np.rint((y - ys[0]) / dx).astype(int)
    K, G, V = len(times), len(grid), len(speeds)
    t_mid = times[len(times) // 2]
    complex_in = np.iscomplexobj(Z)
    imgs = xp.zeros((K, len(ys), len(xs)), dtype=Z.dtype)
    imgs[:, xp.asarray(iy), xp.asarray(ix)] = xp.asarray(Z).T
    if xp is np:
        from scipy.ndimage import map_coordinates as mapc
    else:
        from cupyx.scipy.ndimage import map_coordinates as mapc
    shift = speeds[:, None, :] * (times - t_mid)[None, :, None]                                # (V, K, 2)
    cx_all = (x[None, None, :] + shift[:, :, 0:1] - xs[0]) / dx                                 # (V, K, G)
    cy_all = (y[None, None, :] + shift[:, :, 1:2] - ys[0]) / dx
    ck = np.broadcast_to(np.arange(K, dtype=float)[None, :, None], (V, K, G))
    acc = xp.zeros((V, G), dtype=np.complex128 if coherent else float)
    vc = max(1, int(budget // (K * G * 8 * 3)))
    for v0 in range(0, V, vc):
        coords = xp.asarray(np.stack([ck[v0:v0 + vc], cy_all[v0:v0 + vc], cx_all[v0:v0 + vc]]))   # (3, vc, K, G)
        re = mapc(imgs.real, coords, order=1, mode="constant")
        im = mapc(imgs.imag, coords, order=1, mode="constant") if complex_in else 0.0
        val = re + 1j * im if complex_in else re
        acc[v0:v0 + vc] = xp.sum(val, axis=1) if coherent else xp.sum(xp.abs(val) ** 2, axis=1)
    m = xp.abs(acc) ** 2 if coherent else acc
    peaks = _asnumpy(m.max(axis=1))
    best = _asnumpy(m[int(np.argmax(peaks))])
    return best, peaks


def _motion_integrate_reference(Z: np.ndarray, grid: np.ndarray, times: np.ndarray, speeds: np.ndarray,
                                coherent: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The original per velocity, per frame loop of :func:`motion_integrate`."""
    x, y = grid[:, 0], grid[:, 1]
    xs, ys = np.unique(x), np.unique(y)
    dx = xs[1] - xs[0]
    ix = np.rint((x - xs[0]) / dx).astype(int)
    iy = np.rint((y - ys[0]) / dx).astype(int)
    shape = (len(ys), len(xs))
    t_mid = times[len(times) // 2]
    best, peaks = None, np.zeros(len(speeds))
    for v_i, v in enumerate(speeds):
        acc = np.zeros(len(grid), dtype=np.complex128 if coherent else float)
        for k, t in enumerate(times):
            img = np.zeros(shape, dtype=Z.dtype)
            img[iy, ix] = Z[:, k]
            shift = v * (t - t_mid)
            cx = (x + shift[0] - xs[0]) / dx
            cy = (y + shift[1] - ys[0]) / dx
            re = map_coordinates(img.real, [cy, cx], order=1, mode="constant")
            im = map_coordinates(img.imag, [cy, cx], order=1, mode="constant") if np.iscomplexobj(Z) else 0.0
            val = re + 1j * im if np.iscomplexobj(Z) else re
            acc += val if coherent else np.abs(val) ** 2
        m = np.abs(acc) ** 2 if coherent else acc
        peaks[v_i] = m.max()
        if best is None or peaks[v_i] > best.max():
            best = m
    return best, peaks

"""Moving body masks evaluated on the device (``HarborFDTDConfig.body_masks = "device"``).

The reference :meth:`HarborFDTDSolver._apply_scatterers` calls
:func:`~acoustic_sim.harbor.scatterer.body_factor` on the host for every body and step (a
meshgrid, a signed distance and a ramp over the body's bounding box) and copies the factor to
the device, about half a millisecond per body and step.  :class:`DeviceBodyMasks` evaluates the
same formulas in one CUDA launch per step for all bodies of all batch members:

* the pose of every body at every step (centre, unit heading, active flag) is tabulated once per
  run on the host with the same interpolations as :meth:`Scatterer.pose_at`;
* one thread per cell of a body's bounding box computes the factor of :func:`body_factor` in
  double precision (same operation order; ``hypot`` and ``sqrt`` are correctly rounded on both
  sides) and multiplies the pressure in place, skipping the write where the factor is 1;
* bodies whose bounding boxes overlap at a step would race for the same cells, so the host
  checks the boxes per step from the pose table and runs those steps one body per launch
  (sequential, exactly the reference order).

Masks agree with ``body_factor`` to the last bit or so (measured below 1e-15) and the traces of a
run with device masks agree with the host path to rounding.
"""

from __future__ import annotations

import math

import numpy as np

from acoustic_sim.harbor.scatterer import Scatterer

_SRC = r"""
extern "C" __global__
void body_masks({T}* __restrict__ p, const double* __restrict__ xs, const double* __restrict__ ys,
                const double* __restrict__ pose,      // (n_steps, B, J, 4): cx, cy, hx, hy
                const signed char* __restrict__ active, // (n_steps, B, J)
                const double* __restrict__ par,       // (B, J, 4): length, width, strength, half
                const signed char* __restrict__ kind, // (B, J, 2): shape (0 rect, 1 ellipse), smooth
                const int n, const int n_steps, const int B, const int J, const int j0,
                const int box, const double dx, const int ny, const int nx)
{{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;      // cell within the box
    const int j = j0 + blockIdx.y;                             // body
    const int b = blockIdx.z;                                  // batch member
    if (c >= box * box || j >= J) return;
    const size_t bj = (size_t)b * J + j;
    if (!active[((size_t)n * B) * J + bj]) return;
    const double* q = pose + (((size_t)n * B) * J + bj) * 4;
    const double cx = q[0], cy = q[1], hx = q[2], hy = q[3];
    const double L = par[bj * 4 + 0], W = par[bj * 4 + 1], strength = par[bj * 4 + 2], half = par[bj * 4 + 3];
    const double x0 = xs[0], y0 = ys[0];
    int ix0 = (int)floor((cx - half - x0) / dx);
    int ix1 = (int)ceil((cx + half - x0) / dx) + 1;
    int iy0 = (int)floor((cy - half - y0) / dx);
    int iy1 = (int)ceil((cy + half - y0) / dx) + 1;
    const int ux0 = ix0, uy0 = iy0;                            // unclipped origin of the box
    ix0 = max(ix0, 0); ix1 = min(ix1, nx);
    iy0 = max(iy0, 0); iy1 = min(iy1, ny);
    if (ix1 <= ix0 || iy1 <= iy0) return;
    const int ix = ux0 + (c % box);
    const int iy = uy0 + (c / box);
    if (ix < ix0 || ix >= ix1 || iy < iy0 || iy >= iy1) return;
    const double X = xs[ix] - cx;
    const double Y = ys[iy] - cy;
    const double along = X * hx + Y * hy;
    const double across = (-X) * hy + Y * hx;
    double sdf;
    if (kind[bj * 2] == 1) {{
        const double ra = 0.5 * L, rb = 0.5 * W;
        const double rho = hypot(along / ra, across / rb);
        sdf = (rho - 1.0) * sqrt(ra * rb);
    }} else {{
        const double qa = fabs(along) - 0.5 * L;
        const double qb = fabs(across) - 0.5 * W;
        const double outside = hypot(fmax(qa, 0.0), fmax(qb, 0.0));
        const double inside = fmin(fmax(qa, qb), 0.0);
        sdf = outside + inside;
    }}
    double ramp;
    if (kind[bj * 2 + 1]) {{
        ramp = sdf / dx + 0.5;
        ramp = fmin(fmax(ramp, 0.0), 1.0);
    }} else {{
        ramp = (sdf > 0.0) ? 1.0 : 0.0;
    }}
    const double factor = 1.0 - strength * (1.0 - ramp);
    if (factor != 1.0) {{
        const size_t i = ((size_t)b * ny + iy) * nx + ix;
        p[i] = ({T})(p[i] * ({T})factor);
    }}
}}
"""

_CTYPE = {np.dtype(np.float64): "double", np.dtype(np.float32): "float"}
_KERNELS: dict = {}


def _kernel(dtype):
    import cupy as cp

    key = np.dtype(dtype).str
    if key not in _KERNELS:
        _KERNELS[key] = cp.RawKernel(_SRC.format(T=_CTYPE[np.dtype(dtype)]), "body_masks", options=("--fmad=false",))
    return _KERNELS[key]


def pose_table(sc: Scatterer, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(centres (n, 2), headings (n, 2), active (n,))`` of *sc* at the times *t*, the same values
    as :meth:`Scatterer.pose_at` and ``Track.active_at`` per step."""
    tr = sc.track
    xy = tr.position_at(t)
    d = tr.position_at(t + 0.5) - tr.position_at(t - 0.5)
    norm = np.hypot(d[:, 0], d[:, 1])
    still = norm < 1e-6
    heading = np.empty_like(xy)
    moving = ~still
    heading[moving] = d[moving] / norm[moving, None]
    if still.any():
        d0 = tr.xy[-1] - tr.xy[0]
        n0 = float(np.hypot(*d0))
        heading[still] = np.array([1.0, 0.0]) if n0 < 1e-6 else d0 / n0
    return xy, heading, tr.active_at(t)


class DeviceBodyMasks:
    """Per step body factors of every body of every batch member, applied in one launch.

    Parameters
    ----------
    body_sets : list of lists of Scatterer
        One list per batch member (may be empty).
    x, y : grid coordinates (host), dx : grid spacing
    n_steps, dt : the run's steps (poses are tabulated at ``n * dt``)
    """

    def __init__(self, body_sets: list[list[Scatterer]], x: np.ndarray, y: np.ndarray, dx: float,
                 n_steps: int, dt: float, dtype) -> None:
        import cupy as cp

        self.xp = cp
        self.dtype = np.dtype(dtype)
        self.B = len(body_sets)
        self.J = max((len(s) for s in body_sets), default=0)
        self.n_steps = n_steps
        self.dx = float(dx)
        self.ny, self.nx = len(y), len(x)
        self.xs = cp.asarray(np.ascontiguousarray(x, dtype=np.float64))
        self.ys = cp.asarray(np.ascontiguousarray(y, dtype=np.float64))
        t = np.arange(n_steps, dtype=np.float64) * dt
        pose = np.zeros((n_steps, self.B, self.J, 4), dtype=np.float64)
        active = np.zeros((n_steps, self.B, self.J), dtype=np.int8)
        par = np.zeros((self.B, self.J, 4), dtype=np.float64)
        kind = np.zeros((self.B, self.J, 2), dtype=np.int8)
        box = 1
        for b, bodies in enumerate(body_sets):
            for j, sc in enumerate(bodies):
                xy, hd, act = pose_table(sc, t)
                pose[:, b, j, 0:2] = xy
                pose[:, b, j, 2:4] = hd
                active[:, b, j] = act
                half = 0.5 * (sc.length + sc.width) + 2.0 * self.dx
                strength = 1.0 if sc.kind in ("soft", "hard") else sc.strength
                par[b, j] = (sc.length, sc.width, strength, half)
                kind[b, j] = (1 if sc.shape == "ellipse" else 0, 1 if sc.smooth else 0)
                box = max(box, int(math.ceil(2.0 * half / self.dx)) + 3)
        self.box = box
        self.pose, self.active, self.par, self.kind = (cp.asarray(a) for a in (pose, active, par, kind))
        self.sequential = self._overlap_steps(pose, active, par)
        self.kernel = _kernel(self.dtype)
        self.threads = 256
        self.blocks = math.ceil(box * box / self.threads)
        self.n_launches = 0

    def _overlap_steps(self, pose: np.ndarray, active: np.ndarray, par: np.ndarray) -> np.ndarray:
        """Steps at which two active bodies of the same batch member have overlapping boxes."""
        seq = np.zeros(self.n_steps, dtype=bool)
        x0, y0, dx = float(self.xs[0]), float(self.ys[0]), self.dx
        for b in range(self.B):
            for j in range(self.J):
                for k in range(j + 1, self.J):
                    hj, hk = par[b, j, 3], par[b, k, 3]
                    cj, ck = pose[:, b, j, :2], pose[:, b, k, :2]
                    # Index boxes as the kernel computes them (before clipping): overlap when the ranges meet.
                    jx0, jx1 = np.floor((cj[:, 0] - hj - x0) / dx), np.ceil((cj[:, 0] + hj - x0) / dx) + 1
                    kx0, kx1 = np.floor((ck[:, 0] - hk - x0) / dx), np.ceil((ck[:, 0] + hk - x0) / dx) + 1
                    jy0, jy1 = np.floor((cj[:, 1] - hj - y0) / dx), np.ceil((cj[:, 1] + hj - y0) / dx) + 1
                    ky0, ky1 = np.floor((ck[:, 1] - hk - y0) / dx), np.ceil((ck[:, 1] + hk - y0) / dx) + 1
                    both = active[:, b, j].astype(bool) & active[:, b, k].astype(bool)
                    seq |= both & (jx0 < kx1) & (kx0 < jx1) & (jy0 < ky1) & (ky0 < jy1)
        return seq

    def apply(self, p, n: int) -> None:
        """Multiply the pressure *p* (``(ny, nx)`` or ``(B, ny, nx)``) by every body factor at step *n*."""
        if self.J == 0:
            return
        args_tail = (np.int32(n), np.int32(self.n_steps), np.int32(self.B), np.int32(self.J))
        if self.sequential[n]:
            for j in range(self.J):
                self.kernel((self.blocks, 1, self.B), (self.threads, 1, 1),
                            (p, self.xs, self.ys, self.pose, self.active, self.par, self.kind, *args_tail,
                             np.int32(j), np.int32(self.box), np.float64(self.dx), np.int32(self.ny), np.int32(self.nx)))
                self.n_launches += 1
        else:
            self.kernel((self.blocks, self.J, self.B), (self.threads, 1, 1),
                        (p, self.xs, self.ys, self.pose, self.active, self.par, self.kind, *args_tail,
                         np.int32(0), np.int32(self.box), np.float64(self.dx), np.int32(self.ny), np.int32(self.nx)))
            self.n_launches += 1

    def factor_field(self, b: int, n: int):
        """The product of all body factors of batch member *b* at step *n* as a dense ``(ny, nx)``
        device array (for tests against :func:`body_factor`)."""
        xp = self.xp
        ones = xp.ones((self.B, self.ny, self.nx), dtype=np.float64)
        saved = self.kernel
        self.kernel = _kernel(np.float64)
        try:
            self.apply(ones, n)
        finally:
            self.kernel = saved
        return ones[b]

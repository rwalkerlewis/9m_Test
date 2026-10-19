"""Fused CUDA time step for :class:`acoustic_sim.fdtd.FDTDSolver` (``FDTDConfig.kernel = "fused"``).

The reference ``FDTDSolver._step`` is about thirty CuPy array operations per step, each a kernel
launch that streams the whole grid through memory, plus a ``zeros_like`` allocation and a
dozen tiny indexing kernels for the source and the receivers.  This module does the same
arithmetic in **one** ``cupy.RawKernel`` launch per step:

* the Laplacian of order ``fd_order`` (stencil weights baked into the kernel source), the
  leapfrog update with ``C2`` and the damping ``sigma``, the Dirichlet edge bands, and the
  optional pressure release mask (the harbor land) multiply, written **in place into the
  previous field buffer** (every cell reads ``p_prev`` only at itself, so the two field
  buffers simply swap roles each step: no allocation);
* the bilinear source injection from a per step table precomputed on the host once per run
  (``(iy, ix)`` and the four weights times the signal), applied by the threads that own the
  four cells, before the mask multiply, exactly as the reference does;
* the bilinear receiver sampling of the field left by the previous step into a chunked
  device trace buffer (the field a kernel writes is read by other threads, so a step samples
  the field of the step before and ``finish`` samples the last one).

Optional extra launches per step: the device body masks of ``fdtd_cuda_bodies`` (stage 2) and
the CPML memory variable update (stage 4) hook in through :class:`FusedFDTD` as well.

Arithmetic is ordered as in the reference expression and compiled with ``--fmad=false`` so the
float64 kernel reproduces the reference to rounding (measured 1e-13 relative over 120 000
steps).  All kernels carry a leading batch dimension (grid ``z``): ``B`` independent runs on
the same grid advance in the same launches (stage 3).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_STEP_SRC = r"""
extern "C" __global__
void fdtd_step(const {T}* __restrict__ p, {T}* __restrict__ pp,
               const {T}* __restrict__ c2, const {T}* __restrict__ sigma,
               const {T}* __restrict__ mask,
               const int* __restrict__ src_iy, const int* __restrict__ src_ix,
               const {T}* __restrict__ src_w,
               const int* __restrict__ r_iy, const int* __restrict__ r_ix,
               const {T}* __restrict__ r_wx, const {T}* __restrict__ r_wy,
               {T}* __restrict__ trace, const int n_recv, const int trace_col, const int chunk,
               const int n, const int n_steps, const int ny, const int nx)
{{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (ix >= nx || iy >= ny) return;
    const size_t plane = (size_t)ny * nx;
    const size_t off = (size_t)b * plane;
    const size_t g = (size_t)iy * nx + ix;
    const size_t i = off + g;

    // Receiver sampling of the field left by the previous step (p), one receiver per thread.
    if (trace_col >= 0) {{
        const int r = iy * nx + ix;      // linear thread id over the grid
        if (r < n_recv) {{
            const size_t q = off + (size_t)r_iy[r] * nx + r_ix[r];
            const {T} wx = r_wx[r], wy = r_wy[r];
            const {T} one = ({T})1.0;
            const {T} v = p[q] * (one - wx) * (one - wy) + p[q + 1] * wx * (one - wy)
                        + p[q + nx] * (one - wx) * wy + p[q + nx + 1] * wx * wy;
            trace[((size_t)b * n_recv + r) * chunk + trace_col] = v;
        }}
    }}

    {T} out;
    if (iy < {M} || iy >= ny - {M} || ix < {M} || ix >= nx - {M}) {{
        out = ({T})0.0;
    }} else {{
        const {T} pc = p[i];
        const {T} ppc = pp[i];
        {T} lap = ({T}){C0X2} * pc;
{LAP_TERMS}
        out = ((({T})2.0 * pc - ppc) + c2[g] * lap) - sigma[g] * (pc - ppc);
    }}
    // Source injection: the four bilinear cells of this step (the table holds weight x amplitude).
    {{
        const size_t s = (size_t)b * n_steps + n;
        const int sy = src_iy[s], sx = src_ix[s];
        if (sy >= 0) {{
            if (iy == sy && ix == sx)             out += src_w[4 * s + 0];
            else if (iy == sy && ix == sx + 1)    out += src_w[4 * s + 1];
            else if (iy == sy + 1 && ix == sx)    out += src_w[4 * s + 2];
            else if (iy == sy + 1 && ix == sx + 1) out += src_w[4 * s + 3];
        }}
    }}
{MASK}
    pp[i] = out;
}}

extern "C" __global__
void fdtd_sample(const {T}* __restrict__ p,
                 const int* __restrict__ r_iy, const int* __restrict__ r_ix,
                 const {T}* __restrict__ r_wx, const {T}* __restrict__ r_wy,
                 {T}* __restrict__ trace, const int n_recv, const int trace_col, const int chunk,
                 const int ny, const int nx)
{{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= n_recv) return;
    const size_t off = (size_t)b * ny * nx;
    const size_t q = off + (size_t)r_iy[r] * nx + r_ix[r];
    const {T} wx = r_wx[r], wy = r_wy[r];
    const {T} one = ({T})1.0;
    const {T} v = p[q] * (one - wx) * (one - wy) + p[q + 1] * wx * (one - wy)
                + p[q + nx] * (one - wx) * wy + p[q + nx + 1] * wx * wy;
    trace[((size_t)b * n_recv + r) * chunk + trace_col] = v;
}}
"""

_CTYPE = {np.dtype(np.float64): "double", np.dtype(np.float32): "float"}
_BLOCK = (32, 8, 1)
TRACE_CHUNK = 4096          # steps held on the device before a copy to the host trace array


def _lit(x: float) -> str:
    """Round trip literal of a Python float (17 significant digits)."""
    return repr(float(x))


def step_kernel_source(coeffs: np.ndarray, dtype, has_mask: bool) -> str:
    """CUDA source of the fused step for the stencil *coeffs* (``fd2_coefficients``)."""
    T = _CTYPE[np.dtype(dtype)]
    M = len(coeffs) - 1
    terms = []
    for k in range(1, M + 1):
        # Reference order: lap += ck * (down + up + right + left) with ((down + up) + right) + left.
        terms.append(f"        lap += ({T}){_lit(coeffs[k])} * (((p[i + {k} * nx] + p[i - {k} * nx]) + p[i + {k}]) + p[i - {k}]);")
    mask = "    out *= mask[g];" if has_mask else ""
    return _STEP_SRC.format(T=T, M=M, C0X2=_lit(coeffs[0] * 2.0), LAP_TERMS="\n".join(terms), MASK=mask)


_MODULE_CACHE: dict[tuple, Any] = {}


def compile_step(coeffs: np.ndarray, dtype, has_mask: bool):
    """``(step_kernel, sample_kernel)`` compiled once per (stencil, dtype, mask)."""
    import cupy as cp

    key = (tuple(float(c) for c in coeffs), np.dtype(dtype).str, bool(has_mask))
    if key not in _MODULE_CACHE:
        mod = cp.RawModule(code=step_kernel_source(coeffs, dtype, has_mask), options=("--fmad=false", "--std=c++11"))
        _MODULE_CACHE[key] = (mod.get_function("fdtd_step"), mod.get_function("fdtd_sample"))
    return _MODULE_CACHE[key]


def source_table(sources: list, n_steps: int, dt: float, x0: float, y0: float, dx: float, ny: int, nx: int,
                 amplitude: float, dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per step bilinear injection of every source: ``(B, n_steps)`` cell indices (``-1`` when the
    source is outside the grid, as the reference skips it) and ``(B, n_steps, 4)`` weights times the
    signal sample, in the order ``(iy, ix), (iy, ix+1), (iy+1, ix), (iy+1, ix+1)``.

    Positions follow ``source.position_at(n, dt)`` for every step; sources with ``times`` and
    ``positions`` (``CustomTrajectorySource``) are interpolated in one call, which is the same
    ``np.interp`` per element, and static sources are constant.
    """
    B = len(sources)
    iy_all = np.full((B, n_steps), -1, dtype=np.int32)
    ix_all = np.full((B, n_steps), -1, dtype=np.int32)
    w_all = np.zeros((B, n_steps, 4), dtype=np.float64)
    t = np.arange(n_steps, dtype=np.float64) * dt
    for b, src in enumerate(sources):
        if hasattr(src, "times") and hasattr(src, "positions"):
            x = np.interp(t, src.times, src.positions[:, 0])
            y = np.interp(t, src.times, src.positions[:, 1])
        elif hasattr(src, "x") and hasattr(src, "y") and not hasattr(src, "speed") and not hasattr(src, "x1"):
            x = np.full(n_steps, float(src.x))
            y = np.full(n_steps, float(src.y))
        else:
            xy = np.array([src.position_at(n, dt) for n in range(n_steps)], dtype=np.float64)
            x, y = xy[:, 0], xy[:, 1]
        fx = (x - x0) / dx
        fy = (y - y0) / dx
        gix = np.floor(fx).astype(np.int64)
        giy = np.floor(fy).astype(np.int64)
        valid = (gix >= 0) & (gix + 1 < nx) & (giy >= 0) & (giy + 1 < ny)
        wx = fx - gix
        wy = fy - giy
        sig = np.asarray(src.signal, dtype=np.float64)
        amp = amplitude * sig[np.minimum(np.arange(n_steps), len(sig) - 1)]
        w = np.stack([(1 - wx) * (1 - wy) * amp, wx * (1 - wy) * amp, (1 - wx) * wy * amp, wx * wy * amp], axis=1)
        iy_all[b, valid] = giy[valid]
        ix_all[b, valid] = gix[valid]
        w_all[b, valid] = w[valid]
    return iy_all, ix_all, w_all.astype(dtype)


class FusedFDTD:
    """One fused launch per step for a (possibly batched) 2-D leapfrog solver on the device.

    Parameters
    ----------
    fields : (p_now, p_prev)
        Device arrays ``(ny, nx)`` or ``(B, ny, nx)`` (the solver keeps rebinding them after every
        swap; :meth:`step` returns the new ``(p_now, p_prev)``).
    c2, sigma : device arrays ``(ny, nx)``
        Shared by the batch.
    mask : device array ``(ny, nx)`` or None
        Pressure release mask (1 water, 0 land), multiplied into the new field.
    sources : list
        One source per batch member (``position_at``, ``signal``).
    receivers : (iy, ix, wx, wy)
        Bilinear sample points (host arrays), the same for every batch member.
    traces : host array ``(n_recv, n_steps)`` or ``(B, n_recv, n_steps)``
        Filled chunk by chunk; complete after :meth:`finish`.
    """

    def __init__(self, coeffs: np.ndarray, fields, c2, sigma, mask, sources: list, receivers, traces: np.ndarray,
                 n_steps: int, dt: float, x0: float, y0: float, dx: float, amplitude: float, dtype) -> None:
        import cupy as cp

        self.xp = cp
        self.dtype = np.dtype(dtype)
        p, pp = fields
        self.batched = p.ndim == 3
        self.B = p.shape[0] if self.batched else 1
        self.ny, self.nx = p.shape[-2], p.shape[-1]
        if len(sources) != self.B:
            raise ValueError(f"{len(sources)} sources for a batch of {self.B}")
        self.step_kernel, self.sample_kernel = compile_step(coeffs, self.dtype, mask is not None)
        self.c2 = cp.ascontiguousarray(c2, dtype=self.dtype)
        self.sigma = cp.ascontiguousarray(sigma, dtype=self.dtype)
        self.mask = cp.ascontiguousarray(mask, dtype=self.dtype) if mask is not None else self.c2   # unused when absent
        iy, ix, w = source_table(sources, n_steps, dt, x0, y0, dx, self.ny, self.nx, amplitude, self.dtype)
        self.src_iy, self.src_ix, self.src_w = cp.asarray(iy), cp.asarray(ix), cp.asarray(w)
        r_iy, r_ix, r_wx, r_wy = receivers
        self.n_recv = len(r_iy)
        self.r_iy = cp.asarray(np.asarray(r_iy, dtype=np.int32))
        self.r_ix = cp.asarray(np.asarray(r_ix, dtype=np.int32))
        self.r_wx = cp.asarray(np.asarray(r_wx, dtype=self.dtype))
        self.r_wy = cp.asarray(np.asarray(r_wy, dtype=self.dtype))
        if self.n_recv > self.ny * self.nx:
            raise ValueError("more receivers than grid cells")
        self.n_steps = n_steps
        self.traces = traces
        self.chunk = int(min(TRACE_CHUNK, max(n_steps, 1)))
        self.trace_buf = cp.zeros((self.B, self.n_recv, self.chunk), dtype=self.dtype)
        self.col0 = 0                       # step index of the first column held in trace_buf
        self.sampled_to = 0                 # steps whose receivers have been sampled
        self.grid = (math.ceil(self.nx / _BLOCK[0]), math.ceil(self.ny / _BLOCK[1]), self.B)
        self.post_hooks: list = []          # callables (p_new, n) run after the update (bodies, ...)

    # ------------------------------------------------------------------

    def _flush(self, force: bool = False) -> None:
        """Copy the full trace chunk (or whatever is pending when *force*) to the host array."""
        held = self.sampled_to - self.col0
        if held <= 0 or (held < self.chunk and not force):
            return
        buf = self.trace_buf[:, :, :held].get()
        if self.batched:
            self.traces[:, :, self.col0:self.col0 + held] = buf
        else:
            self.traces[:, self.col0:self.col0 + held] = buf[0]
        self.col0 += held

    def step(self, p, pp, n: int):
        """Advance from ``(p, pp)`` at step *n*; returns ``(p_new, p_old)`` = the swapped buffers."""
        col = n - 1 - self.col0 if n >= 1 and self.sampled_to == n - 1 else -1
        self.step_kernel(
            self.grid, _BLOCK,
            (p, pp, self.c2, self.sigma, self.mask, self.src_iy, self.src_ix, self.src_w,
             self.r_iy, self.r_ix, self.r_wx, self.r_wy, self.trace_buf,
             np.int32(self.n_recv), np.int32(col), np.int32(self.chunk),
             np.int32(n), np.int32(self.n_steps), np.int32(self.ny), np.int32(self.nx)),
        )
        if col >= 0:
            self.sampled_to = n
            self._flush()
        for hook in self.post_hooks:
            hook(pp, n)
        return pp, p

    def sample(self, p, n: int) -> None:
        """Sample the receivers of the field *p* (the state after step *n*) if not done yet."""
        if self.sampled_to >= n + 1:
            return
        if self.sampled_to != n:
            raise RuntimeError(f"receiver samples out of order: {self.sampled_to} steps sampled, asked for step {n}")
        col = n - self.col0
        self.sample_kernel(
            (math.ceil(self.n_recv / 256), self.B, 1), (256, 1, 1),
            (p, self.r_iy, self.r_ix, self.r_wx, self.r_wy, self.trace_buf,
             np.int32(self.n_recv), np.int32(col), np.int32(self.chunk), np.int32(self.ny), np.int32(self.nx)),
        )
        self.sampled_to = n + 1
        self._flush()

    def finish(self, p, n_done: int) -> None:
        """Sample the last field and copy the pending traces to the host (idempotent)."""
        if n_done > 0:
            self.sample(p, n_done - 1)
        self._flush(force=True)

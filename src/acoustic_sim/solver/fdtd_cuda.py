"""Fused CUDA time step for :class:`acoustic_sim.solver.fdtd.FDTDSolver` (``FDTDConfig.kernel = "fused"``).

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
               const {T}* __restrict__ psix, const {T}* __restrict__ psiy,
               {T}* __restrict__ zetax, {T}* __restrict__ zetay,
               const {T}* __restrict__ bx, const {T}* __restrict__ ax, const {T}* __restrict__ mx,
               const {T}* __restrict__ by, const {T}* __restrict__ ay, const {T}* __restrict__ my,
               const {T}* __restrict__ q,
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
{LAPLACIAN}
        out = ((({T})2.0 * pc - ppc) + {TERM}) - sigma[g] * (pc - ppc);
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
void lw4_q(const {T}* __restrict__ p, {T}* __restrict__ q, const {T}* __restrict__ c2, const {T}* __restrict__ mask,
           const int ny, const int nx)
{{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (ix >= nx || iy >= ny) return;
    const size_t g = (size_t)iy * nx + ix;
    const size_t i = (size_t)b * ny * nx + g;
    {T} out;
    if (iy < {M} || iy >= ny - {M} || ix < {M} || ix >= nx - {M}) {{
        out = ({T})0.0;
    }} else {{
        const {T} pc = p[i];
        {T} lap = ({T}){C0X2} * pc;
{LAP_P}
        out = c2[g] * lap;
    }}
{MASK}
    q[i] = out;
}}

extern "C" __global__
void cpml_psi(const {T}* __restrict__ p, {T}* __restrict__ psix, {T}* __restrict__ psiy,
              const {T}* __restrict__ bx, const {T}* __restrict__ ax, const {T}* __restrict__ mx,
              const {T}* __restrict__ by, const {T}* __restrict__ ay, const {T}* __restrict__ my,
              const int ny, const int nx)
{{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (ix < {M} || ix >= nx - {M} || iy < {M} || iy >= ny - {M}) return;
    const bool in_x = mx[ix] != ({T})0.0;
    const bool in_y = my[iy] != ({T})0.0;
    if (!in_x && !in_y) return;
    const size_t i = (size_t)b * ny * nx + (size_t)iy * nx + ix;
    if (in_x) {{
        {T} dpx = ({T}){D1_1} * (p[i + 1] - p[i - 1]);
{DPX_TERMS}
        psix[i] = bx[ix] * psix[i] + ax[ix] * dpx;
    }}
    if (in_y) {{
        {T} dpy = ({T}){D1_1} * (p[i + nx] - p[i - nx]);
{DPY_TERMS}
        psiy[i] = by[iy] * psiy[i] + ay[iy] * dpy;
    }}
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
TRACE_BUDGET = 64 * 2 ** 20  # bytes of device trace buffer (shrinks the chunk for large batches)


def _lit(x: float) -> str:
    """Round trip literal of a Python float (17 significant digits)."""
    return repr(float(x))


def step_kernel_source(coeffs: np.ndarray, dtype, has_mask: bool, cpml: bool = False, d1: np.ndarray | None = None,
                       scheme: str = "leapfrog") -> str:
    """CUDA source of the fused step for the stencil *coeffs* (``fd2_coefficients``); with *cpml* the
    Laplacian carries the layer terms of ``FDTDSolver._cpml_laplacian`` (*d1*: ``fd1_coefficients``);
    *scheme* ``"lw4"`` adds the fourth order time term from the scratch field ``q = C2 L p`` and
    ``"kspace"`` takes the k-space Laplacian from ``q`` (``c2`` then holds ``(c / c0)^2``)."""
    T = _CTYPE[np.dtype(dtype)]
    M = len(coeffs) - 1
    if d1 is None:
        d1 = np.zeros(M + 1)
        d1[1] = 0.5
    lines = []
    term = "c2[g] * lap"
    # Reference order: lap += ck * (down + up + right + left) with ((down + up) + right) + left.
    lap_p = [f"        lap += ({T}){_lit(coeffs[k])} * (((p[i + {k} * nx] + p[i - {k} * nx]) + p[i + {k}]) + p[i - {k}]);"
             for k in range(1, M + 1)]
    if scheme == "lw4":
        lines.append(f"        const {T} qc = q[i];")
        lines.append(f"        {T} lap = ({T}){_lit(coeffs[0] * 2.0)} * qc;")
        for k in range(1, M + 1):
            lines.append(f"        lap += ({T}){_lit(coeffs[k])} * (((q[i + {k} * nx] + q[i - {k} * nx]) + q[i + {k}]) + q[i - {k}]);")
        term = f"(qc + (c2[g] * ({T}){_lit(1.0 / 12.0)}) * lap)"
    elif scheme == "kspace":
        term = "c2[g] * q[i]"
    elif not cpml:
        lines.append(f"        {T} lap = ({T}){_lit(coeffs[0] * 2.0)} * pc;")
        lines.extend(lap_p)
    else:
        # Reference order of _cpml_laplacian: lapx, lapy accumulated per k; dqx, dqy likewise; then
        # ((lapx + dqx) + zetax) + ((lapy + dqy) + zetay).
        lines.append(f"        {T} lapx = ({T}){_lit(coeffs[0])} * pc;")
        lines.append(f"        {T} lapy = ({T}){_lit(coeffs[0])} * pc;")
        for k in range(1, M + 1):
            lines.append(f"        lapx = lapx + ({T}){_lit(coeffs[k])} * (p[i + {k}] + p[i - {k}]);")
            lines.append(f"        lapy = lapy + ({T}){_lit(coeffs[k])} * (p[i + {k} * nx] + p[i - {k} * nx]);")
        lines.append(f"        {T} tx = lapx, ty = lapy;")
        lines.append(f"        if (mx[ix] != ({T})0.0) {{")
        lines.append(f"            {T} dqx = ({T}){_lit(d1[1])} * (psix[i + 1] - psix[i - 1]);")
        for k in range(2, M + 1):
            lines.append(f"            dqx = dqx + ({T}){_lit(d1[k])} * (psix[i + {k}] - psix[i - {k}]);")
        lines.append("            dqx = dqx * mx[ix];")
        lines.append("            const {T} zx = bx[ix] * zetax[i] + ax[ix] * (lapx + dqx);".replace("{T}", T))
        lines.append("            zetax[i] = zx;")
        lines.append("            tx = (lapx + dqx) + zx;")
        lines.append("        }")
        lines.append(f"        if (my[iy] != ({T})0.0) {{")
        lines.append(f"            {T} dqy = ({T}){_lit(d1[1])} * (psiy[i + nx] - psiy[i - nx]);")
        for k in range(2, M + 1):
            lines.append(f"            dqy = dqy + ({T}){_lit(d1[k])} * (psiy[i + {k} * nx] - psiy[i - {k} * nx]);")
        lines.append("            dqy = dqy * my[iy];")
        lines.append("            const {T} zy = by[iy] * zetay[i] + ay[iy] * (lapy + dqy);".replace("{T}", T))
        lines.append("            zetay[i] = zy;")
        lines.append("            ty = (lapy + dqy) + zy;")
        lines.append("        }")
        lines.append(f"        const {T} lap = tx + ty;")
    dpx = [f"        dpx = dpx + ({T}){_lit(d1[k])} * (p[i + {k}] - p[i - {k}]);" for k in range(2, M + 1)]
    dpy = [f"        dpy = dpy + ({T}){_lit(d1[k])} * (p[i + {k} * nx] - p[i - {k} * nx]);" for k in range(2, M + 1)]
    mask = "    out *= mask[g];" if has_mask else ""
    return _STEP_SRC.format(T=T, M=M, LAPLACIAN="\n".join(lines), TERM=term, MASK=mask, D1_1=_lit(d1[1]),
                            DPX_TERMS="\n".join(dpx), DPY_TERMS="\n".join(dpy),
                            C0X2=_lit(coeffs[0] * 2.0), LAP_P="\n".join(lap_p))


_MODULE_CACHE: dict[tuple, Any] = {}


def compile_step(coeffs: np.ndarray, dtype, has_mask: bool, cpml: bool = False, d1: np.ndarray | None = None,
                 scheme: str = "leapfrog"):
    """``(step_kernel, sample_kernel, psi_kernel, lw4_kernel)`` compiled once per (stencil, dtype, mask, cpml, scheme)."""
    import cupy as cp

    key = (tuple(float(c) for c in coeffs), np.dtype(dtype).str, bool(has_mask), bool(cpml),
           None if d1 is None else tuple(float(c) for c in d1), scheme)
    if key not in _MODULE_CACHE:
        mod = cp.RawModule(code=step_kernel_source(coeffs, dtype, has_mask, cpml, d1, scheme), options=("--fmad=false", "--std=c++11"))
        _MODULE_CACHE[key] = (mod.get_function("fdtd_step"), mod.get_function("fdtd_sample"), mod.get_function("cpml_psi"),
                              mod.get_function("lw4_q"))
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
                 n_steps: int, dt: float, x0: float, y0: float, dx: float, amplitude: float, dtype,
                 cpml: dict | None = None, scheme: dict | None = None) -> None:
        import cupy as cp

        self.xp = cp
        self.dtype = np.dtype(dtype)
        p, pp = fields
        self.batched = p.ndim == 3
        self.B = p.shape[0] if self.batched else 1
        self.ny, self.nx = p.shape[-2], p.shape[-1]
        if len(sources) != self.B:
            raise ValueError(f"{len(sources)} sources for a batch of {self.B}")
        self.cpml = cpml
        self.scheme = "leapfrog" if scheme is None else scheme["name"]
        if self.scheme != "leapfrog" and cpml is not None:
            raise ValueError("lw4 / kspace schemes do not combine with the CPML")
        self.step_kernel, self.sample_kernel, self.psi_kernel, self.lw4_kernel = compile_step(
            coeffs, self.dtype, mask is not None, cpml is not None, None if cpml is None else cpml["d1"], self.scheme)
        if self.scheme == "kspace":
            c2 = scheme["ratio"]                                    # the kernel multiplies the k-space Laplacian by (c / c0)^2
            self.k_kernel = cp.asarray(scheme["kernel"], dtype=self.dtype)   # single precision FFT product with float32 fields
        self.q = cp.zeros(p.shape, dtype=self.dtype) if self.scheme != "leapfrog" else None
        self.c2 = cp.ascontiguousarray(c2, dtype=self.dtype)
        self.sigma = cp.ascontiguousarray(sigma, dtype=self.dtype)
        self.mask = cp.ascontiguousarray(mask, dtype=self.dtype) if mask is not None else self.c2   # unused when absent
        if cpml is not None:
            shape = p.shape
            self.psix, self.psiy, self.zetax, self.zetay = (cp.zeros(shape, dtype=self.dtype) for _ in range(4))
            self.cp_prof = tuple(cp.ascontiguousarray(cpml[k], dtype=self.dtype) for k in ("bx", "ax", "mx", "by", "ay", "my"))
        else:
            self.psix = self.psiy = self.zetax = self.zetay = self.c2          # unused
            self.cp_prof = (self.c2,) * 6
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
        # Steps held on the device before a copy to the host: at most TRACE_CHUNK, within TRACE_BUDGET bytes.
        budget = max(16, TRACE_BUDGET // max(self.B * self.n_recv * self.dtype.itemsize, 1))
        self.chunk = int(min(TRACE_CHUNK, budget, max(n_steps, 1)))
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
        if self.cpml is not None:
            self.psi_kernel(self.grid, _BLOCK, (p, self.psix, self.psiy, *self.cp_prof, np.int32(self.ny), np.int32(self.nx)))
        q = self.c2                                                 # unused placeholder for the leapfrog
        if self.scheme == "lw4":
            self.lw4_kernel(self.grid, _BLOCK, (p, self.q, self.c2, self.mask, np.int32(self.ny), np.int32(self.nx)))
            q = self.q
        elif self.scheme == "kspace":
            # The k space step waits for the device after every operation.  Without the waits the float32 runs on the
            # 601 x 801 grids differed between processes on the Quadro (CUDA 12.9, cuFFT 11.4) in about one fresh
            # process in three, one in ten by a disturbance of up to 2e-2 of the trace's peak (once 1.4) that starts at
            # one point of the field; with them, 1 in 240 (output/studies/float32/gpu_determinism.md).  The cause was
            # not found; float64 was never affected.
            xp = self.xp
            wait = xp.cuda.Device().synchronize
            spec = xp.fft.rfft2(p, axes=(-2, -1))
            wait()
            prod = spec * self.k_kernel
            wait()
            lk = xp.fft.irfft2(prod, s=(self.ny, self.nx), axes=(-2, -1))
            wait()
            self.q[...] = lk
            wait()
            q = self.q
        self.step_kernel(
            self.grid, _BLOCK,
            (p, pp, self.c2, self.sigma, self.mask, self.src_iy, self.src_ix, self.src_w,
             self.r_iy, self.r_ix, self.r_wx, self.r_wy, self.trace_buf,
             np.int32(self.n_recv), np.int32(col), np.int32(self.chunk),
             self.psix, self.psiy, self.zetax, self.zetay, *self.cp_prof, q,
             np.int32(n), np.int32(self.n_steps), np.int32(self.ny), np.int32(self.nx)),
        )
        if self.scheme == "kspace":
            self.xp.cuda.Device().synchronize()
        if col >= 0:
            self.sampled_to = n
            self._flush()
        for hook in self.post_hooks:
            hook(pp, n)
        if self.scheme == "kspace" and self.post_hooks:
            self.xp.cuda.Device().synchronize()
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

"""Frequency domain (Helmholtz) forward model of the harbor.

A second forward model next to the FDTD of ``fdtd_runner.py``, on the same
:class:`HarborEnvironment` grid and the same land and shore loss conventions:

* land nodes are pressure release (p = 0), exactly as the FDTD zeroes them every step;
* water within ``shore_loss_width`` of land gets the same quadratic damping ramp as the FDTD
  (``fdtd.shore_loss_max`` per step at ``fdtd.fs``), plus ``fdtd.background_absorption``;
  a per step damping ``sigma`` of the FDTD leapfrog is a damping rate ``gamma = sigma * fs``,
  which enters the Helmholtz equation as ``k^2 = (w^2 + i w gamma) / c^2``;
* the grid edge is a perfectly matched layer instead of the FDTD sponge.

For each frequency the operator is factorised once (sparse LU) and reused for every source.
Because the operator is complex symmetric, the field at ``x`` from a source at ``r`` equals
the field at ``r`` from a source at ``x`` (reciprocity): one solve per receiver sample point
gives the Green's function from that receiver to every cell of the scan grid, which is the
matched field replica library, multipath, shadowing and all.

Stencil
-------
``"9pt"`` (default) is the optimal 9 point stencil of Jo, Shin and Suh (1996, Geophysics 61,
529): a weighted mix of the axis and the rotated Laplacian with a lumped mass term, with phase
velocity errors near 0.1 percent at 8 nodes per wavelength instead of about 2.5 percent for the
plain 5 point stencil.  Over the hundreds of wavelengths of a harbor that is the difference
between a fraction of a radian and several radians of phase error.  Inside the PML the stencil
blends to the 5 point form, where the stretched coordinate operator is exact in symmetric form.

Conventions
-----------
The solve is done for ``exp(-i w t)`` time dependence (outgoing ``(i/4) H0^(1)(kr)``) and the
result is returned as ``4 i conj(u)``, which for a point source in open water is
``H0^(2)(kr)``: the same function and scale as ``detection.green2d``, so Helmholtz replicas are
drop in replacements for the free field replicas of ``detection.py`` and ``bistatic.py``.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import distance_transform_edt

from acoustic_sim.harbor.environment import HarborEnvironment
from acoustic_sim.harbor.sensors import SensorLayout

# Jo, Shin and Suh (1996) optimal coefficients.
JSS_A = 0.5461
JSS_C = 0.6248
JSS_D = 0.09381
JSS_E = (1.0 - JSS_C - 4.0 * JSS_D) / 4.0


@dataclass
class HelmholtzConfig:
    """Settings of the Helmholtz model.

    Attributes
    ----------
    f_lo, f_hi, df : float
        Frequencies ``f_lo, f_lo + df, ..., <= f_hi`` [Hz].
    npml : int
        PML thickness in grid nodes.
    pml_reflection : float
        Target normal incidence reflection of the PML; sets the damping profile.
    blend : int
        Nodes over which the stencil goes from 9 point to 5 point at the inner PML edge.
    stencil : str
        ``"9pt"`` or ``"5pt"``.
    fs, background_absorption, shore_loss_width, shore_loss_max : float
        Copied from the FDTD settings so both models see the same losses
        (:meth:`from_fdtd`).  Per step damping at ``fs``.
    batch : int
        Right hand sides per triangular solve batch (memory: ``batch`` full complex fields).
    """

    f_lo: float = 100.0
    f_hi: float = 380.0
    df: float = 20.0
    npml: int = 40
    pml_reflection: float = 1e-6
    blend: int = 8
    stencil: str = "9pt"
    fs: float = 6000.0
    background_absorption: float = 1e-4
    shore_loss_width: float = 40.0
    shore_loss_max: float = 0.15
    batch: int = 16

    @property
    def freqs(self) -> np.ndarray:
        n = int(np.floor((self.f_hi - self.f_lo) / self.df + 1e-9)) + 1
        return self.f_lo + self.df * np.arange(n)

    @classmethod
    def from_fdtd(cls, fdtd, **kw) -> "HelmholtzConfig":
        """Take the loss settings from a :class:`HarborFDTDConfig`."""
        return cls(fs=fdtd.fs, background_absorption=fdtd.background_absorption,
                   shore_loss_width=fdtd.shore_loss_width, shore_loss_max=fdtd.shore_loss_max, **kw)


class HelmholtzSolver:
    """Sparse Helmholtz operator on the water nodes of a harbor environment."""

    def __init__(self, env: HarborEnvironment, cfg: HelmholtzConfig | None = None):
        self.env = env
        self.cfg = cfg or HelmholtzConfig()
        if self.cfg.stencil not in ("9pt", "5pt"):
            raise ValueError("stencil must be '9pt' or '5pt'")
        ny, nx = env.ny, env.nx
        self.shape = (ny, nx)
        self.h = env.dx
        self.water = env.water.copy()
        self.index = -np.ones((ny, nx), dtype=np.int64)
        self.index[self.water] = np.arange(int(self.water.sum()))
        self.n = int(self.water.sum())
        self.c = env.model.values
        self.gamma = self._damping_rate()
        self._lu: dict[float, object] = {}

    # ------------------------------------------------------------------
    # profiles
    # ------------------------------------------------------------------
    def _damping_rate(self) -> np.ndarray:
        """FDTD shore and background loss as a damping rate [1/s] (``sigma * fs``)."""
        cfg = self.cfg
        sigma = np.full(self.shape, cfg.background_absorption)
        if cfg.shore_loss_width > 0 and cfg.shore_loss_max > 0:
            dist = distance_transform_edt(self.water) * self.h
            ramp = np.clip((cfg.shore_loss_width - dist) / cfg.shore_loss_width, 0.0, 1.0) ** 2
            sigma = np.maximum(sigma, cfg.shore_loss_max * ramp)
        return sigma * cfg.fs

    def _pml_profile(self, n: int, w: float) -> tuple[np.ndarray, np.ndarray]:
        """Damping ``sigma(i)`` at nodes and at half nodes ``i + 1/2`` along one axis, and the blend."""
        cfg = self.cfg
        L = cfg.npml * self.h
        cmax = float(np.max(self.c))
        smax = 3.0 * cmax * np.log(1.0 / cfg.pml_reflection) / (2.0 * L)

        def sig(pos):
            d = np.maximum(np.maximum(cfg.npml - pos, pos - (n - 1 - cfg.npml)), 0.0)  # nodes into the PML
            return smax * (d / cfg.npml) ** 2

        i = np.arange(n, dtype=float)
        return sig(i), sig(i + 0.5)

    def _blend(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """0 in the interior (9 point), 1 inside the PML (5 point), linear over ``blend`` nodes."""
        cfg = self.cfg
        if cfg.stencil == "5pt":
            return np.ones(n), np.ones(n)

        def b(pos):
            d = np.maximum(np.maximum(cfg.npml + cfg.blend - pos, pos - (n - 1 - cfg.npml - cfg.blend)), 0.0)
            return np.clip(d / max(cfg.blend, 1), 0.0, 1.0)

        i = np.arange(n, dtype=float)
        return b(i), b(i + 0.5)

    # ------------------------------------------------------------------
    # operator
    # ------------------------------------------------------------------
    def operator(self, f: float) -> sp.csc_matrix:
        """Complex symmetric matrix ``A`` with ``A u = -delta`` for ``exp(-i w t)``."""
        ny, nx = self.shape
        h2 = self.h ** 2
        w = 2.0 * np.pi * f
        sx_n, sx_h = (1.0 + 1j * s / w for s in self._pml_profile(nx, w))
        sy_n, sy_h = (1.0 + 1j * s / w for s in self._pml_profile(ny, w))
        bx_n, bx_h = self._blend(nx)
        by_n, by_h = self._blend(ny)
        k2 = (w ** 2 + 1j * w * self.gamma) / self.c ** 2                     # (ny, nx)
        SX, SY = np.meshgrid(sx_n, sy_n)
        beta = np.maximum(bx_n[None, :], by_n[:, None])                        # node blend

        rows, cols, vals = [], [], []
        diag = np.zeros(self.shape, dtype=complex)
        idx = self.index

        def couple(sl_p, sl_q, wgt):
            """Edge between node sets p and q with stiffness weight *wgt* (Laplacian part).
            The diagonal gets -w on both sides; a land neighbour (p = 0) still counts."""
            ip, iq = idx[sl_p], idx[sl_q]
            wp = np.broadcast_to(wgt, ip.shape)
            okp = ip >= 0
            okq = iq >= 0
            d = np.zeros(self.shape, dtype=complex)
            d[sl_p] -= np.where(okp, wp, 0)
            d[sl_q] -= np.where(okq, wp, 0)
            both = okp & okq
            rows.append(ip[both]); cols.append(iq[both]); vals.append(wp[both])
            rows.append(iq[both]); cols.append(ip[both]); vals.append(wp[both])
            return d

        def mass(sl_p, sl_q, wgt):
            ip, iq = idx[sl_p], idx[sl_q]
            wp = np.broadcast_to(wgt, ip.shape)
            both = (ip >= 0) & (iq >= 0)
            rows.append(ip[both]); cols.append(iq[both]); vals.append(wp[both])
            rows.append(iq[both]); cols.append(ip[both]); vals.append(wp[both])

        # axis edges in x: between (j, i) and (j, i+1); half node i+1/2
        P, Q = (slice(None), slice(0, nx - 1)), (slice(None), slice(1, nx))
        ax_x = (JSS_A + (1.0 - JSS_A) * np.maximum(bx_h[None, :nx - 1], by_n[:, None])) if self.cfg.stencil == "9pt" \
            else np.ones((ny, nx - 1))
        wx = ax_x * (sy_n[:, None] / sx_h[None, :nx - 1]) / h2
        diag += couple(P, Q, wx)
        # axis edges in y
        P, Q = (slice(0, ny - 1), slice(None)), (slice(1, ny), slice(None))
        ax_y = (JSS_A + (1.0 - JSS_A) * np.maximum(by_h[:ny - 1, None], bx_n[None, :])) if self.cfg.stencil == "9pt" \
            else np.ones((ny - 1, nx))
        wy = ax_y * (sx_n[None, :] / sy_h[:ny - 1, None]) / h2
        diag += couple(P, Q, wy)

        # mass: node term, plus neighbour terms in the 9 point interior
        if self.cfg.stencil == "9pt":
            c_node = JSS_C + (1.0 - JSS_C) * beta
            diag += c_node * SX * SY * k2
            k2m_x = 0.5 * (k2[:, :-1] + k2[:, 1:])
            bm_x = np.maximum(beta[:, :-1], beta[:, 1:])
            mass((slice(None), slice(0, nx - 1)), (slice(None), slice(1, nx)), JSS_D * (1 - bm_x) * k2m_x)
            k2m_y = 0.5 * (k2[:-1, :] + k2[1:, :])
            bm_y = np.maximum(beta[:-1, :], beta[1:, :])
            mass((slice(0, ny - 1), slice(None)), (slice(1, ny), slice(None)), JSS_D * (1 - bm_y) * k2m_y)
            # rotated Laplacian and diagonal mass, interior only (weight -> 0 through the blend)
            for (P, Q) in (((slice(0, ny - 1), slice(0, nx - 1)), (slice(1, ny), slice(1, nx))),   # NE
                           ((slice(0, ny - 1), slice(1, nx)), (slice(1, ny), slice(0, nx - 1)))):  # NW
                bd = np.maximum(beta[P], beta[Q])
                wd = (1.0 - JSS_A) * (1.0 - bd) / (2.0 * h2)
                diag += couple(P, Q, wd)
                mass(P, Q, JSS_E * (1 - bd) * 0.5 * (k2[P] + k2[Q]))
            # In the blend the neighbour mass weight moves onto the node (c_node), keeping the
            # mass row sum at k^2.
        else:
            diag += SX * SY * k2

        rows.append(idx[self.water]); cols.append(idx[self.water]); vals.append(diag[self.water])
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n))
        return A

    def factor(self, f: float, keep: bool = False) -> "_Factor":
        """Sparse LU of the operator at *f*, in nested dissection order.

        SuperLU's own orderings are slow on a 2-D grid (minutes for 160 000 nodes); a geometric
        nested dissection ordering with no pivoting factorises the same matrix in about a second.
        No pivoting is safe here because the PML and the losses keep the matrix well away from
        singular; the tests check the residual.
        """
        if f in self._lu:
            return self._lu[f]
        A = self.operator(f)
        p = self.perm
        lu = spla.splu(A[p][:, p].tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
        fac = _Factor(lu, p)
        if keep:
            self._lu[f] = fac
        return fac

    @property
    def perm(self) -> np.ndarray:
        if getattr(self, "_perm", None) is None:
            self._perm = nested_dissection(self.index)
        return self._perm

    def release(self) -> None:
        self._lu.clear()

    # ------------------------------------------------------------------
    # sources and sampling (bilinear, mutually adjoint so reciprocity is exact)
    # ------------------------------------------------------------------
    def interp_matrix(self, points: np.ndarray) -> sp.csr_matrix:
        """``(n_points, n_water)`` bilinear interpolation onto *points* (land nodes count as 0)."""
        pts = np.atleast_2d(np.asarray(points, float))
        fx = (pts[:, 0] - self.env.model.x[0]) / self.h
        fy = (pts[:, 1] - self.env.model.y[0]) / self.h
        i0 = np.clip(np.floor(fx).astype(int), 0, self.shape[1] - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, self.shape[0] - 2)
        tx, ty = fx - i0, fy - j0
        rows, cols, vals = [], [], []
        for dj, di, wgt in ((0, 0, (1 - tx) * (1 - ty)), (0, 1, tx * (1 - ty)), (1, 0, (1 - tx) * ty), (1, 1, tx * ty)):
            col = self.index[j0 + dj, i0 + di]
            ok = col >= 0
            rows.append(np.flatnonzero(ok)); cols.append(col[ok]); vals.append(wgt[ok])
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(pts), self.n))

    def _to_repo(self, u: np.ndarray) -> np.ndarray:
        return 4j * np.conj(u)

    def solve(self, f: float, sources: np.ndarray, sample: sp.csr_matrix | None = None,
              lu=None, full: bool = False) -> np.ndarray:
        """Fields from unit point sources at *sources* ``(n_src, 2)``.

        Returns ``(n_src, n_sample)`` values at the rows of *sample* (an :meth:`interp_matrix`),
        or full ``(n_src, ny, nx)`` fields when ``full`` is True.  Scaled so that in open water a
        source gives ``H0^(2)(k r)`` (the ``detection.green2d`` convention).
        """
        lu = lu if lu is not None else self.factor(f)
        S = self.interp_matrix(sources)
        out = []
        for b0 in range(0, S.shape[0], self.cfg.batch):
            rhs = -(S[b0:b0 + self.cfg.batch].T.toarray().astype(complex)) / self.h ** 2
            u = lu.solve(rhs)                                                     # (n_water, nb)
            if full:
                blk = np.zeros((u.shape[1],) + self.shape, dtype=complex)
                blk[:, self.water] = self._to_repo(u).T
                out.append(blk)
            else:
                out.append(self._to_repo(sample @ u).T)
        return np.concatenate(out, axis=0)

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(self.env.fingerprint().encode())
        h.update(repr(sorted(asdict(self.cfg).items())).encode())
        return h.hexdigest()[:16]


class _Factor:
    """LU of the permuted operator with a solve in the original ordering."""

    def __init__(self, lu, perm: np.ndarray):
        self.lu, self.perm = lu, perm

    def solve(self, b: np.ndarray) -> np.ndarray:
        y = self.lu.solve(np.ascontiguousarray(b[self.perm]))
        u = np.empty_like(y)
        u[self.perm] = y
        return u


def nested_dissection(index: np.ndarray, leaf: int = 64) -> np.ndarray:
    """Geometric nested dissection ordering of the unknowns of a masked grid.

    *index* is ``(ny, nx)`` with the unknown number at water nodes and -1 on land.  The node set
    is split by the grid line through its median along its longer extent; the two halves are
    ordered recursively and the separator line goes last.  A grid line separates a 5 point
    and a 9 point stencil alike only if the 9 point diagonals cannot jump it, which holds
    because a separator is one full line of nodes.
    """
    jj, ii = np.nonzero(index >= 0)
    order = np.argsort(index[jj, ii])
    ii, jj = ii[order], jj[order]
    out: list[np.ndarray] = []
    stack = [(np.arange(len(ii)), 0)]
    # Iterative post order: (ids, state) with state 0 = split, 1 = emit separator.
    while stack:
        ids, state = stack.pop()
        if state == 1:
            out.append(ids)
            continue
        if len(ids) <= leaf:
            out.append(ids)
            continue
        xs, ys = ii[ids], jj[ids]
        v = xs if (xs.max() - xs.min()) >= (ys.max() - ys.min()) else ys
        m = np.floor(np.median(v))
        lo, hi, sep = ids[v < m], ids[v > m], ids[v == m]
        if len(lo) == 0 or len(hi) == 0:
            out.append(ids)
            continue
        stack.append((sep, 1))
        stack.append((hi, 0))
        stack.append((lo, 0))
    return np.concatenate(out)


# ----------------------------------------------------------------------
# Replica library
# ----------------------------------------------------------------------

@dataclass
class ReplicaLibrary:
    """Green's functions between the receivers, a scan grid, extra points and illuminators.

    ``G[f, cell, receiver]``: channel data from a unit source at each scan cell, in the receiver
    order of :class:`SensorLayout` (cables, then hydrophones), with the strain second difference
    already applied for strain cables.  ``E[f, point, receiver]``: the same for extra points
    (vessel and scatterer positions, which need not lie on the grid).
    ``F[f, source, target]``: the field at every grid cell and extra point (targets = grid, then
    extra) from a unit source at each illuminator position: the incident field for the echo
    model and for the bistatic processor.
    """

    freqs: np.ndarray
    grid: np.ndarray
    G: np.ndarray
    extra: np.ndarray
    E: np.ndarray
    sources: np.ndarray
    F: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez(path, **{k: getattr(self, k) for k in ("freqs", "grid", "G", "extra", "E", "sources", "F")})

    @classmethod
    def load(cls, path: str | Path) -> "ReplicaLibrary":
        z = np.load(path)
        return cls(*(z[k] for k in ("freqs", "grid", "G", "extra", "E", "sources", "F")))

    def incident_grid(self, src: int) -> np.ndarray:
        """``(n_f, n_cell)`` field on the grid from illuminator *src*."""
        return self.F[:, src, :len(self.grid)]

    def incident_extra(self, src: int, point: int) -> np.ndarray:
        """``(n_f,)`` field at extra point *point* from illuminator *src*."""
        return self.F[:, src, len(self.grid) + point]


def replica_library(
    solver: HelmholtzSolver,
    layout: SensorLayout,
    grid: np.ndarray,
    extra: np.ndarray | None = None,
    sources: np.ndarray | None = None,
    freqs: np.ndarray | None = None,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> ReplicaLibrary:
    """Build a :class:`ReplicaLibrary` by reciprocity.

    One solve per distinct receiver sample point and frequency (strain channels at a spacing
    equal to the gauge length share their end points, which removes a third of the solves), plus
    one solve per illuminator position.  Cached on disk by a hash of everything that enters.
    """
    freqs = solver.cfg.freqs if freqs is None else np.asarray(freqs, float)
    extra = np.empty((0, 2)) if extra is None else np.atleast_2d(np.asarray(extra, float))
    sources = np.empty((0, 2)) if sources is None else np.atleast_2d(np.asarray(sources, float))
    key = None
    if cache_dir is not None:
        h = hashlib.sha1()
        for part in (solver.fingerprint(), layout.fingerprint()):
            h.update(part.encode())
        for arr in (grid, extra, sources, freqs):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        key = Path(cache_dir) / f"replicas_{h.hexdigest()[:16]}.npz"
        if key.exists():
            if verbose:
                print(f"  replica library from cache {key.name}")
            return ReplicaLibrary.load(key)

    pts, inv = np.unique(np.round(layout.sample_points, 6), axis=0, return_inverse=True)
    inv = inv.ravel()
    targets = np.vstack([grid, extra])
    sample = solver.interp_matrix(targets)
    nG = len(grid)
    G = np.zeros((len(freqs), nG, layout.n_receivers), dtype=np.complex64)
    E = np.zeros((len(freqs), len(extra), layout.n_receivers), dtype=np.complex64)
    F = np.zeros((len(freqs), len(sources), len(targets)), dtype=np.complex64)
    for fi, f in enumerate(freqs):
        t0 = time.time()
        lu = solver.factor(f)
        t1 = time.time()
        vals = solver.solve(f, pts, sample, lu=lu)            # (n_unique, n_targets)
        ch = layout.collapse(vals[inv].T)                     # (n_targets, n_receivers)
        G[fi] = ch[:nG]
        E[fi] = ch[nG:]
        if len(sources):
            F[fi] = solver.solve(f, sources, sample, lu=lu)
        del lu, vals, ch
        if verbose:
            print(f"  replicas {f:6.1f} Hz: factor {t1 - t0:5.1f} s, {len(pts) + len(sources)} solves "
                  f"{time.time() - t1:6.1f} s", flush=True)
    lib = ReplicaLibrary(freqs, np.asarray(grid, float), G, extra, E, sources, F)
    if key is not None:
        key.parent.mkdir(parents=True, exist_ok=True)
        lib.save(key)
    return lib


def free_field_replicas(layout: SensorLayout, grid: np.ndarray, freqs: np.ndarray, c: float) -> np.ndarray:
    """``(n_f, n_cell, n_receivers)`` free field replicas (``detection.green2d`` at the sample
    points, collapsed): what ``bistatic.py`` uses, for comparison with the Helmholtz library."""
    from acoustic_sim.harbor.bistatic import ReplicaModel

    model = ReplicaModel(layout, c)
    out = np.zeros((len(freqs), len(grid), layout.n_receivers), dtype=np.complex64)
    for fi, f in enumerate(freqs):
        for b0 in range(0, len(grid), 4096):
            out[fi, b0:b0 + 4096] = model(grid[b0:b0 + 4096], 2 * np.pi * f / c)
    return out


# ----------------------------------------------------------------------
# Processing on frequency domain snapshots
# ----------------------------------------------------------------------

def project_out(d: np.ndarray, vecs: list[np.ndarray]) -> np.ndarray:
    """Remove the span of all *vecs* jointly, per frequency.

    *d* is ``(n_snap, n_f, n_ch)``; each entry of *vecs* is ``(n_f, n_ch)`` or ``(n_f, m, n_ch)``.
    Sequential removal is wrong (the projections do not commute); this orthonormalises all
    vectors together (QR) and projects once.
    """
    out = d.copy()
    for fi in range(d.shape[1]):
        cols = []
        for v in vecs:
            v = np.asarray(v)
            cols.append(v[fi][None, :] if v.ndim == 2 else v[fi])
        V = np.concatenate(cols, axis=0).T                     # (n_ch, m)
        Q, R = np.linalg.qr(V)
        keep = np.abs(np.diag(R)) > 1e-8 * np.abs(R).max()
        Q = Q[:, keep]
        X = out[:, fi, :]
        out[:, fi, :] = X - (X @ Q.conj()) @ Q.T
    return out


def bartlett(d: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Incoherent passive Bartlett power per cell, summed over snapshots and frequencies.

    *d* ``(n_snap, n_f, n_ch)``, *G* ``(n_f, n_cell, n_ch)``.
    """
    P = np.zeros(G.shape[1])
    for fi in range(G.shape[0]):
        Gf = G[fi]
        y = Gf.conj() @ d[:, fi, :].T                          # (n_cell, n_snap)
        P += np.sum(np.abs(y) ** 2, axis=1) / (np.sum(np.abs(Gf) ** 2, axis=1) + 1e-30)
    return P / (d.shape[0] * G.shape[0])


def bistatic_coherent(d_res: np.ndarray, ref: np.ndarray, Gin: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Coherent bistatic matched filter over frequency.

    ``Z(x) = sum_f conj(r_f) conj(g_in(x, f)) w(x, f)^H d_res(f)``, power summed over snapshots,
    normalised by the expected noise power so the floor is flat.  *ref* ``(n_snap, n_f)`` is the
    illuminator spectrum, *Gin* ``(n_f, n_cell)`` the field at each cell from the illuminator,
    *G* ``(n_f, n_cell, n_ch)`` the replicas.
    """
    n_snap, n_f, _ = d_res.shape
    Z = np.zeros((G.shape[1], n_snap), complex)
    norm = np.zeros(G.shape[1])
    for fi in range(n_f):
        y = G[fi].conj() @ d_res[:, fi, :].T                   # (n_cell, n_snap)
        Z += y * np.conj(Gin[fi])[:, None] * np.conj(ref[:, fi])[None, :]
        norm += np.sum(np.abs(G[fi]) ** 2, axis=1) * np.abs(Gin[fi]) ** 2 * np.mean(np.abs(ref[:, fi]) ** 2)
    return np.sum(np.abs(Z) ** 2, axis=1) / (norm * n_snap + 1e-30)


def sparse_image(d_res: np.ndarray, ref: np.ndarray, Gin: np.ndarray, G: np.ndarray, mask: np.ndarray | None = None,
                 n_iter: int = 80, lam_frac: float = 0.35) -> np.ndarray:
    """Sparse L1 inversion of the echo data for a scattering strength per cell (FISTA).

    Model: ``d[s, f, :] = ref[s, f] * sum_x m(x) Gin[f, x] G[f, x, :]``, with one complex
    ``m(x)`` per cell shared by all snapshots and frequencies.  Solves
    ``min 0.5 ||d - A m||^2 + lam ||m||_1`` with ``lam = lam_frac * max |A^H d|``.  The
    Bartlett map is ``|A^H d|^2``: its sidelobes are the point spread of the aperture, and the
    inversion removes them because it explains the data with as few cells as it can.
    Returns ``|m|`` per cell (0 where *mask* is False).
    """
    n_snap, n_f, n_ch = d_res.shape
    nc = G.shape[1]
    mask = np.ones(nc, bool) if mask is None else np.asarray(mask, bool)
    # Masked cells get zero weight instead of being cut out, so G (often > 1 GB) is never copied.
    pw = np.sum(np.abs(ref) ** 2, axis=0)                                     # (n_f,)
    col2 = np.zeros(nc)
    for fi in range(n_f):
        col2 += pw[fi] * np.abs(Gin[fi]) ** 2 * np.sum(np.abs(G[fi]) ** 2, axis=1)
    wcol = np.where(mask, 1.0 / (np.sqrt(col2) + 1e-30), 0.0)

    def A(m):
        out = np.empty((n_snap, n_f, n_ch), complex)
        mm = m * wcol
        for fi in range(n_f):
            out[:, fi, :] = ref[:, fi][:, None] * ((mm * Gin[fi]) @ G[fi])[None, :]
        return out

    def AH(r):
        g = np.zeros(nc, complex)
        for fi in range(n_f):
            g += np.conj(Gin[fi]) * (G[fi].conj() @ (r[:, fi, :].T @ np.conj(ref[:, fi])))
        return g * wcol

    # Lipschitz constant by power iteration.
    v = np.random.default_rng(0).standard_normal(nc) * mask + 0j
    for _ in range(20):
        v = AH(A(v)); L = np.linalg.norm(v); v /= L
    b = AH(d_res)
    lam = lam_frac * np.abs(b).max()
    m = np.zeros_like(b); y = m.copy(); t = 1.0
    for _ in range(n_iter):
        g = y - AH(A(y) - d_res) / L
        mag = np.abs(g)
        m_new = np.where(mag > lam / L, (1 - lam / L / np.maximum(mag, 1e-30)) * g, 0)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = m_new + (t - 1) / t_new * (m_new - m)
        m, t = m_new, t_new
    return np.where(mask, np.abs(m), 0.0)

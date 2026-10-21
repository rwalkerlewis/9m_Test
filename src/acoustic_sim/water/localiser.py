"""Two-stage empirical localiser: analytic MFP first-pass -> empirical replica refinement.

The replicas are measured transfer functions (for synthetic data: FDTD runs), so a localiser is
simulator knowledge on synthetic data and every run that uses one says so in its provenance
(``"empirical_replicas"``).

Two replica modes
-----------------
``vessel_path``
    The per-vessel unit gathers of the dataset's own FDTD run; built by the forward model, so it lives
    in ``water/diagnostics.py`` (:class:`~acoustic_sim.water.diagnostics.VesselPathLocaliser`).

``grid_survey``
    Loads per-cell RFFT snapshots produced by ``examples/harbor/90_history/harbor_survey_grid.py``, stored
    as ``{cache_dir}/survey_cache/{ix}_{iy}.npz`` (fields: ``traces`` (N, T),
    ``source_xy`` (2,)).  Covers the full scan grid regardless of vessel paths.

In both modes :meth:`EmpiricalLocaliser.refine` takes a set of candidate positions (from
the coarse analytic scan) and a :class:`~acoustic_sim.water.detection.Whitened` STFT
block, then returns refined ``(x, y)`` positions by scoring each candidate against the
empirical FDTD transfer function replica at the nearest cached location.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Avoid circular imports: detection imports from this module only at call sites.


@dataclass
class EmpiricalLocaliser:
    """Two-stage localiser: analytic MFP first-pass -> empirical replica refinement.

    Parameters
    ----------
    cache_dir : Path
        For ``vessel_path``: the gather cache directory (e.g. ``out/cache``).
        The dataset NPZ is expected at ``cache_dir.parent / "dataset.npz"``.
        For ``grid_survey``: the output directory that contains ``survey_cache/``.
    layout : SensorLayout
        Sensor layout (used for receiver count).
    fs : float
        Sample rate [Hz] of the cached traces.
    pc : ProcessingConfig
        Only ``.f_lo``, ``.f_hi``, ``.c``, ``.win_s`` are used.
    mode : str
        ``"vessel_path"`` or ``"grid_survey"``.
    refine_radius : float
        Maximum distance [m] from a candidate to its nearest replica point.
        Candidates with no replica within this radius are returned unchanged.
    """

    cache_dir: Path
    layout: object          # SensorLayout — typed as object to avoid circular imports
    fs: float
    pc: object              # ProcessingConfig — typed as object to avoid circular imports
    mode: str = "vessel_path"   # "vessel_path" | "grid_survey"
    refine_radius: float = 50.0

    # Internal: populated by build()
    _source_xy: np.ndarray = field(default=None, repr=False)   # (G_cache, 2)
    _H: np.ndarray = field(default=None, repr=False)           # (G_cache, N_recv, F_band) complex64
    _freqs: np.ndarray = field(default=None, repr=False)       # (F_band,)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> "EmpiricalLocaliser":
        """Load / compute the replica store.  Returns *self* for chaining."""
        if self.mode == "vessel_path":
            self._build_vessel_path()
        elif self.mode == "grid_survey":
            self._build_grid_survey()
        else:
            raise ValueError(f"Unknown mode {self.mode!r}; expected 'vessel_path' or 'grid_survey'")
        return self

    def refine(self, candidates_xy: np.ndarray, wh: object, frame_idx: int) -> np.ndarray:
        """Return refined positions for each candidate in *candidates_xy* (C, 2).

        For each candidate the nearest cached replica within ``refine_radius`` metres is
        found.  Among those neighbours the one with the highest normalised matched-field
        score against ``wh.Z[:, :, frame_idx]`` is returned as the refined position.
        If no cached point lies within the radius the candidate is returned unchanged.

        Parameters
        ----------
        candidates_xy : ndarray (C, 2)
            Candidate positions from the coarse analytic scan.
        wh : Whitened
            Whitened STFT block (``wh.Z`` has shape ``(N, F, K)``).
        frame_idx : int
            Frame index into the ``K`` axis of ``wh.Z``.

        Returns
        -------
        refined_xy : ndarray (C, 2)
        """
        if self._H is None or self._source_xy is None:
            raise RuntimeError("Call build() before refine().")

        candidates_xy = np.asarray(candidates_xy, dtype=float)
        if candidates_xy.ndim == 1:
            candidates_xy = candidates_xy[None]

        # Z_frame : (N, F_all) — full in-band STFT for this frame
        Z_frame = np.asarray(wh.Z[:, :, frame_idx], dtype=np.complex64)  # (N, F_all)

        # Restrict to the same frequency band as the stored replicas.
        freqs_all = np.asarray(wh.freqs)                                  # (F_all,)
        band_mask = (freqs_all >= self._freqs[0]) & (freqs_all <= self._freqs[-1])
        Z_band = Z_frame[:, band_mask]  # (N, F_band)

        out = candidates_xy.copy()
        for ci, cxy in enumerate(candidates_xy):
            dists = np.hypot(self._source_xy[:, 0] - cxy[0], self._source_xy[:, 1] - cxy[1])
            nearby = np.flatnonzero(dists <= self.refine_radius)
            if len(nearby) == 0:
                continue  # no empirical replica close enough — keep analytic candidate

            best_score = -1.0
            best_g = int(nearby[0])
            for g in nearby:
                H_g = self._H[g]                  # (N, F_band) complex64
                # score = |sum_n sum_f conj(H[g,n,f]) * Z[n,f]|^2 / |H[g]|^2
                inner = np.sum(np.conj(H_g) * Z_band)
                norm2 = float(np.sum(np.abs(H_g) ** 2))
                if norm2 < 1e-30:
                    continue
                score = float(np.abs(inner) ** 2) / norm2
                if score > best_score:
                    best_score = score
                    best_g = int(g)
            out[ci] = self._source_xy[best_g]
        return out

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _rfft_band(self, traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """RFFT of ``traces (N, T)`` restricted to [f_lo, f_hi].

        Returns
        -------
        H : ndarray (N, F_band) complex64
        freqs : ndarray (F_band,)
        """
        T = traces.shape[1]
        spec = np.fft.rfft(traces.astype(np.float64), axis=1)   # (N, T//2+1)
        freqs = np.fft.rfftfreq(T, 1.0 / self.fs)
        band = (freqs >= self.pc.f_lo) & (freqs <= self.pc.f_hi)
        return spec[:, band].astype(np.complex64), freqs[band]

    def _build_vessel_path(self) -> None:
        raise NotImplementedError(
            "vessel_path replicas are rendered by the forward model (FDTD gathers of the dataset's own "
            "vessels): use acoustic_sim.water.diagnostics.VesselPathLocaliser")

    def _build_grid_survey(self) -> None:
        """Build replica store from ``cache_dir/survey_cache/{ix}_{iy}.npz`` files."""
        survey_dir = self.cache_dir / "survey_cache"
        if not survey_dir.exists():
            raise FileNotFoundError(
                f"grid_survey mode requires {survey_dir}. "
                "Run examples/harbor/90_history/harbor_survey_grid.py first."
            )

        npz_files = sorted(survey_dir.glob("*.npz"))
        if not npz_files:
            raise RuntimeError(f"No .npz files found in {survey_dir}.")

        all_xy, all_H, common_freqs = [], [], None
        for p in npz_files:
            with np.load(p) as d:
                traces = d["traces"]           # (N, T)
                source_xy = d["source_xy"]     # (2,)
            H, freqs = self._rfft_band(traces)
            all_xy.append(np.asarray(source_xy, dtype=float))
            all_H.append(H)
            if common_freqs is None:
                common_freqs = freqs

        if not all_xy:
            raise RuntimeError(f"No survey cells loaded from {survey_dir}.")

        self._source_xy = np.array(all_xy, dtype=float)   # (G, 2)
        self._H = np.stack(all_H, axis=0)                 # (G, N, F)
        self._freqs = common_freqs

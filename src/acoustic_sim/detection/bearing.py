"""Bearing estimation algorithms for acoustic source localisation.

Provides a common interface for direction-of-arrival (DOA) estimation,
with multiple back-end algorithms selectable at runtime.

Algorithms
----------
SRP-PHAT
    Fast, single-peak by default.  Extended here with iterative peak
    finding and circular exclusion zones so it can return multiple
    bearings when ``max_sources > 1``.  Works well when sources are
    angularly well separated (> ~15°) and the array aperture is small
    relative to source range.

MUSIC
    Eigenvalue-based subspace method.  Resolves closely spaced sources
    (**super-resolution**) and naturally estimates the number of active
    sources via the eigenvalue gap.  More expensive per window than
    SRP-PHAT because it requires an eigendecomposition, but still
    real-time feasible for typical array sizes (≤ 24 mics).

MVDR (Capon)
    Minimum Variance Distortionless Response beamformer.  Wraps the
    broadband polar-grid matched-field processor in
    :mod:`acoustic_sim.processor`.  Returns bearing *and* range jointly,
    making it the most informative but also the most expensive method.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg as sp_linalg


# -----------------------------------------------------------------------
# Detection result container
# -----------------------------------------------------------------------

@dataclass
class BearingDetection:
    """A single bearing estimate with metadata."""

    bearing_rad: float
    """Bearing in radians, measured counter-clockwise from +x."""

    power: float = 0.0
    """Normalised power / pseudo-spectrum height at this bearing."""

    @property
    def bearing_deg(self) -> float:
        d = math.degrees(self.bearing_rad)
        return d if d >= 0 else d + 360.0


@dataclass
class BearingResult:
    """Output of a single DOA estimation call."""

    detections: list[BearingDetection]
    """Detected sources, ordered by descending power."""

    spectrum: np.ndarray | None = None
    """Full angular pseudo-spectrum (length ``n_bearings``)."""

    bearings_rad: np.ndarray | None = None
    """Scan angles corresponding to *spectrum*."""

    n_sources_estimated: int = 0
    """Number of sources estimated by the algorithm (e.g. eigenvalue gap)."""


# -----------------------------------------------------------------------
# Abstract base
# -----------------------------------------------------------------------

class BearingEstimator(ABC):
    """Interface every DOA algorithm must implement."""

    @abstractmethod
    def estimate(
        self,
        segment: np.ndarray,
        max_sources: int = 1,
    ) -> BearingResult:
        """Estimate bearing(s) from an ``(n_mics, n_samples)`` segment.

        Parameters
        ----------
        segment : (n_mics, n_samples) array
        max_sources : int
            Maximum number of sources to report.

        Returns
        -------
        BearingResult
        """
        ...


# -----------------------------------------------------------------------
# SRP-PHAT
# -----------------------------------------------------------------------

class SRPBeamformer(BearingEstimator):
    """SRP-PHAT (Steered Response Power with Phase Transform).

    Pre-computes steering vectors on a circular scan of ``n_bearings``
    directions over ``[0, 2π)``.  At call time, applies PHAT whitening
    and evaluates steered power via ``einsum``.

    Multi-source extension
    ~~~~~~~~~~~~~~~~~~~~~~
    When ``max_sources > 1`` in :meth:`estimate`, the spectrum is searched
    for multiple peaks with circular exclusion zones (default 15°).  Each
    peak after the first must exceed ``secondary_threshold`` times the
    global maximum.

    Parameters
    ----------
    mic_positions : (n_mics, 2+) array
        Only the first two columns (x, y) are used.
    fs : float
        Sample rate [Hz].
    window_samples : int
        Expected segment length for FFT sizing.
    c : float
        Speed of sound [m/s].
    n_bearings : int
        Angular resolution of the scan.
    freq_lo, freq_hi : float
        Passband for PHAT processing [Hz].
    min_peak_sep_deg : float
        Minimum angular separation between reported peaks.
    secondary_threshold : float
        Fraction of global-max power a secondary peak must exceed.
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        fs: float,
        window_samples: int,
        c: float = 343.0,
        n_bearings: int = 360,
        freq_lo: float = 100.0,
        freq_hi: float = 2000.0,
        min_peak_sep_deg: float = 15.0,
        secondary_threshold: float = 0.3,
    ) -> None:
        self.n_mics = mic_positions.shape[0]
        self.fs = fs
        self.c = c
        self.n_bearings = n_bearings
        self.min_peak_sep_deg = min_peak_sep_deg
        self.secondary_threshold = secondary_threshold

        mic_xy = np.asarray(mic_positions[:, :2], dtype=np.float64)
        center = mic_xy.mean(axis=0)
        mic_rel = mic_xy - center

        self.nfft = int(2 ** np.ceil(np.log2(window_samples)))
        freqs = np.fft.rfftfreq(self.nfft, d=1.0 / fs)
        fmask = (freqs >= freq_lo) & (freqs <= freq_hi)
        self.fmask = fmask
        omega = 2.0 * np.pi * freqs[fmask]

        self.bearings = np.linspace(0, 2 * np.pi, n_bearings, endpoint=False)
        look = np.column_stack([np.cos(self.bearings), np.sin(self.bearings)])

        # taus: (n_mics, n_bearings)  —  inter-mic delay for each look dir
        taus = -(mic_rel @ look.T) / c
        # steering: (n_mics, n_bearings, n_freq)
        self.steering = np.exp(
            1j * taus[:, :, np.newaxis] * omega[np.newaxis, np.newaxis, :]
        ).astype(np.complex64)

    # -- core spectrum computation ------------------------------------------

    def _compute_power(self, segment: np.ndarray) -> np.ndarray:
        """Return SRP-PHAT power over all scan bearings."""
        X = np.fft.rfft(segment.astype(np.float32), n=self.nfft, axis=1)
        X_bp = X[:, self.fmask]
        mag = np.abs(X_bp)
        mag[mag < 1e-30] = 1e-30
        X_phat = (X_bp / mag).astype(np.complex64)
        steered = np.einsum("mf,mbf->bf", X_phat, self.steering)
        return np.sum(np.abs(steered) ** 2, axis=1).real

    # -- peak finding -------------------------------------------------------

    def _find_peaks(
        self, power: np.ndarray, max_sources: int,
    ) -> list[BearingDetection]:
        """Iterative peak finding with circular exclusion zones."""
        detections: list[BearingDetection] = []
        mask = np.ones(len(power), dtype=bool)
        global_max = float(power.max()) if power.max() > 0 else 1.0
        sep_bins = max(1, int(self.min_peak_sep_deg / 360.0 * self.n_bearings))

        for _ in range(max_sources):
            masked_power = power * mask
            if masked_power.max() <= 0:
                break
            idx = int(np.argmax(masked_power))
            pk_power = float(masked_power[idx])
            if detections and pk_power < self.secondary_threshold * global_max:
                break

            detections.append(BearingDetection(
                bearing_rad=float(self.bearings[idx]),
                power=pk_power / global_max,
            ))

            # Circular exclusion zone.
            for d in range(-sep_bins, sep_bins + 1):
                mask[(idx + d) % self.n_bearings] = False

        return detections

    # -- public interface ---------------------------------------------------

    def estimate(
        self,
        segment: np.ndarray,
        max_sources: int = 1,
    ) -> BearingResult:
        power = self._compute_power(segment)

        if max_sources <= 1:
            idx = int(np.argmax(power))
            dets = [BearingDetection(
                bearing_rad=float(self.bearings[idx]),
                power=1.0,
            )]
        else:
            dets = self._find_peaks(power, max_sources)

        return BearingResult(
            detections=dets,
            spectrum=power,
            bearings_rad=self.bearings.copy(),
            n_sources_estimated=len(dets),
        )


# -----------------------------------------------------------------------
# MUSIC (Multiple Signal Classification)
# -----------------------------------------------------------------------

class MUSICEstimator(BearingEstimator):
    """MUSIC (MUltiple SIgnal Classification) DOA estimator.

    Resolves multiple closely spaced sources via eigenvalue decomposition
    of the spatial covariance matrix.

    The number of sources is estimated from the eigenvalue profile using
    the MDL (Minimum Description Length) criterion unless the caller
    specifies a fixed count.

    Parameters
    ----------
    mic_positions : (n_mics, 2+) array
    fs : float
        Sample rate [Hz].
    window_samples : int
    c : float
        Speed of sound.
    n_bearings : int
        Angular resolution.
    freq_lo, freq_hi : float
        Passband [Hz].
    n_subbands : int
        Number of frequency bins to average for the CSDM.
    diagonal_loading : float
        Regularisation added to the covariance diagonal.
    min_peak_sep_deg : float
        Minimum angular separation between reported peaks.
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        fs: float,
        window_samples: int,
        c: float = 343.0,
        n_bearings: int = 360,
        freq_lo: float = 100.0,
        freq_hi: float = 2000.0,
        n_subbands: int = 0,
        diagonal_loading: float = 0.01,
        min_peak_sep_deg: float = 10.0,
    ) -> None:
        self.n_mics = mic_positions.shape[0]
        self.fs = fs
        self.c = c
        self.n_bearings = n_bearings
        self.diagonal_loading = diagonal_loading
        self.min_peak_sep_deg = min_peak_sep_deg

        mic_xy = np.asarray(mic_positions[:, :2], dtype=np.float64)
        self.center = mic_xy.mean(axis=0)
        self.mic_rel = mic_xy - self.center

        self.nfft = int(2 ** np.ceil(np.log2(window_samples)))
        freqs = np.fft.rfftfreq(self.nfft, d=1.0 / fs)
        fmask = (freqs >= freq_lo) & (freqs <= freq_hi)
        self.fmask = fmask
        self.freqs_bp = freqs[fmask]

        if n_subbands <= 0:
            n_subbands = max(1, int(np.sum(fmask)))
        self.n_subbands = min(n_subbands, int(np.sum(fmask)))

        # Pre-compute steering vectors: (n_bearings, n_freq, n_mics)
        self.bearings = np.linspace(0, 2 * np.pi, n_bearings, endpoint=False)
        look = np.column_stack([np.cos(self.bearings), np.sin(self.bearings)])
        taus = -(self.mic_rel @ look.T) / c  # (n_mics, n_bearings)
        # steering_vectors[b, f] = exp(-j 2π f τ) for each mic
        self._sv = np.exp(
            -1j * 2.0 * np.pi
            * self.freqs_bp[np.newaxis, :, np.newaxis]
            * taus.T[:, np.newaxis, :]
        ).astype(np.complex128)  # (n_bearings, n_freq, n_mics)

    # -- covariance estimation ----------------------------------------------

    def _compute_csdm(
        self, segment: np.ndarray,
    ) -> np.ndarray:
        """Compute cross-spectral density matrix averaged over sub-bands.

        Returns (n_mics, n_mics) complex Hermitian matrix.
        """
        X = np.fft.rfft(segment, n=self.nfft, axis=1)
        X_bp = X[:, self.fmask]  # (n_mics, n_freq_bp)

        # Average CSDM over selected frequency bins.
        n_freq = X_bp.shape[1]
        step = max(1, n_freq // self.n_subbands)
        R = np.zeros((self.n_mics, self.n_mics), dtype=np.complex128)
        count = 0
        for start in range(0, n_freq, step):
            end = min(start + step, n_freq)
            D = X_bp[:, start:end]  # (n_mics, sub_band)
            R += D @ D.conj().T
            count += end - start
        R /= max(count, 1)

        # Diagonal loading for numerical stability.
        R += self.diagonal_loading * np.trace(R).real / self.n_mics * np.eye(self.n_mics)
        return R

    # -- source count estimation (MDL) --------------------------------------

    @staticmethod
    def _estimate_n_sources_mdl(
        eigenvalues: np.ndarray, n_samples: int,
    ) -> int:
        """Minimum Description Length criterion for source enumeration.

        Parameters
        ----------
        eigenvalues : sorted descending eigenvalues of the CSDM.
        n_samples : number of snapshots used to form the CSDM.
        """
        M = len(eigenvalues)
        N = max(n_samples, 1)
        ev = np.maximum(eigenvalues.real, 1e-30)

        best_k = 0
        best_mdl = float("inf")

        for k in range(M - 1):
            noise_ev = ev[k + 1:]
            p = len(noise_ev)
            if p == 0:
                break
            geo_mean = np.exp(np.mean(np.log(noise_ev)))
            arith_mean = np.mean(noise_ev)
            if arith_mean < 1e-30:
                break

            # Log-likelihood term.
            ll = -N * p * np.log(geo_mean / arith_mean)
            # Penalty: number of free parameters.
            n_free = k * (2 * M - k)
            penalty = 0.5 * n_free * np.log(N)
            mdl = ll + penalty

            if mdl < best_mdl:
                best_mdl = mdl
                best_k = k

        return max(best_k, 1)

    # -- MUSIC pseudo-spectrum ----------------------------------------------

    def _music_spectrum(
        self,
        R: np.ndarray,
        n_sources: int,
    ) -> np.ndarray:
        """Compute broadband MUSIC pseudo-spectrum.

        Averages the per-frequency MUSIC spectrum across the passband.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        # eigh returns ascending order; we want descending.
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Noise subspace: columns after the first n_sources.
        En = eigenvectors[:, n_sources:]  # (n_mics, n_mics - n_sources)

        # For computational efficiency, compute the projection matrix once.
        Pn = En @ En.conj().T  # (n_mics, n_mics)

        # Broadband: average per-frequency MUSIC spectrum.
        n_freq = self._sv.shape[1]
        spectrum = np.zeros(self.n_bearings)

        for fi in range(n_freq):
            a = self._sv[:, fi, :]  # (n_bearings, n_mics)
            # MUSIC denominator: a^H P_n a   for each bearing.
            denom = np.real(np.sum((a @ Pn) * a.conj(), axis=1))
            denom = np.maximum(denom, 1e-30)
            spectrum += 1.0 / denom

        spectrum /= max(n_freq, 1)
        return spectrum

    # -- peak finding -------------------------------------------------------

    def _find_peaks(
        self,
        spectrum: np.ndarray,
        max_sources: int,
    ) -> list[BearingDetection]:
        detections: list[BearingDetection] = []
        mask = np.ones(len(spectrum), dtype=bool)
        global_max = float(spectrum.max()) if spectrum.max() > 0 else 1.0
        sep_bins = max(1, int(self.min_peak_sep_deg / 360.0 * self.n_bearings))

        for _ in range(max_sources):
            masked = spectrum * mask
            if masked.max() <= 0:
                break
            idx = int(np.argmax(masked))
            pk = float(masked[idx])

            detections.append(BearingDetection(
                bearing_rad=float(self.bearings[idx]),
                power=pk / global_max,
            ))
            for d in range(-sep_bins, sep_bins + 1):
                mask[(idx + d) % self.n_bearings] = False

        return detections

    # -- public interface ---------------------------------------------------

    def estimate(
        self,
        segment: np.ndarray,
        max_sources: int = 1,
    ) -> BearingResult:
        R = self._compute_csdm(segment)

        # Eigenvalue decomposition for source count.
        eigenvalues = np.linalg.eigvalsh(R)[::-1]
        n_est = self._estimate_n_sources_mdl(eigenvalues, segment.shape[1])
        n_sources = min(n_est, max_sources)

        spectrum = self._music_spectrum(R, n_sources)
        detections = self._find_peaks(spectrum, max_sources)

        return BearingResult(
            detections=detections,
            spectrum=spectrum,
            bearings_rad=self.bearings.copy(),
            n_sources_estimated=n_est,
        )


# -----------------------------------------------------------------------
# MVDR wrapper  (delegates to processor.matched_field_process)
# -----------------------------------------------------------------------

class MVDRBeamformer(BearingEstimator):
    """Broadband MVDR (Capon) beamformer wrapping the MFP in processor.py.

    Unlike SRP-PHAT and MUSIC which estimate bearing only, MVDR searches
    a 2-D polar grid (azimuth × range) and returns the joint optimum.
    The ``BearingDetection.power`` field carries the broadband beam power
    and the range is stored as an extra attribute on the result.

    Parameters
    ----------
    mic_positions : (n_mics, 2) array
    dt : float
        Sampling interval [s].
    sound_speed : float
    azimuth_spacing_deg : float
    range_min, range_max, range_spacing : float
    fundamental : float
        Source fundamental frequency for harmonic bin selection.
    n_harmonics : int
    harmonic_bandwidth : float
    window_length : float
        MFP window duration in seconds.
    detection_threshold : float
    diagonal_loading : float
    max_sources : int
        Passed to MFP peak finder.
    min_source_separation_deg : float
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        dt: float,
        sound_speed: float = 343.0,
        azimuth_spacing_deg: float = 1.0,
        range_min: float = 20.0,
        range_max: float = 500.0,
        range_spacing: float = 5.0,
        fundamental: float = 150.0,
        n_harmonics: int = 6,
        harmonic_bandwidth: float = 10.0,
        window_length: float = 0.2,
        window_overlap: float = 0.5,
        n_subwindows: int = 4,
        detection_threshold: float = 0.25,
        diagonal_loading: float = 0.01,
        max_sources: int = 1,
        min_source_separation_deg: float = 10.0,
        stationary_history: int = 10,
        stationary_cv_threshold: float = 0.15,
    ) -> None:
        self.mic_positions = np.asarray(mic_positions, dtype=np.float64)
        self.dt = dt
        self.sound_speed = sound_speed
        self._mfp_kwargs = dict(
            azimuth_spacing_deg=azimuth_spacing_deg,
            range_min=range_min,
            range_max=range_max,
            range_spacing=range_spacing,
            window_length=window_length,
            window_overlap=window_overlap,
            n_subwindows=n_subwindows,
            detection_threshold=detection_threshold,
            fundamental=fundamental,
            n_harmonics=n_harmonics,
            harmonic_bandwidth=harmonic_bandwidth,
            stationary_history=stationary_history,
            stationary_cv_threshold=stationary_cv_threshold,
            diagonal_loading=diagonal_loading,
            max_sources=max_sources,
            min_source_separation_deg=min_source_separation_deg,
        )

    def estimate(
        self,
        segment: np.ndarray,
        max_sources: int = 1,
    ) -> BearingResult:
        from acoustic_sim.processor import matched_field_process

        kwargs = dict(self._mfp_kwargs)
        kwargs["max_sources"] = max_sources

        result = matched_field_process(
            segment,
            self.mic_positions,
            self.dt,
            sound_speed=self.sound_speed,
            **kwargs,
        )

        detections: list[BearingDetection] = []
        for det in result.get("detections", []):
            detections.append(BearingDetection(
                bearing_rad=float(np.radians(det.get("azimuth_deg", 0.0))),
                power=float(det.get("power", 0.0)),
            ))

        # Also pull from multi_detections if present.
        if max_sources > 1:
            for md in result.get("multi_detections", []):
                for det in md:
                    already = any(
                        abs(d.bearing_rad - np.radians(det.get("azimuth_deg", 0.0))) < 0.01
                        for d in detections
                    )
                    if not already:
                        detections.append(BearingDetection(
                            bearing_rad=float(np.radians(det.get("azimuth_deg", 0.0))),
                            power=float(det.get("power", 0.0)),
                        ))

        # Sort by power descending.
        detections.sort(key=lambda d: d.power, reverse=True)

        return BearingResult(
            detections=detections[:max_sources],
            spectrum=None,
            bearings_rad=None,
            n_sources_estimated=len(detections),
        )


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------

_BEARING_METHODS = {
    "srp_phat": SRPBeamformer,
    "music": MUSICEstimator,
    "mvdr": MVDRBeamformer,
}


def available_bearing_methods() -> list[str]:
    """Return names of all registered bearing estimation methods."""
    return list(_BEARING_METHODS.keys())


def create_bearing_estimator(
    method: str,
    **kwargs,
) -> BearingEstimator:
    """Create a bearing estimator by name.

    Parameters
    ----------
    method : str
        One of ``"srp_phat"``, ``"music"``, ``"mvdr"``.
    **kwargs
        Forwarded to the estimator constructor.
    """
    cls = _BEARING_METHODS.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown bearing method {method!r}. "
            f"Available: {available_bearing_methods()}"
        )
    return cls(**kwargs)

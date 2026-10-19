"""Range estimation algorithms for acoustic source localisation.

Provides:

* **RMS-based** ranging exploiting 1/r amplitude decay (fast, needs CPA
  calibration).
* **TDOA multilateration** using GCC-PHAT time delays and bearing-
  constrained least-squares to resolve range from microphone pair
  cross-correlations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class RangeEstimate:
    """Single range estimate with uncertainty."""

    range_m: float
    """Estimated range [m]."""

    uncertainty_m: float = float("nan")
    """1-σ uncertainty [m], if available."""


class RangeEstimator(ABC):
    """Interface for range estimation algorithms."""

    @abstractmethod
    def estimate(
        self,
        segment: np.ndarray,
        bearing_rad: float | None = None,
    ) -> RangeEstimate:
        """Estimate range from a windowed segment.

        Parameters
        ----------
        segment : (n_mics, n_samples)
        bearing_rad : optional bearing hint.
        """
        ...


class RMSRangeEstimator(RangeEstimator):
    r"""Estimate range from RMS amplitude via the inverse-square law.

    Assumes free-field spherical spreading.  The range estimate is:

    .. math::

        \hat{R} = R_{\text{ref}} \sqrt{\frac{A_{\text{ref}}}{A_{\text{obs}}}}

    where :math:`A_{\text{ref}}` is a calibrated RMS amplitude at known
    range :math:`R_{\text{ref}}`.

    Calibration
    ~~~~~~~~~~~
    Call :meth:`calibrate` with the peak RMS and the corresponding
    ground-truth distance (closest point of approach), *or* supply
    ``ref_rms`` and ``ref_range`` directly at construction.

    Parameters
    ----------
    ref_range : float
        Reference range [m].
    ref_rms : float or None
        RMS amplitude at ``ref_range``.  Set via :meth:`calibrate` if
        not known in advance.
    range_min, range_max : float
        Hard clamp on returned estimates.
    """

    def __init__(
        self,
        ref_range: float = 10.0,
        ref_rms: float | None = None,
        range_min: float = 5.0,
        range_max: float = 100.0,
    ) -> None:
        self.ref_range = ref_range
        self.ref_rms = ref_rms
        self.range_min = range_min
        self.range_max = range_max

    def calibrate(
        self,
        peak_rms: float,
        cpa_distance: float,
    ) -> None:
        """Set reference from CPA (closest point of approach).

        Parameters
        ----------
        peak_rms : float
            Maximum observed RMS across the simulation.
        cpa_distance : float
            True distance at the moment of peak RMS.
        """
        rms_times_range = peak_rms * max(cpa_distance, 1.0)
        self.ref_rms = rms_times_range / self.ref_range

    @property
    def is_calibrated(self) -> bool:
        return self.ref_rms is not None

    def estimate(
        self,
        segment: np.ndarray,
        bearing_rad: float | None = None,
    ) -> RangeEstimate:
        if self.ref_rms is None:
            return RangeEstimate(range_m=(self.range_min + self.range_max) / 2)

        window_rms = float(np.sqrt(np.mean(segment ** 2)))
        if window_rms < 1e-12:
            return RangeEstimate(range_m=self.range_max)

        est = self.ref_range * math.sqrt(self.ref_rms / window_rms)
        est = max(self.range_min, min(self.range_max, est))

        # Rough uncertainty: proportional to range (large range = noisy).
        unc = est * 0.3
        return RangeEstimate(range_m=est, uncertainty_m=unc)


class TDOARangeEstimator(RangeEstimator):
    r"""Range estimator using GCC-PHAT TDOA multilateration.

    Given a bearing from the DOA estimator, searches along the bearing
    ray for the range that best explains the observed time-delay-of-
    arrival (TDOA) across all microphone pairs.

    .. note:: Requires a sample rate high enough that the maximum
       inter-microphone TDOA spans many samples (rule of thumb:
       :math:`f_s \cdot d_{\max}/c \gtrsim 50`).  At low sample rates
       the GCC-PHAT peaks are unreliable and
       :class:`BearingRateRangeEstimator` is preferred.

    Parameters
    ----------
    mic_positions : (n_mics, 2+)
    fs, window_samples, c : float
    range_min, range_max, n_range_bins : float / int
    freq_lo, freq_hi : float
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        fs: float,
        window_samples: int,
        c: float = 343.0,
        range_min: float = 2.0,
        range_max: float = 200.0,
        n_range_bins: int = 80,
        freq_lo: float = 100.0,
        freq_hi: float = 2000.0,
        **_kwargs,
    ) -> None:
        self.mic_positions = np.asarray(mic_positions, dtype=float)[:, :2]
        self.array_center = self.mic_positions.mean(axis=0)
        self.fs = fs
        self.c = c
        self.n_fft = window_samples
        self.range_min = range_min
        self.range_max = range_max
        self.n_range_bins = n_range_bins

        freqs = np.fft.rfftfreq(window_samples, d=1.0 / fs)
        mask = (freqs >= freq_lo) & (freqs <= freq_hi)
        self.freq_idx = np.where(mask)[0]

        n_mics = self.mic_positions.shape[0]
        pairs_i, pairs_j = [], []
        for i in range(n_mics):
            for j in range(i + 1, n_mics):
                pairs_i.append(i)
                pairs_j.append(j)
        self._pair_i = np.array(pairs_i)
        self._pair_j = np.array(pairs_j)

        max_dist = 0.0
        for pi, pj in zip(pairs_i, pairs_j):
            d = float(np.linalg.norm(
                self.mic_positions[pi] - self.mic_positions[pj]))
            if d > max_dist:
                max_dist = d
        self._max_lag = int(math.ceil(max_dist / c * fs)) + 2
        self._mic_x = self.mic_positions[:, 0]
        self._mic_y = self.mic_positions[:, 1]

    def _gcc_phat_tdoas(self, segment: np.ndarray) -> np.ndarray:
        X = np.fft.rfft(segment, n=self.n_fft, axis=1)
        X_band = X[:, self.freq_idx]
        G_i = X_band[self._pair_i]
        G_j = X_band[self._pair_j]
        G = G_i * np.conj(G_j)
        denom = np.abs(G)
        denom[denom < 1e-12] = 1e-12
        G_phat = G / denom

        n_rfft = self.n_fft // 2 + 1
        G_full = np.zeros((len(self._pair_i), n_rfft), dtype=complex)
        G_full[:, self.freq_idx] = G_phat
        cc = np.fft.irfft(G_full, n=self.n_fft)

        max_lag = min(self._max_lag, self.n_fft // 2)
        lags = np.concatenate([
            np.arange(0, max_lag + 1),
            np.arange(self.n_fft - max_lag, self.n_fft),
        ])
        lag_values = np.concatenate([
            np.arange(0, max_lag + 1),
            np.arange(-max_lag, 0),
        ])
        cc_window = cc[:, lags]
        peak_idx = np.argmax(cc_window, axis=1)
        tdoa_samples = lag_values[peak_idx].astype(float)

        for p in range(len(self._pair_i)):
            pk = peak_idx[p]
            if 0 < pk < cc_window.shape[1] - 1:
                y_m = cc_window[p, pk - 1]
                y_0 = cc_window[p, pk]
                y_p = cc_window[p, pk + 1]
                denom_para = y_m - 2.0 * y_0 + y_p
                if abs(denom_para) > 1e-15:
                    delta = 0.5 * (y_m - y_p) / denom_para
                    tdoa_samples[p] = lag_values[pk] + delta
        return tdoa_samples / self.fs

    def _tdoa_cost(self, R, cos_b, sin_b, obs_tdoa):
        R = np.atleast_1d(R)
        src_x = self.array_center[0] + R * cos_b
        src_y = self.array_center[1] + R * sin_b
        dx_i = self._mic_x[self._pair_i][None, :] - src_x[:, None]
        dy_i = self._mic_y[self._pair_i][None, :] - src_y[:, None]
        dx_j = self._mic_x[self._pair_j][None, :] - src_x[:, None]
        dy_j = self._mic_y[self._pair_j][None, :] - src_y[:, None]
        d_i = np.sqrt(dx_i**2 + dy_i**2)
        d_j = np.sqrt(dx_j**2 + dy_j**2)
        pred_tdoa = (d_i - d_j) / self.c
        residuals = pred_tdoa - obs_tdoa[None, :]
        return np.sum(residuals**2, axis=1)

    def estimate(self, segment, bearing_rad=None):
        if bearing_rad is None:
            return RangeEstimate(
                range_m=(self.range_min + self.range_max) / 2)
        obs_tdoa = self._gcc_phat_tdoas(segment)
        cos_b, sin_b = math.cos(bearing_rad), math.sin(bearing_rad)
        ranges = np.geomspace(
            self.range_min, self.range_max, self.n_range_bins)
        costs = self._tdoa_cost(ranges, cos_b, sin_b, obs_tdoa)
        best_idx = int(np.argmin(costs))
        est_range = float(ranges[best_idx])
        if 0 < best_idx < self.n_range_bins - 1:
            lo = float(ranges[max(best_idx - 1, 0)])
            hi = float(ranges[min(best_idx + 1, self.n_range_bins - 1)])
            gr = (math.sqrt(5) + 1) / 2
            for _ in range(20):
                if hi - lo < 1e-4:
                    break
                c1 = hi - (hi - lo) / gr
                c2 = lo + (hi - lo) / gr
                f1 = float(self._tdoa_cost(
                    np.array([c1]), cos_b, sin_b, obs_tdoa)[0])
                f2 = float(self._tdoa_cost(
                    np.array([c2]), cos_b, sin_b, obs_tdoa)[0])
                if f1 < f2:
                    hi = c2
                else:
                    lo = c1
            est_range = (lo + hi) / 2.0
        est_range = max(self.range_min, min(self.range_max, est_range))
        unc = est_range * 0.5
        return RangeEstimate(range_m=est_range, uncertainty_m=unc)


class BearingRateRangeEstimator(RangeEstimator):
    r"""Range estimator from bearing rate of change.

    When the source speed is approximately known, range can be inferred
    from the rate at which the bearing changes:

    .. math::

        R \approx \frac{v_\perp}{|\dot\theta|}

    where :math:`v_\perp = v \sin\alpha` is the component of source
    velocity perpendicular to the current line-of-sight, and
    :math:`\alpha` is the angle between the velocity vector and the
    line-of-sight (estimated from consecutive bearings).

    The method is model-free — it requires no microphone-pair cross-
    correlation and is therefore immune to the low-sample-rate TDOA
    resolution issues that plague small arrays.

    A causal EMA filter smooths the estimated bearing rate to reduce
    noise from single-window bearing jitter.

    Parameters
    ----------
    source_speed : float
        Estimated source speed [m/s].
    hop_sec : float
        Time between consecutive windows [s].
    ema_alpha : float
        EMA smoothing factor for bearing rate (0 < α ≤ 1).
    range_min, range_max : float
        Hard clamps on returned estimates [m].
    min_rate_dps : float
        Minimum bearing rate [deg/s] below which range is clamped to
        ``range_max`` (avoids division-by-near-zero far from CPA).
    """

    def __init__(
        self,
        source_speed: float = 50.0,
        hop_sec: float = 0.025,
        ema_alpha: float = 0.30,
        range_min: float = 2.0,
        range_max: float = 200.0,
        min_rate_dps: float = 5.0,
        **_kwargs,
    ) -> None:
        self.source_speed = source_speed
        self.hop_sec = hop_sec
        self.range_min = range_min
        self.range_max = range_max
        self.ema_alpha = ema_alpha
        self.min_rate_rad = math.radians(min_rate_dps)

        # State: previous bearing and smoothed bearing rate.
        self._prev_bearing: float | None = None
        self._smoothed_rate: float = 0.0

    def reset(self) -> None:
        self._prev_bearing = None
        self._smoothed_rate = 0.0

    def estimate(
        self,
        segment: np.ndarray,
        bearing_rad: float | None = None,
    ) -> RangeEstimate:
        if bearing_rad is None:
            return RangeEstimate(
                range_m=(self.range_min + self.range_max) / 2,
            )

        if self._prev_bearing is None:
            self._prev_bearing = bearing_rad
            return RangeEstimate(range_m=self.range_max, uncertainty_m=self.range_max * 0.5)

        # Bearing difference (handle wrap-around).
        d_theta = bearing_rad - self._prev_bearing
        if d_theta > math.pi:
            d_theta -= 2.0 * math.pi
        elif d_theta < -math.pi:
            d_theta += 2.0 * math.pi
        self._prev_bearing = bearing_rad

        raw_rate = d_theta / self.hop_sec  # rad/s

        # EMA smoothing of the bearing rate.
        alpha = self.ema_alpha
        self._smoothed_rate = (
            alpha * raw_rate + (1.0 - alpha) * self._smoothed_rate
        )

        abs_rate = abs(self._smoothed_rate)
        if abs_rate < self.min_rate_rad:
            # Bearing barely changing → source is far away or radial.
            return RangeEstimate(
                range_m=self.range_max,
                uncertainty_m=self.range_max * 0.5,
            )

        # R ≈ v / |dθ/dt|  (assuming crossing geometry, v_⊥ ≈ v).
        # For non-crossing geometries this over-estimates range, but
        # that is the safe direction (conservative engagement).
        est = self.source_speed / abs_rate
        est = max(self.range_min, min(self.range_max, est))

        # Rough uncertainty: proportional to range.
        unc = est * 0.35

        return RangeEstimate(range_m=est, uncertainty_m=unc)


_RANGE_METHODS = {
    "rms": RMSRangeEstimator,
    "tdoa": TDOARangeEstimator,
    "bearing_rate": BearingRateRangeEstimator,
    "nearfield": TDOARangeEstimator,  # backward-compat alias
}


def available_range_methods() -> list[str]:
    return list(_RANGE_METHODS.keys())


def create_range_estimator(method: str = "rms", **kwargs) -> RangeEstimator:
    """Create a range estimator by name."""
    cls = _RANGE_METHODS.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown range method {method!r}. "
            f"Available: {available_range_methods()}"
        )
    return cls(**kwargs)

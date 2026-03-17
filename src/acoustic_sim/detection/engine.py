"""Unified detection engine with pluggable algorithm selection.

Combines bearing estimation, range estimation, and causal tracking into
a single streaming processor.  The user selects which algorithms to use
(defaulting to SRP-PHAT + RMS ranging + WLS tracking) while retaining
automatic fallback to MUSIC when multiple sources are detected.

Example
-------
::

    from acoustic_sim.detection import DetectionEngine

    engine = DetectionEngine(
        mic_positions=mic_pos,
        fs=1.0 / dt,
        window_samples=win_len,
        bearing_method="srp_phat",   # or "music", "mvdr"
        max_sources=2,               # auto-switch to MUSIC if > 1 detected
    )
    engine.calibrate_range(peak_rms, cpa_distance)

    for seg in windows:
        result = engine.process_window(seg, t_center)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from acoustic_sim.detection.bearing import (
    BearingDetection,
    BearingEstimator,
    BearingResult,
    MUSICEstimator,
    SRPBeamformer,
    create_bearing_estimator,
)
from acoustic_sim.detection.ranging import (
    RangeEstimate,
    RangeEstimator,
    RMSRangeEstimator,
    create_range_estimator,
)
from acoustic_sim.detection.tracking import (
    CausalWLSTracker,
    EMABearingSmoother,
    TrackState,
)


@dataclass
class WindowDetection:
    """Result of processing a single window."""

    time: float
    detected: bool
    window_rms: float

    # Per-source detections (primary is index 0).
    bearings: list[BearingDetection] = field(default_factory=list)

    # Primary source localisation (after EMA + range).
    bearing_rad: float = float("nan")
    bearing_deg: float = float("nan")
    range_m: float = float("nan")
    x: float = float("nan")
    y: float = float("nan")
    z: float = float("nan")

    # Track state (may be None if not enough detections yet).
    track: TrackState | None = None

    # Multi-source flag.
    n_sources: int = 0

    # Algorithm that produced the bearing.
    bearing_method: str = ""


class DetectionEngine:
    """Streaming detection processor with pluggable algorithms.

    Parameters
    ----------
    mic_positions : (n_mics, 2+) array
    fs : float
        Sample rate [Hz].
    window_samples : int
    bearing_method : str
        Primary DOA algorithm: ``"srp_phat"`` (default), ``"music"``,
        or ``"mvdr"``.  When ``max_sources > 1`` and the primary method
        is SRP-PHAT, the engine automatically promotes to MUSIC for the
        multi-source search.
    range_method : str
        Range estimation algorithm: ``"rms"`` (default).
    max_sources : int
        Maximum number of source bearings to report per window.
    min_signal_rms : float
        Windows below this RMS are skipped.
    ema_alpha : float
        EMA smoothing factor for bearing.
    source_z_estimate : float
        Assumed source altitude when converting bearing+range to (x, y, z).
    c : float
        Speed of sound [m/s].
    bearing_kwargs : dict
        Extra keyword arguments forwarded to the bearing estimator.
    range_kwargs : dict
        Extra keyword arguments forwarded to the range estimator.
    tracker_min_detections : int
        Minimum detections before the WLS tracker produces a fit.
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        fs: float,
        window_samples: int,
        bearing_method: str = "srp_phat",
        range_method: str = "rms",
        max_sources: int = 1,
        min_signal_rms: float = 5e-5,
        ema_alpha: float = 0.35,
        source_z_estimate: float = 0.0,
        c: float = 343.0,
        bearing_kwargs: dict | None = None,
        range_kwargs: dict | None = None,
        tracker_min_detections: int = 5,
        tracker_max_history: int = 20,
    ) -> None:
        self.mic_positions = np.asarray(mic_positions)
        self.array_center = self.mic_positions.mean(axis=0)
        self.fs = fs
        self.window_samples = window_samples
        self.max_sources = max_sources
        self.min_signal_rms = min_signal_rms
        self.source_z_estimate = source_z_estimate
        self.bearing_method_name = bearing_method

        # Build primary bearing estimator.
        bkw = dict(
            mic_positions=mic_positions,
            fs=fs,
            window_samples=window_samples,
            c=c,
        )
        if bearing_kwargs:
            bkw.update(bearing_kwargs)
        self.bearing_estimator = create_bearing_estimator(bearing_method, **bkw)

        # If primary is SRP-PHAT and max_sources > 1, also build MUSIC
        # as a fallback for multi-source resolution.
        self._music_fallback: MUSICEstimator | None = None
        if bearing_method == "srp_phat" and max_sources > 1:
            music_kw = dict(bkw)
            # Remove SRP-specific keys that MUSIC doesn't accept.
            music_kw.pop("min_peak_sep_deg", None)
            music_kw.pop("secondary_threshold", None)
            self._music_fallback = MUSICEstimator(**music_kw)

        # Range estimator.
        rkw = dict(range_kwargs or {})
        # Near-field and future range methods need array geometry.
        if range_method != "rms":
            rkw.setdefault("mic_positions", mic_positions)
            rkw.setdefault("fs", fs)
            rkw.setdefault("window_samples", window_samples)
            rkw.setdefault("c", c)
        self.range_method_name = range_method
        self.range_estimator = create_range_estimator(range_method, **rkw)

        # Tracking.
        self.smoother = EMABearingSmoother(alpha=ema_alpha)
        self.tracker = CausalWLSTracker(min_detections=tracker_min_detections,
                                        max_history=tracker_max_history)

    def calibrate_range(self, peak_rms: float, cpa_distance: float) -> None:
        """Calibrate the range estimator from CPA observation."""
        if isinstance(self.range_estimator, RMSRangeEstimator):
            self.range_estimator.calibrate(peak_rms, cpa_distance)

    def reset(self) -> None:
        """Clear all internal state for a new scenario."""
        self.smoother.reset()
        self.tracker.reset()

    def process_window(
        self,
        segment: np.ndarray,
        t_center: float,
    ) -> WindowDetection:
        """Process one window of microphone data.

        Parameters
        ----------
        segment : (n_mics, n_samples)
        t_center : float
            Wall-clock time of the window centre.

        Returns
        -------
        WindowDetection
        """
        window_rms = float(np.sqrt(np.mean(segment ** 2)))
        detected = window_rms >= self.min_signal_rms

        result = WindowDetection(
            time=t_center,
            detected=detected,
            window_rms=window_rms,
        )

        if not detected:
            return result

        # -- Bearing estimation -----------------------------------------------
        bearing_result = self.bearing_estimator.estimate(
            segment, max_sources=self.max_sources,
        )
        method_used = self.bearing_method_name

        # Auto-promote to MUSIC if SRP-PHAT found > 1 source.
        if (
            self._music_fallback is not None
            and bearing_result.n_sources_estimated > 1
        ):
            bearing_result = self._music_fallback.estimate(
                segment, max_sources=self.max_sources,
            )
            method_used = "music"

        result.bearings = bearing_result.detections
        result.n_sources = len(bearing_result.detections)
        result.bearing_method = method_used

        if not bearing_result.detections:
            return result

        # Use the strongest detection as the primary.
        primary = bearing_result.detections[0]
        smoothed_bearing = self.smoother.update(primary.bearing_rad)

        # -- Range estimation -------------------------------------------------
        range_est = self.range_estimator.estimate(segment, smoothed_bearing)

        # -- Localisation -----------------------------------------------------
        cx, cy = float(self.array_center[0]), float(self.array_center[1])
        est_x = cx + range_est.range_m * math.cos(smoothed_bearing)
        est_y = cy + range_est.range_m * math.sin(smoothed_bearing)
        est_z = self.source_z_estimate

        result.bearing_rad = smoothed_bearing
        result.bearing_deg = math.degrees(smoothed_bearing)
        if result.bearing_deg < 0:
            result.bearing_deg += 360.0
        result.range_m = range_est.range_m
        result.x = est_x
        result.y = est_y
        result.z = est_z

        # -- Track update -----------------------------------------------------
        self.tracker.add_detection(t_center, est_x, est_y, est_z, window_rms)
        result.track = self.tracker.fit()

        return result

"""Acoustic source detection and localisation algorithms.

This package provides pluggable bearing estimation, range estimation,
and causal tracking for real-time acoustic pipelines.

Algorithms
----------
**Bearing estimation** (DOA):

* ``srp_phat`` — Steered Response Power with Phase Transform.  Fast,
  default choice.  Extended with multi-peak search and automatic MUSIC
  fallback when multiple sources are detected.
* ``music`` — Multiple Signal Classification.  Eigenvalue-based
  super-resolution; resolves closely spaced sources.
* ``mvdr`` — Minimum Variance Distortionless Response (Capon).  Joint
  bearing + range via the matched-field processor in
  :mod:`acoustic_sim.processor`.

**Range estimation**:

* ``rms`` — Amplitude-based 1/r range via RMS calibration.

**Tracking**:

* ``CausalWLSTracker`` — Causal weighted least-squares constant-velocity
  track fitter.
* ``EMABearingSmoother`` — Exponential moving average on the unit circle.

Quick start
-----------
::

    from acoustic_sim.detection import DetectionEngine

    engine = DetectionEngine(
        mic_positions=mics, fs=fs, window_samples=win,
        bearing_method="srp_phat", max_sources=2,
    )
    engine.calibrate_range(peak_rms, cpa_dist)

    for seg in windows:
        det = engine.process_window(seg, t)
        if det.detected:
            print(det.bearing_deg, det.range_m)
"""

from acoustic_sim.detection.bearing import (
    BearingDetection,
    BearingEstimator,
    BearingResult,
    MUSICEstimator,
    MVDRBeamformer,
    SRPBeamformer,
    available_bearing_methods,
    create_bearing_estimator,
)
from acoustic_sim.detection.engine import DetectionEngine, WindowDetection
from acoustic_sim.detection.ranging import (
    BearingRateRangeEstimator,
    TDOARangeEstimator,
    RMSRangeEstimator,
    RangeEstimate,
    RangeEstimator,
    available_range_methods,
    create_range_estimator,
)
from acoustic_sim.detection.tracking import (
    CausalWLSTracker,
    EMABearingSmoother,
    TrackState,
)

__all__ = [
    # Engine
    "DetectionEngine",
    "WindowDetection",
    # Bearing
    "BearingEstimator",
    "BearingDetection",
    "BearingResult",
    "SRPBeamformer",
    "MUSICEstimator",
    "MVDRBeamformer",
    "available_bearing_methods",
    "create_bearing_estimator",
    # Range
    "RangeEstimator",
    "RangeEstimate",
    "RMSRangeEstimator",
    "TDOARangeEstimator",
    "BearingRateRangeEstimator",
    "available_range_methods",
    "create_range_estimator",
    # Tracking
    "CausalWLSTracker",
    "EMABearingSmoother",
    "TrackState",
]

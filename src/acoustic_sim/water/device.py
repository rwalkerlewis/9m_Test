"""CPU / GPU device selection for the harbor processing chain.

The chain (STFT, whitening, matched-field scan, ship suppression, occupancy filter)
is written against an array module ``xp`` that is NumPy on the CPU and CuPy on the
GPU.  On the GPU everything runs in single precision (complex64): consumer and
workstation Turing-class cards execute FP64 at a small fraction of the FP32 rate, and
single precision is ample here (phases ``k r`` stay below ~1e3 rad, i.e. 1e-4 rad error).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

from acoustic_sim.backend import get_backend


@dataclass(frozen=True)
class Device:
    """Array module plus the dtype policy used with it."""

    xp: ModuleType
    is_cuda: bool

    @property
    def cdtype(self):
        return np.complex64 if self.is_cuda else np.complex128

    @property
    def fdtype(self):
        return np.float32 if self.is_cuda else np.float64

    def asnumpy(self, a: Any) -> np.ndarray:
        return a.get() if self.is_cuda and hasattr(a, "get") else np.asarray(a)

    def median_filter(self, a, size, mode: str = "nearest"):
        if self.is_cuda:
            from cupyx.scipy.ndimage import median_filter
        else:
            from scipy.ndimage import median_filter
        return median_filter(a, size=size, mode=mode)

    def gaussian_filter(self, a, sigma: float, mode: str = "constant"):
        if self.is_cuda:
            from cupyx.scipy.ndimage import gaussian_filter
        else:
            from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma, mode=mode)

    def synchronize(self) -> None:
        if self.is_cuda:
            self.xp.cuda.Stream.null.synchronize()


CPU = Device(np, False)


def get_device(use_cuda: bool = False) -> Device:
    """GPU device if requested and available (warns and falls back to the CPU otherwise)."""
    if not use_cuda:
        return CPU
    xp, is_cuda = get_backend(True)
    return Device(xp, True) if is_cuda else CPU

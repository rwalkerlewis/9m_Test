"""NumPy / CuPy backend abstraction.

All array-heavy code in the FDTD solver uses ``xp`` (returned by
:func:`get_backend`) so that the same stencil kernels run on CPU (NumPy)
or GPU (CuPy) without code changes.
"""

from __future__ import annotations

import warnings
from types import ModuleType

import numpy as np


def get_backend(use_cuda: bool = False) -> tuple[ModuleType, bool]:
    """Return ``(xp, is_cuda)`` where *xp* is numpy or cupy.

    Parameters
    ----------
    use_cuda : bool
        If *True*, attempt to import CuPy.  Falls back to NumPy with a
        warning if CuPy is unavailable or no GPU is detected.
    """
    if use_cuda:
        try:
            import cupy  # type: ignore[import-untyped]

            # Quick sanity check — will raise if no usable GPU.
            cupy.cuda.Device(0).compute_capability
            return cupy, True
        except Exception:
            warnings.warn(
                "CUDA requested but CuPy is unavailable or no GPU detected. "
                "Falling back to NumPy.",
                stacklevel=2,
            )
    return np, False

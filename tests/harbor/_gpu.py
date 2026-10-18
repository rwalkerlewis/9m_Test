"""Helpers shared by the GPU-aware harbor tests."""

import pytest


def _has_gpu() -> bool:
    try:
        import cupy

        cupy.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


HAS_GPU = _has_gpu()

# Parametrise a test over the CPU and (when available) the GPU.
DEVICES = [
    pytest.param(False, id="cpu"),
    pytest.param(True, id="gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="CuPy / GPU not available")),
]

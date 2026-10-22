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

# A test that needs CuPy and a GPU: marked ``gpu`` (``pytest -m gpu`` runs them, the GPU workflow does) and skipped
# without one.
requires_gpu = [pytest.mark.gpu, pytest.mark.skipif(not HAS_GPU, reason="CuPy / GPU not available")]

# Parametrise a test over the CPU and (when available) the GPU: the GPU case is marked ``gpu``.
DEVICES = [
    pytest.param(False, id="cpu"),
    pytest.param(True, id="gpu", marks=requires_gpu),
]

"""Compatibility alias for :mod:`acoustic_sim.air.detection_main`."""

import importlib as _importlib
import sys as _sys

_module = _importlib.import_module(".air.detection_main", __package__)
_sys.modules[__name__] = _module

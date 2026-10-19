"""Compatibility alias for :mod:`acoustic_sim.air.studies`."""

import importlib as _importlib
import sys as _sys

_module = _importlib.import_module(".air.studies", __package__)
_sys.modules[__name__] = _module

"""Compatibility package; water-domain code now lives in :mod:`acoustic_sim.water`."""

from __future__ import annotations

import importlib
import pkgutil
import sys

from acoustic_sim.water import *  # noqa: F403
from acoustic_sim.water import __all__

_water = importlib.import_module("acoustic_sim.water")
for _module in pkgutil.iter_modules(_water.__path__):
    sys.modules[f"{__name__}.{_module.name}"] = importlib.import_module(
        f"acoustic_sim.water.{_module.name}"
    )

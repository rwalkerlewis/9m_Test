"""Compatibility package; water-domain code now lives in :mod:`acoustic_sim.water`.

``import acoustic_sim.harbor.<name>`` returns the very same module object as
``acoustic_sim.water.<name>``.  Submodules are aliased on first import rather
than eagerly, so modules with optional dependencies (``classifier`` needs
PyTorch) are only loaded when they are actually requested.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

from acoustic_sim.water import *  # noqa: F403
from acoustic_sim.water import __all__

_WATER = "acoustic_sim.water"


class _WaterAliasLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # The import system re-reads sys.modules after exec_module, so the
        # canonical module replaces the placeholder (identity is preserved).
        canonical = module.__name__.replace(__name__, _WATER, 1)
        sys.modules[module.__name__] = importlib.import_module(canonical)


class _WaterAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        parent, _, child = fullname.rpartition(".")
        if parent != __name__ or importlib.util.find_spec(f"{_WATER}.{child}") is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _WaterAliasLoader())


if not any(isinstance(finder, _WaterAliasFinder) for finder in sys.meta_path):
    sys.meta_path.append(_WaterAliasFinder())

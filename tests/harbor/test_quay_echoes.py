"""The false echo tracks against the chart (``harbor_quay_echoes.py``): the classes, the decision rule, the clusters
and the scan grid's boundary on inputs with known answers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location("harbor_quay_echoes",
                                                  ROOT / "examples" / "harbor" / "05_detection" / "harbor_quay_echoes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_classes_take_the_first_that_applies():
    m = _script()
    assert m.classify(5.0, 3.0, 1.0) == "quay_image"
    assert m.classify(50.0, 3.0, 1.0) == "land_edge"
    assert m.classify(None, 30.0, 15.0) == "body"
    assert m.classify(50.0, 30.0, None) == "elsewhere"


def test_decision_rule():
    m = _script()
    assert m.decide({"quay_image": 5, "land_edge": 0, "body": 1, "elsewhere": 4}) == "quay_image"
    assert m.decide({"quay_image": 2, "land_edge": 4, "body": 0, "elsewhere": 4}) == "land_edge"
    assert m.decide({"quay_image": 3, "land_edge": 2, "body": 0, "elsewhere": 5}) == "neither"
    assert m.decide({"quay_image": 0, "land_edge": 0, "body": 3, "elsewhere": 25}) == "neither"


def test_clusters_group_within_15_m():
    m = _script()
    rows = [{"recording": f"bistatic_seed{s}", "noise_model": "old", "track": f"e{s}", "median_xy": [100.0 + s, 42.0]}
            for s in (1, 2, 3)] + [{"recording": "bistatic", "noise_model": "old", "track": "e9", "median_xy": [200.0, 42.0]}]
    cl = m.clusters(rows)
    assert len(cl) == 2
    big = max(cl.values(), key=lambda c: c["n"])
    assert big["n"] == 3 and big["recordings"] == 3


def test_grid_edge_of_a_square_grid():
    m = _script()
    x, y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    g = np.column_stack([x.ravel(), y.ravel()])
    edge = m.grid_edge(g)
    assert len(edge) == 16                          # the 5 x 5 grid's boundary
    assert not any(np.allclose(p, [2.0, 2.0]) for p in edge)

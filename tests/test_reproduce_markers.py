"""The ``reproduce`` markers of ``scripts/reproduce.py``: every number in the prose of ``docs/harbor.md`` is written
through a marker filled from ``output/reproduce/summary.json``; these tests resolve keys of known answers on a small
summary and fill a small page."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("reproduce_script", ROOT / "scripts" / "reproduce.py")
rp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rp
_spec.loader.exec_module(rp)

SUMMARY = {
    "recordings": {
        "simple": {"objects": {"wreck": {"pd": {"echo": 0.86}}, "ferry": {"pd": {"radiator": 1.0}}},
                   "false_tracks": {"per_minute": 6.0, "radiator": 2}},
        "simple_seed1": {"objects": {"wreck": {"pd": {"echo": 0.67}}}, "false_tracks": {"per_minute": 3.0, "radiator": 1}},
        "simple_seed2": {"objects": {"wreck": {"pd": {"echo": 0.98}}}, "false_tracks": {"per_minute": 3.0, "radiator": 0}},
    },
    "criteria": [{"id": "a", "passed": False, "values": {"x": 0.796}}, {"id": "b", "passed": True}],
    "source_levels": [{"recording": "simple", "object": "ferry", "error_db": 0.75},
                      {"recording": "simple_seed1", "object": "boat", "error_db": -0.66},
                      {"recording": "bistatic", "object": "ferry", "error_db": 2.4}],
    "propagation": {"h": [{"e": -0.42}, {"e": 0.21}]},
}


@pytest.mark.parametrize("key, text", [
    ("recordings/simple/objects/wreck/pd/echo|pct", "86 %"),
    ("range:recordings/simple_seed*/objects/wreck/pd/echo|pct", "67 to 98 %"),
    ("min:recordings/*/false_tracks/per_minute|f1", "3.0"),
    ("max:recordings/*/false_tracks/per_minute|f1", "6.0"),
    ("range:recordings/simple_seed*/false_tracks/per_minute|f1", "3.0"),
    ("sum:recordings/*/false_tracks/radiator", "3"),
    ("count:recordings/*/false_tracks/radiator", "3"),
    ("ntrue:criteria[*]/passed", "1"),
    ("criteria[0]/values/x|f3", "0.796"),
    ("criteria[0]/values/x|frac2", "0.80"),
    ("source_levels[recording=simple*][object=ferry,boat]/error_db|+f1", None),
    ("range:source_levels[recording=simple*]/error_db|+f1", "-0.7 to +0.8"),
    ("source_levels[recording=bistatic]/error_db|+f1", "+2.4"),
    ("absmax:propagation/h[*]/e|absf2", "0.42"),
    ("range:propagation/h[*]/e|+f2", "-0.42 to +0.21"),
])
def test_resolve(key, text):
    if text is None:                                           # two values without an aggregate: refused
        with pytest.raises(KeyError):
            rp.resolve(key, SUMMARY)
    else:
        assert rp.resolve(key, SUMMARY) == text


def test_resolve_missing_key_is_refused():
    with pytest.raises(KeyError):
        rp.resolve("recordings/nowhere/pd", SUMMARY)


def test_resolve_from_another_file(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "recordings" / "simple").mkdir(parents=True)
    (out / "recordings" / "simple" / "build.json").write_text(json.dumps({"fdtd_wall_s": 355.75}))
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    assert rp.resolve("@recordings/simple/build.json:fdtd_wall_s|f0", SUMMARY) == "356"


def test_fill_docs_writes_and_reports_stale(tmp_path, monkeypatch):
    doc = tmp_path / "page.md"
    doc.write_text("The wreck in <!-- reproduce:recordings/simple/objects/wreck/pd/echo|pct -->90 %<!-- /reproduce --> "
                   "of the frames.\n")
    monkeypatch.setattr(rp, "DOCS", [doc])
    monkeypatch.setattr(rp, "tables_by_name", lambda s: {})
    stale = rp.fill_docs(SUMMARY, write=False)
    assert len(stale) == 1 and "90 %" in stale[0] and "86 %" in stale[0]
    assert "90 %" in doc.read_text()                           # not written
    rp.fill_docs(SUMMARY, write=True)
    assert "-->86 %<!--" in doc.read_text()
    assert rp.fill_docs(SUMMARY, write=False) == []


def test_fill_docs_table_block(tmp_path, monkeypatch):
    doc = tmp_path / "page.md"
    doc.write_text("Before\n\n<!-- reproduce-table:t -->\n| old |\n<!-- /reproduce-table -->\n\nAfter\n")
    monkeypatch.setattr(rp, "DOCS", [doc])
    monkeypatch.setattr(rp, "tables_by_name", lambda s: {"t": "| a |\n|---|\n| 1 |\n"})
    assert rp.fill_docs(SUMMARY, write=True) == ["page.md: table t"]
    assert doc.read_text() == "Before\n\n<!-- reproduce-table:t -->\n| a |\n|---|\n| 1 |\n<!-- /reproduce-table -->\n\nAfter\n"


def test_harbor_md_is_filled_from_the_committed_summary():
    """``docs/harbor.md`` as committed agrees with ``output/reproduce/summary.json`` (what ``--check`` tests)."""
    summary = json.loads((ROOT / "output" / "reproduce" / "summary.json").read_text())
    assert rp.fill_docs(summary, write=False) == []

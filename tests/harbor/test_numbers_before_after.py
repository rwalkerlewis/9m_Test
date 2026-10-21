"""``harbor_numbers_before_after.py``: the before and after table on two small summaries with known differences."""

import importlib.util
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "examples" / "harbor" / "01_datasets" / "harbor_numbers_before_after.py"
_spec = importlib.util.spec_from_file_location("harbor_numbers_before_after", _P)
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)


def _summary(pd_ship, err_ship, pd_body, false_count, prop):
    return {"recordings": {"r": {"family": "f", "false_tracks": {"count": false_count, "per_minute": 3.0 * false_count},
                                 "objects": {"ship": {"radiates": True, "pd": {"radiator": pd_ship}, "median_error_m": {"radiator": err_ship}},
                                             "body": {"radiates": False, "pd": {"echo": pd_body}, "median_error_m": {"echo": None}},
                                             "quiet": {"radiates": False, "pd": {"echo": 0.0}, "median_error_m": {"echo": None}}}}},
            "propagation_error_ms": {"f": {"hydrophones": [{"error_ms": e} for e in prop]}}}


def test_moved_marks_and_worst_hydrophone():
    before = _summary(0.64, 2.2, 0.0, 3, [-1.0, -3.17, 2.0])
    after = _summary(0.65, 0.7, 0.83, 2, [0.05, -0.21])
    md = ba.table(before, after).splitlines()
    assert md[2] == "| r | -3.17 / -0.21 | ship (radiator) | 64 % / 65 % | 2.2 / 0.7 | error |"
    assert md[3] == "| r | -3.17 / -0.21 | body (echo) | 0 % / 83 % | - / - | Pd |"
    assert md[4] == "| r | -3.17 / -0.21 | false tracks (per minute) | 3 (9.0) / 2 (6.0) | | count |"
    assert len(md) == 5                                             # an object at 0 before and after is left out

"""The sensing contract: the processing side never imports the forward side.

See ``docs/architecture.md`` (Sensing contract).  The import walk is static (``ast``): every
``import`` and ``from ... import`` anywhere in a processing module, at module level or inside a
function, is resolved to a module of ``acoustic_sim`` (relative imports included, and a name
imported from a package ``__init__`` is followed to the module it comes from) and checked against
the forward side list.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "acoustic_sim"

PROCESSING = [
    "water/detection.py", "water/suppression.py", "water/occupancy.py", "water/bistatic.py",
    "water/motion.py", "water/localiser.py", "water/classifier.py", "water/evaluate.py",
    "water/propagation.py",
    "air/detection/*.py", "air/processor.py", "air/tracker.py", "air/fire_control.py",
    # ml inference paths (training and data generation are forward side or offline)
    "ml/features.py", "ml/acoustic_classifier.py", "ml/fusion_classifier.py", "ml/maneuver_classifier.py",
    "ml/anomaly_detector.py", "ml/anomaly_integration.py",
]

FORWARD = [
    "solver/*.py", "model.py", "sources.py",
    "water/fdtd_runner.py", "water/fdtd_cuda_bodies.py", "water/scatterer.py", "water/dispersion.py",
    "water/signatures.py", "water/vessels.py", "water/noise.py", "water/dataset.py",
    "water/scenario.py", "water/environment.py",
    "air/forward.py", "air/domains.py", "air/noise.py", "air/setup.py",
    "ml/data_generation.py", "ml/fno*.py",
]

# Observation header keys that would leak the forward model (section 1 of the contract).
FORBIDDEN_HEADER_KEYS = {
    "fdtd", "config", "scenario", "dx", "dt", "grid", "grid_spacing", "time_step", "fd_order",
    "tracks", "track", "level_pa", "levels", "components", "gain", "dead", "vessels",
}


def _expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(sorted(p for p in PKG.glob(pat) if p.is_file()))
    return out


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_path(name: str) -> Path | None:
    base = SRC.joinpath(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _resolve_from(module: str, level: int, target: str | None, is_package: bool) -> str:
    """Absolute module name of ``from <level dots><target> import ...`` inside *module*."""
    if level == 0:
        return target or ""
    parts = module.split(".")
    if not is_package:
        parts = parts[:-1]
    parts = parts[: len(parts) - (level - 1)] if level > 1 else parts
    return ".".join(parts + ([target] if target else []))


def _package_reexports(pkg: str) -> dict[str, str]:
    """Names a package ``__init__`` imports, mapped to the module they come from."""
    path = _module_path(pkg)
    if path is None or path.name != "__init__.py":
        return {}
    out: dict[str, str] = {}
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.ImportFrom):
            src = _resolve_from(pkg, node.level, node.module, True)
            for a in node.names:
                out[a.asname or a.name] = src
    return out


def imported_modules(path: Path) -> list[tuple[int, str]]:
    """``(line, module)`` of every ``acoustic_sim`` module imported anywhere in *path*."""
    module = _module_name(path)
    is_package = path.name == "__init__.py"
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(module, node.level, node.module, is_package)
            reexports = _package_reexports(base)
            for a in node.names:
                sub = f"{base}.{a.name}"
                if _module_path(sub) is not None:
                    found.append((node.lineno, sub))
                elif a.name in reexports:
                    found.append((node.lineno, reexports[a.name]))
                else:
                    found.append((node.lineno, base))
    return sorted({(ln, m) for ln, m in found if m == "acoustic_sim" or m.startswith("acoustic_sim.")})


def contract_violations() -> list[str]:
    forward = {_module_name(p) for p in _expand(FORWARD)}
    out = []
    for path in _expand(PROCESSING):
        for line, mod in imported_modules(path):
            if mod in forward:
                out.append(f"{path.relative_to(SRC.parent)}:{line}: imports forward side module {mod}")
    return out


def test_lists_name_existing_modules():
    for pat in PROCESSING + FORWARD:
        assert list(PKG.glob(pat)), f"no module matches {pat}"
    assert not {p for p in _expand(PROCESSING)} & {p for p in _expand(FORWARD)}


def test_processing_side_does_not_import_forward_side():
    violations = contract_violations()
    assert not violations, "sensing contract violated:\n" + "\n".join(violations)


def test_checker_catches_function_level_and_relative_imports(tmp_path, monkeypatch):
    """The walk sees imports inside functions and relative imports (guards the checker itself)."""
    import acoustic_sim  # noqa: F401  (the package must be importable for the paths to mean anything)

    src = "def f():\n    from .dispersion import DispersionModel\n    import acoustic_sim.solver.fdtd\n"
    probe = PKG / "water" / "_contract_probe.py"
    probe.write_text(src)
    try:
        mods = [m for _, m in imported_modules(probe)]
    finally:
        probe.unlink()
    assert "acoustic_sim.water.dispersion" in mods
    assert "acoustic_sim.solver.fdtd" in mods


# ----------------------------------------------------------------------
# Observation / Truth files
# ----------------------------------------------------------------------

def _keys(obj) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | set().union(*(_keys(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(_keys(v) for v in obj)) if obj else set()
    return set()


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    from tests.harbor.test_dataset import _tiny

    from acoustic_sim.water import assemble_dataset, build_track, render_vessel_gathers

    cfg, env, layout = _tiny()
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks)
    out = tmp_path_factory.mktemp("contract")
    ds = assemble_dataset(cfg, layout, tracks, gathers, keep_components=True)
    ds.save(out / "tiny.npz")
    return ds, out


def test_observation_header_has_no_forward_keys(tiny_dataset):
    from acoustic_sim.contract import Observation, Truth

    ds, out = tiny_dataset
    obs_path = out / "tiny.observation.npz"
    truth_path = out / "tiny.truth.npz"
    assert obs_path.exists() and truth_path.exists()
    with np.load(obs_path) as d:
        header = json.loads(str(d["header"]))
        files = set(d.files)
    leaked = _keys(header) & FORBIDDEN_HEADER_KEYS
    assert not leaked, f"observation header leaks {sorted(leaked)}"
    assert not any(f.startswith(("track", "comp:", "gain", "dead", "clean")) for f in files), files

    obs = Observation.load(obs_path)
    np.testing.assert_array_equal(obs.traces, ds.traces)
    assert obs.fs == ds.fs
    np.testing.assert_array_equal(obs.geometry.positions(), ds.layout.receivers)
    np.testing.assert_array_equal(obs.reports.xy, ds.ais.xy)
    truth = Truth.load(truth_path)
    assert set(truth.tracks) == set(ds.tracks)
    np.testing.assert_array_equal(truth.gain, ds.gain)
    assert truth.fdtd["dt"] == ds.cfg.fdtd.dt
    assert set(truth.components) == set(ds.components)


def test_split_cli_writes_both_files(tiny_dataset, tmp_path):
    import subprocess
    import sys

    ds, out = tiny_dataset
    old = tmp_path / "dataset.npz"
    ds.save(old)
    for p in (tmp_path / "dataset.observation.npz", tmp_path / "dataset.truth.npz"):
        p.unlink()
    subprocess.run([sys.executable, "-m", "acoustic_sim.contract", "split", str(old)], check=True,
                   capture_output=True, text=True)
    assert (tmp_path / "dataset.observation.npz").exists()
    assert (tmp_path / "dataset.truth.npz").exists()


def test_old_single_file_datasets_still_load(tiny_dataset, tmp_path):
    from acoustic_sim.water import HarborDataset

    ds, out = tiny_dataset
    back = HarborDataset.load(out / "tiny.npz")
    np.testing.assert_array_equal(back.traces, ds.traces)
    assert back.observation.fs == ds.fs
    assert set(back.truth.tracks) == set(ds.tracks)

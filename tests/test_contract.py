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
    "water/propagation.py", "water/ais.py", "water/sensors.py", "water/scoreboard.py",
    "air/detection/*.py", "air/processor.py", "air/tracker.py", "air/fire_control.py",
    # ml inference paths (training and data generation are forward side or offline)
    "ml/features.py", "ml/acoustic_classifier.py", "ml/fusion_classifier.py", "ml/maneuver_classifier.py",
    "ml/anomaly_detector.py", "ml/anomaly_integration.py",
]

FORWARD = [
    "solver/*.py", "model.py", "sources.py",
    "water/fdtd_runner.py", "water/fdtd_cuda_bodies.py", "water/scatterer.py", "water/dispersion.py",
    "water/signatures.py", "water/vessels.py", "water/noise.py", "water/dataset.py",
    "water/scenario.py", "water/environment.py", "water/ais_sim.py", "water/sensor_sampling.py",
    "air/forward.py", "air/domains.py", "air/noise.py", "air/setup.py",
    "ml/data_generation.py", "ml/fno*.py",
]

# Scoring modules read the Truth: processing side (no forward import) but no other processing module imports them.
SCORING = ["water/evaluate.py", "water/scoreboard.py"]

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


def test_processing_side_does_not_import_scoring():
    """Truth enters after inference only: no processing module outside ``SCORING`` imports a scoring module."""
    scoring = {_module_name(p) for p in _expand(SCORING)}
    assert scoring and {_module_name(p) for p in _expand(SCORING)} <= {_module_name(p) for p in _expand(PROCESSING)}
    out = []
    for path in _expand(PROCESSING):
        if _module_name(path) in scoring:
            continue
        out += [f"{path.relative_to(SRC.parent)}:{ln}: imports {m}" for ln, m in imported_modules(path) if m in scoring]
    assert not out, "\n".join(out)


def test_observation_kinds_resolve_to_processing_modules():
    """The geometry, environment and report classes an observation names are read from processing side modules."""
    from acoustic_sim import contract

    processing = {_module_name(p) for p in _expand(PROCESSING)} | {"acoustic_sim.contract"}
    for kind, (mod, name) in contract._KINDS.items():
        assert mod in processing, f"{kind}: {mod} is not a processing side module"
        assert contract._cls(kind).__name__ == name


def test_checker_catches_function_level_and_relative_imports(tmp_path, monkeypatch):
    """The walk sees imports inside functions and relative imports (guards the checker itself)."""
    probe = tmp_path / "acoustic_sim" / "water" / "probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("def f():\n    from .dispersion import DispersionModel\n    import acoustic_sim.solver.fdtd\n")
    import sys

    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path)
    mods = [m for _, m in imported_modules(probe)]
    assert "acoustic_sim.water.dispersion" in mods
    assert "acoustic_sim.solver.fdtd" in mods


def test_contract_module_is_domain_neutral():
    """``contract.py`` imports no forward side module (the ``split`` command line imports the old
    format reader lazily, in ``_main``, which no library code calls)."""
    forward = {_module_name(p) for p in _expand(FORWARD)}
    path = PKG / "contract.py"
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_main")
    in_main = {id(n) for n in ast.walk(main)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in in_main:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not set(names) & forward, f"contract.py:{node.lineno} imports {names}"


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
    """A 400 m x 300 m open water scene, one strain cable and a hydrophone, a ship and a silent body (CPU, seconds)."""
    from acoustic_sim.water import (Cable, HarborFDTDConfig, HarborScenarioConfig, SensorLayout, VesselConfig,
                                    assemble_dataset, build_track, render_vessel_gathers, uniform_water_environment)
    from acoustic_sim.water.scenario import CableConfig

    env = uniform_water_environment(400.0, 300.0, 5.0, 1500.0)
    cable = Cable("cable1", np.array([[100.0, 100.0], [300.0, 100.0]]), channel_spacing=20.0, das_mode="strain", gauge_length=10.0)
    layout = SensorLayout(cables=[cable], hydrophones=np.array([[200.0, 220.0]]))
    cfg = HarborScenarioConfig(
        fdtd=HarborFDTDConfig(fs=600.0, total_time=1.5, use_cuda=False),
        cables=[CableConfig("cable1", [[100.0, 100.0], [300.0, 100.0]], 20.0, das_mode="strain", gauge_length=10.0)],
        hydrophones=[[200.0, 220.0]],
        vessels=[VesselConfig("ship", "ship", [[120, 200], [280, 200]], speed=40.0, level_pa=1.0, seed=1),
                 VesselConfig("uuv", "uuv", [[200, 160]], level_pa=0.0, body_length=10.0, body_width=5.0, seed=2)],
        seed=5,
    )
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks)
    out = tmp_path_factory.mktemp("contract")
    ds = assemble_dataset(cfg, layout, tracks, gathers, keep_components=True, env=env, out=out / "tiny.npz")
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
    assert [c.das_mode for c in obs.geometry.cables] == ["strain"] and obs.geometry.cables[0].gauge_length == 10.0
    assert obs.environment.c_nominal == 1500.0 and obs.environment.is_water(np.array([[200.0, 150.0]]))[0]
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
    ds.save_legacy(old)
    assert not (tmp_path / "dataset.observation.npz").exists()
    subprocess.run([sys.executable, "-m", "acoustic_sim.contract", "split", str(old)], check=True,
                   capture_output=True, text=True)
    from acoustic_sim.contract import Observation

    obs = Observation.load(tmp_path / "dataset.observation.npz")
    np.testing.assert_array_equal(obs.traces, ds.traces)
    assert (tmp_path / "dataset.truth.npz").exists()


def test_old_single_file_datasets_still_load(tiny_dataset, tmp_path):
    from acoustic_sim.water import HarborDataset

    ds, out = tiny_dataset
    ds.save_legacy(tmp_path / "old.npz")
    old = HarborDataset.load(tmp_path / "old.npz")
    np.testing.assert_array_equal(old.traces, ds.traces)
    assert [c.das_mode for c in old.layout.cables] == ["pressure"]                 # as before: the old header drops it
    assert [c.das_mode for c in old.observation.geometry.cables] == ["strain"]      # restored from the scenario
    pair = HarborDataset.load(out / "tiny.npz")                                      # the pair written by assemble_dataset
    np.testing.assert_array_equal(pair.traces, ds.traces)
    np.testing.assert_array_equal(pair.clean, ds.clean)
    assert pair.cfg.to_dict() == ds.cfg.to_dict() and set(pair.truth.tracks) == set(ds.tracks)
    assert np.array_equal(pair.ais.t, ds.ais.t)


def test_public_chart_matches_the_forward_environment():
    """``HarborEnvironment.public()`` answers ``is_water`` and ``distance_to_land`` exactly as the forward
    environment does (so processing on the chart is bit identical)."""
    from acoustic_sim.water import build_synthetic_harbor

    rng = np.random.default_rng(0)
    for env in (build_synthetic_harbor(dx=5.0), build_synthetic_harbor(dx=1.0, crop=(700.0, 1300.0, 900.0, 1400.0))):
        chart = env.public()
        assert chart.dx == env.dx and chart.extent == env.extent
        x0, x1, y0, y1 = env.extent
        pts = np.column_stack([rng.uniform(x0 - 20, x1 + 20, 5000), rng.uniform(y0 - 20, y1 + 20, 5000)])
        for margin in (0.0, 5.0, 40.0):
            np.testing.assert_array_equal(chart.is_water(pts, margin), env.is_water(pts, margin))
        np.testing.assert_array_equal(chart.distance_to_land(pts), env.distance_to_land(pts))
        assert chart.c_nominal == float(np.median(env.model.values[env.water]))


# ----------------------------------------------------------------------
# Real data entry point: plain arrays, no scenario anywhere
# ----------------------------------------------------------------------

def _field_recording():
    """What a field recording is converted into (``docs/architecture.md``, Real data entry point):
    a 2 kHz strain DAS cable (10 m channels, 10 m gauge) and two hydrophones, an AIS ship moving east at
    4 m/s with an echo from a silent body, written here as plain arrays with a free field formula."""
    from acoustic_sim.water.ais import AISReports
    from acoustic_sim.water.sensors import Cable, SensorLayout

    fs, T, c = 2000.0, 3.0, 1495.0
    ship0, body = np.array([150.0, 250.0]), np.array([260.0, 180.0])
    layout = SensorLayout(cables=[Cable("cable1", np.array([[100.0, 100.0], [210.0, 100.0]]), 10.0, "strain", 10.0)],
                          hydrophones=np.array([[60.0, 200.0], [300.0, 80.0]]))
    n = int(T * fs)
    tt = np.arange(n) / fs
    rng = np.random.default_rng(11)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[(f < 80) | (f > 450)] = 0
    s = np.fft.irfft(spec, n)
    s /= s.std()

    def at(points, delay):
        return np.interp(tt - delay, tt, s, left=0.0, right=0.0)

    pts = layout.sample_points
    ship_t = ship0[None, :] + np.outer(tt, [4.0, 0.0])
    sampled = np.stack([at(p, np.hypot(*(ship_t - p).T) / c) / np.sqrt(np.hypot(*(ship0 - p))) for p in pts])
    r_e = np.hypot(*(body - ship0)) + np.hypot(*(pts - body).T)
    sampled += np.stack([0.3 * at(p, r / c) / np.sqrt(r) for p, r in zip(pts, r_e)])
    traces = layout.collapse(sampled.T).T + 2e-4 * rng.standard_normal((layout.n_receivers, n))
    x, y = np.arange(0.0, 401.0, 5.0), np.arange(0.0, 401.0, 5.0)
    k = 6
    t_ais = np.arange(k) * 0.5
    reports = AISReports(vessel=np.array(["ship"] * k), t=t_ais, xy=ship0[None, :] + np.outer(t_ais, [4.0, 0.0]),
                         sog=np.full(k, 4.0), cog=np.zeros(k))
    return dict(traces=traces, fs=fs, layout=layout, x=x, y=y, c=c, reports=reports, body=body)


def test_processing_runs_on_an_observation_built_from_plain_arrays(tmp_path):
    """The proof that the processing side works without the forward model: an Observation from plain
    arrays, saved and loaded, then ``run_bistatic`` and ``run_ais_aware`` to completion with the free
    field propagator, provenance with everything estimated, and the body found near its position."""
    from acoustic_sim.contract import ChartEnvironment, Observation, Provenance
    from acoustic_sim.water.bistatic import BistaticConfig, run_bistatic
    from acoustic_sim.water.detection import ProcessingConfig
    from acoustic_sim.water.propagation import FreeFieldPropagator
    from acoustic_sim.water.suppression import SuppressionConfig, run_ais_aware

    rec = _field_recording()
    chart = ChartEnvironment(water=np.ones((len(rec["y"]), len(rec["x"])), bool), depth=np.full((len(rec["y"]), len(rec["x"])), 15.0),
                             x=rec["x"], y=rec["y"], c_nominal=rec["c"])
    obs = Observation.from_arrays(rec["traces"], rec["fs"], rec["layout"], chart, rec["reports"],
                                  meta={"instrument": "DAS + hydrophones", "units": "strain-like, Pa"})
    obs.save(tmp_path / "field.observation.npz")
    obs = Observation.load(tmp_path / "field.observation.npz")
    prov = Provenance(data="observation")
    prop = FreeFieldPropagator.from_environment(obs.environment)

    f_lo, f_hi = 80.0, 450.0                     # the processing band: a choice of the processing, not read from the data
    bc = BistaticConfig(c=prop.c, propagator=prop, f_lo=f_lo, f_hi=f_hi, grid_dx=5.0, edge=20.0, exclusion_radius=30.0,
                        coherent_norm="snr")
    run = run_bistatic(obs.traces, obs.geometry, obs.fs, obs.environment, obs.reports, bc, provenance=prov)
    assert run.provenance.all_estimated and run.provenance.data == "observation"
    B, near = run.scan.B, np.hypot(*(run.grid - rec["body"]).T) <= 6.0
    contrast = 10 * np.log10(B[near].max(axis=0) / np.median(B, axis=0))
    peak_at_body = np.hypot(*(run.grid[B.argmax(axis=0)] - rec["body"]).T) <= 8.0
    assert np.median(contrast) > 10.0 and peak_at_body.mean() >= 0.4, (contrast, peak_at_body)

    pc = ProcessingConfig(win_s=0.5, hop_s=0.25, f_lo=80.0, f_hi=450.0, c=prop.c, propagator=prop, grid_dx=10.0, grid_margin=0.0,
                          grid_bounds=(20.0, 380.0, 20.0, 380.0), min_separation=40.0, refine_dx=2.0, refine_half=6.0)
    passive = run_ais_aware(obs.traces, obs.geometry, obs.fs, obs.environment, obs.reports, pc,
                            SuppressionConfig(refine_half=6.0, refine_step=2.0, min_significance=2.0), floor=0.0, provenance=prov)
    assert passive.provenance.all_estimated and len(passive.scan.times) > 0
    assert passive.ships.ids == ["ship"]


def test_a_greens_only_propagator_fits_the_protocol():
    """A non analytic propagator (only ``greens`` / ``path``, as a later Helmholtz replica library) drives
    the bistatic, passive and suppression chains; wrapping the free field gives the free field result."""
    from acoustic_sim.contract import ChartEnvironment
    from acoustic_sim.water.bistatic import BistaticConfig, run_bistatic
    from acoustic_sim.water.detection import ProcessingConfig, run_detection
    from acoustic_sim.water.propagation import FreeFieldPropagator
    from acoustic_sim.water.suppression import SuppressionConfig, run_ais_aware

    class GreensOnly:
        analytic, needs_angles, knowledge = False, False, ()

        def __init__(self, c):
            self.c, self._free = c, FreeFieldPropagator(c)

        def greens(self, sources_xy, receivers_xy, freqs, xp=np, dtype=np.float64):
            return self._free.greens(sources_xy, receivers_xy, freqs, xp, dtype)

        def path(self, points_xy, source_xy, freqs, xp=np, dtype=np.float64):
            return self._free.path(points_xy, source_xy, freqs, xp, dtype)

    rec = _field_recording()
    chart = ChartEnvironment(water=np.ones((len(rec["y"]), len(rec["x"])), bool), depth=np.zeros((len(rec["y"]), len(rec["x"]))),
                             x=rec["x"], y=rec["y"], c_nominal=rec["c"])
    args = (rec["traces"], rec["layout"], rec["fs"], chart)
    maps = {}
    for name, prop in (("analytic", None), ("greens", GreensOnly(rec["c"]))):
        bc = BistaticConfig(c=rec["c"], propagator=prop, f_lo=80.0, f_hi=450.0, grid_dx=10.0, edge=20.0, removal="none")
        pc = ProcessingConfig(win_s=0.5, hop_s=0.25, f_lo=80.0, f_hi=450.0, c=rec["c"], propagator=prop, grid_dx=20.0,
                              grid_margin=0.0, grid_bounds=(20.0, 380.0, 20.0, 380.0), min_range=1.0, refine_dx=4.0, refine_half=4.0)
        maps[name] = (run_bistatic(*args, rec["reports"], bc).scan.B, run_detection(*args, pc).scan.B,
                      run_ais_aware(*args, rec["reports"], pc, SuppressionConfig(refine_half=4.0, refine_step=2.0), floor=0.0).scan.B)
    for a, b in zip(maps["analytic"], maps["greens"]):
        np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-6 * np.abs(a).max())


def test_dataset_exists_sees_the_pair(tiny_dataset, tmp_path):
    from acoustic_sim.contract import dataset_exists

    ds, out = tiny_dataset
    assert dataset_exists(out / "tiny.npz") and not (out / "tiny.npz").exists()
    assert not dataset_exists(tmp_path / "missing.npz")

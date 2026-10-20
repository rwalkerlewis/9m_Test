"""The sensing contract between the forward side and the processing side.

The **forward side** (solvers, scenarios, sources, datasets) generates synthetic data and knows
everything.  The **processing side** (detection, localisation, tracking, classification) consumes an
:class:`Observation`, real or synthetic, and cannot tell the difference: recorded traces, the sample
rate, the sensor geometry, public environment knowledge (land, depth, a nominal sound speed) and, in
water, the AIS reports.  Everything else is in :class:`Truth`, which the processing side only reads
through the scoring calls (``water/evaluate.py``) or through the diagnostics modules
(``water/diagnostics.py``, ``air/diagnostics.py``), and every result a diagnostic touches says so in its
:class:`Provenance`.

This module is domain neutral and imports no forward side module (``tests/test_contract.py``).

``python -m acoustic_sim.contract split <dataset.npz>`` writes ``<name>.observation.npz`` and
``<name>.truth.npz`` from an old single file harbor dataset.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

# ----------------------------------------------------------------------
# Geometry, environment, propagation
# ----------------------------------------------------------------------


@runtime_checkable
class Geometry(Protocol):
    """Receiver geometry: channel positions in recording order."""

    @property
    def n_receivers(self) -> int: ...

    def positions(self) -> np.ndarray: ...


@dataclass
class MicArray:
    """Microphone positions ``(n_mics, 2 or 3)`` [m], in channel order (the air geometry)."""

    mic_positions: np.ndarray

    def __post_init__(self) -> None:
        self.mic_positions = np.asarray(self.mic_positions, dtype=np.float64)

    @property
    def n_receivers(self) -> int:
        return len(self.mic_positions)

    def positions(self) -> np.ndarray:
        return self.mic_positions

    def to_header(self) -> dict:
        return {"kind": "mic_array", "positions": self.mic_positions.tolist()}

    @classmethod
    def from_header(cls, h: dict, arrays: dict) -> "MicArray":
        return cls(np.asarray(h["positions"], dtype=np.float64))


@dataclass
class ChartEnvironment:
    """Public knowledge of a water area: a land / water chart, depth, a nominal sound speed.

    ``water`` and ``depth`` are ``(ny, nx)`` rasters on the coordinates ``x`` (nx,) and ``y`` (ny,) [m];
    ``polygons`` are the land outlines (for plotting).  Nothing here is a solver parameter: a real
    chart comes at its own resolution (for synthetic data it happens to be the forward model's raster).
    """

    water: np.ndarray
    depth: np.ndarray
    x: np.ndarray
    y: np.ndarray
    c_nominal: float
    polygons: list[np.ndarray] = field(default_factory=list)
    name: str = "chart"

    @property
    def dx(self) -> float:
        """Chart cell size [m]."""
        return float(self.x[1] - self.x[0]) if len(self.x) > 1 else 1.0

    @property
    def nx(self) -> int:
        return len(self.x)

    @property
    def ny(self) -> int:
        return len(self.y)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1])

    @property
    def land(self) -> np.ndarray:
        return ~self.water

    def _index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        ix = np.clip(np.rint((xy[:, 0] - self.x[0]) / self.dx).astype(int), 0, self.nx - 1)
        iy = np.clip(np.rint((xy[:, 1] - self.y[0]) / self.dx).astype(int), 0, self.ny - 1)
        return iy, ix

    def distance_to_land(self, xy: np.ndarray) -> np.ndarray:
        """Distance [m] from each point to the nearest land cell (nearest cell lookup)."""
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(self.water) * self.dx
        return dist[self._index(xy)]

    def is_water(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """True where *xy* is inside the chart, in water and at least *margin* m from land."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        x0, x1, y0, y1 = self.extent
        inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
        ok = inside & self.water[self._index(xy)]
        if margin > 0:
            ok &= self.distance_to_land(xy) >= margin
        return ok

    def to_header(self, arrays: dict) -> dict:
        arrays["env_water"], arrays["env_depth"] = self.water, self.depth
        arrays["env_x"], arrays["env_y"] = self.x, self.y
        return {"kind": "chart", "c_nominal": self.c_nominal, "name": self.name,
                "polygons": [np.asarray(p).tolist() for p in self.polygons]}

    @classmethod
    def from_header(cls, h: dict, arrays: dict) -> "ChartEnvironment":
        return cls(water=np.asarray(arrays["env_water"], bool), depth=arrays["env_depth"], x=arrays["env_x"], y=arrays["env_y"],
                   c_nominal=float(h["c_nominal"]), polygons=[np.asarray(p, float) for p in h["polygons"]], name=h["name"])


@dataclass
class AirEnvironment:
    """Public knowledge of an air scene: nominal sound speed, the wind a met report would give, the
    terrain type and the area of interest ``(x0, x1, y0, y1)``."""

    c_nominal: float = 343.0
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    terrain: str | None = None
    extent: tuple[float, float, float, float] | None = None

    def to_header(self, arrays: dict) -> dict:
        return {"kind": "air", "c_nominal": self.c_nominal, "wind": list(self.wind), "terrain": self.terrain,
                "extent": None if self.extent is None else list(self.extent)}

    @classmethod
    def from_header(cls, h: dict, arrays: dict) -> "AirEnvironment":
        return cls(float(h["c_nominal"]), tuple(h["wind"]), h["terrain"], None if h["extent"] is None else tuple(h["extent"]))


class Propagator(Protocol):
    """Replica model: Green's functions between points, injected into the processing chains.

    ``greens(sources_xy, receivers_xy, freqs)`` is the field at *receivers* (N, 2) of unit sources at
    *sources* (P, 2), shape ``(F, P, N)``; ``path(points_xy, source_xy, freqs)`` is the propagation
    factor ``(F, P)`` from one source to many points (the illuminator to cell leg of the bistatic model).
    A propagator built from the public environment alone (free field, or a Helmholtz replica library
    from ``solver/helmholtz.py``) implements just these two.

    An **analytic** propagator (``analytic = True``) also exposes ``wavenumbers(freqs, theta, xp, dtype)``:
    the wavenumber ``(F, P, N)`` of every point pair, with ``theta = atan2(|dy|, |dx|)`` needed only when
    ``needs_angles``.  The chains use that hook in their cached geometry loops; a non analytic propagator
    goes through ``greens``.  ``c`` is its nominal sound speed, ``knowledge`` the simulator knowledge it
    uses (``()`` for a blind propagator), which the chains copy into the result's :class:`Provenance`.
    """

    c: float
    analytic: bool
    needs_angles: bool
    knowledge: tuple[str, ...]

    def greens(self, sources_xy, receivers_xy, freqs, xp=np): ...

    def path(self, points_xy, source_xy, freqs, xp=np): ...


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

ORACLE_INPUTS = ("waveform", "track", "direct_removal", "components", "range_calibration", "range_method",
                 "altitude", "hit_feedback", "labels")
SIMULATOR_KNOWLEDGE = ("dispersion", "empirical_replicas", "band")


@dataclass
class Provenance:
    """What a result knew beyond the observation.

    ``oracle_inputs``: truth the processing read (``"waveform"``: the true illuminator waveform,
    ``"track"``: the exact illuminator track, ``"direct_removal"``: the direct path removed with the
    truth component, ``"components"``: truth components passed through a stage for diagnostics, ...).
    ``simulator_knowledge``: forward model parameters the processing used (``"dispersion"``: the FDTD
    scheme's numerical wavenumber, ``"empirical_replicas"``: FDTD rendered replicas).  ``data``:
    ``"observation"`` (recorded) or ``"forward_model"`` (synthetic, processed as if recorded).
    ``held_out``: the data were not used to choose any parameter of the processing.
    """

    oracle_inputs: list[str] = field(default_factory=list)
    simulator_knowledge: list[str] = field(default_factory=list)
    data: str = "forward_model"
    held_out: bool = False

    def __post_init__(self) -> None:
        if self.data not in ("observation", "forward_model"):
            raise ValueError(f"data must be 'observation' or 'forward_model', got {self.data!r}")
        self.oracle_inputs = _unique(self.oracle_inputs)
        self.simulator_knowledge = _unique(self.simulator_knowledge)

    @property
    def blind(self) -> bool:
        return not self.oracle_inputs and not self.simulator_knowledge

    def with_flags(self, oracle_inputs=(), simulator_knowledge=()) -> "Provenance":
        """A copy with the given flags added."""
        return Provenance(list(self.oracle_inputs) + list(oracle_inputs),
                          list(self.simulator_knowledge) + list(simulator_knowledge), self.data, self.held_out)

    def to_dict(self) -> dict:
        return {"oracle_inputs": list(self.oracle_inputs), "simulator_knowledge": list(self.simulator_knowledge),
                "data": self.data, "held_out": bool(self.held_out)}

    @classmethod
    def from_dict(cls, d: dict) -> "Provenance":
        return cls(list(d.get("oracle_inputs", [])), list(d.get("simulator_knowledge", [])),
                   d.get("data", "forward_model"), bool(d.get("held_out", False)))

    def label(self) -> str:
        """One line for figure titles and table captions."""
        parts = ["blind" if self.blind else "NOT BLIND"]
        if self.oracle_inputs:
            parts.append("oracle: " + ", ".join(self.oracle_inputs))
        if self.simulator_knowledge:
            parts.append("simulator: " + ", ".join(self.simulator_knowledge))
        parts.append("synthetic data" if self.data == "forward_model" else "recorded data")
        return " | ".join(parts)


def _unique(items) -> list[str]:
    out: list[str] = []
    for s in items:
        if s not in out:
            out.append(str(s))
    return out


def annotate_figure(fig, prov: Provenance, y: float = 0.005, fontsize: float = 8.0) -> None:
    """Print the provenance flags at the bottom of a matplotlib figure (red when not blind)."""
    fig.text(0.5, y, prov.label(), ha="center", va="bottom", fontsize=fontsize, color="0.25" if prov.blind else "firebrick")


# ----------------------------------------------------------------------
# Observation and Truth
# ----------------------------------------------------------------------

# Geometry, environment and report classes are resolved by kind, so this module needs no domain import.
_KINDS = {
    "mic_array": (__name__, "MicArray"),
    "sensor_layout": ("acoustic_sim.water.sensors", "SensorLayout"),
    "chart": (__name__, "ChartEnvironment"),
    "air": (__name__, "AirEnvironment"),
    "ais": ("acoustic_sim.water.ais", "AISReports"),
}


def _cls(kind: str):
    mod, name = _KINDS[kind]
    return getattr(sys.modules[__name__] if mod == __name__ else import_module(mod), name)


def _header(d) -> dict:
    return json.loads(str(d["header"]))


@dataclass
class Observation:
    """What a deployed system records and knows.

    ``traces`` (n_receivers, n_samples) in channel order of ``geometry``; ``fs`` [Hz]; ``geometry``
    (``positions()``, ``n_receivers``); ``environment`` (public knowledge only); ``reports`` (AIS in
    water, ``None`` in air); ``t0`` [s] of the first sample; ``meta``: free text about the recording
    (instrument, units, band), never scenario parameters.
    """

    traces: np.ndarray
    fs: float
    geometry: Any
    environment: Any = None
    reports: Any = None
    t0: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.traces.ndim != 2 or self.traces.shape[0] != self.geometry.n_receivers:
            raise ValueError(f"traces must be (n_receivers={self.geometry.n_receivers}, n_samples), got {self.traces.shape}")

    @property
    def t(self) -> np.ndarray:
        return self.t0 + np.arange(self.traces.shape[1]) / self.fs

    @classmethod
    def from_arrays(cls, traces, fs, geometry, environment=None, reports=None, t0: float = 0.0, meta: dict | None = None) -> "Observation":
        """An observation from plain arrays (the real data entry point; see ``docs/architecture.md``).

        *geometry* is a geometry object or a ``(n, 2 or 3)`` array of microphone positions.
        """
        if not hasattr(geometry, "positions"):
            geometry = MicArray(np.asarray(geometry, dtype=np.float64))
        return cls(np.asarray(traces), float(fs), geometry, environment, reports, float(t0), dict(meta or {}))

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {"traces": self.traces, "fs": np.float64(self.fs)}
        header: dict = {"t0": self.t0, "meta": self.meta, "geometry": self.geometry.to_header(),
                        "environment": None if self.environment is None else self.environment.to_header(arrays),
                        "reports": None}
        if self.reports is not None:
            r = self.reports
            arrays.update(rep_vessel=r.vessel, rep_t=r.t, rep_xy=r.xy, rep_sog=r.sog, rep_cog=r.cog)
            header["reports"] = {"kind": "ais"}
        arrays["header"] = np.array(json.dumps(header))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Observation":
        with np.load(path) as d:
            h = _header(d)
            arrays = {k: d[k] for k in d.files}
        geom = _cls(h["geometry"]["kind"]).from_header(h["geometry"], arrays)
        env = None if h["environment"] is None else _cls(h["environment"]["kind"]).from_header(h["environment"], arrays)
        reports = None
        if h["reports"] is not None:
            reports = _cls(h["reports"]["kind"])(vessel=arrays["rep_vessel"], t=arrays["rep_t"], xy=arrays["rep_xy"],
                                                 sog=arrays["rep_sog"], cog=arrays["rep_cog"])
        return cls(arrays["traces"], float(arrays["fs"]), geom, env, reports, float(h["t0"]), h["meta"])


@dataclass
class Track:
    """Ground-truth trajectory sampled on a regular time grid (a contract type: AIS prediction tests
    and scoring both use it)."""

    id: str
    kind: str
    ais: bool
    t: np.ndarray                  # (M,) seconds
    xy: np.ndarray                 # (M, 2) metres
    speed: np.ndarray              # (M,) m/s
    active: np.ndarray             # (M,) bool, vessel moving and radiating
    level_pa: float = 1.0
    meta: dict = field(default_factory=dict)

    def position_at(self, t: np.ndarray | float) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.column_stack([np.interp(t, self.t, self.xy[:, 0]), np.interp(t, self.t, self.xy[:, 1])])

    def speed_at(self, t: np.ndarray | float) -> np.ndarray:
        return np.interp(np.atleast_1d(np.asarray(t, dtype=float)), self.t, self.speed)

    def active_at(self, t: np.ndarray | float) -> np.ndarray:
        return np.interp(np.atleast_1d(np.asarray(t, dtype=float)), self.t, self.active.astype(float)) > 0.5


@dataclass
class Truth:
    """Everything the forward model knew: ``tracks``, ``components`` (direct and scattered per vessel,
    noise), ``gain``, ``dead``, ``scenario`` (the full scenario config including the solver config),
    ``fdtd`` (the solver config used, with the derived ``dt``), ``clean`` (noise-free sum) and ``meta``."""

    tracks: dict[str, Track] = field(default_factory=dict)
    components: dict[str, np.ndarray] | None = None
    gain: np.ndarray | None = None
    dead: np.ndarray | None = None
    scenario: dict = field(default_factory=dict)
    fdtd: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    clean: np.ndarray | None = None

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name in ("gain", "dead", "clean"):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        for key, arr in (self.components or {}).items():
            arrays["comp:" + key] = arr
        track_meta = {}
        for k, tr in self.tracks.items():
            arrays[f"track_{k}_t"], arrays[f"track_{k}_xy"] = tr.t, tr.xy
            arrays[f"track_{k}_speed"], arrays[f"track_{k}_active"] = tr.speed, tr.active
            track_meta[k] = {"kind": tr.kind, "ais": tr.ais, "level_pa": tr.level_pa, "meta": tr.meta}
        arrays["header"] = np.array(json.dumps({"scenario": self.scenario, "fdtd": self.fdtd, "tracks": track_meta,
                                                "meta": self.meta}))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Truth":
        with np.load(path) as d:
            h = _header(d)
            tracks = {k: Track(id=k, kind=m["kind"], ais=m["ais"], t=d[f"track_{k}_t"], xy=d[f"track_{k}_xy"],
                               speed=d[f"track_{k}_speed"], active=d[f"track_{k}_active"], level_pa=m["level_pa"], meta=m["meta"])
                      for k, m in h["tracks"].items()}
            return cls(tracks=tracks, components={k[5:]: d[k] for k in d.files if k.startswith("comp:")} or None,
                       gain=d["gain"] if "gain" in d else None, dead=d["dead"] if "dead" in d else None,
                       scenario=h["scenario"], fdtd=h["fdtd"], meta=h["meta"], clean=d["clean"] if "clean" in d else None)


def contract_paths(path: str | Path) -> tuple[Path, Path]:
    """``(<name>.observation.npz, <name>.truth.npz)`` for a dataset path ``<name>.npz`` (or either file)."""
    p = Path(path)
    name = p.name
    for suffix in (".observation.npz", ".truth.npz", ".npz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return p.with_name(name + ".observation.npz"), p.with_name(name + ".truth.npz")


def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m acoustic_sim.contract", description="Sensing contract tools.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split", help="write <name>.observation.npz and <name>.truth.npz from an old <name>.npz")
    sp.add_argument("dataset", nargs="+", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "split":
        # The forward side owns the old format; it is imported here, at the command line, never by the library.
        from acoustic_sim.water.dataset import split_dataset

        for p in args.dataset:
            obs, tru = split_dataset(p)
            print(f"{p} -> {obs.name}, {tru.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

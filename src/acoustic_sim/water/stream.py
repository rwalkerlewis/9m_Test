"""The harbor picture: the real-time water chain on an observation, frame by frame, with what each object is.

One implementation of every step (``streaming.py`` and what it calls), run on traces as they arrive:

1. ``StreamingPassive`` (no AIS) locates the radiating objects, ships included;
2. ``RadiatorTracker`` turns its peaks into tracks;
3. ``StreamingEcho`` removes the direct path of the strong tracks and maps the echoes with each as an illuminator;
4. a second ``RadiatorTracker`` (its own ``TrackerConfig``) turns the echo peaks into echo tracks, the peaks within
   the exclusion radius of a confirmed radiator track left out, so a residual direct path is not confirmed as a body;
5. ``characteristics.py`` measures every track (position and its uncertainty, velocity, how it is seen, AIS identity,
   status; level and tonal lines of a radiator; echo strength, size indicator and extent of a body);
6. one record per frame goes to ``picture.jsonl`` as the stream runs (:class:`HarborPicture`, schema in
   ``docs/harbor.md``).

AIS enters only as a label (step 5, reports received by the frame).  Every parameter comes from one processing
config file (``examples/configs/processing/<recording>.json``, :func:`load_processing`), with ``--set key=value``
overrides that are recorded in the ``run`` block of every output.

``python -m acoustic_sim.water.stream --observation <file> --processing <config> --out <dir>`` runs it on an
observation file alone (a field recording converted with ``Observation.from_arrays``); a batch run is the stream
with one chunk (``--chunk`` longer than the record).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import get_window

from acoustic_sim.contract import Observation, Provenance, file_sha256, run_info
from acoustic_sim.water.ais import AISReports
from acoustic_sim.water.characteristics import (
    CharacteristicsConfig,
    ais_match,
    echo_strength,
    kinematics,
    line_persistence,
    peak_shape,
    received_level,
    size_class,
    source_level_100m,
    source_spectrum,
    tonal_lines,
)
from acoustic_sim.water.detection import ProcessingConfig
from acoustic_sim.water.propagation import FreeFieldPropagator
from acoustic_sim.water.streaming import (
    EchoStreamConfig,
    FrameResult,
    RadiatorTracker,
    StreamConfig,
    StreamingEcho,
    StreamingPassive,
    TrackerConfig,
)
from acoustic_sim.water.suppression import SuppressionConfig

SCHEMA = "harbor_picture/1"
NO_REPORTS = AISReports(np.array([], dtype=str), np.empty(0), np.empty((0, 2)), np.empty(0), np.empty(0))


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_override(raw: dict, item: str) -> None:
    """``key=value`` on the nested config *raw* (``tracker.gate=8``, ``echo.bistatic.eca_images=true``); the value
    is read as JSON when it parses, else as a string.  The key must exist."""
    key, sep, value = item.partition("=")
    if not sep:
        raise ValueError(f"override must be key=value, got {item!r}")
    parts = key.strip().split(".")
    d = raw
    for p in parts[:-1]:
        if not isinstance(d, dict) or p not in d or d[p] is None:
            raise KeyError(f"override {item!r}: no section {p!r}")
        d = d[p]
    if parts[-1] not in d:
        raise KeyError(f"override {item!r}: no key {parts[-1]!r} in {'.'.join(parts[:-1]) or 'the config'}")
    d[parts[-1]] = _parse_value(value.strip())


@dataclass
class ChainConfig:
    """Every processing parameter of one recording (one ``examples/configs/processing/<recording>.json``)."""

    raw: dict
    path: str | None = None
    sha256: str | None = None
    overrides: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.raw.get("recording", "")

    @property
    def band(self) -> tuple[float, float]:
        return tuple(self.raw["band_hz"])

    @property
    def chunk_s(self) -> float:
        return float(self.raw.get("chunk_s", 0.25))

    @property
    def has_echo(self) -> bool:
        return self.raw.get("echo") is not None

    def processing(self, env, use_cuda: bool) -> ProcessingConfig:
        x0, x1, y0, y1 = env.extent
        e = float(self.raw["edge_m"])
        prop = FreeFieldPropagator.from_environment(env)
        return ProcessingConfig(f_lo=self.band[0], f_hi=self.band[1], c=prop.c, propagator=prop, use_cuda=use_cuda,
                                grid_bounds=(x0 + e, x1 - e, y0 + e, y1 - e), **self.raw["processing"])

    def stream(self) -> StreamConfig:
        return StreamConfig(**self.raw["stream"])

    def tracker(self) -> TrackerConfig:
        return TrackerConfig(**self.raw["tracker"])

    def echo_stream(self) -> EchoStreamConfig:
        return EchoStreamConfig(**self.raw["echo"]["stream"])

    def bistatic(self, env, use_cuda: bool):
        from acoustic_sim.water.bistatic import BistaticConfig

        prop = FreeFieldPropagator.from_environment(env)
        return BistaticConfig(f_lo=self.band[0], f_hi=self.band[1], c=prop.c, propagator=prop, use_cuda=use_cuda,
                              **self.raw["echo"]["bistatic"])

    def echo_tracker(self) -> TrackerConfig:
        return TrackerConfig(**self.raw["echo"].get("tracker", {}))

    def characteristics(self) -> CharacteristicsConfig:
        return CharacteristicsConfig(**self.raw.get("characteristics", {}))

    @property
    def scoring(self) -> dict:
        return dict(self.raw.get("scoring", {}))


def load_processing(path: str | Path, overrides=()) -> ChainConfig:
    """The processing config of a recording, with ``key=value`` *overrides* applied (and recorded)."""
    path = Path(path)
    raw = json.loads(path.read_text())
    raw = copy.deepcopy(raw)
    for item in overrides:
        apply_override(raw, item)
    return ChainConfig(raw, str(path), file_sha256(path), list(overrides))


# ----------------------------------------------------------------------
# The picture
# ----------------------------------------------------------------------

def _r(x, n: int = 2):
    """Round floats (and nested lists of them) for the JSON records."""
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), n)
    return x


def _status(tr) -> str:
    if tr.lost_t is not None:
        return "lost"
    if tr.static:
        return "static"
    return "confirmed" if tr.confirmed else "tentative"


class HarborPicture:
    """The chain of the module docstring on one recording, pushed chunk by chunk (:meth:`push`); every frame gives one
    picture record (:attr:`records`; ``on_record`` is called with each as it is made)."""

    def __init__(self, layout, fs: float, env, cfg: ChainConfig, use_cuda: bool = True, reports: AISReports | None = None,
                 keep_frames: bool = False, keep_clean: bool = False, on_record=None) -> None:
        self.layout, self.fs, self.env, self.cfg = layout, float(fs), env, cfg
        self.reports = reports if reports is not None else NO_REPORTS
        self.pc = cfg.processing(env, use_cuda)
        self.passive = StreamingPassive(layout, fs, env, self.pc, SuppressionConfig(), cfg.stream(), keep_maps=True,
                                        keep_spectra=True)
        self.tracker = RadiatorTracker(cfg.tracker(), prefix="r")
        self.echo = self.echo_tracker = None
        if cfg.has_echo:
            self.bc = cfg.bistatic(env, use_cuda)
            self.echo = StreamingEcho(layout, fs, env, self.bc, cfg.echo_stream(), keep_maps=True, keep_clean=keep_clean,
                                      keep_spectra=True)
            self.echo_tracker = RadiatorTracker(cfg.echo_tracker(), prefix="e")
        self.cc = cfg.characteristics()
        self.c = self.pc.c
        self.window = get_window("hann", self.passive.nper)
        self.keep_frames = keep_frames
        self.on_record = on_record
        self.records: list[dict] = []
        self.p_frames: list[FrameResult] = []
        self.e_frames: list[FrameResult] = []
        self.e_used: list[FrameResult] = []                 # echo peaks the echo tracker was given
        self.meas: dict[str, dict] = {}                     # per track: measurements over the frames it was measured in
        self.snaps: list[tuple[float, dict]] = []           # radiator picture per passive frame, waiting for the echo frame
        self.merge_since: dict[tuple[str, str], float] = {}
        self.sent: dict[str, int] = {}
        self.ever: dict[str, list] = {}                     # every track that was confirmed: its fixes (for scoring)
        self.seconds = {"passive": 0.0, "echo": 0.0, "characteristics": 0.0}
        self.n_in = 0
        self._snap_t, self._snap = -np.inf, {}

    # ------------------------------------------------------------------

    def push(self, samples: np.ndarray) -> list[dict]:
        n0 = len(self.records)
        self.n_in += samples.shape[1]
        t0 = time.perf_counter()
        new = self.passive.push(samples)
        self.seconds["passive"] += time.perf_counter() - t0
        for f in new:
            t1 = time.perf_counter()
            self.tracker.update(f)
            self.seconds["passive"] += time.perf_counter() - t1
            t1 = time.perf_counter()
            self._measure_radiators(f)                         # the tracks as they stand after this frame
            self.snaps.append((f.t, self._radiator_snapshot(f)))
            self.seconds["characteristics"] += time.perf_counter() - t1
            if not self.keep_frames:
                f.spectrum = f.weights = None
            self.p_frames.append(f if self.keep_frames else _light(f))
            if self.echo is None:
                self._emit(f.t, f.latency_s + time.perf_counter() - t1, self.snaps.pop()[1], None, passive=True, echo_out=False)
        if self.echo is not None:
            t1 = time.perf_counter()
            strong = self.tracker.strong()
            fixes = {}
            for k, tr in strong.items():
                fixes[k] = [(f[0], f[1]) for f in tr.fixes[self.sent.get(k, 0):]]
                self.sent[k] = len(tr.fixes)
            got = self.echo.push(samples, fixes)
            self.seconds["echo"] += time.perf_counter() - t1
            for e in got:
                self._echo_frame(e)
        return self.records[n0:]

    # ------------------------------------------------------------------ radiators

    def _measure_radiators(self, f: FrameResult) -> None:
        cc = self.cc
        freqs = self.passive.freqs
        live = np.flatnonzero(f.weights > 0)
        rec = self.layout.receivers[live]
        for k, tr in self.tracker.tracks.items():
            if not tr.confirmed or tr.fixes[-1][0] != f.t:
                continue
            xy = tr.fixes[-1][1]
            m = self.meas.setdefault(k, {"frames": [], "lines": []})
            shape = peak_shape(f.B, self.passive.grid, xy, cc.peak_radius_m)
            s = source_spectrum(f.spectrum, f.weights, self.layout, xy, freqs, self.c)
            rng = float(np.median(np.hypot(*(rec - xy).T)))
            rl = received_level(s, freqs, self.c, rng, self.passive.nper, self.window)
            lines = tonal_lines(np.abs(s) ** 2, freqs, cc.n_lines, cc.line_min_db, cc.line_floor_bins)
            m["lines"].append([ln["bin"] for ln in lines])
            m["frames"].append({"t": f.t, "shape": shape, "level": rl, "range": rng,
                                "source_level": source_level_100m(rl, rng, cc.level_calibration), "lines": lines})

    def _radiator_snapshot(self, f: FrameResult) -> dict:
        """Every radiator track as it stands after frame *f*, measured (picture fields without the merge and AIS)."""
        out = {}
        tracks = dict(self.tracker.tracks)
        tracks.update(self.tracker.lost)
        for k, tr in tracks.items():
            st = _status(tr)
            if st != "tentative":
                self.ever[k] = tr.fixes
            out[k] = self._track_fields(k, tr, st, f.t, self.tracker, "radiator")
        return out

    def _track_fields(self, k: str, tr, status: str, t: float, tracker: RadiatorTracker, kind: str) -> dict:
        last = tr.fixes[-1]
        obj = {"track": k, "kind": kind, "status": status, "fix_t": last[0], "fix_xy": np.asarray(last[1]).tolist(),
               "detected": bool(last[0] == t), "score": float(last[2])}
        if status == "tentative":
            obj["xy"] = obj["fix_xy"]
            return obj
        kin = kinematics(tr.fixes, min(t, last[0]) if status == "lost" else t, self.cc.fit_s)
        obj["kin"] = kin
        obj["xy"] = kin["xy"]
        ct = tr.confirmed_t
        obj["frames_since_confirmation"] = int(sum(1 for x in tracker.frames if x >= ct)) if ct is not None else 0
        obj["detected_frames"] = int(sum(1 for fx in tr.fixes if fx[0] >= ct)) if ct is not None else 0
        obj["lost_t"] = tr.lost_t
        obj["fixes"] = [(fx[0], np.asarray(fx[1]).tolist()) for fx in tr.fixes[-8:]]
        m = self.meas.get(k)
        if m and m["frames"]:
            win = m["frames"][-self.cc.report_frames:]
            obj["shape"] = win[-1]["shape"]
            obj["shape_t"] = win[-1]["t"]
            if kind == "radiator":
                obj["radiator"] = self._radiator_fields(m, win)
            else:
                obj["echo"] = self._echo_fields(m, win)
        return obj

    def _radiator_fields(self, m: dict, win: list[dict]) -> dict:
        cc = self.cc
        latest = win[-1]["lines"]
        return {"received_level": float(np.median([w["level"] for w in win])),
                "range_m": float(np.median([w["range"] for w in win])),
                "source_level_100m_pa": float(np.median([w["source_level"] for w in win])),
                "calibration": cc.level_calibration,
                "tonal_lines": [{"f_hz": ln["f_hz"], "level_db": ln["level_db"],
                                 "frames": line_persistence(m["lines"], ln["bin"], cc.line_tolerance_bins)} for ln in latest],
                "lines_frames_measured": len(m["lines"]),
                "window_s": [win[0]["t"], win[-1]["t"]], "n_frames": len(win)}

    # ------------------------------------------------------------------ echoes

    def _echo_frame(self, e: FrameResult) -> None:
        t1 = time.perf_counter()
        while self.snaps and self.snaps[0][0] <= e.t + 1e-6:      # the radiator picture of the same frame (passive is ahead)
            self._snap_t, self._snap = self.snaps.pop(0)
        snap = self._snap
        # Echo peaks near a confirmed radiator track are its residual direct path, not a body.
        rad = [np.asarray(o["xy"]) for o in snap.values() if o["status"] in ("confirmed", "static")]
        keep = [i for i, p in enumerate(e.xy) if all(np.hypot(*(p - q)) >= self.bc.exclusion_radius for q in rad)] if e.ships else []
        fe = FrameResult(e.t, e.xy[keep].reshape(-1, 2), e.score[keep], e.ships, e.compute_s, e.latency_s)
        self.echo_tracker.update(fe)
        self.e_used.append(fe)
        self._measure_echoes(e)
        echo_snap = {}
        tracks = dict(self.echo_tracker.tracks)
        tracks.update(self.echo_tracker.lost)
        for k, tr in tracks.items():
            st = _status(tr)
            if st != "tentative":
                self.ever[k] = tr.fixes
            echo_snap[k] = self._track_fields(k, tr, st, e.t, self.echo_tracker, "echo")
        if not self.keep_frames:
            e.spectrum = e.weights = e.refs = None
        self.e_frames.append(e if self.keep_frames else _light(e))
        self.seconds["characteristics"] += time.perf_counter() - t1
        self._emit(e.t, e.latency_s + time.perf_counter() - t1, snap, echo_snap, passive=abs(self._snap_t - e.t) < 1e-6,
                   echo_out=bool(e.ships))

    def _measure_echoes(self, e: FrameResult) -> None:
        if not e.ships or e.spectrum is None:
            return
        cc = self.cc
        for k, tr in self.echo_tracker.tracks.items():
            if not tr.confirmed or tr.fixes[-1][0] != e.t:
                continue
            xy = np.asarray(tr.fixes[-1][1])
            strength = {n: echo_strength(e.spectrum, e.weights, self.layout, xy, e.ships[n], e.refs[n], self.echo.freqs, self.c)
                        for n in e.ships if n in e.refs}
            if not strength:
                continue
            top = max(strength, key=lambda n: strength[n]["strength_db"])
            shape = peak_shape(e.B, self.echo.grid, xy, cc.peak_radius_m, direction=xy - np.asarray(e.ships[top]))
            m = self.meas.setdefault(k, {"frames": [], "lines": []})
            m["frames"].append({"t": e.t, "shape": shape, "strength": strength, "top": top})

    def _echo_fields(self, m: dict, win: list[dict]) -> dict:
        cc = self.cc
        ills = sorted({n for w in win for n in w["strength"]})
        strength = {n: float(np.median([w["strength"][n]["strength_db"] for w in win if n in w["strength"]])) for n in ills}
        ratio = float(np.median([w["strength"][w["top"]]["size_ratio_db"] for w in win]))
        return {"strength_db": strength, "size_ratio_db": ratio, "size": size_class(ratio, cc.size_threshold_db),
                "extent_m": {"along": float(np.median([w["shape"]["along_m"] for w in win])),
                             "across": float(np.median([w["shape"]["across_m"] for w in win]))},
                "illuminator": win[-1]["top"], "window_s": [win[0]["t"], win[-1]["t"]], "n_frames": len(win)}

    # ------------------------------------------------------------------ records

    def _emit(self, t: float, latency: float, rad: dict, echo: dict | None, passive: bool, echo_out: bool) -> None:
        cc = self.cc
        objects = []
        used_e: set[str] = set()
        echo = echo or {}
        merged: dict[str, str] = {}
        firm = ("confirmed", "static")
        for rk, ro in rad.items():                                  # a radiator and an echo track close for merge_s
            if ro["status"] not in firm:
                continue
            for ek, eo in echo.items():
                if eo["status"] not in firm or ek in used_e:
                    continue
                if np.hypot(*(np.asarray(ro["xy"]) - np.asarray(eo["xy"]))) <= cc.merge_radius_m:
                    since = self.merge_since.setdefault((rk, ek), t)
                    if t - since >= cc.merge_s:
                        merged[rk] = ek
                        used_e.add(ek)
                        break
                else:
                    self.merge_since.pop((rk, ek), None)
        for rk, ro in rad.items():
            objects.append(self._object(ro, echo.get(merged.get(rk)) if rk in merged else None, t))
        for ek, eo in echo.items():
            if ek not in used_e:
                objects.append(self._object(None, eo, t))
        rec = {"t": round(t, 4), "latency_s": round(latency, 3), "chains": {"passive": passive, "echo": echo_out},
               "objects": objects}
        self.records.append(rec)
        if self.on_record is not None:
            self.on_record(rec)

    def _object(self, ro: dict | None, eo: dict | None, t: float) -> dict:
        main = ro if ro is not None else eo
        seen = "both" if ro is not None and eo is not None else ("own noise" if ro is not None else "echo")
        tracks = {k: o["track"] for k, o in (("radiator", ro), ("echo", eo)) if o is not None}
        oid = "+".join(tracks.values())
        if main["status"] == "tentative":
            return {"id": oid, "seen": seen, "tracks": tracks, "status": "tentative", "position": {"xy": _r(main["xy"]), "fix_t": _r(main["fix_t"], 3)}}
        # Each track's own state (the scoring reads it; a merged object shows the radiator track's position).
        per_track = {o["track"]: {"status": o["status"], "xy": _r(o["xy"]), "fix_t": _r(o["fix_t"], 3), "fix_xy": _r(o["fix_xy"]),
                                  "detected": o["detected"]} for o in (ro, eo) if o is not None}
        status = main["status"]
        if ro is not None and eo is not None and "static" in (ro["status"], eo["status"]):
            status = "static"
        kin = main["kin"]
        pos = {"xy": _r(kin["xy"]), "fix_t": _r(main["fix_t"], 3), "fix_xy": _r(main["fix_xy"]), "detected": main["detected"],
               "residual_m": _r(kin["residual_m"]), "window_s": _r(kin["window_s"], 3), "n_fixes": kin["n_fixes"]}
        if "shape" in main:
            pos.update(halfmax_width_m=_r(main["shape"]["widths_m"]), halfmax_major_deg=_r(main["shape"]["major_deg"], 1),
                       width_measured_t=_r(main["shape_t"], 3))
        obj = {"id": oid, "seen": seen, "tracks": tracks, "track_state": per_track, "status": status,
               "detected_frames": main["detected_frames"], "frames_since_confirmation": main["frames_since_confirmation"],
               "position": pos,
               "velocity": {"vxy": _r(kin["velocity"]), "speed": _r(kin["speed"]), "speed_se": _r(kin["speed_se"]),
                            "heading_deg": _r(kin["heading_deg"], 1), "heading_se_deg": _r(kin["heading_se_deg"], 1),
                            "window_s": _r(kin["window_s"], 3), "n_fixes": kin["n_fixes"]}}
        if main.get("lost_t") is not None:
            obj["lost_t"] = _r(main["lost_t"], 3)
        if ro is not None:
            a = ais_match(ro["fixes"], self.reports, t, self.cc.ais_gate_m)
            obj["ais"] = {"id": a["id"], "distance_m": _r(a.get("distance_m")), "fix_t": _r(a.get("t_fix"), 3)} if a["id"] else \
                {"id": None, "note": "no report"}
        else:
            obj["ais"] = {"id": None, "note": "no report"}
        if ro is not None and "radiator" in ro:
            r = ro["radiator"]
            obj["radiator"] = {"received_level": _r(r["received_level"], 5), "range_m": _r(r["range_m"], 1),
                               "source_level_100m_pa": _r(r["source_level_100m_pa"], 4), "calibration": r["calibration"],
                               "tonal_lines": [{"f_hz": _r(ln["f_hz"], 1), "level_db": _r(ln["level_db"], 1), "frames": ln["frames"]}
                                               for ln in r["tonal_lines"]],
                               "lines_frames_measured": r["lines_frames_measured"],
                               "window_s": _r(r["window_s"], 3), "n_frames": r["n_frames"]}
        if eo is not None and "echo" in eo:
            x = eo["echo"]
            obj["echo"] = {"strength_db": {k: _r(v, 1) for k, v in x["strength_db"].items()}, "size_ratio_db": _r(x["size_ratio_db"], 1),
                           "size": x["size"], "extent_m": {k: _r(v, 1) for k, v in x["extent_m"].items()},
                           "illuminator": x["illuminator"], "window_s": _r(x["window_s"], 3), "n_frames": x["n_frames"]}
            if eo is not main:
                obj["echo"]["xy"] = _r(eo["xy"])
        return obj


def _light(f: FrameResult) -> FrameResult:
    """The frame without its spectra (kept: peaks, times, map)."""
    return FrameResult(f.t, f.xy, f.score, f.ships, f.compute_s, f.latency_s, f.B)


# ----------------------------------------------------------------------
# Running on a recording
# ----------------------------------------------------------------------

@dataclass
class RunResult:
    picture: HarborPicture
    records: list[dict]
    record_s: float
    wall_s: float
    run: dict

    @property
    def compute_per_second(self) -> dict:
        s = self.picture.seconds
        return {"passive": s["passive"] / self.record_s, "echo": s["echo"] / self.record_s,
                "characteristics": s["characteristics"] / self.record_s,
                "total": (s["passive"] + s["echo"] + s["characteristics"]) / self.record_s}


def header(run: dict, cfg: ChainConfig) -> dict:
    return {"header": {"schema": SCHEMA, "recording": cfg.name, "run": run,
                       "fields": "docs/harbor.md#the-operator-output"}}


def run(obs: Observation, cfg: ChainConfig, use_cuda: bool = True, chunk_s: float | None = None, out: str | Path | None = None,
        provenance: Provenance | None = None, observation_path: str | Path | None = None, seed: int | None = None,
        keep_frames: bool = False, keep_clean: bool = False, reports: AISReports | None = None, on_record=None) -> RunResult:
    """Run the picture on *obs*, chunks of *chunk_s* (default: the config's) seconds; with *out*, ``picture.jsonl`` is
    written there as the stream runs (header first).  AIS (*reports*, default the observation's) only labels."""
    chunk_s = cfg.chunk_s if chunk_s is None else chunk_s
    prov = provenance or Provenance(data=obs.meta.get("data", "forward_model") if isinstance(obs.meta, dict) else "forward_model")
    info = run_info(cfg.path, observation_path, seed=seed, provenance=prov, use_cuda=use_cuda, overrides=cfg.overrides,
                    chunk_s=chunk_s)
    fh = None
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        fh = open(out / "picture.jsonl", "w")
        fh.write(json.dumps(header(info, cfg)) + "\n")

    def emit(rec: dict) -> None:
        if fh is not None:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        if on_record is not None:
            on_record(rec)

    pic = HarborPicture(obs.geometry, obs.fs, obs.environment, cfg, use_cuda=use_cuda,
                        reports=obs.reports if reports is None else reports, keep_frames=keep_frames, keep_clean=keep_clean,
                        on_record=emit)
    n = max(int(round(chunk_s * obs.fs)), 1)
    t0 = time.perf_counter()
    for s0 in range(0, obs.traces.shape[1], n):
        pic.push(obs.traces[:, s0:s0 + n])
    wall = time.perf_counter() - t0
    if fh is not None:
        fh.close()
    info["wall_s"] = wall
    return RunResult(pic, pic.records, obs.traces.shape[1] / obs.fs, wall, info)


def read_picture(path: str | Path) -> tuple[dict, list[dict]]:
    """``(header, records)`` of a ``picture.jsonl``."""
    lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    return lines[0]["header"], lines[1:]


def summary(res: RunResult) -> dict:
    """What the run found, from the processing alone (no truth): every track that was confirmed, compute and latency."""
    lat = [r["latency_s"] for r in res.records]
    tracks = {}
    for rec in res.records:
        for o in rec["objects"]:
            if o["status"] == "tentative":
                continue
            t = tracks.setdefault(o["id"], {"id": o["id"], "seen": o["seen"], "first_t": rec["t"], "statuses": []})
            t["last_t"] = rec["t"]
            t["seen"] = o["seen"]
            if o["status"] not in t["statuses"]:
                t["statuses"].append(o["status"])
            if o.get("ais", {}).get("id"):
                t["ais"] = o["ais"]["id"]
    return {"recording": res.picture.cfg.name, "record_s": res.record_s, "frames": len(res.records),
            "compute_per_second_of_data": res.compute_per_second,
            "latency_s": {"median": float(np.median(lat)) if lat else None, "max": float(np.max(lat)) if lat else None},
            "objects": list(tracks.values()), "ais_used_in_processing": False, "run": res.run}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m acoustic_sim.water.stream", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observation", required=True, help="<name>.observation.npz")
    ap.add_argument("--processing", required=True, help="processing config (examples/configs/processing/<recording>.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (recorded)")
    ap.add_argument("--cpu", action="store_true", help="process on the CPU (default: the GPU, falling back to the CPU)")
    ap.add_argument("--chunk", type=float, default=None, help="seconds per push (default: the config's)")
    ap.add_argument("--held-out", action="store_true", help="the recording was not used to set any parameter (test set)")
    ap.add_argument("--recorded", action="store_true", help="the observation is a field recording, not forward model data")
    ap.add_argument("--no-video", action="store_true", help="skip picture.mp4")
    ap.add_argument("--show", action="store_true", help="show the operator view live")
    ap.add_argument("--truth", default=None, help="<name>.truth.npz: draw the true positions dashed (off by default)")
    a = ap.parse_args(argv)
    cfg = load_processing(a.processing, a.set)
    obs = Observation.load(a.observation)
    prov = Provenance(data="observation" if a.recorded else "forward_model", held_out=a.held_out)
    view = None
    if a.show or not a.no_video:
        from acoustic_sim.water.plotting import HarborPictureView

        truth = None
        if a.truth:
            from acoustic_sim.contract import Truth

            truth = Truth.load(a.truth)
        view = HarborPictureView(obs.environment, obs.geometry, title=cfg.name, truth=truth, show=a.show)
    res = run(obs, cfg, use_cuda=not a.cpu, chunk_s=a.chunk, out=a.out, provenance=prov, observation_path=a.observation,
              on_record=view.update if view is not None else None)
    out = Path(a.out)
    (out / "summary.json").write_text(json.dumps(summary(res), indent=2))
    if view is not None and not a.no_video:
        view.save(out / "picture.mp4", frame_png=out / "picture_frame.png")
    c = res.compute_per_second
    print(f"{cfg.name}: {res.record_s:.0f} s of data, {len(res.records)} frames, compute per s of data {c['total']:.2f}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

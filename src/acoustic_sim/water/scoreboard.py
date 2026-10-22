"""The scoreboard every stage of the harbor picture is measured on.

A chain produces a map over time, ``B`` (n_cells, n_frames) on scan cells ``grid`` at frame times
``times``.  The scoreboard turns maps into per object detection probability, false alarm rate and
localisation error against the :class:`~acoustic_sim.contract.Truth` (it reads the truth, so it sits
with ``evaluate.py`` on the scoring side; processing modules never import it).  Definitions
(``docs/architecture.md``, Scoreboard):

* **Threshold** ``T``: from runs of the identical chain on traces with one object's contribution
  left out of the mix (:class:`ChainMaps` with ``removed``).  The statistic of a frame is its
  largest local maximum outside the ship exclusion zones and more than ``fa_radius`` from every
  object still in the mix; ``T`` is the smallest value exceeded in at most ``pfa`` (5 %) of the
  frames of all those runs together.  It is set on the tuning set and then frozen.
* **Detection** of object *i* in frame *k*: a local maximum above ``T`` within ``detect_radius``
  (5 m) of the object's true position, the peak being assigned to the nearest true object (so one
  peak cannot detect two pilings 5 m apart).
* **Pd** per object: detected frames over frames in which the object is active.
* **False alarms**: local maxima above ``T`` more than ``fa_radius`` (10 m) from every true object
  and outside the ship exclusion zones, per frame (``false_alarm_rate``), and the fraction of frames with at
  least one (``false_alarm_frame_rate``), the measured counterpart of ``pfa`` on the scene scored.  The two
  agree with ``pfa`` only for a map whose background far from the objects does not depend on which objects
  are in the scene (a calibrated map, e.g. the whitened matched field statistic); a map normalised over the
  scene every frame (the occupancy posterior) moves mass between objects and background, so its false alarm
  rate on the full scene is not the rate its threshold was set for and has to be read from these fields.
* **Localisation error**: median and 90th percentile distance from the detecting peak to the true
  position, over detected frames.
* **Full record map**: the sum of the frame maps; its threshold is the largest local maximum of the
  object free full record maps (no false peak on the tuning set).  Static objects are also scored
  on it, and the pilings resolved as distinct peaks on it are the **separation**.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label, maximum_filter

from acoustic_sim.contract import Provenance, Truth

DETECT_RADIUS = 5.0
FA_RADIUS = 10.0
ELLIPSE95_CHI2 = float(-2.0 * np.log(0.05))      # chi squared, 2 degrees of freedom, 0.95
PFA = 0.05


@dataclass
class ChainMaps:
    """One run of a chain: maps ``B`` (G, K) on ``grid`` (G, 2) at ``times`` (K,).

    ``exclusion`` (G, K) marks the cells inside the exclusion zones around the reporting ships
    (ignored by the threshold and the false alarms).  ``removed`` lists the objects whose
    contribution was left out of the traces (an object free run).  ``present`` lists the objects in
    the scene of this run (default: every object of the truth not in ``removed``).
    """

    grid: np.ndarray
    times: np.ndarray
    B: np.ndarray
    exclusion: np.ndarray | None = None
    provenance: Provenance = field(default_factory=Provenance)
    removed: tuple[str, ...] = ()
    present: tuple[str, ...] | None = None
    label: str = ""

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=float)
        self.times = np.asarray(self.times, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        if self.B.shape != (len(self.grid), len(self.times)):
            raise ValueError(f"B must be (n_cells={len(self.grid)}, n_frames={len(self.times)}), got {self.B.shape}")
        if self.exclusion is None:
            self.exclusion = np.zeros(self.B.shape, dtype=bool)
        if self.exclusion.shape != self.B.shape:
            raise ValueError(f"exclusion must have the shape of B {self.B.shape}, got {self.exclusion.shape}")

    @property
    def full_record(self) -> np.ndarray:
        """The full record map: the sum of the frame maps (G,)."""
        return np.sum(np.where(self.exclusion, 0.0, self.B), axis=1)

    def objects(self, truth: Truth) -> list[str]:
        if self.present is not None:
            return [k for k in self.present if k in truth.tracks]
        return [k for k in truth.tracks if k not in self.removed]


def local_maxima(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Boolean mask of the local maxima of *values* (G, ...) on the regular, possibly land masked grid
    of cell centres *grid* (G, 2): a cell is a maximum when no cell of its 3 x 3 neighbourhood is larger.
    A flat maximum (connected cells of equal value, none with a larger neighbour) is one maximum: only its
    first cell is kept, so one flat false peak counts as one false alarm."""
    grid = np.asarray(grid, dtype=float)
    steps = np.diff(np.unique(grid[:, 0]))
    dx = float(steps.min()) if len(steps) else 1.0
    ix = np.rint((grid[:, 0] - grid[:, 0].min()) / dx).astype(int)
    iy = np.rint((grid[:, 1] - grid[:, 1].min()) / dx).astype(int)
    v = np.asarray(values, dtype=float)
    flat = v.reshape(len(grid), -1)
    out = np.zeros(flat.shape, dtype=bool)
    img = np.full((iy.max() + 1, ix.max() + 1), -np.inf)
    marked = np.zeros(img.shape, dtype=bool)
    eight = np.ones((3, 3), dtype=bool)
    for j in range(flat.shape[1]):
        img[iy, ix] = flat[:, j]
        mx = maximum_filter(img, size=3, mode="constant", cval=-np.inf)
        m = (flat[:, j] >= mx[iy, ix]) & np.isfinite(flat[:, j])
        # Adjacent maxima have equal values (each is >= the other): one per connected component.
        marked[:] = False
        marked[iy[m], ix[m]] = True
        lab, n = label(marked, structure=eight)
        if n < m.sum():
            cells = np.flatnonzero(m)
            _, first = np.unique(lab[iy[cells], ix[cells]], return_index=True)
            m[:] = False
            m[cells[first]] = True
        out[:, j] = m
    return out.reshape(v.shape)


def _positions(truth: Truth, ids: list[str], times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True positions (O, K, 2) and active flags (O, K) of the objects *ids* at *times*."""
    if not ids:
        return np.empty((0, len(times), 2)), np.empty((0, len(times)), dtype=bool)
    xy = np.stack([truth.tracks[i].position_at(times) for i in ids])
    active = np.stack([truth.tracks[i].active_at(times) for i in ids])
    return xy, active


def _far_from(xy_cells: np.ndarray, obj_xy: np.ndarray, radius: float) -> np.ndarray:
    """True for cells (P, 2) more than *radius* from every point in *obj_xy* (O, 2)."""
    if len(obj_xy) == 0:
        return np.ones(len(xy_cells), dtype=bool)
    d = np.hypot(xy_cells[:, None, 0] - obj_xy[None, :, 0], xy_cells[:, None, 1] - obj_xy[None, :, 1])
    return d.min(axis=1) > radius


def free_frame_maxima(run: ChainMaps, truth: Truth, fa_radius: float = FA_RADIUS) -> np.ndarray:
    """Per frame, the largest local maximum of an object free run outside the exclusion zones and more
    than *fa_radius* from every object still in the scene (``-inf`` for an empty frame)."""
    ids = run.objects(truth)
    xy, _ = _positions(truth, ids, run.times)
    peaks = local_maxima(run.B, run.grid) & ~run.exclusion
    out = np.full(len(run.times), -np.inf)
    for k in range(len(run.times)):
        g = np.flatnonzero(peaks[:, k])
        g = g[_far_from(run.grid[g], xy[:, k], fa_radius)]
        if len(g):
            out[k] = float(run.B[g, k].max())
    return out


def free_full_record_maximum(run: ChainMaps, truth: Truth, fa_radius: float = FA_RADIUS) -> float:
    """Largest local maximum of an object free run's full record map more than *fa_radius* from every
    position any object still in the scene takes during the record."""
    ids = run.objects(truth)
    xy, _ = _positions(truth, ids, run.times)
    full = run.full_record
    g = np.flatnonzero(local_maxima(full, run.grid))
    pts = xy.reshape(-1, 2)
    g = g[_far_from(run.grid[g], pts, fa_radius)] if len(pts) else g
    return float(full[g].max()) if len(g) else -np.inf


@dataclass
class Threshold:
    """The frozen threshold of one chain configuration."""

    T: float
    T_full: float
    pfa: float
    n_frames: int
    frame_maxima: np.ndarray
    runs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"T": self.T, "T_full_record": self.T_full, "pfa": self.pfa, "n_frames": self.n_frames,
                "frames_above_T": int(np.sum(self.frame_maxima > self.T)), "object_free_runs": list(self.runs)}


def set_threshold(free_runs: list[ChainMaps], truth: Truth, pfa: float = PFA, fa_radius: float = FA_RADIUS) -> Threshold:
    """``T`` such that at most *pfa* of the frames of the object free runs have a local maximum above it
    (outside the exclusion zones and away from the objects still in the scene), and the full record
    threshold (the largest such maximum of their full record maps)."""
    if not free_runs:
        raise ValueError("set_threshold needs at least one object free run")
    m = np.concatenate([free_frame_maxima(r, truth, fa_radius) for r in free_runs])
    if not np.any(np.isfinite(m) & (m > 0)):
        raise ValueError("set_threshold: no object free frame has a positive local maximum outside the exclusion zones; "
                         "there is no noise statistic to set the threshold on")
    s = np.sort(m)
    n_above = int(np.floor(pfa * len(s)))
    T = float(s[len(s) - n_above - 1])
    T_full = max(free_full_record_maximum(r, truth, fa_radius) for r in free_runs)
    return Threshold(T, float(T_full), pfa, len(s), m, [r.label for r in free_runs])


@dataclass
class ObjectScore:
    """One object's row: Pd over the active frames of every run it was scored on, the errors of the
    detecting peaks, and (static objects) the fraction of runs whose full record map detects it."""

    id: str
    kind: str
    radiates: bool
    ais: bool
    moving: bool
    frames: int
    detected: int
    pd: float
    median_error_m: float
    p90_error_m: float
    full_record_detected: float | None = None
    full_record_error_m: float | None = None
    full_record_hits: list[bool] | None = None
    runs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in self.__dict__.items()}


@dataclass
class Scoreboard:
    """Per object scores of one chain configuration on one dataset and one data split."""

    per_object: list[ObjectScore]
    false_alarm_rate: float
    threshold: Threshold
    provenance: Provenance
    false_alarms: int = 0
    n_frames: int = 0
    separation: dict | None = None
    label: str = ""
    meta: dict = field(default_factory=dict)
    false_alarm_frame_rate: float = float("nan")

    def get(self, oid: str) -> ObjectScore:
        return next(o for o in self.per_object if o.id == oid)

    def to_dict(self) -> dict:
        return {"label": self.label, "threshold": self.threshold.to_dict(), "false_alarm_rate": self.false_alarm_rate,
                "false_alarm_frame_rate": self.false_alarm_frame_rate, "false_alarms": self.false_alarms, "n_frames": self.n_frames, "separation": self.separation,
                "per_object": [o.to_dict() for o in self.per_object], "provenance": self.provenance.to_dict(), **self.meta}


def _peaks_above(run: ChainMaps, T: float) -> list[np.ndarray]:
    peaks = local_maxima(run.B, run.grid) & ~run.exclusion & (run.B > T)
    return [np.flatnonzero(peaks[:, k]) for k in range(len(run.times))]


def _assign(cells_xy: np.ndarray, obj_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest object index and its distance for every peak (P, 2) against objects (O, 2)."""
    if len(cells_xy) == 0 or len(obj_xy) == 0:
        return np.empty(0, int), np.empty(0)
    d = np.hypot(cells_xy[:, None, 0] - obj_xy[None, :, 0], cells_xy[:, None, 1] - obj_xy[None, :, 1])
    j = d.argmin(axis=1)
    return j, d[np.arange(len(j)), j]


def _object_meta(truth: Truth, oid: str) -> dict:
    tr = truth.tracks[oid]
    moving = bool(np.ptp(tr.xy[:, 0]) > 0 or np.ptp(tr.xy[:, 1]) > 0)
    return {"kind": tr.kind, "radiates": bool(tr.level_pa > 0), "ais": bool(tr.ais), "moving": moving}


def _as_list(x) -> list:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def score(
    maps: dict[str, ChainMaps | list[ChainMaps]],
    truth: Truth | list[Truth],
    threshold: Threshold,
    primary: ChainMaps | list[ChainMaps] | None = None,
    detect_radius: float = DETECT_RADIUS,
    fa_radius: float = FA_RADIUS,
    pilings: list[str] | None = None,
    label: str = "",
) -> Scoreboard:
    """Score every object in *maps* (object id -> the run, or one run per data realisation, its detection
    is read from; with several runs *truth* is the matching list of truths, and the frames are pooled).

    *primary* (default: the runs most objects are read from) gives the false alarm rate.  Objects that
    never move are also scored on the full record map of each run; *pilings* (ids) get the separation
    count (pilings detected on the full record map, each peak assigned to its nearest object).
    """
    maps = {k: _as_list(v) for k, v in maps.items()}
    n_real = {len(v) for v in maps.values()}
    if len(n_real) != 1:
        raise ValueError("every object needs the same number of runs")
    n_real = n_real.pop()
    truths = _as_list(truth)
    if len(truths) == 1 and n_real > 1:
        truths = truths * n_real
    if len(truths) != n_real:
        raise ValueError(f"{n_real} runs per object but {len(truths)} truths")
    if primary is None:
        counts: dict[tuple, int] = {}
        for v in maps.values():
            counts[tuple(id(r) for r in v)] = counts.get(tuple(id(r) for r in v), 0) + 1
        best = max(counts, key=counts.get)
        primary = next(v for v in maps.values() if tuple(id(r) for r in v) == best)
    primary = _as_list(primary)

    cache: dict[int, list] = {}

    def frame_assignments(run: ChainMaps, tru: Truth):
        if id(run) not in cache:
            ids = run.objects(tru)
            xy, _ = _positions(tru, ids, run.times)
            cache[id(run)] = [(*_assign(run.grid[p], xy[:, k]), ids, p) for k, p in enumerate(_peaks_above(run, threshold.T))]
        return cache[id(run)]

    rows = []
    for oid, runs in maps.items():
        meta = _object_meta(truths[0], oid)
        errs, frames, full_hits, full_errs = [], 0, [], []
        for run, tru in zip(runs, truths):
            _, act = _positions(tru, [oid], run.times)
            frames += int(act[0].sum())
            fa = frame_assignments(run, tru)
            for k in np.flatnonzero(act[0]):
                j, d, ids, _ = fa[k]
                hit = [dd for jj, dd in zip(j, d) if ids[jj] == oid and dd <= detect_radius]
                if hit:
                    errs.append(min(hit))
            if not meta["moving"]:
                full = run.full_record
                g = np.flatnonzero(local_maxima(full, run.grid) & (full > threshold.T_full))
                ids = run.objects(tru)
                pos = np.stack([tru.tracks[i].xy[0] for i in ids]) if ids else np.empty((0, 2))
                j, d = _assign(run.grid[g], pos)
                hit = [dd for jj, dd in zip(j, d) if ids[jj] == oid and dd <= detect_radius]
                full_hits.append(bool(hit))
                if hit:
                    full_errs.append(min(hit))
        errs = np.array(errs)
        row = ObjectScore(oid, meta["kind"], meta["radiates"], meta["ais"], meta["moving"], frames, len(errs),
                          len(errs) / frames if frames else float("nan"),
                          float(np.median(errs)) if len(errs) else float("nan"),
                          float(np.percentile(errs, 90)) if len(errs) else float("nan"), runs=[r.label for r in runs])
        if not meta["moving"]:
            row.full_record_detected = float(np.mean(full_hits))
            row.full_record_hits = full_hits
            row.full_record_error_m = float(np.median(full_errs)) if full_errs else float("nan")
        rows.append(row)

    # False alarms on the primary runs.
    n_fa, k_fa, K = 0, 0, 0
    for run, tru in zip(primary, truths):
        ids = run.objects(tru)
        xy, _ = _positions(tru, ids, run.times)
        for k, (_, _, _, p) in enumerate(frame_assignments(run, tru)):
            n = int(np.sum(_far_from(run.grid[p], xy[:, k], fa_radius)))
            n_fa += n
            k_fa += n > 0
        K += len(run.times)
    sep = None
    if pilings:
        per_run = []
        for i, (run, tru) in enumerate(zip(maps[pilings[0]], truths)):
            per_run.append(sum(1 for p in pilings if p in maps and _full_hit(maps[p][i], tru, p, threshold, detect_radius)))
        sep = {"pilings": len([p for p in pilings if p in maps]), "resolved_per_run": per_run,
               "resolved_median": float(np.median(per_run))}
    return Scoreboard(rows, n_fa / max(K, 1), threshold, primary[0].provenance, n_fa, K, sep, label,
                      false_alarm_frame_rate=k_fa / max(K, 1))


def _full_hit(run: ChainMaps, tru: Truth, oid: str, threshold: Threshold, detect_radius: float) -> bool:
    full = run.full_record
    g = np.flatnonzero(local_maxima(full, run.grid) & (full > threshold.T_full))
    ids = run.objects(tru)
    pos = np.stack([tru.tracks[i].xy[0] for i in ids]) if ids else np.empty((0, 2))
    j, d = _assign(run.grid[g], pos)
    return any(ids[jj] == oid and dd <= detect_radius for jj, dd in zip(j, d))


# ----------------------------------------------------------------------
# Tracks: the headline scoring from stage 2 on
# ----------------------------------------------------------------------
#
# Fixed in stage 2 (``docs/architecture.md``, Scoreboard), replacing the definitions above where they differ; the
# threshold method above stays as a diagnostic.
#
# * A detection (a map peak) within ``hit_radius`` (5 m) of an object's true position is a hit.
# * A confirmed track is matched to the object nearest over its fixes (median distance) if that is within
#   ``assoc_radius`` (10 m; 20 m on the 5 m harbor), otherwise it is a false track.
# * The false alarm rate is the number of confirmed false tracks per minute of recording.
# * Pd per object (and per kind of track: radiator, echo, either): the fraction of the picture frames, from the
#   first frame the object could have been confirmed, in which a confirmed (or static) track matched to it is within
#   ``hit_radius`` of it.  The first frame it could have been confirmed: the first frame in which the chain feeding
#   that tracker (passive for radiator tracks, echo for echo tracks) has an output and the object is active, plus
#   ``confirm - 1`` frames of that chain (``confirm`` of the tracker's ``TrackerConfig``).

FIRM = ("confirmed", "static")


def static_body_cells(truth: Truth, oid: str) -> np.ndarray | None:
    """The cell centres (M, 2) of a static body's mask as the forward model drew it (the scenario's body of a vessel
    with one waypoint, on the solver grid, ``scatterer.body_factor``); ``None`` for a moving object or a radiator
    without a body."""
    from acoustic_sim.water.scatterer import Scatterer, body_factor

    v = next((x for x in truth.scenario.get("vessels", []) if x.get("id") == oid), None)
    if v is None or not v.get("body_length") or len(v.get("waypoints", [])) != 1 or oid not in truth.tracks:
        return None
    dx = float(truth.scenario.get("environment", {}).get("dx", 1.0))
    sc = Scatterer(oid, truth.tracks[oid], float(v["body_length"]), float(v.get("body_width") or v["body_length"]),
                   v.get("body_kind", "soft"), float(v.get("body_strength", 0.5)), v.get("body_shape", "rect"))
    centre, heading = sc.pose_at(float(truth.tracks[oid].t[0]))
    half = sc.length + sc.width + 4 * dx
    x = np.arange(np.floor((centre[0] - half) / dx), np.ceil((centre[0] + half) / dx) + 1) * dx
    y = np.arange(np.floor((centre[1] - half) / dx), np.ceil((centre[1] + half) / dx) + 1) * dx
    got = body_factor(sc, centre, heading, x, y, dx)
    if got is None:
        return None
    rows, cols, f = got
    iy, ix = np.nonzero(f < 0.999)
    return np.column_stack([x[cols][ix], y[rows][iy]])


class ObjectDistance:
    """Distance from a point to each true object: to the outline of a static body's mask (0 inside it), to the
    centre of everything else (moving bodies and radiators keep their centre)."""

    def __init__(self, truth: Truth) -> None:
        self.truth = truth
        self.cells = {oid: static_body_cells(truth, oid) for oid in truth.tracks}

    def __call__(self, oid: str, t: float, xy) -> float:
        cells = self.cells.get(oid)
        xy = np.asarray(xy, dtype=float)
        if cells is not None and len(cells):
            return float(np.min(np.hypot(*(cells - xy[None, :]).T)))
        return float(np.hypot(*(self.truth.tracks[oid].position_at(t)[0] - xy)))

    def nearest_point(self, oid: str, t: float, xy) -> np.ndarray:
        """The point of the object the distance is measured to."""
        cells = self.cells.get(oid)
        xy = np.asarray(xy, dtype=float)
        if cells is not None and len(cells):
            return cells[int(np.argmin(np.hypot(*(cells - xy[None, :]).T)))]
        return self.truth.tracks[oid].position_at(t)[0]


def match_tracks(tracks: dict, truth: Truth, radius: float, objects: list[str] | None = None,
                 dist: ObjectDistance | None = None) -> list[dict]:
    """Every track against the truth: matched to the object nearest over its fixes (median distance, to the outline
    of a static body and the centre of anything else, :class:`ObjectDistance`) when that is within *radius*, else a
    false track.  *tracks*: id -> fixes ``(t, xy, ...)``."""
    ids = objects if objects is not None else list(truth.tracks)
    dist = dist or ObjectDistance(truth)
    out = []
    for k, fixes in tracks.items():
        d = {oid: float(np.median([dist(oid, f[0], f[1]) for f in fixes])) for oid in ids}
        nearest = min(d, key=d.get)
        xy = np.array([f[1] for f in fixes], dtype=float)
        out.append({"track": k, "fixes": len(fixes), "first_t": float(fixes[0][0]), "last_t": float(fixes[-1][0]),
                    "true_object": nearest if d[nearest] <= radius else None, "false_track": d[nearest] > radius,
                    "nearest_object": nearest, "median_error_m": d[nearest], "first_xy": xy[0].round(1).tolist(),
                    "last_xy": xy[-1].round(1).tolist(), "median_xy": np.median(xy, axis=0).round(1).tolist()})
    return out


def picture_tracks(records: list[dict]) -> dict[str, dict]:
    """Per track of a picture (``picture.jsonl`` records) that was ever confirmed: its kind and its fixes
    ``(t, xy)`` from the frames it was confirmed, static or lost in."""
    out: dict[str, dict] = {}
    for rec in records:
        for o in rec["objects"]:
            for kind, tid in o.get("tracks", {}).items():
                st = o.get("track_state", {}).get(tid)
                if st is None or st["status"] not in FIRM + ("lost",):
                    continue
                d = out.setdefault(tid, {"kind": kind, "fixes": {}, "first_t": rec["t"]})
                d["fixes"][st["fix_t"]] = st["fix_xy"]
    return {k: {"kind": v["kind"], "first_t": v["first_t"], "fixes": sorted(v["fixes"].items())} for k, v in out.items()}


def score_picture(records: list[dict], truth: Truth, record_s: float, confirm: dict[str, int], assoc_radius: float = 10.0,
                  hit_radius: float = DETECT_RADIUS) -> dict:
    """The stage 2 scoring of a picture (definitions above).  *confirm*: the ``confirm`` of each tracker
    (``{"radiator": 6, "echo": 8}``)."""
    tracks = picture_tracks(records)
    dist = ObjectDistance(truth)
    matches = {m["track"]: dict(m, kind=tracks[m["track"]]["kind"])
               for m in match_tracks({k: v["fixes"] for k, v in tracks.items()}, truth, assoc_radius, dist=dist)}
    times = np.array([r["t"] for r in records])
    chain_of = {"radiator": "passive", "echo": "echo"}
    per = {}
    for oid, tr in truth.tracks.items():
        act = tr.active_at(times) if len(times) else np.zeros(0, bool)
        pos = tr.position_at(times) if len(times) else np.empty((0, 2))
        row = {"kind": tr.kind, "radiates": bool(tr.level_pa > 0), "ais": bool(tr.ais), "pd": {}, "frames": {}, "hits": {},
               "median_error_m": {}, "tracks": sorted(k for k, m in matches.items() if m["true_object"] == oid)}
        starts = {}
        for kind in ("radiator", "echo"):
            out_k = np.array([bool(r["chains"].get(chain_of[kind])) for r in records]) & act
            first = np.flatnonzero(out_k)
            if len(first) == 0 or kind not in confirm:
                continue
            chain_frames = np.flatnonzero(np.array([bool(r["chains"].get(chain_of[kind])) for r in records]))
            later = chain_frames[chain_frames >= first[0]]
            starts[kind] = int(later[min(confirm[kind] - 1, len(later) - 1)])
        for kind in ("radiator", "echo", "any"):
            k0 = min(starts.values()) if kind == "any" and starts else starts.get(kind)
            if k0 is None:
                continue
            frames = [k for k in range(k0, len(records)) if act[k]]
            errs = []
            for k in frames:
                best = None
                for o in records[k]["objects"]:
                    for tk, tid in o.get("tracks", {}).items():
                        st = o.get("track_state", {}).get(tid)
                        if st is None or st["status"] not in FIRM or (kind != "any" and tk != kind):
                            continue
                        if matches.get(tid, {}).get("true_object") != oid:
                            continue
                        d = dist(oid, records[k]["t"], st["xy"])
                        best = d if best is None else min(best, d)
                if best is not None and best <= hit_radius:
                    errs.append(best)
            row["frames"][kind] = len(frames)
            row["hits"][kind] = len(errs)
            row["pd"][kind] = len(errs) / len(frames) if frames else None
            row["median_error_m"][kind] = float(np.median(errs)) if errs else None
        per[oid] = row
    false = [m for m in matches.values() if m["false_track"]]
    # The 95 % ellipse of every confirmed track's position at the frames it had a fix: does it hold the truth?
    n_in = n_all = 0
    seen: set = set()
    for rec in records:
        for o in rec["objects"]:
            for tk, tid in o.get("tracks", {}).items():
                st = o.get("track_state", {}).get(tid)
                if st is None or st["status"] not in FIRM or not st.get("detected") or st.get("cov") is None:
                    continue
                m = matches.get(tid)
                if m is None or m["true_object"] is None or (tid, rec["t"]) in seen:
                    continue
                seen.add((tid, rec["t"]))
                d = dist.nearest_point(m["true_object"], rec["t"], st["xy"]) - np.asarray(st["xy"], float)
                n_all += 1
                n_in += int(d @ np.linalg.solve(np.asarray(st["cov"], float), d) <= ELLIPSE95_CHI2)
    return {"objects": per, "tracks": list(matches.values()),
            "ellipse_coverage": {"fixes": n_all, "inside": n_in, "fraction": n_in / n_all if n_all else None},
            "false_tracks": {"count": len(false), "per_minute": len(false) / (record_s / 60.0),
                             "radiator": sum(1 for m in false if m["kind"] == "radiator"),
                             "echo": sum(1 for m in false if m["kind"] == "echo"), "tracks": false},
            "definitions": {"hit_radius_m": hit_radius, "assoc_radius_m": assoc_radius, "confirm": dict(confirm)}}


def peak_hits(frames, truth: Truth, hit_radius: float = DETECT_RADIUS) -> dict:
    """Diagnostic per object: the fraction of the frames with a map peak within *hit_radius* (each peak labelled
    with its nearest object) and the median distance of those peaks.  *frames*: ``(t, peaks (P, 2))``."""
    ids = list(truth.tracks)
    hit = {oid: [] for oid in ids}
    n = 0
    for t, xy in frames:
        n += 1
        if len(xy) == 0:
            continue
        pos = np.stack([truth.tracks[i].position_at(t)[0] for i in ids])
        best: dict[str, float] = {}
        for p in np.asarray(xy).reshape(-1, 2):
            d = np.hypot(*(pos - p).T)
            j = int(np.argmin(d))
            if d[j] <= hit_radius:
                best[ids[j]] = min(best.get(ids[j], np.inf), float(d[j]))
        for oid, d in best.items():
            hit[oid].append(d)
    return {oid: {"frames": n, "hit": len(v) / max(n, 1), "median_error_m": float(np.median(v)) if v else None}
            for oid, v in hit.items()}


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def _fmt(v, f="{:.2f}") -> str:
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def _wrap(text: str, width: int = 100) -> str:
    import textwrap

    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def plot_table(sb: Scoreboard, path, title: str = "", recording: str | None = None) -> None:
    """``scoreboard.png``: Pd per object as horizontal bars (coloured by Pd), with each object's median and 90th
    percentile error printed beside its bar; the threshold, the false alarms and the provenance in the title."""
    import matplotlib.pyplot as plt

    from acoustic_sim.plotting import new_figure, save_figure

    objs = sb.per_object
    fig, ax = new_figure((12.0, max(6.0, 1.5 + 0.55 * len(objs))))
    y = np.arange(len(objs))[::-1]
    pd = np.array([o.pd if np.isfinite(o.pd) else 0.0 for o in objs])
    ax.barh(y, pd, color=[plt.cm.RdYlGn(0.15 + 0.7 * o.pd) if np.isfinite(o.pd) else "0.7" for o in objs], ec="k", lw=0.6)
    for yi, o, v in zip(y, objs, pd):
        txt = f"Pd {_fmt(o.pd)}, median err {_fmt(o.median_error_m, '{:.1f}')} m, p90 {_fmt(o.p90_error_m, '{:.1f}')} m"
        if o.full_record_detected is not None:
            txt += f", full record {_fmt(o.full_record_detected)}"
        ax.text(min(v, 1.0) + 0.01, yi, txt, va="center", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{o.id} ({o.kind}{', AIS' if o.ais else ''}{', moving' if o.moving else ''})" for o in objs])
    ax.set_xlim(0.0, 1.9)
    ax.set_xlabel("Pd (fraction of frames)")
    ax.set_ylabel("object")
    sep = (f", pilings resolved {sb.separation['resolved_per_run']} of {sb.separation['pilings']}" if sb.separation else "")
    ax.set_title(_wrap(f"{title or sb.label}\nthreshold T = {sb.threshold.T:.4g} (at most {100 * sb.threshold.pfa:.0f} % of "
                       f"{sb.threshold.n_frames} object free frames above), false alarms {sb.false_alarm_rate:.2f} per frame, "
                       f"frames with a false alarm {100 * sb.false_alarm_frame_rate:.0f} %{sep}\n{sb.provenance.label()}"))
    save_figure(fig, path, recording=recording)


def plot_map(sb: Scoreboard, truth: Truth, env, path, layout=None, title: str = "", recording: str | None = None) -> None:
    """Every object's track (or position) with its Pd printed beside it, on the chart."""
    import matplotlib.pyplot as plt

    from acoustic_sim.plotting import new_figure, save_figure
    from acoustic_sim.water.plotting import declutter_labels

    fig, ax = new_figure((12.0, 10.0))
    x0, x1, y0, y1 = env.extent
    ax.imshow(np.where(env.water, 1.0, 0.0), origin="lower", extent=(x0, x1, y0, y1), cmap="Greys_r", vmin=-0.6, vmax=1.2)
    if layout is not None:
        r = layout.receivers
        ax.plot(r[:, 0], r[:, 1], ".", color="0.35", ms=2, label="sensors")
    anns = []
    for o in sb.per_object:
        tr = truth.tracks[o.id]
        c = plt.cm.RdYlGn(0.15 + 0.7 * o.pd) if np.isfinite(o.pd) else "0.5"
        if o.moving:
            ax.plot(tr.xy[:, 0], tr.xy[:, 1], "-", color=c, lw=2.5)
            p = tr.xy[len(tr.xy) // 2]
        else:
            p = tr.xy[0]
            ax.plot(*p, "o", color=c, mec="k", ms=7)
        txt = f"{o.id}: Pd {_fmt(o.pd)}"
        if o.full_record_detected is not None:
            txt += f", full record {o.full_record_detected:.2f}"
        anns.append(ax.annotate(txt, p, xytext=(5, 5), textcoords="offset points", fontsize=11,
                                bbox=dict(boxstyle="round,pad=0.2", fc="w", ec=c, alpha=0.85)))
    ax.plot([], [], "-", color="0.4", lw=2.5, label="true track of a moving object (colour: Pd)")
    ax.plot([], [], "o", color="0.4", mec="k", ms=7, ls="none", label="static object (colour: Pd)")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (east) [m]")
    ax.set_ylabel("y (north) [m]")
    ax.legend(loc="lower left")
    ax.set_title(_wrap(f"{title or sb.label}\n{sb.provenance.label()}"))
    declutter_labels(ax, anns)
    save_figure(fig, path, recording=recording)

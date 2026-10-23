#!/usr/bin/env python3
"""Where the false tracks come from (prompt ``false_tracks``, section 1): every peak that started a radiator track, and
every confirmed false track, on the tuning recordings, labelled from the truth (a diagnostic; nothing here feeds the
processing) and measured with what the processing has.

The chain runs unchanged (``harbor_stream.run_recording``) with the radiator tracker's ``update`` wrapped.  For every
radiator track, from the frame of its first fix:

* ``ratio``: the largest, over the other confirmed radiator tracks of that frame, of the track's predicted map at the peak
  scaled to the track's own peak, over the data map at the peak (the quantity the sidelobe test compares with
  ``explain_fraction``), and that track; ``ratio_median``: the same at every later fix of the track, the median;
* ``widths_m``: the half maximum widths (major, minor) of the map peak;
* ``coherence``: the least squares source spectrum of the track over its first six fixes (about what the echo test sees
  when a track is about to confirm) against the nearest
  illuminator's at that illuminator's prediction (``stream.spectral_coherence``, the quantity of ``echo_coherence``);
* ``persistence``: its number of fixes and the time from its first to its last fix; whether it confirmed.

``results.json`` also gives every distribution over the 0.5 m and 1 m recordings alone (``distributions_fine``, and of
the confirmed tracks ``confirmed_fine``) and the largest ratio at any fix of a true ship there
(``radiator_fix_ratio_max_fine``): the 5 m harbor's grid and false tracks are of another kind.

Classes (the first that applies, at the first fix): ``radiator`` (within 5 m of a radiating object), ``silent_body``
(within 5 m of a silent body: an echo located passively), ``sidelobe_near_miss`` (``ratio`` between 0.3 and 0.5),
``direct_path`` (within 5 m of the straight line from a ship to its nearest cable channel), ``elsewhere``.  The
confirmed false tracks of the run (radiator and echo, ``summary.json``) get the same classes from their median
position.  Outputs in ``output/studies/false_tracks/``: ``peaks.json`` (one row per track), ``results.json`` (per class
and quantity the distribution and, for radiator against silent body, the separation of each quantity: the area under
the ROC curve), one figure per quantity, ``run.log``.

    python examples/harbor/05_detection/harbor_false_tracks.py [--recordings simple bistatic multistatic harbor]
    python examples/harbor/05_detection/harbor_false_tracks.py --tuning-set      # run_all_3, task 4

``--tuning-set`` (prompt ``run_all_3``, task 4) runs the same measurement on the whole tuning set as
``scripts/reproduce.py`` defines it: noise seed 0 of the four families and noise seeds 4 to 9 of ``simple``,
``bistatic``, ``multistatic`` and ``multistatic_b``, each with both noise models (the old one and harbor_v2), into
``output/studies/false_tracks/tuning_set/``; it sets no threshold and changes no setting.  Criterion, fixed before
it ran: at least 20 echo samples (radiator track starts of class ``silent_body``) in the tuning set, both noise models
together; ``results.json`` gives the counts per class, per noise model and the separation of each quantity with
those counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harbor_stream as hs  # noqa: E402

from acoustic_sim.contract import Truth, contract_paths  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water import stream as S  # noqa: E402
from acoustic_sim.water import streaming as SM  # noqa: E402
from acoustic_sim.water.characteristics import peak_shape, source_spectrum  # noqa: E402
from acoustic_sim.water.scoreboard import ObjectDistance  # noqa: E402

CLASSES = ("radiator", "silent_body", "sidelobe_near_miss", "direct_path", "elsewhere")
NEAR = 5.0
K_COH = 6                     # fixes of the coherence: about what the echo test sees when a track is about to confirm
CONFIGS = ROOT / "examples" / "configs" / "processing"
PROCESSING = {"simple": "simple", "bistatic": "bistatic", "multistatic": "multistatic", "harbor": "harbor"}


def seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    u = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.hypot(*(p - (a + u * ab))))


def classify(xy, ratio: float | None, radiating: dict[str, float], silent: dict[str, float],
             ships_to_cable: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[str, str | None]:
    """The class of a peak at *xy* (the first that applies): ``radiator`` (*radiating*: object -> distance [m], one
    within ``NEAR``), ``silent_body`` (*silent*, the same), ``sidelobe_near_miss`` (*ratio* in [0.3, 0.5)),
    ``direct_path`` (within ``NEAR`` of a segment of *ships_to_cable*: ship -> (ship position, nearest channel)),
    ``elsewhere``."""
    xy = np.asarray(xy, float)
    for group, cls in ((radiating, "radiator"), (silent, "silent_body")):
        if group:
            k = min(group, key=group.get)
            if group[k] <= NEAR:
                return cls, k
    if ratio is not None and 0.3 <= ratio < 0.5:
        return "sidelobe_near_miss", None
    for s, (p, c) in ships_to_cable.items():
        if seg_dist(xy, np.asarray(p, float), np.asarray(c, float)) <= NEAR:
            return "direct_path", s
    return "elsewhere", None


class Labeller:
    """The truth classes of a position at a time (a diagnostic)."""

    def __init__(self, truth: Truth, cable: np.ndarray) -> None:
        self.truth, self.dist, self.cable = truth, ObjectDistance(truth), cable
        self.radiating = [k for k, tr in truth.tracks.items() if tr.level_pa > 0]
        self.silent = [k for k, tr in truth.tracks.items() if tr.level_pa <= 0]

    def __call__(self, t: float, xy, ratio: float | None) -> tuple[str, str | None]:
        xy = np.asarray(xy, float)
        seg = {}
        for k in self.radiating:
            p = self.truth.tracks[k].position_at(t)[0]
            seg[k] = (p, self.cable[int(np.argmin(np.hypot(*(self.cable - p).T)))])
        return classify(xy, ratio, {k: self.dist(k, t, xy) for k in self.radiating},
                        {k: self.dist(k, t, xy) for k in self.silent}, seg)


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Area under the ROC curve of *pos* above *neg* (0.5: no separation, 1: complete)."""
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(np.mean((p > n) + 0.5 * (p == n)))


def record(rec: str, out: Path, use_cuda: bool, log, obs_path: Path | None = None, processing: str | None = None,
           seed: int = 0) -> list[dict]:
    """Run the chain on *rec* with the radiator tracker wrapped; one row per radiator track and per confirmed false
    track.  *obs_path* (default ``output/recordings/<rec>/dataset.observation.npz``) and *processing* (default
    ``PROCESSING[rec]``) for the recordings of the tuning set."""
    obs_path = obs_path or ROOT / "output" / "recordings" / rec / "dataset.observation.npz"
    processing = processing or PROCESSING[rec]
    truth_path = contract_paths(obs_path)[1]
    truth = Truth.load(truth_path)
    rows: dict[str, dict] = {}
    pic: dict = {}
    orig_init, orig_update = S.HarborPicture.__init__, SM.RadiatorTracker.update

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        pic["p"] = self

    def update(self, frame):
        if self.prefix != "r" or "p" not in pic or frame.spectrum is None:
            return orig_update(self, frame)
        p = pic["p"]
        before = set(self.tracks)
        conf = {k: tr for k, tr in self.tracks.items() if tr.confirmed}
        orig_update(self, frame)
        t = frame.t
        all_ills = self.illuminators(frame)
        for k, tr in self.tracks.items():
            if tr.fixes[-1][0] != t:
                continue
            ills = {i: v for i, v in all_ills.items() if i != k}     # as _hold_echo: never the track itself
            xy = np.asarray(tr.fixes[-1][1], float)
            value = float(tr.fixes[-1][2])
            best, by = None, None
            for ck, ctr in conf.items():                           # the sidelobe ratio of this fix
                if ck == k:
                    continue
                src = ctr.predict(t)
                r = self._response(frame, src, xy, ck)
                if r is not None and self.grid is not None and frame.B is not None:
                    q = r * self._own_peak(frame, src) / max(value, 1e-30)
                    if best is None or q > best:
                        best, by = q, ck
            if k not in before:                                    # a new track: the peak that started it
                sh = peak_shape(frame.B, self.grid, xy, self.cfg.shape_radius) if frame.B is not None else {}
                il = min(ills, key=lambda i: np.hypot(*(ills[i].predict(t) - xy))) if ills else None
                rows[k] = {"recording": rec, "track": k, "t": t, "xy": xy.tolist(), "value": value, "ratio": best,
                           "ratio_track": by, "widths_m": sh.get("widths_m"), "illuminator": il, "sc": [], "si": [],
                           "ratios": []}
            row = rows.get(k)
            if row is None:
                continue
            if best is not None:
                row["ratios"].append(best)
            if row["illuminator"] is None and ills:
                row["illuminator"] = min(ills, key=lambda i: np.hypot(*(ills[i].predict(t) - xy)))
            il = row["illuminator"]
            if il is not None and len(row["sc"]) < K_COH and (il in self.tracks or il in self.lost):
                ilt = self.tracks.get(il) or self.lost.get(il)
                row["sc"].append(source_spectrum(frame.spectrum, frame.weights, p.layout, xy, p.passive.freqs, p.c))
                row["si"].append(source_spectrum(frame.spectrum, frame.weights, p.layout, ilt.predict(t), p.passive.freqs, p.c))
        for k, row in rows.items():
            tr = self.tracks.get(k) or self.lost.get(k)
            if tr is not None:
                row.update(n_fixes=len(tr.fixes), duration_s=float(tr.fixes[-1][0] - tr.fixes[0][0]),
                           confirmed=bool(tr.confirmed_t is not None), held=bool(tr.held))

    S.HarborPicture.__init__, SM.RadiatorTracker.update = init, update
    try:
        s = hs.run_recording(rec, CONFIGS / f"{processing}.json", obs_path, truth_path, out / "runs" / rec, False,
                             use_cuda=use_cuda, seed=seed, video=False, figures=False)
    finally:
        S.HarborPicture.__init__, SM.RadiatorTracker.update = orig_init, orig_update
    from acoustic_sim.contract import Observation

    obs = Observation.load(obs_path)
    geo = obs.geometry
    cable = geo.receivers[geo.slices[geo.cables[0].name]]
    lab = Labeller(truth, cable)
    out_rows = []
    for row in rows.values():
        sc, si = row.pop("sc"), row.pop("si")
        row["coherence"] = S.spectral_coherence(np.array(sc), np.array(si)) if len(sc) >= 3 else None
        row["coherence_frames"] = len(sc)
        k_ = len(sc)
        row["coherence_debiased"] = (k_ * row["coherence"] - 1.0) / (k_ - 1.0) if row["coherence"] is not None else None
        rs = row.pop("ratios")
        row["ratio_median"] = float(np.median(rs)) if rs else None           # over every fix with a confirmed other track
        row["ratio_list"] = [float(x) for x in rs]
        row["ratio_max"] = float(np.max(rs)) if rs else None
        row["ratio_fixes"] = len(rs)
        row["class"], row["object"] = lab(row["t"], row["xy"], row["ratio"])
        row["kind"] = "radiator_start"
        row["false"] = False
        out_rows.append(row)
    for f in s["false_tracks"]["tracks"]:
        src = rows.get(f["track"])
        cls, obj = lab(0.5 * (f["first_t"] + f["last_t"]), f["median_xy"], None)
        out_rows.append({"recording": rec, "track": f["track"], "kind": "false_echo" if f["track"].startswith("e") else "false_radiator",
                         "xy": f["median_xy"], "class": cls, "object": obj, "false": True,
                         "coherence": src["coherence"] if src else None, "ratio": src["ratio"] if src else None,
                         "ratio_median": src["ratio_median"] if src else None,
                         "widths_m": src["widths_m"] if src else None, "n_fixes": src.get("n_fixes") if src else None,
                         "duration_s": f["last_t"] - f["first_t"], "n_fixes_track": f["fixes"]})
    n = {c: sum(r["class"] == c for r in out_rows if r["kind"] == "radiator_start") for c in CLASSES}
    log(f"{rec}: {sum(n.values())} radiator track starts {n}; false tracks {[(r['track'], r['class']) for r in out_rows if r['false']]}")
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", nargs="*", default=list(PROCESSING))
    ap.add_argument("--out", default="output/studies/false_tracks")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="redraw from peaks.json")
    ap.add_argument("--tuning-set", action="store_true", help="the whole tuning set of reproduce.py, both noise models")
    a = ap.parse_args()
    jobs = [(r, None, None, 0) for r in a.recordings]
    if a.tuning_set:
        sys.path.insert(0, str(ROOT / "scripts"))
        import reproduce as R

        if a.out == "output/studies/false_tracks":
            a.out = "output/studies/false_tracks/tuning_set"
        jobs = []
        for r in R.recordings():
            if r.held_out:
                continue
            jobs.append((r.name, contract_paths(r.dataset.with_suffix(".npz"))[0], r.processing, r.seed))
            jobs.append((f"{r.name}_noise2", contract_paths(r.dataset_noise2.with_suffix(".npz"))[0], r.processing, r.seed))
        a.recordings = [j[0] for j in jobs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    if a.plot_only:
        rows = json.loads((out / "peaks.json").read_text())
    else:
        rows = []
        for rec, obs_path, proc, seed in jobs:
            rows += record(rec, out, not a.cpu, log, obs_path, proc, seed)
        (out / "peaks.json").write_text(json.dumps(rows, indent=1, default=float))

    starts = [r for r in rows if r["kind"] == "radiator_start"]
    quantities = {
        "coherence": lambda r: r["coherence"],
        "ratio": lambda r: r["ratio"],
        "ratio_median": lambda r: r.get("ratio_median"),
        "ratio_max": lambda r: r.get("ratio_max"),
        "coherence_debiased": lambda r: r.get("coherence_debiased"),
        "width_major_m": lambda r: r["widths_m"][0] if r["widths_m"] else None,
        "width_minor_m": lambda r: r["widths_m"][1] if r["widths_m"] else None,
        "n_fixes": lambda r: r.get("n_fixes"),
        "duration_s": lambda r: r.get("duration_s"),
    }
    per_fix = [x for r in starts if r["class"] == "radiator" and not r["recording"].startswith("harbor") for x in r.get("ratio_list", [])]
    res0 = {"radiator_fix_ratio_max_fine": float(np.max(per_fix)) if per_fix else None,
            "radiator_fix_ratio_n_fine": len(per_fix)}
    res = {"recordings": a.recordings, "tuning_only": True, "classes": list(CLASSES), "distributions": {}, "separation": {},
           "provenance": {"truth_inputs": ["object positions (the classes, a diagnostic)"], "simulator_knowledge": [],
                          "data": "forward_model", "held_out": False},
           "counts": {c: sum(r["class"] == c for r in starts) for c in CLASSES},
           "confirmed_counts": {c: sum(r["class"] == c and r.get("confirmed") for r in starts) for c in CLASSES},
           "false_tracks": [{k: r[k] for k in ("recording", "track", "kind", "class", "object", "xy", "coherence", "ratio")}
                            for r in rows if r["false"]]}
    for q, f in quantities.items():
        per = {c: [f(r) for r in starts if r["class"] == c and f(r) is not None] for c in CLASSES}
        fine = {c: [f(r) for r in starts if r["class"] == c and f(r) is not None and not r["recording"].startswith("harbor")] for c in CLASSES}
        res.setdefault("distributions_fine", {})[q] = {c: {"n": len(v), "min": float(np.min(v)) if v else None,
                                                           "max": float(np.max(v)) if v else None,
                                                           "values": [float(x) for x in v]} for c, v in fine.items()}
        conf = {c: [f(r) for r in starts if r["class"] == c and f(r) is not None and r.get("confirmed")
                    and not r["recording"].startswith("harbor")] for c in CLASSES}
        res.setdefault("confirmed_fine", {})[q] = {c: {"n": len(v), "min": float(np.min(v)) if v else None,
                                                       "max": float(np.max(v)) if v else None} for c, v in conf.items()}
        res["distributions"][q] = {c: {"n": len(v), "median": float(np.median(v)) if v else None,
                                       "p10": float(np.percentile(v, 10)) if v else None,
                                       "p90": float(np.percentile(v, 90)) if v else None,
                                       "min": float(np.min(v)) if v else None, "max": float(np.max(v)) if v else None,
                                       "values": [float(x) for x in v]} for c, v in per.items()}
        # Separation of echoes located passively (silent body) from radiators, and of sidelobes (the near misses and
        # the starts elsewhere) from radiators: the area under the ROC curve (0.5 none, 0 or 1 complete).
        res["separation"][q] = {"silent_body_vs_radiator": auc(per["silent_body"], per["radiator"]),
                                "sidelobe_vs_radiator": auc(per["sidelobe_near_miss"] + per["elsewhere"], per["radiator"])}
        fig, ax = new_figure((12, 8))
        data = [per[c] for c in CLASSES]
        ax.boxplot([d if d else [np.nan] for d in data], showfliers=True)
        for i, d in enumerate(data):
            ax.plot(np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), d, ".", alpha=0.5)
        ax.set_xticks(range(1, len(CLASSES) + 1), [f"{c.replace('_', ' ')}\n(n = {len(d)})" for c, d in zip(CLASSES, data)])
        unit = {"coherence": "coherence with the nearest illuminator", "ratio": "predicted map / data map at the peak",
                "ratio_median": "predicted map / data map, median over the track's fixes",
                "ratio_max": "predicted map / data map, largest over the track's fixes",
                "coherence_debiased": "coherence with the nearest illuminator, bias removed: (K c - 1) / (K - 1)",
                "width_major_m": "half maximum width, major axis [m]", "width_minor_m": "half maximum width, minor axis [m]",
                "n_fixes": "fixes of the track", "duration_s": "first to last fix [s]"}[q]
        ax.set_xlabel("class of the peak that started a radiator track (from the truth, a diagnostic)")
        ax.set_ylabel(unit)
        ax.set_title(f"tuning recordings ({'the tuning set: seeds 0 and 4 to 9, both noise models' if a.tuning_set else ', '.join(a.recordings)}): "
                     f"{unit} per class")
        save_figure(fig, out / f"distribution_{q}.png", recording=None)
    res.update(res0)
    if a.tuning_set:
        n_echo = res["counts"]["silent_body"]
        res["criterion"] = {"text": "at least 20 echo samples (radiator track starts of class silent_body) in the tuning set, "
                                    "both noise models together", "value": n_echo, "passed": n_echo >= 20}
        res["counts_by_model"] = {m: {c: sum(r["class"] == c for r in starts if r["recording"].endswith("_noise2") == (m == "harbor_v2"))
                                      for c in CLASSES} for m in ("old", "harbor_v2")}
        res["confirmed_counts_by_model"] = {m: {c: sum(r["class"] == c and bool(r.get("confirmed")) for r in starts
                                                       if r["recording"].endswith("_noise2") == (m == "harbor_v2"))
                                                for c in CLASSES} for m in ("old", "harbor_v2")}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    log(f"counts {res['counts']}; confirmed {res['confirmed_counts']}")
    for q, sep in res["separation"].items():
        log(f"{q}: separation (AUC) silent body vs radiator {sep['silent_body_vs_radiator']}, sidelobe vs radiator "
            f"{sep['sidelobe_vs_radiator']}")


if __name__ == "__main__":
    main()

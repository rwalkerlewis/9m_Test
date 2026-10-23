#!/usr/bin/env python3
"""The boat of ``multistatic_b`` around its stop (prompt ``false_tracks``, section 2): its true track, its radiator
tracks' fixes and the frames in which it is detected, against time, and what the tracker did with every passive peak
near it (a diagnostic from the truth; nothing here feeds the processing).

The chain runs unchanged with the radiator tracker's ``update`` wrapped.  Per frame: the passive peaks within 10 m of
the true boat, and for each whether a confirmed track explained it as a sidelobe (and which), whether it joined a track
(which) or started one; the boat's radiator track's prediction and gate.  Outputs in
``output/studies/false_tracks/boat/``: ``boat.json``, ``run.log`` and one figure per quantity against time.

    python examples/harbor/05_detection/harbor_boat_stop.py [--recording multistatic_b] [--set KEY=VALUE ...] [--name before]
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
from acoustic_sim.water import streaming as SM  # noqa: E402

NEAR = 10.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default="multistatic_b")
    ap.add_argument("--object", default="boat")
    ap.add_argument("--processing", default="multistatic")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--name", default="before")
    ap.add_argument("--out", default="output/studies/false_tracks/boat")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    obs_path = ROOT / "output" / "recordings" / a.recording / "dataset.observation.npz"
    truth_path = contract_paths(obs_path)[1]
    truth = Truth.load(truth_path)
    boat = truth.tracks[a.object]
    frames = []
    orig = SM.RadiatorTracker.update

    def update(self, frame):
        if self.prefix != "r":
            return orig(self, frame)
        t = frame.t
        n_expl = len(self.explained)
        before = {k: len(tr.fixes) for k, tr in self.tracks.items()}
        pred = {k: (tr.predict(t).tolist(), tr.confirmed) for k, tr in self.tracks.items()}
        orig(self, frame)
        truth_xy = boat.position_at(t)[0]
        peaks = []
        for xy, sc in zip(frame.xy, frame.score):
            xy = np.asarray(xy, float)
            if np.hypot(*(xy - truth_xy)) > NEAR:
                continue
            row = {"xy": xy.tolist(), "score": float(sc), "explained_by": None, "joined": None, "started": False}
            for te, pe, by in self.explained[n_expl:]:
                if np.allclose(pe, xy):
                    row["explained_by"] = by
            for k, tr in self.tracks.items():
                if tr.fixes[-1][0] == t and np.allclose(tr.fixes[-1][1], xy):
                    row["joined"] = k
                    row["started"] = k not in before
            peaks.append(row)
        frames.append({"t": t, "truth": truth_xy.tolist(), "peaks": peaks, "n_peaks": len(frame.xy),
                       "predictions": {k: v for k, v in pred.items() if np.hypot(*(np.asarray(v[0]) - truth_xy)) <= 3 * NEAR},
                       "tracks_on_boat": [k for k, tr in self.tracks.items() if tr.fixes[-1][0] == t
                                          and np.hypot(*(np.asarray(tr.fixes[-1][1]) - truth_xy)) <= 5.0]})

    SM.RadiatorTracker.update = update
    try:
        s = hs.run_recording(a.recording, ROOT / "examples" / "configs" / "processing" / f"{a.processing}.json", obs_path,
                             truth_path, out / "runs" / a.name, True, overrides=a.set, video=False, figures=False)
    finally:
        SM.RadiatorTracker.update = orig
    o = s["objects"][a.object]
    res = {"recording": a.recording, "object": a.object, "overrides": a.set, "pd": o["pd"], "tracks": o["tracks"],
           "track_list": [t for t in s.get("tracks", []) if t.get("true_object") == a.object],
           "frames": frames}
    (out / f"boat_{a.name}.json").write_text(json.dumps(res, indent=1, default=float))
    t = np.array([f["t"] for f in frames])
    tt = np.linspace(0, t[-1], 400)
    true_xy = boat.position_at(tt)
    stop = boat.position_at(np.array([t[-1]]))[0]
    speed = np.hypot(*np.gradient(true_xy, tt, axis=0).T)
    print(f"{a.object}: Pd radiator {o['pd']['radiator']:.3f}, tracks {o['tracks']}; true speed below 0.1 m/s from "
          f"{tt[np.argmax(speed < 0.1)] if (speed < 0.1).any() else None:.2f} s", flush=True)
    # One figure: distance of the true boat, of every peak near it and of the radiator tracks' fixes from where it stops,
    # against time, with the peaks coloured by what the tracker did.
    fig, ax = new_figure((14, 8))
    ax.plot(tt, np.hypot(*(true_xy - stop).T), "k--", lw=2, label="true boat")
    styles = {"joined": ("o", "tab:green", "peak joined a track"), "started": ("^", "tab:blue", "peak started a track"),
              "explained": ("x", "tab:red", "peak explained as a sidelobe"), "none": (".", "0.5", "peak, no track")}
    for key, (mk, col, lab) in styles.items():
        pts = [(f["t"], np.hypot(*(np.asarray(p["xy"]) - stop))) for f in frames for p in f["peaks"]
               if (key == "explained" and p["explained_by"]) or (key == "started" and p["started"])
               or (key == "joined" and p["joined"] and not p["started"]) or (key == "none" and not (p["explained_by"] or p["joined"]))]
        if pts:
            ax.plot(*np.array(pts).T, mk, color=col, label=lab)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance from where the boat stops [m]")
    ax.set_title(f"{a.recording}: the {a.object}'s passive peaks within {NEAR:g} m and what the tracker did ({a.name})")
    ax.legend()
    save_figure(fig, out / f"boat_peaks_{a.name}.png", recording=a.recording)
    fig, ax = new_figure((14, 8))
    det = [f["t"] for f in frames if f["tracks_on_boat"]]
    ax.plot(tt, speed, "k-", label="true speed [m/s]")
    ax.plot(det, np.full(len(det), -0.3), "|", color="tab:green", ms=14, label="frames with a radiator fix within 5 m")
    expl = [f["t"] for f in frames if any(p["explained_by"] for p in f["peaks"])]
    ax.plot(expl, np.full(len(expl), -0.6), "|", color="tab:red", ms=14, label="frames with a peak near it explained as a sidelobe")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.set_title(f"{a.recording}: the {a.object}'s speed and the frames it is detected in ({a.name})")
    ax.legend()
    save_figure(fig, out / f"boat_frames_{a.name}.png", recording=a.recording)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Two measurements of the tracker change on the tuning recordings (diagnostics from the truth, nothing here feeds the
processing):

* what became of the passive chain's peaks at a silent body (the simple harbor's wreck, the bistatic UUV): how many
  frames have a passive peak within 25 m of it, how many of those peaks a confirmed track's predicted map explained (a
  sidelobe, by which track), and which radiator tracks formed there;
* the echo test of every radiator track about to confirm: its coherence with every illuminator (``echo_coherence``)
  and whether it was held as an echo.

    python examples/harbor/05_detection/harbor_tracker_diagnostics.py [--recordings simple bistatic multistatic]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Observation, Truth  # noqa: E402
from acoustic_sim.water import stream as S  # noqa: E402

BODIES = {"simple": ["wreck", "uuv1"], "bistatic": ["uuv1"], "multistatic": ["uuv1", "uuv2", "moored"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", nargs="*", default=list(BODIES))
    ap.add_argument("--out", default="output/studies/tracker_uncertainty/diagnostics.json")
    a = ap.parse_args()
    out = {}
    orig = S.HarborPicture._hold_echo
    for rec in a.recordings:
        log = []

        def hold(self, tr, t):
            frames = {round(ft, 6): (X, w) for ft, X, w in self.spec_buf}
            fx = [f for f in tr.fixes if round(f[0], 6) in frames][-8:]
            coh = {}
            if self.echo is not None and len(fx) >= 3:
                for k, il in self.tracker.illuminators(self._last_frame).items():
                    if k == tr.id:
                        continue
                    sc = [S.source_spectrum(frames[round(f[0], 6)][0], frames[round(f[0], 6)][1], self.layout, f[1], self.passive.freqs, self.c)
                          for f in fx]
                    si = [S.source_spectrum(frames[round(f[0], 6)][0], frames[round(f[0], 6)][1], self.layout, il.predict(f[0]),
                                            self.passive.freqs, self.c) for f in fx]
                    coh[k] = S.spectral_coherence(np.array(sc), np.array(si))
            held = orig(self, tr, t)
            log.append({"t": t, "track": tr.id, "xy": np.asarray(tr.fixes[-1][1]).tolist(), "frames": len(fx), "coherence": coh,
                        "held": held})
            return held

        S.HarborPicture._hold_echo = hold
        cfg = S.load_processing(ROOT / "examples" / "configs" / "processing" / f"{rec}.json")
        obs = Observation.load(ROOT / "output" / "recordings" / rec / "dataset.observation.npz")
        tru = Truth.load(ROOT / "output" / "recordings" / rec / "dataset.truth.npz")
        pic = S.run(obs, cfg, use_cuda=True).picture
        S.HarborPicture._hold_echo = orig
        tracks = dict(pic.tracker.tracks)
        tracks.update(pic.tracker.lost)
        bodies = {}
        for b in BODIES.get(rec, []):
            pos = lambda t: tru.tracks[b].position_at(t)[0]
            near = [f.t for f in pic.p_frames if any(np.hypot(*(p - pos(f.t))) < 25.0 for p in f.xy)]
            ex = [e for e in pic.tracker.explained if np.hypot(*(e[1] - pos(e[0]))) < 25.0]
            bodies[b] = {"frames": len(pic.p_frames), "frames_with_a_peak_within_25m": len(near),
                         "peaks_explained_as_sidelobes": len(ex), "explained_by": sorted({e[2] for e in ex}),
                         "radiator_tracks_within_25m": [k for k, tr in tracks.items()
                                                        if np.median([np.hypot(*(f[1] - pos(f[0]))) for f in tr.fixes]) < 25.0]}
        out[rec] = {"bodies": bodies, "confirmation_tests": log, "held_as_echo": pic.echo_held}
        print(rec, json.dumps(bodies), [(x["track"], {k: round(v, 2) for k, v in x["coherence"].items()}, x["held"]) for x in log], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()

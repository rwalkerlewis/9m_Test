#!/usr/bin/env python3
"""Batch detection: the real-time chain run over the whole record at once, checked against the truth.

A batch run is the stream with one chunk: the same objects as ``harbor_stream.py`` (``acoustic_sim.water.stream``:
passive chain without AIS, radiator tracker, echo chain on the strong tracks, echo tracker, characteristics), pushed the
whole recording in one call, so every step sees the record in the order it was recorded but the tracks are handed to the
echo chain only at the end of the passive pass (the strong tracks of the whole record).  There is one implementation of
every step; the batch chains of before stage 2 (``run_ais_aware``, ``run_bistatic`` with an AIS illuminator) are history
(``docs/harbor_history.md``).

Outputs in ``--out/<dataset>/`` as ``harbor_stream.py``'s.

Usage::

    python examples/harbor/05_detection/harbor_detect.py --dataset simple|bistatic|multistatic|harbor [--cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harbor_stream as hs  # noqa: E402

from acoustic_sim.contract import contract_paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=hs.RECORDINGS)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a processing parameter (recorded)")
    ap.add_argument("--cpu", action="store_true", help="process on the CPU (default: the GPU, falling back to the CPU)")
    ap.add_argument("--cuda", action="store_true", help="kept for old command lines: the GPU is the default")
    ap.add_argument("--out", default="output/harbor_detect")
    a = ap.parse_args()
    processing = hs.CONFIGS / f"{a.dataset}.json"
    observation = ROOT / json.loads(processing.read_text())["observation"]
    s = hs.run_recording(a.dataset, processing, observation, contract_paths(observation)[1], ROOT / a.out / a.dataset, False,
                         use_cuda=not a.cpu, overrides=a.set, chunk=1e9, video=False)
    hs.report(s)


if __name__ == "__main__":
    main()

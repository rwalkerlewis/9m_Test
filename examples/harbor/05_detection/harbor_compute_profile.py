#!/usr/bin/env python3
"""Where the chain's time goes (prompt ``compute_budget``, section 1): every stage of the harbor picture timed per
frame on a recording (``water/profile.py``), with the GPU memory high water mark and the time spent copying between
host and device.

Per recording two runs of ``stream.run`` with the recording's processing config: one as it runs (its compute per second
of data and median latency, the chain's own numbers) and one profiled (each stage between waits for the GPU, so the
stages add up to a little more than the first run's total).  Outputs in ``output/studies/compute/``: ``<name>.json``
(one block per recording: per stage the seconds per frame and per second of data, the share of the profiled total,
the memory and transfer numbers), one bar figure per recording, ``run.log``.

    python examples/harbor/05_detection/harbor_compute_profile.py [--recordings multistatic bistatic] [--name profile]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_sim.contract import Observation  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water.profile import PROFILE  # noqa: E402
from acoustic_sim.water.stream import load_processing, run  # noqa: E402

PROCESSING = {"simple": "simple", "bistatic": "bistatic", "multistatic": "multistatic", "multistatic_b": "multistatic",
              "harbor": "harbor"}


def profile(rec: str, overrides: list[str], use_cuda: bool) -> dict:
    obs = Observation.load(ROOT / "output" / "recordings" / rec / "dataset.observation.npz")
    cfg = load_processing(ROOT / "examples" / "configs" / "processing" / f"{PROCESSING[rec]}.json", overrides)
    plain = run(obs, cfg, use_cuda=use_cuda)
    record_s = obs.traces.shape[1] / obs.fs
    lat = [r["latency_s"] for r in plain.records]
    PROFILE.start(transfers=True)
    t0 = time.perf_counter()
    prof = run(obs, cfg, use_cuda=use_cuda)
    wall = time.perf_counter() - t0
    rep = PROFILE.stop()
    frames = len(prof.picture.p_frames)
    total = sum(v for k, v in rep["seconds"].items() if " / " not in k)             # sub-stages are inside their stage
    stages = {k: {"seconds": v, "per_frame_ms": 1e3 * v / frames, "per_second_of_data": v / record_s,
                  "share": v / total if total else None, "calls": rep["calls"][k]}
              for k, v in sorted(rep["seconds"].items(), key=lambda kv: -kv[1]) if " / " not in k}
    parts = {k: {"seconds": v, "per_frame_ms": 1e3 * v / frames, "per_second_of_data": v / record_s, "calls": rep["calls"][k]}
             for k, v in sorted(rep["seconds"].items(), key=lambda kv: -kv[1]) if " / " in k}
    picture_s = sum(prof.picture.seconds.values())
    return {"recording": rec, "record_s": record_s, "passive_frames": frames, "overrides": overrides,
            "compute_per_second_of_data": sum(plain.picture.seconds.values()) / record_s,
            "compute_by_chain": {k: v / record_s for k, v in plain.picture.seconds.items()},
            "latency_median_s": float(np.median(lat)) if lat else None,
            "profiled": {"compute_per_second_of_data": picture_s / record_s, "wall_s": wall,
                         "stages_total_per_second_of_data": total / record_s,
                         "outside_the_stages_per_second_of_data": (wall - total) / record_s},
            "stages": stages, "sub_stages": parts, "gpu_memory_high_mb": rep["gpu_memory_high_bytes"] / 2**20,
            "transfer": {**rep["transfer"], "h2d_per_second_of_data": rep["transfer"]["h2d_s"] / record_s,
                         "d2h_per_second_of_data": rep["transfer"]["d2h_s"] / record_s}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", nargs="*", default=["multistatic", "bistatic"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--name", default="profile")
    ap.add_argument("--out", default="output/studies/compute")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.name}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    log = open(out / "run.log", "a")
    for rec in a.recordings:
        r = profile(rec, a.set, not a.cpu)
        res[rec] = r
        path.write_text(json.dumps(res, indent=1))
        lines = [f"{a.name} {rec}: compute {r['compute_per_second_of_data']:.2f} s per s of data, latency "
                 f"{r['latency_median_s']:.2f} s; profiled stages {r['profiled']['stages_total_per_second_of_data']:.2f}, "
                 f"GPU memory high {r['gpu_memory_high_mb']:.0f} MB, host to device {r['transfer']['h2d_per_second_of_data']:.3f} "
                 f"s per s ({r['transfer']['h2d_bytes'] / 2**20:.0f} MB), device to host {r['transfer']['d2h_per_second_of_data']:.3f}"]
        lines += [f"    {k:40s} {v['per_frame_ms']:8.1f} ms per frame  {v['per_second_of_data']:6.3f} s/s  {100 * v['share']:5.1f} %"
                  for k, v in r["stages"].items()]
        lines += [f"      {k:38s} {v['per_frame_ms']:8.1f} ms per frame  {v['per_second_of_data']:6.3f} s/s"
                  for k, v in r["sub_stages"].items()]
        for ln in lines:
            print(ln, flush=True)
            log.write(ln + "\n")
        log.flush()
        fig, ax = new_figure((13, 8))
        names = list(r["stages"])[::-1]
        ax.barh(range(len(names)), [r["stages"][k]["per_frame_ms"] for k in names], color="tab:blue")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("time per passive frame (0.25 s of data) [ms]")
        ax.set_ylabel("stage")
        ax.set_title(f"{rec}: compute per stage of the harbor picture ({a.name}, Quadro RTX 4000)")
        save_figure(fig, out / f"{a.name}_{rec}.png", recording=rec)


if __name__ == "__main__":
    main()

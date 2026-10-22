"""The one end to end run CI does: a tiny scenario (50 x 50 m, 1 m grid, 2 s, one ferry, one wreck) through the dataset
build, the stream and the characteristics, on the CPU (the full reproduce, ``scripts/reproduce.py``, is run by hand on
the GPU).  The processing config is ``tests/harbor/ci_processing.json`` (the simple harbor's, scaled to the scene)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from acoustic_sim.contract import Observation
from acoustic_sim.water import HarborScenarioConfig, simulate_scenario
from acoustic_sim.water.scoreboard import match_tracks, picture_tracks
from acoustic_sim.water.stream import load_processing, main, read_picture, run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "ci_processing.json"
RUN_KEYS = {"git_commit", "git_dirty", "processing_config", "processing_config_sha256", "observation", "observation_sha256",
            "python", "numpy", "cupy", "gpu", "seed", "wall_s", "provenance"}

SCENARIO = {
    "environment": {"kind": "simple", "width": 50.0, "height": 50.0, "quay_width": 2.0, "dx": 1.0, "c0": 1500.0},
    "fdtd": {"fs": 3000.0, "total_time": 2.0, "fd_order": 4, "damping_width": 10, "damping_max": 0.15,
             "background_absorption": 0.0001, "shore_loss_width": 2.0, "shore_loss_max": 0.15, "use_cuda": False},
    "cables": [{"name": "cable1", "waypoints": [[12, 12], [38, 12], [38, 38], [12, 38], [12, 14]], "channel_spacing": 1.0,
                "das_mode": "pressure", "gauge_length": 1.0}],
    "hydrophones": [[11, 11], [39, 11], [11, 39], [39, 39]],
    "vessels": [
        {"id": "ferry", "kind": "ship", "waypoints": [[16, 30], [40, 30]], "speed": 4.0, "level_pa": 1.0, "ais": True, "seed": 1},
        {"id": "wreck", "kind": "static", "waypoints": [[27, 19]], "speed": 0.0, "level_pa": 0.0, "ais": None, "seed": 2,
         "body_length": 3.0, "body_width": 3.0, "body_kind": "soft", "body_strength": 0.5, "body_shape": "ellipse"}],
    "noise": {"ambient_rms": 0.002, "shipping_rms": 0.0, "n_shipping": 0, "hydrophone_self_rms": 0.001, "das_noise_rms": 0.005,
              "das_common_mode_rms": 0.0, "das_gain_sigma": 0.0, "das_dead_fraction": 0.0, "c": 1500.0, "band_scale": 5.0},
    "seed": 0,
}


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    d = tmp_path_factory.mktemp("ci")
    ds = simulate_scenario(HarborScenarioConfig.from_dict(SCENARIO), cache_dir=d / "cache", out=d / "ci.npz")
    return d, ds


def test_tiny_scenario_end_to_end_on_the_cpu(recording):
    """The ferry is tracked within 3 m (its matched radiator track over its last fixes), the wreck is the peak of the echo
    map summed over the record (each frame over its median) within 3 m, and every confirmed object carries the
    characteristics (a radiator its level and lines, an echo track its strength and size indicator)."""
    d, ds = recording
    obs, truth = ds.observation, ds.truth
    res = run(obs, load_processing(CONFIG), use_cuda=False)
    tracks = picture_tracks(res.records)
    m = match_tracks({k: v["fixes"] for k, v in tracks.items() if v["kind"] == "radiator"}, truth, 5.0)
    ferry = [x for x in m if x["true_object"] == "ferry"]
    assert ferry, m
    fx = tracks[ferry[0]["track"]]["fixes"][-4:]
    err = [np.hypot(*(truth.tracks["ferry"].position_at(t)[0] - np.asarray(xy))) for t, xy in fx]
    assert np.median(err) <= 3.0, err
    pic = res.picture
    S = sum(f.B / np.median(f.B[f.B > 0]) for f in pic.e_frames if f.ships and f.B is not None and f.B.max() > 0)
    assert np.hypot(*(pic.echo.grid[np.argmax(S)] - truth.tracks["wreck"].xy[0])) <= 3.0
    firm = [o for r in res.records for o in r["objects"] if o["status"] in ("confirmed", "static")]
    assert any("radiator" in o and o["radiator"]["source_level_100m_pa"] > 0 and "tonal_lines" in o["radiator"] for o in firm)
    assert any("echo" in o and o["echo"]["size"] and o["echo"]["strength_db"] for o in firm)
    assert all("halfmax_width_m" in o["position"] or o["detected_frames"] == 0 for o in firm if "radiator" in o or "echo" in o)


def test_every_entry_point_writes_the_run_block(recording, tmp_path):
    """``python -m acoustic_sim.water.stream`` (picture.jsonl header, summary.json) and ``harbor_stream.run_recording``
    (picture.jsonl, summary.json, characteristics.json) carry the run block with the config and observation hashes."""
    d, _ = recording
    obs_path = d / "ci.observation.npz"
    out = tmp_path / "cli"
    assert main(["--observation", str(obs_path), "--processing", str(CONFIG), "--out", str(out), "--cpu", "--no-video",
                 "--set", "tracker.gate_probability=0.995"]) == 0
    header, records = read_picture(out / "picture.jsonl")
    assert records and RUN_KEYS <= set(header["run"])
    assert header["run"]["processing_overrides"] == ["tracker.gate_probability=0.995"]
    assert header["run"]["observation_sha256"] and header["run"]["gpu"] == "cpu"
    assert RUN_KEYS <= set(json.loads((out / "summary.json").read_text())["run"])

    sys.path.insert(0, str(ROOT / "examples" / "harbor" / "05_detection"))
    import harbor_stream

    out2 = tmp_path / "script"
    s = harbor_stream.run_recording("ci", CONFIG, obs_path, d / "ci.truth.npz", out2, held_out=True, use_cuda=False, seed=3,
                                    video=False, figures=False)
    assert s["run"]["provenance"]["held_out"] is True and s["run"]["seed"] == 3
    for name in ("summary.json", "characteristics.json"):
        assert RUN_KEYS <= set(json.loads((out2 / name).read_text())["run"])
    assert RUN_KEYS <= set(read_picture(out2 / "picture.jsonl")[0]["run"])

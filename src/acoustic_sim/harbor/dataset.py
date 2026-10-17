"""Scenario simulation: per-vessel FDTD gathers + noise + AIS -> labelled dataset."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from acoustic_sim.harbor.ais import AISReports, generate_ais
from acoustic_sim.harbor.environment import HarborEnvironment
from acoustic_sim.harbor.fdtd_runner import Gather, simulate_source
from acoustic_sim.harbor.noise import NoiseRealisation, generate_noise
from acoustic_sim.harbor.scenario import HarborScenarioConfig, build_environment, build_layout
from acoustic_sim.harbor.sensors import Cable, SensorLayout
from acoustic_sim.harbor.signatures import InjectionCalibration, make_reference_signal, to_injection
from acoustic_sim.harbor.vessels import Track, VesselConfig, build_track


@dataclass
class HarborDataset:
    """Everything a detector sees (``traces``, ``ais``, ``layout``) and the ground truth (``tracks``)."""

    cfg: HarborScenarioConfig
    layout: SensorLayout
    fs: float
    traces: np.ndarray                      # (n_receivers, n_steps) noisy measurements
    tracks: dict[str, Track]                # truth
    ais: AISReports
    clean: np.ndarray | None = None         # noise-free sum of all sources
    gain: np.ndarray | None = None          # coupling gain per receiver
    dead: np.ndarray | None = None          # dead-channel mask (truth for QC)
    meta: dict = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.traces.shape[1]) / self.fs

    @property
    def unknown_ids(self) -> list[str]:
        """Vessels with no AIS (the detection targets)."""
        return [k for k, tr in self.tracks.items() if not tr.ais]

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "traces": self.traces, "fs": np.float64(self.fs),
            "ais_vessel": self.ais.vessel, "ais_t": self.ais.t, "ais_xy": self.ais.xy,
            "ais_sog": self.ais.sog, "ais_cog": self.ais.cog,
        }
        for name in ("clean", "gain", "dead"):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        track_meta = {}
        for k, tr in self.tracks.items():
            arrays[f"track_{k}_t"], arrays[f"track_{k}_xy"] = tr.t, tr.xy
            arrays[f"track_{k}_speed"], arrays[f"track_{k}_active"] = tr.speed, tr.active
            track_meta[k] = {"kind": tr.kind, "ais": tr.ais, "level_pa": tr.level_pa, "meta": tr.meta}
        arrays["header"] = np.array(json.dumps({
            "config": self.cfg.to_dict(), "tracks": track_meta, "meta": self.meta,
            "layout": {
                "cables": [{"name": c.name, "waypoints": c.waypoints.tolist(), "channel_spacing": c.channel_spacing}
                           for c in self.layout.cables],
                "hydrophones": self.layout.hydrophones.tolist(),
            },
        }))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "HarborDataset":
        with np.load(path) as d:
            header = json.loads(str(d["header"]))
            cfg = HarborScenarioConfig.from_dict(header["config"])
            tracks = {
                k: Track(id=k, kind=m["kind"], ais=m["ais"], t=d[f"track_{k}_t"], xy=d[f"track_{k}_xy"],
                         speed=d[f"track_{k}_speed"], active=d[f"track_{k}_active"],
                         level_pa=m["level_pa"], meta=m["meta"])
                for k, m in header["tracks"].items()
            }
            ais = AISReports(vessel=d["ais_vessel"], t=d["ais_t"], xy=d["ais_xy"], sog=d["ais_sog"], cog=d["ais_cog"])
            lay = header["layout"]
            layout = SensorLayout(
                cables=[Cable(c["name"], np.asarray(c["waypoints"], float), c["channel_spacing"]) for c in lay["cables"]],
                hydrophones=np.asarray(lay["hydrophones"], float),
            )
            return cls(
                cfg=cfg, layout=layout, fs=float(d["fs"]), traces=d["traces"], tracks=tracks, ais=ais,
                clean=d["clean"] if "clean" in d else None,
                gain=d["gain"] if "gain" in d else None,
                dead=d["dead"] if "dead" in d else None,
                meta=header["meta"],
            )


# ----------------------------------------------------------------------
# FDTD rendering (one job per vessel, cached at unit level)
# ----------------------------------------------------------------------

def vessel_injection(
    vcfg: VesselConfig, track: Track, env: HarborEnvironment, cfg: HarborScenarioConfig
) -> np.ndarray:
    """Unit-level FDTD injection waveform (1 Pa rms at 100 m at nominal speed)."""
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps
    t = np.arange(n) / fs
    ref = make_reference_signal(vcfg.kind, track.speed_at(t), track.active_at(t), fs, vcfg.seed)
    cal = InjectionCalibration(dx=env.dx, dt=cfg.fdtd.dt, c=float(np.median(env.model.values[env.water])))
    return to_injection(ref, fs, cal)


def _job(args):
    env, layout, fdtd_cfg, signal, position, trajectory, cache_dir = args
    return simulate_source(env, layout, fdtd_cfg, signal, position=position, trajectory=trajectory, cache_dir=cache_dir)


def render_vessel_gathers(
    cfg: HarborScenarioConfig,
    env: HarborEnvironment,
    layout: SensorLayout,
    tracks: dict[str, Track],
    cache_dir: str | Path | None = None,
    workers: int = 1,
) -> dict[str, Gather]:
    """Unit-level sensor gathers for every vessel (cached; FDTD jobs can run in parallel)."""
    jobs = []
    for v in cfg.vessels:
        track = tracks[v.id]
        signal = vessel_injection(v, track, env, cfg)
        if len(v.waypoints) == 1:
            position, trajectory = (float(v.waypoints[0][0]), float(v.waypoints[0][1])), None
        else:
            position, trajectory = None, (track.t, track.xy)
        jobs.append((env, layout, cfg.fdtd, signal, position, trajectory, None if cache_dir is None else str(cache_dir)))
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            results = list(pool.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    return {v.id: g for v, g in zip(cfg.vessels, results)}


def assemble_dataset(
    cfg: HarborScenarioConfig,
    layout: SensorLayout,
    tracks: dict[str, Track],
    gathers: dict[str, Gather],
    seed: int | None = None,
    levels: dict[str, float] | None = None,
    noise: NoiseRealisation | None = None,
    keep_clean: bool = True,
) -> HarborDataset:
    """Mix cached unit-level gathers into a noisy dataset.

    *levels* overrides ``level_pa`` per vessel (``0`` removes a vessel), which allows
    level sweeps and ablations without re-running FDTD.  A pre-drawn *noise*
    realisation can be passed to hold noise fixed across mixes.
    """
    levels = levels or {}
    ss = np.random.SeedSequence(cfg.seed if seed is None else seed)
    rng_noise, rng_ais = (np.random.default_rng(s) for s in ss.spawn(2))
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps

    clean = np.zeros((layout.n_receivers, n), dtype=np.float64)
    used: dict[str, Track] = {}
    for v in cfg.vessels:
        level = levels.get(v.id, v.level_pa)
        tr = tracks[v.id]
        used[v.id] = Track(tr.id, tr.kind, tr.ais, tr.t, tr.xy, tr.speed, tr.active, level_pa=level, meta=tr.meta)
        if level != 0.0:
            clean += level * gathers[v.id].traces.astype(np.float64)

    real = noise if noise is not None else generate_noise(layout, fs, n, cfg.noise, rng_noise)
    traces = real.apply(clean)
    ais = generate_ais({k: t for k, t in used.items() if t.level_pa > 0}, cfg.ais, rng_ais)
    return HarborDataset(
        cfg=cfg, layout=layout, fs=fs, traces=traces, tracks=used, ais=ais,
        clean=clean.astype(np.float32) if keep_clean else None,
        gain=real.gain, dead=real.dead,
        meta={"shipping_azimuth_deg": real.shipping_azimuth_deg.tolist()},
    )


def simulate_scenario(
    cfg: HarborScenarioConfig,
    env: HarborEnvironment | None = None,
    layout: SensorLayout | None = None,
    cache_dir: str | Path | None = None,
    workers: int = 1,
    seed: int | None = None,
    levels: dict[str, float] | None = None,
) -> HarborDataset:
    """End-to-end: tracks -> FDTD gathers -> mix with noise -> AIS."""
    env = env if env is not None else build_environment(cfg)
    layout = layout if layout is not None else build_layout(cfg, env)
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=cache_dir, workers=workers)
    return assemble_dataset(cfg, layout, tracks, gathers, seed=seed, levels=levels)

"""Scenario simulation: per-vessel FDTD gathers + noise + AIS -> labelled dataset.

A dataset is written as two files of the sensing contract (``acoustic_sim.contract``):
``<name>.observation.npz`` (what a deployed system records and knows) and ``<name>.truth.npz``
(everything else).  :meth:`HarborDataset.load` reads that pair and the old single ``<name>.npz``.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from acoustic_sim.contract import Observation, Truth, contract_paths
from acoustic_sim.water.ais import AISReports
from acoustic_sim.water.ais_sim import generate_ais
from acoustic_sim.water.environment import HarborEnvironment
from acoustic_sim.water.fdtd_runner import Gather, _fused_available, simulate_source, simulate_sources
from acoustic_sim.water.noise import NoiseRealisation, generate_noise
from acoustic_sim.water.scatterer import Scatterer
from acoustic_sim.water.scenario import HarborScenarioConfig, build_environment, build_layout
from acoustic_sim.water.sensors import Cable, SensorLayout
from acoustic_sim.water.signatures import InjectionCalibration, f_max_for_grid, make_reference_signal, to_injection
from acoustic_sim.water.vessels import Track, VesselConfig, build_track


@dataclass
class HarborDataset:
    """The forward side's convenience object: observation and truth in one place.

    The processing side gets :attr:`observation` (traces, sample rate, geometry, public chart, AIS) and scoring gets :attr:`truth` (tracks, components, gains, dead channels, the scenario and
    solver config); :meth:`save` writes them as two files.
    """

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
    components: dict[str, np.ndarray] | None = None
    """Scaled noise-free parts of ``clean``: ``"direct:<vessel>"`` for each radiating vessel and
    ``"scattered:<vessel>><body>"`` for the field of *vessel* scattered by *body*, plus
    ``"noise"`` (the measurement minus the clean field).  Truth for measuring processing gain."""
    env: HarborEnvironment | None = field(default=None, repr=False)
    """Forward model environment (default: built from ``cfg``); its public chart goes into the observation."""
    _geometry: SensorLayout | None = field(default=None, repr=False)
    _environment: object | None = field(default=None, repr=False)

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.traces.shape[1]) / self.fs

    @property
    def unknown_ids(self) -> list[str]:
        """Vessels with no AIS (the detection targets)."""
        return [k for k, tr in self.tracks.items() if not tr.ais]

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    @property
    def observation(self) -> Observation:
        """What the processing side may read: traces, ``fs``, the layout (with its DAS mode and gauge
        length), the public chart and the AIS reports.  The processing band is a choice of the processing
        (``BistaticConfig`` / ``ProcessingConfig``), not part of the recording."""
        if self._environment is None:
            env = self.env if self.env is not None else build_environment(self.cfg)
            self._environment = env.public()
        env = self._environment
        geometry = self._geometry if self._geometry is not None else self.layout
        return Observation(self.traces, float(self.fs), geometry, env, self.ais, 0.0, {
            "instrument": "synthetic: DAS cables and hydrophones (" + ", ".join(
                f"{c.name}: {c.das_mode}" for c in geometry.cables) + ")",
            "units": "Pa (pressure channels); second difference of pressure over the gauge length (strain channels)",
        })

    @property
    def truth(self) -> Truth:
        """Everything else: tracks, components, gains, dead channels, scenario and solver config."""
        fdtd = asdict(self.cfg.fdtd)
        fdtd.update(dt=float(self.cfg.fdtd.dt), n_steps=int(self.cfg.fdtd.n_steps))
        return Truth(tracks=self.tracks, components=self.components, gain=self.gain, dead=self.dead,
                     scenario=self.cfg.to_dict(), fdtd=fdtd, meta=self.meta, clean=self.clean)

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Write ``<name>.observation.npz`` and ``<name>.truth.npz`` for *path* ``<name>.npz``."""
        obs_path, truth_path = contract_paths(path)
        self.observation.save(obs_path)
        self.truth.save(truth_path)
        return obs_path, truth_path

    def save_legacy(self, path: str | Path) -> None:
        """The old single file format (observation and truth together); kept for tests and tools."""
        arrays: dict[str, np.ndarray] = {
            "traces": self.traces, "fs": np.float64(self.fs),
            "ais_vessel": self.ais.vessel, "ais_t": self.ais.t, "ais_xy": self.ais.xy,
            "ais_sog": self.ais.sog, "ais_cog": self.ais.cog,
        }
        for name in ("clean", "gain", "dead"):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        for key, arr in (self.components or {}).items():
            arrays["comp:" + key] = arr
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
        """Load ``<name>.npz`` (the old single file, if it exists) or the pair ``<name>.observation.npz``
        and ``<name>.truth.npz`` (*path* may name either of them)."""
        p = Path(path)
        if p.exists() and not p.name.endswith((".observation.npz", ".truth.npz")):
            return cls._load_legacy(p)
        obs_path, truth_path = contract_paths(p)
        if not obs_path.exists():
            raise FileNotFoundError(f"no dataset at {p} (neither the file nor {obs_path.name} / {truth_path.name})")
        obs, truth = Observation.load(obs_path), Truth.load(truth_path)
        return cls(cfg=HarborScenarioConfig.from_dict(truth.scenario), layout=obs.geometry, fs=obs.fs, traces=obs.traces,
                   tracks=truth.tracks, ais=obs.reports, clean=truth.clean, gain=truth.gain, dead=truth.dead,
                   meta=truth.meta, components=truth.components, _environment=obs.environment)

    @classmethod
    def _load_legacy(cls, path: Path) -> "HarborDataset":
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
                components={k[5:]: d[k] for k in d.files if k.startswith("comp:")} or None,
                _geometry=_with_cable_modes(layout, cfg),
            )


def _with_cable_modes(layout: SensorLayout, cfg: HarborScenarioConfig) -> SensorLayout:
    """The old header dropped the DAS mode and gauge length: take them from the scenario's cable of the same name."""
    modes = {c.name: (c.das_mode, c.gauge_length) for c in cfg.cables}
    return SensorLayout(cables=[Cable(c.name, c.waypoints, c.channel_spacing, *modes.get(c.name, (c.das_mode, c.gauge_length)))
                                for c in layout.cables], hydrophones=layout.hydrophones)


def split_dataset(path: str | Path) -> tuple[Path, Path]:
    """Write ``<name>.observation.npz`` and ``<name>.truth.npz`` next to an old single file ``<name>.npz``
    (``python -m acoustic_sim.contract split``).  The old file is left in place."""
    return HarborDataset.load(path).save(path)


# ----------------------------------------------------------------------
# FDTD rendering (one job per vessel, cached at unit level)
# ----------------------------------------------------------------------

def vessel_injection(
    vcfg: VesselConfig, track: Track, env: HarborEnvironment, cfg: HarborScenarioConfig
) -> np.ndarray:
    """Unit-level FDTD injection waveform (1 Pa rms at 100 m at nominal speed).

    The radiated band ends at :func:`f_max_for_grid` of the environment's grid spacing
    (80 Hz on the default 5 m grid) and never above the Nyquist frequency.
    """
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps
    t = np.arange(n) / fs
    f_max = min(f_max_for_grid(env.dx), 0.45 * fs)
    ref = make_reference_signal(vcfg.kind, track.speed_at(t), track.active_at(t), fs, vcfg.seed, f_max)
    cal = InjectionCalibration(dx=env.dx, dt=cfg.fdtd.dt, c=float(np.median(env.model.values[env.water])))
    return to_injection(ref, fs, cal, f_max)


def make_scatterer(vcfg: VesselConfig, track: Track) -> Scatterer:
    """The scattering body of a vessel with ``body_length > 0``."""
    return Scatterer(vcfg.id, track, vcfg.body_length, vcfg.body_width, vcfg.body_kind, vcfg.body_strength, vcfg.body_shape)


COMBINED = "bodies"
"""Body id of the combined gather (all bodies present at once, ``render_vessel_gathers(combined_bodies=True)``)."""


def scattered_key(source_id: str, body_id: str) -> str:
    """Gather key of the field of *source_id* scattered by *body_id* (or by all, ``COMBINED``)."""
    return f"{source_id}>{body_id}"


def _job(args):
    env, layout, fdtd_cfg, signal, position, trajectory, cache_dir, scatterers = args
    return simulate_source(env, layout, fdtd_cfg, signal, position=position, trajectory=trajectory, cache_dir=cache_dir,
                           scatterers=scatterers)


def render_vessel_gathers(
    cfg: HarborScenarioConfig,
    env: HarborEnvironment,
    layout: SensorLayout,
    tracks: dict[str, Track],
    cache_dir: str | Path | None = None,
    workers: int = 1,
    include_silent: bool = False,
    combined_bodies: bool = False,
) -> dict[str, Gather]:
    """Unit-level sensor gathers (cached; FDTD jobs can run in parallel).

    One gather per radiating vessel (``level_pa != 0``, or every vessel with *include_silent*),
    keyed by its id, plus, for every scattering body in the scenario, one gather per radiating
    vessel keyed ``scattered_key(vessel, body)``: the field of that vessel **scattered** by the
    body, i.e. the run with the body present minus the run without it.  Multiple scattering
    between two bodies is not represented (each body is run on its own).  A body that also
    radiates (``level_pa != 0``) is run as a source without itself as a body.

    With *combined_bodies* every radiating vessel is instead run once with **all** the other
    bodies present and once with none, giving a single gather ``scattered_key(vessel, "bodies")``
    per vessel that includes the multiple scattering between the bodies (two FDTD runs per
    vessel however many bodies there are; the per body split is lost).
    """
    cache = None if cache_dir is None else str(cache_dir)
    bodies = {v.id: make_scatterer(v, tracks[v.id]) for v in cfg.vessels if v.is_body}
    radiating = [v for v in cfg.vessels if include_silent or v.level_pa != 0.0]
    jobs, keys = [], []
    for v in radiating:
        track = tracks[v.id]
        signal = vessel_injection(v, track, env, cfg)
        if len(v.waypoints) == 1:
            position, trajectory = (float(v.waypoints[0][0]), float(v.waypoints[0][1])), None
        else:
            position, trajectory = None, (track.t, track.xy)
        jobs.append((env, layout, cfg.fdtd, signal, position, trajectory, cache, ()))
        keys.append(v.id)
        others = [body for bid, body in bodies.items() if bid != v.id]
        if combined_bodies:
            if others:
                jobs.append((env, layout, cfg.fdtd, signal, position, trajectory, cache, tuple(others)))
                keys.append(scattered_key(v.id, COMBINED))
        else:
            for body in others:
                jobs.append((env, layout, cfg.fdtd, signal, position, trajectory, cache, (body,)))
                keys.append(scattered_key(v.id, body.id))
    if _fused_available(cfg.fdtd):
        # One batched GPU pass over every job that shares the grid (stage 3); cached jobs are loaded.
        results = simulate_sources(env, layout, cfg.fdtd, [
            dict(signal=signal, position=position, trajectory=trajectory, scatterers=scatterers)
            for (_, _, _, signal, position, trajectory, _, scatterers) in jobs], cache_dir=cache)
    elif workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            results = list(pool.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    out = dict(zip(keys, results))
    for key in list(out):
        if ">" in key:
            src = key.split(">")[0]
            out[key] = Gather(traces=(out[key].traces.astype(np.float64) - out[src].traces.astype(np.float64)).astype(np.float32),
                              fs=out[key].fs)
    return out


def assemble_dataset(
    cfg: HarborScenarioConfig,
    layout: SensorLayout,
    tracks: dict[str, Track],
    gathers: dict[str, Gather],
    seed: int | None = None,
    levels: dict[str, float] | None = None,
    noise: NoiseRealisation | None = None,
    keep_clean: bool = True,
    scatter_gain: dict[str, float] | None = None,
    keep_components: bool = False,
    env: HarborEnvironment | None = None,
    out: str | Path | None = None,
) -> HarborDataset:
    """Mix cached unit-level gathers into a noisy dataset.

    *env* is the forward environment the gathers were run on (default: built from *cfg*); its public
    chart is the observation's environment.  With *out* (``<name>.npz``) the observation and truth files
    are written (:meth:`HarborDataset.save`).

    *levels* overrides ``level_pa`` per vessel (``0`` removes a vessel as a source), which
    allows level sweeps and ablations without re-running FDTD.  *scatter_gain* scales the
    scattered field of each body (``0`` removes the body; default 1).  A pre-drawn *noise*
    realisation can be passed to hold noise fixed across mixes.  With *keep_components* the
    dataset carries every scaled component separately (see ``HarborDataset.components``).

    A silent body (``level_pa == 0``) is present in the scene through its scattered gathers
    and keeps its truth track; an AIS vessel reports only while it radiates.  A combined
    gather (``scattered_key(vessel, COMBINED)``, all bodies at once) is used when present; it
    cannot be scaled per body, so *scatter_gain* must then be all ones.
    """
    levels = levels or {}
    scatter_gain = scatter_gain or {}
    ss = np.random.SeedSequence(cfg.seed if seed is None else seed)
    rng_noise, rng_ais = (np.random.default_rng(s) for s in ss.spawn(2))
    fs, n = cfg.fdtd.fs, cfg.fdtd.n_steps

    clean = np.zeros((layout.n_receivers, n), dtype=np.float64)
    comps: dict[str, np.ndarray] = {}
    used: dict[str, Track] = {}
    bodies = [v.id for v in cfg.vessels if v.is_body and scatter_gain.get(v.id, 1.0) != 0.0]
    for v in cfg.vessels:
        level = levels.get(v.id, v.level_pa)
        tr = tracks[v.id]
        used[v.id] = Track(tr.id, tr.kind, tr.ais, tr.t, tr.xy, tr.speed, tr.active, level_pa=level, meta=tr.meta)
        if level != 0.0:
            direct = level * gathers[v.id].traces.astype(np.float64)
            clean += direct
            if keep_components:
                comps["direct:" + v.id] = direct.astype(np.float32)
            combined = scattered_key(v.id, COMBINED)
            if combined in gathers:
                if any(scatter_gain.get(b.id, 1.0) != 1.0 for b in cfg.vessels if b.is_body):
                    raise ValueError("scatter_gain per body needs per body gathers, not the combined gather")
                sc = level * gathers[combined].traces.astype(np.float64)
                clean += sc
                if keep_components:
                    comps["scattered:" + combined] = sc.astype(np.float32)
                continue
            for bid in bodies:
                key = scattered_key(v.id, bid)
                if bid == v.id or key not in gathers:
                    continue
                sc = level * scatter_gain.get(bid, 1.0) * gathers[key].traces.astype(np.float64)
                clean += sc
                if keep_components:
                    comps["scattered:" + key] = sc.astype(np.float32)

    real = noise if noise is not None else generate_noise(layout, fs, n, cfg.noise, rng_noise)
    traces = real.apply(clean)
    if keep_components:
        comps["noise"] = (traces.astype(np.float64) - real.gain[:, None] * clean).astype(np.float32)
    ais = generate_ais({k: t for k, t in used.items() if t.level_pa > 0}, cfg.ais, rng_ais)
    ds = HarborDataset(
        cfg=cfg, layout=layout, fs=fs, traces=traces, tracks=used, ais=ais,
        clean=clean.astype(np.float32) if keep_clean else None,
        gain=real.gain, dead=real.dead,
        meta={"shipping_azimuth_deg": real.shipping_azimuth_deg.tolist(),
              "bodies": bodies, "scatter_gain": {b: scatter_gain.get(b, 1.0) for b in bodies}},
        components=comps or None,
        env=env,
    )
    if out is not None:
        ds.save(out)
    return ds


def simulate_scenario(
    cfg: HarborScenarioConfig,
    env: HarborEnvironment | None = None,
    layout: SensorLayout | None = None,
    cache_dir: str | Path | None = None,
    workers: int = 1,
    seed: int | None = None,
    levels: dict[str, float] | None = None,
    keep_components: bool = False,
    out: str | Path | None = None,
) -> HarborDataset:
    """End-to-end: tracks -> FDTD gathers -> mix with noise -> AIS (and, with *out*, the two files)."""
    env = env if env is not None else build_environment(cfg)
    layout = layout if layout is not None else build_layout(cfg, env)
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=cache_dir, workers=workers)
    return assemble_dataset(cfg, layout, tracks, gathers, seed=seed, levels=levels, keep_components=keep_components,
                            env=env, out=out)

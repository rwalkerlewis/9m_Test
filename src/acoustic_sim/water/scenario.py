"""Scenario configuration (JSON-serialisable) for the harbor simulation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from acoustic_sim.water.ais_sim import AISConfig
from acoustic_sim.water.environment import HarborEnvironment, build_synthetic_harbor
from acoustic_sim.water.fdtd_runner import HarborFDTDConfig
from acoustic_sim.water.noise import NoiseConfig
from acoustic_sim.water.sensor_sampling import validate_layout
from acoustic_sim.water.sensors import Cable, SensorLayout
from acoustic_sim.water.vessels import VesselConfig


@dataclass
class EnvironmentConfig:
    """Grid spacing, sound speed, and an optional crop ``[x0, x1, y0, y1]`` of the harbor
    (the crop becomes its own domain with the origin at ``(x0, y0)``; all other scenario
    coordinates are then in that local frame)."""

    dx: float = 5.0
    c0: float = 1500.0
    shallow_c_drop: float = 10.0
    crop: list[float] | None = None


@dataclass
class CableConfig:
    name: str
    waypoints: list[list[float]]
    channel_spacing: float = 10.0
    das_mode: str = "pressure"
    gauge_length: float = 10.0


def _default_cables() -> list[CableConfig]:
    # Lands on the north shore, crosses the basin with two bends, exits through the entrance.
    return [
        CableConfig(
            name="cable1",
            waypoints=[[1100, 1430], [1100, 1000], [1350, 780], [1350, 540], [1330, 300]],
        )
    ]


def _default_hydrophones() -> list[list[float]]:
    return [[450.0, 1100.0], [1250.0, 700.0], [1850.0, 1100.0], [1500.0, 380.0]]


def default_vessels() -> list[VesselConfig]:
    """AIS ships (ferry, tug, moored) plus hidden unknowns: uuv1, uuv2, diver, static_scatter."""
    return [
        VesselConfig("ferry", "ship", [[1380, 260], [1380, 500], [1380, 700], [1500, 800], [1700, 1000]],
                     speed=6.0, level_pa=1.0, seed=11),
        VesselConfig("tug", "ship", [[600, 1000], [1000, 900], [1250, 720], [1300, 600], [1310, 300]],
                     speed=4.0, level_pa=0.7, seed=12),
        VesselConfig("moored", "ship", [[1650, 1300]], level_pa=0.3, seed=13),
        VesselConfig("uuv1", "uuv", [[1600, 700], [1500, 850], [1450, 1000], [1300, 1100], [1150, 1250]],
                     speed=2.5, level_pa=0.03, wander_sigma=10.0, seed=21),
        # --- unknown-class objects added for Sub-Task 1 ---
        # uuv2: second UUV on the opposite (west) side of the basin, faster, different path
        # Waypoints chosen to clear peninsula at y≈520 (x=530-700); path stays y≥600 through that region
        VesselConfig("uuv2", "uuv", [[400, 820], [480, 700], [620, 650], [800, 560], [950, 420]],
                     speed=3.5, level_pa=0.05, wander_sigma=0.0, seed=31),
        # diver: slow swimmer near the north quay, nearly silent broadband profile
        VesselConfig("diver", "diver", [[1000, 1200], [1050, 1100], [1100, 1000], [1150, 950]],
                     speed=0.8, level_pa=0.01, wander_sigma=5.0, seed=41),
        # static_scatter: stationary hard-body reflector (mine-like); level_pa=0 → passive reflector only
        VesselConfig("static_scatter", "uuv", [[1200, 600]],
                     speed=0.0, level_pa=0.0, seed=51,
                     body_kind="hard", body_length=10.0, body_width=5.0, body_shape="rect"),
    ]


@dataclass
class HarborScenarioConfig:
    """Environment, FDTD settings, sensor layout, vessels, AIS behaviour and noise."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    fdtd: HarborFDTDConfig = field(default_factory=HarborFDTDConfig)
    cables: list[CableConfig] = field(default_factory=_default_cables)
    hydrophones: list[list[float]] = field(default_factory=_default_hydrophones)
    vessels: list[VesselConfig] = field(default_factory=default_vessels)
    ais: AISConfig = field(default_factory=AISConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HarborScenarioConfig":
        return cls(
            environment=EnvironmentConfig(**d.get("environment", {})),
            fdtd=HarborFDTDConfig(**d.get("fdtd", {})),
            cables=[CableConfig(**c) for c in d["cables"]] if "cables" in d else _default_cables(),
            hydrophones=d.get("hydrophones", _default_hydrophones()),
            vessels=[VesselConfig(**v) for v in d["vessels"]] if "vessels" in d else default_vessels(),
            ais=AISConfig(**d.get("ais", {})),
            noise=NoiseConfig(**d.get("noise", {})),
            seed=d.get("seed", 0),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "HarborScenarioConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def build_environment(cfg: HarborScenarioConfig) -> HarborEnvironment:
    e = cfg.environment
    crop = None if e.crop is None else tuple(float(v) for v in e.crop)
    return build_synthetic_harbor(dx=e.dx, c0=e.c0, shallow_c_drop=e.shallow_c_drop, crop=crop)


def build_layout(cfg: HarborScenarioConfig, env: HarborEnvironment | None = None, margin: float = 10.0) -> SensorLayout:
    """Build the sensor layout; if *env* is given, check all sensors are in water."""
    layout = SensorLayout(
        cables=[Cable(c.name, np.asarray(c.waypoints, float), c.channel_spacing, c.das_mode, c.gauge_length)
                for c in cfg.cables],
        hydrophones=np.asarray(cfg.hydrophones, float),
    )
    if env is not None:
        validate_layout(layout, env, margin=margin)
    return layout

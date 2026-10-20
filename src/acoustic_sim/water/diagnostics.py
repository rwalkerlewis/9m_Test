"""Diagnostics: processing that uses forward model knowledge, and says so.

This module may import both sides of the sensing contract (``docs/architecture.md``).  Library code on
the processing side never imports it; a script that wants simulator knowledge on the inference path
has to load the :class:`~acoustic_sim.contract.Truth` file and build one of these objects from it,
which is what makes that use visible.  Every result built with them carries the flags in its
:class:`~acoustic_sim.contract.Provenance`.

* :class:`SimulatorAwarePropagator`: replicas with the numerical, direction dependent wavenumber of the
  FDTD scheme that produced the data (``simulator_knowledge: ["dispersion"]``).
* :class:`VesselPathLocaliser`: empirical replicas from the dataset's own FDTD vessel gathers
  (``simulator_knowledge: ["empirical_replicas"]``).
* :func:`true_illuminator_waveform`: the true illuminator waveform for ``reference="true_waveform"``
  (``truth_inputs: ["waveform"]``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from acoustic_sim.contract import Truth
from acoustic_sim.water.dispersion import DispersionModel, for_scenario, replica_wavenumbers
from acoustic_sim.water.localiser import EmpiricalLocaliser
from acoustic_sim.water.propagation import AnalyticPropagator


class SimulatorAwarePropagator(AnalyticPropagator):
    """The FDTD scheme's numerical wavenumber per direction (:class:`DispersionModel`).

    Construct it with :meth:`from_truth` (the scheme of the scenario that produced the data);
    wrapping a bare :class:`DispersionModel` is for tests and studies of the scheme itself.
    """

    needs_angles = True
    knowledge = ("dispersion",)

    def __init__(self, model: DispersionModel) -> None:
        self.model = model
        self.c = float(model.c)

    @classmethod
    def from_truth(cls, truth: Truth) -> "SimulatorAwarePropagator":
        """Dispersion of the solver in ``truth.scenario`` (grid spacing, time step, order, median water speed)."""
        from acoustic_sim.water.scenario import HarborScenarioConfig, build_environment

        cfg = HarborScenarioConfig.from_dict(truth.scenario)
        return cls(for_scenario(build_environment(cfg), cfg.fdtd))

    @classmethod
    def from_scenario(cls, env, fdtd_cfg) -> "SimulatorAwarePropagator":
        """From a forward model environment and solver config already in hand (forward side scripts)."""
        return cls(for_scenario(env, fdtd_cfg))

    def wavenumbers(self, freqs, theta, xp=np, dtype=np.float64):
        return replica_wavenumbers(self.model, freqs, self.c, theta, xp, dtype)

    def __repr__(self) -> str:
        return f"SimulatorAwarePropagator({self.model})"


@dataclass
class VesselPathLocaliser(EmpiricalLocaliser):
    """:class:`EmpiricalLocaliser` whose replicas are the dataset's own per vessel FDTD unit gathers."""

    mode: str = "vessel_path"

    def _build_vessel_path(self) -> None:
        """Build replica store from M2 vessel-track unit gathers.

        Loads ``cache_dir.parent/dataset.npz`` for track positions, then calls
        ``render_vessel_gathers`` (which reads from the cache) to obtain traces.
        """
        from acoustic_sim.water.dataset import render_vessel_gathers
        from acoustic_sim.water.scenario import HarborScenarioConfig, build_environment, build_layout
        from acoustic_sim.water.vessels import build_track

        dataset_path = self.cache_dir.parent / "dataset.npz"
        truth_path = self.cache_dir.parent / "dataset.truth.npz"
        if dataset_path.exists():
            with np.load(dataset_path) as d:
                header = json.loads(str(d["header"]))
                track_xy = {k: d[f"track_{k}_xy"] for k in header["tracks"]}
            config = header["config"]
        elif truth_path.exists():
            truth = Truth.load(truth_path)
            track_xy = {k: tr.xy for k, tr in truth.tracks.items()}
            config = truth.scenario
        else:
            raise FileNotFoundError(
                f"vessel_path mode requires the dataset at {dataset_path} (or {truth_path}). "
                "Run harbor_m2_dataset.py first."
            )

        cfg = HarborScenarioConfig.from_dict(config)
        env = build_environment(cfg)
        layout = build_layout(cfg, env)
        tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
        gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=self.cache_dir)

        all_xy, all_H, common_freqs = [], [], None
        for vessel_id, gather in gathers.items():
            # For scattered gathers (key contains ">") use the radiating vessel's midpoint.
            src_id = vessel_id.split(">")[0] if ">" in vessel_id else vessel_id
            if src_id not in track_xy:
                continue
            xy_path = track_xy[src_id]       # (T_steps, 2)
            midpoint = xy_path[len(xy_path) // 2]

            H, freqs = self._rfft_band(gather.traces)
            all_xy.append(midpoint)
            all_H.append(H)
            if common_freqs is None:
                common_freqs = freqs

        if not all_xy:
            raise RuntimeError("No vessel gathers found in cache_dir.")

        self._source_xy = np.array(all_xy, dtype=float)   # (G, 2)
        self._H = np.stack(all_H, axis=0)                 # (G, N, F)
        self._freqs = common_freqs


def true_illuminator_waveform(truth: Truth, vessel_id: str) -> np.ndarray:
    """The true radiated waveform of *vessel_id* at the emission time (``level_pa`` times the unit
    FDTD injection), for ``BistaticConfig(reference="true_waveform")``: a truth input."""
    from acoustic_sim.water.dataset import vessel_injection
    from acoustic_sim.water.scenario import HarborScenarioConfig, build_environment

    cfg = HarborScenarioConfig.from_dict(truth.scenario)
    v = next(v for v in cfg.vessels if v.id == vessel_id)
    return v.level_pa * vessel_injection(v, truth.tracks[vessel_id], build_environment(cfg), cfg)

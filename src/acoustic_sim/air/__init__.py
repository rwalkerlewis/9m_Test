"""Air-domain acoustic simulation, detection, tracking and engagement tools."""

from importlib import import_module

from acoustic_sim.air.config import DetectionConfig, sound_speed_from_temperature

_EXPORTS = {
    "evaluate_results": ".detection_main",
    "evaluate_results_3d": ".detection_main",
    "run_detection": ".detection_main",
    "run_detection_3d": ".detection_main",
    "run_detection_pipeline": ".detection_main",
    "simulate_scenario": ".detection_main",
    "compute_engagement": ".fire_control",
    "compute_engagement_3d": ".fire_control",
    "compute_lead": ".fire_control",
    "compute_lead_3d": ".fire_control",
    "run_fire_control": ".fire_control",
    "run_fire_control_3d": ".fire_control",
    "simulate_3d_traces": ".forward",
    "simulate_scenario_3d": ".forward",
}

__all__ = [
    "DetectionConfig",
    "sound_speed_from_temperature",
    *_EXPORTS,
]


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

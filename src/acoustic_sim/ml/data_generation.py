"""Training data generation for source classification and maneuver detection.

All training data is generated from the 3D forward model to ensure
consistency with the propagation environment.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal as sp_signal

from acoustic_sim.sources_3d import (
    CircularOrbitSource3D,
    EvasiveSource3D,
    MovingSource3D,
    StaticSource3D,
    source_velocity_at_3d,
)
from acoustic_sim.forward_3d import simulate_3d_traces
from acoustic_sim.noise import generate_sensor_noise, generate_wind_noise

_P_REF = 20e-6

# Source classes.
SOURCE_CLASSES = [
    "quadcopter", "hexacopter", "fixed_wing",
    "bird", "ground_vehicle", "unknown",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(SOURCE_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Maneuver classes.
MANEUVER_CLASSES = [
    "steady", "turning", "accelerating", "diving", "evasive", "hovering",
]
MANEUVER_TO_IDX = {c: i for i, c in enumerate(MANEUVER_CLASSES)}


def _make_multi_rotor_signal(
    n_steps: int,
    dt: float,
    n_rotors: int,
    fundamental: float,
    n_harmonics: int,
    rotor_freq_spread: float,
    harmonic_decay_power: float = 1.0,
    source_level_dB: float = 90.0,
    seed: int = 42,
) -> np.ndarray:
    """Synthesize a multi-rotor drone signal with beat modulation."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * dt
    sig = np.zeros(n_steps, dtype=np.float64)

    for rotor in range(n_rotors):
        freq_offset = 1.0 + rng.uniform(-rotor_freq_spread, rotor_freq_spread)
        f0 = fundamental * freq_offset
        for h in range(1, n_harmonics + 1):
            amp = 1.0 / (h ** harmonic_decay_power)
            phase = rng.uniform(0, 2 * math.pi)
            sig += amp * np.sin(2 * math.pi * f0 * h * t + phase)

    p_source = _P_REF * 10.0 ** (source_level_dB / 20.0)
    mx = np.max(np.abs(sig))
    if mx > 1e-30:
        sig *= p_source / mx
    return sig


def _make_bird_signal(
    n_steps: int,
    dt: float,
    wing_beat_freq: float = 6.0,
    pulse_width_ms: float = 20.0,
    source_level_dB: float = 75.0,
    vocalization_prob: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Synthesize a bird signal: wing beats + optional vocalizations."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * dt
    fs = 1.0 / dt
    sig = np.zeros(n_steps, dtype=np.float64)

    # Wing beats: periodic Gaussian pulses.
    beat_period = 1.0 / wing_beat_freq
    pulse_sigma = pulse_width_ms * 1e-3 / 2.355  # FWHM to sigma
    beat_times = np.arange(0, t[-1], beat_period)
    for bt in beat_times:
        pulse = np.exp(-0.5 * ((t - bt) / pulse_sigma) ** 2)
        # Broadband content via noise modulation.
        noise = rng.standard_normal(n_steps)
        sig += pulse * noise * 0.3

    # Occasional narrowband vocalization.
    duration = t[-1]
    n_vocalizations = rng.poisson(vocalization_prob * duration)
    for _ in range(n_vocalizations):
        voc_time = rng.uniform(0, duration)
        voc_freq = rng.uniform(1000, 8000)
        voc_dur = rng.uniform(0.1, 0.5)
        voc_env = np.exp(-0.5 * ((t - voc_time) / (voc_dur / 4)) ** 2)
        sig += 0.5 * voc_env * np.sin(2 * math.pi * voc_freq * t)

    p_source = _P_REF * 10.0 ** (source_level_dB / 20.0)
    mx = np.max(np.abs(sig))
    if mx > 1e-30:
        sig *= p_source / mx
    return sig


def _make_ground_vehicle_signal(
    n_steps: int,
    dt: float,
    engine_fundamental: float = 40.0,
    speed: float = 15.0,
    source_level_dB: float = 80.0,
    seed: int = 42,
) -> np.ndarray:
    """Synthesize a ground vehicle signal."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * dt
    fs = 1.0 / dt
    sig = np.zeros(n_steps, dtype=np.float64)

    # Engine harmonics.
    for h in range(1, 5):
        freq = engine_fundamental * h
        amp = 1.0 / h
        sig += amp * np.sin(2 * math.pi * freq * t + rng.uniform(0, 2 * math.pi))

    # Tire noise: broadband 200-1000 Hz, proportional to speed.
    noise = rng.standard_normal(n_steps)
    nyq = fs / 2
    lo = min(200.0, nyq * 0.9)
    hi = min(1000.0, nyq * 0.9)
    if hi > lo > 0:
        sos = sp_signal.butter(2, [lo, hi], btype="bandpass", fs=fs, output="sos")
        tire_noise = sp_signal.sosfilt(sos, noise)
        sig += (speed / 30.0) * 0.3 * tire_noise

    # Pink noise rumble.
    pink = rng.standard_normal(n_steps)
    hi_pink = min(500.0, nyq * 0.9)
    lo_pink = min(10.0, hi_pink * 0.5)
    if hi_pink > lo_pink > 0:
        sos = sp_signal.butter(2, [lo_pink, hi_pink], btype="bandpass", fs=fs, output="sos")
        pink = sp_signal.sosfilt(sos, pink)
    sig += 0.2 * pink

    p_source = _P_REF * 10.0 ** (source_level_dB / 20.0)
    mx = np.max(np.abs(sig))
    if mx > 1e-30:
        sig *= p_source / mx
    return sig


def generate_source_signal(
    class_name: str,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Generate a source signal for the given class.

    Returns (signal, params_dict).
    """
    seed = int(rng.integers(0, 2**31))

    if class_name == "quadcopter":
        fundamental = rng.uniform(100, 250)
        sig = _make_multi_rotor_signal(
            n_steps, dt, n_rotors=4, fundamental=fundamental,
            n_harmonics=6, rotor_freq_spread=0.02,
            harmonic_decay_power=1.0, source_level_dB=90.0, seed=seed,
        )
        params = {"fundamental": fundamental, "n_rotors": 4,
                  "altitude_range": (20, 100), "min_speed": 0}

    elif class_name == "hexacopter":
        fundamental = rng.uniform(80, 200)
        sig = _make_multi_rotor_signal(
            n_steps, dt, n_rotors=6, fundamental=fundamental,
            n_harmonics=6, rotor_freq_spread=0.015,
            harmonic_decay_power=1.0, source_level_dB=90.0, seed=seed,
        )
        params = {"fundamental": fundamental, "n_rotors": 6,
                  "altitude_range": (20, 100), "min_speed": 0}

    elif class_name == "fixed_wing":
        fundamental = rng.uniform(50, 150)
        sig = _make_multi_rotor_signal(
            n_steps, dt, n_rotors=1, fundamental=fundamental,
            n_harmonics=8, rotor_freq_spread=0.0,
            harmonic_decay_power=0.7, source_level_dB=90.0, seed=seed,
        )
        params = {"fundamental": fundamental, "n_rotors": 1,
                  "altitude_range": (30, 150), "min_speed": 10}

    elif class_name == "bird":
        wing_beat_freq = rng.uniform(3, 12)
        sig = _make_bird_signal(
            n_steps, dt, wing_beat_freq=wing_beat_freq,
            pulse_width_ms=rng.uniform(10, 30),
            source_level_dB=75.0, seed=seed,
        )
        params = {"wing_beat_freq": wing_beat_freq,
                  "altitude_range": (5, 200), "min_speed": 5}

    elif class_name == "ground_vehicle":
        engine_fund = rng.uniform(25, 60)
        sig = _make_ground_vehicle_signal(
            n_steps, dt, engine_fundamental=engine_fund,
            source_level_dB=80.0, seed=seed,
        )
        params = {"engine_fundamental": engine_fund,
                  "altitude_range": (0, 0), "min_speed": 2}

    else:  # unknown — random noise
        sig = rng.standard_normal(n_steps) * _P_REF * 10
        params = {"altitude_range": (0, 200), "min_speed": 0}

    return sig, params


def generate_classification_dataset(
    n_samples_per_class: int = 200,
    dt: float = 1.0 / 4000,
    window_duration: float = 0.5,
    sound_speed: float = 343.0,
    seed: int = 42,
) -> dict:
    """Generate a full classification training dataset.

    Returns dict with:
        signals: list of 1D arrays (beamformed traces)
        labels: list of int class indices
        class_names: list of str
        params: list of dicts with sample metadata
        snr_dbs: list of float SNR values
    """
    rng = np.random.default_rng(seed)
    n_steps = int(window_duration / dt)

    # Simple microphone array (L-shaped, 9 elements).
    from acoustic_sim.receivers_3d import create_receiver_l_shaped_3d
    mics = create_receiver_l_shaped_3d(5, 5, spacing=0.3,
                                        origin_x=0.0, origin_y=0.0, z=0.0)

    signals = []
    labels = []
    snr_dbs = []
    params_list = []

    for cls_name in SOURCE_CLASSES:
        cls_idx = CLASS_TO_IDX[cls_name]
        for i in range(n_samples_per_class):
            sig, params = generate_source_signal(cls_name, n_steps, dt, rng)

            # Random source position.
            horiz_dist = rng.uniform(50, 400)
            bearing = rng.uniform(0, 2 * math.pi)
            alt_lo, alt_hi = params.get("altitude_range", (20, 100))
            altitude = rng.uniform(max(alt_lo, 0), max(alt_hi, 0.1))
            if cls_name == "ground_vehicle":
                altitude = 0.0

            sx = horiz_dist * math.cos(bearing)
            sy = horiz_dist * math.sin(bearing)

            # Speed.
            min_speed = params.get("min_speed", 0)
            speed = rng.uniform(max(min_speed, 5), 25)
            if cls_name == "fixed_wing":
                speed = max(speed, 10)
            heading = rng.uniform(0, 2 * math.pi)

            src = MovingSource3D(
                x0=sx, y0=sy, z0=altitude,
                x1=sx + speed * window_duration * math.cos(heading),
                y1=sy + speed * window_duration * math.sin(heading),
                z1=altitude,
                speed=speed, signal=sig,
            )

            # Forward model.
            traces = simulate_3d_traces(
                src, mics, dt, n_steps, sound_speed,
                air_absorption=0.005,
            )

            # Add noise at random SNR.
            target_snr_dB = rng.uniform(-5, 30)
            sig_power = np.mean(traces ** 2)
            noise_power = sig_power / max(10 ** (target_snr_dB / 10), 1e-30)
            noise_std = math.sqrt(max(noise_power, 1e-30))
            noise = rng.standard_normal(traces.shape) * noise_std
            noisy_traces = traces + noise

            # Simple beamforming: sum all channels (delay-and-sum at origin).
            beamformed = np.mean(noisy_traces, axis=0)

            signals.append(beamformed)
            labels.append(cls_idx)
            snr_dbs.append(target_snr_dB)
            params_list.append({
                "class": cls_name,
                "distance": horiz_dist,
                "altitude": altitude,
                "speed": speed,
                "snr_dB": target_snr_dB,
            })

    return {
        "signals": signals,
        "labels": labels,
        "class_names": SOURCE_CLASSES,
        "params": params_list,
        "snr_dbs": snr_dbs,
        "dt": dt,
    }


def generate_maneuver_dataset(
    n_samples_per_class: int = 400,
    window_size: int = 20,
    dt_tracker: float = 0.1,
    seed: int = 42,
) -> dict:
    """Generate labeled track segments for maneuver detection.

    Each segment is N=window_size consecutive tracker outputs.
    Features: [x, y, z, vx, vy, vz] normalized.

    Returns dict with:
        features: (N_total, window_size, 6) array
        labels: (N_total,) int array
    """
    rng = np.random.default_rng(seed)
    features = []
    labels = []

    for cls_name in MANEUVER_CLASSES:
        cls_idx = MANEUVER_TO_IDX[cls_name]
        for _ in range(n_samples_per_class):
            segment = _generate_maneuver_segment(
                cls_name, window_size, dt_tracker, rng,
            )
            features.append(segment)
            labels.append(cls_idx)

    return {
        "features": np.array(features),
        "labels": np.array(labels),
        "class_names": MANEUVER_CLASSES,
    }


def _generate_maneuver_segment(
    maneuver_type: str,
    window_size: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one maneuver segment: (window_size, 6)."""
    n = window_size
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))

    speed = rng.uniform(5, 25)
    heading = rng.uniform(0, 2 * math.pi)
    alt = rng.uniform(20, 100)
    x0, y0 = rng.uniform(-200, 200), rng.uniform(-200, 200)

    if maneuver_type == "steady":
        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        for i in range(n):
            t = i * dt
            positions[i] = [x0 + vx * t, y0 + vy * t, alt]
            velocities[i] = [vx, vy, 0]

    elif maneuver_type == "turning":
        radius = rng.uniform(30, 100)
        omega = speed / radius
        for i in range(n):
            t = i * dt
            theta = heading + omega * t
            positions[i] = [x0 + radius * math.cos(theta),
                           y0 + radius * math.sin(theta), alt]
            velocities[i] = [-speed * math.sin(theta),
                            speed * math.cos(theta), 0]

    elif maneuver_type == "accelerating":
        accel = rng.uniform(2, 8) * rng.choice([-1, 1])
        vx0 = speed * math.cos(heading)
        vy0 = speed * math.sin(heading)
        for i in range(n):
            t = i * dt
            s = speed + accel * t
            scale = s / max(speed, 1e-6)
            velocities[i] = [vx0 * scale, vy0 * scale, 0]
            positions[i] = [x0 + vx0 * t + 0.5 * accel * t ** 2 * math.cos(heading),
                           y0 + vy0 * t + 0.5 * accel * t ** 2 * math.sin(heading),
                           alt]

    elif maneuver_type == "diving":
        vz = rng.uniform(-10, -2)
        vx = speed * math.cos(heading) * 0.5
        vy = speed * math.sin(heading) * 0.5
        for i in range(n):
            t = i * dt
            positions[i] = [x0 + vx * t, y0 + vy * t, max(alt + vz * t, 0)]
            velocities[i] = [vx, vy, vz]

    elif maneuver_type == "evasive":
        for i in range(n):
            t = i * dt
            heading_i = heading + rng.normal(0, 0.5)
            speed_i = max(speed + rng.normal(0, 3), 1)
            vz_i = rng.normal(0, 2)
            velocities[i] = [speed_i * math.cos(heading_i),
                            speed_i * math.sin(heading_i), vz_i]
            if i > 0:
                positions[i] = positions[i - 1] + velocities[i] * dt
            else:
                positions[i] = [x0, y0, alt]

    elif maneuver_type == "hovering":
        noise_std = 0.3
        for i in range(n):
            positions[i] = [x0 + rng.normal(0, noise_std),
                           y0 + rng.normal(0, noise_std),
                           alt + rng.normal(0, noise_std * 0.5)]
            velocities[i] = [rng.normal(0, 0.2), rng.normal(0, 0.2),
                            rng.normal(0, 0.1)]

    # Normalize: subtract mean position.
    mean_pos = np.mean(positions, axis=0)
    positions -= mean_pos

    # Add tracker noise.
    pos_noise_std = rng.uniform(1.0, 5.0)
    vel_noise_std = rng.uniform(0.5, 2.0)
    positions += rng.normal(0, pos_noise_std, positions.shape)
    velocities += rng.normal(0, vel_noise_std, velocities.shape)

    return np.hstack([positions, velocities])  # (n, 6)

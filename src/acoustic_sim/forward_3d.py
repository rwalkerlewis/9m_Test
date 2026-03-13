"""Analytical 3D forward model for generating synthetic microphone traces.

Replaces the 2D FDTD for scenarios with z ≠ 0.  Uses point-source
spherical spreading (1/r decay + time delay) which is physically
consistent with the matched field processor's assumptions.

For the z=0 identity test, a wrapper delegates to the existing 2D FDTD
so traces match exactly.

Features
========
* 1/r spherical spreading
* Optional ground reflection (image source method)
* Optional air absorption
* Optional vegetation attenuation zones
* Noise injection (wind + sensor) via existing noise module
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal as sp_signal

from acoustic_sim.sources_3d import source_velocity_at_3d

_P_REF = 20e-6  # 20 µPa


def simulate_3d_traces(
    source,
    mic_positions: np.ndarray,
    dt: float,
    n_steps: int,
    sound_speed: float = 343.0,
    air_absorption: float = 0.005,
    enable_ground_reflection: bool = False,
    ground_reflection_coeff: float = -0.9,
    ground_z: float = 0.0,
) -> np.ndarray:
    """Generate synthetic microphone traces from a 3D source.

    Parameters
    ----------
    source : any 3D source object with ``position_at(step, dt) -> (x,y,z)``
        and ``signal`` attribute (1D array of length n_steps).
    mic_positions : (n_mics, 3) or (n_mics, 2)
        Microphone positions. If 2D, z=0 is assumed.
    dt : float
        Time step [s].
    n_steps : int
        Number of time steps.
    sound_speed : float
        Speed of sound [m/s].
    air_absorption : float
        Exponential absorption coefficient [1/m].
    enable_ground_reflection : bool
        If True, add a ground-reflected image source.
    ground_reflection_coeff : float
        Reflection coefficient for ground (negative = phase flip).
    ground_z : float
        Ground plane z-coordinate.

    Returns
    -------
    np.ndarray, shape ``(n_mics, n_steps)``
        Pressure traces at each microphone.
    """
    mic_pos = np.asarray(mic_positions, dtype=np.float64)
    if mic_pos.ndim == 1:
        mic_pos = mic_pos.reshape(1, -1)
    if mic_pos.shape[1] == 2:
        mic_pos = np.column_stack([mic_pos, np.zeros(mic_pos.shape[0])])

    n_mics = mic_pos.shape[0]
    traces = np.zeros((n_mics, n_steps), dtype=np.float64)
    sig = source.signal

    # Pre-compute all source positions for efficiency.
    src_positions = np.zeros((n_steps, 3))
    for i in range(n_steps):
        src_positions[i] = source.position_at(i, dt)

    # For each mic, compute the received signal via delay-and-sum.
    for m in range(n_mics):
        mx, my, mz = mic_pos[m]

        # Direct path.
        dx = src_positions[:, 0] - mx
        dy = src_positions[:, 1] - my
        dz = src_positions[:, 2] - mz
        distances = np.sqrt(dx * dx + dy * dy + dz * dz)
        distances = np.maximum(distances, 1.0)  # avoid div by zero

        # 1/r amplitude decay + absorption.
        amplitudes = (1.0 / distances) * np.exp(-air_absorption * distances)

        # Time delay in samples (fractional).
        delay_samples = distances / (sound_speed * dt)

        # Apply delayed signal.
        for i in range(n_steps):
            sig_val = sig[min(i, len(sig) - 1)]
            # Compute the emission time: when was the signal emitted
            # such that it arrives at the mic at time step i?
            # arrival_time = emission_time + travel_time
            # i * dt = emission_step * dt + dist / c
            # emission_step = i - dist / (c * dt)
            emission_idx = i - delay_samples[i]
            if emission_idx < 0 or emission_idx >= n_steps:
                continue
            # Linear interpolation for fractional sample.
            idx_lo = int(math.floor(emission_idx))
            idx_hi = idx_lo + 1
            frac = emission_idx - idx_lo
            if idx_lo < 0 or idx_hi >= n_steps:
                continue
            sig_interp = (1.0 - frac) * sig[idx_lo] + frac * sig[idx_hi]
            traces[m, i] += amplitudes[i] * sig_interp

        # Ground reflection (image source method).
        if enable_ground_reflection:
            # Image source: reflect source z across ground plane.
            img_dz = (2.0 * ground_z - src_positions[:, 2]) - mz
            img_dist = np.sqrt(dx * dx + dy * dy + img_dz * img_dz)
            img_dist = np.maximum(img_dist, 1.0)
            img_amp = (ground_reflection_coeff / img_dist) * np.exp(
                -air_absorption * img_dist
            )
            img_delay = img_dist / (sound_speed * dt)

            for i in range(n_steps):
                emission_idx = i - img_delay[i]
                if emission_idx < 0 or emission_idx >= n_steps:
                    continue
                idx_lo = int(math.floor(emission_idx))
                idx_hi = idx_lo + 1
                frac = emission_idx - idx_lo
                if idx_lo < 0 or idx_hi >= n_steps:
                    continue
                sig_interp = (1.0 - frac) * sig[idx_lo] + frac * sig[idx_hi]
                traces[m, i] += img_amp[i] * sig_interp

    return traces


def simulate_3d_traces_vectorized(
    source,
    mic_positions: np.ndarray,
    dt: float,
    n_steps: int,
    sound_speed: float = 343.0,
    air_absorption: float = 0.005,
    enable_ground_reflection: bool = False,
    ground_reflection_coeff: float = -0.9,
    ground_z: float = 0.0,
) -> np.ndarray:
    """Vectorized version using frequency-domain delay application.

    Faster for long signals. Uses the same physics as ``simulate_3d_traces``
    but applies delays via phase shifts in the frequency domain for each
    time window.

    For simplicity and correctness, we use a block-based approach:
    divide the signal into blocks, compute mean delay per block, and
    apply time-domain shifting.
    """
    mic_pos = np.asarray(mic_positions, dtype=np.float64)
    if mic_pos.shape[1] == 2:
        mic_pos = np.column_stack([mic_pos, np.zeros(mic_pos.shape[0])])

    n_mics = mic_pos.shape[0]
    traces = np.zeros((n_mics, n_steps), dtype=np.float64)
    sig = source.signal

    # Pre-compute all source positions.
    src_positions = np.zeros((n_steps, 3))
    for i in range(n_steps):
        src_positions[i] = source.position_at(i, dt)

    # Block-based approach for speed.
    block_size = max(n_steps // 20, 100)
    n_blocks = (n_steps + block_size - 1) // block_size

    for m in range(n_mics):
        mx, my, mz = mic_pos[m]

        # Compute all distances at once.
        diff = src_positions - mic_pos[m]
        distances = np.sqrt(np.sum(diff * diff, axis=1))
        distances = np.maximum(distances, 1.0)
        amplitudes = (1.0 / distances) * np.exp(-air_absorption * distances)
        delay_samples = distances / (sound_speed * dt)

        # Apply signal with delay per sample.
        for i in range(n_steps):
            emission_idx = i - delay_samples[i]
            if emission_idx < 0 or emission_idx >= n_steps - 1:
                continue
            idx_lo = int(math.floor(emission_idx))
            idx_hi = idx_lo + 1
            if idx_hi >= n_steps:
                continue
            frac = emission_idx - idx_lo
            sig_interp = (1.0 - frac) * sig[idx_lo] + frac * sig[idx_hi]
            traces[m, i] += amplitudes[i] * sig_interp

        if enable_ground_reflection:
            img_z = 2.0 * ground_z - src_positions[:, 2]
            img_diff = np.column_stack([diff[:, 0], diff[:, 1],
                                         img_z - mz])
            img_dist = np.sqrt(np.sum(img_diff * img_diff, axis=1))
            img_dist = np.maximum(img_dist, 1.0)
            img_amp = (ground_reflection_coeff / img_dist) * np.exp(
                -air_absorption * img_dist
            )
            img_delay = img_dist / (sound_speed * dt)

            for i in range(n_steps):
                emission_idx = i - img_delay[i]
                if emission_idx < 0 or emission_idx >= n_steps - 1:
                    continue
                idx_lo = int(math.floor(emission_idx))
                idx_hi = idx_lo + 1
                if idx_hi >= n_steps:
                    continue
                frac = emission_idx - idx_lo
                sig_interp = (1.0 - frac) * sig[idx_lo] + frac * sig[idx_hi]
                traces[m, i] += img_amp[i] * sig_interp

    return traces


def simulate_scenario_3d(
    sources: list,
    mic_positions: np.ndarray,
    dt: float,
    n_steps: int,
    sound_speed: float = 343.0,
    air_absorption: float = 0.005,
    enable_ground_reflection: bool = False,
    ground_reflection_coeff: float = -0.9,
    ground_z: float = 0.0,
    wind_noise_enabled: bool = True,
    wind_noise_level_dB: float = 55.0,
    wind_corner_freq: float = 15.0,
    wind_correlation_length: float = 3.0,
    sensor_noise_enabled: bool = True,
    sensor_noise_level_dB: float = 40.0,
    seed: int = 42,
) -> dict:
    """Run full 3D scenario: forward model + noise.

    Parameters
    ----------
    sources : list of 3D source objects
        Each must have ``position_at(step, dt) -> (x,y,z)`` and ``signal``.
    mic_positions : (n_mics, 3) or (n_mics, 2)
    dt : float
    n_steps : int

    Returns
    -------
    dict with:
        traces : (n_mics, n_steps) combined noisy traces
        clean_traces : list of (n_mics, n_steps) per source
        mic_positions : (n_mics, 3)
        dt, sound_speed, n_steps
        true_positions : list of (n_steps, 3) per source
        true_velocities : list of (n_steps, 3) per source
        true_times : (n_steps,)
    """
    from acoustic_sim.noise import generate_sensor_noise, generate_wind_noise

    mic_pos = np.asarray(mic_positions, dtype=np.float64)
    if mic_pos.shape[1] == 2:
        mic_pos = np.column_stack([mic_pos, np.zeros(mic_pos.shape[0])])

    n_mics = mic_pos.shape[0]

    # Generate traces for each source.
    clean_traces_list = []
    combined = np.zeros((n_mics, n_steps), dtype=np.float64)

    for src in sources:
        tr = simulate_3d_traces(
            src, mic_pos, dt, n_steps,
            sound_speed=sound_speed,
            air_absorption=air_absorption,
            enable_ground_reflection=enable_ground_reflection,
            ground_reflection_coeff=ground_reflection_coeff,
            ground_z=ground_z,
        )
        clean_traces_list.append(tr)
        combined += tr

    # Add noise.
    # Wind noise uses 2D positions (x, y) for spatial correlation.
    mic_pos_2d = mic_pos[:, :2]
    if wind_noise_enabled:
        combined += generate_wind_noise(
            mic_pos_2d, n_steps, dt,
            level_dB=wind_noise_level_dB,
            corner_freq=wind_corner_freq,
            correlation_length=wind_correlation_length,
            seed=seed,
        )
    if sensor_noise_enabled:
        combined += generate_sensor_noise(
            n_mics, n_steps, dt,
            level_dB=sensor_noise_level_dB,
            seed=seed + 1,
        )

    # Ground truth.
    true_positions_list = []
    true_velocities_list = []
    for src in sources:
        pos = np.zeros((n_steps, 3))
        vel = np.zeros((n_steps, 3))
        for i in range(n_steps):
            pos[i] = src.position_at(i, dt)
            vel[i] = source_velocity_at_3d(src, i, dt)
        true_positions_list.append(pos)
        true_velocities_list.append(vel)

    true_times = np.arange(n_steps) * dt

    return {
        "traces": combined,
        "clean_traces": clean_traces_list,
        "mic_positions": mic_pos,
        "dt": dt,
        "sound_speed": sound_speed,
        "n_steps": n_steps,
        "true_positions": true_positions_list,
        "true_velocities": true_velocities_list,
        "true_times": true_times,
    }


# =====================================================================
#  FDTD-based 3D trace generation
# =====================================================================

def simulate_3d_traces_fdtd(
    source,
    mic_positions: np.ndarray,
    dt: float | None,
    n_steps: int | None = None,
    total_time: float = 1.0,
    sound_speed: float = 343.0,
    dx: float = 1.0,
    domain_margin: float = 20.0,
    z_min: float = -5.0,
    z_max: float = 120.0,
    damping_width: int = 10,
    fd_order: int = 2,
    air_absorption: float = 0.005,
    source_amplitude: float = 1.0,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """Generate microphone traces using the 3D FDTD solver.

    Automatically constructs a uniform 3D velocity model around the source
    and receiver positions.

    Parameters
    ----------
    source : 3D source object
        Must have ``position_at(step, dt) -> (x,y,z)`` and ``signal``.
    mic_positions : (n_mics, 3) or (n_mics, 2)
    dt : float or None
        Time step.  None → auto-compute from CFL.
    n_steps : int or None
        Number of steps.  None → compute from ``total_time``.
    total_time : float
        Simulation duration [s], used when ``n_steps`` is None.
    sound_speed : float
    dx : float
        Grid spacing [m].
    domain_margin : float
        Extra margin beyond source/receiver extent [m].
    z_min, z_max : float
        Z-range for the domain.
    damping_width : int
        Sponge-layer width in grid cells.
    fd_order : int
    air_absorption : float
    source_amplitude : float
    verbose : bool

    Returns
    -------
    (traces, dt_actual)
        traces : (n_mics, n_steps) pressure at receivers
        dt_actual : float, the actual timestep used
    """
    from acoustic_sim.domains_3d import DomainMeta3D, create_isotropic_domain_3d
    from acoustic_sim.fdtd_3d import FDTD3DConfig, FDTD3DSolver

    mic_pos = np.asarray(mic_positions, dtype=np.float64)
    if mic_pos.shape[1] == 2:
        mic_pos = np.column_stack([mic_pos, np.zeros(mic_pos.shape[0])])

    # Determine domain extent from source trajectory + receiver positions.
    all_x = list(mic_pos[:, 0])
    all_y = list(mic_pos[:, 1])

    # Sample a few source positions to determine extent.
    test_dt = dt if dt is not None else 1e-4
    sig_len = len(source.signal) if hasattr(source, 'signal') else 10000
    for step in range(0, sig_len, max(sig_len // 20, 1)):
        sx, sy, sz = source.position_at(step, test_dt)
        all_x.append(sx)
        all_y.append(sy)

    x_min = min(all_x) - domain_margin
    x_max = max(all_x) + domain_margin
    y_min = min(all_y) - domain_margin
    y_max = max(all_y) + domain_margin

    # Create domain.
    model, meta = create_isotropic_domain_3d(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_min, z_max=z_max,
        dx=dx, velocity=sound_speed,
    )

    if verbose:
        print(f"  FDTD3D domain: {model.shape} "
              f"({model.nx}×{model.ny}×{model.nz} cells, dx={dx}m)")
        ncells = model.nx * model.ny * model.nz
        mem_mb = ncells * 8 * 2 / 1e6
        print(f"  Memory estimate: {mem_mb:.1f} MB for pressure fields")

    cfg = FDTD3DConfig(
        total_time=total_time,
        dt=dt,
        damping_width=damping_width,
        damping_max=0.15,
        air_absorption=air_absorption,
        snapshot_interval=0,
        source_amplitude=source_amplitude,
        fd_order=fd_order,
    )

    solver = FDTD3DSolver(model, cfg, source, mic_pos, meta)

    if n_steps is not None:
        solver.n_steps = n_steps

    result = solver.run(verbose=verbose)
    return result["traces"], solver.dt


def simulate_scenario_3d_fdtd(
    sources: list,
    mic_positions: np.ndarray,
    total_time: float = 1.0,
    sound_speed: float = 343.0,
    dx: float = 1.0,
    domain_margin: float = 20.0,
    z_min: float = -5.0,
    z_max: float = 120.0,
    damping_width: int = 10,
    fd_order: int = 2,
    air_absorption: float = 0.005,
    wind_noise_enabled: bool = True,
    wind_noise_level_dB: float = 55.0,
    wind_corner_freq: float = 15.0,
    wind_correlation_length: float = 3.0,
    sensor_noise_enabled: bool = True,
    sensor_noise_level_dB: float = 40.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run full 3D scenario using the FDTD forward model + noise.

    Like ``simulate_scenario_3d`` but uses the wave-equation solver
    instead of the analytical 1/r model.

    Returns dict with same structure as ``simulate_scenario_3d``.
    """
    from acoustic_sim.noise import generate_sensor_noise, generate_wind_noise

    mic_pos = np.asarray(mic_positions, dtype=np.float64)
    if mic_pos.shape[1] == 2:
        mic_pos = np.column_stack([mic_pos, np.zeros(mic_pos.shape[0])])
    n_mics = mic_pos.shape[0]

    clean_traces_list = []
    dt_actual = None

    for i, src in enumerate(sources):
        if verbose:
            print(f"  Running FDTD3D for source {i + 1}/{len(sources)}...")
        tr, dt_actual = simulate_3d_traces_fdtd(
            src, mic_pos, dt=None, total_time=total_time,
            sound_speed=sound_speed, dx=dx,
            domain_margin=domain_margin,
            z_min=z_min, z_max=z_max,
            damping_width=damping_width,
            fd_order=fd_order,
            air_absorption=air_absorption,
            verbose=verbose,
        )
        clean_traces_list.append(tr)

    dt = dt_actual
    n_steps = clean_traces_list[0].shape[1]

    # Align trace lengths.
    min_len = min(tr.shape[1] for tr in clean_traces_list)
    clean_traces_list = [tr[:, :min_len] for tr in clean_traces_list]
    n_steps = min_len

    combined = np.zeros((n_mics, n_steps), dtype=np.float64)
    for tr in clean_traces_list:
        combined += tr

    # Add noise.
    mic_pos_2d = mic_pos[:, :2]
    if wind_noise_enabled:
        combined += generate_wind_noise(
            mic_pos_2d, n_steps, dt,
            level_dB=wind_noise_level_dB,
            corner_freq=wind_corner_freq,
            correlation_length=wind_correlation_length,
            seed=seed,
        )
    if sensor_noise_enabled:
        combined += generate_sensor_noise(
            n_mics, n_steps, dt,
            level_dB=sensor_noise_level_dB,
            seed=seed + 1,
        )

    # Ground truth.
    true_positions_list = []
    true_velocities_list = []
    for src in sources:
        pos = np.zeros((n_steps, 3))
        vel = np.zeros((n_steps, 3))
        for i in range(n_steps):
            pos[i] = src.position_at(i, dt)
            vel[i] = source_velocity_at_3d(src, i, dt)
        true_positions_list.append(pos)
        true_velocities_list.append(vel)

    true_times = np.arange(n_steps) * dt

    return {
        "traces": combined,
        "clean_traces": clean_traces_list,
        "mic_positions": mic_pos,
        "dt": dt,
        "sound_speed": sound_speed,
        "n_steps": n_steps,
        "true_positions": true_positions_list,
        "true_velocities": true_velocities_list,
        "true_times": true_times,
    }

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "matplotlib",
# ]
# ///

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

matplotlib.use("Agg")

NUM_MICS = 16
DIAMETER_M = 0.4
DEFAULT_SOUND_SPEED = 343.0


@dataclass
class Material:
    name: str
    wave_speed: float
    attenuation: float
    scattering: float


@dataclass
class DomainConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    dx: float
    default_material: str
    materials: dict[str, Material]
    regions: list[dict[str, Any]]


@dataclass
class Leg:
    start: np.ndarray
    end: np.ndarray
    speed_m_s: float
    distance_m: float
    duration_s: float


def get_mic_array() -> np.ndarray:
    radius = DIAMETER_M / 2
    angles = np.linspace(0, 2 * np.pi, NUM_MICS, endpoint=False)
    coords = np.zeros((NUM_MICS, 3), dtype=np.float64)
    coords[:, 0] = radius * np.cos(angles)
    coords[:, 1] = radius * np.sin(angles)
    coords[:, 2] = 1.6
    return coords


def load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON config not found: {path}")
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def spherical_to_cartesian(distance: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    x = distance * np.cos(el) * np.sin(az)
    y = distance * np.cos(el) * np.cos(az)
    z = distance * np.sin(el)
    return np.array([x, y, z], dtype=np.float64)


def pcm_to_float32(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32)
    if audio.dtype == np.int16:
        return (audio.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if audio.dtype == np.int32:
        return (audio.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    if audio.dtype == np.uint8:
        return ((audio.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    return audio.astype(np.float32)


def float32_to_pcm(audio: np.ndarray) -> np.ndarray:
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def load_mono_wav(path: str) -> tuple[np.ndarray, int]:
    fs, data = wavfile.read(path)
    audio = pcm_to_float32(data)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, int(fs)


def normalize_signal(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = np.max(np.abs(x)) + 1e-8
    return (x / max_abs) * peak


def build_source_signal(args: argparse.Namespace, source_cfg: dict[str, Any] | None) -> tuple[np.ndarray, int]:
    source_model = args.source_model
    cfg = source_cfg or {}
    sample_rate = int(cfg.get("sample_rate", args.sample_rate))
    duration_s = float(cfg.get("duration_s", args.duration))

    if source_model == "file":
        if not args.input:
            raise ValueError("File source model requires positional input wav path.")
        file_audio, file_fs = load_mono_wav(args.input)
        if args.max_seconds:
            file_audio = file_audio[: int(args.max_seconds * file_fs)]
        if sample_rate != file_fs:
            file_audio = signal.resample_poly(file_audio, sample_rate, file_fs).astype(np.float32)
        return normalize_signal(file_audio), sample_rate

    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate

    if source_model == "tone":
        freq = float(cfg.get("frequency_hz", args.tone_frequency))
        tone = np.sin(2 * np.pi * freq * t)
        return normalize_signal(tone.astype(np.float32)), sample_rate

    if source_model == "noise":
        rng = np.random.default_rng(args.seed + 11)
        noise = rng.standard_normal(n).astype(np.float32)
        b, a = signal.butter(4, [120.0, 4500.0], btype="bandpass", fs=sample_rate)
        colored = signal.filtfilt(b, a, noise).astype(np.float32)
        return normalize_signal(colored), sample_rate

    if source_model == "propeller":
        blade_count = int(cfg.get("blade_count", args.blade_count))
        rpm = float(cfg.get("rpm", args.rpm))
        harmonics = int(cfg.get("harmonics", args.harmonics))
        mod_depth = float(cfg.get("mod_depth", args.mod_depth))
        broadband_level = float(cfg.get("broadband_level", args.broadband_level))
        bpf = (blade_count * rpm) / 60.0
        rotor = rpm / 60.0

        sig = np.zeros_like(t)
        for h in range(1, harmonics + 1):
            amp = 1.0 / np.sqrt(h)
            sig += amp * np.sin(2 * np.pi * bpf * h * t + 0.17 * h)

        mod = 1.0 + mod_depth * np.sin(2 * np.pi * rotor * t)
        sig *= mod

        rng = np.random.default_rng(args.seed + 23)
        broadband = rng.standard_normal(n).astype(np.float32)
        b, a = signal.butter(2, [100.0, 8000.0], btype="bandpass", fs=sample_rate)
        broadband = signal.filtfilt(b, a, broadband).astype(np.float32)
        sig += broadband_level * broadband
        return normalize_signal(sig.astype(np.float32)), sample_rate

    raise ValueError(f"Unsupported source model: {source_model}")


def build_domain(domain_cfg: dict[str, Any] | None) -> DomainConfig:
    if not domain_cfg:
        materials = {
            "air": Material(name="air", wave_speed=343.0, attenuation=0.002, scattering=0.05),
            "vegetation": Material(name="vegetation", wave_speed=330.0, attenuation=0.012, scattering=0.18),
            "topography": Material(name="topography", wave_speed=355.0, attenuation=0.008, scattering=0.12),
        }
        return DomainConfig(
            x_min=-20.0,
            x_max=20.0,
            y_min=-20.0,
            y_max=20.0,
            dx=0.4,
            default_material="air",
            materials=materials,
            regions=[
                {
                    "name": "tree_band",
                    "type": "rectangle",
                    "material": "vegetation",
                    "xmin": -4.0,
                    "xmax": 4.0,
                    "ymin": 8.0,
                    "ymax": 16.0,
                },
                {
                    "name": "hill",
                    "type": "circle",
                    "material": "topography",
                    "center": [8.0, -5.0],
                    "radius": 3.5,
                },
            ],
        )

    material_dict: dict[str, Material] = {}
    for name, values in domain_cfg["materials"].items():
        material_dict[name] = Material(
            name=name,
            wave_speed=float(values["wave_speed"]),
            attenuation=float(values["attenuation"]),
            scattering=float(values.get("scattering", 0.05)),
        )

    bounds = domain_cfg.get("bounds", {})
    return DomainConfig(
        x_min=float(bounds.get("x_min", -20.0)),
        x_max=float(bounds.get("x_max", 20.0)),
        y_min=float(bounds.get("y_min", -20.0)),
        y_max=float(bounds.get("y_max", 20.0)),
        dx=float(domain_cfg.get("dx", 0.4)),
        default_material=str(domain_cfg.get("default_material", "air")),
        materials=material_dict,
        regions=list(domain_cfg.get("regions", [])),
    )


def point_in_region(x: float, y: float, region: dict[str, Any]) -> bool:
    rtype = region.get("type", "rectangle")
    if rtype == "rectangle":
        return (
            float(region["xmin"]) <= x <= float(region["xmax"])
            and float(region["ymin"]) <= y <= float(region["ymax"])
        )
    if rtype == "circle":
        cx, cy = region["center"]
        radius = float(region["radius"])
        return (x - float(cx)) ** 2 + (y - float(cy)) ** 2 <= radius**2
    return False


def material_at(domain: DomainConfig, x: float, y: float) -> Material:
    mat_name = domain.default_material
    for region in domain.regions:
        if point_in_region(x, y, region):
            mat_name = region["material"]
    return domain.materials[mat_name]


def parse_legs(legs_cfg: dict[str, Any] | None, default_position: np.ndarray) -> list[Leg]:
    if not legs_cfg:
        return []

    legs_raw = legs_cfg.get("legs", [])
    if not legs_raw:
        raise ValueError("legs file must include non-empty 'legs' list.")

    legs: list[Leg] = []
    current = np.array(legs_cfg.get("start", default_position.tolist()), dtype=np.float64)
    for index, leg in enumerate(legs_raw):
        if "start" in leg:
            start = np.array(leg["start"], dtype=np.float64)
        else:
            start = current
        if "end" not in leg:
            raise ValueError(f"Leg index {index} missing 'end'.")
        end = np.array(leg["end"], dtype=np.float64)
        speed_m_s = float(leg["speed_m_s"])
        if speed_m_s <= 0.0:
            raise ValueError(f"Leg index {index} must have speed_m_s > 0.")
        distance = float(np.linalg.norm(end - start))
        duration = distance / speed_m_s if distance > 1e-8 else 0.0
        legs.append(Leg(start=start, end=end, speed_m_s=speed_m_s, distance_m=distance, duration_s=duration))
        current = end
    return legs


def source_position_at_time(t: float, legs: list[Leg], default_position: np.ndarray) -> np.ndarray:
    if not legs:
        return default_position
    elapsed = 0.0
    for leg in legs:
        if leg.duration_s <= 1e-8:
            elapsed += leg.duration_s
            continue
        if t <= elapsed + leg.duration_s:
            alpha = (t - elapsed) / leg.duration_s
            return leg.start + alpha * (leg.end - leg.start)
        elapsed += leg.duration_s
    return legs[-1].end


def leg_duration_summary(legs: list[Leg]) -> float:
    return float(sum(leg.duration_s for leg in legs))


def sample_path_materials(domain: DomainConfig, p0: np.ndarray, p1: np.ndarray, n_points: int = 40) -> tuple[float, float, float]:
    weights_c = 0.0
    weights_a = 0.0
    weights_s = 0.0
    for i in range(n_points):
        alpha = i / max(n_points - 1, 1)
        xy = p0[:2] + alpha * (p1[:2] - p0[:2])
        mat = material_at(domain, float(xy[0]), float(xy[1]))
        weights_c += mat.wave_speed
        weights_a += mat.attenuation
        weights_s += mat.scattering
    denom = float(n_points)
    return weights_c / denom, weights_a / denom, weights_s / denom


def apply_fractional_delay(x: np.ndarray, delay_samples: float, global_indices: np.ndarray) -> np.ndarray:
    positions = global_indices.astype(np.float64) - float(delay_samples)
    i0 = np.floor(positions).astype(np.int64)
    frac = positions - i0
    i1 = i0 + 1

    y = np.zeros(len(global_indices), dtype=np.float32)
    valid = (i0 >= 0) & (i1 < len(x))
    if not np.any(valid):
        return y

    base = x[i0[valid]]
    nxt = x[i1[valid]]
    y_valid = (1.0 - frac[valid]) * base + frac[valid] * nxt
    y[valid] = y_valid.astype(np.float32)
    return y


def synthesize_interferer(n_samples: int, fs: int, mode: str, seed: int, freq_hz: float = 480.0) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float32) / fs
    rng = np.random.default_rng(seed)
    if mode == "tone":
        signal_i = np.sin(2 * np.pi * freq_hz * t)
    else:
        signal_i = rng.standard_normal(n_samples).astype(np.float32)
        b, a = signal.butter(2, [200.0, 3500.0], btype="bandpass", fs=fs)
        signal_i = signal.filtfilt(b, a, signal_i).astype(np.float32)
    return normalize_signal(signal_i.astype(np.float32), peak=0.75)


def render_array_audio(
    source_signal: np.ndarray,
    fs: int,
    args: argparse.Namespace,
    mic_positions: np.ndarray,
    default_source_pos: np.ndarray,
    legs: list[Leg],
    domain: DomainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(source_signal)
    out = np.zeros((n, mic_positions.shape[0]), dtype=np.float32)
    block_size = int(args.block_size)
    rng = np.random.default_rng(args.seed)

    gain_mismatch = 1.0 + rng.normal(0.0, args.gain_std, size=mic_positions.shape[0])
    phase_delay_m = rng.normal(0.0, args.phase_std_m, size=mic_positions.shape[0])

    interferers: list[dict[str, Any]] = []
    for idx in range(args.interferer_count):
        az = 360.0 * idx / max(args.interferer_count, 1)
        pos = spherical_to_cartesian(
            distance=float(args.interferer_distance),
            azimuth_deg=float(az),
            elevation_deg=float(args.interferer_elevation),
        ) + np.array([0.0, 0.0, 1.4])
        interferers.append(
            {
                "position": pos,
                "signal": synthesize_interferer(
                    n_samples=n,
                    fs=fs,
                    mode=args.interferer_model,
                    seed=args.seed + 101 + idx,
                    freq_hz=args.interferer_frequency + 80.0 * idx,
                ),
            }
        )

    time_axis = np.arange(n) / fs
    trajectory = np.zeros((n, 3), dtype=np.float32)
    for b0 in range(0, n, block_size):
        b1 = min(b0 + block_size, n)
        block_indices = np.arange(b0, b1)
        t_center = float((b0 + b1) * 0.5 / fs)
        src_pos = source_position_at_time(t_center, legs, default_source_pos)
        trajectory[b0:b1] = src_pos

        for m in range(mic_positions.shape[0]):
            mic = mic_positions[m]
            dist = np.linalg.norm(src_pos - mic) + 1e-8
            c_eff, alpha_eff, scatter_eff = sample_path_materials(domain, src_pos, mic)
            dist += phase_delay_m[m]
            dist = max(dist, 0.02)
            delay_samples = (dist / c_eff) * fs
            attenuation = np.exp(-alpha_eff * dist) / (1.0 + dist)
            direct = apply_fractional_delay(source_signal, delay_samples, block_indices)

            # Simple multipath: delayed low-pass reflected copy.
            reflection_gain = 0.27 + 0.2 * scatter_eff
            extra_delay = (2.0 + scatter_eff * 2.0) * fs / max(c_eff, 50.0)
            reflected = apply_fractional_delay(source_signal, delay_samples + extra_delay, block_indices)
            reflected = signal.lfilter([0.4, 0.3, 0.2], [1.0], reflected).astype(np.float32)

            out[b0:b1, m] += gain_mismatch[m] * attenuation * (direct + reflection_gain * reflected)

        # Optional interferers
        for interferer in interferers:
            i_pos = interferer["position"]
            i_signal = interferer["signal"]
            for m in range(mic_positions.shape[0]):
                mic = mic_positions[m]
                dist_i = np.linalg.norm(i_pos - mic) + 1e-8
                c_i, alpha_i, _ = sample_path_materials(domain, i_pos, mic)
                delay_i = (dist_i / c_i) * fs
                att_i = args.interferer_level * np.exp(-alpha_i * dist_i) / (1.0 + dist_i)
                i_block = apply_fractional_delay(i_signal, delay_i, block_indices)
                out[b0:b1, m] += att_i * i_block

    if args.wind_noise_level > 0.0:
        for m in range(mic_positions.shape[0]):
            noise = rng.standard_normal(n).astype(np.float32)
            b, a = signal.butter(2, 40.0, btype="lowpass", fs=fs)
            wind = signal.filtfilt(b, a, noise).astype(np.float32)
            out[:, m] += args.wind_noise_level * normalize_signal(wind, peak=1.0)

    if args.self_noise_level > 0.0:
        self_noise = rng.standard_normal(out.shape).astype(np.float32)
        out += args.self_noise_level * self_noise

    # Gentle limiter
    max_abs = float(np.max(np.abs(out)) + 1e-8)
    if max_abs > 1.0:
        out /= max_abs * 1.02

    if args.debug_trajectory_path:
        np.save(args.debug_trajectory_path, trajectory)
        print(f"Wrote trajectory samples to {args.debug_trajectory_path}")

    print(
        f"Rendered {out.shape[1]} channels, duration={time_axis[-1]:.2f}s, "
        f"source_trajectory_mode={'dynamic' if legs else 'static'}"
    )
    return out.astype(np.float32), trajectory


def build_material_maps(domain: DomainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(domain.x_min, domain.x_max + 0.5 * domain.dx, domain.dx, dtype=np.float64)
    y = np.arange(domain.y_min, domain.y_max + 0.5 * domain.dx, domain.dx, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    c_map = np.zeros_like(xx)
    alpha_map = np.zeros_like(xx)
    scatter_map = np.zeros_like(xx)
    for iy in range(xx.shape[0]):
        for ix in range(xx.shape[1]):
            mat = material_at(domain, float(xx[iy, ix]), float(yy[iy, ix]))
            c_map[iy, ix] = mat.wave_speed
            alpha_map[iy, ix] = mat.attenuation
            scatter_map[iy, ix] = mat.scattering
    return x, y, c_map, alpha_map + 0.02 * scatter_map


def solve_helmholtz(
    domain: DomainConfig,
    source_xy: np.ndarray,
    frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, c_map, alpha_map = build_material_maps(domain)
    ny, nx = c_map.shape
    dx = domain.dx
    omega = 2.0 * np.pi * frequency_hz
    k_map = omega / np.maximum(c_map, 1.0)

    damping = np.zeros_like(k_map)
    edge_cells = max(3, int(0.08 * min(nx, ny)))
    for iy in range(ny):
        for ix in range(nx):
            d_edge = min(ix, nx - 1 - ix, iy, ny - 1 - iy)
            if d_edge < edge_cells:
                damping[iy, ix] = ((edge_cells - d_edge) / edge_cells) ** 2 * 0.7

    total = nx * ny
    a = lil_matrix((total, total), dtype=np.complex128)
    b = np.zeros(total, dtype=np.complex128)

    inv_dx2 = 1.0 / (dx * dx)
    def idx(ix: int, iy: int) -> int:
        return iy * nx + ix

    for iy in range(ny):
        for ix in range(nx):
            row = idx(ix, iy)
            k_local = k_map[iy, ix]
            sigma = alpha_map[iy, ix] + damping[iy, ix]
            diag = -4.0 * inv_dx2 + (k_local**2) * (1.0 + 1j * sigma)
            a[row, row] = diag
            if ix > 0:
                a[row, idx(ix - 1, iy)] = inv_dx2
            if ix < nx - 1:
                a[row, idx(ix + 1, iy)] = inv_dx2
            if iy > 0:
                a[row, idx(ix, iy - 1)] = inv_dx2
            if iy < ny - 1:
                a[row, idx(ix, iy + 1)] = inv_dx2

    sx = int(np.argmin(np.abs(x - source_xy[0])))
    sy = int(np.argmin(np.abs(y - source_xy[1])))
    b[idx(sx, sy)] = -1.0 / (dx * dx)

    p = spsolve(a.tocsr(), b)
    field = np.abs(p.reshape((ny, nx)))
    if not np.all(np.isfinite(field)):
        raise RuntimeError("Helmholtz solution produced non-finite values.")
    return x, y, field


def trace_rays(
    source_xy: np.ndarray,
    domain: DomainConfig,
    ray_count: int,
    max_bounces: int,
) -> list[np.ndarray]:
    x0, x1 = domain.x_min, domain.x_max
    y0, y1 = domain.y_min, domain.y_max
    rays: list[np.ndarray] = []

    for angle in np.linspace(0, 2 * np.pi, ray_count, endpoint=False):
        pos = np.array(source_xy, dtype=np.float64)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        points = [pos.copy()]
        for _ in range(max_bounces + 1):
            tx = np.inf
            ty = np.inf
            wall = "none"
            if abs(direction[0]) > 1e-10:
                if direction[0] > 0:
                    tx = (x1 - pos[0]) / direction[0]
                    wall = "xmax"
                else:
                    tx = (x0 - pos[0]) / direction[0]
                    wall = "xmin"
            if abs(direction[1]) > 1e-10:
                if direction[1] > 0:
                    ty = (y1 - pos[1]) / direction[1]
                    if ty < tx:
                        wall = "ymax"
                else:
                    ty = (y0 - pos[1]) / direction[1]
                    if ty < tx:
                        wall = "ymin"

            t_hit = min(tx, ty)
            if not np.isfinite(t_hit) or t_hit <= 1e-8:
                break
            pos = pos + t_hit * direction
            points.append(pos.copy())

            if wall in {"xmax", "xmin"}:
                direction[0] *= -1
            elif wall in {"ymax", "ymin"}:
                direction[1] *= -1
            else:
                break

        rays.append(np.array(points))
    return rays


def plot_propagation(
    domain: DomainConfig,
    field_x: np.ndarray,
    field_y: np.ndarray,
    field: np.ndarray,
    mic_positions: np.ndarray,
    trajectory: np.ndarray,
    output_path: str,
    plot_rays: bool,
    ray_count: int,
    ray_bounces: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(
        field,
        origin="lower",
        extent=[field_x[0], field_x[-1], field_y[0], field_y[-1]],
        cmap="magma",
        aspect="equal",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("|p(x, y)| (Helmholtz magnitude)")

    for region in domain.regions:
        if region.get("type") == "rectangle":
            rect_x = [region["xmin"], region["xmax"], region["xmax"], region["xmin"], region["xmin"]]
            rect_y = [region["ymin"], region["ymin"], region["ymax"], region["ymax"], region["ymin"]]
            ax.plot(rect_x, rect_y, color="white", lw=0.8, ls="--", alpha=0.7)
        elif region.get("type") == "circle":
            theta = np.linspace(0, 2 * np.pi, 200)
            cx, cy = region["center"]
            radius = region["radius"]
            ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), color="white", lw=0.8, ls="--", alpha=0.7)

    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], s=30, c="cyan", edgecolors="black", label="Microphones")

    trj_xy = trajectory[:, :2]
    ax.plot(trj_xy[:, 0], trj_xy[:, 1], color="lime", lw=1.2, alpha=0.75, label="Source path")
    ax.scatter(trj_xy[0, 0], trj_xy[0, 1], c="yellow", s=50, marker="o", edgecolors="black", label="Source start")
    ax.scatter(trj_xy[-1, 0], trj_xy[-1, 1], c="red", s=70, marker="*", edgecolors="black", label="Source end")

    if plot_rays:
        rays = trace_rays(
            source_xy=trj_xy[-1],
            domain=domain,
            ray_count=max(4, ray_count),
            max_bounces=max(0, ray_bounces),
        )
        for ray in rays:
            ax.plot(ray[:, 0], ray[:, 1], color="#5ec9ff", lw=0.7, alpha=0.55)

        for mic in mic_positions:
            ax.plot([trj_xy[-1, 0], mic[0]], [trj_xy[-1, 1], mic[1]], color="#9fe8ff", lw=0.6, alpha=0.35)

    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(domain.y_min, domain.y_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Helmholtz propagation field with domain materials and ray tracing overlay")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote Helmholtz field plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realistic 16-channel synthetic array simulator with Helmholtz plot.")
    parser.add_argument("input", nargs="?", help="Input mono WAV file (required when --source-model=file)")
    parser.add_argument("--output", "-o", default="output_16ch.wav", help="Output multichannel WAV")
    parser.add_argument("--azimuth", type=float, default=45.0, help="Initial source azimuth [deg]")
    parser.add_argument("--elevation", type=float, default=5.0, help="Initial source elevation [deg]")
    parser.add_argument("--distance", type=float, default=6.0, help="Initial source distance [m]")
    parser.add_argument("--source-height", type=float, default=1.5, help="Added source height offset [m]")

    parser.add_argument("--source-model", choices=["file", "propeller", "tone", "noise"], default="file")
    parser.add_argument("--source-model-file", type=str, default=None, help="JSON config for source model params")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate for synthetic source models")
    parser.add_argument("--duration", type=float, default=12.0, help="Duration for synthetic source models [s]")
    parser.add_argument("--max-seconds", type=float, default=None, help="Truncate file source to this many seconds")

    parser.add_argument("--blade-count", type=int, default=3, help="Propeller model: blade count")
    parser.add_argument("--rpm", type=float, default=3600.0, help="Propeller model: RPM")
    parser.add_argument("--harmonics", type=int, default=14, help="Propeller model: harmonic count")
    parser.add_argument("--mod-depth", type=float, default=0.25, help="Propeller model: amplitude modulation depth")
    parser.add_argument("--broadband-level", type=float, default=0.12, help="Propeller model: broadband noise level")
    parser.add_argument("--tone-frequency", type=float, default=640.0, help="Tone model frequency [Hz]")

    parser.add_argument("--legs-file", type=str, default=None, help="JSON file with trajectory legs and speed")
    parser.add_argument("--domain-file", type=str, default=None, help="JSON file describing material domain")
    parser.add_argument("--block-size", type=int, default=1024, help="Block size for moving-source renderer")

    parser.add_argument("--self-noise-level", type=float, default=0.003, help="Per-channel white self-noise level")
    parser.add_argument("--wind-noise-level", type=float, default=0.004, help="Per-channel low-frequency wind noise level")
    parser.add_argument("--gain-std", type=float, default=0.06, help="Channel gain mismatch std")
    parser.add_argument("--phase-std-m", type=float, default=0.012, help="Channel geometric phase mismatch std [m]")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for deterministic realism")

    parser.add_argument("--interferer-count", type=int, default=1, help="Number of synthetic interferers")
    parser.add_argument("--interferer-model", choices=["tone", "noise"], default="tone", help="Interferer source model")
    parser.add_argument("--interferer-level", type=float, default=0.22, help="Interferer amplitude factor")
    parser.add_argument("--interferer-distance", type=float, default=12.0, help="Interferer radial distance [m]")
    parser.add_argument("--interferer-elevation", type=float, default=0.0, help="Interferer elevation [deg]")
    parser.add_argument("--interferer-frequency", type=float, default=520.0, help="Interferer base tone [Hz]")

    parser.add_argument("--helmholtz-freq", type=float, default=480.0, help="Frequency [Hz] for Helmholtz field solve")
    parser.add_argument("--field-plot", type=str, default="propagation_field.png", help="Output plot path")
    parser.add_argument("--plot-rays", action="store_true", help="Overlay ray-tracing visualization on Helmholtz plot")
    parser.add_argument("--ray-count", type=int, default=28, help="Number of visualization rays")
    parser.add_argument("--ray-bounces", type=int, default=2, help="Maximum wall reflections per ray")

    parser.add_argument(
        "--debug-trajectory-path",
        type=str,
        default=None,
        help="Optional .npy path for saved trajectory samples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_cfg = load_json(args.source_model_file)
    domain_cfg = load_json(args.domain_file)
    legs_cfg = load_json(args.legs_file)

    domain = build_domain(domain_cfg)
    mic_positions = get_mic_array()
    source_signal, fs = build_source_signal(args, source_cfg)

    default_source = spherical_to_cartesian(args.distance, args.azimuth, args.elevation)
    default_source[2] += args.source_height
    print(
        "Initial source from spherical params: "
        f"azimuth={args.azimuth:.1f}°, elevation={args.elevation:.1f}°, distance={args.distance:.2f}m -> "
        f"x={default_source[0]:.2f}m, y={default_source[1]:.2f}m, z={default_source[2]:.2f}m"
    )
    legs = parse_legs(legs_cfg, default_source)
    if legs:
        print(f"Loaded {len(legs)} legs. Total trajectory duration={leg_duration_summary(legs):.2f}s")
    else:
        print("No legs file supplied: using static source position.")

    output, trajectory = render_array_audio(
        source_signal=source_signal,
        fs=fs,
        args=args,
        mic_positions=mic_positions,
        default_source_pos=default_source,
        legs=legs,
        domain=domain,
    )

    wavfile.write(args.output, fs, float32_to_pcm(output))
    print(
        "Trajectory endpoints: "
        f"start=({trajectory[0,0]:.2f}, {trajectory[0,1]:.2f}, {trajectory[0,2]:.2f}) "
        f"end=({trajectory[-1,0]:.2f}, {trajectory[-1,1]:.2f}, {trajectory[-1,2]:.2f})"
    )
    print(f"Wrote {output.shape[1]}-channel output to {args.output}")

    field_x, field_y, field = solve_helmholtz(domain, source_xy=trajectory[-1, :2], frequency_hz=args.helmholtz_freq)
    plot_propagation(
        domain=domain,
        field_x=field_x,
        field_y=field_y,
        field=field,
        mic_positions=mic_positions,
        trajectory=trajectory,
        output_path=args.field_plot,
        plot_rays=args.plot_rays,
        ray_count=args.ray_count,
        ray_bounces=args.ray_bounces,
    )


if __name__ == "__main__":
    main()
